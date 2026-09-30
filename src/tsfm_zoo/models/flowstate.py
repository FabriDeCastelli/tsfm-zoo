"""FlowState (ibm-granite/granite-tsfm 0.3.9, tsfm_public)."""
import numpy as np
import torch
from tsfm_public import FlowStateForPrediction
from tsfm_public.models.flowstate.utils.utils import get_fixed_factor

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, context_loc_scale
from ..metrics import base_freq, to_offset


def infer_freq(timestamps: torch.Tensor) -> str:
    """Pandas frequency of the first row's epoch-second timestamps."""
    import pandas as pd
    index = pd.to_datetime(timestamps[0].cpu().numpy(), unit="s")
    freq = pd.infer_freq(index)
    if freq is None:
        raise MissingInputError("timestamps are irregular; pass freq explicitly")
    return freq


def old_style(freq: str) -> str:
    """'10min' -> '10T', 'h' -> 'H', 'W-SUN' -> 'W-SUN': the spelling get_fixed_factor expects."""
    offset = to_offset(freq)
    base = base_freq(freq)
    if base == "W":
        return offset.name.upper()
    return f"{offset.n}{base}" if offset.n > 1 else base


class FlowState(TSFM):
    """FlowState: per-step S5 state-space encoder with a functional-basis decoder.

    The time scale comes from `scale_factor = 24 / season` (upstream's
    `get_fixed_factor` of the frequency); pass `freq`, `timestamps` or
    `scale_factor`. Preprocessing follows IBM's GIFT-Eval wrapper: leading
    NaNs are dropped, the context is the last `4096 / scale_factor` steps,
    and with `make_positive=True` forecasts of non-negative series are
    clamped at zero. Univariate: channels are folded into the batch. `z` is
    the encoder state at the last context step, one token of 512.
    """

    name = "flowstate"
    uses = frozenset({"mask", "timestamps", "freq", "scale_factor"})
    needs_one_of = frozenset({"timestamps", "freq", "scale_factor"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "last"
    training_corpora = "GIFT-Eval Pretrain + TiRex's corpus minus synthetic (contains PEMS03/04/07/08)"
    BATCH_SIZE = 16  # series per call at the full 4096-step context, as in IBM's GIFT-Eval run

    def __init__(self, model: FlowStateForPrediction, seed=0, make_positive=True):
        super().__init__(seed)
        self.model = model
        self.embed_dim = model.config.encoder_state_dim
        self.patch_size = 1
        self.pretrain_context = model.config.context_length
        self.max_context = 16 * 1024  # upstream's cap; the effective context is 4096 / scale_factor
        self.native_levels = [float(q) for q in model.config.quantiles]
        self.make_positive = make_positive
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", make_positive=True):
        model = FlowStateForPrediction.from_pretrained(str(fetch("flowstate", variant))).to(device).eval()
        return cls(model, seed=seed, make_positive=make_positive)

    def _scale_factor(self, freq, timestamps, scale_factor):
        if scale_factor is not None:
            return float(scale_factor)
        return get_fixed_factor(old_style(freq or infer_freq(timestamps)))

    def _encode(self, x, mask, timestamps, *, horizon=None, freq=None, scale_factor=None, **_):
        B, T, C = x.shape
        sf = self._scale_factor(freq, timestamps, scale_factor)
        context = int(self.pretrain_context / sf)
        H = horizon or int(self.model.config.decoder_patch_len / sf)
        if T > context:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {context} (scale_factor {sf:.3g})")

        # IBM's GIFT wrapper: drop leading NaNs, keep the last `context` steps, batch equal lengths.
        rows = x.transpose(1, 2).flatten(0, 1).float().cpu().numpy()                # [N, T]
        series = []
        for r in rows:
            observed = ~np.isnan(r)
            series.append(r[np.argmax(observed):][-context:] if observed.any() else np.zeros(1, np.float32))
        device = next(self.model.parameters()).device
        z = torch.empty(len(series), self.embed_dim)
        preds = torch.empty(len(series), H, len(self.native_levels))
        groups = []
        for length in sorted({len(s) for s in series}):
            same = [i for i, s in enumerate(series) if len(s) == length]
            size = max(1, int(self.BATCH_SIZE * self.pretrain_context / length))  # the wrapper's dynamic batch size
            groups += [same[i:i + size] for i in range(0, len(same), size)]
        for idx in groups:
            batch = torch.from_numpy(np.stack([series[i] for i in idx], axis=1)).unsqueeze(-1)   # [L, n, 1]
            with torch.no_grad():
                out = self.model(past_values=batch.to(device), scale_factor=sf, prediction_length=H,
                                 batch_first=False)
            q = out.quantile_outputs.squeeze(-1).transpose(-1, -2).float().cpu()    # [n, H, Q]
            if self.make_positive:
                positive = (torch.nan_to_num(batch.squeeze(-1), 1) >= 0).all(0)
                q[positive] = q[positive].clamp(min=0.0)
            z[idx] = out.backbone_hidden_state[0].float().cpu()  # the state after the last context step
            preds[idx] = q

        hidden = z.view(B, C, 1, self.embed_dim)
        self.token_kinds = ["patch"]
        self.token_span = torch.tensor([[max(T - context, 0), T]])
        self.token_observed = (~x.isnan()).any(1).view(B, C, 1)
        values = torch.from_numpy(rows)
        loc, scale = context_loc_scale(torch.nan_to_num(values), values.isnan())
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        self._state = dict(hidden=hidden, preds=preds.view(B, C, H, -1), horizon=horizon, layer=None)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        preds = state["preds"].permute(0, 2, 1, 3)                                   # [B, H, C, Q]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )


class FlowStateFamily:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "r1.1"
        if variant == "r1.1":
            return FlowState.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown flowstate variant {variant!r}")


MODEL = FlowStateFamily
