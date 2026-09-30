"""TiRex (NX-AI): TiRex-1 (tirex-ts 1.4.2) and TiRex-2 (tirex-2 0.3.0)."""
import contextlib
import io

import torch
import tirex2
from tirex.api_adapter.forecast import _resample_fc_func_wrapper
from tirex.models.tirex import TiRexZero
from tirex.util import frequency_resample

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, context_loc_scale, patch_layout


def on_device(device: str):
    """Make `device` the current CUDA device: TiRex-2 only accepts "cuda", meaning the current one."""
    device = torch.device(device)
    return torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()


def strip_leading_nans(row: torch.Tensor) -> torch.Tensor:
    observed = (~row.isnan()).nonzero()
    return row[observed[0, 0]:] if len(observed) else row[-1:]


class TiRex1(TSFM):
    """TiRex-1: sLSTM over patches of 32 with a NaN mask channel; context fixed at 2048 steps.

    Univariate: the C channels are folded into the batch. The recurrence is
    causal, so `z` does not depend on the horizon. Past 32 steps upstream
    appends NaN patches and runs again. With `resample_strategy="frequency"`
    (upstream's option, off by default) series are downsampled by their
    dominant period when the horizon is at least 100 steps; `z` is then the
    embedding of the downsampled series and `token_span` is None.
    """

    name = "tirex-1"
    uses = frozenset({"mask", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "last"
    training_corpora = "GiftEvalPretrain + autogluon/chronos_datasets + synthetic (contains PEMS03/04/07/08)"

    def __init__(self, model: TiRexZero, seed=0, resample_strategy=None):
        super().__init__(seed)
        self.model = model
        self.embed_dim = model.config.block_kwargs["embedding_dim"]
        self.patch_size = model.config.input_patch_size
        self.max_context = model.config.train_ctx_len
        self.native_levels = [float(q) for q in model.config.quantiles]
        self.resample_strategy = resample_strategy
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", backend="torch", resample_strategy=None):
        path = fetch("tirex", variant) / "model.ckpt"
        with contextlib.redirect_stdout(io.StringIO()):  # upstream prints the load path
            model = TiRexZero.from_pretrained(str(path), backend=backend, device=str(device))
        return cls(model.eval(), seed=seed, resample_strategy=resample_strategy)

    def _resampled(self, rows, horizon):
        """Upstream's frequency resampling; it is a no-op for horizons under 100."""
        out = [frequency_resample(strip_leading_nans(r), prediction_length=horizon,
                                  max_period=(self.model.max_context_length // 1000) * 500)[0] for r in rows]
        length = max(len(r) for r in out)
        return torch.stack([torch.nn.functional.pad(r, (length - len(r), 0), value=torch.nan) for r in out])

    def _encode(self, x, mask, timestamps, *, horizon=None, layer=None, **_):
        B, T, C = x.shape
        rows = x.transpose(1, 2).flatten(0, 1).float()                            # [B*C, T]
        resample = self.resample_strategy is not None and horizon is not None
        context = self._resampled(rows, horizon) if resample else rows
        if context.shape[-1] > self.max_context:
            self.warn_once("truncate", f"context of {context.shape[-1]} steps truncated to the last {self.max_context}")
        # _prepare_context_for_embedding pads/truncates to train_ctx_len, as a forecast step does.
        embeds, _ = self.model._prepare_context_for_embedding(context, self.max_context)
        with torch.inference_mode():
            _, hidden = self.model._forward_model(embeds, return_all_hidden=True)   # [N, P, layers, D]
        hidden = hidden[:, :, -1 if layer is None else layer].clone().view(B, C, -1, self.embed_dim)

        n_patches = hidden.shape[2]
        self.token_kinds = ["patch"] * n_patches
        observed = embeds[..., self.patch_size:].amax(-1).bool().view(B, C, n_patches)
        if resample:
            self.token_span = None
        else:
            self.token_span, _ = patch_layout(mask[:, -self.max_context:], T, self.patch_size, n_patches)
        self.token_observed = observed
        loc, scale = context_loc_scale(torch.nan_to_num(rows), rows.isnan())
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        self._state = dict(hidden=hidden, rows=rows, layer=layer)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None:
            raise MissingInputError(f"{self.name}: task='forecast' needs horizon")
        B, C = z.shape[:2]
        rows = state["rows"]
        # Upstream's own forecast path from the same context (see the class docstring).
        with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
            if self.resample_strategy is not None:
                quantiles, _ = _resample_fc_func_wrapper(
                    self.model._forecast_quantiles, [strip_leading_nans(r) for r in rows], self.resample_strategy,
                    max_context=self.model.max_context_length, prediction_length=horizon)
            else:
                quantiles, _ = self.model._forecast_quantiles(rows, prediction_length=horizon)
        preds = quantiles.float().cpu().view(B, C, horizon, -1).permute(0, 2, 1, 3)       # [B, H, C, Q]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
            rollout_steps=-(-horizon // self.patch_size) - 1,
        )


class TiRex2(TSFM):
    """TiRex-2: alternating mLSTM/sLSTM time mixers and variate attention over patches of 32.

    One pass covers `context_len` steps (left-padded with NaN) plus
    `future_len` future steps, so `z` always has the same tokens and does not
    depend on the horizon. The native head is upstream's `predict` (with the
    checkpoint's sign-flip and differencing settings), run in the same pass as
    `z`. With differencing on, trending series are differenced before they are
    encoded.
    """

    name = "tirex-2"
    reads_full_history = True  # trend detection for differencing runs on the whole series
    uses = frozenset({"mask", "past_covariates", "future_covariates", "channel_groups", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = True
    summary = "last"
    training_corpora = "GiftEvalPretrain + autogluon/chronos_datasets + synthetic (contains PEMS03/04/07/08); " \
                       "gifteval-* variants exclude GIFT-Eval test data"

    def __init__(self, forecaster, device, seed=0):
        super().__init__(seed)
        self.forecaster = forecaster
        self.model = forecaster.model  # the TiRex2 nn.Module
        self.device_str = str(device)
        self.embed_dim = self.model.embedding_dim
        self.patch_size = self.model.input_patch_size
        self.max_context = self.model.context_len
        self.max_horizon = self.model.future_len
        self.native_levels = [float(q) for q in self.model.quantiles]
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu"):
        kind = "cuda" if torch.device(device).type == "cuda" else "cpu"
        with on_device(device):
            forecaster = tirex2.load_model(fetch("tirex", variant), device=kind)
        return cls(forecaster, device, seed=seed)

    def _encode(self, x, mask, timestamps, *, horizon=None, past_covariates=None, future_covariates=None,
                channel_groups=None, layer=None, **_):
        B, T, C = x.shape
        if horizon is not None and horizon > self.max_horizon:
            raise ValueError(f"{self.name}: this checkpoint forecasts at most {self.max_horizon} steps")
        if future_covariates is not None and horizon is None:
            horizon = future_covariates.shape[1] - T
        if T > self.max_context:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {self.max_context}")
        if channel_groups is not None and (past_covariates is not None or future_covariates is not None):
            raise ValueError(f"{self.name}: channel_groups cannot be combined with covariates")

        # One upstream sample per item, or per (item, channel group).
        groups = [torch.arange(C)] if channel_groups is None else \
            [(channel_groups == g).nonzero().flatten() for g in channel_groups.unique()]
        samples, target_rows, row = [], [], 0
        for b in range(B):
            for cols in groups:
                pc = None if past_covariates is None else past_covariates[b].T.float()
                fc = None if future_covariates is None else future_covariates[b].T.float()
                samples.append(tirex2.TimeseriesType(target=x[b][:, cols].T.float(), past_covariates=pc,
                                                     future_covariates=fc))
                target_rows.append(torch.arange(row, row + len(cols)))
                row += len(cols) + (0 if pc is None else len(pc)) + (0 if fc is None else len(fc))

        captured = []
        if layer is None:
            hook = self.model.stack_out_norm.register_forward_pre_hook(lambda m, args: captured.append(args[0]))
        else:
            hook = self.model.stack[layer].register_forward_hook(lambda m, args, out: captured.append(out[0]))
        try:
            with on_device(self.device_str), torch.no_grad():
                forecasts = self.model.predict(samples, horizon or 1)
        finally:
            hook.remove()
        hidden = captured[0]                                   # the un-flipped pass; [rows, tokens, D]
        z = torch.empty(B, C, hidden.shape[1], self.embed_dim, device=hidden.device)
        preds = torch.empty(B, C, len(self.native_levels), horizon or 1)
        i = 0
        for b in range(B):
            for cols in groups:
                z[b, cols] = hidden[target_rows[i]]
                preds[b, cols] = forecasts[i].float().cpu()
                i += 1

        n_future = self.max_horizon // self.patch_size
        n_ctx = z.shape[2] - n_future
        spans, observed = patch_layout(torch.nn.functional.pad(mask[:, -self.max_context:],
                                                               (0, 0, n_ctx * self.patch_size - min(T, self.max_context), 0)),
                                       T, self.patch_size, n_ctx)
        future_spans = torch.tensor([[T + self.patch_size * j, T + self.patch_size * (j + 1)] for j in range(n_future)])
        self.token_kinds = ["patch"] * n_ctx + ["future"] * n_future
        self.token_span = torch.cat([spans, future_spans])
        self.token_observed = torch.cat([observed, torch.ones(B, C, n_future, dtype=torch.bool)], dim=-1)
        rows = x.transpose(1, 2).flatten(0, 1).float()
        loc, scale = context_loc_scale(torch.nan_to_num(rows), rows.isnan())
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        self._state = dict(hidden=z, preds=preds, horizon=horizon, layer=layer)
        return z

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        preds = state["preds"].permute(0, 3, 1, 2)                                  # [B, H, C, Q]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )


class TiRex:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "tirex-2"
        if variant.startswith("tirex-2"):
            return TiRex2.from_pretrained(variant, **kwargs)
        if variant.startswith("tirex-1"):
            return TiRex1.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown tirex variant {variant!r}")


MODEL = TiRex
