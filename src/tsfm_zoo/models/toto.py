"""Toto 2.0 (DataDog/toto, toto-2 2.0.0)."""
import math

import numpy as np
import torch
from toto2 import Toto2Model
from toto2.model import ffill_imputation

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, context_loc_scale, patch_layout


class Toto2(TSFM):
    """Toto 2.0: causal patch-32 transformer alternating time and variate attention.

    Inputs follow upstream's GluonTS adapter (Toto2GluonTSModel): missing
    values are forward-filled and masked, and the context is the last
    `context_length - (ceil(H / 32) - 1) * 32` steps (zero-padded and masked
    when shorter), so `z` depends on the horizon through the context length.
    `z` covers those context patches plus one token per future patch; the
    native head is upstream's `forecast`, run in the same pass.
    """

    name = "toto-2"
    uses = frozenset({"mask", "past_covariates", "future_covariates", "channel_groups", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = True
    summary = "last"
    training_corpora = "GIFT-Eval Pretrain + Chronos datasets + Datadog internal + synthetic (contains PEMS03/04/07/08)"
    # Toto2GluonTSModelConfig defaults, as used by the GIFT-Eval notebook.
    FORECAST_FLAGS = dict(decode_block_size=None, has_missing_values=True, scaler_fallback_min_obs=8,
                          quantile_real_cap_k=1e4)

    def __init__(self, model: Toto2Model, seed=0, context_length=4096):
        super().__init__(seed)
        self.model = model
        self.embed_dim = model.config.d_model
        self.patch_size = model.config.patch_size
        self.max_context = context_length
        self.native_levels = [float(q) for q in model.output_head.knots]
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", context_length=4096):
        model = Toto2Model.from_pretrained(str(fetch("toto", variant)), map_location=str(device))
        return cls(model.to(device).eval(), seed=seed, context_length=context_length)

    def context_steps(self, horizon: int) -> int:
        """Upstream's instance splitter: the context shrinks as the horizon grows past one patch."""
        p = self.patch_size
        budget = self.max_context - (math.ceil(horizon / p) - 1) * p
        return max(p, budget // p * p)

    def _rows(self, a, L):
        """[B, T, K] with NaN -> values, observed: forward-filled, last L steps, zero-padded on the left."""
        a = a.transpose(1, 2).cpu().numpy().astype(np.float32)                 # [B, K, T]
        observed = ~np.isnan(a)
        values = np.stack([ffill_imputation(v) for v in a])
        pad = max(L - a.shape[-1], 0)
        values = np.pad(values[..., -L:], ((0, 0), (0, 0), (pad, 0)))
        observed = np.pad(observed[..., -L:], ((0, 0), (0, 0), (pad, 0)))
        return torch.from_numpy(values), torch.from_numpy(observed)

    def _encode(self, x, mask, timestamps, *, horizon=None, past_covariates=None, future_covariates=None,
                channel_groups=None, layer=None, **_):
        B, T, C = x.shape
        if future_covariates is not None and horizon is None:
            horizon = future_covariates.shape[1] - T
        H = horizon or self.patch_size
        L = self.context_steps(H)
        if T > L:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {L} (horizon {H})")
        device = next(self.model.parameters()).device

        target, target_mask = self._rows(x, L)
        if past_covariates is not None:  # upstream appends past-only covariates as extra target rows
            pv, po = self._rows(past_covariates, L)
            target, target_mask = torch.cat([target, pv], 1), torch.cat([target_mask, po], 1)
        series_ids = torch.zeros(B, target.shape[1], dtype=torch.long)
        if channel_groups is not None:  # variate attention only within equal series ids
            if past_covariates is not None or future_covariates is not None:
                raise ValueError(f"{self.name}: channel_groups cannot be combined with covariates")
            series_ids = channel_groups.to(torch.long).expand(B, C).clone()
        inputs = {"target": target.to(device), "target_mask": target_mask.to(device), "series_ids": series_ids.to(device)}
        if future_covariates is not None:
            fut = future_covariates[:, max(T - L, 0):].transpose(1, 2).float()
            fut = torch.nn.functional.pad(fut, (max(L - T, 0), 0), value=torch.nan)
            inputs |= {"known_dynamic": torch.nan_to_num(fut).to(device),
                       "known_dynamic_mask": (~fut.isnan()).to(device),
                       "known_dynamic_series_ids": torch.zeros(B, fut.shape[1], dtype=torch.long, device=device)}

        captured = []
        module = self.model.transformer if layer is None else self.model.transformer.layers[layer]
        hook = module.register_forward_hook(lambda m, args, out: captured.append(out))
        try:
            quantiles = self.model.forecast(inputs, H, **self.FORECAST_FLAGS)          # [Q, B, V, H]
        finally:
            hook.remove()
        hidden = captured[0]
        if layer is not None:  # layers run on [(B V), S, D]; the stack output is [B, V, S, D]
            hidden = hidden.view(B, -1, *hidden.shape[-2:])
        hidden = hidden[:, :C]

        n_ctx = L // self.patch_size
        n_future = hidden.shape[2] - n_ctx
        observed_steps = target_mask[:, :C].transpose(1, 2)
        spans, observed = patch_layout(observed_steps, T, self.patch_size, n_ctx)
        future_spans = torch.tensor([[T + self.patch_size * j, T + self.patch_size * (j + 1)] for j in range(n_future)])
        self.token_kinds = ["patch"] * n_ctx + ["future"] * n_future
        self.token_span = torch.cat([spans, future_spans.reshape(-1, 2)])
        self.token_observed = torch.cat([observed, torch.ones(B, C, n_future, dtype=torch.bool)], dim=-1)
        rows = target[:, :C].flatten(0, 1)
        loc, scale = context_loc_scale(rows, ~target_mask[:, :C].flatten(0, 1))
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        self._state = dict(hidden=hidden, quantiles=quantiles[:, :, :C].float().cpu(), horizon=horizon, layer=layer)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        preds = state["quantiles"].permute(1, 3, 2, 0)                                # [B, H, C, Q]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )


class Toto:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "2.0-313m"
        if variant.startswith("2.0-"):
            return Toto2.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown toto variant {variant!r}")


MODEL = Toto
