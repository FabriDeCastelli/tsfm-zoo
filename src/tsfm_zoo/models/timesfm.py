"""TimesFM (google-research/timesfm 3.0.2): TimesFM 3.0 and 2.5."""
import math

import numpy as np
import torch
from timesfm import configs as timesfm_configs
from timesfm.timesfm_2p5 import timesfm_2p5_base
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.torch.util import revin
from timesfm3.torch.model import TimesFM3Torch
from timesfm3.torch.timesfm3_forecaster import linear_interpolation

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, context_loc_scale, patch_layout, run_encoder

MAX_VARIATES = 32  # timesfm3.torch.evaluator._MAX_VARIATES_PER_FORWARD


def strip_and_interpolate(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Upstream's per-series cleaning: drop leading steps where every variate is NaN, then
    fill the remaining NaNs by linear interpolation. x [v, T] -> (cleaned, number dropped)."""
    all_nan = np.isnan(x).all(axis=0)
    if all_nan.all():
        return np.zeros_like(x), 0
    first = int(np.argmax(~all_nan))
    return linear_interpolation(x[:, first:]), first


class TimesFM3(TSFM):
    """TimesFM 3.0: causal patch-32 transformer with variate attention, 64-step output patches.

    `z` and the native forecast come from the same upstream `decode` pass;
    the head reads that pass's forecast rather than re-deriving it from `z`.
    With `symmetric=True` (the official evaluator default) the head also runs
    `-x` and averages, and with `make_positive=True` it clamps forecasts of
    non-negative series at zero.
    """

    name = "timesfm-3.0"
    uses = frozenset({"mask", "past_covariates", "future_covariates", "channel_groups", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = True
    summary = "last"
    training_corpora = "undisclosed mix incl. GIFT-Eval-Pretrain (contains PEMS03/04/07/08)"

    def __init__(self, model: TimesFM3Torch, seed=0, symmetric=True, make_positive=True):
        super().__init__(seed)
        self.model = model
        self.embed_dim = model.transformer_config.transformer.model_dims
        self.patch_size = model.input_patch_len
        self.output_patch = model.output_patch_len
        self.max_context = 15360  # timesfm3 _MAX_CONTEXT_LENGTH, a multiple of 32
        self.native_levels = [float(q) for q in model.quantiles]
        self.symmetric = symmetric
        self.make_positive = make_positive
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", symmetric=True, make_positive=True):
        model = TimesFM3Torch.from_pretrained(str(fetch("timesfm", variant))).to(device).eval()
        return cls(model, seed=seed, symmetric=symmetric, make_positive=make_positive)

    def _prepare(self, x, past_covariates, future_covariates):
        """Rows as upstream's predict_batch builds them: cleaned, left-padded with zeros and a
        True (= padded) mask to the batch context, rounded up to the patch length."""
        B, T, C = x.shape
        targets, past, future, lengths = [], [], [], []
        for b in range(B):
            tgt, first = strip_and_interpolate(x[b].T.cpu().numpy().astype(np.float32))
            targets.append(tgt)
            past.append(None if past_covariates is None else
                        linear_interpolation(past_covariates[b, first:].T.cpu().numpy().astype(np.float32)))
            future.append(None if future_covariates is None else
                          linear_interpolation(future_covariates[b, first:].T.cpu().numpy().astype(np.float32)))
            lengths.append(tgt.shape[-1])
        L = min(math.ceil(max(lengths) / self.patch_size) * self.patch_size, self.max_context)
        L = max(L, self.patch_size)

        def fit(a, length):
            if a is None:
                return None
            n_future = a.shape[-1] - length
            ctx = a[:, :length][:, -L:]
            ctx = np.pad(ctx, [(0, 0), (L - ctx.shape[-1], 0)])
            return np.concatenate([ctx, a[:, length:]], axis=-1) if n_future > 0 else ctx

        def stack(arrays):
            return None if arrays[0] is None else torch.from_numpy(np.stack(arrays))

        pad = torch.tensor([[i < L - min(n, L) for i in range(L)] for n in lengths])
        tgt = stack([fit(t, n) for t, n in zip(targets, lengths)])
        return tgt, stack([fit(p, n) for p, n in zip(past, lengths)]), \
            stack([fit(f, n) for f, n in zip(future, lengths)]), pad

    def _decode(self, target, past, future, pad, horizon, layer):
        device = next(self.model.parameters()).device
        send = lambda t: None if t is None else t.to(device)
        (logits, aux), at_layer = run_encoder(self.model.transformer_stack.layers, layer, lambda: self.model.decode(
            target=send(target), horizon=horizon, past_only_covariates=send(past),
            past_future_covariates=send(future), mask=send(pad), return_aux_outputs=True))
        hidden = aux["__call__:transformer_output"] if layer is None else at_layer
        return logits, hidden

    def _encode(self, x, mask, timestamps, *, horizon=None, past_covariates=None, future_covariates=None,
                channel_groups=None, layer=None, **_):
        B, T, C = x.shape
        if channel_groups is not None:
            return self._encode_groups(x, mask, timestamps, channel_groups, horizon=horizon, layer=layer)
        n_variates = C + sum(c.shape[-1] for c in (past_covariates, future_covariates) if c is not None)
        if n_variates > MAX_VARIATES:
            raise ValueError(f"{self.name} attends over at most {MAX_VARIATES} variates per item (targets + "
                             f"covariates), got {n_variates}; split C with channel_groups")
        if future_covariates is not None:
            horizon = future_covariates.shape[1] - T
        # Upstream decodes a whole number of output patches and slices the horizon afterwards.
        global_horizon = math.ceil((horizon or 1) / self.output_patch) * self.output_patch
        if future_covariates is not None:
            global_horizon = horizon
        if T > self.max_context:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {self.max_context}")

        target, past, future, pad = self._prepare(x, past_covariates, future_covariates)
        logits, hidden = self._decode(target, past, future, pad, global_horizon, layer)
        hidden = hidden[:, :C]                                             # drop covariate variates

        L = target.shape[-1]
        n_ctx = L // self.patch_size
        n_future = hidden.shape[2] - n_ctx
        spans, observed = patch_layout(mask[:, -L:] if T >= L else torch.nn.functional.pad(
            mask, (0, 0, L - T, 0), value=False), T, self.patch_size, n_ctx)
        future_spans = [[T + self.patch_size * j, T + self.patch_size * (j + 1)] for j in range(n_future)]
        self.token_kinds = ["patch"] * n_ctx + ["future"] * n_future
        self.token_span = torch.cat([spans, torch.tensor(future_spans).reshape(-1, 2)])
        self.token_observed = torch.cat(
            [observed, torch.ones(B, C, n_future, dtype=torch.bool)], dim=-1)
        loc, scale = context_loc_scale(target.flatten(0, 1), pad.repeat_interleave(target.shape[1], 0))
        self.last_scale = (loc.view(B, -1)[:, :C], scale.view(B, -1)[:, :C])
        self._state = dict(hidden=hidden, logits=logits[:, :C], x=x, past=past_covariates, future=future_covariates,
                           global_horizon=global_horizon, layer=layer)
        return hidden

    def _encode_groups(self, x, mask, timestamps, channel_groups, **inputs):
        """Each group of channels is its own item: no attention across groups."""
        parts, order = [], []
        for g in channel_groups.unique():
            cols = (channel_groups == g).nonzero().flatten()
            z = self._encode(x[..., cols], mask[..., cols], timestamps, **inputs)
            parts.append((z, self._state["logits"], self.token_observed, *self.last_scale))
            order.append(cols)
        inverse = torch.argsort(torch.cat(order))
        z, logits, observed, loc, scale = (torch.cat(t, dim=1)[:, inverse] for t in zip(*parts))
        self.token_observed, self.last_scale = observed, (loc, scale)
        self._state.update(hidden=z, logits=logits, x=x, groups=channel_groups)
        return z

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None:
            raise MissingInputError(f"{self.name}: task='forecast' needs horizon")
        logits = state["logits"].float().cpu()                              # [B, C, H', Qn]
        if logits.shape[2] < horizon:
            raise ValueError(f"{self.name}: encode ran for {logits.shape[2]} steps; pass the same horizon to encode")
        preds = torch.sort(logits, dim=-1).values
        if self.symmetric:
            neg = self._flipped_logits(state)
            preds = (preds - torch.sort(neg, dim=-1).values.flip(-1)) / 2
        x = state["x"].cpu()
        if self.make_positive:
            nonneg = ((x >= 0) | x.isnan()).all(1) & (~x.isnan()).any(1)       # [B, C]
            preds = torch.where(nonneg[:, :, None, None], preds.clamp_min(0), preds)
        preds = preds[:, :, :horizon].permute(0, 2, 1, 3)                     # [B, H, C, Qn]

        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )

    def _flipped_logits(self, state):
        """The same decode on -x (and negated covariates), for symmetric averaging."""
        neg = lambda t: None if t is None else -t
        x = state["x"]
        groups = state.get("groups")
        cols = [torch.arange(x.shape[-1])] if groups is None else \
            [(groups == g).nonzero().flatten() for g in groups.unique()]
        out = torch.empty(x.shape[0], x.shape[-1], state["logits"].shape[2], len(self.native_levels))
        for c in cols:
            target, past, future, pad = self._prepare(-x[..., c], neg(state["past"]), neg(state["future"]))
            logits, _ = self._decode(target, past, future, pad, state["global_horizon"], None)
            out[:, c] = logits[:, :len(c)].float().cpu()
        return out


class TimesFM25(TSFM):
    """TimesFM 2.5 (200M): causal patch-32 decoder, 128-step output patches, continuous quantile head.

    Univariate: the C channels are folded into the batch. `z` is the prefill
    output embedding of the context patches; the decoder is causal, so `z`
    does not depend on the horizon. The native head is upstream's compiled
    decode with the GIFT-Eval notebook's flags (`FORECAST_FLAGS`), run in the
    same pass as `z`.
    """

    name = "timesfm-2.5"
    uses = frozenset({"mask", "layer"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "last"
    training_corpora = "undisclosed"
    FORECAST_FLAGS = dict(infer_is_positive=True, use_continuous_quantile_head=True, fix_quantile_crossing=True,
                          force_flip_invariance=True, return_backcast=False, normalize_inputs=True)
    MAX_HORIZON = 1024  # the continuous quantile head covers at most this many steps
    CHUNK = 128         # rows per decode, the notebook's per_core_batch_size

    def __init__(self, tfm: TimesFM_2p5_200M_torch, seed=0, forecast_flags=None):
        super().__init__(seed)
        self.tfm = tfm
        self.model = tfm.model  # the nn.Module; registered so .to()/.eval() reach it
        self.embed_dim = self.model.md
        self.patch_size = self.model.p
        self.max_context = 15360  # upstream's maximum with max_horizon = 1024 (limit 16384)
        self.native_levels = [float(q) for q in self.model.config.quantiles]
        self.flags = {**self.FORECAST_FLAGS, **(forecast_flags or {})}
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", forecast_flags=None):
        tfm = TimesFM_2p5_200M_torch(torch_compile=False)
        # Upstream pins the module to the first visible GPU whenever CUDA is available (PATCHES.md).
        tfm.model.device = torch.device(device)
        tfm.model.device_count = 1
        tfm.load_checkpoint(str(fetch("timesfm", variant)), torch_compile=False)
        return cls(tfm, seed=seed, forecast_flags=forecast_flags)

    def _prepare(self, x):
        """Rows as upstream's forecast() builds them: leading NaNs stripped, NaNs interpolated,
        left-padded with zeros and a True (= padded) mask to the longest row, rounded up to 32."""
        rows = [timesfm_2p5_base.linear_interpolation(timesfm_2p5_base.strip_leading_nans(r))
                for r in x.transpose(1, 2).flatten(0, 1).cpu().numpy().astype(np.float64)]
        L = min(math.ceil(max(len(r) for r in rows) / self.patch_size) * self.patch_size, self.max_context)
        values, masks = [], []
        for r in rows:
            r = r[-L:]
            values.append(np.pad(r, (L - len(r), 0)))
            masks.append(np.arange(L) < L - len(r))
        return values, masks, L

    def _encode(self, x, mask, timestamps, *, horizon=None, layer=None, **_):
        B, T, C = x.shape
        if horizon is not None and horizon > self.MAX_HORIZON:
            raise ValueError(f"{self.name}: the continuous quantile head covers at most {self.MAX_HORIZON} steps")
        if T > self.max_context:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {self.max_context}")
        values, masks, L = self._prepare(x)

        captured = []
        target = self.model if layer is None else self.model.stacked_xf[layer]
        pick = (lambda out: out[0][1]) if layer is None else (lambda out: out[0])
        hook = target.register_forward_hook(lambda module, args, out: captured.append(pick(out)))
        hidden, forecasts = [], []
        if horizon is not None:
            self.tfm.compile(timesfm_configs.ForecastConfig(
                max_context=L, max_horizon=self.MAX_HORIZON, per_core_batch_size=self.CHUNK, **self.flags))
        try:
            # Rows are independent; decode them in upstream forecast()'s chunks to bound memory.
            for start in range(0, len(values), self.CHUNK):
                vals, msks = values[start:start + self.CHUNK], masks[start:start + self.CHUNK]
                captured.clear()
                with torch.no_grad():
                    if horizon is None:
                        # Upstream's compiled decode without the flip and post-processing. Decoding
                        # MAX_HORIZON sizes the attention cache as upstream does, so z is bit-identical.
                        inputs = torch.tensor(np.array(vals), dtype=torch.float32, device=self.model.device)
                        if self.flags["normalize_inputs"]:
                            inputs = revin(inputs, inputs.mean(-1, keepdim=True), inputs.std(-1, keepdim=True))
                        self.model.decode(self.MAX_HORIZON, inputs, torch.tensor(np.array(msks), device=self.model.device))
                    else:
                        forecasts.append(self.tfm.compiled_decode(horizon, vals, msks)[1])
                # The first call is the prefill of the input; later calls are AR steps or the flipped input.
                hidden.append(captured[0])
        finally:
            hook.remove()
        hidden = torch.cat(hidden).view(B, C, -1, self.embed_dim)
        forecast = np.concatenate(forecasts) if forecasts else None

        n_patches = hidden.shape[2]
        observed_steps = torch.tensor(~np.array(masks)).view(B, C, L).transpose(1, 2)
        self.token_span, self.token_observed = patch_layout(observed_steps, T, self.patch_size, n_patches)
        self.token_kinds = ["patch"] * n_patches
        loc, scale = context_loc_scale(torch.tensor(np.array(values)), torch.tensor(np.array(masks)))
        self.last_scale = (loc.view(B, C).float(), scale.view(B, C).float())
        self._state = dict(hidden=hidden, forecast=forecast, layer=layer)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None or state["forecast"] is None:
            raise MissingInputError(f"{self.name}: task='forecast' needs horizon (also in encode)")
        B, C = z.shape[:2]
        # Upstream's output is [mean, q0.1 .. q0.9].
        preds = torch.from_numpy(state["forecast"][..., 1:]).float().view(B, C, horizon, -1).permute(0, 2, 1, 3)
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )


class TimesFM:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "3.0"
        if variant == "3.0":
            return TimesFM3.from_pretrained(variant, **kwargs)
        if variant == "2.5":
            return TimesFM25.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown timesfm variant {variant!r}")


MODEL = TimesFM
