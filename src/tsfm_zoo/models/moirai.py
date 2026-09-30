"""Moirai (SalesforceAIResearch/uni2ts 2.0.0): Moirai 1.x, Moirai-MoE and Moirai 2.0."""
import numpy as np
import torch
from einops import rearrange
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast, select_quantiles
from ._common import check_head_state, context_loc_scale, patch_layout

CONTEXT = 4000  # the GIFT-Eval notebooks' context length


def capture_first_input(module):
    """Register a hook that stores the first positional input of `module`'s first call."""
    captured = []
    hook = module.register_forward_pre_hook(lambda m, args: captured.append(args[0]) if not captured else None)
    return captured, hook


class Moirai1(TSFM):
    """Moirai 1.x / Moirai-MoE: any-variate transformer over patches, sampled forecasts.

    Items are natively multivariate: all C channels of an item share one
    flattened token sequence. Inputs follow uni2ts' GluonTS predictor:
    missing values become 0 with an observed mask, the context is the last
    4000 steps, and steps before a series starts (left padding where every
    channel is NaN) are `past_is_pad`. Moirai 1.x is a masked encoder, so `z`
    comes from its own pass with one masked patch and holds the context tokens
    only; it does not depend on the horizon. The native head runs upstream's
    forward for the horizon, sampling from the model's seed.
    """

    uses = frozenset({"mask", "num_samples", "patch_size"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = True
    summary = "mean"
    training_corpora = "LOTSA (contains PEMS03/04/07/08)"

    def __init__(self, forecaster, name, seed=0, patch_size=32, num_samples=100):
        super().__init__(seed)
        self.name = name
        # MoE feeds sampled patches back in; its negative-binomial component is sampled by
        # rejection, so a rescaled input consumes a different random stream.
        self.affine_equivariant = not name.startswith("moirai-moe")
        self.forecaster = forecaster
        self.model = forecaster.module
        self.embed_dim = self.model.d_model
        self.patch_size = patch_size
        self.max_context = CONTEXT
        self.num_samples = num_samples
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", patch_size=None, num_samples=100):
        path = str(fetch("moirai", variant))
        if variant.startswith("moe"):
            patch_size = patch_size or 16
            forecaster = MoiraiMoEForecast(module=MoiraiMoEModule.from_pretrained(path), prediction_length=1,
                                           context_length=CONTEXT, patch_size=patch_size, num_samples=num_samples,
                                           target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        else:
            patch_size = patch_size or 32
            forecaster = MoiraiForecast(module=MoiraiModule.from_pretrained(path), prediction_length=1,
                                        context_length=CONTEXT, patch_size=patch_size, num_samples=num_samples,
                                        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return cls(forecaster.to(device).eval(), f"moirai-{variant}", seed=seed, patch_size=patch_size,
                   num_samples=num_samples)

    def _inputs(self, x):
        """past_target, past_observed_target, past_is_pad as the TFT instance splitter builds them."""
        B, T, C = x.shape
        x = x[:, -CONTEXT:]
        observed = ~x.isnan()
        is_pad = ~observed.any(-1)
        is_pad = torch.cummin(is_pad.int(), dim=1).values.bool()        # only leading all-NaN steps
        pad = CONTEXT - x.shape[1]
        target = torch.nn.functional.pad(torch.nan_to_num(x), (0, 0, pad, 0))
        observed = torch.nn.functional.pad(observed, (0, 0, pad, 0), value=False)
        is_pad = torch.nn.functional.pad(is_pad, (pad, 0), value=True)
        device = next(self.model.parameters()).device
        return target.float().to(device), observed.to(device), is_pad.to(device)

    def _encode(self, x, mask, timestamps, *, horizon=None, num_samples=None, patch_size=None, **_):
        B, T, C = x.shape
        if patch_size is not None and patch_size != self.patch_size:
            raise ValueError(f"{self.name}: patch_size is fixed at load ({self.patch_size})")
        if T > CONTEXT:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {CONTEXT}")
        target, observed, is_pad = self._inputs(x)
        p = self.patch_size
        n_ctx = self.forecaster.context_token_length(p)

        captured, hook = capture_first_input(self.model.param_proj)
        try:
            with torch.no_grad(), self.forecaster.hparams_context(prediction_length=p, target_dim=C), \
                    torch.random.fork_rng(devices=[target.device] if target.is_cuda else []):
                torch.manual_seed(self.seed)
                self.forecaster(target, observed, is_pad, num_samples=1)
        finally:
            hook.remove()
        hidden = rearrange(captured[0][:, :C * n_ctx].float(), "b (c n) d -> b c n d", c=C)

        samples = None
        if horizon is not None:
            with torch.no_grad(), self.forecaster.hparams_context(prediction_length=horizon, target_dim=C), \
                    torch.random.fork_rng(devices=[target.device] if target.is_cuda else []):
                torch.manual_seed(self.seed)
                samples = self.forecaster(target, observed, is_pad, num_samples=num_samples or self.num_samples)
            samples = samples.float().cpu()
            samples = samples.unsqueeze(-1) if C == 1 else samples                   # [B, S, H, C]

        observed_steps = torch.nn.functional.pad(observed.cpu(), (0, 0, n_ctx * p - CONTEXT, 0), value=False) \
            if n_ctx * p > CONTEXT else observed.cpu()[:, -n_ctx * p:]
        self.token_kinds = ["patch"] * n_ctx
        self.token_span, self.token_observed = patch_layout(observed_steps, T, p, n_ctx)
        rows = target.cpu().transpose(1, 2).flatten(0, 1)
        loc, scale = context_loc_scale(rows, ~observed.cpu().transpose(1, 2).flatten(0, 1))
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        self._state = dict(hidden=hidden, samples=samples, horizon=horizon, layer=None)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        samples = state["samples"]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        quantiles = torch.from_numpy(np.quantile(samples.numpy(), levels, axis=1)).permute(1, 2, 3, 0).float()
        return Forecast(
            median=torch.from_numpy(np.quantile(samples.numpy(), 0.5, axis=1)).float(),
            quantiles=quantiles,
            quantile_levels=torch.tensor(levels),
            samples=samples,
        )


class Moirai2(TSFM):
    """Moirai 2.0: causal decoder over patches of 16 with multi-token quantile output.

    Univariate: channels are folded into the batch. The native head is
    upstream's `Moirai2Forecast.predict` (causal-mean imputation of NaNs,
    first-value padding to 4000 steps), run in the same pass as `z`. Our
    left NaN padding is removed first, so upstream sees each series at its
    own length. `z` holds the context tokens of that pass.
    """

    name = "moirai-2.0-small"
    uses = frozenset({"mask"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "last"
    training_corpora = "GIFT-Eval Pretrain + synthetic (contains PEMS03/04/07/08)"

    def __init__(self, forecaster, seed=0):
        super().__init__(seed)
        self.forecaster = forecaster
        self.model = forecaster.module
        self.embed_dim = self.model.d_model
        self.patch_size = self.model.patch_size
        self.max_context = CONTEXT
        self.native_levels = [float(q) for q in self.model.quantile_levels]
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu"):
        module = Moirai2Module.from_pretrained(str(fetch("moirai", variant)))
        forecaster = Moirai2Forecast(module=module, prediction_length=1, context_length=CONTEXT, target_dim=1,
                                     feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return cls(forecaster.to(device).eval(), seed=seed)

    def _encode(self, x, mask, timestamps, *, horizon=None, **_):
        B, T, C = x.shape
        if T > CONTEXT:
            self.warn_once("truncate", f"context of {T} steps truncated to the last {CONTEXT}")
        rows = x.transpose(1, 2).flatten(0, 1).float().cpu().numpy()
        series = []
        for r in rows:
            observed = ~np.isnan(r)
            series.append(r[np.argmax(observed):] if observed.any() else np.zeros(1, np.float32))
        H = horizon or self.patch_size
        captured, hook = capture_first_input(self.model.out_proj)
        try:
            with self.forecaster.hparams_context(prediction_length=H):
                preds = self.forecaster.predict(series)                              # [N, Q, H]
        finally:
            hook.remove()
        n_ctx = self.forecaster.context_token_length(self.patch_size)
        hidden = captured[0][:, :n_ctx].float().view(B, C, n_ctx, self.embed_dim)

        steps = torch.nn.functional.pad(mask[:, -CONTEXT:], (0, 0, n_ctx * self.patch_size - min(T, CONTEXT), 0),
                                        value=False)
        self.token_kinds = ["patch"] * n_ctx
        self.token_span, self.token_observed = patch_layout(steps, T, self.patch_size, n_ctx)
        values = torch.from_numpy(rows)
        loc, scale = context_loc_scale(torch.nan_to_num(values), values.isnan())
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        preds = torch.from_numpy(np.asarray(preds)).float().view(B, C, len(self.native_levels), H)
        self._state = dict(hidden=hidden, preds=preds, horizon=horizon, layer=None)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, interpolate=False, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        preds = state["preds"].permute(0, 3, 1, 2)                                   # [B, H, C, Q]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        return Forecast(
            median=preds[..., self.native_levels.index(0.5)],
            quantiles=select_quantiles(preds, self.native_levels, levels, interpolate, self.name),
            quantile_levels=torch.tensor(levels),
        )


class Moirai:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "2.0-small"
        if variant == "2.0-small":
            return Moirai2.from_pretrained(variant, **kwargs)
        if variant.startswith(("1.0-", "1.1-", "moe-")):
            return Moirai1.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown moirai variant {variant!r}")


MODEL = Moirai
