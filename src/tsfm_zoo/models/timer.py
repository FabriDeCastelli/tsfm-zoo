"""thuml Timer family via HF remote code: Sundial (sundial-base-128m) and Timer (timer-base-84m).

The remote code needs transformers 4.40 (the `legacy` env); under
transformers 5 it fails on `DynamicCache.seen_tokens`.
"""
import contextlib

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM

from ..base import DEFAULT_QUANTILES, TSFM, MissingInputError
from ..checkpoints import fetch
from ..forecast import Forecast
from ._common import check_head_state, context_loc_scale, patch_layout

if int(transformers.__version__.split(".")[0]) >= 5:
    raise ImportError("thuml remote code needs transformers 4.40; use the legacy env")


def last_value_imputation(row: np.ndarray) -> np.ndarray:
    """gluonts LastValueImputation: forward fill, leading NaNs take the first observed value."""
    if len(row) == 1 or np.isnan(row).all():
        return np.zeros_like(row)
    idx = np.where(~np.isnan(row), np.arange(len(row)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = row[idx]
    missing = np.isnan(out)
    out[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), out[~missing])
    return out


class Thuml(TSFM):
    """Decoder-only patch transformer; `z` is the last hidden state of the context patches.

    Univariate: channels are folded into the batch. Missing values are
    imputed with gluonts' LastValueImputation, as in the GIFT-Eval notebook,
    and the context is the last 2880 steps. The causal decoder makes `z`
    independent of the horizon. The native head is upstream's `generate`,
    run in the same pass: Sundial draws `num_samples` flow-matching samples
    (seeded, not from the global RNG), Timer returns a point forecast.
    """

    uses = frozenset({"mask", "layer", "num_samples"})
    tasks = {"forecast": "native", "reconstruct": "unsupported", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "last"
    training_corpora = "UTSD / TimeBench (Sundial: 1T points incl. LOTSA, which contains PEMS03/04/07/08)"

    def __init__(self, model, name, sampled: bool, seed=0, autocast=None):
        super().__init__(seed)
        self.name = name
        self.model = model
        self.embed_dim = model.config.hidden_size
        self.patch_size = model.config.input_token_len
        self.max_context = 2880
        self.sampled = sampled
        self.autocast = autocast
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu", autocast=None):
        model = AutoModelForCausalLM.from_pretrained(str(fetch("timer", variant)), trust_remote_code=True)
        return cls(model.to(device).eval(), variant, sampled=variant.startswith("sundial"), seed=seed,
                   autocast=autocast)

    def _encode(self, x, mask, timestamps, *, horizon=None, num_samples=None, layer=None, **_):
        B, T, C = x.shape
        x, mask = self._truncate(x, mask, self.max_context)
        rows = x.transpose(1, 2).flatten(0, 1).cpu().numpy().astype(np.float32)
        if np.isnan(rows).any():
            rows = np.stack([last_value_imputation(r) for r in rows])
        device = next(self.model.parameters()).device
        inputs = torch.from_numpy(rows).to(device)
        H = horizon or self.patch_size

        captured = []
        module = self.model.model if layer is None else self.model.model.layers[layer]
        hook = module.register_forward_hook(lambda m, args, out: captured.append(out[0]))
        autocast = torch.autocast(device_type=device.type, dtype=getattr(torch, self.autocast)) \
            if self.autocast else contextlib.nullcontext()
        kwargs = dict(num_samples=num_samples or 20) if self.sampled else {}
        try:
            with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), autocast:
                torch.manual_seed(self.seed)
                out = self.model.generate(inputs, max_new_tokens=H, **kwargs)
        finally:
            hook.remove()
        hidden = captured[0].float().view(B, C, -1, self.embed_dim)

        n_patches = hidden.shape[2]
        self.token_kinds = ["patch"] * n_patches
        self.token_span, self.token_observed = patch_layout(mask, T, self.patch_size, n_patches)
        values = torch.from_numpy(rows)
        loc, scale = context_loc_scale(values, torch.zeros_like(values, dtype=torch.bool))
        self.last_scale = (loc.view(B, C), scale.view(B, C))
        out = out.float().cpu()[..., :H]
        self._state = dict(hidden=hidden, out=out, horizon=horizon, layer=layer)
        return hidden

    def _native_head(self, z, *, task, horizon=None, quantile_levels=None, **_):
        state = check_head_state(self, z)
        if horizon is None or state["horizon"] != horizon:
            raise MissingInputError(f"{self.name}: task='forecast' needs the same horizon in encode and the head")
        B, C = z.shape[:2]
        if not self.sampled:  # Timer: point forecast only
            return Forecast(median=state["out"].view(B, C, horizon).transpose(1, 2))
        samples = state["out"].view(B, C, -1, horizon).permute(0, 2, 3, 1)           # [B, S, H, C]
        levels = list(quantile_levels or DEFAULT_QUANTILES)
        quantiles = torch.from_numpy(np.quantile(samples.numpy(), levels, axis=1)).permute(1, 2, 3, 0).float()
        return Forecast(
            median=torch.from_numpy(np.quantile(samples.numpy(), 0.5, axis=1)).float(),
            quantiles=quantiles,
            quantile_levels=torch.tensor(levels),
            samples=samples,
        )


class Timer:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "sundial-base-128m"
        if variant in ("sundial-base-128m", "timer-base-84m"):
            return Thuml.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown timer variant {variant!r}")


MODEL = Timer
