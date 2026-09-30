"""MOMENT (moment-timeseries-foundation-model/moment 0.1.5, vendored)."""
import torch

from .._vendor.moment.models.moment import MOMENTPipeline
from ..base import TSFM
from ..checkpoints import fetch
from ._common import check_head_state, patch_layout


class Moment(TSFM):
    """MOMENT: T5 encoder over 64 patches of 8 steps, a fixed 512-step window, with a
    pretrained reconstruction head.

    Univariate: channels are folded into the batch. Shorter series are
    left-padded and excluded through `input_mask`, as upstream recommends.
    Missing values are MOMENT's pretraining mask: a patch with any missing
    step is replaced by the mask token, and the head reconstructs it. `z` is
    the encoder output of that pass; with nothing missing it equals upstream's
    `embed(reduction="none")`. There is no zero-shot forecast (the forecasting
    head must be trained), so the native task is `reconstruct`, which returns
    `[B, min(T, 512), C]` in the original scale.
    """

    name = "moment"
    uses = frozenset({"mask", "layer"})
    tasks = {"forecast": "unsupported", "reconstruct": "native", "classify": "probe", "regress": "probe"}
    cross_channel = False
    summary = "mean"
    training_corpora = "Time Series Pile (Informer long-horizon sets incl. ETT, Monash, UCR/UEA, TSB-UAD)"

    def __init__(self, pipeline: MOMENTPipeline, seed=0):
        super().__init__(seed)
        self.model = pipeline
        self.embed_dim = pipeline.config.d_model
        self.patch_size = pipeline.config.patch_len
        self.max_context = pipeline.config.seq_len
        self._state = None

    @classmethod
    def from_pretrained(cls, variant, *, seed=0, device="cpu"):
        pipeline = MOMENTPipeline.from_pretrained(str(fetch("moment", variant)),
                                                  model_kwargs={"task_name": "reconstruction"})
        pipeline.init()
        return cls(pipeline.to(device).float().eval(), seed=seed)

    def _encode(self, x, mask, timestamps, *, layer=None, **_):
        B, T, C = x.shape
        x, mask = self._truncate(x, mask, self.max_context)
        L, S = x.shape[1], self.max_context
        device = next(self.model.parameters()).device
        rows = torch.nn.functional.pad(x.transpose(1, 2).flatten(0, 1), (S - L, 0))           # [N, 512]
        observed = torch.nn.functional.pad(mask.transpose(1, 2).flatten(0, 1), (S - L, 0))
        input_mask = torch.zeros(rows.shape[0], S)
        input_mask[:, S - L:] = 1

        captured = {}
        target = self.model.head if layer is None else self.model.encoder.block[layer]
        pick = (lambda args, out: args[0]) if layer is None else (lambda args, out: out[0])
        hook = target.register_forward_hook(lambda m, args, out: captured.__setitem__("h", pick(args, out)))
        try:
            with torch.no_grad():
                out = self.model.reconstruction(
                    x_enc=torch.nan_to_num(rows).unsqueeze(1).to(device), input_mask=input_mask.to(device),
                    mask=observed.long().to(device))
        finally:
            hook.remove()
        hidden = captured["h"].reshape(B, C, -1, self.embed_dim)

        n_patches = hidden.shape[2]
        self.token_kinds = ["patch"] * n_patches
        padded = torch.nn.functional.pad(mask, (0, 0, S - L, 0), value=False)
        self.token_span, self.token_observed = patch_layout(padded, T, self.patch_size, n_patches)
        stats = self.model.normalizer
        self.last_scale = (stats.mean.view(B, C).cpu(), stats.stdev.view(B, C).cpu())
        recon = out.reconstruction[:, 0, S - L:].float().cpu().view(B, C, L).transpose(1, 2)
        self._state = dict(hidden=hidden, reconstruction=recon, layer=layer)
        return hidden

    def _native_head(self, z, *, task, **_):
        return check_head_state(self, z)["reconstruction"]


class MomentFamily:
    @staticmethod
    def from_pretrained(variant=None, **kwargs):
        variant = variant or "large"
        if variant in ("small", "base", "large"):
            return Moment.from_pretrained(variant, **kwargs)
        raise KeyError(f"unknown moment variant {variant!r}")


MODEL = MomentFamily
