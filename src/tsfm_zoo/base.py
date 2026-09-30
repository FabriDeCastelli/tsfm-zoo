import warnings
from typing import ClassVar

import torch
from torch import nn

# Inputs that carry data; providing one the model does not read warns once.
DATA_INPUTS = frozenset({
    "mask", "timestamps", "freq", "past_covariates", "future_covariates", "static_covariates",
    "channel_groups", "num_samples", "layer", "scale_factor", "patch_size",
})
# Inputs that control the call; every model accepts them.
CONTROL_INPUTS = frozenset({"horizon", "quantile_levels", "interpolate", "return_scale"})
TASKS = ("forecast", "reconstruct", "classify", "regress")
DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class MissingInputError(ValueError):
    pass


class TSFM(nn.Module):
    """Base class of every wrapper.

    Subclasses set the class attributes and implement `_encode` and
    `_native_head`. `encode` and `forward` validate inputs and never switch
    train/eval mode.

    Per-call state that the native head needs besides `z` (the scale, the
    token layout) is stored on the instance by `_encode`: `last_scale`,
    `token_kinds`, `token_span`, `token_observed`.
    """

    name: ClassVar[str]
    uses: ClassVar[frozenset[str]]          # data inputs the model reads, incl. "mask"
    requires: ClassVar[frozenset[str]] = frozenset()
    needs_one_of: ClassVar[frozenset[str]] = frozenset()  # at least one of these must be given
    input_docs: ClassVar[dict[str, str]] = {}
    tasks: ClassVar[dict[str, str]]         # task -> "native" | "probe" | "unsupported"
    cross_channel: ClassVar[bool]           # backbone mixes information across C
    summary: ClassVar[str]                  # recommended pooling: "last" | "mean" | "reg"
    normalization: ClassVar[str] = "affine"  # internal instance norm: "affine" (loc and scale) | "scale"
    reads_full_history: ClassVar[bool] = False  # preprocessing sees steps beyond max_context
    affine_equivariant: ClassVar[bool] = True   # forecast(a*x + b) == a*forecast(x) + b under a fixed seed
    training_corpora: ClassVar[str]

    # Set per instance, since they depend on the variant.
    embed_dim: int
    max_context: int
    patch_size: int | None
    spec: dict | None = None  # what `load_tsfm` loaded (family, variant, seed, load options), for cache keys

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed
        self.head: nn.Module | None = None
        self.last_scale: tuple[torch.Tensor, torch.Tensor] | None = None
        self.token_kinds: list[str] = []
        self.token_span: torch.Tensor | None = None      # [N, 2] time indices [start, end) in x; -1 for special tokens
        self.token_observed: torch.Tensor | None = None  # [B, C, N] bool
        self._warned: set[str] = set()

    def encode(self, x, mask=None, timestamps=None, **inputs):
        """x [B, T, C] -> z [B, C, N, D]; with return_scale=True also (loc, scale), each [B, C]."""
        self._check_inputs(x, mask, timestamps, inputs)
        x, mask = self._merge_mask(x, mask)
        z = self._encode(x, mask, timestamps, **inputs)
        if inputs.get("return_scale"):
            return z, self.last_scale
        return z

    def forward(self, x, mask=None, timestamps=None, *, task: str = "forecast", **inputs):
        support = self.tasks.get(task, "unsupported")
        if task not in TASKS or (support != "native" and self.head is None):
            raise ValueError(f"{self.name} has no {task!r} head; tasks: {self.tasks}. Use set_head for a probe.")
        if task == "forecast" and inputs.get("horizon") is None:
            raise MissingInputError(f"{self.name}: task='forecast' needs horizon")
        z = self.encode(x, mask, timestamps, **{k: v for k, v in inputs.items() if k != "return_scale"})
        return self.output_head(z, task=task, x=x, mask=mask, timestamps=timestamps, **inputs)

    def output_head(self, z, *, task: str, **inputs):
        if self.head is not None:
            return self.head(z, task=task, **inputs)
        return self._native_head(z, task=task, **inputs)

    def pool(self, z, how: str | None = None):
        """[B, C, N, D] -> [B, C, D] with the token layout of the latest encode call."""
        from .pooling import pool
        return pool(z, how or self.summary, self.token_kinds, self.token_observed)

    def set_head(self, head: nn.Module | None):
        """Replace the native head. `head(z, task=..., **inputs)`; None restores it."""
        self.head = head

    def generator(self, device) -> torch.Generator:
        """A fresh generator per call, so identical inputs give identical outputs."""
        return torch.Generator(device=device).manual_seed(self.seed)

    def warn_once(self, key: str, message: str):
        if key not in self._warned:
            warnings.warn(f"{self.name}: {message}", stacklevel=3)
            self._warned.add(key)

    def _check_inputs(self, x, mask, timestamps, inputs):
        if x.ndim != 3:
            raise ValueError(f"{self.name}: x must be [B, T, C], got shape {tuple(x.shape)}")
        unknown = set(inputs) - DATA_INPUTS - CONTROL_INPUTS
        if unknown:
            raise TypeError(f"{self.name} got unknown inputs {sorted(unknown)}")
        provided = {k for k, v in inputs.items() if v is not None and k in DATA_INPUTS}
        if mask is not None:
            provided.add("mask")
        if timestamps is not None:
            provided.add("timestamps")
        missing = self.requires - provided
        if missing:
            details = "; ".join(f"{k}: {self.input_docs.get(k, 'required')}" for k in sorted(missing))
            raise MissingInputError(f"{self.name} needs {sorted(missing)} ({details})")
        if self.needs_one_of and not provided & self.needs_one_of:
            raise MissingInputError(f"{self.name} needs one of {sorted(self.needs_one_of)}")
        for k in sorted(provided - self.uses):
            self.warn_once(f"ignored:{k}", f"ignores input {k!r}")

    @staticmethod
    def _merge_mask(x, mask):
        """NaN and mask == 0 mean the same thing; return x with NaN at missing values and the mask."""
        observed = ~torch.isnan(x)
        if mask is not None:
            observed = observed & mask.bool()
        return torch.where(observed, x, torch.nan), observed

    def _truncate(self, x, mask, max_context: int):
        """Keep the most recent `max_context` steps, as every upstream pipeline does."""
        if x.shape[1] > max_context:
            self.warn_once("truncate", f"context of {x.shape[1]} steps truncated to the last {max_context}")
            return x[:, -max_context:], mask[:, -max_context:]
        return x, mask

    def _encode(self, x, mask, timestamps, **inputs) -> torch.Tensor:
        raise NotImplementedError

    def _native_head(self, z, *, task: str, **inputs):
        raise NotImplementedError
