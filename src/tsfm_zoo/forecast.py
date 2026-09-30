from dataclasses import dataclass

import torch


@dataclass
class Forecast:
    """Unified forecast. H is the horizon, C the channels of x, Q the quantile levels."""

    median: torch.Tensor                         # [B, H, C], the point forecast
    quantiles: torch.Tensor | None = None        # [B, H, C, Q]
    quantile_levels: torch.Tensor | None = None  # [Q]
    samples: torch.Tensor | None = None          # [B, S, H, C], only when native
    dist: torch.distributions.Distribution | None = None  # only when native
    rollout_steps: int = 0                       # autoregressive steps past the native horizon


def select_quantiles(values, native_levels, levels, interpolate: bool, name: str):
    """values [..., Qn] at `native_levels` -> [..., Q] at `levels`.

    Levels the model does not emit are an error unless interpolate=True, so
    the zoo never quietly invents quantiles.
    """
    native = [round(q, 6) for q in native_levels]
    index = [native.index(round(q, 6)) if round(q, 6) in native else None for q in levels]
    if None not in index:
        return values[..., index]
    if not interpolate:
        extra = [q for q, i in zip(levels, index) if i is None]
        raise ValueError(f"{name} does not emit quantile levels {extra}; native: {native}. Pass interpolate=True.")
    return interpolate_quantiles(values, native_levels, levels)


def interpolate_quantiles(values, native_levels, levels):
    """Linear interpolation between native levels, clamped at the outermost ones."""
    native = torch.tensor(native_levels, dtype=values.dtype, device=values.device)
    query = torch.tensor(levels, dtype=values.dtype, device=values.device).clamp(native[0], native[-1])
    hi = torch.searchsorted(native, query).clamp(1, len(native) - 1)
    lo = hi - 1
    w = (query - native[lo]) / (native[hi] - native[lo])
    return values[..., lo] * (1 - w) + values[..., hi] * w
