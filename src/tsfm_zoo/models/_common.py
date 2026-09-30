"""Helpers shared by the wrappers; no upstream package imports here."""
import torch


def run_encoder(blocks, layer, encode):
    """Call `encode()`; with `layer`, also capture that block's hidden state."""
    if layer is None:
        return encode(), None
    captured = {}
    hook = blocks[layer].register_forward_hook(lambda module, args, out: captured.__setitem__("h", out[0]))
    try:
        return encode(), captured["h"]
    finally:
        hook.remove()


def patch_layout(mask, T: int, patch: int, n_patches: int):
    """Spans [P, 2] of the context patches in x's time index, and observed [B, C, P]."""
    T_ctx = mask.shape[1]
    ends = T - patch * torch.arange(n_patches - 1, -1, -1)
    spans = torch.stack([(ends - patch).clamp_min(max(T - T_ctx, 0)), ends], dim=1)
    pad = n_patches * patch - T_ctx
    observed = torch.nn.functional.pad(mask.transpose(1, 2), (pad, 0), value=False)
    return spans, observed.reshape(*observed.shape[:2], n_patches, patch).any(-1)


def special_tokens(spans, observed, extra_spans):
    """Append special tokens (always observed) after the context patches."""
    extra = torch.as_tensor(extra_spans, dtype=spans.dtype).reshape(-1, 2)
    ones = torch.ones(*observed.shape[:2], len(extra), dtype=torch.bool, device=observed.device)
    return torch.cat([spans, extra]), torch.cat([observed, ones], dim=-1)


def check_head_state(model, z):
    state = model._state
    if state is None or state["hidden"].data_ptr() != z.data_ptr():
        raise RuntimeError(f"{model.name}: the native head needs the z returned by the latest encode call")
    if state["layer"] is not None:
        raise ValueError(f"{model.name}: the native head reads the last layer; drop `layer` to forecast")
    return state


def context_loc_scale(values: torch.Tensor, padded: torch.Tensor):
    """Mean and std of the observed context, i.e. the model's running statistics after the
    last context patch. values, padded [N, L] -> two [N] tensors."""
    w = (~padded).float()
    n = w.sum(-1).clamp_min(1)
    loc = (values * w).sum(-1) / n
    return loc, (((values - loc[:, None]) ** 2 * w).sum(-1) / n).sqrt()
