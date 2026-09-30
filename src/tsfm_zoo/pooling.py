import torch


def pool(z: torch.Tensor, how: str, token_kinds: list[str], observed: torch.Tensor | None = None) -> torch.Tensor:
    """z [B, C, N, D] -> [B, C, D].

    "mean" averages the context patches (only the observed ones when
    `observed` [B, C, N] is given, so left-padding does not dilute it),
    "last" takes the last context patch, "reg" takes the REG token. Special
    tokens are never averaged in.
    """
    if how == "reg":
        return z[:, :, token_kinds.index("reg")]
    context = [i for i, kind in enumerate(token_kinds) if kind == "patch"]
    if how == "last":
        return z[:, :, context[-1]]
    if how == "mean":
        if observed is None:
            return z[:, :, context].mean(2)
        w = observed[:, :, context].to(z.dtype).unsqueeze(-1)
        return (z[:, :, context] * w).sum(2) / w.sum(2).clamp_min(1)
    raise ValueError(f"unknown pooling {how!r}; use 'mean', 'last' or 'reg'")
