from __future__ import annotations

import torch

__all__ = ['gather_operation']


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Per-batch gather along the point dimension.

    Semantics match `pointnet2_ops.pointnet2_utils.gather_operation`: gathers
    `features[b, :, idx[b, j]]` for every batch `b` and output position `j`.
    Implemented with `torch.gather`, which is already a native GPU op and comes
    with autograd support, so no custom backward is needed.

    Args:
        features: (B, C, N) tensor.
        idx: (B, npoint) integer tensor of indices into the N dimension.
    Returns:
        (B, C, npoint) gathered tensor.
    """
    idx = idx.long()
    B, C, _ = features.shape
    npoint = idx.shape[1]
    idx_expanded = idx.view(B, 1, npoint).expand(B, C, npoint)
    return torch.gather(features, 2, idx_expanded)
