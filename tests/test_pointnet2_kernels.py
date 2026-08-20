"""Parity tests for src/kernel/fps.py and src/kernel/gather.py against the
reference CUDA-kernel algorithm from pointnet2_ops (erikwijmans/Pointnet2_PyTorch,
pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu).

The reference functions below are plain, unvectorized Python ports of that
.cu source (not imports from the pointnet2_ops package, which requires a
CUDA-toolkit build step we are trying to avoid). They exist purely as a
correctness oracle for the Triton kernel.
"""
import pytest
import torch

from src.kernel.fps import furthest_point_sample
from src.kernel.gather import gather_operation

CUDA = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA, reason="Triton kernel requires a CUDA device")


def reference_furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Direct port of furthest_point_sampling_kernel (sampling_gpu.cu)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, npoint, dtype=torch.long)
    for b in range(B):
        pts = xyz[b].tolist()
        valid = [(x * x + y * y + z * z) > 1e-3 for x, y, z in pts]
        temp = [1e10] * N
        old = 0
        idx[b, 0] = old
        for j in range(1, npoint):
            x1, y1, z1 = pts[old]
            best = -1.0
            besti = 0
            for k in range(N):
                if not valid[k]:
                    continue
                x2, y2, z2 = pts[k]
                d = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
                d2 = min(d, temp[k])
                temp[k] = d2
                if d2 > best:
                    best = d2
                    besti = k
            old = besti
            idx[b, j] = old
    return idx


def reference_gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Direct port of gather_points_kernel (sampling_gpu.cu)."""
    B, C, _ = features.shape
    npoint = idx.shape[1]
    out = torch.zeros(B, C, npoint, dtype=features.dtype)
    for b in range(B):
        for j in range(npoint):
            out[b, :, j] = features[b, :, idx[b, j]]
    return out


@skip_no_cuda
@pytest.mark.parametrize("B,N,npoint,seed", [(1, 32, 8, 0), (2, 50, 16, 1), (4, 100, 32, 2), (1, 4096, 128, 3)])
def test_furthest_point_sample_matches_reference(B, N, npoint, seed):
    xyz = torch.randn(B, N, 3, generator=torch.Generator().manual_seed(seed)).cuda()
    got = furthest_point_sample(xyz, npoint)
    ref = reference_furthest_point_sample(xyz.cpu(), npoint)
    assert torch.equal(got.cpu(), ref), f"mismatch:\n got={got.cpu()}\n ref={ref}"


@skip_no_cuda
def test_furthest_point_sample_always_starts_at_zero():
    xyz = torch.randn(3, 40, 3).cuda()
    idx = furthest_point_sample(xyz, 10)
    assert torch.all(idx[:, 0] == 0)


@skip_no_cuda
def test_furthest_point_sample_excludes_origin_padding():
    xyz = torch.randn(1, 20, 3).cuda()
    xyz[0, 5] = 0.0
    xyz[0, 11] = 0.0
    idx = furthest_point_sample(xyz, 15)[0].tolist()
    assert 5 not in idx
    assert 11 not in idx
    ref = reference_furthest_point_sample(xyz.cpu(), 15)[0].tolist()
    assert idx == ref


@skip_no_cuda
@pytest.mark.parametrize("B,C,N,npoint,seed", [(1, 3, 20, 5, 0), (3, 8, 50, 12, 1)])
def test_gather_operation_matches_reference(B, C, N, npoint, seed):
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(B, C, N, generator=g).cuda()
    idx = torch.randint(0, N, (B, npoint), generator=g).cuda()
    got = gather_operation(features, idx)
    ref = reference_gather_operation(features.cpu(), idx.cpu())
    assert torch.allclose(got.cpu(), ref)


@skip_no_cuda
def test_gather_operation_is_differentiable():
    features = torch.randn(2, 4, 10, requires_grad=True, device="cuda")
    idx = torch.randint(0, 10, (2, 3), device="cuda")
    out = gather_operation(features, idx)
    out.sum().backward()
    assert features.grad is not None
    assert features.grad.abs().sum().item() > 0


@skip_no_cuda
def test_furthest_point_sample_then_gather_end_to_end():
    B, N, C, npoint = 2, 64, 6, 16
    xyz = torch.randn(B, N, 3).cuda()
    features = torch.randn(B, C, N).cuda()
    fps_idx = furthest_point_sample(xyz, npoint)
    sampled = gather_operation(features, fps_idx)
    assert sampled.shape == (B, C, npoint)
    for b in range(B):
        for j in range(npoint):
            assert torch.allclose(sampled[b, :, j], features[b, :, fps_idx[b, j]])
