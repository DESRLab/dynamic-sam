# DynamicSAM

Interactive 3D point cloud segmentation model (XYZ-only, click-prompted). Model class is `DynamicSAM` (`src/models/dynamic_sam.py`); Lightning task wrapper is `SegmentPcdTask` (`src/task/segment_pcd_task.py`, name kept from an earlier iteration — same project).

## Layout

- `src/models/` — encoder/prompt-encoder/mask-decoder (`DynamicSAM`, `PointCloudEncoder`, `PromptEncoder`, `MaskDecoder`)
- `src/kernel/` — Triton/native-PyTorch replacements for what used to be MinkowskiEngine, `pointnet2_ops`, and `KNN_CUDA` (`fps.py`, `gather.py`, `knn.py`, `quantization.py`); no CUDA-toolkit build step needed
- `src/task/` — `SegmentPcdTask` (train/val/test steps, simulated clicks, adaptive threshold) and `cli.py` (shared `LightningCLI` entry point)
- `src/datasets/` — per-dataset loaders + `DataModule`s
- `src/serve/` — FastAPI app (`create_fast_api`) serving a trained checkpoint over HTTP; sessions are keyed by client-sent `X-User-Id`/`X-Pcd-Id` headers (see `UserFrameCache`), not stored on the model, so concurrent users don't clobber each other's encoded point cloud
- `src/cli/` — the `dynamic-sam` console command (`serve`, `train`, `test`)
- `configs/*.yaml` — one `LightningCLI` config per dataset
- `tests/` — pytest suite (kernel parity tests + `src/serve` session/concurrency tests)

## Commands

```bash
dynamic-sam train -c configs/scannet.yaml            # requires the `training` extra
dynamic-sam test -c configs/scannet.yaml --ckpt_path /path/to/last.ckpt
dynamic-sam serve /path/to/checkpoint.ckpt --port 8000  # requires the `inference` extra
```

Install with `pip install -e ".[training,inference,dev]"` (see `pyproject.toml` for the exact extras split and the pinned CUDA 12.4 torch/torchvision index).

## Things that will surprise a new reader

- No ONNX export/serving path exists anymore (removed); `src/serve` talks PyTorch directly.
- `group_size`/`num_group` on `PointCloudEncoder` can be overridden per-call (`encode_pcd(..., group_size=..., num_group=...)`) without mutating shared state — needed so concurrent `/encode_pcd` requests with different point-cloud sizes don't race.
- The mask decoder has no IoU-prediction head; adaptive thresholding happens via `AdaptiveIoU`'s τ-EMA in `src/task/utils/metrics.py`.
