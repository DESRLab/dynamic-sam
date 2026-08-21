# DynamicSAM

Interactive 3D point cloud segmentation with adaptive masking.

## Environment

Tested with:
- Ubuntu 22.04
- Python 3.10
- PyTorch 2.6.0, on CUDA 11.8 through 12.6 (see [Installation](#1-installation) for picking the right build for your GPU driver)

## 1. Installation

`torch`/`triton` need a build matching your machine's NVIDIA driver. `uv` can detect this automatically:

```shell
$ uv venv dynamic-sam --python 3.10
$ source dynamic-sam/bin/activate

# pick the extras you need:
$ uv pip install -e ".[training]" --torch-backend=auto                # to train/evaluate
$ uv pip install -e ".[inference]" --torch-backend=auto               # to serve a trained checkpoint
$ uv pip install -e ".[training,inference,dev]" --torch-backend=auto  # everything, including the test suite
```

`--torch-backend=auto` inspects your driver and picks a matching torch/torchvision build (cu118, cu124, cu126, ...) instead of whatever plain PyPI happens to default to. If detection picks the wrong one, or you're installing for a different machine than the one running the command, pass an explicit backend instead, e.g. `--torch-backend=cu118`. This can't be set as a persistent default in `pyproject.toml` (not supported by `uv` 0.8), so it has to be passed on every install command; alternatively, export `UV_TORCH_BACKEND=auto` (or a specific `cuXXX`) once per shell session.

### Using pip

```shell
$ python3.10 -m venv dynamic-sam
$ source dynamic-sam/bin/activate
$ pip install -e ".[training,inference,dev]" --extra-index-url https://download.pytorch.org/whl/cu118  # replace cu118 with your driver's supported CUDA version
```

## 2. Dataset Setup

### Scannet & S3DIS
Special thanks to Pointcept framework contributers for providing preprocessed version of these two dataset.

Download their preprocessed data from their [repository](https://github.com/Pointcept/Pointcept/tree/main).

### Kitti360
Download accumulated Point Clouds for Train & Val (12G)/Test Semantic (1.2G) [here](https://www.cvlibs.net/datasets/kitti-360/download.php) in their official website.

### STPLS3D
Download synthetic and real instance segmentation [here](https://www.stpls3d.com/data) in their official website and follow their instruction on how to prepare STPL dataset.

### Semantic Kitti
Download semantic kitti from their official website [here](http://www.semantic-kitti.org/).

## 3. Training / Evaluation

Requires the `training` extra (see [Installation](#1-installation)). Use the `dynamic-sam` CLI:

```shell
# for training
$ dynamic-sam train -c configs/scannet.yaml

# for testing/evaluation from a checkpoint
$ dynamic-sam test -c configs/scannet.yaml --ckpt_path /path/to/last.ckpt
```

Anything after `-c CONFIG` is forwarded to LightningCLI verbatim, so any Trainer/model override supported by the config schema (e.g. `--trainer.max_epochs=10`) also works on the command line.

## 4. Serving

Requires the `inference` extra (see [Installation](#1-installation)). Serve a local checkpoint (`.ckpt` or `.safetensors`):

```shell
$ dynamic-sam serve /path/to/checkpoint.ckpt --port 8000
```

Or serve directly from a Hugging Face Hub repo, without downloading anything by hand:

```shell
$ dynamic-sam serve --hf-repo-id Marali/dynamic-sam --port 8000
```

This downloads through the local Hugging Face cache (`HF_HOME`), reusing an already-cached file rather than re-downloading, and uses your `hf auth login` token automatically for private repos.

See `dynamic-sam serve --help` for the full option list, including `--max-users`/`--max-frames-per-user` to size the per-user encoded-point-cloud cache (defaults: 1 user, 5 cached frames each).

## Acknowledgment

Parts of our code are built by borrowing ideas and being inspired by following repositiories:
- [Segment Anything Model](https://github.com/facebookresearch/segment-anything.git): Mask decoder architecture
- [PointBERT](https://github.com/lulutang0608/Point-BERT.git): Point Cloud encoder architecture
- [Reviving Iterative Training with Mask Guidance for Interactive Segmentation](https://github.com/SamsungLabs/ritm_interactive_segmentation.git): Loss, evaluation metrics and training loop for interactive semgnetation.
