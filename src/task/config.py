
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from torch import device
from typing import Optional

__all__ = ['Configs']


@dataclass(frozen=True)
class Configs():
    ngpus: int
    distributed: bool
    batch_size: int
    val_batch_size: int
    workers: int
    multi_gpu: bool
    start_epoch: int
    device: device
    checkpoints_path: Path
    gpus: str
    gpu_ids: list[int]
    logs_path: Path
    local_rank: int
    model_path: Path
    dataset_root: str
    drop_path_rate: float
    weight_decay: float
    lr: float
    num_epochs: int
    resume_exp: Optional[bool]
    exp_name: str
    temp_model_path: Optional[str]
    exps_path: str
