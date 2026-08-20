from __future__ import annotations

import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .segment_pcd_task import SegmentPcdTask

__all__ = ['run_lightning_cli']


def run_lightning_cli(args=None) -> None:
    """Runs the LightningCLI (fit/validate/test/predict) for SegmentPcdTask.

    `args`, when given, overrides sys.argv (e.g. ['fit', '-c', 'config.yaml']) --
    this is what lets `dynamic-sam train`/`dynamic-sam test` (src/cli) build
    args explicitly and call straight into this, instead of duplicating the
    Trainer/callback configuration.
    """
    torch.set_float32_matmul_precision('medium')
    LightningCLI(
        model_class=SegmentPcdTask,
        trainer_defaults={
            'accelerator': 'gpu',
            'callbacks': [
                LearningRateMonitor(),
                ModelCheckpoint(
                    filename='{epoch}-{step}-{Val_Metrics_EvalIoU_epoch:.5f}',
                    monitor='Val_Metrics_EvalIoU_epoch',
                    save_last=True,
                    save_top_k=3,
                    mode='max',
                ),
            ],
        },
        auto_configure_optimizers=False,
        args=args,
    )
