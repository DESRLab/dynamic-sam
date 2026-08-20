from __future__ import annotations

from ..dataops import HM3D, HM3DConfig, Split

from .base import BaseDataModule

__all__ = ['HM3DDataModule']

class HM3DDataModule(BaseDataModule[HM3DConfig]):
    def __init__(self, dataset_config: HM3DConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_ds = HM3D(Split.TRAIN, self.ds_config)
            self.val_ds = HM3D(Split.VAL, self.ds_config)
        elif stage == 'test':
            self.test_ds = HM3D(Split.VAL, self.ds_config)
        else:
            self.full_ds = HM3D(Split.VAL, self.ds_config)