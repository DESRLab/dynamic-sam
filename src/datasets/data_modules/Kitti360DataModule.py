from __future__ import annotations

from ..dataops import Kitti360, Kitti360V2, Kitti360Config, Split

from .base import BaseDataModule

__all__ = ['Kitti360DataModule']

class Kitti360DataModule(BaseDataModule[Kitti360Config]):
    def __init__(self, dataset_config: Kitti360Config, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            if self.use_v2:
                raise ValueError('Training based on V2 is not supported')
            self.train_ds = Kitti360(Split.TRAIN, self.ds_config)
            self.val_ds = Kitti360(Split.VAL, self.ds_config)
        elif stage == 'test':
            if self.use_v2:
                self.test_ds = Kitti360V2(Split.TEST, self.ds_config)
            else:
                self.test_ds = Kitti360(Split.VAL, self.ds_config)
        else:
            self.full_ds = Kitti360(Split.TEST, self.ds_config)