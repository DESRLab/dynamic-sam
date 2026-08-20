from __future__ import annotations

from ..dataops import Scannet, ScannetV2, ScannetConfig, Split

from .base import BaseDataModule

__all__ = ['ScannetDataModule']

class ScannetDataModule(BaseDataModule[ScannetConfig]):
    def __init__(self, dataset_config: ScannetConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            if self.use_v2:
                self.train_ds = ScannetV2(Split.TRAIN, self.ds_config)
                self.val_ds = ScannetV2(Split.VAL, self.ds_config)
            else:
                self.train_ds = Scannet(Split.TRAIN, self.ds_config)
                self.val_ds = Scannet(Split.VAL, self.ds_config)
        elif stage == 'test':
            if self.use_v2:
                self.test_ds = ScannetV2(Split.TEST, self.ds_config)
            else:
                self.test_ds = Scannet(Split.VAL, self.ds_config)
        else:
            if self.use_v2:
                self.full_ds = ScannetV2(Split.VAL, self.ds_config)
            else:
                self.full_ds = Scannet(Split.VAL, self.ds_config)