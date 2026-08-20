from __future__ import annotations

from ..dataops import MixModelNet, DSConfig, Split

from .base import BaseDataModule

__all__ = ['MixModelNetDataModule']

class MixModelNetDataModule(BaseDataModule[DSConfig]):
    def __init__(self, dataset_config: DSConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_ds = MixModelNet(Split.TRAIN,self.ds_config)
            self.val_ds = MixModelNet(Split.VAL, self.ds_config)
        elif stage == 'test':
            self.test_ds = MixModelNet(Split.TEST, self.ds_config)
        else:
            self.full_ds = MixModelNet(Split.TEST, self.ds_config)