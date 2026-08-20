from __future__ import annotations

from ..dataops import S3DIS, S3DISConfig, Split, S3DISV2

from .base import BaseDataModule

__all__ = ['S3DISDataModule']

class S3DISDataModule(BaseDataModule[S3DISConfig]):
    def __init__(self, dataset_config: S3DISConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            if self.use_v2:
                raise ValueError('Training based on V2 is not supported')
            self.train_ds = S3DIS(Split.TRAIN, self.ds_config)
            self.val_ds = S3DIS(Split.VAL, self.ds_config)
        elif stage == 'test':
            if self.use_v2:
                self.test_ds = S3DISV2(Split.TEST, self.ds_config)
            else:
                self.test_ds = S3DIS(Split.TEST, self.ds_config)
        else:
            self.full_ds = S3DIS(Split.TEST, self.ds_config)