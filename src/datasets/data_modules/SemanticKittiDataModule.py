from __future__ import annotations

from ..dataops import SemanticKitti, KittiConfig, Split

from .base import BaseDataModule

__all__ = ['SemanticKittiDataModule']

class SemanticKittiDataModule(BaseDataModule[KittiConfig]):
    def __init__(self, dataset_config: KittiConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_ds = SemanticKitti(Split.TRAIN, self.ds_config)
            self.val_ds = SemanticKitti(Split.VAL, self.ds_config)
        elif stage == 'test':
            self.test_ds = SemanticKitti(Split.VAL, self.ds_config)
        else:
            self.full_ds = SemanticKitti(Split.VAL, self.ds_config)