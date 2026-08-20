from __future__ import annotations

from ..dataops import ARKit, ARKitConfig, Split

from .base import BaseDataModule

__all__ = ['ARKitDataModule']

class ARKitDataModule(BaseDataModule[ARKitConfig]):
    def __init__(self, dataset_config: ARKitConfig, batch_size: int = 1, use_v2: bool = False, num_workers: int | None = 1) -> None:
        super().__init__(dataset_config, batch_size, use_v2, num_workers)

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_ds = ARKit(Split.TRAIN, self.ds_config)
            self.val_ds = ARKit(Split.VAL, self.ds_config)
        elif stage == 'test':
            self.test_ds = ARKit(Split.VAL, self.ds_config)
        else:
            self.full_ds = ARKit(Split.VAL, self.ds_config)