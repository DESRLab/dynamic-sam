from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Generic, TypeVar
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as pl

from ..dataops import DSConfig, DSOutput

DSCONFIG = TypeVar('DSCONFIG', bound=DSConfig) 

class BaseDataModule(pl.LightningDataModule, Generic[DSCONFIG]):
    """
    Parameters
    ----------
    dataset_config : DSCONFIG
        The dataset parameters to run.
    batch_size : int
        The number of batches to
        train/validate the data before terminating.
        Defaults to 1.
    num_workers: int | None
        The number of workers to load the dataset.
        Defaults to 1 if not provided.
    """
    def __init__(self, dataset_config: DSCONFIG, batch_size: int = 1, use_v2: bool = False, num_workers: Optional[int] = 1) -> None:
        super().__init__()

        self._ds_config = dataset_config
        self._batch_size = batch_size
        self._use_v2 = use_v2
        self._num_workers = 1 if num_workers is None else num_workers
        self.train_ds: Dataset
        self.test_ds: Dataset
        self.val_ds: Dataset
        self.full_ds: Dataset

    @property
    def ds_config(self) -> DSCONFIG:
        return self._ds_config
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def num_workers(self) -> int:
        return self._num_workers
    
    @property
    def use_v2(self) -> bool:
        return self._use_v2

    @abstractmethod
    def setup(self, stage: str):
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[DSOutput]:
        return DataLoader(self.train_ds, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=True, pin_memory=True)

    def val_dataloader(self) -> DataLoader[DSOutput]:
        return DataLoader(self.val_ds, num_workers=self.num_workers, batch_size=self.batch_size, shuffle=False, pin_memory=True)

    def test_dataloader(self) -> DataLoader[DSOutput]:
        return DataLoader(self.test_ds, num_workers=self.num_workers, shuffle=False, batch_size=1, pin_memory=True)

    def predict_dataloader(self) -> DataLoader[DSOutput]:
        return DataLoader(self.full_ds, num_workers=self.num_workers, shuffle=False, batch_size=self.batch_size, pin_memory=True)
