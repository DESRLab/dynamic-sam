from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset
from typing import TypedDict, TypeVar, Generic
from typing_extensions import TypeAlias

from ..utils.preprocess import augment

Tensor: TypeAlias = torch.Tensor

__all__ = ['DSOutput', 'DSConfig', 'PointCloudBaseDataset', 'InferenceInput']

class DSOutput(TypedDict):
    prompts_labels: Tensor
    point_prompt: Tensor
    mask: Tensor
    pcd: Tensor
    test_class_labels: list[str]
    test_class_labels_count: dict[str, int]

class TestLabels(TypedDict):
    test_class_labels: list[str]
    test_class_labels_count: dict[str, int]
class InferenceInput(TypedDict):
    prompts_labels: Tensor
    point_prompt: Tensor
    pcd: Tensor

class Split(Enum):
    TRAIN = 'Train'
    VAL = 'Val'
    TEST = 'Test'
    ALL = 'All'

@dataclass(frozen=True)
class DSConfig():
    root_path: str
    voxelize_only: bool


CONFIG = TypeVar('CONFIG', bound=DSConfig)
class PointCloudBaseDataset(Dataset[DSOutput], Generic[CONFIG]):
    MIN_NUMBER_OF_POINTS_PER_MASK = 5
    NAME = ''
    def __init__(self, split: Split, config: CONFIG) -> None:
        super().__init__()
        self._split = split
        self._voxelize_only = config.voxelize_only
        self.setup_files(split, config)
        self._masks_ds = self.get_mask_ds()
        self.dataset_size = len(self.masks_ds)

    @property
    def masks_ds(self) -> list[tuple[NDArray[np.float32], NDArray[np.int8]]]:
        return self._masks_ds

    @property
    def split(self) -> Split:
        return self._split
    
    @property
    def voxelize_only(self) -> bool:
        return self._voxelize_only

    @abstractmethod
    def setup_files(self, split: Split, config: CONFIG) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_mask_ds(self) -> list[tuple[NDArray[np.float32], NDArray[np.int8]]]:
        raise NotImplementedError

    def get_test_class_labels(self) -> TestLabels:
        return TestLabels(test_class_labels=[], test_class_labels_count={})

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index:int) -> DSOutput:
        pcd, mask = self.masks_ds[index]
        if self.split is Split.TRAIN:
            pcd = augment(pcd)

        test_lable_info = self.get_test_class_labels()
        if self.split is Split.TRAIN:
            bg_indices = np.nonzero(mask == 0)[0]
            bg_random_index = np.random.choice(bg_indices)
            bg_point = pcd[bg_random_index:bg_random_index+1,:]
            bg_label = np.array([0])

            fg_indices = np.nonzero(mask==1)[0]
            fg_random_index = np.random.choice(fg_indices)
            fg_point = pcd[fg_random_index:fg_random_index+1,:]
            fg_label = np.array([1])

            return DSOutput(
                pcd=torch.from_numpy(pcd.astype(np.float32)).float(),
                point_prompt=torch.from_numpy(np.concatenate((fg_point, bg_point), axis=0)).to(torch.bfloat16),
                prompts_labels=torch.from_numpy(np.concatenate((fg_label, bg_label), axis=0)).to(torch.bfloat16),
                mask= torch.from_numpy(mask[None,:]),
                test_class_labels=test_lable_info['test_class_labels'],
                test_class_labels_count=test_lable_info['test_class_labels_count']
            )

        else:
            fg_point = np.mean(pcd[mask==1], axis=0)
            fg_label = np.array([1])

            distances = np.linalg.norm(pcd - fg_point, axis=1)
            sorted_indices = np.argsort(distances)
            closest_point = pcd[sorted_indices[0]][None,:]

            return DSOutput(
                pcd=torch.from_numpy(pcd.astype(np.float32)).float(),
                point_prompt=torch.from_numpy(closest_point).float(),
                prompts_labels=torch.from_numpy(fg_label),
                mask= torch.from_numpy(mask[None,:]),
                test_class_labels=test_lable_info['test_class_labels'],
                test_class_labels_count=test_lable_info['test_class_labels_count']
            )
