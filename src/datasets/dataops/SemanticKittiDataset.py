from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm


from ..utils.preprocess import voxelize_fps, voxelize_only
from .base import PointCloudBaseDataset, Split, DSConfig


__all__ = ['KittiSeq', 'KittiConfig', 'SemanticKitti']

class KittiSeq(Enum):
    S1 = '01'
    S2 = '02'
    S3 = '03'
    S4 = '04'
    S5 = '05'
    S6 = '06'
    S7 = '07'
    S8 = '08'
    S9 = '09'
    S10 = '10'
    S11 = '11'
    S12 = '12'
    S13 = '13'
    S14 = '14'
    S15 = '15'
    S16 = '16'
    S17 = '17'
    S18 = '18'
    S19 = '19'
    S20 = '20'
    S21 = '21'
    
    @classmethod
    def to_list(cls) -> list[str]:
        return [e.value for e in cls]

    @classmethod
    def to_list_except(cls, sequences: list[KittiSeq]) -> list[str]:
        return [e.value for e in cls if e is not sequences]
    
    @classmethod
    def default_val(cls) -> list[KittiSeq]:
        return [cls.S8]

    @classmethod
    def default_test(cls) -> list[KittiSeq]:
        return [cls.S11, cls.S12, cls.S13, cls.S14, cls.S15, cls.S16, cls.S17, cls.S18, cls.S19, cls.S20, cls.S21]

@dataclass(frozen=True)
class KittiConfig(DSConfig):
    quantization: float
    num_sampling_points: int
    objects_only: bool
    val_seq: list[KittiSeq]
    train_seq: list[KittiSeq]

class SemanticKitti(PointCloudBaseDataset[KittiConfig]):
    IGNORE_CLASS_ID = 0

    def __init__(self, split: Split, config: KittiConfig) -> None:
        super().__init__(split, config)
        
    def setup_files(self, split: Split, config: KittiConfig) -> None:
        root = Path(config.root_path)

        if split is Split.TRAIN:
            sequences = [s.value for s in config.train_seq]
        elif split is Split.VAL:
            sequences = [s.value for s in config.val_seq]
        elif split is Split.TEST:
            sequences = [s.value for s in KittiSeq.default_test()]
        else:
            sequences = KittiSeq.to_list()

        pcd_files: list[str] = []
        for seq in sequences:
            pcd_files.extend([str(pcd_path) for pcd_path in Path(f'{root}/{seq}/velodyne').iterdir()])

        self._pcd_files = pcd_files
        self._num_points = config.num_sampling_points
        self._quantization = config.quantization
        self._objects_only = config.objects_only

    @property
    def pcd_files(self) -> list[str]:
        return self._pcd_files

    @property 
    def objects_only(self) -> bool:
        return self._objects_only

    @property
    def num_points(self) -> int:
        return self._num_points

    @property
    def quantization(self) -> float:
        return self._quantization

    def get_mask_ds(self) -> list[tuple[NDArray[np.float32], NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        print('> Loading bin files....')

        assert self.pcd_files is not None

        for pcd_file in tqdm(self.pcd_files, ascii=True):
            label_file = pcd_file.replace("velodyne", "labels").replace(".bin", ".label")

            pcd_data = np.fromfile(pcd_file, dtype=np.float32).reshape(-1,4)
            pcd = pcd_data[:,:3]

            instances= np.zeros(pcd_data.shape[0]).astype(np.uint32)
            if Path(label_file).exists():
                with open(label_file, 'rb') as f:
                    instances = np.fromfile(f, dtype=np.uint32).reshape(-1,1)

            if self.voxelize_only:
                pcd, instances = voxelize_only(pcd.astype(np.float32), instances.astype(np.int64), self.quantization)
            else:
                pcd, instances = voxelize_fps(pcd.astype(np.float32), instances.astype(np.int64), self.num_points, self.quantization)

            print(pcd.shape)
            unique_labels = np.unique(instances)
            # unique_labels = np.delete(unique_labels, np.where(unique_labels==self.IGNORE_CLASS_ID))
            if self.objects_only:
                unique_labels = np.array([i for i in unique_labels if self.get_obj_classes().get(i, False)])

            masks = np.zeros((pcd.shape[0], unique_labels.shape[0]), dtype=np.int8)
            for i, label in enumerate(unique_labels):
                indices = np.where(instances==label)
                masks[indices, i] = 1
                # points_in_mask = pcd[indices]
                # if points_in_mask.shape[0] > self.MIN_NUMBER_OF_POINTS_PER_MASK:
                masks_ds.append((pcd, masks[:,i].transpose()))

        return masks_ds
    
    @staticmethod
    def get_learning_map() -> dict[int,int]:
        return {
            0 : 0 ,    # "unlabeled"
            1 : 0 ,    # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 1 ,    # "car"
            11: 2 ,    # "bicycle"
            13: 5 ,    # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 3 ,    # "motorcycle"
            16: 5 ,    # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 4 ,    # "truck"
            20: 5 ,    # "other-vehicle"
            30: 6 ,    # "person"
            31: 7 ,    # "bicyclist"
            32: 8 ,    # "motorcyclist"
            40: 9 ,    # "road"
            44: 10,    # "parking"
            48: 11,    # "sidewalk"
            49: 12,    # "other-ground"
            50: 13,    # "building"
            51: 14,    # "fence"
            52: 0 ,    # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 9 ,    # "lane-marking" to "road" ---------------------------------mapped
            70: 15,    # "vegetation"
            71: 16,    # "trunk"
            72: 17,    # "terrain"
            80: 18,    # "pole"
            81: 19,    # "traffic-sign"
            99: 0 ,    # "other-object" to "unlabeled" ----------------------------mapped
            252: 1,    # "moving-car" to "car" ------------------------------------mapped
            253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 6,    # "moving-person" to "person" ------------------------------mapped
            255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 4,    # "moving-truck" to "truck" --------------------------------mapped
            259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }

    @staticmethod
    def get_obj_classes():
        return {
            0: False  ,    # "unlabeled", and others ignored
            1: True   ,  # "car"
            2: True   ,  # "bicycle"
            3: True   ,  # "motorcycle"
            4: True   ,  # "truck"
            5: True   ,  # "other-vehicle"
            6: True   ,  # "person"
            7: True   ,  # "bicyclist"
            8: True   ,  # "motorcyclist"
            9: False  ,   # "road"
            10: False ,   # "parking"
            11: False ,   # "sidewalk"
            12: False ,   # "other-ground"
            13: False ,   # "building"
            14: False ,   # "fence"
            15: False ,   # "vegetation"
            16: False ,   # "trunk"
            17: False ,   # "terrain"
            18: False ,   # "pole"
            19: False ,   # "traffic-sign" 
        }