from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from .base import PointCloudBaseDataset, Split, DSConfig


__all__ = ['MixModelNet']

class MixModelNet(PointCloudBaseDataset[DSConfig]):
    def __init__(self, split: Split, config: DSConfig) -> None:
        super().__init__(split, config)
    
    def setup_files(self, split: Split, config: DSConfig) -> None:
        root = config.root_path
        if split is Split.TRAIN:
            scenes_no: NDArray[np.int32] = np.loadtxt(f'{root}/train_scene_nums.txt', dtype=np.int32)
            self._scenes = [f'{root}/train/scene_{scene_n}.npy' for scene_n in scenes_no]
            self._masks = [f'{root}/train/mask_{scene_n}.npy' for scene_n in scenes_no]
        elif split is Split.VAL:
            scenes_no = np.loadtxt(f'{root}/val_scene_nums.txt', dtype=np.int32)
            self._scenes = [f'{root}/train/scene_{scene_n}.npy' for scene_n in scenes_no]
            self._masks = [f'{root}/train/mask_{scene_n}.npy' for scene_n in scenes_no]
        else:
            scenes_path = Path(root).joinpath('test/scenes')
            self._scenes = [str(file) for file in scenes_path.iterdir() if file.is_file()]
            masks_path = Path(root).joinpath('test/masks')
            self._masks = [str(file) for file in masks_path.iterdir() if file.is_file()]

    @property
    def masks(self) -> list[str]:
        return self._masks

    @property
    def scenes(self) -> list[str]:
        return self._scenes

    def get_mask_ds(self) -> list[tuple[NDArray[np.float32],NDArray[np.int8]]]:
        masks_ds: list[tuple[NDArray[np.float32],NDArray[np.int8]]] = []

        for scene_path, mask_path in tqdm(zip(self.scenes, self.masks), ascii=True):
            pcd: NDArray[np.float32] = np.load(scene_path)
            masks: NDArray[np.int8] = np.load(mask_path)

            for num_ins in range(masks.shape[0]):
                masks_ds.append((pcd, masks[num_ins,:]))
                
        return masks_ds