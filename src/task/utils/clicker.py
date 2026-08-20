from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch
from typing import Tuple
from typing_extensions import TypeAlias
from scipy import ndimage

Tensor: TypeAlias = torch.Tensor

__all__ = ['Clicker']

class Click:
    def __init__(self, coords: Tensor, label: Tensor, indx: int):
        self._label = label
        self._coords = coords
        self._index = indx

    @property
    def coords(self) -> Tensor:
        return self._coords

    @property
    def label(self) -> Tensor:
        return self._label
    
    @property
    def index(self) -> int:
        return self._index

    @property
    def is_positive(self) -> bool:
        return torch.sum(self.label).item() == 1

class Clicker:
    def __init__(self, gt_mask: Tensor, init_click:Tuple[Tensor,...], pcd: Tensor, device: torch.device):

        self._gt_mask = gt_mask.squeeze(0)

        self._pcd = pcd

        self.reset_clicks()

        coords, labels = init_click

        self._clicks_list: list[Click] = []

        click = Click(coords, labels, 0)
        self.add_click(click, first_click=True)
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def gt_mask(self) -> Tensor:
        return self._gt_mask

    @property
    def pcd(self) -> Tensor:
        return self._pcd

    @property
    def clicks_list(self) -> list[Click]:
        return self._clicks_list

    def make_next_click(self, pred_mask: Tensor):
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self) -> Tuple[Tensor,...]:
        all_points = [click.coords for click in self.clicks_list]
        all_labels = [click.label for click in self.clicks_list]


        return torch.concat(all_points, dim=0).to(self.device)[None,:], torch.concat(all_labels, dim=0).to(self.device)[None,:]

    def _get_next_click(self, pred_mask: Tensor):
        pred = pred_mask.transpose(1,0).squeeze(1).cpu().numpy()
        gt = self.gt_mask.cpu().numpy() == 1

        new_coords = torch.zeros(size=(1, 3), device=self.device)
        new_label = torch.zeros(size=(1,), device=self.device)

        fn_mask = np.logical_and(gt, np.logical_not(pred)) # type: ignore
        fp_mask = np.logical_and(np.logical_not(gt), pred) # type: ignore

        fn_mask_dt: NDArray[np.float64] = ndimage.distance_transform_edt(fn_mask) # type: ignore
        fp_mask_dt: NDArray[np.float64] = ndimage.distance_transform_edt(fp_mask) # type: ignore

        fn_mask_dt = fn_mask_dt * self._not_clicked_map
        fp_mask_dt = fp_mask_dt * self._not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            index = np.where(fn_mask_dt == fn_max_dist)
            new_coords = self.pcd[index[0][0], :][None, :]

            new_label = torch.ones(size=(1,)).to(self.device)
        else:
            index = np.where(fp_mask_dt == fp_max_dist)
            new_coords = self.pcd[index[0][0], :][None, :]

        return Click(new_coords, new_label, index[0][0])

    def add_click(self, click: Click, first_click: bool = False):
        self._clicks_list.append(click)
        if not first_click:
            self._not_clicked_map[click.index] = False

    def reset_clicks(self):
        self._not_clicked_map = np.ones(shape=self.gt_mask.shape, dtype=np.bool_)

        self._clicks_list = []

    def __len__(self):
        return len(self.clicks_list)



