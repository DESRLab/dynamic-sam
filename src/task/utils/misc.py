from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import random

from src.kernel.fps import furthest_point_sample
from src.kernel.gather import gather_operation

from torch import Tensor
from typing import Dict, List

def fps(data: Tensor, number: Tensor):
    '''
        data B N 3
        number int
    '''
    fps_idx = furthest_point_sample(data, number)
    fps_data = gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous() #type: ignore
    return fps_data

def random_vis_ptcloud_with_masks_prompts(ptcloud:NDArray[np.float64], points: NDArray[np.float64], labels: NDArray[np.int32], 
                                   gt_mask:NDArray[np.int32], pred_mask:NDArray[np.int32], pred_logits:NDArray[np.float64]
                                   ) -> Dict[str,Dict[str,NDArray]]:
    b, n,_ = ptcloud.shape
    gt_mask = gt_mask.transpose(0,2,1).squeeze(2) # [b,1,N] => [b,N]
    pred_mask = pred_mask.transpose(0,2,1).squeeze(2) # [b,1,N] => [b,N] 
    pred_logits = pred_logits.transpose(0,2,1).squeeze(2)

    rand_b = random.randint(0, b -1)

    scene, gt, pred, pred_logit = ptcloud[rand_b] ,gt_mask[rand_b], pred_mask[rand_b], pred_logits[rand_b]
    prompts = points[rand_b] #[N,3]
    prompt_label = labels[rand_b] #[N,1]
    
    return vis_ptcloud_with_masks_prompts(scene, prompts, prompt_label,gt, pred, pred_logit)

def vis_ptcloud_with_masks_prompts_all(ptcloud:NDArray[np.float64], points: NDArray[np.float64], labels: NDArray[np.int32], 
                                   gt_mask:NDArray[np.int32], pred_mask:NDArray[np.int32], pred_logits:NDArray[np.float64]) -> List[Dict[str,Dict[str,NDArray]]]:
    b_size, _,_ = ptcloud.shape
    gt_mask = gt_mask.transpose(0,2,1).squeeze(2) # [b,1,N] => [b,N]
    pred_mask = pred_mask.transpose(0,2,1).squeeze(2) # [b,1,N] => [b,N] 
    pred_logits = pred_logits.transpose(0,2,1).squeeze(2)

    viz_list = []
    for bidx in range(b_size):
        viz_list.append(vis_ptcloud_with_masks_prompts(ptcloud[bidx], prompts=points[bidx], prompt_label=labels[bidx],gt=gt_mask[bidx], pred=pred_mask[bidx], pred_logit=pred_logits[bidx]))

    return viz_list

def vis_ptcloud_with_masks_prompts(scene:NDArray[np.float64], prompts: NDArray[np.float64], prompt_label: NDArray[np.int32], 
                                   gt:NDArray[np.int32], pred:NDArray[np.int32], pred_logit:NDArray[np.float64]) -> Dict[str,Dict[str,NDArray]]:
    n,_ = scene.shape

    gt_points = scene[gt == 1]
    pred_points = scene[pred == 1]

    pred_points_colors = np.broadcast_to(np.array([1.0, 0.0, 0.0])[None, :], (pred_points.shape[0], 3))
    if not pred_points.size:
        pred_points = scene.copy()

        pred_neg = 1 - pred_logit
        pred_points_colors = np.zeros((n, 3))
        pred_points_colors[:,0] = pred_logit
        pred_points_colors[:,1] = pred_neg


    prompt_pos_color = [0.0, 1.0, 0.0]
    prompt_neg_color = [1.0, 1.0, 0.0]

    prompt_pos_points = []
    prompt_neg_points = []
    for i, label in enumerate(prompt_label):
        if label == 0:
            prompt_neg_points.append(prompts[i])
        else:
            prompt_pos_points.append(prompts[i])

    if len(prompt_neg_points) == 0:
        prompt_neg_points.append(np.zeros((1,3)))

    pos_prompts = np.stack(prompt_pos_points)
    neg_prompts = np.stack(prompt_neg_points)

    output = {
        'gt_points': {
            'vertex_positions': gt_points,
            'vertex_colors': np.broadcast_to(np.array([0.0, 0.0, 1.0])[None, :], (gt_points.shape[0], 3)),
        },
        'scene': {
            'vertex_positions': scene,
            'vertex_colors': np.broadcast_to(np.array([0.5, 0.5, 0.5])[None, :], (scene.shape[0], 3)),
        },
        'pred_points': {
            'vertex_positions': pred_points,
            'vertex_colors': pred_points_colors,
        },
        'prompt_pos': {
            'vertex_positions': pos_prompts,
            'vertex_colors': np.broadcast_to(np.array(prompt_pos_color)[None, :], (pos_prompts.shape[0], 3))
        },
        'prompt_neg': {
            'vertex_positions': neg_prompts,
            'vertex_colors': np.broadcast_to(np.array(prompt_neg_color)[None, :], (neg_prompts.shape[0], 3))
        }
    }

    return output

def get_dims_with_exclusion(dim, exclude=None) -> list[int]:
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims
