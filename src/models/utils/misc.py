
import torch

import random
from typing import Dict
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import seaborn as sns

from open3d.visualization.tensorboard_plugin import summary  # noqa: F401
from torch.utils.tensorboard.writer import SummaryWriter

def vis_ptcloud_with_masks_prompts(writer: SummaryWriter, step:int, scene:NDArray[np.float32], prompts: NDArray[np.float32], prompt_label: NDArray[np.int32], pred:NDArray[np.int32], pred_logit:NDArray[np.float64]) -> None:
    print('in viz')
    print(scene.shape)
    print(pred_logit.shape)
    print(prompts.shape)
    print(prompt_label.shape)
    n,_ = scene.shape

    pred_points = scene[pred == 1]
    # print(pred_points.shape)
    pred_points_colors = np.broadcast_to(np.array([1.0, 0.0, 0.0])[None, :], (pred_points.shape[0], 3))
    # logit_points_color = np.broadcast_to(np.array([1.0, 0.0, 0.0])[None, :], (n, 3))
    # if not pred_points.size:
    #     print('no pred')
    #     pred_points = scene.copy()
    log_points = scene.copy()
    pred_neg = 1 - pred_logit
    logit_points_color = np.zeros((n, 3))
    logit_points_color[:,0] = pred_logit
    logit_points_color[:,1] = pred_neg
    print(logit_points_color.shape)


    prompt_pos_color = [0.0, 1.0, 0.0]
    prompt_neg_color = [1.0, 1.0, 0.0]

    prompt_pos_points = []
    prompt_neg_points = []
    for i, label in enumerate(prompt_label):
        if label == 0:
            print('bg point')
            prompt_neg_points.append(prompts[i])
        else:
            print('foreg point')
            prompt_pos_points.append(prompts[i])

    if len(prompt_neg_points) == 0:
        print('no bg point')
        prompt_neg_points.append(np.zeros((1,3)))

    pos_prompts = np.stack(prompt_pos_points)
    print(pos_prompts.shape)
    neg_prompts = np.stack(prompt_neg_points)
    print(neg_prompts.shape)

    output = {
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
        },
        'logits': {
            'vertex_positions': log_points,
            'vertex_colors': logit_points_color,
        }
    }

    for key, value in output.items():
        print('in items')
        print(key)
        writer.add_3d(key, value, step=step) # type: ignore

def visualize_attention(attention_weights: NDArray, attention_type: str, layer=None, p_dir: str=''):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)

    title = f"Attention Weights: {attention_type}"
    if layer is not None:
        title += f" (Layer {layer})"
    
    random_n = random.randint(0,10)
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.savefig(f'{p_dir}/{attention_type}_{layer}_b{random_n}.jpg')

def visualize_all_attention_types(all_attention_weights: list[Dict[str, torch.Tensor]], p_dir:str=''):
    print(p_dir)
    for layer, layer_attention in enumerate(all_attention_weights):
        # Visualize self-attention
        if 'token_self_attn' in layer_attention:
            self_attn = layer_attention['token_self_attn'].mean(dim=1).squeeze().detach().cpu().numpy()
            visualize_attention(self_attn, 'token_self_attn', layer)
        
        # Visualize cross-attention from tokens to pcd
        if 'cross_token_to_pcd_attn' in layer_attention:
            cross_attn_t2i = layer_attention['cross_token_to_pcd_attn'].mean(dim=1).squeeze().detach().cpu().numpy()
            visualize_attention(cross_attn_t2i, 'cross_token_to_pcd_attn', layer, p_dir)
        
        # Visualize cross-attention from pcd to tokens
        if 'cross_pcd_to_token_attn' in layer_attention:
            cross_attn_i2t = layer_attention['cross_pcd_to_token_attn'].mean(dim=1).squeeze().detach().cpu().numpy()
            visualize_attention(cross_attn_i2t, 'cross_pcd_to_token_attn', layer, p_dir)
    
    # Visualize final attention
    if 'final_token_to_pcd_attn' in all_attention_weights[-1]:
        final_attn = all_attention_weights[-1]['final_token_to_pcd_attn'].mean(dim=1).squeeze().detach().cpu().numpy()
        visualize_attention(final_attn, 'final_token_to_pcd_attn', p_dir)