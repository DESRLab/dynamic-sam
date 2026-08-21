from __future__ import annotations

import torch

from src.models import DynamicSAM, MaskDecoderParams, PcdEncoderParams, PromptEncoderParams

__all__ = ['load_dynamic_sam']

# The three DynamicSAM sub-modules a Lightning SegmentPcdTask checkpoint's
# flat state_dict holds under a `network.<name>.` (or, for torch.compile'd
# checkpoints, `network._orig_mod.<name>.`) key prefix.
_SUBMODULES = ('pcd_encoder', 'mask_decoder', 'prompt_encoder')


def _build_model(group_size: int, num_group: int) -> DynamicSAM:
    pcd_encoder_params = PcdEncoderParams(group_size=group_size, num_group=num_group)
    prompt_encoder_params = PromptEncoderParams(embedding_dim=pcd_encoder_params.trans_dim)
    mask_decoder_params = MaskDecoderParams(trans_dim=pcd_encoder_params.trans_dim)
    return DynamicSAM(pcd_encoder_params, prompt_encoder_params, mask_decoder_params)


def _move_to_device(model: DynamicSAM, device: int) -> None:
    model.pcd_encoder.to(f'cuda:{device}')
    model.mask_decoder.to(f'cuda:{device}')
    model.prompt_encoder.to(f'cuda:{device}')
    model.to(f'cuda:{device}')


def _extract_submodule_state_dicts(state_dict: dict, optimized: bool) -> dict[str, dict[str, torch.Tensor]]:
    keyword = '._orig_mod' if optimized else ''
    result = {}
    for name in _SUBMODULES:
        prefix = f"network{keyword}.{name}."
        result[name] = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return result


def _load_state_dicts(model: DynamicSAM, state_dicts: dict[str, dict[str, torch.Tensor]]) -> None:
    model.load_modules_state_dict(state_dicts['pcd_encoder'], state_dicts['prompt_encoder'], state_dicts['mask_decoder'])


def load_from_lightning_checkpoint(path: str, device: int, group_size: int, num_group: int, optimized: bool = True) -> DynamicSAM:
    ck = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
    model = _build_model(group_size, num_group)
    _load_state_dicts(model, _extract_submodule_state_dicts(ck["state_dict"], optimized))
    _move_to_device(model, device)
    model.eval()
    return model


def load_from_safetensors(path: str, device: int) -> DynamicSAM:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(path, framework='pt') as f:
        metadata = f.metadata() or {}
    group_size = int(metadata.get('group_size', 32))
    num_group = int(metadata.get('num_group', 128))

    flat = load_file(path, device=f'cuda:{device}')
    state_dicts: dict[str, dict[str, torch.Tensor]] = {name: {} for name in _SUBMODULES}
    for key, tensor in flat.items():
        name, sub_key = key.split('.', 1)
        state_dicts[name][sub_key] = tensor

    # prompt_encoder.pos_embedding is a shared reference to pcd_encoder.pos_embed
    # (the "same learned MLP encodes both sub-cloud centers and user clicks"
    # design), so a safetensors export is expected to have skipped saving it a
    # second time under its own name (safetensors refuses to serialize the
    # same underlying storage under two different tensor names). Reconstruct
    # it here so prompt_encoder's state_dict is still complete.
    if not any(k.startswith('pos_embedding.') for k in state_dicts['prompt_encoder']):
        for k, v in state_dicts['pcd_encoder'].items():
            if k.startswith('pos_embed.'):
                state_dicts['prompt_encoder']['pos_embedding.' + k[len('pos_embed.'):]] = v

    model = _build_model(group_size, num_group)
    _load_state_dicts(model, state_dicts)
    _move_to_device(model, device)
    model.eval()
    return model


def load_dynamic_sam(path: str, device: int, group_size: int = 32, num_group: int = 128, optimized: bool = True) -> DynamicSAM:
    """Loads a DynamicSAM ready for inference from either a full Lightning
    checkpoint (.ckpt) or an inference-only safetensors file (flat
    `<submodule>.<key>` tensor names, `group_size`/`num_group` string metadata).

    group_size/num_group only matter for the .ckpt path; a safetensors file
    carries them in its own metadata.
    """
    if path.endswith('.safetensors'):
        return load_from_safetensors(path, device)
    return load_from_lightning_checkpoint(path, device, group_size, num_group, optimized)
