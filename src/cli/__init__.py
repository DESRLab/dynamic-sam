from __future__ import annotations

import click

__all__ = ['cli']


@click.group()
def cli():
    """dynamic-sam command-line interface."""


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=str))
@click.option('--group-size', 'group_size', type=int, required=False, default=32,
              help='Number of points per group the point cloud encoder was trained with.')
@click.option('--num-group', 'num_group', type=int, required=False, default=128,
              help='Number of groups the point cloud encoder was trained with.')
@click.option('--trained-optimized/--not-trained-optimized', 'optimized', default=True,
              help='Whether the checkpoint was trained with torch.compile (changes the state_dict key prefix).')
@click.option('--device', 'device', type=int, required=False, default=0, help='CUDA device index to load the model onto.')
@click.option('--port', 'port', type=int, required=False, default=8000)
@click.option('--workers', 'workers', type=int, required=False, default=1)
@click.option('--max-users', 'max_users', type=int, required=False, default=1,
              help='Maximum number of distinct users whose encoded point clouds are cached '
                   'concurrently. Beyond this, the least-recently-encoded-for user is evicted '
                   'entirely (all of its cached frames) to make room for a new one.')
@click.option('--max-frames-per-user', 'max_frames_per_user', type=int, required=False, default=5,
              help='Maximum number of distinct point clouds (frames) cached per user. Beyond '
                   'this, that user\'s least-recently-encoded frame is evicted to make room '
                   'for a new one.')
def serve(
    model_path: str,
    group_size: int,
    num_group: int,
    optimized: bool,
    device: int,
    port: int,
    workers: int,
    max_users: int,
    max_frames_per_user: int,
):
    """Serve a DynamicSAM checkpoint at MODEL_PATH behind a FastAPI HTTP API.

    Requires the `inference` extra: pip install "dynamic-sam[inference]"
    """
    # imported here (not at module level) so that `dynamic-sam` itself, and
    # the train/test commands, don't require fastapi/uvicorn/pydantic to be
    # installed -- only running `serve` does.
    import torch
    import uvicorn

    from src.models import DynamicSAM, MaskDecoderParams, PcdEncoderParams, PromptEncoderParams
    from src.serve import create_fast_api

    print("Loading model...")
    ck = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))

    keyword = ''
    if optimized:
        keyword = '._orig_mod'

    pcd_encoder_weights = {k[len(f"network{keyword}.pcd_encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith(f"network{keyword}.pcd_encoder.")}
    mask_decoder_weights = {k[len(f"network{keyword}.mask_decoder."):]: v for k, v in ck["state_dict"].items() if k.startswith(f"network{keyword}.mask_decoder.")}
    prompt_encoder_weights = {k[len(f"network{keyword}.prompt_encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith(f"network{keyword}.prompt_encoder.")}

    pcd_encoder_params = PcdEncoderParams(group_size=group_size, num_group=num_group)
    prompt_encoder_params = PromptEncoderParams(embedding_dim=pcd_encoder_params.trans_dim)
    mask_decoder_params = MaskDecoderParams(trans_dim=pcd_encoder_params.trans_dim)
    model = DynamicSAM(pcd_encoder_params, prompt_encoder_params, mask_decoder_params)
    model.pcd_encoder.to(f'cuda:{device}')
    model.mask_decoder.to(f'cuda:{device}')
    model.prompt_encoder.to(f'cuda:{device}')
    model.to(f'cuda:{device}')
    model.load_modules_state_dict(pcd_encoder_weights, prompt_encoder_weights, mask_decoder_weights)
    model.eval()

    print(f"Serving with max_users={max_users}, max_frames_per_user={max_frames_per_user}")
    app = create_fast_api(model, device=f'cuda:{device}', max_users=max_users, max_frames_per_user=max_frames_per_user)
    uvicorn.run(app, host="127.0.0.1", port=port, reload=False, log_level="debug", workers=workers)


def _run_lightning_subcommand(subcommand: str, config: str, extra_args: tuple[str, ...]):
    # imported here (not at module level) so that `dynamic-sam` itself, and
    # the serve command, don't require lightning/jsonargparse to be
    # installed -- only running train/test does.
    from src.task.cli import run_lightning_cli

    run_lightning_cli(args=[subcommand, '-c', config, *extra_args])


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option('-c', '--config', 'config', type=click.Path(exists=True, dir_okay=False, path_type=str), required=True,
              help='Path to the LightningCLI training config YAML (see configs/*.yaml).')
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def train(config: str, extra_args: tuple[str, ...]):
    """Train a DynamicSAM model. Equivalent to LightningCLI's `fit -c CONFIG`.

    Any EXTRA_ARGS are forwarded to LightningCLI verbatim (e.g. --ckpt_path
    to resume, or --trainer.max_epochs=10 to override the config).

    Requires the `training` extra: pip install "dynamic-sam[training]"
    """
    _run_lightning_subcommand('fit', config, extra_args)


@cli.command(context_settings=dict(ignore_unknown_options=True))
@click.option('-c', '--config', 'config', type=click.Path(exists=True, dir_okay=False, path_type=str), required=True,
              help='Path to the LightningCLI config YAML (see configs/*.yaml).')
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def test(config: str, extra_args: tuple[str, ...]):
    """Evaluate a DynamicSAM checkpoint. Equivalent to LightningCLI's `test -c CONFIG`.

    Any EXTRA_ARGS are forwarded to LightningCLI verbatim, most commonly
    --ckpt_path /path/to/checkpoint.ckpt.

    Requires the `training` extra: pip install "dynamic-sam[training]"
    """
    _run_lightning_subcommand('test', config, extra_args)


if __name__ == '__main__':
    cli()
