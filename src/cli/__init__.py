from __future__ import annotations

import click

__all__ = ['cli']


@click.group()
def cli():
    """dynamic-sam command-line interface."""


@cli.command()
@click.argument('model_path', required=False, default=None,
                 type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=str))
@click.option('--hf-repo-id', 'hf_repo_id', type=str, required=False, default=None,
              help='Hugging Face Hub repo to download the model from when MODEL_PATH is not '
                   'given (e.g. your-username/dynamic-sam-checkpoint). Downloads go through the '
                   'local HF cache (HF_HOME), so a file already cached there is reused rather '
                   'than re-downloaded. Uses your `hf auth login` token automatically for '
                   'private repos.')
@click.option('--hf-filename', 'hf_filename', type=str, required=False, default='model.safetensors',
              help='Filename within --hf-repo-id to download.')
@click.option('--hf-revision', 'hf_revision', type=str, required=False, default=None,
              help='Optional branch/tag/commit to download from --hf-repo-id.')
@click.option('--group-size', 'group_size', type=int, required=False, default=32,
              help='Number of points per group the point cloud encoder was trained with. '
                   'Ignored for a .safetensors model, which carries its own group_size.')
@click.option('--num-group', 'num_group', type=int, required=False, default=128,
              help='Number of groups the point cloud encoder was trained with. '
                   'Ignored for a .safetensors model, which carries its own num_group.')
@click.option('--trained-optimized/--not-trained-optimized', 'optimized', default=True,
              help='Whether the checkpoint was trained with torch.compile (changes the state_dict '
                   'key prefix). Ignored for a .safetensors model.')
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
    model_path: str | None,
    hf_repo_id: str | None,
    hf_filename: str,
    hf_revision: str | None,
    group_size: int,
    num_group: int,
    optimized: bool,
    device: int,
    port: int,
    workers: int,
    max_users: int,
    max_frames_per_user: int,
):
    """Serve a DynamicSAM checkpoint behind a FastAPI HTTP API.

    MODEL_PATH may be a full Lightning checkpoint (.ckpt) or an inference-only
    safetensors file. If omitted, pass --hf-repo-id to download the model
    from Hugging Face Hub instead (through the local HF cache).

    Requires the `inference` extra: pip install "dynamic-sam[inference]"
    """
    # imported here (not at module level) so that `dynamic-sam` itself, and
    # the train/test commands, don't require fastapi/uvicorn/pydantic to be
    # installed -- only running `serve` does.
    import uvicorn

    from src.serve import create_fast_api
    from src.serve.checkpoint import load_dynamic_sam

    if model_path is None:
        if hf_repo_id is None:
            raise click.UsageError(
                'Provide MODEL_PATH, or --hf-repo-id (and optionally --hf-filename/--hf-revision) '
                'to load from Hugging Face Hub instead.'
            )
        from huggingface_hub import hf_hub_download

        print(f"Fetching {hf_filename} from {hf_repo_id} (Hugging Face cache: HF_HOME)...")
        model_path = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, revision=hf_revision)

    print("Loading model...")
    model = load_dynamic_sam(model_path, device, group_size=group_size, num_group=num_group, optimized=optimized)

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
