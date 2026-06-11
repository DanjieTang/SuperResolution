# SuperResolution

A super resolution model that upsamples 64x64 images to 256x256, trained on ImageNet. The architecture is a stack of residual blocks with a self-attention block (with 2D sinusoidal positional encoding) and learned upsampling.

## Pipeline

- `model.py` — `ResBlock`, `SelfAttentionBlock`, and the `SuperResolution` network.
- `dataloader.py` — builds train/val dataloaders from an ImageFolder-style dataset.
- `train.py` — training entry point with CLI arguments and optional W&B logging.
- `run_sweep.py` / `sweep_config.yaml` — grid search over hyperparameters with resume support.
- `deprecated/` — the original notebooks this pipeline was refactored from.

## Training

```bash
python train.py
```

Defaults train on the `ImagenetHighResolution` folder using MPS if available, falling back to CUDA then CPU. Mixed precision is used automatically on CUDA.

Common overrides:

```bash
python train.py --dataset_path /path/to/imagenet --device cuda:0 --batch_size 8 --epochs 1
```

Enable W&B logging by passing both `--project` and `--entity`.

## Hyperparameter sweep

```bash
python run_sweep.py
```

Runs `train.py` over every combination of the list-valued keys in `sweep_config.yaml`. Progress is checkpointed to `sweep_progress.json`, so an interrupted sweep resumes where it left off; use `--force-rerun` to redo completed runs and `--dry-run` to preview the commands.
