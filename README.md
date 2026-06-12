# Outpainting

A latent flow matching model that expands images beyond their borders (outpainting), trained on ImageNet. A frozen pretrained VAE compresses the canvas into latent space; a DiT (diffusion transformer) conditioned on the known-region latent and mask learns the flow matching velocity field. The known region is randomly sized and placed during training, supporting expansion in any direction up to 2x per dimension (e.g. 512x512 -> 1024x1024).

## Pipeline

- `model.py` — `TimestepEmbedding`, `DiTBlock` (adaLN-zero), and the `DiT` network.
- `dataloader.py` — canvas dataloaders plus random known-region mask sampling.
- `train.py` — flow matching training entry point with CLI arguments and optional W&B logging.
- `run_sweep.py` / `sweep_config.yaml` — grid search over hyperparameters with resume support.
- `super_resolution/` — the previous project: a 4x super resolution model (64x64 -> 256x256).

## Training

```bash
python train.py
```

Defaults train on the `ImagenetHighResolution` folder using CUDA if available, falling back to MPS then CPU. The pretrained VAE (`stabilityai/sd-vae-ft-ema`) is downloaded from Hugging Face on first run. bf16 autocast is used automatically on CUDA.

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

To sweep the old super resolution project instead:

```bash
python run_sweep.py --config super_resolution/sweep_config.yaml --train-script super_resolution/train.py --state-file super_resolution/sweep_progress.json
```
