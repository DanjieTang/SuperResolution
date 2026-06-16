# Outpainting

A pixel-space flow matching model that expands images beyond their borders (outpainting), trained from scratch on ImageNet with no pretrained components. A DiT (diffusion transformer) folds each 16x16 pixel tile into one token and, conditioned on the masked image and known-region mask, learns the flow matching velocity field. The known region is randomly sized and placed during training, supporting expansion in any direction up to 2x per dimension (e.g. 512x512 -> 1024x1024).

## Pipeline

- `model.py` — `TimestepEmbedding`, `DiTBlock` (adaLN-zero), and the `DiT` network.
- `dataloader.py` — canvas dataloaders, latent loaders, plus random known-region mask sampling.
- `train.py` — flow matching training entry point with CLI arguments and optional W&B logging.
- `precompute_latents.py` — one-time script that encodes the dataset to VAE latents on disk (latent mode only).
- `latents.py` — shared helpers for saving, loading, and sampling those cached latents.
- `run_sweep.py` / `sweep_config.yaml` — grid search over hyperparameters with resume support.
- `super_resolution/` — the previous project: a 4x super resolution model (64x64 -> 256x256).

## Training

```bash
python train.py
```

Defaults train on the `ImagenetHighResolution` folder using CUDA if available, falling back to MPS then CPU. bf16 autocast is used automatically on CUDA.

Common overrides:

```bash
python train.py --dataset_path /path/to/imagenet --device cuda:0 --batch_size 8 --epochs 1
```

Enable W&B logging by passing both `--project` and `--entity`.

### Optional: latent diffusion

Passing `--use_vae` trains in the latent space of a frozen pretrained VAE (`stabilityai/sd-vae-ft-ema`, downloaded from Hugging Face on first run) instead of pixel space. The patch size defaults to 2 on the 8x-smaller latents, keeping the token count identical. Requires the optional dependency: `uv sync --extra vae`.

Because the VAE is frozen, latents are **precomputed once to disk** rather than re-encoded every step. This is a two-step workflow:

```bash
# 1. Encode the dataset to latents on disk (run once per dataset/canvas/VAE)
python precompute_latents.py --dataset_path ImagenetHighResolution --output latent_cache/latents.pt

# 2. Train, reading latents from disk (no VAE encoding in the loop)
python train.py --use_vae --latent_cache latent_cache/latents.pt
```

`--latent_cache` defaults to `latent_cache/latents.pt`, so if you keep the defaults the second command is just `python train.py --use_vae`. If the cache is missing, training stops and prints the exact precompute command to run.

> **Why two steps:** encoding the dataset is the expensive part of latent training. Doing it every step (twice per step, on the full-resolution image) made `--use_vae` roughly 7.5x slower than pixel mode. Precomputing it once brings latent training back to roughly pixel-mode speed, and the cache is reused across epochs and runs.

**Latent-space masking.** In latent mode the known region is formed by masking the cached latent directly (`latent * mask`), rather than encoding the masked pixels. This is what makes caching possible and is the standard latent-inpainting approach; it differs slightly from pixel mode, which still masks in pixel space.

## Hyperparameter sweep

```bash
python run_sweep.py
```

Runs `train.py` over every combination of the list-valued keys in `sweep_config.yaml`. Progress is checkpointed to `sweep_progress.json`, so an interrupted sweep resumes where it left off; use `--force-rerun` to redo completed runs and `--dry-run` to preview the commands.

To sweep the old super resolution project instead:

```bash
python run_sweep.py --config super_resolution/sweep_config.yaml --train-script super_resolution/train.py --state-file super_resolution/sweep_progress.json
```
