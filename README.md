# Outpainting

A pixel-space flow matching model that expands images beyond their borders (outpainting), trained from scratch on ImageNet with no pretrained components. A DiT (diffusion transformer) folds each 16x16 pixel tile into one token and, conditioned on the masked image and known-region mask, learns the flow matching velocity field. The known region is randomly sized and placed during training, supporting expansion in any direction up to 2x per dimension (e.g. 512x512 -> 1024x1024).

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

Defaults train on the `ImagenetHighResolution` folder using CUDA if available, falling back to MPS then CPU. bf16 autocast is used automatically on CUDA.

Common overrides:

```bash
python train.py --dataset_path /path/to/imagenet --device cuda:0 --batch_size 8 --epochs 1
```

Enable W&B logging by passing both `--project` and `--entity`.

### Optional: latent diffusion

Passing `--use_vae` trains in the latent space of a frozen pretrained VAE (`stabilityai/sd-vae-ft-ema`, downloaded from Hugging Face on first run) instead of pixel space. The patch size defaults to 2 on the 8x-smaller latents, keeping the token count identical. Requires the optional dependency: `uv sync --extra vae`.

```bash
python train.py --use_vae
```

The VAE encodes each batch to latents on the fly inside the training loop (and decodes samples for visualization); nothing is stored to disk. This is slower per step than pixel mode, but avoids maintaining a large on-disk latent cache.

> **Note:** an earlier version precomputed the latents to disk to skip in-loop encoding. That cache grew very large (one tiny file per image) and a write failed partway through a multi-hour run, so the project reverted to on-the-fly encoding. The current loop does a single encode per step (masking happens in latent space, below), not the two-encode path that originally motivated the cache.

**Latent-space masking.** In latent mode the known region is formed by masking the encoded latent directly (`latent * mask`), rather than encoding the masked pixels. This is the standard latent-inpainting approach; it differs slightly from pixel mode, which still masks in pixel space.

## Hyperparameter sweep

```bash
python run_sweep.py
```

Runs `train.py` over every combination of the list-valued keys in `sweep_config.yaml`. Progress is checkpointed to `sweep_progress.json`, so an interrupted sweep resumes where it left off; use `--force-rerun` to redo completed runs and `--dry-run` to preview the commands.

To sweep the old super resolution project instead:

```bash
python run_sweep.py --config super_resolution/sweep_config.yaml --train-script super_resolution/train.py --state-file super_resolution/sweep_progress.json
```
