# CLAUDE.md

Guidance for AI agents working in this repo. Humans should read `README.md`.

## What this project is

An **outpainting** model: a flow matching DiT that expands images beyond their
borders. Trained from scratch on ImageNet, no GANs, no pretrained components
(except an optional frozen VAE for latent mode). The repo was previously a 4x
super resolution project, which now lives untouched under `super_resolution/`.

## Core abstraction: the "working space"

The model trains on a **working tensor** plus a **working mask**, and is
agnostic to whether that space is pixels or VAE latents:

- **Pixel mode** (default): working tensor is the image `(B, 3, 512, 512)`;
  patch size 16 → 32x32 = 1024 tokens.
- **Latent mode** (`--use_vae`): working tensor is a VAE latent
  `(B, 4, 64, 64)`; patch size 2 → 32x32 = 1024 tokens.

The token count is identical in both modes, so the DiT itself costs the same.
`flow_matching_loss(model, working, working_mask, criterion)` and
`visualize(model, vae, working, mask, ...)` both operate purely in working
space. The training loop produces `working` before calling them:
`working = encode(vae, x)`.

`encode(vae, pixels)` maps pixels into working space (sampled, scaled VAE latent)
and `decode(vae, tensor)` maps working space back to pixels for visualization.
Both are the identity when `vae is None` (pixel mode).

## Latent mode encodes on the fly

The VAE is frozen, so it is loaded once and used for both directions:
`train.py:encode` maps a pixel batch into working space and `decode` maps
samples back for visualization. `train.py --use_vae` loads the same pixel
dataloader as pixel mode (`dataloader.prepare_dataset`); each step encodes that
batch to a latent in the loop:
`working = encode(vae, x)` (identity when `vae is None`, i.e. pixel mode).

`encode` samples a fresh latent every call
(`vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor`), so the
encoder's sampling stochasticity is preserved across epochs.

> Earlier this repo precomputed latents to disk to avoid encoding in the loop.
> That cache grew to ~75 GB of tiny files and a write failed mid-run, so we
> reverted to on-the-fly encoding. The current path does **one** encode per step
> (mask in latent space, see below), not the two-encode path that originally
> motivated the cache, so the slowdown is roughly half what it was.

### Invariants to keep when editing latent mode

- `scaling_factor` comes from `vae.config.scaling_factor` and must be applied in
  `encode` and undone in `decode` (they are inverses). The VAE must be loaded
  for training, not just visualization.
- **Latent-space masking**: known region is `working * working_mask` in latent
  space. Do NOT add a second `encode(masked_pixels)` in the loop — encode the
  full image once and mask the resulting latent. Pixel mode still masks in pixel
  space; this asymmetry is intentional.
- The VAE encode is wrapped in the same bf16 autocast as the rest of the step on
  CUDA, and runs under `@torch.no_grad` (we never backprop through the encoder).
- Masks are sampled at the working resolution directly
  (`sample_known_region_mask(B, working.shape[-1], device)`), not at canvas size
  then interpolated.

## Conventions

- Images are normalized to `[-1, 1]` (mean/std 0.5). Views are un-normalized
  with `* 0.5 + 0.5` before plotting.
- bf16 autocast is used on CUDA only (`nullcontext` elsewhere).
- `--use_vae` requires the optional `diffusers` dependency (`uv sync --extra vae`).
- Flow matching convention: `t=0` is data, `t=1` is noise; the model predicts
  velocity `noise - data`. Sampling integrates from `t=1` to `t=0` with Euler
  steps. Keep this direction consistent across `sample` and the loss.

## Gotchas

- `dataloader.py:valid_image_folder` filters macOS `._*` / `.DS_Store` files.
- Latent mode re-encodes every image each epoch (no caching), so `--use_vae` is
  slower per step than pixel mode; this is the deliberate trade for not storing a
  large latent cache to disk.
- Training and inference must agree on masking semantics; if you change one of
  `flow_matching_loss` / `visualize` / `sample`, check the others.

## Verifying changes

There is no test suite. After edits, at minimum:

```bash
python -m py_compile train.py dataloader.py
```

End-to-end runs need the ImageNet dataset and a CUDA GPU (DGX Spark); the
MacBook is edit-only.
