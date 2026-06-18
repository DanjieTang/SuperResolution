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
`working = sample_latent(x, scaling_factor) if args.use_vae else x`.

`decode(vae, tensor)` maps working space back to pixels for visualization and is
the identity when `vae is None` (pixel mode).

## Latent mode is a precompute-then-train workflow

The VAE is frozen, so its encodings never change. Encoding on the fly was the
entire performance problem (two full-res encodes per step ≈ 7.5x slowdown), so:

1. `precompute_latents.py` encodes the dataset **once**, writing one file per
   image (raw 8-channel VAE moments, mean/logvar) into a directory tree that
   mirrors the source dataset, via `latents.save_latent_moment`. A single
   `cache_meta.pt` (scaling factor + provenance) is written at the cache root by
   `latents.save_cache_meta`. Each image is encoded and written as it is read,
   so memory stays flat, and re-running skips images whose `.pt` already exists.
2. `train.py --use_vae` reads `scaling_factor` once via `latents.load_cache_meta`
   and streams moments through `dataloader.prepare_latent_loaders`, which builds
   a `LatentFolderDataset` that loads one moments file per access (the full cache
   is never resident). Each step draws a fresh latent with `latents.sample_latent`
   (so VAE sampling stochasticity is preserved — we cache moments, not a sample).

The training loop performs **zero VAE encodes**. The VAE is still loaded in
`train.py` but only to `decode` samples during visualization.

### Invariants to keep when editing latent mode

- Moments are stored as fp16, one file per image of shape `(8, H, W)`; the
  loader stacks them to `(B, 8, H, W)` and `sample_latent` chunks into
  mean/logvar, clamps logvar to `[-30, 20]`, and scales by `scaling_factor`.
  This must stay consistent with diffusers' `DiagonalGaussianDistribution`.
- `scaling_factor` is read from `cache_meta.pt`, not from `vae.config`, so
  training does not depend on the VAE being loaded for sampling.
- The cache dir mirrors the dataset tree (`<root>/<class>/<name>.pt`).
  `cache_meta.pt` lives at the root and is excluded from the moment-file scan
  (`dataloader.list_latent_files`). Deleting an image's `.pt` re-encodes just
  that image on the next precompute run.
- **Latent-space masking**: known region is `working * working_mask` in latent
  space. Do NOT reintroduce `encode(masked_pixels)` in the loop — that is the
  uncacheable, slow path we removed. Pixel mode still masks in pixel space; this
  asymmetry is intentional.
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
- Changing dataset, `--canvas_size`, or `--vae` requires regenerating the latent
  cache; there is no automatic invalidation — the cache path is whatever
  `--output` / `--latent_cache` points at.
- Training and inference must agree on masking semantics; if you change one of
  `flow_matching_loss` / `visualize` / `sample`, check the others.

## Verifying changes

There is no test suite. After edits, at minimum:

```bash
python -m py_compile train.py dataloader.py latents.py precompute_latents.py
```

End-to-end runs need the ImageNet dataset and a CUDA GPU (DGX Spark); the
MacBook is edit-only.
