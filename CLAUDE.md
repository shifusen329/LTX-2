# LTX-2 Monorepo

This file provides guidance to AI coding assistants (Claude, Cursor, etc.) working in this
repository. Package-specific guidance lives in each package's own `CLAUDE.md`.

## Layout

```
LTX-2/
├── packages/
│   ├── ltx-core/       # Core model implementations (transformer, VAEs, text encoder, scheduler)
│   ├── ltx-pipelines/  # High-level inference pipelines (distilled, two-stage, IC-LoRA, etc.)
│   └── ltx-trainer/    # LoRA / full fine-tuning / IC-LoRA training
└── pyproject.toml      # uv workspace root
```

All three packages share a single `uv` workspace. Install from the repo root with `uv sync`.

## Package-level guidance

- **[ltx-trainer/CLAUDE.md](packages/ltx-trainer/CLAUDE.md)** — the most comprehensive; covers
  model loading, training strategies, modalities, LTX-2 vs LTX-2.3 differences, and the config
  system. Read this before touching anything under `packages/ltx-trainer/`.

When editing cross-package code (e.g. adding a model component in `ltx-core` that the trainer
consumes), update both packages in the same change and run `uv run pytest` in each affected
package.

## Supported models

Only **LTX-2 / LTX-2.3** (audio-video). Older LTXV (video-only) checkpoints are not supported.
Version detection is automatic — `ltx-core` inspects the checkpoint config and selects the
correct feature extractor, caption projection path, embeddings connectors, prompt-AdaLN policy,
and vocoder. No version-specific code paths in downstream packages.

## Platform requirements

- Linux only (`triton` dependency).
- CUDA GPU with 24 GB+ VRAM for LoRA; 80 GB+ recommended for full fine-tuning.
- Multi-GPU training uses Accelerate (DDP or FSDP). For LoRA training on 96 GB cards, DDP is
  the right choice — see `packages/ltx-trainer/CLAUDE.md` for the rationale.

## Conventions

- Python 3.10+ syntax (`list[str]`, `str | Path`).
- `pathlib.Path` for filesystem operations.
- `@torch.inference_mode()` for inference code paths (preferred over `@torch.no_grad()`).
- Logging via the package logger (e.g. `from ltx_trainer import logger`); no `print` in
  production code.
- Run `uv run ruff check .` and `uv run ruff format .` before committing.

## Local overrides

`packages/ltx-trainer/configs/nsfw_ltx2_av_lora.yaml` is a local training config and is
gitignored. Do not commit personal or dataset-specific configs to the shared tree.
