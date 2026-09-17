# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.3] - 2026-09-18
### Changed
- Replaced the README hero with a text-free CycleGAN transformation visual that reflects PixelDrift's bidirectional grayscale translation workflow.

## [0.1.2] - 2026-09-15
### Changed
- README badges brought to linked shields.io style (CI, Version, Python, License all
  clickable), matching llm_eval/ocr_docker.

## [0.1.1] - 2026-09-14
### Added
- `Dockerfile`/`docker-compose.yml`/`.env.example` for one-click `docker compose up --build` deploy.
- `GOVERNANCE.md` — intended use, explainability boundary, fairness scope (not applicable to this domain), LLM controls, audit trail, regulatory framing.
- Permanent governance banner and a "Run Info" audit block (LLM provider, epochs trained, image size, checkpoint dir, timestamp) embedded in every generated report; commentary content is now HTML-escaped before embedding.

### Fixed
- `narrative_service.py`'s commentary LLM call now passes `temperature=0.0` explicitly — previously omitted, so commentary used non-deterministic default sampling instead of reproducible output.

## [Unreleased]

## [0.1.0] - 2026-08-26

### Added
- `src/core/interfaces.py` — `GeneratorFactory` abstract interface
- `src/providers/pix2pix_factory.py` — `Pix2PixGeneratorFactory` wrapping `tensorflow_examples`' pix2pix U-Net, with a self-adapting channel-tiling wrapper for non-RGB inputs
- `src/services/data_service.py` — domain image loading and normalization
- `src/services/training_service.py` — `CycleGANTrainer` with checkpointing and auto-resume
- `src/services/narrative_service.py` — LLM-generated translation-quality commentary via `meerax.llm`'s multimodal support
- `src/services/report_service.py` — self-contained HTML report assembly via `meerax.report`
- `src/app.py` — CLI driver (`--domain-x-dir`/`--domain-y-dir`, `--epochs`, `--checkpoint-interval`, `--sample-interval`, `--llm-provider`, `--output`)
- 16 tests, zero real network/LLM calls

[0.1.0]: https://github.com/mauryasameer/pixel-drift/releases/tag/v0.1.0
