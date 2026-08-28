# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
