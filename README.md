# PixelDrift

![Version](https://img.shields.io/badge/version-0.1.0-blue)

CycleGAN-based unpaired grayscale image-to-image translation, built on
[sameer-forge](https://github.com/mauryasameer/the-forge). Adds real checkpointing and a GenAI
translation-quality commentary layer via `forge.llm`'s multimodal support, rendered into a
self-contained HTML report via `forge.report`.

## Setup

```bash
pip install -r requirements.txt
```

Place two unpaired grayscale image domains in `src/data/domain_x` and `src/data/domain_y`
(gitignored — not included in this repo).

## Usage

```bash
python -m src.app --epochs 500 --llm-provider ollama
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--domain-x-dir` | `src/data/domain_x` | First image domain |
| `--domain-y-dir` | `src/data/domain_y` | Second image domain |
| `--image-size` | `256` | Square image size images are resized to |
| `--epochs` | `10` | Training epochs (notebook's original scale was 500) |
| `--checkpoint-dir` | `checkpoints` | Where to save/resume TensorFlow checkpoints |
| `--checkpoint-interval` | `5` | Save a checkpoint every N epochs |
| `--sample-interval` | `5` | Generate a sample grid + LLM commentary every N epochs |
| `--llm-provider` | `ollama` | Commentary backend: `ollama` / `claude` / `openai` |
| `--output` | `reports/model_report.html` | Report output path |

Output is a single HTML report with one section per sampled epoch: current losses
(generator/discriminator/cycle/identity), a translation sample grid, and an LLM-generated
qualitative description of translation quality.

Training auto-resumes from the latest checkpoint in `--checkpoint-dir` if one exists.

## Architecture

- `src/core/interfaces.py` — `GeneratorFactory` abstract interface
- `src/providers/pix2pix_factory.py` — `Pix2PixGeneratorFactory`, wraps `tensorflow_examples`
- `src/services/data_service.py` — domain loading and normalization
- `src/services/training_service.py` — `CycleGANTrainer`: training loop + checkpointing
- `src/services/narrative_service.py` — LLM vision commentary on sample grids
- `src/services/report_service.py` — HTML report assembly

## Testing

```bash
pytest tests/ -v
```

15 tests, zero real network/LLM/TensorFlow-training calls beyond tiny synthetic smoke tests.
