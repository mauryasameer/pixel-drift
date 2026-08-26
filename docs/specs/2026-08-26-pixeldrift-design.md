# PixelDrift

## Problem

`Projects/style_gan/Cycle-GAN+solution.ipynb` is a monolithic TensorFlow/Keras CycleGAN
notebook (unpaired grayscale image-to-image translation: two U-Net generators with instance
norm, two PatchGAN discriminators, adversarial + cycle-consistency + identity losses, custom
`train_step` under `@tf.function`, 500-epoch loop). It imports a local `pix2pix_1` module that
doesn't exist anywhere in `dev/` (almost certainly a copy of TensorFlow's official
`tensorflow_examples.models.pix2pix` tutorial helper), reads from `data/Tr1`/`data/Tr2`
(two unpaired domains with no recoverable source), and has its checkpointing code commented
out — a crash during a long training run loses all progress.

## Goals

- A standalone, PROJECT_STANDARDS.md-compliant, `sameer-forge`-dependent repo that reproduces
  the notebook's CycleGAN training pipeline with real, working checkpointing (save every N
  epochs, auto-resume from latest on restart).
- A GenAI layer: real LLM-generated translation-quality commentary that actually looks at the
  sample translation grid (not just numeric losses) — the first project to exercise
  `forge.llm`'s multimodal `generate(images=...)` support (added in `sameer-forge` v0.5.0) and
  `forge.vision.gridplot`'s numpy/TF image support (added in v0.4.0).
- Fetchable/replaceable dependency: `tensorflow_examples` installed from its real upstream
  source rather than a guessed/vendored reimplementation.

## Non-goals

- Reproducing the actual `data/Tr1`/`data/Tr2` dataset — its source is unrecoverable. The
  retrofit parameterizes the two domain directories; the user supplies images later.
- Swapping generator/discriminator architectures. Only one architecture
  (`tensorflow_examples.models.pix2pix`'s U-Net + PatchGAN) is in scope; the abstraction
  boundary exists to isolate that one external dependency, not to support hypothetical
  alternative architectures with no current second implementation.
- Training at the notebook's original scale (500 epochs, 256×256 images) inside the test suite.
  Tests use tiny synthetic images and 1-2 epochs; a real training run is an operational concern
  for whoever runs `src/app.py` against real data, not something CI exercises.

## Architecture

New top-level repo `/Users/sameermaurya/Downloads/dev/pixel-drift` (own GitHub repo,
`github.com/mauryasameer/pixel-drift`), scaffolded via `forge new pixel-drift`.

- **`src/core/interfaces.py`** — `GeneratorFactory(ABC)`:
  - `build_generator(self, output_channels: int) -> tf.keras.Model`
  - `build_discriminator(self) -> tf.keras.Model`
  This isolates the one genuine external dependency needing abstraction — an unversioned,
  git-installed third-party model-construction library — the same way `LLMProvider` isolates
  `anthropic`/`openai`. Not a speculative multi-provider abstraction: the notebook only ever
  uses one architecture, so this interface has exactly one concrete implementation.

- **`src/providers/pix2pix_factory.py`** — `Pix2PixGeneratorFactory(GeneratorFactory)`, wrapping
  `tensorflow_examples.models.pix2pix.{unet_generator, discriminator}` with
  `norm_type='instancenorm'` (matching the notebook exactly).

- **`src/services/data_service.py`** — loads the two unpaired grayscale domains via
  `tf.keras.preprocessing.image_dataset_from_directory` (configurable `--domain-x-dir`/
  `--domain-y-dir`, defaulting to `src/data/domain_x`/`src/data/domain_y`, gitignored),
  normalizes to `[-1, 1]` (matching the notebook's `normalize()`).

- **`src/services/training_service.py`** — the CycleGAN training loop: two generators
  (G: X→Y, F: Y→X), two discriminators, adversarial loss (`BinaryCrossentropy`), cycle-
  consistency loss (`LAMBDA=10`), identity loss — all matching the notebook's loss functions
  exactly. `@tf.function`-decorated `train_step`. **Checkpointing enabled for real** (the
  notebook's commented-out code, now live): `tf.train.Checkpoint`/`CheckpointManager` saves
  every `--checkpoint-interval` epochs; on startup, automatically restores the latest checkpoint
  if one exists, logging whether it resumed or started fresh.

- **`src/services/narrative_service.py`** — every `--sample-interval` epochs: renders a sample
  translation grid via `forge.vision.gridplot.plot_translation_grid` (passing the TF sample
  tensors' `.numpy()` output — the numpy path added in forge v0.4.0), encodes it to PNG bytes,
  and calls `forge.llm`'s `generate(prompt, images=[grid_png_bytes])` (the vision path added in
  v0.5.0) for real qualitative commentary — not a numeric-metrics proxy, an actual look at the
  image.

- **`src/services/report_service.py`** — assembles a `forge.report.ReportBuilder` with one
  section per sample interval: the translation grid image, current epoch's loss values
  (generator/discriminator/cycle/identity), and the LLM's commentary.

- **`src/app.py`** — CLI driver. Flags: `--domain-x-dir`/`--domain-y-dir` (defaults above),
  `--epochs` (default a practical smoke-test value, e.g. 10 — not the notebook's 500, which
  remains available by passing `--epochs 500` for a real run), `--checkpoint-dir` (default
  `checkpoints/`, gitignored), `--checkpoint-interval` (default 5), `--sample-interval`
  (default 5), `--llm-provider` (`ollama`/`claude`/`openai`, default `ollama`, matching
  TrendWhisperer's convention), `--output` (default `reports/model_report.html`).

- **`requirements.txt`** — `sameer-forge[llm,vision] @ git+https://github.com/mauryasameer/the-forge.git@v0.5.1`
  (vision extra needed for `forge.vision.gridplot`, even though this project's own model code
  stays TensorFlow), plus `tensorflow`, `tensorflow_examples @ git+https://github.com/tensorflow/examples.git`.

## Data Flow

User places images in `src/data/domain_x`/`src/data/domain_y` → `data_service.py` loads both
via `image_dataset_from_directory`, normalizes to `[-1,1]` → `training_service.py` runs the
epoch loop: `train_step` (adversarial + cycle + identity losses, gradient updates for both
generators and discriminators) → checkpoint saved every `--checkpoint-interval` epochs → every
`--sample-interval` epochs, `narrative_service.py` renders a sample grid and gets LLM commentary
→ `report_service.py` accumulates one section per sample interval → final HTML report written
at `--output`.

## Error Handling

- `data_service.py`: missing domain directory → `FileNotFoundError` propagates with a clear
  message (matches `image_dataset_from_directory`'s own behavior — no need to wrap it).
- `training_service.py`: checkpoint restore failure (corrupted/incompatible checkpoint) is
  caught and logged as a warning; training starts fresh rather than crashing — a bad checkpoint
  should never block progress, only cost the epochs since the last good one.
- `narrative_service.py`: LLM failures (unreachable Ollama, missing API key) are caught per
  interval and produce a "commentary unavailable" fallback in that section — matches
  TrendWhisperer's narrative-service precedent — training itself is never blocked by the LLM
  layer being down.

## Testing

- `tests/unit/test_pix2pix_factory.py` — `Pix2PixGeneratorFactory` builds real
  `tf.keras.Model` instances with the right input/output channel shapes (a real, fast
  construction test — building the graph is cheap, only training is expensive).
- `tests/unit/test_data_service.py` — loads a tiny synthetic image directory (2-3 tiny PNGs
  per domain), asserts normalized `[-1,1]` range and correct tensor shape.
- `tests/unit/test_training_service.py` — a stub `GeneratorFactory` (returns trivial
  `tf.keras.Model`s) drives 1-2 epochs on tiny (e.g. 8×8) synthetic images; asserts a checkpoint
  file is written, and that restarting with the same checkpoint directory resumes rather than
  re-initializing (checkpoint step counter advances, not reset). One additional real-factory
  smoke test (`Pix2PixGeneratorFactory` + 1 epoch on tiny images) confirms actual integration,
  kept separate so the stub-based tests stay fast.
- `tests/unit/test_narrative_service.py` — stub `LLMProvider` (matching
  TrendWhisperer's `_StubLLM` pattern); asserts the rendered grid is passed as `images=[...]`,
  and the LLM-failure fallback path.
- `tests/unit/test_report_service.py` — assembles a report from fake results, asserts HTML
  contains the expected sections.
- `tests/integration/test_pipeline.py` — end-to-end on tiny synthetic domains, stub
  `GeneratorFactory` and stub `LLMProvider`, 1-2 epochs, asserts a report file is written and a
  checkpoint exists.

## Versioning

New repo starts at `0.1.0` per `forge new`'s scaffold default. Standard PROJECT_STANDARDS.md
branch/PR/version rules apply from the first feature branch onward. Single Python version only
(3.12, per standing project-wide rule) — the scaffold already generates this correctly as of
`sameer-forge` v0.5.1.
