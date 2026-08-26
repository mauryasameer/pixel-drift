# PixelDrift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PixelDrift — a standalone, forge-dependent repo that refactors `Projects/style_gan/Cycle-GAN+solution.ipynb`'s TensorFlow CycleGAN (unpaired grayscale image translation) into a tested `src/{core,providers,services}` pipeline with real checkpointing and an LLM vision-commentary layer.

**Architecture:** `src/core/interfaces.py` defines `GeneratorFactory`; `src/providers/pix2pix_factory.py` wraps `tensorflow_examples.models.pix2pix` (self-adapting to the data's actual channel count); `src/services/` holds data loading, the CycleGAN training loop (with real checkpointing), narrative generation (via `forge.llm`'s multimodal `generate()`), and report assembly (via `forge.report` + `forge.vision.gridplot`); `src/app.py` is the CLI driver.

**Tech Stack:** Python 3.12, TensorFlow >=2.16, `tensorflow_examples` (git), `sameer-forge[llm,vision]` (pinned to v0.5.1), pytest.

**Spec:** `docs/specs/2026-08-26-pixeldrift-design.md` (staged pre-Task-1 at `/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/590a04fc-1a1c-4940-9374-ef8c02322e96/scratchpad/2026-08-26-pixeldrift-design.md`)

## Global Constraints

- Python `>=3.12` only — no version matrix, no `>=3.11` range. `target-version = "py312"` in ruff.
- No `typing.List`/`Dict`/`Optional` — use `list[str]`, `dict[str, Any]`, `X | None`.
- No `print()` outside `src/app.py` (the only CLI-facing entry point) — everything in `src/services/`, `src/providers/` uses `logging`.
- Commit messages use `feat:`/`fix:`/`test:`/`docs:`/`chore:`/`init:` imperative-mood prefixes, no AI/Claude/Anthropic attribution anywhere.
- Task 1 is repo genesis: the GitHub repo `mauryasameer/pixel-drift` already exists (created empty, no commits, default branch `main`) — Task 1 pushes into it via `git remote add` + push, NOT `gh repo create`.
- After Task 1, all further work (Tasks 2-9) happens on a single branch `feature/cyclegan-pipeline`, branched from `dev` — never committed directly to `dev`/`main`. One PR into `dev` at the end of this plan; promoting `dev` to `main` (tag, release) is a separate follow-up step, out of scope for this plan.
- Real network calls (TensorFlow Hub, live LLM calls to Ollama/Claude/OpenAI) are never made in tests — every test uses synthetic in-memory/tiny-file data and stub providers.
- Working directory for Task 1: `/Users/sameermaurya/Downloads/dev`. Working directory for Tasks 2-9: `/Users/sameermaurya/Downloads/dev/pixel-drift`.

---

### Task 1: Scaffold the repo, push into the existing GitHub remote, land the spec/plan

**Files:**
- Create: the entire `pixel-drift` repo via `forge new`
- Modify: `requirements.txt` (add TensorFlow/tensorflow_examples/forge extras `forge new` doesn't know about)
- Create: `docs/specs/2026-08-26-pixeldrift-design.md`, `docs/plans/2026-08-26-pixeldrift-plan.md`

**Interfaces:**
- Produces: a working `pixel-drift` repo at `/Users/sameermaurya/Downloads/dev/pixel-drift`, with `dev` and `main` branches pushed to the pre-existing empty GitHub repo `github.com/mauryasameer/pixel-drift`.

- [ ] **Step 1: Bootstrap a Python 3.12 venv with sameer-forge installed, to run `forge new`**

```bash
/opt/homebrew/bin/python3.12 -m venv /tmp/pixeldrift-bootstrap
source /tmp/pixeldrift-bootstrap/bin/activate
pip install -q --upgrade pip
pip install -e /Users/sameermaurya/Downloads/dev/the-forge
python -c "import forge; print(forge.__version__)"
```

Expected: prints `0.5.1` (the-forge's current released version — if it prints something else, STOP and report NEEDS_CONTEXT, since Step 3's `requirements.txt` dependency line depends on this being `0.5.1`).

- [ ] **Step 2: Scaffold the project**

```bash
cd /Users/sameermaurya/Downloads/dev
forge new pixel-drift
```

Expected: prints a created-files list; `ls pixel-drift` shows `src/`, `tests/`, `scripts/`, `.github/`, `conftest.py`, `pyproject.toml`, `requirements.txt`, `VERSION`, `CHANGELOG.md`, `task.md`, `README.md`, `.gitignore`; `pixel-drift/.git` exists. `pyproject.toml` should already show `requires-python = ">=3.12"` and `target-version = "py312"`, and `.github/workflows/ci.yml`'s test job should have NO `strategy.matrix` block (both fixed in sameer-forge v0.5.1) — if either is wrong, STOP and report NEEDS_CONTEXT, since that means an older forge version got installed in Step 1.

- [ ] **Step 3: Extend the generated `requirements.txt`**

Read `pixel-drift/requirements.txt` first — it will contain exactly one line:
`sameer-forge @ git+https://github.com/mauryasameer/the-forge.git@v0.5.1`

Replace that line's package spec to include the `llm` and `vision` extras (needed for
`OllamaProvider`/`ClaudeProvider`/`OpenAIProvider` and `forge.vision.gridplot` respectively),
and add `tensorflow` and `tensorflow_examples` (the latter has no PyPI release, installed
directly from its GitHub source):

```
sameer-forge[llm,vision] @ git+https://github.com/mauryasameer/the-forge.git@v0.5.1
tensorflow>=2.16
tensorflow_examples @ git+https://github.com/tensorflow/examples.git
```

- [ ] **Step 4: Land the spec and plan documents**

```bash
cd /Users/sameermaurya/Downloads/dev/pixel-drift
mkdir -p docs/specs docs/plans
cp "/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/590a04fc-1a1c-4940-9374-ef8c02322e96/scratchpad/2026-08-26-pixeldrift-design.md" docs/specs/
cp "/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/590a04fc-1a1c-4940-9374-ef8c02322e96/scratchpad/2026-08-26-pixeldrift-plan.md" docs/plans/
```

(If either source path no longer exists because the scratchpad was cleaned up, STOP and report NEEDS_CONTEXT — do not fabricate replacement content.)

- [ ] **Step 5: Commit the scaffold**

```bash
cd /Users/sameermaurya/Downloads/dev/pixel-drift
git add -A
git status --short
git commit -m "init: scaffold pixel-drift via forge new"
git log --oneline -1
git branch --show-current
```

Expected: commit succeeds; the branch shown is whatever `forge new`'s `git init` created as the default (almost certainly `main`).

- [ ] **Step 6: Create the `dev` branch, push both into the existing GitHub repo**

The GitHub repo `mauryasameer/pixel-drift` already exists (created empty via the GitHub UI, no commits, no branches, default branch `main`) — do NOT run `gh repo create`, it will fail or conflict. Add it as a remote instead:

```bash
cd /Users/sameermaurya/Downloads/dev/pixel-drift
git checkout -b dev
git remote add origin https://github.com/mauryasameer/pixel-drift.git
git push -u origin main
git push -u origin dev
```

Expected: both pushes succeed. If `git push` fails because the remote already has commits (meaning the repo wasn't actually empty), STOP and report BLOCKED with the exact error — do not force-push.

- [ ] **Step 7: Verify the venv can install from the real generated requirements.txt**

This step installs TensorFlow, which is a large download (several hundred MB) — expect it to take a few minutes.

```bash
deactivate
rm -rf /tmp/pixeldrift-bootstrap
/opt/homebrew/bin/python3.12 -m venv /Users/sameermaurya/Downloads/dev/pixel-drift/.venv
source /Users/sameermaurya/Downloads/dev/pixel-drift/.venv/bin/activate
pip install -q --upgrade pip
pip install -r /Users/sameermaurya/Downloads/dev/pixel-drift/requirements.txt
python -c "import forge, tensorflow, tensorflow_examples; print('deps ok')"
```

Expected: `deps ok` prints with no errors. Verify `.venv/` is covered by `pixel-drift/.gitignore` (it should be, from the scaffold) — if not, add `.venv/` and commit that as a follow-up in this same step.

- [ ] **Step 8: Report**

Report back with:
- **Status:** DONE | BLOCKED | NEEDS_CONTEXT
- Commits created (short SHA + subject)
- Confirmation of: `forge.__version__` printed in Step 1, no-matrix `ci.yml` confirmed in Step 2, both pushes succeeded in Step 6, `deps ok` from Step 7
- The report file path (write full details to `/private/tmp/claude-501/-Users-sameermaurya-Downloads-dev/590a04fc-1a1c-4940-9374-ef8c02322e96/scratchpad/pixeldrift-task-1-report.md`)

---

### Task 2: Core interface — `GeneratorFactory`

**Files:**
- Create: `src/core/interfaces.py`
- Test: `tests/unit/test_interfaces.py`

**Interfaces:**
- Produces: `GeneratorFactory(ABC)` with abstract methods `build_generator(self, output_channels: int) -> tf.keras.Model` and `build_discriminator(self) -> tf.keras.Model`. Consumed by `Pix2PixGeneratorFactory` (Task 3) and `CycleGANTrainer` (Task 5, referenced as a type).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_interfaces.py
import pytest
import tensorflow as tf

from src.core.interfaces import GeneratorFactory


class _StubFactory(GeneratorFactory):
    def build_generator(self, output_channels: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(8, 8, 1))
        outputs = tf.keras.layers.Conv2D(output_channels, 1, padding="same")(inputs)
        return tf.keras.Model(inputs, outputs)

    def build_discriminator(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(8, 8, 1))
        outputs = tf.keras.layers.Conv2D(1, 1, padding="same")(inputs)
        return tf.keras.Model(inputs, outputs)


def test_cannot_instantiate_abstract_factory():
    with pytest.raises(TypeError):
        GeneratorFactory()


def test_stub_factory_implements_interface():
    factory = _StubFactory()
    generator = factory.build_generator(output_channels=1)
    discriminator = factory.build_discriminator()
    assert isinstance(generator, tf.keras.Model)
    assert isinstance(discriminator, tf.keras.Model)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/unit/test_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.interfaces'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/core/interfaces.py
from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf


class GeneratorFactory(ABC):
    """Abstract interface for CycleGAN generator/discriminator construction.

    Concrete implementations live in src/providers/. Isolates the one external
    model-construction dependency (tensorflow_examples) from the training service.
    """

    @abstractmethod
    def build_generator(self, output_channels: int) -> tf.keras.Model:
        """Build a U-Net-style generator producing output_channels output channels."""
        ...

    @abstractmethod
    def build_discriminator(self) -> tf.keras.Model:
        """Build a PatchGAN-style discriminator."""
        ...
```

Also create `src/core/__init__.py` (empty) if it doesn't already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_interfaces.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/core/interfaces.py src/core/__init__.py tests/unit/test_interfaces.py
git commit -m "feat: add GeneratorFactory interface"
```

---

### Task 3: Pix2Pix provider (wraps `tensorflow_examples`, self-adapts to grayscale)

**Files:**
- Create: `src/providers/pix2pix_factory.py`
- Test: `tests/unit/test_pix2pix_factory.py`

**Interfaces:**
- Consumes: `GeneratorFactory` from `src/core/interfaces.py` (Task 2); `tensorflow_examples.models.pix2pix.pix2pix.{unet_generator, discriminator}` (external dependency, installed via `requirements.txt` in Task 1).
- Produces: `Pix2PixGeneratorFactory(GeneratorFactory)`, constructor `__init__(self, norm_type: str = "instancenorm")`.

**Important — read before implementing:** `tensorflow_examples`' `pix2pix.unet_generator`/`pix2pix.discriminator` are built for the library's original RGB (3-channel) tutorial use case, and their `Input` layers may hardcode 3 input channels regardless of the `output_channels` argument (which only controls the final output layer's channel count). Our data is single-channel grayscale. Rather than guessing whether the installed version needs this, the factory below **self-adapts**: it inspects the real constructed model's `input_shape[-1]` and, if it doesn't match the channel count we actually need, wraps it with an input adapter that tiles the single channel to match. This works correctly whether or not the real package turns out to already be channel-flexible.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_pix2pix_factory.py
import tensorflow as tf

from src.providers.pix2pix_factory import Pix2PixGeneratorFactory


def test_build_generator_runs_on_grayscale_input():
    factory = Pix2PixGeneratorFactory()
    generator = factory.build_generator(output_channels=1)

    dummy_input = tf.zeros((1, 256, 256, 1))
    output = generator(dummy_input, training=False)

    assert output.shape == (1, 256, 256, 1)


def test_build_discriminator_runs_on_grayscale_input():
    factory = Pix2PixGeneratorFactory()
    discriminator = factory.build_discriminator()

    dummy_input = tf.zeros((1, 256, 256, 1))
    output = discriminator(dummy_input, training=False)

    assert output.shape[0] == 1
    assert output.shape[-1] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pix2pix_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.providers.pix2pix_factory'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/pix2pix_factory.py
from __future__ import annotations

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from src.core.interfaces import GeneratorFactory


class Pix2PixGeneratorFactory(GeneratorFactory):
    """Wraps tensorflow_examples' pix2pix U-Net generator / PatchGAN discriminator.

    Self-adapts input channel count: tensorflow_examples' models are built for
    3-channel RGB; if the constructed model doesn't already accept our actual
    channel count, wraps it with a channel-tiling input adapter.
    """

    def __init__(self, norm_type: str = "instancenorm") -> None:
        self._norm_type = norm_type

    def build_generator(self, output_channels: int) -> tf.keras.Model:
        base = pix2pix.unet_generator(output_channels, norm_type=self._norm_type)
        return self._adapt_input_channels(base, input_channels=output_channels)

    def build_discriminator(self) -> tf.keras.Model:
        base = pix2pix.discriminator(norm_type=self._norm_type, target=False)
        return self._adapt_input_channels(base, input_channels=1)

    @staticmethod
    def _adapt_input_channels(model: tf.keras.Model, input_channels: int) -> tf.keras.Model:
        expected_channels = model.input_shape[-1]
        if expected_channels == input_channels:
            return model
        inputs = tf.keras.Input(shape=(None, None, input_channels))
        repeats = expected_channels // input_channels
        tiled = tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 1, 1, repeats]))(inputs)
        return tf.keras.Model(inputs, model(tiled))
```

Also create `src/providers/__init__.py` (empty) if it doesn't already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_pix2pix_factory.py -v`
Expected: PASS (2 passed). This test builds real `tensorflow_examples` models and runs a real forward pass — expect it to take up to 30 seconds (TF graph tracing overhead), not because anything is wrong.

- [ ] **Step 5: Commit**

```bash
git add src/providers/pix2pix_factory.py src/providers/__init__.py tests/unit/test_pix2pix_factory.py
git commit -m "feat: add Pix2Pix generator factory with grayscale channel adaptation"
```

---

### Task 4: Data service

**Files:**
- Create: `src/services/data_service.py`
- Test: `tests/unit/test_data_service.py`

**Interfaces:**
- Produces: `normalize(image: tf.Tensor) -> tf.Tensor` (maps `[0,255]` to `[-1,1]`), `load_domain(directory: str | Path, image_size: int = 256, batch_size: int = 1) -> tf.data.Dataset` (raises `FileNotFoundError` if the directory doesn't exist). Consumed by `src/app.py` (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_data_service.py
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from src.services.data_service import load_domain, normalize


def _write_tiny_images(directory, count=3, size=8):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        array = np.full((size, size), fill_value=i * 50, dtype=np.uint8)
        Image.fromarray(array, mode="L").save(directory / f"img{i}.png")


def test_load_domain_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_domain(tmp_path / "does-not-exist")


def test_load_domain_returns_normalized_batches(tmp_path):
    domain_dir = tmp_path / "domain_x"
    _write_tiny_images(domain_dir, count=3, size=8)

    dataset = load_domain(domain_dir, image_size=8, batch_size=1)
    batch = next(iter(dataset))

    assert batch.shape == (1, 8, 8, 1)
    assert tf.reduce_min(batch) >= -1.0
    assert tf.reduce_max(batch) <= 1.0


def test_normalize_maps_black_and_white_to_minus1_and_1():
    black = tf.zeros((1, 1, 1))
    white = tf.fill((1, 1, 1), 255.0)

    assert normalize(black).numpy()[0][0][0] == pytest.approx(-1.0)
    assert normalize(white).numpy()[0][0][0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_data_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.data_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/data_service.py
from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def normalize(image: tf.Tensor) -> tf.Tensor:
    """Convert a [0, 255] image tensor to [-1, 1]."""
    image = tf.cast(image, tf.float32)
    return (image / 127.5) - 1.0


def load_domain(directory: str | Path, image_size: int = 256, batch_size: int = 1) -> tf.data.Dataset:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image domain directory not found: {directory}")
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        str(directory),
        color_mode="grayscale",
        batch_size=batch_size,
        label_mode=None,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=42,
        interpolation="nearest",
    )
    return dataset.map(normalize)
```

Create `src/services/__init__.py` (empty) if it doesn't already exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_data_service.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/data_service.py src/services/__init__.py tests/unit/test_data_service.py
git commit -m "feat: add data service (domain loading, normalization)"
```

---

### Task 5: Training service (CycleGAN loop + real checkpointing)

**Files:**
- Create: `src/services/training_service.py`
- Test: `tests/unit/test_training_service.py`

**Interfaces:**
- Consumes: `GeneratorFactory` (Task 2).
- Produces: `CycleGANTrainer` class:
  - `__init__(self, factory: GeneratorFactory, checkpoint_dir: str | Path, output_channels: int = 1) -> None` — builds `generator_g`, `generator_f`, `discriminator_x`, `discriminator_y` (all public attributes), optimizers, `self.checkpoint` (a `tf.train.Checkpoint` with an `epoch: tf.Variable` field), `self.checkpoint_manager`, and auto-restores the latest checkpoint if present.
  - `train_step(self, real_x: tf.Tensor, real_y: tf.Tensor) -> dict[str, tf.Tensor]` — one `@tf.function`-decorated training step, returns a dict with keys `gen_g_loss`, `gen_f_loss`, `disc_x_loss`, `disc_y_loss`.
  - `fit(self, domain_x: tf.data.Dataset, domain_y: tf.data.Dataset, epochs: int, checkpoint_interval: int = 5, on_epoch_end: Callable[[int, dict[str, tf.Tensor]], None] | None = None) -> None`.
  Consumed by `src/app.py` (Task 8, uses `CycleGANTrainer`, `trainer.generator_g` directly, and the `on_epoch_end` callback hook).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_training_service.py
import numpy as np
import tensorflow as tf

from src.core.interfaces import GeneratorFactory
from src.services.training_service import CycleGANTrainer


class _TinyFactory(GeneratorFactory):
    def build_generator(self, output_channels: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(None, None, 1))
        outputs = tf.keras.layers.Conv2D(output_channels, 3, padding="same", activation="tanh")(inputs)
        return tf.keras.Model(inputs, outputs)

    def build_discriminator(self) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(None, None, 1))
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same")(inputs)
        return tf.keras.Model(inputs, outputs)


def _tiny_dataset(count=2, size=8):
    images = np.random.uniform(-1, 1, size=(count, size, size, 1)).astype("float32")
    return tf.data.Dataset.from_tensor_slices(images).batch(1)


def test_fit_runs_and_saves_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = CycleGANTrainer(_TinyFactory(), checkpoint_dir=checkpoint_dir)

    trainer.fit(_tiny_dataset(), _tiny_dataset(), epochs=1, checkpoint_interval=1)

    assert checkpoint_dir.exists()
    assert trainer.checkpoint_manager.latest_checkpoint is not None


def test_fit_resumes_from_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"

    first_trainer = CycleGANTrainer(_TinyFactory(), checkpoint_dir=checkpoint_dir)
    first_trainer.fit(_tiny_dataset(), _tiny_dataset(), epochs=2, checkpoint_interval=1)
    assert int(first_trainer.checkpoint.epoch) == 2

    second_trainer = CycleGANTrainer(_TinyFactory(), checkpoint_dir=checkpoint_dir)
    assert int(second_trainer.checkpoint.epoch) == 2


def test_restore_failure_falls_back_to_fresh_start(tmp_path, monkeypatch, caplog):
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = CycleGANTrainer(_TinyFactory(), checkpoint_dir=checkpoint_dir)

    monkeypatch.setattr(
        type(trainer.checkpoint_manager),
        "latest_checkpoint",
        property(lambda self: "bogus-path"),
    )

    def _raise(*args, **kwargs):
        raise ValueError("corrupted checkpoint")

    monkeypatch.setattr(trainer.checkpoint, "restore", _raise)

    with caplog.at_level("WARNING"):
        trainer._restore_latest()

    assert "Checkpoint restore failed" in caplog.text
    assert int(trainer.checkpoint.epoch) == 0


def test_fit_invokes_on_epoch_end_callback(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = CycleGANTrainer(_TinyFactory(), checkpoint_dir=checkpoint_dir)
    calls = []

    trainer.fit(
        _tiny_dataset(),
        _tiny_dataset(),
        epochs=2,
        checkpoint_interval=5,
        on_epoch_end=lambda epoch, losses: calls.append(epoch),
    )

    assert calls == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_training_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.training_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/training_service.py
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import tensorflow as tf

from src.core.interfaces import GeneratorFactory

logger = logging.getLogger(__name__)

LAMBDA = 10
_loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real: tf.Tensor, generated: tf.Tensor) -> tf.Tensor:
    real_loss = _loss_obj(tf.ones_like(real), real)
    generated_loss = _loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5


def generator_loss(generated: tf.Tensor) -> tf.Tensor:
    return _loss_obj(tf.ones_like(generated), generated)


def cycle_loss(real_image: tf.Tensor, cycled_image: tf.Tensor) -> tf.Tensor:
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image: tf.Tensor, same_image: tf.Tensor) -> tf.Tensor:
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))


class CycleGANTrainer:
    """Orchestrates CycleGAN training: two generators, two discriminators, checkpointing."""

    def __init__(
        self,
        factory: GeneratorFactory,
        checkpoint_dir: str | Path,
        output_channels: int = 1,
    ) -> None:
        self.generator_g = factory.build_generator(output_channels)
        self.generator_f = factory.build_generator(output_channels)
        self.discriminator_x = factory.build_discriminator()
        self.discriminator_y = factory.build_discriminator()

        self.generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint = tf.train.Checkpoint(
            generator_g=self.generator_g,
            generator_f=self.generator_f,
            discriminator_x=self.discriminator_x,
            discriminator_y=self.discriminator_y,
            generator_g_optimizer=self.generator_g_optimizer,
            generator_f_optimizer=self.generator_f_optimizer,
            discriminator_x_optimizer=self.discriminator_x_optimizer,
            discriminator_y_optimizer=self.discriminator_y_optimizer,
            epoch=tf.Variable(0),
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, str(self.checkpoint_dir), max_to_keep=5
        )
        self._restore_latest()

    def _restore_latest(self) -> None:
        if self.checkpoint_manager.latest_checkpoint:
            try:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                logger.info("Resumed from checkpoint at epoch %d", int(self.checkpoint.epoch))
            except Exception:
                self.checkpoint.epoch.assign(0)
                logger.warning("Checkpoint restore failed, starting fresh", exc_info=True)
        else:
            logger.info("No checkpoint found, starting fresh")

    @tf.function
    def train_step(self, real_x: tf.Tensor, real_y: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_g(real_x, training=True)
            cycled_x = self.generator_f(fake_y, training=True)

            fake_x = self.generator_f(real_y, training=True)
            cycled_y = self.generator_g(fake_x, training=True)

            same_x = self.generator_f(real_x, training=True)
            same_y = self.generator_g(real_y, training=True)

            disc_real_x = self.discriminator_x(real_x, training=True)
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_x = self.discriminator_x(fake_x, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)

            gen_g_loss = generator_loss(disc_fake_y)
            gen_f_loss = generator_loss(disc_fake_x)

            total_cycle = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + total_cycle + identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle + identity_loss(real_x, same_x)

            disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

        self.generator_g_optimizer.apply_gradients(
            zip(
                tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables),
                self.generator_g.trainable_variables,
                strict=True,
            )
        )
        self.generator_f_optimizer.apply_gradients(
            zip(
                tape.gradient(total_gen_f_loss, self.generator_f.trainable_variables),
                self.generator_f.trainable_variables,
                strict=True,
            )
        )
        self.discriminator_x_optimizer.apply_gradients(
            zip(
                tape.gradient(disc_x_loss, self.discriminator_x.trainable_variables),
                self.discriminator_x.trainable_variables,
                strict=True,
            )
        )
        self.discriminator_y_optimizer.apply_gradients(
            zip(
                tape.gradient(disc_y_loss, self.discriminator_y.trainable_variables),
                self.discriminator_y.trainable_variables,
                strict=True,
            )
        )

        return {
            "gen_g_loss": total_gen_g_loss,
            "gen_f_loss": total_gen_f_loss,
            "disc_x_loss": disc_x_loss,
            "disc_y_loss": disc_y_loss,
        }

    def fit(
        self,
        domain_x: tf.data.Dataset,
        domain_y: tf.data.Dataset,
        epochs: int,
        checkpoint_interval: int = 5,
        on_epoch_end: Callable[[int, dict[str, tf.Tensor]], None] | None = None,
    ) -> None:
        start_epoch = int(self.checkpoint.epoch)
        for epoch in range(start_epoch, epochs):
            losses: dict[str, tf.Tensor] = {}
            for image_x, image_y in tf.data.Dataset.zip((domain_x, domain_y)):
                losses = self.train_step(image_x, image_y)
            self.checkpoint.epoch.assign(epoch + 1)
            logger.info("Epoch %d losses: %s", epoch + 1, {k: float(v) for k, v in losses.items()})
            if (epoch + 1) % checkpoint_interval == 0:
                save_path = self.checkpoint_manager.save()
                logger.info("Saved checkpoint at %s", save_path)
            if on_epoch_end is not None:
                on_epoch_end(epoch + 1, losses)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_training_service.py -v`
Expected: PASS (4 passed). `@tf.function` tracing on the first call adds a few seconds of one-time overhead per test — this is expected, not a hang.

- [ ] **Step 5: Commit**

```bash
git add src/services/training_service.py tests/unit/test_training_service.py
git commit -m "feat: add CycleGAN training service with checkpointing"
```

---

### Task 6: Narrative service (LLM vision commentary)

**Files:**
- Create: `src/services/narrative_service.py`
- Test: `tests/unit/test_narrative_service.py`

**Interfaces:**
- Consumes: `forge.llm.base.LLMProvider` (`.generate(prompt: str, system: str | None = None, images: list[bytes] | None = None, **kwargs) -> LLMResponse`, already exists in `sameer-forge` v0.5.1), `forge.vision.gridplot.plot_translation_grid(rows: list[tuple[str, np.ndarray]]) -> plt.Figure` (already exists, accepts numpy images as of v0.4.0).
- Produces: `CommentaryResult` (attributes: `commentary: str`, `grid_fig: plt.Figure`), `generate_commentary(input_image: np.ndarray, translated_image: np.ndarray, reference_image: np.ndarray, llm: LLMProvider) -> CommentaryResult`. Consumed by `src/app.py` (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_narrative_service.py
import numpy as np

from forge.llm.base import LLMResponse

from src.services.narrative_service import generate_commentary


class _StubLLM:
    def __init__(self, content="a description", raise_error=False):
        self._content = content
        self._raise_error = raise_error
        self.calls = []

    def generate(self, prompt, system=None, images=None, **kwargs):
        self.calls.append({"prompt": prompt, "images": images})
        if self._raise_error:
            raise RuntimeError("LLM unreachable")
        return LLMResponse(content=self._content, model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _tiny_image():
    return np.zeros((8, 8, 1), dtype=np.float32) - 1.0


def test_generate_commentary_passes_grid_image_to_llm():
    llm = _StubLLM(content="looks good")
    result = generate_commentary(_tiny_image(), _tiny_image(), _tiny_image(), llm)

    assert result.commentary == "looks good"
    assert result.grid_fig is not None
    assert len(llm.calls) == 1
    assert llm.calls[0]["images"] is not None
    assert len(llm.calls[0]["images"]) == 1
    assert isinstance(llm.calls[0]["images"][0], bytes)


def test_generate_commentary_falls_back_on_llm_failure():
    llm = _StubLLM(raise_error=True)
    result = generate_commentary(_tiny_image(), _tiny_image(), _tiny_image(), llm)

    assert result.commentary == "commentary unavailable"
    assert result.grid_fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_narrative_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.narrative_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/narrative_service.py
from __future__ import annotations

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
from forge.llm.base import LLMProvider
from forge.vision.gridplot import plot_translation_grid

logger = logging.getLogger(__name__)

COMMENTARY_PROMPT = (
    "This image shows a CycleGAN translation sample: input image, translated output, and "
    "reference image side by side. In 2-3 sentences, describe the translation quality — "
    "artifacts, texture transfer, and how close the translation looks to the reference."
)


class CommentaryResult:
    def __init__(self, commentary: str, grid_fig: plt.Figure) -> None:
        self.commentary = commentary
        self.grid_fig = grid_fig


def _figure_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def generate_commentary(
    input_image: np.ndarray,
    translated_image: np.ndarray,
    reference_image: np.ndarray,
    llm: LLMProvider,
) -> CommentaryResult:
    fig = plot_translation_grid(
        [
            ("Input", input_image),
            ("Translated", translated_image),
            ("Reference", reference_image),
        ]
    )
    try:
        png_bytes = _figure_to_png_bytes(fig)
        response = llm.generate(COMMENTARY_PROMPT, images=[png_bytes])
        commentary = response.content
    except Exception:
        logger.exception("Narrative commentary generation failed")
        commentary = "commentary unavailable"
    return CommentaryResult(commentary=commentary, grid_fig=fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_narrative_service.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/narrative_service.py tests/unit/test_narrative_service.py
git commit -m "feat: add narrative service with LLM vision commentary"
```

---

### Task 7: Report service

**Files:**
- Create: `src/services/report_service.py`
- Test: `tests/unit/test_report_service.py`

**Interfaces:**
- Consumes: `forge.report.builder.{ReportBuilder, ReportSection}` (already exist — `ReportBuilder(title: str, subtitle: str = "")`, `.add_section(ReportSection) -> ReportBuilder`, `.save(path) -> Path`; `ReportSection(title: str, content: str = "", figures: list = [], metrics: dict = {})`).
- Produces: `EpochReport` (attributes: `epoch: int`, `losses: dict[str, float]`, `grid_fig`, `commentary: str`), `build_report(title: str, epoch_reports: list[EpochReport]) -> ReportBuilder`. Consumed by `src/app.py` (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_report_service.py
import matplotlib.pyplot as plt

from src.services.report_service import EpochReport, build_report


def test_build_report_includes_each_epoch_section():
    fig, _ = plt.subplots()
    reports = [
        EpochReport(epoch=5, losses={"gen_g_loss": 1.2}, grid_fig=fig, commentary="improving"),
        EpochReport(epoch=10, losses={"gen_g_loss": 0.9}, grid_fig=fig, commentary="sharper edges"),
    ]

    report = build_report("Test Report", reports)
    html = report.to_html()

    assert "Epoch 5" in html
    assert "Epoch 10" in html
    assert "improving" in html
    assert "sharper edges" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_report_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.services.report_service'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/services/report_service.py
from __future__ import annotations

from forge.report.builder import ReportBuilder, ReportSection


class EpochReport:
    def __init__(self, epoch: int, losses: dict[str, float], grid_fig, commentary: str) -> None:
        self.epoch = epoch
        self.losses = losses
        self.grid_fig = grid_fig
        self.commentary = commentary


def build_report(title: str, epoch_reports: list[EpochReport]) -> ReportBuilder:
    report = ReportBuilder(title)
    for entry in epoch_reports:
        section = ReportSection(
            title=f"Epoch {entry.epoch}",
            content=entry.commentary,
            metrics=entry.losses,
            figures=[entry.grid_fig],
        )
        report.add_section(section)
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_report_service.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/services/report_service.py tests/unit/test_report_service.py
git commit -m "feat: add report assembly service"
```

---

### Task 8: CLI driver + integration test

**Files:**
- Create: `src/app.py`
- Test: `tests/integration/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 2-7 — `src.providers.pix2pix_factory.Pix2PixGeneratorFactory`, `src.services.data_service.load_domain`, `src.services.training_service.CycleGANTrainer`, `src.services.narrative_service.generate_commentary`, `src.services.report_service.{EpochReport, build_report}`, plus `forge.llm.{ollama.OllamaProvider, claude.ClaudeProvider, openai_provider.OpenAIProvider}` (all three constructible with zero required arguments).
- Produces: `src/app.py`'s `main(argv: list[str] | None = None) -> int`, module-level `FACTORY_CLS: type` and `LLM_PROVIDERS: dict[str, type]` (deliberately patchable module attributes — the integration test monkeypatches them to avoid real TensorFlow model training and real LLM calls).

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_pipeline.py
import numpy as np
import tensorflow as tf
from PIL import Image

from forge.llm.base import LLMResponse

from src import app
from src.core.interfaces import GeneratorFactory


class _StubFactory(GeneratorFactory):
    def build_generator(self, output_channels):
        inputs = tf.keras.Input(shape=(None, None, 1))
        outputs = tf.keras.layers.Conv2D(output_channels, 3, padding="same", activation="tanh")(inputs)
        return tf.keras.Model(inputs, outputs)

    def build_discriminator(self):
        inputs = tf.keras.Input(shape=(None, None, 1))
        outputs = tf.keras.layers.Conv2D(1, 3, padding="same")(inputs)
        return tf.keras.Model(inputs, outputs)


class _StubLLM:
    def generate(self, prompt, system=None, images=None, **kwargs):
        return LLMResponse(content="stub commentary", model="stub", input_tokens=1, output_tokens=1)

    def chat(self, messages, system=None, **kwargs):
        return self.generate(messages[-1]["content"])


def _write_domain(directory, count=3, size=8):
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        array = np.full((size, size), fill_value=i * 40, dtype=np.uint8)
        Image.fromarray(array, mode="L").save(directory / f"img{i}.png")


def test_pipeline_writes_report_and_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "FACTORY_CLS", _StubFactory)
    monkeypatch.setitem(app.LLM_PROVIDERS, "ollama", _StubLLM)

    domain_x_dir = tmp_path / "domain_x"
    domain_y_dir = tmp_path / "domain_y"
    _write_domain(domain_x_dir)
    _write_domain(domain_y_dir)

    checkpoint_dir = tmp_path / "checkpoints"
    output_path = tmp_path / "report.html"

    exit_code = app.main(
        [
            "--domain-x-dir", str(domain_x_dir),
            "--domain-y-dir", str(domain_y_dir),
            "--image-size", "8",
            "--epochs", "1",
            "--checkpoint-dir", str(checkpoint_dir),
            "--checkpoint-interval", "1",
            "--sample-interval", "1",
            "--llm-provider", "ollama",
            "--output", str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert checkpoint_dir.exists()
    html = output_path.read_text()
    assert "Epoch 1" in html
    assert "stub commentary" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.app'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/app.py
from __future__ import annotations

import argparse
import logging

from forge.llm.claude import ClaudeProvider
from forge.llm.ollama import OllamaProvider
from forge.llm.openai_provider import OpenAIProvider

from src.providers.pix2pix_factory import Pix2PixGeneratorFactory
from src.services.data_service import load_domain
from src.services.narrative_service import generate_commentary
from src.services.report_service import EpochReport, build_report
from src.services.training_service import CycleGANTrainer

logger = logging.getLogger(__name__)

FACTORY_CLS: type = Pix2PixGeneratorFactory

LLM_PROVIDERS: dict[str, type] = {
    "ollama": OllamaProvider,
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pixel-drift")
    parser.add_argument("--domain-x-dir", default="src/data/domain_x")
    parser.add_argument("--domain-y-dir", default="src/data/domain_y")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--llm-provider", choices=list(LLM_PROVIDERS.keys()), default="ollama")
    parser.add_argument("--output", default="reports/model_report.html")
    args = parser.parse_args(argv)

    domain_x = load_domain(args.domain_x_dir, image_size=args.image_size)
    domain_y = load_domain(args.domain_y_dir, image_size=args.image_size)

    sample_x = next(iter(domain_x))
    sample_y = next(iter(domain_y))

    trainer = CycleGANTrainer(FACTORY_CLS(), checkpoint_dir=args.checkpoint_dir)
    llm = LLM_PROVIDERS[args.llm_provider]()

    epoch_reports: list[EpochReport] = []

    def on_epoch_end(epoch: int, losses: dict) -> None:
        if epoch % args.sample_interval != 0:
            return
        translated = trainer.generator_g(sample_x, training=False)
        result = generate_commentary(
            sample_x[0].numpy(), translated[0].numpy(), sample_y[0].numpy(), llm
        )
        epoch_reports.append(
            EpochReport(
                epoch=epoch,
                losses={k: float(v) for k, v in losses.items()},
                grid_fig=result.grid_fig,
                commentary=result.commentary,
            )
        )

    trainer.fit(
        domain_x,
        domain_y,
        epochs=args.epochs,
        checkpoint_interval=args.checkpoint_interval,
        on_epoch_end=on_epoch_end,
    )

    report = build_report("PixelDrift — CycleGAN Training Report", epoch_reports)
    report.save(args.output)
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_pipeline.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests across every task pass, no regressions. This full run includes Task 3's real `tensorflow_examples` model construction and Task 5's real `tf.function` tracing — expect it to take a minute or two, not because anything is wrong.

- [ ] **Step 6: Commit**

```bash
git add src/app.py tests/integration/test_pipeline.py
git commit -m "feat: add CLI driver"
```

---

### Task 9: README and task.md

**Files:**
- Modify: `README.md`, `task.md`, `pyproject.toml` (fill in the empty `description` field)

**Interfaces:**
- Consumes: nothing new — documents Tasks 1-8.

- [ ] **Step 1: Get the actual test count**

Run: `pytest tests/ --collect-only -q | tail -1`

- [ ] **Step 2: Rewrite `README.md`**

Replace the scaffold's placeholder content with:

```markdown
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

<!-- test count filled in from Step 1's actual pytest output -->
```

(Replace the HTML comment placeholder with the real count as a sentence, e.g. "24 tests, zero real network/LLM/TensorFlow-training calls beyond tiny synthetic smoke tests." — use the number from Step 1, not a guess.)

- [ ] **Step 3: Update `task.md`**

Replace the empty `## Backlog` section with:

```markdown
## Backlog

- [x] Core `GeneratorFactory` interface
- [x] Pix2Pix provider (tensorflow_examples wrapper, grayscale channel adaptation)
- [x] Data service (domain loading, normalization)
- [x] Training service (CycleGAN loop, real checkpointing)
- [x] Narrative service (LLM vision commentary)
- [x] Report service
- [x] CLI driver
- [ ] Real end-to-end run against actual domain_x/domain_y image data (source unrecoverable from the original notebook, not exercised in CI)
- [ ] Real LLM commentary review against actual Ollama/Claude/OpenAI vision output (stubbed in tests)
```

- [ ] **Step 4: Fill in `pyproject.toml`'s description**

Change:
```toml
description = ""
```
to:
```toml
description = "CycleGAN unpaired image translation with GenAI translation-quality commentary, built on sameer-forge."
```

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/ -v` (confirm nothing broke) and re-read the edited `README.md` once to confirm no broken code fences or headers.

```bash
git add README.md task.md pyproject.toml
git commit -m "docs: document PixelDrift usage and architecture"
```

---

## Self-Review Notes

- Spec coverage: `GeneratorFactory` interface (Task 2), `Pix2PixGeneratorFactory` wrapping `tensorflow_examples` with self-adapting grayscale channel handling (Task 3 — resolves the uncertainty about whether the real package hardcodes 3-channel input, discovered during plan-writing, without blocking on it or guessing wrong), data loading/normalization (Task 4), CycleGAN training loop with real checkpointing and auto-resume (Task 5), LLM vision commentary via `forge.llm`'s multimodal `generate()` and `forge.vision.gridplot`'s numpy support (Task 6), report assembly (Task 7), CLI driver wiring everything together with `--image-size` added during planning so tiny synthetic test data doesn't force a 256×256 resize (Task 8), README/task.md consistency (Task 9). Repo genesis pushing into the pre-existing empty GitHub repo (Task 1) covers the "repo is now created" state discovered before this plan was written.
- Type consistency checked: `CycleGANTrainer.fit()`'s `on_epoch_end` signature (`Callable[[int, dict[str, tf.Tensor]], None]`, Task 5) matches how `src/app.py` (Task 8) defines and passes its `on_epoch_end` closure. `CommentaryResult`'s `commentary`/`grid_fig` fields (Task 6) match `EpochReport`'s consumption of `result.grid_fig`/`result.commentary` (Task 8). `GeneratorFactory.build_generator`/`build_discriminator` signatures match across the interface (Task 2), `Pix2PixGeneratorFactory` (Task 3), and every stub factory used in tests (Tasks 2, 5, 8). `FACTORY_CLS`/`LLM_PROVIDERS` module attributes (Task 8) match what the integration test monkeypatches.
- Fixed during planning: discovered that `tensorflow_examples`' pix2pix generator/discriminator likely hardcode 3-channel (RGB) input regardless of the `output_channels` argument, which only controls output — our data is 1-channel grayscale. Rather than guessing or blocking on this, `Pix2PixGeneratorFactory` self-adapts by inspecting the real constructed model's `input_shape[-1]` at construction time and wrapping with a channel-tiling adapter only if needed — correct whether or not the installed package turns out to already be channel-flexible.
