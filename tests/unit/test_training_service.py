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
