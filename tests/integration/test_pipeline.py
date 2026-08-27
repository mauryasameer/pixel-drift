import numpy as np
import tensorflow as tf
from meerax.llm.base import LLMResponse
from PIL import Image

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
            "--image-size", "256",
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
