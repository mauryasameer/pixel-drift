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
