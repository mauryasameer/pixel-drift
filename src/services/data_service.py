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
