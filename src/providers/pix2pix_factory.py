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
