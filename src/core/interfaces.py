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
