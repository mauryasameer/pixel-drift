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
