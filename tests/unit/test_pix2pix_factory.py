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
