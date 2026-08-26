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
