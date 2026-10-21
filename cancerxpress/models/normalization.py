"""Channels-last group normalization compatible with the released checkpoints."""
import tensorflow as tf

from .._keras import keras


@keras.utils.register_keras_serializable(package="CancerXpress")
class GroupNormalization(keras.layers.Layer):
    """Preserve the original groups, epsilon and gamma/beta checkpoint attributes."""

    def __init__(self, groups=32, epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(groups, int) or groups < 1:
            raise ValueError("groups must be a positive integer")
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None or channels % self.groups:
            raise ValueError("The channel count must be known and divisible by groups")
        self.channels = int(channels)
        self.input_spec = keras.layers.InputSpec(ndim=len(input_shape),
                                                axes={-1: self.channels})
        self.gamma = self.add_weight(name="gamma", shape=(self.channels,),
                                     initializer="ones")
        self.beta = self.add_weight(name="beta", shape=(self.channels,),
                                    initializer="zeros")
        super().build(input_shape)

    def call(self, inputs):
        shape = tf.shape(inputs)
        grouped = tf.reshape(inputs, tf.concat(
            [shape[:-1], [self.groups, self.channels // self.groups]], axis=0))
        rank = inputs.shape.rank
        axes = list(range(1, rank - 1)) + [rank]
        mean, variance = tf.nn.moments(grouped, axes=axes, keepdims=True)
        weight_shape = [1] * (rank - 1) + [self.groups, self.channels // self.groups]
        normalized = tf.nn.batch_normalization(
            grouped, mean, variance, tf.reshape(self.beta, weight_shape),
            tf.reshape(self.gamma, weight_shape), self.epsilon)
        return tf.reshape(normalized, shape)

    def get_config(self):
        return dict(super().get_config(), groups=self.groups, epsilon=self.epsilon)
