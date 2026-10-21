import tempfile
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf

from cancerxpress._keras import keras
from cancerxpress.models.normalization import GroupNormalization


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_group_norm_values_and_gradients(self):
        values = np.random.RandomState(9).normal(size=(2, 3, 4, 64)).astype("float32")
        layer = GroupNormalization()
        x = tf.Variable(values)
        with tf.GradientTape() as tape:
            actual = layer(x)
            loss = tf.reduce_sum(actual * x)
        self.assertTrue(np.isfinite(tape.gradient(loss, x).numpy()).all())
        groups = values.reshape(2, 3, 4, 32, 2)
        expected = (groups - groups.mean(axis=(1, 2, 4), keepdims=True))
        expected /= np.sqrt(groups.var(axis=(1, 2, 4), keepdims=True) + 1e-3)
        np.testing.assert_allclose(actual.numpy(), expected.reshape(values.shape),
                                   rtol=1e-5, atol=1e-6)

    def test_group_norm_checkpoint_and_config(self):
        x = tf.ones((1, 2, 2, 64))
        layer = GroupNormalization(groups=16)
        layer(x)
        layer.gamma.assign(tf.fill((64,), 2.0))
        layer.beta.assign(tf.fill((64,), 0.5))
        restored = GroupNormalization.from_config(layer.get_config())
        restored(x)
        with tempfile.TemporaryDirectory() as directory:
            path = tf.train.Checkpoint(layer=layer).save(str(Path(directory) / "model"))
            tf.train.Checkpoint(layer=restored).restore(path).assert_consumed()
        np.testing.assert_allclose(restored(x), layer(x))
        self.assertIsInstance(restored, keras.layers.Layer)

    def test_group_norm_bad_channels(self):
        with self.assertRaises(ValueError):
            GroupNormalization(groups=3)(tf.ones((1, 2, 2, 64)))


if __name__ == "__main__":
    unittest.main()
