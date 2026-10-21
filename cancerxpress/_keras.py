"""Use checkpoint-compatible Keras without changing TensorFlow's global backend."""
import tensorflow as tf

if tuple(int(v) for v in tf.__version__.split(".")[:2]) >= (2, 16):
    try:
        import tf_keras as keras
    except ImportError as exc:
        raise ImportError(
            "CancerXpress needs tf-keras with TensorFlow >=2.16 to load its "
            "Keras 2 checkpoints. Install CancerXpress with its dependencies."
        ) from exc
else:
    keras = tf.keras
