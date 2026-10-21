from __future__ import annotations

import tensorflow as tf
from .._keras import keras

from ..resources import ModelPaths
from ..utils.model_loader import load_model





class FCN(keras.Model):
    def __init__(self, encoder_layer, output_dim, activation):
        super().__init__()
        self.encoder = encoder_layer
        self.fc_layer = keras.Sequential([
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64),
            keras.layers.LeakyReLU(alpha=0.1),
        ])
        self.out_layer = keras.Sequential([keras.layers.Dense(output_dim, activation=activation)])

    def call(self, x, training=False):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        logits = self.fc_layer(mean, training=training)
        return self.out_layer(logits, training=training)


def get_encoder_layer(trainable_layer='none', ckpt_path=None):
    ckpt_path = ckpt_path or str(ModelPaths().batch_correction)
    bio_generator = load_model(ckpt_path=ckpt_path)
    encoder = bio_generator.get_layer(index=0)
    if trainable_layer == 'all':
        encoder.trainable = True
    elif trainable_layer == 'none':
        encoder.trainable = False
    elif trainable_layer < len(encoder.layers):
        encoder.trainable = True
        for layer in encoder.layers[:(-1 * trainable_layer)]:
            layer.trainable = False
    return encoder


def make_finetune_model(output_dim, activation='softmax', trainable_layer='none', pretrain='v2', encoder_ckpt_path=None):
    if encoder_ckpt_path is not None:
        encoder_layer = get_encoder_layer(trainable_layer=trainable_layer, ckpt_path=encoder_ckpt_path)
    elif pretrain == 'v1':
        encoder_layer = get_encoder_layer(trainable_layer=trainable_layer)
    elif pretrain == 'v2':
        encoder_layer = get_encoder_layer(trainable_layer=trainable_layer, ckpt_path=str(ModelPaths().batch_correction))
    else:
        encoder_layer = get_encoder_layer(trainable_layer=trainable_layer)
    return FCN(encoder_layer, output_dim, activation)


class AlignmentPredictor(keras.Model):
    def __init__(self, encoder_layer, hidden_dims=[256, 128], output_dim=None, dropout_rate=0.3, activation='relu'):
        super().__init__()
        self.encoder = encoder_layer
        self.hidden_layers = []
        for h_dim in hidden_dims:
            self.hidden_layers.append(keras.layers.Dense(h_dim, activation=activation))
            self.hidden_layers.append(keras.layers.BatchNormalization())
            self.hidden_layers.append(keras.layers.Dropout(dropout_rate))
        self.output_layer = keras.layers.Dense(output_dim, activation='softmax') if output_dim is not None else None

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def call(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        latent = self.reparameterize(mean, logvar)
        for layer in self.hidden_layers:
            latent = layer(latent)
        output = self.output_layer(latent) if self.output_layer is not None else None
        return latent, output


def make_AlignmentPredictor(output_dim, activation='softmax', trainable_layer='none', pretrain='v2'):
    encoder_layer = get_encoder_layer(trainable_layer=trainable_layer, ckpt_path=str(ModelPaths().batch_correction))
    return AlignmentPredictor(encoder_layer=encoder_layer, output_dim=output_dim, activation=activation)
