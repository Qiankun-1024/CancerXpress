from __future__ import annotations

import tensorflow as tf
from .._keras import keras


class ClinicalAxisRiskModel(keras.Model):
    """Run1 Cox model using predicted functional-axis scores and clinical covariates."""

    def __init__(self, n_sex: int, n_race: int, n_stage: int, n_cancer: int):
        super().__init__()
        self.sex_emb = keras.layers.Embedding(n_sex, 2)
        self.race_emb = keras.layers.Embedding(n_race, 4)
        self.stage_emb = keras.layers.Embedding(n_stage, 4)
        self.cancer_emb = keras.layers.Embedding(n_cancer, 16)
        self.core = keras.Sequential([
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.35),
            keras.layers.Dense(64),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32),
            keras.layers.LeakyReLU(alpha=0.1),
        ])
        self.out = keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        features = tf.concat([
            inputs["functional_axes"],
            inputs["age"],
            self.sex_emb(inputs["sex"]),
            self.race_emb(inputs["race"]),
            self.stage_emb(inputs["stage"]),
            self.cancer_emb(inputs["cancer"]),
        ], axis=1)
        return self.out(self.core(features, training=training))
