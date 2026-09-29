from __future__ import annotations

import tensorflow as tf


class ClinicalMERiskModel(tf.keras.Model):
    """Run1 Cox model using predicted ME scores and clinical covariates."""

    def __init__(self, n_sex: int, n_race: int, n_stage: int, n_cancer: int):
        super().__init__()
        self.sex_emb = tf.keras.layers.Embedding(n_sex, 2)
        self.race_emb = tf.keras.layers.Embedding(n_race, 4)
        self.stage_emb = tf.keras.layers.Embedding(n_stage, 4)
        self.cancer_emb = tf.keras.layers.Embedding(n_cancer, 16)
        self.core = tf.keras.Sequential([
            tf.keras.layers.Dense(128),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(64),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(alpha=0.1),
        ])
        self.out = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training=False):
        features = tf.concat([
            inputs["me"],
            inputs["age"],
            self.sex_emb(inputs["sex"]),
            self.race_emb(inputs["race"]),
            self.stage_emb(inputs["stage"]),
            self.cancer_emb(inputs["cancer"]),
        ], axis=1)
        return self.out(self.core(features, training=training))
