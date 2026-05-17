from __future__ import annotations

import os
import tensorflow as tf

from ..models.rnaimage_model import Bio_Generator, Noise_Encoder, Reconstructor
from ..models import task_model, drug_model
from ..resources import ModelPaths


DEFAULT_MODELS = ModelPaths()


def _resolve_ckpt_path(ckpt_path: str | None, default_path: str) -> str:
    return str(ckpt_path or default_path)


def load_model(ckpt='latest', ckpt_path=None, z_dim=100, only_encoder=False):
    ckpt_path = _resolve_ckpt_path(ckpt_path, str(DEFAULT_MODELS.batch_correction))
    bio_generator = Bio_Generator((-1, 128, 256, 1), z_dim)
    checkpoint = tf.train.Checkpoint(bio_generator=bio_generator)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return bio_generator.get_layer(index=0) if only_encoder else bio_generator


def load_batch_model(ckpt='latest', ckpt_path=None, z_dim=100):
    ckpt_path = _resolve_ckpt_path(ckpt_path, str(DEFAULT_MODELS.batch_correction))
    batch_encoder = Noise_Encoder((-1, 128, 256, 1), z_dim)
    checkpoint = tf.train.Checkpoint(batch_encoder=batch_encoder)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return batch_encoder


def load_generator_model(ckpt='latest', ckpt_path=None, z_dim=100):
    ckpt_path = _resolve_ckpt_path(ckpt_path, str(DEFAULT_MODELS.batch_correction))
    reconstructor = Reconstructor((-1, 128, 256, 1), z_dim)
    checkpoint = tf.train.Checkpoint(reconstructor=reconstructor)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return reconstructor


def load_predict_model(ckpt, ckpt_path, out_dim, activation):
    classifier = task_model.make_finetune_model(out_dim, activation=activation)
    checkpoint = tf.train.Checkpoint(classifier=classifier)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return classifier


def load_risk_model(ckpt, ckpt_path, out_dim, activation):
    model = task_model.make_finetune_model(out_dim, activation=activation)
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return model


def load_drug_model(ckpt, ckpt_path, activation):
    drug_gcn = drug_model.GCNModel()
    regression_model = drug_model.drug_FCN(activation=activation)
    checkpoint = tf.train.Checkpoint(regression_model=regression_model, drug_gcn=drug_gcn)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return drug_gcn, regression_model


def load_alignment_predictor(ckpt, ckpt_path, output_dim):
    model = task_model.make_AlignmentPredictor(output_dim=output_dim, activation='softmax', trainable_layer='none', pretrain='v2')
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_full = tf.train.latest_checkpoint(ckpt_path) if ckpt == 'latest' else os.path.join(ckpt_path, ckpt)
    checkpoint.restore(ckpt_full).expect_partial()
    return model
