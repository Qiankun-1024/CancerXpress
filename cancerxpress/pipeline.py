from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .models.rnaimage_model import Bio_Generator
from .models.task_model import FCN, make_finetune_model
from .attribution import AttributionResult, CancerXpressAttributor
from .predictor import Predictor
from .preprocess import ExpressionPreprocessor
from .resources import ModelPaths, ResourcePaths, load_risk_cancer_vocab


ME_COLUMNS = [
    'MEblack', 'MEblue', 'MEbrown', 'MEgreen',
    'MEpink', 'MEred', 'MEturquoise', 'MEyellow',
]


@dataclass
class CancerXpressResult:
    latent: Optional[pd.DataFrame] = None
    corrected_expression: Optional[pd.DataFrame] = None
    me: Optional[pd.DataFrame] = None
    primary_site: Optional[pd.DataFrame] = None
    cancer_type: Optional[pd.DataFrame] = None
    survival_risk: Optional[pd.DataFrame] = None


class CancerXpress:
    def __init__(
        self,
        model_paths: ModelPaths | None = None,
        resource_paths: ResourcePaths | None = None,
    ):
        self.model_paths = model_paths or ModelPaths()
        self.resource_paths = resource_paths or ResourcePaths()
        self.preprocessor = ExpressionPreprocessor()
        self.attributor = CancerXpressAttributor(self.model_paths, self.resource_paths)

    @staticmethod
    def _latest_checkpoint(ckpt_dir: Path) -> str:
        ckpt = tf.train.latest_checkpoint(str(ckpt_dir))
        if ckpt is None:
            raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
        return ckpt

    def _load_generator(self) -> Bio_Generator:
        model = Bio_Generator((-1, 128, 256, 1), 100)
        ckpt = tf.train.Checkpoint(bio_generator=model)
        ckpt.restore(self._latest_checkpoint(self.model_paths.batch_correction)).expect_partial()
        return model

    def _load_classifier(self, out_dim: int, activation: Optional[str], finetune_dir: Path) -> FCN:
        model = make_finetune_model(out_dim, activation=activation, trainable_layer='none', pretrain='v2')
        latest = self._latest_checkpoint(finetune_dir)
        checkpoint = tf.train.Checkpoint(classifier=model)
        checkpoint.restore(latest).expect_partial()
        return model

    @staticmethod
    def _batch_predict(model: tf.keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        out = []
        for batch in ds:
            pred = model(batch, training=False)
            out.append(pred.numpy())
        return np.concatenate(out, axis=0)

    def harmonize_expression(self, expr: pd.DataFrame, gene_id_type: str = 'auto') -> pd.DataFrame:
        return self.preprocessor.harmonize_expression(expr, gene_id_type=gene_id_type)

    def batch_correct(self, expr_tpm: pd.DataFrame, batch_size: int = 256, gene_id_type: str = 'auto') -> tuple[pd.DataFrame, pd.DataFrame]:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        x = self.preprocessor.gene2img(expr, gene_id_type='ensembl')
        generator = self._load_generator()
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        latent_list, observed_list = [], []
        for batch in ds:
            latent = generator.encode(batch, reparameterize=True)
            observed = generator.decode(latent, apply_sigmoid=True)
            latent_list.append(latent.numpy())
            observed_list.append(observed.numpy())
        latent = np.concatenate(latent_list, axis=0)
        observed = np.concatenate(observed_list, axis=0)
        observed_1d = self.preprocessor.trans_2d_to_1d(observed)
        observed_log2 = self.preprocessor.reverse_log2_feature(observed_1d)
        observed_tpm = self.preprocessor.reverse_tpm(observed_log2)
        latent_df = pd.DataFrame(latent, index=expr.index, columns=[f'latent_{i}' for i in range(latent.shape[1])])
        observed_tpm.index = expr.index
        return latent_df, observed_tpm

    def predict_me(self, expr_tpm: pd.DataFrame, batch_size: int = 256, gene_id_type: str = 'auto') -> pd.DataFrame:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr)
        return predictor.predict_pathway_axis(batch_size=batch_size, return_type='score')

    def predict_primary_site(self, expr_tpm: pd.DataFrame, batch_size: int = 256, gene_id_type: str = 'auto', return_type: str = 'label') -> pd.DataFrame:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr)
        return predictor.predict(label_types='primary_site', batch_size=batch_size, return_type=return_type)

    def predict_cancer_type(self, expr_tpm: pd.DataFrame, batch_size: int = 256, gene_id_type: str = 'auto', return_type: str = 'label') -> pd.DataFrame:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr)
        return predictor.predict(label_types='cancer_type', batch_size=batch_size, return_type=return_type)

    def predict_survival_risk(
        self,
        expr_tpm: pd.DataFrame,
        cancer_type: Optional[Sequence[str] | pd.Series] = None,
        batch_size: int = 256,
        gene_id_type: str = 'auto',
    ) -> pd.DataFrame:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr)
        vocab = load_risk_cancer_vocab(self.resource_paths.risk_cancer_vocab)
        return predictor.predict_survival_risk(
            batch_size=batch_size,
            cancer_type=cancer_type,
            cancer_vocab=vocab,
            ckpt_path=str(self.model_paths.survival_risk),
        )

    def attribute_me(self, expr_tpm: pd.DataFrame, me_name: str, steps: int = 50, gene_id_type: str = 'auto') -> AttributionResult:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_me(expr, me_name=me_name, steps=steps)

    def attribute_primary_site(
        self,
        expr_tpm: pd.DataFrame,
        target_name: Optional[str] = None,
        steps: int = 50,
        gene_id_type: str = 'auto',
    ) -> AttributionResult:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_primary_site(expr, target_name=target_name, steps=steps)

    def attribute_cancer_type(
        self,
        expr_tpm: pd.DataFrame,
        target_name: Optional[str] = None,
        steps: int = 50,
        gene_id_type: str = 'auto',
    ) -> AttributionResult:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_cancer_type(expr, target_name=target_name, steps=steps)

    def attribute_survival_risk(
        self,
        expr_tpm: pd.DataFrame,
        cancer_type: Optional[Sequence[str] | pd.Series] = None,
        steps: int = 50,
        gene_id_type: str = 'auto',
    ) -> AttributionResult:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        if cancer_type is None:
            raise ValueError('Survival risk attribution requires cancer_type.')
        return self.attributor.attribute_survival_risk(expr, cancer_type=cancer_type, steps=steps)

    def run_all(
        self,
        expr_tpm: pd.DataFrame,
        cancer_type: Optional[Sequence[str] | pd.Series] = None,
        batch_size: int = 256,
        gene_id_type: str = 'auto',
    ) -> CancerXpressResult:
        latent, corrected = self.batch_correct(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        me = self.predict_me(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        primary = self.predict_primary_site(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        cancer = self.predict_cancer_type(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        risk = self.predict_survival_risk(expr_tpm, cancer_type=cancer_type, batch_size=batch_size, gene_id_type=gene_id_type)
        return CancerXpressResult(
            latent=latent,
            corrected_expression=corrected,
            me=me,
            primary_site=primary,
            cancer_type=cancer,
            survival_risk=risk,
        )
