from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .predictor import Predictor
from .resources import ModelPaths, ResourcePaths, load_risk_cancer_vocab
from .utils import model_loader


@dataclass
class AttributionResult:
    attributions: pd.DataFrame
    predicted_output: pd.DataFrame
    target_name: str


class IntegratedGradients:
    def __init__(self, model, baseline=None, steps: int = 50):
        self.model = model
        self.steps = steps
        self.baseline = baseline

    def _interpolate_inputs(self, inputs: tf.Tensor) -> tf.Tensor:
        baseline = self.baseline if self.baseline is not None else tf.zeros_like(inputs)
        alphas = tf.linspace(0.0, 1.0, self.steps + 1)
        interpolated_inputs = [baseline + alpha * (inputs - baseline) for alpha in alphas]
        return tf.concat(interpolated_inputs, axis=0)

    def compute_attributions(self, inputs: tf.Tensor, target_fn) -> np.ndarray:
        interpolated_inputs = self._interpolate_inputs(inputs)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(interpolated_inputs)
            target_predictions = target_fn(interpolated_inputs)
        gradients = tape.gradient(target_predictions, interpolated_inputs)
        grads_per_step = tf.split(gradients, self.steps + 1)
        avg_gradients = tf.reduce_mean(grads_per_step[:-1], axis=0)
        baseline = self.baseline if self.baseline is not None else tf.zeros_like(inputs)
        attributions = (inputs - baseline) * avg_gradients
        return attributions.numpy()


class CancerXpressAttributor:
    def __init__(self, model_paths: ModelPaths | None = None, resource_paths: ResourcePaths | None = None):
        self.model_paths = model_paths or ModelPaths()
        self.resource_paths = resource_paths or ResourcePaths()

    @staticmethod
    def _ensure_single_sample(expr: pd.DataFrame) -> pd.DataFrame:
        if len(expr) != 1:
            raise ValueError("Attribution currently expects exactly one sample.")
        return expr

    @staticmethod
    def _to_gene_level(expr: pd.DataFrame, attrs_4d: np.ndarray) -> pd.DataFrame:
        from .utils import data_normalizer
        gene_level = data_normalizer.trans_2d_to_1d(attrs_4d)
        gene_level.index = expr.index
        return gene_level

    @staticmethod
    def _classifier_target_fn(model, class_idx: int):
        def fn(x):
            preds = model(x, training=False)
            return preds[:, class_idx]
        return fn

    @staticmethod
    def _risk_target_fn(model, cancer_idx: Optional[tf.Tensor] = None):
        def fn(x):
            if cancer_idx is None:
                preds = model(x, training=False)
            else:
                tiled_idx = tf.repeat(cancer_idx, repeats=tf.shape(x)[0], axis=0)
                preds = model((x, tiled_idx), training=False)
            return tf.reshape(preds, (-1,))
        return fn

    def attribute_me(self, expr: pd.DataFrame, me_name: str, steps: int = 50) -> AttributionResult:
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr)
        model = model_loader.load_predict_model("latest", str(self.model_paths.me), 8, None)
        preds = model(predictor.data, training=False).numpy()
        me_names = ['MEblack', 'MEblue', 'MEbrown', 'MEgreen', 'MEpink', 'MEred', 'MEturquoise', 'MEyellow']
        if me_name not in me_names:
            raise ValueError(f"Unsupported me_name: {me_name}")
        me_idx = me_names.index(me_name)
        ig = IntegratedGradients(model, steps=steps)
        attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), self._classifier_target_fn(model, me_idx))
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame(preds, index=expr.index, columns=me_names),
            target_name=me_name,
        )

    def attribute_primary_site(self, expr: pd.DataFrame, target_name: Optional[str] = None, steps: int = 50) -> AttributionResult:
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr)
        model = model_loader.load_predict_model("latest", str(self.model_paths.primary_site), 35, "softmax")
        preds = model(predictor.data, training=False).numpy()
        encoder = joblib.load(self.resource_paths.primary_site_encoder)
        class_names = list(encoder.categories_[0])
        if target_name is None:
            target_idx = int(np.argmax(preds[0]))
            target_name = class_names[target_idx]
        else:
            target_idx = class_names.index(target_name)
        ig = IntegratedGradients(model, steps=steps)
        attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), self._classifier_target_fn(model, target_idx))
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame(preds, index=expr.index, columns=class_names),
            target_name=target_name,
        )

    def attribute_cancer_type(self, expr: pd.DataFrame, target_name: Optional[str] = None, steps: int = 50) -> AttributionResult:
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr)
        model = model_loader.load_predict_model("latest", str(self.model_paths.cancer_type), 41, "softmax")
        preds = model(predictor.data, training=False).numpy()
        encoder = joblib.load(self.resource_paths.cancer_type_encoder)
        class_names = list(encoder.categories_[0])
        if target_name is None:
            target_idx = int(np.argmax(preds[0]))
            target_name = class_names[target_idx]
        else:
            target_idx = class_names.index(target_name)
        ig = IntegratedGradients(model, steps=steps)
        attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), self._classifier_target_fn(model, target_idx))
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame(preds, index=expr.index, columns=class_names),
            target_name=target_name,
        )

    def attribute_survival_risk(
        self,
        expr: pd.DataFrame,
        cancer_type: Sequence[str] | pd.Series,
        steps: int = 50,
    ) -> AttributionResult:
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr)
        vocab = load_risk_cancer_vocab(self.resource_paths.risk_cancer_vocab)
        mapping = {ct: i for i, ct in enumerate(vocab)}
        cancer_series = predictor._normalize_cancer_type(cancer_type, expr.index)
        unknown_idx = len(vocab)
        cancer_idx = cancer_series.map(mapping).fillna(unknown_idx).astype(np.int32).to_numpy()
        ckpt_path = str(self.model_paths.survival_risk)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        ckpt_vars = dict(tf.train.list_variables(latest_ckpt))
        has_cancer_embedding = 'model/cancer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE' in ckpt_vars
        if has_cancer_embedding:
            model = predictor._build_survival_model_with_cancer_cov(len(vocab), str(self.model_paths.batch_correction))
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(latest_ckpt).expect_partial()
            risk_scores = model((predictor.data, cancer_idx), training=False).numpy().reshape(-1)
            ig = IntegratedGradients(model, steps=steps)
            attrs = ig.compute_attributions(
                tf.convert_to_tensor(predictor.data),
                self._risk_target_fn(model, tf.convert_to_tensor(cancer_idx, dtype=tf.int32)),
            )
        else:
            model, _ = predictor._load_survival_model_single_input(ckpt_path)
            risk_scores = model(predictor.data, training=False).numpy().reshape(-1)
            ig = IntegratedGradients(model, steps=steps)
            attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), self._risk_target_fn(model))
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame({'risk_score': risk_scores}, index=expr.index),
            target_name='survival_risk',
        )
