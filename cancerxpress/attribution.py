from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from .predictor import Predictor
from .preprocess import ExpressionPreprocessor, PreprocessAssets
from .resources import ModelPaths, ResourcePaths, load_risk_preprocessing
from .utils import model_loader
from .axes import FUNCTIONAL_AXIS_NAMES, validate_axis_name
from .validation import validate_expression
from .risk_attribution import (
    AxisRiskAttributionResult, exact_axis_shapley,
    gene_axis_integrated_gradients, allocate_axis_risk,
)


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

    def attribute_axis_risk(self, expr, clinical, axis_name, reference_tpm,
                            steps=64, internal_batch_size=32,
                            maximum_relative_error=0.1):
        """Use reference TPM samples to construct a consistent internal baseline.

        Reference samples must already have the same harmonized genes as expr.
        The baseline is the mean of preprocessed reference images; its model
        prediction supplies the Shapley reference, NOT the mean predicted functional-axis.
        """
        expr = self._ensure_single_sample(expr)
        if (not isinstance(reference_tpm, pd.DataFrame) or reference_tpm.empty
                or reference_tpm.index.has_duplicates or reference_tpm.columns.has_duplicates):
            raise ValueError('reference_tpm must be a nonempty DataFrame with unique labels')
        if set(reference_tpm.columns) != set(expr.columns):
            raise ValueError('Reference and target must have the same harmonized genes')
        reference_tpm = reference_tpm.reindex(columns=expr.columns)
        values = reference_tpm.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('reference_tpm must contain finite nonnegative TPM')
        # Preprocess each sample before averaging; TPM normalization is nonlinear.
        image_sum = None
        for start in range(0, len(reference_tpm), 32):
            images = Predictor(reference_tpm.iloc[start:start + 32], model_paths=self.model_paths, resource_paths=self.resource_paths).data
            batch_sum = np.asarray(images, dtype=np.float64).sum(axis=0, keepdims=True)
            image_sum = batch_sum if image_sum is None else image_sum + batch_sum
        baseline_image = (image_sum / len(reference_tpm)).astype(np.float32)
        metadata = load_risk_preprocessing(self.resource_paths.risk_preprocessing)
        axis_model = model_loader.load_predict_model('latest', str(self.model_paths.functional_axes), 8, None, encoder_ckpt_path=str(self.model_paths.batch_correction))
        reference = pd.Series(np.asarray(axis_model(baseline_image, training=False))[0],
                              index=metadata['functional_axis_columns'])
        result = self.attribute_axis_risk_from_baseline(
            expr, clinical, axis_name, baseline_image, reference, steps,
            internal_batch_size, maximum_relative_error, _axis_model=axis_model)
        result.diagnostics['reference_n_samples'] = len(reference_tpm)
        result.diagnostics['baseline_method'] = 'mean_preprocessed_reference_TPM'
        result.diagnostics['baseline_axis_value'] = reference[axis_name]
        return result

    def attribute_axis_risk_from_baseline(self, expr, clinical, axis_name, baseline_image,
                            baseline_axes, steps=64, internal_batch_size=32,
                            maximum_relative_error=0.1, baseline_tolerance=1e-3,
                            _axis_model=None):
        """Single-sample gene -> functional-axis -> log-risk allocation with frozen baselines.

        baseline_axes is an axis-indexed Series in unscaled functional-axis units;
        baseline_image is a preprocessed model-input image (not raw TPM).
        All clinical covariates remain fixed during coalition evaluation.
        """
        expr = self._ensure_single_sample(expr)
        metadata = load_risk_preprocessing(self.resource_paths.risk_preprocessing)
        names = metadata['functional_axis_columns']
        if axis_name not in names:
            raise ValueError('Unsupported axis_name: ' + str(axis_name))
        if not isinstance(baseline_axes, pd.Series) or baseline_axes.index.has_duplicates:
            raise ValueError('baseline_axes must be a uniquely axis-indexed pandas Series')
        reference = baseline_axes.reindex(names).to_numpy(np.float32)
        if not np.isfinite(reference).all():
            raise ValueError('baseline_axes must contain all finite model functional-axis columns')
        if not np.isfinite(baseline_tolerance) or baseline_tolerance < 0:
            raise ValueError('baseline_tolerance must be finite and nonnegative')
        predictor = Predictor(expr, model_paths=self.model_paths, resource_paths=self.resource_paths)
        axis_model = _axis_model if _axis_model is not None else model_loader.load_predict_model(
            'latest', str(self.model_paths.functional_axes), 8, None, encoder_ckpt_path=str(self.model_paths.batch_correction))
        values = axis_model(predictor.data, training=False).numpy()
        frame = pd.DataFrame(values, index=expr.index, columns=names)
        features = predictor._prepare_run1_features(frame, clinical, metadata)
        risk_model, _ = predictor._load_run1_survival_model(str(self.model_paths.survival_risk), metadata)
        shapley = exact_axis_shapley(risk_model, features, reference)
        index = names.index(axis_name)
        ig = gene_axis_integrated_gradients(axis_model, predictor.data, baseline_image,
                                            index, steps, internal_batch_size)
        mismatch = ig.baseline_output - reference[index]
        if np.any(np.abs(mismatch) > baseline_tolerance):
            raise ValueError('IG baseline output does not match the selected Shapley functional-axis baseline; '
                             'use a matched baseline image and inspect baseline optimization QC')
        genes = self._to_gene_level(expr, ig.values)
        allocated, qc = allocate_axis_risk(
            genes.to_numpy(), shapley.values[:, index],
            ig.predicted_output - ig.baseline_output,
            maximum_relative_error=maximum_relative_error)
        qc.index = expr.index
        qc['baseline_axis_error'] = mismatch
        qc['baseline_log_risk'] = shapley.baseline_log_risk
        qc['predicted_log_risk'] = shapley.predicted_log_risk
        qc['shapley_completeness_error'] = shapley.completeness_error
        qc['image_ig_completeness_error'] = ig.completeness_error
        return AxisRiskAttributionResult(
            pd.DataFrame(shapley.values, index=expr.index, columns=names), genes,
            pd.DataFrame(allocated, index=expr.index, columns=genes.columns), qc, axis_name)

    @staticmethod
    def _ensure_single_sample(expr: pd.DataFrame) -> pd.DataFrame:
        validate_expression(expr)
        if len(expr) != 1:
            raise ValueError("Attribution currently expects exactly one sample.")
        return expr

    def _to_gene_level(self, expr: pd.DataFrame, attrs_4d: np.ndarray) -> pd.DataFrame:
        from .utils import data_normalizer
        rp = self.resource_paths
        processor = ExpressionPreprocessor(PreprocessAssets(rp.pixel_coords, rp.max_norm_tpm, rp.gene_id_map))
        gene_level = processor.trans_2d_to_1d(attrs_4d)
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

    def attribute_functional_axis(self, expr: pd.DataFrame, axis_name: str, steps: int = 50) -> AttributionResult:
        validate_axis_name(axis_name)
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr, model_paths=self.model_paths, resource_paths=self.resource_paths)
        model = model_loader.load_predict_model("latest", str(self.model_paths.functional_axes), 8, None, encoder_ckpt_path=str(self.model_paths.batch_correction))
        preds = model(predictor.data, training=False).numpy()
        axis_names = list(FUNCTIONAL_AXIS_NAMES)
        if axis_name not in axis_names:
            raise ValueError(f"Unsupported axis_name: {axis_name}")
        axis_index = axis_names.index(axis_name)
        ig = IntegratedGradients(model, steps=steps)
        attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), self._classifier_target_fn(model, axis_index))
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame(preds, index=expr.index, columns=axis_names),
            target_name=axis_name,
        )

    def attribute_survival_risk(
        self,
        expr: pd.DataFrame,
        clinical: pd.DataFrame,
        steps: int = 50,
    ) -> AttributionResult:
        expr = self._ensure_single_sample(expr)
        predictor = Predictor(expr, model_paths=self.model_paths, resource_paths=self.resource_paths)
        metadata = load_risk_preprocessing(self.resource_paths.risk_preprocessing)
        axis_model = model_loader.load_predict_model('latest', str(self.model_paths.functional_axes), 8, None, encoder_ckpt_path=str(self.model_paths.batch_correction))
        axis_values = axis_model(predictor.data, training=False).numpy()
        axis_frame = pd.DataFrame(axis_values, index=expr.index, columns=metadata['functional_axis_columns'])
        base_features = predictor._prepare_run1_features(axis_frame, clinical, metadata)
        risk_model, _ = predictor._load_run1_survival_model(str(self.model_paths.survival_risk), metadata)
        risk_scores = risk_model(base_features, training=False).numpy().reshape(-1)

        def target_fn(x):
            functional_axes = axis_model(x, training=False)
            repeats = tf.shape(x)[0]
            features = {'functional_axes': functional_axes}
            for name, value in base_features.items():
                if name != 'functional_axes':
                    features[name] = tf.repeat(value, repeats=repeats, axis=0)
            return tf.reshape(risk_model(features, training=False), (-1,))

        ig = IntegratedGradients(risk_model, steps=steps)
        attrs = ig.compute_attributions(tf.convert_to_tensor(predictor.data), target_fn)
        gene_level = self._to_gene_level(expr, attrs)
        return AttributionResult(
            attributions=gene_level,
            predicted_output=pd.DataFrame({'risk_score': risk_scores}, index=expr.index),
            target_name='survival_risk',
        )
