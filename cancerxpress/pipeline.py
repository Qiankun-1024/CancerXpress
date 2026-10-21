from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from ._keras import keras

from .models.rnaimage_model import Bio_Generator
from .attribution import AttributionResult, CancerXpressAttributor
from .predictor import Predictor
from .preprocess import ExpressionPreprocessor, PreprocessAssets
from .resources import ModelPaths, ResourcePaths, resource_paths_from_directory
from .validation import validate_expression, align_clinical
from .axes import FUNCTIONAL_AXIS_NAMES, validate_axis_name


@dataclass
class CancerXpressResult:
    latent: Optional[pd.DataFrame] = None
    corrected_expression: Optional[pd.DataFrame] = None
    functional_axes: Optional[pd.DataFrame] = None
    survival_risk: Optional[pd.DataFrame] = None


class CancerXpress:
    def __init__(
        self,
        model_paths: ModelPaths | None = None,
        resource_paths: ResourcePaths | None = None,
        model_dir=None, resource_dir=None,
    ):
        if model_paths is not None and model_dir is not None:
            raise ValueError('Use model_paths or model_dir, not both')
        if resource_paths is not None and resource_dir is not None:
            raise ValueError('Use resource_paths or resource_dir, not both')
        self.model_paths = model_paths or (ModelPaths.from_directory(model_dir) if model_dir else
                                          ModelPaths.from_directory(Path(resource_dir) / 'models') if resource_dir else ModelPaths())
        self.resource_paths = resource_paths or (resource_paths_from_directory(resource_dir) if resource_dir else ResourcePaths())
        self.preprocessor = ExpressionPreprocessor(PreprocessAssets(
            self.resource_paths.pixel_coords, self.resource_paths.max_norm_tpm,
            self.resource_paths.gene_id_map))
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

    @staticmethod
    def _batch_predict(model: keras.Model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        out = []
        for batch in ds:
            pred = model(batch, training=False)
            out.append(pred.numpy())
        return np.concatenate(out, axis=0)

    def harmonize_expression(self, expr: pd.DataFrame, gene_id_type: str = 'auto') -> pd.DataFrame:
        return self.preprocessor.harmonize_expression(expr, gene_id_type=gene_id_type)

    def batch_correct(
        self,
        expr_tpm: pd.DataFrame,
        batch_size: int = 256,
        gene_id_type: str = 'auto',
        stochastic: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return latent and corrected expression; deterministic unless sampling is requested."""
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        x = self.preprocessor.gene2img(expr, gene_id_type='ensembl')
        generator = self._load_generator()
        ds = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        latent_list, observed_list = [], []
        for batch in ds:
            latent = generator.encode(batch, reparameterize=stochastic)
            if not stochastic:
                latent = latent[0]
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

    def predict_functional_axes(self, expr_tpm: pd.DataFrame, batch_size: int = 256, gene_id_type: str = 'auto') -> pd.DataFrame:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr, model_paths=self.model_paths, resource_paths=self.resource_paths)
        return predictor.predict_pathway_axis(batch_size=batch_size, return_type='score')

    def predict_survival_risk(
        self,
        expr_tpm: pd.DataFrame,
        clinical: pd.DataFrame,
        batch_size: int = 256,
        gene_id_type: str = 'auto',
    ) -> pd.DataFrame:
        validate_expression(expr_tpm)
        clinical = align_clinical(clinical, expr_tpm.index)
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        predictor = Predictor(expr, model_paths=self.model_paths, resource_paths=self.resource_paths)
        return predictor.predict_survival_risk(
            clinical=clinical,
            batch_size=batch_size,
            ckpt_path=str(self.model_paths.survival_risk),
        )

    def attribute_functional_axis(self, expr_tpm: pd.DataFrame, axis_name: str, steps: int = 50, gene_id_type: str = 'auto') -> AttributionResult:
        validate_axis_name(axis_name)
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_functional_axis(expr, axis_name=axis_name, steps=steps)

    def attribute_survival_risk(
        self,
        expr_tpm: pd.DataFrame,
        clinical: pd.DataFrame,
        steps: int = 50,
        gene_id_type: str = 'auto',
    ) -> AttributionResult:
        validate_expression(expr_tpm)
        clinical = align_clinical(clinical, expr_tpm.index)
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_survival_risk(expr, clinical=clinical, steps=steps)

    def attribute_axis_risk(self, expr_tpm, clinical, axis_name, reference_tpm,
                            steps=64, internal_batch_size=32,
                            maximum_relative_error=0.1,
                            gene_id_type='auto'):
        """Attribute one sample's selected functional-axis risk contribution to all genes.

        See README.md, Axis-Specific Risk Attribution, for baselines and QC.
        """
        validate_axis_name(axis_name)
        validate_expression(expr_tpm)
        clinical = align_clinical(clinical, expr_tpm.index)
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        if not isinstance(reference_tpm, pd.DataFrame) or reference_tpm.empty:
            raise ValueError('reference_tpm must be a nonempty samples-by-genes TPM DataFrame')
        reference = self.harmonize_expression(reference_tpm, gene_id_type=gene_id_type)
        return self.attributor.attribute_axis_risk(
            expr, clinical, axis_name, reference, steps,
            internal_batch_size, maximum_relative_error)

    def _feature_signature(self):
        """Hash the actual functional-axis/encoder checkpoints and preprocessing, not filenames."""
        import hashlib
        from .downloads import sha256_file
        paths = []
        for directory in (self.model_paths.batch_correction, self.model_paths.functional_axes):
            checkpoint = Path(self._latest_checkpoint(directory))
            paths.extend([Path(str(checkpoint) + '.index')])
            paths.extend(sorted(checkpoint.parent.glob(checkpoint.name + '.data-*')))
        paths.extend([self.resource_paths.pixel_coords, self.resource_paths.max_norm_tpm,
                      self.resource_paths.gene_id_map])
        state = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
        if getattr(self, '_signature_state', None) != state:
            self._signature_value = hashlib.sha256(
                ''.join(sha256_file(p) for p in paths).encode()).hexdigest()
            self._signature_state = state
        return self._signature_value

    def _reference_axis_model(self):
        from .utils.model_loader import load_predict_model
        signature = self._feature_signature()
        if getattr(self, '_reference_model_signature', None) != signature:
            self._reference_model = load_predict_model(
                'latest', str(self.model_paths.functional_axes), 8, None,
                encoder_ckpt_path=str(self.model_paths.batch_correction))
            self._reference_model_signature = signature
        return self._reference_model

    def encode_features(self, expr_tpm, batch_size=32, gene_id_type='auto',
                        adapter=None, metadata=None):
        """Frozen functional-axis-encoder latent mean and functional-axis features; optional target adapter."""
        validate_expression(expr_tpm)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be positive')
        model = self._reference_axis_model()
        if adapter is not None:
            if adapter.model_signature != self._feature_signature():
                raise ValueError('Adapter was fitted with a different model/preprocessing')
            if metadata is None:
                raise ValueError('Metadata is required when applying an adapter')
            adapter.validate_metadata(metadata, expr_tpm.index)
        latent, axes = [], []
        for start in range(0, len(expr_tpm), batch_size):
            images = self.preprocessor.gene2img(expr_tpm.iloc[start:start+batch_size], gene_id_type=gene_id_type)
            z, _ = tf.split(model.encoder(images, training=False), 2, axis=1)
            if adapter is not None:
                z = adapter(z, training=False)
            functional_axes = model.out_layer(model.fc_layer(z, training=False), training=False)
            latent.append(z.numpy())
            axes.append(functional_axes.numpy())
        z, functional_axes = np.concatenate(latent), np.concatenate(axes)
        return (pd.DataFrame(z, index=expr_tpm.index, columns=['latent_%d' % i for i in range(z.shape[1])]),
                pd.DataFrame(functional_axes, index=expr_tpm.index, columns=FUNCTIONAL_AXIS_NAMES))

    def build_reference(self, expr_tpm, metadata, batch_size=32, gene_id_type='auto'):
        """Create a reference atlas from TPM and sample-indexed biological labels."""
        from .reference import ReferenceAtlas
        z, functional_axes = self.encode_features(expr_tpm, batch_size, gene_id_type)
        if metadata.index.has_duplicates:
            raise ValueError('Duplicate metadata IDs')
        return ReferenceAtlas(z, functional_axes, metadata.reindex(expr_tpm.index), self._feature_signature())

    def adapt(self, expr_tpm, metadata, reference, biological_labels=None,
              batch_size=32, gene_id_type='auto', **fit_options):
        """Fit a target-only residual adapter, stratified by cancer and optional labels."""
        from .reference import ReferenceAtlas
        from .adaptation import fit_latent_adapter
        atlas = ReferenceAtlas.load(reference) if isinstance(reference, (str, Path)) else reference
        if atlas.model_signature != self._feature_signature():
            raise ValueError('Reference atlas was built with different model/preprocessing')
        z, _ = self.encode_features(expr_tpm, batch_size, gene_id_type)
        return fit_latent_adapter(z, metadata, atlas, biological_labels, **fit_options)

    def query(self, expr_tpm, reference, metadata=None, space='latent', k=10,
              metric='cosine', standardize=True, same_cancer=False, exclude_self=True,
              adapter=None, batch_size=32, gene_id_type='auto'):
        """Return nearest database samples, not classifications or clinical matches."""
        from .reference import ReferenceAtlas
        atlas = ReferenceAtlas.load(reference) if isinstance(reference, (str, Path)) else reference
        if atlas.model_signature != self._feature_signature():
            raise ValueError('Reference atlas was built with different model/preprocessing')
        z, functional_axes = self.encode_features(expr_tpm, batch_size, gene_id_type, adapter, metadata)
        if space not in ('latent', 'functional_axes'):
            raise ValueError("space must be 'latent' or 'functional_axes'")
        return atlas.query(z if space == 'latent' else functional_axes, space, k, metric,
                           standardize, metadata, same_cancer, exclude_self)

    def run_all(
        self,
        expr_tpm: pd.DataFrame,
        clinical: pd.DataFrame,
        batch_size: int = 256,
        gene_id_type: str = 'auto',
    ) -> CancerXpressResult:
        latent, corrected = self.batch_correct(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        functional_axes = self.predict_functional_axes(expr_tpm, batch_size=batch_size, gene_id_type=gene_id_type)
        risk = self.predict_survival_risk(expr_tpm, clinical=clinical, batch_size=batch_size, gene_id_type=gene_id_type)
        return CancerXpressResult(
            latent=latent,
            corrected_expression=corrected,
            functional_axes=functional_axes,
            survival_risk=risk,
        )
