from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .models import task_model
from .resources import ModelPaths, ResourcePaths
from .utils import data_normalizer, model_loader


class Predictor:
    def __init__(self, data: pd.DataFrame, dtype='TPM'):
        self.sample_id = data.index
        self.data = data_normalizer.gene2img(data, dtype=dtype)
        self.model_paths = ModelPaths()
        self.resource_paths = ResourcePaths()

    @staticmethod
    def _normalize_cancer_type(cancer_type: Union[pd.Series, np.ndarray, List[str]], sample_id: pd.Index) -> pd.Series:
        if isinstance(cancer_type, pd.Series):
            out = cancer_type.reindex(sample_id) if not cancer_type.index.equals(sample_id) else cancer_type.copy()
        else:
            arr = np.asarray(cancer_type)
            if arr.shape[0] != len(sample_id):
                raise ValueError(f'cancer_type长度({arr.shape[0]})与样本数({len(sample_id)})不一致。')
            out = pd.Series(arr, index=sample_id)
        if out.isna().any():
            raise ValueError('cancer_type包含缺失值，无法完成双输入风险预测。')
        out = out.astype(str)
        out.loc[out == 'LAML'] = 'AML'
        return out

    @staticmethod
    def _build_training_like_cancer_mapping(reference_label_file: str):
        if not os.path.exists(reference_label_file):
            raise FileNotFoundError(f'参考标签文件不存在: {reference_label_file}')
        label = pd.read_csv(reference_label_file, sep='\t', index_col=0)
        need_cols = {'OS_time_months', 'OS_Status', 'batch', 'cancer_type'}
        if not need_cols.issubset(label.columns):
            raise ValueError(f'参考标签文件缺少必要列: {need_cols - set(label.columns)}')
        label = label[label['cancer_type'] != 'Normal'].copy()
        label.loc[label['cancer_type'] == 'LAML', 'cancer_type'] = 'AML'
        label = label[['OS_time_months', 'OS_Status', 'batch', 'cancer_type']].dropna(how='any')
        label = label[label['OS_time_months'] > 0]
        cancer_counts = label['cancer_type'].value_counts()
        sufficient_cancers = cancer_counts[cancer_counts >= 30].index
        label = label[label['cancer_type'].isin(sufficient_cancers)]
        cancer_types_list = sorted(label['cancer_type'].unique())
        return {ct: i for i, ct in enumerate(cancer_types_list)}

    @staticmethod
    def _load_survival_model_single_input(ckpt_path: str):
        model = task_model.make_finetune_model(1, activation=None)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt is None:
            raise FileNotFoundError(f'未找到checkpoint: {ckpt_path}')
        var_names = [name for name, _ in tf.train.list_variables(latest_ckpt)]
        ckpt = tf.train.Checkpoint(model=model) if any(name.startswith('model/') for name in var_names) else tf.train.Checkpoint(classifier=model)
        ckpt.restore(latest_ckpt).expect_partial()
        return model, latest_ckpt

    @staticmethod
    def _build_survival_model_with_cancer_cov(n_cancer_types: int, batch_ckpt_path: str):
        class SurvivalModelWithCancerType(tf.keras.Model):
            def __init__(self, encoder_layer, n_cancer_types, output_dim=1):
                super().__init__()
                self.encoder = encoder_layer
                self.cancer_embedding = tf.keras.layers.Embedding(input_dim=n_cancer_types + 1, output_dim=16, name='cancer_embedding')
                self.fc_layer = tf.keras.Sequential([
                    tf.keras.layers.Dense(256),
                    tf.keras.layers.LeakyReLU(alpha=0.1),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(128),
                    tf.keras.layers.LeakyReLU(alpha=0.2),
                    tf.keras.layers.Dropout(0.3),
                    tf.keras.layers.Dense(64),
                    tf.keras.layers.LeakyReLU(alpha=0.1),
                ])
                self.out_layer = tf.keras.layers.Dense(output_dim, activation=None)

            def call(self, inputs, training=False):
                x, cancer_idx = inputs
                mean, logvar = tf.split(self.encoder(x, training=training), num_or_size_splits=2, axis=1)
                gene_features = mean
                cancer_feat = self.cancer_embedding(cancer_idx)
                combined = tf.concat([gene_features, cancer_feat], axis=1)
                logits = self.fc_layer(combined, training=training)
                return self.out_layer(logits)

        encoder_layer = task_model.get_encoder_layer(trainable_layer=3, ckpt_path=batch_ckpt_path)
        return SurvivalModelWithCancerType(encoder_layer=encoder_layer, n_cancer_types=n_cancer_types, output_dim=1)

    def _batch_predict(self, model, data, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
        out = []
        for batch in ds:
            out.append(model(batch, training=False).numpy())
        return np.concatenate(out, axis=0)

    def predict(self, label_types: Union[str, List[str]], batch_size=256, return_type='label'):
        pred_labels = pd.DataFrame(index=self.sample_id)
        if not isinstance(label_types, list):
            label_types = [label_types]
        task_config = {
            'primary_site': {
                'ckpt_path': str(self.model_paths.primary_site),
                'out_dim': 35,
                'activation': 'softmax',
                'encoder_path': str(self.resource_paths.primary_site_encoder),
                'output_name': 'primary_site',
            },
            'primary_site_v2': {
                'ckpt_path': str(self.model_paths.primary_site),
                'out_dim': 35,
                'activation': 'softmax',
                'encoder_path': str(self.resource_paths.primary_site_encoder),
                'output_name': 'primary_site',
            },
            'cancer_type': {
                'ckpt_path': str(self.model_paths.cancer_type),
                'out_dim': 41,
                'activation': 'softmax',
                'encoder_path': str(self.resource_paths.cancer_type_encoder),
                'output_name': 'cancer_type',
            },
            'cancer_type_v2': {
                'ckpt_path': str(self.model_paths.cancer_type),
                'out_dim': 41,
                'activation': 'softmax',
                'encoder_path': str(self.resource_paths.cancer_type_encoder),
                'output_name': 'cancer_type',
            },
        }
        for lt in label_types:
            cfg = task_config[lt]
            model = model_loader.load_predict_model('latest', cfg['ckpt_path'], cfg['out_dim'], cfg['activation'])
            preds = self._batch_predict(model, self.data, batch_size)
            encoder = joblib.load(cfg['encoder_path'])
            if return_type == 'label':
                preds = pd.DataFrame(encoder.inverse_transform(preds), columns=[cfg['output_name']], index=self.sample_id)
            else:
                class_names = encoder.categories_[0]
                preds = pd.DataFrame(np.asarray(preds), columns=class_names, index=self.sample_id)
            pred_labels = pd.concat([pred_labels, preds], axis=1)
        return pred_labels

    def predict_pathway_axis(self, batch_size=256, return_type='score'):
        model = model_loader.load_predict_model('latest', str(self.model_paths.me), 8, None)
        preds = self._batch_predict(model, self.data, batch_size)
        class_names = ['MEblack', 'MEblue', 'MEbrown', 'MEgreen', 'MEpink', 'MEred', 'MEturquoise', 'MEyellow']
        if return_type == 'label':
            binary_preds = []
            for i, _ in enumerate(class_names):
                median_val = np.median(preds[:, i])
                binary_preds.append(np.where(preds[:, i] > median_val, 'High', 'Low'))
            preds = pd.DataFrame(np.column_stack(binary_preds), columns=class_names, index=self.sample_id)
        else:
            preds = pd.DataFrame(preds, columns=class_names, index=self.sample_id)
        return preds

    def predict_survival_risk(
        self,
        batch_size=256,
        cancer_type: Optional[Union[pd.Series, np.ndarray, List[str]]] = None,
        cancer_type2idx: Optional[Dict[str, int]] = None,
        cancer_vocab: Optional[Sequence[str]] = None,
        reference_label_file: Optional[str] = None,
        ckpt_path: Optional[str] = None,
    ):
        ckpt_path = ckpt_path or str(self.model_paths.survival_risk)
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt is None:
            raise FileNotFoundError(f'未找到checkpoint: {ckpt_path}')
        ckpt_vars = dict(tf.train.list_variables(latest_ckpt))
        has_cancer_embedding = 'model/cancer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE' in ckpt_vars
        if has_cancer_embedding:
            if cancer_type is None:
                raise ValueError('当前checkpoint为双输入风险模型，请提供cancer_type参数。')
            if cancer_type2idx is not None:
                mapping = dict(cancer_type2idx)
            elif cancer_vocab is not None:
                mapping = {ct: i for i, ct in enumerate(cancer_vocab)}
            else:
                if reference_label_file is None:
                    raise ValueError('未提供cancer_vocab/cancer_type2idx时，需要reference_label_file。')
                mapping = self._build_training_like_cancer_mapping(reference_label_file)
            embed_shape = ckpt_vars['model/cancer_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE']
            n_cancer_types = int(embed_shape[0]) - 1
            cancer_series = self._normalize_cancer_type(cancer_type, self.sample_id)
            unknown_idx = n_cancer_types
            cancer_idx = cancer_series.map(mapping).fillna(unknown_idx).astype(np.int32).to_numpy()
            model = self._build_survival_model_with_cancer_cov(n_cancer_types, str(self.model_paths.batch_correction))
            ckpt = tf.train.Checkpoint(model=model)
            ckpt.restore(latest_ckpt).expect_partial()
            ds = tf.data.Dataset.from_tensor_slices((self.data, cancer_idx)).batch(batch_size)
            risk_scores = []
            for batch_x, batch_cancer in ds:
                risk_scores.append(model((batch_x, batch_cancer), training=False).numpy())
            risk_scores = np.concatenate(risk_scores, axis=0).flatten()
        else:
            model, _ = self._load_survival_model_single_input(ckpt_path)
            risk_scores = self._batch_predict(model, self.data, batch_size).flatten()
        return pd.DataFrame({'risk_score': risk_scores}, index=self.sample_id)
