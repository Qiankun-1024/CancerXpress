from __future__ import annotations

from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from .models import task_model
from .models.survival_model import ClinicalMERiskModel
from .resources import ModelPaths, ResourcePaths, load_risk_preprocessing
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
    def _normalize_sex(value: str) -> str:
        value = str(value).strip().lower()
        if value in {'male', 'm'}:
            return 'male'
        if value in {'female', 'f'}:
            return 'female'
        return 'unknown'

    @staticmethod
    def _normalize_race(value: str) -> str:
        value = str(value).strip().lower()
        allowed = {
            'white', 'black or african american', 'asian',
            'american indian or alaska native',
            'native hawaiian or other pacific islander',
        }
        return value if value in allowed else 'unknown'

    @staticmethod
    def _normalize_stage(value: str) -> str:
        value = str(value).strip().lower()
        mapping = {
            'stage i': 'stage i', 'stage ia': 'stage i', 'stage ib': 'stage i',
            'stage ii': 'stage ii', 'stage iia': 'stage ii', 'stage iib': 'stage ii', 'stage iic': 'stage ii',
            'stage iii': 'stage iii', 'stage iiia': 'stage iii', 'stage iiib': 'stage iii', 'stage iiic': 'stage iii',
            'stage iv': 'stage iv', 'stage iva': 'stage iv', 'stage ivb': 'stage iv', 'stage ivc': 'stage iv',
        }
        return mapping.get(value, 'unknown')

    def _prepare_run1_features(self, me: pd.DataFrame, clinical: pd.DataFrame, metadata: dict):
        required = set(metadata['clinical_columns'])
        missing_columns = required - set(clinical.columns)
        if missing_columns:
            raise ValueError(f'clinical缺少run1风险模型所需列: {sorted(missing_columns)}')
        if clinical.index.has_duplicates:
            raise ValueError('clinical样本ID不能重复。')
        clinical = clinical.reindex(self.sample_id).copy()
        if clinical[list(required)].isna().all(axis=1).any():
            missing_ids = clinical.index[clinical[list(required)].isna().all(axis=1)].tolist()
            raise ValueError(f'clinical缺少表达矩阵中的样本: {missing_ids[:5]}')

        age = pd.to_numeric(clinical['age'], errors='coerce')
        age = age.fillna(float(metadata['age_fill_median']))
        age_z = (age.to_numpy(np.float32) - float(metadata['age_mean'])) / float(metadata['age_std'])

        sex = clinical['sex'].map(self._normalize_sex)
        race = clinical['race'].map(self._normalize_race)
        stage = clinical['tumor_stage'].map(self._normalize_stage)
        cancer = clinical['cancer_type'].astype(str).str.strip()
        cancer.loc[cancer == 'LAML'] = 'AML'
        unsupported = sorted(set(cancer) - set(metadata['cancer_vocab']))
        if unsupported:
            raise ValueError(f'run1风险模型不支持这些癌种: {unsupported}')

        def encode(values, vocab, fallback=None):
            encoded = values.map(vocab)
            if fallback is not None:
                encoded = encoded.fillna(vocab[fallback])
            return encoded.astype(np.int32).to_numpy()

        return {
            'me': tf.convert_to_tensor(me[metadata['me_columns']].to_numpy(np.float32)),
            'age': tf.convert_to_tensor(age_z.reshape(-1, 1), dtype=tf.float32),
            'sex': tf.convert_to_tensor(encode(sex, metadata['sex_vocab'], fallback='female')),
            'race': tf.convert_to_tensor(encode(race, metadata['race_vocab'], fallback='unknown')),
            'stage': tf.convert_to_tensor(encode(stage, metadata['stage_vocab'], fallback='unknown')),
            'cancer': tf.convert_to_tensor(encode(cancer, metadata['cancer_vocab'])),
        }

    @staticmethod
    def _load_run1_survival_model(ckpt_path: str, metadata: dict):
        latest_ckpt = tf.train.latest_checkpoint(ckpt_path)
        if latest_ckpt is None:
            raise FileNotFoundError(f'未找到checkpoint: {ckpt_path}')
        variables = dict(tf.train.list_variables(latest_ckpt))
        marker = 'model/cancer_emb/embeddings/.ATTRIBUTES/VARIABLE_VALUE'
        if marker not in variables:
            raise ValueError(f'风险checkpoint不是run1 ME+clinical模型: {latest_ckpt}')
        expected = len(metadata['cancer_vocab'])
        if int(variables[marker][0]) != expected:
            raise ValueError('风险checkpoint与预处理癌种词表维度不一致。')

        model = ClinicalMERiskModel(
            n_sex=len(metadata['sex_vocab']),
            n_race=len(metadata['race_vocab']),
            n_stage=len(metadata['stage_vocab']),
            n_cancer=expected,
        )
        dummy = {
            'me': tf.zeros((1, 8), tf.float32),
            'age': tf.zeros((1, 1), tf.float32),
            'sex': tf.zeros((1,), tf.int32),
            'race': tf.zeros((1,), tf.int32),
            'stage': tf.zeros((1,), tf.int32),
            'cancer': tf.zeros((1,), tf.int32),
        }
        model(dummy, training=False)
        status = tf.train.Checkpoint(model=model).restore(latest_ckpt)
        status.expect_partial()
        status.assert_existing_objects_matched()
        return model, latest_ckpt

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
        clinical: pd.DataFrame,
        batch_size=256,
        ckpt_path: Optional[str] = None,
    ):
        ckpt_path = ckpt_path or str(self.model_paths.survival_risk)
        metadata = load_risk_preprocessing(self.resource_paths.risk_preprocessing)
        me = self.predict_pathway_axis(batch_size=batch_size, return_type='score')
        features = self._prepare_run1_features(me, clinical, metadata)
        model, _ = self._load_run1_survival_model(ckpt_path, metadata)
        risk_scores = model(features, training=False).numpy().reshape(-1)
        return pd.DataFrame({'risk_score': risk_scores}, index=self.sample_id)
