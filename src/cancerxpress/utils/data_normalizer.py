from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..resources import ResourcePaths


def _pixel_coords_path() -> Path:
    return ResourcePaths().pixel_coords


def _max_norm_path() -> Path:
    return ResourcePaths().max_norm_tpm


def trans_1d_to_2d(x_data: pd.DataFrame) -> np.ndarray:
    pixel_coords = pd.read_csv(_pixel_coords_path(), index_col=0)
    x_len = int(pixel_coords['x_coord'].to_list()[-1])
    y_len = int(pixel_coords['y_coord'].to_list()[-1])
    data_df = pd.merge(pixel_coords, x_data.T, how='left', left_index=True, right_index=True, sort=False)
    data_df.drop(['x_coord', 'y_coord'], axis=1, inplace=True)
    data = data_df.fillna(0).T.values
    num = data.shape[0]
    data = data.reshape((num, x_len, y_len))
    return data[:, ::-1]


def trans_2d_to_1d(data: np.ndarray) -> pd.DataFrame:
    pixel_coords = pd.read_csv(_pixel_coords_path(), index_col=0)
    data = np.array(data)
    num = data.shape[0]
    data = data[:, ::-1].reshape(num, -1)
    data_df = pd.DataFrame(data).T
    data_df.index = pixel_coords.index
    data_df = pd.concat([pixel_coords, data_df], axis=1)
    data_df.drop(['x_coord', 'y_coord'], axis=1, inplace=True)
    data_df = data_df[data_df.index.str.startswith('ENSG')]
    return data_df.T


def log2_feature(x_data: pd.DataFrame, dtype: str = 'FPKM') -> pd.DataFrame:
    if dtype not in ['FPKM', 'TPM']:
        raise ValueError(f"Allowed values are 'FPKM' or 'TPM', got {dtype}")
    if dtype == 'FPKM':
        sum_counts = x_data.sum(axis=1)
        tpm_data = x_data.div(sum_counts, axis=0) * 10**6
        return np.log2(tpm_data + 1)
    return np.log2(x_data + 1)


def normalize_feature(log2_tpm: pd.DataFrame, max_matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    if max_matrix is None:
        max_matrix = pd.read_csv(_max_norm_path(), index_col=0, sep='\t')
    max_matrix.columns = ['max_log2tpm']
    log2_tpm_t = pd.merge(log2_tpm.T, max_matrix + 1e-8, how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = log2_tpm_t.dropna(how='any', axis=0)
    norm_tpm_t = log2_tpm_t.div(log2_tpm_t['max_log2tpm'], axis=0)
    norm_tpm = norm_tpm_t.drop(['max_log2tpm'], axis=1).T
    norm_tpm[norm_tpm > 1] = 1
    return norm_tpm


def reverse_log2_feature(norm_tpm: pd.DataFrame, max_matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    if max_matrix is None:
        max_matrix = pd.read_csv(_max_norm_path(), index_col=0, sep='\t')
    max_matrix.columns = ['max_log2tpm']
    norm_tpm_t = pd.merge(norm_tpm.T, max_matrix + 1e-8, how='left', left_index=True, right_index=True, sort=False)
    log2_tpm_t = norm_tpm_t.mul(norm_tpm_t['max_log2tpm'], axis=0)
    return log2_tpm_t.drop(['max_log2tpm'], axis=1).T


def reverse_tpm_feature(log2_tpm: pd.DataFrame) -> pd.DataFrame:
    return 2 ** log2_tpm - 1


def gene2img(data: pd.DataFrame, dtype: str = 'TPM', max_matrix: pd.DataFrame | None = None) -> np.ndarray:
    if dtype not in ['FPKM', 'TPM', 'log2TPM', 'normalizedTPM']:
        raise ValueError(f"Unsupported dtype: {dtype}")
    if dtype in ['FPKM', 'TPM']:
        data = log2_feature(data, dtype=dtype)
    if dtype != 'normalizedTPM':
        data = normalize_feature(data, max_matrix)
    data = trans_1d_to_2d(data).astype(np.float32)
    return np.expand_dims(data, axis=3)


class Gtf:
    @classmethod
    def _load_gtf(cls, gtf_file: str, filter_gene: bool = False) -> pd.DataFrame:
        column_names = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        gtf_data = pd.read_csv(gtf_file, sep='\t', comment='#', names=column_names)
        if filter_gene:
            gtf_data = gtf_data[gtf_data['feature'] == 'gene']
        return gtf_data

    @classmethod
    def _split_attributes(cls, attribute: str) -> dict:
        attr_dict = {}
        for attr in attribute.split(';'):
            if attr.strip():
                key_value = re.split(r'[\s\"]+', attr.strip())
                if len(key_value) > 1:
                    attr_dict[key_value[0]] = key_value[1]
        return attr_dict

    @classmethod
    def _parse_attributes(cls, df: pd.DataFrame) -> pd.DataFrame:
        attributes = df['attribute'].apply(cls._split_attributes)
        attributes_df = pd.json_normalize(attributes)
        if 'gene_id' in attributes_df.columns:
            attributes_df['gene_id'] = attributes_df['gene_id'].str.replace(r'\.\d+$', '', regex=True)
        df = df.reset_index(drop=True)
        attributes_df = attributes_df.reset_index(drop=True)
        return pd.concat([df.drop(columns=['attribute']), attributes_df], axis=1)

    @classmethod
    def get_gtf(cls, gtf_file: str, filter_gene: bool = False) -> pd.DataFrame:
        return cls._parse_attributes(cls._load_gtf(gtf_file, filter_gene=filter_gene))


def counts2tpm(data: pd.DataFrame, gtf_path: str) -> pd.DataFrame:
    gtf = Gtf.get_gtf(gtf_path, filter_gene=True)
    gtf = gtf[gtf['gene_id'].isin(data.columns)]
    gtf = gtf.drop_duplicates('gene_id', keep=False)
    gtf.index = gtf['gene_id']
    length = gtf['end'] - gtf['start'] + 1
    sum_counts = data.sum(axis=1)
    data = data.div(length, axis=1).div(sum_counts, axis=0) * 10**9
    sum_fpkm = data.sum(axis=1)
    return data.div(sum_fpkm, axis=0) * 10**6


def tpm2counts(tpm: pd.DataFrame, counts: pd.DataFrame, gtf_path: str, use_fpkm: bool = False) -> pd.DataFrame:
    counts = counts.loc[tpm.index]
    gtf = Gtf.get_gtf(gtf_path, filter_gene=True)
    gtf = gtf.drop_duplicates('gene_id', keep=False)
    gtf.index = gtf['gene_id']
    intersect_genes = set(tpm.columns).intersection(set(counts.columns), set(gtf['gene_id']))
    tpm = tpm[list(intersect_genes)]
    counts = counts[list(intersect_genes)]
    gtf = gtf.loc[list(intersect_genes)]
    length = gtf['end'] - gtf['start'] + 1
    quantile_thresholds = counts.quantile(0.75, axis=1)
    mask = counts.gt(quantile_thresholds, axis=0)
    sum_counts = counts.where(~mask, other=0).sum(axis=1)
    fpkm = counts.div(length, axis=1).div(sum_counts, axis=0) * 10 ** 7
    sum_fpkm = fpkm.where(~mask, other=0).sum(axis=1)
    sum_tpm = tpm.where(~mask, other=0).sum(axis=1)
    tpm = tpm.div(sum_tpm, axis=0) * 10**6
    return tpm.multiply(sum_fpkm, axis=0).multiply(length, axis=1).div(10**6)
