from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .resources import ResourcePaths


@dataclass
class PreprocessAssets:
    pixel_coords_path: Path
    max_norm_tpm_path: Path
    gene_id_map_path: Path


class GeneIDConverter:
    def __init__(self, gene_id_map_path: Path | None = None):
        path = gene_id_map_path or ResourcePaths().gene_id_map
        mapping = pd.read_csv(path, sep='\t')
        mapping = mapping.dropna(subset=['Gene stable ID', 'HGNC symbol']).copy()
        mapping['Gene stable ID'] = mapping['Gene stable ID'].astype(str).str.replace(r'\.\d+$', '', regex=True)
        mapping['HGNC symbol'] = mapping['HGNC symbol'].astype(str)
        mapping = mapping.drop_duplicates(subset=['Gene stable ID'], keep='first')
        self.ensembl_to_symbol = dict(zip(mapping['Gene stable ID'], mapping['HGNC symbol']))
        symbol_df = mapping[~mapping['HGNC symbol'].duplicated(keep=False)].copy()
        self.symbol_to_ensembl = dict(zip(symbol_df['HGNC symbol'], symbol_df['Gene stable ID']))

    @staticmethod
    def _strip_ensembl_version(columns: pd.Index) -> pd.Index:
        return pd.Index(columns.astype(str).str.replace(r'\.\d+$', '', regex=True))

    def infer_id_type(self, columns: pd.Index) -> Literal['ensembl', 'symbol']:
        stripped = self._strip_ensembl_version(columns)
        ensembl_hits = stripped.isin(self.ensembl_to_symbol).sum()
        symbol_hits = columns.astype(str).isin(self.symbol_to_ensembl).sum()
        return 'ensembl' if ensembl_hits >= symbol_hits else 'symbol'

    def to_ensembl(self, expr: pd.DataFrame, gene_id_type: str = 'auto') -> pd.DataFrame:
        if gene_id_type == 'auto':
            gene_id_type = self.infer_id_type(expr.columns)

        out = expr.copy()
        if gene_id_type == 'ensembl':
            out.columns = self._strip_ensembl_version(out.columns)
        elif gene_id_type == 'symbol':
            mapped = pd.Index(out.columns.astype(str)).map(self.symbol_to_ensembl)
            keep = ~mapped.isna()
            out = out.loc[:, keep].copy()
            out.columns = mapped[keep]
        else:
            raise ValueError(f'Unsupported gene_id_type: {gene_id_type}')

        out = out.groupby(out.columns, axis=1).sum()
        return out


class ExpressionPreprocessor:
    def __init__(self, assets: PreprocessAssets | None = None):
        if assets is None:
            rp = ResourcePaths()
            assets = PreprocessAssets(
                pixel_coords_path=rp.pixel_coords,
                max_norm_tpm_path=rp.max_norm_tpm,
                gene_id_map_path=rp.gene_id_map,
            )
        self.assets = assets
        self.pixel_coords = pd.read_csv(assets.pixel_coords_path, index_col=0)
        self.max_matrix = pd.read_csv(assets.max_norm_tpm_path, index_col=0, sep='\t')
        self.max_matrix.columns = ['max_log2tpm']
        self.converter = GeneIDConverter(assets.gene_id_map_path)
        self.training_gene_order = self.pixel_coords.index[self.pixel_coords.index.str.startswith('ENSG')]

    def harmonize_expression(self, expr: pd.DataFrame, gene_id_type: str = 'auto', fill_value: float = 0.0) -> pd.DataFrame:
        expr_ensembl = self.converter.to_ensembl(expr, gene_id_type=gene_id_type)
        expr_ensembl = expr_ensembl.reindex(columns=self.training_gene_order, fill_value=fill_value)
        return expr_ensembl

    @staticmethod
    def log2_tpm(tpm: pd.DataFrame) -> pd.DataFrame:
        return np.log2(tpm + 1.0)

    def normalize(self, log2_tpm: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            log2_tpm.T,
            self.max_matrix + 1e-8,
            how='left',
            left_index=True,
            right_index=True,
            sort=False,
        ).dropna(axis=0, how='any')
        norm = merged.div(merged['max_log2tpm'], axis=0).drop(columns=['max_log2tpm']).T
        norm[norm > 1.0] = 1.0
        return norm

    def trans_1d_to_2d(self, x_data: pd.DataFrame) -> np.ndarray:
        x_len = int(self.pixel_coords['x_coord'].to_list()[-1])
        y_len = int(self.pixel_coords['y_coord'].to_list()[-1])
        data_df = pd.merge(
            self.pixel_coords,
            x_data.T,
            how='left',
            left_index=True,
            right_index=True,
            sort=False,
        )
        data_df = data_df.drop(columns=['x_coord', 'y_coord'])
        data = data_df.fillna(0).T.values
        num = data.shape[0]
        data = data.reshape((num, x_len, y_len))
        return data[:, ::-1]

    def gene2img(self, expr_tpm: pd.DataFrame, gene_id_type: str = 'auto') -> np.ndarray:
        expr = self.harmonize_expression(expr_tpm, gene_id_type=gene_id_type)
        log2_data = self.log2_tpm(expr)
        norm_data = self.normalize(log2_data)
        arr = self.trans_1d_to_2d(norm_data).astype(np.float32)
        return np.expand_dims(arr, axis=3)

    def trans_2d_to_1d(self, data: np.ndarray) -> pd.DataFrame:
        data = np.array(data)
        num = data.shape[0]
        flat = data[:, ::-1].reshape(num, -1)
        data_df = pd.DataFrame(flat).T
        data_df.index = self.pixel_coords.index
        data_df = pd.concat([self.pixel_coords, data_df], axis=1)
        data_df = data_df.drop(columns=['x_coord', 'y_coord'])
        data_df = data_df[data_df.index.str.startswith('ENSG')]
        return data_df.T

    def reverse_log2_feature(self, norm_tpm: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            norm_tpm.T,
            self.max_matrix + 1e-8,
            how='left',
            left_index=True,
            right_index=True,
            sort=False,
        )
        log2_tpm = merged.mul(merged['max_log2tpm'], axis=0).drop(columns=['max_log2tpm']).T
        return log2_tpm

    @staticmethod
    def reverse_tpm(log2_tpm: pd.DataFrame) -> pd.DataFrame:
        return (2.0 ** log2_tpm) - 1.0
