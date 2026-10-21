"""Versioned, labelled reference features and deterministic nearest-neighbour lookup."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def validate_features(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(name + ' must be a nonempty sample-indexed DataFrame')
    if frame.index.has_duplicates or frame.columns.has_duplicates or frame.index.hasnans:
        raise ValueError(name + ' needs unique nonmissing sample and feature labels')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(name + ' contains nonfinite features')


class ReferenceAtlas:
    """A frozen feature database, not a collection of fitted patient outcomes."""

    def __init__(self, latent, me, metadata, model_signature):
        validate_features(latent, 'latent')
        validate_features(me, 'me')
        if not latent.index.equals(me.index):
            raise ValueError('Latent and ME sample order must agree')
        if metadata.index.has_duplicates or not metadata.index.equals(latent.index):
            raise ValueError('Metadata sample order must agree with features')
        if 'cancer_type' not in metadata:
            raise ValueError('metadata requires a cancer_type column; missing labels may remain NA')
        if not isinstance(model_signature, str) or not model_signature:
            raise ValueError('A model/preprocessing signature is required')
        self.latent = latent.copy()
        self.me = me.copy()
        self.metadata = metadata.copy()
        self.metadata['cancer_type'] = self.metadata.cancer_type.replace(r'^\s*$', np.nan, regex=True)
        self.model_signature = model_signature

    def save(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        self.metadata.to_csv(root / 'samples.tsv', sep='\t', index_label='sample_id')
        np.savez_compressed(root / 'features.npz', latent=self.latent.values, me=self.me.values)
        (root / 'atlas.json').write_text(json.dumps({
            'format_version': 1, 'model_signature': self.model_signature,
            'latent_columns': list(self.latent.columns), 'me_columns': list(self.me.columns),
            'note': 'Fixed-model features; not OOF risk attributions.'}, indent=2))

    @classmethod
    def load(cls, directory):
        """Load the three local atlas files; never download missing resources."""
        root = Path(directory).expanduser()
        missing = [name for name in ('atlas.json', 'samples.tsv', 'features.npz')
                   if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError('Place the reference atlas files in ' + str(root)
                                    + ' manually. Missing: ' + ', '.join(missing))
        spec = json.loads((root / 'atlas.json').read_text())
        if spec['format_version'] != 1:
            raise ValueError('Unsupported atlas format')
        meta = pd.read_csv(root / 'samples.tsv', sep='\t', index_col=0, dtype={'sample_id': str})
        with np.load(root / 'features.npz', allow_pickle=False) as arrays:
            return cls(pd.DataFrame(arrays['latent'], index=meta.index, columns=spec['latent_columns']),
                       pd.DataFrame(arrays['me'], index=meta.index, columns=spec['me_columns']),
                       meta, spec['model_signature'])

    def query(self, features, space='latent', k=10, metric='cosine',
              standardize=True, query_metadata=None, same_cancer=False,
              exclude_self=True, batch_size=128):
        if space not in ('latent', 'me'):
            raise ValueError("space must be 'latent' or 'me'")
        if metric not in ('cosine', 'euclidean'):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if not isinstance(k, int) or k < 1 or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('k and batch_size must be positive integers')
        validate_features(features, 'query features')
        reference = getattr(self, space)
        if set(features.columns) != set(reference.columns):
            raise ValueError('Query feature columns do not match the reference')
        x = reference.to_numpy(dtype=float)
        y = features.reindex(columns=reference.columns).to_numpy(dtype=float)
        if query_metadata is not None:
            if query_metadata.index.has_duplicates:
                raise ValueError('Duplicate query metadata IDs')
            labels = query_metadata.reindex(features.index)
            if 'cancer_type' not in labels:
                raise ValueError('Query metadata requires cancer_type')
            labels = labels.copy()
            labels['cancer_type'] = labels.cancer_type.replace(r'^\s*$', np.nan, regex=True)
            if same_cancer and labels.cancer_type.isna().any():
                raise ValueError('same_cancer=True requires nonmissing cancer_type for every query')
            unknown = set(labels.cancer_type.dropna()) - set(self.metadata.cancer_type.dropna())
            if unknown:
                raise ValueError('Cancer types absent from reference: ' + str(sorted(unknown)))
        elif same_cancer:
            raise ValueError('same_cancer=True requires query_metadata')
        if standardize:
            mean, sd = x.mean(0), x.std(0)
            sd = np.where(sd > 1e-8, sd, 1.)
            x, y = (x - mean) / sd, (y - mean) / sd
        if metric == 'cosine' and (np.any(np.linalg.norm(x, axis=1) < 1e-12)
                                   or np.any(np.linalg.norm(y, axis=1) < 1e-12)):
            raise ValueError('Cosine undefined for a zero vector; use euclidean or change standardization')
        rows = []
        for start in range(0, len(y), batch_size):
            distances = cdist(y[start:start + batch_size], x, metric=metric)
            for j, distance in enumerate(distances):
                sid = features.index[start + j]
                allowed = np.ones(len(x), dtype=bool)
                if same_cancer:
                    allowed &= self.metadata.cancer_type.eq(labels.loc[sid, 'cancer_type']).fillna(False).to_numpy(dtype=bool)
                if exclude_self:
                    allowed &= self.metadata.index.to_numpy() != sid
                candidates = np.flatnonzero(allowed)
                if not len(candidates):
                    raise ValueError('No eligible reference samples for ' + str(sid))
                order = candidates[np.argsort(distance[candidates], kind='stable')[:k]]
                for rank, pos in enumerate(order, 1):
                    rows.append({'query_sample': sid, 'reference_sample': self.metadata.index[pos],
                                 'reference_cancer_type': self.metadata.cancer_type.iloc[pos],
                                 'rank': rank, 'space': space, 'metric': metric,
                                 'standardized': bool(standardize), 'distance': distance[pos],
                                 'similarity': 1. - distance[pos] if metric == 'cosine' else np.nan})
        return pd.DataFrame(rows)
