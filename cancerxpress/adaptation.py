"""Stratified target-only residual adaptation without outcome labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from ._keras import keras
from .reference import validate_features


def _distance(a, b):
    return tf.maximum(tf.reduce_sum(a*a, 1)[:, None] + tf.reduce_sum(b*b, 1)[None, :]
                      - 2 * tf.matmul(a, b, transpose_b=True), 0.) / tf.cast(tf.shape(a)[1], tf.float32)


def _mmd(a, b):
    xx, yy, xy = _distance(a, a), _distance(b, b), _distance(a, b)
    return tf.add_n([tf.reduce_mean(tf.exp(-xx/(2*s*s))) +
                     tf.reduce_mean(tf.exp(-yy/(2*s*s))) -
                     2*tf.reduce_mean(tf.exp(-xy/(2*s*s))) for s in (.5, 1., 2.)]) / 3.


class LatentAdapter(keras.Model):
    def __init__(self, mean, sd, model_signature, columns, label_columns, groups, hidden_dim=32):
        super().__init__()
        self.reference_mean = np.asarray(mean, np.float32)
        self.reference_sd = np.asarray(sd, np.float32)
        self.model_signature = model_signature
        self.columns = list(columns)
        self.label_columns = list(label_columns)
        self.groups = [tuple(g) for g in groups]
        self.hidden_dim = hidden_dim
        self.hidden = keras.layers.Dense(hidden_dim, activation='tanh')
        self.delta = keras.layers.Dense(len(mean), kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, z, training=False):
        u = (z - self.reference_mean) / self.reference_sd
        return z + self.delta(self.hidden(u)) * self.reference_sd

    def validate_metadata(self, metadata, index):
        labels = _labels(metadata, index, self.label_columns)
        if set(labels) - set(self.groups):
            raise ValueError('Adapter cannot be applied to unseen biological strata')
        return labels

    def transform(self, latent, metadata):
        validate_features(latent, 'latent')
        if list(latent.columns) != self.columns:
            raise ValueError('Adapter latent columns/order differ')
        self.validate_metadata(metadata, latent.index)
        return pd.DataFrame(self(tf.constant(latent.values, dtype=tf.float32)).numpy(),
                            index=latent.index, columns=latent.columns)

    def save_adapter(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=False)
        self(tf.zeros((1, len(self.columns))))
        np.savez(root / 'adapter_weights.npz', *self.get_weights())
        (root / 'adapter.json').write_text(json.dumps({
            'format_version': 1, 'mean': self.reference_mean.tolist(), 'sd': self.reference_sd.tolist(),
            'model_signature': self.model_signature, 'columns': self.columns,
            'label_columns': self.label_columns, 'groups': self.groups, 'hidden_dim': self.hidden_dim}, indent=2))

    @classmethod
    def load_adapter(cls, directory):
        root = Path(directory)
        spec = json.loads((root / 'adapter.json').read_text())
        if spec.pop('format_version') != 1:
            raise ValueError('Unsupported adapter format')
        adapter = cls(**spec)
        adapter(tf.zeros((1, len(adapter.columns))))
        with np.load(root / 'adapter_weights.npz', allow_pickle=False) as weights:
            adapter.set_weights([weights['arr_%d' % i] for i in range(len(adapter.get_weights()))])
        return adapter


def _labels(metadata, index, columns):
    if metadata.index.has_duplicates or not set(columns).issubset(metadata.columns):
        raise ValueError('Metadata requires unique IDs and all biological label columns')
    subset = metadata.reindex(index)[list(columns)]
    if subset.replace(r'^\s*$', np.nan, regex=True).isna().any().any():
        raise ValueError('Missing biological labels')
    return [tuple(str(x) for x in row) for row in subset.values]


def _splits(metadata, labels, seed):
    # Group repeated assays from one patient before partitioning.
    patients = metadata['patient_id'].astype(str) if 'patient_id' in metadata else pd.Series(metadata.index, index=metadata.index)
    if 'patient_id' in metadata and metadata.patient_id.isna().any():
        raise ValueError('patient_id cannot be missing')
    units = {}
    for i, patient in enumerate(patients):
        if patient in units and units[patient][0] != labels[i]:
            raise ValueError('One patient has inconsistent biological labels')
        units.setdefault(patient, (labels[i], []))[1].append(i)
    rng = np.random.RandomState(seed)
    result = np.empty(len(metadata), dtype=object)
    for label in sorted(set(labels)):
        ids = [p for p, (g, _) in units.items() if g == label]
        if len(ids) < 5:
            raise ValueError('Each stratum needs at least 5 independent patients per cohort: ' + str(label))
        rng.shuffle(ids)
        count = max(1, int(len(ids) * .2))
        for split, selected in [('test', ids[:count]), ('validation', ids[count:2*count]),
                                ('train', ids[2*count:])]:
            for patient in selected:
                result[units[patient][1]] = split
    return result


class AdaptationResult:
    def __init__(self, adapter, history, splits, metrics):
        self.adapter, self.history, self.splits, self.metrics = adapter, history, splits, metrics

    def save(self, directory):
        self.adapter.save_adapter(directory)
        for name in ('history', 'splits', 'metrics'):
            getattr(self, name).to_csv(Path(directory) / (name + '.tsv'), sep='\t', index=False)


def fit_latent_adapter(target, target_metadata, reference, biological_labels=None,
                       epochs=200, patience=25, batch_size=96, learning_rate=.001, seed=42):
    """Fit on training patients only; select epoch on validation, report test MMD.

    Reference is a ReferenceAtlas; target columns must match its frozen encoder.
    cancer_type is always included, even when extra biological labels are used.
    """
    validate_features(target, 'target latent')
    if list(target.columns) != list(reference.latent.columns):
        raise ValueError('Target and reference latent columns must agree')
    if any(not isinstance(v, int) or v < 1 for v in (epochs, patience, batch_size)):
        raise ValueError('epochs, patience, batch_size must be positive integers')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    if isinstance(biological_labels, str):
        biological_labels = [biological_labels]
    columns = list(dict.fromkeys(['cancer_type'] + list(biological_labels or [])))
    target_meta = target_metadata.reindex(target.index)
    tl = _labels(target_metadata, target.index, columns)
    if not set(columns).issubset(reference.metadata.columns):
        raise ValueError('Reference metadata lacks requested biological label columns')
    labelled = reference.metadata[columns].replace(r'^\s*$', np.nan, regex=True).notna().all(axis=1)
    labelled_meta = reference.metadata.loc[labelled]
    rl = _labels(labelled_meta, labelled_meta.index, columns)
    groups = sorted(set(tl))
    if set(groups) - set(rl):
        raise ValueError('Target contains cancer/biological strata absent from reference')
    # Exclude unrelated reference strata rather than pulling target toward them.
    eligible_ids = labelled_meta.index[np.array([g in groups for g in rl], dtype=bool)]
    keep = reference.metadata.index.isin(eligible_ids)
    ref_meta = reference.metadata.loc[keep]
    rl = [g for g in rl if g in groups]
    ts = _splits(target_meta, tl, seed)
    rs = _splits(ref_meta, rl, seed + 1)
    x = reference.latent.loc[keep].to_numpy(np.float32)
    y = target.to_numpy(np.float32)
    mu, sd = x[rs == 'train'].mean(0), np.maximum(x[rs == 'train'].std(0), 1e-5)
    tf.random.set_seed(seed)
    adapter = LatentAdapter(mu, sd, reference.model_signature, target.columns, columns, groups)
    adapter(tf.constant(y[:1]))
    optimizer = keras.optimizers.Adam(learning_rate)
    indices = {}
    for g in groups:
        for split in ('train', 'validation', 'test'):
            indices[g, split] = (np.flatnonzero((ts == split) & np.array([t == g for t in tl])),
                                  np.flatnonzero((rs == split) & np.array([r == g for r in rl])))
    def objective(t, r):
        before = (t-mu)/sd
        after = (adapter(t)-mu)/sd
        return (_mmd(after, (r-mu)/sd) + .05*tf.reduce_mean((after-before)**2)
                + .02*tf.reduce_mean((_distance(after, after)-_distance(before, before))**2))
    rng = np.random.RandomState(seed)
    best, wait, weights, history = np.inf, 0, None, []
    for epoch in range(epochs):
        for step in range(max(1, int(np.ceil(np.sum(ts == 'train')/batch_size)))):
            with tf.GradientTape() as tape:
                terms = []
                for g in groups:
                    ti, ri = indices[g, 'train']
                    ti = rng.choice(ti, min(len(ti), batch_size), replace=False)
                    ri = rng.choice(ri, min(len(ri), batch_size), replace=False)
                    terms.append(objective(tf.constant(y[ti]), tf.constant(x[ri])))
                loss = tf.add_n(terms)/len(terms)
            if not np.isfinite(float(loss)):
                raise ValueError('Nonfinite adaptation loss')
            optimizer.apply_gradients(zip(tape.gradient(loss, adapter.trainable_variables), adapter.trainable_variables))
        # Fixed deterministic validation subset bounds quadratic memory; never use test.
        vals = []
        for g in groups:
            ti, ri = indices[g, 'validation']
            vals.append(float(objective(tf.constant(y[ti[:256]]), tf.constant(x[ri[:256]]))))
        value = float(np.mean(vals))
        if not np.isfinite(value):
            raise ValueError('Nonfinite validation loss')
        history.append({'epoch': epoch + 1, 'validation_loss': value})
        if value < best - 1e-6:
            best, weights, wait = value, adapter.get_weights(), 0
        else:
            wait += 1
        if wait >= patience:
            break
    adapter.set_weights(weights)
    metrics = []
    for g in groups:
        ti, ri = indices[g, 'test']
        # Deterministic cap is reported, not silently treated as all-test evaluation.
        ti, ri = ti[:256], ri[:256]
        z = adapter(tf.constant(y[ti])).numpy()
        for name, values in [('before', y[ti]), ('after', z)]:
            metrics.append({'stratum': json.dumps(g), 'version': name, 'n_target_test': len(ti),
                            'n_reference_test': len(ri),
                            'MMD': float(_mmd(tf.constant((values-mu)/sd), tf.constant((x[ri]-mu)/sd))),
                            'mean_squared_displacement_SD': float(np.mean(((values-y[ti])/sd)**2))})
    splits = pd.concat([target_meta.assign(cohort='target', split=ts),
                        ref_meta.assign(cohort='reference', split=rs)]).rename_axis('sample_id').reset_index()
    return AdaptationResult(adapter, pd.DataFrame(history), splits, pd.DataFrame(metrics))
