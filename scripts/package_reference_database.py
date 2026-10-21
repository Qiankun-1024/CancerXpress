"""Prepare a resumable local release, without uploading or granting sharing rights."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cancerxpress._legacy import axis_payload, axis_labels


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def compressed_copy(src, dest):
    """Reuse only a complete gzip whose decompressed digest matches the source."""
    source_hash = digest(src)
    if dest.exists():
        print('Validating existing compressed file', dest.name, flush=True)
        h = hashlib.sha256()
        with gzip.open(dest, 'rb') as inp:
            for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                h.update(block)
        if h.hexdigest() != source_hash:
            raise ValueError('Existing compressed file differs from source: ' + dest.name)
    else:
        print('Compressing', src.name, flush=True)
        tmp = dest.with_name(dest.name + '.partial')
        with src.open('rb') as inp, gzip.open(tmp, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(inp, out, length=8*1024*1024)
        tmp.replace(dest)
    return {'path': dest.name, 'size': dest.stat().st_size, 'sha256': digest(dest),
            'source_file': src.name, 'source_sha256': source_hash}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--checkpoint-dir', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    if (a.output / 'COMPLETE').exists():
        raise FileExistsError('Release is already complete; do not silently overwrite it')
    cache = a.checkpoint_dir or a.output.parent / ('.' + a.output.name + '-checkpoints')
    cache.mkdir(parents=True, exist_ok=True)
    # Prevent two processes writing the same release at once.
    import fcntl
    lock = (cache / 'export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta = pd.read_csv(a.source / 'sample_metadata.tsv', sep='\t', index_col=0)
    if not meta.index.is_unique or meta.index.hasnans or 'cancer_type' not in meta:
        raise ValueError('Metadata requires unique IDs and a cancer_type column')
    missing = meta.cancer_type.replace(r'^\s*$', np.nan, regex=True).isna()
    print('Samples:', len(meta), 'missing cancer_type:', int(missing.sum()), flush=True)
    (a.output / 'README.txt').write_text('CancerXpress reference database v1.0.0\n'
        'Matrices are samples x features; join by sample ID, never assume row order.\n'
        'Expression and original metadata are individual gzip files.\n'
        'historical_biological_representation is original, NOT the frozen-model query atlas.\n'
        'reference_atlas contains latent and functional-axis recomputed from the released model.\n'
        'Missing cancer labels remain missing: no imputation, no sample removal.\n'
        'Unlabelled samples support unrestricted queries, not cancer-stratified adaptation.\n'
        'Original metadata includes clinical fields: verify source redistribution permissions.\n'
        'This bundle is not a declaration of redistribution permission.\n'
        'Reference samples are not independent validation data.\n')
    meta.loc[missing].to_csv(a.output / 'samples_missing_cancer_type.tsv', sep='\t', index_label='sample_id')
    records = []
    for src in sorted(a.source.iterdir()):
        if src.is_file():
            name = ('historical_' if src.name == 'biological_representation.tsv' else '') + src.name
            records.append(compressed_copy(src, a.output / (name + '.gz')))
            write_json(cache / 'compression_records.json', records)
    from cancerxpress import CancerXpress, ReferenceAtlas
    model = CancerXpress(resource_dir=a.resources)
    signature = model._feature_signature()
    source_hashes = {r['source_file']: r['source_sha256'] for r in records}
    state = {'checkpoint_format': 1, 'model_signature': signature,
             'expression_sha256': source_hashes['raw_tpm.tsv'],
             'metadata_sha256': source_hashes['sample_metadata.tsv'], 'chunk_size': 64,
             'gene_id_type': 'ensembl'}
    state_path = cache / 'feature_state.json'
    if state_path.exists() and json.loads(state_path.read_text()) != state:
        raise ValueError('Checkpoint data/model signature changed; choose a new checkpoint directory')
    write_json(state_path, state)
    z_all, axes_all, seen = [], [], []
    for number, batch in enumerate(pd.read_csv(a.source / 'raw_tpm.tsv', sep='\t', index_col=0, chunksize=64)):
        checkpoint = cache / ('features_%06d.npz' % number)
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                if not np.array_equal(saved['sample_ids'], batch.index.astype(str)):
                    raise ValueError('Checkpoint sample order mismatch')
                z = pd.DataFrame(saved['latent'], index=batch.index, columns=saved['latent_columns'])
                functional_axes = pd.DataFrame(axis_payload(saved), index=batch.index,
                    columns=axis_labels(axis_payload(saved, 'functional_axis_columns')))
        else:
            z, functional_axes = model.encode_features(batch, batch_size=16, gene_id_type='ensembl')
            tmp = checkpoint.with_name(checkpoint.name + '.partial')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, sample_ids=np.asarray(batch.index, dtype=str),
                                    latent=z.values, functional_axes=functional_axes.values,
                                    latent_columns=np.asarray(z.columns, dtype=str),
                                    functional_axis_columns=np.asarray(functional_axes.columns, dtype=str))
            tmp.replace(checkpoint)
        if not np.isfinite(z.values).all() or not np.isfinite(functional_axes.values).all():
            raise ValueError('Nonfinite frozen features')
        z_all.append(z); axes_all.append(functional_axes); seen.extend(batch.index)
        print('Frozen features', len(seen), '/', len(meta), flush=True)
    if len(seen) != len(set(seen)) or set(seen) != set(meta.index):
        raise ValueError('Expression/metadata sample mismatch')
    # String column names are consistent for newly computed and cached batches.
    for z in z_all: z.columns = z.columns.astype(str)
    for functional_axes in axes_all: functional_axes.columns = functional_axes.columns.astype(str)
    atlas = ReferenceAtlas(pd.concat(z_all), pd.concat(axes_all), meta.loc[seen], signature)
    atlas_dir = a.output / 'reference_atlas'
    if atlas_dir.exists():
        old = ReferenceAtlas.load(atlas_dir)
        if (old.model_signature != signature or not old.latent.equals(atlas.latent)
                or not old.functional_axes.equals(atlas.functional_axes) or not old.metadata.index.equals(atlas.metadata.index)):
            raise ValueError('Existing atlas does not match computed features')
    else:
        stage = cache / 'atlas_stage'
        if stage.exists(): shutil.rmtree(stage)
        atlas.save(stage)
        stage.rename(atlas_dir)
    check = ReferenceAtlas.load(atlas_dir)
    if len(check.metadata) != len(meta) or check.metadata.cancer_type.isna().sum() != missing.sum():
        raise ValueError('Atlas roundtrip sample/label mismatch')
    records_by_path = {r['path']: r for r in records}
    for file in sorted(a.output.rglob('*')):
        if file.is_file() and file.name not in ('manifest.json','SHA256SUMS.txt','COMPLETE'):
            name = str(file.relative_to(a.output))
            if name not in records_by_path:
                records_by_path[name] = {'path': name, 'size': file.stat().st_size, 'sha256': digest(file)}
    records = list(records_by_path.values())
    write_json(a.output / 'manifest.json', {'version': '1.0.0', 'n_samples': len(meta),
               'n_missing_cancer_type': int(missing.sum()), 'model_signature': signature, 'files': records})
    (a.output / 'SHA256SUMS.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in records))
    (a.output / 'COMPLETE').write_text('Local packaging complete; permissions require review before publication.\n')
    print('READY', a.output, flush=True)


if __name__ == '__main__':
    main()
