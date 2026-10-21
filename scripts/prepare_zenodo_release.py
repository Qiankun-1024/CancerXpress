"""Export active inference checkpoints and build a data-free resource bundle."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(source, stage, task):
    import numpy as np
    import tensorflow as tf
    from cancerxpress.utils import model_loader
    from cancerxpress.predictor import Predictor
    from cancerxpress.resources import load_risk_preprocessing
    encoder = source / 'models/pretrained/batch_correction'
    dummy = tf.zeros((1, 128, 256, 1))
    if task == 'batch_correction':
        model = model_loader.load_model(ckpt_path=str(encoder))
        ckpt = tf.train.Checkpoint(bio_generator=model)
        probe = model.decode(model.encode(dummy)[0], apply_sigmoid=True).numpy()
        relative = 'models/pretrained/batch_correction'
    elif task == 'survival_risk':
        meta = load_risk_preprocessing(source / 'mappings/survival_risk_preprocessing.json')
        model, _ = Predictor._load_run1_survival_model(str(source / 'models/finetuned/survival_risk'), meta)
        ckpt = tf.train.Checkpoint(model=model)
        dummy = {'me': tf.zeros((1, 8)), 'age': tf.zeros((1, 1))}
        dummy.update({k: tf.zeros((1,), tf.int32) for k in ['sex', 'race', 'stage', 'cancer']})
        probe = model(dummy, training=False).numpy()
        relative = 'models/finetuned/survival_risk'
    else:
        dim = {'me': 8, 'primary_site': 35, 'cancer_type': 41}[task]
        model = model_loader.load_predict_model('latest', str(source / 'models/finetuned' / task),
                                                dim, None if task == 'me' else 'softmax',
                                                encoder_ckpt_path=str(encoder))
        probe = model(dummy, training=False).numpy()
        ckpt = tf.train.Checkpoint(classifier=model)
        relative = 'models/finetuned/' + task
    target = stage / relative
    target.mkdir(parents=True, exist_ok=True)
    path = ckpt.save(str(target / 'ckpt'))
    # TensorFlow state files must be portable, never retain export machine paths.
    name = Path(path).name
    (target / 'checkpoint').write_text('model_checkpoint_path: "' + name + '"\nall_model_checkpoint_paths: "' + name + '"\n')
    variables = tf.train.list_variables(path)
    assert not any('optimizer' in key.lower() for key, _ in variables)
    assert np.isfinite(probe).all()
    print(task, 'exported', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'resources')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default=None)
    args = parser.parse_args()
    stage = args.output / 'resources'
    if args.task:
        worker(args.source, stage, args.task)
        return
    if args.output.exists():
        raise FileExistsError('Choose a fresh release output directory')
    stage.mkdir(parents=True)
    for folder in ['assets', 'encoders', 'gene_annotation', 'mappings']:
        (stage / folder).mkdir()
    # Explicit allowlist: never collect datasets, examples, caches or patient files.
    files = [
        'assets/tsne_sorted_pixels_coords.csv', 'assets/max_norm_tpm.tsv',
        'encoders/primary_site_encoder.pkl', 'encoders/cancer_type_encoder.pkl',
        'encoders/sample_type_encoder.pkl', 'gene_annotation/ensembl_symbol_id.txt',
        'mappings/survival_risk_preprocessing.json',
    ]
    for relative in files:
        shutil.copy2(args.source / relative, stage / relative)
    for task in ['batch_correction', 'me', 'primary_site', 'cancer_type', 'survival_risk']:
        subprocess.run([sys.executable, __file__, '--source', str(args.source),
                        '--output', str(args.output), '--task', task], check=True)
    from cancerxpress.downloads import sha256_file, verify_resources
    files = [{'path': str(p.relative_to(stage)), 'size': p.stat().st_size, 'sha256': sha256_file(p)}
             for p in sorted(stage.rglob('*')) if p.is_file()]
    (stage / 'resource_manifest.json').write_text(json.dumps({
        'resource_version': '1.0.0', 'package_version': '1.0.0',
        'content': 'Inference weights and model preprocessing resources; no patient data',
        'files': files}, indent=2))
    verify_resources(stage)
    archive = args.output / 'CancerXpress-models-v1.0.0.tar.gz'
    with tarfile.open(str(archive), 'w:gz', compresslevel=1) as tar:
        for file in sorted(stage.rglob('*')):
            if file.is_file():
                tar.add(str(file), arcname=str(file.relative_to(stage)), recursive=False)
    shutil.copy2(stage / 'resource_manifest.json', args.output / 'resource_manifest.json')
    checksum = sha256_file(archive)
    (args.output / 'SHA256SUMS.txt').write_text(checksum + '  ' + archive.name + '\n')
    print('READY', archive, checksum, flush=True)


if __name__ == '__main__':
    main()
