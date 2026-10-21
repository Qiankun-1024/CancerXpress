"""Run against an installed wheel from outside the source checkout."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import cancerxpress as cx

parser = argparse.ArgumentParser()
parser.add_argument('--resources', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--source-resources')
args = parser.parse_args()
assert 'site-packages' in cx.__file__, cx.__file__
model = cx.CancerXpress(resource_dir=args.resources)
genes = model.preprocessor.training_gene_order
rng = np.random.RandomState(123)
expr = pd.DataFrame(rng.gamma(2, 5, size=(2, len(genes))), columns=genes, index=['synthetic1', 'synthetic2'])
clinical = pd.DataFrame({'age': [60, 61], 'sex': ['female'] * 2, 'race': ['white'] * 2,
                         'tumor_stage': ['stage ii'] * 2, 'cancer_type': ['BRCA'] * 2}, index=expr.index)
report = {'package': cx.__file__, 'synthetic_samples': True}
functional_axes = model.predict_functional_axes(expr, gene_id_type='ensembl')
risk = model.predict_survival_risk(expr, clinical, gene_id_type='ensembl')
assert functional_axes.shape == (2, 8) and np.isfinite(functional_axes.values).all()
assert risk.shape == (2, 1) and np.isfinite(risk.values).all()
report['functional_axis_prediction'] = 'passed'
report['clinical_risk_prediction'] = 'passed'
for name in ['primary_site', 'cancer_type']:
    assert not hasattr(model, 'predict_' + name)
report['classification_prediction_removed'] = True
latent, corrected = model.batch_correct(expr.iloc[:1], gene_id_type='ensembl')
assert latent.shape == (1, 100) and np.isfinite(corrected.values).all()
report['batch_correction'] = 'passed'
result = model.attribute_axis_risk(expr.iloc[:1], clinical.iloc[:1], 'developmental_regulation', expr.iloc[1:],
                                   steps=64, internal_batch_size=8, gene_id_type='ensembl')
assert np.isfinite(result.gene_to_axis_ig.values).all()
assert np.abs(result.diagnostics.shapley_completeness_error.iloc[0]) < 1e-5
report['attribution_QC'] = result.diagnostics.reset_index(drop=True).to_dict(orient='records')
report['gene_IG_shape'] = list(result.gene_to_axis_ig.shape)
# A second instance with a different root must not change the first instance.
if args.source_resources:
    original = cx.CancerXpress(resource_dir=args.source_resources)
    source_axes = original.predict_functional_axes(expr, gene_id_type='ensembl')
    source_risk = original.predict_survival_risk(expr, clinical, gene_id_type='ensembl')
    np.testing.assert_allclose(functional_axes.values, source_axes.values, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(risk.values, source_risk.values, rtol=1e-5, atol=1e-6)
    report['export_prediction_equivalence'] = 'passed'
manifest = cx.verify_resources(args.resources)
report['verified_resource_files'] = len(manifest['files'])
Path(args.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
