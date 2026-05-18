# CancerXpress

CancerXpress is a pan-cancer transcriptomic representation learning framework for bulk RNA-seq data. It learns de-batched expression profiles and transferable biological representations that can be used for downstream prediction, functional state analysis, and interpretable single-sample characterization.

It is designed for use cases where transcriptomic models need to generalize across cohorts, sequencing workflows, and cancer types rather than only perform well within a single dataset.

## Key Features

- batch effect correction and corrected expression reconstruction
- latent embedding extraction for downstream analysis
- pan-cancer representation learning across cohorts and cancer types
- Module Eigenpathway prediction
- primary site prediction
- cancer type prediction
- survival risk prediction with external cohort evaluation
- functional state characterization through low-dimensional transcriptomic axes
- sample-level attribution through integrated gradients
- automatic gene ID conversion between `Ensembl` IDs and `HGNC symbols`
- training and fine-tuning scripts under `training/`

## Quick Start

Install the package:

```bash
cd CancerXpress
pip install -r requirements.txt
pip install -e .
```

Run the full inference pipeline on the example data:

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --cancer-type-file examples/scanb_demo_3samples_cancer_type.tsv \
  --outdir output/demo \
  --task all \
  --gene-id-type ensembl
```

Use the Python API:

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('examples/scanb_demo_3samples.tsv', sep='\t', index_col=0)
cancer_type = pd.read_csv(
    'examples/scanb_demo_3samples_cancer_type.tsv',
    sep='\t',
    index_col=0,
)['cancer_type']

model = cx.CancerXpress()
latent, corrected = model.batch_correct(expr, gene_id_type='ensembl')
module_eigenpathway_predictions = model.predict_me(expr, gene_id_type='ensembl')
risk_scores = model.predict_survival_risk(expr, cancer_type=cancer_type, gene_id_type='ensembl')
```

## Project Layout

- `cancerxpress/`: installable Python package
- `resources/`: required assets, encoders, mappings, and model manifests/checkpoints
- `training/`: key training and fine-tuning scripts
- `examples/`: small example input files

## Packaged Models

- `resources/models/pretrained/batch_correction/`
  Model parameters for batch correction, latent embedding extraction, and corrected expression reconstruction.
- `resources/models/finetuned/me/`
  Model parameters for Module Eigenpathway prediction.
- `resources/models/finetuned/primary_site/`
  Model parameters for primary site classification.
- `resources/models/finetuned/cancer_type/`
  Model parameters for cancer type classification.
- `resources/models/finetuned/survival_risk/`
  Model parameters for survival risk prediction.

For GitHub publication, large model checkpoints should be distributed via external links instead of being committed to the repository. See [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md).
For a repository-level keep/exclude policy under `resources/`, see [RESOURCES_POLICY.md](RESOURCES_POLICY.md).

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Python API

CancerXpress provides a single high-level entry point for preprocessing, prediction, and attribution:

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('examples/scanb_demo_3samples.tsv', sep='\t', index_col=0)
cancer_type = pd.read_csv(
    'examples/scanb_demo_3samples_cancer_type.tsv',
    sep='\t',
    index_col=0,
)['cancer_type']

model = cx.CancerXpress()
latent, corrected = model.batch_correct(expr, gene_id_type='ensembl')
module_eigenpathway_predictions = model.predict_me(expr, gene_id_type='ensembl')
primary = model.predict_primary_site(expr, gene_id_type='ensembl')
cancer = model.predict_cancer_type(expr, gene_id_type='ensembl')
risk_scores = model.predict_survival_risk(
    expr,
    cancer_type=cancer_type,
    gene_id_type='ensembl',
)

module_eigenpathway_attribution = model.attribute_me(expr.iloc[[0]], me_name='MEblue', gene_id_type='ensembl')
survival_risk_attribution = model.attribute_survival_risk(
    expr.iloc[[0]],
    cancer_type='BRCA',
    gene_id_type='ensembl',
)
```

Main API methods:

- `batch_correct(...)`
- `predict_me(...)`
- `predict_primary_site(...)`
- `predict_cancer_type(...)`
- `predict_survival_risk(...)`
- `attribute_me(...)`
- `attribute_primary_site(...)`
- `attribute_cancer_type(...)`
- `attribute_survival_risk(...)`

## CLI

### Run the Full Inference Pipeline

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --cancer-type-file examples/scanb_demo_3samples_cancer_type.tsv \
  --outdir output/demo \
  --task all \
  --gene-id-type ensembl
```

Expected outputs:

- `latent.tsv`
- `corrected_expression.tsv`
- `module_eigenpathway_predictions.tsv`
- `primary_site_predictions.tsv`
- `cancer_type_predictions.tsv`
- `survival_risk.tsv`

### Run Attribution from CLI

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --outdir output/attr_demo \
  --task attribution \
  --attribution-task me \
  --target-name MEblue \
  --sample-id TCGA-A2-A0T2-01A \
  --gene-id-type ensembl
```

Expected outputs:

- `attribution.tsv`
- `attribution_prediction.tsv`

### Input Expression with HGNC Symbols

If the input matrix uses gene symbols instead of Ensembl IDs:

```bash
cancerxpress \
  --input your_symbol_expression.tsv \
  --cancer-type-file your_cancer_type.tsv \
  --outdir output/run_symbol \
  --task all \
  --gene-id-type symbol
```

## Training and Fine-Tuning

The `training/` directory currently keeps three key scripts:

- `training/ME_regression.py`: regression training for eight Module Eigenpathway targets
- `training/classifier.py`: fine-tuning for primary site or cancer type classification
- `training/improved_risk_prediction.py`: survival risk model training

### Example: Module Eigenpathway Training

```bash
python training/ME_regression.py \
  --data /path/to/expression_matrix.tsv \
  --label /path/to/me_targets.tsv \
  --ckpt-dir training_checkpoints/me_regressor
```

### Example: Primary Site Classifier Training

```bash
python training/classifier.py \
  --data /path/to/expression_matrix.tsv \
  --label /path/to/primary_site_labels.tsv \
  --encoder resources/encoders/primary_site_encoder.pkl \
  --class-type multiclass \
  --ckpt-dir training_checkpoints/primary_site_classifier
```

### Example: Survival Risk Training

```bash
python training/improved_risk_prediction.py \
  --data /path/to/expression_matrix.tsv \
  --label /path/to/survival_labels.tsv \
  --test-batches external_batch_1 external_batch_2 \
  --holdout-cancer-types optional_cancer_type
```

## What Was Not Migrated

The following content was intentionally not copied into CancerXpress:

- most prediction outputs under `output/`
- historical training logs
- historical checkpoints
- very large internal training matrices
- attribution scripts and attribution intermediate results

This keeps the project much cleaner for packaging, publication, and long-term maintenance.

## Notes

- The current survival risk model is a dual-input model and requires `cancer_type` during prediction.
- Training scripts do not automatically download datasets. You need to provide your own expression matrices and labels.
