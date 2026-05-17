# CancerXpress

CancerXpress is a cleaned, package-ready version of the original `RNA-IMAGE` project. It is organized for GitHub release and Python packaging, while keeping only the essential code, resources, and latest model checkpoints needed for practical use.

CancerXpress currently supports:

- pretrained model loading
- batch correction of gene expression data
- latent embedding extraction
- corrected expression reconstruction
- ME prediction
- primary site prediction
- cancer type prediction
- survival risk prediction
- integrated gradients attribution for ME, primary site, cancer type, and survival risk
- automatic gene ID conversion between `Ensembl` IDs and `HGNC symbols`
- key training and fine-tuning scripts under `training/`

## Project Layout

- `src/cancerxpress/`: installable Python package
- `resources/`: required assets, encoders, mappings, and model manifests/checkpoints
- `training/`: key training and fine-tuning scripts
- `examples/`: small example input files

## Packaged Models

- `batch_correction`: source checkpoint `distill_pretrain_v3`
- `me`: source checkpoint `ME`
- `primary_site`: source checkpoint `primary_site_v2`
- `cancer_type`: source checkpoint `cancer_type_v2`
- `survival_risk`: source checkpoint `improved_survival_all_batches`

Only the latest checkpoint is kept for each model. Historical checkpoints are not included.

For GitHub publication, large model checkpoints should be distributed via external links instead of being committed to the repository. See [MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md).
For a repository-level keep/exclude policy under `resources/`, see [RESOURCES_POLICY.md](RESOURCES_POLICY.md).

## Installation

CancerXpress has been validated in a fresh Conda environment with Python 3.8 and TensorFlow 2.4 GPU dependencies.

### Recommended: Conda Environment from Scratch

```bash
cd CancerXpress
conda env create -f environment.yml
conda activate cancerxpress
```

### Alternative: Manual Installation

```bash
cd CancerXpress
pip install -r requirements.txt
pip install -e .
```

### Verified Smoke Test

After installation, run:

```bash
python scripts/smoke_test.py
```

This checks:

- batch correction
- latent embedding extraction
- corrected expression reconstruction
- ME prediction
- primary site prediction
- cancer type prediction
- survival risk prediction
- gene ID conversion
- integrated gradients attribution

## Python API

CancerXpress is intended to be used like this:

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
me = model.predict_me(expr, gene_id_type='ensembl')
primary = model.predict_primary_site(expr, gene_id_type='ensembl')
cancer = model.predict_cancer_type(expr, gene_id_type='ensembl')
risk = model.predict_survival_risk(
    expr,
    cancer_type=cancer_type,
    gene_id_type='ensembl',
)

me_attr = model.attribute_me(expr.iloc[[0]], me_name='MEblue', gene_id_type='ensembl')
risk_attr = model.attribute_survival_risk(
    expr.iloc[[0]],
    cancer_type='BRCA',
    gene_id_type='ensembl',
)
```

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
- `me_predictions.tsv`
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

- `training/ME_regression.py`: regression training for the 8 ME outputs
- `training/classifier.py`: fine-tuning for primary site or cancer type classification
- `training/improved_risk_prediction.py`: survival risk model training

### Example: ME Training

```bash
python training/ME_regression.py \
  --data /path/to/merged_data.tsv \
  --label /path/to/moduleEigenpathways.csv \
  --ckpt-dir training_checkpoints/ME_new
```

### Example: Primary Site Classifier Training

```bash
python training/classifier.py \
  --data /path/to/merged_data.tsv \
  --label /path/to/primary_site_label.tsv \
  --encoder resources/encoders/primary_site_encoder.pkl \
  --class-type multiclass \
  --ckpt-dir training_checkpoints/primary_site_new
```

### Example: Survival Risk Training

```bash
python training/improved_risk_prediction.py \
  --data /path/to/merged_data.tsv \
  --label /path/to/merged_label.tsv \
  --test-batches META-PRISM MMRF CPTAC \
  --holdout-cancer-types MM
```

## What Was Not Migrated

The following content was intentionally not copied into CancerXpress:

- most prediction outputs under `output/`
- historical training logs
- historical checkpoints
- very large training matrices such as `dataset/merged_data/merged_data.tsv`
- attribution scripts and attribution intermediate results

This keeps the project much cleaner for packaging, publication, and long-term maintenance.

## Notes

- The current survival risk model is a dual-input model and requires `cancer_type` during prediction.
- Training scripts do not automatically download datasets. You need to provide your own expression matrices and labels.
- GPU support for TensorFlow 2.4 depends on matching CUDA libraries. The provided `environment.yml` includes `cudatoolkit=11.0.3` and `cudnn=8.0.5`, which matches the validated development environment.
