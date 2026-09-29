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
- hierarchical gene-to-axis-to-risk attribution with exact ME Shapley,
  matched baselines, and completeness diagnostics
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
  --clinical-file examples/scanb_demo_3samples_label.tsv \
  --outdir output/demo \
  --task all \
  --gene-id-type ensembl
```

Run survival risk prediction only:

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --clinical-file examples/scanb_demo_3samples_label.tsv \
  --outdir output/risk_demo \
  --task risk \
  --gene-id-type ensembl
```

Use the Python API:

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('examples/scanb_demo_3samples.tsv', sep='\t', index_col=0)
clinical = pd.read_csv('examples/scanb_demo_3samples_label.tsv', sep='\t', index_col=0)

model = cx.CancerXpress()
latent, corrected = model.batch_correct(expr, gene_id_type='ensembl')
module_eigenpathway_predictions = model.predict_me(expr, gene_id_type='ensembl')
risk_scores = model.predict_survival_risk(expr, clinical=clinical, gene_id_type='ensembl')
```

Run the standalone survival risk example:

```bash
python examples/risk_prediction_demo.py
```

Run the standalone survival risk attribution example:

```bash
python examples/risk_attribution_demo.py
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
clinical = pd.read_csv('examples/scanb_demo_3samples_label.tsv', sep='\t', index_col=0)

model = cx.CancerXpress()
latent, corrected = model.batch_correct(expr, gene_id_type='ensembl')
module_eigenpathway_predictions = model.predict_me(expr, gene_id_type='ensembl')
primary = model.predict_primary_site(expr, gene_id_type='ensembl')
cancer = model.predict_cancer_type(expr, gene_id_type='ensembl')
risk_scores = model.predict_survival_risk(
    expr,
    clinical=clinical,
    gene_id_type='ensembl',
)

module_eigenpathway_attribution = model.attribute_me(expr.iloc[[0]], me_name='MEblue', gene_id_type='ensembl')
survival_risk_attribution = model.attribute_survival_risk(
    expr.iloc[[0]],
    clinical=clinical.iloc[[0]],
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

## Survival Risk Prediction

The packaged survival risk model is the run1 ME-and-clinical Cox model. CancerXpress first predicts the eight Module Eigenpathway scores from expression and then combines them with the clinical covariates.
It requires:

- an expression matrix
- a clinical table containing `age`, `sex`, `race`, `tumor_stage`, and `cancer_type`

### Required Input Format

Expression matrix:

- rows: samples
- columns: genes
- values: TPM-like expression matrix accepted by CancerXpress

Clinical file:

- same sample order as the expression matrix, or matching sample IDs in the first column
- must contain `age`, `sex`, `race`, `tumor_stage`, and `cancer_type`

Example clinical file:

```tsv
sample_id	age	sex	race	tumor_stage	cancer_type
sample_1	62	female	white	Stage II	BRCA
sample_2	55	male	asian	Stage III	LUAD
```

### Python Example

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('examples/scanb_demo_3samples.tsv', sep='\t', index_col=0)
clinical = pd.read_csv('examples/scanb_demo_3samples_label.tsv', sep='\t', index_col=0)

model = cx.CancerXpress()
risk_scores = model.predict_survival_risk(
    expr_tpm=expr,
    clinical=clinical,
    gene_id_type='ensembl',
)
print(risk_scores)
```

### Survival Risk Attribution Example

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('examples/scanb_demo_3samples.tsv', sep='\t', index_col=0)
clinical = pd.read_csv('examples/scanb_demo_3samples_label.tsv', sep='\t', index_col=0)

sample_id = expr.index[0]

model = cx.CancerXpress()
attribution = model.attribute_survival_risk(
    expr_tpm=expr.loc[[sample_id]],
    clinical=clinical.loc[[sample_id]],
    gene_id_type='ensembl',
)
```

### CLI Example

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --clinical-file examples/scanb_demo_3samples_label.tsv \
  --outdir output/risk_demo \
  --task risk \
  --gene-id-type ensembl
```

### Output

The risk prediction output is a table with one row per sample and one column:

- `risk_score`: relative Cox-model risk score

Interpretation:

- a higher `risk_score` means a higher predicted relative risk
- the score is mainly useful for ranking samples within a cohort or cancer setting
- it is not a direct survival probability

For attribution output:

- `survival_risk_prediction.tsv`: predicted risk score for the input sample
- `survival_risk_attribution.tsv`: gene-level integrated gradients attribution for survival risk

### Axis-Specific Risk Attribution

`attribute_axis_risk` explains one patient's selected functional-axis risk
contribution: exact ME Shapley followed by gene-to-axis integrated gradients
(IG). Clinical inputs remain fixed. This differs from the direct gene-to-risk
IG returned by `attribute_survival_risk`.

```python
import numpy as np
import pandas as pd
import cancerxpress as cx

# expr and clinical: sample-indexed inputs as in the examples above.
# Use the frozen training reference for this cancer and model/fold.
baseline_me = pd.read_csv('baseline_me.tsv', sep='\t', index_col=0).iloc[0]
result = cx.CancerXpress().attribute_axis_risk(
    expr.iloc[[0]], clinical.loc[expr.index[:1]], me_name='MEred',
    baseline_image=np.load('baseline_image.npy'),
    baseline_me=baseline_me,  # Series indexed by the eight ME names
    steps=64, internal_batch_size=16, gene_id_type='ensembl',
)
result.axis_shapley.to_csv('axis_shapley.tsv', sep='\t')
result.gene_risk_contribution.to_csv('gene_axis_risk.tsv', sep='\t')
print(result.diagnostics)
```

- `baseline_image` is a normalized model-input image, not raw TPM. Its selected
  ME prediction must match `baseline_me` (default tolerance: 0.001 ME units).
  Use training-only references; do not recenter on the external cohort.
- Gene contributions are `axis_shapley * gene_IG / sum(gene_IG)`, using all
  mapped genes. They are signed log-risk allocations, not percentages, causal
  effects, or exact gene-level Shapley values.
- `result.gene_to_axis_ig` retains the original gene IG. Diagnostics report
  completeness and allocation validity. A near-zero denominator or excessive
  IG error produces NaN contributions, not zero importance.
- Lower-level functions are also available: `exact_axis_shapley`,
  `gene_axis_integrated_gradients`, `allocate_axis_risk`, and
  `optimize_axis_baseline`. The last fits a synthetic baseline from a training
  mean input image, a binary mapped-gene pixel mask, and a frozen ME reference;
  inspect its returned optimization history before use.

## CLI

### Run the Full Inference Pipeline

```bash
cancerxpress \
  --input examples/scanb_demo_3samples.tsv \
  --clinical-file examples/scanb_demo_3samples_label.tsv \
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
  --clinical-file your_clinical_data.tsv \
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

- The packaged run1 survival model requires expression plus `age`, `sex`, `race`, `tumor_stage`, and `cancer_type` for every sample.
- Training scripts do not automatically download datasets. You need to provide your own expression matrices and labels.
