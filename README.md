# CancerXpress

CancerXpress is a pan-cancer transcriptomic representation learning framework for bulk RNA-seq data. It learns de-batched expression profiles and transferable biological representations that can be used for downstream prediction, functional state analysis, and interpretable single-sample characterization.

It is designed for use cases where transcriptomic models need to generalize across cohorts, sequencing workflows, and cancer types rather than only perform well within a single dataset.

## Key Features

- batch effect correction and corrected expression reconstruction
- latent embedding extraction for downstream analysis
- pan-cancer representation learning across cohorts and cancer types
- Module Eigenpathway prediction
- survival risk prediction with external cohort evaluation
- functional state characterization through low-dimensional transcriptomic axes
- sample-level attribution through integrated gradients
- hierarchical gene-to-axis-to-risk attribution with exact ME Shapley,
  matched baselines, and completeness diagnostics
- automatic gene ID conversion between `Ensembl` IDs and `HGNC symbols`
- legacy research training scripts under `training/` (outside the package runtime compatibility matrix)

This release provides representation/batch correction, functional-axis and risk
prediction, attribution, reference search and target-cohort adaptation. It does
not provide cancer-type or primary-site prediction, plotting functions, or
automatic reference-data installation. Cancer type remains a user-supplied
clinical/biological label. Outputs are for research, not clinical diagnosis.

## Quick Start

Choose one of the two [installation methods](#installation):

| Method | Current version | Python |
| --- | --- | --- |
| Download the GitHub project and install locally | 1.1 development series | 3.8-3.13; 3.11 or 3.12 recommended |
| Install from PyPI | 1.0.0 released version | 3.8 only |

Prepare the separate model bundle before inference. Follow the
[step-by-step tutorial](tutorials/01_quickstart.md) for TPM inputs,
prediction, attribution, local reference queries and optional cohort adaptation.
The tutorial includes a reproducible synthetic-data walkthrough and explains
how to substitute real data; synthetic results are not biological validation.

The published PyPI `1.0.0` release remains Python-3.8-only until a new release
is published. Do not apply its historical `requirements-release-lock.txt`
constraints to this development version.

The examples below use files from the GitHub project. A PyPI installation does
not include those example files; download the project or use your own TPM and
clinical files.

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
- `tutorials/`: step-by-step workflows and interpretation notes

## Packaged Models

- `resources/models/pretrained/batch_correction/`
  Model parameters for batch correction, latent embedding extraction, and corrected expression reconstruction.
- `resources/models/finetuned/me/`
  Model parameters for Module Eigenpathway prediction.
- `resources/models/finetuned/survival_risk/`
  Model parameters for survival risk prediction.

Model parameters and reference data are installed manually and are not included
in this package. Keep the supplied bundle layout unchanged; older bundles may
contain unused classification checkpoints, which this API does not load.

## Installation

The 1.1 development series targets **Python 3.8-3.13**, with TensorFlow 2.13.x
on Python 3.8 and TensorFlow / tf-keras 2.20.x on Python 3.9-3.13.
TensorFlow Addons is no longer required. Keras 2 checkpoint compatibility is
handled internally without changing the process-wide Keras selection.

Python 3.14 and later are not yet advertised as supported. See
[Installation and Compatibility](tutorials/installation.md) for runtime
requirements and the model-regression test procedure.

### Option 1: Download the GitHub Project and Install Locally

Use a dedicated Python 3.8-3.13 environment. This option includes the source,
tutorials, and example files. Download the project using Git:

```bash
git clone https://github.com/Qiankun-1024/CancerXpress.git
cd CancerXpress
python -m pip install --upgrade pip
python -m pip install .
```

Alternatively, select **Code > Download ZIP** on the
[GitHub repository](https://github.com/Qiankun-1024/CancerXpress), extract it,
and run `python -m pip install .` inside the extracted `CancerXpress-main`
directory. Git is not needed for the ZIP method.

If you plan to edit the source, use `python -m pip install -e .` instead.
For reproducibility, use a specific Git commit or release tag rather than a
moving branch.

### Option 2: Install from PyPI

The currently published [PyPI release](https://pypi.org/project/cancerxpress/)
is **1.0.0 and requires Python 3.8**. It does not yet include the multi-Python
runtime changes on GitHub. In a dedicated Python 3.8 environment, run:

```bash
python -m pip install --upgrade pip
python -m pip install "cancerxpress==1.0.0"
python -m pip check
```

For Python 3.9-3.13, use Option 1 until a compatible release is published on
PyPI. The historical `requirements-release-lock.txt` can be used as pip
constraints to reproduce the tested 1.0.0 dependency versions; do not apply
that file to the GitHub development version.

**Both methods install code and dependencies only.** Model parameters and
reference data remain separate and must be installed manually. Prepare the
local model files as described in [Model Resources and Paths](#model-resources-and-paths)
before running predictions.

### Read the Docs

The repository includes `mkdocs.yml` and `.readthedocs.yaml`. The documentation
build installs only `docs/requirements.txt`, not TensorFlow or model resources.
See [Documentation Deployment](tutorials/readthedocs.md) for local preview and
the required account-level GitHub connection. The configuration alone does not
create a hosted Read the Docs project.

### Model Resources and Paths

The wheel and source distribution contain **code only**, not model weights or
patient data. The resource bundle includes inference checkpoints, gene mapping,
normalization assets, label encoders and clinical preprocessing metadata.

```python
import cancerxpress as cx

# Manually extract the model bundle to ~/.cache/cancerxpress/resources first.
# Construction and prediction never download missing files automatically.
model = cx.CancerXpress()

# Alternatively, use your own extracted bundle:
model = cx.CancerXpress(resource_dir="/path/to/resources")
# Override only weights if needed (assets still come from resource_dir):
model = cx.CancerXpress(
    resource_dir="/path/to/resources", model_dir="/path/to/other/models"
)
cx.verify_resources("/path/to/resources")
```

Set `CANCERXPRESS_HOME` to change the default resource root. Paths do not depend
on the working directory. CLI equivalents are `--resource-dir` and `--model-dir`.
For a local checkout using its existing resources, pass `resource_dir="resources"`
explicitly. Do not mix weights and preprocessing assets from different releases.

The downloader requires HTTPS and the expected SHA-256, verifies every file,
rejects unsafe archive paths, and never overwrites an existing destination.
Only install bundles from a trusted publisher: label encoders use Python pickle.
See [Manually Install Reference Data](#manually-install-reference-data) for the
resource and reference-directory layout. No published download URL is assumed.

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
- `predict_survival_risk(...)`
- `attribute_me(...)`
- `attribute_survival_risk(...)`
- `attribute_axis_risk(...)`

Attribution covers functional axes and survival risk, not classification.

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

### Reference Queries and Target-Cohort Adaptation

#### Manually Install Reference Data

The supported setup is currently **local files supplied separately**. No data
hosting service, DOI, account or automatic download is required. Installing the
Python package does not install model parameters or the reference database.
Missing files produce an error rather than trigger a download.

Place the files in this layout (or choose any absolute directory):

```text
~/.cache/cancerxpress/
  resources/                     # Model bundle, required for inference
    assets/                      # Keep the supplied bundle layout unchanged
    models/
    encoders/
    gene_annotation/
    mappings/
    resource_manifest.json
  reference/                     # Indexed data release, for sample retrieval
    reference_index.json.gz
    expression_samples_001.zip
    expression_samples_002.zip
    ...                          # Every ZIP named in reference_index.json.gz
    reference_atlas/              # Optional fast-loading query/adaptation atlas
      atlas.json
      samples.tsv
      features.npz
```

Do not unzip or rename the indexed ZIP files. They contain raw TPM, corrected
expression, latent, ME and sample metadata; these five components need not be
copied again as separate tables. ZIP counts are defined by the release index,
not hard-coded in the package.

| Use case | Required reference files |
| --- | --- |
| Predict or attribute your own TPM | None; only the matching model/resource bundle |
| Query or adapt against a supplied atlas | `atlas.json`, `samples.tsv`, `features.npz` |
| Retrieve stored sample expression/features | `reference_index.json.gz` and all indexed ZIP shards |

`atlas.json` records the format, feature column names and model/preprocessing
signature. `features.npz` contains numeric `latent` and `me` arrays in exactly the
row order of `samples.tsv`. The latter is a sample-indexed TSV with `sample_id`
and `cancer_type`; additional biological labels are optional. Preserve missing
reference labels. Do not combine feature files and models from different releases.

#### Read Specific Reference Samples

The indexed reference release uses uniform sample shards, not cancer-specific
archives. Each sample's TPM, corrected expression and features are independently
compressed inside an uncompressed ZIP container. A small index maps sample IDs
to exact byte ranges and SHA-256 checksums. This is different from the historical
whole-file `.tar.gz` release; those files cannot use this interface.

```python
from cancerxpress import SampleDatabase

# Default: ~/.cache/cancerxpress/reference, or CANCERXPRESS_REFERENCE_DIR.
db = SampleDatabase.from_directory()
# Alternatively: db = SampleDatabase.from_directory("/data/cancerxpress/reference")
ids = db.sample_ids[:2]  # Replace with the desired IDs from this release.
data = db.fetch(ids, components=["raw_tpm", "me", "metadata"])
# data["raw_tpm"]: samples x genes; data["me"]: samples x functional axes.
```

Supported components: `raw_tpm`, `corrected_expression`, `latent`, `me`, `metadata`.
Raw TPM and corrected expression preserve their original, potentially different
gene columns. Missing metadata labels are not inferred. Latent/ME use the frozen
release model recorded in the index. Opening the directory checks shard presence
and sizes; reading a component verifies its SHA-256. Local reads do not use the
network or create duplicate component caches. Only requested components are read.
`CANCERXPRESS_REFERENCE_DIR` controls reference data only; `CANCERXPRESS_HOME`
controls model resources. Neither default depends on the working directory.
Automatic data download can be added later without changing the local format.
The existing explicit-URL transport is optional and is not needed for this workflow.

A reference atlas stores sample-indexed latent means, eight ME values and
biological labels generated with one frozen model. It does not ship inside the
Python wheel. Use a supplied TCGA atlas or build one from reference TPM:

```python
atlas = cx.ReferenceAtlas.load("/data/cancerxpress/reference/reference_atlas")
# If only the indexed release is available, read its features (not expression):
# atlas = db.to_atlas()  # All samples; can take time on a large reference.
# atlas.save("/data/cancerxpress/reference/reference_atlas")  # New directory only.
# Or build your own atlas from reference TPM with model.build_reference(...).

hits = model.query(
    new_tpm, atlas, metadata=new_metadata,
    space="latent", k=10, metric="cosine", same_cancer=True,
    gene_id_type="ensembl",
)
# Use space="me" for functional-axis similarity; same_cancer=False for cross-cancer search.
```

Expression tables are samples x genes in TPM; metadata is indexed by the same
sample IDs and contains `cancer_type`. Missing reference cancer labels remain
missing (never inferred or imputed) and are allowed in unrestricted queries.
`same_cancer=True` requires a nonmissing query label; unlabelled reference rows
are excluded. Nonmissing query cancer labels must occur in the atlas.
Extra labels such as `subtype` or `grade` are optional.
Feature spaces and checkpoint/preprocessing hashes must match. Query distances
default to reference-only z-scored features; cosine similarity is 1-distance.
Euclidean mode reports distance (similarity is NaN). These are not confidence
scores or prognostic predictions. Self-exclusion uses exact sample ID.

Fit a shared target-only residual adapter while keeping the encoder/ME head frozen:

```python
fit = model.adapt(
    new_tpm, new_metadata, atlas,
    biological_labels=["subtype"],  # Omit for cancer-only alignment, including a single cancer.
    epochs=200, patience=25, gene_id_type="ensembl",
)
fit.save("cohort_adapter")
print(fit.metrics)  # Before/after held-out MMD and displacement, not clinical validation.

adapter = cx.LatentAdapter.load_adapter("cohort_adapter")
latent, me = model.encode_features(
    new_tpm, adapter=adapter, metadata=new_metadata, gene_id_type="ensembl"
)
hits = model.query(
    new_tpm, atlas, metadata=new_metadata, adapter=adapter,
    space="me", same_cancer=True, gene_id_type="ensembl"
)
```

Cancer type is ALWAYS part of the alignment strata. Additional labels must exist
in both cohorts; unseen strata are rejected. No survival or NMF program labels
are used automatically. Use genuine biology labels, not outcome-derived groups.
Reference rows missing any requested biological label are excluded from fitting;
target rows missing these labels are rejected rather than silently reassigned.
Each stratum needs at least five independent patients in each cohort to create
60/20/20-style training/validation/test splits; that minimum is a software check,
not evidence that a tiny cohort is statistically adequate.
Provide `patient_id` to keep repeated assays in one partition. Otherwise IDs
are assumed to identify independent patients.

The objective is equal-weight per-stratum RBF MMD + 0.05 latent displacement
+ 0.02 pairwise-distance distortion. Only reference TRAINING samples determine
normalization. Validation-only early stopping selects weights. Validation/test
MMD uses at most the first 256 samples per stratum, deterministically; metrics
record evaluated counts. Inspect saved split tables and biological preservation;
smaller MMD is not proof of biological recovery. Adapter use is explicit and
does not change the reference atlas or ordinary prediction/attribution calls.
Current adapter support covers `encode_features` and `query`; risk/IG APIs do
not automatically apply it. Do not mix adapted ME with old gene attribution.
Training/validation patients are not untouched external validation.

Historical TCGA OOF attribution tables and fixed-release-model query features
are distinct outputs. Use the companion TCGA handoff README for definitions,
QC, patient/gene ordering and the historical baseline differences.

## CLI Example

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

Pass the patient's TPM, clinical information, and reference TPM samples.
The package handles internal image conversion and computes exact ME Shapley,
gene-to-axis IG, and the selected axis's gene-level risk allocation.

```python
import pandas as pd
import cancerxpress as cx

expr = pd.read_csv('patient_tpm.tsv', sep='\t', index_col=0)
clinical = pd.read_csv('clinical.tsv', sep='\t', index_col=0)
reference = pd.read_csv('training_reference_tpm.tsv', sep='\t', index_col=0)

result = cx.CancerXpress().attribute_axis_risk(
    expr.iloc[[0]], clinical.loc[expr.index[:1]],
    me_name='MEred', reference_tpm=reference, gene_id_type='ensembl',
)
result.axis_shapley.to_csv('axis_shapley.tsv', sep='\t')
result.gene_risk_contribution.to_csv('gene_axis_risk.tsv', sep='\t')
print(result.diagnostics)
```

- Both TPM matrices have samples as rows and genes as columns. Use a fixed
  training reference from the corresponding cancer/model/fold, not the test
  patient itself or an external cohort recentered separately. Reference patients
  do not need clinical information; the target patient's clinical inputs stay fixed.
- Internally, reference samples are individually preprocessed and their model-input
  images averaged. The model prediction of this average supplies the ME baseline.
  No image or ME-reference file is required from the user. This baseline is NOT
  the mean of individual ME predictions and may differ from earlier research
  results using an optimized training-mean-ME baseline.
- Outputs: `axis_shapley` (eight axes), `gene_to_axis_ig`,
  `gene_risk_contribution`, and `diagnostics` (completeness errors and validity).
  Allocation is `axis_shapley * gene_IG / sum(gene_IG)`, over all mapped genes.
  Values are signed log-risk allocations, not percentages or causal effects.
  Invalid allocations become NaN.
- This differs from direct gene-to-risk IG (`attribute_survival_risk`).
  Advanced primitives remain available: `exact_axis_shapley`,
  `gene_axis_integrated_gradients`, `allocate_axis_risk`, and
  `optimize_axis_baseline`.

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
- `me_predictions.tsv`
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

The GitHub checkout contains research training scripts under `training/`.
They are not installed by the wheel or source distribution and are not a
unified, supported pretraining API. The supported package adaptation entry point
is `model.adapt(...)`, described above. Repository-only scripts include:

- `training/ME_regression.py`: regression training for eight Module Eigenpathway targets
- `training/improved_risk_prediction.py`: survival risk model training

### Example: Module Eigenpathway Training

```bash
python training/ME_regression.py \
  --data /path/to/expression_matrix.tsv \
  --label /path/to/me_targets.tsv \
  --ckpt-dir training_checkpoints/me_regressor
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
