# CancerXpress Quickstart

This tutorial covers batch correction, functional-axis and risk prediction, attribution, and sample similarity search, followed by optional target-cohort domain adaptation. The input is a **samples-by-genes matrix of nonnegative TPM values**, not an expression image. Do not log-transform, z-score, or apply additional cohort-wise normalization before passing TPM to these interfaces.

The core walkthrough runs on synthetic inputs so you can check the software without obtaining patient data first. Synthetic outputs are not evidence of biological risk, pathway activity, or clinical validity. For research, substitute your own TPM matrix and genuine clinical labels. This release does not provide cancer-type or primary-site prediction, or plotting interfaces.

## 1. Install the Package and Prepare Model Resources

This tutorial now targets the **1.1 development series**, with a dependency matrix for Python 3.8-3.13. Use Python 3.11 or 3.12 for a new environment. The published PyPI `1.0.0` wheel remains Python-3.8-only; install the development checkout for the new runtime. See [Installation and Compatibility](installation.md).

```bash
conda create -n cancerxpress python=3.11 -y
conda activate cancerxpress
python -m pip install --upgrade pip
cd /path/to/CancerXpress
python -m pip install --only-binary=:all: .
python -m pip check
```

The package contains code only, not model parameters or the reference database. Manually extract the matching model bundle and preserve its directory structure:

```text
/your/data/cancerxpress/models_v1/
    assets/
    encoders/
    gene_annotation/
    mappings/
    models/
    resource_manifest.json
```

Set the resource and output paths so the code does not depend on the working directory:

```bash
export CANCERXPRESS_HOME=/your/data/cancerxpress/models_v1
export CANCERXPRESS_TUTORIAL_OUTPUT=/your/work/cancerxpress_tutorial
# CPU walkthrough. For GPU use, configure a runtime matching your TensorFlow version.
export CUDA_VISIBLE_DEVICES=-1
```

Load model bundles only from trusted sources: they include serialized preprocessing resources. Execute the core walkthrough blocks below sequentially in the same Python session.

<!-- runnable -->
```python
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import cancerxpress as cx

resource_dir = Path(os.environ["CANCERXPRESS_HOME"]).expanduser()
out = Path(os.environ["CANCERXPRESS_TUTORIAL_OUTPUT"]).expanduser()
out.mkdir(parents=True, exist_ok=True)

print("CancerXpress:", version("cancerxpress"))
manifest = cx.verify_resources(resource_dir)
print("Verified model/resource files:", len(manifest["files"]))
model = cx.CancerXpress(resource_dir=resource_dir)
```

## 2. Prepare TPM and Sample Labels

### Core Walkthrough: Create Synthetic Inputs

Create four explicitly labeled synthetic samples. The nonnegative expression values are simulated inputs, not TPM measured by sequencing; all clinical labels are fictional. These inputs are for software checks only and must not replace real observations in research.

<!-- runnable -->
```python
rng = np.random.RandomState(123)
genes = model.preprocessor.training_gene_order
sample_ids = ["synthetic1", "synthetic2", "synthetic3", "synthetic4"]
expr = pd.DataFrame(
    rng.gamma(2.0, 5.0, size=(4, len(genes))),
    index=sample_ids, columns=genes,
)
clinical = pd.DataFrame({
    "age": [60, 61, 55, 67],
    "sex": ["female"] * 4,
    "race": ["white"] * 4,
    "tumor_stage": ["stage ii"] * 4,
    "cancer_type": ["BRCA"] * 4,
}, index=sample_ids)
metadata = clinical[["cancer_type"]].copy()
gene_id_type = "ensembl"
```

### Real Analysis: Replace the Synthetic Inputs

Use the following code instead of the synthetic input block when working with real data. It is not required for the synthetic walkthrough:

```python
expr = pd.read_csv("/your/data/expression_tpm.tsv", sep="\t", index_col=0)
clinical = pd.read_csv("/your/data/clinical.tsv", sep="\t", index_col=0)
assert set(expr.index).issubset(clinical.index)
clinical = clinical.loc[expr.index]
metadata = clinical[["cancer_type"]].copy()
gene_id_type = "ensembl"  # Use "symbol" for HGNC gene symbols.
```

The first column of the TPM table contains sample IDs; the remaining columns contain genes. Transpose the matrix after loading if your file uses genes as rows and samples as columns. The clinical table must contain at least `age`, `sex`, `race`, `tumor_stage`, and `cancer_type`, with matching sample IDs. Cancer type, such as `BRCA`, is a user-supplied label, not a model prediction.

Provide the full set of measured genes whenever possible, not just the top 100 genes or selected markers. Successful execution does not establish reliability when many model genes are missing. Do not invent clinical labels to make the workflow run.

<!-- runnable -->
```python
assert expr.index.is_unique and expr.columns.is_unique
assert clinical.index.is_unique
assert np.isfinite(expr.to_numpy()).all()
assert (expr.to_numpy() >= 0).all()
assert set(expr.index).issubset(clinical.index)
clinical = clinical.loc[expr.index]
metadata = metadata.loc[expr.index]
```

## 3. Run Batch Correction and Prediction

<!-- runnable -->
```python
latent_correction, corrected = model.batch_correct(
    expr, gene_id_type=gene_id_type, batch_size=2,
)
me = model.predict_me(expr, gene_id_type=gene_id_type, batch_size=2)
risk = model.predict_survival_risk(
    expr, clinical, gene_id_type=gene_id_type, batch_size=2,
)

for name, table in {
    "batch_correction_latent": latent_correction,
    "corrected_expression": corrected,
    "me_predictions": me,
    "survival_risk": risk,
}.items():
    assert np.isfinite(table.to_numpy()).all()
    table.to_csv(out / (name + ".tsv"), sep="\t", index_label="sample_id")
    print(name, table.shape)
```

Outputs:

| Output | Dimensions | Interpretation |
| --- | --- | --- |
| `latent_correction` | Samples x 100 | Latent means from the batch-correction generator |
| `corrected` | Samples x model genes | Reconstructed, corrected expression, not raw measurements |
| `me` | Samples x 8 | Predicted functional-axis values, not attribution contributions |
| `risk` | Samples x 1 | `risk_score` from the ME + clinical model |

The risk score is not a probability between 0 and 1, nor a five-year mortality probability. With the same model and preprocessing, higher scores indicate higher predicted risk. Clinical validity requires evaluation on independent data.

Each interface above receives the original TPM input separately. Do not substitute `corrected` for the original TPM in the prediction interfaces unless that workflow has been separately validated; doing so could introduce repeated processing.

## 4. Attribute a Single Sample: Gene to Functional Axis to Risk

Hierarchical attribution requires an explicit reference TPM set. Define the reference population before examining results and keep the reference strategy consistent when comparing samples or cohorts. Below, one other synthetic sample serves as the reference for a software demonstration; this is not a recommended biological baseline.

<!-- runnable -->
```python
sample_id = expr.index[0]
reference_tpm = expr.iloc[[1]]
attribution = model.attribute_axis_risk(
    expr_tpm=expr.loc[[sample_id]],
    clinical=clinical.loc[[sample_id]],
    me_name="MEred",
    reference_tpm=reference_tpm,
    steps=64,
    internal_batch_size=8,
    gene_id_type=gene_id_type,
)

attribution.axis_shapley.to_csv(out / "axis_shapley.tsv", sep="\t")
attribution.gene_to_axis_ig.to_csv(out / "MEred_gene_ig.tsv", sep="\t")
attribution.gene_risk_contribution.to_csv(
    out / "MEred_gene_risk_contribution.tsv", sep="\t",
)
attribution.diagnostics.to_csv(out / "attribution_diagnostics.tsv", sep="\t")
print(attribution.diagnostics)

valid = attribution.diagnostics["allocation_valid"].astype(bool).all()
if not valid:
    raise RuntimeError(
        "Attribution allocation failed QC. Inspect diagnostics; "
        "do not interpret the gene risk contribution table."
    )
scores = attribution.gene_risk_contribution.loc[sample_id]
top_genes = scores.loc[scores.abs().sort_values(ascending=False).head(10).index]
top_genes.to_csv(out / "MEred_top10_gene_contributions.tsv", sep="\t")
print(top_genes)
```

The four main outputs are:

- `axis_shapley`: Risk contributions of the eight functional axes relative to the reference baseline, with clinical inputs held fixed.
- `gene_to_axis_ig`: Integrated gradients from genes to the selected functional axis. These are not expression values and are not restricted to 0-1.
- `gene_risk_contribution`: An allocation of the selected axis's risk contribution to genes, not an exact gene-level Shapley calculation over all genes.
- `diagnostics`: Shapley completeness, IG numerical error, and allocation validity.

Positive and negative risk contributions indicate an increase or decrease in predicted risk relative to the reference baseline. They do not establish pathway activation or inhibition, or causal effects. A small contribution does not mean a gene is unexpressed. If QC fails, inspect the baseline and consider increasing the integration steps. Near-zero allocation denominators cannot necessarily be fixed by adding steps; do not replace invalid values with zero and interpret them.

For gene contributions to MEred alone or to overall predicted risk, use `attribute_me()` or `attribute_survival_risk()`, respectively. Their attribution targets and baseline definitions differ from the hierarchical allocation above; the outputs should not be treated as interchangeable.

## 5. Search for Similar Samples

### Core Walkthrough: Build a Four-Sample Reference Atlas

In practice, you can load a supplied reference atlas directly. Here, build a small atlas to check saving, loading, and querying. Matches among these synthetic samples have no biological interpretation.

<!-- runnable -->
```python
import tempfile

atlas = model.build_reference(expr, metadata, gene_id_type=gene_id_type)
with tempfile.TemporaryDirectory(prefix="cancerxpress_atlas_", dir=str(out)) as tmp:
    atlas_dir = Path(tmp) / "reference_atlas"
    atlas.save(atlas_dir)
    loaded_atlas = cx.ReferenceAtlas.load(atlas_dir)
    for space in ("latent", "me"):
        hits = model.query(
            expr.iloc[[0]], loaded_atlas,
            metadata=metadata.iloc[[0]],
            space=space, k=2, same_cancer=True,
            exclude_self=True, gene_id_type=gene_id_type,
        )
        assert len(hits) == 2
        assert sample_id not in hits["reference_sample"].tolist()
        hits.to_csv(out / ("query_" + space + ".tsv"), sep="\t", index=False)
        print(space, hits)
```

By default, features are standardized using reference-atlas statistics before cosine similarity is calculated. Results contain reference sample IDs, cancer-type labels, ranks, distances, and similarities. Set `same_cancer=False` to allow cross-cancer searches; `True` requires valid cancer-type labels. Similarity is not a diagnosis and does not imply identical prognosis.

**Keep the latent representations distinct:** `build_reference()`, `query()`, and `encode_features()` use the matching ME model's encoder. Do not mix their latent features with those returned by `batch_correct()`, even though both have 100 dimensions. The query model and reference atlas must have matching model/preprocessing signatures.

### Optional: Load a Real Atlas and Retrieve Selected Samples

The following code requires separately supplied local reference data. Installing the Python package does not download these files.

```python
atlas = cx.ReferenceAtlas.load("/your/data/reference_atlas")
hits = model.query(
    expr, atlas, metadata=metadata, space="me",
    k=10, same_cancer=False, gene_id_type=gene_id_type,
)

# Atlas queries require atlas.json, samples.tsv, and features.npz.
# Expression retrieval requires reference_index.json.gz and all indexed ZIP shards.
db = cx.SampleDatabase.from_directory("/your/data/indexed_reference")
ids = hits["reference_sample"].drop_duplicates().head(2).tolist()
data = db.fetch(ids, components=[
    "raw_tpm", "corrected_expression", "latent", "me", "metadata",
])
```

These interfaces read local data and do not upload inputs in the background. Do not extract or rename the ZIP shards. Raw TPM and corrected expression may have different gene columns; align them by gene ID, not column position.

## 6. Optional: Stratified Latent Domain Adaptation

This section is not intended for the four synthetic samples above. It requires a target cohort, a reference atlas generated by the same model, and compatible biological labels in both cohorts.

- `cancer_type` is the base stratification label. Target cancer types must exist in the reference atlas; a single-cancer target cohort is supported.
- Additional genuine labels, such as `subtype`, can define finer strata. Do not invent labels to encourage alignment.
- Each stratum needs at least five independent patients in each cohort. This is a software minimum, not evidence of adequate statistical power.
- Supply `patient_id` to keep repeated measurements from the same patient within one split. Otherwise, sample IDs define the splitting units.
- Survival outcome labels are not required. Do not select adaptation settings based on test-set survival performance.

```python
target_expr = pd.read_csv("/your/data/target_tpm.tsv", sep="\t", index_col=0)
target_meta = pd.read_csv("/your/data/target_metadata.tsv", sep="\t", index_col=0)
atlas = cx.ReferenceAtlas.load("/your/data/reference_atlas")

fit = model.adapt(
    target_expr, target_meta, atlas,
    biological_labels=None,  # Cancer only; use ["subtype"] if both cohorts have it.
    gene_id_type="ensembl",
    epochs=200, patience=25, seed=42,
)
fit.save("/your/work/target_adapter")  # The destination must not already exist.
print(fit.metrics)

adapter = cx.LatentAdapter.load_adapter("/your/work/target_adapter")
z_aligned, me_aligned = model.encode_features(
    target_expr, metadata=target_meta,
    adapter=adapter, gene_id_type="ensembl",
)
hits_aligned = model.query(
    target_expr, atlas, metadata=target_meta,
    adapter=adapter, same_cancer=True, gene_id_type="ensembl",
)
```

The adapter changes only target-sample latent representations; the base encoder and ME head remain frozen. Training combines stratified distribution alignment with constraints that preserve the original structure. `fit.history`, `fit.splits`, and `fit.metrics` contain the training history, patient splits, and diagnostics, respectively, and are included when saving the result.

**Current scope:** An adapter takes effect only when explicitly passed to `encode_features()` or `query()`. It does not automatically modify `predict_survival_risk()` or the attribution interfaces. Do not describe unadapted risk predictions as adapted results. This interface also does not provide pretraining from scratch.

## 7. Check the Outputs

The core walkthrough in Sections 1-5 saves prediction, attribution, and query TSV files to your output directory. Check that:

1. Predictions are finite, ME output has eight columns, and sample IDs remain consistent.
2. Attribution has `allocation_valid=True` and acceptable numerical errors.
3. Queries exclude the query's own sample ID from the returned neighbors.
4. For real data, separately assess gene coverage, clinical labels, cross-cohort shifts, and performance against independent outcomes.

The first three checks establish software and numerical behavior, not biological or clinical validity. A small workflow test does not establish that the model is valid for a new cancer type.

### Validation Scope

The walkthrough was validated on CPU using installed `1.1.0.dev0` wheels on Python 3.12 and 3.13, matching model parameters, and synthetic inputs. It covered all three attribution interfaces, adapter training/save/reload, and queries and retrieval against the local 51,052-sample reference database. Unit tests and fixed-input model comparisons additionally cover Python 3.8-3.13; see [Installation and Compatibility](installation.md) for the exact scope.

The original PyPI `cancerxpress==1.0.0` / Python 3.8 environment provides the numerical reference. Its `requirements-release-lock.txt` is historical and must not be applied to the new runtime.

These checks validate installation and interface behavior, not the biology of synthetic samples. They do not validate GPU execution or predictive performance in new cohorts.
