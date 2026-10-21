# API Guide

Construct the main entry point using a local model bundle:

```python
import cancerxpress as cx

model = cx.CancerXpress(resource_dir="/your/data/cancerxpress/models_v1")
```

All expression inputs below are samples-by-genes TPM DataFrames with unique
sample IDs. Use `gene_id_type="ensembl"` or `"symbol"` as appropriate.

## Prediction and Batch Correction

| Method | Additional inputs | Output |
| --- | --- | --- |
| `harmonize_expression(expr)` | Gene ID type | Matrix aligned to model genes |
| `batch_correct(expr)` | Optional batch size | Latent means and corrected expression |
| `predict_me(expr)` | Optional batch size | Eight functional-axis scores |
| `predict_survival_risk(expr, clinical)` | Age, sex, race, stage, cancer type | Model risk score |
| `run_all(expr, clinical)` | Same clinical table | Result object containing all four output tables |

The survival risk score is not an absolute survival probability. Cancer type
is an observed input label, not a classification output.

## Attribution

| Method | Question |
| --- | --- |
| `attribute_me(expr, me_name)` | Which genes contribute to this functional-axis prediction? |
| `attribute_survival_risk(expr, clinical)` | Which genes contribute to overall predicted risk? |
| `attribute_axis_risk(expr, clinical, me_name, reference_tpm)` | How is one axis's risk contribution allocated to genes? |

These interfaces currently take one query sample at a time. Hierarchical
attribution requires reference TPM and returns `axis_shapley`,
`gene_to_axis_ig`, `gene_risk_contribution`, and `diagnostics`. Inspect QC before
interpreting the allocation. Attribution sign is not expression activation or
inhibition, and attribution does not establish causality.

## Reference Search

- `encode_features(expr)` returns ME-model encoder features and ME scores.
- `build_reference(expr, metadata)` constructs a `ReferenceAtlas`.
- `query(expr, reference, space="latent")` retrieves similar reference samples.
- Set `space="me"` to query functional-axis space instead.
- Set `same_cancer=True` only when valid matching cancer-type labels are supplied.

The ME-model encoder and the batch-correction generator have distinct latent
representations. Do not combine them just because their dimensions agree.

`ReferenceAtlas.save()` / `.load()` persist the small feature atlas.
`SampleDatabase.from_directory()` / `.fetch()` read selected samples from
the separately supplied indexed reference release.

## Target-Cohort Adaptation

`adapt(expr, metadata, reference, biological_labels=None)` fits a target-only
latent adapter within shared biological strata. Cancer type is always included;
additional biological labels are optional. Supply patient IDs to prevent
repeated measurements from leaking across splits.

The result provides `adapter`, `history`, `splits`, and `metrics`.
Use `result.save()` and `LatentAdapter.load_adapter()` to persist the adapter.

Pass the adapter explicitly to `encode_features()` or `query()`. It is not
automatically applied to risk prediction or attribution. The interface does
not provide training from scratch.

See the [Quickstart](01_quickstart.md) for executable examples and interpretation
limits.
