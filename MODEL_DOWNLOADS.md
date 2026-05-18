# Model Downloads

This repository does not include large model checkpoints in GitHub.

The following resources should be hosted externally, for example via:

- GitHub Releases
- Hugging Face Hub
- Zenodo
- institutional object storage

## Required Checkpoint Groups

- `batch_correction`: batch correction, latent embedding extraction, and corrected expression reconstruction
- `module_eigenpathway`: Module Eigenpathway prediction
- `primary_site`: primary site classification
- `cancer_type`: cancer type classification
- `survival_risk`: survival risk prediction

## Suggested Layout After Download

Place downloaded files under:

```text
resources/models/
  model_manifest.json
  pretrained/
    batch_correction/
  finetuned/
    me/
    primary_site/
    cancer_type/
    survival_risk/
```

The public task name `module_eigenpathway` currently maps to the local checkpoint directory `resources/models/finetuned/me/`.

## External Links

Replace the placeholders below with your real public download URLs.

- `batch_correction`: `TO_BE_FILLED`
- `module_eigenpathway`: `TO_BE_FILLED`
- `primary_site`: `TO_BE_FILLED`
- `cancer_type`: `TO_BE_FILLED`
- `survival_risk`: `TO_BE_FILLED`

## Notes

- Keep `model_manifest.json` in the repository.
- Do not commit TensorFlow checkpoint shards, optimizer states, or large exported outputs to GitHub.
- If you later publish to PyPI, large checkpoints should also stay outside the wheel and source distribution.
