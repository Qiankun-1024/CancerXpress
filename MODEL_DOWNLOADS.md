# Model Downloads

This repository does not include large model checkpoints in GitHub.

The following resources should be hosted externally, for example via:

- GitHub Releases
- Hugging Face Hub
- Zenodo
- institutional object storage

## Required Checkpoint Groups

- `batch_correction`
- `me`
- `primary_site`
- `cancer_type`
- `survival_risk`

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

## External Links

Replace the placeholders below with your real public download URLs.

- `batch_correction`: `TO_BE_FILLED`
- `me`: `TO_BE_FILLED`
- `primary_site`: `TO_BE_FILLED`
- `cancer_type`: `TO_BE_FILLED`
- `survival_risk`: `TO_BE_FILLED`

## Notes

- Keep `model_manifest.json` in the repository.
- Do not commit TensorFlow checkpoint shards, optimizer states, or large exported outputs to GitHub.
- If you later publish to PyPI, large checkpoints should also stay outside the wheel and source distribution.
