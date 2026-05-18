# Resources Policy

This file defines which files under `resources/` should stay in the GitHub repository and which files should be distributed externally.

## Keep in GitHub

These files are typically small, stable, and required for inference or preprocessing:

- `resources/assets/tsne_sorted_pixels_coords.csv`
- `resources/assets/max_norm_tpm.tsv`
- `resources/encoders/`
- `resources/gene_annotation/`
- `resources/mappings/`
- `resources/models/model_manifest.json`

## Do Not Commit to GitHub

These files are large and should be hosted externally:

- `resources/models/pretrained/`
- `resources/models/finetuned/`

This includes:

- TensorFlow checkpoint shards
- optimizer state files
- large exported weights
- temporary fine-tuning checkpoints

## Recommended External Hosting

Use one of the following:

- GitHub Releases
- Hugging Face Hub
- Zenodo
- institutional object storage

## Recommended Repository State

The repository should contain:

```text
resources/
  assets/
  encoders/
  gene_annotation/
  mappings/
  models/
    model_manifest.json
```

Large checkpoint directories should be downloaded after cloning the repository.

## After Cloning

Users should:

1. install the package environment
2. download model checkpoints from public links
3. place them under `resources/models/`
