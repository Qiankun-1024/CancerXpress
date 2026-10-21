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

## Installed Package

Neither the wheel nor source distribution includes `resources/`. The complete
versioned resource bundle is published separately; the small source copies
remain for maintainers, not as an installation-path fallback.

Users download the resource bundle into `~/.cache/cancerxpress/resources`,
set `CANCERXPRESS_HOME`, or pass `CancerXpress(resource_dir=...)`. An existing
checkout can still be used with `resource_dir="resources"`.

The release exporter uses an explicit resource allowlist and the five active
inference models. It excludes patient expression, clinical records, optimizer
state, stale checkpoints and analysis outputs.
