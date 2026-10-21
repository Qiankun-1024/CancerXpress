# Model Resources

Weights and required preprocessing assets are distributed separately from the
Python wheel and source distribution. The official Zenodo URL/DOI is pending
publication; no placeholder URL is used for automatic downloads.

## Download and use

```python
import cancerxpress as cx
cx.download_models(url=zenodo_file_url, sha256=release_sha256)
model = cx.CancerXpress()
```

The default destination is `~/.cache/cancerxpress/resources`, or
`CANCERXPRESS_HOME` when set. A custom download destination can be passed as
`resource_dir`; pass the same directory to `CancerXpress(resource_dir=...)`.
The archive contains the following paths directly (no extra top-level folder):

```text
resource_manifest.json
assets/
encoders/
gene_annotation/
mappings/survival_risk_preprocessing.json
models/pretrained/batch_correction/
models/finetuned/functional_axes/
models/finetuned/primary_site/
models/finetuned/cancer_type/
models/finetuned/survival_risk/
```

The survival model consumes predicted functional-axis scores and clinical covariates:
age, sex, race, tumor stage and cancer type. Model order and preprocessing
parameters are frozen in the bundle.

## Maintainer release

Run `scripts/prepare_zenodo_release.py --output /path/to/new-release` in the
supported TensorFlow environment. It explicitly selects inference resources,
re-saves the five active checkpoint groups without optimizer state, rewrites
checkpoint paths to relative paths, computes per-file hashes, and creates:

- `CancerXpress-models-v1.0.0.tar.gz`
- `resource_manifest.json`
- `SHA256SUMS.txt`

Upload these three files to one Zenodo record. Review licensing for both model
weights and third-party reference assets before publishing. Do not upload
source training directories, datasets, example patient expression, or logs.
After publication, add the permanent file URL and SHA-256 to this document.
Test a real remote download before announcing public availability.

`verify_resources` checks integrity, not biological validity or publisher
identity. Obtain the archive checksum through the trusted release record.
