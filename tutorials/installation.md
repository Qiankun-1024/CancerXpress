# Installation and Compatibility

## Development Version

The 1.1 development series targets **Python 3.8 through 3.13**. Python 3.11 or
3.12 is recommended for new environments. The package selects runtime
dependencies using Python-version markers:

| Python | TensorFlow | Keras implementation |
| --- | --- | --- |
| 3.8 | 2.13.x | TensorFlow's bundled Keras 2 |
| 3.9-3.13 | 2.20.x | Explicit `tf-keras` 2.20.x |

Python 3.14 and future versions are not advertised as supported until the
required upstream wheels and model checks are available. Dependency resolution
alone is not a model-compatibility test.

```bash
conda create -n cancerxpress python=3.11 -y
conda activate cancerxpress
cd /path/to/CancerXpress
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: .
python -m pip check
```

Use a checkout containing the 1.1 changes. Once those changes are pushed to the
repository, installation from the corresponding Git commit is also possible.
Do not install `1.0.0` and expect it to include these changes.

The `environment.yml` file provides an alternative conda setup. GPU libraries
are not pinned to the old CUDA 11.0 stack. CPU is sufficient for the walkthrough;
GPU setup and validation must match the selected TensorFlow version and platform.
The binary-only option avoids silently compiling large dependencies on older
Linux systems. It does not prevent pip from building the local CancerXpress
package, which contains Python code only.

## Why a Compatibility Layer?

The released model bundle uses Keras 2 TensorFlow checkpoints. CancerXpress
imports `tf_keras` explicitly on newer TensorFlow versions rather than changing
`TF_USE_LEGACY_KERAS` globally. This avoids changing another library's Keras
selection in the same process. Do not mix Keras 3 layers with these model objects.

TensorFlow Addons is no longer required. The package supplies the channels-last
group-normalization operation needed by the released models, preserving its
groups, epsilon, gamma/beta variables, and checkpoint attributes.

The implementation follows the
[Keras legacy-package guidance](https://keras.io/keras_3/).
TensorFlow Addons' retirement is documented in its
[maintainer announcement](https://github.com/tensorflow/addons/issues/2807).

## Model Resources

Manually extract the matching model bundle and set:

```bash
export CANCERXPRESS_HOME=/your/data/cancerxpress/models_v1
```

Preserve `assets/`, `models/`, `encoders/`, `gene_annotation/`, `mappings/`, and
`resource_manifest.json`. Do not mix checkpoints and preprocessing assets from
different bundles. No model or patient data is installed with the wheel.

For reference queries, separately supply an atlas containing `atlas.json`,
`features.npz`, and `samples.tsv`. Indexed expression retrieval additionally
requires the reference index and all indexed ZIP shards.

## Reproducing the Original 1.0 Release

PyPI `cancerxpress==1.0.0` and `requirements-release-lock.txt` describe the
historical Python 3.8 / TensorFlow 2.4 environment. Keep it separate from the
1.1 environment. Do not apply that old lock file as constraints when installing
the new runtime.

## Validation

Local Linux CPU checks for `1.1.0.dev0` tested installed wheels outside the
source checkout, not just imports from source:

| Python | Unit tests | Released-model comparison |
| --- | --- | --- |
| 3.8 | 44 passed | Passed |
| 3.9 | 44 passed | Passed |
| 3.10 | 44 passed | Passed |
| 3.11 | 44 passed | Passed |
| 3.12 | 44 passed | Passed |
| 3.13 | 44 passed | Passed |

The model comparison uses two fixed synthetic TPM profiles and the original
1.0.0 / TensorFlow 2.4 environment as its reference. It requires element-wise
agreement at `rtol=2e-4, atol=2e-5`; this is numerical agreement, not bitwise
identity. The full quickstart, adapter training/save/reload, reference search,
and indexed retrieval were additionally checked on Python 3.12 and 3.13 using
the local 51,052-sample reference database. Only a small number of synthetic
queries and retrieved samples were used, not an exhaustive database scan.

GPU execution, other operating systems, Python 3.14, and pretraining scripts
have not been validated by these checks.

The GitHub Actions matrix is configured to build and test the installed wheel
on Python 3.8-3.13 using synthetic fixtures. This configuration is not a claim
that a remote CI run or Read the Docs deployment has completed.
Real-model validation additionally
requires the matching resource bundle:

```bash
# Run from outside the source checkout, using the environment with the wheel installed.
python /path/to/CancerXpress/scripts/check_model_compatibility.py \
  --resources /your/data/cancerxpress/models_v1 \
  --output /your/work/runtime_check \
  --baseline /your/work/original_runtime/outputs.npz
```

Generate the baseline with the same script in the original 1.0 environment,
omitting `--baseline`. The comparison covers ME, risk, corrected expression,
both latent representations, axis Shapley values, and gene attribution.
It checks numerical compatibility, not clinical predictive performance.

The compatibility matrix applies to the installed package and its public
inference, attribution, search, and adaptation interfaces. Historical research
scripts under `training/` are not shipped in the wheel and are not covered by
this runtime migration.
