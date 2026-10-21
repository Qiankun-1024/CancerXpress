# CancerXpress

CancerXpress provides transcriptomic batch correction, functional-axis and
survival-risk prediction, model attribution, reference-sample search, and
target-cohort latent adaptation.

Start with [Installation and Compatibility](installation.md), then follow the
[Quickstart](01_quickstart.md). The [API Guide](api.md) explains which interface
to use and how to interpret its outputs.

## Inputs and Outputs

The main input is a samples-by-genes TPM matrix. Risk prediction also requires
sample-matched clinical covariates. Models and reference data are supplied
separately from the Python package.

- Predict eight functional-axis scores and a model-based risk score.
- Attribute a functional axis or predicted risk to input genes.
- Search a local reference atlas in latent or functional-axis space.
- Adapt a target cohort using biological strata shared with a reference atlas.

This release does not predict cancer type or primary site and does not include
plotting functions. Results are intended for research, not clinical diagnosis.

## Version Notice

These pages describe the **1.1 development series**. The published PyPI release
`1.0.0` remains Python-3.8-only; editing this repository does not change that
published wheel. See the installation page for the source-install workflow.

The website displays documentation and examples. It does not run models on
uploaded patient data or provide an online prediction service.
