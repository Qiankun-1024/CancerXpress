"""Export fixed-input outputs and compare checkpoint behavior across runtimes."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import cancerxpress as cx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    assert "site-packages" in cx.__file__, cx.__file__
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    model = cx.CancerXpress(resource_dir=args.resources)
    rng = np.random.RandomState(123)
    genes = model.preprocessor.training_gene_order
    expr = pd.DataFrame(rng.gamma(2, 5, size=(2, len(genes))),
                        columns=genes, index=["synthetic1", "synthetic2"])
    clinical = pd.DataFrame({
        "age": [60, 61], "sex": ["female"] * 2, "race": ["white"] * 2,
        "tumor_stage": ["stage ii"] * 2, "cancer_type": ["BRCA"] * 2,
    }, index=expr.index)
    latent, corrected = model.batch_correct(expr, gene_id_type="ensembl")
    query_latent, query_axes = model.encode_features(expr, gene_id_type="ensembl")
    attribution = model.attribute_axis_risk(
        expr.iloc[:1], clinical.iloc[:1], "developmental_regulation", expr.iloc[1:],
        steps=64, internal_batch_size=8, gene_id_type="ensembl")
    tables = {
        "latent": latent, "corrected": corrected, "query_latent": query_latent,
        "query_axes": query_axes, "functional_axes": model.predict_functional_axes(expr, gene_id_type="ensembl"),
        "risk": model.predict_survival_risk(expr, clinical, gene_id_type="ensembl"),
        "axis_shapley": attribution.axis_shapley,
        "gene_ig": attribution.gene_to_axis_ig,
        "gene_risk": attribution.gene_risk_contribution,
    }
    arrays = {name: value.to_numpy() for name, value in tables.items()}
    for value in arrays.values():
        assert np.isfinite(value).all()
    assert attribution.diagnostics.allocation_valid.all()
    np.savez_compressed(root / "outputs.npz", **arrays)
    report = {"package": cx.__file__, "cancerxpress": version("cancerxpress"),
              "tensorflow": tf.__version__, "numpy": np.__version__,
              "pandas": pd.__version__, "synthetic_inputs": True,
              "attribution_qc": attribution.diagnostics.to_dict("records")}
    if args.baseline:
        differences, relative_errors = {}, {}
        with np.load(args.baseline, allow_pickle=False) as baseline:
            for name, actual in arrays.items():
                # Historical regression fixtures retain the pre-migration keys.
                old_key = {'functional_axes': 'me', 'query_axes': 'query_me'}.get(name, name)
                expected = baseline[name] if name in baseline else baseline[old_key]
                # CPU kernel changes can alter the last few floating-point digits.
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5,
                                           err_msg=name)
                differences[name] = float(np.max(np.abs(actual - expected)))
                relative_errors[name] = float(
                    np.linalg.norm(actual - expected)
                    / max(float(np.linalg.norm(expected)), 1e-12))
        report["max_absolute_difference"] = differences
        report["relative_l2_error"] = relative_errors
        report["comparison"] = "passed"
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
