#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cancerxpress as cx


def main() -> None:
    expr_file = PROJECT_ROOT / "examples" / "scanb_demo_3samples.tsv"
    cancer_type_file = PROJECT_ROOT / "examples" / "scanb_demo_3samples_cancer_type.tsv"
    outdir = PROJECT_ROOT / "output" / "risk_attribution_demo"
    outdir.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(expr_file, sep="\t", index_col=0)
    cancer_type = pd.read_csv(cancer_type_file, sep="\t", index_col=0)["cancer_type"]

    sample_id = expr.index[0]
    sample_expr = expr.loc[[sample_id]]
    sample_cancer_type = cancer_type.loc[[sample_id]]

    model = cx.CancerXpress()
    result = model.attribute_survival_risk(
        expr_tpm=sample_expr,
        cancer_type=sample_cancer_type,
        gene_id_type="ensembl",
    )

    result.predicted_output.to_csv(outdir / "survival_risk_prediction.tsv", sep="\t")
    result.attributions.to_csv(outdir / "survival_risk_attribution.tsv", sep="\t")

    print("Sample ID:", sample_id)
    print("Saved prediction to:", outdir / "survival_risk_prediction.tsv")
    print("Saved attribution to:", outdir / "survival_risk_attribution.tsv")
    print(result.predicted_output)


if __name__ == "__main__":
    main()
