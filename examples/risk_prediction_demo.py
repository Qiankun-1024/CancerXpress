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
    outdir = PROJECT_ROOT / "output" / "risk_demo"
    outdir.mkdir(parents=True, exist_ok=True)

    expr = pd.read_csv(expr_file, sep="\t", index_col=0)
    cancer_type = pd.read_csv(cancer_type_file, sep="\t", index_col=0)["cancer_type"]

    model = cx.CancerXpress()
    risk = model.predict_survival_risk(
        expr_tpm=expr,
        cancer_type=cancer_type,
        gene_id_type="ensembl",
    )
    risk.to_csv(outdir / "survival_risk.tsv", sep="\t")

    print("Saved survival risk predictions to:", outdir / "survival_risk.tsv")
    print(risk)


if __name__ == "__main__":
    main()
