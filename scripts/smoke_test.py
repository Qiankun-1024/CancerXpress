from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import cancerxpress as cx


ROOT = Path(__file__).resolve().parents[1]
EXPR_FILE = ROOT / "examples" / "scanb_demo_3samples.tsv"
CANCER_TYPE_FILE = ROOT / "examples" / "scanb_demo_3samples_cancer_type.tsv"
OUTDIR = ROOT / "output" / "smoke_test"


def build_symbol_expression(expr: pd.DataFrame) -> pd.DataFrame:
    converter = cx.GeneIDConverter()
    keep_cols = []
    new_cols = []
    for gene in expr.columns:
        gene_clean = str(gene).split(".")[0]
        if gene_clean in converter.ensembl_to_symbol:
            keep_cols.append(gene)
            new_cols.append(converter.ensembl_to_symbol[gene_clean])
    symbol_expr = expr[keep_cols].copy()
    symbol_expr.columns = new_cols
    symbol_expr = symbol_expr.groupby(symbol_expr.columns, axis=1).sum()
    return symbol_expr


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(EXPR_FILE, sep="\t", index_col=0).iloc[:1]
    cancer_type = pd.read_csv(CANCER_TYPE_FILE, sep="\t", index_col=0).loc[expr.index, "cancer_type"]
    expr_symbol = build_symbol_expression(expr)

    model = cx.CancerXpress()

    latent, corrected = model.batch_correct(expr, batch_size=1, gene_id_type="ensembl")
    me = model.predict_me(expr, batch_size=1, gene_id_type="ensembl")
    primary = model.predict_primary_site(expr, batch_size=1, gene_id_type="ensembl")
    cancer = model.predict_cancer_type(expr, batch_size=1, gene_id_type="ensembl")
    risk = model.predict_survival_risk(expr, cancer_type=cancer_type, batch_size=1, gene_id_type="ensembl")

    latent_symbol, corrected_symbol = model.batch_correct(expr_symbol, batch_size=1, gene_id_type="symbol")
    me_symbol = model.predict_me(expr_symbol, batch_size=1, gene_id_type="symbol")

    me_attr = model.attribute_me(expr, me_name="MEblue", steps=8, gene_id_type="ensembl")
    primary_attr = model.attribute_primary_site(expr, steps=8, gene_id_type="ensembl")
    cancer_attr = model.attribute_cancer_type(expr, steps=8, gene_id_type="ensembl")
    risk_attr = model.attribute_survival_risk(expr, cancer_type=cancer_type, steps=8, gene_id_type="ensembl")

    summary = {
        "latent_shape": list(latent.shape),
        "corrected_shape": list(corrected.shape),
        "me_shape": list(me.shape),
        "primary_site_columns": primary.columns.tolist(),
        "cancer_type_columns": cancer.columns.tolist(),
        "risk_shape": list(risk.shape),
        "symbol_latent_shape": list(latent_symbol.shape),
        "symbol_corrected_shape": list(corrected_symbol.shape),
        "symbol_me_shape": list(me_symbol.shape),
        "me_attr_shape": list(me_attr.attributions.shape),
        "primary_attr_shape": list(primary_attr.attributions.shape),
        "cancer_attr_shape": list(cancer_attr.attributions.shape),
        "risk_attr_shape": list(risk_attr.attributions.shape),
        "me_prediction": me.iloc[0].to_dict(),
        "primary_site_prediction": primary.iloc[0].to_dict(),
        "cancer_type_prediction": cancer.iloc[0].to_dict(),
        "risk_prediction": risk.iloc[0].to_dict(),
        "primary_attr_target": primary_attr.target_name,
        "cancer_attr_target": cancer_attr.target_name,
    }

    with open(OUTDIR / "smoke_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    me_attr.attributions.to_csv(OUTDIR / "me_attribution.tsv", sep="\t")
    primary_attr.attributions.to_csv(OUTDIR / "primary_site_attribution.tsv", sep="\t")
    cancer_attr.attributions.to_csv(OUTDIR / "cancer_type_attribution.tsv", sep="\t")
    risk_attr.attributions.to_csv(OUTDIR / "survival_risk_attribution.tsv", sep="\t")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
