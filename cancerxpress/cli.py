from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import CancerXpress
from .validation import validate_expression, align_clinical


def main() -> None:
    parser = argparse.ArgumentParser(description='CancerXpress inference CLI')
    parser.add_argument('--input', required=True, help='Input expression TSV, rows=samples, cols=genes')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--task', default='all', choices=['batch_correct', 'functional_axes', 'risk', 'attribution', 'all'])
    parser.add_argument('--gene-id-type', default='auto', choices=['auto', 'ensembl', 'symbol'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--resource-dir', default=None, help='Extracted resource bundle root')
    parser.add_argument('--model-dir', default=None, help='Override checkpoint models directory')
    parser.add_argument('--clinical-file', default=None, help='TSV with age, sex, race, tumor_stage, and cancer_type columns')
    parser.add_argument('--sample-id', default=None, help='Optional sample ID for single-sample attribution')
    parser.add_argument('--attribution-task', default='functional_axis', choices=['functional_axis', 'survival_risk'])
    parser.add_argument('--target-name', default=None, help='functional-axis name for functional-axis attribution')
    parser.add_argument('--steps', type=int, default=50, help='Integrated gradients steps for attribution')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(args.input, sep='\t', index_col=0)
    validate_expression(expr)
    model = CancerXpress(resource_dir=args.resource_dir, model_dir=args.model_dir)

    clinical = None
    if args.clinical_file:
        clinical = pd.read_csv(args.clinical_file, sep='\t', index_col=0)
        clinical = align_clinical(clinical, expr.index)

    if args.sample_id is not None:
        expr = expr.loc[[args.sample_id]]
        if clinical is not None:
            clinical = clinical.loc[expr.index]

    if args.task in ['batch_correct', 'all']:
        latent, corrected = model.batch_correct(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type)
        latent.to_csv(outdir / 'latent.tsv', sep='\t')
        corrected.to_csv(outdir / 'corrected_expression.tsv', sep='\t')

    if args.task in ['functional_axes', 'all']:
        model.predict_functional_axes(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'functional_axis_predictions.tsv', sep='\t')

    if args.task in ['risk', 'all']:
        if clinical is None:
            raise ValueError('Risk prediction requires --clinical-file for the run1 survival model.')
        model.predict_survival_risk(expr, clinical=clinical, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'survival_risk.tsv', sep='\t')

    if args.task == 'attribution':
        if len(expr) != 1:
            raise ValueError('Attribution requires a single sample. Use --sample-id.')
        if args.attribution_task == 'functional_axis':
            result = model.attribute_functional_axis(expr, axis_name=args.target_name or 'immune_inflammation', steps=args.steps, gene_id_type=args.gene_id_type)
        else:
            if clinical is None:
                raise ValueError('Survival risk attribution requires --clinical-file.')
            result = model.attribute_survival_risk(expr, clinical=clinical, steps=args.steps, gene_id_type=args.gene_id_type)
        result.attributions.to_csv(outdir / 'attribution.tsv', sep='\t')
        result.predicted_output.to_csv(outdir / 'attribution_prediction.tsv', sep='\t')


if __name__ == '__main__':
    main()
