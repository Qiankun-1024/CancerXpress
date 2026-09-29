from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .pipeline import CancerXpress


def main() -> None:
    parser = argparse.ArgumentParser(description='CancerXpress inference CLI')
    parser.add_argument('--input', required=True, help='Input expression TSV, rows=samples, cols=genes')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--task', default='all', choices=['batch_correct', 'me', 'primary_site', 'cancer_type', 'risk', 'attribution', 'all'])
    parser.add_argument('--gene-id-type', default='auto', choices=['auto', 'ensembl', 'symbol'])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--clinical-file', default=None, help='TSV with age, sex, race, tumor_stage, and cancer_type columns')
    parser.add_argument('--sample-id', default=None, help='Optional sample ID for single-sample attribution')
    parser.add_argument('--attribution-task', default='me', choices=['me', 'primary_site', 'cancer_type', 'survival_risk'])
    parser.add_argument('--target-name', default=None, help='Optional target class name or ME name for attribution')
    parser.add_argument('--steps', type=int, default=50, help='Integrated gradients steps for attribution')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    expr = pd.read_csv(args.input, sep='\t', index_col=0)
    model = CancerXpress()

    clinical = None
    if args.clinical_file:
        clinical = pd.read_csv(args.clinical_file, sep='\t', index_col=0).reindex(expr.index)

    if args.sample_id is not None:
        expr = expr.loc[[args.sample_id]]
        if clinical is not None:
            clinical = clinical.loc[expr.index]

    if args.task in ['batch_correct', 'all']:
        latent, corrected = model.batch_correct(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type)
        latent.to_csv(outdir / 'latent.tsv', sep='\t')
        corrected.to_csv(outdir / 'corrected_expression.tsv', sep='\t')

    if args.task in ['me', 'all']:
        model.predict_me(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'me_predictions.tsv', sep='\t')

    if args.task in ['primary_site', 'all']:
        model.predict_primary_site(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'primary_site_predictions.tsv', sep='\t')

    if args.task in ['cancer_type', 'all']:
        model.predict_cancer_type(expr, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'cancer_type_predictions.tsv', sep='\t')

    if args.task in ['risk', 'all']:
        if clinical is None:
            raise ValueError('Risk prediction requires --clinical-file for the run1 survival model.')
        model.predict_survival_risk(expr, clinical=clinical, batch_size=args.batch_size, gene_id_type=args.gene_id_type).to_csv(outdir / 'survival_risk.tsv', sep='\t')

    if args.task == 'attribution':
        if len(expr) != 1:
            raise ValueError('Attribution requires a single sample. Use --sample-id.')
        if args.attribution_task == 'me':
            result = model.attribute_me(expr, me_name=args.target_name or 'MEblue', steps=args.steps, gene_id_type=args.gene_id_type)
        elif args.attribution_task == 'primary_site':
            result = model.attribute_primary_site(expr, target_name=args.target_name, steps=args.steps, gene_id_type=args.gene_id_type)
        elif args.attribution_task == 'cancer_type':
            result = model.attribute_cancer_type(expr, target_name=args.target_name, steps=args.steps, gene_id_type=args.gene_id_type)
        else:
            if clinical is None:
                raise ValueError('Survival risk attribution requires --clinical-file.')
            result = model.attribute_survival_risk(expr, clinical=clinical, steps=args.steps, gene_id_type=args.gene_id_type)
        result.attributions.to_csv(outdir / 'attribution.tsv', sep='\t')
        result.predicted_output.to_csv(outdir / 'attribution_prediction.tsv', sep='\t')


if __name__ == '__main__':
    main()
