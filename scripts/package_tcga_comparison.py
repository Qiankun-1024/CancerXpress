"""Build a separate, traceable TCGA attribution handoff; never package raw TPM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='Resume this script-created staging directory')
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=args.resume)
    root = args.research_root
    base = root / 'prognostic_mechanism'
    data = base / 'data/contribution_cluster_gene_risk'
    provenance = []
    def copy(src, relative):
        dst = out / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        provenance.append({'file': relative, 'source': str(src.relative_to(root))})
    sample = pd.read_csv(data / 'sample_order.tsv', sep='\t').sample_id
    genes = pd.read_csv(data / 'gene_order.tsv', sep='\t').ensembl_id
    meta = pd.read_csv(data / 'all_cluster_target_samples.tsv', sep='\t').set_index('sample_id').loc[sample]
    selected = meta[['fold','cancer_type','cluster','dominant_axis','direction','dominant_axis_shapley','risk_score','risk_fold_cancer_z']].copy()
    selected['patient_id'] = selected.index.str[:12]
    (out / 'attribution').mkdir(exist_ok=args.resume)
    selected.to_csv(out / 'attribution/samples.tsv', sep='\t')
    for filename in ['dominant_axis_gene_logrisk_contribution.npy','gene_order.tsv',
                     'sample_order.tsv','all_cluster_attribution_qc.tsv.gz','cluster_definitions.tsv']:
        copy(data / filename, 'attribution/' + filename)
    axis_path = root / 'output/run1_survival_main_figure_curated/sample_axis_contribution_clustering/sample_axis_contribution_clusters.tsv.gz'
    axes = pd.read_csv(axis_path, sep='\t')
    axes = axes.drop(columns=['OS_time_months','OS_Status'], errors='ignore')
    axes.to_csv(out / 'attribution/all_axis_values_and_shapley.tsv.gz', sep='\t', index=False)
    provenance.append({'file':'attribution/all_axis_values_and_shapley.tsv.gz',
                       'source':str(axis_path.relative_to(root)), 'processing':'Removed OS fields'})
    for folder in sorted((base / 'results/contribution_cluster_nmf').glob('C[0-9][0-9]')):
        if not (folder / 'summary.json').exists():
            continue
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix in ('.json','.tsv','.gz','.npy'):
                copy(file, 'programs/' + folder.name + '/' + file.name)
    copy(base / 'results/contribution_cluster_nmf/all_cluster_nmf_summary.tsv','programs/all_cluster_nmf_summary.tsv')
    for name in ['Figure_5A_cross_cancer_functional_axis_contribution_clusters.pdf',
                 'Figure_5B_cluster_gene_attribution_NMF.pdf',
                 'Figure_5C_program_directional_ranked_GSEA.pdf',
                 'Supplementary_Figure_5_all_cluster_NMF_profiles.pdf',
                 'Supplementary_immune_program_leading_edge_comparison.pdf']:
        copy(base / 'results/figure5' / name, 'figures/' + name)
    for file in sorted((base / 'results/figure5').glob('Supplementary_Table_*.tsv')):
        copy(file, 'tables/' + file.name)
    for name in ['Supplementary_enrichment_README.md','Supplementary_C11_stability_methods.txt',
                 'Supplementary_immune_overlap_methods.txt']:
        copy(base / 'results/figure5' / name, 'tables/' + name)
    # Include the implementation needed to understand historical normalization.
    for name in ['15_compute_all_cluster_gene_risk_attributions.py','16_cluster_consensus_nmf.py',
                 '17_plot_cluster_nmf_profiles.py','64_plot_representative_figure5c.py']:
        copy(base / 'scripts' / name, 'research_scripts/' + name)
    meta_info = json.loads((data / 'run_metadata.json').read_text())
    for key in ('cluster_file','expression_source'):
        meta_info.pop(key, None)
    (out / 'attribution/methods.json').write_text(json.dumps(meta_info, indent=2))
    (out / 'README.md').write_text("""# CancerXpress TCGA comparison data v1.0.0

This is a research-result handoff, separate from the installable Python package
and inference-weight bundle. It is not a clinical diagnostic reference.

## Contents and row alignment
- attribution/samples.tsv and sample_order.tsv define matrix row order.
- attribution/gene_order.tsv defines matrix column order (Ensembl IDs).
- dominant_axis_gene_logrisk_contribution.npy is float32, patients x genes.
  It contains ONLY each patient's CLUSTER-DOMINANT axis gene-risk allocation,
  NOT all 8 gene-to-axis channels and NOT expression.
- all_axis_values_and_shapley.tsv.gz contains all 8 ME values and axis Shapley
  contributions, risk scores, cluster and fold. Do not treat signed shares
  as ME expression values.
- all_cluster_attribution_qc.tsv.gz has explicit validity and IG completeness.
  Failed rows remain NaN; never replace them with biological zero.
- programs/Cxx contains the original pan-cancer program loadings, usage,
  selected genes, stability, rank diagnostics and summary. P1 in one cluster
  is not P1 in another. These are not the cancer-refitted Figure5E programs.
- figures/ contains Figures5A/B/C and associated TCGA supplements.
- tables/ retains enrichment sources, NES, FDR, leading edges and stability.
- reference_atlas/ contains NEW frozen-release-model latent/ME features for
  nearest-neighbour queries. Its feature fingerprint is enforced by the API.

## Historical attribution versus query features
Published risk attributions use the OOF fold-specific risk model and
fold/cancer training-mean reference, with image-baseline optimization.
Gene risk = axis Shapley * gene IG / sum(gene IG); these are allocations,
not exact gene Shapley, probabilities or causal effects.
NMF first filters finite/nonzero rows, normalizes each patient's full gene
vector by its L1 mass, selects 1000 genes per cluster, splits positive and
negative channels and scales the matrix by 1000 for fitting. Inspect the
included historical scripts for exact H/W scaling and ordering.
The public TPM reference API uses a different baseline unless explicitly
reproducing the historical procedure. Do not compare absolute magnitudes
without matching model, preprocessing, reference and channel definitions.

The query atlas is recomputed from the frozen released model and is not an
OOF validation product. Its ME coordinates can differ from historical columns.
Use query results as phenotypic neighbours, not as proof of shared prognosis.
Latent coordinates and ME values are different from risk contributions.
Use the same weights/assets and API fingerprint when comparing new samples.
Adaptation training/validation patients are not untouched external validation.

## Privacy and sharing
Raw TPM, survival endpoints and clinical records are excluded. Public TCGA
sample IDs, cancer labels and derived predictions remain necessary for linkage.
Before any public redistribution, verify the source-data terms and your
institution's requirements; this preparation is not a legal clearance.
Included research scripts document provenance and may depend on local project
inputs. They are not standalone re-execution pipelines.

## Loading
Use np.load(..., mmap_mode='r') for the gene matrix. Join tables by sample_id;
do not assume other TSV tables have the same row order. Load reference_atlas
with cancerxpress.ReferenceAtlas.load(path). See package README for query/adapt.
Manifest checksums cover every delivered file except the manifest itself.
""")
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    import cancerxpress as cx
    model = cx.CancerXpress(resource_dir=args.resources)
    expression = np.load(data / 'tcga_oof_raw_tpm.npy', mmap_mode='r')
    z_all, me_all = [], []
    for start in range(0, len(sample), 64):
        batch = pd.DataFrame(np.asarray(expression[start:start+64]), index=sample.iloc[start:start+64], columns=genes)
        z, me = model.encode_features(batch, batch_size=16, gene_id_type='ensembl')
        z_all.append(z);me_all.append(me)
        if start % 640 == 0:
            print('Reference features',start,'/',len(sample),flush=True)
    atlas = cx.ReferenceAtlas(pd.concat(z_all), pd.concat(me_all),
                              selected, model._feature_signature())
    atlas.save(out / 'reference_atlas')
    from cancerxpress.downloads import sha256_file
    listing = [{'path':str(f.relative_to(out)), 'size':f.stat().st_size, 'sha256':sha256_file(f)}
               for f in sorted(out.rglob('*')) if f.is_file()]
    (out/'manifest.json').write_text(json.dumps({'version':'1.0.0','n_samples':len(sample),
                                               'n_genes':len(genes),'files':listing},indent=2))
    archive = Path(str(out) + '.tar.gz')
    with tarfile.open(str(archive),'w:gz',compresslevel=1) as tar:
        tar.add(str(out),arcname=out.name)
    archive.with_suffix('.sha256').write_text(sha256_file(archive)+'  '+archive.name+'\n')
    from organize_tcga_comparison import organize
    organize(out, root)
    print('READY',archive,flush=True)


if __name__ == '__main__':
    main()
