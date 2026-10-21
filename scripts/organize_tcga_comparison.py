"""Organize the exported handoff by analysis, without recomputing results."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

STEPS = {
 '01_risk_states': 'All-axis ME values, Shapley contributions and cluster definitions; Figure 5A. Values and risk contributions are different quantities.',
 '02_gene_attribution': 'Dominant-axis gene-risk allocations and QC. Rows follow sample_order.tsv, columns follow gene_order.tsv. This is NOT a full eight-axis gene tensor; invalid rows remain NaN.',
 '03_nmf_programs': 'Original pan-cancer cluster-specific NMF usage, loadings and rank diagnostics; Figure 5B. Program numbering is local to each cluster. Inputs come from step 02.',
 '04_program_enrichment': 'Complete GSEA results, representative Figure 5C evidence and redundancy audit. Read the included methods for database-wise BH correction and evidence tiers. Inputs are step 03 loadings.',
 '05_immune_overlap': 'Immune-program leading-edge overlap and shared genes. Low overlap alone does not prove distinct mechanisms; gene universes and set sizes matter.',
 '06_stability': 'Conditional C11 bootstrap stability, not a new independent cohort or repeated GSEA significance test. Gene universe and rank are fixed; see supplied methods.',
 '07_reference_features': 'New frozen-release-model latent/ME reference atlas, NOT historical OOF features. Load the reference_atlas subdirectory using ReferenceAtlas.load().'
}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024), b''):h.update(b)
 return h.hexdigest()

def organize(root, research):
 changes={}
 def move(src, step):
  dst=root/step/src.name
  if dst.exists():raise FileExistsError(dst)
  dst.parent.mkdir(parents=True,exist_ok=True)
  old=str(src.relative_to(root));shutil.move(str(src),str(dst));changes[old]=str(dst.relative_to(root))
 for step in STEPS:(root/step).mkdir(exist_ok=True)
 for folder in ['attribution','programs','figures','tables','research_scripts']:
  base=root/folder
  if not base.exists():continue
  for f in list(base.iterdir()):
   n=f.name
   if folder=='attribution':step='01_risk_states' if n in ['cluster_definitions.tsv','all_axis_values_and_shapley.tsv.gz'] else '02_gene_attribution'
   elif folder=='programs':step='03_nmf_programs'
   elif folder=='figures':step='01_risk_states' if '5A_' in n else '05_immune_overlap' if 'immune' in n else '04_program_enrichment' if '5C_' in n else '03_nmf_programs'
   elif folder=='tables':step='06_stability' if 'bootstrap' in n or 'stability' in n else '05_immune_overlap' if 'immune' in n else '04_program_enrichment'
   else:step='02_gene_attribution' if n.startswith('15_') else '04_program_enrichment' if n.startswith('64_') else '03_nmf_programs'
   move(f,step)
  base.rmdir()
 if (root/'reference_atlas').exists():move(root/'reference_atlas','07_reference_features')
 scripts={
 '01_risk_states':['14_plot_figure5_shared_risk_architecture.py'],
 '04_program_enrichment':['50_pid_dorothea_program_annotation.py','51_cancer_library_program_annotation.py','55_red_regulatory_evidence.py','65_package_enrichment_supplement.py'],
 '05_immune_overlap':['63_immune_overlap_and_evidence_tiers.py'],
 '06_stability':['57_red_regulatory_support.py']}
 for step,names in scripts.items():
  for name in names:shutil.copy2(research/'prognostic_mechanism/scripts'/name,root/step/name)
 provenance=root/'provenance.json'
 if provenance.exists():
  records=json.loads(provenance.read_text())
  for record in records:
   old=record['file']
   for src,dst in changes.items():
    if old==src or old.startswith(src+'/'):
     record['file']=dst+old[len(src):];break
  provenance.write_text(json.dumps(records,indent=2))
 for step,description in STEPS.items():
  files=sorted(p.name for p in (root/step).iterdir() if p.name!='README.md')
  (root/step/'README.md').write_text('# '+step+'\n\n'+description+'\n\n## Files\n'+''.join('- `'+n+'`\n' for n in files)+'\n## Reuse\nJoin tables by sample IDs. Original scripts are provenance implementations and can depend on original project paths, intermediate files, libraries and gene-set resources; they are not standalone rerunnable pipelines. No analysis or significance threshold was changed during packaging.\n')
 readme=root/'README.md'
 if readme.exists() and not (root/'METHODS_original_export.md').exists():
  methods=readme.read_text()
  for src,dst in sorted(changes.items(),key=lambda x:-len(x[0])):methods=methods.replace(src,dst)
  methods+='\nLegacy generic folder references refer to the step-specific directories listed in README.md.\n'
  (root/'METHODS_original_export.md').write_text(methods)
 readme.write_text('# CancerXpress TCGA comparison v1.0.0\n\nOrganized by analysis step. Each directory contains its data, available figures, implementation scripts and README.\n\n'+''.join('- ['+s+']('+s+'/README.md): '+d+'\n' for s,d in STEPS.items())+'\nHistorical OOF attribution and new frozen-model features must not be conflated. Source redistribution permissions must be reviewed before public release. See METHODS_original_export.md for original export details.\n')
 # Preserve detailed Chinese documentation in all future exports.
 source_files = {
  '01_risk_states': [research/'visu/cluster_oof_sample_axis_contributions.py'] +
      [research/'output/run1_survival_main_figure_curated/sample_axis_contribution_clustering'/n for n in
       ['cluster_number_selection.tsv','selected_cluster_bootstrap_stability.tsv',
        'cluster_axis_contribution_profiles.tsv','cluster_cancer_composition.tsv']],
  '06_stability': [research/'prognostic_mechanism/results/figure5/Supplementary_C11_regulatory_stability_expression_support.pdf']}
 records=json.loads(provenance.read_text()) if provenance.exists() else []
 for step, sources in source_files.items():
  for src in sources:
   if not src.is_file():
    raise FileNotFoundError(src)
   dst=root/step/src.name
   shutil.copy2(src,dst)
   relative=str(dst.relative_to(root))
   if not any(r['file']==relative for r in records):
    records.append({'file':relative,'source':str(src.relative_to(research))})
 for step, names in scripts.items():
  for name in names:
   relative=step+'/'+name
   if not any(r['file']==relative for r in records):
    records.append({'file':relative,'source':'prognostic_mechanism/scripts/'+name})
 provenance.write_text(json.dumps(records,indent=2))
 templates=Path(__file__).resolve().parent/'tcga_handoff_docs'
 for step in STEPS:
  description=(templates/(step+'.md')).read_text(encoding='utf-8')
  description+='\n## 本目录文件索引与表头\n\n以下为实际交付文件。字段名保留原始英文，定义以上文和原始方法为准；并非所有表中的统计量都可互换。\n\n'
  for f in sorted((root/step).iterdir()):
   if f.name=='README.md':continue
   description+='- `'+f.name+'`'+('：子目录。' if f.is_dir() else '。')+'\n'
   if f.name.endswith(('.tsv','.tsv.gz')):
    opener=gzip.open if f.name.endswith('.gz') else open
    with opener(f,'rt',encoding='utf-8') as stream:
     columns=next(csv.reader(stream,delimiter='\t'),[])
    description+='\n表头：`'+'`、`'.join(columns)+'`。\n\n'
  (root/step/'README.md').write_text(description,encoding='utf-8')
 readme.write_text((templates/'README.md').read_text(encoding='utf-8'),encoding='utf-8')
 files=[{'path':str(f.relative_to(root)),'size':f.stat().st_size,'sha256':digest(f)} for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json']
 (root/'manifest.json').write_text(json.dumps({'version':'1.0.0','layout':'analysis_steps','files':files},indent=2))
 archive=Path(str(root)+'.tar.gz');tmp=Path(str(archive)+'.partial')
 with tarfile.open(str(tmp),'w:gz',compresslevel=1) as tar:tar.add(str(root),arcname=root.name)
 tmp.replace(archive)
 archive.with_suffix('.sha256').write_text(digest(archive)+'  '+archive.name+'\n')
 print('READY',archive,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--research-root',type=Path,required=True);a=p.parse_args();organize(a.root,a.research_root)
