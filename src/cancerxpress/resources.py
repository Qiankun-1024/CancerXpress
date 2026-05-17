from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
RESOURCE_ROOT = PROJECT_ROOT / 'resources'
ASSET_ROOT = RESOURCE_ROOT / 'assets'
MODEL_ROOT = RESOURCE_ROOT / 'models'
ENCODER_ROOT = RESOURCE_ROOT / 'encoders'
MAPPING_ROOT = RESOURCE_ROOT / 'mappings'
GENE_ANNOTATION_ROOT = RESOURCE_ROOT / 'gene_annotation'


@dataclass(frozen=True)
class ResourcePaths:
    pixel_coords: Path = ASSET_ROOT / 'tsne_sorted_pixels_coords.csv'
    max_norm_tpm: Path = ASSET_ROOT / 'max_norm_tpm.tsv'
    gene_id_map: Path = GENE_ANNOTATION_ROOT / 'ensembl_symbol_id.txt'
    primary_site_encoder: Path = ENCODER_ROOT / 'primary_site_encoder.pkl'
    cancer_type_encoder: Path = ENCODER_ROOT / 'cancer_type_encoder.pkl'
    sample_type_encoder: Path = ENCODER_ROOT / 'sample_type_encoder.pkl'
    risk_cancer_vocab: Path = MAPPING_ROOT / 'survival_risk_cancer_vocab.json'


@dataclass(frozen=True)
class ModelPaths:
    batch_correction: Path = MODEL_ROOT / 'pretrained' / 'batch_correction'
    me: Path = MODEL_ROOT / 'finetuned' / 'me'
    primary_site: Path = MODEL_ROOT / 'finetuned' / 'primary_site'
    cancer_type: Path = MODEL_ROOT / 'finetuned' / 'cancer_type'
    survival_risk: Path = MODEL_ROOT / 'finetuned' / 'survival_risk'


def load_risk_cancer_vocab(path: Path | None = None) -> list[str]:
    target = path or ResourcePaths().risk_cancer_vocab
    with open(target, 'r', encoding='utf-8') as f:
        return json.load(f)
