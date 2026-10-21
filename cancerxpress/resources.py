from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os
from ._legacy import axis_checkpoint_directory, risk_metadata


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
def default_resource_root():
    configured = os.environ.get('CANCERXPRESS_HOME')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / '.cache' / 'cancerxpress' / 'resources'


RESOURCE_ROOT = default_resource_root()
ASSET_ROOT = RESOURCE_ROOT / 'assets'
MODEL_ROOT = RESOURCE_ROOT / 'models'
ENCODER_ROOT = RESOURCE_ROOT / 'encoders'
MAPPING_ROOT = RESOURCE_ROOT / 'mappings'
GENE_ANNOTATION_ROOT = RESOURCE_ROOT / 'gene_annotation'


@dataclass(frozen=True)
class ResourcePaths:
    pixel_coords: Path = field(default_factory=lambda: default_resource_root() / 'assets' / 'tsne_sorted_pixels_coords.csv')
    max_norm_tpm: Path = field(default_factory=lambda: default_resource_root() / 'assets' / 'max_norm_tpm.tsv')
    gene_id_map: Path = field(default_factory=lambda: default_resource_root() / 'gene_annotation' / 'ensembl_symbol_id.txt')
    primary_site_encoder: Path = field(default_factory=lambda: default_resource_root() / 'encoders' / 'primary_site_encoder.pkl')
    cancer_type_encoder: Path = field(default_factory=lambda: default_resource_root() / 'encoders' / 'cancer_type_encoder.pkl')
    sample_type_encoder: Path = field(default_factory=lambda: default_resource_root() / 'encoders' / 'sample_type_encoder.pkl')
    risk_preprocessing: Path = field(default_factory=lambda: default_resource_root() / 'mappings' / 'survival_risk_preprocessing.json')


@dataclass(frozen=True)
class ModelPaths:
    batch_correction: Path = field(default_factory=lambda: default_resource_root() / 'models' / 'pretrained' / 'batch_correction')
    functional_axes: Path = field(default_factory=lambda: axis_checkpoint_directory(default_resource_root() / 'models'))
    primary_site: Path = field(default_factory=lambda: default_resource_root() / 'models' / 'finetuned' / 'primary_site')
    cancer_type: Path = field(default_factory=lambda: default_resource_root() / 'models' / 'finetuned' / 'cancer_type')
    survival_risk: Path = field(default_factory=lambda: default_resource_root() / 'models' / 'finetuned' / 'survival_risk')

    @classmethod
    def from_directory(cls, directory):
        root = Path(directory).expanduser().resolve()
        return cls(**{name: axis_checkpoint_directory(root) if name == 'functional_axes' else
                      root / ('pretrained' if name == 'batch_correction' else 'finetuned') / name
                      for name in cls.__dataclass_fields__})


def resource_paths_from_directory(directory):
    root = Path(directory).expanduser().resolve()
    defaults = ResourcePaths()
    base = default_resource_root()
    return ResourcePaths(**{name: root / getattr(defaults, name).relative_to(base)
                            for name in defaults.__dataclass_fields__})


def load_risk_preprocessing(path: Path | None = None) -> dict:
    target = path or ResourcePaths().risk_preprocessing
    with open(target, 'r', encoding='utf-8') as f:
        return risk_metadata(json.load(f))
