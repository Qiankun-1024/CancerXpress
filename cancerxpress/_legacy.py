"""Read historical on-disk names without exposing them in the public API.

Do not rename files in released bundles: their manifests and hashes are frozen.
"""
from pathlib import Path
from .axes import FUNCTIONAL_AXIS_NAMES


_COLORS = ('black', 'blue', 'brown', 'green', 'pink', 'red', 'turquoise', 'yellow')
_OLD_NAMES = dict(zip(_COLORS, FUNCTIONAL_AXIS_NAMES))
_OLD_NAMES.update({'ME' + color: name for color, name in zip(_COLORS, FUNCTIONAL_AXIS_NAMES)})


def axis_labels(labels):
    return [_OLD_NAMES.get(label, label) for label in labels]


def axis_checkpoint_directory(models_root):
    root = Path(models_root) / 'finetuned'
    preferred = root / 'functional_axes'
    historical = root / 'me'
    return preferred if preferred.exists() or not historical.exists() else historical


def axis_payload(mapping, key='functional_axes'):
    historical = {'functional_axes': 'me', 'functional_axis_columns': 'me_columns'}[key]
    return mapping[key] if key in mapping else mapping[historical]


def risk_metadata(metadata):
    result = dict(metadata)
    result['functional_axis_columns'] = axis_labels(axis_payload(result, 'functional_axis_columns'))
    result.pop('me_columns', None)
    return result
