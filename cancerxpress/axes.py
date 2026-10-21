"""Stable functional-axis names in the frozen model's output order."""
FUNCTIONAL_AXIS_NAMES = (
    'repair_proteostasis',
    'immune_inflammation',
    'hypoxia_myogenesis',
    'inflammatory_stress',
    'stromal_remodeling',
    'developmental_regulation',
    'proliferation',
    'metabolism_detoxification',
)

FUNCTIONAL_AXIS_LABELS = {
    'repair_proteostasis': {'en': 'Repair–proteostasis', 'zh': '修复–蛋白稳态'},
    'immune_inflammation': {'en': 'Immune inflammation', 'zh': '免疫炎症'},
    'hypoxia_myogenesis': {'en': 'Hypoxia–myogenesis', 'zh': '缺氧–肌样'},
    'inflammatory_stress': {'en': 'Inflammatory stress', 'zh': '炎症应激'},
    'stromal_remodeling': {'en': 'Stromal remodeling', 'zh': '基质重塑'},
    'developmental_regulation': {'en': 'Developmental regulation', 'zh': '发育调控'},
    'proliferation': {'en': 'Proliferation', 'zh': '细胞增殖'},
    'metabolism_detoxification': {'en': 'Metabolism–detoxification', 'zh': '代谢解毒'},
}


def validate_axis_name(axis_name):
    if axis_name not in FUNCTIONAL_AXIS_NAMES:
        raise ValueError('axis_name must be one of: ' + ', '.join(FUNCTIONAL_AXIS_NAMES))
