"""Input checks shared by preprocessing and public prediction interfaces."""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_complex_dtype, is_bool_dtype


def validate_frame_labels(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(name + ' must be a nonempty samples-by-features DataFrame')
    for labels, kind in ((frame.index, 'sample IDs'), (frame.columns, 'column labels')):
        if isinstance(labels, pd.MultiIndex) or labels.has_duplicates or labels.hasnans:
            raise ValueError(name + ' requires unique, nonmissing ' + kind)
        if any(isinstance(label, str) and not label.strip() for label in labels):
            raise ValueError(name + ' contains empty ' + kind)


def validate_expression(expression, name='expr_tpm'):
    validate_frame_labels(expression, name)
    if any(not is_numeric_dtype(dtype) or is_complex_dtype(dtype) or is_bool_dtype(dtype)
           for dtype in expression.dtypes):
        raise ValueError(name + ' must contain real numeric TPM values')
    try:
        values = expression.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must contain real numeric TPM values') from exc
    if not np.isfinite(values).all():
        raise ValueError(name + ' contains missing or infinite TPM values')
    if (values < 0).any():
        raise ValueError(name + ' must contain nonnegative TPM values')


def align_clinical(clinical, sample_ids, required=None):
    validate_frame_labels(clinical, 'clinical')
    required = list(required if required is not None else
                    ('age', 'sex', 'race', 'tumor_stage', 'cancer_type'))
    missing_columns = set(required) - set(clinical.columns)
    if missing_columns:
        raise ValueError('clinical is missing required columns: ' + str(sorted(missing_columns)))
    missing_ids = sample_ids[~sample_ids.isin(clinical.index)]
    if len(missing_ids):
        raise ValueError('clinical is missing expression samples: ' + str(list(missing_ids[:5])))
    # Reordering is safe only after checking identifiers, never by row position.
    return clinical.reindex(sample_ids).copy()
