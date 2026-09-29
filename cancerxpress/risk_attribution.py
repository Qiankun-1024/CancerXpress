"""Axis-specific attribution primitives, independent of project data paths.

All models must be deterministic, sample-independent inference callables.
Risk models must return a scalar log-risk per sample.
"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import tensorflow as tf


@dataclass
class AxisShapleyResult:
    values: np.ndarray
    baseline_log_risk: np.ndarray
    predicted_log_risk: np.ndarray
    completeness_error: np.ndarray


@dataclass
class GeneAxisIGResult:
    values: np.ndarray
    baseline_output: np.ndarray
    predicted_output: np.ndarray
    completeness_error: np.ndarray


@dataclass
class AxisRiskAttributionResult:
    axis_shapley: pd.DataFrame
    gene_to_axis_ig: pd.DataFrame
    gene_risk_contribution: pd.DataFrame
    diagnostics: pd.DataFrame
    axis: str


def _finite(value, name):
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise ValueError(name + ' must contain only finite values')
    return arr


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name + ' must be a positive integer')


def exact_axis_shapley(model, features, baseline_me, coalition_batch_size=256):
    """Exact conditional Shapley, holding all non-ME inputs fixed.

    ``features['me']`` is (samples, axes); baseline is (axes,), (1, axes)
    or (samples, axes). Other feature arrays must have the same sample count.
    Enumerates all 2**axes coalitions, with a safety limit of 12 axes.
    """
    _positive_integer(coalition_batch_size, 'coalition_batch_size')
    me = _finite(features['me'], 'me').astype(np.float32)
    if me.ndim != 2 or not len(me) or not 1 <= me.shape[1] <= 12:
        raise ValueError('me must be nonempty (samples, axes), with 1..12 axes')
    base = np.broadcast_to(_finite(baseline_me, 'baseline_me'), me.shape).astype(np.float32)
    others = {}
    for name, value in features.items():
        if name == 'me':
            continue
        value = _finite(value, name)
        if value.ndim == 0 or value.shape[0] != len(me):
            raise ValueError(name + ' must have the same sample count as me')
        others[name] = value
    axes = me.shape[1]
    ids = np.arange(1 << axes)
    masks = ((ids[:, None] >> np.arange(axes)) & 1).astype(np.float32)
    phi = np.zeros(me.shape, dtype=np.float64)
    risks = np.empty((len(me), 2), dtype=np.float64)
    for i in range(len(me)):
        values = []
        for start in range(0, len(ids), coalition_batch_size):
            mask = masks[start:start + coalition_batch_size]
            inputs = {'me': tf.convert_to_tensor(base[i] + mask * (me[i] - base[i]))}
            inputs.update({k: tf.convert_to_tensor(np.repeat(v[i:i+1], len(mask), axis=0))
                           for k, v in others.items()})
            output = _finite(model(inputs, training=False), 'model output')
            if output.shape not in ((len(mask),), (len(mask), 1)):
                raise ValueError('risk model must return one scalar per sample')
            values.extend(output.reshape(-1))
        values = np.asarray(values, dtype=np.float64)
        risks[i] = values[[0, -1]]
        for axis in range(axes):
            bit = 1 << axis
            for coalition in ids[(ids & bit) == 0]:
                size = bin(int(coalition)).count('1')
                weight = 1.0 / (axes * math.comb(axes - 1, size))
                phi[i, axis] += weight * (values[coalition | bit] - values[coalition])
    error = phi.sum(axis=1) - (risks[:, 1] - risks[:, 0])
    return AxisShapleyResult(phi, risks[:, 0], risks[:, 1], error)


def gene_axis_integrated_gradients(model, inputs, baseline, output_index, steps=64,
                                   internal_batch_size=32):
    """Trapezoidal IG in model-input space with bounded interpolation batches.

    Returns attributions with the input shape and a per-sample completeness
    residual. Images must be mapped to genes before gene-risk allocation.
    """
    _positive_integer(steps, 'steps')
    _positive_integer(internal_batch_size, 'internal_batch_size')
    x = _finite(inputs, 'inputs').astype(np.float32)
    if x.ndim < 2 or not len(x):
        raise ValueError('inputs must have a nonempty sample dimension')
    b = np.broadcast_to(_finite(baseline, 'baseline'), x.shape).astype(np.float32)
    pred = _finite(model(tf.convert_to_tensor(x), training=False), 'model output')
    if (pred.ndim != 2 or pred.shape[0] != len(x) or
            not isinstance(output_index, (int, np.integer)) or
            not 0 <= output_index < pred.shape[1]):
        raise ValueError('output_index must select a column in (samples, outputs)')
    baseline_pred = _finite(model(tf.convert_to_tensor(b), training=False), 'baseline output')
    delta = x - b
    gradient_sum = np.zeros(x.shape, dtype=np.float64)
    # Flatten (integration step, sample) indices without materializing the path.
    total = (steps + 1) * len(x)
    for start in range(0, total, internal_batch_size):
        indices = np.arange(start, min(start + internal_batch_size, total))
        step_index, sample_index = np.divmod(indices, len(x))
        alpha = (step_index / float(steps)).reshape((-1,) + (1,) * (x.ndim - 1))
        path = tf.convert_to_tensor(b[sample_index] + alpha * delta[sample_index], dtype=tf.float32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(path)
            output = model(path, training=False)[:, output_index]
        gradient = tape.gradient(output, path)
        if gradient is None:
            raise ValueError('model output is disconnected from inputs')
        gradient = _finite(gradient, 'gradients')
        weights = np.where((step_index == 0) | (step_index == steps), 0.5, 1.0)
        weights = weights.reshape((-1,) + (1,) * (x.ndim - 1))
        np.add.at(gradient_sum, sample_index, gradient * weights)
    attrs = delta * gradient_sum / steps
    prediction = pred[:, output_index]
    reference = baseline_pred[:, output_index]
    error = attrs.reshape(len(x), -1).sum(axis=1) - (prediction - reference)
    return GeneAxisIGResult(attrs, reference, prediction, error)


def allocate_axis_risk(gene_ig, axis_shapley, axis_delta, minimum_denominator=1e-6,
                       maximum_relative_error=0.1):
    """Allocate phi * IG / sum(IG), with explicit QC; invalid rows become NaN.

    This is a proportional hierarchical allocation, NOT end-to-end gene Shapley.
    IG must contain all mapped genes, not a preselected top-gene subset.
    """
    if not np.isfinite(minimum_denominator) or minimum_denominator <= 0:
        raise ValueError('minimum_denominator must be finite and positive')
    if not np.isfinite(maximum_relative_error) or maximum_relative_error < 0:
        raise ValueError('maximum_relative_error must be finite and nonnegative')
    ig = np.asarray(gene_ig, dtype=np.float64)
    if ig.ndim != 2 or not len(ig) or not ig.shape[1]:
        raise ValueError('gene_ig must be nonempty (samples, genes)')
    phi = np.asarray(axis_shapley, dtype=np.float64)
    delta = np.asarray(axis_delta, dtype=np.float64)
    if phi.shape != (len(ig),) or delta.shape != (len(ig),):
        raise ValueError('axis_shapley and axis_delta must be (samples,)')
    sums = ig.sum(axis=1)
    error = sums - delta
    relative = np.abs(error) / (np.abs(delta) + 1e-8)
    valid = (np.isfinite(ig).all(axis=1) & np.isfinite(phi) & np.isfinite(delta)
             & (np.abs(sums) >= minimum_denominator) & (relative <= maximum_relative_error))
    allocated = np.full(ig.shape, np.nan)
    allocated[valid] = ig[valid] * (phi[valid] / sums[valid])[:, None]
    qc = pd.DataFrame({'gene_ig_sum': sums, 'axis_delta': delta,
                       'ig_completeness_error': error, 'ig_relative_error': relative,
                       'allocation_valid': valid,
                       'allocation_error': allocated.sum(axis=1) - phi})
    return allocated, qc


def optimize_axis_baseline(model, initial_image, gene_pixel_mask, target_me,
                           steps=500, learning_rate=0.01, regularization=1e-3):
    """Match a frozen training ME reference, updating only mapped gene pixels.

    Initial image must be (1, ...), normalized to [0, 1]. The target is a
    single ME vector. Returns (image, optimization_history); matching is not
    guaranteed. This synthetic baseline must never be fitted using outcomes.
    """
    _positive_integer(steps, 'steps')
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive')
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError('regularization must be finite and nonnegative')
    initial = _finite(initial_image, 'initial_image').astype(np.float32)
    if initial.ndim < 2 or initial.shape[0] != 1 or np.any((initial < 0) | (initial > 1)):
        raise ValueError('initial_image must be (1, ...) with values in [0, 1]')
    mask = np.broadcast_to(_finite(gene_pixel_mask, 'gene_pixel_mask'), initial.shape).astype(np.float32)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('gene_pixel_mask must be binary')
    target = _finite(target_me, 'target_me').astype(np.float32)
    if target.ndim == 1:
        target = target[None, :]
    prediction = np.asarray(model(tf.convert_to_tensor(initial), training=False))
    if prediction.shape != target.shape or target.ndim != 2 or target.shape[0] != 1:
        raise ValueError('target_me must match the single-sample model output')
    baseline = tf.Variable(initial)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    history = []
    for step in range(steps + 1):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(baseline)
            prediction = model(baseline, training=False)
            mse = tf.reduce_mean(tf.square(prediction - target))
            deviation = tf.reduce_mean(tf.square(baseline - initial))
            loss = mse + regularization * deviation
        if step % 50 == 0 or step == steps:
            history.append({'step': step, 'total_loss': float(loss.numpy()),
                            'me_output_mse': float(mse.numpy()),
                            'mean_squared_input_deviation': float(deviation.numpy()),
                            'maximum_abs_me_error': float(tf.reduce_max(tf.abs(prediction - target)).numpy())})
        if step < steps:
            gradient = tape.gradient(loss, baseline)
            if gradient is None:
                raise ValueError('model output is disconnected from baseline')
            _finite(gradient, 'baseline gradient')
            optimizer.apply_gradients([(gradient * mask, baseline)])
            baseline.assign(initial + mask * (tf.clip_by_value(baseline, 0.0, 1.0) - initial))
    _finite(baseline, 'optimized baseline')
    return baseline.numpy(), pd.DataFrame(history)
