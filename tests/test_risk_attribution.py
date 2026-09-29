import unittest
from unittest.mock import patch
import numpy as np
import tensorflow as tf
import pandas as pd

from cancerxpress.risk_attribution import (
    exact_axis_shapley, gene_axis_integrated_gradients,
    allocate_axis_risk, optimize_axis_baseline,
)


class RiskAttributionTests(unittest.TestCase):
    def test_high_level_wrapper_and_baseline_guard(self):
        from cancerxpress.attribution import CancerXpressAttributor
        names = ['MEblack', 'MEblue', 'MEbrown', 'MEgreen', 'MEpink',
                 'MEred', 'MEturquoise', 'MEyellow']

        def me_model(x, training=False):
            return tf.convert_to_tensor(x, dtype=tf.float32)

        def risk_model(f, training=False):
            return tf.reduce_sum(f['me'], axis=1)

        class ToyPredictor:
            def __init__(self, expr):
                self.data = expr.to_numpy(np.float32)

            def _prepare_run1_features(self, frame, clinical, metadata):
                return {'me': frame.to_numpy(np.float32)}

            def _load_run1_survival_model(self, path, metadata):
                return risk_model, 'toy'

        expr = pd.DataFrame(np.arange(1, 9, dtype=np.float32)[None, :],
                            index=['patient'], columns=['g' + str(i) for i in range(8)])
        with patch('cancerxpress.attribution.Predictor', ToyPredictor), \
                patch('cancerxpress.attribution.load_risk_preprocessing', return_value={'me_columns': names}), \
                patch('cancerxpress.attribution.model_loader.load_predict_model', return_value=me_model), \
                patch.object(CancerXpressAttributor, '_to_gene_level',
                             side_effect=lambda frame, values: pd.DataFrame(values, index=frame.index, columns=frame.columns)):
            attributor = CancerXpressAttributor()
            ref = pd.Series(0., index=names[::-1])
            result = attributor.attribute_axis_risk(expr, pd.DataFrame(), 'MEred',
                                                     np.zeros((1, 8)), ref, steps=4)
            self.assertTrue(result.diagnostics.allocation_valid.iloc[0])
            self.assertEqual(result.axis_shapley.index.tolist(), ['patient'])
            self.assertAlmostEqual(result.gene_risk_contribution.sum(axis=1).iloc[0], 6)
            with self.assertRaises(ValueError):
                attributor.attribute_axis_risk(expr, pd.DataFrame(), 'MEred',
                                               np.ones((1, 8)), ref, steps=4)

    def test_shapley_interaction_and_fixed_covariates(self):
        def model(f, training=False):
            x = f['me']
            return x[:, 0] * x[:, 1] + 2 * x[:, 0] + tf.cast(f['cancer'], tf.float32)
        x = np.array([[2, 3], [4, 1]], np.float32)
        result = exact_axis_shapley(model, {'me': x, 'cancer': np.array([5, 9])}, [0, 0], 3)
        np.testing.assert_allclose(result.values, [[7, 3], [10, 2]])
        np.testing.assert_allclose(result.baseline_log_risk, [5, 9])
        np.testing.assert_allclose(result.completeness_error, 0, atol=1e-6)

    def test_eight_axis_nonzero_baseline(self):
        def model(f, training=False):
            return tf.reduce_sum(f['me'] ** 2, axis=1, keepdims=True)
        x = np.arange(16, dtype=np.float32).reshape(2, 8)
        r = exact_axis_shapley(model, {'me': x}, np.ones(8), 17)
        np.testing.assert_allclose(r.values, x ** 2 - 1, atol=1e-6)

    def test_ig_quadratic_and_chunk_invariance(self):
        def model(x, training=False):
            return tf.reduce_sum(x ** 2, axis=1, keepdims=True)
        x = np.array([[2., -3.], [4., 5.]], np.float32)
        a = gene_axis_integrated_gradients(model, x, [1., 1.], 0, steps=8, internal_batch_size=3)
        b = gene_axis_integrated_gradients(model, x, [1., 1.], 0, steps=8, internal_batch_size=100)
        np.testing.assert_allclose(a.values, x ** 2 - 1, atol=1e-5)
        np.testing.assert_allclose(a.values, b.values)
        np.testing.assert_allclose(a.completeness_error, 0, atol=1e-5)

    def test_allocation_qc(self):
        values, qc = allocate_axis_risk([[2, -1], [1, -1], [2, 1], [np.nan, 1]],
                                        [0.5, 1, 1, 1], [1, 0, 2, 1])
        np.testing.assert_allclose(values[0], [1, -0.5])
        self.assertTrue(np.isnan(values[1:]).all())
        self.assertEqual(qc.allocation_valid.tolist(), [True, False, False, False])

    def test_validation(self):
        def model(x, training=False):
            return x
        for steps in [0, -1, 1.5]:
            with self.assertRaises(ValueError):
                gene_axis_integrated_gradients(model, [[1., 2.]], [0, 0], 0, steps)
        with self.assertRaises(ValueError):
            gene_axis_integrated_gradients(model, [[1., 2.]], [0, 0], 2)
        with self.assertRaises(ValueError):
            exact_axis_shapley(model, {'me': [[1., np.nan]]}, [0, 0])
        with self.assertRaises(ValueError):
            allocate_axis_risk([[1, 2]], [1], [3], minimum_denominator=0)

    def test_disconnected_output(self):
        def model(x, training=False):
            return tf.ones((tf.shape(x)[0], 1))
        with self.assertRaises(ValueError):
            gene_axis_integrated_gradients(model, [[1., 2.]], [0, 0], 0)

    def test_baseline_mask_and_history(self):
        def model(x, training=False):
            return tf.reduce_sum(x, axis=1, keepdims=True)
        initial = np.array([[0.1, 0.2]], np.float32)
        b, history = optimize_axis_baseline(model, initial, [[1, 0]], [0.7],
                                            steps=150, regularization=0)
        self.assertAlmostEqual(float(b[0, 1]), 0.2, places=6)
        self.assertLess(history.iloc[-1].maximum_abs_me_error, 0.002)
        self.assertLess(history.iloc[-1].total_loss, history.iloc[0].total_loss)


if __name__ == '__main__':
    unittest.main()
