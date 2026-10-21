"""Regression checks for the supported attribution surface."""
import contextlib
import io
import unittest
from unittest.mock import patch

from cancerxpress import CancerXpress, CancerXpressAttributor, CancerXpressResult
from cancerxpress.predictor import Predictor
from cancerxpress.cli import main


class AttributionAPITests(unittest.TestCase):
    def test_supported_methods(self):
        for cls in (CancerXpress, CancerXpressAttributor):
            for name in ('attribute_primary_site', 'attribute_cancer_type'):
                self.assertFalse(hasattr(cls, name))
            for name in ('attribute_functional_axis', 'attribute_survival_risk', 'attribute_axis_risk'):
                self.assertTrue(callable(getattr(cls, name)))
        for name in ('predict_primary_site', 'predict_cancer_type'):
            self.assertFalse(hasattr(CancerXpress, name))
        self.assertFalse(hasattr(Predictor, 'predict'))
        self.assertNotIn('primary_site', CancerXpressResult.__dataclass_fields__)
        self.assertNotIn('cancer_type', CancerXpressResult.__dataclass_fields__)

    def test_run_all_only_supported_predictions(self):
        model = CancerXpress.__new__(CancerXpress)
        with patch.object(model, 'batch_correct', return_value=('latent', 'expression')), \
                patch.object(model, 'predict_functional_axes', return_value='functional_axes'), \
                patch.object(model, 'predict_survival_risk', return_value='risk'):
            result = model.run_all(None, None)
        self.assertEqual(result.latent, 'latent')
        self.assertEqual(result.corrected_expression, 'expression')
        self.assertEqual(result.functional_axes, 'functional_axes')
        self.assertEqual(result.survival_risk, 'risk')

    def test_cli_rejects_classification_prediction(self):
        for task in ('primary_site', 'cancer_type'):
            with patch('sys.argv', ['cancerxpress', '--input', 'unused.tsv',
                                   '--outdir', 'unused', '--task', task]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)

    def test_cli_rejects_classification_attribution_before_io(self):
        for task in ('primary_site', 'cancer_type'):
            with self.subTest(task=task), patch('sys.argv', [
                'cancerxpress', '--input', 'unused.tsv', '--outdir', 'unused',
                '--task', 'attribution', '--attribution-task', task,
            ]), patch('cancerxpress.cli.CancerXpress') as model, \
                    patch('cancerxpress.cli.pd.read_csv') as read, \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as exc:
                    main()
                self.assertEqual(exc.exception.code, 2)
                self.assertIn('invalid choice', errors.getvalue())
                model.assert_not_called()
                read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
