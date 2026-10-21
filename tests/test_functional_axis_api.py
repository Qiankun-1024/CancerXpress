"""Functional-axis naming, automatic input checks, and old bundle readers."""
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cancerxpress import CancerXpress, CancerXpressResult, ReferenceAtlas, FUNCTIONAL_AXIS_NAMES, FUNCTIONAL_AXIS_LABELS
from cancerxpress.axes import validate_axis_name
from cancerxpress._legacy import axis_labels
from cancerxpress.preprocess import ExpressionPreprocessor, GeneIDConverter
from cancerxpress.resources import ModelPaths, load_risk_preprocessing
from cancerxpress.validation import validate_expression, align_clinical


class FunctionalAxisAPITests(unittest.TestCase):
    def setUp(self):
        self.expression = pd.DataFrame([[1., 2.], [3., 4.]],
                                       index=['a', 'b'], columns=['ENSG1', 'ENSG2'])
        self.clinical = pd.DataFrame({
            'age': [40, 50], 'sex': ['female'] * 2, 'race': ['white'] * 2,
            'tumor_stage': ['stage ii'] * 2, 'cancer_type': ['BRCA'] * 2,
        }, index=['a', 'b'])

    def processor(self):
        converter = GeneIDConverter.__new__(GeneIDConverter)
        converter.ensembl_to_symbol = {'ENSG1': 'G1', 'ENSG2': 'G2'}
        converter.symbol_to_ensembl = {'G1': 'ENSG1', 'G2': 'ENSG2'}
        processor = ExpressionPreprocessor.__new__(ExpressionPreprocessor)
        processor.converter = converter
        processor.training_gene_order = pd.Index(['ENSG1', 'ENSG2'])
        return processor

    def test_public_names(self):
        self.assertTrue(callable(CancerXpress.predict_functional_axes))
        self.assertTrue(callable(CancerXpress.attribute_functional_axis))
        self.assertFalse(hasattr(CancerXpress, 'predict_me'))
        self.assertFalse(hasattr(CancerXpress, 'attribute_me'))
        self.assertIn('functional_axes', CancerXpressResult.__dataclass_fields__)
        self.assertIn('axis_name', inspect.signature(CancerXpress.attribute_axis_risk).parameters)
        self.assertEqual(FUNCTIONAL_AXIS_NAMES[5], 'developmental_regulation')

    def test_biological_names_and_display_labels(self):
        self.assertEqual(tuple(FUNCTIONAL_AXIS_LABELS), FUNCTIONAL_AXIS_NAMES)
        self.assertEqual(FUNCTIONAL_AXIS_LABELS['developmental_regulation']['en'],
                         'Developmental regulation')
        for name in FUNCTIONAL_AXIS_NAMES:
            validate_axis_name(name)
            self.assertTrue(FUNCTIONAL_AXIS_LABELS[name]['zh'])
        for name in ('red', 'MEred', 'blue', 'MEblue'):
            with self.assertRaisesRegex(ValueError, 'axis_name'):
                validate_axis_name(name)

    def test_legacy_names_preserve_frozen_axis_order(self):
        colors = ['black', 'blue', 'brown', 'green', 'pink', 'red', 'turquoise', 'yellow']
        self.assertEqual(axis_labels(colors), list(FUNCTIONAL_AXIS_NAMES))
        self.assertEqual(axis_labels(['ME'+x for x in colors]), list(FUNCTIONAL_AXIS_NAMES))
        self.assertEqual(axis_labels(FUNCTIONAL_AXIS_NAMES), list(FUNCTIONAL_AXIS_NAMES))

    def test_expression_numeric_checks(self):
        for value in (np.nan, np.inf, -np.inf, -1., 'invalid', 1+2j, True):
            frame = self.expression.astype(object)
            frame.iloc[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'TPM'):
                validate_expression(frame)
        validate_expression(self.expression)

    def test_numeric_nan_and_negative(self):
        for value in (np.nan, np.inf, -1.):
            frame = self.expression.copy()
            frame.iloc[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'TPM'):
                self.processor().harmonize_expression(frame)

    def test_nullable_missing_value(self):
        frame = self.expression.astype('Float64')
        frame.iloc[0, 0] = pd.NA
        with self.assertRaisesRegex(ValueError, 'missing or infinite'):
            validate_expression(frame)

    def test_expression_labels(self):
        for field in ('index', 'columns'):
            for labels in (['x', 'x'], ['x', None], ['x', ' ']):
                frame = self.expression.copy()
                setattr(frame, field, labels)
                with self.subTest(field=field, labels=labels), self.assertRaises(ValueError):
                    self.processor().harmonize_expression(frame)

    def test_empty_and_wrong_type(self):
        for value in (None, [[1, 2]], pd.DataFrame(), self.expression.iloc[:0]):
            with self.subTest(value=type(value)), self.assertRaisesRegex(ValueError, 'DataFrame'):
                validate_expression(value)

    def test_automatic_clinical_alignment(self):
        original = self.clinical.iloc[::-1].copy()
        result = align_clinical(original, self.expression.index)
        pd.testing.assert_frame_equal(result, self.clinical)
        self.assertEqual(list(original.index), ['b', 'a'])

    def test_missing_clinical_id_and_columns(self):
        with self.assertRaisesRegex(ValueError, 'missing expression samples'):
            align_clinical(self.clinical.iloc[:1], self.expression.index)
        with self.assertRaisesRegex(ValueError, 'missing required columns'):
            align_clinical(self.clinical.drop(columns='age'), self.expression.index)
        duplicated = pd.concat([self.clinical, self.clinical.iloc[:1]])
        with self.assertRaisesRegex(ValueError, 'unique'):
            align_clinical(duplicated, self.expression.index)
        duplicated = pd.concat([self.clinical, self.clinical[['age']]], axis=1)
        with self.assertRaisesRegex(ValueError, 'unique'):
            align_clinical(duplicated, self.expression.index)

    def test_no_gene_matches(self):
        frame = self.expression.rename(columns={'ENSG1': 'BAD1', 'ENSG2': 'BAD2'})
        with self.assertRaisesRegex(ValueError, 'No input genes match'):
            self.processor().harmonize_expression(frame, 'ensembl')

    def test_mapped_versions_still_sum(self):
        frame = self.expression.rename(columns={'ENSG1': 'ENSG1.1', 'ENSG2': 'ENSG1.2'})
        result = self.processor().harmonize_expression(frame, 'ensembl')
        np.testing.assert_array_equal(result['ENSG1'], [3., 7.])

    def test_high_level_rejects_before_inference(self):
        model = CancerXpress.__new__(CancerXpress)
        model.preprocessor = self.processor()
        bad = self.expression.copy(); bad.iloc[0, 0] = np.nan
        with patch('cancerxpress.pipeline.Predictor') as predictor:
            with self.assertRaises(ValueError): model.predict_functional_axes(bad)
            with self.assertRaises(ValueError): model.predict_survival_risk(bad, self.clinical)
            with self.assertRaises(ValueError): model.attribute_survival_risk(bad, self.clinical)
            with self.assertRaises(ValueError): model.encode_features(bad)
            with self.assertRaises(ValueError): model.predict_survival_risk(self.expression, self.clinical.iloc[:1])
            predictor.assert_not_called()

    def test_old_checkpoint_directory_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old = root/'finetuned/me'; old.mkdir(parents=True)
            self.assertEqual(ModelPaths.from_directory(root).functional_axes, old)
            new = root/'finetuned/functional_axes'; new.mkdir()
            self.assertEqual(ModelPaths.from_directory(root).functional_axes, new)
            path = root/'risk.json'
            path.write_text(json.dumps({'me_columns': [
                'MEblack', 'MEblue', 'MEbrown', 'MEgreen', 'MEpink', 'MEred', 'MEturquoise', 'MEyellow']}))
            metadata = load_risk_preprocessing(path)
            self.assertEqual(metadata['functional_axis_columns'], list(FUNCTIONAL_AXIS_NAMES))
            self.assertNotIn('me_columns', metadata)

    def test_old_atlas_load_new_atlas_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.clinical.to_csv(root/'samples.tsv', sep='\t', index_label='sample_id')
            np.savez(root/'features.npz', latent=[[1., 0.], [0., 1.]], me=[[1., 2.], [2., 1.]])
            (root/'atlas.json').write_text(json.dumps({
                'format_version': 1, 'model_signature': 'fixed',
                'latent_columns': ['z0', 'z1'], 'me_columns': ['MEred', 'MEblue'],
            }))
            atlas = ReferenceAtlas.load(root)
            self.assertEqual(list(atlas.functional_axes.columns), ['developmental_regulation', 'immune_inflammation'])
            atlas.save(root/'new')
            self.assertEqual(json.loads((root/'new/atlas.json').read_text())['format_version'], 2)
            restored = ReferenceAtlas.load(root/'new')
            pd.testing.assert_frame_equal(restored.functional_axes, atlas.functional_axes)
            hits = restored.query(atlas.functional_axes.iloc[:1], space='functional_axes',
                                  metric='euclidean', k=1)
            self.assertEqual(hits.reference_sample.iloc[0], 'b')


if __name__ == '__main__':
    unittest.main()
