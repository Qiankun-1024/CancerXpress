import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from cancerxpress import ReferenceAtlas, LatentAdapter, fit_latent_adapter


def atlas_fixture():
    rng = np.random.RandomState(10)
    index = ['r%d' % i for i in range(20)]
    z = pd.DataFrame(rng.normal(size=(20, 4)), index=index, columns=['z0','z1','z2','z3'])
    functional_axes = pd.DataFrame(rng.normal(size=(20, 2)), index=index, columns=['developmental_regulation','immune_inflammation'])
    meta = pd.DataFrame({'cancer_type':['BRCA']*20, 'subtype':['A']*10+['B']*10},index=index)
    return ReferenceAtlas(z, functional_axes, meta, 'test-signature')


class ReferenceTests(unittest.TestCase):
    def test_missing_local_atlas(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, 'atlas.json, samples.tsv, features.npz'):
                ReferenceAtlas.load(tmp)

    def test_query_and_roundtrip(self):
        ref = atlas_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ref.save(Path(tmp)/'ref')
            ref = ReferenceAtlas.load(Path(tmp)/'ref')
        q = ref.latent.iloc[[3]].copy()
        q.index = ['new']
        hits = ref.query(q, k=2, metric='euclidean')
        self.assertEqual(hits.reference_sample.iloc[0], 'r3')
        self.assertAlmostEqual(hits.distance.iloc[0], 0)
        hits = ref.query(ref.latent.iloc[[3]], k=2, metric='euclidean')
        self.assertNotIn('r3', hits.reference_sample.tolist())
        with self.assertRaises(ValueError):
            ref.query(q,query_metadata=pd.DataFrame({'cancer_type':['UNKNOWN']},index=q.index))

    def test_column_order(self):
        ref=atlas_fixture()
        a=ref.query(ref.functional_axes.iloc[:1],space='functional_axes')
        b=ref.query(ref.functional_axes.iloc[:1,::-1],space='functional_axes')
        np.testing.assert_allclose(a.distance,b.distance)

    def test_missing_reference_labels(self):
        ref = atlas_fixture()
        meta = ref.metadata.copy()
        meta.loc['r3', 'cancer_type'] = np.nan
        meta.loc['r4', 'cancer_type'] = ' '
        ref = ReferenceAtlas(ref.latent, ref.functional_axes, meta, ref.model_signature)
        with tempfile.TemporaryDirectory() as tmp:
            ref.save(Path(tmp) / 'ref')
            ref = ReferenceAtlas.load(Path(tmp) / 'ref')
        self.assertEqual(ref.metadata.cancer_type.isna().sum(), 2)
        q = ref.latent.iloc[[3]].copy(); q.index = ['new']
        qm = pd.DataFrame({'cancer_type': [np.nan]}, index=q.index)
        hits = ref.query(q, metric='euclidean', query_metadata=qm)
        self.assertEqual(hits.reference_sample.iloc[0], 'r3')
        self.assertTrue(pd.isna(hits.reference_cancer_type.iloc[0]))
        with self.assertRaises(ValueError):
            ref.query(q, same_cancer=True, query_metadata=qm)
        qm['cancer_type'] = 'BRCA'
        hits = ref.query(q, same_cancer=True, query_metadata=qm)
        self.assertFalse(hits.reference_cancer_type.isna().any())

    def test_adaptation_excludes_unlabelled_reference(self):
        ref = atlas_fixture()
        target = ref.latent.copy(); target.index = ['t%d' % i for i in range(20)]
        meta = ref.metadata.copy(); meta.index = target.index
        ref.metadata.loc['r0', 'cancer_type'] = np.nan
        result = fit_latent_adapter(target, meta, ref, epochs=1)
        used = result.splits[result.splits.cohort == 'reference'].sample_id
        self.assertNotIn('r0', set(used))
        meta.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_latent_adapter(target, meta, ref, epochs=1)

    def test_adaptation_and_serialization(self):
        ref=atlas_fixture()
        target=ref.latent.copy()+.4
        target.index=['t%d'%i for i in range(20)]
        meta=ref.metadata.copy();meta.index=target.index
        result=fit_latent_adapter(target,meta,ref,biological_labels=['subtype'],
                                  epochs=3,patience=2,batch_size=5)
        self.assertTrue(np.isfinite(result.metrics.MMD).all())
        for (cohort, sample), group in result.splits.groupby(['cohort','sample_id']):
            self.assertEqual(len(group),1)
        transformed=result.adapter.transform(target,meta)
        with tempfile.TemporaryDirectory() as tmp:
            result.save(Path(tmp)/'adapter')
            restored=LatentAdapter.load_adapter(Path(tmp)/'adapter')
            np.testing.assert_allclose(transformed,restored.transform(target,meta))
        meta['cancer_type']='UNKNOWN'
        with self.assertRaises(ValueError):
            restored.transform(target,meta)

    def test_unseen_stratum_and_small_groups(self):
        ref=atlas_fixture()
        target=ref.latent.copy();target.index=['t%d'%i for i in range(20)]
        meta=ref.metadata.copy();meta.index=target.index
        meta['subtype']='new'
        with self.assertRaises(ValueError):
            fit_latent_adapter(target,meta,ref,['subtype'],epochs=1)
        with self.assertRaises(ValueError):
            fit_latent_adapter(target.iloc[:3],meta.iloc[:3],ref,epochs=1)

    def test_repeated_patient_split(self):
        ref=atlas_fixture()
        target=ref.latent.copy();target.index=['t%d'%i for i in range(20)]
        meta=ref.metadata.copy();meta.index=target.index
        meta['patient_id']=['p%d'%(i//2) for i in range(20)]
        result=fit_latent_adapter(target,meta,ref,epochs=1)
        groups=result.splits[result.splits.cohort=='target'].groupby('patient_id').split.nunique()
        self.assertTrue((groups==1).all())


if __name__=='__main__':
    unittest.main()
