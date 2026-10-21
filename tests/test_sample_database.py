import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from package_indexed_reference import package
from cancerxpress.sample_database import SampleDatabase, RangeNotSupportedError

class SampleDatabaseTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  source=self.root/'source';source.mkdir();atlas=self.root/'atlas';atlas.mkdir()
  (source/'sample_metadata.tsv').write_text('sample_id\tcancer_type\nA\tBRCA\nB\t\nC\tNormal\n')
  (source/'raw_tpm.tsv').write_text('\tg1\tg2\nA\t1.00\t2e-3\nB\t3\t4\nC\t5\t6\n')
  (source/'corrected_expression.tsv').write_text('\tg2\tg3\tg1\nA\t.1\t.2\t.3\nB\t.4\t.5\t.6\nC\t.7\t.8\t.9\n')
  (atlas/'samples.tsv').write_text('sample_id\tcancer_type\nA\tBRCA\nB\t\nC\tNormal\n')
  (atlas/'atlas.json').write_text(json.dumps({'model_signature':'test','latent_columns':['z1','z2'],'me_columns':['MEred']}))
  np.savez(atlas/'features.npz',latent=np.arange(6).reshape(3,2),me=np.arange(3).reshape(3,1))
  self.out=self.root/'packed';package(source,atlas,self.out,2)
 def tearDown(self):self.tmp.cleanup()
 def db(self,remote=False):
  return SampleDatabase(self.out/'reference_index.json.gz',base_url='https://example.org/files/' if remote else None,cache_dir=self.root/'cache')
 def response(self,url,status,body,headers=None):
  r=requests.Response();r.status_code=status;r.url=url;r.headers.update(headers or {});r.raw=io.BytesIO(body);return r
 def test_local_components(self):
  db=self.db();data=db.fetch(['B','A'],components=['raw_tpm','corrected_expression','metadata','latent','me'])
  self.assertEqual(list(data['raw_tpm'].index),['B','A'])
  np.testing.assert_allclose(data['raw_tpm'].loc['A'],[1,.002])
  self.assertEqual(list(data['corrected_expression'].columns),['g2','g3','g1'])
  self.assertEqual(data['metadata'].loc['B','cancer_type'],'')
  np.testing.assert_equal(data['latent'].loc['A'].values,[0,1])
  with self.assertRaises(KeyError):db.fetch(['absent'])
 def test_local_directory_offline_and_atlas(self):
  with patch.dict('os.environ',{'CANCERXPRESS_REFERENCE_DIR':str(self.out)}), \
       patch.object(requests.sessions.Session,'request',side_effect=AssertionError('Network forbidden')):
   db=SampleDatabase.from_directory()
   self.assertEqual(db.sample_ids,['A','B','C'])
   with patch.object(SampleDatabase,'_store',side_effect=AssertionError('Local cache write forbidden')):
    result=db.fetch(['C','A'],components=['raw_tpm'])
    atlas=db.to_atlas(['B','A'])
   self.assertEqual(list(result['raw_tpm'].index),['C','A'])
   self.assertEqual(list(atlas.latent.index),['B','A'])
   self.assertTrue(pd.isna(atlas.metadata.loc['B','cancer_type']))
   self.assertEqual(atlas.model_signature,'test')
   query=atlas.latent.iloc[[1]].copy();query.index=['new']
   hits=atlas.query(query,k=1,metric='euclidean')
   self.assertEqual(hits.reference_sample.iloc[0],'A')
 def test_local_directory_validation(self):
  with self.assertRaisesRegex(FileNotFoundError,'manually'):
   SampleDatabase.from_directory(self.root/'absent')
  shard=next(self.out.glob('*.zip'))
  content=shard.read_bytes();shard.write_bytes(content[:-1])
  with self.assertRaisesRegex(ValueError,'size mismatch'):
   SampleDatabase.from_directory(self.out)
  shard.unlink()
  with self.assertRaisesRegex(FileNotFoundError,'missing reference shards'):
   SampleDatabase.from_directory(self.out)
 def test_local_component_checksum(self):
  db=SampleDatabase.from_directory(self.out)
  row=db.index['samples']['A'];entry=row['components']['raw_tpm']
  with (self.out/row['shard']).open('r+b') as stream:
   stream.seek(entry['offset']);stream.write(b'broken')
  with self.assertRaisesRegex(ValueError,'checksum mismatch'):
   db.fetch(['A'],components=['raw_tpm'])
 def test_http_range_and_cache(self):
  db=self.db(True)
  def get(url,**kwargs):
   start,end=map(int,kwargs['headers']['Range'].split('=')[1].split('-'))
   content=(self.out/url.rsplit('/',1)[1]).read_bytes()
   return self.response(url,206,content[start:end+1],{'Content-Range':'bytes %d-%d/%d'%(start,end,len(content))})
  db.session.get=Mock(side_effect=get)
  x=db.fetch(['A'],components=['me','latent'])
  self.assertEqual(db.session.get.call_count,1)
  db.fetch(['A'],components=['me'])
  self.assertEqual(db.session.get.call_count,1)
  self.assertEqual(x['me'].iloc[0,0],0)
 def test_ignored_range_requires_opt_in(self):
  db=self.db(True)
  db.session.get=Mock(side_effect=lambda url,**kw:self.response(url,200,b''))
  with self.assertRaises(RangeNotSupportedError):db.fetch(['A'],components=['raw_tpm'])
  self.assertEqual(db.session.get.call_count,1)
 def test_full_shard_fallback(self):
  db=self.db(True)
  db.session.get=Mock(side_effect=lambda url,**kw:self.response(url,200,(self.out/url.rsplit('/',1)[1]).read_bytes()))
  with self.assertWarns(RuntimeWarning):
   x=db.fetch(['A'],components=['me'],allow_full_shard=True)
  self.assertEqual(x['me'].iloc[0,0],0)
  self.assertEqual(db.session.get.call_count,2)
 def test_bad_range_rejected(self):
  db=self.db(True)
  db.session.get=Mock(side_effect=lambda url,**kw:self.response(url,206,b'bad',{'Content-Range':'bytes 0-2/3'}))
  with self.assertRaises(ValueError):db.fetch(['A'],components=['me'])

if __name__=='__main__':unittest.main()
