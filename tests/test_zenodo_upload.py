import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('uploader',Path(__file__).resolve().parents[1]/'scripts/upload_zenodo_reference.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)

class UploadTests(unittest.TestCase):
    def test_curl_secret_not_in_arguments(self):
        client=u.DraftClient('test-secret',123,transport='curl')
        with patch.object(u.subprocess,'run',return_value=Mock(returncode=0,stdout='201 7 1.2')) as run:
            status=client.put_file('https://zenodo.org/api/files/test/data',Path('data'))
        self.assertEqual(status,201)
        args=run.call_args[0][0]
        self.assertNotIn('test-secret',' '.join(args))
        self.assertIn('Authorization: Bearer test-secret',run.call_args[1]['input'])

    def test_verified_skip_without_put(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';p.write_bytes(b'content')
            md5=u.hashes(p)[1]
            c=u.DraftClient('fake-test-token',123)
            c.draft=Mock(return_value={'files':[{'filename':'data','filesize':7,'checksum':'md5:'+md5}]})
            c.session.put=Mock()
            c.upload(p,md5)
            c.session.put.assert_not_called()

    def test_conflict_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';p.write_bytes(b'content')
            c=u.DraftClient('fake-test-token',123)
            c.draft=Mock(return_value={'files':[{'filename':'data','filesize':7,'checksum':'md5:wrong'}]})
            c.session.put=Mock()
            with self.assertRaises(ValueError):c.upload(p,u.hashes(p)[1])
            c.session.put.assert_not_called()

    def test_published_record_rejected(self):
        c=u.DraftClient('fake-test-token',123)
        r=Mock(status_code=200);r.json.return_value={'submitted':True,'state':'done'}
        c.session.get=Mock(return_value=r)
        with self.assertRaises(ValueError):c.draft()

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ('../secret','/secret','a\\secret'):
                with self.assertRaises(ValueError):u.safe_path(Path(tmp),bad)

    def test_success_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';p.write_bytes(b'content');md5=u.hashes(p)[1]
            c=u.DraftClient('fake-test-token',123)
            c.draft=Mock(side_effect=[{'files':[],'links':{'bucket':'https://zenodo.org/api/files/test'}},
                {'files':[{'filename':'data','filesize':7,'checksum':'md5:'+md5}]}])
            c.session.put=Mock(return_value=Mock(status_code=201))
            c.upload(p,md5)
            self.assertEqual(c.session.put.call_count,1)

if __name__=='__main__': unittest.main()
