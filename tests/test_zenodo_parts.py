import hashlib
import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from upload_zenodo_parts import split_file
from reassemble_reference_parts import restore, local_file

class SplitTests(unittest.TestCase):
    def test_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'data.gz';data=b'0123456789'*10+b'x';src.write_bytes(data)
            out=root/'parts';out.mkdir()
            expected={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            record=split_file(src,out,16,expected)
            again=split_file(src,out,16,expected)
            self.assertEqual(record,again)
            self.assertEqual(len(record['parts']),7)
            restore(out,{'expression_files':[record]})
            self.assertEqual((out/'data.gz').read_bytes(),data)
            with self.assertRaises(FileExistsError):restore(out,{'expression_files':[record]})

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'data.gz';src.write_bytes(b'123456')
            out=root/'parts';out.mkdir()
            record=split_file(src,out,4,{'size':6,'sha256':hashlib.sha256(b'123456').hexdigest()})
            (out/record['parts'][0]['filename']).write_bytes(b'xxxx')
            with self.assertRaises(ValueError):restore(out,{'expression_files':[record]})
            self.assertFalse((out/'data.gz').exists())
            with self.assertRaises(ValueError):local_file(out,'../bad')

if __name__=='__main__':unittest.main()
