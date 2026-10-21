import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('reference_packaging', Path(__file__).resolve().parents[1] / 'scripts/package_reference_database.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PackagingTests(unittest.TestCase):
    def test_verified_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'source.tsv'; dst = Path(tmp) / 'data.gz'
            src.write_bytes(b'gene\tvalue\nA\t1\n')
            first = module.compressed_copy(src, dst)
            stamp = dst.stat().st_mtime_ns
            second = module.compressed_copy(src, dst)
            self.assertEqual(first, second)
            self.assertEqual(stamp, dst.stat().st_mtime_ns)
            src.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                module.compressed_copy(src, dst)
            with gzip.open(dst, 'rb') as stream:
                self.assertEqual(stream.read(), b'gene\tvalue\nA\t1\n')

    def test_corrupt_gzip_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'source'; dst = Path(tmp) / 'data.gz'
            src.write_bytes(b'data'); dst.write_bytes(b'not gzip')
            with self.assertRaises(OSError):
                module.compressed_copy(src, dst)


if __name__ == '__main__':
    unittest.main()
