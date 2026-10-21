import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from cancerxpress.downloads import download_models, verify_resources
from cancerxpress.resources import ModelPaths, ResourcePaths, resource_paths_from_directory


class Response(io.BytesIO):
    def geturl(self):
        return 'https://example.org/bundle.tar.gz'


def archive_bytes(unsafe=False):
    data = b'test asset'
    manifest = json.dumps({'files': [{'path': 'assets/file.txt', 'size': len(data),
                                     'sha256': hashlib.sha256(data).hexdigest()}]}).encode()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as tar:
        for name, value in [('assets/file.txt', data), ('resource_manifest.json', manifest)]:
            info = tarfile.TarInfo('../escape' if unsafe and name == 'assets/file.txt' else name)
            info.size = len(value)
            tar.addfile(info, io.BytesIO(value))
    return out.getvalue()


class ResourceTests(unittest.TestCase):
    def test_dynamic_environment_paths(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'CANCERXPRESS_HOME': tmp}):
            self.assertEqual(ResourcePaths().pixel_coords, Path(tmp) / 'assets/tsne_sorted_pixels_coords.csv')
            self.assertEqual(ModelPaths().functional_axes, Path(tmp) / 'models/finetuned/functional_axes')
            self.assertEqual(resource_paths_from_directory(tmp).max_norm_tpm, Path(tmp) / 'assets/max_norm_tpm.tsv')

    def test_install_and_tamper(self):
        data = archive_bytes()
        with tempfile.TemporaryDirectory() as tmp, patch('cancerxpress.downloads.urlopen', return_value=Response(data)):
            target = Path(tmp) / 'installed'
            download_models('https://example.org/model', hashlib.sha256(data).hexdigest(), target)
            self.assertEqual(len(verify_resources(target)['files']), 1)
            with self.assertRaises(FileExistsError):
                download_models('https://example.org/model', hashlib.sha256(data).hexdigest(), target)
            (target / 'assets/file.txt').write_text('bad')
            with self.assertRaises(ValueError):
                verify_resources(target)

    def test_checksum_and_traversal(self):
        for unsafe in (False, True):
            data = archive_bytes(unsafe)
            with tempfile.TemporaryDirectory() as tmp, patch('cancerxpress.downloads.urlopen', return_value=Response(data)):
                target = Path(tmp) / 'installed'
                with self.assertRaises(ValueError):
                    download_models('https://example.org/model',
                                    hashlib.sha256(data).hexdigest() if unsafe else '0' * 64, target)
                self.assertFalse(target.exists())
                self.assertFalse((Path(tmp) / 'escape').exists())


if __name__ == '__main__':
    unittest.main()
