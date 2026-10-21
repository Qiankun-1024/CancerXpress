"""Download and verify trusted resource bundles without loading model files."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.request import urlopen

from .resources import default_resource_root


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise ValueError('Unsafe resource path: ' + name)
    return path


def verify_resources(resource_dir=None):
    """Verify every file and size in resource_manifest.json; raise on mismatch."""
    root = Path(resource_dir or default_resource_root()).expanduser().resolve()
    manifest = json.loads((root / 'resource_manifest.json').read_text())
    seen = set()
    for entry in manifest['files']:
        relative = _relative(entry['path'])
        if str(relative) in seen:
            raise ValueError('Duplicate manifest path')
        seen.add(str(relative))
        path = root.joinpath(*relative.parts)
        if root not in path.resolve().parents or path.is_symlink():
            raise ValueError('Resource escapes root')
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Missing or wrong-size resource: ' + str(relative))
        if sha256_file(path) != entry['sha256']:
            raise ValueError('SHA-256 mismatch: ' + str(relative))
    if not seen:
        raise ValueError('Empty resource manifest')
    return manifest


def download_models(url, sha256, resource_dir=None):
    """Fetch a trusted HTTPS .tar.gz bundle, verify, and atomically install.

    Obtain URL and SHA-256 from the published release, not an untrusted archive.
    Existing destinations are never overwritten. Pickled encoders must only
    be loaded from a trusted publisher. No official URL is assumed yet.
    """
    if not url.startswith('https://'):
        raise ValueError('A trusted HTTPS download URL is required')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', sha256):
        raise ValueError('Expected a 64-character SHA-256 checksum')
    destination = Path(resource_dir or default_resource_root()).expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Resource directory already exists; verify it or choose a new directory')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cancerxpress-', dir=str(destination.parent)) as tmp:
        tmp = Path(tmp)
        archive = tmp / 'download.tar.gz'
        with urlopen(url, timeout=120) as response, archive.open('wb') as out:
            if not response.geturl().startswith('https://'):
                raise ValueError('Refusing non-HTTPS redirect')
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if sha256_file(archive) != sha256.lower():
            raise ValueError('Archive SHA-256 mismatch')
        stage = tmp / 'resources'
        stage.mkdir()
        with tarfile.open(str(archive), 'r:gz') as tar:
            seen = set()
            for member in tar:
                relative = _relative(member.name)
                if member.name in seen or not member.isfile():
                    raise ValueError('Only unique regular files are permitted')
                seen.add(member.name)
                target = stage.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('xb') as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
        manifest = verify_resources(stage)
        expected = {x['path'] for x in manifest['files']} | {'resource_manifest.json'}
        if seen != expected:
            raise ValueError('Archive has files not declared in manifest')
        if destination.exists():
            raise FileExistsError(str(destination))
        stage.rename(destination)
    return destination
