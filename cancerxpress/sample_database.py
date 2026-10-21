"""Local indexed reference data, with optional explicit remote access."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urljoin, urlparse
import warnings
import numpy as np
import pandas as pd
import requests
from ._legacy import axis_payload, axis_labels


class RangeNotSupportedError(RuntimeError):
    """Host ignored a byte range; whole-shard fallback requires explicit opt-in."""


def _unzip(data, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:
        result = f.read(limit + 1)
    if len(result) > limit:
        raise ValueError('Compressed resource exceeds size limit')
    return result


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class SampleDatabase:
    """Local-first sample access; network access requires an explicit URL."""

    def __init__(self, index_path, base_url=None, cache_dir=None):
        self.index_path = Path(index_path).expanduser()
        if not self.index_path.is_file():
            raise FileNotFoundError('Place reference_index.json.gz and its ZIP shards in the '
                                    'reference directory manually. Missing: ' + str(self.index_path))
        self.index = json.loads(_unzip(self.index_path.read_bytes(), 256*1024*1024))
        if self.index.get('format') not in ('cancerxpress-sample-ranges-v1', 'cancerxpress-sample-ranges-v2'):
            raise ValueError('Unsupported sample index format')
        if self.index.get('n_samples') != len(self.index['samples']):
            raise ValueError('Sample index count mismatch')
        self.index['columns']['functional_axes'] = axis_labels(axis_payload(self.index['columns']))
        if base_url is not None and not base_url.startswith('https://'):
            raise ValueError('Reference downloads require HTTPS')
        self.base_url = base_url.rstrip('/')+'/' if base_url else None
        self.cache = Path(cache_dir or Path.home()/'.cache/cancerxpress/sample_data').expanduser()
        if self.base_url is not None:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        for name, spec in self.index['shards'].items():
            if not re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', name) or spec['size'] <= 0:
                raise ValueError('Invalid shard filename or size')
        for row in self.index['samples'].values():
            shard = self.index['shards'][row['shard']]
            for entry in row['components'].values():
                if (not isinstance(entry['offset'], int) or not isinstance(entry['length'], int)
                        or entry['offset'] < 0 or not 0 < entry['length'] <= 16*1024*1024
                        or entry['offset']+entry['length'] > shard['size']
                        or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])):
                    raise ValueError('Invalid component range')

    @classmethod
    def from_directory(cls, directory=None):
        """Open a manually installed release without downloading or caching data.

        Default: CANCERXPRESS_REFERENCE_DIR, then ~/.cache/cancerxpress/reference.
        All indexed ZIP shards must be present and have their expected sizes.
        Component SHA256 checks are performed when the selected data are read.
        """
        root = Path(directory if directory is not None else os.environ.get(
            'CANCERXPRESS_REFERENCE_DIR', '~/.cache/cancerxpress/reference')).expanduser()
        db = cls(root / 'reference_index.json.gz')
        missing = [name for name in db.index['shards'] if not (root / name).is_file()]
        if missing:
            raise FileNotFoundError('Copy the missing reference shards into ' + str(root)
                                    + ': ' + ', '.join(missing))
        for name, spec in db.index['shards'].items():
            if (root / name).stat().st_size != spec['size']:
                raise ValueError('Reference shard size mismatch; replace the incomplete file: '
                                 + str(root / name))
        return db

    @property
    def sample_ids(self):
        """Sample IDs in index order; no expression data are read."""
        return list(self.index['samples'])

    def to_atlas(self, sample_ids=None):
        """Read only features and metadata for use with query/adapt, not expression."""
        from .reference import ReferenceAtlas
        data = self.fetch(self.sample_ids if sample_ids is None else sample_ids,
                          components=['latent', 'functional_axes', 'metadata'])
        return ReferenceAtlas(data['latent'], data['functional_axes'], data['metadata'],
                              self.index['model_signature'])

    @classmethod
    def from_url(cls, index_url, sha256, cache_dir=None):
        """Download a small index with a publisher-provided SHA256, not the data."""
        if not index_url.startswith('https://') or not re.fullmatch('[0-9a-fA-F]{64}', sha256):
            raise ValueError('HTTPS index URL and SHA256 required')
        cache = Path(cache_dir or Path.home()/'.cache/cancerxpress/sample_data').expanduser()
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache/(sha256.lower()+'.index.json.gz')
        if not destination.exists() or _sha(destination.read_bytes()) != sha256.lower():
            with requests.get(index_url, stream=True, timeout=(20,120)) as response:
                response.raise_for_status()
                if not response.url.startswith('https://'):
                    raise ValueError('Non-HTTPS redirect')
                data = response.raw.read(64*1024*1024+1)
                if len(data)>64*1024*1024 or _sha(data)!=sha256.lower():
                    raise ValueError('Index checksum or size mismatch')
            cls._store(destination, data)
        parsed = urlparse(index_url)
        base = parsed._replace(path=parsed.path.rsplit('/',1)[0]+'/', query='', fragment='').geturl()
        return cls(destination, base_url=base, cache_dir=cache)

    @staticmethod
    def _store(path, data):
        with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
        tmp.replace(path)

    def _whole_shard(self, name):
        spec = self.index['shards'][name]
        dest = self.cache/(spec['sha256']+'.zip')
        def valid(path):
            if not path.exists() or path.stat().st_size != spec['size']: return False
            h=hashlib.sha256()
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
            return h.hexdigest()==spec['sha256']
        if valid(dest): return dest
        warnings.warn('Server ignored Range; explicitly downloading the entire shard: '+name, RuntimeWarning)
        with tempfile.NamedTemporaryFile(dir=str(self.cache),delete=False) as out:
            temp=Path(out.name)
            try:
                with self.session.get(urljoin(self.base_url,name), stream=True,
                                      headers={'Accept-Encoding':'identity'},timeout=(20,120)) as response:
                    response.raise_for_status()
                    if not response.url.startswith('https://'):raise ValueError('Non-HTTPS redirect')
                    size=0
                    for block in response.iter_content(1024*1024):
                        size+=len(block)
                        if size>spec['size']:raise ValueError('Shard exceeds expected size')
                        out.write(block)
                out.flush()
                if not valid(temp):raise ValueError('Shard checksum mismatch')
            except Exception:
                temp.unlink()
                raise
        temp.replace(dest)
        return dest

    def _component(self, row, name, allow_full_shard):
        entry=row['components'][name];shard=row['shard']
        target=self.cache/(entry['sha256']+'.component.gz')
        if self.base_url is not None and target.exists():
            data=target.read_bytes()
            if len(data)==entry['length'] and _sha(data)==entry['sha256']:
                return _unzip(data,16*1024*1024)
        if self.base_url is None:
            with (self.index_path.parent/shard).open('rb') as f:
                f.seek(entry['offset']);data=f.read(entry['length'])
        else:
            start=entry['offset'];end=start+entry['length']-1
            # Close an ignored Range response before consuming the full body.
            with self.session.get(urljoin(self.base_url,shard), stream=True,
                  headers={'Range':'bytes=%d-%d'%(start,end),'Accept-Encoding':'identity'},
                  timeout=(20,120)) as response:
                if not response.url.startswith('https://'):raise ValueError('Non-HTTPS redirect')
                full=response.status_code==200
                if full:
                    if not allow_full_shard:
                        raise RangeNotSupportedError('Server ignored Range. Set allow_full_shard=True to explicitly download '+shard)
                else:
                    response.raise_for_status()
                    expected='bytes %d-%d/%d'%(start,end,self.index['shards'][shard]['size'])
                    if (response.status_code!=206 or response.headers.get('Content-Range')!=expected
                            or response.headers.get('Content-Encoding','identity')!='identity'):
                        raise ValueError('Invalid byte-range response')
                    data=response.raw.read(entry['length']+1)
            if full:
                with self._whole_shard(shard).open('rb') as f:
                    f.seek(start);data=f.read(entry['length'])
        if len(data)!=entry['length'] or _sha(data)!=entry['sha256']:
            raise ValueError('Sample component checksum mismatch')
        if self.base_url is not None:
            self._store(target,data)
        return _unzip(data,16*1024*1024)

    def fetch(self, sample_ids, components=('raw_tpm','corrected_expression','latent','functional_axes','metadata'),
              allow_full_shard=False):
        """Return sample-indexed DataFrames; download only requested components."""
        ids=[sample_ids] if isinstance(sample_ids,str) else list(sample_ids)
        components=[components] if isinstance(components,str) else list(components)
        allowed={'raw_tpm','corrected_expression','latent','functional_axes','metadata'}
        if not ids or len(set(ids))!=len(ids):raise ValueError('Supply unique nonempty sample IDs')
        if not components or len(set(components))!=len(components) or set(components)-allowed:
            raise ValueError('Invalid components')
        missing=set(ids)-set(self.index['samples'])
        if missing:raise KeyError('Sample IDs absent from index: '+str(sorted(missing)))
        values={name:[] for name in components}
        for sid in ids:
            row=self.index['samples'][sid]
            for name in set(components)&{'raw_tpm','corrected_expression'}:
                text=self._component(row,name,allow_full_shard).decode()
                arr=np.array(text.split('\t'),dtype=float)
                if len(arr)!=len(self.index['columns'][name]):raise ValueError('Gene count mismatch')
                values[name].append(arr)
            if set(components)&{'latent','functional_axes','metadata'}:
                features=json.loads(self._component(row,'features',allow_full_shard))
                if features['sample_id']!=sid:raise ValueError('Sample ID mismatch')
                for name in set(components)&{'latent','functional_axes','metadata'}:
                    values[name].append(axis_payload(features) if name == 'functional_axes' else features[name])
        return {name:pd.DataFrame(data,index=pd.Index(ids,name='sample_id'),
                    columns=None if name=='metadata' else self.index['columns'][name])
                for name,data in values.items()}
