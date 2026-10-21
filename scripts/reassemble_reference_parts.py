"""Reassemble downloaded expression gzip parts, verifying every SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path


def local_file(root, name):
    if not name or Path(name).name != name or name in ('.','..') or '\\' in name:
        raise ValueError('Unsafe filename in split manifest')
    return root/name


def restore(root, manifest):
    for original in manifest['expression_files']:
        target=local_file(root,original['filename'])
        if target.exists():
            raise FileExistsError('Will not overwrite existing file: '+str(target))
        partial=target.with_name(target.name+'.reassembling')
        whole=hashlib.sha256();size=0
        try:
            with partial.open('xb') as out:
                for part in original['parts']:
                    path=local_file(root,part['filename']);h=hashlib.sha256();n=0
                    with path.open('rb') as inp:
                        for data in iter(lambda:inp.read(8*1024*1024),b''):
                            h.update(data);whole.update(data);out.write(data);n+=len(data);size+=len(data)
                    if n!=part['size'] or h.hexdigest()!=part['sha256']:
                        raise ValueError('Part checksum mismatch: '+path.name)
            if size!=original['size'] or whole.hexdigest()!=original['sha256']:
                raise ValueError('Reassembled file checksum mismatch: '+target.name)
            partial.replace(target)
            print('VERIFIED',target.name,flush=True)
        except Exception:
            # Keep invalid partial output out of the final filename.
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path('.'))
    args=p.parse_args()
    restore(args.directory,json.loads((args.directory/'SPLIT_DATA_MANIFEST.json').read_text()))
