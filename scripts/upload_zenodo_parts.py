"""Split selected expression files and upload verified parts to a draft only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
from upload_zenodo_reference import DraftClient, hashes, same_file


def split_file(src, dest, part_bytes, expected):
    if part_bytes<=0:raise ValueError('Positive part size required')
    whole=hashlib.sha256();total=0;parts=[]
    with src.open('rb') as inp:
        number=0
        while True:
            first=inp.read(min(part_bytes,8*1024*1024))
            if not first:break
            number+=1
            name=src.name+'.part%04d'%number
            path=dest/name;tmp=dest/(name+'.partial')
            sha=hashlib.sha256();md5=hashlib.md5();n=0
            with tmp.open('wb') as out:
                block=first
                while block:
                    out.write(block);sha.update(block);md5.update(block);whole.update(block)
                    n+=len(block);total+=len(block)
                    if n==part_bytes:break
                    block=inp.read(min(8*1024*1024,part_bytes-n))
            if path.exists():
                if path.stat().st_size!=n or hashes(path)[0]!=sha.hexdigest():
                    raise ValueError('Existing local part differs: '+name)
                tmp.unlink()
            else:tmp.replace(path)
            parts.append({'filename':name,'size':n,'sha256':sha.hexdigest(),'md5':md5.hexdigest()})
            print('PART READY',name,n,flush=True)
    if whole.hexdigest()!=expected['sha256'] or total!=expected['size']:
        raise ValueError('Source does not match release manifest: '+src.name)
    return {'filename':src.name,'size':total,'sha256':whole.hexdigest(),'parts':parts}


def write_json(path,data):
    tmp=path.with_suffix('.partial');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--state-dir',type=Path,required=True)
    p.add_argument('--atlas',type=Path,required=True)
    p.add_argument('--draft-id',type=int,required=True)
    p.add_argument('--token-file',type=Path,default=Path.home()/'.config/zenodo/token')
    p.add_argument('--part-mib',type=int,default=256)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--minimal',action='store_true',help='Upload only expression parts, frozen atlas and metadata; retain tools and checksums locally')
    a=p.parse_args();a.state_dir.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.state_dir/'upload.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mode=a.token_file.stat()
    if not stat.S_ISREG(mode.st_mode) or mode.st_uid!=os.getuid() or mode.st_mode&0o077:
        raise ValueError('Token must be in a private user-owned regular file')
    token=a.token_file.read_text().strip()
    if not token:raise ValueError('Empty token')
    client=DraftClient(token,a.draft_id,transport='curl')
    draft=client.draft();print('DRAFT CONFIRMED',a.draft_id,flush=True)
    root=a.directory
    if not (root/'COMPLETE').is_file():raise ValueError('Reference release incomplete')
    specs={e['path']:e for e in json.loads((root/'manifest.json').read_text())['files']}
    names=['raw_tpm.tsv.gz','corrected_expression.tsv.gz'];part_bytes=a.part_mib*1024*1024
    if part_bytes<=0:raise ValueError('Invalid part size')
    planned=set()
    for name in names:
        planned.update(name+'.part%04d'%i for i in range(1,(specs[name]['size']+part_bytes-1)//part_bytes+1))
    planned.update(['reference_atlas.tar.gz','sample_metadata.tsv.gz'] if a.minimal else
                   ['reference_atlas.tar.gz','SPLIT_DATA_MANIFEST.json','SPLIT_DATA_README.txt','reassemble_reference_parts.py','SPLIT_DATA_SHA256SUMS.txt'])
    remote={e['filename']:e for e in draft.get('files',[])}
    if len(planned|set(remote))>100:raise ValueError('Would exceed Zenodo 100-file limit; increase part size')
    print('PLANNED',len(planned),'files; total including existing',len(planned|set(remote)),flush=True)
    originals=[split_file(root/name,a.state_dir,part_bytes,specs[name]) for name in names]
    sha,md5=hashes(a.atlas)
    # This atlas must match the previously verified upload; no replacement is permitted.
    atlas={'filename':a.atlas.name,'size':a.atlas.stat().st_size,'sha256':sha,'md5':md5}
    if atlas['filename'] in remote and not same_file(remote[atlas['filename']],atlas['size'],md5):
        raise ValueError('Atlas differs from existing remote file')
    manifest={'version':'1.0.0','draft_id':a.draft_id,'part_bytes':part_bytes,
              'expression_files':originals,'reference_atlas':atlas}
    write_json(a.state_dir/'SPLIT_DATA_MANIFEST.json',manifest)
    shutil.copy2(Path(__file__).with_name('reassemble_reference_parts.py'),a.state_dir/'reassemble_reference_parts.py')
    (a.state_dir/'SPLIT_DATA_README.txt').write_text(
        'CancerXpress selected reference data: split-file delivery\n\n'
        'This document supersedes earlier whole-file upload instructions.\n'
        'Included: raw TPM, batch-corrected expression, frozen-model latent and ME.\n'
        'Raw counts and historical latent are intentionally NOT included.\n'
        'Expression .gz files were split at binary byte boundaries (256 MiB per part, except tails).\n'
        'Parts are not independent gzip archives and cannot be decompressed individually.\n'
        'Download all .partNNNN files, SPLIT_DATA_MANIFEST.json and reassemble_reference_parts.py\n'
        'into one directory. Run: python reassemble_reference_parts.py --directory .\n'
        'Python 3 standard library only. Each part and the final compressed file are verified by SHA256.\n'
        'Do not overwrite existing original files. Allow approximately 18 GB additional disk for assembly.\n'
        'Then read raw_tpm.tsv.gz / corrected_expression.tsv.gz with gzip-aware software.\n'
        'Extract reference_atlas.tar.gz to obtain features.npz (latent and ME), samples.tsv, atlas.json.\n'
        'Original row IDs and column IDs are unchanged. Missing labels are not imputed.\n'
        'Earlier manifest.json/COMPLETE describe the full LOCAL bundle, not remote upload completeness.\n'
        'Use SPLIT_DATA_MANIFEST.json for assembly and SPLIT_DATA_SHA256SUMS.txt for transfer verification.\n'
        'Previously uploaded documentation is retained; no remote files were deleted.\n'
        'This task uploads to a draft only and never publishes.\n')
    transfers=[p for original in originals for p in original['parts']]+[atlas]
    for name in ['SPLIT_DATA_MANIFEST.json','SPLIT_DATA_README.txt','reassemble_reference_parts.py']:
        path=a.state_dir/name;sh,md=hashes(path)
        transfers.append({'filename':name,'size':path.stat().st_size,'sha256':sh,'md5':md})
    checks=a.state_dir/'SPLIT_DATA_SHA256SUMS.txt'
    checks.write_text(''.join(e['sha256']+'  '+e['filename']+'\n' for e in transfers))
    sh,md=hashes(checks);transfers.append({'filename':checks.name,'size':checks.stat().st_size,'sha256':sh,'md5':md})
    if a.minimal:
        metadata=root/'sample_metadata.tsv.gz'
        sh,md=hashes(metadata)
        if sh!=specs[metadata.name]['sha256'] or metadata.stat().st_size!=specs[metadata.name]['size']:
            raise ValueError('Metadata checksum mismatch')
        transfers=[p for original in originals for p in original['parts']]+[atlas,
                   {'filename':metadata.name,'size':metadata.stat().st_size,'sha256':sh,'md5':md}]
    write_json(a.state_dir/'upload_manifest.json',{'draft_id':a.draft_id,'files':transfers})
    if sum(e['size'] for e in transfers)+sum(int(e.get('filesize',e.get('size',0))) for n,e in remote.items() if n not in planned)>50_000_000_000:
        raise ValueError('Would exceed 50GB record limit')
    pending=sorted(transfers,key=lambda e:(e['filename'].endswith(tuple(['.json','.txt','.py'])) is False,e['size']))
    for round_number in range(1,a.rounds+1):
        failed=[]
        print('UPLOAD ROUND',round_number,'remaining',len(pending),flush=True)
        for entry in pending:
            path=(a.atlas if entry['filename']==a.atlas.name else root/entry['filename']
                  if entry['filename']=='sample_metadata.tsv.gz' else a.state_dir/entry['filename'])
            try:client.upload(path,entry['md5'])
            except RuntimeError as exc:
                print('DEFERRED',entry['filename'],str(exc),flush=True);failed.append(entry)
            write_json(a.state_dir/'progress.json',{'round':round_number,'last_file':entry['filename'],
                       'deferred':[e['filename'] for e in failed],'published':False})
        pending=failed
        if not pending:break
        if round_number<a.rounds:time.sleep(300)
    remote={e['filename']:e for e in client.draft().get('files',[])}
    missing=[e['filename'] for e in transfers if not same_file(remote.get(e['filename'],{}),e['size'],e['md5'])]
    write_json(a.state_dir/'verification.json',{'verified':len(transfers)-len(missing),'total':len(transfers),'pending':missing,'published':False})
    if missing:raise RuntimeError('Some files remain incomplete; see verification.json')
    write_json(a.state_dir/'UPLOAD_COMPLETE.json',{'draft_id':a.draft_id,'verified_files':len(transfers),'published':False})
    print('UPLOAD COMPLETE; NOT PUBLISHED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAILED',type(exc).__name__,str(exc) if isinstance(exc,(ValueError,RuntimeError,FileNotFoundError)) else '',flush=True)
        raise SystemExit(1)
