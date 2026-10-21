"""Uniform sample shards with independently compressed, byte-addressable members."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import struct
import sys
import zipfile
from itertools import zip_longest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cancerxpress._legacy import axis_payload, axis_labels


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def add_member(archive, name, raw):
 data=gzip.compress(raw,compresslevel=3,mtime=0)
 info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED
 archive.writestr(info,data)
 # All member names are short ASCII, with no ZIP64 per-member extension.
 offset=info.header_offset+30+len(name.encode('ascii'))+len(info.extra)
 return {'offset':offset,'length':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def package(source, atlas_root, output, samples_per_shard=1000):
 if samples_per_shard<1:raise ValueError('Positive shard size required')
 output.mkdir(parents=True,exist_ok=False)
 with (source/'sample_metadata.tsv').open(newline='') as f:
  reader=csv.DictReader(f,delimiter='\t');id_column=reader.fieldnames[0]
  metadata={}
  for row in reader:
   sid=row.pop(id_column)
   if not sid or sid in metadata:raise ValueError('Missing or duplicate metadata ID')
   metadata[sid]=row
 meta=pd.read_csv(atlas_root/'samples.tsv',sep='\t',index_col=0)
 spec=json.loads((atlas_root/'atlas.json').read_text())
 if meta.index.has_duplicates or set(meta.index)!=set(metadata):raise ValueError('Atlas IDs mismatch')
 arrays=np.load(atlas_root/'features.npz',allow_pickle=False)
 latent=arrays['latent'];functional_axes=axis_payload(arrays);arrays.close()
 if len(latent)!=len(meta) or len(functional_axes)!=len(meta):raise ValueError('Atlas dimensions mismatch')
 if not np.isfinite(latent).all() or not np.isfinite(functional_axes).all():raise ValueError('Nonfinite atlas')
 lookup={sid:i for i,sid in enumerate(meta.index)}
 shard_count=(len(meta)+samples_per_shard-1)//samples_per_shard
 if not shard_count:raise ValueError('Empty reference database')
 quotient,remainder=divmod(len(meta),shard_count)
 sizes=[quotient+(i<remainder) for i in range(shard_count)]
 starts={int(start):i+1 for i,start in enumerate(np.cumsum([0]+sizes[:-1]))}
 records={};shards={};archive=None;seen=set();n=0
 try:
  with (source/'raw_tpm.tsv').open('rb') as raw, (source/'corrected_expression.tsv').open('rb') as corrected:
   rh=raw.readline().decode().rstrip('\r\n').split('\t')[1:]
   ch=corrected.readline().decode().rstrip('\r\n').split('\t')[1:]
   if len(set(rh))!=len(rh) or len(set(ch))!=len(ch):raise ValueError('Duplicate gene IDs')
   for r,c in zip_longest(raw,corrected):
    if r is None or c is None:raise ValueError('Expression row counts differ')
    rid,rv=r.rstrip(b'\r\n').split(b'\t',1);cid,cv=c.rstrip(b'\r\n').split(b'\t',1)
    sid=rid.decode()
    if rid!=cid:raise ValueError('Expression row orders differ at '+sid)
    if sid in seen or sid not in metadata:raise ValueError('Unexpected/duplicate sample '+sid)
    if rv.count(b'\t')+1!=len(rh) or cv.count(b'\t')+1!=len(ch):raise ValueError('Expression column count mismatch')
    if n in starts:
     if archive:
      archive.close();shards[shard]={'size':path.stat().st_size,'sha256':digest(path)}
      print('SHARD READY',shard,'samples',n,flush=True)
     shard='expression_samples_%03d.zip'%starts[n]
     path=output/shard;archive=zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED)
    prefix='sample_%06d'%n
    row={'shard':shard,'components':{}}
    row['components']['raw_tpm']=add_member(archive,prefix+'.raw_tpm.tsv.gz',rv)
    row['components']['corrected_expression']=add_member(archive,prefix+'.corrected_expression.tsv.gz',cv)
    i=lookup[sid]
    payload={'sample_id':sid,'latent':latent[i].tolist(),'functional_axes':functional_axes[i].tolist(),'metadata':metadata[sid]}
    row['components']['features']=add_member(archive,prefix+'.features.json.gz',json.dumps(payload,allow_nan=False).encode())
    records[sid]=row;seen.add(sid);n+=1
   if seen!=set(metadata):raise ValueError('Expression missing metadata samples')
   if archive:
    archive.close();archive=None;shards[shard]={'size':path.stat().st_size,'sha256':digest(path)}
  index={'format':'cancerxpress-sample-ranges-v2','n_samples':n,'samples_per_shard':samples_per_shard,
         'model_signature':spec['model_signature'],
         'columns':{'raw_tpm':rh,'corrected_expression':ch,'latent':spec['latent_columns'],'functional_axes':axis_labels(axis_payload(spec, 'functional_axis_columns'))},
         'shards':shards,'samples':records}
  # Validate all byte locations and hashes without extracting or unpickling data.
  for shard in shards:
   with (output/shard).open('rb') as f:
    for row in records.values():
     if row['shard']!=shard:continue
     for entry in row['components'].values():
      f.seek(entry['offset']);data=f.read(entry['length'])
      if hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Bad byte offset')
  compressed=gzip.compress(json.dumps(index,separators=(',',':')).encode(),compresslevel=6,mtime=0)
  (output/'reference_index.json.gz').write_bytes(compressed)
  (output/'SHA256SUMS.txt').write_text(''.join(v['sha256']+'  '+k+'\n' for k,v in shards.items())+
                                    hashlib.sha256(compressed).hexdigest()+'  reference_index.json.gz\n')
  (output/'COMPLETE').write_text('Uniform indexed sample shards validated; not uploaded or published.\n')
  print('READY',n,'samples',len(shards),'shards',output,flush=True)
 finally:
  if archive:archive.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
 p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--samples-per-shard',type=int,default=1000)
 a=p.parse_args();package(a.source,a.atlas,a.output,a.samples_per_shard)
