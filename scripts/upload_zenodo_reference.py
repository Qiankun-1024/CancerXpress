"""Wait for a verified reference release and upload to an existing Zenodo draft.

Never publishes. Credentials are read from a private file, never logged.
Retries restart whole files; this is not byte-range resumable uploading.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import tarfile
import tempfile
import time
from urllib.parse import quote, urlparse
import requests


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(chunk); md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name or not rel.parts:
        raise ValueError('Unsafe manifest path')
    file = root.joinpath(*rel.parts)
    if file.is_symlink() or root.resolve() not in file.resolve().parents or not file.is_file():
        raise ValueError('Missing or unsafe release file: ' + name)
    return file


def checksum(entry):
    value = str(entry.get('checksum', ''))
    return value[4:] if value.startswith('md5:') else value


def same_file(entry, size, md5):
    return int(entry.get('filesize', entry.get('size', -1))) == size and checksum(entry) == md5


class DraftClient:
    def __init__(self, token, draft, transport='requests'):
        self.session = requests.Session()
        self.session.headers['Authorization'] = 'Bearer ' + token
        self.url = 'https://zenodo.org/api/deposit/depositions/' + str(draft)
        self.transport = transport

    def put_file(self, url, path):
        if self.transport == 'requests':
            with path.open('rb') as stream:
                response = self.session.put(url, data=stream, timeout=(120, 900),
                    allow_redirects=False, headers={'Content-Type': 'application/octet-stream'})
            return response.status_code
        # Pass the secret through stdin, never argv, URLs or progress logs.
        header = self.session.headers['Authorization']
        if '\n' in header or '\r' in header:
            raise ValueError('Invalid token format')
        config = 'header = ' + json.dumps('Authorization: ' + header) + '\n'
        config += 'header = "Content-Type: application/octet-stream"\n'
        with tempfile.TemporaryDirectory(prefix='zenodo-response-') as tmp:
            args = ['curl', '--config', '-', '--http1.1', '--silent', '--show-error',
                    '--connect-timeout', '60', '--speed-limit', '1024', '--speed-time', '600',
                    '--keepalive-time', '60', '--upload-file', str(path),
                    '--output', str(Path(tmp)/'response.json'),
                    '--write-out', '%{http_code} %{size_upload} %{time_total}', url]
            run = subprocess.run(args, input=config, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            parts = run.stdout.strip().split()
            print('TRANSFER', path.name, 'curl_exit=', run.returncode,
                  'http/bytes/seconds=', ' '.join(parts), flush=True)
            if run.returncode:
                return 503
            return int(parts[0])

    def draft(self):
        for attempt in range(6):
            try:
                response = self.session.get(self.url, timeout=(20, 120), allow_redirects=False)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('submitted') or data.get('state') not in ('unsubmitted',):
                        raise ValueError('Destination is not an unpublished draft')
                    return data
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise ValueError('Draft request HTTP ' + str(response.status_code))
            except requests.RequestException:
                pass
            if attempt < 5: time.sleep(min(30 * 2**attempt, 300))
        raise RuntimeError('Could not retrieve draft after retries')

    def upload(self, path, md5):
        size = path.stat().st_size
        for attempt in range(6):
            data = self.draft()
            existing = {e['filename']: e for e in data.get('files', [])}
            if path.name in existing:
                if same_file(existing[path.name], size, md5):
                    print('VERIFIED', path.name, '(already uploaded)', flush=True)
                    return
                raise ValueError('Remote name conflict; will not overwrite: ' + path.name)
            bucket = data.get('links', {}).get('bucket', '')
            parsed = urlparse(bucket)
            if (parsed.scheme != 'https' or parsed.netloc != 'zenodo.org'
                    or not parsed.path.startswith('/api/files/') or parsed.query or parsed.fragment):
                raise ValueError('Untrusted or missing draft bucket URL')
            print('UPLOAD', path.name, size, 'bytes; attempt', attempt + 1, flush=True)
            try:
                status = self.put_file(bucket.rstrip('/') + '/' + quote(path.name, safe=''), path)
                if status not in (200, 201, 429, 500, 502, 503, 504):
                    raise ValueError('Upload HTTP ' + str(status) + ': ' + path.name)
            except requests.RequestException as exc:
                print('Transient upload connection failure:', path.name, type(exc).__name__, flush=True)
            # After success or an uncertain response, check server-side bytes.
            for _ in range(3):
                current = {e['filename']: e for e in self.draft().get('files', [])}
                if path.name in current:
                    if not same_file(current[path.name], size, md5):
                        raise ValueError('Remote checksum mismatch: ' + path.name)
                    print('VERIFIED', path.name, flush=True)
                    return
                time.sleep(10)
            time.sleep(min(30 * 2**attempt, 300))
        raise RuntimeError('Upload retries exhausted: ' + path.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--draft-id', type=int, required=True)
    p.add_argument('--token-file', type=Path, default=Path.home()/'.config/zenodo/token')
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--producer-pid', type=int)
    p.add_argument('--wait-hours', type=float, default=48)
    p.add_argument('--data-only', action='store_true', help='Only TPM, corrected expression and frozen latent/ME atlas, plus selection documentation')
    p.add_argument('--transport', choices=['requests','curl'], default='requests')
    a = p.parse_args()
    a.state_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (a.state_dir/'upload.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    info = a.token_file.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
        raise ValueError('Token file must be owned by this user with private permissions')
    token = a.token_file.read_text().strip()
    if not token: raise ValueError('Empty token')
    client = DraftClient(token, a.draft_id, a.transport)
    client.draft()
    print('Confirmed unpublished draft', a.draft_id, flush=True)
    started = time.monotonic()
    while not (a.directory/'COMPLETE').is_file():
        if time.monotonic()-started > a.wait_hours*3600:
            raise TimeoutError('Release did not complete before timeout')
        if a.producer_pid:
            try: os.kill(a.producer_pid, 0)
            except ProcessLookupError: raise RuntimeError('Packaging stopped without COMPLETE; upload aborted')
        print('WAIT: reference release has not completed', flush=True)
        time.sleep(60)
    root = a.directory
    manifest = json.loads((root/'manifest.json').read_text())
    specs = manifest['files']
    named = [s['path'] for s in specs]
    if len(named) != len(set(named)) or not named:
        raise ValueError('Empty or duplicate manifest entries')
    local_hashes = {}
    for entry in specs:
        if a.data_only and entry['path'] not in ('raw_tpm.tsv.gz','corrected_expression.tsv.gz') and not entry['path'].startswith('reference_atlas/'):
            continue
        path = safe_path(root, entry['path'])
        if path.stat().st_size != entry['size']: raise ValueError('Size mismatch: '+entry['path'])
        sha, md5 = hashes(path)
        if sha != entry['sha256']: raise ValueError('SHA256 mismatch: '+entry['path'])
        local_hashes[entry['path']] = (sha, md5)
        print('LOCAL VERIFIED', entry['path'], flush=True)
    allowed = set(named) | {'manifest.json', 'SHA256SUMS.txt', 'COMPLETE'}
    actual = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
    if actual != allowed: raise ValueError('Unexpected or missing release files')
    for required in ('reference_atlas/atlas.json','reference_atlas/features.npz','reference_atlas/samples.tsv'):
        if required not in named: raise ValueError('Incomplete reference atlas')
    # Zenodo file names are flat; keep the atlas directory structure in a tarball.
    archive = a.state_dir/'reference_atlas.tar.gz'
    partial = archive.with_suffix('.partial')
    with partial.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for relative in sorted(n for n in named if n.startswith('reference_atlas/')):
                    path = root/relative
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
    partial.replace(archive)
    guide = a.state_dir/'UPLOAD_README.txt'
    guide.write_text('CancerXpress reference database v1.0.0\n'
        'Download all files to one directory. Extract reference_atlas.tar.gz there;\n'
        'this creates reference_atlas/{samples.tsv,features.npz,atlas.json}.\n'
        'Other gzip files can be read directly or decompressed individually.\n'
        'manifest.json checksums refer to the original layout after atlas extraction.\n'
        'ZENODO_SHA256SUMS.txt checksums refer to files as uploaded.\n'
        'Missing cancer labels remain missing; see README.txt.\n'
        'This upload is a draft; publication requires separate manual review.\n')
    files = [root/n for n in named if '/' not in n]
    files += [root/'manifest.json',root/'SHA256SUMS.txt',root/'COMPLETE',archive,guide]
    if a.data_only:
        guide = a.state_dir/'SELECTED_DATA_README.txt'
        guide.write_text('Selected CancerXpress reference release\n'
            'Current upload scope supersedes the earlier whole-database upload plan.\n'
            'Required data: raw_tpm.tsv.gz, corrected_expression.tsv.gz, reference_atlas.tar.gz.\n'
            'The atlas contains frozen-model latent and ME in features.npz, aligned sample metadata\n'
            'in samples.tsv and model/column definitions in atlas.json. Extract it before loading.\n'
            'Raw counts and historical latent are deliberately excluded, not missing uploads.\n'
            'Existing manifest.json, COMPLETE and SHA256SUMS.txt describe the FULL LOCAL bundle,\n'
            'not completion of this remote upload. Use SELECTED_DATA_SHA256SUMS.txt for this subset.\n'
            'Already uploaded labels and documentation are retained; no existing files are deleted.\n')
        files = [root/'raw_tpm.tsv.gz',root/'corrected_expression.tsv.gz',archive,guide]
    transfer = []
    for path in files:
        sha, md5 = local_hashes.get(path.name, (None, None))
        if sha is None: sha, md5 = hashes(path)
        transfer.append({'filename':path.name,'size':path.stat().st_size,'sha256':sha,'md5':md5})
    checks = a.state_dir/('SELECTED_DATA_SHA256SUMS.txt' if a.data_only else 'ZENODO_SHA256SUMS.txt')
    checks.write_text(''.join(e['sha256']+'  '+e['filename']+'\n' for e in transfer))
    sha, md5 = hashes(checks)
    transfer.append({'filename':checks.name,'size':checks.stat().st_size,'sha256':sha,'md5':md5})
    files.append(checks)
    (a.state_dir/'upload_manifest.json').write_text(json.dumps({'draft_id':a.draft_id,'files':transfer},indent=2))
    paths = {f.name:f for f in files}
    for entry in sorted(transfer, key=lambda e:e['size']):
        client.upload(paths[entry['filename']],entry['md5'])
    remote = {e['filename']:e for e in client.draft().get('files',[])}
    if any(not same_file(remote.get(e['filename'],{}),e['size'],e['md5']) for e in transfer):
        raise ValueError('Final remote verification failed')
    (a.state_dir/'UPLOAD_COMPLETE.json').write_text(json.dumps({'draft_id':a.draft_id,'published':False,
        'verified_files':len(transfer),'data_only':a.data_only,
        'draft_url':'https://zenodo.org/uploads/'+str(a.draft_id)},indent=2))
    print('UPLOAD COMPLETE; draft only, NOT PUBLISHED', a.draft_id, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Do not expose request headers, tokens or remote error bodies in tracebacks.
        if isinstance(exc, requests.RequestException):
            print('FAILED:', type(exc).__name__, flush=True)
        else:
            print('FAILED:', type(exc).__name__, str(exc), flush=True)
        raise SystemExit(1)
