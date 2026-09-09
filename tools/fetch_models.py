"""Materialize the two fixed public model packages; never execute model code.

Metadata is checked against the published Git blob identity; weights against
the published LFS SHA-256. Existing different files are never overwritten.
Use --local-weights frontier=PATH to copy a known local weight without a download.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def hashes(path):
    size = path.stat().st_size
    git = hashlib.sha1(f"blob {size}\0".encode())
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            git.update(block)
            sha.update(block)
    return size, git.hexdigest(), sha.hexdigest()


def check(path, entry):
    size, git, sha = hashes(path)
    if size != entry['size']:
        raise ValueError(f"Size mismatch: {entry['rfilename']}")
    if 'lfs' in entry:
        if sha != entry['lfs']['sha256']:
            raise ValueError(f"Weight hash mismatch: {entry['rfilename']}")
    elif git != entry['blobId']:
        raise ValueError(f"Public Git blob mismatch: {entry['rfilename']}")
    return {'bytes': size, 'sha256': sha}


def acquire(model, entry, local_weights):
    name = entry['rfilename']
    if Path(name).name != name:
        raise ValueError('Only package-root files are imported')
    directory = ROOT / 'models' / model['directory']
    directory.mkdir(exist_ok=True)
    target = directory / name
    if target.is_symlink():
        raise ValueError('Refusing symlink destination')
    if target.exists():
        return str(target.relative_to(ROOT)), check(target, entry)
    fd, temp_name = tempfile.mkstemp(prefix='.fetch-', dir=directory)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as output:
            local = local_weights.get(model['directory']) if name == 'model.safetensors' else None
            if local:
                source = Path(local)
                if source.is_symlink() or not source.is_file():
                    raise ValueError('Local weights must be a regular, non-symlink file')
                before = source.stat()
                with source.open('rb') as stream:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
                after = source.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise ValueError('Source changed during copy')
            else:
                url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{name}"
                request = urllib.request.Request(url, headers={'User-Agent': 'Qiushi-BabyLM-Release/1'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        record = check(temp, entry)
        os.chmod(temp, 0o644)
        # link is atomic and fails if another writer already created the target.
        os.link(temp, target)
        return str(target.relative_to(ROOT)), record
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', action='store_true', help='also materialize model weights')
    parser.add_argument('--local-weights', action='append', default=[], metavar='MODEL=FILE')
    args = parser.parse_args()
    local = dict(item.split('=', 1) for item in args.local_weights)
    if set(local) - {'frontier', 'principle_guided'}:
        parser.error('Unknown model in --local-weights')
    if local and not args.weights:
        parser.error('--local-weights requires --weights')
    models = json.loads((ROOT/'models/manifest.json').read_text())['models']
    jobs = [(m, f) for m in models for f in m['files']
            if args.weights or f['rfilename'] != 'model.safetensors']
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(acquire, m, f, local) for m, f in jobs]
        for future in concurrent.futures.as_completed(pending):
            path, record = future.result()
            records[path] = record
            print('Verified:', path, flush=True)
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    (build/'model_verification.json').write_text(json.dumps(records, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
