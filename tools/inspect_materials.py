"""Inspect archived research artifacts without importing experimental code."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common import ROOT


def contained_path(relative):
    path = ROOT / relative
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Artifact is outside the release: {relative}')
    return path


def load_manifest():
    return json.loads((ROOT / 'evidence/materials_manifest.json').read_text())


def verify_materials(public_only=False):
    manifest = load_manifest()
    seen = set()
    counts = Counter()
    for record in manifest['files']:
        relative = record['path']
        if relative in seen:
            raise ValueError(f'Duplicate artifact identity: {relative}')
        seen.add(relative)
        if public_only and (record.get('distribution') == 'local_only'
                            or Path(relative).suffix == '.safetensors'):
            continue
        path = contained_path(relative)
        if not path.is_file():
            raise ValueError(f'Missing artifact: {relative}')
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(chunk)
        if path.stat().st_size != record['bytes'] or checksum.hexdigest() != record['sha256']:
            raise ValueError(f'Artifact differs from its recorded version: {relative}')
        counts[record['kind']] += 1
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Verify every listed local file and content digest')
    check.add_argument('--public-only', action='store_true',
                       help='Check the source distribution, excluding weights and local-only assets')
    listing = sub.add_parser('list', help='List original implementations, notes and results')
    listing.add_argument('--kind', choices=['code', 'note', 'result', 'config', 'data_builder', 'figure', 'input', 'checkpoint'])
    listing.add_argument('--collection')
    listing.add_argument('--contains', default='')
    sub.add_parser('entrypoints', help='Show named study entrypoints and input requirements')
    args = parser.parse_args()
    if args.command == 'check':
        print(json.dumps(verify_materials(public_only=args.public_only), indent=2))
    elif args.command == 'entrypoints':
        data = json.loads((ROOT / 'experiments/entrypoints.json').read_text())
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for record in load_manifest()['files']:
            if args.kind and args.kind != record['kind']:
                continue
            if args.collection and args.collection != record['collection']:
                continue
            if args.contains.lower() not in record['path'].lower():
                continue
            print(f"{record['kind']}\t{record['scientific_status']}\t{record['path']}")


if __name__ == '__main__':
    main()
