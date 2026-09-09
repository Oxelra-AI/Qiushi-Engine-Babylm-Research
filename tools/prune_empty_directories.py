"""Remove empty archival directories without deleting any file or symlink."""
from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path

from common import ROOT


def empty_directories(root):
    empty = set()
    for area in ('experiments/archive', 'dist'):
        start = root / area
        if start.is_symlink() or not start.exists() or not start.resolve().is_relative_to(root.resolve()):
            continue
        for folder, directories, files in os.walk(start, topdown=False, followlinks=False):
            path = Path(folder)
            if any(part in {'.git', 'build', '__pycache__'} for part in path.relative_to(root).parts):
                continue
            if any(path.joinpath(name).is_symlink() for name in directories + files):
                continue
            if not files and all(path / name in empty for name in directories):
                if path != root / 'experiments/archive':
                    empty.add(path)
    return sorted(empty, key=lambda p: (-len(p.parts), str(p)))


def prune(root, apply=False):
    candidates = empty_directories(root)
    removed = []
    if apply:
        for path in candidates:
            if path.is_symlink():
                continue
            try:
                path.rmdir()
                removed.append(str(path.relative_to(root)))
            except OSError as error:
                if error.errno not in (errno.ENOTEMPTY, errno.ENOENT, errno.EEXIST):
                    raise
    return {'empty_directories': len(candidates), 'removed': len(removed),
            'paths': [str(path.relative_to(root)) for path in candidates]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Use rmdir on empty paths only')
    parser.add_argument('--list', action='store_true', help='Include each path in the result')
    args = parser.parse_args()
    result = prune(ROOT, apply=args.apply)
    if not args.list:
        result.pop('paths')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
