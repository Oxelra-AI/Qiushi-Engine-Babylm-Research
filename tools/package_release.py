"""Create a local review archive from the public file allowlist; never upload."""
import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from common import ROOT, public_files


def require_release_review(root):
    path = root/'evidence/release_review.json'
    if not path.is_file():
        raise SystemExit('Release review is missing; no distribution archive was created.')
    review = json.loads(path.read_text(encoding='utf-8'))
    checks = ['content_review_complete', 'source_fidelity_review_complete',
              'coverage_review_complete', 'report_alignment_complete',
              'redistribution_review_complete']
    if review.get('status') != 'approved' or not all(review.get(key) is True for key in checks):
        raise SystemExit('Release review is pending; no distribution archive was created.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-weights', action='store_true')
    parser.add_argument('--both', action='store_true', help='Build the source archive once, then add weights to a second archive')
    parser.add_argument('--compresslevel', type=int, choices=range(1, 10), default=4)
    args = parser.parse_args()
    require_release_review(ROOT)
    subprocess.run(['python3', 'tools/build_catalog.py'], cwd=ROOT, check=True)
    subprocess.run(['python3', 'tools/validate_repository.py'], cwd=ROOT, check=True)
    if args.include_weights or args.both:
        for name in ['frontier', 'principle_guided']:
            if not (ROOT/'models'/name/'model.safetensors').is_file():
                raise SystemExit('Missing weights; run make models first')
    files = public_files(include_weights=args.include_weights and not args.both)
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    stem = 'qiushi-engine-babylm-research'+('-with-models' if args.include_weights and not args.both else '')
    target = output/(stem+'.zip')
    with tempfile.NamedTemporaryFile(prefix='.package-', suffix='.zip', dir=output, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=args.compresslevel) as archive:
            for path in files:
                archive.write(path, str(Path('Qiushi-Engine-Babylm-Research')/path.relative_to(ROOT)))
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
        # dist contains rebuildable outputs, not archived research originals.
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print('Created:', target.relative_to(ROOT))
    if args.both:
        full_target = output/'qiushi-engine-babylm-research-with-models.zip'
        with tempfile.NamedTemporaryFile(prefix='.package-', suffix='.zip', dir=output, delete=False) as temp:
            full_temporary = Path(temp.name)
        try:
            shutil.copyfile(target, full_temporary)
            with zipfile.ZipFile(full_temporary, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=args.compresslevel) as archive:
                source_paths = set(files)
                for path in public_files(include_weights=True):
                    if path in source_paths:
                        continue
                    archive.write(path, str(Path('Qiushi-Engine-Babylm-Research')/path.relative_to(ROOT)))
            with zipfile.ZipFile(full_temporary) as archive:
                assert len(archive.namelist()) == len(set(archive.namelist()))
                assert archive.testzip() is None
            full_temporary.replace(full_target)
        finally:
            full_temporary.unlink(missing_ok=True)
        print('Created:', full_target.relative_to(ROOT))


if __name__ == '__main__':
    main()
