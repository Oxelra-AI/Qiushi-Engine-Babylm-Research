"""Check portable public files and frozen results; no research execution."""
import ast
import csv
import hashlib
import json
import re
import struct
import subprocess
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

from common import ROOT, local_only_paths, public_files
from fetch_models import check as check_model_file
from inspect_materials import verify_materials

LOCAL_ASSETS = local_only_paths()


def rows(name, delimiter=','):
    with (ROOT/'results'/name).open() as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def check_results():
    experiments = rows('training_strategy_comparison.csv')
    fields = ['BLiMP','Supplement','EWoK','Entity','COMPS','SuperGLUE','GlobalPIQA','Reading','AoA']
    assert len(experiments) == 9
    for r in experiments:
        assert r['complete'] == 'true'
        assert abs(sum(float(r[k]) for k in fields)/9-float(r['Overall'])) < 1e-9
    board = rows('leaderboard_comparison.csv')
    assert len(board) == 10
    assert all('rank' not in key.lower() for key in board[0])
    assert len([r for r in board if '/leslie721007/' in r['model_url']]) == 2
    assert not any('Qiushi-BabyLM-' in r['model'] for r in board)
    catalogue = rows('research_catalog.tsv', '\t')
    assert len(catalogue) == 74 and len({r['id'] for r in catalogue}) == 74
    assert next(r for r in catalogue if r['id'] == 'R36')['status'] == 'Withdrawn'
    cfg = json.loads((ROOT/'experiments/configs/stage3.json').read_text())
    assert cfg['parent_counted_words']+cfg['acquisition']['counted_words']+cfg['preservation']['additional_counted_words'] == cfg['full_strategy_total_words']
    with (ROOT/'experiments/index.csv').open() as stream:
        for r in csv.DictReader(stream):
            for field in ['method','result','report_section']:
                assert (ROOT/r[field]).is_file(), r[field]


def inspect_text(path, text):
    # Match actual identifiers/paths, not generic scientific terminology.
    patterns = [r'(?i)S\d{4}A\d{2}', r'(?i)\bstep[_-]\d{2,}',
                r'(?i)(?<![A-Za-z0-9])s(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])(?![A-Za-z0-9])',
                r'/(?:data|home)/(?:qiushi|ubuntu|admin)/',
                r'\.\.\./users/admin\b',
                r'\b(?:ghp_|gho_|github_pat_)[A-Za-z0-9_]{20,}',
                r'\bsk-[A-Za-z0-9_-]{24,}',
                r'\bhf_[A-Za-z0-9]{25,}',
                r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    for pattern in patterns:
        if re.search(pattern, text):
            raise ValueError('Private material in '+str(path.relative_to(ROOT)))
    if path.suffix == '.md':
        if not path.is_relative_to(ROOT/'reports/zh'):
            # Preserve verbatim corpus/tokenizer examples inside code spans.
            prose = re.sub(r'(?ms)^\s*```[^\n]*\n.*?^\s*```\s*$', '', text)
            prose = re.sub(r'`+[^`\n]*`+', '', prose)
            if re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', prose):
                raise ValueError('Non-English public prose in '+str(path.relative_to(ROOT)))
            if re.search(r'(?i)\b(?:six-session|peer-session|session-local|AI Lab|Qiushi LLM Research Ontology|Qiushi (?:AI |Python )?runtime)\b', prose):
                raise ValueError('Operational wording in '+str(path.relative_to(ROOT)))
        for target in re.findall(r'\[[^\]\n]*\]\(([^)]+)\)', text):
            target = target.split(' "', 1)[0].strip('<>')
            if urlsplit(target).scheme or target.startswith('#'):
                continue
            local = (path.parent/unquote(target.split('#')[0])).resolve()
            assert local.is_relative_to(ROOT), f'Link escapes repository: {path.relative_to(ROOT)}'
            assert local.exists() or str(local.relative_to(ROOT)) in LOCAL_ASSETS, f'Broken link: {path.relative_to(ROOT)} -> {target}'


def check_public_file(path):
    inspect_text(path, str(path.relative_to(ROOT)))
    if path.suffix in {'.md','.txt','.tex','.bib','.py','.sh','.json','.jsonl','.csv','.tsv','.cff','.yaml','.yml','.toml','.cfg','.ini','.svg'} or path.name in {'LICENSE','NOTICE','Makefile','AGENTS.md'}:
        text = path.read_text(encoding='utf-8')
        inspect_text(path, text)
        if path.suffix == '.py':
            ast.parse(text, filename=str(path.relative_to(ROOT)))
    elif path.suffix == '.pdf':
        text = subprocess.check_output(['pdftotext', str(path), '-'], text=True)
        inspect_text(path, text)
    elif path.suffix == '.safetensors':
        with path.open('rb') as stream:
            length_bytes = stream.read(8)
            assert len(length_bytes) == 8, f'Incomplete tensor header: {path.relative_to(ROOT)}'
            length = struct.unpack('<Q', length_bytes)[0]
            assert 0 < length < 2_000_000, f'Invalid tensor header: {path.relative_to(ROOT)}'
            header = json.loads(stream.read(length))
        inspect_text(path, json.dumps(header))
    elif path.suffix == '.npz':
        import numpy as np
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                inspect_text(path, key)
                if arrays[key].dtype.kind not in 'biufc':
                    raise ValueError('Non-numeric array requires separate review: '+str(path.relative_to(ROOT)))
    elif path.suffix == '.zip':
        if path.parent != ROOT/'data/protected':
            raise ValueError('Unclassified archive: '+str(path.relative_to(ROOT)))
        index_path = path.with_suffix('.json')
        records = None
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == index['sha256']
            records = {r['path']: r for r in index['files']}
        seen = set()
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                member = Path(item.filename)
                if member.is_absolute() or '..' in member.parts:
                    raise ValueError('Unsafe protected-data path: '+item.filename)
                if item.is_dir():
                    continue
                if not item.flag_bits & 1:
                    raise ValueError('Invalid protected-data member: '+item.filename)
                # This is the upstream dataset's public indexing-protection password.
                content = archive.read(item, pwd=b'ewok')
                if records is not None:
                    record = records[item.filename]
                    assert hashlib.sha256(content).hexdigest() == record['original_scientific_sha256']
                    assert hashlib.sha256((ROOT/item.filename).read_bytes()).hexdigest() == record['projection_sha256']
                    seen.add(item.filename)
                text = content.decode('utf-8')
                inspect_text(path, text)
        if records is not None:
            assert seen == set(records), 'Protected archive index differs from its members'


def check_files():
    with ProcessPoolExecutor(max_workers=8) as executor:
        for _ in executor.map(check_public_file, public_files(include_weights=True), chunksize=32):
            pass
    # Check the English entry point explicitly as well as the selected file set.
    source = ROOT/'reports/en/main.tex'
    if source.exists():
        inspect_text(source, source.read_text())


def check_models():
    manifest = json.loads((ROOT/'models/manifest.json').read_text())
    count = 0
    for m in manifest['models']:
        directory = ROOT/'models'/m['directory']
        for entry in m['files']:
            path = directory/entry['rfilename']
            if entry['rfilename'] == 'model.safetensors' and not path.exists():
                continue
            assert path.is_file(), path.relative_to(ROOT)
            check_model_file(path, entry)
        weight = directory/'model.safetensors'
        if weight.exists():
            with weight.open('rb') as stream:
                length = struct.unpack('<Q', stream.read(8))[0]
                assert 0 < length < 2_000_000
                header = json.loads(stream.read(length))
            total = 0
            for name, tensor in header.items():
                if name == '__metadata__':
                    inspect_text(weight, json.dumps(tensor))
                    continue
                n = 1
                for d in tensor['shape']:
                    n *= d
                total += n
            assert total == 36_458_592
            count += 1
    return count


def main():
    check_results()
    check_files()
    count = check_models()
    materials = verify_materials(public_only=True) if (ROOT/'evidence/materials_manifest.json').exists() else {}
    print(f'Passed public-source static checks: relative links, identifier patterns, result arithmetic, 74 research topics; {count} local models; distributed materials: {materials}.')
    print(f'{len(LOCAL_ASSETS)} declared local-only assets are excluded from the source distribution; use tools/inspect_materials.py check to verify the complete local archive.')
    print('Static checks do not certify semantic review, source fidelity, coverage completeness or release approval.')


if __name__ == '__main__':
    main()
