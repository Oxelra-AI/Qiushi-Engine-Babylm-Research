"""Small shared file-selection rules for checking and packaging."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {'reports', 'research', 'methods', 'experiments', 'results',
               'models', 'data', 'evidence', 'reproducibility', 'assets', 'tools'}
ROOT_FILES = {'README.md', 'LICENSE',
              'CITATION.cff', 'CHANGELOG.md', 'Makefile', '.gitignore', '.gitattributes'}
SUFFIXES = {'.md', '.tex', '.bib', '.pdf', '.png', '.json', '.jsonl', '.csv', '.tsv',
            '.py', '.txt', '.sh', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.svg',
            '.safetensors', '.model', '.vocab', '.npz', '.zip'}


def local_only_paths():
    manifest = ROOT / 'evidence/materials_manifest.json'
    if not manifest.is_file():
        return set()
    records = json.loads(manifest.read_text(encoding='utf-8'))['files']
    return {r['path'] for r in records if r.get('distribution') == 'local_only'}


def public_files(include_weights=False):
    selected = []
    local_only = local_only_paths()
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if rel.parts[0] not in DIRECTORIES and str(rel) not in ROOT_FILES:
            continue
        if any(part in {'build', '__pycache__'} for part in rel.parts):
            continue
        if str(rel) in local_only:
            continue
        if str(rel).startswith('reports/zh/tables/'):
            continue
        if str(rel).startswith(('reports/en/tables/', 'reports/en/reference/', 'reports/en/figures/')):
            continue
        if str(rel) in {'reports/en/references.bib', 'reports/en/qiushi-engine-babylm-report-en-reference.pdf'}:
            continue
        if path.is_symlink():
            raise ValueError('A public asset must not be a symlink: ' + str(rel))
        if not path.is_file():
            continue
        if path.suffix == '.safetensors' and not include_weights:
            continue
        if str(rel) in ROOT_FILES or path.suffix in SUFFIXES or path.name in {'LICENSE', 'NOTICE'}:
            selected.append(path)
        else:
            raise ValueError('Unclassified public file: ' + str(rel))
    return selected
