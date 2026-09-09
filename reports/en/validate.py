"""Check the English report build and alignment with the Chinese edition."""
import csv
import json
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ZH = BASE.parent / 'zh'
log = (BASE / 'build/main.log').read_text()
for marker in ['Overfull', 'Missing character:', 'undefined references',
               'undefined citations', 'Font Warning:', 'There were undefined']:
    assert marker not in log, marker
assert not re.search(r'^!', log, re.M), 'LaTeX error'
assert (BASE / 'build/main.bbl').is_file()
first_citations = list(dict.fromkeys(
    key for group in re.findall(r'\\citation\{([^}]+)\}',
                                (BASE / 'build/main.aux').read_text())
    for key in group.split(',')))
bibliography_keys = re.findall(
    r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',
    (BASE / 'build/main.bbl').read_text())
assert bibliography_keys == first_citations, 'References must follow first-citation order.'
text = (BASE / 'build/main.txt').read_text()
assert 'Strict-Small (this study)' in text, 'Keep the track label on one line.'
cover = subprocess.check_output([
    'pdftotext', '-f', '1', '-l', '1', '-layout',
    str(BASE / 'build/main.pdf'), '-'], text=True)
assert any('yangyihao@zju.edu.cn' in line and 'hansomchen@zju.edu.cn' in line
           for line in cover.splitlines()), 'Correspondence must fit on one line.'
assert 'Keywords' in cover, 'The abstract and keywords must fit on the first page.'
for pattern in [r'S\d{4}A\d{2}', r'Sessions/', r'/(?:data|home)/[^\s/]+/',
                r'\bstep\d+\b', r'Qiushi-BabyLM-\S+-v\d+', r'[\u3400-\u4dbf\u4e00-\u9fff]']:
    assert not re.search(pattern, text, re.I), pattern
for phrase in ['Research RSI', 'AI Lab', 'NVIDIA H100', '11.835',
               'yangyihao@zju.edu.cn', 'hansomchen@zju.edu.cn']:
    assert phrase in text, phrase
assert re.search(r'make\s+report-en', text)
author_source = (BASE / 'latex/authors.tex').read_text()
author_rows = [line for line in author_source.splitlines() if ' & ' in line]
assert len(author_rows) == 4 and all(row.count(' & ') == 3 for row in author_rows)
assert 'Hongsheng Chen' in author_rows[-1].split(' & ')[-2]
assert 'Yihao Yang' in author_rows[-1].split(' & ')[-1]
sources = [BASE / 'main.tex', *sorted((BASE / 'chapters').glob('*.tex'))]
english = '\n'.join(p.read_text() for p in sources)
chinese = '\n'.join(p.read_text() for p in [ZH / 'main.tex', *sorted((ZH / 'chapters').glob('*.tex'))])
labels = lambda s: set(re.findall(r'\\label\{([^}]+)\}', s))
assert labels(english) == labels(chinese), ('Label mismatch', labels(english) ^ labels(chinese))
def citations(source):
    return {key.strip() for match in re.findall(r'\\cite[a-z]*\{([^}]+)\}', source)
            for key in match.split(',')}
assert citations(english) == citations(chinese), ('Citation mismatch', citations(english) ^ citations(chinese))
def citation_order(source):
    return list(dict.fromkeys(key.strip() for match in re.findall(
        r'\\cite[a-z]*\{([^}]+)\}', source) for key in match.split(',')))
assert citation_order(english) == citation_order(chinese), 'Bilingual citation order differs.'
for main_source in [BASE / 'main.tex', ZH / 'main.tex']:
    source = main_source.read_text()
    assert r'\bibliographystyle{unsrtnat}' in source
    assert r'\bibliography{../references}' in source
figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', english)
assert len(figures) == 8
for name in figures:
    assert (ZH / 'figures' / name).is_file(), name
material_roots = {'reports', 'research', 'methods', 'experiments', 'results',
                  'models', 'data', 'evidence', 'reproducibility', 'assets'}
paths = set()
for value in re.findall(r'\\nolinkurl\{([^}]+)\}', english):
    value = value.replace(r'\_', '_')
    if value.split('/')[0] in material_roots:
        target = (ROOT / value).resolve()
        assert target.is_relative_to(ROOT) and target.exists(), value
        paths.add(value)
with (ROOT / 'results/research_catalog.tsv').open(newline='') as stream:
    catalog = list(csv.DictReader(stream, delimiter='\t'))
assert len(catalog) == 74 and catalog[35]['status'] == 'Withdrawn'
assert len({row['group'] for row in catalog}) == 14
for row in catalog:
    assert (ROOT / 'research/materials' / (row['id'] + '.md')).is_file()
# Generated numerical cells must agree, not merely share similar prose.
for name in ['strategies', 'local_language', 'local_remaining', 'strategy_differences',
             'leaderboard', 'board_language', 'board_remaining', 'relation_windows',
             'natural_restatement', 'finetuning_seeds']:
    def numeric_cells(path):
        rows = [line.split(' & ')[1:] for line in path.read_text().splitlines() if ' & ' in line]
        return [re.findall(r'[+-]?\d+(?:[,.]\d+)*', ' & '.join(row)) for row in rows]
    assert numeric_cells(BASE / 'tables' / (name + '.tex')) == numeric_cells(ZH / 'tables' / (name + '.tex')), name
fonts = subprocess.check_output(['pdffonts', str(BASE / 'build/main.pdf')], text=True)
assert 'Type 3' not in fonts
for line in fonts.splitlines()[2:]:
    assert re.search(r'\byes\s+yes\s+(?:yes|no)\s+\d+\s+\d+\s*$', line), line
info = subprocess.check_output(['pdfinfo', str(BASE / 'build/main.pdf')], text=True)
pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
print(json.dumps({'pages': pages, 'latex_clean': True, 'embedded_fonts': True,
                  'references_in_first_citation_order': len(bibliography_keys),
                  'cover_correspondence_single_line': True,
                  'shared_figures': len(figures), 'bilingual_labels_matched': len(labels(english)),
                  'bilingual_numeric_tables_matched': 10, 'research_topics': len(catalog),
                  'material_paths_checked': len(paths), 'model_experiments_run': False}, indent=2))
