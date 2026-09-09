"""Check build integrity and report-only scientific arithmetic (no experiments)."""
import csv
import json
import re
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE.parents[1] / 'results'
log = (BASE/'build/main.log').read_text()
for marker in ['Overfull', 'Missing character:', 'undefined references',
               'undefined citations', 'Font Warning:', 'There were undefined']:
    assert marker not in log, marker
assert not re.search(r'^!', log, flags=re.M), 'LaTeX error'
assert (BASE/'build/main.bbl').exists()
first_citations = list(dict.fromkeys(
    key for group in re.findall(r'\\citation\{([^}]+)\}',
                                (BASE/'build/main.aux').read_text())
    for key in group.split(',')))
bibliography_keys = re.findall(
    r'\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}',
    (BASE/'build/main.bbl').read_text())
assert bibliography_keys == first_citations, 'References must follow first-citation order.'
text = (BASE/'build/main.txt').read_text()
assert re.search(r'\bmake\s+report\b', text), 'Build command lost its space in the PDF'
for pattern in [r'S\d{4}A\d{2}', r'Sessions/', r'/(?:data|home)/[^\s/]+/',
                r'\bstep\d+\b', r'Qiushi-BabyLM-\S+-v\d+']:
    assert not re.search(pattern, text, flags=re.I), pattern
for phrase in ['第一阶段：前沿突破', '第二阶段：原理发现', '第三阶段：原理指导的模型改进',
               '本地完整评测向量', '干预目标选择率', '研究谱系与独立发现',
               '研究主题与成果目录', '11.835', '计算环境与实验支持',
               'Qiushi Engine 的研究组织与积累', '科研过程的递归自我改进',
               'AI Lab', 'NVIDIA H100']:
    assert phrase in text, phrase
author_source = (BASE/'latex/authors-zh.tex').read_text()
author_block = author_source.split(r'\newcommand{\ReportAffiliation}')[0]
author_names = re.findall(r'（([A-Za-z -]+)）', author_block)
assert len(author_names) == 16
assert author_names[-2:] == ['Hongsheng Chen', 'Yihao Yang']
author_rows = [line for line in author_block.splitlines() if ' & ' in line]
assert len(author_rows) == 4 and all(line.count(' & ') == 3 for line in author_rows)
for contact in ['Yihao Yang', 'Hongsheng Chen',
                'yangyihao@zju.edu.cn', 'hansomchen@zju.edu.cn']:
    assert contact in text, contact
for phrase in ['待公开', '待发布', '本地发布准备稿']:
    assert phrase not in text, phrase
fonts=subprocess.check_output(['pdffonts',str(BASE/'build/main.pdf')],text=True)
assert 'Type 3' not in fonts
for line in fonts.splitlines()[2:]:
    assert re.search(r'\byes\s+yes\s+(?:yes|no)\s+\d+\s+\d+\s*$',line), line
info=subprocess.check_output(['pdfinfo',str(BASE/'build/main.pdf')],text=True)
pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
assert pages >= 30
with (DATA/'training_strategy_comparison.csv').open() as stream:
    rows=list(csv.DictReader(stream))
assert len(rows)==9 and all(r['complete']=='true' for r in rows)
keys=['BLiMP','Supplement','EWoK','Entity','COMPS','SuperGLUE','GlobalPIQA','Reading','AoA']
for row in rows:
    assert abs(sum(float(row[k]) for k in keys)/9-float(row['Overall']))<1e-9
assert 86_005_295+3_162_742+517_332==89_685_369
with (BASE/'data/research_catalog.tsv').open() as stream:
    catalog=list(csv.DictReader(stream,delimiter='\t'))
assert [r['id'] for r in catalog]==[f'R{i:02d}' for i in range(1,75)]
assert catalog[35]['status']=='已撤回'
assert len({r['group'] for r in catalog})==14
assert all(all(r.values()) for r in catalog)
main=(BASE/'main.tex').read_text()
chapters=re.findall(r'\\input\{chapters/(\d+_[^}]+)\}',main)
assert len(chapters)==7
opening=(BASE/'chapters/01_background.tex').read_text()
expected=['有限经验学习的科学问题','研究背景与赛道设置','语言能力与学习行为的评价体系',
          '综合评价与排行榜解释','有限经验学习的关键挑战','性能增量与科学贡献',
          '研究价值与应用前景','Qiushi Engine 的三阶段自主研究']
assert re.findall(r'\\subsection\{([^}]+)\}',opening)==expected
with (DATA/'leaderboard_comparison.csv').open() as stream:
    board_reader=csv.DictReader(stream)
    assert 'official_overall_rank' not in board_reader.fieldnames
    board=list(board_reader)
assert len(board)==10
assert len([r for r in board if '/leslie721007/' in r['model_url']])==2
assert {r['model'] for r in board[:2]}=={
    'Qiushi-Engine-Frontier-Advancement',
    'Qiushi-Engine-Principle-Guided-Frontier-Advancement'}
assert not re.search(r'Qiushi-BabyLM-|babylm-strict-small-scale1p75-chck82',str(board),re.I)
for name in ['leaderboard.tex','board_language.tex','board_remaining.tex']:
    table=(BASE/'tables'/name).read_text()
    assert '名次' not in table and 'official_overall_rank' not in table
    assert len([line for line in table.splitlines() if ' & ' in line])==10
assert not any(term in text for term in ['全榜名次','全榜序位','原始排名'])
assert all((BASE/'chapters'/f'{n}.tex').exists() for n in chapters)
# Verify the concrete repository paths printed in the report, not only its
# Markdown reading guides. Commands and external model names are not paths.
root = BASE.parents[1]
material_roots = {'reports', 'research', 'methods', 'experiments', 'results',
                  'models', 'data', 'evidence', 'reproducibility', 'assets'}
material_paths = set()
for source in [BASE/'main.tex', *sorted((BASE/'chapters').glob('*.tex'))]:
    for value in re.findall(r'\\nolinkurl\{([^}]+)\}', source.read_text()):
        value = value.replace(r'\_', '_')
        if value.split('/')[0] not in material_roots:
            continue
        target = (root/value).resolve()
        assert target.is_relative_to(root), f'Report path escapes repository: {value}'
        assert target.exists(), f'Missing report material: {value}'
        material_paths.add(value)
figure_files = sorted((BASE/'figures').glob('*.pdf'))
assert len(figure_files) == 8, 'Expected the eight shared scientific figures'
for figure in figure_files:
    figure_text = subprocess.check_output(['pdftotext', str(figure), '-'], text=True)
    assert not re.search(r'[\u3400-\u4dbf\u4e00-\u9fff]', figure_text), figure.name
    figure_fonts = subprocess.check_output(['pdffonts', str(figure)], text=True)
    assert 'Type 3' not in figure_fonts, figure.name
print(json.dumps({'pages':pages,'latex_clean':True,'fonts_embedded_no_type3':True,
                  'references_in_first_citation_order':len(bibliography_keys),
                  'private_run_identifiers_absent_from_pdf':True,
                  'local_endpoints':9,'local_metrics_per_endpoint':9,
                  'main_chapters':len(chapters),'background_sections':len(expected),
                  'leaderboard_models':len(board),'old_qiushi_entries_absent':True,
                  'catalog_topics':len(catalog),'english_scientific_figures':len(figure_files),
                  'report_material_paths_checked':len(material_paths),
                  'arithmetic_checks_passed':True},indent=2))
