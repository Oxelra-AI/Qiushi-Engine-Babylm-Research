"""Build reader indexes from existing scientific records, without copying them."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from common import ROOT

COLLECTIONS = {
    'initial_model_studies': 'Model structure, tokenization and learning baselines',
    'compact_experience': 'Compact experience, data budgets and training objectives',
    'representation_and_objectives': 'Representation, supervision and relational structure',
    'frontier_consolidation': 'Incremental learning and frontier model consolidation',
    'relation_learning': 'Relation structure, context use and transfer',
    'functional_learning': 'Functional access, acquisition and preservation',
}
STATUS = {
    'historical': 'Historical record', 'conditional': 'Conditional result',
    'superseded': 'Superseded', 'proposal': 'Research proposal',
    'withdrawn': 'Withdrawn', 'supported': 'Experimentally supported',
    'curated_method_correction': 'Method correction',
}
KINDS = {
    'note': 'Analysis, protocols and interpretation', 'code': 'Original programs',
    'data_builder': 'Data construction', 'input': 'Experimental inputs',
    'config': 'Configuration', 'result': 'Results and measurements',
    'checkpoint': 'Model checkpoints', 'figure': 'Figures',
}


def cell(value):
    return re.sub(r'\s+', ' ', str(value)).replace('|', '&#124;').replace('[', '&#91;').replace(']', '&#93;').replace('<', '&lt;').replace('>', '&gt;')


def link(label, source, target):
    relative = Path(os.path.relpath(target, Path(source).parent)).as_posix()
    return '[' + cell(label) + '](' + quote(relative, safe='/') + ')'


def title(path):
    stem = path.stem.replace('_', ' ')
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream):
            if number > 60:
                break
            if line.startswith('# '):
                heading = line[2:].strip().strip('#').strip()
                if heading.lower().startswith(stem.lower()) and len(heading) > len(stem):
                    remainder = heading[len(stem):].lstrip(' :-\u2014\u2013')
                    if len(remainder) > 8:
                        heading = remainder
                return heading or stem
    return stem


def document_role(path):
    parts = Path(path).parts
    if parts[0] == 'research' and parts[1] in {'notes', 'plans', 'documents'}:
        return parts[1]
    area = parts[3]
    if area == 'plans':
        return 'plans'
    if area in {'notes', 'analysis'}:
        return 'notes'
    return 'documents'


def document_rows(root, records, topics):
    memberships = defaultdict(list)
    for topic in topics:
        for path in topic['materials']:
            memberships[path].append(topic['id'])
    documents = []
    for record in records:
        if record['kind'] != 'note' or not record['path'].endswith('.md'):
            continue
        path = record['path']
        if not (root / path).is_file():
            raise ValueError('Missing indexed document: ' + path)
        documents.append({
            'path': path, 'title': title(root / path),
            'collection': record['collection'], 'role': document_role(path),
            'scientific_status': record['scientific_status'],
            'topics': memberships[path],
        })
    if len({row['path'] for row in documents}) != len(documents):
        raise ValueError('Duplicate document identity')
    return sorted(documents, key=lambda row: row['path'])


def record_table(rows, page):
    lines = ['| Original material | Recorded status | Related topics |', '| --- | --- | --- |']
    for row in rows:
        topics = ', '.join(link(t, page, f'research/materials/{t}.md') for t in row['topics'])
        lines.append('| ' + link(row['title'], page, row['path']) + ' | ' +
                     STATUS.get(row['scientific_status'], row['scientific_status']) +
                     ' | ' + (topics or 'Supporting record') + ' |')
    return lines


def render_documents(root, records, topics):
    documents = document_rows(root, records, topics)
    outputs = {'research/document_index.json': json.dumps(
        {'schema_version': 1, 'documents': documents}, ensure_ascii=False, indent=2) + '\n'}
    titles = {'notes': 'Research notes and analysis', 'plans': 'Experimental protocols and proposals',
              'documents': 'Measurements, data descriptions and technical records'}
    intros = {
        'notes': 'Analyses of research questions, competing explanations, experimental decisions and revised conclusions. Start with the reading guide, then follow individual questions to the original records.',
        'plans': 'Original hypotheses, competing explanations, matched controls and decision criteria. A protocol is not evidence that an experiment was completed; consult the associated results.',
        'documents': 'Data descriptions, measurement summaries, checkpoint records and technical documentation. These supporting records are not counted as additional independent discoveries.',
    }
    for role, label in titles.items():
        page = f'research/{role}/README.md'
        selected = [row for row in documents if row['role'] == role]
        lines = ['# ' + label, '', intros[role], '',
                 '[Reading guide](../reading_paths.md) | [Three research stages](../README.md) | [Glossary](../terms.md) | [Topic materials](../materials.md)', '',
                 f'This directory contains **{len(selected)} research documents**, grouped by scientific topic.', '',
                 'Recorded statuses come from the material manifest and do not override later corrections. Formal model scores are maintained in `results/`.', '']
        lines += ['| Research area | Documents |', '| --- | ---: |']
        for collection, name in COLLECTIONS.items():
            group = [row for row in selected if row['collection'] == collection]
            if not group:
                continue
            subpage = f'research/{role}/{collection}/README.md'
            lines.append('| ' + link(name, page, subpage) + f' | {len(group)} |')
            body = ['# ' + name, '', '## ' + label, '',
                    '[Directory](../README.md) | [Reading guide](../../reading_paths.md) | [Glossary](../../terms.md)', '',
                    f'This folder contains {len(group)} research documents, listed by their original titles. Programs, inputs and numerical results remain in their respective directories.', '',
                    'Recorded statuses describe historical material, not necessarily the final conclusion.', '']
            if role == 'documents':
                grouped = defaultdict(list)
                for row in group:
                    parts = Path(row['path']).parts
                    key = '/'.join(parts[3:5])
                    grouped[key].append(row)
                for key, rows in grouped.items():
                    body += ['<details>', f'<summary>{cell(key)} ({len(rows)})</summary>', '']
                    body += record_table(rows, subpage) + ['', '</details>', '']
            else:
                body += record_table(group, subpage)
            outputs[subpage] = '\n'.join(body) + '\n'
        lines += ['', '[Complete document index](../document_index.json) | [Research notes](../notes/README.md) | [Protocols](../plans/README.md) | [Technical records](../documents/README.md)', '']
        outputs[page] = '\n'.join(lines)
    return outputs


def render_topics(records, topics):
    by_path = {record['path']: record for record in records}
    lines = ['# Research topics and original materials', '',
             'Start with the [reading guide](reading_paths.md) for the key scientific decisions, then use this table to find individual experiments. Each topic separates original notes, protocols, programs, inputs and results.', '',
             '[Research notes](notes/README.md) | [Protocols](plans/README.md) | [Technical records](documents/README.md) | [Three stages](README.md) | [Core programs](../experiments/CORE_PROGRAMS.md)', '',
             'The 74 entries index scientific questions, not 74 independent innovations. Supported results, limitations, withdrawn interpretations and untested proposals retain their distinct statuses.', '']
    outputs = {}
    for family in dict.fromkeys(topic['research_family'] for topic in topics):
        lines += ['## ' + family, '', '| Topic | Status | Materials |', '| --- | --- | ---: |']
        for topic in [t for t in topics if t['research_family'] == family]:
            page = f"research/materials/{topic['id']}.md"
            anchor = topic['id'].lower()
            label = topic['id'] + ' - ' + topic['title']
            lines.append('| ' + f'<a id="{anchor}"></a>' + link(label, 'research/materials.md', page) + ' | ' +
                         cell(topic['status']) + ' | ' + str(len(topic['materials'])) + ' |')
            body = ['# ' + label, '',
                    '[All topics](../materials.md) | [Reading guide](../reading_paths.md) | [Scientific synthesis](../scientific_guide.md)', '',
                    '**' + topic['status'] + '**', '', topic['explanation'], '',
                    topic.get('report_guide', 'Report section: `' + topic['report_section'] + '`. Interpret historical claims within their stated conditions and in light of later corrections.'), '']
            groups = defaultdict(list)
            for path in topic['materials']:
                if path not in by_path:
                    raise ValueError('Topic material absent from manifest: ' + path)
                groups[by_path[path]['kind']].append(path)
            for kind, paths in groups.items():
                body += ['## ' + KINDS.get(kind, kind), '']
                for path in paths:
                    if by_path[path].get('distribution') == 'local_only':
                        body.append('- `' + path + '`: archived locally, not distributed with the source package; see ' +
                                    link('data sources and access conditions', page, 'data/README.md') + '.')
                    else:
                        body.append('- ' + link(Path(path).name, page, path))
                body.append('')
            outputs[page] = '\n'.join(body)
        lines.append('')
    lines += ['[Material manifest](../evidence/materials_manifest.json) | [Programs and input dependencies](../experiments/entrypoints.json)', '']
    outputs['research/materials.md'] = '\n'.join(lines)
    return outputs


def build(root):
    records = json.loads((root / 'evidence/materials_manifest.json').read_text())['files']
    topics = json.loads((root / 'research/topics.json').read_text())['topics']
    return render_documents(root, records, topics) | render_topics(records, topics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check generated navigation without writing')
    args = parser.parse_args()
    outputs = build(ROOT)
    stale = []
    for name, content in outputs.items():
        target = ROOT / name
        if args.check:
            if not target.is_file() or target.read_text() != content:
                stale.append(name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding='utf-8')
    if stale:
        raise SystemExit('Navigation needs rebuilding: ' + ', '.join(stale))
    print(json.dumps({'pages': len(outputs), 'check': args.check}))


if __name__ == '__main__':
    main()
