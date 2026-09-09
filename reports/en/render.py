"""Generate English report tables from shared results; no model execution."""
import csv
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
DATA = ROOT / 'results'
TABLES = BASE / 'tables'
METRICS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS',
           'GlobalPIQA', 'SuperGLUE', 'Reading', 'AoA']
POLICIES = [
    ('ORDINARY_CONTINUATION', 'Ordinary continuation'),
    ('SPARSE_TARGET_ACQUISITION', r'Dense masking, sparse supervision $(M,S)$'),
    ('DENSE_ACQUISITION', r'Dense masking, dense supervision $(M,M)$'),
    ('PRINCIPLE_GUIDED', r'Dense masking, sparse supervision + ordinary-input preservation'),
]


def read(name, delimiter=','):
    with (DATA / name).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def escape(value):
    return re.sub(r'[&%_#]', lambda match: '\\' + match[0], value)


def write_table(name, rows):
    content = '\n'.join(' & '.join(row) + r' \\' for row in rows)
    (TABLES / (name + '.tex')).write_text(content + '\n\\bottomrule\n', encoding='utf-8')


def catalog():
    entries = read('research_catalog.tsv', '\t')
    assert [row['id'] for row in entries] == [f'R{i:02d}' for i in range(1, 75)]
    with (BASE.parent / 'zh/data/research_catalog.tsv').open(newline='') as stream:
        chinese = list(csv.DictReader(stream, delimiter='\t'))
    assert [(r['id'], r['section']) for r in entries] == [(r['id'], r['section']) for r in chinese]
    status_map = {'Modeling method': 'Model method', 'Methods and diagnostics': 'Method or analysis tool',
                  'Not yet run': 'Not run'}
    parts = []
    for group in dict.fromkeys(row['group'] for row in entries):
        parts.extend([
            r'\Needspace{8\baselineskip}', r'\subsection*{' + escape(group) + '}',
            r'\begin{longtable}{@{}P{39mm}P{111mm}@{}}',
            r'\toprule Topic & Intervention, finding, and status\\\midrule\endfirsthead',
            r'\toprule Topic (continued) & Intervention, finding, and status\\\midrule\endhead',
            r'\bottomrule\endfoot',
        ])
        for row in (entry for entry in entries if entry['group'] == group):
            target = row['section']
            left = escape(row['id']) + r'\quad ' + escape(row['title'])
            status = escape(status_map.get(row['status'], row['status']))
            right = r'\textbf{' + status + '}. ' + escape(row['intervention'] + ' ' + row['finding'])
            guide = '' if row['id'] == 'R26' else (
                ('Appendix' if target.startswith('app:') else 'Section') + r'~\ref{' + target + '}; ')
            url = 'https://github.com/Oxelra-AI/Qiushi-Engine-Babylm-Research/blob/main/research/materials/' + row['id'] + '.md'
            right += ' (' + guide + r'\href{' + url + '}{Materials: ' + row['id'] + '}.)'
            parts.append(left + ' & ' + right + r' \\[3pt]')
        parts.append(r'\end{longtable}')
    (TABLES / 'research_catalog.tex').write_text('\n'.join(parts) + '\n', encoding='utf-8')


def main():
    TABLES.mkdir(exist_ok=True)
    catalog()
    trials = read('training_strategy_comparison.csv')
    by_id = {row['experiment_id']: row for row in trials}
    parent = by_id['FRONTIER_REFERENCE']
    assert len(trials) == 9
    for row in trials:
        assert row['complete'] == 'true'
        assert abs(sum(float(row[key]) for key in METRICS) / 9 - float(row['Overall'])) < 1e-9

    def trial(stem, seed):
        key = f'{stem}_S{seed}'
        if stem == 'PRINCIPLE_GUIDED' and seed == 65:
            key = 'PRINCIPLE_GUIDED_REPLICATION_S65'
        return by_id[key]

    summary = [['Parent', f"{float(parent['Overall']):.4f}", 'Same parent', '86,005,295']]
    ordered = [(parent, 'Parent', '---')]
    for stem, label in POLICIES:
        a, b = trial(stem, 64), trial(stem, 65)
        summary.append([label, f"{float(a['Overall']):.4f}", f"{float(b['Overall']):.4f}",
                        f"{int(a['counted_exposure_words_reported']):,}"])
        for seed in (64, 65):
            short = dict(POLICIES)[stem]
            if stem == 'PRINCIPLE_GUIDED':
                short = r'$(M,S)$ + ordinary-input preservation'
            ordered.append((trial(stem, seed), short, f'620{seed}'))
    write_table('strategies', summary)
    for name, keys in [('local_language', METRICS[:5]),
                       ('local_remaining', ['GlobalPIQA', 'SuperGLUE', 'Reading', 'AoA', 'Overall'])]:
        write_table(name, [[label, seed] + [f"{float(row[key]):.4f}" for key in keys]
                           for row, label, seed in ordered])
    differences = []
    for method, reference, label in [
        ('SPARSE_TARGET_ACQUISITION', 'ORDINARY_CONTINUATION', r'$(M,S)$ minus ordinary continuation'),
        ('PRINCIPLE_GUIDED', 'ORDINARY_CONTINUATION', 'Complete minus ordinary continuation'),
        ('PRINCIPLE_GUIDED', 'SPARSE_TARGET_ACQUISITION', r'Complete minus $(M,S)$'),
        ('PRINCIPLE_GUIDED', None, 'Complete minus parent'),
    ]:
        values = []
        for seed in (64, 65):
            a, b = trial(method, seed), trial(reference, seed) if reference else parent
            delta = float(a['Overall']) - float(b['Overall'])
            assert abs(sum((float(a[k]) - float(b[k])) / 9 for k in METRICS) - delta) < 1e-9
            values.append(f'{delta:.4f}')
        differences.append([label] + values)
    write_table('strategy_differences', differences)

    windows = read('relation_window_controls.csv')
    rows = []
    for relation, label in [('exact_repetition', 'Exact repetition'), ('aligned_restatement', 'Aligned restatement')]:
        for window, name in [('same_window', 'Same window'), ('split_window', 'Split windows')]:
            values = []
            for seed in (43022, 43122):
                row = next(r for r in windows if r['relation'] == relation and r['window'] == window
                           and int(r['training_seed']) == seed)
                assert int(row['targets_per_checkpoint']) == 2732
                values.append(f"${float(row['delta_true_source_advantage_nats']):+.3f}$")
            rows.append([label, name] + values)
    write_table('relation_windows', rows)

    natural = read('natural_restatement_transfer.csv')
    rows = []
    for token_class, label in [('overlap', 'Target token present in source'),
                               ('nonoverlap', 'Target token absent from source')]:
        values = []
        for relation in ('exact_repetition', 'aligned_restatement'):
            row = next(r for r in natural if r['token_class'] == token_class and r['relation'] == relation)
            assert int(row['n_training_seeds']) == 3 and int(row['pairs_per_seed']) == 1200
            values.append(f"${float(row['delta_true_source_advantage_nats']):+.3f}"
                          rf"\pm {float(row['training_seed_sd']):.3f}$")
        rows.append([label] + values)
    write_table('natural_restatement', rows)

    finetuning = read('finetuning_seed_comparison.csv')
    for row in finetuning:
        tasks = ['boolq', 'multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']
        assert abs(sum(float(row[k]) for k in tasks) / 7 - float(row['SuperGLUE'])) < 1e-9
        if row['finetuning_seed'] == '42':
            assert float(row['SuperGLUE']) == float(by_id[row['experiment_id']]['SuperGLUE'])
    rows = []
    for identifier, label in [('FRONTIER_REFERENCE', 'Parent'),
                               ('SPARSE_TARGET_ACQUISITION_S64', r'Dense masking, sparse supervision $(M,S)$'),
                               ('PRINCIPLE_GUIDED_S64', r'$(M,S)$ + ordinary-input preservation')]:
        rows.append([label] + [f"{float(next(r for r in finetuning if r['experiment_id'] == identifier and int(r['finetuning_seed']) == seed)['SuperGLUE']):.4f}"
                               for seed in (42, 44)])
    write_table('finetuning_seeds', rows)

    board = read('leaderboard_comparison.csv')
    labels = ['Qiushi Engine / Principle-guided', 'Qiushi Engine / Frontier'] + [r['display_label'] for r in board[2:]]
    for name, keys in [('leaderboard', ['Overall', 'NLP', 'HumanLike']),
                       ('board_language', METRICS[:5]),
                       ('board_remaining', ['GlobalPIQA', 'SuperGLUE', 'Reading', 'AoA'])]:
        write_table(name, [[escape(label)] + [f"{float(row[k]):.2f}" for k in keys]
                           for row, label in zip(board, labels)])
    print('Generated 11 English tables from shared results; nine-metric arithmetic and catalog alignment passed.')


if __name__ == '__main__':
    main()
