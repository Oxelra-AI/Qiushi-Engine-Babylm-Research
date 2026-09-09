#!/usr/bin/env python3
"""research: quantify confounds in the legal BSM 1M screen.

Reads the actual research corpora, research training metrics, and research eval
results. Summarizes row packing, update count, official/binding word exposure,
and exact control weaknesses so route decisions do not over-attribute official
score changes to consistent binding.
"""
from __future__ import annotations
import json, pathlib, statistics, math
from collections import Counter

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
CORPUS_DIR = ROOT / 'data/legal_bsm_corpus'
RUN_DIR = ROOT / 'training/runs'
OUT_JSON = ROOT / 'data/bsm_1m_confounds.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/bsm_1m_route_control.md')
BATCH_SIZE = 64
MAX_WORDS = 1_000_000

CORPORA = {
    'official_control': CORPUS_DIR / 'official_control.jsonl',
    'bsm_coherent_20pct': CORPUS_DIR / 'bsm_coherent_20pct.jsonl',
    'bsm_swapped_20pct': CORPUS_DIR / 'bsm_swapped_20pct.jsonl',
}
RUNS = {
    'official_control': RUN_DIR / 'legal_bsm_1m_official_control_seed42' / 'scientific_metrics.json',
    'bsm_coherent_20pct': RUN_DIR / 'legal_bsm_1m_bsm_coherent_20pct_seed42' / 'scientific_metrics.json',
    'bsm_swapped_20pct': RUN_DIR / 'legal_bsm_1m_bsm_swapped_20pct_seed42' / 'scientific_metrics.json',
}
EVAL = ROOT / 'data/legal_bsm_1m_eval.json'


def load_prefix(path: pathlib.Path, max_words: int = MAX_WORDS):
    rows = []
    cum = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            w = int(r['words'])
            if cum + w > max_words:
                break
            rows.append(r); cum += w
    return rows, cum


def summarize_rows(rows):
    out = {'rows': len(rows), 'words': sum(int(r['words']) for r in rows), 'kinds': {}}
    for kind in sorted(set(r.get('kind','official') for r in rows)):
        ks = [r for r in rows if r.get('kind','official') == kind]
        lens = [int(r['words']) for r in ks]
        out['kinds'][kind] = {
            'rows': len(ks), 'words': sum(lens),
            'mean_words_per_row': statistics.mean(lens) if lens else 0,
            'median_words_per_row': statistics.median(lens) if lens else 0,
            'min_words_per_row': min(lens) if lens else None,
            'max_words_per_row': max(lens) if lens else None,
        }
    out['mean_words_per_row'] = out['words'] / max(1, out['rows'])
    out['estimated_updates_batch64'] = math.ceil(out['rows']/BATCH_SIZE)
    ent = Counter(r.get('query_entity','') for r in rows if r.get('kind') == 'binding')
    val = Counter(r.get('bound_value','') for r in rows if r.get('kind') == 'binding')
    out['top_binding_entities'] = ent.most_common(12)
    out['top_binding_values'] = val.most_common(20)
    return out


def score_table(eval_data):
    out = {}
    for name, rec in eval_data['results'].items():
        off = rec['official_fast']
        gp = (off['global_piqa_parallel']['score'] + off['global_piqa_nonparallel']['score'])/2
        out[name] = {
            'BLiMP': off['blimp_fast']['score'],
            'Supplement': off['supplement_fast']['score'],
            'Entity': off['entity_tracking_fast']['score'],
            'EWoK': off['ewok_fast']['score'],
            'GlobalPIQA': gp,
            'Reading': off['reading']['scores'].get('reading_mean'),
        }
    return out


def main():
    corp = {}
    for name, path in CORPORA.items():
        rows, cum = load_prefix(path)
        corp[name] = summarize_rows(rows)
        corp[name]['prefix_words_loaded_by_script'] = cum
    metrics = {k: json.loads(v.read_text()) for k, v in RUNS.items()}
    ev = json.loads(EVAL.read_text())
    scores = score_table(ev)
    payload = {
        'status': 'BSM_1M_CONFOUND_STATS',
        'batch_size': BATCH_SIZE,
        'max_words_prefix_rule': MAX_WORDS,
        'corpus_prefix_stats': corp,
        'training_metrics': metrics,
        'official_score_table': scores,
        'deltas': ev['deltas'],
        'scientific_points': [
            'All research binding probes are 0.0 both-correct for all three arms; official score movement is not evidence of learned binding.',
            'BSM arms have about 3x update count because binding rows are short: official_control 98 steps vs coherent 292 and swapped 291.',
            'BSM arms also replace 20% official words with generated binding words and use metadata-targeted answer masks; coherent-official deltas combine content, update count, row packing, and mask supervision.',
            'Coherent-swapped is cleaner but not exact: the materializer generated the two arms independently, so entity/value distributions are close but not row-identical counterfactual text multisets.',
            'Before scaling, either construct exact row-matched BSM controls and a matched-update official-experience reference, or redirect to mechanisms that make binding learnable earlier.'
        ]
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

    c = corp
    lines = ['# research — route-control analysis for legal BSM 1M', '', f'Evidence JSON: `{OUT_JSON}`', '',
             '## Actual 1M prefix packing and exposure', '',
             '| arm | rows | words | binding words | official words | mean words/row | estimated updates (batch64) | recorded updates |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['official_control','bsm_coherent_20pct','bsm_swapped_20pct']:
        kinds = c[name]['kinds']
        bw = kinds.get('binding',{}).get('words',0)
        ow = kinds.get('official',{}).get('words',0)
        lines.append(f"| {name} | {c[name]['rows']} | {c[name]['words']} | {bw} | {ow} | {c[name]['mean_words_per_row']:.2f} | {c[name]['estimated_updates_batch64']} | {metrics[name]['total_steps']} |")
    lines += ['', '## Binding-row length contrast', '',
              '| arm | binding rows | mean binding words/row | official rows | mean official words/row |',
              '|---|---:|---:|---:|---:|']
    for name in ['bsm_coherent_20pct','bsm_swapped_20pct']:
        b = c[name]['kinds'].get('binding',{})
        o = c[name]['kinds'].get('official',{})
        lines.append(f"| {name} | {b.get('rows',0)} | {b.get('mean_words_per_row',0):.2f} | {o.get('rows',0)} | {o.get('mean_words_per_row',0):.2f} |")
    lines += ['', '## Official fast profile (research)', '',
              '| arm | BLiMP | Supplement | Entity | EWoK | GlobalPIQA | Reading |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['official_control','bsm_coherent_20pct','bsm_swapped_20pct']:
        s = scores[name]
        lines.append(f"| {name} | {s['BLiMP']:.2f} | {s['Supplement']:.2f} | {s['Entity']:.2f} | {s['EWoK']:.2f} | {s['GlobalPIQA']:.3f} | {s['Reading']:.2f} |")
    lines += ['', '## Route implication', '',
              'The current legal BSM recipe should not be scaled as a mechanism-positive result. It did not form binding in probes, and the official task movements are entangled with row length, update count, official-data replacement, and targeted-mask supervision. Coherent-vs-swapped is the best current contrast but it is not an exact text-multiset control because the two BSM corpora were generated independently. A stronger next experiment must either build exact row-matched coherent/swapped controls plus a matched-update official-experience reference, or redirect toward an architecture/objective that makes binding learnable earlier.']
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True)
    OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT_JSON), 'note': str(OUT_NOTE)}, indent=2))

if __name__ == '__main__':
    main()
