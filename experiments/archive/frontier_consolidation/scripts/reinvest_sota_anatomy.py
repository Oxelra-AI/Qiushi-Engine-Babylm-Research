#!/usr/bin/env python3
"""research CPU-only anatomy of the compact_view_reinvest SOTA endpoint.

Compares the compact_view_reinvest full official-compatible result
with inherited clean-Qwen and with the compact_view_core aggregate from research.
Reads existing prediction/report files only; no GPU work.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = pathlib.Path('.')
EVAL = ROOT / 'experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/reinvest_sota_anatomy'
NOTE = ROOT / 'research/notes/frontier_consolidation/reinvest_sota_anatomy.md'

MODELS = {
    'clean_qwen': {
        'label': 'COMPACT_EXPERIENCE clean-Qwen aligned full endpoint',
        'per_target': ROOT / 'experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json',
    },
    'compact_view_reinvest': {
        'label': 'frontier_consolidation compact_view_reinvest endpoint evaluated by representation_and_objectives',
        'per_target': ROOT / 'experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/per_target/compact_view_reinvest.json',
    },
}
FULL_SUMMARY = ROOT / 'experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json'
SOTA_VERIFY = ROOT / 'experiments/archive/representation_and_objectives/data/reinvest_sota_verification/compact_reinvest_sota_verification.json'
CORE = ROOT / 'experiments/archive/frontier_consolidation/data/compact_core_full_loss_anatomy/compact_core_full_loss_anatomy.json'
SUPPLEMENT_FILES = ['qa_congruence_easy', 'qa_congruence_tricky', 'turn_taking', 'subject_aux_inversion', 'hypernym']
GLOBAL_FILES = {
    'GlobalPIQA_parallel': EVAL / 'global_piqa_parallel/eng_latn.jsonl',
    'GlobalPIQA_nonparallel': EVAL / 'global_piqa_nonparallel/eng_latn.jsonl',
}
SUPERGLUE_TASKS = ['boolq', 'multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']
VISIBLE_LEADER = {
    'BLiMP': 67.2,
    'Supplement': 56.01,
    'EWoK': 56.07,
    'Entity': 28.45,
    'COMPS': 53.57,
    'SuperGLUE': 69.79,
    'GlobalPIQA': 39.67,
    'Reading': 5.42,
    'AoA': 0.0,
    'Overall': 41.8,
}


def read_json(path: pathlib.Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x).strip())


def short(x: Any, n: int = 220) -> str:
    s = norm(x)
    return s if len(s) <= n else s[: n - 1] + '…'


def pct(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def official_scores(model_key: str) -> Dict[str, float]:
    d = read_json(MODELS[model_key]['per_target'])
    return {k: float(v) for k, v in d['official_overall']['scores'].items()}


def official_overall(model_key: str) -> Dict[str, Any]:
    return read_json(MODELS[model_key]['per_target'])['official_overall']


def parse_uid_report(path: pathlib.Path) -> Tuple[Dict[str, float], Optional[float]]:
    uid_scores: Dict[str, float] = {}
    average: Optional[float] = None
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    in_uid = False
    for i, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith('### UID ACCURACY'):
            in_uid = True
            continue
        if in_uid and line.startswith('###'):
            in_uid = False
        if in_uid and ':' in line:
            k, v = line.split(':', 1)
            try:
                uid_scores[k.strip()] = float(v.strip())
            except ValueError:
                pass
        if line.startswith('### AVERAGE ACCURACY') and i + 1 < len(lines):
            try:
                average = float(lines[i + 1].strip())
            except ValueError:
                pass
    return uid_scores, average


def supplement_bundle(model_key: str) -> Dict[str, Any]:
    d = read_json(MODELS[model_key]['per_target'])
    task = d['tasks']['Supplement']
    preds = read_json(pathlib.Path(task['predictions']))
    flat: Dict[str, str] = {}
    for uid in SUPPLEMENT_FILES:
        for rec in preds[uid]['predictions']:
            flat[rec['id']] = norm(rec['pred'])
    uid_scores, avg = parse_uid_report(pathlib.Path(task['report']))
    return {'preds': flat, 'uid_scores': uid_scores, 'average_score': avg}


def summarize_pair(rows: Iterable[Dict[str, Any]], a: str, b: str, max_examples: int = 12) -> Dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    a_correct = sum(int(r[f'{a}_correct']) for r in rows)
    b_correct = sum(int(r[f'{b}_correct']) for r in rows)
    lost = [r for r in rows if r[f'{a}_correct'] and not r[f'{b}_correct']]
    gained = [r for r in rows if (not r[f'{a}_correct']) and r[f'{b}_correct']]
    return {
        'n': n,
        f'{a}_correct': a_correct,
        f'{b}_correct': b_correct,
        f'{a}_acc': pct(a_correct, n),
        f'{b}_acc': pct(b_correct, n),
        'delta_acc_b_minus_a': round((pct(b_correct, n) or 0.0) - (pct(a_correct, n) or 0.0), 6),
        'lost_from_a_to_b': len(lost),
        'gained_from_a_to_b': len(gained),
        'lost_examples': [{k: r[k] for k in r if k in {'id', 'uid', 'contrast', 'fragment', 'template', 'ambiguous', 'good', 'bad', 'category', 'inspiration', 'prompt', 'answer', 'clean_qwen_pred', 'compact_view_reinvest_pred'}} for r in lost[:max_examples]],
        'gained_examples': [{k: r[k] for k in r if k in {'id', 'uid', 'contrast', 'fragment', 'template', 'ambiguous', 'good', 'bad', 'category', 'inspiration', 'prompt', 'answer', 'clean_qwen_pred', 'compact_view_reinvest_pred'}} for r in gained[:max_examples]],
    }


def analyze_supplement() -> Dict[str, Any]:
    gold: Dict[str, Dict[str, Any]] = {}
    for uid in SUPPLEMENT_FILES:
        for i, row in enumerate(read_jsonl(EVAL / f'supplement_filtered/{uid}.jsonl')):
            rid = f'{uid}_{i}'
            gold[rid] = {**row, 'uid': uid, 'id': rid}
    bundles = {m: supplement_bundle(m) for m in MODELS}
    rows: List[Dict[str, Any]] = []
    for rid, row in gold.items():
        good = norm(row['sentence_good'])
        rec: Dict[str, Any] = {
            'id': rid,
            'uid': row['uid'],
            'row': row.get('row'),
            'contrast': row.get('contrast'),
            'fragment': row.get('fragment'),
            'template': row.get('template'),
            'ambiguous': row.get('ambiguous'),
            'good': short(row['sentence_good']),
            'bad': short(row['sentence_bad']),
        }
        for m, bundle in bundles.items():
            pred = norm(bundle['preds'][rid])
            rec[f'{m}_pred'] = short(pred)
            rec[f'{m}_correct'] = pred == good
        rows.append(rec)
    by_uid: Dict[str, Any] = {}
    for uid in SUPPLEMENT_FILES:
        subset = [r for r in rows if r['uid'] == uid]
        entry: Dict[str, Any] = {'n': len(subset)}
        for m, bundle in bundles.items():
            c = sum(int(r[f'{m}_correct']) for r in subset)
            entry[f'{m}_correct'] = c
            entry[f'{m}_acc'] = pct(c, len(subset))
            entry[f'{m}_report_uid_score'] = bundle['uid_scores'].get(uid)
        entry['reinvest_minus_clean'] = round(entry['compact_view_reinvest_acc'] - entry['clean_qwen_acc'], 6)
        entry['clean_to_reinvest'] = summarize_pair(subset, 'clean_qwen', 'compact_view_reinvest')
        by_uid[uid] = entry
    overall: Dict[str, Any] = {'n': len(rows)}
    for m, bundle in bundles.items():
        c = sum(int(r[f'{m}_correct']) for r in rows)
        overall[f'{m}_all_item_acc'] = pct(c, len(rows))
        overall[f'{m}_official_equal_uid_average'] = bundle['average_score']
    overall['reinvest_minus_clean_equal_uid'] = round(overall['compact_view_reinvest_official_equal_uid_average'] - overall['clean_qwen_official_equal_uid_average'], 6)
    return {'overall': overall, 'by_uid': by_uid, 'clean_to_reinvest_overall': summarize_pair(rows, 'clean_qwen', 'compact_view_reinvest')}


def global_predictions(model_key: str, task_col: str) -> Dict[str, str]:
    d = read_json(MODELS[model_key]['per_target'])
    task = d['tasks'][task_col]
    preds = read_json(pathlib.Path(task['predictions']))
    flat: Dict[str, str] = {}
    for exid, block in preds.items():
        plist = block.get('predictions', [])
        if plist:
            flat[exid] = norm(plist[0]['pred'])
    return flat


def analyze_globalpiqa() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col, path in GLOBAL_FILES.items():
        gold_rows = read_jsonl(path)
        preds = {m: global_predictions(m, col) for m in MODELS}
        rows: List[Dict[str, Any]] = []
        by_slice = defaultdict(lambda: {'n': 0, 'clean_qwen_correct': 0, 'compact_view_reinvest_correct': 0})
        for row in gold_rows:
            exid = row['example_id']
            sol = norm(row[f'solution{row["label"]}'])
            rec: Dict[str, Any] = {
                'id': exid,
                'category': row.get('categories'),
                'prompt': short(row.get('prompt'), 260),
                'answer': short(sol, 180),
                'inspiration': '(none)',
            }
            try:
                rec['inspiration'] = json.loads(row.get('supplement') or '{}').get('example_inspiration', '(none)')
            except Exception:
                rec['inspiration'] = '(parse_error)'
            for m in MODELS:
                p = preds[m][exid]
                rec[f'{m}_pred'] = short(p, 180)
                rec[f'{m}_correct'] = p == sol
            cats = [c.strip() for c in str(row.get('categories', '')).split(',') if c.strip()] or ['(none)']
            keys = [f'category={c}' for c in cats] + [f'inspiration={rec["inspiration"]}']
            for key in keys:
                s = by_slice[key]
                s['n'] += 1
                for m in MODELS:
                    s[f'{m}_correct'] += int(rec[f'{m}_correct'])
            rows.append(rec)
        slice_rows: List[Dict[str, Any]] = []
        for key, s in by_slice.items():
            rr = {'slice': key, **s}
            for m in MODELS:
                rr[f'{m}_acc'] = pct(s[f'{m}_correct'], s['n'])
            rr['reinvest_minus_clean'] = round(rr['compact_view_reinvest_acc'] - rr['clean_qwen_acc'], 6)
            slice_rows.append(rr)
        slice_rows.sort(key=lambda r: (r['reinvest_minus_clean'], -r['n']))
        out[col] = {
            'n': len(rows),
            'clean_to_reinvest': summarize_pair(rows, 'clean_qwen', 'compact_view_reinvest', max_examples=25),
            'worst_slices': slice_rows[:20],
            'best_slices': slice_rows[-20:][::-1],
        }
    return out


def coerce_label(x: Any) -> Any:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        y = x.strip()
        if y.isdigit() or (y.startswith('-') and y[1:].isdigit()):
            return int(y)
        return {'false': 0, 'true': 1, 'entailment': 0, 'not_entailment': 1, 'neutral': 1, 'contradiction': 2}.get(y.lower(), y)
    return x


def superglue_records(model_key: str) -> Dict[str, Dict[str, Any]]:
    d = read_json(MODELS[model_key]['per_target'])
    return {r['task']: r for r in d['tasks']['SuperGLUE']['tasks']}


def super_preds(rec: Dict[str, Any], task: str) -> List[Any]:
    data = read_json(pathlib.Path(rec['predictions']))
    block = data.get(task, data)
    preds = block['predictions'] if isinstance(block, dict) and 'predictions' in block else block
    return [coerce_label(p['pred'] if isinstance(p, dict) and 'pred' in p else p) for p in preds]


def extract_task_fields(task: str, row: Dict[str, Any]) -> Dict[str, Any]:
    keys = ['idx', 'sentence1', 'sentence2', 'premise', 'hypothesis', 'question', 'passage', 'paragraph', 'text', 'span1_text', 'span2_text', 'word', 'query', 'answers', 'answer', 'question_text']
    out: Dict[str, Any] = {}
    for k in keys:
        if k in row:
            out[k] = short(row[k], 220)
    if not out:
        for k, v in row.items():
            if k != 'label':
                out[k] = short(v, 220)
                if len(out) >= 5:
                    break
    return out


def analyze_superglue() -> Dict[str, Any]:
    recs = {m: superglue_records(m) for m in MODELS}
    out: Dict[str, Any] = {}
    for task in SUPERGLUE_TASKS:
        rows_gold = read_jsonl(EVAL / f'glue_filtered/{task}.valid.jsonl')
        labels = [coerce_label(r['label']) for r in rows_gold]
        preds = {m: super_preds(recs[m][task], task) for m in MODELS}
        n = min([len(labels)] + [len(preds[m]) for m in MODELS])
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            rec: Dict[str, Any] = {
                'i': i,
                'gold': labels[i],
                'fields': extract_task_fields(task, rows_gold[i]),
            }
            for m in MODELS:
                rec[f'{m}_pred'] = preds[m][i]
                rec[f'{m}_correct'] = preds[m][i] == labels[i]
            rows.append(rec)
        entry: Dict[str, Any] = {'n': n}
        for m in MODELS:
            c = sum(int(r[f'{m}_correct']) for r in rows)
            entry[f'{m}_correct'] = c
            entry[f'{m}_acc'] = pct(c, n)
            entry[f'{m}_record_accuracy'] = recs[m][task]['accuracy']
            entry[f'{m}_pred_counts'] = recs[m][task].get('pred_counts')
        entry['reinvest_minus_clean'] = round(entry['compact_view_reinvest_acc'] - entry['clean_qwen_acc'], 6)
        entry['clean_to_reinvest'] = summarize_pair(rows, 'clean_qwen', 'compact_view_reinvest', max_examples=10)
        out[task] = entry
    return out


def arithmetic() -> Dict[str, Any]:
    clean = official_scores('clean_qwen')
    reinv = official_scores('compact_view_reinvest')
    core = read_json(CORE)['official_vectors']['compact_view_core']['scores']
    deltas_vs_clean = {k: round(reinv[k] - clean[k], 6) for k in reinv if k in clean}
    deltas_vs_leader = {k: round(reinv[k] - VISIBLE_LEADER[k], 6) for k in VISIBLE_LEADER if k in reinv}
    deltas_vs_core = {k: round(reinv[k] - core[k], 6) for k in reinv if k in core}
    deltas_vs_clean['Overall'] = round(official_overall('compact_view_reinvest')['Overall'] - official_overall('clean_qwen')['Overall'], 6)
    deltas_vs_leader['Overall'] = round(official_overall('compact_view_reinvest')['Overall'] - VISIBLE_LEADER['Overall'], 6)
    # research core Overall was official and contains negative AoA.
    core_overall = read_json(CORE)['official_vectors']['compact_view_core']['Overall']
    deltas_vs_core['Overall'] = round(official_overall('compact_view_reinvest')['Overall'] - core_overall, 6)
    return {
        'clean_scores': clean,
        'compact_view_reinvest_scores': reinv,
        'compact_view_core_scores_from_step020': core,
        'reinvest_minus_clean': deltas_vs_clean,
        'reinvest_minus_visible_leader': deltas_vs_leader,
        'reinvest_minus_compact_view_core': deltas_vs_core,
        'clean_official_overall': official_overall('clean_qwen'),
        'reinvest_official_overall': official_overall('compact_view_reinvest'),
        'compact_view_core_overall_from_step020': core_overall,
    }


def make_note(result: Dict[str, Any]) -> str:
    ar = result['arithmetic']
    supp = result['supplement']
    glob = result['globalpiqa']
    sg = result['superglue']
    verify = result['verification_excerpt']
    lines: List[str] = []
    lines.append('# research compact-view-reinvest SOTA anatomy')
    lines.append('')
    lines.append('CPU-only reading of the peer-evaluated compact_view_reinvest full official-compatible endpoint. This note does not launch training or evaluation; it parses existing reports/predictions and A01 verification records.')
    lines.append('')
    lines.append('## Score status')
    ro = ar['reinvest_official_overall']
    lines.append(f"- compact_view_reinvest Overall {ro['Overall']:.6f}; NLP average {ro['NLP_average']:.6f}; Human-like average {ro['Human_like_average']:.6f}; AoA status {ro.get('aoa_status')}.")
    lines.append('- Scores: ' + ', '.join(f"{k} {v:.3f}" for k, v in ar['compact_view_reinvest_scores'].items()))
    lines.append('- Delta vs clean-Qwen: ' + ', '.join(f"{k} {v:+.3f}" for k, v in ar['reinvest_minus_clean'].items()))
    lines.append('- Delta vs visible 41.8 leader: ' + ', '.join(f"{k} {v:+.3f}" for k, v in ar['reinvest_minus_visible_leader'].items()))
    lines.append('- Delta vs compact_view_core: ' + ', '.join(f"{k} {v:+.3f}" for k, v in ar['reinvest_minus_compact_view_core'].items()))
    lines.append('')
    lines.append('## Verification signals read from A01')
    lines.append(f"- train 100M hash matches train command: {verify['training_hash_ok']}; exact 10M words: {verify['exact_10M']}; required AoA checkpoints present: {verify['aoa_required_present']}; missing columns: {verify['missing_columns']}; unexpected missing/empty artifacts: {verify['unexpected_missing_artifacts']}.")
    lines.append(f"- refined overlap audit changed block vs heldout official rows: score-bearing overlap rows delta {verify['changed_minus_heldout_score_bearing_rows']}, GLUE-valid rows delta {verify['changed_minus_heldout_glue_valid_rows']}, unique 7-gram delta {verify['changed_minus_heldout_unique_ngrams']}. This is a safeguard result, not proof that all overlap is harmless.")
    lines.append('')
    lines.append('## Supplement anatomy')
    ov = supp['overall']
    lines.append(f"- Official equal-UID Supplement: clean {ov['clean_qwen_official_equal_uid_average']:.2f}, reinvest {ov['compact_view_reinvest_official_equal_uid_average']:.2f}, delta {ov['reinvest_minus_clean_equal_uid']:+.2f}.")
    lines.append('| UID | clean | reinvest | delta |')
    lines.append('|---|---:|---:|---:|')
    for uid, s in supp['by_uid'].items():
        lines.append(f"| {uid} | {s['clean_qwen_acc']:.2f} | {s['compact_view_reinvest_acc']:.2f} | {s['reinvest_minus_clean']:+.2f} |")
    lines.append('')
    lines.append('## GlobalPIQA anatomy')
    for col, s in glob.items():
        pair = s['clean_to_reinvest']
        lines.append(f"- {col}: clean {pair['clean_qwen_acc']:.2f}, reinvest {pair['compact_view_reinvest_acc']:.2f}, delta {pair['delta_acc_b_minus_a']:+.2f}; lost {pair['lost_from_a_to_b']}, gained {pair['gained_from_a_to_b']}, n={pair['n']}.")
    lines.append('')
    lines.append('## SuperGLUE anatomy')
    lines.append('| task | clean | reinvest | delta |')
    lines.append('|---|---:|---:|---:|')
    for task, s in sorted(sg.items(), key=lambda kv: kv[1]['reinvest_minus_clean']):
        lines.append(f"| {task} | {s['clean_qwen_acc']:.3f} | {s['compact_view_reinvest_acc']:.3f} | {s['reinvest_minus_clean']:+.3f} |")
    lines.append('')
    lines.append('## Scientific reading')
    lines.append('- This is a real endpoint update, not merely a fast-screen projection: all nine official-like columns are present, AoA is submit-ready under the local official-compatible helper, and the recomputed arithmetic exceeds both clean-Qwen and the visible 41.8 leader.')
    lines.append('- Reinvestment changes the meaning of compact density: the failed neutral compact-core did not show aggregate NLP gain, but reinvest turns compact savings into added source exposure and recovers SuperGLUE/AoA while preserving large EWoK and Entity gains. The principle is better stated as redundancy-reduced semantic second views plus reinvested source diversity under a fixed word budget, not compacting alone.')
    lines.append('- Remaining weakness is GlobalPIQA/practical commonsense relative to both the visible leader and the target of a durable SOTA. Because the endpoint margin over 41.8 is only +0.2868, seed43122 and independent verification matter before packaging or trying to improve. Do not start a new 100M route until the pending seed/AoA/compact-repeat evidence is read; use this endpoint as the new reference.')
    lines.append('')
    lines.append(f"Machine-readable JSON: `{OUT_DIR / 'reinvest_sota_anatomy.json'}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    verify = read_json(SOTA_VERIFY)
    overlap = read_json(ROOT / 'experiments/archive/representation_and_objectives/data/reinvest_overlap_refined/compact_reinvest_refined_eval_overlap_audit.json')
    ch = overlap.get('changed_block_minus_heldout_clean_rows')
    if ch is None and 'comparisons' in overlap:
        ch = overlap['comparisons'].get('candidate_changed_block_only_minus_heldout_clean_rows_replaced_by_changed_block')
    if ch is None:
        ch = {}
    result = {
        'status': 'REINVEST_SOTA_ANATOMY',
        'method': 'CPU-only parsing of existing full evaluation predictions/reports plus A01 verification records; no new GPU work.',
        'inputs': {
            'clean_qwen_per_target': str(MODELS['clean_qwen']['per_target']),
            'compact_view_reinvest_per_target': str(MODELS['compact_view_reinvest']['per_target']),
            'full_summary': str(FULL_SUMMARY),
            'sota_verification': str(SOTA_VERIFY),
            'compact_core_anatomy': str(CORE),
        },
        'arithmetic': arithmetic(),
        'supplement': analyze_supplement(),
        'globalpiqa': analyze_globalpiqa(),
        'superglue': analyze_superglue(),
        'verification_excerpt': {
            'training_hash_ok': verify['training_provenance']['hashes']['train_command_hash_ok'] and verify['training_provenance']['hashes']['recomputed_100M_matches_train_command'],
            'exact_10M': verify['training_provenance']['corpus_10M_counts']['exact_10M_words'],
            'aoa_required_present': verify['checkpoint_and_aoa_status']['required_steps_present_in_train_metrics'],
            'missing_columns': verify['returncode_and_column_status']['missing_columns'],
            'unexpected_missing_artifacts': verify['artifact_status']['missing_or_unexpected_empty_artifacts'],
            'changed_minus_heldout_score_bearing_rows': ch.get('rows_with_score_bearing_overlap'),
            'changed_minus_heldout_glue_valid_rows': ch.get('rows_with_glue_valid_overlap'),
            'changed_minus_heldout_unique_ngrams': ch.get('unique_matching_ngrams'),
        },
    }
    out_json = OUT_DIR / 'reinvest_sota_anatomy.json'
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    NOTE.write_text(make_note(result), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'json': str(out_json), 'note': str(NOTE)}, indent=2))


if __name__ == '__main__':
    main()
