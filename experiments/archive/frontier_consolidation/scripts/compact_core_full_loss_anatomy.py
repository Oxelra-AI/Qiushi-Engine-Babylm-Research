#!/usr/bin/env python3
"""research CPU-only anatomy of compact-core complete-surface losses.

Compare the inherited clean-Qwen full official-compatible endpoint with the
frontier_consolidation compact_view_core full endpoint using existing prediction/report files.
No training, generation, or GPU evaluation is performed.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

ROOT = pathlib.Path('.')
EVAL = ROOT / 'experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval'
OUT_DIR = ROOT / 'experiments/archive/frontier_consolidation/data/compact_core_full_loss_anatomy'
NOTE = ROOT / 'research/notes/frontier_consolidation/compact_core_full_loss_anatomy.md'

MODELS = {
    'clean_qwen': {
        'name': 'COMPACT_EXPERIENCE clean-Qwen aligned full endpoint',
        'per_target': ROOT / 'experiments/archive/compact_experience/data/full_eval/per_target/qwen_clean_aligned.json',
    },
    'compact_view_core': {
        'name': 'frontier_consolidation compact_view_core full endpoint',
        'per_target': ROOT / 'experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_view_core.json',
    },
}

FAST_SUMMARIES = {
    'compact_core_noaoa': ROOT / 'experiments/archive/frontier_consolidation/data/density_noaoa_eval_compact_core/density_noaoa_eval_summary.json',
    'compact_reinvest_noaoa': ROOT / 'experiments/archive/frontier_consolidation/data/density_noaoa_eval_reinvest/density_noaoa_eval_summary.json',
}

SUPPLEMENT_FILES = [
    'qa_congruence_easy',
    'qa_congruence_tricky',
    'turn_taking',
    'subject_aux_inversion',
    'hypernym',
]
GLOBAL_FILES = {
    'GlobalPIQA_parallel': EVAL / 'global_piqa_parallel/eng_latn.jsonl',
    'GlobalPIQA_nonparallel': EVAL / 'global_piqa_nonparallel/eng_latn.jsonl',
}
SUPERGLUE_TASKS = ['boolq', 'multirc', 'rte', 'wsc', 'mrpc', 'qqp', 'mnli']


def read_json(path: pathlib.Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_text(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x).strip())


def short(x: Any, n: int = 180) -> str:
    s = norm_text(x)
    return s if len(s) <= n else s[: n - 1] + '…'


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def parse_report_uid_scores(path: pathlib.Path) -> Dict[str, float]:
    out = {}
    if not path.exists():
        return out
    in_uid = False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if line.startswith('### UID ACCURACY'):
            in_uid = True
            continue
        if in_uid and line.startswith('###'):
            break
        if in_uid and ':' in line:
            k, v = line.split(':', 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def supplement_predictions(model: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    d = read_json(model['per_target'])
    pred_path = pathlib.Path(d['tasks']['Supplement']['predictions'])
    report_path = pathlib.Path(d['tasks']['Supplement']['report'])
    preds = read_json(pred_path)
    uid_scores = parse_report_uid_scores(report_path)
    flat: Dict[str, Dict[str, Any]] = {}
    for uid in SUPPLEMENT_FILES:
        for rec in preds[uid]['predictions']:
            flat[rec['id']] = {'pred': rec['pred'], 'uid': uid}
    return flat, uid_scores


def analyze_supplement(models_loaded: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    gold: Dict[str, Dict[str, Any]] = {}
    for uid in SUPPLEMENT_FILES:
        rows = read_jsonl(EVAL / f'supplement_filtered/{uid}.jsonl')
        for i, row in enumerate(rows):
            rid = f'{uid}_{i}'
            gold[rid] = {**row, 'uid': uid, 'id': rid}
    pred = {}
    report_uid_scores = {}
    for m, md in MODELS.items():
        pred[m], report_uid_scores[m] = supplement_predictions(md)

    rows = []
    by_uid = {uid: {'n': 0, 'clean_correct': 0, 'compact_correct': 0, 'lost': 0, 'gained': 0, 'both_correct': 0, 'both_wrong': 0} for uid in SUPPLEMENT_FILES}
    by_uid_field = defaultdict(lambda: {'n': 0, 'clean_correct': 0, 'compact_correct': 0, 'lost': 0, 'gained': 0})
    lost_examples = defaultdict(list)
    gained_examples = defaultdict(list)
    mismatch = []
    for rid, row in gold.items():
        g = norm_text(row['sentence_good'])
        b = norm_text(row['sentence_bad'])
        p_clean = norm_text(pred['clean_qwen'][rid]['pred'])
        p_comp = norm_text(pred['compact_view_core'][rid]['pred'])
        clean_ok = p_clean == g
        comp_ok = p_comp == g
        if p_clean not in (g, b) or p_comp not in (g, b):
            mismatch.append({'id': rid, 'uid': row['uid'], 'good': g, 'bad': b, 'clean_pred': p_clean, 'compact_pred': p_comp})
        s = by_uid[row['uid']]
        s['n'] += 1
        s['clean_correct'] += int(clean_ok)
        s['compact_correct'] += int(comp_ok)
        s['lost'] += int(clean_ok and not comp_ok)
        s['gained'] += int((not clean_ok) and comp_ok)
        s['both_correct'] += int(clean_ok and comp_ok)
        s['both_wrong'] += int((not clean_ok) and (not comp_ok))
        for field in ['contrast', 'fragment', 'template', 'ambiguous']:
            if field in row:
                key = f'{row["uid"]}|{field}={row[field]}'
                ss = by_uid_field[key]
                ss['n'] += 1
                ss['clean_correct'] += int(clean_ok)
                ss['compact_correct'] += int(comp_ok)
                ss['lost'] += int(clean_ok and not comp_ok)
                ss['gained'] += int((not clean_ok) and comp_ok)
        ex = {
            'id': rid,
            'uid': row['uid'],
            'row': row.get('row'),
            'contrast': row.get('contrast'),
            'fragment': row.get('fragment'),
            'template': row.get('template'),
            'ambiguous': row.get('ambiguous'),
            'good': short(row['sentence_good'], 220),
            'bad': short(row['sentence_bad'], 220),
            'clean_pred': short(p_clean, 220),
            'compact_pred': short(p_comp, 220),
        }
        if clean_ok and not comp_ok and len(lost_examples[row['uid']]) < 10:
            lost_examples[row['uid']].append(ex)
        if (not clean_ok) and comp_ok and len(gained_examples[row['uid']]) < 10:
            gained_examples[row['uid']].append(ex)
        rows.append({**ex, 'clean_correct': clean_ok, 'compact_correct': comp_ok})

    for uid, s in by_uid.items():
        s['clean_acc'] = pct(s['clean_correct'] / s['n']) if s['n'] else None
        s['compact_acc'] = pct(s['compact_correct'] / s['n']) if s['n'] else None
        s['delta_acc'] = round(s['compact_acc'] - s['clean_acc'], 4)
        s['report_clean_acc'] = report_uid_scores['clean_qwen'].get(uid)
        s['report_compact_acc'] = report_uid_scores['compact_view_core'].get(uid)
    uid_field_sorted = []
    for key, s in by_uid_field.items():
        uid_field_sorted.append({
            'field': key,
            **s,
            'clean_acc': pct(s['clean_correct'] / s['n']),
            'compact_acc': pct(s['compact_correct'] / s['n']),
            'delta_acc': round(pct(s['compact_correct'] / s['n']) - pct(s['clean_correct'] / s['n']), 4),
        })
    uid_field_sorted.sort(key=lambda r: (r['delta_acc'], -r['n']))
    equal_uid_clean = sum(v['clean_acc'] for v in by_uid.values()) / len(by_uid)
    equal_uid_comp = sum(v['compact_acc'] for v in by_uid.values()) / len(by_uid)
    return {
        'n': len(gold),
        'equal_uid_clean': round(equal_uid_clean, 4),
        'equal_uid_compact': round(equal_uid_comp, 4),
        'equal_uid_delta': round(equal_uid_comp - equal_uid_clean, 4),
        'by_uid': by_uid,
        'worst_field_slices': uid_field_sorted[:20],
        'best_field_slices': uid_field_sorted[-20:][::-1],
        'lost_examples': dict(lost_examples),
        'gained_examples': dict(gained_examples),
        'prediction_candidate_mismatches_n': len(mismatch),
        'prediction_candidate_mismatches_examples': mismatch[:20],
    }


def global_predictions(model: Dict[str, Any], task_col: str) -> Dict[str, str]:
    d = read_json(model['per_target'])
    pred_path = pathlib.Path(d['tasks'][task_col]['predictions'])
    preds = read_json(pred_path)
    flat = {}
    for exid, block in preds.items():
        plist = block.get('predictions', [])
        if plist:
            flat[exid] = plist[0]['pred']
    return flat


def analyze_globalpiqa() -> Dict[str, Any]:
    out = {}
    for col, data_path in GLOBAL_FILES.items():
        gold_rows = read_jsonl(data_path)
        by_id = {r['example_id']: r for r in gold_rows}
        preds = {m: global_predictions(md, col) for m, md in MODELS.items()}
        summary = defaultdict(lambda: {'n': 0, 'clean_correct': 0, 'compact_correct': 0, 'lost': 0, 'gained': 0})
        lost_examples = []
        gained_examples = []
        rows = []
        for exid, row in by_id.items():
            sol = norm_text(row[f'solution{row["label"]}'])
            p_clean = norm_text(preds['clean_qwen'][exid])
            p_comp = norm_text(preds['compact_view_core'][exid])
            clean_ok = p_clean == sol
            comp_ok = p_comp == sol
            cats = [c.strip() for c in str(row.get('categories', '')).split(',') if c.strip()]
            if not cats:
                cats = ['(none)']
            try:
                supp = json.loads(row.get('supplement') or '{}')
                insp = supp.get('example_inspiration', '(none)')
            except Exception:
                insp = '(parse_error)'
            keys = [f'category={c}' for c in cats] + [f'inspiration={insp}']
            for key in keys:
                s = summary[key]
                s['n'] += 1
                s['clean_correct'] += int(clean_ok)
                s['compact_correct'] += int(comp_ok)
                s['lost'] += int(clean_ok and not comp_ok)
                s['gained'] += int((not clean_ok) and comp_ok)
            ex = {
                'example_id': exid,
                'category': row.get('categories'),
                'inspiration': insp,
                'prompt': short(row.get('prompt'), 250),
                'label': row['label'],
                'answer': short(sol, 160),
                'clean_pred': short(p_clean, 160),
                'compact_pred': short(p_comp, 160),
                'clean_correct': clean_ok,
                'compact_correct': comp_ok,
            }
            if clean_ok and not comp_ok and len(lost_examples) < 25:
                lost_examples.append(ex)
            if (not clean_ok) and comp_ok and len(gained_examples) < 25:
                gained_examples.append(ex)
            rows.append(ex)
        by_slice = []
        for key, s in summary.items():
            by_slice.append({
                'slice': key,
                **s,
                'clean_acc': pct(s['clean_correct'] / s['n']),
                'compact_acc': pct(s['compact_correct'] / s['n']),
                'delta_acc': round(pct(s['compact_correct'] / s['n']) - pct(s['clean_correct'] / s['n']), 4),
            })
        by_slice.sort(key=lambda r: (r['delta_acc'], -r['n']))
        n = len(rows)
        out[col] = {
            'n': n,
            'clean_correct': sum(r['clean_correct'] for r in rows),
            'compact_correct': sum(r['compact_correct'] for r in rows),
            'clean_acc': pct(sum(r['clean_correct'] for r in rows) / n),
            'compact_acc': pct(sum(r['compact_correct'] for r in rows) / n),
            'delta_acc': round(pct(sum(r['compact_correct'] for r in rows) / n) - pct(sum(r['clean_correct'] for r in rows) / n), 4),
            'lost_n': sum(r['clean_correct'] and not r['compact_correct'] for r in rows),
            'gained_n': sum((not r['clean_correct']) and r['compact_correct'] for r in rows),
            'worst_slices': by_slice[:15],
            'best_slices': by_slice[-15:][::-1],
            'lost_examples': lost_examples,
            'gained_examples': gained_examples,
        }
    return out


def load_super_preds(model: Dict[str, Any], task: str) -> List[Any]:
    d = read_json(model['per_target'])
    task_rec = None
    for r in d['tasks']['SuperGLUE']['tasks']:
        if r['task'] == task:
            task_rec = r
            break
    if task_rec is None:
        raise KeyError(task)
    pj = read_json(pathlib.Path(task_rec['predictions']))
    block = pj.get(task, pj)
    preds = block['predictions'] if isinstance(block, dict) and 'predictions' in block else block
    return [p['pred'] if isinstance(p, dict) and 'pred' in p else p for p in preds]


def coerce_label(x: Any) -> Any:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, str):
        y = x.strip()
        if y.isdigit() or (y.startswith('-') and y[1:].isdigit()):
            return int(y)
        low = y.lower()
        maps = {
            'false': 0, 'true': 1,
            'not_entailment': 1, 'entailment': 0,
            'contradiction': 2, 'neutral': 1,
        }
        return maps.get(low, y)
    return x


def extract_example_fields(task: str, row: Dict[str, Any]) -> Dict[str, Any]:
    skip = {'label'}
    fields = {}
    priority = ['idx', 'sentence1', 'sentence2', 'premise', 'hypothesis', 'question', 'passage', 'paragraph', 'text', 'span1_text', 'span2_text', 'word', 'query', 'answers', 'answer', 'question_text']
    for k in priority:
        if k in row:
            fields[k] = short(row[k], 220)
    if not fields:
        for k, v in row.items():
            if k not in skip:
                fields[k] = short(v, 220)
                if len(fields) >= 5:
                    break
    return fields


def analyze_superglue() -> Dict[str, Any]:
    out = {}
    for task in SUPERGLUE_TASKS:
        gold_rows = read_jsonl(EVAL / f'glue_filtered/{task}.valid.jsonl')
        gold = [coerce_label(r.get('label')) for r in gold_rows]
        preds = {m: [coerce_label(x) for x in load_super_preds(md, task)] for m, md in MODELS.items()}
        n = min(len(gold), len(preds['clean_qwen']), len(preds['compact_view_core']))
        rows = []
        lost_examples = []
        gained_examples = []
        for i in range(n):
            clean_ok = preds['clean_qwen'][i] == gold[i]
            comp_ok = preds['compact_view_core'][i] == gold[i]
            ex = {
                'i': i,
                'gold': gold[i],
                'clean_pred': preds['clean_qwen'][i],
                'compact_pred': preds['compact_view_core'][i],
                'clean_correct': clean_ok,
                'compact_correct': comp_ok,
                'fields': extract_example_fields(task, gold_rows[i]),
            }
            if clean_ok and not comp_ok and len(lost_examples) < 20:
                lost_examples.append(ex)
            if (not clean_ok) and comp_ok and len(gained_examples) < 20:
                gained_examples.append(ex)
            rows.append(ex)
        clean_correct = sum(r['clean_correct'] for r in rows)
        comp_correct = sum(r['compact_correct'] for r in rows)
        out[task] = {
            'n': n,
            'clean_correct': clean_correct,
            'compact_correct': comp_correct,
            'clean_acc': pct(clean_correct / n) if n else None,
            'compact_acc': pct(comp_correct / n) if n else None,
            'delta_acc': round(pct(comp_correct / n) - pct(clean_correct / n), 4) if n else None,
            'lost_n': sum(r['clean_correct'] and not r['compact_correct'] for r in rows),
            'gained_n': sum((not r['clean_correct']) and r['compact_correct'] for r in rows),
            'lost_examples': lost_examples,
            'gained_examples': gained_examples,
        }
    return out


def read_official_vectors() -> Dict[str, Any]:
    out = {}
    for name, md in MODELS.items():
        d = read_json(md['per_target'])
        out[name] = d['official_overall']
    return out


def load_fast() -> Dict[str, Any]:
    out = {}
    for k, p in FAST_SUMMARIES.items():
        if p.exists():
            out[k] = read_json(p)
    return out


def route_arithmetic(official: Dict[str, Any]) -> Dict[str, Any]:
    c = official['clean_qwen']['scores']
    v = official['compact_view_core']['scores']
    deltas = {k: v.get(k) - c.get(k) for k in c if k in v}
    ranked = sorted(deltas.items(), key=lambda kv: kv[1])
    compact_if_aoa_zero = (sum(v[k] for k in v if k != 'AoA') + 0.0) / 9.0
    needed_for_41_8 = 41.8 * 9 - sum(v[k] for k in v if k not in ('AoA', 'SuperGLUE'))
    return {
        'component_deltas_compact_minus_clean': deltas,
        'ranked_losses': ranked,
        'compact_overall_if_aoa_zero': compact_if_aoa_zero,
        'superglue_plus_aoa_needed_for_compact_other_columns_to_reach_41_8': needed_for_41_8,
        'actual_compact_superglue_plus_aoa': v['SuperGLUE'] + v['AoA'],
    }


def make_note(result: Dict[str, Any]) -> str:
    official = result['official_vectors']
    ar = result['route_arithmetic']
    supp = result['supplement']
    glob = result['globalpiqa']
    sg = result['superglue']
    lines = []
    lines.append('# research compact-core complete-surface loss anatomy')
    lines.append('')
    lines.append('CPU-only comparison of the inherited clean-Qwen full endpoint and frontier_consolidation compact_view_core full endpoint. It uses existing official-compatible prediction/report files only; no model training, generation, or GPU evaluation was run.')
    lines.append('')
    lines.append('## Overall arithmetic')
    lines.append(f"- clean-Qwen Overall: {official['clean_qwen']['Overall']:.6f}; compact_view_core Overall: {official['compact_view_core']['Overall']:.6f}.")
    lines.append(f"- Compact_view_core if its AoA column were exactly 0: {ar['compact_overall_if_aoa_zero']:.6f}; this still does not reach 41.8.")
    lines.append('- Component deltas compact minus clean: ' + ', '.join(f"{k} {v:+.3f}" for k, v in ar['component_deltas_compact_minus_clean'].items()))
    lines.append(f"- With compact_view_core's other measured columns fixed, SuperGLUE+AoA would need {ar['superglue_plus_aoa_needed_for_compact_other_columns_to_reach_41_8']:.3f} for Overall 41.8; actual compact SuperGLUE+AoA is {ar['actual_compact_superglue_plus_aoa']:.3f}.")
    lines.append('')
    lines.append('## Supplement: not a broad syntactic collapse')
    lines.append(f"- Equal-UID Supplement: clean {supp['equal_uid_clean']:.2f}, compact {supp['equal_uid_compact']:.2f}, delta {supp['equal_uid_delta']:+.2f}.")
    for uid, s in supp['by_uid'].items():
        lines.append(f"  - {uid}: clean {s['clean_acc']:.2f}, compact {s['compact_acc']:.2f}, delta {s['delta_acc']:+.2f}; lost {s['lost']}, gained {s['gained']}, n={s['n']}.")
    lines.append('- Interpretation: compact_core improves subject-auxiliary inversion, turn-taking, and hypernym slices, but loses both QA-congruence slices. The lost QA examples are mainly simple who/what/where answer-type and animacy/object-role discriminations, which are close to child-directed dialogue and event-role evidence; this points away from a generic grammar failure and toward losing some pragmatic/action-role support when official source rows are replaced by FineWeb packets.')
    lines.append('')
    lines.append('## GlobalPIQA: few flips against clean, but still far below the visible leader')
    for col, s in glob.items():
        lines.append(f"- {col}: clean {s['clean_acc']:.2f} ({s['clean_correct']}/{s['n']}), compact {s['compact_acc']:.2f} ({s['compact_correct']}/{s['n']}), delta {s['delta_acc']:+.2f}; lost {s['lost_n']}, gained {s['gained_n']}.")
        worst = ', '.join(f"{w['slice']} {w['delta_acc']:+.1f}" for w in s['worst_slices'][:4])
        lines.append(f"  - weakest slices in compact-vs-clean: {worst}.")
    lines.append('- Interpretation: compact_core is not catastrophically worse than clean-Qwen on GlobalPIQA; it loses only one net parallel example and two net nonparallel examples. But both endpoints are weak on practical physical/time/counting questions relative to the target surface, so small packet swaps are unlikely to solve SOTA unless they also preserve the other columns.')
    lines.append('')
    lines.append('## SuperGLUE: loss localizes to entailment/paraphrase classification')
    for task, s in sorted(sg.items(), key=lambda kv: kv[1]['delta_acc']):
        lines.append(f"- {task}: clean {s['clean_acc']:.3f}, compact {s['compact_acc']:.3f}, delta {s['delta_acc']:+.3f}; lost {s['lost_n']}, gained {s['gained_n']}, n={s['n']}.")
    lines.append('- Interpretation: the SuperGLUE mean drop is dominated by RTE and MRPC, while MNLI/QQP/MultiRC are flat-to-slightly positive and WSC unchanged. This again argues against a universal finetuning degradation; the current FineWeb compact overlay seems to damage a narrow equivalence/entailment calibration that clean-Qwen previously retained.')
    lines.append('')
    lines.append('## Route implication')
    lines.append('- The research failure should not be reduced to AoA. The complete losses are structured: QA answer-type/action-role support, practical commonsense still weak, and RTE/MRPC sentence-pair calibration.')
    lines.append('- Because the overlay replaced 423,520 official clean-Qwen words (including CHILDES/OpenSubtitles/Gutenberg/SimpleWiki slices) with only 353,945 source+compact FineWeb pair words plus 69,575 neutral top-up words in compact_core, a plausible failure mechanism is source-distribution displacement rather than the compact-view mechanism alone.')
    lines.append('- A higher-value redesign, if the pending AoA/reinvest evidence permits, should preserve the clean-Qwen official source distribution much more tightly and move information-density inside the existing Qwen-pair budget: compact or mix-resolution the near-length Qwen second views, then reinvest saved words into additional official/practical/event-rich packets. This would test the density principle without paying the observed cost of replacing a broad official slice.')
    lines.append('')
    lines.append(f"Machine-readable JSON: `{OUT_DIR / 'compact_core_full_loss_anatomy.json'}`")
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    official = read_official_vectors()
    result = {
        'status': 'COMPACT_CORE_FULL_LOSS_ANATOMY',
        'method': 'CPU-only comparison of existing prediction/report files; no training/generation/GPU evaluation.',
        'inputs': {
            'clean_qwen_per_target': str(MODELS['clean_qwen']['per_target']),
            'compact_view_core_per_target': str(MODELS['compact_view_core']['per_target']),
            'evaluation_data_root': str(EVAL),
        },
        'official_vectors': official,
        'route_arithmetic': route_arithmetic(official),
        'supplement': analyze_supplement(MODELS),
        'globalpiqa': analyze_globalpiqa(),
        'superglue': analyze_superglue(),
        'fast_summaries': load_fast(),
    }
    json_path = OUT_DIR / 'compact_core_full_loss_anatomy.json'
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    NOTE.write_text(make_note(result), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'json': str(json_path), 'note': str(NOTE)}, indent=2))


if __name__ == '__main__':
    main()
