#!/usr/bin/env python3
"""Interpret the research/74 legal40k true 12x384 depth endpoint when ready.

CPU-only.  This script never launches training or evaluation and never infers a
missing score.  It waits for the depth pristine collation summary, validates the
same official-coordinate invariants used for research legal40k, then compares the
single seed43022 vector against:
  * research legal40k 8x480 fixed-WWM seed43022 (matched tokenizer/data/seed),
  * research/58 legal16k fixed-WWM seed43022,
  * inherited non-submission 16k seed43022 mechanism coordinate,
  * the visible 41.80 leader vector.

Scientific purpose: decide whether the unmeasured leader-shape depth-over-width
factor repairs the legal40k GlobalPIQA/Entity/relational deficits enough to merit
seed43122 reproduction or later combination with masking/sequence factors.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/legal40k_12x384_depth_endpoint_interpretation'
OUT_JSON = OUT_DIR / 'legal40k_12x384_depth_endpoint_interpretation.json'
OUT_CSV = OUT_DIR / 'legal40k_12x384_depth_column_comparison.csv'
OUT_DETAIL_CSV = OUT_DIR / 'legal40k_12x384_depth_detail_comparison.csv'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/legal40k_12x384_depth_endpoint_interpretation.md')

LEADER_VECTOR = {
    'BLiMP': 67.20,
    'Supplement': 56.01,
    'EWoK': 56.07,
    'Entity': 28.45,
    'COMPS': 53.57,
    'SuperGLUE': 69.79,
    'GlobalPIQA': 39.67,
    'Reading': 5.42,
    'AoA': 0.0,
    'Overall': 41.80,
}
TASKS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA', 'Overall']
SUPERGLUE_TASKS = ['boolq', 'mnli', 'mrpc', 'multirc', 'qqp', 'rte', 'wsc']
EXPECTED_GLOBALPIQA = {
    'global_piqa_parallel': {'sha256': 'cb9513ed5096ad8115becbe42fdfc97711980f519ada87808e3ddc5f8d5e4453', 'num_lines': 103},
    'global_piqa_nonparallel': {'sha256': 'aeec831d3adb09bf268ce1b847a854d105b3922c440a164a2c5893b96b228a8f', 'num_lines': 100},
}

SUMMARIES = {
    'depth12x384_43022': WS / 'data/legal40k_12x384_depth_seed43022_pristine_collate/pristine_collate_legal40k_12x384_depth_seed43022_summary.json',
    'legal40k8x480_43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    'legal16k8x480_43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json',
    'inherited16k_non_submission_43022': WS / 'data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json',
}
CONTROLLER_SUMMARY = WS / 'data/legal40k_12x384_depth_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json'
TRAIN_RESULT = WS / 'data/legal40k_12x384_depth_training/legal40k_12x384_depth_train_result_seed43022.json'
TRAIN_METRICS = WS / 'training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/scientific_metrics.json'


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def finite(v: Any) -> float | None:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def vector_candidates(rec: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    def add(scores_obj: Any, top_obj: Any | None = None) -> None:
        if not isinstance(scores_obj, dict):
            return
        cand = dict(scores_obj)
        if isinstance(top_obj, dict):
            for k in ['Overall', 'NLP_average', 'Human_like_average', 'margin_over_visible_leader_41p8', 'margin_over_visible_leader']:
                if k in top_obj and k not in cand:
                    cand[k] = top_obj[k]
        out.append(cand)
    if isinstance(rec, dict):
        ss = rec.get('score_summary')
        if isinstance(ss, dict):
            oo = ss.get('official_overall')
            if isinstance(oo, dict):
                add(oo.get('scores'), oo)
                add(oo, oo)
            add(ss.get('scores'), ss)
            add(ss, ss)
        add(rec.get('scores'), rec)
        add(rec, rec)
    return out


def extract_vector(path: Path) -> dict[str, float]:
    rec = load_json(path)
    for cand in vector_candidates(rec):
        vec = {}
        for k in TASKS:
            x = finite(cand.get(k))
            if x is not None:
                vec[k] = x
        if all(k in vec for k in TASKS):
            return vec
    raise RuntimeError({'cannot_extract_vector': rel(path)})


def score_summary(path: Path) -> dict[str, Any]:
    rec = load_json(path)
    ss = rec.get('score_summary') if isinstance(rec, dict) else None
    return ss if isinstance(ss, dict) else {}


def details(path: Path) -> dict[str, Any]:
    ss = score_summary(path)
    d = ss.get('details')
    return d if isinstance(d, dict) else {}


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in TASKS if k in a and k in b}


def top_abs(d: dict[str, float], n: int = 12) -> list[dict[str, Any]]:
    return [{'key': k, 'delta': d[k]} for k in sorted(d, key=lambda k: abs(d[k]), reverse=True)[:n]]


def count_fraction_scores(entity_detail: dict[str, Any]) -> dict[str, float]:
    out = {}
    if not isinstance(entity_detail, dict):
        return out
    frac = entity_detail.get('subtask_fraction_scores')
    if isinstance(frac, dict):
        for k, v in frac.items():
            x = finite(v)
            if x is not None:
                out[k] = 100.0 * x
        return out
    for k, v in entity_detail.items():
        if isinstance(v, dict) and v.get('total'):
            out[k] = 100.0 * float(v.get('correct', 0)) / float(v['total'])
    return out


def superglue_scores(d: dict[str, Any]) -> dict[str, float]:
    sg = d.get('SuperGLUE') if isinstance(d, dict) else None
    out = {}
    if isinstance(sg, dict):
        for task, rec in sg.items():
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[task] = float(rec['score'])
    return out


def globalpiqa_scores(d: dict[str, Any]) -> dict[str, float]:
    gp = d.get('GlobalPIQA') if isinstance(d, dict) else None
    out = {}
    if isinstance(gp, dict):
        for k, rec in gp.items():
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[k] = float(rec['score'])
    return out


def detail_scores(path: Path) -> dict[str, dict[str, float]]:
    d = details(path)
    return {
        'Entity': count_fraction_scores(d.get('Entity') if isinstance(d, dict) else {}),
        'SuperGLUE': superglue_scores(d),
        'GlobalPIQA': globalpiqa_scores(d),
    }


def detail_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    labels = [k for k in paths if paths[k].exists()]
    scored = {label: detail_scores(path) for label, path in paths.items() if path.exists()}
    rows = []
    families = sorted({fam for v in scored.values() for fam in v})
    for fam in families:
        keys = sorted({key for v in scored.values() for key in v.get(fam, {})})
        for key in keys:
            row: dict[str, Any] = {'family': fam, 'subtask': key}
            for label in labels:
                row[label] = scored[label].get(fam, {}).get(key)
            if 'depth12x384_43022' in labels and 'legal40k8x480_43022' in labels:
                a = row.get('depth12x384_43022')
                b = row.get('legal40k8x480_43022')
                row['delta_depth_minus_legal40k8x480'] = None if a is None or b is None else a - b
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def validate_depth_summary(path: Path) -> list[str]:
    rec = load_json(path)
    errors: list[str] = []
    if rec.get('status') != 'PRISTINE_COLLATE':
        errors.append(f'unexpected collation status {rec.get("status")}')
    if rec.get('validation_errors') not in ([], None):
        errors.append(f'validation_errors={rec.get("validation_errors")}')
    val = rec.get('validation') or {}
    if isinstance(val, dict) and val.get('errors') not in ([], None):
        errors.append(f'validation.errors={val.get("errors")}')
    cs = rec.get('collated_summary') or {}
    if sorted(cs.get('null_keys') or []) != []:
        errors.append(f'null collated keys {cs.get("null_keys")}')
    if sum((cs.get('ewok_lengths') or {}).values()) != 7618:
        errors.append(f'EWoK count {sum((cs.get("ewok_lengths") or {}).values())} != 7618')
    if cs.get('aoa_surprisal_num_steps') != 19:
        errors.append(f'AoA steps {cs.get("aoa_surprisal_num_steps")} != 19')
    if cs.get('aoa_surprisal_row_count_values') != [8005]:
        errors.append(f'AoA row-count values {cs.get("aoa_surprisal_row_count_values")} != [8005]')
    if finite(cs.get('aoa_value')) is None:
        errors.append('missing finite AoA value')
    vec = extract_vector(path)
    for k in TASKS:
        if finite(vec.get(k)) is None:
            errors.append(f'missing/nonfinite score {k}')
    det = details(path)
    gp = det.get('GlobalPIQA') if isinstance(det, dict) else None
    for task_name, exp in EXPECTED_GLOBALPIQA.items():
        g = gp.get(task_name) if isinstance(gp, dict) else None
        if not isinstance(g, dict):
            errors.append(f'missing GlobalPIQA detail {task_name}')
        elif g.get('sha256') != exp['sha256'] or g.get('num_lines') != exp['num_lines']:
            errors.append(f'GlobalPIQA {task_name} target {g.get("sha256")}/{g.get("num_lines")} != expected {exp}')
    sg = det.get('SuperGLUE') if isinstance(det, dict) else None
    for task in SUPERGLUE_TASKS:
        if not isinstance(sg, dict) or task not in sg or finite((sg.get(task) or {}).get('score')) is None:
            errors.append(f'missing SuperGLUE detail {task}')
    return errors


def build_waiting(missing: list[str]) -> dict[str, Any]:
    train_result = load_json(TRAIN_RESULT) if TRAIN_RESULT.exists() else None
    train_metrics = load_json(TRAIN_METRICS) if TRAIN_METRICS.exists() else None
    return {
        'status': 'WAITING_FOR_LEGAL40K_12X384_DEPTH_COLLATION',
        'created_utc': now_utc(),
        'missing_collation_summaries': missing,
        'active_or_expected_tasks': {
            'training_task': 's74_t5_tool1 legal40k 12x384 depth seed43022 training',
            'training_result_path': rel(TRAIN_RESULT),
            'expected_eval_controller': rel(WS / 'scripts/legal40k_12x384_depth_posttrain_eval_controller.py'),
        },
        'train_result_seen': train_result,
        'train_metrics_seen': train_metrics,
        'script_purpose': 'Run after the depth hardened official collation exists; no score is inferred from missing outputs.',
        'files': {'json': rel(OUT_JSON), 'note': rel(OUT_MD)},
    }


def ready_payload() -> dict[str, Any]:
    paths = SUMMARIES
    vectors = {label: extract_vector(path) for label, path in paths.items() if path.exists()}
    depth = vectors['depth12x384_43022']
    deltas = {
        'depth_minus_legal40k8x480_43022': diff(depth, vectors['legal40k8x480_43022']),
        'depth_minus_legal16k8x480_43022': diff(depth, vectors['legal16k8x480_43022']),
        'depth_minus_inherited16k_non_submission_43022': diff(depth, vectors['inherited16k_non_submission_43022']),
        'depth_minus_visible_leader': diff(depth, LEADER_VECTOR),
    }
    integrity_errors = validate_depth_summary(paths['depth12x384_43022'])
    train_result = load_json(TRAIN_RESULT) if TRAIN_RESULT.exists() else None
    train_metrics = load_json(TRAIN_METRICS) if TRAIN_METRICS.exists() else None
    controller = load_json(CONTROLLER_SUMMARY) if CONTROLLER_SUMMARY.exists() else None
    col_rows = []
    for k in TASKS:
        row = {'task': k}
        for label, vec in vectors.items():
            row[label] = vec[k]
        row['visible_leader'] = LEADER_VECTOR[k]
        for name, dv in deltas.items():
            row[name] = dv[k]
        col_rows.append(row)
    write_csv(OUT_CSV, col_rows)
    dr = detail_rows(paths)
    if dr:
        write_csv(OUT_DETAIL_CSV, dr)
    reading = []
    if integrity_errors:
        reading.append('Depth collation has integrity errors; repair official-coordinate artifacts before interpreting the endpoint.')
    else:
        reading.append('Depth collation integrity checks passed on the same hardened official coordinate used for legal40k.')
    margin = depth['Overall'] - LEADER_VECTOR['Overall']
    matched_delta = deltas['depth_minus_legal40k8x480_43022']['Overall']
    if depth['Overall'] >= LEADER_VECTOR['Overall']:
        reading.append('The single depth seed clears the visible leader; protect the endpoint and reproduce seed43122 before any mechanism combination.')
    elif depth['Overall'] >= 41.55 or matched_delta >= 0.35:
        reading.append('The depth seed is close enough or improves enough over matched legal40k 8x480 to justify seed43122 reproduction or a focused combination, depending on the column pattern.')
    elif matched_delta <= -0.10:
        reading.append('Depth-over-width did not improve the matched legal40k seed; do not spend a second seed on depth alone.')
    else:
        reading.append('Depth is roughly flat versus matched legal40k; use column-level movement and WWM/A02 evidence before deciding reproduction.')
    if deltas['depth_minus_legal40k8x480_43022']['GlobalPIQA'] > 0 and deltas['depth_minus_legal40k8x480_43022']['Entity'] > 0:
        reading.append('Depth repairs the two main legal40k preserve-column losses for seed43022, making architecture a live mechanism even if Overall remains below frontier.')
    if deltas['depth_minus_legal40k8x480_43022']['EWoK'] > 0 and deltas['depth_minus_legal40k8x480_43022']['COMPS'] > 0:
        reading.append('Depth also helps relational/structural columns; relation-bias may be postponed until after reproduction evidence.')
    return {
        'status': 'LEGAL40K_12X384_DEPTH_ENDPOINT_READY' if not integrity_errors else 'LEGAL40K_12X384_DEPTH_ENDPOINT_HAS_INTEGRITY_ERRORS',
        'created_utc': now_utc(),
        'visible_leader': LEADER_VECTOR,
        'integrity_errors': integrity_errors,
        'vectors': vectors,
        'deltas': deltas,
        'largest_abs_delta_vs_legal40k8x480_43022': top_abs(deltas['depth_minus_legal40k8x480_43022'], 10),
        'margin_over_visible_leader_41p8': margin,
        'train_result': train_result,
        'train_metrics': train_metrics,
        'controller_summary_status': controller.get('status') if isinstance(controller, dict) else None,
        'controller_summary_path': rel(CONTROLLER_SUMMARY),
        'scientific_reading': reading,
        'files': {'json': rel(OUT_JSON), 'column_csv': rel(OUT_CSV), 'detail_csv': rel(OUT_DETAIL_CSV), 'note': rel(OUT_MD)},
    }


def write_md(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append('# research legal40k 12x384 depth endpoint interpretation\n\n')
    lines.append(f"Status: `{payload['status']}`\n\n")
    if payload['status'] == 'WAITING_FOR_LEGAL40K_12X384_DEPTH_COLLATION':
        lines.append('The depth official collation is not present. No score is inferred from missing outputs.\n\n')
        lines.append('Missing summaries:\n')
        for p in payload.get('missing_collation_summaries') or []:
            lines.append(f'- `{p}`\n')
        lines.append('\n')
    else:
        vec = payload['vectors']['depth12x384_43022']
        lines.append(f"Overall: `{vec['Overall']:.6f}`; margin vs 41.80 `{payload['margin_over_visible_leader_41p8']:+.6f}`\n\n")
        lines.append('## Matched depth movement vs legal40k 8x480 seed43022\n')
        d = payload['deltas']['depth_minus_legal40k8x480_43022']
        for k in TASKS:
            lines.append(f'- {k}: `{d[k]:+.6f}`\n')
        lines.append('\n## Integrity\n')
        if payload['integrity_errors']:
            for e in payload['integrity_errors']:
                lines.append(f'- ERROR: {e}\n')
        else:
            lines.append('- Hardened collation integrity checks passed.\n')
        lines.append('\n## Scientific reading\n')
        for item in payload['scientific_reading']:
            lines.append(f'- {item}\n')
    lines.append('\n## Files\n')
    for k, v in (payload.get('files') or {}).items():
        lines.append(f'- {k}: `{v}`\n')
    OUT_MD.write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [rel(SUMMARIES['depth12x384_43022'])] if not SUMMARIES['depth12x384_43022'].exists() else []
    refs_missing = [rel(p) for k, p in SUMMARIES.items() if k != 'depth12x384_43022' and not p.exists()]
    if missing or refs_missing:
        payload = build_waiting(missing + refs_missing)
    else:
        payload = ready_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_md(payload)
    print(json.dumps({
        'status': payload['status'],
        'missing': payload.get('missing_collation_summaries'),
        'depth_overall': ((payload.get('vectors') or {}).get('depth12x384_43022') or {}).get('Overall'),
        'margin_over_visible_leader_41p8': payload.get('margin_over_visible_leader_41p8'),
        'integrity_errors': payload.get('integrity_errors'),
        'out_json': rel(OUT_JSON),
        'note': rel(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
