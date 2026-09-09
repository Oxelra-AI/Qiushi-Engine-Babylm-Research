#!/usr/bin/env python3
"""Interpret the exact-prefix SGCR(K=50,d=64) endpoint when official collation exists.

CPU-only. This script never launches training/evaluation and never infers a
missing score. It reads the managed SGCR posttrain outputs only from files, then
compares the full official-compatible nine-column endpoint against:
  * the matched legal40k 12x384 depth seed43022 endpoint,
  * the shallower legal40k 8x480 seed43022 endpoint,
  * the legal16k 8x480 seed43022 endpoint,
  * the inherited non-submission 16k seed43022 mechanism coordinate,
  * the visible public 41.80 leader vector,
  * the research exact-prefix SGCR official-span burden map.

Scientific purpose: when the pending managed endpoint arrives, decide whether
support-gated component sharing expresses the compact-view mechanism under legal
40k tokenization, whether any improvement is on the measured support-sharing
mechanism (especially COMPS and GlobalPIQA), or whether SGCR should be closed or
reframed before spending another full run.
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
OUT_DIR = WS / 'data/sgcr_endpoint_interpretation'
OUT_JSON = OUT_DIR / 'sgcr_endpoint_interpretation.json'
OUT_CSV = OUT_DIR / 'sgcr_endpoint_column_comparison.csv'
OUT_DETAIL_CSV = OUT_DIR / 'sgcr_endpoint_detail_comparison.csv'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/sgcr_endpoint_interpretation.md')

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
    'sgcr12x384_k50d64_43022': WS / 'data/legal40k_12x384_sgcrK50d64_seed43022_pristine_collate/pristine_collate_legal40k_12x384_sgcrK50d64_seed43022_summary.json',
    'depth12x384_43022': WS / 'data/legal40k_12x384_depth_seed43022_pristine_collate/pristine_collate_legal40k_12x384_depth_seed43022_summary.json',
    'legal40k8x480_43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    'legal16k8x480_43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json',
    'inherited16k_non_submission_43022': WS / 'data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json',
}
CONTROLLER_SUMMARY = WS / 'data/legal40k_12x384_sgcrK50d64_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json'
TRAIN_RUN_DIR = WS / 'training/runs/legal40k_12x384_sgcrK50d64_compact_view_reinvest_seed43022'
TRAIN_METRICS = TRAIN_RUN_DIR / 'scientific_metrics.json'
TRAIN_LOG = TRAIN_RUN_DIR / 'training_log.jsonl'
EXAMPLE_ORDER_MANIFEST = TRAIN_RUN_DIR / 'example_order_manifest.json'
SGCR_DIAG = TRAIN_RUN_DIR / 'sgcr_diagnostic.json'
BURDEN_JSON = WS / 'data/sgcr_official_span_burden/sgcr_official_span_burden.json'

EXPECTED_SGCR = {
    'source_tokenizer_sha256': '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758',
    'component_tokenizer_sha256': '4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738',
    'train_stream_sha256': '3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691',
    'decomposition_map_sha256': 'b450cc45f7d66564a894fb8cc12e0b0ab8e3ba7db0339cad5eec6f30c6edfd8b',
    'base_param_count': 38_421_952,
    'sgcr_new_params': 1_073_536,
    'total_param_count': 39_495_488,
    'word_exposure': 100_000_000,
    'total_steps': 2529,
    'sgcr_K': 50.0,
    'sgcr_d_comp': 64,
    'sgcr_decomposition_max_components': 7,
    'sgcr_decomposition_component_slots': 68_660,
    'sgcr_pool_count_total_tokens': 13_942_644,
    'sgcr_pool_count_used_types': 39_320,
    'gradient_clip_norm': 1.0,
    'train_rng_restored_after_sgcr': True,
    'zero_sharing_verified': True,
    'sgcr_exact_prefix_decomposition': True,
    'sgcr_uniform_gate': False,
}

BURDEN_FAMILIES = ['COMPS', 'GlobalPIQA', 'GlobalPIQA_parallel', 'GlobalPIQA_nonparallel', 'Entity', 'EWoK']


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


def load_json_if_exists(path: Path) -> Any:
    return load_json(path) if path.exists() else None


def file_rec(path: Path) -> dict[str, Any]:
    return {'path': rel(path), 'exists': path.exists(), 'size_bytes': path.stat().st_size if path.exists() else None}


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
    out: dict[str, float] = {}
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
    out: dict[str, float] = {}
    if isinstance(sg, dict):
        for task, rec in sg.items():
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[task] = float(rec['score'])
    return out


def globalpiqa_scores(d: dict[str, Any]) -> dict[str, float]:
    gp = d.get('GlobalPIQA') if isinstance(d, dict) else None
    out: dict[str, float] = {}
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
    rows: list[dict[str, Any]] = []
    families = sorted({fam for v in scored.values() for fam in v})
    for fam in families:
        keys = sorted({key for v in scored.values() for key in v.get(fam, {})})
        for key in keys:
            row: dict[str, Any] = {'family': fam, 'subtask': key}
            for label in labels:
                row[label] = scored[label].get(fam, {}).get(key)
            if 'sgcr12x384_k50d64_43022' in labels and 'depth12x384_43022' in labels:
                a = row.get('sgcr12x384_k50d64_43022')
                b = row.get('depth12x384_43022')
                row['delta_sgcr_minus_depth12x384'] = None if a is None or b is None else a - b
            if 'sgcr12x384_k50d64_43022' in labels and 'legal40k8x480_43022' in labels:
                a = row.get('sgcr12x384_k50d64_43022')
                b = row.get('legal40k8x480_43022')
                row['delta_sgcr_minus_legal40k8x480'] = None if a is None or b is None else a - b
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


def last_jsonl(path: Path) -> tuple[int, Any | None]:
    n = 0
    last: Any | None = None
    if not path.exists():
        return n, last
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                last = json.loads(line)
            except Exception:
                last = {'raw_tail': line[-500:]}
    return n, last


def training_snapshot() -> dict[str, Any]:
    n, last = last_jsonl(TRAIN_LOG)
    return {
        'run_dir': file_rec(TRAIN_RUN_DIR),
        'training_log': file_rec(TRAIN_LOG),
        'training_log_records': n,
        'last_training_record': last,
        'scientific_metrics_file': file_rec(TRAIN_METRICS),
        'scientific_metrics': load_json_if_exists(TRAIN_METRICS),
        'example_order_manifest_file': file_rec(EXAMPLE_ORDER_MANIFEST),
        'example_order_manifest': load_json_if_exists(EXAMPLE_ORDER_MANIFEST),
        'sgcr_diagnostic_file': file_rec(SGCR_DIAG),
        'sgcr_diagnostic': load_json_if_exists(SGCR_DIAG),
    }


def validate_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        errors.append('missing or invalid example_order_manifest.json')
        return errors
    if manifest.get('seed') != 43:
        errors.append(f"example_order_manifest seed {manifest.get('seed')} != 43")
    if manifest.get('selected_for_training_words') != 100_000_000:
        errors.append(f"selected_for_training_words {manifest.get('selected_for_training_words')} != 100000000")
    if manifest.get('num_consumed_examples') != 647_400:
        errors.append(f"num_consumed_examples {manifest.get('num_consumed_examples')} != 647400")
    if manifest.get('data_source_type') != 'example_jsonl':
        errors.append(f"data_source_type {manifest.get('data_source_type')} != example_jsonl")
    if manifest.get('example_jsonl') != 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl':
        errors.append(f"unexpected example_jsonl {manifest.get('example_jsonl')}")
    if manifest.get('masking_curriculum') != 'wwm_fixed' or manifest.get('mask_prob_start') != 0.15 or manifest.get('mask_prob_end') != 0.15:
        errors.append(f"masking contract changed: {manifest.get('masking_curriculum')} {manifest.get('mask_prob_start')}->{manifest.get('mask_prob_end')}")
    stream_sha = None
    for mf in manifest.get('manifest_files') or []:
        if isinstance(mf, dict) and mf.get('name') == 'cleanqwen_fineweb_compact_view_reinvest_100M.jsonl':
            stream_sha = mf.get('sha256')
            if mf.get('whitespace_words') != 100_000_000 or mf.get('rows') != 647_400:
                errors.append(f"stream manifest accounting changed: words={mf.get('whitespace_words')} rows={mf.get('rows')}")
    meta_sha = (((manifest.get('example_jsonl_meta') or {}).get('output_files') or {}).get('sha256') or {}).get('cleanqwen_fineweb_compact_view_reinvest_100M.jsonl')
    stream_sha = stream_sha or meta_sha
    if stream_sha != EXPECTED_SGCR['train_stream_sha256']:
        errors.append(f"stream sha {stream_sha} != expected {EXPECTED_SGCR['train_stream_sha256']}")
    return errors


def validate_training(metrics: Any, manifest: Any | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(metrics, dict):
        errors.append('missing or invalid scientific_metrics.json')
        return errors + validate_manifest(manifest)
    key_aliases = {
        'source_tokenizer_sha256': ['tokenizer_sha256', 'tok40_sha256', 'sgcr_tok40_sha256', 'sgcr_tokenizer40_sha256'],
        'component_tokenizer_sha256': ['sgcr_component_tokenizer_sha256', 'tok16_sha256', 'sgcr_tokenizer16_sha256'],
        # The research trainer records the stream path in scientific_metrics;
        # the exact 100M stream SHA is checked by the research evaluation
        # controller preflight through example_order_manifest.  Do not require
        # a non-emitted trainer metric here.
        'decomposition_map_sha256': ['sgcr_decomposition_map_sha256', 'decomposition_map_sha256'],
    }
    skip_metric_keys = {'train_stream_sha256'}
    for key, exp in EXPECTED_SGCR.items():
        if key in skip_metric_keys:
            continue
        candidates = [key] + key_aliases.get(key, [])
        got = None
        got_key = None
        for c in candidates:
            if c in metrics:
                got = metrics.get(c)
                got_key = c
                break
        if got is None:
            errors.append(f'metrics missing {key}')
            continue
        if got != exp:
            errors.append(f'metrics {got_key}={got!r} != expected {exp!r}')
    hist = metrics.get('sgcr_decomposition_length_histogram')
    expected_hist_str = {'1': 16384, '2': 19333, '3': 3629, '4': 571, '5': 63, '6': 16, '7': 4}
    expected_hist_int = {1: 16384, 2: 19333, 3: 3629, 4: 571, 5: 63, 6: 16, 7: 4}
    if hist not in (expected_hist_str, expected_hist_int):
        errors.append(f'SGCR decomposition histogram unexpected: {hist}')
    errors.extend(validate_manifest(manifest))
    return errors


def validate_collation(path: Path) -> list[str]:
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


def burden_index() -> dict[str, Any]:
    rec = load_json(BURDEN_JSON)
    idx: dict[str, Any] = {}
    for fam in rec.get('family_summaries') or []:
        if isinstance(fam, dict) and fam.get('label'):
            idx[fam['label']] = fam
    for label, fam in (rec.get('all_group_summaries') or {}).items():
        if label not in idx:
            idx[label] = fam
    return {
        'record': rec,
        'by_label': idx,
        'selected': {k: idx.get(k) for k in BURDEN_FAMILIES if k in idx},
    }


def burden_rows(burden: dict[str, Any], sgcr_vs_depth: dict[str, float] | None = None) -> list[dict[str, Any]]:
    idx = burden.get('selected') or {}
    rows: list[dict[str, Any]] = []
    for label in BURDEN_FAMILIES:
        fam = idx.get(label)
        if not isinstance(fam, dict):
            continue
        score_key = 'GlobalPIQA' if label.startswith('GlobalPIQA') else label
        row = {
            'label': label,
            'score_key': score_key,
            'sgcr_minus_depth_score_delta': None if sgcr_vs_depth is None else sgcr_vs_depth.get(score_key),
            'disc_over_shared_mean_residual': fam.get('disc_over_shared_mean_residual'),
            'disc_over_shared_frac_lt50': fam.get('disc_over_shared_frac_lt50'),
            'disc_token_share': fam.get('discriminating_token_share'),
            'disc_residual_share': fam.get('discriminating_residual_share'),
            'disc_lt50_occurrence_share': fam.get('disc_share_of_lt50_occurrences'),
            'disc_lt50_residual_share': fam.get('disc_share_of_lt50_residual'),
            'disc_component_ge50_occurrence_fraction': (((fam.get('discriminating') or {}).get('lt50_component_ge50_occurrence_fraction')) if isinstance(fam.get('discriminating'), dict) else None),
            'disc_mean_decomp_len': (((fam.get('discriminating') or {}).get('mean_decomp_len')) if isinstance(fam.get('discriminating'), dict) else None),
        }
        rows.append(row)
    return rows


def safe_gt(d: dict[str, float], key: str, thresh: float = 0.0) -> bool:
    return d.get(key, -1e9) > thresh


def ready_payload() -> dict[str, Any]:
    paths = SUMMARIES
    vectors = {label: extract_vector(path) for label, path in paths.items() if path.exists()}
    sgcr = vectors['sgcr12x384_k50d64_43022']
    deltas = {
        'sgcr_minus_depth12x384_43022': diff(sgcr, vectors['depth12x384_43022']),
        'sgcr_minus_legal40k8x480_43022': diff(sgcr, vectors['legal40k8x480_43022']),
        'sgcr_minus_legal16k8x480_43022': diff(sgcr, vectors['legal16k8x480_43022']),
        'sgcr_minus_inherited16k_non_submission_43022': diff(sgcr, vectors['inherited16k_non_submission_43022']),
        'sgcr_minus_visible_leader': diff(sgcr, LEADER_VECTOR),
    }
    train = training_snapshot()
    train_errors = validate_training(train.get('scientific_metrics'), train.get('example_order_manifest'))
    collate_errors = validate_collation(paths['sgcr12x384_k50d64_43022'])
    burden = burden_index()
    br = burden_rows(burden, deltas['sgcr_minus_depth12x384_43022'])

    col_rows: list[dict[str, Any]] = []
    for k in TASKS:
        row: dict[str, Any] = {'task': k}
        for label, vec in vectors.items():
            row[label] = vec[k]
        row['visible_leader'] = LEADER_VECTOR[k]
        for name, dv in deltas.items():
            row[name] = dv[k]
        # Attach one family-level burden when there is a direct family match.
        fam = (burden.get('selected') or {}).get(k)
        if isinstance(fam, dict):
            row['burden_disc_over_shared_mean_residual'] = fam.get('disc_over_shared_mean_residual')
            row['burden_disc_over_shared_frac_lt50'] = fam.get('disc_over_shared_frac_lt50')
            row['burden_disc_residual_share'] = fam.get('discriminating_residual_share')
        col_rows.append(row)
    write_csv(OUT_CSV, col_rows)
    dr = detail_rows(paths)
    if dr:
        write_csv(OUT_DETAIL_CSV, dr)

    reading: list[str] = []
    if collate_errors or train_errors:
        reading.append('The SGCR files have integrity mismatches; repair provenance or official-coordinate artifacts before using the endpoint scientifically.')
    else:
        reading.append('SGCR collation and recorded training provenance match the repaired exact-prefix legal40k support-sharing coordinate.')

    margin = sgcr['Overall'] - LEADER_VECTOR['Overall']
    d_depth = deltas['sgcr_minus_depth12x384_43022']
    d_8x480 = deltas['sgcr_minus_legal40k8x480_43022']
    if sgcr['Overall'] >= LEADER_VECTOR['Overall']:
        reading.append('SGCR clears the visible 41.80 leader on the official-compatible coordinate; protect this endpoint and reproduce seed43122 before adding mechanisms.')
    elif sgcr['Overall'] >= 41.55 or d_depth['Overall'] >= 0.35 or d_8x480['Overall'] >= 0.35:
        reading.append('SGCR is close enough or improves enough over matched baselines that a reproduction or a single attribution control may change the route; choose using the column pattern rather than launching by inertia.')
    elif d_depth['Overall'] <= -0.10 and d_8x480['Overall'] <= -0.10:
        reading.append('SGCR underperforms both matched depth and shallower legal40k baselines; close exact-prefix SGCR as implemented unless a specific column-level result reveals a different useful mechanism.')
    else:
        reading.append('SGCR is roughly flat in Overall; interpret column movement against the burden map before deciding whether any one control is worth a full run.')

    on_mech_cols = ['COMPS', 'GlobalPIQA']
    on_mech_delta = {k: d_depth[k] for k in on_mech_cols}
    broad_language_cols = ['BLiMP', 'Supplement', 'SuperGLUE', 'Reading']
    broad_delta = {k: d_depth[k] for k in broad_language_cols}
    if all(safe_gt(d_depth, k, 0.0) for k in on_mech_cols):
        reading.append('COMPS and GlobalPIQA both improve over matched depth; this is the clearest on-mechanism support-sharing pattern in the research burden map.')
    elif any(safe_gt(d_depth, k, 0.0) for k in on_mech_cols):
        reading.append('Only part of the strongest on-mechanism pair moves over depth; inspect COMPS/GlobalPIQA subtasks and consider whether the gain is support-local or generic.')
    else:
        reading.append('The strongest measured support-sharing targets, COMPS and GlobalPIQA, do not improve over matched depth; exact-prefix SGCR does not express the intended mechanism at endpoint scale.')
    if safe_gt(d_depth, 'Entity', 0.0):
        reading.append('Entity improves over depth; this is compatible with low-support option enrichment, but research showed weak mean-residual separation and mostly length-1 paths, so it is not decisive evidence for multi-piece support transfer.')
    if abs(d_depth.get('EWoK', 0.0)) >= 0.75:
        reading.append('EWoK moves substantially even though research found weak low-support/discriminating alignment; explain this through generic representation/optimization, broader lexical effects, or noise rather than simple rare-token support sharing.')
    if any(v < -0.5 for v in broad_delta.values()):
        reading.append('At least one broad language column falls by more than 0.5 over depth; if SGCR is otherwise promising, any next comparison must separate support-dependent gains from interference caused by the added auxiliary estimator.')
    if (d_depth['Overall'] > 0.0 and (safe_gt(d_depth, 'COMPS', 0.0) or safe_gt(d_depth, 'GlobalPIQA', 0.0)) and sgcr['Overall'] < LEADER_VECTOR['Overall']):
        reading.append('If this sub-frontier improvement is considered worth another full run, the most informative comparison is the mass-matched uniform-residual control, because SGCR adds 1,073,536 trainable parameters beyond depth.')

    controller = load_json_if_exists(CONTROLLER_SUMMARY)
    return {
        'status': 'SGCR_ENDPOINT_READY' if not (collate_errors or train_errors) else 'SGCR_ENDPOINT_HAS_INTEGRITY_ERRORS',
        'created_utc': now_utc(),
        'script_never_launches_training_or_evaluation': True,
        'visible_leader': LEADER_VECTOR,
        'vectors': vectors,
        'deltas': deltas,
        'margin_over_visible_leader_41p8': margin,
        'largest_abs_delta_vs_depth12x384_43022': top_abs(d_depth, 10),
        'largest_abs_delta_vs_legal40k8x480_43022': top_abs(d_8x480, 10),
        'on_mechanism_delta_vs_depth': on_mech_delta,
        'broad_language_delta_vs_depth': broad_delta,
        'collation_integrity_errors': collate_errors,
        'training_integrity_errors': train_errors,
        'training_snapshot': train,
        'controller_summary_status': controller.get('status') if isinstance(controller, dict) else None,
        'controller_summary_path': rel(CONTROLLER_SUMMARY),
        'burden_summary_selected': br,
        'burden_record_path': rel(BURDEN_JSON),
        'scientific_reading': reading,
        'files': {'json': rel(OUT_JSON), 'column_csv': rel(OUT_CSV), 'detail_csv': rel(OUT_DETAIL_CSV), 'note': rel(OUT_MD)},
    }


def build_waiting(missing: list[str]) -> dict[str, Any]:
    return {
        'status': 'WAITING_FOR_SGCR_OFFICIAL_COLLATION',
        'created_utc': now_utc(),
        'script_never_launches_training_or_evaluation': True,
        'no_managed_task_state_query': True,
        'missing_required_files': missing,
        'expected_tasks_from_step87': {
            'training_task': 's87_t29_tool1 sgcr_exactprefix_12x384_seed43022_train',
            'evaluation_task': 's87_t37_tool1 sgcr_exactprefix_12x384_seed43022_eval',
        },
        'training_snapshot_not_read_because_endpoint_missing': True,
        'expected_training_artifacts_after_completion': {
            'run_dir': rel(TRAIN_RUN_DIR),
            'training_log': rel(TRAIN_LOG),
            'scientific_metrics': rel(TRAIN_METRICS),
            'example_order_manifest': rel(EXAMPLE_ORDER_MANIFEST),
            'sgcr_diagnostic': rel(SGCR_DIAG),
        },
        'reference_summaries_present': {label: file_rec(path) for label, path in SUMMARIES.items() if label != 'sgcr12x384_k50d64_43022'},
        'burden_record': file_rec(BURDEN_JSON),
        'script_purpose': 'Run after the SGCR hardened official collation exists; no score is inferred from missing outputs.',
        'files': {'json': rel(OUT_JSON), 'note': rel(OUT_MD)},
    }


def write_md(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append('# research — SGCR endpoint interpretation\n\n')
    lines.append(f"Status: `{payload['status']}`\n\n")
    if payload['status'] == 'WAITING_FOR_SGCR_OFFICIAL_COLLATION':
        lines.append('The SGCR official-compatible endpoint collation is not present. No score is inferred from missing files or from training progress.\n\n')
        lines.append('Missing required files:\n')
        for p in payload.get('missing_required_files') or []:
            lines.append(f'- `{p}`\n')
        lines.append('\nReference artifacts are present for depth/legal baselines and research burden map; rerun this script after the managed evaluation writes the SGCR pristine collation summary.\n')
    else:
        vec = payload['vectors']['sgcr12x384_k50d64_43022']
        lines.append(f"SGCR Overall: `{vec['Overall']:.6f}`; margin vs 41.80 `{payload['margin_over_visible_leader_41p8']:+.6f}`\n\n")
        lines.append('## SGCR movement vs matched depth 12x384 seed43022\n')
        d = payload['deltas']['sgcr_minus_depth12x384_43022']
        for k in TASKS:
            lines.append(f'- {k}: `{d[k]:+.6f}`\n')
        lines.append('\n## SGCR movement vs shallower legal40k 8x480 seed43022\n')
        d2 = payload['deltas']['sgcr_minus_legal40k8x480_43022']
        for k in TASKS:
            lines.append(f'- {k}: `{d2[k]:+.6f}`\n')
        lines.append('\n## research burden-map anchors\n')
        for row in payload.get('burden_summary_selected') or []:
            lines.append(
                f"- {row['label']}: score_delta_vs_depth `{row.get('sgcr_minus_depth_score_delta')}`, "
                f"residual ratio `{row.get('disc_over_shared_mean_residual')}`, "
                f"lt50 ratio `{row.get('disc_over_shared_frac_lt50')}`, "
                f"disc residual share `{row.get('disc_residual_share')}`\n"
            )
        lines.append('\n## Integrity\n')
        errs = (payload.get('collation_integrity_errors') or []) + (payload.get('training_integrity_errors') or [])
        if errs:
            for e in errs:
                lines.append(f'- ERROR: {e}\n')
        else:
            lines.append('- SGCR training provenance and official-compatible collation checks passed.\n')
        lines.append('\n## Scientific reading\n')
        for item in payload.get('scientific_reading') or []:
            lines.append(f'- {item}\n')
    lines.append('\n## Files\n')
    for k, v in (payload.get('files') or {}).items():
        lines.append(f'- {k}: `{v}`\n')
    OUT_MD.write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [rel(p) for label, p in SUMMARIES.items() if not p.exists()]
    if not BURDEN_JSON.exists():
        missing.append(rel(BURDEN_JSON))
    if missing:
        payload = build_waiting(missing)
    else:
        payload = ready_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_md(payload)
    print(json.dumps({
        'status': payload['status'],
        'missing_required_files': payload.get('missing_required_files'),
        'sgcr_overall': ((payload.get('vectors') or {}).get('sgcr12x384_k50d64_43022') or {}).get('Overall'),
        'margin_over_visible_leader_41p8': payload.get('margin_over_visible_leader_41p8'),
        'collation_integrity_errors': payload.get('collation_integrity_errors'),
        'training_integrity_errors': payload.get('training_integrity_errors'),
        'out_json': rel(OUT_JSON),
        'note': rel(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
