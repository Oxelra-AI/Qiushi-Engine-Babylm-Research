#!/usr/bin/env python3
"""research: interpret legal-40k official vectors as soon as collations exist.

CPU-only. This script does not run evaluation and does not query managed task
state. It reads the hardened research legal-40k collation summaries if present,
validates the official-coordinate integrity fields, compares them with the
legal-16k corrected-tokenizer endpoints and the inherited-tokenizer mechanism
coordinate, and writes a durable waiting or ready record.

The scientific question is whether legal 40k repaired the legal-16k failure by
recovering Supplement/EWoK while preserving the compact-view gains in
GlobalPIQA/Entity/COMPS, or whether the result points to a genuinely different
post-40k route such as relation-focused masking or 12x384 depth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path('.').resolve()
WS = ROOT / 'experiments/archive/representation_and_objectives'
OUT_DIR = WS / 'data/legal40k_endpoint_interpretation'
OUT_JSON = OUT_DIR / 'legal40k_endpoint_interpretation.json'
OUT_CSV = OUT_DIR / 'legal40k_column_comparison.csv'
OUT_DETAIL_CSV = OUT_DIR / 'legal40k_detail_comparison.csv'
OUT_MD = (ROOT / 'research/notes/representation_and_objectives/legal40k_endpoint_interpretation.md')

LEADER = 41.8
TASKS = ['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'SuperGLUE', 'GlobalPIQA', 'Reading', 'AoA', 'Overall']
CORE_TASKS = ['Supplement', 'EWoK']
PRESERVE_TASKS = ['GlobalPIQA', 'Entity', 'COMPS']
SUPERGLUE_TASKS = ['boolq', 'mnli', 'mrpc', 'multirc', 'qqp', 'rte', 'wsc']
EXPECTED_GLOBALPIQA = {
    'global_piqa_parallel': {'sha256': 'cb9513ed5096ad8115becbe42fdfc97711980f519ada87808e3ddc5f8d5e4453', 'num_lines': 103},
    'global_piqa_nonparallel': {'sha256': 'aeec831d3adb09bf268ce1b847a854d105b3922c440a164a2c5893b96b228a8f', 'num_lines': 100},
}
COLLATE_SUMMARIES = {
    '43022': WS / 'data/legal40k_accum_seed43022_pristine_collate/pristine_collate_legal40k_seed43022_summary.json',
    '43122': WS / 'data/legal40k_accum_seed43122_pristine_collate/pristine_collate_legal40k_seed43122_summary.json',
}
CONTROLLER_SUMMARIES = {
    '43022': WS / 'data/legal40k_accum_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json',
    '43122': WS / 'data/legal40k_accum_seed43122_posttrain_eval_controller/posttrain_eval_seed43122_summary.json',
}
LEGAL16_SUMMARIES = {
    '43022': WS / 'data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json',
    '43122': WS / 'data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json',
}
INHERITED_SUMMARIES = {
    '43022': WS / 'data/pristine_collate_seed43022/pristine_collate_seed43022_summary.json',
    '43122': WS / 'data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json',
}
ROUTE_ASSETS = WS / 'data/post40k_route_assets/post40k_route_assets.json'
SUPPORT_SPECTRUM = WS / 'data/tokenizer_support_spectrum/tokenizer_support_spectrum.json'
TARGET_BURDEN_DIR = WS / 'data/wwm_target_burden_map'
CANDIDATE_MANIFEST = WS / 'data/post40k_candidate_launch_manifests/post40k_candidate_launch_manifests.json'
RELATION_AUDIT = WS / 'data/relation_bias_large_stream_audit/relation_bias_large_stream_audit.json'
RELATION_PREFIX_IDENTITY = WS / 'data/relation_bias_prefix_identity/relation_bias_prefix_identity.json'
RELATION_EXECUTION_REPLAY = WS / 'data/relation_bias_execution_replay/relation_bias_execution_replay.json'
OVERLAP_SCAN = WS / 'data/generated_view_overlap_scan/generated_view_overlap_scan.json'
REWRITE_ATTRIBUTION = WS / 'data/generated_view_overlap_scan/rewrite_overlap_attribution.json'


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


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


def extract_vector(rec: Any, path: Path) -> dict[str, float]:
    for cand in vector_candidates(rec):
        vec: dict[str, float] = {}
        for k in TASKS:
            x = finite(cand.get(k))
            if x is not None:
                vec[k] = x
        if all(k in vec for k in TASKS):
            return vec
    raise RuntimeError({'cannot_extract_vector': rel(path), 'candidate_keys': sorted({k for c in vector_candidates(rec) for k in c})})


def score_summary(rec: Any) -> dict[str, Any]:
    ss = rec.get('score_summary') if isinstance(rec, dict) else None
    if isinstance(ss, dict):
        return ss
    return {}


def detail_obj(rec: Any) -> dict[str, Any]:
    ss = score_summary(rec)
    details = ss.get('details')
    if isinstance(details, dict):
        return details
    return {}


def load_summary(path: Path) -> dict[str, Any]:
    rec = load_json(path)
    return {
        'path': rel(path),
        'record': rec,
        'vector': extract_vector(rec, path),
        'details': detail_obj(rec),
        'collated_sha256': rec.get('collated_sha256') if isinstance(rec, dict) else None,
        'validation_errors': rec.get('validation_errors') if isinstance(rec, dict) else None,
        'status': rec.get('status') if isinstance(rec, dict) else None,
    }


def diff(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {k: a[k] - b[k] for k in TASKS if k in a and k in b}


def mean_vec(vs: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(v[k] for v in vs) / len(vs) for k in TASKS}


def population_sd(xs: list[float]) -> float:
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0


def top_abs(d: dict[str, float], n: int = 12) -> list[dict[str, Any]]:
    return [{'key': k, 'delta': d[k]} for k in sorted(d, key=lambda k: abs(d[k]), reverse=True)[:n]]


def validate_legal40_summary(seed: str, rec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    raw = rec['record']
    if raw.get('status') != 'PRISTINE_COLLATE':
        errors.append(f'seed {seed}: unexpected collation status {raw.get("status")}')
    if raw.get('validation_errors') not in ([], None):
        errors.append(f'seed {seed}: validation_errors={raw.get("validation_errors")}')
    val = raw.get('validation') or {}
    if isinstance(val, dict) and val.get('errors') not in ([], None):
        errors.append(f'seed {seed}: validation.errors={val.get("errors")}')
    cs = raw.get('collated_summary') or {}
    if sorted(cs.get('null_keys') or []) != []:
        errors.append(f'seed {seed}: null collated keys {cs.get("null_keys")}')
    if sum((cs.get('ewok_lengths') or {}).values()) != 7618:
        errors.append(f'seed {seed}: EWoK count {sum((cs.get("ewok_lengths") or {}).values())} != 7618')
    if cs.get('aoa_surprisal_num_steps') != 19:
        errors.append(f'seed {seed}: AoA steps {cs.get("aoa_surprisal_num_steps")} != 19')
    if cs.get('aoa_surprisal_row_count_values') != [8005]:
        errors.append(f'seed {seed}: AoA row-count values {cs.get("aoa_surprisal_row_count_values")} != [8005]')
    if finite((cs.get('aoa_value'))) is None:
        errors.append(f'seed {seed}: missing finite AoA value')
    for k in TASKS:
        if finite(rec['vector'].get(k)) is None:
            errors.append(f'seed {seed}: missing/nonfinite score {k}')
    details = rec.get('details') or {}
    gp = details.get('GlobalPIQA') if isinstance(details, dict) else None
    for task_name, exp in EXPECTED_GLOBALPIQA.items():
        g = gp.get(task_name) if isinstance(gp, dict) else None
        if not isinstance(g, dict):
            errors.append(f'seed {seed}: missing GlobalPIQA detail {task_name}')
            continue
        if g.get('sha256') != exp['sha256'] or g.get('num_lines') != exp['num_lines']:
            errors.append(f'seed {seed}: GlobalPIQA {task_name} target {g.get("sha256")}/{g.get("num_lines")} != expected {exp}')
    sg = details.get('SuperGLUE') if isinstance(details, dict) else None
    for task in SUPERGLUE_TASKS:
        if not isinstance(sg, dict) or task not in sg or finite((sg.get(task) or {}).get('score')) is None:
            errors.append(f'seed {seed}: missing SuperGLUE detail {task}')
    return errors


def validate_controller_summary(seed: str) -> dict[str, Any]:
    p = CONTROLLER_SUMMARIES[seed]
    out = {'path': rel(p), 'exists': p.exists(), 'status': None, 'returncode_failures': [], 'collate_summary_path_matches': None, 'overall_visible': None}
    if not p.exists():
        return out
    rec = load_json(p)
    out['status'] = rec.get('status')
    runs = rec.get('runs') or []
    out['returncode_failures'] = [r for r in runs if isinstance(r, dict) and r.get('returncode') != 0]
    out['collate_summary_path_matches'] = Path(str(rec.get('collate_summary_path'))).as_posix() == COLLATE_SUMMARIES[seed].as_posix()
    ss = rec.get('score_summary') if isinstance(rec, dict) else None
    if isinstance(ss, dict):
        out['overall_visible'] = ss.get('Overall')
    return out


def count_fraction_scores(entity_detail: dict[str, Any]) -> dict[str, float]:
    out = {}
    if not isinstance(entity_detail, dict):
        return out
    # research details are counts; older summaries may include subtask_fraction_scores.
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


def superglue_scores(details: dict[str, Any]) -> dict[str, float]:
    sg = details.get('SuperGLUE') if isinstance(details, dict) else None
    out = {}
    if isinstance(sg, dict):
        for task, rec in sg.items():
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[task] = float(rec['score'])
    return out


def globalpiqa_scores(details: dict[str, Any]) -> dict[str, float]:
    gp = details.get('GlobalPIQA') if isinstance(details, dict) else None
    out = {}
    if isinstance(gp, dict):
        for k, rec in gp.items():
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[k] = float(rec['score'])
    else:
        for k in ['GlobalPIQA_parallel', 'GlobalPIQA_nonparallel']:
            rec = details.get(k) if isinstance(details, dict) else None
            if isinstance(rec, dict) and finite(rec.get('score')) is not None:
                out[k] = float(rec['score'])
    return out


def detail_scores(summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    d = summary.get('details') or {}
    out = {
        'Entity': count_fraction_scores(d.get('Entity') if isinstance(d, dict) else {}),
        'SuperGLUE': superglue_scores(d),
        'GlobalPIQA': globalpiqa_scores(d),
    }
    return out


def detail_delta_rows(legal40: dict[str, Any], legal16: dict[str, Any], inherited: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in ['43022', '43122']:
        s40 = detail_scores(legal40[seed])
        s16 = detail_scores(legal16[seed])
        sin = detail_scores(inherited[seed])
        for family in sorted(set(s40) | set(s16) | set(sin)):
            keys = sorted(set(s40.get(family, {})) | set(s16.get(family, {})) | set(sin.get(family, {})))
            for key in keys:
                v40 = s40.get(family, {}).get(key)
                v16 = s16.get(family, {}).get(key)
                vin = sin.get(family, {}).get(key)
                rows.append({
                    'seed': seed,
                    'family': family,
                    'subtask': key,
                    'legal40k': v40,
                    'legal16k': v16,
                    'inherited16k_non_submission': vin,
                    'delta_40_minus_16': None if v40 is None or v16 is None else v40 - v16,
                    'delta_40_minus_inherited': None if v40 is None or vin is None else v40 - vin,
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_waiting(missing: list[str]) -> dict[str, Any]:
    payload = {
        'status': 'WAITING_FOR_LEGAL40K_FULL_COLLATIONS',
        'created_utc': now_utc(),
        'missing_collation_summaries': missing,
        'active_evaluation_tasks': ['s65_t22_tool1 legal40k seed43022 full official evaluation', 's66_t9_tool1 legal40k seed43122 full official evaluation'],
        'script_purpose': 'Run after one or both hardened research legal-40k official collations exist; no scores are inferred from missing outputs.',
        'files': {'json': rel(OUT_JSON), 'note': rel(OUT_MD)},
    }
    return payload


def route_assets_digest() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if ROUTE_ASSETS.exists():
        r = load_json(ROUTE_ASSETS)
        pool = r.get('relation_pool_summary') or {}
        out['relation_pool'] = {
            'relation_cue_word_frac': pool.get('relation_cue_word_frac'),
            'top_categories': dict(list((pool.get('category_word_counts') or {}).items())[:10]) if isinstance(pool.get('category_word_counts'), dict) else None,
        }
        toks = r.get('tokenizer_interface_summaries') or {}
        out['tokenizer_interface'] = {
            k: {
                'raw_tokens_per_word': v.get('raw_tokens_per_word_weighted'),
                'visible_tokens_per_word': v.get('visible_tokens_per_word_weighted'),
                'raw_over256_rows': v.get('raw_over256_rows'),
                'visible_relation_group_frac': v.get('visible_relation_group_frac'),
                'visible_relation_token_frac': v.get('visible_relation_token_frac'),
            } for k, v in toks.items() if k in ['legal16k', 'legal40k', 'legal40k_minfreq50'] and isinstance(v, dict)
        }
    if SUPPORT_SPECTRUM.exists():
        s = load_json(SUPPORT_SPECTRUM)
        out['token_support_note'] = s.get('interpretation') or s.get('support_summary') or 'see research tokenizer_support_spectrum.json'
    if RELATION_AUDIT.exists():
        a = load_json(RELATION_AUDIT)
        g = a.get('global_summary') or {}
        boosts = g.get('boosts') or {}
        out['relation_audit'] = {
            'path': rel(RELATION_AUDIT),
            'pool_rows': (a.get('checks') or {}).get('rows_seen'),
            'pool_words': (a.get('checks') or {}).get('words_seen'),
            'pool10_sha_matches': (a.get('checks') or {}).get('pool10_sha_matches'),
            'tokenizer_sha_matches': (a.get('checks') or {}).get('tokenizer_sha_matches'),
            'visible_relation_group_frac': g.get('relation_group_frac'),
            'visible_relation_token_frac': g.get('relation_token_frac'),
            'boost2': {
                'relation_selected_group_frac': (boosts.get('2.0') or {}).get('relation_selected_group_frac'),
                'relation_target_token_frac': (boosts.get('2.0') or {}).get('relation_target_token_frac'),
                'selected_group_multiplier_vs_uniform': (boosts.get('2.0') or {}).get('selected_group_multiplier_vs_uniform'),
                'target_token_multiplier_vs_uniform': (boosts.get('2.0') or {}).get('target_token_multiplier_vs_uniform'),
                'rows_clipped_frac': (boosts.get('2.0') or {}).get('rows_clipped_frac'),
            },
            'boost3': {
                'relation_selected_group_frac': (boosts.get('3.0') or {}).get('relation_selected_group_frac'),
                'target_token_multiplier_vs_uniform': (boosts.get('3.0') or {}).get('target_token_multiplier_vs_uniform'),
                'rows_clipped_frac': (boosts.get('3.0') or {}).get('rows_clipped_frac'),
            },
            'boost4': {
                'relation_selected_group_frac': (boosts.get('4.0') or {}).get('relation_selected_group_frac'),
                'target_token_multiplier_vs_uniform': (boosts.get('4.0') or {}).get('target_token_multiplier_vs_uniform'),
                'rows_clipped_frac': (boosts.get('4.0') or {}).get('rows_clipped_frac'),
            },
            'interpretation': a.get('interpretation'),
        }
    if RELATION_PREFIX_IDENTITY.exists():
        p = load_json(RELATION_PREFIX_IDENTITY)
        out['relation_prefix_identity'] = {
            'path': rel(RELATION_PREFIX_IDENTITY),
            'row_multiset_equal': p.get('row_multiset_equal'),
            'source_words_equal': p.get('source_words_equal'),
            'stream_minus_pool_row_count': p.get('stream_minus_pool_row_count'),
            'pool_minus_stream_row_count': p.get('pool_minus_stream_row_count'),
            'pool_rows_words': [(p.get('pool10') or {}).get('rows'), (p.get('pool10') or {}).get('words')],
            'stream_prefix_rows_words': [(p.get('stream_prefix') or {}).get('rows'), (p.get('stream_prefix') or {}).get('words')],
            'interpretation': p.get('interpretation'),
        }
    if RELATION_EXECUTION_REPLAY.exists():
        r = load_json(RELATION_EXECUTION_REPLAY)
        g = r.get('global_realized_masking') or {}
        out['relation_execution_replay'] = {
            'path': rel(RELATION_EXECUTION_REPLAY),
            'max_rows': (r.get('inputs') or {}).get('max_rows'),
            'mask_seeds': (r.get('inputs') or {}).get('mask_seeds'),
            'prefix_row_multiset_equal_for_subset': (r.get('prefix_identity') or {}).get('row_multiset_equal'),
            'relation_dataset_matches_base_first256': (r.get('checks') or {}).get('relation_dataset_matches_base_first256'),
            'base_selected_relation_group_frac': g.get('base_selected_relation_group_frac'),
            'boost_selected_relation_group_frac': g.get('boost_selected_relation_group_frac'),
            'base_selected_relation_token_frac': g.get('base_selected_relation_token_frac'),
            'boost_selected_relation_token_frac': g.get('boost_selected_relation_token_frac'),
            'boost_selected_group_multiplier_vs_base_realized': g.get('boost_selected_group_multiplier_vs_base_realized'),
            'boost_selected_token_multiplier_vs_base_realized': g.get('boost_selected_token_multiplier_vs_base_realized'),
            'boost_expected_group_multiplier_vs_uniform': g.get('boost_expected_group_multiplier_vs_uniform'),
            'boost_expected_token_multiplier_vs_uniform': g.get('boost_expected_token_multiplier_vs_uniform'),
            'boost_rows_clipped_frac': g.get('boost_rows_clipped_frac'),
            'base_rows_zero_selected_frac': g.get('base_rows_zero_selected_frac'),
            'boost_rows_zero_selected_frac': g.get('boost_rows_zero_selected_frac'),
            'base_special_random_replacements_total': g.get('base_special_random_replacements_total'),
            'boost_special_random_replacements_total': g.get('boost_special_random_replacements_total'),
            'interpretation': r.get('interpretation'),
        }
    if OVERLAP_SCAN.exists():
        o = load_json(OVERLAP_SCAN)
        rewrite = o.get('rewrite_component_summary') or {}
        rows = o.get('row_source_summary') or []
        def slim_component(name: str) -> dict[str, Any]:
            d = rewrite.get(name)
            if not isinstance(d, dict):
                all_components = o.get('component_kind_summary') or []
                d = next((x for x in all_components if isinstance(x, dict) and x.get('component_kind') == name), {})
            return {
                'components': d.get('components'),
                'words_declared': d.get('words_declared'),
                'components_with_any_match_n7': d.get('components_with_any_match_n7'),
                'components_with_any_match_n8': d.get('components_with_any_match_n8'),
                'components_with_any_match_n10': d.get('components_with_any_match_n10'),
                'unique_matched_ngrams_n7': d.get('unique_matched_ngrams_n7'),
                'unique_matched_ngrams_n8': d.get('unique_matched_ngrams_n8'),
                'unique_matched_ngrams_n10': d.get('unique_matched_ngrams_n10'),
            }
        out['overlap_scan'] = {
            'path': rel(OVERLAP_SCAN),
            'strict_eval_records_total': (o.get('official_coordinate') or {}).get('strict_eval_records_total'),
            'fineweb_compact_qwen_rewrite': slim_component('fineweb_compact_qwen_rewrite'),
            'official_source_qwen_paraphrase_rewrite': slim_component('official_source_qwen_paraphrase_rewrite'),
            'row_source_summary': [
                {
                    'row_source_class': r.get('row_source_class'),
                    'rows': r.get('rows'),
                    'words': r.get('words'),
                    'rows_with_any_match_n7': r.get('rows_with_any_match_n7'),
                    'rows_with_any_match_n8': r.get('rows_with_any_match_n8'),
                    'rows_with_any_match_n10': r.get('rows_with_any_match_n10'),
                    'unique_matched_ngrams_n7': r.get('unique_matched_ngrams_n7'),
                } for r in rows if isinstance(r, dict)
            ],
            'interpretation_limits': o.get('interpretation_limits'),
        }
    if REWRITE_ATTRIBUTION.exists():
        a = load_json(REWRITE_ATTRIBUTION)
        summary = a.get('summary') or {}
        out['rewrite_overlap_attribution'] = {
            'path': rel(REWRITE_ATTRIBUTION),
            'raw_detail_rows': summary.get('raw_detail_rows'),
            'unique_pair_ngrams': summary.get('unique_pair_ngrams'),
            'by_attribution': summary.get('by_attribution'),
            'interpretation': a.get('interpretation'),
        }
    if CANDIDATE_MANIFEST.exists():
        m = load_json(CANDIDATE_MANIFEST)
        out['dormant_routes'] = [{'route': r.get('route'), 'seed_key': r.get('seed_key'), 'run_dir': r.get('run_dir')} for r in (m.get('routes') or [])]
    return out


def ready_payload() -> dict[str, Any]:
    legal40 = {s: load_summary(p) for s, p in COLLATE_SUMMARIES.items()}
    legal16 = {s: load_summary(p) for s, p in LEGAL16_SUMMARIES.items()}
    inherited = {s: load_summary(p) for s, p in INHERITED_SUMMARIES.items()}
    controller = {s: validate_controller_summary(s) for s in ['43022', '43122']}
    validation_errors = []
    for s in ['43022', '43122']:
        validation_errors.extend(validate_legal40_summary(s, legal40[s]))
        if controller[s]['exists']:
            if controller[s]['status'] != 'LEGAL40K_POSTTRAIN_EVAL_DONE':
                validation_errors.append(f'seed {s}: controller status {controller[s]["status"]}')
            if controller[s]['returncode_failures']:
                validation_errors.append(f'seed {s}: controller returncode failures present')
            if controller[s]['collate_summary_path_matches'] is not True:
                validation_errors.append(f'seed {s}: controller collate path mismatch')

    legal40_vecs = {s: legal40[s]['vector'] for s in ['43022', '43122']}
    legal16_vecs = {s: legal16[s]['vector'] for s in ['43022', '43122']}
    inherited_vecs = {s: inherited[s]['vector'] for s in ['43022', '43122']}
    means = {
        'legal40k': mean_vec(list(legal40_vecs.values())),
        'legal16k': mean_vec(list(legal16_vecs.values())),
        'inherited16k_non_submission': mean_vec(list(inherited_vecs.values())),
    }
    mean_deltas = {
        'legal40k_minus_legal16k': diff(means['legal40k'], means['legal16k']),
        'legal40k_minus_inherited16k': diff(means['legal40k'], means['inherited16k_non_submission']),
    }
    by_seed = {}
    for s in ['43022', '43122']:
        by_seed[s] = {
            'legal40k': legal40_vecs[s],
            'legal16k': legal16_vecs[s],
            'inherited16k_non_submission': inherited_vecs[s],
            'delta_legal40k_minus_legal16k': diff(legal40_vecs[s], legal16_vecs[s]),
            'delta_legal40k_minus_inherited16k': diff(legal40_vecs[s], inherited_vecs[s]),
            'margin_over_visible_leader_41p8': legal40_vecs[s]['Overall'] - LEADER,
            'largest_abs_delta_vs_legal16k': top_abs(diff(legal40_vecs[s], legal16_vecs[s]), 10),
        }
    core_recovery = {k: mean_deltas['legal40k_minus_legal16k'][k] for k in CORE_TASKS}
    preserve = {k: mean_deltas['legal40k_minus_legal16k'][k] for k in PRESERVE_TASKS}
    both_above = all(v['Overall'] > LEADER for v in legal40_vecs.values())
    any_above = any(v['Overall'] > LEADER for v in legal40_vecs.values())
    overall_values = [v['Overall'] for v in legal40_vecs.values()]
    reading = []
    if validation_errors:
        reading.append('Legal-40k collation integrity has errors; inspect and repair before any scientific endpoint interpretation.')
    elif both_above:
        reading.append('Both legal-40k seeds clear the visible 41.8 leader; protect the compliant route and reproduce/package only after independent review of provenance and artifacts.')
    elif any_above:
        reading.append('One legal-40k seed clears the visible leader; preserve it as a compliant endpoint candidate but treat seed-sensitive columns as load-bearing before submission packaging.')
    else:
        reading.append('Neither legal-40k seed clears the visible leader; vocabulary-breadth/embedding-capacity alone did not solve the compliant SOTA problem.')
    if all(core_recovery[k] > 0 for k in CORE_TASKS):
        reading.append('Mean Supplement and EWoK both recovered relative to legal16k; interpret 40k as partially repairing the legal representation coordinate even if Overall remains below frontier.')
    else:
        reading.append('Mean Supplement/EWoK did not jointly recover relative to legal16k; favor a genuinely different learning-signal or architecture route rather than another vocabulary sweep.')
    if all(preserve[k] >= -0.25 for k in PRESERVE_TASKS):
        reading.append('GlobalPIQA/Entity/COMPS were largely preserved relative to legal16k, so remaining weakness is concentrated rather than a broad collapse.')
    else:
        reading.append('At least one compact-view-preserve column fell materially relative to legal16k; inspect full vector before choosing relation masking versus depth.')

    col_rows = []
    for s in ['43022', '43122']:
        for k in TASKS:
            col_rows.append({
                'seed': s,
                'task': k,
                'legal40k': legal40_vecs[s][k],
                'legal16k': legal16_vecs[s][k],
                'inherited16k_non_submission': inherited_vecs[s][k],
                'delta_40_minus_16': legal40_vecs[s][k] - legal16_vecs[s][k],
                'delta_40_minus_inherited': legal40_vecs[s][k] - inherited_vecs[s][k],
            })
    for k in TASKS:
        col_rows.append({
            'seed': 'mean',
            'task': k,
            'legal40k': means['legal40k'][k],
            'legal16k': means['legal16k'][k],
            'inherited16k_non_submission': means['inherited16k_non_submission'][k],
            'delta_40_minus_16': mean_deltas['legal40k_minus_legal16k'][k],
            'delta_40_minus_inherited': mean_deltas['legal40k_minus_inherited16k'][k],
        })
    detail_rows = detail_delta_rows(legal40, legal16, inherited)
    write_csv(OUT_CSV, col_rows)
    write_csv(OUT_DETAIL_CSV, detail_rows)
    return {
        'status': 'LEGAL40K_ENDPOINT_INTERPRETATION_READY' if not validation_errors else 'LEGAL40K_ENDPOINT_INTERPRETATION_HAS_VALIDATION_ERRORS',
        'created_utc': now_utc(),
        'visible_leader_overall': LEADER,
        'collation_integrity_errors': validation_errors,
        'controller_summaries': controller,
        'collation_summaries': {s: {'path': legal40[s]['path'], 'collated_sha256': legal40[s]['collated_sha256'], 'validation_errors': legal40[s]['validation_errors']} for s in ['43022', '43122']},
        'legal40k_mean': means['legal40k'],
        'legal16k_mean': means['legal16k'],
        'inherited16k_non_submission_mean': means['inherited16k_non_submission'],
        'mean_delta_legal40k_minus_legal16k': mean_deltas['legal40k_minus_legal16k'],
        'mean_delta_legal40k_minus_inherited16k': mean_deltas['legal40k_minus_inherited16k'],
        'restore_focus_delta_vs_legal16k': core_recovery,
        'preserve_focus_delta_vs_legal16k': preserve,
        'legal40k_overall_spread_abs': abs(legal40_vecs['43022']['Overall'] - legal40_vecs['43122']['Overall']),
        'legal40k_overall_population_sd': population_sd(overall_values),
        'both_legal40k_seeds_above_visible_leader': both_above,
        'any_legal40k_seed_above_visible_leader': any_above,
        'by_seed': by_seed,
        'largest_abs_mean_delta_vs_legal16k': top_abs(mean_deltas['legal40k_minus_legal16k'], 10),
        'scientific_reading': reading,
        'route_asset_digest': route_assets_digest(),
        'files': {
            'json': rel(OUT_JSON),
            'column_csv': rel(OUT_CSV),
            'detail_csv': rel(OUT_DETAIL_CSV),
            'note': rel(OUT_MD),
            'top_level_comparison_script': rel(WS / 'scripts/compare_legal40k_two_seed_results.py'),
        },
    }


def write_md(payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append('# research legal-40k endpoint interpretation\n\n')
    lines.append(f"Status: `{payload['status']}`\n\n")
    if payload['status'] == 'WAITING_FOR_LEGAL40K_FULL_COLLATIONS':
        lines.append('The full official legal-40k collations have not both landed. No score is inferred from missing outputs.\n\n')
        lines.append('Missing summaries:\n')
        for p in payload['missing_collation_summaries']:
            lines.append(f'- `{p}`\n')
    else:
        lines.append(f"Visible leader Overall: `{LEADER}`\n\n")
        lines.append('## Overall\n')
        for seed, rec in payload['by_seed'].items():
            lines.append(f"- seed {seed}: legal40k `{rec['legal40k']['Overall']:.6f}`; legal16k `{rec['legal16k']['Overall']:.6f}`; delta `{rec['delta_legal40k_minus_legal16k']['Overall']:+.6f}`; margin vs 41.8 `{rec['margin_over_visible_leader_41p8']:+.6f}`\n")
        lines.append(f"- legal40k mean `{payload['legal40k_mean']['Overall']:.6f}`; delta vs legal16k mean `{payload['mean_delta_legal40k_minus_legal16k']['Overall']:+.6f}`\n\n")
        lines.append('## Restore/preserve pattern vs legal16k\n')
        for k in CORE_TASKS + PRESERVE_TASKS:
            lines.append(f"- {k}: `{payload['mean_delta_legal40k_minus_legal16k'][k]:+.6f}`\n")
        lines.append('\n## Largest mean movements vs legal16k\n')
        for item in payload['largest_abs_mean_delta_vs_legal16k']:
            lines.append(f"- {item['key']}: `{item['delta']:+.6f}`\n")
        lines.append('\n## Integrity\n')
        if payload['collation_integrity_errors']:
            for e in payload['collation_integrity_errors']:
                lines.append(f'- ERROR: {e}\n')
        else:
            lines.append('- Hardened collation integrity checks passed for both seeds.\n')
        lines.append('\n## Scientific reading\n')
        for item in payload['scientific_reading']:
            lines.append(f'- {item}\n')
    lines.append('\n## Files\n')
    for k, v in (payload.get('files') or {}).items():
        lines.append(f'- {k}: `{v}`\n')
    OUT_MD.write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [rel(p) for p in COLLATE_SUMMARIES.values() if not p.exists()]
    if missing:
        payload = build_waiting(missing)
    else:
        payload = ready_payload()
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    write_md(payload)
    print(json.dumps({
        'status': payload['status'],
        'missing': payload.get('missing_collation_summaries'),
        'legal40k_mean_overall': (payload.get('legal40k_mean') or {}).get('Overall'),
        'any_above': payload.get('any_legal40k_seed_above_visible_leader'),
        'integrity_errors': payload.get('collation_integrity_errors'),
        'out_json': rel(OUT_JSON),
        'note': rel(OUT_MD),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
