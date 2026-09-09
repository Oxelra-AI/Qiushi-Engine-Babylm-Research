#!/usr/bin/env python3
"""research: large CPU audit for relation-biased masking on the legal-40k route.

This script performs no model training, no GPU work, no official evaluation, and no
benchmark-text-based selection.  It reads the exact allowed 10M compact_view_reinvest
pool and the legal 40k tokenizer, reconstructs the research relation-group marking
used by the dormant relation-biased trainer, and computes how WWM relation boosts
would reallocate masked-LM target pressure across the real pool.

The purpose is to make a possible post-40k route safer before any expensive launch:
verify that boost=2.0 is not only valid on the first 256 rows, quantify row-level
clipping and source/category effects, and compare it with more aggressive boosts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import time
from typing import Any

ROOT = Path('.').resolve()
STUDY = ROOT / "experiments/archive" / 'representation_and_objectives'
WS = STUDY
SCRIPT = WS / 'scripts' / 'relation_bias_accumulated_trainer.py'
POOL10 = ROOT / "experiments/archive" / 'frontier_consolidation' / 'data' / 'density_cleanqwen_overlay_medium_riskhard' / 'cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
TOKENIZER_40K = WS / 'data' / 'legal_representation_route_map' / 'tokenizers' / 'legal_byte_bpe_40k'
ROUTE_ASSETS = WS / 'data' / 'post40k_route_assets' / 'post40k_route_assets.json'
OUT_DIR = WS / 'data' / 'relation_bias_large_stream_audit'
OUT_JSON = OUT_DIR / 'relation_bias_large_stream_audit.json'
OUT_SOURCE_CSV = OUT_DIR / 'boost_budget_by_source.csv'
OUT_CATEGORY_CSV = OUT_DIR / 'boost_budget_by_category.csv'
OUT_ROW_CSV = OUT_DIR / 'row_relation_density_quantiles.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/relation_bias_large_stream_audit.md')

EXPECTED_POOL10_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_TOKENIZER_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
BOOSTS = [1.0, 1.5, 2.0, 3.0, 4.0]
BASE_PROB = 0.15
PROB_MAX = 0.6


def now_utc() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def q(xs: list[float], p: float) -> float | None:
    vals = sorted(float(x) for x in xs if math.isfinite(float(x)))
    if not vals:
        return None
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def sd(xs: list[float]) -> float | None:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else None


def add_num(d: dict[str, float], k: str, v: float) -> None:
    d[k] = d.get(k, 0.0) + float(v)


def import_trainer():
    spec = importlib.util.spec_from_file_location('relation_bias_accumulated_trainer', SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {SCRIPT}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def new_bucket() -> dict[str, Any]:
    return {
        'rows': 0,
        'words': 0,
        'visible_groups': 0,
        'visible_tokens': 0,
        'relation_groups': 0,
        'relation_tokens': 0,
        'rows_with_relation_group': 0,
        'row_relation_group_fracs': [],
        'row_relation_token_fracs': [],
        'category_candidate_groups': defaultdict(float),
        'category_candidate_tokens': defaultdict(float),
        'boosts': defaultdict(lambda: {
            'expected_selected_groups': 0.0,
            'expected_selected_tokens': 0.0,
            'expected_relation_selected_groups': 0.0,
            'expected_relation_selected_tokens': 0.0,
            'rows_clipped': 0,
            'rows_no_nonrelation': 0,
            'category_expected_groups': defaultdict(float),
            'category_expected_tokens': defaultdict(float),
        }),
    }


def finalize_bucket(b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        'rows': b['rows'],
        'words': b['words'],
        'visible_groups': b['visible_groups'],
        'visible_tokens': b['visible_tokens'],
        'relation_groups': b['relation_groups'],
        'relation_tokens': b['relation_tokens'],
        'rows_with_relation_group': b['rows_with_relation_group'],
        'visible_groups_per_word': b['visible_groups'] / b['words'] if b['words'] else None,
        'visible_tokens_per_word': b['visible_tokens'] / b['words'] if b['words'] else None,
        'relation_group_frac': b['relation_groups'] / b['visible_groups'] if b['visible_groups'] else None,
        'relation_token_frac': b['relation_tokens'] / b['visible_tokens'] if b['visible_tokens'] else None,
        'row_relation_group_frac_mean': mean(b['row_relation_group_fracs']),
        'row_relation_group_frac_p50': q(b['row_relation_group_fracs'], 0.50),
        'row_relation_group_frac_p90': q(b['row_relation_group_fracs'], 0.90),
        'row_relation_group_frac_p99': q(b['row_relation_group_fracs'], 0.99),
        'row_relation_token_frac_mean': mean(b['row_relation_token_fracs']),
        'row_relation_token_frac_p50': q(b['row_relation_token_fracs'], 0.50),
        'row_relation_token_frac_p90': q(b['row_relation_token_fracs'], 0.90),
        'row_relation_token_frac_p99': q(b['row_relation_token_fracs'], 0.99),
        'category_candidate_groups': dict(sorted(b['category_candidate_groups'].items())),
        'category_candidate_tokens': dict(sorted(b['category_candidate_tokens'].items())),
    }
    boost_out: dict[str, Any] = {}
    for boost, rec in sorted(b['boosts'].items(), key=lambda kv: float(kv[0])):
        boost_f = float(boost)
        expected_uniform_groups = BASE_PROB * b['visible_groups']
        expected_uniform_tokens = BASE_PROB * b['visible_tokens']
        boost_out[str(boost_f)] = {
            'expected_selected_groups': rec['expected_selected_groups'],
            'expected_selected_tokens': rec['expected_selected_tokens'],
            'expected_relation_selected_groups': rec['expected_relation_selected_groups'],
            'expected_relation_selected_tokens': rec['expected_relation_selected_tokens'],
            'selected_group_multiplier_vs_uniform': rec['expected_selected_groups'] / expected_uniform_groups if expected_uniform_groups else None,
            'target_token_multiplier_vs_uniform': rec['expected_selected_tokens'] / expected_uniform_tokens if expected_uniform_tokens else None,
            'relation_selected_group_frac': rec['expected_relation_selected_groups'] / rec['expected_selected_groups'] if rec['expected_selected_groups'] else None,
            'relation_target_token_frac': rec['expected_relation_selected_tokens'] / rec['expected_selected_tokens'] if rec['expected_selected_tokens'] else None,
            'rows_clipped': rec['rows_clipped'],
            'rows_clipped_frac': rec['rows_clipped'] / b['rows'] if b['rows'] else None,
            'rows_no_nonrelation': rec['rows_no_nonrelation'],
            'category_expected_groups': dict(sorted(rec['category_expected_groups'].items())),
            'category_expected_tokens': dict(sorted(rec['category_expected_tokens'].items())),
        }
    out['boosts'] = boost_out
    return out


def normalized_relation_probs(rb, boost: float, n_rel: int, n_non: int) -> tuple[float, float, bool]:
    return rb.normalized_relation_probs(BASE_PROB, boost, n_rel, n_non, PROB_MAX)


def analyze_pool(max_rows: int | None = None) -> dict[str, Any]:
    rb = import_trainer()
    tok = rb.base.make_portable_tokenizer(str(TOKENIZER_40K))
    cue_lexicon, multi_cues = rb.load_step54_lexicon()
    cue_single = rb.build_cue_single(cue_lexicon)
    categories = sorted(cue_lexicon.keys())
    special_ids = set(int(x) for x in tok.all_special_ids)
    word_start_cache: dict[int, bool] = {}

    def is_word_start(token_id: int) -> bool:
        v = word_start_cache.get(token_id)
        if v is None:
            s = tok.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and rb.base.is_word_start(str(s)))
            word_start_cache[token_id] = v
        return v

    global_bucket = new_bucket()
    by_source: dict[str, dict[str, Any]] = defaultdict(new_bucket)
    row_records: list[dict[str, Any]] = []
    exact_word_total = 0
    rows_seen = 0
    start = time.time()

    with POOL10.open('r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if max_rows is not None and idx >= max_rows:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            words = int(obj.get('words', len(text.split())))
            if len(text.split()) != words:
                raise RuntimeError({'row': idx, 'word_mismatch': [len(text.split()), words]})
            source = str(obj.get('source', 'unknown'))
            exact_word_total += words
            rows_seen += 1

            enc = tok(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_offsets_mapping=True,
            )
            input_ids = [int(x) for x in enc['input_ids']]
            attn = [int(x) for x in enc['attention_mask']]
            offsets = enc['offset_mapping']
            spans, cats_by_word = rb.relation_categories_by_word(text, cue_single, multi_cues)
            starts = [s for s, _e, _w in spans]
            groups: list[dict[str, Any]] = []
            gid = -1
            for i, tid in enumerate(input_ids):
                if attn[i] == 0 or tid in special_ids:
                    continue
                if gid < 0 or is_word_start(tid) or i == 0:
                    gid += 1
                    groups.append({'tokens': 0, 'cats': set()})
                groups[gid]['tokens'] += 1
                off_start, off_end = offsets[i]
                wi = rb.word_index_for_offset(starts, spans, int(off_start), int(off_end))
                if wi is not None and 0 <= wi < len(cats_by_word):
                    groups[gid]['cats'].update(cats_by_word[wi])

            visible_groups = len(groups)
            visible_tokens = sum(int(g['tokens']) for g in groups)
            relation_groups = sum(1 for g in groups if g['cats'])
            relation_tokens = sum(int(g['tokens']) for g in groups if g['cats'])
            row_rg_frac = relation_groups / visible_groups if visible_groups else 0.0
            row_rt_frac = relation_tokens / visible_tokens if visible_tokens else 0.0
            row_records.append({
                'row_index': idx,
                'source': source,
                'words': words,
                'visible_groups': visible_groups,
                'visible_tokens': visible_tokens,
                'relation_groups': relation_groups,
                'relation_tokens': relation_tokens,
                'relation_group_frac': row_rg_frac,
                'relation_token_frac': row_rt_frac,
            })

            buckets = [global_bucket, by_source[source]]
            for b in buckets:
                b['rows'] += 1
                b['words'] += words
                b['visible_groups'] += visible_groups
                b['visible_tokens'] += visible_tokens
                b['relation_groups'] += relation_groups
                b['relation_tokens'] += relation_tokens
                b['rows_with_relation_group'] += int(relation_groups > 0)
                b['row_relation_group_fracs'].append(row_rg_frac)
                b['row_relation_token_fracs'].append(row_rt_frac)
                for g in groups:
                    cats = g['cats']
                    for cat in cats:
                        b['category_candidate_groups'][cat] += 1.0
                        b['category_candidate_tokens'][cat] += float(g['tokens'])

            n_non = visible_groups - relation_groups
            for boost in BOOSTS:
                rel_p, nonrel_p, clipped = normalized_relation_probs(rb, boost, relation_groups, n_non)
                for b in buckets:
                    br = b['boosts'][str(boost)]
                    br['expected_selected_groups'] += rel_p * relation_groups + nonrel_p * n_non
                    # target tokens count every token in a selected WWM group, same objective geometry as trainer labels.
                    rel_tok = 0.0
                    nonrel_tok = 0.0
                    for g in groups:
                        if g['cats']:
                            rel_tok += rel_p * float(g['tokens'])
                        else:
                            nonrel_tok += nonrel_p * float(g['tokens'])
                    br['expected_selected_tokens'] += rel_tok + nonrel_tok
                    br['expected_relation_selected_groups'] += rel_p * relation_groups
                    br['expected_relation_selected_tokens'] += rel_tok
                    br['rows_clipped'] += int(clipped)
                    br['rows_no_nonrelation'] += int(n_non == 0 and visible_groups > 0)
                    for g in groups:
                        if not g['cats']:
                            continue
                        for cat in g['cats']:
                            br['category_expected_groups'][cat] += rel_p
                            br['category_expected_tokens'][cat] += rel_p * float(g['tokens'])

            if rows_seen % 10000 == 0:
                print(json.dumps({'event': 'progress', 'rows_seen': rows_seen, 'words': exact_word_total, 'elapsed_sec': round(time.time() - start, 1)}), flush=True)

    if max_rows is None:
        if rows_seen != 64740 or exact_word_total != 10_000_000:
            raise RuntimeError({'pool_count_mismatch': {'rows_seen': rows_seen, 'words': exact_word_total}})

    finalized_global = finalize_bucket(global_bucket)
    finalized_by_source = {k: finalize_bucket(v) for k, v in sorted(by_source.items())}

    # Row-density quantiles overall and by source, including clipping fractions by boost.
    row_quantiles: list[dict[str, Any]] = []
    for source_name, vals in [('ALL', row_records)] + sorted([(s, [r for r in row_records if r['source'] == s]) for s in by_source]):
        rg = [float(r['relation_group_frac']) for r in vals]
        rt = [float(r['relation_token_frac']) for r in vals]
        row_quantiles.append({
            'source': source_name,
            'rows': len(vals),
            'words': sum(int(r['words']) for r in vals),
            'relation_group_frac_mean': mean(rg),
            'relation_group_frac_p50': q(rg, 0.50),
            'relation_group_frac_p90': q(rg, 0.90),
            'relation_group_frac_p95': q(rg, 0.95),
            'relation_group_frac_p99': q(rg, 0.99),
            'relation_group_frac_max': max(rg) if rg else None,
            'relation_token_frac_mean': mean(rt),
            'relation_token_frac_p50': q(rt, 0.50),
            'relation_token_frac_p90': q(rt, 0.90),
            'relation_token_frac_p95': q(rt, 0.95),
            'relation_token_frac_p99': q(rt, 0.99),
            'relation_token_frac_max': max(rt) if rt else None,
        })

    # Source rows for CSV.
    source_rows: list[dict[str, Any]] = []
    for source, b in [('ALL', finalized_global)] + list(finalized_by_source.items()):
        for boost in BOOSTS:
            rec = b['boosts'][str(float(boost))]
            source_rows.append({
                'source': source,
                'boost': boost,
                'rows': b['rows'],
                'words': b['words'],
                'relation_group_frac': b['relation_group_frac'],
                'relation_token_frac': b['relation_token_frac'],
                'selected_group_multiplier_vs_uniform': rec['selected_group_multiplier_vs_uniform'],
                'target_token_multiplier_vs_uniform': rec['target_token_multiplier_vs_uniform'],
                'relation_selected_group_frac': rec['relation_selected_group_frac'],
                'relation_target_token_frac': rec['relation_target_token_frac'],
                'rows_clipped': rec['rows_clipped'],
                'rows_clipped_frac': rec['rows_clipped_frac'],
            })

    category_rows: list[dict[str, Any]] = []
    all_cats = categories
    for cat in all_cats:
        cand_g = float(finalized_global['category_candidate_groups'].get(cat, 0.0))
        cand_t = float(finalized_global['category_candidate_tokens'].get(cat, 0.0))
        for boost in BOOSTS:
            rec = finalized_global['boosts'][str(float(boost))]
            exp_g = float(rec['category_expected_groups'].get(cat, 0.0))
            exp_t = float(rec['category_expected_tokens'].get(cat, 0.0))
            category_rows.append({
                'category': cat,
                'boost': boost,
                'candidate_groups': cand_g,
                'candidate_tokens': cand_t,
                'candidate_group_frac_of_all_visible_groups': cand_g / finalized_global['visible_groups'] if finalized_global['visible_groups'] else None,
                'candidate_token_frac_of_all_visible_tokens': cand_t / finalized_global['visible_tokens'] if finalized_global['visible_tokens'] else None,
                'expected_selected_groups': exp_g,
                'expected_selected_tokens': exp_t,
                'expected_group_multiplier_vs_uniform_category': exp_g / (BASE_PROB * cand_g) if cand_g else None,
                'expected_token_multiplier_vs_uniform_category': exp_t / (BASE_PROB * cand_t) if cand_t else None,
                'expected_group_frac_of_all_selected_groups': exp_g / rec['expected_selected_groups'] if rec['expected_selected_groups'] else None,
                'expected_token_frac_of_all_selected_tokens': exp_t / rec['expected_selected_tokens'] if rec['expected_selected_tokens'] else None,
            })

    return {
        'status': 'RELATION_BIAS_LARGE_STREAM_AUDIT',
        'created_utc': now_utc(),
        'elapsed_sec': round(time.time() - start, 1),
        'purpose': 'CPU-only audit of dormant legal40k relation-biased WWM masking before any expensive post-40k launch.',
        'inputs': {
            'pool10': rel(POOL10),
            'pool10_sha256': sha256_file(POOL10),
            'expected_pool10_sha256': EXPECTED_POOL10_SHA,
            'tokenizer_40k': rel(TOKENIZER_40K),
            'tokenizer_json_sha256': sha256_file(TOKENIZER_40K / 'tokenizer.json'),
            'expected_tokenizer_json_sha256': EXPECTED_TOKENIZER_SHA,
            'trainer': rel(SCRIPT),
            'trainer_sha256': sha256_file(SCRIPT),
            'route_assets': rel(ROUTE_ASSETS),
            'max_rows': max_rows,
            'boosts': BOOSTS,
            'base_prob': BASE_PROB,
            'relation_prob_max': PROB_MAX,
            'notes': 'Relation cue lexicon is imported from research and applied only to training text; official evaluation text is not read.',
        },
        'checks': {
            'pool10_sha_matches': sha256_file(POOL10) == EXPECTED_POOL10_SHA,
            'tokenizer_sha_matches': sha256_file(TOKENIZER_40K / 'tokenizer.json') == EXPECTED_TOKENIZER_SHA,
            'rows_seen': rows_seen,
            'words_seen': exact_word_total,
            'full_pool_counts_match': (max_rows is not None) or (rows_seen == 64740 and exact_word_total == 10_000_000),
        },
        'global_summary': finalized_global,
        'by_source': finalized_by_source,
        'row_relation_density_quantiles': row_quantiles,
        'csv_files': {
            'source': rel(OUT_SOURCE_CSV),
            'category': rel(OUT_CATEGORY_CSV),
            'row_quantiles': rel(OUT_ROW_CSV),
        },
        'interpretation': build_interpretation(finalized_global, finalized_by_source),
    }


def build_interpretation(global_summary: dict[str, Any], by_source: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    b2 = global_summary['boosts']['2.0']
    b3 = global_summary['boosts']['3.0']
    b4 = global_summary['boosts']['4.0']
    lines.append(
        f"Boost 2.0 gives relation selected-group fraction {b2['relation_selected_group_frac']:.4f} "
        f"and relation target-token fraction {b2['relation_target_token_frac']:.4f}; "
        f"selected-group multiplier {b2['selected_group_multiplier_vs_uniform']:.6f}, "
        f"target-token multiplier {b2['target_token_multiplier_vs_uniform']:.6f}."
    )
    lines.append(
        f"Boost 2.0 row clipping fraction is {b2['rows_clipped_frac']:.6f}; "
        "low clipping supports it as the cleanest relation-pressure intervention if legal40k endpoints motivate this route."
    )
    lines.append(
        f"Boost 3.0 raises relation selected-group fraction to {b3['relation_selected_group_frac']:.4f} "
        f"but clips {b3['rows_clipped_frac']:.6f} of rows and lowers total target tokens to multiplier {b3['target_token_multiplier_vs_uniform']:.6f}."
    )
    lines.append(
        f"Boost 4.0 raises relation selected-group fraction to {b4['relation_selected_group_frac']:.4f} "
        f"with row clipping {b4['rows_clipped_frac']:.6f} and stronger target-token reduction {b4['target_token_multiplier_vs_uniform']:.6f}; it is less clean as a first route."
    )
    source_b2 = []
    for source, rec in by_source.items():
        br = rec['boosts']['2.0']
        source_b2.append((source, br['relation_selected_group_frac'], br['target_token_multiplier_vs_uniform'], br['rows_clipped_frac']))
    source_b2.sort(key=lambda x: x[1])
    if source_b2:
        lo = source_b2[0]
        hi = source_b2[-1]
        lines.append(
            f"Across source classes under boost 2.0, relation selected-group fraction ranges from {lo[1]:.4f} ({lo[0]}) "
            f"to {hi[1]:.4f} ({hi[0]}), so source composition changes the dose but the intervention remains broad rather than confined to the compact block."
        )
    return lines


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Rebuild CSV rows from payload to keep JSON complete and CSV easy to inspect.
    source_rows: list[dict[str, Any]] = []
    for source, b in [('ALL', payload['global_summary'])] + list(payload['by_source'].items()):
        for boost in BOOSTS:
            rec = b['boosts'][str(float(boost))]
            source_rows.append({
                'source': source,
                'boost': boost,
                'rows': b['rows'],
                'words': b['words'],
                'relation_group_frac': b['relation_group_frac'],
                'relation_token_frac': b['relation_token_frac'],
                'selected_group_multiplier_vs_uniform': rec['selected_group_multiplier_vs_uniform'],
                'target_token_multiplier_vs_uniform': rec['target_token_multiplier_vs_uniform'],
                'relation_selected_group_frac': rec['relation_selected_group_frac'],
                'relation_target_token_frac': rec['relation_target_token_frac'],
                'rows_clipped': rec['rows_clipped'],
                'rows_clipped_frac': rec['rows_clipped_frac'],
            })
    category_rows: list[dict[str, Any]] = []
    cats = sorted(payload['global_summary']['category_candidate_groups'].keys())
    for cat in cats:
        cand_g = float(payload['global_summary']['category_candidate_groups'].get(cat, 0.0))
        cand_t = float(payload['global_summary']['category_candidate_tokens'].get(cat, 0.0))
        for boost in BOOSTS:
            rec = payload['global_summary']['boosts'][str(float(boost))]
            exp_g = float(rec['category_expected_groups'].get(cat, 0.0))
            exp_t = float(rec['category_expected_tokens'].get(cat, 0.0))
            category_rows.append({
                'category': cat,
                'boost': boost,
                'candidate_groups': cand_g,
                'candidate_tokens': cand_t,
                'candidate_group_frac_of_all_visible_groups': cand_g / payload['global_summary']['visible_groups'] if payload['global_summary']['visible_groups'] else None,
                'candidate_token_frac_of_all_visible_tokens': cand_t / payload['global_summary']['visible_tokens'] if payload['global_summary']['visible_tokens'] else None,
                'expected_selected_groups': exp_g,
                'expected_selected_tokens': exp_t,
                'expected_group_multiplier_vs_uniform_category': exp_g / (BASE_PROB * cand_g) if cand_g else None,
                'expected_token_multiplier_vs_uniform_category': exp_t / (BASE_PROB * cand_t) if cand_t else None,
                'expected_group_frac_of_all_selected_groups': exp_g / rec['expected_selected_groups'] if rec['expected_selected_groups'] else None,
                'expected_token_frac_of_all_selected_tokens': exp_t / rec['expected_selected_tokens'] if rec['expected_selected_tokens'] else None,
            })
    row_rows = payload['row_relation_density_quantiles']
    write_csv(OUT_SOURCE_CSV, source_rows)
    write_csv(OUT_CATEGORY_CSV, category_rows)
    write_csv(OUT_ROW_CSV, row_rows)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    g = payload['global_summary']
    b2 = g['boosts']['2.0']
    b3 = g['boosts']['3.0']
    b4 = g['boosts']['4.0']
    lines = [
        '# research relation-biased masking large-pool audit\n\n',
        'CPU-only audit on the exact allowed 10M compact_view_reinvest pool with the legal 40k tokenizer. No model training, no GPU work, no official evaluation text.\n\n',
        '## Integrity\n\n',
        f"- Pool SHA matched expected: `{payload['checks']['pool10_sha_matches']}`; tokenizer SHA matched expected: `{payload['checks']['tokenizer_sha_matches']}`.\n",
        f"- Rows/words audited: `{payload['checks']['rows_seen']}` / `{payload['checks']['words_seen']}`.\n",
        '\n## Global visible relation substrate\n\n',
        f"- Visible groups: `{g['visible_groups']}`; visible tokens: `{g['visible_tokens']}`.\n",
        f"- Relation-bearing groups: `{g['relation_groups']}` (`{g['relation_group_frac']:.6f}`); relation-bearing tokens: `{g['relation_tokens']}` (`{g['relation_token_frac']:.6f}`).\n",
        f"- Row relation-group fraction p50/p90/p99: `{g['row_relation_group_frac_p50']:.4f}` / `{g['row_relation_group_frac_p90']:.4f}` / `{g['row_relation_group_frac_p99']:.4f}`.\n",
        '\n## Candidate boosts\n\n',
        f"- Boost 2.0: relation selected-group fraction `{b2['relation_selected_group_frac']:.6f}`, relation target-token fraction `{b2['relation_target_token_frac']:.6f}`, selected-group multiplier `{b2['selected_group_multiplier_vs_uniform']:.6f}`, target-token multiplier `{b2['target_token_multiplier_vs_uniform']:.6f}`, clipped rows `{b2['rows_clipped']}` (`{b2['rows_clipped_frac']:.6f}`).\n",
        f"- Boost 3.0: relation selected-group fraction `{b3['relation_selected_group_frac']:.6f}`, relation target-token fraction `{b3['relation_target_token_frac']:.6f}`, selected-group multiplier `{b3['selected_group_multiplier_vs_uniform']:.6f}`, target-token multiplier `{b3['target_token_multiplier_vs_uniform']:.6f}`, clipped rows `{b3['rows_clipped']}` (`{b3['rows_clipped_frac']:.6f}`).\n",
        f"- Boost 4.0: relation selected-group fraction `{b4['relation_selected_group_frac']:.6f}`, relation target-token fraction `{b4['relation_target_token_frac']:.6f}`, selected-group multiplier `{b4['selected_group_multiplier_vs_uniform']:.6f}`, target-token multiplier `{b4['target_token_multiplier_vs_uniform']:.6f}`, clipped rows `{b4['rows_clipped']}` (`{b4['rows_clipped_frac']:.6f}`).\n",
        '\n## Interpretation\n\n',
    ]
    for item in payload['interpretation']:
        lines.append(f'- {item}\n')
    lines.extend([
        '\n## Files\n\n',
        f"- JSON: `{rel(OUT_JSON)}`\n",
        f"- Source CSV: `{rel(OUT_SOURCE_CSV)}`\n",
        f"- Category CSV: `{rel(OUT_CATEGORY_CSV)}`\n",
        f"- Row quantiles CSV: `{rel(OUT_ROW_CSV)}`\n",
    ])
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(''.join(lines), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_rows', type=int, default=0, help='0 means full 10M pool')
    args = ap.parse_args()
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    payload = analyze_pool(max_rows=max_rows)
    write_outputs(payload)
    print(json.dumps({
        'status': payload['status'],
        'rows_seen': payload['checks']['rows_seen'],
        'words_seen': payload['checks']['words_seen'],
        'elapsed_sec': payload['elapsed_sec'],
        'boost2_relation_selected_group_frac': payload['global_summary']['boosts']['2.0']['relation_selected_group_frac'],
        'boost2_target_token_multiplier': payload['global_summary']['boosts']['2.0']['target_token_multiplier_vs_uniform'],
        'boost2_rows_clipped_frac': payload['global_summary']['boosts']['2.0']['rows_clipped_frac'],
        'json': rel(OUT_JSON),
        'note': rel(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
