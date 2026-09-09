#!/usr/bin/env python3
"""research: build a source/frequency-matched non-relation cue control audit.

No model training, no GPU work, no official evaluation text.  This script reads the
exact 10M BabyLM training pool and the legal40k tokenizer, labels relation groups
with the research lexicon, then builds a lexical non-relation cue set whose visible
word-group occurrence count and source distribution approximate the relation set.

Purpose: if relation-cue WWM boost later improves official scores, a matched
non-relation cue control can distinguish relation/predicate pressure from merely
boosting a frequent source-distributed subset of words.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import random
import statistics
import time
from typing import Any

ROOT = Path('.').resolve()
STUDY = ROOT / 'experiments/archive/representation_and_objectives'
WS = STUDY
REL_TRAINER = WS / 'scripts/relation_bias_accumulated_trainer.py'
POOL10 = ROOT / 'experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl'
TOKENIZER_40K = WS / 'data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
OUT_DIR = WS / 'data/matched_nonrelation_control_audit'
OUT_JSON = OUT_DIR / 'matched_nonrelation_control_audit.json'
OUT_WORDS_CSV = OUT_DIR / 'matched_nonrelation_words.csv'
OUT_SOURCE_CSV = OUT_DIR / 'relation_vs_matched_control_by_source.csv'
OUT_BOOST_CSV = OUT_DIR / 'matched_control_boost_budget.csv'
NOTE = (ROOT / 'research/notes/representation_and_objectives/68_matched_nonrelation_control_audit.md')

EXPECTED_POOL10_SHA = '215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23'
EXPECTED_TOKENIZER_SHA = '94b44bcf5901d097e20294e8220740758eac29af3ec20d58d8da867c85197758'
BASE_PROB = 0.15
BOOST = 2.0
PROB_MAX = 0.6
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")


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


def import_trainer():
    spec = importlib.util.spec_from_file_location('relation_bias_accumulated_trainer', REL_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {REL_TRAINER}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def primary_norm(word: str, rb) -> str:
    norms = rb.word_norms(word)
    return norms[0] if norms else ''


def source_vec(counter: Counter[str], sources: list[str]) -> list[float]:
    total = sum(counter.values())
    if total <= 0:
        return [0.0 for _ in sources]
    return [float(counter.get(s, 0)) / float(total) for s in sources]


def l1(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def choose_control_words(type_records: dict[str, dict[str, Any]], target: dict[str, Any], max_words: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    sources = target['sources']
    target_vec = target['source_vec']
    target_count = target['groups']
    target_mean_len = target['tokens'] / target['groups']
    # Candidate pool excludes relation-marked, numeric-only, one-character, and ultra-rare words.
    candidates = []
    for w, r in type_records.items():
        if r.get('is_relation'):
            continue
        if r['groups'] < 50:
            continue
        if len(w) < 2:
            continue
        if all(ch.isdigit() for ch in w):
            continue
        vec = source_vec(r['source_counts'], sources)
        freq_ratio = math.log((r['groups'] + 1.0) / (target_count / max(1, max_words) + 1.0))
        # This is only for ordering; final greedy objective uses actual aggregate counts.
        r['candidate_source_l1_to_relation'] = l1(vec, target_vec)
        r['candidate_mean_tokens_per_group'] = r['tokens'] / r['groups']
        r['candidate_freq_order_score'] = abs(freq_ratio) + 0.5 * r['candidate_source_l1_to_relation']
        candidates.append(w)
    rng = random.Random(seed)
    # Start with a frequency-stratified shortlist to avoid all mass coming from a few function words.
    candidates.sort(key=lambda w: (type_records[w]['candidate_freq_order_score'], -type_records[w]['groups']))
    shortlist = candidates[:max(5000, min(len(candidates), max_words * 80))]
    rng.shuffle(shortlist)

    selected: list[str] = []
    agg_sources = Counter()
    agg_groups = 0
    agg_tokens = 0
    remaining = set(shortlist)

    def objective(groups: int, tokens: int, sources_counter: Counter[str]) -> float:
        if groups <= 0:
            return 1e9
        vec = source_vec(sources_counter, sources)
        count_err = abs(groups - target_count) / target_count
        source_err = l1(vec, target_vec)
        mean_len = tokens / groups
        len_err = abs(mean_len - target_mean_len) / target_mean_len
        # Count and source are primary; token length matters because MLM target count differs.
        return 4.0 * count_err + 2.0 * source_err + 0.75 * len_err

    best_obj = objective(agg_groups, agg_tokens, agg_sources)
    while remaining and len(selected) < max_words:
        # Consider a bounded random+best slice each iteration for speed.
        cand_list = list(remaining)
        cand_list.sort(key=lambda w: abs((agg_groups + type_records[w]['groups']) - target_count))
        probe = cand_list[:400]
        if len(cand_list) > 400:
            probe += rng.sample(cand_list[400:], min(400, len(cand_list) - 400))
        best_w = None
        best_new = None
        best_score = None
        for w in probe:
            r = type_records[w]
            ns = agg_sources.copy(); ns.update(r['source_counts'])
            ng = agg_groups + int(r['groups'])
            nt = agg_tokens + int(r['tokens'])
            sc = objective(ng, nt, ns)
            if best_score is None or sc < best_score:
                best_score = sc; best_w = w; best_new = (ng, nt, ns)
        if best_w is None or best_new is None:
            break
        # Allow additions until count is close; after that stop if objective stops improving strongly.
        selected.append(best_w)
        remaining.remove(best_w)
        agg_groups, agg_tokens, agg_sources = best_new
        best_obj = float(best_score)
        if agg_groups >= target_count * 0.995 and agg_groups <= target_count * 1.005:
            # Try a few more substitutions would be better, but for a control audit this is close enough.
            break
        if agg_groups > target_count * 1.02:
            break

    summary = {
        'selected_count': len(selected),
        'groups': agg_groups,
        'tokens': agg_tokens,
        'source_counts': dict(agg_sources),
        'source_vec': source_vec(agg_sources, sources),
        'groups_ratio_to_relation': agg_groups / target_count if target_count else None,
        'tokens_ratio_to_relation': agg_tokens / target['tokens'] if target['tokens'] else None,
        'source_l1_to_relation': l1(source_vec(agg_sources, sources), target_vec),
        'mean_tokens_per_group': agg_tokens / agg_groups if agg_groups else None,
        'relation_mean_tokens_per_group': target_mean_len,
        'objective': best_obj,
    }
    return selected, summary


def normalized_relation_probs(boost: float, n_target: int, n_other: int) -> tuple[float, float, bool]:
    n_total = n_target + n_other
    if n_total <= 0:
        return BASE_PROB, BASE_PROB, False
    rel_prob = min(PROB_MAX, BASE_PROB * boost)
    clipped = rel_prob < BASE_PROB * boost - 1e-12
    if n_target == 0:
        return BASE_PROB, BASE_PROB, clipped
    if n_other == 0:
        return BASE_PROB, 0.0, clipped or rel_prob != BASE_PROB
    nonrel = (BASE_PROB * n_total - rel_prob * n_target) / n_other
    if nonrel < 0:
        rel_prob = BASE_PROB * n_total / n_target
        nonrel = 0.0
        clipped = True
    if nonrel > 1:
        nonrel = 1.0
        clipped = True
    return rel_prob, nonrel, clipped


def boost_budget(rows: list[dict[str, Any]], target_key: str) -> dict[str, Any]:
    out = {
        'rows': len(rows),
        'candidate_groups': 0,
        'candidate_tokens': 0,
        'target_groups': 0,
        'target_tokens': 0,
        'expected_selected_groups': 0.0,
        'expected_selected_tokens': 0.0,
        'expected_target_selected_groups': 0.0,
        'expected_target_selected_tokens': 0.0,
        'rows_clipped': 0,
    }
    for r in rows:
        total_g = int(r['visible_groups'])
        total_t = int(r['visible_tokens'])
        target_g = int(r[target_key + '_groups'])
        target_t = int(r[target_key + '_tokens'])
        other_g = total_g - target_g
        other_t = total_t - target_t
        p_t, p_o, clipped = normalized_relation_probs(BOOST, target_g, other_g)
        out['candidate_groups'] += total_g
        out['candidate_tokens'] += total_t
        out['target_groups'] += target_g
        out['target_tokens'] += target_t
        out['expected_selected_groups'] += p_t * target_g + p_o * other_g
        out['expected_selected_tokens'] += p_t * target_t + p_o * other_t
        out['expected_target_selected_groups'] += p_t * target_g
        out['expected_target_selected_tokens'] += p_t * target_t
        out['rows_clipped'] += int(clipped)
    out['target_group_frac'] = out['target_groups'] / out['candidate_groups'] if out['candidate_groups'] else None
    out['target_token_frac'] = out['target_tokens'] / out['candidate_tokens'] if out['candidate_tokens'] else None
    out['selected_group_multiplier_vs_uniform'] = out['expected_selected_groups'] / (BASE_PROB * out['candidate_groups']) if out['candidate_groups'] else None
    out['target_token_multiplier_vs_uniform'] = out['expected_selected_tokens'] / (BASE_PROB * out['candidate_tokens']) if out['candidate_tokens'] else None
    out['target_selected_group_frac'] = out['expected_target_selected_groups'] / out['expected_selected_groups'] if out['expected_selected_groups'] else None
    out['target_selected_token_frac'] = out['expected_target_selected_tokens'] / out['expected_selected_tokens'] if out['expected_selected_tokens'] else None
    out['rows_clipped_frac'] = out['rows_clipped'] / len(rows) if rows else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_control_words', type=int, default=400)
    ap.add_argument('--seed', type=int, default=6801)
    args = ap.parse_args()
    start = time.time()
    rb = import_trainer()
    tok = rb.base.make_portable_tokenizer(str(TOKENIZER_40K))
    cue_lexicon, multi_cues = rb.load_step54_lexicon()
    cue_single = rb.build_cue_single(cue_lexicon)
    special_ids = set(int(x) for x in tok.all_special_ids)
    word_start_cache: dict[int, bool] = {}
    def is_word_start(token_id: int) -> bool:
        v = word_start_cache.get(token_id)
        if v is None:
            s = tok.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and rb.base.is_word_start(str(s)))
            word_start_cache[token_id] = v
        return v

    type_records: dict[str, dict[str, Any]] = defaultdict(lambda: {'groups': 0, 'tokens': 0, 'source_counts': Counter(), 'source_tokens': Counter(), 'is_relation': False})
    row_records: list[dict[str, Any]] = []
    relation_source_counts = Counter()
    relation_source_tokens = Counter()
    all_source_words = Counter()
    rows = 0; words = 0

    with POOL10.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj['text'])
            src = str(obj.get('source', 'unknown'))
            w = int(obj.get('words', len(text.split())))
            if len(text.split()) != w:
                raise RuntimeError({'row': rows, 'word_mismatch': [len(text.split()), w]})
            rows += 1; words += w; all_source_words[src] += w
            enc = tok(text, add_special_tokens=False, truncation=True, max_length=256, padding='max_length', return_offsets_mapping=True)
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
                    groups.append({'tokens': 0, 'norms': [], 'cats': set()})
                groups[gid]['tokens'] += 1
                off_start, off_end = offsets[i]
                wi = rb.word_index_for_offset(starts, spans, int(off_start), int(off_end))
                if wi is not None and 0 <= wi < len(cats_by_word):
                    groups[gid]['cats'].update(cats_by_word[wi])
                    if wi < len(spans):
                        n = primary_norm(spans[wi][2], rb)
                        if n:
                            groups[gid]['norms'].append(n)
            relation_groups = 0; relation_tokens = 0
            row_norms = []
            for g in groups:
                norms = g['norms'] or []
                norm = norms[0] if norms else ''
                if not norm:
                    continue
                is_rel = bool(g['cats'])
                tr = type_records[norm]
                tr['groups'] += 1
                tr['tokens'] += int(g['tokens'])
                tr['source_counts'][src] += 1
                tr['source_tokens'][src] += int(g['tokens'])
                tr['is_relation'] = bool(tr['is_relation'] or is_rel)
                row_norms.append((norm, int(g['tokens']), is_rel))
                if is_rel:
                    relation_groups += 1
                    relation_tokens += int(g['tokens'])
                    relation_source_counts[src] += 1
                    relation_source_tokens[src] += int(g['tokens'])
            row_records.append({
                'source': src,
                'words': w,
                'visible_groups': len(groups),
                'visible_tokens': sum(int(g['tokens']) for g in groups),
                'relation_groups': relation_groups,
                'relation_tokens': relation_tokens,
                # filled after selecting matched words
                '_norms': row_norms,
            })
            if rows % 10000 == 0:
                print(json.dumps({'event': 'progress', 'rows': rows, 'words': words, 'elapsed_sec': round(time.time() - start, 1)}), flush=True)

    if rows != 64740 or words != 10_000_000:
        raise RuntimeError({'count_mismatch': {'rows': rows, 'words': words}})
    sources = sorted(all_source_words)
    target = {
        'groups': sum(relation_source_counts.values()),
        'tokens': sum(relation_source_tokens.values()),
        'source_counts': dict(relation_source_counts),
        'source_tokens': dict(relation_source_tokens),
        'sources': sources,
        'source_vec': source_vec(relation_source_counts, sources),
    }
    selected, control_summary = choose_control_words(type_records, target, args.max_control_words, args.seed)
    selected_set = set(selected)

    # Fill row control counts.
    control_source_counts = Counter(); control_source_tokens = Counter()
    for rr in row_records:
        cg = 0; ct = 0
        for norm, ntoks, is_rel in rr['_norms']:
            if (not is_rel) and norm in selected_set:
                cg += 1; ct += int(ntoks)
        rr['control_groups'] = cg
        rr['control_tokens'] = ct
        control_source_counts[rr['source']] += cg
        control_source_tokens[rr['source']] += ct
        del rr['_norms']

    relation_budget = boost_budget(row_records, 'relation')
    control_budget = boost_budget(row_records, 'control')
    relation_by_source = {}
    control_by_source = {}
    source_rows = []
    for src in sources:
        sub = [r for r in row_records if r['source'] == src]
        rbgt = boost_budget(sub, 'relation')
        cbgt = boost_budget(sub, 'control')
        relation_by_source[src] = rbgt
        control_by_source[src] = cbgt
        source_rows.append({
            'source': src,
            'rows': len(sub),
            'words': sum(int(r['words']) for r in sub),
            'relation_groups': rbgt['target_groups'],
            'control_groups': cbgt['target_groups'],
            'relation_group_frac': rbgt['target_group_frac'],
            'control_group_frac': cbgt['target_group_frac'],
            'relation_boost2_selected_group_frac': rbgt['target_selected_group_frac'],
            'control_boost2_selected_group_frac': cbgt['target_selected_group_frac'],
            'relation_target_token_multiplier': rbgt['target_token_multiplier_vs_uniform'],
            'control_target_token_multiplier': cbgt['target_token_multiplier_vs_uniform'],
        })

    selected_rows = []
    for w in selected:
        r = type_records[w]
        selected_rows.append({
            'word': w,
            'groups': r['groups'],
            'tokens': r['tokens'],
            'mean_tokens_per_group': r['tokens'] / r['groups'] if r['groups'] else None,
            **{f'source_groups_{s}': int(r['source_counts'].get(s, 0)) for s in sources},
        })
    selected_rows.sort(key=lambda r: int(r['groups']), reverse=True)

    boost_rows = [
        {'target': 'relation', **{k: v for k, v in relation_budget.items() if k != 'rows'}},
        {'target': 'matched_nonrelation_control', **{k: v for k, v in control_budget.items() if k != 'rows'}},
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, table in [(OUT_WORDS_CSV, selected_rows), (OUT_SOURCE_CSV, source_rows), (OUT_BOOST_CSV, boost_rows)]:
        with path.open('w', encoding='utf-8', newline='') as f:
            fields = list(table[0].keys()) if table else []
            wtr = csv.DictWriter(f, fieldnames=fields)
            wtr.writeheader()
            for row in table:
                wtr.writerow(row)

    payload = {
        'status': 'MATCHED_NONRELATION_CONTROL_AUDIT',
        'created_utc': now_utc(),
        'elapsed_sec': round(time.time() - start, 1),
        'purpose': 'Training-only lexical-control audit for relation-cue WWM route; no model training/evaluation.',
        'inputs': {
            'pool10': rel(POOL10),
            'pool10_sha256': sha256_file(POOL10),
            'expected_pool10_sha256': EXPECTED_POOL10_SHA,
            'tokenizer_40k': rel(TOKENIZER_40K),
            'tokenizer_json_sha256': sha256_file(TOKENIZER_40K / 'tokenizer.json'),
            'expected_tokenizer_json_sha256': EXPECTED_TOKENIZER_SHA,
            'relation_trainer': rel(REL_TRAINER),
            'max_control_words': args.max_control_words,
            'seed': args.seed,
        },
        'checks': {
            'pool10_sha_matches': sha256_file(POOL10) == EXPECTED_POOL10_SHA,
            'tokenizer_sha_matches': sha256_file(TOKENIZER_40K / 'tokenizer.json') == EXPECTED_TOKENIZER_SHA,
            'rows_seen': rows,
            'words_seen': words,
            'counts_match': rows == 64740 and words == 10_000_000,
        },
        'relation_target': target,
        'matched_control_summary': control_summary,
        'relation_boost2_budget': relation_budget,
        'matched_control_boost2_budget': control_budget,
        'relation_by_source_boost2': relation_by_source,
        'matched_control_by_source_boost2': control_by_source,
        'selected_control_words_top40': selected_rows[:40],
        'files': {
            'json': rel(OUT_JSON),
            'selected_words_csv': rel(OUT_WORDS_CSV),
            'source_csv': rel(OUT_SOURCE_CSV),
            'boost_csv': rel(OUT_BOOST_CSV),
            'note': rel(NOTE),
        },
        'interpretation': [],
    }
    payload['interpretation'].append(
        f"Selected {control_summary['selected_count']} non-relation word types with group count ratio {control_summary['groups_ratio_to_relation']:.4f}, token count ratio {control_summary['tokens_ratio_to_relation']:.4f}, and source-distribution L1 distance {control_summary['source_l1_to_relation']:.4f} to the relation cue set."
    )
    payload['interpretation'].append(
        f"Under boost2, matched control target selected-group fraction is {control_budget['target_selected_group_frac']:.4f} versus relation {relation_budget['target_selected_group_frac']:.4f}; target-token multipliers are {control_budget['target_token_multiplier_vs_uniform']:.4f} versus relation {relation_budget['target_token_multiplier_vs_uniform']:.4f}."
    )
    payload['interpretation'].append(
        'This control is not a proposed first H100 route; it is a future mechanism control if relation-cue boosting itself produces useful official gains.'
    )

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [
        '# research matched non-relation lexical-control audit\n\n',
        'CPU-only, training-text-only audit. No model training, GPU, or official evaluation text.\n\n',
        '## Match quality\n\n',
        f"- Selected control word types: `{control_summary['selected_count']}`.\n",
        f"- Control/relation group ratio: `{control_summary['groups_ratio_to_relation']:.6f}`; token ratio: `{control_summary['tokens_ratio_to_relation']:.6f}`.\n",
        f"- Source-distribution L1 distance to relation set: `{control_summary['source_l1_to_relation']:.6f}`.\n",
        f"- Mean tokens/group control vs relation: `{control_summary['mean_tokens_per_group']:.4f}` vs `{control_summary['relation_mean_tokens_per_group']:.4f}`.\n",
        '\n## Boost2 comparison\n\n',
        f"- Relation boost2 selected target-group fraction: `{relation_budget['target_selected_group_frac']:.6f}`; target-token multiplier `{relation_budget['target_token_multiplier_vs_uniform']:.6f}`.\n",
        f"- Matched-control boost2 selected target-group fraction: `{control_budget['target_selected_group_frac']:.6f}`; target-token multiplier `{control_budget['target_token_multiplier_vs_uniform']:.6f}`.\n",
        f"- Rows clipped relation/control: `{relation_budget['rows_clipped']}` / `{control_budget['rows_clipped']}`.\n",
        '\n## Interpretation\n\n',
    ]
    for item in payload['interpretation']:
        lines.append(f'- {item}\n')
    lines.extend([
        '\n## Files\n\n',
        f"- JSON: `{rel(OUT_JSON)}`\n",
        f"- Selected words CSV: `{rel(OUT_WORDS_CSV)}`\n",
        f"- Source CSV: `{rel(OUT_SOURCE_CSV)}`\n",
        f"- Boost CSV: `{rel(OUT_BOOST_CSV)}`\n",
    ])
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({
        'status': payload['status'],
        'elapsed_sec': payload['elapsed_sec'],
        'selected_control_words': control_summary['selected_count'],
        'groups_ratio_to_relation': control_summary['groups_ratio_to_relation'],
        'source_l1_to_relation': control_summary['source_l1_to_relation'],
        'relation_boost2_group_frac': relation_budget['target_selected_group_frac'],
        'control_boost2_group_frac': control_budget['target_selected_group_frac'],
        'json': rel(OUT_JSON),
        'note': rel(NOTE),
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
