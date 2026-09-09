#!/usr/bin/env python3
"""research: materialize the missing MAX-dose source-breadth vertex.

The existing 2.64x MAX arms compare compact semantic second views with
hash-rotated source repetition and with clean-Qwen filler.  This script builds
the practitioner-relevant third allocation arm without generation:

    same MAX FineWeb source sentences + independent unused FineWeb sentences

where the independent source sentences spend exactly the compact-rewrite word
budget in each changed row.  The suffix/topup/filler and 100M pass order are
kept identical to the research MAX view/repeat arms.  The arm is a mechanism
instrument, not a leaderboard candidate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import time
from typing import Any, Iterable

ROOT = pathlib.Path('.').resolve()
WS = ROOT / 'experiments/archive/frontier_consolidation'
FULL_SOURCE_DEFAULT = ROOT / 'experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl'
PAIRS_DEFAULT = WS / 'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
VIEW_POOL_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'
VIEW_META_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
META_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json'
OUT_DIR_DEFAULT = WS / 'data/dose_2p64x_breadth_rowholdout_pools'
TOTAL_WORDS = 10_000_000
PASSES = 10
DEFAULT_SEED = 82926101
DEFAULT_STREAM_SEED = 82914124 + 7000
LABEL = 'compact_breadth_dose2p64x_matched_rowholdout'


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def norm(text: Any) -> str:
    return ' '.join(str(text or '').split())


def wc(text: Any) -> int:
    return len(norm(text).split())


def text_hash(text: Any) -> str:
    return hashlib.sha256(norm(text).lower().encode('utf-8')).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def iter_jsonl(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def stats(vals: list[int | float]) -> dict[str, Any]:
    if not vals:
        return {'n': 0}
    xs = sorted(float(v) for v in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(pos); hi = min(lo + 1, len(xs) - 1)
        a = pos - lo
        return xs[lo] * (1 - a) + xs[hi] * a
    return {
        'n': len(xs), 'sum': float(sum(xs)), 'min': xs[0], 'p01': q(0.01), 'p05': q(0.05),
        'p10': q(0.10), 'p25': q(0.25), 'median': q(0.50), 'p75': q(0.75),
        'p90': q(0.90), 'p95': q(0.95), 'p99': q(0.99), 'max': xs[-1],
        'mean': float(statistics.mean(xs)),
    }


def load_pairs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for d in iter_jsonl(path):
        pid = str(d.get('pair_id') or '')
        if not pid:
            raise RuntimeError('pair without pair_id')
        src = norm(d.get('source_text'))
        rew = norm(d.get('rewrite_text'))
        sw = int(d.get('source_words') or wc(src))
        rw = int(d.get('rewrite_words') or wc(rew))
        if sw != wc(src) or rw != wc(rew):
            raise RuntimeError(f'pair word mismatch {pid}')
        if pid in pairs:
            raise RuntimeError(f'duplicate pair_id {pid}')
        pairs[pid] = {
            **d,
            'pair_id': pid,
            'source_text': src,
            'rewrite_text': rew,
            'source_words': sw,
            'rewrite_words': rw,
            'pair_words': sw + rw,
            'source_hash': text_hash(src),
            'sentence_id': str(d.get('sentence_id') or ''),
            'doc_id': str(d.get('doc_id') or ''),
        }
    return pairs


def parse_pair_rows(meta_path: pathlib.Path, pairs: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    topup_meta: list[dict[str, Any]] = []
    for m in iter_jsonl(meta_path):
        pids = [str(x) for x in (m.get('pair_ids') or [])]
        if not pids:
            topup_meta.append(m)
            continue
        missing = [p for p in pids if p not in pairs]
        if missing:
            raise RuntimeError(f'row {m.get("row_index")} references missing pairs {missing[:3]}')
        sw = sum(int(pairs[p]['source_words']) for p in pids)
        rw = sum(int(pairs[p]['rewrite_words']) for p in pids)
        total = int(m.get('words') or 0)
        if sw + rw != total:
            raise RuntimeError(f'row {m.get("row_index")} words mismatch: {sw}+{rw}!={total}')
        source_text = norm(' '.join(str(pairs[p]['source_text']) for p in pids))
        if wc(source_text) != sw:
            raise RuntimeError(f'row {m.get("row_index")} source text count mismatch')
        pair_rows.append({
            'row_index': int(m['row_index']),
            'example_id': int(m['example_id']),
            'pair_ids': pids,
            'source_text': source_text,
            'source_words': sw,
            'companion_words': rw,
            'total_words': total,
            'component_sources': dict(m.get('component_sources') or {}),
        })
    if [r['row_index'] for r in pair_rows] != list(range(len(pair_rows))):
        raise RuntimeError('pair row indices are not the initial contiguous prefix')
    return pair_rows, topup_meta


def read_view_pool(path: pathlib.Path, pair_row_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    if sum(int(r.get('words') or wc(r.get('text'))) for r in rows) != TOTAL_WORDS:
        raise RuntimeError('view pool is not exact 10M')
    if len(rows) <= pair_row_count:
        raise RuntimeError('view pool too short')
    prefix = rows[:pair_row_count]
    suffix = rows[pair_row_count:]
    return prefix, suffix


def candidate_domain(d: dict[str, Any]) -> str:
    ents = d.get('entities') or []
    nums = d.get('numbers') or []
    text = norm(d.get('text')).lower()
    if nums:
        return 'quant_numeric'
    if any(w in text for w in ['because', 'therefore', 'caused', 'leads to', 'due to', 'after', 'before', 'while']):
        return 'causal_temporal'
    if any(w in text for w in ['science', 'energy', 'chemical', 'temperature', 'species', 'software', 'system']):
        return 'science_technical'
    if ents:
        return 'entity_factual'
    return 'no_domain'


def load_candidates(path: pathlib.Path, used_sentence_ids: set[str], used_hashes: set[str], seed: int, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    seen_hash: set[str] = set()
    candidates: list[dict[str, Any]] = []
    skipped = collections.Counter()
    docs = collections.Counter()
    for d in iter_jsonl(path):
        text = norm(d.get('text'))
        if not text:
            skipped['empty_text'] += 1
            continue
        sid = str(d.get('sentence_id') or '')
        h = text_hash(text)
        words = int(d.get('words') or wc(text))
        if words != wc(text):
            raise RuntimeError(f'candidate word mismatch sentence_id={sid}')
        if sid in used_sentence_ids or h in used_hashes:
            skipped['selected_max_source'] += 1
            continue
        if h in seen_hash:
            skipped['duplicate_hash'] += 1
            continue
        seen_hash.add(h)
        if words <= 0 or words > max_len:
            skipped['length_outside_allowed'] += 1
            continue
        flags = list(d.get('source_row_quality_flags') or [])
        c = {
            'sentence_id': sid,
            'source': str(d.get('source') or ''),
            'source_row': d.get('source_row'),
            'doc_id': str(d.get('doc_id') or f'docless::{h}'),
            'sent_index_in_row': d.get('sent_index_in_row'),
            'text': text,
            'words': words,
            'entities': list(d.get('entities') or []),
            'numbers': list(d.get('numbers') or []),
            'source_row_quality_flags': flags,
            'norm_hash': h,
            'primary_domain': candidate_domain(d),
            '_rand': rng.random(),
        }
        docs[c['doc_id']] += 1
        candidates.append(c)
    candidates.sort(key=lambda c: (len(c['source_row_quality_flags']), float(c['_rand'])))
    summary = {
        'rows': len(candidates),
        'words': int(sum(int(c['words']) for c in candidates)),
        'unique_docs': len(docs),
        'skipped': dict(skipped),
        'word_stats': stats([int(c['words']) for c in candidates]),
        'domain_rows': dict(collections.Counter(str(c['primary_domain']) for c in candidates).most_common()),
        'domain_words': dict(collections.Counter({}).most_common()),
    }
    dom_words = collections.Counter()
    flag_rows = collections.Counter()
    for c in candidates:
        dom_words[str(c['primary_domain'])] += int(c['words'])
        flag_rows['flagged' if c['source_row_quality_flags'] else 'no_flags'] += 1
    summary['domain_words'] = dict(dom_words.most_common())
    summary['flag_rows'] = dict(flag_rows.most_common())
    return candidates, summary


def generate_patterns(total: int, allowed_lengths: set[int], max_parts: int) -> list[tuple[int, ...]]:
    allowed = sorted(x for x in allowed_lengths if 0 < x <= total)
    out: list[tuple[int, ...]] = []
    def rec(start: int, remaining: int, parts: list[int]) -> None:
        if remaining == 0:
            out.append(tuple(parts))
            return
        if len(parts) >= max_parts:
            return
        for i in range(start, len(allowed)):
            w = allowed[i]
            if w > remaining:
                break
            parts.append(w)
            rec(i, remaining - w, parts)
            parts.pop()
    rec(0, total, [])
    out.sort(key=lambda p: (len(p), max(p)-min(p) if len(p) > 1 else 0, p))
    return out


def make_buckets(candidates: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        buckets[int(c['words'])].append(c)
    return buckets


def compact_bucket(bucket: list[dict[str, Any]], used: set[str], doc_counts: collections.Counter[str], doc_cap: int) -> list[dict[str, Any]]:
    return [c for c in bucket if str(c['norm_hash']) not in used and doc_counts[str(c['doc_id'])] < doc_cap]


def try_pattern(pattern: tuple[int, ...], buckets: dict[int, list[dict[str, Any]]], used: set[str], doc_counts: collections.Counter[str], doc_cap: int) -> list[dict[str, Any]] | None:
    chosen: list[dict[str, Any]] = []
    chosen_hashes: set[str] = set()
    temp_docs: collections.Counter[str] = collections.Counter()
    for L, need in sorted(collections.Counter(pattern).items()):
        found: list[dict[str, Any]] = []
        for c in buckets.get(L, []):
            h = str(c['norm_hash'])
            if h in used or h in chosen_hashes:
                continue
            doc = str(c['doc_id'])
            if doc_counts[doc] + temp_docs[doc] >= doc_cap:
                continue
            found.append(c)
            chosen_hashes.add(h)
            temp_docs[doc] += 1
            if len(found) == need:
                break
        if len(found) < need:
            return None
        chosen.extend(found)
    chosen.sort(key=lambda c: (int(c['words']), float(c['_rand'])))
    return chosen


def select_by_row(pair_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], doc_cap: int, max_parts: int, out_trace_dir: pathlib.Path) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    buckets = make_buckets(candidates)
    allowed = set(buckets)
    budgets = [int(r['companion_words']) for r in pair_rows]
    pattern_cache: dict[int, list[tuple[int, ...]]] = {b: generate_patterns(b, allowed, max_parts) for b in sorted(set(budgets))}
    missing = {b: 0 for b, pats in pattern_cache.items() if not pats}
    if missing:
        raise RuntimeError(f'no exact whole-sentence patterns for budgets: {sorted(missing)[:20]}')
    budget_counts = collections.Counter(budgets)
    # Harder and rarer budgets first; restore original row order in the output.
    row_order = sorted(range(len(pair_rows)), key=lambda i: (len(pattern_cache[budgets[i]]), budget_counts[budgets[i]], budgets[i], i))
    initial_counts = collections.Counter(int(c['words']) for c in candidates)
    used: set[str] = set()
    doc_counts: collections.Counter[str] = collections.Counter()
    chosen_by_row: list[list[dict[str, Any]]] = [[] for _ in pair_rows]
    fail_trace: list[dict[str, Any]] = []

    avail = {L: len(arr) for L, arr in buckets.items()}
    stale = {L: 0 for L in buckets}
    for step_i, row_i in enumerate(row_order):
        budget = budgets[row_i]
        feasible: list[tuple[tuple[float, ...], tuple[int, ...], list[dict[str, Any]]]] = []
        for pat in pattern_cache[budget]:
            pc = collections.Counter(pat)
            if any(avail.get(L, 0) < need for L, need in pc.items()):
                continue
            chosen = try_pattern(pat, buckets, used, doc_counts, doc_cap)
            if chosen is None:
                continue
            scarcity_now = sum(pc[L] / max(1, avail.get(L, 0)) for L in pc)
            scarcity_initial = sum(pc[L] / max(1, initial_counts.get(L, 0)) for L in pc)
            flagged = sum(1 for c in chosen if c.get('source_row_quality_flags'))
            domains = len({str(c.get('primary_domain')) for c in chosen})
            score = (len(pat), scarcity_now, scarcity_initial, flagged, -domains, max(pat)-min(pat) if len(pat) > 1 else 0)
            feasible.append((score, pat, chosen))
            if len(feasible) >= 24:
                break
        if not feasible:
            fail_trace.append({'selection_step': step_i, 'row_index': row_i, 'budget': budget, 'available_by_length': {str(k): int(v) for k, v in sorted(avail.items()) if v}})
            (out_trace_dir / 'max_breadth_fill_failure_trace.json').write_text(json.dumps({'fail_trace': fail_trace, 'rows_filled': step_i, 'doc_cap': doc_cap, 'max_parts': max_parts}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            raise RuntimeError(f'failed exact fill for row {row_i} budget {budget} after {step_i} rows; trace saved')
        feasible.sort(key=lambda x: x[0])
        _, pat, chosen = feasible[0]
        if sum(int(c['words']) for c in chosen) != budget:
            raise RuntimeError(f'internal pattern sum mismatch row {row_i}')
        chosen_by_row[row_i] = chosen
        for c in chosen:
            h = str(c['norm_hash'])
            if h in used:
                raise RuntimeError(f'duplicate selected hash {h}')
            used.add(h)
            doc_counts[str(c['doc_id'])] += 1
            L = int(c['words'])
            avail[L] = max(0, avail.get(L, 0) - 1)
            stale[L] = stale.get(L, 0) + 1
            if stale[L] >= 64:
                buckets[L] = compact_bucket(buckets[L], used, doc_counts, doc_cap)
                avail[L] = len(buckets[L])
                stale[L] = 0

    flat = [c for row in chosen_by_row for c in row]
    if sum(int(c['words']) for c in flat) != sum(budgets):
        raise RuntimeError('selected word total mismatch')
    doc_overlap_with_max_sources = 0  # filled by caller if needed
    summary = {
        'seeded_greedy_rule': 'harder_row_budgets_first; fewest whole sentences; avoid currently scarce lengths; prefer unflagged candidates; doc capped',
        'doc_cap': doc_cap,
        'max_parts_per_changed_row': max_parts,
        'selected_sentences': len(flat),
        'selected_words': int(sum(int(c['words']) for c in flat)),
        'selected_unique_docs': len(doc_counts),
        'max_doc_use': max(doc_counts.values()) if doc_counts else 0,
        'per_row_sentence_count_stats': stats([len(row) for row in chosen_by_row]),
        'selected_sentence_word_stats': stats([int(c['words']) for c in flat]),
        'selected_words_by_domain': dict(collections.Counter({}).most_common()),
        'selected_rows_by_domain': dict(collections.Counter(str(c['primary_domain']) for c in flat).most_common()),
        'selected_flag_rows': dict(collections.Counter('flagged' if c.get('source_row_quality_flags') else 'no_flags' for c in flat).most_common()),
        'row_budgets_by_part_count': {str(k): int(v) for k, v in sorted(collections.Counter(len(row) for row in chosen_by_row).items())},
        'length_pattern_availability': {str(k): len(v) for k, v in sorted(pattern_cache.items())},
    }
    dom_words = collections.Counter()
    for c in flat:
        dom_words[str(c['primary_domain'])] += int(c['words'])
    summary['selected_words_by_domain'] = dict(dom_words.most_common())
    return chosen_by_row, summary


def build_breadth_rows(pair_rows: list[dict[str, Any]], chosen_by_row: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    row_meta: list[dict[str, Any]] = []
    selected_sidecar: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row, chosen in zip(pair_rows, chosen_by_row):
        companion_text = norm(' '.join(str(c['text']) for c in chosen))
        if wc(companion_text) != int(row['companion_words']):
            raise RuntimeError(f'companion count mismatch row {row["row_index"]}')
        text = norm(f"{row['source_text']} {companion_text}")
        if wc(text) != int(row['total_words']):
            raise RuntimeError(f'total count mismatch row {row["row_index"]}')
        rows.append({'text': text, 'words': int(row['total_words']), 'example_id': int(row['example_id']), 'source': LABEL})
        row_meta.append({
            'row_index': int(row['row_index']),
            'example_id': int(row['example_id']),
            'words': int(row['total_words']),
            'pair_ids': list(row['pair_ids']),
            'common_max_source_words': int(row['source_words']),
            'breadth_companion_words': int(row['companion_words']),
            'breadth_source_ids': [str(c['sentence_id']) for c in chosen],
            'breadth_source_hashes': [str(c['norm_hash']) for c in chosen],
            'breadth_source_words': [int(c['words']) for c in chosen],
            'whole_sentences': True,
            'component_sources': {'max_source_words': int(row['source_words']), 'independent_breadth_words': int(row['companion_words'])},
        })
        for c in chosen:
            h = str(c['norm_hash'])
            if h not in seen:
                selected_sidecar.append({k: c.get(k) for k in ['sentence_id', 'source', 'source_row', 'doc_id', 'sent_index_in_row', 'text', 'words', 'entities', 'numbers', 'source_row_quality_flags', 'norm_hash', 'primary_domain']})
                seen.add(h)
    return rows, row_meta, selected_sidecar


def write_training(path: pathlib.Path, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    n = len(rows)
    total = 0
    order_hashes: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            order_hashes.append(sha256_text(','.join(map(str, order))))
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({'text': r['text'], 'words': int(r['words']), 'example_id': int(r['example_id']), 'source': str(r['source'])}, ensure_ascii=False) + '\n')
                total += int(r['words'])
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f'training stream total {total}')
    return {'path': rel(path), 'rows': n * PASSES, 'words': total, 'sha256': sha256_file(path), 'pass_order_hashes': order_hashes, 'seed': seed}


def word_total(rows: list[dict[str, Any]]) -> int:
    total = 0
    for i, r in enumerate(rows):
        text = norm(r.get('text'))
        w = int(r.get('words') or wc(text))
        if w != wc(text):
            raise RuntimeError(f'row {i} word field mismatch')
        total += w
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--full-source', default=str(FULL_SOURCE_DEFAULT))
    ap.add_argument('--pairs', default=str(PAIRS_DEFAULT))
    ap.add_argument('--view-pool', default=str(VIEW_POOL_DEFAULT))
    ap.add_argument('--view-meta', default=str(VIEW_META_DEFAULT))
    ap.add_argument('--research-meta', default=str(META_DEFAULT))
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--stream-seed', type=int, default=DEFAULT_STREAM_SEED)
    ap.add_argument('--doc-cap', type=int, default=6)
    ap.add_argument('--max-parts', type=int, default=5)
    ap.add_argument('--max-candidate-len', type=int, default=80)
    ap.add_argument('--write-training', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(pathlib.Path(args.pairs))
    pair_rows, topup_meta = parse_pair_rows(pathlib.Path(args.view_meta), pairs)
    view_prefix, suffix_rows = read_view_pool(pathlib.Path(args.view_pool), len(pair_rows))
    if [int(r['words']) for r in view_prefix] != [int(r['total_words']) for r in pair_rows]:
        raise RuntimeError('view prefix word sequence does not match pair row metadata')
    used_sentence_ids = {str(p['sentence_id']) for p in pairs.values()}
    used_hashes = {str(p['source_hash']) for p in pairs.values()}
    used_docs = {str(p['doc_id']) for p in pairs.values() if str(p.get('doc_id') or '')}
    candidates, cand_summary = load_candidates(pathlib.Path(args.full_source), used_sentence_ids, used_hashes, args.seed, args.max_candidate_len)
    budget_words = sum(int(r['companion_words']) for r in pair_rows)
    plan = {
        'status': 'MAX_BREADTH_ROWHOLDOUT_PLAN',
        'created_utc': now(),
        'scientific_purpose': 'Build the MAX-dose budget-allocation vertex: keep the 33,291 MAX FineWeb sources and replace compact rewrite words with additional independent FineWeb sentences at identical row word geometry.',
        'inputs': {'full_source': rel(pathlib.Path(args.full_source)), 'pairs': rel(pathlib.Path(args.pairs)), 'view_pool': rel(pathlib.Path(args.view_pool)), 'view_meta': rel(pathlib.Path(args.view_meta)), 'meta': rel(pathlib.Path(args.meta))},
        'budget_words_to_replace_rewrites': budget_words,
        'pair_rows': len(pair_rows),
        'topup_meta_rows': len(topup_meta),
        'suffix_rows_reused_from_view_pool': len(suffix_rows),
        'candidate_summary': cand_summary,
        'doc_cap': args.doc_cap,
        'max_parts': args.max_parts,
        'max_candidate_len': args.max_candidate_len,
        'write_training': bool(args.write_training),
        'no_generation_training_evaluation_superglue_aoa_upload_or_leaderboard': True,
    }
    (out_dir / 'max_breadth_rowholdout_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': plan['status'], 'budget_words': budget_words, 'candidate_words': cand_summary['words'], 'pair_rows': len(pair_rows), 'out_dir': rel(out_dir)}, indent=2), flush=True)
    if args.plan_only:
        return
    if cand_summary['words'] < budget_words:
        raise RuntimeError(f'candidate words {cand_summary["words"]} < needed {budget_words}')

    chosen_by_row, selection = select_by_row(pair_rows, candidates, args.doc_cap, args.max_parts, out_dir)
    selected_hashes = {str(c['norm_hash']) for row in chosen_by_row for c in row}
    if selected_hashes & used_hashes:
        raise RuntimeError('selected breadth source overlaps MAX source hashes')
    selected_docs = {str(c['doc_id']) for row in chosen_by_row for c in row}
    selection['selected_docs_overlapping_max_source_docs'] = len(selected_docs & used_docs)
    selection['selected_doc_overlap_fraction'] = len(selected_docs & used_docs) / max(1, len(selected_docs))

    breadth_prefix, row_meta, selected_sidecar = build_breadth_rows(pair_rows, chosen_by_row)
    breadth_pool = breadth_prefix + suffix_rows
    if [int(r['words']) for r in breadth_pool] != [int(r['words']) for r in view_prefix + suffix_rows]:
        raise RuntimeError('breadth row length sequence differs from MAX view')
    if word_total(breadth_pool) != TOTAL_WORDS:
        raise RuntimeError('breadth pool not exact 10M')

    pool_path = out_dir / 'compact_breadth_dose2p64x_10M.jsonl'
    meta_path = out_dir / 'compact_breadth_dose2p64x_changed_block_rows_meta.jsonl'
    sidecar_path = out_dir / 'compact_breadth_dose2p64x_selected_companion_sources.jsonl'
    write_jsonl(pool_path, breadth_pool)
    write_jsonl(meta_path, row_meta)
    write_jsonl(sidecar_path, selected_sidecar)
    training_rec: dict[str, Any] = {}
    if args.write_training:
        training_rec = write_training(out_dir / 'compact_breadth_dose2p64x_100M.jsonl', breadth_pool, args.stream_seed)

    meta = json.loads(pathlib.Path(args.meta).read_text(encoding='utf-8')) if pathlib.Path(args.meta).exists() else {}
    view_training_sha = ((meta.get('sha256') or {}).get('compact_view_dose2p64x_100M.jsonl'))
    repeat_training_sha = ((meta.get('sha256') or {}).get('compact_repeat_dose2p64x_100M.jsonl'))
    hashes = {
        pool_path.name: sha256_file(pool_path),
        meta_path.name: sha256_file(meta_path),
        sidecar_path.name: sha256_file(sidecar_path),
    }
    if training_rec:
        hashes['compact_breadth_dose2p64x_100M.jsonl'] = training_rec['sha256']

    payload = {
        'status': 'MAX_BREADTH_ROWHOLDOUT_MATERIALIZED',
        'created_utc': now(),
        'scientific_purpose': 'MAX-dose budget-allocation instrument: compare source+compact view against keeping the same MAX sources and buying additional distinct FineWeb sentences with the rewrite-word budget.',
        'inputs': plan['inputs'],
        'source_allocation': {
            'max_sources_preserved': len(pairs),
            'max_source_words_preserved': int(sum(int(p['source_words']) for p in pairs.values())),
            'compact_rewrite_words_replaced_by_breadth': budget_words,
            'selected_breadth_sentences': selection['selected_sentences'],
            'selected_breadth_words': selection['selected_words'],
            'selected_breadth_unique_docs': selection['selected_unique_docs'],
        },
        'dose': (meta.get('dose') or {}),
        'candidate_summary': cand_summary,
        'selection': selection,
        'audit': {
            'all_exact_10M': word_total(breadth_pool) == TOTAL_WORDS,
            'row_count_matches_max_view': len(breadth_pool) == len(view_prefix) + len(suffix_rows),
            'row_length_sequence_matches_max_view': [int(r['words']) for r in breadth_pool] == [int(r['words']) for r in view_prefix + suffix_rows],
            'changed_prefix_rows': len(breadth_prefix),
            'suffix_rows_reused_from_view_pool': len(suffix_rows),
            'selected_breadth_hash_overlap_with_max_sources': len(selected_hashes & used_hashes),
            'all_breadth_companions_whole_sentences': True,
            'training_stream_written': bool(training_rec),
            'training_order_seed_matches_step256_formula': args.stream_seed == DEFAULT_STREAM_SEED,
            'view_training_sha_reference': view_training_sha,
            'repeat_training_sha_reference': repeat_training_sha,
            'mechanism_instrument_not_leaderboard_submission': True,
        },
        'files': {
            'breadth_10M': rel(pool_path),
            'breadth_100M': training_rec.get('path'),
            'row_meta': rel(meta_path),
            'selected_companion_sources': rel(sidecar_path),
            'plan': rel(out_dir / 'max_breadth_rowholdout_plan.json'),
            'metadata': rel(out_dir / 'max_breadth_rowholdout_metadata.json'),
        },
        'sha256': hashes,
        'boundary': 'This arm reuses the research MAX sources, topup, common filler, fixed research tokenizer policy, and pass-order seed. It is not semantic isolation and not a leaderboard endpoint; it tests compact-view allocation against buying additional independent FineWeb sentences at fixed words and row geometry.',
        'elapsed_sec': round(time.time() - t0, 2),
    }
    meta_out = out_dir / 'max_breadth_rowholdout_metadata.json'
    meta_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = [
        '# research MAX-dose source-breadth row-holdout arm',
        '',
        'This arm preserves the 33,291 MAX FineWeb source sentences and replaces the compact rewrite words with independent whole FineWeb sentences.',
        '',
        f"Rewrite/breadth budget: {budget_words:,} words across {len(pair_rows):,} changed rows.",
        f"Selected breadth material: {selection['selected_sentences']:,} sentences / {selection['selected_words']:,} words from {selection['selected_unique_docs']:,} docs (doc cap {args.doc_cap}).",
        f"Exact 10M and row-length matched to MAX view: {payload['audit']['all_exact_10M']} / {payload['audit']['row_length_sequence_matches_max_view']}.",
        f"Hash overlap with MAX sources: {payload['audit']['selected_breadth_hash_overlap_with_max_sources']}.",
        '',
        f"Metadata: `{rel(meta_out)}`",
        f"10M pool: `{rel(pool_path)}`",
        f"100M stream: `{training_rec.get('path')}`",
    ]
    (out_dir / 'max_breadth_rowholdout_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'selected_sentences': selection['selected_sentences'], 'selected_words': selection['selected_words'], 'all_exact_10M': payload['audit']['all_exact_10M'], 'row_length_sequence_matches_max_view': payload['audit']['row_length_sequence_matches_max_view'], 'training': training_rec.get('path'), 'metadata': rel(meta_out)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
