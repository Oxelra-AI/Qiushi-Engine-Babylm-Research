#!/usr/bin/env python3
"""research: materialize a MAX-dose correspondence-broken compact companion arm.

This creates the tight control requested after the MAX dose readout: keep the
same 33,291 source sentences and the exact same multiset of accepted compact
rewrites, but reassign each source to another source's compact rewrite, excluding
same-document and same-row partners.  The changed-block row lengths, topup,
common filler, heldout split, fixed tokenizer policy, and 100M row-order formula
are inherited from research.  No training is launched by this script.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import itertools
import json
import pathlib
import random
import statistics
import time
from typing import Any, Iterable

ROOT = pathlib.Path('.').resolve()
WS = ROOT / 'experiments/archive/frontier_consolidation'
PAIRS_DEFAULT = WS / 'data/dose_distribution_select/selected_matched_max_pairs.jsonl'
VIEW_POOL_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl'
VIEW_META_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl'
META_DEFAULT = WS / 'data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json'
TOKENIZER_DIR = WS / 'data/compliant_tokenizer'
OUT_DIR_DEFAULT = WS / 'data/dose_2p64x_permuted_companion_rowholdout_pools'
TOTAL_WORDS = 10_000_000
PASSES = 10
STREAM_SEED = 82914124 + 7000
DEFAULT_ASSIGN_SEED = 82927011
LABEL = 'compact_permuted_view_dose2p64x_matched_rowholdout'
SEQ_LENGTH = 256
EXPECTED_TOKENIZER_SHA = '91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9'


@dataclasses.dataclass(frozen=True)
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    doc_id: str
    sentence_id: str
    key: str
    domain_hits: tuple[str, ...]
    origin: str


@dataclasses.dataclass(frozen=True)
class Slot:
    pair_id: str
    row_index: int
    slot_index: int
    doc_id: str
    source_words: int
    rewrite_words: int


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


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def multiset_digest(items: Iterable[str]) -> str:
    hs = sorted(hashlib.sha256(norm(x).encode('utf-8')).hexdigest() for x in items)
    h = hashlib.sha256()
    for s in hs:
        h.update(s.encode('ascii'))
        h.update(b'\n')
    return h.hexdigest()


def iter_jsonl(path: pathlib.Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


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
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        a = pos - lo
        return xs[lo] * (1 - a) + xs[hi] * a
    return {
        'n': len(xs), 'sum': float(sum(xs)), 'min': xs[0], 'p01': q(0.01),
        'p05': q(0.05), 'p10': q(0.10), 'p25': q(0.25), 'median': q(0.50),
        'p75': q(0.75), 'p90': q(0.90), 'p95': q(0.95), 'p99': q(0.99),
        'max': xs[-1], 'mean': float(statistics.mean(xs)),
    }


def load_pairs(path: pathlib.Path) -> dict[str, Pair]:
    pairs: dict[str, Pair] = {}
    for d in iter_jsonl(path):
        pid = str(d.get('pair_id') or '').strip()
        if not pid:
            raise RuntimeError('pair without pair_id')
        src = norm(d.get('source_text'))
        rew = norm(d.get('rewrite_text'))
        sw = int(d.get('source_words') or wc(src))
        rw = int(d.get('rewrite_words') or wc(rew))
        if sw != wc(src) or rw != wc(rew):
            raise RuntimeError(f'word mismatch for pair {pid}: {sw}/{wc(src)} {rw}/{wc(rew)}')
        if pid in pairs:
            raise RuntimeError(f'duplicate pair_id {pid}')
        pairs[pid] = Pair(
            pair_id=pid,
            source_text=src,
            rewrite_text=rew,
            source_words=sw,
            rewrite_words=rw,
            doc_id=str(d.get('doc_id') or ''),
            sentence_id=str(d.get('sentence_id') or ''),
            key=str(d.get('key') or ''),
            domain_hits=tuple(str(x) for x in (d.get('domain_hits') or ())),
            origin=str(d.get('origin') or d.get('regime') or ''),
        )
    if len(pairs) != 33291:
        raise RuntimeError(f'unexpected MAX pair count {len(pairs)}')
    return pairs


def parse_pair_rows(meta_path: pathlib.Path, pairs: dict[str, Pair]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    other_meta: list[dict[str, Any]] = []
    for m in iter_jsonl(meta_path):
        pids = [str(x) for x in (m.get('pair_ids') or [])]
        if not pids:
            other_meta.append(m)
            continue
        missing = [p for p in pids if p not in pairs]
        if missing:
            raise RuntimeError(f'row {m.get("row_index")} references missing pairs {missing[:5]}')
        slots: list[Slot] = []
        sw = 0
        rw = 0
        for j, pid in enumerate(pids):
            p = pairs[pid]
            sw += p.source_words
            rw += p.rewrite_words
            slots.append(Slot(pid, int(m['row_index']), j, p.doc_id, p.source_words, p.rewrite_words))
        total = int(m.get('words') or 0)
        if sw + rw != total:
            raise RuntimeError(f'row {m.get("row_index")} word mismatch {sw}+{rw}!={total}')
        pair_rows.append({
            'row_index': int(m['row_index']),
            'example_id': int(m['example_id']),
            'words': total,
            'source_words': sw,
            'companion_words': rw,
            'pair_ids': pids,
            'slots': slots,
            'row_doc_ids': sorted({pairs[p].doc_id for p in pids}),
            'component_sources': dict(m.get('component_sources') or {}),
        })
    if [r['row_index'] for r in pair_rows] != list(range(len(pair_rows))):
        raise RuntimeError('pair rows are not the initial contiguous prefix')
    return pair_rows, other_meta


def read_view_pool(path: pathlib.Path, pair_row_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    total = sum(int(r.get('words') or wc(r.get('text'))) for r in rows)
    if total != TOTAL_WORDS:
        raise RuntimeError(f'view pool word total {total} != {TOTAL_WORDS}')
    if len(rows) <= pair_row_count:
        raise RuntimeError('view pool shorter than pair row prefix')
    return rows[:pair_row_count], rows[pair_row_count:]


def row_doc_sets(pair_rows: list[dict[str, Any]]) -> dict[int, set[str]]:
    return {int(r['row_index']): set(str(x) for x in r['row_doc_ids']) for r in pair_rows}


def valid_assignment(target: Slot, donor: Slot, target_row_docs: dict[int, set[str]]) -> bool:
    if target.pair_id == donor.pair_id:
        return False
    if target.row_index == donor.row_index:
        return False
    if donor.doc_id in target_row_docs[target.row_index]:
        return False
    return True


def valid_subset_exchange(target_slots: list[Slot], donor_slots: list[Slot], target_row_docs: dict[int, set[str]]) -> bool:
    if len(target_slots) != len(donor_slots):
        return False
    if sum(s.rewrite_words for s in target_slots) != sum(s.rewrite_words for s in donor_slots):
        return False
    for t, d in zip(target_slots, donor_slots):
        # zip order is only a quick check; row-doc disjointness below makes all bijections valid.
        if t.row_index == d.row_index:
            return False
    target_docs = set().union(*(target_row_docs[s.row_index] for s in target_slots)) if target_slots else set()
    donor_docs = set().union(*(target_row_docs[s.row_index] for s in donor_slots)) if donor_slots else set()
    if target_docs & donor_docs:
        return False
    return True


def residual_bins_ok(slots: list[Slot], target_row_docs: dict[int, set[str]]) -> tuple[bool, list[dict[str, Any]]]:
    by_len: dict[int, list[Slot]] = collections.defaultdict(list)
    for s in slots:
        by_len[s.rewrite_words].append(s)
    bad: list[dict[str, Any]] = []
    for L, group in sorted(by_len.items()):
        n = len(group)
        row_counts = collections.Counter(s.row_index for s in group)
        doc_counts = collections.Counter(s.doc_id for s in group)
        reason: list[str] = []
        if n < 2:
            reason.append('fewer_than_two_slots')
        if row_counts and max(row_counts.values()) > n - max(row_counts.values()):
            reason.append('row_hall_violation_under_no_same_row')
        if doc_counts and max(doc_counts.values()) > n - max(doc_counts.values()):
            reason.append('doc_hall_risk_under_no_same_doc')
        if reason:
            bad.append({
                'rewrite_words': L,
                'n': n,
                'reason': reason,
                'top_rows': row_counts.most_common(5),
                'top_docs': doc_counts.most_common(5),
            })
    return not bad, bad


def choose_repair_exchanges(pair_rows: list[dict[str, Any]], all_slots: list[Slot], target_row_docs: dict[int, set[str]], rng: random.Random) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Assign rows containing globally singleton rewrite lengths by exact row-sum bundles.

    Most slots can be deranged within an exact rewrite-length bin.  Three MAX
    rewrites have globally unique lengths (39, 40, 42), so those rows cannot be
    handled by length bins.  For this tiny subset we solve a row-bundle
    derangement: every selected target row receives the same number of donor
    rewrites with exactly the same total companion words, with no donor from the
    same row or from any source document represented in that target row.  This
    preserves row geometry and the compact-text multiset while breaking
    correspondence.
    """
    length_counts = collections.Counter(s.rewrite_words for s in all_slots)
    singleton_lengths = {L for L, c in length_counts.items() if c == 1}
    exception_rows = [r for r in pair_rows if any(s.rewrite_words in singleton_lengths for s in r['slots'])]
    assignments: dict[str, str] = {}
    assignment_kind: dict[str, str] = {}
    exchanges: list[dict[str, Any]] = []
    if not exception_rows:
        return assignments, assignment_kind, exchanges

    exception_row_ids = {int(r['row_index']) for r in exception_rows}

    def valid_row_donors(row: dict[str, Any], donor_slots: tuple[Slot, ...]) -> bool:
        if len(donor_slots) != len(row['slots']):
            return False
        if sum(s.rewrite_words for s in donor_slots) != int(row['companion_words']):
            return False
        docs = target_row_docs[int(row['row_index'])]
        for s in donor_slots:
            if s.row_index == int(row['row_index']):
                return False
            if s.doc_id in docs:
                return False
        return True

    def candidate_bundles(row: dict[str, Any], donor_slots: list[Slot]) -> list[tuple[Slot, ...]]:
        out: list[tuple[Slot, ...]] = []
        k = len(row['slots'])
        need = int(row['companion_words'])
        for comb in itertools.combinations(donor_slots, k):
            if sum(s.rewrite_words for s in comb) != need:
                continue
            if valid_row_donors(row, comb):
                out.append(tuple(comb))
        return out

    def solve_subrows(subrows: list[dict[str, Any]], seed: int, limit: int = 1_000_000) -> tuple[dict[int, tuple[Slot, ...]] | None, dict[str, Any]]:
        local_rng = random.Random(seed)
        donor_slots = [s for r in subrows for s in r['slots']]
        cand = {int(r['row_index']): candidate_bundles(r, donor_slots) for r in subrows}
        if any(not v for v in cand.values()):
            return None, {'candidate_counts': {str(k): len(v) for k, v in cand.items()}, 'empty_candidate_rows': [k for k, v in cand.items() if not v]}
        order = sorted(subrows, key=lambda r: (len(cand[int(r['row_index'])]), len(r['slots']), int(r['companion_words']), int(r['row_index'])))
        used: set[str] = set()
        out: dict[int, tuple[Slot, ...]] = {}
        nodes = 0

        def rec(i: int) -> bool:
            nonlocal nodes
            nodes += 1
            if nodes > limit:
                return False
            if i == len(order):
                return True
            row = order[i]
            rid = int(row['row_index'])
            cs = [c for c in cand[rid] if not any(s.pair_id in used for s in c)]
            local_rng.shuffle(cs)
            cs.sort(key=lambda c: (len({s.row_index for s in c}), sum(abs(s.row_index - rid) for s in c), sum(abs(s.rewrite_words - t.rewrite_words) for s, t in zip(c, row['slots']))))
            for comb in cs:
                for s in comb:
                    used.add(s.pair_id)
                out[rid] = comb
                if rec(i + 1):
                    return True
                for s in comb:
                    used.remove(s.pair_id)
                out.pop(rid, None)
            return False

        ok = rec(0)
        diag = {
            'candidate_counts': {str(k): len(v) for k, v in cand.items()},
            'search_order': [int(r['row_index']) for r in order],
            'nodes': nodes,
            'limit': limit,
            'selected_rows': [int(r['row_index']) for r in subrows],
        }
        if not ok:
            return None, diag
        if len(used) != len(donor_slots):
            return None, {**diag, 'error': 'solution did not consume every selected donor slot'}
        return out, diag

    # A deterministic seed set is tried first because search_all_exception_repairs.py
    # found it as the smallest exact bundle repair; the randomized loop below keeps the
    # script robust if row order or inputs are regenerated in the same coordinate.
    by_row = {int(r['row_index']): r for r in pair_rows}
    fixed_helper_rows = [6007, 2231, 3501, 4382, 4496]
    helper_pool = []
    for r in pair_rows:
        if int(r['row_index']) in exception_row_ids:
            continue
        k = len(r['slots'])
        cw = int(r['companion_words'])
        lens = [s.rewrite_words for s in r['slots']]
        if k in {2, 3, 4} and 25 <= cw <= 95 and any(8 <= x <= 45 for x in lens):
            helper_pool.append(r)
    helper_pool = sorted(helper_pool, key=lambda r: (abs(int(r['companion_words']) - 60), len(r['slots']), int(r['row_index'])))

    attempts: list[list[dict[str, Any]]] = []
    if all(x in by_row for x in fixed_helper_rows):
        attempts.append([by_row[x] for x in fixed_helper_rows])
    for m in range(3, 11):
        attempts.append(helper_pool[:m])
        for _ in range(800):
            attempts.append(rng.sample(helper_pool, m))
    seen: set[tuple[int, ...]] = set()
    solution = None
    solution_diag: dict[str, Any] = {}
    solution_helpers: list[dict[str, Any]] = []
    for helpers in attempts:
        key = tuple(sorted(int(r['row_index']) for r in helpers))
        if key in seen:
            continue
        seen.add(key)
        subrows = list(exception_rows) + list(helpers)
        sol, diag = solve_subrows(subrows, seed=10_000 + len(seen))
        if sol is not None:
            solution = sol
            solution_diag = diag
            solution_helpers = list(helpers)
            break
    if solution is None:
        raise RuntimeError(f'no exact row-sum repair found for singleton-length exception rows; last_diag={solution_diag}')

    def min_abs_pairing(target_slots: list[Slot], donor_slots: tuple[Slot, ...]) -> list[tuple[Slot, Slot]]:
        best: tuple[int, tuple[int, ...]] | None = None
        for perm in itertools.permutations(range(len(donor_slots))):
            score = sum(abs(target_slots[i].rewrite_words - donor_slots[perm[i]].rewrite_words) for i in range(len(target_slots)))
            if best is None or (score, perm) < best:
                best = (score, perm)
        assert best is not None
        return [(target_slots[i], donor_slots[best[1][i]]) for i in range(len(target_slots))]

    selected_rows = list(exception_rows) + solution_helpers
    for row in selected_rows:
        rid = int(row['row_index'])
        target_slots = sorted(list(row['slots']), key=lambda s: (s.slot_index, s.pair_id))
        donor_slots = solution[rid]
        for t, d in min_abs_pairing(target_slots, donor_slots):
            if not valid_assignment(t, d, target_row_docs):
                raise RuntimeError(f'invalid bundle repair assignment target={t} donor={d}')
            assignments[t.pair_id] = d.pair_id
            assignment_kind[t.pair_id] = 'row_sum_exception_bundle_derangement'
    if set(assignments) != set(assignments.values()):
        raise RuntimeError('exception repair is not a permutation over selected slots')
    exchanges.append({
        'singleton_rewrite_lengths': sorted(singleton_lengths),
        'exception_row_indices': [int(r['row_index']) for r in exception_rows],
        'helper_row_indices': [int(r['row_index']) for r in solution_helpers],
        'selected_row_indices': [int(r['row_index']) for r in selected_rows],
        'selected_row_sums': {str(int(r['row_index'])): int(r['companion_words']) for r in selected_rows},
        'selected_row_slot_counts': {str(int(r['row_index'])): len(r['slots']) for r in selected_rows},
        'selected_row_rewrite_lengths': {str(int(r['row_index'])): [s.rewrite_words for s in r['slots']] for r in selected_rows},
        'candidate_counts': solution_diag.get('candidate_counts'),
        'search_order': solution_diag.get('search_order'),
        'search_nodes': solution_diag.get('nodes'),
        'per_row_donor_pair_ids': {str(rid): [s.pair_id for s in slots] for rid, slots in solution.items()},
        'per_row_donor_rewrite_lengths': {str(rid): [s.rewrite_words for s in slots] for rid, slots in solution.items()},
        'row_companion_words_preserved': True,
        'donor_docs_excluded_from_target_row_docset': True,
    })
    return assignments, assignment_kind, exchanges


def brute_match(slots: list[Slot], target_row_docs: dict[int, set[str]]) -> dict[str, str] | None:
    idx = list(range(len(slots)))
    # Try constrained backtracking rather than all permutations in arbitrary order.
    targets = sorted(slots, key=lambda s: sum(1 for d in slots if valid_assignment(s, d, target_row_docs)))
    donors = list(slots)
    out: dict[str, str] = {}
    used: set[str] = set()
    def rec(i: int) -> bool:
        if i == len(targets):
            return True
        t = targets[i]
        cands = [d for d in donors if d.pair_id not in used and valid_assignment(t, d, target_row_docs)]
        cands.sort(key=lambda d: (collections.Counter(x.doc_id for x in donors)[d.doc_id], d.row_index, d.pair_id))
        for d in cands:
            used.add(d.pair_id)
            out[t.pair_id] = d.pair_id
            if rec(i + 1):
                return True
            used.remove(d.pair_id)
            out.pop(t.pair_id, None)
        return False
    return out if rec(0) else None


def random_cycle_match(slots: list[Slot], target_row_docs: dict[int, set[str]], rng: random.Random, max_attempts: int = 6000) -> dict[str, str] | None:
    n = len(slots)
    if n <= 12:
        return brute_match(slots, target_row_docs)
    seq = list(slots)
    # For large bins a random Hamiltonian cycle nearly always avoids the sparse forbidden row/doc arcs.
    for attempt in range(max_attempts):
        rng.shuffle(seq)
        if n <= 80:
            shifts = list(range(1, n))
            rng.shuffle(shifts)
        else:
            shifts = [1]
            shifts.extend(rng.randrange(1, n) for _ in range(128))
        for shift in shifts:
            ok = True
            for i, t in enumerate(seq):
                if not valid_assignment(t, seq[(i + shift) % n], target_row_docs):
                    ok = False
                    break
            if ok:
                return {seq[i].pair_id: seq[(i + shift) % n].pair_id for i in range(n)}
    return None


def assign_remaining_by_length(pair_rows: list[dict[str, Any]], all_slots: list[Slot], existing: dict[str, str], existing_kind: dict[str, str], target_row_docs: dict[int, set[str]], rng: random.Random) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    assigned_targets = set(existing)
    assigned_donors = set(existing.values())
    if assigned_targets != assigned_donors:
        # Repair exchanges are reciprocal and should use the same set as targets and donors.
        raise RuntimeError('repair assignment target/donor sets differ')
    remaining = [s for s in all_slots if s.pair_id not in assigned_targets]
    ok, bad = residual_bins_ok(remaining, target_row_docs)
    if not ok:
        raise RuntimeError(f'residual length bins cannot be deranged: {bad}')
    by_len: dict[int, list[Slot]] = collections.defaultdict(list)
    for s in remaining:
        by_len[s.rewrite_words].append(s)
    diagnostics: dict[str, Any] = {'length_bins': {}, 'random_seed': DEFAULT_ASSIGN_SEED}
    assignments = dict(existing)
    kinds = dict(existing_kind)
    for L, group in sorted(by_len.items()):
        match = random_cycle_match(group, target_row_docs, rng)
        if match is None:
            raise RuntimeError(f'failed to find derangement for rewrite length {L} with n={len(group)}')
        for target_pid, donor_pid in match.items():
            assignments[target_pid] = donor_pid
            kinds[target_pid] = 'exact_rewrite_length_derangement'
        row_counts = collections.Counter(s.row_index for s in group)
        doc_counts = collections.Counter(s.doc_id for s in group)
        diagnostics['length_bins'][str(L)] = {
            'n': len(group),
            'top_row_count': row_counts.most_common(1)[0][1],
            'top_doc_count': doc_counts.most_common(1)[0][1],
        }
    return assignments, kinds, diagnostics


def build_permuted_rows(pair_rows: list[dict[str, Any]], pairs: dict[str, Pair], assignment: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for r in pair_rows:
        segs: list[str] = []
        row_donor_words = 0
        source_words = 0
        for slot in r['slots']:
            src = pairs[slot.pair_id]
            donor = pairs[assignment[slot.pair_id]]
            segs.append(src.source_text)
            segs.append(donor.rewrite_text)
            source_words += src.source_words
            row_donor_words += donor.rewrite_words
            assignment_rows.append({
                'row_index': int(r['row_index']),
                'slot_index': int(slot.slot_index),
                'target_pair_id': slot.pair_id,
                'target_doc_id': src.doc_id,
                'target_sentence_id': src.sentence_id,
                'target_source_words': src.source_words,
                'original_rewrite_words': src.rewrite_words,
                'donor_pair_id': donor.pair_id,
                'donor_doc_id': donor.doc_id,
                'donor_sentence_id': donor.sentence_id,
                'donor_rewrite_words': donor.rewrite_words,
                'same_doc': donor.doc_id == src.doc_id,
                'same_pair': donor.pair_id == src.pair_id,
                'same_row': False,  # filled in audit from donor slot lookup
            })
        text = norm(' '.join(segs))
        words = wc(text)
        if row_donor_words != int(r['companion_words']):
            raise RuntimeError(f'row {r["row_index"]} companion sum changed {row_donor_words} != {r["companion_words"]}')
        if source_words != int(r['source_words']) or words != int(r['words']):
            raise RuntimeError(f'row {r["row_index"]} total changed: source={source_words}, words={words}, expected={r["words"]}')
        rows.append({'text': text, 'words': words, 'example_id': int(r['example_id']), 'source': LABEL})
    return rows, assignment_rows


def write_training(path: pathlib.Path, rows: list[dict[str, Any]], seed: int = STREAM_SEED) -> None:
    total = 0
    n = len(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                total += int(r['words'])
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f'training word total {total} != {TOTAL_WORDS * PASSES}')


class TokenGeometry:
    def __init__(self, tokenizer, seq_length: int = SEQ_LENGTH) -> None:
        self.tok = tokenizer
        self.seq = seq_length
        self.special = set(tokenizer.all_special_ids)
        self._ws: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        val = self._ws.get(int(tid))
        if val is None:
            s = self.tok.convert_ids_to_tokens(int(tid))
            val = bool(s is not None and (str(s).startswith('\u0120') or str(s).startswith('\u2581')))
            self._ws[int(tid)] = val
        return val

    def pool(self, path: pathlib.Path, batch: int = 512) -> dict[str, Any]:
        rows = words = tokens_full = tokens_visible = groups = truncated_rows = tokens_dropped = 0
        buf: list[str] = []
        def flush(texts: list[str]) -> None:
            nonlocal tokens_full, tokens_visible, groups, truncated_rows, tokens_dropped
            if not texts:
                return
            enc = self.tok(texts, add_special_tokens=False)['input_ids']
            for ids in enc:
                tokens_full += len(ids)
                vis = ids[: self.seq]
                tokens_visible += len(vis)
                if len(ids) > self.seq:
                    truncated_rows += 1
                    tokens_dropped += len(ids) - self.seq
                gid = -1
                for i, tid in enumerate(vis):
                    if tid in self.special:
                        continue
                    if gid < 0 or self.word_start(int(tid)) or i == 0:
                        gid += 1
                        groups += 1
        for obj in iter_jsonl(path):
            text = str(obj.get('text') or '')
            rows += 1
            words += wc(text)
            buf.append(text)
            if len(buf) >= batch:
                flush(buf)
                buf = []
        flush(buf)
        return {
            'path': rel(path), 'rows': rows, 'words': words,
            'tokens_legal16k': tokens_full, 'tokens_visible_seq256': tokens_visible,
            'wwm_groups_visible': groups, 'truncated_rows': truncated_rows,
            'tokens_dropped_by_truncation': tokens_dropped,
            'tokens_per_word': tokens_full / words if words else None,
            'wwm_groups_per_word': groups / words if words else None,
        }


def token_audit(paths: dict[str, pathlib.Path]) -> dict[str, Any]:
    from transformers import AutoTokenizer
    tok_sha = sha256_file(TOKENIZER_DIR / 'tokenizer.json')
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f'tokenizer sha mismatch {tok_sha}')
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR), use_fast=True)
    geo = TokenGeometry(tok)
    out: dict[str, Any] = {'tokenizer_dir': rel(TOKENIZER_DIR), 'tokenizer_json_sha256': tok_sha, 'seq_length': SEQ_LENGTH, 'pools': {}}
    for name, path in paths.items():
        t0 = time.time()
        out['pools'][name] = geo.pool(path)
        out['pools'][name]['elapsed_sec'] = round(time.time() - t0, 1)
    ref = out['pools'].get('max_view') or {}
    shifts: dict[str, Any] = {}
    for name, rec in out['pools'].items():
        if name == 'max_view' or not rec.get('rows'):
            continue
        row: dict[str, Any] = {}
        for key in ['tokens_legal16k', 'tokens_visible_seq256', 'wwm_groups_visible', 'words']:
            a = ref.get(key); b = rec.get(key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a:
                row[f'{key}_minus_max_view'] = b - a
                row[f'{key}_pct_vs_max_view'] = 100.0 * (b - a) / a
        shifts[name] = row
    out['relative_shift_vs_max_view'] = shifts
    return out


def count_jsonl(path: pathlib.Path) -> dict[str, Any]:
    rows = words = 0
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows += 1
            words += int(obj.get('words') or wc(obj.get('text')))
    return {'rows': rows, 'words': words, 'exact_10M': words == TOTAL_WORDS, 'exact_100M': words == TOTAL_WORDS * PASSES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default=str(PAIRS_DEFAULT))
    ap.add_argument('--view-pool', default=str(VIEW_POOL_DEFAULT))
    ap.add_argument('--view-meta', default=str(VIEW_META_DEFAULT))
    ap.add_argument('--research-meta', default=str(META_DEFAULT))
    ap.add_argument('--out-dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--assignment-seed', type=int, default=DEFAULT_ASSIGN_SEED)
    ap.add_argument('--write-training', action='store_true')
    ap.add_argument('--token-audit', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(pathlib.Path(args.pairs))
    pair_rows, other_meta = parse_pair_rows(pathlib.Path(args.view_meta), pairs)
    view_prefix, suffix = read_view_pool(pathlib.Path(args.view_pool), len(pair_rows))
    meta = json.loads(pathlib.Path(args.meta).read_text(encoding='utf-8'))
    all_slots: list[Slot] = [s for r in pair_rows for s in r['slots']]
    slot_lookup = {s.pair_id: s for s in all_slots}
    target_row_docs = row_doc_sets(pair_rows)
    rng = random.Random(args.assignment_seed)

    repair_assign, repair_kind, repair_exchanges = choose_repair_exchanges(pair_rows, all_slots, target_row_docs, rng)
    assignment, assignment_kind, length_diag = assign_remaining_by_length(pair_rows, all_slots, repair_assign, repair_kind, target_row_docs, rng)
    if set(assignment) != {s.pair_id for s in all_slots} or set(assignment.values()) != {s.pair_id for s in all_slots}:
        raise RuntimeError('assignment is not a permutation over all pair ids')

    perm_prefix, assignment_rows = build_permuted_rows(pair_rows, pairs, assignment)
    for ar in assignment_rows:
        ar['same_row'] = slot_lookup[ar['donor_pair_id']].row_index == int(ar['row_index'])
        ar['assignment_kind'] = assignment_kind[ar['target_pair_id']]
    pool_rows = perm_prefix + suffix
    row_words = [int(r['words']) for r in pool_rows]
    view_rows_all = view_prefix + suffix
    view_row_words = [int(r['words']) for r in view_rows_all]
    if row_words != view_row_words:
        raise RuntimeError('permuted pool row-length sequence differs from MAX view')
    if sum(row_words) != TOTAL_WORDS:
        raise RuntimeError(f'pool words {sum(row_words)} != {TOTAL_WORDS}')

    original_rewrites = [pairs[s.pair_id].rewrite_text for s in all_slots]
    assigned_rewrites = [pairs[assignment[s.pair_id]].rewrite_text for s in all_slots]
    kind_counts = collections.Counter(assignment_kind.values())
    length_preserved = sum(1 for s in all_slots if pairs[assignment[s.pair_id]].rewrite_words == s.rewrite_words)
    row_changed_stats = stats([sum(1 for s in r['slots'] if pairs[assignment[s.pair_id]].rewrite_words != s.rewrite_words) for r in pair_rows])
    donor_row_distance = [abs(slot_lookup[assignment[s.pair_id]].row_index - s.row_index) for s in all_slots]
    same_doc = [s for s in all_slots if pairs[assignment[s.pair_id]].doc_id == s.doc_id]
    same_row = [s for s in all_slots if slot_lookup[assignment[s.pair_id]].row_index == s.row_index]
    donor_doc_in_target_row = [s for s in all_slots if pairs[assignment[s.pair_id]].doc_id in target_row_docs[s.row_index]]
    row_companion_residuals = []
    for r in pair_rows:
        orig = int(r['companion_words'])
        new = sum(pairs[assignment[s.pair_id]].rewrite_words for s in r['slots'])
        row_companion_residuals.append(new - orig)
    source_text_digest = multiset_digest([pairs[s.pair_id].source_text for s in all_slots])
    original_rewrite_digest = multiset_digest(original_rewrites)
    assigned_rewrite_digest = multiset_digest(assigned_rewrites)

    pool_path = out_dir / 'compact_permuted_view_dose2p64x_10M.jsonl'
    train_path = out_dir / 'compact_permuted_view_dose2p64x_100M.jsonl'
    row_meta_path = out_dir / 'compact_permuted_view_dose2p64x_changed_block_rows_meta.jsonl'
    assignment_path = out_dir / 'compact_permuted_view_dose2p64x_companion_assignment.jsonl'
    meta_path = out_dir / 'permuted_companion_rowholdout_metadata.json'
    summary_path = out_dir / 'permuted_companion_rowholdout_summary.md'

    row_meta_rows = []
    for r in pair_rows:
        row_meta_rows.append({
            'row_index': int(r['row_index']),
            'example_id': int(r['example_id']),
            'words': int(r['words']),
            'pair_ids': [str(x) for x in r['pair_ids']],
            'source_words': int(r['source_words']),
            'companion_words': int(r['companion_words']),
            'target_row_doc_ids': list(r['row_doc_ids']),
            'component_sources': dict(r['component_sources']),
            'donor_pair_ids': [assignment[s.pair_id] for s in r['slots']],
            'donor_rewrite_words': [pairs[assignment[s.pair_id]].rewrite_words for s in r['slots']],
            'assignment_kinds': [assignment_kind[s.pair_id] for s in r['slots']],
        })

    if not args.plan_only:
        write_jsonl(pool_path, pool_rows)
        write_jsonl(row_meta_path, row_meta_rows)
        write_jsonl(assignment_path, assignment_rows)
        if args.write_training:
            write_training(train_path, pool_rows, STREAM_SEED)

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for key, path in [('pool_10M', pool_path), ('row_meta', row_meta_path), ('assignment', assignment_path)]:
        if path.exists() and not args.plan_only:
            files[key] = rel(path); hashes[path.name] = sha256_file(path)
    if args.write_training and train_path.exists() and not args.plan_only:
        files['training_100M'] = rel(train_path); hashes[train_path.name] = sha256_file(train_path)

    counts = {}
    if not args.plan_only and pool_path.exists():
        counts['pool_10M'] = count_jsonl(pool_path)
    if args.write_training and train_path.exists():
        counts['training_100M'] = count_jsonl(train_path)

    tok_audit: dict[str, Any] | None = None
    if args.token_audit and not args.plan_only:
        paths = {'max_view': pathlib.Path(args.view_pool), 'permuted': pool_path}
        # Include repeat if present to keep the future V-P-R triangle interpretable in one object.
        rep = pathlib.Path(args.view_pool).with_name('compact_repeat_dose2p64x_10M.jsonl')
        if rep.exists():
            paths['max_repeat'] = rep
        tok_audit = token_audit(paths)
        (out_dir / 'permuted_companion_token_geometry.json').write_text(json.dumps(tok_audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        hashes['permuted_companion_token_geometry.json'] = sha256_file(out_dir / 'permuted_companion_token_geometry.json')
        files['token_geometry'] = rel(out_dir / 'permuted_companion_token_geometry.json')

    audit = {
        'pair_count': len(all_slots),
        'pair_row_count': len(pair_rows),
        'suffix_rows_reused_from_max_view': len(suffix),
        'row_count_matches_max_view': len(pool_rows) == len(view_rows_all),
        'word_total_exact_10M': sum(row_words) == TOTAL_WORDS,
        'row_length_sequence_matches_max_view': row_words == view_row_words,
        'suffix_exactly_reused_from_max_view_after_changed_block': suffix == view_rows_all[len(pair_rows):],
        'assignment_is_permutation': set(assignment) == {s.pair_id for s in all_slots} and set(assignment.values()) == {s.pair_id for s in all_slots},
        'companion_text_multiset_identical_to_max_view': original_rewrite_digest == assigned_rewrite_digest,
        'source_text_multiset_identical_to_max_view': source_text_digest == multiset_digest([pairs[s.pair_id].source_text for s in all_slots]),
        'all_sources_receive_different_pair_rewrite': not any(assignment[s.pair_id] == s.pair_id for s in all_slots),
        'same_doc_assignment_count': len(same_doc),
        'same_row_assignment_count': len(same_row),
        'donor_doc_in_target_row_docset_count': len(donor_doc_in_target_row),
        'all_row_companion_word_sums_preserved': all(x == 0 for x in row_companion_residuals),
        'row_companion_residual_minmax': [min(row_companion_residuals), max(row_companion_residuals)],
        'per_slot_rewrite_length_preserved_count': length_preserved,
        'per_slot_rewrite_length_changed_count': len(all_slots) - length_preserved,
        'assignment_kind_counts': dict(kind_counts.most_common()),
        'repair_exchanges': repair_exchanges,
        'donor_row_distance_stats': stats(donor_row_distance),
        'changed_slots_per_row_stats': row_changed_stats,
        'training_order_seed_matches_step256_formula': bool(args.write_training),
        'no_training_launched_by_this_script': True,
        'mechanism_instrument_not_leaderboard_submission': True,
    }
    if audit['same_doc_assignment_count'] or audit['same_row_assignment_count'] or audit['donor_doc_in_target_row_docset_count']:
        raise RuntimeError(f'correspondence-breaking audit failed: {audit}')

    payload = {
        'status': 'PERMUTED_COMPANION_MAX_ROWHOLDOUT_MATERIALIZED' if not args.plan_only else 'PERMUTED_COMPANION_MAX_PLAN_OK',
        'created_utc': now(),
        'scientific_purpose': 'Correspondence-breaking MAX-dose control: hold source multiset, compact-rewrite multiset, changed-block row lengths, common filler, heldout split, tokenizer policy, and 100M order fixed while destroying source-to-own-view correspondence within the same window.',
        'expensive_work_policy': 'CPU materialization/audit only now. DeBERTa training should launch only if the already-running second-basin MAX view/repeat readout reproduces the Entity carrier strongly enough to make aligned-versus-permuted decisive.',
        'inputs': {
            'pairs': rel(pathlib.Path(args.pairs)),
            'pairs_sha256': sha256_file(pathlib.Path(args.pairs)),
            'view_pool': rel(pathlib.Path(args.view_pool)),
            'view_pool_sha256': sha256_file(pathlib.Path(args.view_pool)),
            'view_meta': rel(pathlib.Path(args.view_meta)),
            'view_meta_sha256': sha256_file(pathlib.Path(args.view_meta)),
            'metadata': rel(pathlib.Path(args.meta)),
            'status': meta.get('status'),
            'view_training_sha256': (meta.get('sha256') or {}).get('compact_view_dose2p64x_100M.jsonl'),
            'repeat_training_sha256': (meta.get('sha256') or {}).get('compact_repeat_dose2p64x_100M.jsonl'),
        },
        'assignment_seed': args.assignment_seed,
        'stream_seed_for_100M_order': STREAM_SEED,
        'constraints': {
            'one_donor_rewrite_per_source': True,
            'donor_rewrite_multiset_identical_to_max_view': True,
            'donor_pair_never_self': True,
            'donor_row_never_target_row': True,
            'donor_doc_never_in_target_row_source_docs': True,
            'per_changed_row_total_words_identical_to_max_view': True,
            'suffix_topup_and_common_filler_exactly_reused_from_max_view': True,
            'fixed_step35_tokenizer_policy': True,
        },
        'audit': audit,
        'length_derangement_diagnostics': length_diag,
        'token_geometry': tok_audit,
        'files': files,
        'sha256': hashes,
        'counts': counts,
        'boundary': 'This is not a new dose, not new generated data, not a legal leaderboard candidate, and not a training result. It isolates source-view correspondence from compact-text multiset and row-budget effects once trained under the predeclared condition.',
        'elapsed_sec': round(time.time() - t0, 2),
    }
    if not args.plan_only:
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        hashes[meta_path.name] = sha256_file(meta_path)
        files['metadata'] = rel(meta_path)
        lines = [
            '# research MAX permuted-compact correspondence control',
            '',
            'This CPU materialization keeps the MAX source rows and the exact compact-rewrite multiset, but reassigns every source to another source\'s compact view with no same-row or same-document donor.',
            '',
            '## Core audit',
            '',
            f"- pair slots: {audit['pair_count']:,}; changed rows: {audit['pair_row_count']:,}; suffix rows reused from MAX view: {audit['suffix_rows_reused_from_max_view']:,}",
            f"- exact 10M words: {audit['word_total_exact_10M']}; row-length sequence matches MAX view: {audit['row_length_sequence_matches_max_view']}",
            f"- companion text multiset identical to MAX view: {audit['companion_text_multiset_identical_to_max_view']}; assignment is a permutation: {audit['assignment_is_permutation']}",
            f"- same-pair assignments: {0 if audit['all_sources_receive_different_pair_rewrite'] else 'nonzero'}; same-doc assignments: {audit['same_doc_assignment_count']}; same-row assignments: {audit['same_row_assignment_count']}; donor doc in target row docset: {audit['donor_doc_in_target_row_docset_count']}",
            f"- per-slot rewrite length changed only for {audit['per_slot_rewrite_length_changed_count']} / {audit['pair_count']} slots; row companion sums preserved: {audit['all_row_companion_word_sums_preserved']}",
            f"- 100M training stream written: {bool(args.write_training)}; stream seed matches research formula: {audit['training_order_seed_matches_step256_formula']}",
            '',
        ]
        if tok_audit:
            shift = (tok_audit.get('relative_shift_vs_max_view') or {}).get('permuted', {})
            lines += [
                '## Token/WWM geometry versus MAX view',
                '',
                f"- legal16k token delta: {shift.get('tokens_legal16k_minus_max_view')} ({shift.get('tokens_legal16k_pct_vs_max_view'):+.6f}%)",
                f"- visible seq256 token delta: {shift.get('tokens_visible_seq256_minus_max_view')} ({shift.get('tokens_visible_seq256_pct_vs_max_view'):+.6f}%)",
                f"- WWM group delta: {shift.get('wwm_groups_visible_minus_max_view')} ({shift.get('wwm_groups_visible_pct_vs_max_view'):+.6f}%)",
                '',
            ]
        lines += [
            '## Expensive-work condition',
            '',
            'Do not launch this arm merely because the file exists. Train it only if the already-running second-basin MAX view/repeat scoring reproduces the Entity carrier strongly enough that aligned-versus-permuted can decide whether record addressability, rather than compact-style filler, is the mechanism.',
            '',
            f"Metadata: `{rel(meta_path)}`",
        ]
        summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        hashes[summary_path.name] = sha256_file(summary_path)
        files['summary'] = rel(summary_path)
        # Re-write metadata once to include hashes of itself and summary path in the returned files block is unnecessary;
        # hashes above are printed separately and the existing JSON is the authoritative scientific metadata.
    print(json.dumps({
        'status': payload['status'],
        'out_dir': rel(out_dir),
        'audit': audit,
        'files': files,
        'sha256': hashes,
        'counts': counts,
        'token_shift_vs_max_view': None if not tok_audit else (tok_audit.get('relative_shift_vs_max_view') or {}).get('permuted'),
        'elapsed_sec': payload['elapsed_sec'],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
