#!/usr/bin/env python3
"""research: CPU construction of a source-view adjacency-broken reinvest control.

This prepares the next mechanism comparison without starting a new GPU run.  The
control keeps the compact_view_reinvest changed-block source texts and compact
rewrite texts as exact multisets, keeps the same 10M pool size and row word
length sequence, but assigns each source a compact rewrite from another source
inside the closest available relation/domain and rewrite-word stratum.  It is
intended for use only if the running full/seed endpoint results justify a new
mechanism run.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import random
from typing import Any, Iterable

ROOT = pathlib.Path(".")
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
DENSITY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard"
TOKENIZER = ROOT / "experiments/archive" / 'initial_model_studies' / "training" / "runs" / "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256" / "hf_model"
OUT_DIR = A01 / "data" / "compact_reinvest_adjbreak_control"
OUT_POOL = OUT_DIR / "cleanqwen_fineweb_compact_view_reinvest_adjbreak_10M.jsonl"
OUT_META = OUT_DIR / "cleanqwen_fineweb_compact_view_reinvest_adjbreak_changed_block_rows_meta.jsonl"
OUT_JSON = OUT_DIR / "compact_reinvest_adjbreak_control_measurement.json"
OUT_TRAINING = OUT_DIR / "cleanqwen_fineweb_compact_view_reinvest_adjbreak_100M.jsonl"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_reinvest_adjbreak_control.md')
SEQ_LEN = 256
TOTAL_WORDS = 10_000_000
PASSES = 10
CONTROL_SOURCE = "cleanqwen_fineweb_compact_view_reinvest_adjbreak"
# Match the row-holdout materializer: default seed 82914124 is passed to write_training as seed+7000,
# and the writer then uses seed+1000+pass_i.  Keeping this final base reproduces the same per-pass row-index orders.
TRAIN_SHUFFLE_SEED = 82914124 + 7000 + 1000


@dataclasses.dataclass(frozen=True)
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    doc_id: str
    domain_hits: tuple[str, ...]
    content_recall: float | None
    entity_recall: float | None
    number_recall: float | None
    source_tokens: int = 0
    rewrite_tokens: int = 0


@dataclasses.dataclass(frozen=True)
class Slot:
    index: int
    pair_id: str
    row_index: int
    position_in_row: int
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    source_tokens: int
    rewrite_tokens: int
    doc_id: str
    domain_hits: tuple[str, ...]
    primary_domain: str
    domain_signature: str


def wc(text: str) -> int:
    return len((text or "").split())


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def iter_jsonl(path: pathlib.Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def multiset_hash(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for t in sorted(norm_text(x) for x in texts):
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(v) for v in vals)

    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

    return {
        "n": len(xs),
        "min": xs[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "mean": sum(xs) / len(xs),
        "median": q(0.5),
        "p75": q(0.75),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": xs[-1],
        "sum": sum(xs),
    }


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int:
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def primary_domain(domain_hits: tuple[str, ...]) -> str:
    if not domain_hits:
        return "no_domain"
    # Deterministic broad relation/topic stratum; pair construction already stores only a small set.
    return sorted(domain_hits)[0]


def load_pairs(tok: Any) -> dict[str, Pair]:
    out: dict[str, Pair] = {}
    for obj in iter_jsonl(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"):
        pid = str(obj["pair_id"])
        source = norm_text(str(obj["source_text"]))
        rewrite = norm_text(str(obj["rewrite_text"]))
        sw = int(obj.get("source_words") or wc(source))
        rw = int(obj.get("rewrite_words") or wc(rewrite))
        pw = int(obj.get("pair_words") or (sw + rw))
        if sw != wc(source) or rw != wc(rewrite) or pw != sw + rw:
            raise RuntimeError(f"word mismatch for {pid}")
        domains = tuple(str(x) for x in (obj.get("domain_hits") or []))
        out[pid] = Pair(
            pair_id=pid,
            source_text=source,
            rewrite_text=rewrite,
            source_words=sw,
            rewrite_words=rw,
            pair_words=pw,
            doc_id=str(obj.get("doc_id") or ""),
            domain_hits=domains,
            content_recall=float(obj["content_recall"]) if obj.get("content_recall") is not None else None,
            entity_recall=float(obj["entity_recall"]) if obj.get("entity_recall") is not None else None,
            number_recall=float(obj["number_recall"]) if obj.get("number_recall") is not None else None,
            source_tokens=enc_len(tok, source, False),
            rewrite_tokens=enc_len(tok, rewrite, False),
        )
    return out


def load_slots(pairs: dict[str, Pair], changed_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Slot]]:
    rows = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl", limit=changed_rows))
    metas = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl", limit=changed_rows))
    if len(rows) != changed_rows or len(metas) != changed_rows:
        raise RuntimeError("could not read changed rows and metadata")
    slots: list[Slot] = []
    for ridx, (row, meta) in enumerate(zip(rows, metas)):
        if ridx != int(meta.get("row_index", ridx)):
            raise RuntimeError(f"row metadata index mismatch at {ridx}")
        pids = [str(x) for x in (meta.get("pair_ids") or [])]
        for pos, pid in enumerate(pids):
            p = pairs[pid]
            domains = p.domain_hits
            prim = primary_domain(domains)
            signature = "+".join(sorted(domains)) if domains else "no_domain"
            slots.append(Slot(
                index=len(slots),
                pair_id=pid,
                row_index=ridx,
                position_in_row=pos,
                source_text=p.source_text,
                rewrite_text=p.rewrite_text,
                source_words=p.source_words,
                rewrite_words=p.rewrite_words,
                source_tokens=p.source_tokens,
                rewrite_tokens=p.rewrite_tokens,
                doc_id=p.doc_id,
                domain_hits=domains,
                primary_domain=prim,
                domain_signature=signature,
            ))
    return rows, metas, slots


def rotate_assignment(indices: list[int], slots: list[Slot]) -> dict[int, int]:
    """Return target-slot-index -> donor-rewrite-slot-index for one equal-word group."""
    if len(indices) == 1:
        return {indices[0]: indices[0]}
    order = sorted(indices, key=lambda i: (slots[i].rewrite_tokens, slots[i].row_index, slots[i].pair_id))
    n = len(order)
    best_shift = 1
    best_score: tuple[float, ...] | None = None
    for shift in range(1, n):
        donor_order = order[shift:] + order[:shift]
        same_pair = sum(1 for a, b in zip(order, donor_order) if a == b)
        same_row = sum(1 for a, b in zip(order, donor_order) if slots[a].row_index == slots[b].row_index)
        tok_abs = [abs(slots[b].rewrite_tokens - slots[a].rewrite_tokens) for a, b in zip(order, donor_order)]
        same_doc = sum(1 for a, b in zip(order, donor_order) if slots[a].doc_id and slots[a].doc_id == slots[b].doc_id)
        # First break true row-local source-view pairing, then keep token lengths close and avoid same-document accidental coherence.
        score = (same_pair, same_row, sum(tok_abs), max(tok_abs) if tok_abs else 0, same_doc, shift)
        if best_score is None or score < best_score:
            best_score = score
            best_shift = shift
    donor_order = order[best_shift:] + order[:best_shift]
    return {a: b for a, b in zip(order, donor_order)}


def repair_fixed_assignments(slots: list[Slot], assigned: dict[int, int]) -> dict[int, int]:
    """Remove singleton true-pair leftovers when the same rewrite-word length has alternatives."""
    fixed = [i for i, d in assigned.items() if i == d]
    for sidx in fixed:
        if assigned.get(sidx) != sidx:
            continue
        same_word = [i for i in assigned if slots[i].rewrite_words == slots[sidx].rewrite_words and i != sidx]
        if not same_word:
            continue
        best: tuple[float, ...] | None = None
        best_t: int | None = None
        for tidx in same_word:
            old_donor = assigned[tidx]
            if old_donor == sidx:
                continue
            # Proposed swap: sidx receives old_donor; tidx receives sidx.
            after_same_pair = int(old_donor == sidx) + int(tidx == sidx)
            after_same_row = int(slots[sidx].row_index == slots[old_donor].row_index) + int(slots[tidx].row_index == slots[sidx].row_index)
            primary_miss = int(slots[sidx].primary_domain != slots[old_donor].primary_domain) + int(slots[tidx].primary_domain != slots[sidx].primary_domain)
            signature_miss = int(slots[sidx].domain_signature != slots[old_donor].domain_signature) + int(slots[tidx].domain_signature != slots[sidx].domain_signature)
            same_doc = int(slots[sidx].doc_id and slots[sidx].doc_id == slots[old_donor].doc_id) + int(slots[tidx].doc_id and slots[tidx].doc_id == slots[sidx].doc_id)
            tok_abs = abs(slots[old_donor].rewrite_tokens - slots[sidx].rewrite_tokens) + abs(slots[sidx].rewrite_tokens - slots[tidx].rewrite_tokens)
            row_distance = -abs(slots[sidx].row_index - slots[tidx].row_index)
            score = (after_same_pair, after_same_row, primary_miss, signature_miss, same_doc, tok_abs, row_distance, tidx)
            if best is None or score < best:
                best = score
                best_t = tidx
        if best_t is None:
            continue
        old = assigned[best_t]
        assigned[sidx] = old
        assigned[best_t] = sidx
    return assigned


def make_domain_word_assignment(slots: list[Slot]) -> dict[int, int]:
    groups: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for s in slots:
        groups[(s.primary_domain, s.rewrite_words)].append(s.index)

    assigned: dict[int, int] = {}
    leftovers_by_word: dict[int, list[int]] = collections.defaultdict(list)
    for key in sorted(groups):
        inds = groups[key]
        if len(inds) >= 2:
            assigned.update(rotate_assignment(inds, slots))
        else:
            leftovers_by_word[key[1]].extend(inds)

    for word_len, inds in sorted(leftovers_by_word.items()):
        if len(inds) >= 2:
            assigned.update(rotate_assignment(inds, slots))
        else:
            # Temporarily assign singleton exact-word leftovers to themselves, then repair by swapping with an already assigned same-word slot when possible.
            assigned[inds[0]] = inds[0]

    assigned = repair_fixed_assignments(slots, assigned)

    if set(assigned) != {s.index for s in slots}:
        raise RuntimeError("assignment does not cover all source slots")
    if sorted(assigned.values()) != sorted(assigned.keys()):
        raise RuntimeError("assignment is not an exact rewrite-slot permutation")
    for target, donor in assigned.items():
        if slots[target].rewrite_words != slots[donor].rewrite_words:
            raise RuntimeError("rewrite word length changed")
    return assigned


def assignment_summary(slots: list[Slot], assignment: dict[int, int]) -> dict[str, Any]:
    token_deltas = []
    pair_token_deltas = []
    same_pair = same_row = same_primary = same_signature = domain_intersect = same_doc = 0
    exact_rewrite_token = 0
    original_rewrite_new_row_distances: list[float] = []
    donor_to_target = {donor: target for target, donor in assignment.items()}
    for target, donor in assignment.items():
        a = slots[target]
        b = slots[donor]
        if a.pair_id == b.pair_id:
            same_pair += 1
        if a.row_index == b.row_index:
            same_row += 1
        if a.primary_domain == b.primary_domain:
            same_primary += 1
        if a.domain_signature == b.domain_signature:
            same_signature += 1
        if set(a.domain_hits or ("no_domain",)) & set(b.domain_hits or ("no_domain",)):
            domain_intersect += 1
        if a.doc_id and a.doc_id == b.doc_id:
            same_doc += 1
        dt = b.rewrite_tokens - a.rewrite_tokens
        token_deltas.append(float(dt))
        if dt == 0:
            exact_rewrite_token += 1
        # Source tokens are fixed for target; replacement rewrite token length approximates pair token shift.
        pair_token_deltas.append(float(dt))
        moved_to = donor_to_target[target]
        original_rewrite_new_row_distances.append(float(abs(slots[moved_to].row_index - a.row_index)))
    n = len(slots)
    return {
        "slot_count": n,
        "same_pair_assignments": same_pair,
        "same_original_row_assignments": same_row,
        "same_primary_domain_assignments": same_primary,
        "same_primary_domain_rate": same_primary / n if n else None,
        "same_domain_signature_assignments": same_signature,
        "same_domain_signature_rate": same_signature / n if n else None,
        "domain_intersection_assignments": domain_intersect,
        "domain_intersection_rate": domain_intersect / n if n else None,
        "same_doc_assignments": same_doc,
        "same_doc_rate": same_doc / n if n else None,
        "exact_rewrite_token_length_assignments": exact_rewrite_token,
        "exact_rewrite_token_length_rate": exact_rewrite_token / n if n else None,
        "rewrite_token_delta_stats": stats(token_deltas),
        "rewrite_token_abs_delta_stats": stats([abs(x) for x in token_deltas]),
        "pair_token_delta_approx_stats": stats(pair_token_deltas),
        "true_rewrite_new_row_distance_stats": stats(original_rewrite_new_row_distances),
        "true_rewrite_stays_in_same_row_count": sum(1 for x in original_rewrite_new_row_distances if x == 0),
    }


def build_control_rows(
    rows: list[dict[str, Any]],
    metas: list[dict[str, Any]],
    pairs: dict[str, Pair],
    slots: list[Slot],
    assignment: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slot_by_pid_row_pos: dict[tuple[str, int, int], int] = {}
    for s in slots:
        slot_by_pid_row_pos[(s.pair_id, s.row_index, s.position_in_row)] = s.index

    out_rows: list[dict[str, Any]] = []
    out_meta: list[dict[str, Any]] = []
    for ridx, (row, meta) in enumerate(zip(rows, metas)):
        pids = [str(x) for x in (meta.get("pair_ids") or [])]
        if not pids:
            out_rows.append(dict(row))
            out_meta.append({
                "row_index": ridx,
                "example_id": row.get("example_id"),
                "words": int(row.get("words", wc(str(row.get("text", ""))))),
                "source_pair_ids": [],
                "donor_rewrite_pair_ids": [],
                "component_sources": meta.get("component_sources") or {},
                "copied_nonpair_changed_row": True,
            })
            continue
        units: list[str] = []
        donor_ids: list[str] = []
        donor_domains: list[str] = []
        for pos, pid in enumerate(pids):
            slot_idx = slot_by_pid_row_pos[(pid, ridx, pos)]
            donor_idx = assignment[slot_idx]
            donor_pid = slots[donor_idx].pair_id
            source_p = pairs[pid]
            donor_p = pairs[donor_pid]
            if donor_p.rewrite_words != source_p.rewrite_words:
                raise RuntimeError("rewrite word length mismatch during row construction")
            units.append(f"{source_p.source_text} {donor_p.rewrite_text}".strip())
            donor_ids.append(donor_pid)
            donor_domains.append(primary_domain(donor_p.domain_hits))
        new_text = " ".join(units)
        new_words = wc(new_text)
        old_words = int(row.get("words", wc(str(row.get("text", "")))))
        if new_words != old_words:
            raise RuntimeError(f"row word length changed at {ridx}: {new_words} != {old_words}")
        out_rows.append({"text": new_text, "words": new_words, "example_id": row.get("example_id"), "source": CONTROL_SOURCE})
        out_meta.append({
            "row_index": ridx,
            "example_id": row.get("example_id"),
            "words": new_words,
            "source_pair_ids": pids,
            "donor_rewrite_pair_ids": donor_ids,
            "source_primary_domains": [primary_domain(pairs[pid].domain_hits) for pid in pids],
            "donor_primary_domains": donor_domains,
            "component_sources": meta.get("component_sources") or {},
            "copied_nonpair_changed_row": False,
        })
    return out_rows, out_meta


def seq_visibility(tok: Any, rows: list[dict[str, Any]], metas: list[dict[str, Any]], pairs: dict[str, Pair]) -> dict[str, Any]:
    special_overhead = enc_len(tok, "hello", True) - enc_len(tok, "hello", False)
    cutoff = SEQ_LEN - special_overhead
    total = full = partial = hidden = source_visible = 0
    row_token_with_special = []
    row_token_no_special = []
    lost_by_position: collections.Counter[int] = collections.Counter()
    for row, meta in zip(rows, metas):
        source_ids = [str(x) for x in (meta.get("source_pair_ids") or [])]
        donor_ids = [str(x) for x in (meta.get("donor_rewrite_pair_ids") or [])]
        row_token_no_special.append(enc_len(tok, str(row["text"]), False))
        row_token_with_special.append(enc_len(tok, str(row["text"]), True))
        if not source_ids:
            continue
        prefix = ""
        for pos, (source_pid, donor_pid) in enumerate(zip(source_ids, donor_ids)):
            source = pairs[source_pid].source_text
            rewrite = pairs[donor_pid].rewrite_text
            unit = f"{source} {rewrite}".strip()
            source_prefix = prefix + (" " if prefix else "") + source
            pair_prefix = prefix + (" " if prefix else "") + unit
            start_tok = enc_len(tok, prefix, False) if prefix else 0
            source_end = enc_len(tok, source_prefix, False)
            pair_end = enc_len(tok, pair_prefix, False)
            total += 1
            visible_inside = max(0, min(pair_end, cutoff) - start_tok)
            sv = source_end <= cutoff
            pv = visible_inside > 0
            fv = pair_end <= cutoff
            if sv:
                source_visible += 1
            if fv:
                full += 1
            elif pv:
                partial += 1
                lost_by_position[pos] += 1
            else:
                hidden += 1
                lost_by_position[pos] += 1
            prefix = pair_prefix
    return {
        "seq_len": SEQ_LEN,
        "special_overhead": special_overhead,
        "cutoff_no_special": cutoff,
        "total_pair_occurrences": total,
        "source_visible_pairs": source_visible,
        "full_source_plus_donor_rewrite_visible_pairs": full,
        "partial_visible_pairs": partial,
        "hidden_pairs": hidden,
        "source_visible_rate": source_visible / total if total else None,
        "full_pair_visible_rate": full / total if total else None,
        "partial_visible_rate": partial / total if total else None,
        "hidden_rate": hidden / total if total else None,
        "rows_over_seq256_with_special": sum(1 for x in row_token_with_special if x > SEQ_LEN),
        "row_tokens_no_special_stats": stats([float(x) for x in row_token_no_special]),
        "row_tokens_with_special_stats": stats([float(x) for x in row_token_with_special]),
        "lost_pair_position_counts": dict(sorted(lost_by_position.items())),
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, pool_rows: list[dict[str, Any]]) -> None:
    total = 0
    n = len(pool_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(TRAIN_SHUFFLE_SEED + pass_i).shuffle(order)
            for idx in order:
                obj = pool_rows[idx]
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += int(obj["words"])
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training word exposure changed: {total}")


def make_note(payload: dict[str, Any]) -> str:
    a = payload["assignment"]
    geom = payload["row_geometry"]
    vis = payload["adjbreak_seq256_visibility"]
    files = payload["files"]
    return f"""# research compact-view reinvest adjacency-broken control

## Purpose

This CPU-only construction prepares the next mechanism comparison without starting a GPU run.  It should be used only if the running `compact_view_reinvest` full evaluation and seed43122 screen keep the endpoint scientifically alive.

The constructed control keeps the exact compact-view reinvest source-text multiset and compact-rewrite multiset, keeps the 10M pool word total and the changed-block row word sequence, and breaks the original source -> its own compact rewrite adjacency by assigning each source a rewrite from another source with the same rewrite word length, usually inside the same broad relation/topic stratum.

## Main matching facts

- slots/source-view units: {a['slot_count']}
- same-pair assignments left: {a['same_pair_assignments']}
- same original row assignments: {a['same_original_row_assignments']}
- same primary domain assignments: {a['same_primary_domain_assignments']} ({a['same_primary_domain_rate']:.6f})
- same domain-signature assignments: {a['same_domain_signature_assignments']} ({a['same_domain_signature_rate']:.6f})
- exact rewrite-token-length assignments: {a['exact_rewrite_token_length_assignments']} ({a['exact_rewrite_token_length_rate']:.6f})
- rewrite token absolute-delta mean/p95/max: {a['rewrite_token_abs_delta_stats']['mean']:.4f} / {a['rewrite_token_abs_delta_stats']['p95']:.4f} / {a['rewrite_token_abs_delta_stats']['max']:.4f}
- true rewrite stays in the same row as its original source: {a['true_rewrite_stays_in_same_row_count']}

## Row and sequence-interface facts

- pool rows / words: {payload['pool']['rows']} / {payload['pool']['words']}
- row word sequence identical to original reinvest view: {geom['row_word_sequence_identical_to_original_view']}
- changed-row token-with-special exact matches / rows: {geom['changed_row_tokens_with_special_exact_match_count']} / {geom['changed_rows']}
- changed-row token-with-special delta mean_abs/p95_abs/max_abs: {geom['changed_row_token_with_special_abs_delta_stats']['mean']:.4f} / {geom['changed_row_token_with_special_abs_delta_stats']['p95']:.4f} / {geom['changed_row_token_with_special_abs_delta_stats']['max']:.4f}
- adjbreak seq256 full source+donor-rewrite visibility: {vis['full_source_plus_donor_rewrite_visible_pairs']}/{vis['total_pair_occurrences']} ({vis['full_pair_visible_rate']:.6f})
- adjbreak source visibility: {vis['source_visible_pairs']}/{vis['total_pair_occurrences']} ({vis['source_visible_rate']:.6f})
- rows over seq256 with special tokens: {vis['rows_over_seq256_with_special']}

## Files

- 10M control pool: `{files['pool_10M']}`
- changed-block row metadata: `{files['changed_block_row_meta']}`
- JSON measurement: `{files['measurement_json']}`

The 100M training file was {'written' if files.get('training_100M_written') else 'not written'} in this step. If written later, it will use the same pass-wise row-index orders as the original A02 reinvest arms. The next expensive action should wait for the endpoint results already running.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-training", action="store_true", help="also materialize the 100M shuffled training stream; leave off until endpoint results justify training")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    overlay_meta = read_json(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json")
    changed_rows = int(overlay_meta["families"]["compact_reinvest"]["changed_block_rows"])
    pairs = load_pairs(tok)
    original_changed_rows, original_metas, slots = load_slots(pairs, changed_rows)
    if len(slots) != int(overlay_meta["families"]["compact_reinvest"]["pair_words"] >= 0) and len(slots) != len(pairs):
        # The equality to len(pairs) is the important one; the first expression only prevents static linters from ignoring metadata use.
        pass
    if len(slots) != len(pairs):
        raise RuntimeError(f"slot/pair mismatch: {len(slots)} != {len(pairs)}")

    assignment = make_domain_word_assignment(slots)
    assign_summary = assignment_summary(slots, assignment)
    adj_changed_rows, adj_metas = build_control_rows(original_changed_rows, original_metas, pairs, slots, assignment)

    original_row_words = [int(r.get("words", wc(str(r.get("text", ""))))) for r in original_changed_rows]
    adj_row_words = [int(r["words"]) for r in adj_changed_rows]
    orig_tok_with = [enc_len(tok, str(r["text"]), True) for r in original_changed_rows]
    adj_tok_with = [enc_len(tok, str(r["text"]), True) for r in adj_changed_rows]
    orig_tok_no = [enc_len(tok, str(r["text"]), False) for r in original_changed_rows]
    adj_tok_no = [enc_len(tok, str(r["text"]), False) for r in adj_changed_rows]
    tok_with_delta = [float(b - a) for a, b in zip(orig_tok_with, adj_tok_with)]
    tok_no_delta = [float(b - a) for a, b in zip(orig_tok_no, adj_tok_no)]

    # Build the full 10M pool by replacing only the changed prefix.  The common clean-Qwen filler is copied byte-semantically as JSON objects from the original view pool.
    original_pool = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"))
    if len(original_pool) <= changed_rows:
        raise RuntimeError("original pool unexpectedly short")
    pool_rows = adj_changed_rows + [dict(r) for r in original_pool[changed_rows:]]
    pool_words = sum(int(r.get("words", wc(str(r.get("text", ""))))) for r in pool_rows)
    if pool_words != TOTAL_WORDS:
        raise RuntimeError(f"pool words changed: {pool_words}")
    if [int(r.get("words", wc(str(r.get("text", ""))))) for r in pool_rows] != [int(r.get("words", wc(str(r.get("text", ""))))) for r in original_pool]:
        raise RuntimeError("full-pool row word sequence changed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_POOL, pool_rows)
    write_jsonl(OUT_META, adj_metas)
    training_written = False
    if args.write_training:
        write_training(OUT_TRAINING, pool_rows)
        training_written = True

    source_original = [s.source_text for s in slots]
    source_control = [pairs[s.pair_id].source_text for s in slots]
    rewrite_original = [s.rewrite_text for s in slots]
    rewrite_control = [pairs[slots[assignment[s.index]].pair_id].rewrite_text for s in slots]

    visibility = seq_visibility(tok, adj_changed_rows, adj_metas, pairs)
    payload: dict[str, Any] = {
        "status": "COMPACT_REINVEST_ADJBREAK_CONTROL",
        "scientific_purpose": "Prepare a low-cost mechanism control that preserves the compact_view_reinvest source and rewrite marginals while breaking true source-view adjacency before any additional GPU training is considered.",
        "inputs": {
            "pairs": str(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"),
            "original_view_pool_10M": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"),
            "original_view_changed_row_meta": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"),
            "overlay_metadata": str(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"),
            "tokenizer": str(TOKENIZER),
        },
        "construction": {
            "control_source_label": CONTROL_SOURCE,
            "assignment_rule": "Rotate compact rewrites within (primary_domain, rewrite_word_length) groups when possible; singleton domain-word groups rotate within rewrite_word_length; every donor rewrite is used exactly once.",
            "changed_rows": changed_rows,
            "pair_slots": len(slots),
            "nonpair_changed_rows_copied": sum(1 for m in adj_metas if m.get("copied_nonpair_changed_row")),
            "rewrite_word_length_preserved_per_slot": all(slots[i].rewrite_words == slots[d].rewrite_words for i, d in assignment.items()),
            "all_donor_rewrites_used_once": sorted(assignment.values()) == sorted(assignment.keys()),
        },
        "assignment": assign_summary,
        "multisets": {
            "source_multiset_hash_original": multiset_hash(source_original),
            "source_multiset_hash_control": multiset_hash(source_control),
            "source_multiset_preserved": multiset_hash(source_original) == multiset_hash(source_control),
            "rewrite_multiset_hash_original": multiset_hash(rewrite_original),
            "rewrite_multiset_hash_control": multiset_hash(rewrite_control),
            "rewrite_multiset_preserved": multiset_hash(rewrite_original) == multiset_hash(rewrite_control),
        },
        "row_geometry": {
            "changed_rows": changed_rows,
            "row_word_sequence_identical_to_original_view": original_row_words == adj_row_words,
            "changed_row_tokens_with_special_exact_match_count": sum(1 for a, b in zip(orig_tok_with, adj_tok_with) if a == b),
            "changed_row_tokens_no_special_exact_match_count": sum(1 for a, b in zip(orig_tok_no, adj_tok_no) if a == b),
            "changed_row_token_with_special_delta_stats": stats(tok_with_delta),
            "changed_row_token_with_special_abs_delta_stats": stats([abs(x) for x in tok_with_delta]),
            "changed_row_token_no_special_delta_stats": stats(tok_no_delta),
            "changed_row_token_no_special_abs_delta_stats": stats([abs(x) for x in tok_no_delta]),
            "original_rows_over_seq256_with_special": sum(1 for x in orig_tok_with if x > SEQ_LEN),
            "adjbreak_rows_over_seq256_with_special": sum(1 for x in adj_tok_with if x > SEQ_LEN),
        },
        "adjbreak_seq256_visibility": visibility,
        "pool": {
            "rows": len(pool_rows),
            "words": pool_words,
            "row_word_sequence_identical_to_original_view_pool": [int(r.get("words", wc(str(r.get("text", ""))))) for r in pool_rows] == [int(r.get("words", wc(str(r.get("text", ""))))) for r in original_pool],
            "filler_rows_copied_after_changed_block": len(pool_rows) - changed_rows,
        },
        "files": {
            "pool_10M": str(OUT_POOL),
            "pool_10M_sha256": sha256_file(OUT_POOL),
            "changed_block_row_meta": str(OUT_META),
            "changed_block_row_meta_sha256": sha256_file(OUT_META),
            "measurement_json": str(OUT_JSON),
            "training_100M": str(OUT_TRAINING),
            "training_100M_written": training_written,
        },
        "scientific_reading": {
            "what_it_would_test": "If compact_view_reinvest survives the running full and seed tests, training this control would ask whether local source-own-rewrite adjacency is necessary after preserving the exact source and rewrite marginals, exact word budget, and close trainer-interface geometry.",
            "what_it_does_not_test": "It is not an endpoint result and not a replacement for the full evaluation or seed43122 evidence now running; it also does not alter tokenizer, architecture, optimizer, or masking schedule.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(make_note(payload), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "pool_10M": str(OUT_POOL),
        "measurement_json": str(OUT_JSON),
        "note": str(NOTE),
        "same_pair_assignments": assign_summary["same_pair_assignments"],
        "same_original_row_assignments": assign_summary["same_original_row_assignments"],
        "same_primary_domain_rate": assign_summary["same_primary_domain_rate"],
        "rewrite_token_abs_delta_mean": assign_summary["rewrite_token_abs_delta_stats"]["mean"],
        "row_word_sequence_identical": payload["row_geometry"]["row_word_sequence_identical_to_original_view"],
        "full_pair_visible_rate": visibility["full_pair_visible_rate"],
        "training_written": training_written,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
