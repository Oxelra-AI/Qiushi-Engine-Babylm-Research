#!/usr/bin/env python3
"""research: Build dose-response training pools at a specified dose level.

Dose = fraction of 10M budget restructured into source+compact-view packets.
  dose=0: clean reference (no changed block)
  dose=1: current 12,155-pair compact-reinvest (base_changed=423,520 words)
  dose=N: N × 423,520 words in the changed block (needs more pairs)

Uses the same materializer logic as research but parameterizes the changed block size
and reads from the combined accepted pairs pool.

Produces two training-ready 100M-word JSONL files: view arm and repeat arm,
plus a metadata/audit JSON for verification.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import pathlib
import random
import statistics
import time
from typing import Any, Iterable

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
BASE_POOL_DEFAULT = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
COMBINED_PAIRS_DEFAULT = WORKSPACE / "data/expansion_analysis/combined_all_accepted_pairs.jsonl"

TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_PACKET_WORDS = 160
PRESERVE_SOURCE = "qwen_pair_packed"
BASE_CHANGED_BLOCK = 423520  # 1x dose


@dataclasses.dataclass
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    sentence_id: str = ""
    doc_id: str = ""
    domain_hits: list[str] = dataclasses.field(default_factory=list)
    content_recall: float | None = None
    entity_recall: float | None = None
    number_recall: float | None = None
    soft_flags: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Row:
    text: str
    words: int
    example_id: int
    source: str
    pair_ids: list[str] | None = None


def wc(text: str) -> int:
    return len((text or "").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pair_from_dict(d: dict[str, Any]) -> Pair:
    src = " ".join(str(d.get("source_text") or "").split())
    rew = " ".join(str(d.get("rewrite_text") or "").split())
    sw = wc(src)
    rw = wc(rew)
    pw = int(d.get("pair_words") or (sw + rw))
    if pw != sw + rw:
        pw = sw + rw
    return Pair(
        pair_id=str(d.get("pair_id") or d.get("prompt_id") or ""),
        source_text=src, rewrite_text=rew,
        source_words=sw, rewrite_words=rw, pair_words=pw,
        sentence_id=str(d.get("sentence_id") or ""),
        doc_id=str(d.get("doc_id") or ""),
        domain_hits=[str(x) for x in (d.get("domain_hits") or [])],
        content_recall=float(d["content_recall"]) if d.get("content_recall") is not None else None,
        entity_recall=float(d["entity_recall"]) if d.get("entity_recall") is not None else None,
        number_recall=float(d["number_recall"]) if d.get("number_recall") is not None else None,
        soft_flags=[str(x) for x in (d.get("soft_flags") or [])],
    )


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    h = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16)
    start = h % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def pack_pair_rows(pairs: list[Pair], use_rewrite: bool, source_label: str, example_base: int) -> list[Row]:
    rows = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    for p in pairs:
        second = p.rewrite_text if use_rewrite else repeat_source_words(p.source_text, p.rewrite_words, p.pair_id)
        text = f"{p.source_text} {second}".strip()
        L = wc(text)
        if L != p.pair_words:
            raise RuntimeError(f"pair {p.pair_id} expected {p.pair_words}, built {L}")
        if cur_words and cur_words + L > MAX_PACKET_WORDS:
            rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows),
                            source=source_label, pair_ids=list(cur_pair_ids)))
            cur_segments, cur_words, cur_pair_ids = [], 0, []
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(p.pair_id)
    if cur_segments:
        rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows),
                        source=source_label, pair_ids=list(cur_pair_ids)))
    for r in rows:
        if r.words != wc(r.text):
            raise RuntimeError("packed row word mismatch")
        if r.words > MAX_PACKET_WORDS:
            raise RuntimeError("packed row exceeds max")
    return rows


def row_to_record(r: Row) -> dict[str, Any]:
    return {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}


def read_base_pool(path: pathlib.Path) -> tuple[list[Row], dict[str, int]]:
    rows = []
    source_counts: collections.Counter[str] = collections.Counter()
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = " ".join(str(obj.get("text") or "").split())
            words = wc(text)
            source = str(obj.get("source") or "base")
            example_id = int(obj.get("example_id") if obj.get("example_id") is not None else i)
            rows.append(Row(text=text, words=words, example_id=example_id, source=source))
            source_counts[source] += words
            total += words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"base pool has {total}, expected {TOTAL_WORDS}")
    return rows, dict(source_counts)


def proportional_drop(by_source: dict[str, list[str]], budget: int) -> dict[str, int]:
    totals = {s: len(ws) for s, ws in by_source.items()}
    grand = sum(totals.values())
    if grand < budget:
        raise RuntimeError(f"not enough words: need {budget}, have {grand}")
    raw = {s: totals[s] * budget / grand for s in totals}
    floor = {s: int(raw[s]) for s in totals}
    rem = budget - sum(floor.values())
    order = sorted(totals, key=lambda s: (raw[s] - floor[s], s), reverse=True)
    out = dict(floor)
    for s in order[:rem]:
        out[s] += 1
    return out


def split_drop_remaining(by_source: dict[str, list[str]], drop_counts: dict[str, int], seed: int):
    drop = []
    remaining = []
    for source in sorted(by_source):
        words = by_source[source]
        n = drop_counts.get(source, 0)
        if n <= 0:
            remaining.extend((w, source) for w in words)
            continue
        h = int(hashlib.sha1(f"{seed}:{source}".encode("utf-8")).hexdigest()[:8], 16)
        start = h % len(words)
        rotated = words[start:] + words[:start]
        drop.extend((w, source) for w in rotated[:n])
        remaining.extend((w, source) for w in rotated[n:])
    rng = random.Random(seed + 777)
    rng.shuffle(drop)
    rng.shuffle(remaining)
    return drop, remaining


def stream_rows(stream, lengths, label_prefix, example_base):
    rows = []
    pos = 0
    for i, L in enumerate(lengths):
        seg = stream[pos:pos+L]
        words = [w for w, _ in seg]
        source_counts = collections.Counter(s for _, s in seg)
        common = source_counts.most_common(1)[0][0] if source_counts else "base"
        rows.append(Row(text=" ".join(words), words=L, example_id=example_base + i,
                        source=f"{label_prefix}::{common}"))
        pos += L
    return rows


def pack_filler(protected, remaining_stream, needed_total, seed):
    protected_words = sum(r.words for r in protected)
    needed_rem = needed_total - protected_words
    lengths = []
    rem = needed_rem
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        lengths.append(L)
        rem -= L
    official = stream_rows(remaining_stream, lengths, "cleanqwen_base_filler", 810000)
    rows = [Row(r.text, r.words, r.example_id, r.source) for r in protected] + official
    random.Random(seed + 12345).shuffle(rows)
    total = sum(r.words for r in rows)
    if total != needed_total:
        raise RuntimeError(f"filler total {total}, expected {needed_total}")
    return rows


def write_training(path, rows, seed):
    total = 0
    n_rows = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n_rows))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training total {total}")


def select_pairs_for_budget(all_pairs: list[Pair], budget: int) -> tuple[list[Pair], int]:
    """Greedily select pairs up to word budget. Returns (selected_pairs, total_pair_words)."""
    selected = []
    used = 0
    for p in all_pairs:
        if p.pair_words > MAX_PACKET_WORDS:
            continue
        if used + p.pair_words > budget:
            continue
        selected.append(p)
        used += p.pair_words
    return selected, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dose", type=float, required=True, help="Dose multiplier (1.0 = base changed block)")
    ap.add_argument("--combined-pairs", default=str(COMBINED_PAIRS_DEFAULT))
    ap.add_argument("--base-pool", default=str(BASE_POOL_DEFAULT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=82914255)
    ap.add_argument("--write-training", action="store_true", help="Write 100M training JSONLs")
    ap.add_argument("--plan-only", action="store_true", help="Only report capacity without writing pools")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dose = args.dose
    changed_budget = int(round(BASE_CHANGED_BLOCK * dose))
    print(f"Dose {dose}x: changed_budget = {changed_budget:,} words ({changed_budget/TOTAL_WORDS*100:.2f}% of 10M)", flush=True)

    # Read combined pairs
    print("Reading pairs...", flush=True)
    raw_pairs = read_jsonl(pathlib.Path(args.combined_pairs))
    all_pairs = []
    for d in raw_pairs:
        try:
            p = pair_from_dict(d)
            if p.source_text and p.rewrite_text and p.pair_words > 0 and p.pair_words <= MAX_PACKET_WORDS:
                all_pairs.append(p)
        except Exception:
            pass
    print(f"  Valid pairs: {len(all_pairs)}, total pair_words: {sum(p.pair_words for p in all_pairs):,}", flush=True)

    # Select pairs for this dose
    selected, used_pw = select_pairs_for_budget(all_pairs, changed_budget)
    topup = changed_budget - used_pw
    print(f"  Selected: {len(selected)} pairs, {used_pw:,} pair_words, topup: {topup}", flush=True)

    if used_pw < changed_budget * 0.95:
        print(f"WARNING: can only fill {used_pw/changed_budget*100:.1f}% of budget. Reduce dose or add pairs.", flush=True)

    meta = {
        "status": "DOSE_POOL_PLAN" if args.plan_only else "DOSE_POOL_MATERIALIZED",
        "dose": dose,
        "changed_budget_words": changed_budget,
        "changed_budget_fraction": changed_budget / TOTAL_WORDS,
        "pairs_selected": len(selected),
        "pair_words_used": used_pw,
        "topup_words": topup,
        "total_pairs_available": len(all_pairs),
        "total_pair_words_available": sum(p.pair_words for p in all_pairs),
        "max_feasible_dose": sum(p.pair_words for p in all_pairs) / BASE_CHANGED_BLOCK,
        "unique_docs": len({p.doc_id for p in selected}),
        "combined_pairs_path": args.combined_pairs,
        "base_pool_path": args.base_pool,
    }

    if args.plan_only:
        meta_path = out_dir / "dose_pool_plan.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
        return

    # Read base pool
    print("Reading base pool...", flush=True)
    base_rows, base_source_counts = read_base_pool(pathlib.Path(args.base_pool))

    # Separate protected and candidate rows
    protected = []
    by_source: dict[str, list[str]] = collections.defaultdict(list)
    for r in base_rows:
        if r.source == PRESERVE_SOURCE:
            protected.append(r)
        else:
            by_source[r.source].extend(r.text.split())

    # Drop words proportionally
    print("Building changed block...", flush=True)
    drop_counts = proportional_drop(by_source, changed_budget)
    drop_stream, remaining_stream = split_drop_remaining(by_source, drop_counts, args.seed)

    # Build common filler
    filler_budget = TOTAL_WORDS - changed_budget
    filler = pack_filler(protected, remaining_stream, filler_budget, args.seed)

    # Build view and repeat arms
    dose_label = f"dose{dose:.1f}x".replace(".", "p")
    view_label = f"compact_view_{dose_label}"
    repeat_label = f"compact_repeat_{dose_label}"

    view_rows = pack_pair_rows(selected, use_rewrite=True, source_label=view_label, example_base=900000)
    repeat_rows = pack_pair_rows(selected, use_rewrite=False, source_label=repeat_label, example_base=900000)
    pair_lengths = [r.words for r in view_rows]

    # Topup rows
    topup_lengths = []
    rem = topup
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        topup_lengths.append(L)
        rem -= L
    topup_rows = stream_rows(drop_stream, topup_lengths, f"neutral_topup_{dose_label}", 850000) if topup_lengths else []

    # Assemble complete pools
    view_pool = view_rows + topup_rows + list(filler)
    repeat_pool = repeat_rows + topup_rows + list(filler)

    # Verify
    view_total = sum(r.words for r in view_pool)
    repeat_total = sum(r.words for r in repeat_pool)
    assert view_total == TOTAL_WORDS, f"view total {view_total}"
    assert repeat_total == TOTAL_WORDS, f"repeat total {repeat_total}"
    print(f"  View pool: {len(view_pool)} rows, {view_total:,} words ✓", flush=True)
    print(f"  Repeat pool: {len(repeat_pool)} rows, {repeat_total:,} words ✓", flush=True)

    # Write pool files (10M single-pass) for audit
    view_pool_path = out_dir / f"compact_view_{dose_label}_10M.jsonl"
    repeat_pool_path = out_dir / f"compact_repeat_{dose_label}_10M.jsonl"
    for path, rows in [(view_pool_path, view_pool), (repeat_pool_path, repeat_pool)]:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")

    # Write training files (10 passes = 100M)
    if args.write_training:
        print("Writing 100M training files...", flush=True)
        view_train_path = out_dir / f"compact_view_{dose_label}_100M.jsonl"
        repeat_train_path = out_dir / f"compact_repeat_{dose_label}_100M.jsonl"
        write_training(view_train_path, view_pool, args.seed + 100)
        write_training(repeat_train_path, repeat_pool, args.seed + 200)
        meta["training_files"] = {
            "view": str(view_train_path),
            "repeat": str(repeat_train_path),
        }
        print(f"  Training files written", flush=True)

    meta.update({
        "view_pool": str(view_pool_path),
        "repeat_pool": str(repeat_pool_path),
        "view_rows": len(view_pool),
        "repeat_rows": len(repeat_pool),
        "filler_rows": len(filler),
        "filler_words": sum(r.words for r in filler),
        "pair_rows": len(view_rows),
        "topup_rows": len(topup_rows),
        "topup_words": sum(r.words for r in topup_rows),
        "protected_qwen_words": sum(r.words for r in protected),
        "elapsed_sec": round(time.time() - t0, 1),
    })

    meta_path = out_dir / f"dose_pool_{dose_label}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k not in ("training_files",)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
