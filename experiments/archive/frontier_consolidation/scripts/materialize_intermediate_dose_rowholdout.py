#!/usr/bin/env python3
"""research: materialize an intermediate compact-dose row-holdout pool.

This is a general-label adaptation of the repaired research MAX materializer.
It builds exact-10M clean, hash-rotated-repeat, and compact-view arms for a
selected nested pair set such as the research midpoint `dose1p82` selection.  The
construction preserves the old 1x selected pair prefix, uses intact 160-word
clean-Qwen row holdout with the same seed/preserved source policy, shares filler
and topup across arms, and writes exact 100M view/repeat training streams.

No training/evaluation/upload is performed.
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
import time
from typing import Any, Iterable

WORKSPACE = pathlib.Path("experiments/archive/frontier_consolidation")
BASE_POOL_DEFAULT = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
PAIRS_DEFAULT = WORKSPACE / "data/dose_intermediate_select/selected_dose1p82_pairs.jsonl"
OLD_HELDOUT_DEFAULT = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl"
OUT_DIR_DEFAULT = WORKSPACE / "data/dose_1p82x_rowholdout_pools"
TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_PACKET_WORDS = 160
PRESERVE_SOURCE = "qwen_pair_packed"
BASE_CHANGED_BLOCK = 423_520
RNG_SEED = 82914124

@dataclasses.dataclass
class Pair:
    pair_id: str
    key: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    pair_words: int
    sentence_id: str = ""
    doc_id: str = ""
    domain_hits: list[str] = dataclasses.field(default_factory=list)
    origin: str = ""
    content_recall: float | None = None
    entity_recall: float | None = None
    number_recall: float | None = None
    soft_flags: list[str] = dataclasses.field(default_factory=list)
    source_risks: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Row:
    text: str
    words: int
    example_id: int
    source: str
    pair_ids: list[str] | None = None
    component_sources: dict[str, int] | None = None


def wc(text: str) -> int:
    return len((text or "").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_key(d: dict[str, Any]) -> str:
    k = str(d.get("key") or "").strip()
    if k.startswith("sid:") and "|doc:" in k:
        return k
    sid = str(d.get("sentence_id") or "").strip()
    doc = str(d.get("doc_id") or "").strip()
    if sid or doc:
        return f"sid:{sid}|doc:{doc}"
    return k or str(d.get("pair_id") or d.get("prompt_id") or "")


def norm_pair_id(d: dict[str, Any], key: str) -> str:
    pid = str(d.get("pair_id") or "").strip()
    if pid:
        return pid
    pr = str(d.get("prompt_id") or "").strip()
    if pr:
        return f"compact:{pr}"
    return f"compact:{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def pair_from_dict(d: dict[str, Any]) -> Pair:
    src = " ".join(str(d.get("source_text") or "").split())
    rew = " ".join(str(d.get("rewrite_text") or "").split())
    sw, rw = wc(src), wc(rew)
    key = norm_key(d)
    return Pair(
        pair_id=norm_pair_id(d, key),
        key=key,
        source_text=src,
        rewrite_text=rew,
        source_words=sw,
        rewrite_words=rw,
        pair_words=sw + rw,
        sentence_id=str(d.get("sentence_id") or ""),
        doc_id=str(d.get("doc_id") or ""),
        domain_hits=[str(x) for x in (d.get("domain_hits") or [])],
        origin=str(d.get("origin") or d.get("regime") or ""),
        content_recall=float(d["content_recall"]) if d.get("content_recall") is not None else None,
        entity_recall=float(d["entity_recall"]) if d.get("entity_recall") is not None else None,
        number_recall=float(d["number_recall"]) if d.get("number_recall") is not None else None,
        soft_flags=[str(x) for x in (d.get("soft_flags") or [])],
        source_risks=[str(x) for x in (d.get("source_risks") or [])],
    )


def read_pairs(path: pathlib.Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = pair_from_dict(json.loads(line))
            if not p.source_text or not p.rewrite_text:
                raise RuntimeError(f"missing text for pair {p.pair_id}")
            if p.pair_words > MAX_PACKET_WORDS:
                raise RuntimeError(f"pair too long for atomic pack {p.pair_id}: {p.pair_words}")
            pairs.append(p)
    if len({p.key for p in pairs}) != len(pairs):
        raise RuntimeError("pair keys are not unique")
    return pairs


def repeat_source_words(source_text: str, n_words: int, salt: str) -> str:
    toks = source_text.split()
    if not toks or n_words <= 0:
        return ""
    start = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16) % len(toks)
    rot = toks[start:] + toks[:start]
    out: list[str] = []
    while len(out) < n_words:
        out.extend(rot[: n_words - len(out)])
    return " ".join(out)


def pack_pair_rows(pairs: list[Pair], use_rewrite: bool, source_label: str, example_base: int) -> list[Row]:
    rows: list[Row] = []
    cur_segments: list[str] = []
    cur_words = 0
    cur_pair_ids: list[str] = []
    cur_domains: collections.Counter[str] = collections.Counter()
    for p in pairs:
        second = p.rewrite_text if use_rewrite else repeat_source_words(p.source_text, p.rewrite_words, p.pair_id)
        text = f"{p.source_text} {second}".strip()
        L = wc(text)
        if L != p.pair_words:
            raise RuntimeError(f"pair {p.pair_id} expected {p.pair_words}, built {L}")
        if cur_words and cur_words + L > MAX_PACKET_WORDS:
            rows.append(Row(" ".join(cur_segments), cur_words, example_base + len(rows), source_label, list(cur_pair_ids), dict(cur_domains)))
            cur_segments, cur_words, cur_pair_ids, cur_domains = [], 0, [], collections.Counter()
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(p.pair_id)
        for d in p.domain_hits or ["no_domain"]:
            cur_domains[d] += L
    if cur_segments:
        rows.append(Row(" ".join(cur_segments), cur_words, example_base + len(rows), source_label, list(cur_pair_ids), dict(cur_domains)))
    for r in rows:
        if r.words != wc(r.text) or r.words > MAX_PACKET_WORDS:
            raise RuntimeError("bad packed pair row")
    return rows


def row_to_record(r: Row) -> dict[str, Any]:
    return {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}


def read_base_pool(path: pathlib.Path) -> tuple[list[Row], dict[str, int]]:
    rows: list[Row] = []
    counts: collections.Counter[str] = collections.Counter()
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = " ".join(str(obj.get("text") or "").split())
            words = int(obj.get("words") or wc(text))
            if words != wc(text):
                raise RuntimeError(f"base row word mismatch {i}")
            source = str(obj.get("source") or "base")
            exid = int(obj.get("example_id") if obj.get("example_id") is not None else i)
            rows.append(Row(text, words, exid, source))
            counts[source] += words
            total += words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"base total {total}, expected {TOTAL_WORDS}")
    return rows, dict(counts)


def proportional_counts(total_by_source: dict[str, int], target_total: int) -> dict[str, int]:
    grand = sum(total_by_source.values())
    raw = {s: total_by_source[s] * target_total / grand for s in total_by_source}
    out = {s: int(raw[s]) for s in total_by_source}
    rem = target_total - sum(out.values())
    order = sorted(total_by_source, key=lambda s: (raw[s] - out[s], s), reverse=True)
    for s in order[:rem]:
        out[s] += 1
    return out


def choose_row_holdout(base_rows: list[Row], changed_budget: int, preserve_source: str, seed: int) -> tuple[list[Row], list[Row], dict[str, Any]]:
    if changed_budget % MAX_PACKET_WORDS:
        raise RuntimeError("changed_budget must be divisible by 160 for intact-row holdout")
    target_rows = changed_budget // MAX_PACKET_WORDS
    by_source: dict[str, list[Row]] = collections.defaultdict(list)
    protected: list[Row] = []
    noncandidate: list[Row] = []
    for r in base_rows:
        if r.source == preserve_source:
            protected.append(r)
        elif r.words == MAX_PACKET_WORDS:
            by_source[r.source].append(r)
        else:
            noncandidate.append(r)
    counts = {s: len(v) for s, v in by_source.items()}
    take_counts = proportional_counts(counts, target_rows)
    if any(take_counts[s] > counts[s] for s in take_counts):
        raise RuntimeError(f"over-asked source in holdout: counts={counts}, take={take_counts}")
    rng = random.Random(seed)
    held_ids: set[int] = set()
    heldout: list[Row] = []
    for s, rows in by_source.items():
        order = list(rows)
        rng.shuffle(order)
        chosen = order[:take_counts.get(s, 0)]
        heldout.extend(chosen)
        held_ids.update(id(r) for r in chosen)
    if len(heldout) != target_rows:
        raise RuntimeError(f"heldout rows {len(heldout)} != {target_rows}")
    remaining = [r for r in base_rows if id(r) not in held_ids]
    rng.shuffle(heldout)
    rng.shuffle(remaining)
    wc_by_source: collections.Counter[str] = collections.Counter()
    for r in heldout:
        wc_by_source[r.source] += r.words
    return heldout, remaining, {
        "target_rows": target_rows,
        "preserved_source": preserve_source,
        "protected_rows": len(protected),
        "protected_words": sum(r.words for r in protected),
        "noncandidate_rows": len(noncandidate),
        "candidate_rows_by_source": counts,
        "heldout_rows_by_source": dict(collections.Counter(r.source for r in heldout).most_common()),
        "heldout_words_by_source": dict(wc_by_source.most_common()),
    }


def stream_from_rows(rows: list[Row]) -> list[tuple[str, str]]:
    stream: list[tuple[str, str]] = []
    for r in rows:
        stream.extend((w, r.source) for w in r.text.split())
    return stream


def stream_rows(stream: list[tuple[str, str]], lengths: list[int], label_prefix: str, example_base: int) -> list[Row]:
    need = sum(lengths)
    if need > len(stream):
        raise RuntimeError(f"stream too short: need {need}, have {len(stream)}")
    rows: list[Row] = []
    pos = 0
    for i, L in enumerate(lengths):
        seg = stream[pos:pos + L]
        counts = collections.Counter(s for _, s in seg)
        common = counts.most_common(1)[0][0] if counts else "base"
        rows.append(Row(" ".join(w for w, _ in seg), L, example_base + i, f"{label_prefix}::{common}", None, dict(counts)))
        pos += L
    return rows


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path) -> None:
    write_jsonl(path, [row_to_record(r) for r in rows])
    with meta_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r.pair_ids or r.component_sources:
                f.write(json.dumps({
                    "row_index": i,
                    "example_id": r.example_id,
                    "words": r.words,
                    "pair_ids": r.pair_ids or [],
                    "component_sources": r.component_sources or {},
                }, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[Row], seed: int) -> None:
    total = 0
    n = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training total {total}")


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {"n": len(xs), "min": xs[0], "mean": sum(vals) / len(vals), "median": xs[len(xs) // 2], "max": xs[-1], "sum": sum(vals)}


def pair_summary(pairs: list[Pair], changed_budget: int, topup: int) -> dict[str, Any]:
    src = sum(p.source_words for p in pairs)
    rew = sum(p.rewrite_words for p in pairs)
    origins: collections.Counter[str] = collections.Counter()
    for p in pairs:
        origins[p.origin or "unknown"] += p.pair_words
    return {
        "pairs": len(pairs),
        "source_words": src,
        "rewrite_words": rew,
        "pair_words": src + rew,
        "changed_block_budget_words": changed_budget,
        "neutral_cleanqwen_topup_words_inside_changed_block": topup,
        "unique_docs": len({p.doc_id for p in pairs}),
        "origin_pair_words": dict(origins.most_common()),
        "rewrite_to_source_ratio_weighted": rew / max(1, src),
        "rewrite_to_source_ratio_stats": stat([p.rewrite_words / max(1, p.source_words) for p in pairs]),
        "content_recall_stats": stat([float(p.content_recall) for p in pairs if p.content_recall is not None]),
        "entity_recall_stats": stat([float(p.entity_recall) for p in pairs if p.entity_recall is not None]),
        "number_recall_stats": stat([float(p.number_recall) for p in pairs if p.number_recall is not None]),
        "domain_hit_counts": dict(collections.Counter(d for p in pairs for d in (p.domain_hits or ["no_domain"])).most_common()),
    }


def source_word_counts(rows: list[Row]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c.most_common())


def read_old_heldout_ids(path: pathlib.Path) -> set[int]:
    if not path.exists():
        return set()
    ids: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("example_id") is not None:
                    ids.add(int(d["example_id"]))
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIRS_DEFAULT))
    ap.add_argument("--base-pool", default=str(BASE_POOL_DEFAULT))
    ap.add_argument("--old-heldout", default=str(OLD_HELDOUT_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--label", default="dose1p82")
    ap.add_argument("--view-label", default="compact_view_dose1p82x")
    ap.add_argument("--repeat-label", default="compact_repeat_dose1p82x")
    ap.add_argument("--clean-label", default="cleanqwen_lengthmatched_dose1p82x")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--write-training", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(pathlib.Path(args.pairs))
    pair_words = sum(p.pair_words for p in pairs)
    changed_budget = int(math.ceil(pair_words / MAX_PACKET_WORDS) * MAX_PACKET_WORDS)
    topup_words = changed_budget - pair_words
    dose_multiple = changed_budget / BASE_CHANGED_BLOCK

    base_rows, base_counts = read_base_pool(pathlib.Path(args.base_pool))
    heldout, filler, holdout_meta = choose_row_holdout(base_rows, changed_budget, PRESERVE_SOURCE, args.seed)
    holdout_stream = stream_from_rows(heldout)
    view_rows = pack_pair_rows(pairs, True, f"{args.view_label}_matched_rowholdout", 1_280_000)
    repeat_rows = pack_pair_rows(pairs, False, f"{args.repeat_label}_matched_rowholdout", 1_280_000)
    pair_lengths = [r.words for r in view_rows]
    topup_lengths: list[int] = []
    rem = topup_words
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        topup_lengths.append(L)
        rem -= L
    topup = stream_rows(holdout_stream, topup_lengths, f"neutral_cleanqwen_topup_{args.label}", 1_380_000) if topup_lengths else []
    clean_changed = stream_rows(holdout_stream, pair_lengths + topup_lengths, f"{args.clean_label}", 1_480_000)
    filler_rows = [Row(r.text, r.words, r.example_id, r.source) for r in filler]
    random.Random(args.seed + 12345).shuffle(filler_rows)
    arms = {
        args.clean_label: clean_changed + filler_rows,
        args.repeat_label: repeat_rows + topup + filler_rows,
        args.view_label: view_rows + topup + filler_rows,
    }
    lengths = {name: tuple(r.words for r in rows) for name, rows in arms.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError("arm row-length sequences differ")
    totals = {name: sum(r.words for r in rows) for name, rows in arms.items()}
    if any(v != TOTAL_WORDS for v in totals.values()):
        raise RuntimeError(f"bad totals {totals}")

    pool_paths: dict[str, str] = {}
    train_paths: dict[str, str] = {}
    meta_paths: dict[str, str] = {}
    train_seed = args.seed + 7000
    for name, rows in arms.items():
        pool_p = out_dir / f"{name}_10M.jsonl"
        meta_p = out_dir / f"{name}_changed_block_rows_meta.jsonl"
        write_pool(pool_p, rows, meta_p)
        pool_paths[name] = str(pool_p)
        meta_paths[name] = str(meta_p)
        if args.write_training and name != args.clean_label:
            train_p = out_dir / f"{name}_100M.jsonl"
            write_training(train_p, rows, train_seed)
            train_paths[name] = str(train_p)

    write_jsonl(out_dir / "heldout_cleanqwen_rows.jsonl", [row_to_record(r) for r in heldout])
    write_jsonl(out_dir / "common_filler_rows.jsonl", [row_to_record(r) for r in filler_rows])

    old_held_ids = read_old_heldout_ids(pathlib.Path(args.old_heldout))
    new_held_ids = {r.example_id for r in heldout}
    old_subset = old_held_ids <= new_held_ids if old_held_ids else None
    hashes = {pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in list(pool_paths.values()) + list(train_paths.values()) + list(meta_paths.values())}
    hashes["heldout_cleanqwen_rows.jsonl"] = sha256_file(out_dir / "heldout_cleanqwen_rows.jsonl")
    hashes["common_filler_rows.jsonl"] = sha256_file(out_dir / "common_filler_rows.jsonl")

    payload = {
        "status": "INTERMEDIATE_DOSE_ROWHOLDOUT_POOLS_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "label": args.label,
        "scientific_purpose": "Third fixed-budget dose instrument to locate whether the value of restructuring turns over between old 1x and research MAX.",
        "inputs": {
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(pathlib.Path(args.pairs)),
            "base_pool": str(args.base_pool),
            "base_pool_sha256": sha256_file(pathlib.Path(args.base_pool)),
            "old_1x_heldout": str(args.old_heldout),
        },
        "dose": {
            "base_changed_block_words": BASE_CHANGED_BLOCK,
            "selected_pair_words": pair_words,
            "changed_block_budget_words": changed_budget,
            "changed_budget_fraction_of_10M": changed_budget / TOTAL_WORDS,
            "dose_multiple_vs_1x_budget": dose_multiple,
            "topup_words": topup_words,
            "pair_rows": len(view_rows),
        },
        "labels": {"clean": args.clean_label, "repeat": args.repeat_label, "view": args.view_label},
        "base_split": {
            "base_source_counts": base_counts,
            "row_holdout": holdout_meta,
            "old_1x_heldout_rows": len(old_held_ids),
            "old_1x_heldout_subset_of_intermediate": old_subset,
            "old_1x_heldout_overlap_rows": len(old_held_ids & new_held_ids) if old_held_ids else None,
            "common_filler_rows": len(filler_rows),
            "common_filler_words": sum(r.words for r in filler_rows),
            "common_filler_source_counts": source_word_counts(filler_rows),
        },
        "pair_summary": pair_summary(pairs, changed_budget, topup_words),
        "audit": {
            "word_totals": totals,
            "all_exact_10M": all(v == TOTAL_WORDS for v in totals.values()),
            "row_counts": {name: len(rows) for name, rows in arms.items()},
            "row_length_sequence_identical_all_arms": len(set(lengths.values())) == 1,
            "view_repeat_suffix_identical_after_changed_block": all(row_to_record(a) == row_to_record(b) for a, b in zip(arms[args.view_label][len(view_rows):], arms[args.repeat_label][len(repeat_rows):])),
            "training_order_seed_identical_for_view_and_repeat": args.write_training,
            "tokenizer_policy": "Hold fixed research compliant16k_reinvest10M tokenizer for all dose arms; do not refit per dose.",
            "mechanism_instrument_not_leaderboard_submission": True,
        },
        "files": {"pools": pool_paths, "training": train_paths, "row_meta": meta_paths, "heldout_cleanqwen_rows": str(out_dir / "heldout_cleanqwen_rows.jsonl"), "common_filler_rows": str(out_dir / "common_filler_rows.jsonl")},
        "sha256": hashes,
        "boundary": "Intermediate arms are scientific mechanism instruments, not leaderboard submissions. Stable-family evaluation only is intended if trained.",
        "elapsed_sec": round(time.time() - t0, 2),
        "no_training_or_evaluation": True,
    }
    meta_path = out_dir / f"{args.label}_rowholdout_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research intermediate row-holdout dose pools: {args.label}",
        "",
        f"Selected pair words: {pair_words:,}; row-holdout changed block: {changed_budget:,} words; dose {dose_multiple:.4f}x; rho={changed_budget / TOTAL_WORDS:.5f}; topup {topup_words} words.",
        f"Arms have exact 10M words and identical row-length sequences: {payload['audit']['row_length_sequence_identical_all_arms']}.",
        f"Old 1x heldout subset of intermediate heldout: {old_subset}; overlap rows {payload['base_split']['old_1x_heldout_overlap_rows']} / {len(old_held_ids)}.",
        "Tokenizer stays fixed to research compliant16k_reinvest10M. These arms are mechanism instruments, not leaderboard submissions.",
        "",
        f"Metadata: `{meta_path}`",
    ]
    (out_dir / f"{args.label}_rowholdout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "label": args.label,
        "dose_multiple_vs_1x_budget": dose_multiple,
        "changed_block_budget_words": changed_budget,
        "selected_pair_words": pair_words,
        "topup_words": topup_words,
        "row_length_sequence_identical_all_arms": payload["audit"]["row_length_sequence_identical_all_arms"],
        "view_repeat_suffix_identical_after_changed_block": payload["audit"]["view_repeat_suffix_identical_after_changed_block"],
        "old_1x_heldout_subset_of_intermediate": old_subset,
        "metadata": str(meta_path),
        "training": train_paths,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
