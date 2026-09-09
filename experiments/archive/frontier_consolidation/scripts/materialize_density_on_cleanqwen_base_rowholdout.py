#!/usr/bin/env python3
"""Repair the clean-Qwen overlay using coherent row-level official holdout.

The first clean-Qwen overlay preserved the inherited Qwen-pair rows but split the
non-Qwen official base into shuffled word streams, which would damage the inherited
base distribution.  This repaired materializer removes an exact number of intact
160-word non-Qwen rows from the clean-Qwen 10M corpus, keeps all remaining base rows
coherent and shared across arms, and uses only the held-out rows as the lengthmatched
changed-block stream/topup.  It therefore maps the FineWeb density contrast closer
to the inherited 41.34 clean-Qwen anchor without destroying most of the base.

No BabyLM evaluation data or labels are read.
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
DENSITY_META_DEFAULT = WORKSPACE / "data/density_core_reinvestment_full/density_core_reinvestment_metadata.json"
BASE_POOL_DEFAULT = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
OUT_DEFAULT = WORKSPACE / "data/density_cleanqwen_overlay_rowholdout_highprecision"
NOTE_DEFAULT = (WORKSPACE.parents[2] / 'research/notes/frontier_consolidation/density_cleanqwen_overlay_rowholdout_highprecision.md')
TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_PACKET_WORDS = 160
PRESERVE_SOURCE = "qwen_pair_packed"
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
    regime: str = ""
    content_recall: float | None = None
    content_overlap: float | None = None
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


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pair_from_dict(d: dict[str, Any]) -> Pair:
    p = Pair(
        pair_id=str(d.get("pair_id") or d.get("prompt_id") or ""),
        key=str(d.get("key") or d.get("sentence_id") or d.get("pair_id") or ""),
        source_text=" ".join(str(d.get("source_text") or "").split()),
        rewrite_text=" ".join(str(d.get("rewrite_text") or "").split()),
        source_words=int(d.get("source_words") or wc(str(d.get("source_text") or ""))),
        rewrite_words=int(d.get("rewrite_words") or wc(str(d.get("rewrite_text") or ""))),
        pair_words=int(d.get("pair_words") or 0),
        sentence_id=str(d.get("sentence_id") or ""),
        doc_id=str(d.get("doc_id") or ""),
        domain_hits=[str(x) for x in (d.get("domain_hits") or [])],
        regime=str(d.get("regime") or ""),
        content_recall=float(d["content_recall"]) if d.get("content_recall") is not None else None,
        content_overlap=float(d["content_overlap"]) if d.get("content_overlap") is not None else None,
        entity_recall=float(d["entity_recall"]) if d.get("entity_recall") is not None else None,
        number_recall=float(d["number_recall"]) if d.get("number_recall") is not None else None,
        soft_flags=[str(x) for x in (d.get("soft_flags") or [])],
        source_risks=[str(x) for x in (d.get("source_risks") or [])],
    )
    if p.pair_words <= 0:
        p.pair_words = p.source_words + p.rewrite_words
    return p


def read_pairs(path: pathlib.Path) -> list[Pair]:
    pairs = [pair_from_dict(d) for d in read_jsonl(path)]
    for p in pairs:
        if not p.source_text or not p.rewrite_text:
            raise RuntimeError(f"malformed pair {p.pair_id}: missing text")
        if p.source_words != wc(p.source_text):
            raise RuntimeError(f"source word mismatch {p.pair_id}")
        if p.rewrite_words != wc(p.rewrite_text):
            raise RuntimeError(f"rewrite word mismatch {p.pair_id}")
        if p.pair_words != p.source_words + p.rewrite_words:
            raise RuntimeError(f"pair word mismatch {p.pair_id}")
        if p.pair_words > MAX_PACKET_WORDS:
            raise RuntimeError(f"pair too long {p.pair_id}: {p.pair_words}")
    return pairs


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
            rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows), source=source_label,
                            pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains)))
            cur_segments, cur_words, cur_pair_ids, cur_domains = [], 0, [], collections.Counter()
        cur_segments.append(text)
        cur_words += L
        cur_pair_ids.append(p.pair_id)
        for d in p.domain_hits or ["no_domain"]:
            cur_domains[d] += L
    if cur_segments:
        rows.append(Row(text=" ".join(cur_segments), words=cur_words, example_id=example_base + len(rows), source=source_label,
                        pair_ids=list(cur_pair_ids), component_sources=dict(cur_domains)))
    for r in rows:
        if r.words != wc(r.text):
            raise RuntimeError("packed row word mismatch")
        if r.words > MAX_PACKET_WORDS:
            raise RuntimeError("packed row exceeds max")
    return rows


def row_to_record(r: Row) -> dict[str, Any]:
    return {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}


def read_base_pool(path: pathlib.Path) -> tuple[list[Row], dict[str, int]]:
    rows: list[Row] = []
    source_counts: collections.Counter[str] = collections.Counter()
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = " ".join(str(obj.get("text") or "").split())
            words = int(obj.get("words") or wc(text))
            if words != wc(text):
                raise RuntimeError(f"base row word mismatch at {i}")
            source = str(obj.get("source") or "base")
            example_id = int(obj.get("example_id") if obj.get("example_id") is not None else i)
            rows.append(Row(text=text, words=words, example_id=example_id, source=source))
            source_counts[source] += words
            total += words
    if total != TOTAL_WORDS:
        raise RuntimeError(f"base pool {path} has {total}, expected {TOTAL_WORDS}")
    return rows, dict(source_counts)


def proportional_counts(total_by_source: dict[str, int], target_total: int) -> dict[str, int]:
    grand = sum(total_by_source.values())
    raw = {s: total_by_source[s] * target_total / grand for s in total_by_source}
    floor = {s: int(raw[s]) for s in total_by_source}
    rem = target_total - sum(floor.values())
    order = sorted(total_by_source, key=lambda s: (raw[s] - floor[s], s), reverse=True)
    out = dict(floor)
    for s in order[:rem]:
        out[s] += 1
    return out


def choose_row_holdout(base_rows: list[Row], changed_budget: int, preserve_source: str, seed: int) -> tuple[list[Row], list[Row], dict[str, Any]]:
    if changed_budget % MAX_PACKET_WORDS != 0:
        raise RuntimeError("rowholdout requires changed_budget divisible by 160")
    target_rows = changed_budget // MAX_PACKET_WORDS
    candidates_by_source: dict[str, list[Row]] = collections.defaultdict(list)
    protected: list[Row] = []
    noncandidate: list[Row] = []
    for r in base_rows:
        if r.source == preserve_source:
            protected.append(r)
        elif r.words == MAX_PACKET_WORDS:
            candidates_by_source[r.source].append(r)
        else:
            noncandidate.append(r)
    counts = {s: len(v) for s, v in candidates_by_source.items()}
    take_counts = proportional_counts(counts, target_rows)
    # Repair rounding if a tiny source is over-asked.
    for s, n in list(take_counts.items()):
        if n > counts[s]:
            excess = n - counts[s]
            take_counts[s] = counts[s]
            for t in sorted(counts, key=lambda x: counts[x] - take_counts.get(x, 0), reverse=True):
                room = counts[t] - take_counts.get(t, 0)
                add = min(room, excess)
                take_counts[t] = take_counts.get(t, 0) + add
                excess -= add
                if excess == 0:
                    break
            if excess:
                raise RuntimeError("cannot allocate row holdout counts")
    rng = random.Random(seed)
    heldout_ids: set[int] = set()
    heldout: list[Row] = []
    for s, rows in candidates_by_source.items():
        order = list(rows)
        rng.shuffle(order)
        take = take_counts.get(s, 0)
        chosen = order[:take]
        heldout.extend(chosen)
        heldout_ids.update(id(r) for r in chosen)
    if len(heldout) != target_rows:
        raise RuntimeError(f"heldout rows {len(heldout)} != target {target_rows}")
    remaining = [r for r in base_rows if id(r) not in heldout_ids]
    if sum(r.words for r in heldout) != changed_budget:
        raise RuntimeError("heldout word total mismatch")
    if sum(r.words for r in remaining) != TOTAL_WORDS - changed_budget:
        raise RuntimeError("remaining word total mismatch")
    rng.shuffle(heldout)
    rng.shuffle(remaining)
    meta = {
        "target_rows": target_rows,
        "preserved_source": preserve_source,
        "protected_rows": len(protected),
        "protected_words": sum(r.words for r in protected),
        "noncandidate_rows": len(noncandidate),
        "candidate_rows_by_source": counts,
        "heldout_rows_by_source": dict(collections.Counter(r.source for r in heldout).most_common()),
        "heldout_words_by_source": dict(collections.Counter({}).most_common()),
    }
    wc_by_source: collections.Counter[str] = collections.Counter()
    for r in heldout:
        wc_by_source[r.source] += r.words
    meta["heldout_words_by_source"] = dict(wc_by_source.most_common())
    return heldout, remaining, meta


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
        seg = stream[pos:pos+L]
        words = [w for w, _ in seg]
        source_counts = collections.Counter(s for _, s in seg)
        common = source_counts.most_common(1)[0][0] if source_counts else "base"
        rows.append(Row(text=" ".join(words), words=L, example_id=example_base + i,
                        source=f"{label_prefix}::{common}", component_sources=dict(source_counts)))
        pos += L
    return rows


def write_pool(path: pathlib.Path, rows: list[Row], meta_path: pathlib.Path | None = None) -> None:
    write_jsonl(path, [row_to_record(r) for r in rows])
    if meta_path is not None:
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


def source_word_counts(rows: list[Row]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c.most_common())


def stat(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(vals)
    return {"n": len(xs), "min": xs[0], "mean": statistics.fmean(xs), "median": statistics.median(xs), "max": xs[-1], "sum": sum(xs)}


def pair_quality_summary(pairs: list[Pair], changed_budget: int, neutral_topup: int = 0) -> dict[str, Any]:
    src = sum(p.source_words for p in pairs)
    rew = sum(p.rewrite_words for p in pairs)
    return {
        "pairs": len(pairs),
        "source_words": src,
        "rewrite_words": rew,
        "pair_words": src + rew,
        "neutral_cleanqwen_topup_words_inside_changed_block": neutral_topup,
        "changed_block_budget_words": changed_budget,
        "unique_docs": len({p.doc_id for p in pairs}),
        "rewrite_to_source_ratio_weighted": rew / max(1, src),
        "rewrite_to_source_ratio_stats": stat([p.rewrite_words / max(1, p.source_words) for p in pairs]),
        "content_recall_stats": stat([p.content_recall for p in pairs if p.content_recall is not None]),
        "entity_recall_stats": stat([p.entity_recall for p in pairs if p.entity_recall is not None]),
        "number_recall_stats": stat([p.number_recall for p in pairs if p.number_recall is not None]),
        "domain_hit_counts": dict(collections.Counter(d for p in pairs for d in (p.domain_hits or ["no_domain"])).most_common()),
        "near_copy_like_soft_count": sum(1 for p in pairs if p.soft_flags and "near_copy_view" in p.soft_flags),
    }


def build_family(family_name: str, clean_name: str, repeat_name: str, view_name: str,
                 pair_list: list[Pair], changed_budget: int, topup_words: int,
                 holdout_stream: list[tuple[str, str]], filler: list[Row], example_base: int) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    view_rows = pack_pair_rows(pair_list, use_rewrite=True, source_label=view_name, example_base=example_base)
    repeat_rows = pack_pair_rows(pair_list, use_rewrite=False, source_label=repeat_name, example_base=example_base)
    pair_lengths = [r.words for r in view_rows]
    topup_lengths: list[int] = []
    rem = topup_words
    while rem > 0:
        L = min(MAX_PACKET_WORDS, rem)
        topup_lengths.append(L)
        rem -= L
    topup = stream_rows(holdout_stream, topup_lengths, f"neutral_cleanqwen_topup_{family_name}", example_base + 100000)
    changed_lengths = pair_lengths + topup_lengths
    clean_changed = stream_rows(holdout_stream, changed_lengths, f"cleanqwen_lengthmatched_{family_name}", example_base + 200000)
    arms = {
        clean_name: clean_changed + filler,
        repeat_name: repeat_rows + topup + filler,
        view_name: view_rows + topup + filler,
    }
    length_sequences = {name: tuple(r.words for r in rows) for name, rows in arms.items()}
    if len(set(length_sequences.values())) != 1:
        raise RuntimeError(f"row length mismatch in {family_name}")
    for name, rows in arms.items():
        total = sum(r.words for r in rows)
        if total != TOTAL_WORDS:
            raise RuntimeError(f"{name} total {total}")
    meta = {
        "family_name": family_name,
        "arm_names": {"cleanqwen_lengthmatched": clean_name, "repeat": repeat_name, "view": view_name},
        "pair_rows": len(view_rows),
        "changed_block_rows": len(view_rows) + len(topup),
        "pair_words": sum(p.pair_words for p in pair_list),
        "neutral_cleanqwen_topup_words_inside_changed_block": topup_words,
        "changed_block_budget_words": changed_budget,
        "row_length_sequence_identical_within_family": len(set(length_sequences.values())) == 1,
        "repeat_view_suffix_identical_after_pair_rows": all(
            row_to_record(a) == row_to_record(b) for a, b in zip(arms[repeat_name][len(repeat_rows):], arms[view_name][len(view_rows):])
        ),
        "common_filler_rows": len(filler),
        "common_filler_words": sum(r.words for r in filler),
        "word_totals": {name: sum(r.words for r in rows) for name, rows in arms.items()},
        "row_counts": {name: len(rows) for name, rows in arms.items()},
    }
    return arms, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--density-metadata", default=str(DENSITY_META_DEFAULT))
    ap.add_argument("--base-pool", default=str(BASE_POOL_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--preserve-source", default=PRESERVE_SOURCE)
    ap.add_argument("--write-training", action="store_true")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    density_meta_path = pathlib.Path(args.density_metadata)
    density_meta = json.loads(density_meta_path.read_text(encoding="utf-8"))
    changed_budget = int(density_meta["changed_block_budget_words"])
    paths = density_meta["files"]
    near_core_pairs = read_pairs(pathlib.Path(paths["near_core_pairs"]))
    compact_core_pairs = read_pairs(pathlib.Path(paths["compact_core_pairs"]))
    compact_reinvest_pairs = read_pairs(pathlib.Path(paths["compact_reinvest_pairs"]))

    base_rows, base_source_counts = read_base_pool(pathlib.Path(args.base_pool))
    heldout_rows, filler, holdout_meta = choose_row_holdout(base_rows, changed_budget, args.preserve_source, args.seed)
    holdout_stream = stream_from_rows(heldout_rows)
    if len(holdout_stream) != changed_budget:
        raise RuntimeError("holdout stream word mismatch")
    if sum(r.words for r in filler) != TOTAL_WORDS - changed_budget:
        raise RuntimeError("filler word mismatch")

    near_topup = changed_budget - sum(p.pair_words for p in near_core_pairs)
    compact_core_topup = changed_budget - sum(p.pair_words for p in compact_core_pairs)
    compact_reinvest_topup = changed_budget - sum(p.pair_words for p in compact_reinvest_pairs)
    if min(near_topup, compact_core_topup, compact_reinvest_topup) < 0:
        raise RuntimeError("negative topup")

    family_specs = [
        ("near_core", "cleanqwen_lengthmatched_near_core", "cleanqwen_fineweb_repeat_near_core", "cleanqwen_fineweb_near_view_core", near_core_pairs, near_topup, 910000),
        ("compact_core_neutral", "cleanqwen_lengthmatched_compact_core_neutral", "cleanqwen_fineweb_repeat_compact_core_neutral", "cleanqwen_fineweb_compact_view_core_neutral", compact_core_pairs, compact_core_topup, 930000),
        ("compact_reinvest", "cleanqwen_lengthmatched_compact_reinvest", "cleanqwen_fineweb_repeat_compact_reinvest", "cleanqwen_fineweb_compact_view_reinvest", compact_reinvest_pairs, compact_reinvest_topup, 950000),
    ]
    arms: dict[str, list[Row]] = {}
    family_meta: dict[str, Any] = {}
    for fam, clean, rep, view, pairs, topup, base in family_specs:
        fam_arms, fam_meta = build_family(fam, clean, rep, view, pairs, changed_budget, topup, holdout_stream, filler, base)
        arms.update(fam_arms)
        family_meta[fam] = fam_meta

    pool_paths: dict[str, str] = {}
    train_paths: dict[str, str] = {}
    row_meta_paths: dict[str, str] = {}
    for name, rows in arms.items():
        p = out_dir / f"{name}_10M.jsonl"
        meta_p = out_dir / f"{name}_changed_block_rows_meta.jsonl"
        write_pool(p, rows, meta_p)
        pool_paths[name] = str(p)
        row_meta_paths[name] = str(meta_p)
        if args.write_training:
            tp = out_dir / f"{name}_100M.jsonl"
            write_training(tp, rows, args.seed + 7000)
            train_paths[name] = str(tp)

    write_jsonl(out_dir / "heldout_cleanqwen_rows.jsonl", [row_to_record(r) for r in heldout_rows])
    write_jsonl(out_dir / "common_filler_rows.jsonl", [row_to_record(r) for r in filler])

    hashes = {pathlib.Path(p).name: sha256_file(pathlib.Path(p)) for p in pool_paths.values()}
    for p in row_meta_paths.values():
        hashes[pathlib.Path(p).name] = sha256_file(pathlib.Path(p))
    for p in train_paths.values():
        hashes[pathlib.Path(p).name] = sha256_file(pathlib.Path(p))
    hashes["heldout_cleanqwen_rows.jsonl"] = sha256_file(out_dir / "heldout_cleanqwen_rows.jsonl")
    hashes["common_filler_rows.jsonl"] = sha256_file(out_dir / "common_filler_rows.jsonl")

    all_word_totals = {name: sum(r.words for r in rows) for name, rows in arms.items()}
    payload = {
        "status": "DENSITY_CLEANQWEN_ROWHOLDOUT_OVERLAY_MATERIALIZED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Map the FineWeb density contrast onto the inherited clean-Qwen paired-view base while preserving coherent non-heldout base rows and all qwen_pair_packed rows.",
        "inputs": {
            "density_metadata": str(density_meta_path),
            "density_metadata_sha256": sha256_file(density_meta_path),
            "base_pool": str(args.base_pool),
            "base_pool_sha256": sha256_file(pathlib.Path(args.base_pool)),
            "preserve_source": args.preserve_source,
        },
        "base_split": {
            "base_source_counts": base_source_counts,
            "row_holdout": holdout_meta,
            "changed_block_budget_words": changed_budget,
            "heldout_words": sum(r.words for r in heldout_rows),
            "heldout_rows": len(heldout_rows),
            "common_filler_words": sum(r.words for r in filler),
            "common_filler_rows": len(filler),
            "common_filler_source_counts": source_word_counts(filler),
        },
        "changed_block_budget_words": changed_budget,
        "total_words_per_pool": TOTAL_WORDS,
        "passes": PASSES,
        "write_training": bool(args.write_training),
        "families": family_meta,
        "pair_summaries": {
            "near_core": pair_quality_summary(near_core_pairs, changed_budget, near_topup),
            "compact_core_neutral": pair_quality_summary(compact_core_pairs, changed_budget, compact_core_topup),
            "compact_reinvest": pair_quality_summary(compact_reinvest_pairs, changed_budget, compact_reinvest_topup),
        },
        "audit": {
            "all_pool_word_totals": all_word_totals,
            "all_exact_10M": all(v == TOTAL_WORDS for v in all_word_totals.values()),
            "common_filler_shared_by_all_arms": True,
            "all_inherited_qwen_pair_words_preserved_in_common_filler": holdout_meta["protected_words"],
            "nonheldout_base_rows_preserved_coherently": True,
            "heldout_rows_are_intact_160_word_base_rows_before_changed_block_repacking": True,
        },
        "source_word_counts_by_arm": {name: source_word_counts(rows) for name, rows in arms.items()},
        "files": {
            "pools": pool_paths,
            "training": train_paths,
            "row_meta": row_meta_paths,
            "heldout_cleanqwen_rows": str(out_dir / "heldout_cleanqwen_rows.jsonl"),
            "common_filler_rows": str(out_dir / "common_filler_rows.jsonl"),
            "note": str(args.note),
        },
        "sha256": hashes,
        "interpretation": {
            "cleanqwen_lengthmatched_control": "same inherited clean-Qwen base minus one coherent held-out official/non-Qwen row slice, with that slice repacked to the same changed-block row lengths",
            "near_view_value_at_cleanqwen_base": "cleanqwen_fineweb_near_view_core minus cleanqwen_fineweb_repeat_near_core",
            "compact_core_view_value_at_cleanqwen_base": "cleanqwen_fineweb_compact_view_core_neutral minus cleanqwen_fineweb_repeat_compact_core_neutral",
            "compact_reinvestment_value_at_cleanqwen_base": "cleanqwen_fineweb_compact_view_reinvest minus cleanqwen_fineweb_compact_view_core_neutral; repeat counterpart isolates added-source effect without generated views",
            "fineweb_substitution_vs_cleanqwen_slice": "FineWeb arms minus corresponding cleanqwen_lengthmatched family control estimate whether replacing a clean-Qwen official slice helps at an inherited second-view base.",
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_path = out_dir / "density_cleanqwen_rowholdout_overlay_metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_path = pathlib.Path(args.note)
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# research repaired FineWeb density contrast on clean-Qwen base\n\n"
        "This row-holdout construction fixes the earlier word-stream overlay risk. It preserves all inherited `qwen_pair_packed` rows and all non-heldout base rows as coherent rows, removes 1,357 intact 160-word official/non-Qwen rows (217,120 words), and uses only that held-out stream for clean lengthmatching/topup.\n\n"
        f"Changed-block budget: {changed_budget:,} words. Preserved inherited Qwen pair words: {holdout_meta['protected_words']:,}. Common filler words: {sum(r.words for r in filler):,}.\n\n"
        f"Metadata: `{meta_path}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "metadata": str(meta_path),
        "changed_block_budget_words": changed_budget,
        "heldout_rows": len(heldout_rows),
        "preserved_qwen_pair_words": holdout_meta["protected_words"],
        "common_filler_words": sum(r.words for r in filler),
        "all_exact_10M": payload["audit"]["all_exact_10M"],
        "write_training": bool(args.write_training),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
