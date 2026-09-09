#!/usr/bin/env python3
"""research: materialize split in-window controls for MAX-dose DeBERTa.

REPEAT_SPLIT and VIEW_SPLIT preserve the selected source text multiset and the
companion text multiset of the original MAX REPEAT/VIEW arms, plus the same
neutral top-up and common filler suffix. The only intended intervention is that
paired source and companion spans no longer co-occur in the same 160-word row.

Scientific fork pre-stated before training:
- If the research/14 active recurrence cost is caused by practicing an in-window
  identity relation, REPEAT_SPLIT should move close to CLEAN on nonidentical
  rewrite-conditioning gain and should lose most of the held-out copy-gain
  advantage.
- If token repetition budget alone is sufficient, REPEAT_SPLIT should retain the
  original R-C pattern: about -0.88 nats on nonoverlap rewrite gain and about
  +0.58 nats on held-out natural copy gain.
- If VIEW's positive content-conditioning requires in-window source/rewrite
  co-occurrence, VIEW_SPLIT should move close to CLEAN on nonoverlap rewrite
  gain. If ordinary cross-row paraphrase augmentation suffices, it should retain
  the original V-C pattern, about +0.81 nats on nonoverlap rewrite gain.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import dataclasses
import hashlib
import json
import pathlib
import random
import time
from typing import Any, Iterable

ROOT = _public_path('experiments/archive/relation_learning/scripts/materialize_split_inwindow_controls.py')
ROOT = _PUBLIC_ROOT

REPRESENTATION_FRONTIER_STUDIES_WS = _public_path('experiments/archive/frontier_consolidation')
FUNCTIONAL_RELATION_STUDIES_WS = _public_path('experiments/archive/relation_learning')
PAIR_PATH = _public_path('experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl')
BASE_POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json')
ORIG_VIEW_POOL_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl')
ORIG_REPEAT_POOL_10M = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl')
OUT_DIR_DEFAULT = _public_path('experiments/archive/relation_learning/data/split_inwindow_rowholdout_pools')
TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
TRAIN_SEED = 82914124 + 7000


@dataclasses.dataclass
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    doc_id: str
    domain_hits: list[str]


@dataclasses.dataclass
class Segment:
    text: str
    words: int
    pair_id: str
    segment_kind: str
    doc_id: str
    domain_hits: list[str]


@dataclasses.dataclass
class Row:
    text: str
    words: int
    example_id: int
    source: str
    pair_ids: list[str] | None = None
    component_sources: dict[str, int] | None = None
    segment_kind_counts: dict[str, int] | None = None


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


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


def norm_pair_id(d: dict[str, Any]) -> str:
    pid = str(d.get("pair_id") or "").strip()
    if pid:
        return pid
    pr = str(d.get("prompt_id") or "").strip()
    if pr:
        return f"compact:{pr}"
    return "compact:" + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def load_pairs(path: pathlib.Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            src = " ".join(str(d.get("source_text") or "").split())
            rew = " ".join(str(d.get("rewrite_text") or "").split())
            sw = int(d.get("source_words") or wc(src))
            rw = int(d.get("rewrite_words") or wc(rew))
            if sw != wc(src) or rw != wc(rew):
                raise RuntimeError(f"word mismatch for pair {norm_pair_id(d)}")
            if sw + rw > MAX_ROW_WORDS:
                raise RuntimeError(f"original pair exceeds row words {norm_pair_id(d)}")
            pairs.append(Pair(
                pair_id=norm_pair_id(d),
                source_text=src,
                rewrite_text=rew,
                source_words=sw,
                rewrite_words=rw,
                doc_id=str(d.get("doc_id") or ""),
                domain_hits=[str(x) for x in (d.get("domain_hits") or ["no_domain"])],
            ))
    if len({p.pair_id for p in pairs}) != len(pairs):
        raise RuntimeError("pair ids are not unique")
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


def pack_segments(segments: list[Segment], label: str, example_base: int) -> list[Row]:
    rows: list[Row] = []
    cur: list[str] = []
    cur_words = 0
    ids: list[str] = []
    comp: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    def flush() -> None:
        nonlocal cur, cur_words, ids, comp, kind_counts
        if not cur:
            return
        rows.append(Row(" ".join(cur), cur_words, example_base + len(rows), label, list(ids), dict(comp), dict(kind_counts)))
        cur, cur_words, ids, comp, kind_counts = [], 0, [], {}, {}
    for s in segments:
        if s.words <= 0 or s.words > MAX_ROW_WORDS:
            raise RuntimeError(f"bad segment length {s.words} for {s.pair_id}")
        if cur_words and cur_words + s.words > MAX_ROW_WORDS:
            flush()
        cur.append(s.text)
        cur_words += s.words
        ids.append(s.pair_id)
        kind_counts[s.segment_kind] = kind_counts.get(s.segment_kind, 0) + s.words
        for d in s.domain_hits or ["no_domain"]:
            comp[d] = comp.get(d, 0) + s.words
    flush()
    for r in rows:
        if r.words != wc(r.text) or r.words > MAX_ROW_WORDS:
            raise RuntimeError("bad split row")
    return rows


def row_to_record(r: Row) -> dict[str, Any]:
    return {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}


def meta_record(i: int, r: Row) -> dict[str, Any]:
    return {
        "row_index": i,
        "example_id": r.example_id,
        "words": r.words,
        "pair_ids": r.pair_ids or [],
        "component_sources": r.component_sources or {},
        "segment_kind_counts": r.segment_kind_counts or {},
    }


def read_jsonl_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_pool(path: pathlib.Path, meta_path: pathlib.Path, rows: list[Row]) -> None:
    write_jsonl(path, [row_to_record(r) for r in rows])
    write_jsonl(meta_path, [meta_record(i, r) for i, r in enumerate(rows) if (r.pair_ids or r.component_sources or r.segment_kind_counts)])


def write_training(path: pathlib.Path, rows: list[Row]) -> dict[str, Any]:
    total = 0
    n = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(TRAIN_SEED + 1000 + pass_i).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_to_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training total {total}")
    return {"rows": n * PASSES, "words": total, "sha256": sha256_file(path)}


def segment_signature(segments: list[Segment]) -> str:
    h = hashlib.sha256()
    for s in segments:
        h.update(json.dumps({"pair_id": s.pair_id, "kind": s.segment_kind, "words": s.words, "text": s.text}, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def count_pool(rows: list[Row]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "words": sum(r.words for r in rows),
        "exact_10M": sum(r.words for r in rows) == TOTAL_WORDS,
        "row_length_stats": {
            "min": min(r.words for r in rows),
            "max": max(r.words for r in rows),
            "mean": sum(r.words for r in rows) / len(rows),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--write-training", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    meta = json.loads(META.read_text(encoding="utf-8"))
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected research metadata status {meta.get('status')}")
    n_pair_rows_orig = int((meta.get("dose") or {}).get("pair_rows") or 0)
    if n_pair_rows_orig <= 0:
        raise RuntimeError("could not recover original pair row count")
    view_pool_rows = read_jsonl_rows(ORIG_VIEW_POOL_10M)
    repeat_pool_rows = read_jsonl_rows(ORIG_REPEAT_POOL_10M)
    if len(view_pool_rows) != len(repeat_pool_rows):
        raise RuntimeError("original view/repeat row counts differ")
    suffix_view = view_pool_rows[n_pair_rows_orig:]
    suffix_repeat = repeat_pool_rows[n_pair_rows_orig:]
    if suffix_view != suffix_repeat:
        raise RuntimeError("original view/repeat suffix differs after changed block")
    suffix_rows = [Row(" ".join(str(r["text"]).split()), int(r["words"]), int(r.get("example_id", 2_000_000+i)), str(r.get("source") or "suffix")) for i, r in enumerate(suffix_view)]
    suffix_words = sum(r.words for r in suffix_rows)

    pairs = load_pairs(PAIR_PATH)
    source_segments: list[Segment] = []
    view_segments: list[Segment] = []
    repeat_segments: list[Segment] = []
    for p in pairs:
        source_segments.append(Segment(p.source_text, p.source_words, p.pair_id, "source_only", p.doc_id, p.domain_hits))
        view_segments.append(Segment(p.rewrite_text, p.rewrite_words, p.pair_id, "view_companion_only", p.doc_id, p.domain_hits))
        rep = repeat_source_words(p.source_text, p.rewrite_words, p.pair_id)
        if wc(rep) != p.rewrite_words:
            raise RuntimeError(f"repeat fragment length mismatch {p.pair_id}")
        repeat_segments.append(Segment(rep, p.rewrite_words, p.pair_id, "repeat_companion_only", p.doc_id, p.domain_hits))

    source_rows = pack_segments(source_segments, "compact_split_source_dose2p64x", 1_270_000)
    view_companion_rows = pack_segments(view_segments, "compact_view_split_companion_dose2p64x", 1_370_000)
    repeat_companion_rows = pack_segments(repeat_segments, "compact_repeat_split_companion_dose2p64x", 1_470_000)

    view_split = source_rows + view_companion_rows + suffix_rows
    repeat_split = source_rows + repeat_companion_rows + suffix_rows
    if [r.words for r in view_split] != [r.words for r in repeat_split]:
        raise RuntimeError("split arm row length sequences differ")

    files: dict[str, str] = {}
    hashes: dict[str, str] = {}
    train: dict[str, Any] = {}
    for name, rows in [
        ("compact_view_split_dose2p64x", view_split),
        ("compact_repeat_split_dose2p64x", repeat_split),
    ]:
        pool_p = out_dir / f"{name}_10M.jsonl"
        meta_p = out_dir / f"{name}_changed_block_rows_meta.jsonl"
        write_pool(pool_p, meta_p, rows)
        files[f"{name}_10M"] = rel(pool_p)
        files[f"{name}_row_meta"] = rel(meta_p)
        hashes[pool_p.name] = sha256_file(pool_p)
        hashes[meta_p.name] = sha256_file(meta_p)
        if args.write_training:
            tr_p = out_dir / f"{name}_100M.jsonl"
            train[name] = write_training(tr_p, rows)
            files[f"{name}_100M"] = rel(tr_p)
            hashes[tr_p.name] = train[name]["sha256"]

    audit = {
        "pair_count": len(pairs),
        "original_pair_rows": n_pair_rows_orig,
        "source_split_rows": len(source_rows),
        "view_companion_split_rows": len(view_companion_rows),
        "repeat_companion_split_rows": len(repeat_companion_rows),
        "suffix_rows_reused": len(suffix_rows),
        "source_words": sum(s.words for s in source_segments),
        "view_companion_words": sum(s.words for s in view_segments),
        "repeat_companion_words": sum(s.words for s in repeat_segments),
        "suffix_words": suffix_words,
        "view_split_pool": count_pool(view_split),
        "repeat_split_pool": count_pool(repeat_split),
        "split_arms_row_length_sequence_identical": [r.words for r in view_split] == [r.words for r in repeat_split],
        "source_and_companion_never_same_row_by_construction": True,
        "topup_and_filler_suffix_exactly_reused_from_step256_view": True,
        "training_order_seed": TRAIN_SEED,
        "training_100M_written": bool(args.write_training),
    }
    payload = {
        "status": "SPLIT_INWINDOW_CONTROLS_MATERIALIZED",
        "created_utc": now(),
        "scientific_purpose": "Remove source-companion co-occurrence while preserving selected text content and exposure budget for the MAX VIEW/REPEAT mechanism arms.",
        "pre_state_predictions": {
            "repeat_split_inwindow_relation": "If in-window identity practice causes the active recurrence cost, Rsplit-CLEAN token-nonoverlap rewrite gain should move near 0 rather than the original DeBERTa R-C mean about -0.88 nats, and Rsplit-CLEAN held-out copy gain should lose most of the original +0.58 nats.",
            "repeat_split_budget_only": "If repeated token budget alone causes the effects, Rsplit-CLEAN should remain near original R-C: roughly -0.88 nats nonoverlap rewrite gain and +0.58 nats copy gain.",
            "view_split_inwindow_relation": "If VIEW's content-conditioning benefit requires source/rewrite co-occurrence, Vsplit-CLEAN nonoverlap rewrite gain should move near 0 rather than the original +0.81 nats.",
            "view_split_crossrow_augmentation": "If cross-row paraphrase augmentation suffices, Vsplit-CLEAN should remain near the original +0.81 nats nonoverlap rewrite gain.",
        },
        "inputs": {
            "pairs": rel(PAIR_PATH),
            "pairs_sha256": sha256_file(PAIR_PATH),
            "metadata": rel(META),
            "view_pool_10M": rel(ORIG_VIEW_POOL_10M),
            "repeat_pool_10M": rel(ORIG_REPEAT_POOL_10M),
        },
        "content_signatures": {
            "source_segments": segment_signature(source_segments),
            "view_companion_segments": segment_signature(view_segments),
            "repeat_companion_segments": segment_signature(repeat_segments),
        },
        "audit": audit,
        "files": files,
        "sha256": hashes,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    meta_p = out_dir / "split_inwindow_rowholdout_metadata.json"
    meta_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = [
        "# research split in-window row-holdout controls",
        "",
        "These pools remove paired source/companion co-occurrence from the MAX VIEW and REPEAT arms while preserving source and companion text exposure.",
        "",
        f"- pair count: {len(pairs):,}",
        f"- original paired changed rows: {n_pair_rows_orig:,}",
        f"- source-only split rows: {len(source_rows):,}",
        f"- view companion-only split rows: {len(view_companion_rows):,}",
        f"- repeat companion-only split rows: {len(repeat_companion_rows):,}",
        f"- suffix rows reused from research: {len(suffix_rows):,}",
        f"- both pools exact 10M words: {audit['view_split_pool']['exact_10M'] and audit['repeat_split_pool']['exact_10M']}",
        f"- split-arm row length sequences identical: {audit['split_arms_row_length_sequence_identical']}",
        "",
        "Pre-stated readout: compare split arms to CLEAN on token-nonoverlap rewrite gain and held-out copy gain at 80M/90M/100M. Near-CLEAN split behavior supports same-window relation practice; original-like split behavior supports cross-row repetition/augmentation budget.",
        "",
        f"Metadata: `{rel(meta_p)}`",
    ]
    (out_dir / "split_inwindow_rowholdout_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "metadata": rel(meta_p),
        "summary": rel(out_dir / "split_inwindow_rowholdout_summary.md"),
        "audit": audit,
        "training_files": {k: files[k] for k in files if k.endswith("_100M")},
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
