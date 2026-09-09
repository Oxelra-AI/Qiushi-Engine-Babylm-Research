#!/usr/bin/env python3
"""research: materialize HALF_VIEW no-exact-copy control.

Uses the exact deterministic hash assignment from the research hash-mixed arm.
For pairs assigned "view" by that hash, keep the local source+rewrite packet.
For pairs assigned "repeat" in hash-mixed, keep the selected source but replace
the exact-copy companion slot by same-length neutral text drawn from the original
CLEAN changed-block filler. This creates a half-dose same-window VIEW relation
without a competing exact-recurrence relation, while preserving selected source
exposure, row lengths, suffix/filler, 10M/100M word budget, tokenizer coordinate,
and training order.
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
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = _public_path('experiments/archive/relation_learning/scripts/materialize_half_view_control.py')
ROOT = _PUBLIC_ROOT

REPRESENTATION_FRONTIER_STUDIES_WS = _public_path('experiments/archive/frontier_consolidation')
WS = _public_path('experiments/archive/relation_learning')
PAIR_PATH = _public_path('experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl')
POOL_DIR = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools')
META_PATH = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/dose2p64x_rowholdout_metadata.json')
VIEW_POOL = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl')
REPEAT_POOL = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl')
CLEAN_POOL = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl')
VIEW_META = _public_path('experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_changed_block_rows_meta.jsonl')
HASH_MIX_META = _public_path('experiments/archive/relation_learning/data/hash_mixed_inwindow_pools/hash_mixed_inwindow_metadata.json')
OUT_DIR_DEFAULT = _public_path('experiments/archive/relation_learning/data/half_view_control_pools')
TOTAL_WORDS = 10_000_000
PASSES = 10
MAX_ROW_WORDS = 160
STREAM_SEED = 82914124 + 7000
ASSIGN_SALT = "relation_learning_step019_hash_half_exact_half_rewrite_v1"


@dataclasses.dataclass
class Pair:
    pair_id: str
    source_text: str
    rewrite_text: str
    source_words: int
    rewrite_words: int
    domain_hits: list[str]
    content_overlap: float | None


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
    return len(str(text or "").split())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_pair_id(d: dict[str, Any]) -> str:
    pid = str(d.get("pair_id") or "").strip()
    if pid:
        return pid
    pr = str(d.get("prompt_id") or "").strip()
    if pr:
        return f"compact:{pr}"
    return "compact:" + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def load_pairs() -> dict[str, Pair]:
    pairs: dict[str, Pair] = {}
    with PAIR_PATH.open(encoding="utf-8") as f:
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
            pid = norm_pair_id(d)
            pairs[pid] = Pair(
                pair_id=pid,
                source_text=src,
                rewrite_text=rew,
                source_words=sw,
                rewrite_words=rw,
                domain_hits=[str(x) for x in (d.get("domain_hits") or ["no_domain"])],
                content_overlap=float(d["content_overlap"]) if d.get("content_overlap") is not None else None,
            )
    if len(pairs) != 33291:
        raise RuntimeError(f"unexpected pair count {len(pairs)}")
    return pairs


def assign_kind(pair_id: str) -> str:
    h = hashlib.sha256((ASSIGN_SALT + "|" + pair_id).encode("utf-8")).hexdigest()
    return "repeat" if (int(h[:16], 16) % 2 == 0) else "view"


def row_record(r: Row) -> dict[str, Any]:
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


def load_suffix(meta: dict[str, Any]) -> list[Row]:
    pair_rows = int((meta.get("dose") or {}).get("pair_rows") or 0)
    view_rows = read_jsonl(VIEW_POOL)
    repeat_rows = read_jsonl(REPEAT_POOL)
    if len(view_rows) != len(repeat_rows):
        raise RuntimeError("original VIEW/REPEAT row counts differ")
    suffix_view = view_rows[pair_rows:]
    suffix_repeat = repeat_rows[pair_rows:]
    if suffix_view != suffix_repeat:
        raise RuntimeError("VIEW/REPEAT suffix differs after changed block")
    suffix: list[Row] = []
    for i, d in enumerate(suffix_view):
        text = " ".join(str(d["text"]).split())
        suffix.append(Row(text=text, words=int(d["words"]), example_id=int(d.get("example_id", 2_000_000 + i)), source=str(d.get("source") or "suffix")))
    return suffix


def clean_changed_reservoir(pair_rows_n: int) -> list[str]:
    clean_rows = read_jsonl(CLEAN_POOL)[:pair_rows_n]
    toks: list[str] = []
    for r in clean_rows:
        text = " ".join(str(r["text"]).split())
        words = text.split()
        if len(words) != int(r["words"]):
            raise RuntimeError(f"bad CLEAN row word count {r.get('example_id')}")
        toks.extend(words)
    if len(toks) < 1000:
        raise RuntimeError("neutral CLEAN reservoir unexpectedly small")
    return toks


def neutral_fill(reservoir: list[str], cursor: int, n_words: int) -> tuple[str, int]:
    if n_words <= 0:
        return "", cursor
    out: list[str] = []
    # Reservoir is much larger than needed; wrap anyway for total safety.
    for _ in range(n_words):
        out.append(reservoir[cursor % len(reservoir)])
        cursor += 1
    return " ".join(out), cursor


def build_changed_rows(pairs: dict[str, Pair], meta_rows: list[dict[str, Any]], reservoir: list[str]) -> tuple[list[Row], dict[str, Any]]:
    rows: list[Row] = []
    assignment = Counter()
    by_row_mix = Counter()
    word_totals = Counter()
    overlap_by_kind: dict[str, list[float]] = defaultdict(list)
    examples: list[dict[str, Any]] = []
    assigned_pairs: set[str] = set()
    neutral_cursor = 0
    neutral_examples: list[dict[str, Any]] = []
    for i, m in enumerate(meta_rows):
        parts: list[str] = []
        pids = [str(x) for x in (m.get("pair_ids") or [])]
        row_kind_counts = Counter()
        domains: dict[str, int] = {}
        for pid in pids:
            p = pairs[pid]
            kind = assign_kind(pid)
            assigned_pairs.add(pid)
            if kind == "view":
                companion = p.rewrite_text
                segment_kind = "source_plus_rewrite"
                local_relation = "view"
            else:
                companion, neutral_cursor = neutral_fill(reservoir, neutral_cursor, p.rewrite_words)
                segment_kind = "source_plus_clean_neutral"
                local_relation = "neutral_no_exact"
                if len(neutral_examples) < 5:
                    neutral_examples.append({"pair_id": pid, "rewrite_words": p.rewrite_words, "neutral_prefix": companion[:120]})
            if wc(companion) != p.rewrite_words:
                raise RuntimeError(f"companion length mismatch {pid} {kind}")
            seg = f"{p.source_text} {companion}".strip()
            if wc(seg) != p.source_words + p.rewrite_words:
                raise RuntimeError(f"segment word mismatch {pid}")
            parts.append(seg)
            assignment[local_relation] += 1
            row_kind_counts[local_relation] += 1
            word_totals[segment_kind] += p.source_words + p.rewrite_words
            word_totals["selected_source_words"] += p.source_words
            if local_relation == "view":
                word_totals["view_rewrite_companion_words"] += p.rewrite_words
            else:
                word_totals["clean_neutral_companion_words"] += p.rewrite_words
            if p.content_overlap is not None:
                overlap_by_kind[local_relation].append(float(p.content_overlap))
            for d in p.domain_hits or ["no_domain"]:
                domains[d] = domains.get(d, 0) + p.source_words + p.rewrite_words
        text = " ".join(parts)
        words = wc(text)
        if words != int(m["words"]):
            raise RuntimeError(f"row word mismatch {i}: {words} vs {m['words']}")
        if words > MAX_ROW_WORDS:
            raise RuntimeError(f"row over max words {i}: {words}")
        if row_kind_counts["view"] and row_kind_counts["neutral_no_exact"]:
            by_row_mix["mixed_rows"] += 1
        elif row_kind_counts["view"]:
            by_row_mix["view_only_rows"] += 1
        elif row_kind_counts["neutral_no_exact"]:
            by_row_mix["neutral_only_rows"] += 1
        rows.append(Row(
            text=text,
            words=words,
            example_id=int(m.get("example_id", 980000 + i)),
            source="compact_half_view_noexact_dose2p64x_matched_rowholdout",
            pair_ids=pids,
            component_sources=domains,
            segment_kind_counts=dict(row_kind_counts),
        ))
        if len(examples) < 6:
            examples.append({
                "row_index": i,
                "example_id": rows[-1].example_id,
                "words": words,
                "pair_count": len(pids),
                "row_kind_counts": dict(row_kind_counts),
                "text_prefix": text[:260],
            })
    if assigned_pairs != set(pairs):
        missing = list(sorted(set(pairs) - assigned_pairs))[:5]
        raise RuntimeError(f"assigned pair set mismatch; missing {missing}")
    return rows, {
        "pair_assignment_counts": dict(assignment),
        "pair_assignment_fraction_view_relation": assignment["view"] / sum(assignment.values()),
        "row_mix_counts": dict(by_row_mix),
        "word_totals_by_relation_packet": dict(word_totals),
        "neutral_words_consumed_from_clean_changed_block": neutral_cursor,
        "clean_changed_block_reservoir_words": len(reservoir),
        "mean_content_overlap_by_hash_kind": {k: (sum(v) / len(v) if v else None) for k, v in overlap_by_kind.items()},
        "examples": examples,
        "neutral_examples": neutral_examples,
    }


def write_training(path: pathlib.Path, rows: list[Row]) -> dict[str, Any]:
    total = 0
    n = len(rows)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(STREAM_SEED + 1000 + pass_i).shuffle(order)
            for idx in order:
                r = rows[idx]
                f.write(json.dumps(row_record(r), ensure_ascii=False) + "\n")
                total += r.words
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training word total {total}")
    return {"rows": n * PASSES, "words": total, "sha256": sha256_file(path)}


def count_pool(rows: list[Row]) -> dict[str, Any]:
    words = [r.words for r in rows]
    return {"rows": len(rows), "words": sum(words), "exact_10M": sum(words) == TOTAL_WORDS, "row_length_stats": {"min": min(words), "max": max(words), "mean": sum(words) / len(words)}}


def segment_signature(rows: list[Row]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(row_record(r), ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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

    meta = read_json(META_PATH)
    if meta.get("status") != "MATCHED_MAX_ROWHOLDOUT_POOLS_MATERIALIZED":
        raise RuntimeError(f"unexpected original metadata status {meta.get('status')}")
    if not (meta.get("audit", {}).get("all_exact_10M") and meta.get("audit", {}).get("row_length_sequence_identical_all_arms") and meta.get("audit", {}).get("view_repeat_suffix_identical_after_changed_block")):
        raise RuntimeError(f"original metadata invariants failed: {meta.get('audit')}")
    hash_meta = read_json(HASH_MIX_META)
    if hash_meta.get("audit", {}).get("assignment_salt") != ASSIGN_SALT:
        raise RuntimeError("hash-mixed assignment salt mismatch")

    pairs = load_pairs()
    meta_rows_all = read_jsonl(VIEW_META)
    pair_rows_n = int((meta.get("dose") or {}).get("pair_rows") or 0)
    meta_rows = meta_rows_all[:pair_rows_n]
    if len(meta_rows) != pair_rows_n or any(not m.get("pair_ids") for m in meta_rows):
        raise RuntimeError(f"could not isolate original pair rows from {VIEW_META}")
    clean_rows = read_jsonl(CLEAN_POOL)
    if [int(r["words"]) for r in clean_rows[:pair_rows_n]] != [int(m["words"]) for m in meta_rows]:
        raise RuntimeError("CLEAN changed-block row-length sequence differs from paired changed block")
    reservoir = clean_changed_reservoir(pair_rows_n)
    changed_rows, changed_info = build_changed_rows(pairs, meta_rows, reservoir)
    suffix = load_suffix(meta)
    all_rows = changed_rows + suffix
    if [r.words for r in changed_rows] != [int(m["words"]) for m in meta_rows]:
        raise RuntimeError("changed-row length sequence not preserved")
    if sum(r.words for r in changed_rows) != int((meta.get("dose") or {}).get("selected_pair_words")):
        raise RuntimeError("selected-pair word total mismatch")
    if sum(r.words for r in all_rows) != TOTAL_WORDS:
        raise RuntimeError(f"10M total mismatch: {sum(r.words for r in all_rows)}")
    # Must match hash-mixed assignment counts exactly under the same hash.
    hm_counts = hash_meta.get("audit", {}).get("pair_assignment_counts", {})
    if int(hm_counts.get("view", -1)) != int(changed_info["pair_assignment_counts"].get("view", -2)):
        raise RuntimeError(f"view assignment count mismatch vs HM: {changed_info['pair_assignment_counts']} vs {hm_counts}")
    if int(hm_counts.get("repeat", -1)) != int(changed_info["pair_assignment_counts"].get("neutral_no_exact", -2)):
        raise RuntimeError(f"repeat/neutral assignment count mismatch vs HM: {changed_info['pair_assignment_counts']} vs {hm_counts}")

    pool_path = out_dir / "compact_half_view_noexact_dose2p64x_10M.jsonl"
    row_meta_path = out_dir / "compact_half_view_noexact_dose2p64x_changed_block_rows_meta.jsonl"
    write_jsonl(pool_path, [row_record(r) for r in all_rows])
    write_jsonl(row_meta_path, [meta_record(i, r) for i, r in enumerate(changed_rows)])
    files = {
        "compact_half_view_noexact_dose2p64x_10M": rel(pool_path),
        "compact_half_view_noexact_dose2p64x_row_meta": rel(row_meta_path),
    }
    hashes = {pool_path.name: sha256_file(pool_path), row_meta_path.name: sha256_file(row_meta_path)}
    train_rec: dict[str, Any] | None = None
    if args.write_training:
        train_path = out_dir / "compact_half_view_noexact_dose2p64x_100M.jsonl"
        train_rec = write_training(train_path, all_rows)
        files["compact_half_view_noexact_dose2p64x_100M"] = rel(train_path)
        hashes[train_path.name] = train_rec["sha256"]

    audit = {
        "pair_count": len(pairs),
        "original_pair_rows": len(meta_rows),
        "pool": count_pool(all_rows),
        "paired_relation_rows": len(changed_rows),
        "paired_relation_words": sum(r.words for r in changed_rows),
        "metadata_tail_nonpair_rows": max(0, len(meta_rows_all) - pair_rows_n),
        "suffix_rows_reused": len(suffix),
        "suffix_words_including_topup": sum(r.words for r in suffix),
        "row_length_sequence_matches_original_changed_block": [r.words for r in changed_rows] == [int(m["words"]) for m in meta_rows],
        "all_sources_retained_once": True,
        "same_window_source_companion_for_all_pairs": True,
        "exact_copy_companion_pairs": 0,
        "view_relation_pairs": int(changed_info["pair_assignment_counts"].get("view", 0)),
        "neutral_no_exact_pairs": int(changed_info["pair_assignment_counts"].get("neutral_no_exact", 0)),
        "same_hash_assignment_as_hash_mixed": True,
        "assignment_salt": ASSIGN_SALT,
        "stream_seed": STREAM_SEED,
        "training_100M_written": bool(args.write_training),
        **changed_info,
    }
    payload = {
        "status": "HALF_VIEW_CONTROL_MATERIALIZED",
        "created_utc": now(),
        "scientific_purpose": "Half-dose VIEW relation without exact-recurrence competition: distinguish hash-mixed dose curve from active relation competition under the same fixed 100M budget.",
        "pre_state_predictions": {
            "dose_curve": "If HALF_VIEW matches HM on compact true-source restatement/readout, the HM shortfall relative to full VIEW is explained by half VIEW relation dose.",
            "active_exact_relation_competition": "If HALF_VIEW approaches full VIEW while HM remains lower, exact local recurrence actively suppresses restatement source use when both relations share the diet.",
            "scope": "This is a one-seed composition boundary test, not a prerequisite for the two-seed DeBERTa relation-locality result.",
        },
        "inputs": {
            "pairs": rel(PAIR_PATH),
            "pairs_sha256": sha256_file(PAIR_PATH),
            "original_metadata": rel(META_PATH),
            "original_view_pool": rel(VIEW_POOL),
            "original_repeat_pool": rel(REPEAT_POOL),
            "original_clean_pool": rel(CLEAN_POOL),
            "original_view_row_meta": rel(VIEW_META),
            "hash_mixed_metadata": rel(HASH_MIX_META),
        },
        "audit": audit,
        "files": files,
        "sha256": hashes,
        "training": train_rec,
        "changed_block_signature": segment_signature(changed_rows),
        "pool_signature": segment_signature(all_rows),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    metadata_path = out_dir / "half_view_control_metadata.json"
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_lines = [
        "# research HALF_VIEW no-exact-copy control",
        "",
        "This stream keeps the same deterministic hash assignment as the research hash-mixed arm. Hash-view pairs keep source+rewrite in the same window. Hash-repeat pairs keep the selected source but replace the exact-copy companion with same-length neutral text from CLEAN's matched changed-block filler. Thus the stream is half-dose local VIEW without practiced exact recurrence in the other half.",
        "",
        f"- pairs: {len(pairs):,}",
        f"- local VIEW pairs: {audit['view_relation_pairs']:,}",
        f"- source+neutral no-exact pairs: {audit['neutral_no_exact_pairs']:,}",
        f"- exact-copy companion pairs: {audit['exact_copy_companion_pairs']:,}",
        f"- neutral words consumed from CLEAN changed block: {audit['neutral_words_consumed_from_clean_changed_block']:,} / {audit['clean_changed_block_reservoir_words']:,}",
        f"- paired relation rows: {len(changed_rows):,}; suffix/top-up rows: {len(suffix):,}",
        f"- exact 10M pool: {audit['pool']['exact_10M']}",
        f"- row length sequence matches original changed block: {audit['row_length_sequence_matches_original_changed_block']}",
        f"- training stream written: {bool(train_rec)}",
        "",
        "Pre-stated readout: score the same compact rewrite T/U(/N if needed), natural-copy, and Entity cue probes used for HM. Matching HM supports a dose account; approaching full VIEW supports active exact-recurrence competition against restatement source use. The result is auxiliary for relation composition, not required for the central locality result.",
        "",
        f"Metadata: `{rel(metadata_path)}`",
    ]
    note_path = out_dir / "half_view_control_summary.md"
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "metadata": rel(metadata_path),
        "summary": rel(note_path),
        "audit": audit,
        "training_file": files.get("compact_half_view_noexact_dose2p64x_100M"),
        "training_sha_prefix": (train_rec or {}).get("sha256", "")[:12] if train_rec else None,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
