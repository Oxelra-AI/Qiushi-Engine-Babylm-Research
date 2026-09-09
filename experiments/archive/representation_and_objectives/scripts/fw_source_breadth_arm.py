#!/usr/bin/env python3
"""research: construct the FineWeb source-breadth arm for the FW mechanism comparison.

The scientific comparison is compact views versus
additional independent FineWeb experience.  This script keeps the research
compact arm's base material, retained-Qwen rows, shared 16k tokenizer, selected
source set, FineWeb block word count, row count, and FineWeb row word sequence.
It replaces only the compact rewrite companion words with independently selected
FineWeb source text.

No training and no official evaluation are run here.  Optional stream writing
materializes aligned 100M streams for later training.
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

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
USABLE_PAIRS = ROOT / "data/fw_full_preservation/full26k_usable_pairs_for_materializer.jsonl"
COMPACT_ARM = ROOT / "data/fw_full_arms/fw_preserved_compact_view_10M.jsonl"
SHARED_TOK = ROOT / "data/shared_tokenizer/shared_16k_tokenizer"
SHARED_TOK_MANIFEST = ROOT / "data/shared_tokenizer/shared_tokenizer_manifest.json"
OUT_DIR = ROOT / "data/fw_source_breadth_arm"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/fw_source_breadth_arm.md')

MAX_ROW_WORDS = 160
TOTAL_WORDS = 10_000_000
PASSES = 10
TARGET_SEED = 10282931
TRAIN_STREAM_SEED = 10289931
FW_COMPACT_LABEL = "fw_preserved_compact_view"
FW_BREADTH_LABEL = "fw_preserved_source_breadth"

# High-quality FineWeb reservoirs collected in earlier steps.  Selection is
# independent of the BabyLM evaluation text; overlap is measured later only as a
# provenance safety measurement.
CANDIDATE_PATHS: list[tuple[str, pathlib.Path, int]] = [
    ("stratified", ROOT / "data/stratified_live_fineweb_blueprint/live_fineweb_stratified_current_seed_300k.jsonl", 0),
    ("v4_high_precision", ROOT / "data/live_fineweb_selector_v4_high_precision/live_fineweb_selector_v4_high_precision_kept_sorted.jsonl", 1),
    ("v5_strictstable", ROOT / "data/live_fineweb_selector_v5_strictstable/live_fineweb_selector_v5_kept_sorted.jsonl", 2),
    ("core_fact_doccap8", ROOT / "data/live_fineweb_core_scale_probe/live_fineweb_core_fact_neardedup_doccap8.jsonl", 3),
    ("priority_unique", ROOT / "data/fineweb_source_pool_audit/fineweb_priority_unique_sources.jsonl", 4),
    ("v3_stable_doccap8", ROOT / "data/live_fineweb_source_selector_v3/live_fineweb_selector_v3_stable_doccap8.jsonl", 5),
    ("quality_v2_doccap8", ROOT / "data/live_fineweb_quality_tiers/live_fineweb_quality_v2_doccap8.jsonl", 6),
    ("strict_anchor_core_true", ROOT / "data/live_fineweb_core_scale_probe/live_fineweb_strict_anchor_like_all.jsonl", 7),
]


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def wc(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def norm_hash(text: str) -> str:
    return hashlib.sha256(" ".join(norm_text(text).lower().split()).encode("utf-8")).hexdigest()[:32]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_pairs() -> list[dict[str, Any]]:
    rows = read_jsonl(USABLE_PAIRS)
    pairs = []
    for r in rows:
        src = norm_text(r.get("source_text") or "")
        rw = norm_text(r.get("rewrite_text") or "")
        if not src or not rw:
            raise RuntimeError(f"empty source/rewrite in usable pair {r.get('norm_hash')}")
        sw = int(r.get("source_words") or wc(src))
        rw_words = int(r.get("rewrite_words") or wc(rw))
        if sw != wc(src) or rw_words != wc(rw):
            raise RuntimeError(f"word mismatch in usable pair {r.get('norm_hash')}")
        pairs.append({
            "norm_hash": r.get("norm_hash") or norm_hash(src),
            "source_text": src,
            "rewrite_text": rw,
            "source_words": sw,
            "rewrite_words": rw_words,
            "pair_words": sw + rw_words,
            "doc_id": r.get("doc_id"),
            "domains": r.get("domains") or [],
            "source_kind": r.get("source_kind") or "unknown",
        })
    return pairs


def pack_pair_metas(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur_sources: list[str] = []
    cur_ids: list[str] = []
    cur_source_words = 0
    cur_companion_words = 0
    cur_total = 0
    eid = 950000
    for p in pairs:
        pw = p["pair_words"]
        if cur_sources and cur_total + pw > MAX_ROW_WORDS:
            rows.append({
                "example_id": eid,
                "pair_ids": cur_ids,
                "source_text": " ".join(cur_sources),
                "source_words": cur_source_words,
                "companion_words": cur_companion_words,
                "total_words": cur_total,
            })
            eid += 1
            cur_sources = []
            cur_ids = []
            cur_source_words = 0
            cur_companion_words = 0
            cur_total = 0
        cur_sources.append(p["source_text"])
        cur_ids.append(str(p["norm_hash"]))
        cur_source_words += int(p["source_words"])
        cur_companion_words += int(p["rewrite_words"])
        cur_total += int(pw)
    if cur_sources:
        rows.append({
            "example_id": eid,
            "pair_ids": cur_ids,
            "source_text": " ".join(cur_sources),
            "source_words": cur_source_words,
            "companion_words": cur_companion_words,
            "total_words": cur_total,
        })
    return rows


def load_compact_arm() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(COMPACT_ARM)
    fw = [r for r in rows if r.get("source") == FW_COMPACT_LABEL]
    filler = [r for r in rows if r.get("source") != FW_COMPACT_LABEL]
    return fw, filler


def primary_domain(domains: Iterable[str]) -> str:
    ds = [str(d) for d in domains if str(d).strip()]
    if not ds:
        return "no_domain"
    joined = " ".join(ds).lower()
    if "science" in joined or "physical" in joined or "object" in joined:
        return "science"
    if "geo" in joined or "spatial" in joined or "place" in joined:
        return "geography"
    if "quant" in joined or "numeric" in joined or "date" in joined:
        return "quant"
    if "history" in joined or "society" in joined or "institution" in joined or "social" in joined:
        return "history_society"
    if "causal" in joined or "relation" in joined or "temporal" in joined:
        return "relational"
    return ds[0]


def row_domains(r: dict[str, Any]) -> list[str]:
    for key in ["domains", "domain_hits", "core_domain_hits", "types"]:
        v = r.get(key)
        if isinstance(v, list) and v:
            return [str(x) for x in v]
    focus = r.get("focus")
    if focus:
        return [str(focus)]
    flags = r.get("flags")
    if isinstance(flags, dict):
        return [k for k, v in flags.items() if v]
    return ["no_domain"]


def source_text_from_row(r: dict[str, Any]) -> str:
    for key in ["source_text", "text", "sentence_text", "core_clean_text", "selector_v4_text", "selector_v3_repaired_text"]:
        val = r.get(key)
        if isinstance(val, str) and val.strip():
            return norm_text(val)
    return ""


def source_words_from_row(r: dict[str, Any], text: str) -> int:
    for key in ["words", "source_words", "selector_v4_words", "selector_v3_words"]:
        if key in r:
            try:
                w = int(r[key])
                if w == wc(text):
                    return w
            except Exception:
                pass
    return wc(text)


def looks_usable_source(r: dict[str, Any], text: str, words: int, pool_name: str) -> bool:
    if words < 8 or words > 80:
        return False
    if len(text) < 25:
        return False
    alpha = sum(ch.isalpha() for ch in text) / max(1, len(text))
    if alpha < 0.55:
        return False
    low = text.lower()
    if any(bad in low for bad in ["click here", "read more", "copyright", "all rights reserved", "subscribe", "sign up"]):
        return False
    # The broad strict-anchor file contains rejected examples too; keep only rows
    # already marked as core facts when that annotation is available.
    if pool_name == "strict_anchor_core_true" and r.get("core_fact_accepted") is not True:
        return False
    rejects = r.get("core_reject_reasons") or r.get("selector_v4_reject_reasons") or r.get("selector_v3_reject_reasons")
    if isinstance(rejects, list) and rejects and pool_name in {"strict_anchor_core_true"}:
        return False
    return True


def load_candidate_sources(exclude_hashes: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[str] = set(exclude_hashes)
    candidates: list[dict[str, Any]] = []
    source_counts: dict[str, dict[str, Any]] = {}
    for pool_name, path, priority in CANDIDATE_PATHS:
        rec = {"path": str(path), "exists": path.exists(), "read_rows": 0, "accepted_rows": 0, "accepted_words": 0}
        if not path.exists():
            source_counts[pool_name] = rec
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec["read_rows"] += 1
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                text = source_text_from_row(r)
                if not text:
                    continue
                w = source_words_from_row(r, text)
                h = norm_hash(text)
                if h in seen:
                    continue
                if not looks_usable_source(r, text, w, pool_name):
                    continue
                seen.add(h)
                domains = row_domains(r)
                doc_id = str(r.get("doc_id") or r.get("document_id") or "")
                qscore = float(r.get("selection_score") or r.get("core_content_score") or r.get("core_relation_count") or 0.0)
                candidates.append({
                    "norm_hash": h,
                    "text": text,
                    "words": w,
                    "doc_id": doc_id,
                    "domains": domains,
                    "primary_domain": primary_domain(domains),
                    "source_pool": pool_name,
                    "source_path": str(path),
                    "source_priority": priority,
                    "quality_score": qscore,
                    "sentence_index": r.get("sentence_index"),
                })
                rec["accepted_rows"] += 1
                rec["accepted_words"] += w
        source_counts[pool_name] = rec
    return candidates, source_counts


def target_domain_words(pairs: list[dict[str, Any]]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for p in pairs:
        c[primary_domain(p.get("domains") or [])] += int(p["rewrite_words"])
    return dict(c)


def subset_sum_indices(cands: list[dict[str, Any]], target: int, limit: int = 5000) -> list[int] | None:
    if target == 0:
        return []
    if target < 0:
        return None
    # Prefer a bounded candidate tail for speed; all word lengths are small.
    usable = [(i, int(c["words"])) for i, c in enumerate(cands[:limit]) if int(c["words"]) <= target]
    parent: dict[int, tuple[int, int]] = {0: (-1, -1)}
    for i, w in usable:
        for s in list(parent.keys())[::-1]:
            ns = s + w
            if ns > target or ns in parent:
                continue
            parent[ns] = (s, i)
            if ns == target:
                out: list[int] = []
                cur = target
                while cur:
                    prev, idx = parent[cur]
                    out.append(idx)
                    cur = prev
                out.reverse()
                return out
    return None


def choose_sources(candidates: list[dict[str, Any]], target_words: int, domain_targets: dict[str, int], seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    # Stable but not corpus-order dominated: priority first, then relation/content
    # score, with a tiny deterministic random key for tie breaking.
    for c in candidates:
        c["_rand"] = rng.random()
    candidates = sorted(candidates, key=lambda c: (int(c["source_priority"]), -float(c["quality_score"]), c["_rand"]))
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for c in candidates:
        groups[str(c["primary_domain"])].append(c)
    ptr = {d: 0 for d in groups}
    doc_cap = 6
    doc_counts: collections.Counter[str] = collections.Counter()
    selected: list[dict[str, Any]] = []
    selected_words = 0
    selected_by_domain: collections.Counter[str] = collections.Counter()
    remaining_pool: list[dict[str, Any]] = []

    def next_candidate(domain: str, remaining: int) -> dict[str, Any] | None:
        arr = groups.get(domain) or []
        i = ptr.get(domain, 0)
        while i < len(arr):
            c = arr[i]
            i += 1
            ptr[domain] = i
            doc = c.get("doc_id") or f"docless::{c['norm_hash']}"
            if doc_counts[str(doc)] >= doc_cap:
                remaining_pool.append(c)
                continue
            if int(c["words"]) > remaining:
                remaining_pool.append(c)
                continue
            return c
        ptr[domain] = i
        return None

    # Greedily satisfy the compact companion domain mix, stopping with a small
    # remainder that exact full-sentence subset search can solve.
    while target_words - selected_words > 500:
        remaining = target_words - selected_words
        live_domains = [d for d, arr in groups.items() if ptr.get(d, 0) < len(arr)]
        if not live_domains:
            break
        def deficit(d: str) -> float:
            target = domain_targets.get(d, 0)
            if target <= 0:
                target = max(1, int(0.02 * target_words))
            return target - selected_by_domain.get(d, 0)
        live_domains.sort(key=lambda d: (deficit(d), len(groups[d]) - ptr[d]), reverse=True)
        chosen = None
        for d in live_domains:
            chosen = next_candidate(d, remaining)
            if chosen is not None:
                break
        if chosen is None:
            break
        selected.append(chosen)
        selected_words += int(chosen["words"])
        selected_by_domain[str(chosen["primary_domain"])] += int(chosen["words"])
        doc_counts[str(chosen.get("doc_id") or f"docless::{chosen['norm_hash']}")] += 1

    # Collect still-available candidates for exact remainder repair.
    unused: list[dict[str, Any]] = []
    selected_hashes = {c["norm_hash"] for c in selected}
    for c in candidates:
        if c["norm_hash"] in selected_hashes:
            continue
        doc = str(c.get("doc_id") or f"docless::{c['norm_hash']}")
        if doc_counts[doc] >= doc_cap:
            continue
        unused.append(c)

    rem = target_words - selected_words
    exact_strategy = "greedy_exact"
    if rem > 0:
        idxs = subset_sum_indices(unused, rem, limit=min(6000, len(unused)))
        if idxs is not None:
            for idx in idxs:
                c = unused[idx]
                selected.append(c)
                selected_words += int(c["words"])
                selected_by_domain[str(c["primary_domain"])] += int(c["words"])
                doc_counts[str(c.get("doc_id") or f"docless::{c['norm_hash']}")] += 1
            exact_strategy = "tail_subset_full_sentences"
        else:
            repaired = False
            # Remove one recently selected sentence and replace it plus the
            # small remainder with a full-sentence subset.
            for remove_pos in range(len(selected) - 1, max(-1, len(selected) - 200), -1):
                removed = selected[remove_pos]
                repair_target = rem + int(removed["words"])
                repair_pool = [removed] + unused[:8000]
                idxs = subset_sum_indices(repair_pool, repair_target, limit=len(repair_pool))
                if idxs is not None:
                    doc = str(removed.get("doc_id") or f"docless::{removed['norm_hash']}")
                    selected_by_domain[str(removed["primary_domain"])] -= int(removed["words"])
                    doc_counts[doc] -= 1
                    selected.pop(remove_pos)
                    selected_words -= int(removed["words"])
                    used_hashes = {x["norm_hash"] for x in selected}
                    for idx in idxs:
                        c = repair_pool[idx]
                        if c["norm_hash"] in used_hashes:
                            continue
                        selected.append(c)
                        selected_words += int(c["words"])
                        selected_by_domain[str(c["primary_domain"])] += int(c["words"])
                        doc_counts[str(c.get("doc_id") or f"docless::{c['norm_hash']}")] += 1
                        used_hashes.add(c["norm_hash"])
                    exact_strategy = "one_sentence_repair_full_sentences"
                    repaired = True
                    break
            if not repaired:
                # Last resort: use a prefix from one unused source.  Record it
                # explicitly; the current cached pools have enough full-sentence
                # combinations in normal runs, so this should rarely happen.
                rem = target_words - selected_words
                for c in unused:
                    if int(c["words"]) > rem:
                        toks = c["text"].split()[:rem]
                        prefix = dict(c)
                        prefix["text"] = " ".join(toks)
                        prefix["words"] = rem
                        prefix["norm_hash"] = norm_hash(prefix["text"])
                        prefix["partial_prefix_from"] = c["norm_hash"]
                        selected.append(prefix)
                        selected_words += rem
                        selected_by_domain[str(prefix["primary_domain"])] += rem
                        exact_strategy = "single_prefix_completion"
                        repaired = True
                        break
                if not repaired:
                    raise RuntimeError(f"could not reach target_words={target_words}, selected={selected_words}, rem={target_words-selected_words}")

    if selected_words != target_words:
        raise RuntimeError(f"selected source words {selected_words} != target {target_words}")
    summary = {
        "target_words": target_words,
        "selected_words": selected_words,
        "selected_rows": len(selected),
        "exact_strategy": exact_strategy,
        "domain_targets": domain_targets,
        "selected_words_by_domain": dict(selected_by_domain),
        "selected_rows_by_pool": dict(collections.Counter(c["source_pool"] for c in selected)),
        "selected_words_by_pool": dict(collections.Counter({})),
        "doc_cap": doc_cap,
        "unique_docs": len({str(c.get("doc_id")) for c in selected}),
    }
    pool_words: collections.Counter[str] = collections.Counter()
    for c in selected:
        pool_words[str(c["source_pool"])] += int(c["words"])
    summary["selected_words_by_pool"] = dict(pool_words)
    return selected, summary


def fill_breadth_rows(row_metas: list[dict[str, Any]], selected_sources: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stream_words: list[str] = []
    source_boundaries: list[tuple[int, int, str]] = []
    cursor = 0
    for src in selected_sources:
        toks = src["text"].split()
        start = cursor
        stream_words.extend(toks)
        cursor += len(toks)
        source_boundaries.append((start, cursor, str(src["norm_hash"])))
    total_companion = sum(int(m["companion_words"]) for m in row_metas)
    if len(stream_words) != total_companion:
        raise RuntimeError(f"companion stream words {len(stream_words)} != {total_companion}")

    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    pos = 0
    boundary_i = 0
    for m in row_metas:
        cw = int(m["companion_words"])
        companion = " ".join(stream_words[pos:pos + cw])
        text = norm_text(f"{m['source_text']} {companion}")
        if wc(text) != int(m["total_words"]):
            raise RuntimeError(f"row word mismatch at {m['example_id']}")
        rows.append({
            "text": text,
            "words": int(m["total_words"]),
            "example_id": int(m["example_id"]),
            "source": FW_BREADTH_LABEL,
        })
        # Track which independent source sentences contribute to this row.
        start_pos, end_pos = pos, pos + cw
        used_ids: list[str] = []
        while boundary_i < len(source_boundaries) and source_boundaries[boundary_i][1] <= start_pos:
            boundary_i += 1
        j = boundary_i
        while j < len(source_boundaries) and source_boundaries[j][0] < end_pos:
            used_ids.append(source_boundaries[j][2])
            j += 1
        meta_rows.append({
            "example_id": int(m["example_id"]),
            "pair_ids": m["pair_ids"],
            "common_source_words": int(m["source_words"]),
            "breadth_companion_words": cw,
            "total_words": int(m["total_words"]),
            "breadth_source_ids": used_ids,
            "breadth_word_start": start_pos,
            "breadth_word_end": end_pos,
        })
        pos += cw
    if pos != len(stream_words):
        raise RuntimeError("not all breadth companion words consumed")
    return rows, meta_rows


def pool_word_count(rows: Iterable[dict[str, Any]]) -> int:
    total = 0
    for r in rows:
        w = int(r.get("words") or wc(r.get("text", "")))
        if w != wc(str(r.get("text") or "")):
            raise RuntimeError(f"row word field mismatch source={r.get('source')} example_id={r.get('example_id')}")
        total += w
    return total


def write_training_stream(path: pathlib.Path, rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    n = len(rows)
    total = 0
    order_hashes: list[str] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pass_i in range(PASSES):
            order = list(range(n))
            random.Random(seed + 1000 + pass_i).shuffle(order)
            order_hashes.append(sha256_text(",".join(map(str, order))))
            for idx in order:
                r = rows[idx]
                rec = {"text": r["text"], "words": int(r["words"]), "example_id": int(r.get("example_id") or 0), "source": str(r.get("source") or "")}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += int(r["words"])
    if total != TOTAL_WORDS * PASSES:
        raise RuntimeError(f"training stream total {total} != {TOTAL_WORDS * PASSES}")
    return {
        "path": str(path),
        "rows": n * PASSES,
        "words": total,
        "sha256": sha256_file(path),
        "pass_order_hashes": order_hashes,
        "seed": seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-training-streams", action="store_true")
    ap.add_argument("--seed", type=int, default=TARGET_SEED)
    ap.add_argument("--stream-seed", type=int, default=TRAIN_STREAM_SEED)
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    row_metas = pack_pair_metas(pairs)
    compact_fw, filler_rows = load_compact_arm()
    if len(compact_fw) != len(row_metas):
        raise RuntimeError(f"FW row count mismatch: compact={len(compact_fw)} metas={len(row_metas)}")
    compact_words = [int(r["words"]) for r in compact_fw]
    meta_words = [int(m["total_words"]) for m in row_metas]
    if compact_words != meta_words:
        raise RuntimeError("FineWeb row word sequence does not match compact arm")

    pair_source_words = sum(int(p["source_words"]) for p in pairs)
    compact_companion_words = sum(int(p["rewrite_words"]) for p in pairs)
    pair_words = pair_source_words + compact_companion_words
    filler_words = pool_word_count(filler_rows)
    if filler_words + pair_words != TOTAL_WORDS:
        raise RuntimeError(f"filler+pair words {filler_words}+{pair_words} != {TOTAL_WORDS}")

    exclude = {str(p["norm_hash"]) for p in pairs}
    candidates, candidate_summary = load_candidate_sources(exclude)
    domain_targets = target_domain_words(pairs)
    selected_sources, selection_summary = choose_sources(candidates, compact_companion_words, domain_targets, args.seed)

    breadth_fw_rows, breadth_row_meta = fill_breadth_rows(row_metas, selected_sources)
    breadth_pool = list(breadth_fw_rows) + list(filler_rows)
    if len(breadth_pool) != len(compact_fw) + len(filler_rows):
        raise RuntimeError("row count changed")
    if [int(r["words"]) for r in breadth_pool] != [int(r["words"]) for r in compact_fw + filler_rows]:
        raise RuntimeError("complete arm row word sequence changed")
    breadth_total = pool_word_count(breadth_pool)
    if breadth_total != TOTAL_WORDS:
        raise RuntimeError(f"breadth total {breadth_total} != {TOTAL_WORDS}")

    # Save the new 10M arm and sidecars.
    arm_path = OUT_DIR / "fw_preserved_source_breadth_10M.jsonl"
    selected_path = OUT_DIR / "source_breadth_companion_sources.jsonl"
    row_meta_path = OUT_DIR / "source_breadth_row_meta.jsonl"
    write_jsonl(arm_path, breadth_pool)
    write_jsonl(selected_path, selected_sources)
    write_jsonl(row_meta_path, breadth_row_meta)

    stream_records: dict[str, Any] = {}
    if args.write_training_streams:
        compact_rows = compact_fw + filler_rows
        compact_stream = OUT_DIR / "fw_preserved_compact_view_100M.jsonl"
        breadth_stream = OUT_DIR / "fw_preserved_source_breadth_100M.jsonl"
        compact_stream_rec = write_training_stream(compact_stream, compact_rows, args.stream_seed)
        breadth_stream_rec = write_training_stream(breadth_stream, breadth_pool, args.stream_seed)
        if compact_stream_rec["pass_order_hashes"] != breadth_stream_rec["pass_order_hashes"]:
            raise RuntimeError("training pass orders differ")
        stream_records = {"compact_view": compact_stream_rec, "source_breadth": breadth_stream_rec}

    source_kind_counts = collections.Counter(str(p.get("source_kind") or "unknown") for p in pairs)
    source_kind_words = collections.Counter()
    for p in pairs:
        source_kind_words[str(p.get("source_kind") or "unknown")] += int(p["pair_words"])

    manifest = {
        "status": "FW_SOURCE_BREADTH_ARM_READY",
        "created_utc": now_utc(),
        "scientific_purpose": "Compare source-aligned compact views against the same word budget spent on independent FineWeb source breadth under the research shared tokenizer.",
        "inputs": {
            "usable_pairs": str(USABLE_PAIRS),
            "usable_pairs_sha256": sha256_file(USABLE_PAIRS),
            "compact_arm": str(COMPACT_ARM),
            "compact_arm_sha256": sha256_file(COMPACT_ARM),
            "shared_tokenizer": str(SHARED_TOK),
            "shared_tokenizer_manifest": str(SHARED_TOK_MANIFEST),
            "shared_tokenizer_manifest_sha256": sha256_file(SHARED_TOK_MANIFEST),
        },
        "budgets": {
            "official_qwen_neutral_filler_words_reused": filler_words,
            "common_fineweb_source_words": pair_source_words,
            "compact_rewrite_words_replaced_by_breadth": compact_companion_words,
            "fineweb_block_words": pair_words,
            "total_words": breadth_total,
            "rows_total": len(breadth_pool),
            "fineweb_rows": len(breadth_fw_rows),
            "row_word_sequence_matches_compact": True,
            "filler_rows_reused": len(filler_rows),
        },
        "pair_set": {
            "pairs": len(pairs),
            "pair_source_words": pair_source_words,
            "pair_rewrite_words": compact_companion_words,
            "pair_total_words": pair_words,
            "source_kind_pairs": dict(source_kind_counts),
            "source_kind_pair_words": dict(source_kind_words),
        },
        "candidate_sources": {
            "total_candidates_after_excluding_pair_sources": len(candidates),
            "total_candidate_words": sum(int(c["words"]) for c in candidates),
            "by_input_pool": candidate_summary,
        },
        "selected_breadth_sources": selection_summary,
        "files": {
            "source_breadth_10m": str(arm_path),
            "source_breadth_selected_sources": str(selected_path),
            "source_breadth_row_meta": str(row_meta_path),
            "manifest": str(OUT_DIR / "fw_source_breadth_arm_manifest.json"),
            "note": str(NOTE),
        },
        "training_streams": stream_records,
        "hashes": {
            "source_breadth_10m": sha256_file(arm_path),
            "selected_sources": sha256_file(selected_path),
            "row_meta": sha256_file(row_meta_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    manifest_path = OUT_DIR / "fw_source_breadth_arm_manifest.json"
    manifest["files"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text(
        "# research — FineWeb source-breadth arm\n\n"
        "The research compact arm is compared against an independent FineWeb source-breadth arm rather than literal repetition as the first expensive data test. "
        "The breadth arm keeps the same base filler rows, the same selected first-source spans, the same 813,005-word FineWeb block, the same row word sequence, and the same shared 16k tokenizer; only the 318,851 compact rewrite words are replaced by independently selected FineWeb source words.\n\n"
        f"- Common FineWeb source words retained: {pair_source_words:,}\n"
        f"- Compact rewrite words replaced by breadth: {compact_companion_words:,}\n"
        f"- FineWeb block words: {pair_words:,}\n"
        f"- Reused filler words: {filler_words:,}\n"
        f"- Total arm words: {breadth_total:,}\n"
        f"- Selected independent breadth sources: {selection_summary['selected_rows']:,} rows / {selection_summary['selected_words']:,} words, exact strategy `{selection_summary['exact_strategy']}`\n"
        f"- Row word sequence matches compact arm: yes ({len(breadth_pool):,} rows)\n"
        f"- Shared tokenizer: `{SHARED_TOK}`\n\n"
        f"Manifest: `{manifest_path}`\n"
        f"10M breadth arm: `{arm_path}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": manifest["status"],
        "fineweb_block_words": pair_words,
        "common_source_words": pair_source_words,
        "breadth_companion_words": compact_companion_words,
        "selected_breadth_sources": selection_summary["selected_rows"],
        "total_words": breadth_total,
        "rows_total": len(breadth_pool),
        "row_word_sequence_matches_compact": True,
        "training_streams_written": bool(stream_records),
        "manifest": str(manifest_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
