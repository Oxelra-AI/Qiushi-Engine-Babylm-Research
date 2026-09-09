#!/usr/bin/env python3
"""research preservation-aware FW arm materializer.

Build compact_view and source_repeat corpora from exactly the same retained
source set after the research semantic-preservation standard has been applied.
This prevents the mechanism-scale data route from silently filling word volume
with damaged compact views.  CPU-only: no BabyLM training or official evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import random
import time
from typing import Any, Iterable

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
DEFAULT_BASE_POOL = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
DEFAULT_USABLE_PAIRS = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_standard/available_usable_pairs_for_materializer.jsonl')
DEFAULT_OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/fw_preservation_aware_arms_available')
DEFAULT_NOTE = _public_path('research/notes/representation_and_objectives/fw_preservation_aware_arm_materializer_available.md')

TOTAL_WORDS = 10_000_000
QWEN_SOURCE = "qwen_pair_packed"
MAX_ROW_WORDS = 160
RNG_SEED = 10082931
TARGET_PAIR_WORDS = 1_494_110  # research full-scale all-pair expectation under the COMPACT_EXPERIENCE-base budget.


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wc(text: str) -> int:
    return len(" ".join((text or "").split()).split())


def repeat_companion(source: str, target_words: int) -> str:
    toks = source.split()
    if target_words <= 0 or not toks:
        return ""
    out = []
    i = 0
    while len(out) < target_words:
        out.append(toks[i % len(toks)])
        i += 1
    return " ".join(out)


def select_qwen_rows_to_retain(qwen_rows: list[dict[str, Any]], target_words: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(qwen_rows)
    rng.shuffle(rows)
    kept = []
    total = 0
    for r in rows:
        w = int(r.get("words") or wc(r.get("text", "")))
        if total + w <= target_words:
            kept.append(r)
            total += w
    return kept


def build_neutral_topup(official_rows: list[dict[str, Any]], words_needed: int, seed: int, example_id: int) -> list[dict[str, Any]]:
    if words_needed <= 0:
        return []
    rng = random.Random(seed)
    rows = list(official_rows)
    rng.shuffle(rows)
    toks: list[str] = []
    for r in rows:
        toks.extend(str(r.get("text") or "").split())
        if len(toks) >= words_needed:
            break
    text = " ".join(toks[:words_needed])
    return [{"text": text, "words": wc(text), "example_id": example_id, "source": "neutral_topup_from_official"}]


def normalize_pair(row: dict[str, Any]) -> dict[str, Any]:
    source = " ".join(str(row.get("source_text") or row.get("text") or "").split())
    rewrite = " ".join(str(row.get("rewrite_text") or "").split())
    sw = wc(source)
    rw = wc(rewrite)
    if not source or not rewrite:
        raise ValueError("empty source/rewrite in usable pair")
    return {
        "norm_hash": row.get("norm_hash") or hashlib.sha256(" ".join(source.lower().split()).encode("utf-8")).hexdigest()[:32],
        "source_text": source,
        "rewrite_text": rewrite,
        "source_words": sw,
        "rewrite_words": rw,
        "pair_words": sw + rw,
        "doc_id": row.get("doc_id"),
        "domains": row.get("domains") or [],
        "source_kind": row.get("source_kind"),
        "content_recall": row.get("content_recall"),
        "content_overlap": row.get("content_overlap"),
        "soft_flags": row.get("soft_flags") or [],
    }


def trim_to_target(pairs: list[dict[str, Any]], target_pair_words: int) -> tuple[list[dict[str, Any]], int]:
    selected = []
    total = 0
    for p in pairs:
        pw = int(p["pair_words"])
        if total + pw <= target_pair_words:
            selected.append(p)
            total += pw
    return selected, total


def pack_pairs(pairs: list[dict[str, Any]], arm: str, base_eid: int) -> list[dict[str, Any]]:
    rows = []
    cur_words: list[str] = []
    cur_ids: list[str] = []
    cur_count = 0
    eid = base_eid
    for p in pairs:
        src = p["source_text"]
        if arm == "compact_view":
            comp = p["rewrite_text"]
        elif arm == "source_repeat":
            comp = repeat_companion(src, p["rewrite_words"])
        else:
            raise ValueError(arm)
        pair_text = f"{src} {comp}".strip()
        pair_wc = wc(pair_text)
        expected = p["source_words"] + p["rewrite_words"]
        if pair_wc != expected:
            raise RuntimeError(f"pair word mismatch {arm} {p['norm_hash']}: {pair_wc} != {expected}")
        if cur_words and cur_count + pair_wc > MAX_ROW_WORDS:
            rows.append({
                "text": " ".join(cur_words),
                "words": cur_count,
                "example_id": eid,
                "source": f"fw_preserved_{arm}",
                "pair_ids": cur_ids,
            })
            eid += 1
            cur_words = []
            cur_ids = []
            cur_count = 0
        cur_words.extend(pair_text.split())
        cur_ids.append(str(p["norm_hash"]))
        cur_count += pair_wc
    if cur_words:
        rows.append({
            "text": " ".join(cur_words),
            "words": cur_count,
            "example_id": eid,
            "source": f"fw_preserved_{arm}",
            "pair_ids": cur_ids,
        })
    return rows


def write_pool(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {"text": r["text"], "words": int(r["words"]), "example_id": int(r.get("example_id") or 0), "source": str(r.get("source") or "")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_shared_tokenizer_pool(official: list[dict[str, Any]], retained_qwen: list[dict[str, Any]], pairs: list[dict[str, Any]], neutral: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    parts.extend(str(r.get("text") or "") for r in official)
    parts.extend(str(r.get("text") or "") for r in retained_qwen)
    parts.extend(p["source_text"] for p in pairs)
    parts.extend(str(r.get("text") or "") for r in neutral)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-pool", default=str(DEFAULT_BASE_POOL))
    ap.add_argument("--usable-pairs", default=str(DEFAULT_USABLE_PAIRS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--target-pair-words", type=int, default=TARGET_PAIR_WORDS)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    t0 = time.time()
    base_path = pathlib.Path(args.base_pool)
    pairs_path = pathlib.Path(args.usable_pairs)
    out_dir = pathlib.Path(args.out_dir)
    note_path = pathlib.Path(args.note)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = read_jsonl(base_path)
    official_rows = [r for r in base_rows if r.get("source") != QWEN_SOURCE]
    qwen_rows = [r for r in base_rows if r.get("source") == QWEN_SOURCE]
    official_words = sum(int(r.get("words") or wc(r.get("text", ""))) for r in official_rows)
    qwen_words = sum(int(r.get("words") or wc(r.get("text", ""))) for r in qwen_rows)

    raw_pairs = read_jsonl(pairs_path)
    pairs = [normalize_pair(r) for r in raw_pairs]
    input_pair_words = sum(p["pair_words"] for p in pairs)
    trimmed = False
    if input_pair_words > args.target_pair_words:
        pairs, pair_words = trim_to_target(pairs, args.target_pair_words)
        trimmed = True
    else:
        pair_words = input_pair_words
    pair_source_words = sum(p["source_words"] for p in pairs)
    pair_rewrite_words = sum(p["rewrite_words"] for p in pairs)

    retained_qwen_target = TOTAL_WORDS - official_words - pair_words
    if retained_qwen_target < 0:
        # Very large pair set; trim enough to keep all official material.
        max_pair_words = TOTAL_WORDS - official_words
        pairs, pair_words = trim_to_target(pairs, max_pair_words)
        trimmed = True
        pair_source_words = sum(p["source_words"] for p in pairs)
        pair_rewrite_words = sum(p["rewrite_words"] for p in pairs)
        retained_qwen_target = TOTAL_WORDS - official_words - pair_words

    retained_qwen = select_qwen_rows_to_retain(qwen_rows, retained_qwen_target, args.seed)
    retained_qwen_words = sum(int(r.get("words") or wc(r.get("text", ""))) for r in retained_qwen)
    neutral_words_needed = TOTAL_WORDS - official_words - retained_qwen_words - pair_words
    neutral_rows = build_neutral_topup(official_rows, neutral_words_needed, args.seed + 17, 990000)
    neutral_words = sum(r["words"] for r in neutral_rows)

    compact_rows = pack_pairs(pairs, "compact_view", 950000)
    repeat_rows = pack_pairs(pairs, "source_repeat", 950000)
    filler = list(official_rows) + list(retained_qwen)
    arms = {}
    for name, fw_rows in [("compact_view", compact_rows), ("source_repeat", repeat_rows)]:
        pool = list(fw_rows) + list(neutral_rows) + list(filler)
        total = sum(int(r.get("words") or wc(r.get("text", ""))) for r in pool)
        arms[name] = {"rows": pool, "fw_rows": len(fw_rows), "fw_words": sum(r["words"] for r in fw_rows), "total_words": total, "n_rows": len(pool)}
        if total != TOTAL_WORDS:
            raise RuntimeError(f"{name} total words {total} != {TOTAL_WORDS}")

    for name, info in arms.items():
        write_pool(out_dir / f"fw_preserved_{name}_10M.jsonl", info["rows"])

    tok_pool = build_shared_tokenizer_pool(official_rows, retained_qwen, pairs, neutral_rows)
    tok_pool_path = out_dir / "shared_tokenizer_pool.txt"
    tok_pool_path.write_text(tok_pool, encoding="utf-8")
    tok_words = wc(tok_pool)

    by_source_kind: dict[str, int] = {}
    by_source_kind_words: dict[str, int] = {}
    for p in pairs:
        k = str(p.get("source_kind") or "unknown")
        by_source_kind[k] = by_source_kind.get(k, 0) + 1
        by_source_kind_words[k] = by_source_kind_words.get(k, 0) + p["pair_words"]

    manifest = {
        "status": "FW_PRESERVATION_AWARE_ARMS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_purpose": "Compare compact_view and source_repeat on the identical retained source set after semantic-preservation filtering; do not fill missing compact volume with damaged rewrites.",
        "base_pool": str(base_path),
        "base_pool_sha256": sha256_file(base_path),
        "usable_pairs": str(pairs_path),
        "usable_pairs_sha256": sha256_file(pairs_path),
        "target_pair_words": args.target_pair_words,
        "input_pair_words": input_pair_words,
        "selected_pair_words": pair_words,
        "yield_fraction_of_target_pair_words": pair_words / max(1, args.target_pair_words),
        "yield_short_of_target": pair_words < args.target_pair_words,
        "pair_words_missing_from_target": max(0, args.target_pair_words - pair_words),
        "trimmed_to_target": trimmed,
        "selected_pairs": len(pairs),
        "by_source_kind_pairs": by_source_kind,
        "by_source_kind_pair_words": by_source_kind_words,
        "budgets": {
            "official_words": official_words,
            "retained_qwen_words": retained_qwen_words,
            "pair_source_words": pair_source_words,
            "pair_rewrite_words": pair_rewrite_words,
            "pair_total_words": pair_words,
            "neutral_topup_words": neutral_words,
            "total_words": TOTAL_WORDS,
        },
        "arms": {name: {k: v for k, v in info.items() if k != "rows"} for name, info in arms.items()},
        "shared_tokenizer_pool_words": tok_words,
        "shared_tokenizer_pool": str(tok_pool_path),
        "retained_qwen_rows": len(retained_qwen),
        "displaced_qwen_words": qwen_words - retained_qwen_words,
        "word_totals": {
            "compact_view_fw_words": arms["compact_view"]["fw_words"],
            "source_repeat_fw_words": arms["source_repeat"]["fw_words"],
            "fw_word_totals_match": arms["compact_view"]["fw_words"] == arms["source_repeat"]["fw_words"],
        },
        "files": {
            "compact_view": str(out_dir / "fw_preserved_compact_view_10M.jsonl"),
            "source_repeat": str(out_dir / "fw_preserved_source_repeat_10M.jsonl"),
            "shared_tokenizer_pool": str(tok_pool_path),
            "manifest": str(out_dir / "fw_preservation_aware_arms_manifest.json"),
            "note": str(note_path),
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }
    manifest_path = out_dir / "fw_preservation_aware_arms_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(
        "# research preservation-aware FW arms\n\n"
        "These corpora compare compact_view and source_repeat on exactly the same retained source set after applying the research semantic-preservation standard. "
        "If selected pair words are far below the intended mechanism-scale block, the next scientific action is to improve or extend generation, not to admit damaged compact views merely to fill volume.\n\n"
        f"- Usable pairs: {len(pairs):,}\n"
        f"- Selected FineWeb pair words: {pair_words:,} / target {args.target_pair_words:,} ({manifest['yield_fraction_of_target_pair_words']:.3f})\n"
        f"- Official words preserved: {official_words:,}\n"
        f"- Retained Qwen words: {retained_qwen_words:,}\n"
        f"- Neutral topup words: {neutral_words:,}\n"
        f"- compact_view FW words: {arms['compact_view']['fw_words']:,}\n"
        f"- source_repeat FW words: {arms['source_repeat']['fw_words']:,}\n"
        f"- Shared tokenizer pool words: {tok_words:,}\n\n"
        f"Manifest: `{manifest_path}`\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": manifest["status"],
        "selected_pairs": len(pairs),
        "selected_pair_words": pair_words,
        "target_pair_words": args.target_pair_words,
        "yield_fraction": manifest["yield_fraction_of_target_pair_words"],
        "yield_short": manifest["yield_short_of_target"],
        "official_words": official_words,
        "retained_qwen_words": retained_qwen_words,
        "neutral_topup_words": neutral_words,
        "fw_word_totals_match": manifest["word_totals"]["fw_word_totals_match"],
        "shared_tokenizer_pool_words": tok_words,
        "manifest": str(manifest_path),
        "note": str(note_path),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
