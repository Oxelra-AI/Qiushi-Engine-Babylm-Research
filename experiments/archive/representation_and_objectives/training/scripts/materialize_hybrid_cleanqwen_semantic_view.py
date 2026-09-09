#!/usr/bin/env python3
"""Materialize a clean-Qwen + semantic-view hybrid candidate corpus.

Scientific purpose
------------------
This CPU/file step prepares, but does not train, the next possible data route if
the running semantic-view packet-local contrast shows a real positive signal.  It
combines two independently verified mechanisms under exact BabyLM Strict-Small
word accounting:

  * COMPACT_EXPERIENCE clean-Qwen aligned official-source rewrite packets, the strongest local
    data coordinate so far under the protected 8x480/baseline16k/WWM recipe.
  * REPRESENTATION_FRONTIER_STUDIES capped SimpleWiki semantic-view packets, already matched against a
    packet-local source-only control.

The script writes two matched hybrid arms:

  1. hybrid_cleanqwen_semantic_view:
       COMPACT_EXPERIENCE clean-Qwen pair rows + REPRESENTATION_FRONTIER_STUDIES semantic-view packets + official filler.
  2. hybrid_cleanqwen_packet_local:
       the exact same clean-Qwen pair rows + REPRESENTATION_FRONTIER_STUDIES packet-local source-only rows
       + the exact same official filler.

Thus any difference between the two hybrid arms is the incremental generated
SimpleWiki simplification/paraphrase view effect on top of the clean-Qwen data
coordinate, not a difference in clean-Qwen pairs, official filler, row lengths,
training exposure, tokenizer, or model recipe.  It is not a broad-factual-data
solution by itself; it is a launch-ready checked candidate contingent on the
basic semantic-view evidence.
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
from dataclasses import dataclass
from typing import Iterable, Any

TOTAL_WORDS = 10_000_000
PASSES = 10
DEFAULT_BASE_QWEN = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
DEFAULT_BASE_META = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json")
DEFAULT_SELECTED_PAIRS = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl")
DEFAULT_SEM_DIR = pathlib.Path("experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/hybrid_cleanqwen_semantic_view")


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_key(text: str) -> str:
    return hashlib.sha256(norm_text(text).lower().encode("utf-8")).hexdigest()


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]
    return {
        "n": len(xs),
        "min": min(xs),
        "p05": q(0.05),
        "mean": round(statistics.mean(xs), 4),
        "median": statistics.median(xs),
        "p95": q(0.95),
        "max": max(xs),
        "sum": sum(xs),
    }


def load_rows(path: pathlib.Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            o = json.loads(line)
            text = norm_text(o["text"])
            words = int(o.get("words", len(text.split())))
            if words != len(text.split()):
                raise RuntimeError(f"{path} row {i} word mismatch: meta={words} actual={len(text.split())}")
            rows.append(Row(text=text, words=words, source=str(o.get("source", "")), example_id=int(o.get("example_id", i))))
    return rows


def choose_filler(rows: list[Row], needed_words: int) -> tuple[list[Row], dict[str, Any]]:
    chosen: list[Row] = []
    total = 0
    for r in rows:
        if total >= needed_words:
            break
        remain = needed_words - total
        if r.words <= remain:
            chosen.append(Row(text=r.text, words=r.words, source=r.source, example_id=r.example_id))
            total += r.words
        else:
            ws = r.text.split()[:remain]
            chosen.append(Row(text=" ".join(ws), words=remain, source=f"{r.source}::partial_hybrid_filler", example_id=950000 + len(chosen)))
            total += remain
            break
    if total != needed_words:
        raise RuntimeError(f"filler unavailable: needed {needed_words}, got {total}")
    return chosen, {
        "needed_words": needed_words,
        "rows": len(chosen),
        "partial_last_row": bool(chosen and chosen[-1].source.endswith("::partial_hybrid_filler")),
        "last_row_words": chosen[-1].words if chosen else 0,
    }


def source_words(rows: Iterable[Row]) -> dict[str, int]:
    c = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c)


def write_pool(path: pathlib.Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.words != len(r.text.split()):
                raise RuntimeError(f"word mismatch before write: {path} example_id={r.example_id}")
            f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[Row], pass_orders: list[list[int]]) -> int:
    total = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                f.write(json.dumps({"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}, ensure_ascii=False) + "\n")
                total += r.words
    expected = sum(r.words for r in rows) * len(pass_orders)
    if total != expected:
        raise RuntimeError(f"training exposure mismatch {total} != {expected}")
    return total


def load_jsonl_dicts(path: pathlib.Path, limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-qwen", default=str(DEFAULT_BASE_QWEN))
    ap.add_argument("--base-meta", default=str(DEFAULT_BASE_META))
    ap.add_argument("--selected-pairs", default=str(DEFAULT_SELECTED_PAIRS))
    ap.add_argument("--semantic-dir", default=str(DEFAULT_SEM_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--total-words", type=int, default=TOTAL_WORDS)
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--seed", type=int, default=70707)
    ap.add_argument("--write-training", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    base_qwen_path = pathlib.Path(args.base_qwen)
    sem_dir = pathlib.Path(args.semantic_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_rows(base_qwen_path)
    if sum(r.words for r in base_rows) != args.total_words:
        raise RuntimeError(f"base qwen pool is not {args.total_words} words")
    qwen_pair_rows = [r for r in base_rows if r.source == "qwen_pair_packed"]
    base_filler_rows = [r for r in base_rows if r.source != "qwen_pair_packed"]
    qwen_pair_words = sum(r.words for r in qwen_pair_rows)
    base_filler_words = sum(r.words for r in base_filler_rows)
    if qwen_pair_words <= 0:
        raise RuntimeError("no qwen_pair_packed rows found in base clean-Qwen pool")

    sem_treat_all = load_rows(sem_dir / "semantic_view_treatment_10M.jsonl")
    sem_ctrl_all = load_rows(sem_dir / "original_packet_local_10M.jsonl")
    sem_treat = [r for r in sem_treat_all if r.source == "simplewiki_semantic_view"]
    sem_ctrl = [r for r in sem_ctrl_all if r.source == "simplewiki_original_packet_local"]
    if len(sem_treat) != len(sem_ctrl):
        raise RuntimeError("semantic treatment/control prefix row count mismatch")
    if [r.words for r in sem_treat] != [r.words for r in sem_ctrl]:
        raise RuntimeError("semantic treatment/control prefix row lengths mismatch")
    sem_words = sum(r.words for r in sem_treat)

    filler_needed = args.total_words - qwen_pair_words - sem_words
    if filler_needed <= 0:
        raise RuntimeError(f"hybrid prefix exceeds total words: qwen={qwen_pair_words} semantic={sem_words}")
    filler, filler_meta = choose_filler(base_filler_rows, filler_needed)

    # Preserve row identity blocks, then use identical pass shuffles for training exposure.
    treat_pool = qwen_pair_rows + sem_treat + filler
    ctrl_pool = qwen_pair_rows + sem_ctrl + filler
    if sum(r.words for r in treat_pool) != args.total_words or sum(r.words for r in ctrl_pool) != args.total_words:
        raise RuntimeError("hybrid pool word total mismatch")
    if [r.words for r in treat_pool] != [r.words for r in ctrl_pool]:
        raise RuntimeError("hybrid row length sequence mismatch")
    qn = len(qwen_pair_rows)
    sn = len(sem_treat)
    identical_qwen_block = all(treat_pool[i].text == ctrl_pool[i].text and treat_pool[i].source == ctrl_pool[i].source for i in range(qn))
    identical_filler_block = all(treat_pool[qn+sn+i].text == ctrl_pool[qn+sn+i].text and treat_pool[qn+sn+i].source == ctrl_pool[qn+sn+i].source for i in range(len(filler)))
    semantic_words_matched = sum(r.words for r in sem_treat) == sum(r.words for r in sem_ctrl)

    rng = random.Random(args.seed)
    base_indices = list(range(len(treat_pool)))
    pass_orders: list[list[int]] = []
    for _ in range(args.passes):
        order = list(base_indices)
        rng.shuffle(order)
        pass_orders.append(order)

    paths = {
        "hybrid_treatment_pool": out_dir / "hybrid_cleanqwen_semantic_view_10M.jsonl",
        "hybrid_control_pool": out_dir / "hybrid_cleanqwen_packet_local_10M.jsonl",
        "hybrid_treatment_training": out_dir / "hybrid_cleanqwen_semantic_view_100M.jsonl",
        "hybrid_control_training": out_dir / "hybrid_cleanqwen_packet_local_100M.jsonl",
        "metadata": out_dir / "hybrid_materialization_metadata.json",
        "audit": out_dir / "hybrid_materialization_audit.json",
        "note": pathlib.Path("research/notes/representation_and_objectives/hybrid_cleanqwen_semantic_view_candidate.md"),
        "pass_manifest": out_dir / "pass_order_manifest.json",
    }

    write_pool(paths["hybrid_treatment_pool"], treat_pool)
    write_pool(paths["hybrid_control_pool"], ctrl_pool)
    training_exposure = None
    if args.write_training:
        exp1 = write_training(paths["hybrid_treatment_training"], treat_pool, pass_orders)
        exp2 = write_training(paths["hybrid_control_training"], ctrl_pool, pass_orders)
        if exp1 != exp2:
            raise RuntimeError("training exposure differs between hybrid arms")
        training_exposure = exp1
    paths["pass_manifest"].write_text(json.dumps({
        "seed": args.seed,
        "passes": args.passes,
        "rows_per_pool": len(treat_pool),
        "same_order_used_for_treatment_and_control": True,
        "orders_sha256": hashlib.sha256(json.dumps(pass_orders, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }, indent=2) + "\n", encoding="utf-8")

    base_meta = json.loads(pathlib.Path(args.base_meta).read_text(encoding="utf-8"))
    sem_meta = json.loads((sem_dir / "semantic_view_materialization_metadata.json").read_text(encoding="utf-8"))
    sem_audit = json.loads((sem_dir / "semantic_view_contrast_audit.json").read_text(encoding="utf-8"))
    sem_visibility = json.loads((sem_dir / "semantic_view_seq256_visibility_audit.json").read_text(encoding="utf-8"))
    selected_pairs = load_jsonl_dicts(pathlib.Path(args.selected_pairs))
    selected_orig_keys = {text_key(p.get("original", "")) for p in selected_pairs}
    sem_meta_rows = load_jsonl_dicts(sem_dir / "semantic_packet_rows_meta.jsonl")
    sem_source_keys_as_text = {text_key(r.get("source_text", "")) for r in sem_meta_rows if r.get("source_text")}
    # `semantic_packet_rows_meta.jsonl` intentionally does not carry source_text in the compact file;
    # overlap will be zero/unknown unless a richer meta is used later.  Keep the field explicit.
    exact_overlap_selected_pair_originals_vs_semantic_meta_source_text = len(selected_orig_keys & sem_source_keys_as_text)

    audit = {
        "status": "HYBRID_CLEANQWEN_SEMANTIC_VIEW_AUDITED",
        "total_words_per_pool": args.total_words,
        "passes": args.passes,
        "write_training": bool(args.write_training),
        "training_exposure_words": training_exposure,
        "hybrid_row_length_sequence_identical": [r.words for r in treat_pool] == [r.words for r in ctrl_pool],
        "qwen_pair_block_identical": identical_qwen_block,
        "official_filler_block_identical": identical_filler_block,
        "semantic_packet_word_totals_matched": semantic_words_matched,
        "qwen_pair_rows": len(qwen_pair_rows),
        "qwen_pair_words": qwen_pair_words,
        "semantic_packet_rows": len(sem_treat),
        "semantic_packet_words": sem_words,
        "base_qwen_filler_words_available": base_filler_words,
        "hybrid_filler": filler_meta,
        "hybrid_rows": {"treatment": len(treat_pool), "control": len(ctrl_pool)},
        "prefix_word_fraction": round((qwen_pair_words + sem_words) / args.total_words, 6),
        "qwen_pair_word_fraction": round(qwen_pair_words / args.total_words, 6),
        "semantic_word_fraction": round(sem_words / args.total_words, 6),
        "row_stats": {
            "qwen_pair_rows": stats([r.words for r in qwen_pair_rows]),
            "semantic_rows": stats([r.words for r in sem_treat]),
            "filler_rows": stats([r.words for r in filler]),
            "hybrid_treatment_rows": stats([r.words for r in treat_pool]),
        },
        "source_word_counts_treatment": source_words(treat_pool),
        "source_word_counts_control": source_words(ctrl_pool),
        "inherited_base_cleanqwen": {
            "metadata": str(args.base_meta),
            "status": base_meta.get("status"),
            "selected_pair_words": base_meta.get("selected_pair_words"),
            "selected_pair_word_fraction": base_meta.get("selected_pair_word_fraction"),
            "qwen_pair_rows": base_meta.get("qwen_pair_rows"),
            "pair_boundary_preserved": base_meta.get("pair_boundary_preserved"),
        },
        "semantic_source_candidate": {
            "metadata": str(sem_dir / "semantic_view_materialization_metadata.json"),
            "audit": str(sem_dir / "semantic_view_contrast_audit.json"),
            "visibility": str(sem_dir / "semantic_view_seq256_visibility_audit.json"),
            "accepted_views_used": sem_meta.get("accepted_views_used"),
            "semantic_packet_rows": sem_meta.get("semantic_packet_rows"),
            "semantic_packet_words": sem_meta.get("semantic_packet_words"),
            "packet_local_deep_ok": sem_audit.get("deep_prefix_construction_check", {}).get("ok"),
            "views_not_visible_seq256": sem_visibility.get("views_not_visible"),
            "view_visible_token_fraction_overall": sem_visibility.get("view_visible_token_fraction_overall"),
        },
        "overlap_probe": {
            "selected_pair_originals": len(selected_orig_keys),
            "semantic_meta_source_text_keys_available": len(sem_source_keys_as_text),
            "exact_overlap_selected_pair_originals_vs_semantic_meta_source_text": exact_overlap_selected_pair_originals_vs_semantic_meta_source_text,
            "note": "Compact semantic_packet_rows_meta lacks source_text, so this overlap probe is only informative if a richer semantic meta is used later.",
        },
    }
    required = {
        "hybrid_row_length_sequence_identical": audit["hybrid_row_length_sequence_identical"],
        "qwen_pair_block_identical": audit["qwen_pair_block_identical"],
        "official_filler_block_identical": audit["official_filler_block_identical"],
        "semantic_packet_word_totals_matched": audit["semantic_packet_word_totals_matched"],
        "total_words_treatment": sum(r.words for r in treat_pool) == args.total_words,
        "total_words_control": sum(r.words for r in ctrl_pool) == args.total_words,
        "semantic_packet_local_source_check_inherited": bool(audit["semantic_source_candidate"]["packet_local_deep_ok"]),
        "semantic_view_visibility_inherited": audit["semantic_source_candidate"].get("views_not_visible_seq256") == 0,
    }
    audit["required_checks"] = required
    audit["all_required_checks_pass"] = all(required.values())
    if not audit["all_required_checks_pass"]:
        raise RuntimeError(f"hybrid audit failed: {required}")

    metadata = {
        "status": "HYBRID_CLEANQWEN_SEMANTIC_VIEW_MATERIALIZED",
        "created_utc_unix": int(time.time()),
        "purpose": "candidate matched hybrid: clean-Qwen all-source pairs plus capped SimpleWiki semantic-view packets versus clean-Qwen plus packet-local same-source repetition",
        "causal_interpretation": "incremental generated SimpleWiki simplification/paraphrase variation on top of COMPACT_EXPERIENCE clean-Qwen, conditional on later matched training/evaluation; not evidence until trained and compared",
        "inputs": {
            "base_qwen_pool": str(base_qwen_path),
            "base_meta": str(args.base_meta),
            "selected_pairs": str(args.selected_pairs),
            "semantic_dir": str(sem_dir),
        },
        "outputs": {k: str(v) for k, v in paths.items()},
        "audit": audit,
        "sha256": {k: sha256_file(v) for k, v in paths.items() if v.exists() and v.is_file() and k not in {"note"}},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paths["audit"].write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research hybrid clean-Qwen + semantic-view candidate\n\n"]
    lines.append("This is a checked candidate corpus, not trained evidence. It should only be trained if the running packet-local semantic-view contrast gives a meaningful positive signal.\n\n")
    lines.append("## Construction\n\n")
    lines.append(f"- Clean-Qwen pair rows inherited from COMPACT_EXPERIENCE: {len(qwen_pair_rows):,} rows, {qwen_pair_words:,} words ({qwen_pair_words/args.total_words:.2%}).\n")
    lines.append(f"- REPRESENTATION_FRONTIER_STUDIES semantic packet rows: {len(sem_treat):,} rows, {sem_words:,} words ({sem_words/args.total_words:.2%}).\n")
    lines.append(f"- Combined generated/paired prefix: {qwen_pair_words + sem_words:,} words ({(qwen_pair_words + sem_words)/args.total_words:.2%}).\n")
    lines.append(f"- Official filler from the clean-Qwen filler pool: {filler_needed:,} words in {len(filler):,} rows; partial last row: {filler_meta['partial_last_row']}.\n")
    lines.append("- Treatment and control have identical clean-Qwen rows, identical official filler, identical row-length sequence, and matched SimpleWiki packet word totals.\n\n")
    lines.append("## Use\n\n")
    lines.append("If the basic semantic-view treatment beats its packet-local control in official-compatible no-AoA evaluation without damaging key columns, this hybrid pair can test whether the mechanism adds on top of the stronger COMPACT_EXPERIENCE clean-Qwen coordinate. If the basic contrast is weak or negative, do not spend H100 time on this hybrid; instead reopen a broader verifiable factual-source route.\n\n")
    lines.append(f"Audit JSON: `{paths['audit']}`\n\n")
    lines.append(f"Metadata JSON: `{paths['metadata']}`\n")
    paths["note"].parent.mkdir(parents=True, exist_ok=True)
    paths["note"].write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": metadata["status"],
        "out_dir": str(out_dir),
        "qwen_pair_words": qwen_pair_words,
        "semantic_packet_words": sem_words,
        "prefix_word_fraction": audit["prefix_word_fraction"],
        "filler_words": filler_needed,
        "rows": audit["hybrid_rows"],
        "write_training": bool(args.write_training),
        "training_exposure_words": training_exposure,
        "all_required_checks_pass": audit["all_required_checks_pass"],
        "metadata": str(paths["metadata"]),
        "audit": str(paths["audit"]),
        "note": str(paths["note"]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
