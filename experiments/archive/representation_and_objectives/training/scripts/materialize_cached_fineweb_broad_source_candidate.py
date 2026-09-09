#!/usr/bin/env python3
"""Materialize a cached-FineWeb broad-source data arm on top of clean-Qwen.

Purpose
-------
If the running same-source semantic-view contrast is weak, the next plausible
data lever is broader factual source distribution rather than more views of the
same SimpleWiki rows.  Fresh FineWeb-Edu streaming is currently unavailable in
representation_and_objectives, but initial_model_studies preserves a public HuggingFaceFW/fineweb-edu sample-10BT
random-quality 3M-word corpus.  This script turns that cached public substrate
into a transparent, matched, non-trained 10M data pair:

  treatment: COMPACT_EXPERIENCE clean-Qwen pair rows + cached FineWeb-Edu random-quality rows
             + identical official filler tail.
  control:   same COMPACT_EXPERIENCE clean-Qwen pair rows + official BabyLM filler text
             chunked to the exact FineWeb row-length sequence
             + the same official filler tail.

Thus a later training pair would test broad public factual source replacement on
top of the strongest local clean-Qwen coordinate, not generated same-source
paraphrases.  The script does not train by default.
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
from dataclasses import dataclass, field
from typing import Any, Iterable

TOTAL_WORDS = 10_000_000
PASSES = 10
DEFAULT_QWEN_POOL = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl")
DEFAULT_QWEN_META = pathlib.Path("experiments/archive/compact_experience/data/qwen_clean_aligned/clean_materialization_metadata.json")
DEFAULT_FINEWEB = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl")
DEFAULT_FINEWEB_META = pathlib.Path("experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_relation_materialization_meta.json")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/cached_fineweb_broad_source_candidate")
DEFAULT_NOTE = pathlib.Path("research/notes/representation_and_objectives/cached_fineweb_broad_source_candidate.md")


@dataclass
class Row:
    text: str
    words: int
    source: str
    example_id: int
    meta: dict[str, Any] = field(default_factory=dict)


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = list(vals)
    if not xs:
        return {"n": 0}
    ys = sorted(xs)
    def q(p: float):
        return ys[min(len(ys) - 1, max(0, round((len(ys) - 1) * p)))]
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


def load_rows(path: pathlib.Path, source_override: str | None = None, id_offset: int = 0) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = norm_text(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"{path} row {i} word mismatch: meta={words} actual={actual}")
            meta = {k: v for k, v in obj.items() if k not in {"text", "words", "source", "example_id"}}
            rows.append(Row(
                text=text,
                words=words,
                source=source_override or str(obj.get("source", "")),
                example_id=int(obj.get("example_id", i)) + id_offset,
                meta=meta,
            ))
    return rows


@dataclass
class StreamState:
    rows: list[Row]
    row_i: int = 0
    word_i: int = 0

    def take_words(self, n: int) -> tuple[list[str], collections.Counter[str], list[dict[str, Any]]]:
        out_words: list[str] = []
        comp: collections.Counter[str] = collections.Counter()
        segs: list[dict[str, Any]] = []
        while len(out_words) < n:
            if self.row_i >= len(self.rows):
                raise RuntimeError(f"stream exhausted while taking {n} words; have {len(out_words)}")
            r = self.rows[self.row_i]
            words = r.text.split()
            remain_row = len(words) - self.word_i
            need = n - len(out_words)
            take = min(need, remain_row)
            if take <= 0:
                self.row_i += 1
                self.word_i = 0
                continue
            out_words.extend(words[self.word_i:self.word_i + take])
            comp[r.source] += take
            segs.append({"source": r.source, "example_id": r.example_id, "start_word": self.word_i, "words": take})
            self.word_i += take
            if self.word_i >= len(words):
                self.row_i += 1
                self.word_i = 0
        return out_words, comp, segs

    def consumed_words(self) -> int:
        total = sum(r.words for r in self.rows[:self.row_i])
        return total + self.word_i


def chunk_stream_to_lengths(state: StreamState, lengths: list[int], source: str, start_example_id: int) -> tuple[list[Row], dict[str, Any]]:
    rows: list[Row] = []
    component_total: collections.Counter[str] = collections.Counter()
    multi_source = 0
    sample_meta = []
    for j, length in enumerate(lengths):
        ws, comp, segs = state.take_words(length)
        if len(ws) != length:
            raise RuntimeError("internal length mismatch")
        component_total.update(comp)
        if len(comp) > 1:
            multi_source += 1
        meta = {"component_sources": dict(comp)}
        if j < 12:
            meta["segments"] = segs
            sample_meta.append({"row_index": j, "words": length, "component_sources": dict(comp), "segments": segs})
        rows.append(Row(text=" ".join(ws), words=length, source=source, example_id=start_example_id + j, meta=meta))
    return rows, {
        "rows": len(rows),
        "words": sum(lengths),
        "component_sources_total": dict(component_total),
        "multi_source_rows": multi_source,
        "multi_source_row_fraction": (multi_source / len(rows)) if rows else 0.0,
        "sample_meta": sample_meta,
        "stream_consumed_words_after": state.consumed_words(),
        "stream_row_i_after": state.row_i,
        "stream_word_i_after": state.word_i,
    }


def take_tail_rows(state: StreamState, needed_words: int, source: str, start_example_id: int) -> tuple[list[Row], dict[str, Any]]:
    rows: list[Row] = []
    total = 0
    partial_rows = 0
    component_total: collections.Counter[str] = collections.Counter()
    while total < needed_words:
        remaining = needed_words - total
        # Preserve the next complete source row when it fits exactly from a row boundary.
        if state.row_i < len(state.rows) and state.word_i == 0 and state.rows[state.row_i].words <= remaining:
            r = state.rows[state.row_i]
            rows.append(Row(text=r.text, words=r.words, source=source, example_id=start_example_id + len(rows), meta={"original_source": r.source, "original_example_id": r.example_id}))
            component_total[r.source] += r.words
            total += r.words
            state.row_i += 1
            continue
        ws, comp, segs = state.take_words(remaining)
        rows.append(Row(text=" ".join(ws), words=len(ws), source=source, example_id=start_example_id + len(rows), meta={"component_sources": dict(comp), "segments": segs, "partial_tail_row": True}))
        component_total.update(comp)
        partial_rows += 1
        total += len(ws)
    if total != needed_words:
        raise RuntimeError(f"tail filler mismatch {total} != {needed_words}")
    return rows, {
        "needed_words": needed_words,
        "rows": len(rows),
        "partial_rows": partial_rows,
        "component_sources_total": dict(component_total),
        "stream_consumed_words_after": state.consumed_words(),
        "stream_row_i_after": state.row_i,
        "stream_word_i_after": state.word_i,
    }


def write_rows(path: pathlib.Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r.words != len(r.text.split()):
                raise RuntimeError(f"word mismatch before write: {path} row={r.example_id}")
            obj = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
            obj.update(r.meta)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_training(path: pathlib.Path, rows: list[Row], pass_orders: list[list[int]]) -> int:
    total = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for order in pass_orders:
            for idx in order:
                r = rows[idx]
                obj = {"text": r.text, "words": r.words, "example_id": r.example_id, "source": r.source}
                obj.update(r.meta)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += r.words
    expected = sum(r.words for r in rows) * len(pass_orders)
    if total != expected:
        raise RuntimeError(f"training exposure mismatch {total} != {expected}")
    return total


def source_words(rows: Iterable[Row]) -> dict[str, int]:
    c: collections.Counter[str] = collections.Counter()
    for r in rows:
        c[r.source] += r.words
    return dict(c)


def first_rows(rows: list[Row], n: int = 5) -> list[dict[str, Any]]:
    out = []
    for r in rows[:n]:
        out.append({
            "example_id": r.example_id,
            "source": r.source,
            "words": r.words,
            "meta": r.meta,
            "text_excerpt": r.text[:600] + ("…" if len(r.text) > 600 else ""),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-pool", default=str(DEFAULT_QWEN_POOL))
    ap.add_argument("--qwen-meta", default=str(DEFAULT_QWEN_META))
    ap.add_argument("--fineweb", default=str(DEFAULT_FINEWEB))
    ap.add_argument("--fineweb-meta", default=str(DEFAULT_FINEWEB_META))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    ap.add_argument("--total-words", type=int, default=TOTAL_WORDS)
    ap.add_argument("--passes", type=int, default=PASSES)
    ap.add_argument("--seed", type=int, default=80808)
    ap.add_argument("--write-training", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qwen_rows_all = load_rows(pathlib.Path(args.qwen_pool))
    if sum(r.words for r in qwen_rows_all) != args.total_words:
        raise RuntimeError(f"qwen pool word total is not {args.total_words}")
    qwen_pair_rows = [r for r in qwen_rows_all if r.source == "qwen_pair_packed"]
    official_filler_rows = [r for r in qwen_rows_all if r.source != "qwen_pair_packed"]
    qwen_words = sum(r.words for r in qwen_pair_rows)
    filler_words_available = sum(r.words for r in official_filler_rows)
    if qwen_words <= 0:
        raise RuntimeError("no qwen_pair_packed rows found")

    fineweb_rows = load_rows(pathlib.Path(args.fineweb), source_override="fineweb_edu_random_quality_cached_initial_model_studies", id_offset=8_000_000)
    fineweb_words = sum(r.words for r in fineweb_rows)
    if qwen_words + fineweb_words >= args.total_words:
        raise RuntimeError(f"qwen + fineweb exceeds pool: {qwen_words}+{fineweb_words}")
    if fineweb_words <= 0:
        raise RuntimeError("fineweb rows empty")

    state = StreamState(official_filler_rows)
    control_block, control_meta = chunk_stream_to_lengths(
        state, [r.words for r in fineweb_rows], "official_lengthmatched_to_cached_fineweb", 8_500_000,
    )
    tail_needed = args.total_words - qwen_words - fineweb_words
    tail_rows, tail_meta = take_tail_rows(state, tail_needed, "official_identical_tail_after_fineweb_block", 9_000_000)

    treatment = qwen_pair_rows + fineweb_rows + tail_rows
    control = qwen_pair_rows + control_block + tail_rows
    if sum(r.words for r in treatment) != args.total_words or sum(r.words for r in control) != args.total_words:
        raise RuntimeError("pool word total mismatch")
    if [r.words for r in treatment] != [r.words for r in control]:
        raise RuntimeError("row length sequence mismatch")
    qn = len(qwen_pair_rows)
    fn = len(fineweb_rows)
    qwen_identical = all(treatment[i].text == control[i].text and treatment[i].source == control[i].source for i in range(qn))
    tail_identical = all(treatment[qn+fn+i].text == control[qn+fn+i].text and treatment[qn+fn+i].source == control[qn+fn+i].source for i in range(len(tail_rows)))
    if not (qwen_identical and tail_identical):
        raise RuntimeError("shared qwen/tail identity mismatch")

    rng = random.Random(args.seed)
    indices = list(range(len(treatment)))
    pass_orders: list[list[int]] = []
    for _ in range(args.passes):
        order = list(indices)
        rng.shuffle(order)
        pass_orders.append(order)

    paths = {
        "treatment_10M": out_dir / "cleanqwen_cached_fineweb_broad_10M.jsonl",
        "control_10M": out_dir / "cleanqwen_official_lengthmatched_control_10M.jsonl",
        "treatment_100M": out_dir / "cleanqwen_cached_fineweb_broad_100M.jsonl",
        "control_100M": out_dir / "cleanqwen_official_lengthmatched_control_100M.jsonl",
        "verification": out_dir / "materialization_verification.json",
        "metadata": out_dir / "materialization_metadata.json",
        "samples": out_dir / "sample_rows.json",
        "pass_manifest": out_dir / "pass_order_manifest.json",
        "note": pathlib.Path(args.note),
    }

    write_rows(paths["treatment_10M"], treatment)
    write_rows(paths["control_10M"], control)
    training_exposure = None
    if args.write_training:
        e1 = write_training(paths["treatment_100M"], treatment, pass_orders)
        e2 = write_training(paths["control_100M"], control, pass_orders)
        if e1 != e2:
            raise RuntimeError("training exposure mismatch between arms")
        training_exposure = e1
    paths["pass_manifest"].write_text(json.dumps({
        "seed": args.seed,
        "passes": args.passes,
        "rows_per_pool": len(treatment),
        "same_order_used_for_treatment_and_control": True,
        "orders_sha256": hashlib.sha256(json.dumps(pass_orders, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    qwen_meta = json.loads(pathlib.Path(args.qwen_meta).read_text(encoding="utf-8"))
    fineweb_meta_path = pathlib.Path(args.fineweb_meta)
    fineweb_meta = json.loads(fineweb_meta_path.read_text(encoding="utf-8")) if fineweb_meta_path.exists() else None
    verification = {
        "status": "CACHED_FINEWEB_BROAD_SOURCE_CANDIDATE_VERIFIED",
        "total_words_per_pool": args.total_words,
        "passes": args.passes,
        "write_training": bool(args.write_training),
        "training_exposure_words": training_exposure,
        "rows": {"treatment": len(treatment), "control": len(control)},
        "row_length_sequence_identical": [r.words for r in treatment] == [r.words for r in control],
        "qwen_pair_block_identical": qwen_identical,
        "tail_filler_identical": tail_identical,
        "qwen_pair_rows": len(qwen_pair_rows),
        "qwen_pair_words": qwen_words,
        "fineweb_rows": len(fineweb_rows),
        "fineweb_words": fineweb_words,
        "tail_filler_words": tail_needed,
        "source_word_counts_treatment": source_words(treatment),
        "source_word_counts_control": source_words(control),
        "word_fractions": {
            "clean_qwen_pairs": round(qwen_words / args.total_words, 6),
            "cached_fineweb": round(fineweb_words / args.total_words, 6),
            "combined_non_tail_prefix": round((qwen_words + fineweb_words) / args.total_words, 6),
            "identical_official_tail": round(tail_needed / args.total_words, 6),
        },
        "row_stats": {
            "qwen_pair": stats([r.words for r in qwen_pair_rows]),
            "fineweb": stats([r.words for r in fineweb_rows]),
            "control_block": stats([r.words for r in control_block]),
            "tail": stats([r.words for r in tail_rows]),
            "pool": stats([r.words for r in treatment]),
        },
        "control_block_from_official_filler": control_meta,
        "identical_tail_from_official_filler": tail_meta,
        "inherited_clean_qwen_metadata": {
            "path": str(args.qwen_meta),
            "status": qwen_meta.get("status"),
            "selected_pair_words": qwen_meta.get("selected_pair_words"),
            "selected_pair_word_fraction": qwen_meta.get("selected_pair_word_fraction"),
            "pair_boundary_preserved": qwen_meta.get("pair_boundary_preserved"),
        },
        "inherited_cached_fineweb_metadata": {
            "path": str(args.fineweb_meta),
            "status": fineweb_meta.get("status") if isinstance(fineweb_meta, dict) else None,
            "dataset_name": fineweb_meta.get("dataset_name") if isinstance(fineweb_meta, dict) else None,
            "config": fineweb_meta.get("config") if isinstance(fineweb_meta, dict) else None,
            "max_stream_docs": fineweb_meta.get("max_stream_docs") if isinstance(fineweb_meta, dict) else None,
            "random_summary": fineweb_meta.get("random_summary") if isinstance(fineweb_meta, dict) else None,
        },
    }
    required = {
        "row_length_sequence_identical": verification["row_length_sequence_identical"],
        "qwen_pair_block_identical": verification["qwen_pair_block_identical"],
        "tail_filler_identical": verification["tail_filler_identical"],
        "total_words_treatment": sum(r.words for r in treatment) == args.total_words,
        "total_words_control": sum(r.words for r in control) == args.total_words,
        "fineweb_words_positive": fineweb_words > 0,
        "official_filler_sufficient": state.consumed_words() <= filler_words_available,
    }
    verification["required_checks"] = required
    verification["all_required_checks_pass"] = all(required.values())
    if not verification["all_required_checks_pass"]:
        raise RuntimeError(f"verification failed: {required}")
    paths["verification"].write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    samples = {
        "treatment_first_rows": first_rows(treatment),
        "treatment_fineweb_block_first_rows": first_rows(treatment[qn:qn+fn]),
        "control_matched_block_first_rows": first_rows(control[qn:qn+fn]),
        "tail_first_rows": first_rows(tail_rows),
    }
    paths["samples"].write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metadata = {
        "status": "CACHED_FINEWEB_BROAD_SOURCE_CANDIDATE_MATERIALIZED",
        "created_unix": int(time.time()),
        "purpose": "checked 10M pair for broad public FineWeb-Edu factual-source replacement on top of clean-Qwen; not trained here",
        "causal_interpretation_if_later_trained": "treatment-control difference tests cached FineWeb-Edu random-quality source replacement versus official BabyLM filler with identical clean-Qwen pair rows, row lengths, official tail, pass order, tokenizer/model/training recipe if launched",
        "inputs": {
            "qwen_pool": str(args.qwen_pool),
            "qwen_meta": str(args.qwen_meta),
            "fineweb": str(args.fineweb),
            "fineweb_meta": str(args.fineweb_meta),
        },
        "outputs": {k: str(v) for k, v in paths.items()},
        "verification": verification,
        "sha256": {k: sha256_file(v) for k, v in paths.items() if v.exists() and v.is_file() and k != "note"},
        "elapsed_sec": round(time.time() - t0, 3),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research cached FineWeb-Edu broad-source candidate\n\n"]
    lines.append("This is a prepared data pair, not trained evidence. It should be used only if the running semantic-view contrast does not provide a strong enough same-source transformation signal or if a separate broad factual-source test becomes scientifically preferable.\n\n")
    lines.append("## Construction\n\n")
    lines.append(f"- Shared clean-Qwen pair block: {len(qwen_pair_rows):,} rows, {qwen_words:,} words ({qwen_words/args.total_words:.2%}).\n")
    lines.append(f"- Treatment-only cached FineWeb-Edu random-quality block: {len(fineweb_rows):,} rows, {fineweb_words:,} words ({fineweb_words/args.total_words:.2%}).\n")
    lines.append(f"- Control block: official BabyLM filler chunked to the exact FineWeb row-length sequence, {control_meta['rows']:,} rows and {control_meta['words']:,} words.\n")
    lines.append(f"- Shared official tail after the replacement block: {len(tail_rows):,} rows, {tail_needed:,} words ({tail_needed/args.total_words:.2%}).\n")
    lines.append("- Treatment and control have identical clean-Qwen rows, identical tail text, exact 10M words, and the same row-length sequence.\n\n")
    lines.append("## Interpretation if later trained\n\n")
    lines.append("This tests broad public factual-source replacement, not generated paraphrastic variation. It inherits INITIAL_MODEL_STUDIES evidence that simple FineWeb relation filtering was unstable at small scale, so a later run should be interpreted against that history and only continued if official-compatible downstream scores improve the hard EWoK/Entity/COMPS/GlobalPIQA region without destroying the clean-Qwen strengths.\n\n")
    lines.append(f"Verification JSON: `{paths['verification']}`\n\n")
    lines.append(f"Metadata JSON: `{paths['metadata']}`\n\n")
    lines.append(f"Sample rows JSON: `{paths['samples']}`\n")
    paths["note"].parent.mkdir(parents=True, exist_ok=True)
    paths["note"].write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": metadata["status"],
        "out_dir": str(out_dir),
        "qwen_pair_words": qwen_words,
        "fineweb_words": fineweb_words,
        "tail_words": tail_needed,
        "rows": verification["rows"],
        "word_fractions": verification["word_fractions"],
        "write_training": bool(args.write_training),
        "training_exposure_words": training_exposure,
        "all_required_checks_pass": verification["all_required_checks_pass"],
        "verification": str(paths["verification"]),
        "metadata": str(paths["metadata"]),
        "note": str(paths["note"]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
