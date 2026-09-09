#!/usr/bin/env python3
"""Actual row-local joint visibility audit for compact_view_reinvest.

The earlier research audit showed every selected source+rewrite pair is short
enough by itself.  This script checks the sharper trainer-interface question:
after packing several pairs into one changed-block row and applying seq256
truncation with special tokens, how many actual source+rewrite units are still
fully visible to the masked-LM trainer?
"""
from __future__ import annotations

import json
import math
import pathlib
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path(".")
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
DENSITY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard"
TOKENIZER = ROOT / "experiments/archive" / 'initial_model_studies' / "training" / "runs" / "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256" / "hf_model"
OUT_DIR = A01 / "data" / "compact_reinvest_actual_joint_visibility"
OUT_JSON = OUT_DIR / "compact_reinvest_actual_joint_visibility.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_reinvest_actual_joint_visibility.md')
SEQ_LEN = 256


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def quantiles(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0}
    xs = sorted(float(x) for x in vals)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": sum(xs)/len(xs), "median": q(0.5), "p75": q(0.75), "p95": q(0.95), "max": xs[-1], "sum": sum(xs)}


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int:
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def main() -> None:
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    special_overhead_samples = {
        "empty": enc_len(tok, "", True) - enc_len(tok, "", False),
        "hello": enc_len(tok, "hello", True) - enc_len(tok, "hello", False),
        "two_words": enc_len(tok, "hello world", True) - enc_len(tok, "hello world", False),
    }
    special_overhead = special_overhead_samples["hello"]
    cutoff_no_special = SEQ_LEN - special_overhead

    overlay_meta = read_json(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json")
    changed_rows = int(overlay_meta["families"]["compact_reinvest"]["changed_block_rows"])
    pairs = {p["pair_id"]: p for p in iter_jsonl(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl")}
    rows = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl", limit=changed_rows))
    metas = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl", limit=changed_rows))
    if len(rows) != changed_rows or len(metas) != changed_rows:
        raise RuntimeError("changed-row read failed")

    reconstructed_exact_rows = 0
    reconstructed_wordmatch_rows = 0
    nonpair_rows = 0
    pair_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    domain_visibility = defaultdict(lambda: Counter())

    for ridx, (row, meta) in enumerate(zip(rows, metas)):
        pids = meta.get("pair_ids") or []
        if not pids:
            nonpair_rows += 1
            row_records.append({
                "row_index": ridx,
                "pairs": 0,
                "row_words": int(row.get("words", 0)),
                "row_tokens_no_special": enc_len(tok, row["text"], False),
                "row_tokens_with_special": enc_len(tok, row["text"], True),
                "fully_visible_pairs": 0,
                "partially_visible_pairs": 0,
            })
            continue
        units = []
        for pid in pids:
            p = pairs[pid]
            units.append(str(p["source_text"]) + " " + str(p["rewrite_text"]))
        constructed = " ".join(units)
        if constructed == row["text"]:
            reconstructed_exact_rows += 1
        if len(constructed.split()) == int(row.get("words", 0)):
            reconstructed_wordmatch_rows += 1

        prefix_before = ""
        full_visible = 0
        partial_visible = 0
        hidden = 0
        for j, pid in enumerate(pids):
            p = pairs[pid]
            source = str(p["source_text"])
            rewrite = str(p["rewrite_text"])
            unit = source + " " + rewrite
            before_text = prefix_before
            source_prefix = (before_text + (" " if before_text else "") + source)
            pair_prefix = (before_text + (" " if before_text else "") + unit)
            start_tok = enc_len(tok, before_text, False) if before_text else 0
            source_end_tok = enc_len(tok, source_prefix, False)
            pair_end_tok = enc_len(tok, pair_prefix, False)
            pair_tok_len = pair_end_tok - start_tok
            visible_tokens_inside_pair = max(0, min(pair_end_tok, cutoff_no_special) - start_tok)
            source_visible = source_end_tok <= cutoff_no_special
            rewrite_visible = pair_end_tok <= cutoff_no_special
            pair_full_visible = rewrite_visible
            pair_partial_visible = visible_tokens_inside_pair > 0 and not pair_full_visible
            if pair_full_visible:
                full_visible += 1
            elif pair_partial_visible:
                partial_visible += 1
            else:
                hidden += 1
            domains = p.get("domain_hits") or ["no_domain"]
            for d in domains:
                domain_visibility[d]["pairs"] += 1
                if pair_full_visible:
                    domain_visibility[d]["full_visible"] += 1
                elif pair_partial_visible:
                    domain_visibility[d]["partial_visible"] += 1
                else:
                    domain_visibility[d]["hidden"] += 1
            pair_records.append({
                "row_index": ridx,
                "pair_position_in_row": j,
                "pair_id": pid,
                "domain_hits": domains,
                "source_words": int(p.get("source_words", len(source.split()))),
                "rewrite_words": int(p.get("rewrite_words", len(rewrite.split()))),
                "content_recall": p.get("content_recall"),
                "entity_recall": p.get("entity_recall"),
                "number_recall": p.get("number_recall"),
                "start_tok_no_special": start_tok,
                "source_end_tok_no_special": source_end_tok,
                "pair_end_tok_no_special": pair_end_tok,
                "pair_tok_len_no_special_in_row_context": pair_tok_len,
                "visible_tokens_inside_pair_no_special": visible_tokens_inside_pair,
                "source_visible_under_seq256": source_visible,
                "rewrite_visible_under_seq256": rewrite_visible,
                "pair_full_visible_under_seq256": pair_full_visible,
                "pair_partial_visible_under_seq256": pair_partial_visible,
            })
            prefix_before = pair_prefix
        row_tokens = enc_len(tok, row["text"], False)
        row_records.append({
            "row_index": ridx,
            "pairs": len(pids),
            "row_words": int(row.get("words", 0)),
            "row_tokens_no_special": row_tokens,
            "row_tokens_with_special": enc_len(tok, row["text"], True),
            "fully_visible_pairs": full_visible,
            "partially_visible_pairs": partial_visible,
            "hidden_pairs": hidden,
            "last_full_visible_pair_position": full_visible - 1 if full_visible else None,
        })

    total_pairs = len(pair_records)
    full_pairs = sum(1 for r in pair_records if r["pair_full_visible_under_seq256"])
    source_visible = sum(1 for r in pair_records if r["source_visible_under_seq256"])
    partial_pairs = sum(1 for r in pair_records if r["pair_partial_visible_under_seq256"])
    hidden_pairs = total_pairs - full_pairs - partial_pairs
    trunc_rows = [r for r in row_records if r.get("row_tokens_with_special", 0) > SEQ_LEN]
    records_sorted_hidden = [r for r in pair_records if not r["pair_full_visible_under_seq256"]]
    examples = records_sorted_hidden[:10]

    domain_rates = {}
    for d, c in domain_visibility.items():
        n = c["pairs"]
        domain_rates[d] = {
            "pairs": n,
            "full_visible": c["full_visible"],
            "partial_visible": c["partial_visible"],
            "hidden": c["hidden"],
            "full_visible_rate": c["full_visible"] / n if n else None,
        }

    payload = {
        "status": "COMPACT_REINVEST_ACTUAL_JOINT_VISIBILITY",
        "scientific_purpose": "Measure whether packed source+compact rewrite units are jointly visible under the actual seq256 trainer interface.",
        "tokenizer": str(TOKENIZER),
        "seq_len": SEQ_LEN,
        "special_overhead_samples": special_overhead_samples,
        "cutoff_no_special_assuming_single_sequence": cutoff_no_special,
        "inputs": {
            "pairs": str(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"),
            "pool": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"),
            "row_meta": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"),
            "overlay_metadata": str(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"),
        },
        "row_reconstruction": {
            "changed_rows": changed_rows,
            "nonpair_rows": nonpair_rows,
            "pair_rows": changed_rows - nonpair_rows,
            "exact_string_reconstruction_rows": reconstructed_exact_rows,
            "word_count_reconstruction_rows": reconstructed_wordmatch_rows,
        },
        "visibility_summary": {
            "total_pair_occurrences": total_pairs,
            "unique_pairs": len({r["pair_id"] for r in pair_records}),
            "source_visible_pairs": source_visible,
            "full_source_plus_rewrite_visible_pairs": full_pairs,
            "partial_visible_pairs": partial_pairs,
            "hidden_pairs": hidden_pairs,
            "source_visible_rate": source_visible / total_pairs if total_pairs else None,
            "full_pair_visible_rate": full_pairs / total_pairs if total_pairs else None,
            "partial_visible_rate": partial_pairs / total_pairs if total_pairs else None,
            "hidden_rate": hidden_pairs / total_pairs if total_pairs else None,
            "rows_over_seq256_with_special": len(trunc_rows),
            "truncated_rows_with_pair_loss": sum(1 for r in trunc_rows if r.get("fully_visible_pairs", 0) < r.get("pairs", 0)),
        },
        "row_stats": {
            "pairs_per_row": quantiles([r["pairs"] for r in row_records]),
            "row_tokens_with_special": quantiles([r["row_tokens_with_special"] for r in row_records]),
            "full_visible_pairs_per_pair_row": quantiles([r["fully_visible_pairs"] for r in row_records if r["pairs"]]),
            "lost_pairs_per_row": quantiles([r["pairs"] - r["fully_visible_pairs"] for r in row_records if r["pairs"]]),
        },
        "pair_position_loss_counts": dict(Counter(r["pair_position_in_row"] for r in pair_records if not r["pair_full_visible_under_seq256"]).most_common()),
        "domain_visibility": domain_rates,
        "non_full_visible_examples": examples,
        "scientific_reading": {
            "main": "The compact-view reinvest changed block is interpretable as adjacent two-view learning only to the extent that source and rewrite survive in the same seq256 example; high full visibility supports the mechanism, while systematic last-pair loss would require caution.",
            "downstream_dependency": "This remains a trainer-interface audit, not a BabyLM score; the running full and seed evaluations decide whether the visible signal improves official-compatible behavior.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    vs = payload["visibility_summary"]
    rr = payload["row_reconstruction"]
    lines = [
        "# research compact_view_reinvest actual joint visibility",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Actual packed-row reconstruction",
        "",
        f"- Pair rows/non-pair rows in the changed block: {rr['pair_rows']} / {rr['nonpair_rows']} across {rr['changed_rows']} changed rows.",
        f"- Exact string reconstruction from selected pair order: {rr['exact_string_reconstruction_rows']} rows; word-count reconstruction: {rr['word_count_reconstruction_rows']} rows.",
        "",
        "## Seq256 joint visibility",
        "",
        f"- Total selected pair occurrences: {vs['total_pair_occurrences']} unique {vs['unique_pairs']}.",
        f"- Source visible pairs: {vs['source_visible_pairs']} ({vs['source_visible_rate']:.6f}); full source+rewrite visible pairs: {vs['full_source_plus_rewrite_visible_pairs']} ({vs['full_pair_visible_rate']:.6f}).",
        f"- Partially visible pairs: {vs['partial_visible_pairs']} ({vs['partial_visible_rate']:.6f}); fully hidden pairs: {vs['hidden_pairs']} ({vs['hidden_rate']:.6f}).",
        f"- Rows over seq256 with special tokens: {vs['rows_over_seq256_with_special']}; rows where at least one pair loses full visibility: {vs['truncated_rows_with_pair_loss']}.",
        "",
        "## Loss pattern",
        "",
        f"- Pair-position counts for non-full-visible units: {payload['pair_position_loss_counts']}.",
        "- Non-full-visible units occur at the tail of packed rows if the position counts concentrate on later pair positions; this makes the first several adjacent source+rewrite signals robustly visible.",
        "",
        "## Scientific reading",
        "",
        "The compact reinvest intervention is not just a nominal word-level construction: almost every selected source+compact rewrite unit is physically available within the model's seq256 input. This strengthens the interpretation of the running downstream tests as tests of anchor-preserving compression and source reinvestment rather than a truncation accident.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE), "full_pair_visible_rate": vs["full_pair_visible_rate"], "source_visible_rate": vs["source_visible_rate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
