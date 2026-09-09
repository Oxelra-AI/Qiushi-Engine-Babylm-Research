#!/usr/bin/env python3
"""Compare inherited Strict-100M tokenizer vs research Strict-Small-trained tokenizer
on compact_view_reinvest trainer-interface geometry.

The corrected research retrains change only the tokenizer. This CPU audit measures
what that representation change does before any downstream evaluation result is
available: token/word density, seq256 truncation risk, whole-word-mask grouping,
and source+compact-rewrite joint visibility inside the changed block. It uses
the actual trainer convention (`add_special_tokens=False`, truncation max_length
256), not the more conservative special-token cutoff used by an older research
visibility script.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

ROOT = Path(".")
A01 = ROOT / "experiments/archive/representation_and_objectives"
DENSITY_DIR = ROOT / "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard"
OLD_TOKENIZER = ROOT / "experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
NEW_TOKENIZER = A01 / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
OUT_DIR = A01 / "data/strictsmall_tokenizer_visibility_audit"
OUT_JSON = OUT_DIR / "strictsmall_tokenizer_visibility_audit.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/strictsmall_tokenizer_visibility_audit.md')
SEQ_LEN = 256


def iter_jsonl(path: Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def tok_len(tok: Any, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def is_word_start_token(s: str) -> bool:
    return s.startswith("Ġ") or s.startswith("▁")


def word_groups_for_ids(tok: Any, ids: list[int]) -> int:
    special = set(tok.all_special_ids)
    gid = -1
    for i, tid in enumerate(ids):
        if tid in special:
            continue
        s = tok.convert_ids_to_tokens(int(tid))
        if gid < 0 or i == 0 or (s is not None and is_word_start_token(str(s))):
            gid += 1
    return max(0, gid + 1)


def summarize_rows(tok: Any, rows: list[dict[str, Any]], limit_groups: int | None = None) -> dict[str, Any]:
    lens = []
    words = []
    tokens_per_word = []
    truncated = 0
    group_counts = []
    pieces_per_group = []
    for i, row in enumerate(rows):
        text = row["text"]
        w = int(row.get("words", len(text.split())))
        ids = tok.encode(text, add_special_tokens=False)
        n = len(ids)
        lens.append(n)
        words.append(w)
        tokens_per_word.append(n / w if w else 0.0)
        if n > SEQ_LEN:
            truncated += 1
        if limit_groups is None or i < limit_groups:
            ids_trunc = ids[:SEQ_LEN]
            g = word_groups_for_ids(tok, ids_trunc)
            group_counts.append(g)
            pieces_per_group.append(len(ids_trunc) / g if g else 0.0)
    return {
        "rows": len(rows),
        "tokenizer_len": len(tok),
        "row_tokens_no_special": quantiles(lens),
        "row_words": quantiles(words),
        "tokens_per_whitespace_word": quantiles(tokens_per_word),
        "rows_over_seq256_no_special": truncated,
        "rows_over_seq256_rate": truncated / len(rows) if rows else None,
        "wwm_groups_in_truncated_input": quantiles(group_counts),
        "tokens_per_wwm_group_in_truncated_input": quantiles(pieces_per_group),
    }


def pair_visibility(tok: Any, changed_rows: list[dict[str, Any]], row_metas: list[dict[str, Any]], pairs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pair_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    domain_visibility = defaultdict(lambda: Counter())
    exact_reconstruct = 0
    word_reconstruct = 0
    nonpair_rows = 0
    for ridx, (row, meta) in enumerate(zip(changed_rows, row_metas)):
        pids = meta.get("pair_ids") or []
        if not pids:
            nonpair_rows += 1
            row_records.append({
                "row_index": ridx,
                "pairs": 0,
                "row_words": int(row.get("words", len(row["text"].split()))),
                "row_tokens_no_special": tok_len(tok, row["text"]),
                "fully_visible_pairs": 0,
                "partially_visible_pairs": 0,
                "hidden_pairs": 0,
                "lost_pairs": 0,
            })
            continue
        units = [str(pairs[pid]["source_text"]) + " " + str(pairs[pid]["rewrite_text"]) for pid in pids]
        constructed = " ".join(units)
        if constructed == row["text"]:
            exact_reconstruct += 1
        if len(constructed.split()) == int(row.get("words", len(row["text"].split()))):
            word_reconstruct += 1
        prefix = ""
        full_visible = 0
        partial_visible = 0
        hidden = 0
        for j, pid in enumerate(pids):
            p = pairs[pid]
            source = str(p["source_text"])
            rewrite = str(p["rewrite_text"])
            unit = source + " " + rewrite
            source_prefix = prefix + (" " if prefix else "") + source
            pair_prefix = prefix + (" " if prefix else "") + unit
            start_tok = tok_len(tok, prefix) if prefix else 0
            source_end_tok = tok_len(tok, source_prefix)
            pair_end_tok = tok_len(tok, pair_prefix)
            visible_inside = max(0, min(pair_end_tok, SEQ_LEN) - start_tok)
            source_visible = source_end_tok <= SEQ_LEN
            pair_full = pair_end_tok <= SEQ_LEN
            pair_partial = visible_inside > 0 and not pair_full
            if pair_full:
                full_visible += 1
            elif pair_partial:
                partial_visible += 1
            else:
                hidden += 1
            domains = p.get("domain_hits") or ["no_domain"]
            for d in domains:
                domain_visibility[d]["pairs"] += 1
                if pair_full:
                    domain_visibility[d]["full_visible"] += 1
                elif pair_partial:
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
                "pair_tok_len_no_special_in_row_context": pair_end_tok - start_tok,
                "visible_tokens_inside_pair_no_special": visible_inside,
                "source_visible_under_seq256": source_visible,
                "pair_full_visible_under_seq256": pair_full,
                "pair_partial_visible_under_seq256": pair_partial,
            })
            prefix = pair_prefix
        row_records.append({
            "row_index": ridx,
            "pairs": len(pids),
            "row_words": int(row.get("words", len(row["text"].split()))),
            "row_tokens_no_special": tok_len(tok, row["text"]),
            "fully_visible_pairs": full_visible,
            "partially_visible_pairs": partial_visible,
            "hidden_pairs": hidden,
            "lost_pairs": len(pids) - full_visible,
        })
    total_pairs = len(pair_records)
    full_pairs = sum(1 for r in pair_records if r["pair_full_visible_under_seq256"])
    source_visible = sum(1 for r in pair_records if r["source_visible_under_seq256"])
    partial_pairs = sum(1 for r in pair_records if r["pair_partial_visible_under_seq256"])
    hidden_pairs = total_pairs - full_pairs - partial_pairs
    domain_rates = {}
    for d, c in domain_visibility.items():
        n = c["pairs"]
        domain_rates[d] = {k: int(c[k]) for k in ["pairs", "full_visible", "partial_visible", "hidden"]}
        domain_rates[d]["full_visible_rate"] = c["full_visible"] / n if n else None
    return {
        "row_reconstruction": {
            "changed_rows": len(changed_rows),
            "nonpair_rows": nonpair_rows,
            "pair_rows": len(changed_rows) - nonpair_rows,
            "exact_string_reconstruction_rows": exact_reconstruct,
            "word_count_reconstruction_rows": word_reconstruct,
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
            "rows_over_seq256_no_special": sum(1 for r in row_records if r["row_tokens_no_special"] > SEQ_LEN),
            "truncated_rows_with_pair_loss": sum(1 for r in row_records if r["lost_pairs"] > 0),
        },
        "row_stats": {
            "pairs_per_row": quantiles([r["pairs"] for r in row_records]),
            "row_tokens_no_special": quantiles([r["row_tokens_no_special"] for r in row_records]),
            "full_visible_pairs_per_pair_row": quantiles([r["fully_visible_pairs"] for r in row_records if r["pairs"]]),
            "lost_pairs_per_pair_row": quantiles([r["lost_pairs"] for r in row_records if r["pairs"]]),
        },
        "pair_position_loss_counts": dict(Counter(r["pair_position_in_row"] for r in pair_records if not r["pair_full_visible_under_seq256"]).most_common()),
        "domain_visibility": domain_rates,
        "non_full_visible_examples": [r for r in pair_records if not r["pair_full_visible_under_seq256"]][:10],
    }


def compare_scalar(new: float | None, old: float | None) -> float | None:
    if new is None or old is None:
        return None
    return new - old


def main() -> None:
    from transformers import AutoTokenizer  # type: ignore

    old_tok = AutoTokenizer.from_pretrained(str(OLD_TOKENIZER), local_files_only=True, use_fast=True)
    new_tok = AutoTokenizer.from_pretrained(str(NEW_TOKENIZER), local_files_only=True, use_fast=True)
    overlay_meta = read_json(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json")
    changed_rows_n = int(overlay_meta["families"]["compact_reinvest"]["changed_block_rows"])
    pool10 = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"))
    changed_rows = pool10[:changed_rows_n]
    row_metas = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl", limit=changed_rows_n))
    pairs = {p["pair_id"]: p for p in iter_jsonl(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl")}

    old_full = summarize_rows(old_tok, pool10, limit_groups=5000)
    new_full = summarize_rows(new_tok, pool10, limit_groups=5000)
    old_changed = summarize_rows(old_tok, changed_rows)
    new_changed = summarize_rows(new_tok, changed_rows)
    old_vis = pair_visibility(old_tok, changed_rows, row_metas, pairs)
    new_vis = pair_visibility(new_tok, changed_rows, row_metas, pairs)

    payload = {
        "status": "STRICTSMALL_TOKENIZER_VISIBILITY_AUDIT",
        "scientific_purpose": "Measure what the compliant 10M-trained tokenizer changes at the actual seq256 trainer interface before corrected-tokenizer downstream scores arrive.",
        "seq_len": SEQ_LEN,
        "trainer_interface": "masking_curriculum_trainer tokenizes with add_special_tokens=False, truncation=True, max_length=256; WWM groups are inferred from tokens starting with Ġ or ▁.",
        "inputs": {
            "old_tokenizer": str(OLD_TOKENIZER),
            "new_tokenizer": str(NEW_TOKENIZER),
            "pool10m": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"),
            "changed_row_meta": str(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"),
            "selected_pairs": str(DENSITY_DIR / "selected_compact_reinvest_pairs.jsonl"),
            "changed_rows": changed_rows_n,
            "pool_rows": len(pool10),
        },
        "old_tokenizer_summary": {
            "full_pool": old_full,
            "changed_block": old_changed,
            "pair_visibility": old_vis,
        },
        "new_tokenizer_summary": {
            "full_pool": new_full,
            "changed_block": new_changed,
            "pair_visibility": new_vis,
        },
        "delta_new_minus_old": {
            "full_pool_mean_tokens_per_word": compare_scalar(new_full["tokens_per_whitespace_word"]["mean"], old_full["tokens_per_whitespace_word"]["mean"]),
            "full_pool_rows_over_seq256": new_full["rows_over_seq256_no_special"] - old_full["rows_over_seq256_no_special"],
            "changed_block_mean_tokens_per_word": compare_scalar(new_changed["tokens_per_whitespace_word"]["mean"], old_changed["tokens_per_whitespace_word"]["mean"]),
            "changed_block_rows_over_seq256": new_changed["rows_over_seq256_no_special"] - old_changed["rows_over_seq256_no_special"],
            "full_pair_visible_pairs": new_vis["visibility_summary"]["full_source_plus_rewrite_visible_pairs"] - old_vis["visibility_summary"]["full_source_plus_rewrite_visible_pairs"],
            "full_pair_visible_rate": compare_scalar(new_vis["visibility_summary"]["full_pair_visible_rate"], old_vis["visibility_summary"]["full_pair_visible_rate"]),
            "source_visible_pairs": new_vis["visibility_summary"]["source_visible_pairs"] - old_vis["visibility_summary"]["source_visible_pairs"],
        },
        "scientific_reading": "The research retrain is not merely a compliance relabeling: the 10M-trained tokenizer changes token density, row truncation, and WWM grouping. These geometry changes should be considered when interpreting the corrected official scores, but only full official evaluation can decide whether compact_view_reinvest survives the corrected Strict-Small coordinate.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ov = old_vis["visibility_summary"]
    nv = new_vis["visibility_summary"]
    lines = [
        "# research Strict-Small tokenizer visibility audit",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Why this audit matters",
        "",
        "The corrected research retrains use a 16k tokenizer trained only on the exact allowed 10M compact-view pool. This audit measures the representation-level change before downstream scores arrive.",
        "",
        "## Main comparison",
        "",
        f"- Full 10M pool mean tokens/word: inherited {old_full['tokens_per_whitespace_word']['mean']:.6f} → strict-small {new_full['tokens_per_whitespace_word']['mean']:.6f} (delta {payload['delta_new_minus_old']['full_pool_mean_tokens_per_word']:.6f}).",
        f"- Full 10M rows over 256 tokens (no special tokens, matching trainer): inherited {old_full['rows_over_seq256_no_special']} → strict-small {new_full['rows_over_seq256_no_special']} (delta {payload['delta_new_minus_old']['full_pool_rows_over_seq256']}).",
        f"- Changed block mean tokens/word: inherited {old_changed['tokens_per_whitespace_word']['mean']:.6f} → strict-small {new_changed['tokens_per_whitespace_word']['mean']:.6f} (delta {payload['delta_new_minus_old']['changed_block_mean_tokens_per_word']:.6f}).",
        f"- Changed block rows over 256 tokens: inherited {old_changed['rows_over_seq256_no_special']} → strict-small {new_changed['rows_over_seq256_no_special']} (delta {payload['delta_new_minus_old']['changed_block_rows_over_seq256']}).",
        f"- Full source+rewrite visible pairs under actual seq256: inherited {ov['full_source_plus_rewrite_visible_pairs']}/{ov['total_pair_occurrences']} ({ov['full_pair_visible_rate']:.6f}) → strict-small {nv['full_source_plus_rewrite_visible_pairs']}/{nv['total_pair_occurrences']} ({nv['full_pair_visible_rate']:.6f}); delta {payload['delta_new_minus_old']['full_pair_visible_pairs']} pairs.",
        f"- Source-visible pairs: inherited {ov['source_visible_pairs']} → strict-small {nv['source_visible_pairs']} (delta {payload['delta_new_minus_old']['source_visible_pairs']}).",
        "",
        "## Reading",
        "",
        "The representation repair changes token density and therefore the exact visible evidence and WWM group structure. It may help or hurt the endpoint independently of compliance. The ongoing H100 retrains are still necessary because this audit is only input geometry, not competence evidence.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "json": str(OUT_JSON),
        "note": str(NOTE),
        "full_pool_mean_tpw_old": old_full["tokens_per_whitespace_word"]["mean"],
        "full_pool_mean_tpw_new": new_full["tokens_per_whitespace_word"]["mean"],
        "changed_block_mean_tpw_old": old_changed["tokens_per_whitespace_word"]["mean"],
        "changed_block_mean_tpw_new": new_changed["tokens_per_whitespace_word"]["mean"],
        "full_pair_visible_old": ov["full_source_plus_rewrite_visible_pairs"],
        "full_pair_visible_new": nv["full_source_plus_rewrite_visible_pairs"],
        "full_pair_visible_rate_old": ov["full_pair_visible_rate"],
        "full_pair_visible_rate_new": nv["full_pair_visible_rate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
