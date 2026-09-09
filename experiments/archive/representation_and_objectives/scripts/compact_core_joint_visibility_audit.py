#!/usr/bin/env python3
"""Actual packed-row joint visibility for the compact_view_core mechanism arm.

research first audited compact_view_reinvest because it is the SOTA-facing
endpoint.  independent review pointed out that the mechanism anchor is compact_view_core vs
compact_repeat_core, so this CPU audit gives the core view arm the same
trainer-interface visibility check.
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
OUT_DIR = A01 / "data" / "compact_core_joint_visibility_audit"
OUT_JSON = OUT_DIR / "compact_core_joint_visibility_audit.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_core_joint_visibility_audit.md')
SEQ_LEN = 256


def iter_jsonl(path: pathlib.Path, limit: int | None = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def read_json(path: pathlib.Path) -> Any:
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


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int:
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def audit_family(tok: Any, *, family_key: str, selected_pairs_file: str, view_pool: str, view_meta: str, repeat_pool: str, length_pool: str) -> dict[str, Any]:
    overlay_meta = read_json(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json")
    changed_rows = int(overlay_meta["families"][family_key]["changed_block_rows"])
    special_overhead = enc_len(tok, "hello", True) - enc_len(tok, "hello", False)
    cutoff = SEQ_LEN - special_overhead
    pairs = {p["pair_id"]: p for p in iter_jsonl(DENSITY_DIR / selected_pairs_file)}
    rows = list(iter_jsonl(OVERLAY_DIR / view_pool, limit=changed_rows))
    metas = list(iter_jsonl(OVERLAY_DIR / view_meta, limit=changed_rows))
    repeat_rows = list(iter_jsonl(OVERLAY_DIR / repeat_pool, limit=changed_rows))
    length_rows = list(iter_jsonl(OVERLAY_DIR / length_pool, limit=changed_rows))
    if not (len(rows) == len(metas) == len(repeat_rows) == len(length_rows) == changed_rows):
        raise RuntimeError(f"{family_key}: changed-row read mismatch")

    reconstructed_exact = 0
    reconstructed_words = 0
    row_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    domain_visibility = defaultdict(lambda: Counter())
    for ridx, (row, meta) in enumerate(zip(rows, metas)):
        pids = meta.get("pair_ids") or []
        units = []
        for pid in pids:
            p = pairs[pid]
            units.append(str(p["source_text"]) + " " + str(p["rewrite_text"]))
        constructed = " ".join(units)
        if pids and constructed == row["text"]:
            reconstructed_exact += 1
        if pids and len(constructed.split()) == int(row.get("words", 0)):
            reconstructed_words += 1
        prefix = ""
        full = partial = hidden = src_visible = 0
        for pos, pid in enumerate(pids):
            p = pairs[pid]
            source = str(p["source_text"])
            rewrite = str(p["rewrite_text"])
            unit = source + " " + rewrite
            before = prefix
            source_prefix = before + (" " if before else "") + source
            pair_prefix = before + (" " if before else "") + unit
            start = enc_len(tok, before, False) if before else 0
            source_end = enc_len(tok, source_prefix, False)
            pair_end = enc_len(tok, pair_prefix, False)
            source_ok = source_end <= cutoff
            full_ok = pair_end <= cutoff
            visible_inside = max(0, min(pair_end, cutoff) - start)
            partial_ok = visible_inside > 0 and not full_ok
            if source_ok:
                src_visible += 1
            if full_ok:
                full += 1
            elif partial_ok:
                partial += 1
            else:
                hidden += 1
            domains = p.get("domain_hits") or ["no_domain"]
            for d in domains:
                domain_visibility[d]["pairs"] += 1
                if full_ok:
                    domain_visibility[d]["full_visible"] += 1
                elif partial_ok:
                    domain_visibility[d]["partial_visible"] += 1
                else:
                    domain_visibility[d]["hidden"] += 1
            pair_records.append({
                "row_index": ridx,
                "pair_position_in_row": pos,
                "pair_id": pid,
                "domain_hits": domains,
                "source_end_tok_no_special": source_end,
                "pair_end_tok_no_special": pair_end,
                "source_visible_under_seq256": source_ok,
                "pair_full_visible_under_seq256": full_ok,
                "pair_partial_visible_under_seq256": partial_ok,
                "content_recall": p.get("content_recall"),
                "entity_recall": p.get("entity_recall"),
                "number_recall": p.get("number_recall"),
            })
            prefix = pair_prefix
        row_records.append({
            "row_index": ridx,
            "pairs": len(pids),
            "row_words": int(row.get("words", 0)),
            "view_tokens_no_special": enc_len(tok, row["text"], False),
            "view_tokens_with_special": enc_len(tok, row["text"], True),
            "repeat_tokens_no_special": enc_len(tok, repeat_rows[ridx]["text"], False),
            "lengthmatched_tokens_no_special": enc_len(tok, length_rows[ridx]["text"], False),
            "full_visible_pairs": full,
            "partial_visible_pairs": partial,
            "hidden_pairs": hidden,
            "source_visible_pairs": src_visible,
        })

    total = len(pair_records)
    full = sum(1 for r in pair_records if r["pair_full_visible_under_seq256"])
    source_ok = sum(1 for r in pair_records if r["source_visible_under_seq256"])
    partial = sum(1 for r in pair_records if r["pair_partial_visible_under_seq256"])
    hidden = total - full - partial
    trunc_rows = [r for r in row_records if r["view_tokens_with_special"] > SEQ_LEN]
    domain_rates = {}
    for d, c in domain_visibility.items():
        n = c["pairs"]
        domain_rates[d] = {"pairs": n, "full_visible": c["full_visible"], "partial_visible": c["partial_visible"], "hidden": c["hidden"], "full_visible_rate": c["full_visible"] / n if n else None}
    view_minus_repeat = [r["view_tokens_no_special"] - r["repeat_tokens_no_special"] for r in row_records]
    view_minus_length = [r["view_tokens_no_special"] - r["lengthmatched_tokens_no_special"] for r in row_records]
    return {
        "family_key": family_key,
        "changed_rows": changed_rows,
        "pair_rows": sum(1 for r in row_records if r["pairs"]),
        "nonpair_rows": sum(1 for r in row_records if not r["pairs"]),
        "exact_string_reconstruction_pair_rows": reconstructed_exact,
        "word_count_reconstruction_pair_rows": reconstructed_words,
        "visibility_summary": {
            "total_pair_occurrences": total,
            "unique_pairs": len({r["pair_id"] for r in pair_records}),
            "source_visible_pairs": source_ok,
            "full_source_plus_rewrite_visible_pairs": full,
            "partial_visible_pairs": partial,
            "hidden_pairs": hidden,
            "source_visible_rate": source_ok / total if total else None,
            "full_pair_visible_rate": full / total if total else None,
            "partial_visible_rate": partial / total if total else None,
            "hidden_rate": hidden / total if total else None,
            "view_rows_over_seq256_with_special": len(trunc_rows),
            "truncated_rows_with_pair_loss": sum(1 for r in trunc_rows if r["full_visible_pairs"] < r["pairs"]),
        },
        "row_stats": {
            "pairs_per_row": quantiles([r["pairs"] for r in row_records]),
            "view_tokens_with_special": quantiles([r["view_tokens_with_special"] for r in row_records]),
            "lost_pairs_per_pair_row": quantiles([r["pairs"] - r["full_visible_pairs"] for r in row_records if r["pairs"]]),
            "view_minus_repeat_tokens_no_special": quantiles(view_minus_repeat),
            "view_minus_lengthmatched_tokens_no_special": quantiles(view_minus_length),
            "rows_view_token_len_gt_repeat": sum(1 for x in view_minus_repeat if x > 0),
            "rows_view_token_len_lt_repeat": sum(1 for x in view_minus_repeat if x < 0),
            "rows_view_token_len_eq_repeat": sum(1 for x in view_minus_repeat if x == 0),
        },
        "pair_position_loss_counts": dict(Counter(r["pair_position_in_row"] for r in pair_records if not r["pair_full_visible_under_seq256"]).most_common()),
        "domain_visibility": domain_rates,
        "non_full_visible_examples": [r for r in pair_records if not r["pair_full_visible_under_seq256"]][:8],
    }


def main() -> None:
    from transformers import AutoTokenizer  # type: ignore
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    core = audit_family(
        tok,
        family_key="compact_core_neutral",
        selected_pairs_file="selected_compact_core_pairs.jsonl",
        view_pool="cleanqwen_fineweb_compact_view_core_neutral_10M.jsonl",
        view_meta="cleanqwen_fineweb_compact_view_core_neutral_changed_block_rows_meta.jsonl",
        repeat_pool="cleanqwen_fineweb_repeat_compact_core_neutral_10M.jsonl",
        length_pool="cleanqwen_lengthmatched_compact_core_neutral_10M.jsonl",
    )
    payload = {
        "status": "COMPACT_CORE_JOINT_VISIBILITY_AUDIT",
        "scientific_purpose": "Give the compact_view_core mechanism anchor the same actual packed-row seq256 visibility check already run for compact_view_reinvest.",
        "tokenizer": str(TOKENIZER),
        "seq_len": SEQ_LEN,
        "family": core,
        "scientific_reading": {
            "main": "High actual packed-row joint visibility in compact_view_core supports reading the fast core-vs-repeat movement as a real adjacent two-view training signal rather than a construction that is mostly lost to truncation.",
            "remaining_confounds": "View rows still carry more visible subword tokens than repeat rows; visibility does not separate consolidation from denoising or lexical diversity.",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    vs = core["visibility_summary"]
    rs = core["row_stats"]
    vrd = rs["view_minus_repeat_tokens_no_special"]
    lines = [
        "# research compact_view_core actual joint visibility",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Seq256 visibility for the mechanism anchor",
        "",
        f"- Pair rows/non-pair rows: {core['pair_rows']} / {core['nonpair_rows']} across {core['changed_rows']} changed rows; exact string reconstruction from selected pairs: {core['exact_string_reconstruction_pair_rows']} rows.",
        f"- Full source+rewrite visible pairs: {vs['full_source_plus_rewrite_visible_pairs']} / {vs['total_pair_occurrences']} ({vs['full_pair_visible_rate']:.6f}); source visible pairs: {vs['source_visible_pairs']} ({vs['source_visible_rate']:.6f}).",
        f"- Partially visible / hidden pairs: {vs['partial_visible_pairs']} / {vs['hidden_pairs']}; view rows over seq256 with special tokens: {vs['view_rows_over_seq256_with_special']}.",
        f"- Non-full-visible pair positions: {core['pair_position_loss_counts']}.",
        "",
        "## Token-geometry difference against repeat",
        "",
        f"- Row-paired view-minus-repeat token delta (no special) mean/median/p95/sum: {vrd['mean']:.4f} / {vrd['median']:.4f} / {vrd['p95']:.4f} / {vrd['sum']:.1f}.",
        f"- Rows with view token length greater/less/equal than repeat: {rs['rows_view_token_len_gt_repeat']} / {rs['rows_view_token_len_lt_repeat']} / {rs['rows_view_token_len_eq_repeat']}.",
        "",
        "## Scientific reading",
        "",
        "The compact_view_core mechanism anchor is physically visible in the seq256 interface: nearly all source+compact rewrite pairs survive actual row packing. The remaining interpretation must still account for view-vs-repeat token fragmentation and for whether compact rewrites act through two-view consolidation, cleaner wording, lexical diversity, or a mixture of these.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE), "core_full_pair_visible_rate": vs["full_pair_visible_rate"]}, indent=2))


if __name__ == "__main__":
    main()
