#!/usr/bin/env python3
"""research CPU audit of the compact_view_reinvest data mechanism.

This is intentionally not a new training or evaluation launch.  The running
research managed tasks already carry the decisive GPU work.  This audit checks a
load-bearing premise of the compact-view hypothesis while those tasks run: the
source and compact rewrite should be jointly visible in short fixed-sequence
training examples, and the reinvest arm should buy more source-view packets
without introducing a large token/truncation confound.
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics as stats
from collections import Counter, defaultdict
from typing import Any, Iterable

ROOT = pathlib.Path(".")
A01 = ROOT / "experiments/archive" / 'representation_and_objectives'
DENSITY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_core_reinvestment_medium_riskhard"
OVERLAY_DIR = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "density_cleanqwen_overlay_medium_riskhard"
EXPOSURE = ROOT / "experiments/archive" / 'frontier_consolidation' / "data" / "trainer_exact_token_exposure_measurement" / "trainer_exact_token_exposure_summary.json"
TOKENIZER = ROOT / "experiments/archive" / 'initial_model_studies' / "training" / "runs" / "fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256" / "hf_model"
OUT_DIR = A01 / "data" / "compact_reinvest_pair_visibility_audit"
OUT_JSON = OUT_DIR / "compact_reinvest_pair_visibility_audit.json"
NOTE = (ROOT / 'research/notes/representation_and_objectives/compact_reinvest_pair_visibility_audit.md')
SEQ_LEN = 256


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: pathlib.Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if line.strip():
                yield json.loads(line)


def quantiles(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    xs = sorted(float(v) for v in values)
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        pos = p * (len(xs) - 1)
        lo = int(math.floor(pos)); hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
    return {
        "n": len(xs),
        "min": xs[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "mean": sum(xs) / len(xs),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": xs[-1],
        "sum": sum(xs),
    }


def try_tokenizer():
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    except Exception as exc:  # keep the non-token audit useful if tokenizer import fails
        return {"error": repr(exc)}


def enc_len(tok: Any, text: str, add_special_tokens: bool = False) -> int | None:
    if isinstance(tok, dict):
        return None
    return len(tok.encode(text, add_special_tokens=add_special_tokens))


def load_pairs(name: str) -> list[dict[str, Any]]:
    return list(iter_jsonl(DENSITY_DIR / name))


def summarize_pairs(label: str, pairs: list[dict[str, Any]], tok: Any) -> dict[str, Any]:
    source_words = [int(p.get("source_words", len(str(p.get("source_text", "")).split()))) for p in pairs]
    rewrite_words = [int(p.get("rewrite_words", len(str(p.get("rewrite_text", "")).split()))) for p in pairs]
    pair_words = [int(p.get("pair_words", sw + rw)) for p, sw, rw in zip(pairs, source_words, rewrite_words)]
    content = [float(p.get("content_recall", 0.0)) for p in pairs]
    entity = [float(p.get("entity_recall", 0.0)) for p in pairs]
    number = [float(p.get("number_recall", 0.0)) for p in pairs]
    domains = Counter()
    for p in pairs:
        hits = p.get("domain_hits") or ["no_domain"]
        for h in hits:
            domains[h] += 1
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "source_words": sum(source_words),
        "rewrite_words": sum(rewrite_words),
        "pair_words": sum(pair_words),
        "rewrite_to_source_word_ratio": (sum(rewrite_words) / sum(source_words)) if sum(source_words) else None,
        "source_word_stats": quantiles(source_words),
        "rewrite_word_stats": quantiles(rewrite_words),
        "pair_word_stats": quantiles(pair_words),
        "content_recall_stats": quantiles(content),
        "entity_recall_stats": quantiles(entity),
        "number_recall_stats": quantiles(number),
        "domain_hit_counts": dict(domains.most_common()),
        "very_low_content_recall_pairs_lt_0p55": sum(1 for v in content if v < 0.55),
        "entity_recall_below_1_pairs": sum(1 for v in entity if v < 1.0),
        "number_recall_below_1_pairs": sum(1 for v in number if v < 1.0),
    }
    if not isinstance(tok, dict):
        src_tok = [enc_len(tok, str(p.get("source_text", "")), False) or 0 for p in pairs]
        rew_tok = [enc_len(tok, str(p.get("rewrite_text", "")), False) or 0 for p in pairs]
        pair_tok = [enc_len(tok, str(p.get("source_text", "")) + " " + str(p.get("rewrite_text", "")), False) or 0 for p in pairs]
        out.update({
            "source_token_stats_no_special": quantiles(src_tok),
            "rewrite_token_stats_no_special": quantiles(rew_tok),
            "pair_token_stats_no_special": quantiles(pair_tok),
            "rewrite_to_source_token_ratio": (sum(rew_tok) / sum(src_tok)) if sum(src_tok) else None,
            "individual_pair_fits_seq256_no_special": sum(1 for v in pair_tok if v <= SEQ_LEN),
            "individual_pair_fits_seq254_no_special": sum(1 for v in pair_tok if v <= SEQ_LEN - 2),
            "individual_pair_over_seq256_no_special": sum(1 for v in pair_tok if v > SEQ_LEN),
            "individual_pair_over_seq254_no_special": sum(1 for v in pair_tok if v > SEQ_LEN - 2),
        })
    return out


def summarize_changed_rows(label: str, pool_file: pathlib.Path, meta_file: pathlib.Path, changed_rows: int, tok: Any) -> dict[str, Any]:
    rows = list(iter_jsonl(pool_file, limit=changed_rows))
    metas = list(iter_jsonl(meta_file, limit=changed_rows))
    if len(rows) != changed_rows or len(metas) != changed_rows:
        raise RuntimeError(f"{label}: expected {changed_rows} rows/metas, got {len(rows)} rows {len(metas)} metas")
    words = [int(r.get("words", len(str(r.get("text", "")).split()))) for r in rows]
    pairs_per_row = [len(m.get("pair_ids") or []) for m in metas]
    domain_keys_per_row = [len((m.get("component_sources") or {}).keys()) for m in metas]
    domain_word_counts = Counter()
    for m in metas:
        for k, v in (m.get("component_sources") or {}).items():
            domain_word_counts[k] += int(v)
    pair_ids = [pid for m in metas for pid in (m.get("pair_ids") or [])]
    out: dict[str, Any] = {
        "pool_file": str(pool_file),
        "meta_file": str(meta_file),
        "changed_rows": changed_rows,
        "row_words_stats": quantiles(words),
        "total_words": sum(words),
        "pairs_listed_in_row_meta": len(pair_ids),
        "unique_pairs_listed_in_row_meta": len(set(pair_ids)),
        "duplicate_pair_ids_in_row_meta": len(pair_ids) - len(set(pair_ids)),
        "pairs_per_row_stats": quantiles(pairs_per_row),
        "rows_with_multiple_pairs": sum(1 for v in pairs_per_row if v > 1),
        "domain_keys_per_row_stats": quantiles(domain_keys_per_row),
        "rows_with_multiple_domain_keys": sum(1 for v in domain_keys_per_row if v > 1),
        "domain_word_counts_from_meta": dict(domain_word_counts.most_common()),
    }
    if not isinstance(tok, dict):
        no_special = [enc_len(tok, str(r.get("text", "")), False) or 0 for r in rows]
        with_special = [enc_len(tok, str(r.get("text", "")), True) or 0 for r in rows]
        out.update({
            "tokens_no_special_stats": quantiles(no_special),
            "tokens_with_special_stats": quantiles(with_special),
            "rows_over_seq256_with_special": sum(1 for v in with_special if v > SEQ_LEN),
            "tokens_hidden_if_seq256_with_special": sum(max(0, v - SEQ_LEN) for v in with_special),
            "visible_token_fraction_with_special": sum(min(v, SEQ_LEN) for v in with_special) / sum(with_special) if sum(with_special) else None,
            "tokens_per_word_no_special_stats": quantiles([t / w for t, w in zip(no_special, words) if w]),
        })
    return out


def row_token_delta(view: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    # This function uses aggregate fields from summarize_changed_rows; row-paired deltas are computed separately in main.
    return {
        "view_total_tokens_no_special": view.get("tokens_no_special_stats", {}).get("sum"),
        "repeat_total_tokens_no_special": repeat.get("tokens_no_special_stats", {}).get("sum"),
        "view_minus_repeat_total_tokens_no_special": None if view.get("tokens_no_special_stats", {}).get("sum") is None else view["tokens_no_special_stats"]["sum"] - repeat["tokens_no_special_stats"]["sum"],
        "view_rows_over_seq256_with_special": view.get("rows_over_seq256_with_special"),
        "repeat_rows_over_seq256_with_special": repeat.get("rows_over_seq256_with_special"),
    }


def main() -> None:
    tok = try_tokenizer()
    density_meta = read_json(DENSITY_DIR / "density_core_reinvestment_metadata.json")
    overlay_meta = read_json(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json")
    exposure = read_json(EXPOSURE) if EXPOSURE.exists() else None

    core_pairs = load_pairs("selected_compact_core_pairs.jsonl")
    added_pairs = load_pairs("selected_compact_added_pairs.jsonl")
    reinvest_pairs = load_pairs("selected_compact_reinvest_pairs.jsonl")
    core_ids = {p["pair_id"] for p in core_pairs}
    added_ids = {p["pair_id"] for p in added_pairs}
    reinvest_ids = {p["pair_id"] for p in reinvest_pairs}

    changed_rows = int(overlay_meta["families"]["compact_reinvest"]["changed_block_rows"])
    row_summaries = {
        "view_reinvest": summarize_changed_rows(
            "view_reinvest",
            OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl",
            OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl",
            changed_rows,
            tok,
        ),
        "repeat_reinvest": summarize_changed_rows(
            "repeat_reinvest",
            OVERLAY_DIR / "cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl",
            OVERLAY_DIR / "cleanqwen_fineweb_repeat_compact_reinvest_changed_block_rows_meta.jsonl",
            changed_rows,
            tok,
        ),
        "lengthmatched_reinvest": summarize_changed_rows(
            "lengthmatched_reinvest",
            OVERLAY_DIR / "cleanqwen_lengthmatched_compact_reinvest_10M.jsonl",
            OVERLAY_DIR / "cleanqwen_lengthmatched_compact_reinvest_changed_block_rows_meta.jsonl",
            changed_rows,
            tok,
        ),
    }

    paired_token_deltas: dict[str, Any] = {}
    if not isinstance(tok, dict):
        view_rows = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl", limit=changed_rows))
        repeat_rows = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl", limit=changed_rows))
        length_rows = list(iter_jsonl(OVERLAY_DIR / "cleanqwen_lengthmatched_compact_reinvest_10M.jsonl", limit=changed_rows))
        v_tok = [enc_len(tok, r["text"], False) or 0 for r in view_rows]
        r_tok = [enc_len(tok, r["text"], False) or 0 for r in repeat_rows]
        l_tok = [enc_len(tok, r["text"], False) or 0 for r in length_rows]
        paired_token_deltas = {
            "view_minus_repeat_row_tokens_no_special_stats": quantiles([v-r for v, r in zip(v_tok, r_tok)]),
            "view_minus_lengthmatched_row_tokens_no_special_stats": quantiles([v-l for v, l in zip(v_tok, l_tok)]),
            "repeat_minus_lengthmatched_row_tokens_no_special_stats": quantiles([r-l for r, l in zip(r_tok, l_tok)]),
            "rows_view_token_len_gt_repeat": sum(1 for v, r in zip(v_tok, r_tok) if v > r),
            "rows_view_token_len_lt_repeat": sum(1 for v, r in zip(v_tok, r_tok) if v < r),
            "rows_view_token_len_eq_repeat": sum(1 for v, r in zip(v_tok, r_tok) if v == r),
        }

    row_meta_ids = set()
    for m in iter_jsonl(OVERLAY_DIR / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl", limit=changed_rows):
        row_meta_ids.update(m.get("pair_ids") or [])

    pair_summaries = {
        "compact_core": summarize_pairs("compact_core", core_pairs, tok),
        "compact_added": summarize_pairs("compact_added", added_pairs, tok),
        "compact_reinvest": summarize_pairs("compact_reinvest", reinvest_pairs, tok),
    }

    exposure_contrasts = (exposure or {}).get("contrasts", {}) if isinstance(exposure, dict) else {}
    payload = {
        "status": "COMPACT_REINVEST_PAIR_VISIBILITY_AUDIT",
        "scientific_purpose": "Check the source+compact-view joint-visibility and token/truncation premise behind compact_view_reinvest while GPU full/seed evidence is running.",
        "tokenizer": str(TOKENIZER),
        "tokenizer_loaded": not isinstance(tok, dict),
        "tokenizer_error": tok.get("error") if isinstance(tok, dict) else None,
        "seq_len": SEQ_LEN,
        "sources": {
            "density_metadata": str(DENSITY_DIR / "density_core_reinvestment_metadata.json"),
            "overlay_metadata": str(OVERLAY_DIR / "density_cleanqwen_rowholdout_overlay_metadata.json"),
            "trainer_exact_exposure": str(EXPOSURE),
        },
        "metadata_audit_reading": {
            "overlay_all_exact_10M": overlay_meta.get("audit", {}).get("all_exact_10M"),
            "overlay_write_training": overlay_meta.get("write_training"),
            "reinvest_word_totals": overlay_meta.get("families", {}).get("compact_reinvest", {}).get("word_totals"),
            "reinvest_row_counts": overlay_meta.get("families", {}).get("compact_reinvest", {}).get("row_counts"),
            "reinvest_row_length_sequence_identical_within_family": overlay_meta.get("families", {}).get("compact_reinvest", {}).get("row_length_sequence_identical_within_family"),
            "reinvest_repeat_view_suffix_identical_after_pair_rows": overlay_meta.get("families", {}).get("compact_reinvest", {}).get("repeat_view_suffix_identical_after_pair_rows"),
            "selected_reinvest_ids_equal_core_union_added": reinvest_ids == (core_ids | added_ids),
            "core_added_disjoint": core_ids.isdisjoint(added_ids),
            "row_meta_pair_ids_equal_selected_reinvest": row_meta_ids == reinvest_ids,
            "row_meta_missing_pair_ids": sorted(reinvest_ids - row_meta_ids)[:10],
            "row_meta_extra_pair_ids": sorted(row_meta_ids - reinvest_ids)[:10],
        },
        "pair_summaries": pair_summaries,
        "changed_row_summaries": row_summaries,
        "paired_changed_row_token_deltas": paired_token_deltas,
        "trainer_exact_exposure_contrasts_reused": {
            "compact_view_reinvest_minus_compact_view_core": exposure_contrasts.get("compact_view_reinvest_minus_compact_view_core"),
            "compact_repeat_reinvest_minus_compact_repeat_core": exposure_contrasts.get("compact_repeat_reinvest_minus_compact_repeat_core"),
            # If absent from the reference file, the row-level audit above supplies the local view-vs-repeat reading.
        },
        "scientific_reading": {
            "joint_visibility": "If individual source+rewrite pairs and changed-block rows fit within seq256, the compact-view mechanism is a genuine adjacent two-view training signal rather than a truncation artifact.",
            "reinvestment": "Core-to-reinvest converts neutral top-up words into additional compact source-view pairs while preserving near-identical visible-token budget; downstream evidence still must come from the running full/seed evaluations.",
            "not_settled_by_this_audit": "This CPU audit does not prove downstream competence or SOTA; it checks that the data mechanism is physically visible to the trainer and that future repeat controls can be interpreted.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def fmt(v: Any, digits: int = 4) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v)
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return str(v)

    reinv = pair_summaries["compact_reinvest"]
    view_row = row_summaries["view_reinvest"]
    repeat_row = row_summaries["repeat_reinvest"]
    length_row = row_summaries["lengthmatched_reinvest"]
    vrd = paired_token_deltas.get("view_minus_repeat_row_tokens_no_special_stats", {}) if paired_token_deltas else {}
    lines = [
        "# research compact_view_reinvest pair visibility audit",
        "",
        "This CPU audit supports the active compact-view-density route while the GPU full-evaluation and independent-seed jobs run. It does not use a GPU and does not add a new model result.",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
        "## Pair-level compact signal",
        "",
        f"- Reinvest selected pairs: {reinv['pairs']} = core {pair_summaries['compact_core']['pairs']} + added {pair_summaries['compact_added']['pairs']}; ID union check: {payload['metadata_audit_reading']['selected_reinvest_ids_equal_core_union_added']}.",
        f"- Source words {reinv['source_words']}, rewrite words {reinv['rewrite_words']}, pair words {reinv['pair_words']}, rewrite/source word ratio {fmt(reinv['rewrite_to_source_word_ratio'])}.",
        f"- Mean content/entity/number recall: {fmt(reinv['content_recall_stats'].get('mean'))} / {fmt(reinv['entity_recall_stats'].get('mean'))} / {fmt(reinv['number_recall_stats'].get('mean'))}; pairs with entity recall <1: {reinv['entity_recall_below_1_pairs']}.",
    ]
    if "individual_pair_fits_seq254_no_special" in reinv:
        lines += [
            f"- Individual source+rewrite pair token length mean/median/p95/max: {fmt(reinv['pair_token_stats_no_special'].get('mean'))} / {fmt(reinv['pair_token_stats_no_special'].get('median'))} / {fmt(reinv['pair_token_stats_no_special'].get('p95'))} / {fmt(reinv['pair_token_stats_no_special'].get('max'))} (no special tokens).",
            f"- Individual pairs fitting within 254/256 tokens: {reinv['individual_pair_fits_seq254_no_special']} / {reinv['individual_pair_fits_seq256_no_special']} of {reinv['pairs']}.",
        ]
    lines += [
        "",
        "## Changed-block row visibility",
        "",
        "| arm | changed rows | words | token mean | token p95 | token max | rows >256 with special | visible token fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("view_reinvest", view_row), ("repeat_reinvest", repeat_row), ("lengthmatched_reinvest", length_row)]:
        t = row.get("tokens_with_special_stats", {})
        lines.append(
            f"| {name} | {row['changed_rows']} | {row['total_words']} | {fmt(t.get('mean'))} | {fmt(t.get('p95'))} | {fmt(t.get('max'))} | {row.get('rows_over_seq256_with_special')} | {fmt(row.get('visible_token_fraction_with_special'), 6)} |"
        )
    lines += [
        "",
        "## View-vs-repeat token geometry within the changed block",
        "",
        f"- Row-paired view-minus-repeat token delta (no special) mean/median/p95/sum: {fmt(vrd.get('mean'))} / {fmt(vrd.get('median'))} / {fmt(vrd.get('p95'))} / {fmt(vrd.get('sum'))}.",
        f"- Rows with view token length greater/less/equal than repeat: {paired_token_deltas.get('rows_view_token_len_gt_repeat')} / {paired_token_deltas.get('rows_view_token_len_lt_repeat')} / {paired_token_deltas.get('rows_view_token_len_eq_repeat')}.",
        "",
        "## Scientific reading",
        "",
        "- The active GPU work will decide downstream value. This audit checks whether the proposed adjacent source+compact-view signal is actually available inside the seq256 trainer interface and whether reinvestment changes token/truncation geometry enough to threaten interpretation.",
        "- If full/seed evaluations preserve the fast component pattern, these visibility facts make a matched repeat-reinvest seed run interpretable as a mechanism test rather than a rescue run.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "note": str(NOTE), "tokenizer_loaded": payload["tokenizer_loaded"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
