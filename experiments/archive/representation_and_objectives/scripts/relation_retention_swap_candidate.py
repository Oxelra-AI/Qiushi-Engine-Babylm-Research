#!/usr/bin/env python3
"""research: exact-budget relation-retention swap candidate inside existing compact pairs.

CPU-only.  Uses the research relation-frame audit features to build a legal,
same-budget pair-selection candidate that removes some selected compact pairs whose
rewrites drop broad relation categories and replaces them with unused accepted Qwen
compact pairs that retain those categories.  It does not materialize a 10M/100M corpus
or launch training; it decides whether an internal accepted-pair repair is large and
clean enough to be worth considering after corrected official scores arrive.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

USER_ROOT = Path(".").resolve()
A01 = USER_ROOT / "experiments/archive/representation_and_objectives"
WORKSPACE = A01
SCRIPTS = WORKSPACE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import relation_frame_retention_audit as audit  # noqa: E402

OUT_DIR = WORKSPACE / "data/relation_retention_swap_candidate"
OUT_JSON = OUT_DIR / "relation_retention_swap_candidate.json"
OUT_FINAL = OUT_DIR / "relation_retention_swap_selected_pairs.jsonl"
OUT_REMOVED = OUT_DIR / "relation_retention_swap_removed_pairs.csv"
OUT_ADDED = OUT_DIR / "relation_retention_swap_added_pairs.csv"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/54_relation_retention_swap_candidate.md')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def load_rows() -> list[dict[str, Any]]:
    selected_reinvest = audit.load_selected_ids(audit.SELECTED_REINVEST)
    selected_core = audit.load_selected_ids(audit.SELECTED_CORE)
    selected_added = audit.load_selected_ids(audit.SELECTED_ADDED)
    rows = []
    for obj in audit.read_jsonl(audit.ALL_ACCEPTED):
        rows.append(audit.enrich_pair(obj, selected_reinvest, selected_core, selected_added))
    return rows


def is_bad_selected(r: dict[str, Any]) -> bool:
    return (
        r["selected_reinvest"]
        and r["n_source_categories"] >= 2
        and r["category_retention_frac"] is not None
        and r["category_retention_frac"] < 0.5
        and r["content_recall"] >= 0.50
        and r["entity_recall"] >= 0.99
        and r["number_recall"] >= 0.99
    )


def is_good_unused(r: dict[str, Any]) -> bool:
    return (
        (not r["selected_reinvest"])
        and r["n_source_categories"] >= 2
        and r["category_retention_frac"] is not None
        and r["category_retention_frac"] >= 0.75
        and r["content_recall"] >= 0.55
        and r["entity_recall"] >= 0.99
        and r["number_recall"] >= 0.99
    )


def bad_value(r: dict[str, Any]) -> float:
    ret = float(r["category_retention_frac"] or 0.0)
    return float(r["relation_loss_score"]) + 0.5 * r["n_dropped_categories"] + 0.25 * (1.0 - ret)


def good_value(r: dict[str, Any]) -> float:
    ret = float(r["category_retention_frac"] or 0.0)
    return 2.0 * ret + float(r["content_recall"]) + 0.15 * r["n_source_categories"] + 0.05 * r["rewrite_cue_total"]


def dp_best_by_sum(items: list[dict[str, Any]], value_fn: Callable[[dict[str, Any]], float], max_sum: int) -> dict[int, tuple[float, int | None, int | None]]:
    """Return sum -> (value, prev_sum, item_index), maximizing value for each exact word sum."""
    dp: dict[int, tuple[float, int | None, int | None]] = {0: (0.0, None, None)}
    for idx, item in enumerate(items):
        w = int(item["pair_words"])
        if w <= 0 or w > max_sum:
            continue
        val = float(value_fn(item))
        current = list(dp.items())
        updates: list[tuple[int, tuple[float, int, int]]] = []
        for s, (v, _prev, _idx) in current:
            ns = s + w
            if ns > max_sum:
                continue
            nv = v + val
            old = dp.get(ns)
            if old is None or nv > old[0]:
                updates.append((ns, (nv, s, idx)))
        for ns, rec in updates:
            old = dp.get(ns)
            if old is None or rec[0] > old[0]:
                dp[ns] = rec
    return dp


def reconstruct(dp: dict[int, tuple[float, int | None, int | None]], items: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    s = target
    seen = set()
    while s != 0:
        val, prev, idx = dp[s]
        if prev is None or idx is None:
            raise RuntimeError({"bad_dp_chain": s})
        if idx in seen:
            raise RuntimeError({"duplicate_item_in_chain": idx, "target": target})
        seen.add(idx)
        out.append(items[idx])
        s = prev
    return out


def choose_domain_swap(domain: str, bad: list[dict[str, Any]], good: list[dict[str, Any]]) -> dict[str, Any]:
    bad_sum = sum(int(r["pair_words"]) for r in bad)
    good_sum = sum(int(r["pair_words"]) for r in good)
    max_sum = min(bad_sum, good_sum)
    if max_sum <= 0:
        return {"domain": domain, "target_pair_words": 0, "bad_available_rows": len(bad), "good_available_rows": len(good), "removed": [], "added": []}
    # Sort to make reconstruction deterministic. DP still chooses by value.
    bad_sorted = sorted(bad, key=lambda r: (-bad_value(r), r["pair_words"], r["pair_id"]))
    good_sorted = sorted(good, key=lambda r: (-good_value(r), r["pair_words"], r["pair_id"]))
    bad_dp = dp_best_by_sum(bad_sorted, bad_value, max_sum)
    good_dp = dp_best_by_sum(good_sorted, good_value, max_sum)
    common = set(bad_dp) & set(good_dp)
    common.discard(0)
    if not common:
        return {"domain": domain, "target_pair_words": 0, "bad_available_rows": len(bad), "good_available_rows": len(good), "bad_available_pair_words": bad_sum, "good_available_pair_words": good_sum, "removed": [], "added": [], "no_common_exact_sum": True}
    # Maximize exact swapped words first; use value only to choose among near ties.
    max_word = max(common)
    near = [s for s in common if s >= max(1, int(0.985 * max_word))]
    target = max(near, key=lambda s: (bad_dp[s][0] + good_dp[s][0], s))
    removed = reconstruct(bad_dp, bad_sorted, target)
    added = reconstruct(good_dp, good_sorted, target)
    return {
        "domain": domain,
        "target_pair_words": target,
        "bad_available_rows": len(bad),
        "bad_available_pair_words": bad_sum,
        "good_available_rows": len(good),
        "good_available_pair_words": good_sum,
        "removed_rows": len(removed),
        "added_rows": len(added),
        "removed_value": bad_dp[target][0],
        "added_value": good_dp[target][0],
        "removed": removed,
        "added": added,
    }


def compact_pair_record(r: dict[str, Any], status: str) -> dict[str, Any]:
    keys = [
        "pair_id", "key", "sentence_id", "doc_id", "source_text", "rewrite_text", "source_words", "rewrite_words", "pair_words",
        "length_ratio", "content_recall", "content_overlap", "entity_recall", "number_recall", "domain_hits", "primary_domain",
        "source_categories", "rewrite_categories", "dropped_source_categories", "category_retention_frac", "relation_loss_score",
    ]
    out = {k: r.get(k) for k in keys}
    out["swap_status"] = status
    return out


def write_pair_jsonl(path: Path, rows: list[dict[str, Any]], status_by_id: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(compact_pair_record(r, status_by_id.get(r["pair_id"], "kept_selected")), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], status: str) -> None:
    fields = [
        "swap_status", "pair_id", "primary_domain", "domain_hits", "pair_words", "source_words", "rewrite_words", "length_ratio", "content_recall",
        "entity_recall", "number_recall", "source_categories", "rewrite_categories", "dropped_source_categories", "category_retention_frac", "relation_loss_score", "source_text", "rewrite_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fields}
            out["swap_status"] = status
            for k in ["domain_hits", "source_categories", "rewrite_categories", "dropped_source_categories"]:
                out[k] = json.dumps(out[k], ensure_ascii=False)
            w.writerow(out)


def summarize_final(original: list[dict[str, Any]], final: list[dict[str, Any]], removed: list[dict[str, Any]], added: list[dict[str, Any]]) -> dict[str, Any]:
    orig_sum = audit.summarize(original, "original_selected_reinvest")
    final_sum = audit.summarize(final, "relation_retention_swap_candidate")
    rem_sum = audit.summarize(removed, "removed_low_retention_selected")
    add_sum = audit.summarize(added, "added_high_retention_unused")
    def val(summary: dict[str, Any], k: str) -> float | None:
        x = summary.get(k)
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else None
    delta = {}
    for k in [
        "n", "pair_words", "mean_pair_words", "mean_content_recall", "source_relation_pair_frac",
        "mean_source_categories", "mean_rewrite_categories", "mean_category_retention_frac_on_source_rel_pairs",
        "drop_any_source_category_frac_on_source_rel_pairs", "drop_majority_source_categories_frac_on_source_rel_pairs", "mean_relation_loss_score",
    ]:
        a = val(final_sum, k)
        b = val(orig_sum, k)
        if a is not None and b is not None:
            delta[k] = a - b
    return {
        "original_selected_summary": orig_sum,
        "final_candidate_summary": final_sum,
        "removed_summary": rem_sum,
        "added_summary": add_sum,
        "final_minus_original": delta,
        "swapped_pair_words_fraction_of_changed_block": sum(r["pair_words"] for r in removed) / max(1, sum(r["pair_words"] for r in original)),
        "swapped_pair_words_fraction_of_10m_pool": sum(r["pair_words"] for r in removed) / 10_000_000,
    }


def write_note(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    d = s["final_minus_original"]
    lines = ["# research relation-retention swap candidate\n\n"]
    lines.append("This is a CPU-only exact pair-word replacement plan inside the already accepted compact-pair pool. It does not launch training.\n\n")
    lines.append("## Exact budget properties\n\n")
    lines.append(f"- Original selected pair-words: `{s['original_selected_summary']['pair_words']}`; final selected pair-words: `{s['final_candidate_summary']['pair_words']}`.\n")
    lines.append(f"- Removed rows/pair-words: `{payload['n_removed']}` / `{payload['pair_words_removed']}`. Added rows/pair-words: `{payload['n_added']}` / `{payload['pair_words_added']}`.\n")
    lines.append(f"- Pair-word replacement fraction of changed block: `{s['swapped_pair_words_fraction_of_changed_block']:.4f}`; fraction of 10M pool: `{s['swapped_pair_words_fraction_of_10m_pool']:.6f}`.\n")
    lines.append("- Replacements are exact within each primary-domain bucket, preserving selected pair-word totals per primary-domain for the swapped part.\n\n")
    lines.append("## Expected corpus-side movement\n\n")
    for k in ["mean_category_retention_frac_on_source_rel_pairs", "drop_any_source_category_frac_on_source_rel_pairs", "drop_majority_source_categories_frac_on_source_rel_pairs", "mean_relation_loss_score", "mean_content_recall", "n"]:
        if k in d:
            lines.append(f"- `{k}` final-minus-original: `{d[k]:+.6f}`.\n")
    lines.append("\n## Interpretation\n\n")
    for x in payload["interpretation"]:
        lines.append(f"- {x}\n")
    lines.append("\n## Files\n\n")
    for k, v in payload["files"].items():
        lines.append(f"- {k}: `{v}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    selected = [r for r in rows if r["selected_reinvest"]]
    bad = [r for r in rows if is_bad_selected(r)]
    good = [r for r in rows if is_good_unused(r)]
    bad_by_dom = defaultdict(list)
    good_by_dom = defaultdict(list)
    for r in bad:
        bad_by_dom[r["primary_domain"]].append(r)
    for r in good:
        good_by_dom[r["primary_domain"]].append(r)
    swaps = []
    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    for dom in sorted(set(bad_by_dom) | set(good_by_dom)):
        rec = choose_domain_swap(dom, bad_by_dom.get(dom, []), good_by_dom.get(dom, []))
        swaps.append({k: v for k, v in rec.items() if k not in {"removed", "added"}})
        removed.extend(rec.get("removed", []))
        added.extend(rec.get("added", []))
    removed_ids = {r["pair_id"] for r in removed}
    added_ids = {r["pair_id"] for r in added}
    if removed_ids & added_ids:
        raise RuntimeError("removed/added overlap")
    if not removed_ids <= {r["pair_id"] for r in selected}:
        raise RuntimeError("removed pair not originally selected")
    if added_ids & {r["pair_id"] for r in selected}:
        raise RuntimeError("added pair originally selected")
    final = [r for r in selected if r["pair_id"] not in removed_ids] + added
    final.sort(key=lambda r: (str(r["primary_domain"]), str(r["pair_id"])))
    orig_pair_words = sum(r["pair_words"] for r in selected)
    final_pair_words = sum(r["pair_words"] for r in final)
    if orig_pair_words != final_pair_words:
        raise RuntimeError({"pair_word_mismatch": [orig_pair_words, final_pair_words]})
    domain_orig = Counter()
    domain_final = Counter()
    for r in selected:
        domain_orig[r["primary_domain"]] += r["pair_words"]
    for r in final:
        domain_final[r["primary_domain"]] += r["pair_words"]
    domain_delta = {d: domain_final[d] - domain_orig[d] for d in sorted(set(domain_orig) | set(domain_final))}
    nonzero_domain_delta = {d: v for d, v in domain_delta.items() if v}
    summary = summarize_final(selected, final, removed, added)
    status_by = {r["pair_id"]: "added_high_retention_unused" for r in added}
    for r in final:
        status_by.setdefault(r["pair_id"], "kept_selected")
    write_pair_jsonl(OUT_FINAL, final, status_by)
    write_csv(OUT_REMOVED, sorted(removed, key=lambda r: (r["primary_domain"], -bad_value(r), r["pair_id"])), "removed_low_retention_selected")
    write_csv(OUT_ADDED, sorted(added, key=lambda r: (r["primary_domain"], -good_value(r), r["pair_id"])), "added_high_retention_unused")
    interpretations = [
        "This swap preserves the compact changed-block pair-word budget exactly and preserves pair-word totals within primary-domain buckets; it is a corpus-side candidate, not evidence of downstream improvement.",
        "It uses only already accepted compact Qwen pairs, so it would not require new teacher generation or new source collection before a future training test.",
        "Because it changes only a small fraction of the 10M pool, it should be trained only if corrected official results localize weakness to relation-sensitive columns and no cheaper analysis can answer the route decision.",
    ]
    d = summary["final_minus_original"]
    if d.get("mean_category_retention_frac_on_source_rel_pairs", 0.0) > 0 and d.get("drop_any_source_category_frac_on_source_rel_pairs", 0.0) < 0:
        interpretations.append("The plan measurably improves relation-frame retention and reduces dropped relation categories at fixed budget; it is a plausible low-cost repair substrate if EWoK/COMPS weakness survives the compliant tokenizer.")
    else:
        interpretations.append("The plan does not strongly improve the relation-retention audit metrics, so it should not be promoted to a GPU experiment without a stronger construction.")
    payload = {
        "status": "RELATION_RETENTION_SWAP_CANDIDATE",
        "created_utc": now_utc(),
        "method": {
            "exact_pair_word_budget": True,
            "same_primary_domain_pair_word_budget_for_swapped_part": len(nonzero_domain_delta) == 0,
            "no_new_generation": True,
            "no_training_launched": True,
            "bad_selected_rule": "selected reinvest pair with >=2 source relation categories, category retention <0.5, content_recall>=0.50, entity/number recall>=0.99",
            "good_unused_rule": "unused accepted pair with >=2 source relation categories, category retention >=0.75, content_recall>=0.55, entity/number recall>=0.99",
            "dp_selection": "per-primary-domain exact subset-sum; maximize swapped word count first, then relation-loss removal plus retained-relation replacement value within the top 1.5% word-count band",
        },
        "n_all_accepted": len(rows),
        "n_original_selected": len(selected),
        "pair_words_original_selected": orig_pair_words,
        "n_bad_selected_available": len(bad),
        "pair_words_bad_selected_available": sum(r["pair_words"] for r in bad),
        "n_good_unused_available": len(good),
        "pair_words_good_unused_available": sum(r["pair_words"] for r in good),
        "n_removed": len(removed),
        "pair_words_removed": sum(r["pair_words"] for r in removed),
        "n_added": len(added),
        "pair_words_added": sum(r["pair_words"] for r in added),
        "n_final_selected": len(final),
        "pair_words_final_selected": final_pair_words,
        "primary_domain_pair_word_delta_final_minus_original": domain_delta,
        "nonzero_primary_domain_pair_word_deltas": nonzero_domain_delta,
        "per_domain_swaps": swaps,
        "summary": summary,
        "interpretation": interpretations,
        "files": {
            "json": rel(OUT_JSON),
            "final_pair_selection_jsonl": rel(OUT_FINAL),
            "removed_pairs_csv": rel(OUT_REMOVED),
            "added_pairs_csv": rel(OUT_ADDED),
            "note": rel(NOTE),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(payload)
    print(json.dumps({
        "status": payload["status"],
        "out_json": rel(OUT_JSON),
        "note": rel(NOTE),
        "n_removed": len(removed),
        "pair_words_removed": payload["pair_words_removed"],
        "n_added": len(added),
        "pair_words_added": payload["pair_words_added"],
        "pair_words_final_selected": final_pair_words,
        "nonzero_domain_pair_word_deltas": nonzero_domain_delta,
        "final_minus_original_key_metrics": {k: summary["final_minus_original"].get(k) for k in ["mean_category_retention_frac_on_source_rel_pairs", "drop_any_source_category_frac_on_source_rel_pairs", "drop_majority_source_categories_frac_on_source_rel_pairs", "mean_relation_loss_score", "mean_content_recall", "n"]},
        "interpretation": interpretations,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
