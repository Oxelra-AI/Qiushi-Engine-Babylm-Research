#!/usr/bin/env python3
"""research: test whether rel_eq0 binding damage is operation-content pull.

research showed that a last-operation recency option accounts for only a small part
of rel_eq0 damage.  The remaining hypothesis is still relation-typed content pull:
answer-only private training may increase probability of options whose object words
were mentioned in irrelevant operation sentences, even when the queried box/entity
was unchanged.  Entity options are length/item-count matched, so this script asks
whether chck82-correct -> endpoint-wrong flips choose options with more operation-
mentioned items or content tokens than the gold option.

The script uses only saved official Entity predictions and the official Entity item
text.  It does not run model evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import json
import math
import pathlib
import re
import sys
import time
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import eval_state_update_entity as S53  # noqa: E402
import entity_recency_diagnostic as S84  # noqa: E402

RECENCY_CSV = _public_path('experiments/archive/relation_learning/data/entity_recency_diagnostic_alpha050_075_100/prediction_rows.csv')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/rel_eq0_operation_content_partition')

LABELS_DEFAULT = [
    "coherent86",
    "binding_ep25_alpha0p50",
    "binding_ep25_alpha0p75",
    "binding_ep25_alpha1p00",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower()).rstrip(" .")


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def bin_irrelevant(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 6:
        return "4-6"
    return "7+"


def op_phrase(op: str) -> tuple[str, list[int], list[int], str]:
    """Return (kind, boxes, item-ish phrases, raw item phrase) for an Entity op."""
    s = op.strip().rstrip(".")
    m = re.match(r"^Put\s+(.+?)\s+into\s+Box\s+(\d+)$", s)
    if m:
        return "Put", [int(m.group(2))], [], m.group(1)
    m = re.match(r"^Remove\s+(.+?)\s+from\s+Box\s+(\d+)$", s)
    if m:
        return "Remove", [int(m.group(2))], [], m.group(1)
    m = re.match(r"^Move\s+(.+?)\s+from\s+Box\s+(\d+)\s+to\s+Box\s+(\d+)$", s)
    if not m:
        m = re.match(r"^Move\s+(.+?)\s+of\s+Box\s+(\d+)\s+to\s+Box\s+(\d+)$", s)
    if m:
        return "Move", [int(m.group(2)), int(m.group(3))], [], m.group(1)
    return "unparsed", [], [], ""


def collect_operation_items(initial_sentence: str, ops: list[str]) -> dict[str, Any]:
    """Simulate operations and collect object phrases mentioned/moved by operations."""
    state = S84.parse_initial_items(initial_sentence)
    op_items: list[str] = []
    op_item_sources: list[str] = []
    op_token_set: set[str] = set()
    unparsed = 0
    for oi, op in enumerate(ops):
        kind, boxes, _unused, phrase = op_phrase(op)
        op_token_set.update(S84.content_tokens_for_match(op))
        mentioned: list[str] = []
        if kind == "Put":
            mentioned = S84.split_items(phrase)
        elif kind == "Remove":
            items = S84.split_items(phrase)
            b = boxes[0] if boxes else -1
            _remaining, matched = S84.remove_with_matches(list(state.get(b, [])), items)
            mentioned = matched or items
        elif kind == "Move":
            src = boxes[0] if boxes else -1
            ph = norm(phrase)
            if ph in {"the contents", "contents", "the content", "content"}:
                mentioned = list(state.get(src, []))
            else:
                items = S84.split_items(phrase)
                _remaining, matched = S84.remove_with_matches(list(state.get(src, [])), items)
                mentioned = matched or items
        else:
            unparsed += 1
        for it in mentioned:
            ni = S84.norm_item(it)
            if ni:
                op_items.append(ni)
                op_item_sources.append(f"op{oi}:{kind}:{ni}")
        # advance state using the repaired research state machine
        affected, got_kind = S84.apply_op(state, op)
        if got_kind == "unparsed":
            unparsed += 1
    return {
        "operation_items": op_items,
        "operation_item_sources": op_item_sources,
        "operation_token_set": op_token_set,
        "unparsed_ops": unparsed,
    }


def option_operation_counts(option: str, op_meta: dict[str, Any]) -> dict[str, Any]:
    items = S84.split_items(option)
    op_items = list(op_meta.get("operation_items") or [])
    op_token_set = set(op_meta.get("operation_token_set") or set())
    item_hits = 0
    hit_items: list[str] = []
    for item in items:
        # Operations often omit adjectives, so reuse the research fuzzy object score.
        if any(S84.item_match_score(op_item, item)[0] > 0 or S84.item_match_score(item, op_item)[0] > 0 for op_item in op_items):
            item_hits += 1
            hit_items.append(item)
    content_toks = S84.content_tokens_for_match(option)
    tok_hits = sum(1 for t in content_toks if t in op_token_set)
    return {
        "item_count": len(items),
        "operation_item_hits": item_hits,
        "operation_item_hit_fraction": (item_hits / len(items)) if items else 0.0,
        "operation_token_hits": tok_hits,
        "content_token_count": len(content_toks),
        "operation_token_hit_fraction": (tok_hits / len(content_toks)) if content_toks else 0.0,
        "hit_items": " || ".join(hit_items),
    }


def load_entity_items() -> dict[str, dict[str, Any]]:
    """Load the exact no-'nothing' official Entity universe keyed by pred_id."""
    out: dict[str, dict[str, Any]] = {}
    for fn in S53.ENTITY_FILES:
        typ = fn[:-6]
        counters: dict[int, int] = collections.defaultdict(int)
        for obj in S53.read_jsonl(S53.ENTITY_DATA / fn):
            if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                continue
            reported = int(obj["numops"])
            idx = counters[reported]
            counters[reported] += 1
            pid = f"{typ}_{reported}_ops_{idx}"
            initial, ops, query = S53.split_entity_prefix(obj.get("input_prefix", ""))
            qbox = S53.parse_query_box(query)
            op_meta = collect_operation_items(initial, ops)
            # Compute operation-content counts for all options once.
            options = [str(x) for x in obj.get("options", [])]
            opt_counts = {norm(o): option_operation_counts(o, op_meta) for o in options}
            out[pid] = {
                "pred_id": pid,
                "entity_type": typ,
                "reported_numops": reported,
                "item_index": idx,
                "input_prefix": obj.get("input_prefix", ""),
                "initial": initial,
                "ops": ops,
                "query": query,
                "query_box": qbox,
                "options": options,
                "gold": options[0] if options else "",
                "op_meta": op_meta,
                "option_counts": opt_counts,
            }
    return out


def read_prediction_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_rows(rows: list[dict[str, Any]], label: str, group_name: str) -> dict[str, Any]:
    n = len(rows)
    def pct_count(key: str) -> float:
        return 100.0 * sum(1 for r in rows if bool(r.get(key))) / n if n else float("nan")
    return {
        "label": label,
        "group": group_name,
        "n": n,
        "mean_pred_item_hits": mean([r["pred_item_hits"] for r in rows]),
        "mean_gold_item_hits": mean([r["gold_item_hits"] for r in rows]),
        "mean_pred_minus_gold_item_hits": mean([r["pred_minus_gold_item_hits"] for r in rows]),
        "pred_more_items_n": sum(1 for r in rows if r["pred_minus_gold_item_hits"] > 0),
        "pred_more_items_pct": pct_count("pred_more_items"),
        "pred_less_items_n": sum(1 for r in rows if r["pred_minus_gold_item_hits"] < 0),
        "mean_pred_token_hits": mean([r["pred_token_hits"] for r in rows]),
        "mean_gold_token_hits": mean([r["gold_token_hits"] for r in rows]),
        "mean_pred_minus_gold_token_hits": mean([r["pred_minus_gold_token_hits"] for r in rows]),
        "pred_more_tokens_n": sum(1 for r in rows if r["pred_minus_gold_token_hits"] > 0),
        "pred_more_tokens_pct": pct_count("pred_more_tokens"),
        "pred_less_tokens_n": sum(1 for r in rows if r["pred_minus_gold_token_hits"] < 0),
        "mean_irrelevant_ops": mean([r["irrelevant_ops"] for r in rows]),
        "mean_option_item_count": mean([r["pred_item_count"] for r in rows]),
    }


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    try:
        y = float(x)
        if math.isnan(y):
            return "NA"
        return f"{y:.3f}"
    except Exception:
        return "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-rows", default=str(RECENCY_CSV))
    ap.add_argument("--baseline-label", default="chck82")
    ap.add_argument("--labels", default=",".join(LABELS_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    items = load_entity_items()
    pred_rows = read_prediction_rows(pathlib.Path(args.prediction_rows) if pathlib.Path(args.prediction_rows).is_absolute() else ROOT / args.prediction_rows)
    by_label: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in pred_rows:
        by_label[str(r["label"])][str(r["pred_id"])] = r
    base_label = args.baseline_label
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    base = by_label[base_label]
    rel_eq0_pids = [pid for pid, r in base.items() if int(r.get("relevant_updates", 999)) == 0]

    detail_rows: list[dict[str, Any]] = []
    all_rel_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for label in labels:
        cur = by_label.get(label, {})
        flip_rows: list[dict[str, Any]] = []
        reverse_rows: list[dict[str, Any]] = []
        all_rows_label: list[dict[str, Any]] = []
        for pid in rel_eq0_pids:
            br = base.get(pid); cr = cur.get(pid); item = items.get(pid)
            if not br or not cr or not item:
                continue
            gold = br.get("gold", item.get("gold", ""))
            pred = cr.get("pred", "")
            gold_counts = item["option_counts"].get(norm(gold), option_operation_counts(gold, item["op_meta"]))
            pred_counts = item["option_counts"].get(norm(pred), option_operation_counts(pred, item["op_meta"]))
            rec = {
                "label": label,
                "pred_id": pid,
                "entity_type": cr.get("entity_type", item.get("entity_type", "")),
                "reported_numops": int(cr.get("reported_numops", item.get("reported_numops", 0))),
                "total_ops": int(cr.get("total_ops", 0)),
                "irrelevant_ops": int(cr.get("irrelevant_ops", 0)),
                "irrelevant_bin": bin_irrelevant(int(cr.get("irrelevant_ops", 0))),
                "base_correct": int(br.get("correct", 0)),
                "label_correct": int(cr.get("correct", 0)),
                "gold": gold,
                "pred": pred,
                "pred_item_count": int(pred_counts["item_count"]),
                "gold_item_count": int(gold_counts["item_count"]),
                "pred_item_hits": float(pred_counts["operation_item_hits"]),
                "gold_item_hits": float(gold_counts["operation_item_hits"]),
                "pred_minus_gold_item_hits": float(pred_counts["operation_item_hits"]) - float(gold_counts["operation_item_hits"]),
                "pred_item_hit_fraction": float(pred_counts["operation_item_hit_fraction"]),
                "gold_item_hit_fraction": float(gold_counts["operation_item_hit_fraction"]),
                "pred_token_hits": float(pred_counts["operation_token_hits"]),
                "gold_token_hits": float(gold_counts["operation_token_hits"]),
                "pred_minus_gold_token_hits": float(pred_counts["operation_token_hits"]) - float(gold_counts["operation_token_hits"]),
                "pred_token_hit_fraction": float(pred_counts["operation_token_hit_fraction"]),
                "gold_token_hit_fraction": float(gold_counts["operation_token_hit_fraction"]),
                "pred_more_items": float(pred_counts["operation_item_hits"]) > float(gold_counts["operation_item_hits"]),
                "pred_more_tokens": float(pred_counts["operation_token_hits"]) > float(gold_counts["operation_token_hits"]),
                "pred_less_items": float(pred_counts["operation_item_hits"]) < float(gold_counts["operation_item_hits"]),
                "pred_less_tokens": float(pred_counts["operation_token_hits"]) < float(gold_counts["operation_token_hits"]),
                "pred_hit_items": pred_counts.get("hit_items", ""),
                "gold_hit_items": gold_counts.get("hit_items", ""),
                "operation_items": " || ".join(item["op_meta"].get("operation_items", [])),
                "operations": " || ".join(item.get("ops", [])),
            }
            all_rows_label.append(rec)
            all_rel_rows.append(rec)
            if rec["base_correct"] == 1 and rec["label_correct"] == 0:
                rec2 = dict(rec); rec2["flip_type"] = "base_correct_to_label_wrong"
                flip_rows.append(rec2); detail_rows.append(rec2)
            elif rec["base_correct"] == 0 and rec["label_correct"] == 1:
                rec2 = dict(rec); rec2["flip_type"] = "base_wrong_to_label_correct"
                reverse_rows.append(rec2); detail_rows.append(rec2)

        summary_rows.append({
            **summarize_rows(flip_rows, label, "flips_base_correct_to_label_wrong"),
            "reverse_flips_to_correct": len(reverse_rows),
            "net_loss_items": len(flip_rows) - len(reverse_rows),
        })
        summary_rows.append(summarize_rows(reverse_rows, label, "reverse_flips_base_wrong_to_label_correct"))
        summary_rows.append(summarize_rows(all_rows_label, label, "all_rel_eq0_predictions"))
        for b in ["0", "1-3", "4-6", "7+"]:
            brs = [r for r in flip_rows if r["irrelevant_bin"] == b]
            if brs:
                summary_rows.append(summarize_rows(brs, label, f"flips_irrelevant_ops_{b}"))

    write_csv(out / "operation_content_detail.csv", detail_rows)
    write_csv(out / "operation_content_all_rel_eq0.csv", all_rel_rows)
    write_csv(out / "operation_content_summary.csv", summary_rows)

    result = {
        "status": "REL_EQ0_OPERATION_CONTENT_PARTITION",
        "created_utc": now(),
        "prediction_rows": rel(args.prediction_rows),
        "baseline_label": base_label,
        "labels": labels,
        "rel_eq0_items": len(rel_eq0_pids),
        "n_detail_rows": len(detail_rows),
        "n_all_rel_eq0_rows": len(all_rel_rows),
        "summary_csv": rel(out / "operation_content_summary.csv"),
        "detail_csv": rel(out / "operation_content_detail.csv"),
        "all_rel_eq0_csv": rel(out / "operation_content_all_rel_eq0.csv"),
        "main_reading": "On rel_eq0 flips where chck82 is correct and the endpoint is wrong, pred_minus_gold operation-item/token hits tests whether the wrong option contains more irrelevant-operation-mentioned content than the gold option. Positive values support operation-content pull; near-zero or mixed values keep the remaining damage as a broader initial-state confidence loss.",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# research rel_eq0 operation-content partition", "", result["main_reading"], "", f"rel_eq0 items in official no-nothing universe: `{len(rel_eq0_pids)}`", "", "## Summary", "", "| label | group | n | pred_item_hits | gold_item_hits | pred-gold item | pred_more_items | pred_token_hits | gold_token_hits | pred-gold token | pred_more_tokens | net_loss |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary_rows:
        net = r.get("net_loss_items", "")
        lines.append(f"| {r['label']} | {r['group']} | {r['n']} | {fmt(r['mean_pred_item_hits'])} | {fmt(r['mean_gold_item_hits'])} | {fmt(r['mean_pred_minus_gold_item_hits'])} | {fmt(r['pred_more_items_pct'])} | {fmt(r['mean_pred_token_hits'])} | {fmt(r['mean_gold_token_hits'])} | {fmt(r['mean_pred_minus_gold_token_hits'])} | {fmt(r['pred_more_tokens_pct'])} | {net} |")
    lines += ["", "## Files", "", f"- Detail flips/reverse flips: `{result['detail_csv']}`", f"- All rel_eq0 prediction rows: `{result['all_rel_eq0_csv']}`", f"- Summary CSV: `{result['summary_csv']}`", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
