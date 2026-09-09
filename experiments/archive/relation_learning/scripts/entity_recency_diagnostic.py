#!/usr/bin/env python3
"""research: quantify transferred Entity recency policy in official predictions.

The single-frame private binding endpoint improved only the Entity strata where the
most recent state-bearing operation is also the answer and damaged strata where later
irrelevant operations must be ignored.  This script makes that mechanism explicit by
simulating the official Entity items, deriving the final state(s) of the box(es)
affected by the last operation, and measuring how often each endpoint predicts that
last-operation state.

For a Put/Remove operation there is one recency box; for Move there are two affected
states (from and to), so `pred_is_recency` is true if the prediction matches either
affected box's final contents.  The key bias coordinate is the subset where a
recency answer is one of the official choices but is not the gold query-box answer.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
sys.path.insert(0, str(SCRIPTS))
import eval_state_update_entity as S53  # noqa: E402

CHCK82_PRED = _public_path('experiments/archive/frontier_consolidation/data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/eval/official_outputs/scale1p75_seed43022_reference_chck_82M/Entity/chck_82M/full_scale1p75_seed43022_reference_chck_82M_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')
COHERENT86_PRED = _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/official_outputs/coherent86_private_scale_0p75/Entity/final/full_coherent86_private_scale_0p75_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')
ALPHA050_PRED = _public_path('experiments/archive/relation_learning/data/eval_binding_ep25_alpha0p50/full_eval/official_outputs/binding_ep25_alpha0p50/Entity/final/full_binding_ep25_alpha0p50_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')
ALPHA075_PRED = _public_path('experiments/archive/relation_learning/data/eval_binding_ep25_alpha0p75/full_eval/official_outputs/binding_ep25_alpha0p75/Entity/final/full_binding_ep25_alpha0p75_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')
ALPHA100_PRED = _public_path('experiments/archive/relation_learning/data/eval_binding_ep25_alpha1p00/full_eval/official_outputs/binding_ep25_alpha1p00/Entity/final/full_binding_ep25_alpha1p00_Entity/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json')

KEY_GROUPS = [
    "ALL", "lastop_recency_gold", "lastop_recency_not_gold", "lastop_recency_not_gold_available",
    "rel_eq0", "rel_eq0_irrelevant_ops_gt0", "rel_eq0_irrelevant_ops_1to3",
    "rel_eq0_irrelevant_ops_4to6", "rel_eq0_irrelevant_ops_ge7", "rel_ge1",
    "rel_ge1_postrel_ops0", "rel_ge1_postrel_ops_gt0", "rel_ge3_postrel_ops0",
    "rel_ge3_postrel_ops_gt0", "stale_available_not_gold",
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path | str) -> str:
    q = pathlib.Path(p)
    try:
        return str(q.resolve().relative_to(ROOT))
    except Exception:
        return str(q)


def norm_answer(s: Any) -> str:
    return S53.norm_answer(s)


def norm_item(s: str) -> str:
    t = str(s or "").strip()
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t)
    return t


def split_items(contents: str) -> list[str]:
    c = norm_item(contents)
    if not c or c.lower() == "nothing":
        return []
    return [norm_item(x) for x in re.split(r"\s+and\s+", c) if norm_item(x) and norm_item(x).lower() != "nothing"]


def join_items(items: list[str]) -> str:
    if not items:
        return "nothing."
    return " and ".join(items) + "."


def parse_initial_items(initial_sentence: str) -> dict[int, list[str]]:
    raw = S53.parse_initial_contents(initial_sentence)
    return {int(k): split_items(v) for k, v in raw.items()}


def content_tokens_for_match(s: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", norm_answer(s))
    return [t for t in toks if t not in {"the", "a", "an"}]


def item_match_score(query_item: str, candidate_item: str) -> tuple[int, int]:
    """Return a deterministic match score for an operation item against a box item.

    Entity operations sometimes omit the adjective present in the initial contents,
    e.g. `Remove the ticket from Box 3` when the box contains `the green ticket`.
    Exact string matching therefore under-parses the official state machine.  Score
    exact normalized matches first, then suffix/head matches, then token-subset
    matches.  The second component prefers shorter candidates when several objects
    share a noun; such rows are rare and remain recorded through the simulation
    agreement check.
    """
    qn = norm_answer(query_item)
    cn = norm_answer(candidate_item)
    if qn == cn:
        return (3, -len(cn))
    qt = content_tokens_for_match(query_item)
    ct = content_tokens_for_match(candidate_item)
    if not qt or not ct:
        return (0, 0)
    if len(qt) <= len(ct) and ct[-len(qt):] == qt:
        return (2, -len(ct))
    if all(t in ct for t in qt):
        return (1, -len(ct))
    return (0, 0)


def remove_with_matches(box_items: list[str], items: list[str]) -> tuple[list[str], list[str]]:
    remaining = list(box_items)
    matched: list[str] = []
    for it in items:
        scored = [(item_match_score(it, cur), j) for j, cur in enumerate(remaining)]
        scored = [(sc, j) for sc, j in scored if sc[0] > 0]
        if scored:
            scored.sort(key=lambda x: (x[0][0], x[0][1]), reverse=True)
            j = scored[0][1]
            matched.append(remaining.pop(j))
        else:
            # For Put this branch is not used; for Move/Remove it records the
            # operation text itself as the moved phrase when the source state is
            # too ambiguous.  The later official-option matching keeps only rows
            # whose affected-box state corresponds to an offered option.
            matched.append(norm_item(it))
    return remaining, matched


def remove_items(box_items: list[str], items: list[str]) -> list[str]:
    return remove_with_matches(box_items, items)[0]


def item_signature(contents: str) -> tuple[tuple[str, ...], ...]:
    sig = []
    for it in split_items(contents):
        toks = tuple(content_tokens_for_match(it))
        if toks:
            sig.append(toks)
    return tuple(sorted(sig))


def state_signature(items: list[str]) -> tuple[tuple[str, ...], ...]:
    return item_signature(join_items(items))


def apply_op(state: dict[int, list[str]], op: str) -> tuple[list[int], str]:
    op_clean = op.strip().rstrip(".")
    m = re.match(r"^Put\s+(.+?)\s+into\s+Box\s+(\d+)$", op_clean)
    if m:
        items = split_items(m.group(1))
        b = int(m.group(2))
        # BabyLM Entity puts new material at the front of the box description.
        # In the Entity data, word order in offered choices is not always the
        # same as the obvious operation-rendering order.  We keep a deterministic
        # state here, but later identify recency options by unordered object
        # signature against the official options.
        state[b] = list(state.get(b, [])) + items
        return [b], "Put"
    m = re.match(r"^Remove\s+(.+?)\s+from\s+Box\s+(\d+)$", op_clean)
    if m:
        items = split_items(m.group(1))
        b = int(m.group(2))
        state[b] = remove_items(state.get(b, []), items)
        return [b], "Remove"
    m = re.match(r"^Move\s+(.+?)\s+from\s+Box\s+(\d+)\s+to\s+Box\s+(\d+)$", op_clean)
    if not m:
        m = re.match(r"^Move\s+(.+?)\s+of\s+Box\s+(\d+)\s+to\s+Box\s+(\d+)$", op_clean)
    if m:
        phrase = norm_item(m.group(1)).lower()
        src = int(m.group(2)); dst = int(m.group(3))
        if phrase in {"the contents", "contents", "the content", "content"}:
            moved = list(state.get(src, []))
            remaining = []
        else:
            items = split_items(m.group(1))
            remaining, moved = remove_with_matches(state.get(src, []), items)
        state[src] = remaining
        state[dst] = list(state.get(dst, [])) + moved
        return [src, dst], "Move"
    return [], "unparsed"


def row_recency_meta(obj: dict[str, Any], typ: str, item_index: int) -> dict[str, Any]:
    initial, ops, query = S53.split_entity_prefix(obj.get("input_prefix", ""))
    qbox = S53.parse_query_box(query)
    state = parse_initial_items(initial)
    last_boxes: list[int] = []
    last_kind = "none"
    for op in ops:
        boxes, kind = apply_op(state, op)
        if boxes:
            last_boxes, last_kind = boxes, kind
        else:
            last_kind = kind
    options = [str(x) for x in obj.get("options", [])]
    option_norms = {norm_answer(o): o for o in options}
    gold = options[0] if options else ""
    gold_norm = norm_answer(gold)
    recency_answers_all = [join_items(state.get(b, [])) for b in last_boxes]
    recency_sigs = {state_signature(state.get(b, [])) for b in last_boxes}
    recency_options = []
    seen_opts = set()
    for opt in options:
        no = norm_answer(opt)
        if no in seen_opts:
            continue
        if no in {norm_answer(a) for a in recency_answers_all} or item_signature(opt) in recency_sigs:
            recency_options.append(opt)
            seen_opts.add(no)
    recency_option_norms = [norm_answer(x) for x in recency_options]
    recency_gold = int(bool(recency_option_norms) and gold_norm in set(recency_option_norms))
    recency_not_gold_available = int(any(na != gold_norm for na in recency_option_norms))
    simulated_gold = join_items(state.get(qbox, [])) if qbox is not None else ""
    return {
        "uid": f"{typ}_{int(obj['numops'])}_ops",
        "item_index": item_index,
        "sample_id": obj.get("sample_id"),
        "example_id": obj.get("example_id"),
        "entity_type": typ,
        "last_op_kind": last_kind,
        "last_op_boxes": ";".join(str(x) for x in last_boxes),
        "last_op_affects_query": int(qbox in set(last_boxes)) if qbox is not None else 0,
        "simulated_gold": simulated_gold,
        "simulation_matches_gold": int(norm_answer(simulated_gold) == gold_norm or item_signature(simulated_gold) == item_signature(gold)),
        "recency_answers": " || ".join(recency_answers_all),
        "recency_options": " || ".join(recency_options),
        "recency_option_norms": " || ".join(recency_option_norms),
        "recency_option_available": int(bool(recency_option_norms)),
        "recency_gold": recency_gold,
        "recency_not_gold_available": recency_not_gold_available,
    }


def load_recency_meta() -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for fn in S53.ENTITY_FILES:
        typ = fn[:-6]
        counters: dict[int, int] = defaultdict(int)
        for obj in S53.read_jsonl(S53.ENTITY_DATA / fn):
            # Mirror S53.load_entity_metadata: remove rows with a "nothing" option.
            if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                continue
            reported = int(obj["numops"])
            idx = counters[reported]
            counters[reported] += 1
            r = row_recency_meta(obj, typ, idx)
            out[(r["uid"], idx)] = r
    return out


def parse_target(spec: str) -> dict[str, str]:
    parts = spec.split(",", 2)
    if len(parts) != 3:
        raise ValueError("target must be label,checkpoint,predictions_path")
    return {"label": parts[0], "checkpoint": parts[1], "predictions": parts[2]}


def read_prediction_rows(targets: list[dict[str, str]]) -> list[dict[str, Any]]:
    meta = S53.load_entity_metadata()
    rec = load_recency_meta()
    rows: list[dict[str, Any]] = []
    for t in targets:
        pred_path = pathlib.Path(t["predictions"])
        if not pred_path.is_absolute():
            pred_path = ROOT / pred_path
        recency_missing = 0
        for uid, item_index, pred_id, pred_text in S53.flatten_predictions(pred_path):
            m = meta.get((uid, item_index))
            rr = rec.get((uid, item_index))
            if not m or not rr:
                recency_missing += 1
                continue
            pred_norm = norm_answer(pred_text)
            rec_norms = [x for x in str(rr.get("recency_option_norms", "")).split(" || ") if x]
            pred_is_recency = int(bool(rec_norms) and pred_norm in set(rec_norms))
            pred_is_recency_not_gold = int(pred_is_recency and pred_norm != norm_answer(m.get("gold", "")))
            row = {
                "label": t["label"],
                "checkpoint": t["checkpoint"],
                "uid": uid,
                "item_index": item_index,
                "pred_id": pred_id,
                "pred": pred_text,
                "correct": int(pred_norm == norm_answer(m.get("gold", ""))),
                "pred_is_recency": pred_is_recency,
                "pred_is_recency_not_gold": pred_is_recency_not_gold,
            }
            row.update(m)
            row.update(rr)
            rows.append(row)
        if recency_missing:
            print(json.dumps({"event": "missing_recency_rows", "target": t["label"], "n": recency_missing}), flush=True)
    return rows


def row_groups(r: dict[str, Any]) -> list[str]:
    groups = S53.item_groups(r)
    if int(r.get("recency_gold", 0)):
        groups.append("lastop_recency_gold")
    if int(r.get("recency_not_gold_available", 0)):
        groups.append("lastop_recency_not_gold_available")
    if int(r.get("recency_option_available", 0)) and not int(r.get("recency_gold", 0)):
        groups.append("lastop_recency_not_gold")
    if int(r.get("last_op_affects_query", 0)):
        groups.append("lastop_affects_query")
    else:
        groups.append("lastop_not_query")
    return groups


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for g in row_groups(r):
            buckets[(str(r["label"]), str(r["checkpoint"]), g)].append(r)
    out = []
    for (label, ck, g), vals in sorted(buckets.items()):
        n = len(vals)
        rec_av = [v for v in vals if int(v.get("recency_option_available", 0))]
        rec_ng = [v for v in vals if int(v.get("recency_not_gold_available", 0))]
        out.append({
            "label": label,
            "checkpoint": ck,
            "group": g,
            "n": n,
            "accuracy_pct": 100.0 * sum(int(v["correct"]) for v in vals) / n if n else float("nan"),
            "recency_available_n": len(rec_av),
            "recency_pick_pct_among_available": 100.0 * sum(int(v["pred_is_recency"]) for v in rec_av) / len(rec_av) if rec_av else float("nan"),
            "recency_not_gold_available_n": len(rec_ng),
            "recency_not_gold_pick_pct": 100.0 * sum(int(v["pred_is_recency_not_gold"]) for v in rec_ng) / len(rec_ng) if rec_ng else float("nan"),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_ops_after_last_relevant": mean([v["ops_after_last_relevant"] for v in vals]),
            "simulation_match_pct": 100.0 * sum(int(v.get("simulation_matches_gold", 0)) for v in vals) / n if n else float("nan"),
        })
    return out


def pairwise(summary: list[dict[str, Any]], baseline: str) -> list[dict[str, Any]]:
    idx = {(r["label"], r["checkpoint"], r["group"]): r for r in summary}
    labels = sorted({r["label"] for r in summary if r["label"] != baseline})
    groups = sorted({r["group"] for r in summary})
    cks = sorted({r["checkpoint"] for r in summary})
    out = []
    for label in labels:
        for ck in cks:
            for g in groups:
                b = idx.get((baseline, ck, g)); a = idx.get((label, ck, g))
                if not a or not b:
                    continue
                out.append({
                    "label": label,
                    "baseline": baseline,
                    "checkpoint": ck,
                    "group": g,
                    "n": int(a["n"]),
                    "base_accuracy_pct": float(b["accuracy_pct"]),
                    "label_accuracy_pct": float(a["accuracy_pct"]),
                    "delta_accuracy_pct": float(a["accuracy_pct"]) - float(b["accuracy_pct"]),
                    "base_recency_pick_pct_among_available": float(b["recency_pick_pct_among_available"]),
                    "label_recency_pick_pct_among_available": float(a["recency_pick_pct_among_available"]),
                    "delta_recency_pick_pct_among_available": float(a["recency_pick_pct_among_available"]) - float(b["recency_pick_pct_among_available"]),
                    "base_recency_not_gold_pick_pct": float(b["recency_not_gold_pick_pct"]),
                    "label_recency_not_gold_pick_pct": float(a["recency_not_gold_pick_pct"]),
                    "delta_recency_not_gold_pick_pct": float(a["recency_not_gold_pick_pct"]) - float(b["recency_not_gold_pick_pct"]),
                })
    return out


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
        return f"{y:.2f}"
    except Exception:
        return "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="append", help="label,checkpoint,predictions_path")
    ap.add_argument("--baseline-label", default="chck82")
    ap.add_argument("--out-dir", default="experiments/archive/relation_learning/data/entity_recency_diagnostic")
    args = ap.parse_args()
    default_targets = [
        {"label": "chck82", "checkpoint": "eval", "predictions": str(CHCK82_PRED)},
        {"label": "coherent86", "checkpoint": "eval", "predictions": str(COHERENT86_PRED)},
        {"label": "binding_ep25_alpha0p50", "checkpoint": "eval", "predictions": str(ALPHA050_PRED)},
        {"label": "binding_ep25_alpha0p75", "checkpoint": "eval", "predictions": str(ALPHA075_PRED)},
        {"label": "binding_ep25_alpha1p00", "checkpoint": "eval", "predictions": str(ALPHA100_PRED)},
    ]
    targets = [parse_target(x) for x in args.target] if args.target else default_targets
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_prediction_rows(targets)
    summary = summarize(rows)
    deltas = pairwise(summary, args.baseline_label)
    write_csv(out_dir / "prediction_rows.csv", rows)
    write_csv(out_dir / "recency_summary.csv", summary)
    write_csv(out_dir / "recency_pairwise_deltas.csv", deltas)
    obj = {
        "status": "ENTITY_RECENCY_DIAGNOSTIC",
        "created_utc": now(),
        "targets": targets,
        "n_prediction_rows": len(rows),
        "summary_csv": rel(out_dir / "recency_summary.csv"),
        "deltas_csv": rel(out_dir / "recency_pairwise_deltas.csv"),
        "prediction_rows_csv": rel(out_dir / "prediction_rows.csv"),
        "interpretation_note": "recency_pick uses final state(s) of box(es) affected by the last operation; recency_not_gold_pick is the main wrong-recency bias when such a state is offered as a non-gold option.",
    }
    (out_dir / "summary.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research Entity last-operation recency diagnostic", "",
             "`rec_avail_pick` is the percent of rows where the prediction equals the final state of a box affected by the last operation, among rows where that state appears as an official option. `rec_not_gold_pick` restricts to cases where such a recency option is present but is not the gold answer.", "",
             "## Summary", "", "| label | group | n | acc | rec_avail_n | rec_avail_pick | rec_not_gold_n | rec_not_gold_pick |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        if r["group"] in KEY_GROUPS:
            lines.append(f"| {r['label']} | {r['group']} | {r['n']} | {fmt(r['accuracy_pct'])} | {r['recency_available_n']} | {fmt(r['recency_pick_pct_among_available'])} | {r['recency_not_gold_available_n']} | {fmt(r['recency_not_gold_pick_pct'])} |")
    lines += ["", "## Deltas vs baseline", "", f"Baseline label: `{args.baseline_label}`", "", "| label | group | n | d_acc | d_rec_avail_pick | d_rec_not_gold_pick |", "|---|---|---:|---:|---:|---:|"]
    for r in deltas:
        if r["group"] in KEY_GROUPS:
            lines.append(f"| {r['label']} | {r['group']} | {r['n']} | {fmt(r['delta_accuracy_pct'])} | {fmt(r['delta_recency_pick_pct_among_available'])} | {fmt(r['delta_recency_not_gold_pick_pct'])} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
