#!/usr/bin/env python3
"""research: stratify format-endpoint item losses by edge versus interior differences.

The coherent-special endpoints improved short isolated likelihood mostly through
special-token/edge geometry, yet lost Supplement and EWoK.  This script tests
whether the lost official items are concentrated in minimal pairs whose alternatives
differ near the first or last content tokens, where boundary-token redistribution
would not cancel, or whether losses are spread across interior-differing pairs.

Only saved official payloads and evaluation data are read; no model inference is
performed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import difflib
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

ROOT = _public_path('.')
DEFAULT_OUT = _public_path('experiments/archive/relation_learning/data/item_edge_flip_stratification')

DEFAULT_PAYLOADS = {
    "chck82": _public_path('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json'),
    "coherent86_s43022": _public_path('experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json'),
    "coherent_special_98097": _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval/coherent_unsplit_special_seed98097_alpha0p75/per_target/coherent_unsplit_special_seed98097_alpha0p75.json'),
    "coherent_special_98098": _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/coherent_unsplit_special/eval/coherent_unsplit_special_seed98098_alpha0p75/per_target/coherent_unsplit_special_seed98098_alpha0p75.json'),
    "half_98097": _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/half_coherent_half_isolated/eval/half_coherent_half_isolated_seed98097_alpha0p75/per_target/half_coherent_half_isolated_seed98097_alpha0p75.json'),
    "half_98098": _public_path('experiments/archive/relation_learning/data/eval_format_replay_corrected/half_coherent_half_isolated/eval/half_coherent_half_isolated_seed98098_alpha0p75/per_target/half_coherent_half_isolated_seed98098_alpha0p75.json'),
}
COLUMNS = ["Supplement", "EWoK"]
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\sA-Za-z0-9]")
SPACE_RE = re.compile(r"\s+")


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def norm_text(x: Any) -> str:
    return SPACE_RE.sub(" ", str(x).strip())


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(norm_text(text)) if not re.fullmatch(r"\s+", m.group(0))]


def pred_nested(path: pathlib.Path) -> dict[str, list[dict[str, Any]]]:
    raw = read_json(path)
    out: dict[str, list[dict[str, Any]]] = {}
    for uid, rec in raw.items():
        if isinstance(rec, dict):
            out[str(uid)] = rec.get("predictions", [])
        elif isinstance(rec, list):
            out[str(uid)] = rec
        else:
            out[str(uid)] = []
    return out


def get_pred(preds: dict[str, list[dict[str, Any]]], uid: str, idx: int) -> str | None:
    arr = preds.get(str(uid))
    if arr is None or idx >= len(arr):
        return None
    return norm_text(arr[idx].get("pred"))


def diff_positions(a: list[str], b: list[str]) -> tuple[list[int], list[int]]:
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    apos: list[int] = []
    bpos: list[int] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if i1 < i2:
            apos.extend(range(i1, i2))
        else:
            # Insertion relative to a: mark the adjacent token position if possible.
            if a:
                apos.append(min(max(i1, 0), len(a) - 1))
        if j1 < j2:
            bpos.extend(range(j1, j2))
        else:
            if b:
                bpos.append(min(max(j1, 0), len(b) - 1))
    return sorted(set(apos)), sorted(set(bpos))


def edge_distance(pos: int, n: int) -> int:
    return int(min(pos, max(0, n - 1 - pos))) if n > 0 else 0


def classify_pair(good: str, bad: str, k: int) -> dict[str, Any]:
    gt = tokens(good)
    bt = tokens(bad)
    gp, bp = diff_positions(gt, bt)
    distances = [edge_distance(p, len(gt)) for p in gp] + [edge_distance(p, len(bt)) for p in bp]
    if not distances:
        cls = "no_word_diff"
        edge_touch = False
        interior_touch = False
        min_dist = None
    else:
        edge_touch = any(d < k for d in distances)
        interior_touch = any(d >= k for d in distances)
        if edge_touch and not interior_touch:
            cls = "edge_only"
        elif edge_touch and interior_touch:
            cls = "edge_and_interior"
        else:
            cls = "interior_only"
        min_dist = int(min(distances))
    return {
        f"edge_class_k{k}": cls,
        f"edge_touch_k{k}": bool(edge_touch),
        f"interior_touch_k{k}": bool(interior_touch),
        f"min_diff_edge_distance_k{k}": min_dist,
        "good_token_len": len(gt),
        "bad_token_len": len(bt),
        "n_good_diff_tokens": len(gp),
        "n_bad_diff_tokens": len(bp),
        "good_diff_positions": " ".join(map(str, gp[:20])),
        "bad_diff_positions": " ".join(map(str, bp[:20])),
    }


def task_score(payload: dict[str, Any], column: str) -> float | None:
    scores = payload.get("official_overall", {}).get("scores", {})
    return scores.get(column)


def load_column_items(payload_path: pathlib.Path, column: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = read_json(payload_path)
    tasks = payload["tasks"]
    if column == "Supplement":
        rec = tasks[column]
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        counters: Counter[str] = Counter()
        rows: dict[str, dict[str, Any]] = {}
        missing = 0
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                uid = str(raw["UID"]) if "field" in raw and "UID" in raw else file_path.stem
                idx = counters[uid]
                counters[uid] += 1
                pred = get_pred(preds, uid, idx)
                if pred is None:
                    missing += 1
                good = norm_text(raw["sentence_good"])
                bad = norm_text(raw["sentence_bad"])
                item_id = f"Supplement:{uid}:{idx}"
                meta = {
                    "column": "Supplement",
                    "uid": uid,
                    "group": uid,
                    "idx": idx,
                    "file": file_path.stem,
                    "contrast": raw.get("contrast", ""),
                    "row": raw.get("row", ""),
                    "good": good,
                    "bad": bad,
                    "pred": pred,
                    "correct": bool(pred == good),
                }
                for k in (3, 5):
                    meta.update(classify_pair(good, bad, k))
                rows[item_id] = meta
        return rows, {"missing_predictions": missing, "payload_score": task_score(payload, column), "n_items": len(rows), "uids": len(counters)}
    if column == "EWoK":
        rec = tasks[column]
        preds = pred_nested(ROOT / rec["predictions"])
        data_dir = ROOT / rec["data_path"]
        counters: Counter[str] = Counter()
        rows: dict[str, dict[str, Any]] = {}
        missing = 0
        for file_path in sorted(data_dir.glob("*.jsonl")):
            for raw in read_jsonl(file_path):
                uid = str(raw["Domain"])
                idx = counters[uid]
                counters[uid] += 1
                pred = get_pred(preds, uid, idx)
                if pred is None:
                    missing += 1
                good = norm_text(" ".join([raw["Context1"], raw["Target1"]]))
                bad = norm_text(" ".join([raw["Context2"], raw["Target1"]]))
                item_id = f"EWoK:{uid}:{idx}"
                meta = {
                    "column": "EWoK",
                    "uid": uid,
                    "group": uid,
                    "idx": idx,
                    "file": file_path.stem,
                    "ContextType": raw.get("ContextType", ""),
                    "ContextDiff": raw.get("ContextDiff", ""),
                    "TargetDiff": raw.get("TargetDiff", ""),
                    "ConceptA": raw.get("ConceptA", ""),
                    "ConceptB": raw.get("ConceptB", ""),
                    "good": good,
                    "bad": bad,
                    "pred": pred,
                    "correct": bool(pred == good),
                }
                for k in (3, 5):
                    meta.update(classify_pair(good, bad, k))
                rows[item_id] = meta
        return rows, {"missing_predictions": missing, "payload_score": task_score(payload, column), "n_items": len(rows), "uids": len(counters)}
    raise KeyError(column)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def stderr_prop(k: int, n: int) -> float:
    if n <= 0:
        return float("nan")
    p = k / n
    return math.sqrt(p * (1 - p) / n)


def compare_items(anchor_label: str, cand_label: str, anchor: dict[str, dict[str, Any]], cand: dict[str, dict[str, Any]], k: int, column: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = sorted(set(anchor) & set(cand))
    event_counts: Counter[str] = Counter()
    strata: dict[str, Counter[str]] = defaultdict(Counter)
    group_rows: dict[str, Counter[str]] = defaultdict(Counter)
    item_rows: list[dict[str, Any]] = []
    for item_id in common:
        a = anchor[item_id]
        c = cand[item_id]
        if a["correct"] and not c["correct"]:
            event = "loss"
        elif (not a["correct"]) and c["correct"]:
            event = "gain"
        elif a["correct"] and c["correct"]:
            event = "both_correct"
        else:
            event = "both_wrong"
        cls = str(a[f"edge_class_k{k}"])
        edge_touch = bool(a[f"edge_touch_k{k}"])
        event_counts[event] += 1
        strata[cls][event] += 1
        strata[cls]["n"] += 1
        if edge_touch:
            strata["edge_touch"][event] += 1
            strata["edge_touch"]["n"] += 1
        else:
            strata["not_edge_touch"][event] += 1
            strata["not_edge_touch"]["n"] += 1
        g = str(a["group"])
        group_rows[g][event] += 1
        group_rows[g]["n"] += 1
        if event in ("loss", "gain"):
            item_rows.append({
                "comparison": f"{cand_label}_minus_{anchor_label}",
                "column": column,
                "k": k,
                "item_id": item_id,
                "event": event,
                "group": g,
                "edge_class": cls,
                "edge_touch": edge_touch,
                "min_diff_edge_distance": a.get(f"min_diff_edge_distance_k{k}"),
                "good_token_len": a.get("good_token_len"),
                "bad_token_len": a.get("bad_token_len"),
                "n_good_diff_tokens": a.get("n_good_diff_tokens"),
                "n_bad_diff_tokens": a.get("n_bad_diff_tokens"),
                "anchor_pred": a.get("pred"),
                "candidate_pred": c.get("pred"),
                "good": a.get("good"),
                "bad": a.get("bad"),
                "file": a.get("file", ""),
                "contrast": a.get("contrast", ""),
                "ContextType": a.get("ContextType", ""),
                "ContextDiff": a.get("ContextDiff", ""),
                "TargetDiff": a.get("TargetDiff", ""),
            })
    base_correct = event_counts["both_correct"] + event_counts["loss"]
    base_wrong = event_counts["both_wrong"] + event_counts["gain"]
    edge_base_correct = strata["edge_touch"]["both_correct"] + strata["edge_touch"]["loss"]
    nonedge_base_correct = strata["not_edge_touch"]["both_correct"] + strata["not_edge_touch"]["loss"]
    edge_base_wrong = strata["edge_touch"]["both_wrong"] + strata["edge_touch"]["gain"]
    nonedge_base_wrong = strata["not_edge_touch"]["both_wrong"] + strata["not_edge_touch"]["gain"]
    loss_edge_frac = strata["edge_touch"]["loss"] / event_counts["loss"] if event_counts["loss"] else float("nan")
    gain_edge_frac = strata["edge_touch"]["gain"] / event_counts["gain"] if event_counts["gain"] else float("nan")
    base_correct_edge_frac = edge_base_correct / base_correct if base_correct else float("nan")
    all_edge_frac = strata["edge_touch"]["n"] / len(common) if common else float("nan")
    edge_loss_rate = strata["edge_touch"]["loss"] / edge_base_correct if edge_base_correct else float("nan")
    nonedge_loss_rate = strata["not_edge_touch"]["loss"] / nonedge_base_correct if nonedge_base_correct else float("nan")
    edge_gain_rate = strata["edge_touch"]["gain"] / edge_base_wrong if edge_base_wrong else float("nan")
    nonedge_gain_rate = strata["not_edge_touch"]["gain"] / nonedge_base_wrong if nonedge_base_wrong else float("nan")
    stratum_rows: list[dict[str, Any]] = []
    for cls, cnt in sorted(strata.items()):
        n = cnt["n"]
        bc = cnt["both_correct"] + cnt["loss"]
        bw = cnt["both_wrong"] + cnt["gain"]
        stratum_rows.append({
            "stratum": cls,
            "n": int(n),
            "both_correct": int(cnt["both_correct"]),
            "both_wrong": int(cnt["both_wrong"]),
            "gain": int(cnt["gain"]),
            "loss": int(cnt["loss"]),
            "net_gain_minus_loss": int(cnt["gain"] - cnt["loss"]),
            "base_correct": int(bc),
            "base_wrong": int(bw),
            "loss_rate_given_base_correct": None if bc == 0 else cnt["loss"] / bc,
            "gain_rate_given_base_wrong": None if bw == 0 else cnt["gain"] / bw,
        })
    group_summaries: list[dict[str, Any]] = []
    for g, cnt in group_rows.items():
        n = cnt["n"]
        group_summaries.append({
            "group": g,
            "n": int(n),
            "gain": int(cnt["gain"]),
            "loss": int(cnt["loss"]),
            "net_gain_minus_loss": int(cnt["gain"] - cnt["loss"]),
            "base_correct": int(cnt["both_correct"] + cnt["loss"]),
            "candidate_correct": int(cnt["both_correct"] + cnt["gain"]),
            "net_pct": 100.0 * (cnt["gain"] - cnt["loss"]) / n if n else 0.0,
        })
    group_summaries.sort(key=lambda r: (r["net_pct"], r["net_gain_minus_loss"]))
    summary = {
        "comparison": f"{cand_label}_minus_{anchor_label}",
        "column": column,
        "k": k,
        "n_common": int(len(common)),
        "events": dict(event_counts),
        "net_gain_minus_loss": int(event_counts["gain"] - event_counts["loss"]),
        "reconstructed_delta_pct_points": 100.0 * (event_counts["gain"] - event_counts["loss"]) / len(common) if common else float("nan"),
        "all_edge_touch_fraction": all_edge_frac,
        "base_correct_edge_touch_fraction": base_correct_edge_frac,
        "loss_edge_touch_fraction": loss_edge_frac,
        "gain_edge_touch_fraction": gain_edge_frac,
        "loss_edge_fraction_minus_base_correct_edge_fraction": None if math.isnan(loss_edge_frac) or math.isnan(base_correct_edge_frac) else loss_edge_frac - base_correct_edge_frac,
        "edge_loss_rate_given_base_correct": edge_loss_rate,
        "not_edge_loss_rate_given_base_correct": nonedge_loss_rate,
        "edge_gain_rate_given_base_wrong": edge_gain_rate,
        "not_edge_gain_rate_given_base_wrong": nonedge_gain_rate,
        "loss_rate_edge_minus_not_edge": None if math.isnan(edge_loss_rate) or math.isnan(nonedge_loss_rate) else edge_loss_rate - nonedge_loss_rate,
        "strata": stratum_rows,
        "worst_groups": group_summaries[:8],
        "best_groups": list(reversed(group_summaries[-8:])),
        "examples": {
            "loss_edge": [r for r in item_rows if r["event"] == "loss" and r["edge_touch"]][:5],
            "loss_interior": [r for r in item_rows if r["event"] == "loss" and not r["edge_touch"]][:5],
            "gain_edge": [r for r in item_rows if r["event"] == "gain" and r["edge_touch"]][:5],
            "gain_interior": [r for r in item_rows if r["event"] == "gain" and not r["edge_touch"]][:5],
        },
    }
    return summary, item_rows


def load_payloads(extra_specs: list[str]) -> dict[str, pathlib.Path]:
    payloads = {k: pathlib.Path(v) for k, v in DEFAULT_PAYLOADS.items()}
    for spec in extra_specs:
        if "=" not in spec:
            raise SystemExit(f"--payload must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        payloads[label] = pathlib.Path(path)
    return {k: v for k, v in payloads.items() if v.exists()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--anchor", default="coherent86_s43022")
    ap.add_argument("--payload", action="append", default=[])
    ap.add_argument("--candidate", action="append", default=[], help="candidate labels; default all non-anchor non-chck82 available endpoints")
    ap.add_argument("--columns", nargs="*", default=COLUMNS)
    ap.add_argument("--edge-k", nargs="*", type=int, default=[3, 5])
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(args.payload)
    if args.anchor not in payloads:
        raise SystemExit(f"anchor {args.anchor!r} not found; available={sorted(payloads)}")
    candidates = args.candidate or [k for k in payloads if k not in {args.anchor, "chck82"}]
    candidates = [k for k in candidates if k in payloads and k != args.anchor]
    if not candidates:
        raise SystemExit("no candidates available")

    loaded: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    needed = [args.anchor] + candidates
    for label in needed:
        for col in args.columns:
            rows, m = load_column_items(payloads[label], col)
            loaded[(label, col)] = rows
            meta[(label, col)] = m
    summaries: list[dict[str, Any]] = []
    item_rows_all: list[dict[str, Any]] = []
    for cand in candidates:
        for col in args.columns:
            for k in args.edge_k:
                s, rows = compare_items(args.anchor, cand, loaded[(args.anchor, col)], loaded[(cand, col)], k, col)
                s["anchor_payload_score"] = meta[(args.anchor, col)].get("payload_score")
                s["candidate_payload_score"] = meta[(cand, col)].get("payload_score")
                if s["anchor_payload_score"] is not None and s["candidate_payload_score"] is not None:
                    s["payload_delta"] = float(s["candidate_payload_score"]) - float(s["anchor_payload_score"])
                else:
                    s["payload_delta"] = None
                summaries.append(s)
                item_rows_all.extend(rows)
    write_csv(out_dir / "flip_items_edge_strata.csv", item_rows_all)
    flat_strata: list[dict[str, Any]] = []
    for s in summaries:
        for r in s["strata"]:
            q = {kk: vv for kk, vv in s.items() if kk not in {"strata", "worst_groups", "best_groups", "examples"}}
            q.update(r)
            flat_strata.append(q)
    write_csv(out_dir / "edge_strata_summary.csv", flat_strata)
    out = {
        "status": "ITEM_EDGE_FLIP_STRATIFICATION_DONE",
        "created_utc": now(),
        "anchor": args.anchor,
        "payloads": {k: rel(v) for k, v in payloads.items()},
        "candidates": candidates,
        "columns": args.columns,
        "edge_k": args.edge_k,
        "payload_meta": {f"{label}:{col}": meta[(label, col)] for label in needed for col in args.columns},
        "summaries": summaries,
        "outputs": {
            "json": rel(out_dir / "item_edge_flip_stratification.json"),
            "md": rel(out_dir / "item_edge_flip_stratification.md"),
            "items_csv": rel(out_dir / "flip_items_edge_strata.csv"),
            "strata_csv": rel(out_dir / "edge_strata_summary.csv"),
        },
        "reading": "If losses are enriched among edge-touching pairs beyond their base-correct fraction, boundary-token redistribution is implicated; if not, losses are not localized by this edge test.",
    }
    out_json = out_dir / "item_edge_flip_stratification.json"
    out_md = out_dir / "item_edge_flip_stratification.md"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research item flip edge stratification",
        "",
        out["reading"],
        "",
        f"Anchor: `{args.anchor}`.",
        "",
        "## Primary summaries for first/last 3-token touch",
        "",
        "| comparison | column | payload Δ | reconstructed Δ | gains | losses | loss edge frac | base-correct edge frac | edge loss rate | non-edge loss rate | worst groups |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        if s["k"] != 3:
            continue
        worst = ", ".join(f"{g['group']} {g['net_gain_minus_loss']:+d}/{g['n']}" for g in s["worst_groups"][:4])
        pd = s.get("payload_delta")
        lines.append(
            f"| {s['comparison']} | {s['column']} | {pd if pd is not None else float('nan'):+.3f} | {s['reconstructed_delta_pct_points']:+.3f} | "
            f"{s['events'].get('gain', 0)} | {s['events'].get('loss', 0)} | {s['loss_edge_touch_fraction']:.3f} | {s['base_correct_edge_touch_fraction']:.3f} | "
            f"{s['edge_loss_rate_given_base_correct']:.3f} | {s['not_edge_loss_rate_given_base_correct']:.3f} | {worst} |"
        )
    lines += ["", "## Notes", "", "- `edge_touch` means at least one word-level difference between the alternatives lies in the first or last k content tokens of either sentence/string.", "- EWoK follows the validated research reconstruction: good is `Context1 Target1`, bad is `Context2 Target1`.", "- Full examples and k=5 summaries are in the JSON and CSV files.", "", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "out_json": rel(out_json),
        "out_md": rel(out_md),
        "n_item_flip_rows": len(item_rows_all),
        "candidates": candidates,
        "summary_excerpt": [s for s in summaries if s["k"] == 3][:8],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
