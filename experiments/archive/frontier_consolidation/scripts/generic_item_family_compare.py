#!/usr/bin/env python3
"""Generic cheap-discrete item/family comparator for official-compatible payloads.

Use after a per-target evaluation payload is complete. It compares a candidate
against a base using the validated research item parser, preserving item gains,
losses, and fragile EWoK/Entity families so aggregate cheap7 movement is not
mistaken for relation/state improvement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import sys
import time
from statistics import mean
from typing import Any

ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import DISCRETE_COLUMNS, PayloadLoader, compare_column  # noqa: E402

CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
FRAGILE_EWOK_TOKENS = [
    "material", "spatial", "space", "quantity", "quantitative", "number", "physical", "social",
    "support", "contact", "contained", "size", "mass", "volume", "distance", "dynamics", "interaction",
]
FRAGILE_ENTITY_TOKENS = ["regular_5", "move_contents_4", "move_contents_3", "ambiref_3", "5_ops", "4_ops", "3_ops"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path | str) -> str:
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def payload_ready(path: pathlib.Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    try:
        data = read_json(path)
    except Exception as e:
        return False, f"json_error:{e}"
    tasks = data.get("tasks", {}) if isinstance(data, dict) else {}
    required = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    missing = [r for r in required if r not in tasks or not isinstance(tasks.get(r), dict) or tasks[r].get("predictions") is None]
    # In normal payloads the task record has a predictions path, not inline predictions.
    # Some old records use score/report fields but still include a predictions path.
    missing = []
    for r in required:
        rec = tasks.get(r)
        if not isinstance(rec, dict):
            missing.append(r)
        elif rec.get("predictions") is None or rec.get("score") is None:
            missing.append(r)
    return (len(missing) == 0), ("ready" if not missing else "missing_or_incomplete:" + ",".join(missing))


def scores_from_payload(path: pathlib.Path) -> dict[str, float | None]:
    data = read_json(path)
    official = data.get("official_overall", {}).get("scores", {}) if isinstance(data, dict) else {}
    scores: dict[str, float | None] = {}
    for c in CHEAP:
        v = official.get(c)
        if v is not None:
            scores[c] = float(v)
        else:
            scores[c] = None
    if scores.get("GlobalPIQA") is None:
        gp = []
        tasks = data.get("tasks", {}) if isinstance(data, dict) else {}
        for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            rec = tasks.get(c, {})
            if isinstance(rec, dict) and rec.get("score") is not None:
                gp.append(float(rec["score"]))
        if len(gp) == 2:
            scores["GlobalPIQA"] = float(mean(gp))
    if scores.get("Reading") is None:
        rd = data.get("tasks", {}).get("Reading", {}) if isinstance(data, dict) else {}
        if isinstance(rd, dict) and isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
            scores["Reading"] = float(rd["scores"]["Reading"])
        elif isinstance(rd, dict) and rd.get("score") is not None:
            scores["Reading"] = float(rd["score"])
    return scores


def cheap7(scores: dict[str, float | None]) -> float | None:
    vals = [scores.get(c) for c in CHEAP]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def group_net_sum(groups: list[dict[str, Any]], tokens: list[str]) -> dict[str, Any]:
    tagged = []
    net = gains = losses = n = 0
    for g in groups:
        name = str(g.get("group", "")).lower()
        if any(tok in name for tok in tokens):
            tagged.append(g)
            net += int(g.get("net_gain_minus_loss", 0))
            gains += int(g.get("gain", 0) or 0)
            losses += int(g.get("loss", 0) or 0)
            n += int(g.get("n", 0) or 0)
    return {"n": n, "gain": gains, "loss": losses, "net_gain_minus_loss": net, "groups": tagged[:40]}


def compare_payloads(base: pathlib.Path, cand: pathlib.Path) -> dict[str, Any]:
    bl = PayloadLoader(base)
    cl = PayloadLoader(cand)
    cols = {col: compare_column(bl, cl, col) for col in DISCRETE_COLUMNS}
    aggregate = {
        "total_common_items": int(sum(cols[c]["n_common"] for c in DISCRETE_COLUMNS)),
        "total_gain_items": int(sum(cols[c]["flip_counts"].get("gain", 0) for c in DISCRETE_COLUMNS)),
        "total_loss_items": int(sum(cols[c]["flip_counts"].get("loss", 0) for c in DISCRETE_COLUMNS)),
        "mean_payload_delta_discrete6": float(mean(float(cols[c]["delta_score_payload"]) for c in DISCRETE_COLUMNS)),
        "mean_reconstructed_delta_discrete6": float(mean(float(cols[c]["delta_score_reconstructed_common"]) for c in DISCRETE_COLUMNS)),
    }
    aggregate["net_gain_minus_loss"] = aggregate["total_gain_items"] - aggregate["total_loss_items"]
    fragile = {
        "EWoK_fragile_tagged": group_net_sum(cols["EWoK"].get("groups_all_by_item_net", []), FRAGILE_EWOK_TOKENS),
        "Entity_highop_tagged": group_net_sum(cols["Entity"].get("groups_all_by_item_net", []), FRAGILE_ENTITY_TOKENS),
        "EWoK_worst12": cols["EWoK"].get("worst_groups_by_item_net", [])[:12],
        "EWoK_best12": cols["EWoK"].get("best_groups_by_item_net", [])[:12],
        "Entity_worst12": cols["Entity"].get("worst_groups_by_item_net", [])[:12],
        "Entity_best12": cols["Entity"].get("best_groups_by_item_net", [])[:12],
        "GlobalPIQA": {k: cols["GlobalPIQA"].get(k) for k in ["n_common", "gain_minus_loss_items", "flip_counts", "delta_score_payload", "delta_score_reconstructed_common"]},
    }
    return {"columns": cols, "aggregate": aggregate, "fragile_readouts": fragile}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-label", required=True)
    ap.add_argument("--base-payload", required=True)
    ap.add_argument("--candidate-label", required=True)
    ap.add_argument("--candidate-payload", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path(args.base_payload)
    cand = pathlib.Path(args.candidate_payload)
    base_ready, base_reason = payload_ready(base)
    cand_ready, cand_reason = payload_ready(cand)
    out: dict[str, Any] = {
        "status": "READY_TO_COMPARE" if base_ready and cand_ready else "PENDING_PAYLOADS",
        "created_utc": now(),
        "base": {"label": args.base_label, "payload": rel(base), "ready": base_ready, "reason": base_reason},
        "candidate": {"label": args.candidate_label, "payload": rel(cand), "ready": cand_ready, "reason": cand_reason},
        "scientific_reading": "Use this before escalating a route: aggregate cheap7 must be decomposed into item gains/losses and fragile EWoK/Entity movement.",
    }
    if base_ready and cand_ready:
        base_scores = scores_from_payload(base)
        cand_scores = scores_from_payload(cand)
        comp = compare_payloads(base, cand)
        out.update({
            "base_scores": base_scores,
            "candidate_scores": cand_scores,
            "base_cheap7": cheap7(base_scores),
            "candidate_cheap7": cheap7(cand_scores),
            "score_deltas": {c: (None if base_scores[c] is None or cand_scores[c] is None else float(cand_scores[c] - base_scores[c])) for c in CHEAP},
            "cheap7_delta": (None if cheap7(base_scores) is None or cheap7(cand_scores) is None else float(cheap7(cand_scores) - cheap7(base_scores))),
            "comparison": comp,
        })
        out["status"] = "COMPLETE"
    js = out_dir / "generic_item_family_compare.json"
    md = out_dir / "generic_item_family_compare.md"
    js.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        f"# research generic item/family compare: {args.candidate_label} minus {args.base_label}",
        "",
        f"Status: **{out['status']}**",
        "",
        f"Base: `{out['base']['payload']}` ({out['base']['reason']})",
        f"Candidate: `{out['candidate']['payload']}` ({out['candidate']['reason']})",
    ]
    if out["status"] == "COMPLETE":
        lines += [
            "",
            f"Cheap7 delta: `{out['cheap7_delta']}`",
            "",
            "| column | score delta | gains | losses | net |",
            "|---|---:|---:|---:|---:|",
        ]
        for col in DISCRETE_COLUMNS:
            c = out["comparison"]["columns"][col]
            lines.append(f"| {col} | {float(c['delta_score_payload']):+.6f} | {c['flip_counts'].get('gain',0)} | {c['flip_counts'].get('loss',0)} | {c['gain_minus_loss_items']:+d} |")
        lines += ["", "## Fragile readouts", "", "```json", json.dumps(out["comparison"]["fragile_readouts"], indent=2, ensure_ascii=False)[:12000], "```"]
    lines += ["", out["scientific_reading"], "", f"JSON: `{rel(js)}`"]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": rel(js), "out_md": rel(md), "cheap7_delta": out.get("cheap7_delta")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
