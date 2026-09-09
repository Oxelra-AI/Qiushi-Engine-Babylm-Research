#!/usr/bin/env python3
"""research: old-tokenizer EWoK focused-margin phase probe at intermediate checkpoints.

Scores the same 553-row focused EWoK subset used in research on the inherited-
tokenizer compact_view_reinvest runs at chck_50M and chck_80M. This lets us
interpret the corrected-tokenizer chck_50M signal from research:
- If old chck_50M already resembles old chck_100M, then the corrected-tokenizer
  reversal is likely a representation/training-coordinate change.
- If old chck_50M also differs strongly from old chck_100M, then focused EWoK
  relation margins are late-forming and mid-training probes are not reliable
  endpoint selectors by themselves.

CPU-only; does not launch or touch training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/oldtok_checkpoint_ewok_phase_probe.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
A02 = _public_path('experiments/archive/frontier_consolidation')

EXPORTER_PATH = _public_path('experiments/archive/representation_and_objectives/scripts/official_ewok_margin_exporter.py')
SELECTION_CSV = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/ewok_margin_focus_selection.csv')
OLD100_JSON = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553.json')
NEW50_JSON = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_50M_focus553_margins.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe')

BASE_PATHS = {
    "oldtok430": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model'),
    "oldtok431": _public_path('experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122/hf_model'),
}


def import_exporter():
    spec = importlib.util.spec_from_file_location("margin_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXPORTER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = statistics.mean(xs); my = statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def sign(x: float, eps: float = 0.0) -> int:
    if x > eps: return 1
    if x < -eps: return -1
    return 0


def pair_summary(records430: list[dict[str, Any]], records431: list[dict[str, Any]], label: str) -> dict[str, Any]:
    b = {r["uid"]: r for r in records431}
    rows = []
    ma, mb, delta = [], [], []
    both_correct = only430 = only431 = both_wrong = 0
    opp = same = zero = 0
    moderate_opp_430pos431neg = confident_opp_430pos431neg = 0
    by_domain: dict[str, dict[str, Any]] = {}
    for r in records430:
        uid = r["uid"]
        q = b[uid]
        m430 = float(r["margin_c0_minus_c1"])
        m431 = float(q["margin_c0_minus_c1"])
        c430 = int(r["correct"]); c431 = int(q["correct"])
        if c430 and c431: both_correct += 1
        elif c430 and not c431: only430 += 1
        elif c431 and not c430: only431 += 1
        else: both_wrong += 1
        s430 = sign(m430); s431 = sign(m431)
        if s430 == 0 or s431 == 0: zero += 1
        elif s430 == s431: same += 1
        else: opp += 1
        if m430 > 1 and m431 < -1: moderate_opp_430pos431neg += 1
        if m430 > 2 and m431 < -2: confident_opp_430pos431neg += 1
        ma.append(m430); mb.append(m431); delta.append(m431 - m430)
        dom = r["domain"]
        bd = by_domain.setdefault(dom, {"n":0, "c430":0, "c431":0, "opp":0, "delta":[]})
        bd["n"] += 1; bd["c430"] += c430; bd["c431"] += c431
        if s430 * s431 < 0: bd["opp"] += 1
        bd["delta"].append(m431 - m430)
        rows.append({
            "label": label, "uid": uid, "domain": dom, "idx": r.get("idx"),
            "ConceptA": r.get("ConceptA"), "ConceptB": r.get("ConceptB"),
            "ContextType": r.get("ContextType"), "ContextDiff": r.get("ContextDiff"), "TargetDiff": r.get("TargetDiff"),
            "selection_pattern": r.get("selection_pattern"), "selection_DiD_item": r.get("selection_DiD_item"),
            "margin_430": m430, "margin_431": m431, "delta_431_minus_430": m431 - m430,
            "correct_430": c430, "correct_431": c431, "pattern": f"{c430}{c431}",
        })
    dom_summary = {}
    for dom, bd in sorted(by_domain.items()):
        dom_summary[dom] = {
            "n": bd["n"],
            "acc_430": bd["c430"] / bd["n"] * 100,
            "acc_431": bd["c431"] / bd["n"] * 100,
            "opp_sign_frac": bd["opp"] / bd["n"],
            "delta_431_minus_430_mean": statistics.mean(bd["delta"]),
            "delta_431_minus_430_median": statistics.median(bd["delta"]),
        }
    return {
        "label": label,
        "n_rows": len(rows),
        "accuracy_430": sum(x["correct_430"] for x in rows) / len(rows) * 100,
        "accuracy_431": sum(x["correct_431"] for x in rows) / len(rows) * 100,
        "both_correct": both_correct,
        "only_430_correct": only430,
        "only_431_correct": only431,
        "both_wrong": both_wrong,
        "same_sign_count": same,
        "opposite_sign_count": opp,
        "zero_involved_count": zero,
        "opposite_sign_frac": opp / len(rows),
        "moderate_430pos_431neg_count": moderate_opp_430pos431neg,
        "confident_430pos_431neg_count": confident_opp_430pos431neg,
        "margin_pearson_430_431": pearson(ma, mb),
        "delta_431_minus_430_mean": statistics.mean(delta),
        "delta_431_minus_430_median": statistics.median(delta),
        "delta_431_minus_430_abs_mean": statistics.mean([abs(x) for x in delta]),
        "by_domain": dom_summary,
        "rows": rows,
    }


def load_pair_from_step49(json_path: Path, model430: str, model431: str, label: str) -> dict[str, Any] | None:
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rbm = data.get("results_by_model", {})
    if model430 not in rbm or model431 not in rbm:
        return None
    return pair_summary(rbm[model430]["records"], rbm[model431]["records"], label)


def compare_pair_deltas(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    bs = {r["uid"]: r for r in source["rows"]}
    bt = {r["uid"]: r for r in target["rows"]}
    uids = sorted(set(bs) & set(bt))
    sd = [bs[u]["delta_431_minus_430"] for u in uids]
    td = [bt[u]["delta_431_minus_430"] for u in uids]
    sign_agree = sum(1 for a,b in zip(sd,td) if sign(a) == sign(b))
    return {
        "source_label": source["label"],
        "target_label": target["label"],
        "n": len(uids),
        "delta_corr": pearson(sd, td),
        "delta_sign_agreement_frac": sign_agree / len(uids) if uids else None,
        "source_delta_mean": statistics.mean(sd) if sd else None,
        "target_delta_mean": statistics.mean(td) if td else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["50M", "80M"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    mod = import_exporter()
    all_rows = mod.load_all_ewok_rows(mod.EWOK_DIR)
    selected = mod.load_selection(SELECTION_CSV, all_rows, max_items=None, domain_filter=None)
    paths = {f"oldtok{seed}_{ck}": base / f"chck_{ck}"
             for ck in args.checkpoints for seed, base in [("430", BASE_PATHS["oldtok430"]), ("431", BASE_PATHS["oldtok431"])]}
    preflight = {
        "selected_rows": len(selected),
        "checkpoints": args.checkpoints,
        "paths": {k: str(v.relative_to(USER_ROOT)) for k,v in paths.items()},
        "exists": {k: v.exists() for k,v in paths.items()},
        "method": "research official-compatible EWoK margin scorer on old-tokenizer intermediate compact_view_reinvest checkpoints.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_phase_preflight.json')).write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status":"OLDTOK_PHASE_PROBE_DRYRUN", **preflight}, indent=2))
        return
    missing = [k for k,e in preflight["exists"].items() if not e]
    if missing:
        raise FileNotFoundError({"missing": missing, "preflight": preflight})

    t0 = time.time()
    device = torch.device(args.device)
    all_results: dict[str, Any] = {}
    phase_pairs: dict[str, Any] = {}
    for ck in args.checkpoints:
        pair_records = {}
        for seed in ["430", "431"]:
            key = f"oldtok{seed}_{ck}"
            mp = paths[key]
            print(json.dumps({"event":"score_model_start", "model_key": key, "path": str(mp.relative_to(USER_ROOT)), "rows": len(selected)}), flush=True)
            res = mod.score_rows_for_model(key, mp, selected, device=device, non_causal_batch_size=args.non_causal_batch_size, official_predictions=None)
            all_results[key] = res
            pair_records[seed] = res["records"]
            print(json.dumps({"event":"score_model_done", "model_key": key, **res["summary"]}, default=str), flush=True)
        phase_pairs[f"oldtok_{ck}"] = pair_summary(pair_records["430"], pair_records["431"], f"oldtok_{ck}")

    old100_pair = load_pair_from_step49(OLD100_JSON, "reinv430", "reinv431", "oldtok_100M")
    new50_pair = None
    if NEW50_JSON.exists():
        nd = json.loads(NEW50_JSON.read_text(encoding="utf-8"))
        rbm = nd.get("results_by_model", {})
        if "strict50_43022" in rbm and "strict50_43122" in rbm:
            new50_pair = pair_summary(rbm["strict50_43022"]["records"], rbm["strict50_43122"]["records"], "strictsmalltok_50M")

    comparisons = []
    labels = list(phase_pairs.keys())
    if old100_pair:
        for p in phase_pairs.values():
            comparisons.append(compare_pair_deltas(p, old100_pair))
    if new50_pair:
        for p in phase_pairs.values():
            comparisons.append(compare_pair_deltas(p, new50_pair))
    if len(labels) >= 2:
        comparisons.append(compare_pair_deltas(phase_pairs[labels[0]], phase_pairs[labels[1]]))

    payload = {
        "status": "OLDTOK_CHECKPOINT_EWOK_PHASE_PROBE",
        "elapsed_sec": round(time.time() - t0, 1),
        "preflight": preflight,
        "model_summaries": {k: v["summary"] for k,v in all_results.items()},
        "phase_pair_summaries": {k: {kk: vv for kk,vv in v.items() if kk != "rows"} for k,v in phase_pairs.items()},
        "comparisons": comparisons,
        "interpretation": [
            "This CPU probe tests phase dynamics on the old inherited-tokenizer coordinate; it is not an endpoint score.",
            "The key comparison is whether oldtok_50M and oldtok_80M seed deltas resemble oldtok_100M endpoint deltas or the corrected-tokenizer 50M deltas.",
            "If old intermediate deltas do not predict old endpoint deltas, mid-training focused EWoK cannot be used alone to choose a final seed.",
        ],
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_focus553_phase_probe.json')
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_csv = _public_path('experiments/archive/representation_and_objectives/data/oldtok_checkpoint_ewok_phase_probe/oldtok_checkpoint_focus553_pair_rows.csv')
    rows = []
    for p in phase_pairs.values():
        rows.extend(p["rows"])
    if rows:
        fields = list(rows[0].keys())
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    compact = {
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "out_csv": str(out_csv.relative_to(USER_ROOT)),
        "phase_pair_summaries": payload["phase_pair_summaries"],
        "comparisons": comparisons,
    }
    print(json.dumps(compact, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
