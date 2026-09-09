#!/usr/bin/env python3
"""Focused EWoK margin probe for research legal-40k 70M checkpoints.

This CPU/GPU-light diagnostic reuses the official-compatible research margin
exporter on the same 553 old-instability EWoK rows. It is directional only. research
showed that 10-20M screens are misleading and that even 70M is not a substitute
for full 100M official evaluation. Use this probe only after both legal-40k 70M
checkpoints exist to interpret relation dynamics, not to stop or select routes.
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

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/legal40k_chck70_ewok_margin_probe.py')
A01 = _public_path('experiments/archive/representation_and_objectives')
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')
EXPORTER_PATH = _public_path('experiments/archive/representation_and_objectives/scripts/official_ewok_margin_exporter.py')
SELECTION_CSV = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/ewok_margin_focus_selection.csv')
OLD_FOCUS_JSON = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553.json')
LEGAL16_100M_JSON = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe/strictsmalltok_chck_100M_focus553_margins.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/legal40k_chck70_ewok_margin_probe')

RUN_DIRS = {
    "legal40k70_43022": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_70M'),
    "legal40k70_43122": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43122/hf_model/chck_70M'),
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
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def sign(x: float, eps: float = 0.0) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def summarize_two_seed(records_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    akey, bkey = "legal40k70_43022", "legal40k70_43122"
    ra = records_by_model[akey]
    rb = records_by_model[bkey]
    by_uid_b = {r["uid"]: r for r in rb}
    rows = []
    m_a, m_b, d_ba = [], [], []
    same_sign = opposite_sign = zero_involved = 0
    both_correct = only_a = only_b = both_wrong = 0
    confident_opp = moderate_opp = near_both = 0
    by_domain: dict[str, dict[str, Any]] = {}
    for r in ra:
        uid = r["uid"]
        q = by_uid_b[uid]
        ma = float(r["margin_c0_minus_c1"])
        mb = float(q["margin_c0_minus_c1"])
        ca = int(r["correct"])
        cb = int(q["correct"])
        sa = sign(ma)
        sb = sign(mb)
        if sa == 0 or sb == 0:
            zero_involved += 1
        elif sa == sb:
            same_sign += 1
        else:
            opposite_sign += 1
        if ca and cb:
            both_correct += 1
        elif ca and not cb:
            only_a += 1
        elif cb and not ca:
            only_b += 1
        else:
            both_wrong += 1
        if ma > 2 and mb < -2:
            confident_opp += 1
        if ma > 1 and mb < -1:
            moderate_opp += 1
        if abs(ma) < 1 and abs(mb) < 1:
            near_both += 1
        m_a.append(ma); m_b.append(mb); d_ba.append(mb - ma)
        dom = r["domain"]
        bd = by_domain.setdefault(dom, {"n": 0, "correct_a": 0, "correct_b": 0, "opp": 0, "delta_margin": []})
        bd["n"] += 1; bd["correct_a"] += ca; bd["correct_b"] += cb
        if sa * sb < 0:
            bd["opp"] += 1
        bd["delta_margin"].append(mb - ma)
        rows.append({
            "uid": uid,
            "domain": dom,
            "idx": r.get("idx"),
            "ConceptA": r.get("ConceptA"),
            "ConceptB": r.get("ConceptB"),
            "ContextType": r.get("ContextType"),
            "ContextDiff": r.get("ContextDiff"),
            "TargetDiff": r.get("TargetDiff"),
            "selection_pattern": r.get("selection_pattern"),
            "selection_DiD_item": r.get("selection_DiD_item"),
            "margin_legal40k70_43022": ma,
            "margin_legal40k70_43122": mb,
            "delta_margin_431_minus_430": mb - ma,
            "correct_legal40k70_43022": ca,
            "correct_legal40k70_43122": cb,
            "pattern_legal40k70": f"{ca}{cb}",
        })
    domain_summary = {}
    for dom, bd in sorted(by_domain.items()):
        domain_summary[dom] = {
            "n": bd["n"],
            "acc_43022": bd["correct_a"] / bd["n"] * 100,
            "acc_43122": bd["correct_b"] / bd["n"] * 100,
            "opp_sign_frac": bd["opp"] / bd["n"],
            "delta_margin_431_minus_430_mean": statistics.mean(bd["delta_margin"]),
            "delta_margin_431_minus_430_median": statistics.median(bd["delta_margin"]),
        }
    return {
        "n_rows": len(rows),
        "accuracy_43022": sum(r["correct_legal40k70_43022"] for r in rows) / len(rows) * 100,
        "accuracy_43122": sum(r["correct_legal40k70_43122"] for r in rows) / len(rows) * 100,
        "both_correct": both_correct,
        "only_43022_correct": only_a,
        "only_43122_correct": only_b,
        "both_wrong": both_wrong,
        "same_sign_count": same_sign,
        "opposite_sign_count": opposite_sign,
        "zero_involved_count": zero_involved,
        "opposite_sign_frac": opposite_sign / len(rows),
        "moderate_opposite_430_gt1_431_lt_minus1_count": moderate_opp,
        "confident_opposite_430_gt2_431_lt_minus2_count": confident_opp,
        "both_abs_margin_lt1_count": near_both,
        "both_abs_margin_lt1_frac": near_both / len(rows),
        "margin_pearson_430_431": pearson(m_a, m_b),
        "delta_margin_431_minus_430_mean": statistics.mean(d_ba),
        "delta_margin_431_minus_430_median": statistics.median(d_ba),
        "delta_margin_431_minus_430_abs_mean": statistics.mean([abs(x) for x in d_ba]),
        "by_domain": domain_summary,
        "rows": rows,
    }


def compare_to_reference(new_rows: list[dict[str, Any]], ref_path: Path, ref_model_keys: tuple[str, str], label: str) -> dict[str, Any]:
    if not ref_path.exists():
        return {"status": "reference_missing", "label": label, "path": str(ref_path)}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    models = ref.get("results_by_model", {})
    if any(k not in models for k in ref_model_keys):
        return {"status": "reference_missing_models", "label": label, "available": sorted(models)}
    r430 = {r["uid"]: r for r in models[ref_model_keys[0]]["records"]}
    r431 = {r["uid"]: r for r in models[ref_model_keys[1]]["records"]}
    nm430=[]; nm431=[]; rm430=[]; rm431=[]; nd=[]; rd=[]; n=0
    for row in new_rows:
        uid = row["uid"]
        if uid not in r430 or uid not in r431:
            continue
        m430 = float(row["margin_legal40k70_43022"])
        m431 = float(row["margin_legal40k70_43122"])
        o430 = float(r430[uid]["margin_c0_minus_c1"])
        o431 = float(r431[uid]["margin_c0_minus_c1"])
        nm430.append(m430); nm431.append(m431); rm430.append(o430); rm431.append(o431)
        nd.append(m431 - m430); rd.append(o431 - o430); n += 1
    return {
        "status": "compared",
        "label": label,
        "n_aligned_rows": n,
        "corr_40k70_430_with_ref_430": pearson(nm430, rm430),
        "corr_40k70_431_with_ref_431": pearson(nm431, rm431),
        "corr_40k70_seed_delta_with_ref_seed_delta": pearson(nd, rd),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    mod = import_exporter()
    all_rows = mod.load_all_ewok_rows(mod.EWOK_DIR)
    selected = mod.load_selection(SELECTION_CSV, all_rows, max_items=None, domain_filter=None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preflight = {
        "selected_rows": len(selected),
        "selection_csv": str(SELECTION_CSV.relative_to(USER_ROOT)),
        "checkpoint": "70M",
        "model_paths": {k: str(v.relative_to(USER_ROOT)) for k, v in RUN_DIRS.items()},
        "model_path_exists": {k: v.exists() for k, v in RUN_DIRS.items()},
        "purpose": "directional 70M relation-dynamics probe only; full 100M official evaluation remains decisive.",
    }
    (_public_path('experiments/archive/representation_and_objectives/data/legal40k_chck70_ewok_margin_probe/legal40k_chck70_focus553_preflight.json')).write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.dry_run or not all(preflight["model_path_exists"].values()):
        status = "LEGAL40K_CHCK70_EWOK_MARGIN_DRYRUN" if args.dry_run else "WAITING_FOR_LEGAL40K_CHCK70"
        print(json.dumps({"status": status, **preflight}, indent=2))
        return
    t0 = time.time()
    device = torch.device(args.device)
    results_by_model: dict[str, Any] = {}
    for mk, mp in RUN_DIRS.items():
        print(json.dumps({"event": "score_model_start", "model_key": mk, "model_path": str(mp.relative_to(USER_ROOT)), "rows": len(selected), "device": str(device)}), flush=True)
        results_by_model[mk] = mod.score_rows_for_model(mk, mp, selected, device=device, non_causal_batch_size=args.non_causal_batch_size, official_predictions=None)
        print(json.dumps({"event": "score_model_done", "model_key": mk, **results_by_model[mk]["summary"]}, default=str), flush=True)
    records_by_model = {k: v["records"] for k, v in results_by_model.items()}
    pair = summarize_two_seed(records_by_model)
    oldcmp = compare_to_reference(pair["rows"], OLD_FOCUS_JSON, ("reinv430", "reinv431"), "inherited16k_endpoint")
    legal16cmp = compare_to_reference(pair["rows"], LEGAL16_100M_JSON, ("strict50_43022", "strict50_43122"), "legal16k_100M_focus_probe")
    payload = {
        "status": "LEGAL40K_CHCK70_EWOK_MARGIN_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "preflight": preflight,
        "results_by_model": results_by_model,
        "two_seed_focus553_analysis": {k: v for k, v in pair.items() if k != "rows"},
        "reference_comparisons": {
            "inherited16k_endpoint": oldcmp,
            "legal16k_100M_focus_probe": legal16cmp,
        },
        "interpretation": [
            "70M focused EWoK margins may help interpret relation dynamics because research calibration found 70M is the first plausibly informative checkpoint in existing ladders.",
            "This is still not endpoint evidence and must not be used to kill, select, or submit a model without full 100M official evaluation.",
        ],
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/legal40k_chck70_ewok_margin_probe/legal40k_chck70_focus553_margins.json')
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_csv = _public_path('experiments/archive/representation_and_objectives/data/legal40k_chck70_ewok_margin_probe/legal40k_chck70_focus553_pair_rows.csv')
    fields = list(pair["rows"][0].keys()) if pair["rows"] else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair["rows"])
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "out_csv": str(out_csv.relative_to(USER_ROOT)),
        "acc_43022": payload["two_seed_focus553_analysis"]["accuracy_43022"],
        "acc_43122": payload["two_seed_focus553_analysis"]["accuracy_43122"],
        "old_cmp": oldcmp,
        "legal16_cmp": legal16cmp,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
