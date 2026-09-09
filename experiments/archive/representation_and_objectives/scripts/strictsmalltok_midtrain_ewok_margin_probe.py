#!/usr/bin/env python3
"""research: corrected-tokenizer mid-training EWoK relation-margin probe.

CPU-only probe on already-saved research corrected-tokenizer checkpoints, defaulting
to chck_50M for seeds 43022 and 43122. It reuses the official-compatible margin
scoring functions validated in research, but does not touch the long-running H100
retrain processes and does not decide endpoint performance.

Scientific purpose:
- Determine whether the Strict-Small-trained tokenizer runs already show a
  seed-dependent relation preference phenotype on the focused 553 EWoK rows.
- Compare that mid-training phenotype to the inherited-tokenizer endpoint margins
  from research, to see whether early focused EWoK margins might be useful later as
  a cheap seed-selection/stability signal. This is not a substitute for full
  official evaluation after 100M completion.
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

HERE = _public_path('experiments/archive/representation_and_objectives/scripts/strictsmalltok_midtrain_ewok_margin_probe.py')
A01 = _public_path('experiments/archive/representation_and_objectives')  # experiments/archive/representation_and_objectives
USER_ROOT = _public_path('.')
WORKSPACE = _public_path('experiments/archive/representation_and_objectives')

EXPORTER_PATH = _public_path('experiments/archive/representation_and_objectives/scripts/official_ewok_margin_exporter.py')
SELECTION_CSV = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/ewok_margin_focus_selection.csv')
OLD_FOCUS_JSON = _public_path('experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553.json')
OUT_DIR = _public_path('experiments/archive/representation_and_objectives/data/strictsmalltok_midtrain_ewok_margin_probe')

RUN_DIRS = {
    "strict50_43022": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_50M'),
    "strict50_43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_50M'),
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
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
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
    akey, bkey = "strict50_43022", "strict50_43122"
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
            "margin_strict50_43022": ma,
            "margin_strict50_43122": mb,
            "delta_margin_431_minus_430": mb - ma,
            "correct_strict50_43022": ca,
            "correct_strict50_43122": cb,
            "pattern_strict50": f"{ca}{cb}",
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
        "accuracy_43022": sum(r["correct_strict50_43022"] for r in rows) / len(rows) * 100,
        "accuracy_43122": sum(r["correct_strict50_43122"] for r in rows) / len(rows) * 100,
        "both_correct": both_correct,
        "only_43022_correct": only_a,
        "only_43122_correct": only_b,
        "both_wrong": both_wrong,
        "same_sign_count": same_sign,
        "opposite_sign_count": opposite_sign,
        "zero_involved_count": zero_involved,
        "opposite_sign_frac": opposite_sign / len(rows),
        "moderate_opposite_reinv430_gt1_reinv431_lt_minus1_count": moderate_opp,
        "confident_opposite_reinv430_gt2_reinv431_lt_minus2_count": confident_opp,
        "both_abs_margin_lt1_count": near_both,
        "both_abs_margin_lt1_frac": near_both / len(rows),
        "margin_pearson_430_431": pearson(m_a, m_b),
        "delta_margin_431_minus_430_mean": statistics.mean(d_ba),
        "delta_margin_431_minus_430_median": statistics.median(d_ba),
        "delta_margin_431_minus_430_abs_mean": statistics.mean([abs(x) for x in d_ba]),
        "by_domain": domain_summary,
        "rows": rows,
    }


def compare_to_old_focus(new_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not OLD_FOCUS_JSON.exists():
        return {"status": "old_focus_json_missing", "path": str(OLD_FOCUS_JSON)}
    old = json.loads(OLD_FOCUS_JSON.read_text(encoding="utf-8"))
    old_models = old.get("results_by_model", {})
    needed = ["reinv430", "reinv431"]
    if any(k not in old_models for k in needed):
        return {"status": "old_focus_missing_models", "available": sorted(old_models)}
    old430 = {r["uid"]: r for r in old_models["reinv430"]["records"]}
    old431 = {r["uid"]: r for r in old_models["reinv431"]["records"]}
    nm430 = []
    nm431 = []
    od430 = []
    od431 = []
    n_delta = []
    o_delta = []
    sign_agree430 = sign_agree431 = 0
    n = 0
    for r in new_rows:
        uid = r["uid"]
        if uid not in old430 or uid not in old431:
            continue
        m430 = float(r["margin_strict50_43022"])
        m431 = float(r["margin_strict50_43122"])
        o430 = float(old430[uid]["margin_c0_minus_c1"])
        o431 = float(old431[uid]["margin_c0_minus_c1"])
        nm430.append(m430); nm431.append(m431); od430.append(o430); od431.append(o431)
        n_delta.append(m431 - m430); o_delta.append(o431 - o430)
        sign_agree430 += int(sign(m430) == sign(o430))
        sign_agree431 += int(sign(m431) == sign(o431))
        n += 1
    return {
        "status": "compared",
        "n_aligned_rows": n,
        "corr_new50_430_with_old100_430": pearson(nm430, od430),
        "corr_new50_431_with_old100_431": pearson(nm431, od431),
        "corr_new50_seed_delta_with_old100_seed_delta": pearson(n_delta, o_delta),
        "sign_agreement_430_frac": sign_agree430 / n if n else None,
        "sign_agreement_431_frac": sign_agree431 / n if n else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="50M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=8)
    ap.add_argument("--non-causal-batch-size", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    mod = import_exporter()
    all_rows = mod.load_all_ewok_rows(mod.EWOK_DIR)
    selected = mod.load_selection(SELECTION_CSV, all_rows, max_items=None, domain_filter=None)

    model_paths = {
        "strict50_43022": WORKSPACE / f"training/runs/strictsmalltok_compact_view_reinvest_seed43022/hf_model/chck_{args.checkpoint}",
        "strict50_43122": WORKSPACE / f"training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_{args.checkpoint}",
    }
    preflight = {
        "selected_rows": len(selected),
        "selection_csv": str(SELECTION_CSV.relative_to(USER_ROOT)),
        "checkpoint": args.checkpoint,
        "model_paths": {k: str(v.relative_to(USER_ROOT)) for k, v in model_paths.items()},
        "model_path_exists": {k: v.exists() for k, v in model_paths.items()},
        "method": "research official-compatible EWoK MLM candidate scorer on focused old-instability rows; CPU-only; not an endpoint score.",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"strictsmalltok_chck_{args.checkpoint}_focus553_preflight.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "STRICTSMALLTOK_MIDTRAIN_EWOK_MARGIN_DRYRUN", **preflight}, indent=2))
        return
    missing = [k for k, exists in preflight["model_path_exists"].items() if not exists]
    if missing:
        raise FileNotFoundError({"missing_model_keys": missing, "preflight": preflight})

    t0 = time.time()
    device = torch.device(args.device)
    results_by_model: dict[str, Any] = {}
    for mk, mp in model_paths.items():
        print(json.dumps({"event": "score_model_start", "model_key": mk, "model_path": str(mp.relative_to(USER_ROOT)), "rows": len(selected), "device": str(device)}), flush=True)
        results_by_model[mk] = mod.score_rows_for_model(
            mk, mp, selected, device=device, non_causal_batch_size=args.non_causal_batch_size, official_predictions=None)
        print(json.dumps({"event": "score_model_done", "model_key": mk, **results_by_model[mk]["summary"]}, default=str), flush=True)

    records_by_model = {k: v["records"] for k, v in results_by_model.items()}
    pair = summarize_two_seed(records_by_model)
    oldcmp = compare_to_old_focus(pair["rows"])
    payload = {
        "status": "STRICTSMALLTOK_MIDTRAIN_EWOK_MARGIN_PROBE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "preflight": preflight,
        "results_by_model": results_by_model,
        "two_seed_focus553_analysis": {k: v for k, v in pair.items() if k != "rows"},
        "old_inherited_tokenizer_endpoint_comparison": oldcmp,
        "interpretation": [
            "This probes already-saved corrected-tokenizer mid-training checkpoints only; it must not stop or replace the ongoing 100M retrains.",
            "If focused EWoK margin signs at 50M correlate with old endpoint relation-polarization, future cheap intermediate downstream probes may be useful for seed triage, unlike MLM loss.",
            "Full official-coordinate evaluation after chck_100M remains decisive for the Strict-Small SOTA goal.",
        ],
    }
    out_json = OUT_DIR / f"strictsmalltok_chck_{args.checkpoint}_focus553_margins.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    out_csv = OUT_DIR / f"strictsmalltok_chck_{args.checkpoint}_focus553_pair_rows.csv"
    fields = list(pair["rows"][0].keys()) if pair["rows"] else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pair["rows"])
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "out_csv": str(out_csv.relative_to(USER_ROOT)),
        "checkpoint": args.checkpoint,
        "acc_43022": payload["two_seed_focus553_analysis"]["accuracy_43022"],
        "acc_43122": payload["two_seed_focus553_analysis"]["accuracy_43122"],
        "opposite_sign_frac": payload["two_seed_focus553_analysis"]["opposite_sign_frac"],
        "margin_corr_430_431": payload["two_seed_focus553_analysis"]["margin_pearson_430_431"],
        "old_cmp": oldcmp,
    }, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
