#!/usr/bin/env python3
"""research v2: Entity evaluation for HM and HV — cooperation test.

Uses the same evaluation pipeline as research with --output_dir to avoid read-only FS.
Then maps predictions to numops for zero-update stratification.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, math, os, pathlib, re, subprocess, sys, time
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/entity_hm_hv.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
STRICT = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')
ENTITY_DATA = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
NLP_DATA_ROOT = _public_path('experiments/archive/initial_model_studies/data/nltk_data')
OUT = _public_path('experiments/archive/relation_learning/data/entity_hm_hv')
FUNCTIONAL_RELATION_STUDIES_RUNS = _public_path('experiments/archive/relation_learning/training/runs')

ARM_CONFIGS = {
    "HM": {"run_dir": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_hash_mixed_dose2p64x_matched_rowholdout_deberta100M_seed43022')},
    "HV": {"run_dir": _public_path('experiments/archive/relation_learning/training/runs/full_p2c_c2p_abs_half_view_noexact_dose2p64x_matched_rowholdout_deberta100M_seed43022')},
}
CKS = ["chck_80M", "chck_90M", "chck_100M"]

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def rel(p): return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
def read_json(p): return json.loads(p.read_text(encoding="utf-8"))
def write_json(p, o): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(o, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")

def parse_score(text):
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)", r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            val = float(m.group(1))
            return val if math.isfinite(val) and -5 <= val <= 105 else None
    return None

def eval_env(arm, ck, gpu):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "NLTK_DATA": str(NLP_DATA_ROOT)}
    cache = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/cache') / f"{arm}_{ck}"
    tmp = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/tmp') / f"{arm}_{ck}"
    for key, p in {"HF_HOME": cache, "HF_HUB_CACHE": cache / "hub",
                   "TRANSFORMERS_CACHE": cache / "transformers",
                   "HF_MODULES_CACHE": cache / "modules",
                   "HF_DATASETS_CACHE": cache / "datasets",
                   "TMPDIR": tmp}.items():
        p.mkdir(parents=True, exist_ok=True)
        env[key] = str(p.resolve())
    return env

def eval_one(arm, ck, gpu, timeout=600):
    model_path = ARM_CONFIGS[arm]["run_dir"] / "hf_model" / ck
    if not model_path.exists():
        return {"arm": arm, "checkpoint": ck, "status": "model_not_found"}
    out_dir = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/eval_output') / f"{arm}_{ck}"
    log_path = _public_path('experiments/archive/relation_learning/data/entity_hm_hv/logs') / f"entity_{arm}_{ck}.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    revision = f"step033_{arm}_{ck}_Entity"
    argv = [
        sys.executable, "-B", "-m", "evaluation_pipeline.sentence_zero_shot.run",
        "--model_path_or_name", str(model_path.resolve()),
        "--backend", "mlm",
        "--task", "entity_tracking",
        "--data_path", str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')),
        "--revision_name", revision,
        "--save_predictions",
        "--batch_size", "128",
        "--non_causal_batch_size", "64",
        "--output_dir", str(out_dir.resolve()),
    ]
    print(f"  [{arm} {ck}] Running Entity eval on GPU {gpu}...", flush=True)
    t0 = time.time()
    try:
        with log_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": "start", "utc": now(), "arm": arm, "ck": ck}) + "\n")
            proc = subprocess.run(argv, cwd=str(_public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict')), env=eval_env(arm, ck, gpu),
                                  stdout=fh, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"arm": arm, "checkpoint": ck, "status": "timeout", "elapsed": round(time.time()-t0, 1)}
    elapsed = round(time.time() - t0, 1)
    report_files = sorted(out_dir.rglob("best_temperature_report.txt"), key=lambda p: p.stat().st_mtime)
    pred_files = sorted(out_dir.rglob("predictions.json"), key=lambda p: p.stat().st_mtime)
    score = parse_score(report_files[-1].read_text(encoding="utf-8", errors="replace")) if report_files else None
    return {"arm": arm, "checkpoint": ck, "status": "ok" if rc == 0 and score is not None else f"exit_{rc}",
            "score": score, "elapsed": elapsed, "rc": rc,
            "report": rel(report_files[-1]) if report_files else None,
            "predictions": rel(pred_files[-1]) if pred_files else None}

def load_entity_items():
    """Load evaluation items with numops to stratify accuracy."""
    items = []
    for fp in sorted(ENTITY_DATA.glob("*.jsonl")):
        category = fp.stem  # regular, ambiref, move_contents
        for line_idx, line in enumerate(fp.read_text(encoding="utf-8").splitlines()):
            if not line.strip(): continue
            obj = json.loads(line)
            items.append({
                "category": category,
                "example_id": obj.get("example_id", line_idx),
                "sample_id": obj.get("sample_id"),
                "numops": int(obj.get("numops", 0)),
                "gold_answer": obj["options"][0] if obj.get("options") else "",  # first option is correct in BabyLM format
                "n_options": len(obj.get("options", [])),
            })
    return items

def analyze_predictions(results, items):
    """Map predictions to numops for zero-update stratification."""
    import csv
    all_rows = []
    for res in results:
        if res.get("predictions") is None: continue
        pred_path = ROOT / res["predictions"]
        if not pred_path.exists(): continue
        preds = read_json(pred_path)
        # Predictions keyed by category, containing list of predictions
        for cat_key, cat_preds in preds.items():
            cat_items = [it for it in items if it["category"] == cat_key.replace("entity_tracking/", "")]
            if not cat_items and "_" in cat_key:
                cat_items = [it for it in items if it["category"] in cat_key]
            pred_list = cat_preds if isinstance(cat_preds, list) else cat_preds.get("predictions", [])
            for idx, pred in enumerate(pred_list):
                if idx >= len(cat_items): break
                item = cat_items[idx]
                pred_text = pred.get("pred", pred) if isinstance(pred, dict) else str(pred)
                gold = item["gold_answer"]
                correct = int(pred_text.strip().lower().rstrip(".") == gold.strip().lower().rstrip("."))
                all_rows.append({
                    "arm": res["arm"], "checkpoint": res["checkpoint"],
                    "category": item["category"], "example_id": item["example_id"],
                    "numops": item["numops"], "correct": correct,
                    "pred": pred_text[:100], "gold": gold[:100],
                })
    if not all_rows: return None
    # Write raw rows
    with open(_public_path('experiments/archive/relation_learning/data/entity_hm_hv/entity_prediction_rows.csv'), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "checkpoint", "category", "example_id", "numops", "correct", "pred", "gold"])
        w.writeheader(); w.writerows(all_rows)
    
    # Aggregate by arm × numops (late checkpoint mean)
    from collections import defaultdict
    by_arm_ck_numops = defaultdict(list)
    for r in all_rows:
        by_arm_ck_numops[(r["arm"], r["checkpoint"], r["numops"])].append(r["correct"])
    
    # Late mean across checkpoints
    late = defaultdict(lambda: defaultdict(list))
    for (arm, ck, numops), vals in by_arm_ck_numops.items():
        acc = 100.0 * sum(vals) / len(vals) if vals else 0
        late[(arm, numops)]["accs"].append(acc)
        late[(arm, numops)]["n"] = len(vals)
    
    summary_rows = []
    for (arm, numops), d in sorted(late.items()):
        accs = d["accs"]
        summary_rows.append({
            "arm": arm, "numops": numops, "n_checkpoints": len(accs),
            "n_items": d["n"], "late_mean_accuracy_pct": sum(accs) / len(accs),
        })
    with open(_public_path('experiments/archive/relation_learning/data/entity_hm_hv/entity_by_numops.csv'), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "numops", "n_checkpoints", "n_items", "late_mean_accuracy_pct"])
        w.writeheader(); w.writerows(summary_rows)
    return summary_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"status": "ENTITY_HM_HV_V2_PLAN", "arms": list(ARM_CONFIGS.keys()), "checkpoints": CKS,
            "model_exists": {f"{arm}_{ck}": (ARM_CONFIGS[arm]["run_dir"]/"hf_model"/ck).exists()
                             for arm in ARM_CONFIGS for ck in CKS}}
    write_json(_public_path('experiments/archive/relation_learning/data/entity_hm_hv/entity_hm_hv_plan.json'), plan)
    if args.plan_only:
        print(json.dumps(plan, indent=2), flush=True); return
    
    results = []
    for arm in ARM_CONFIGS:
        for ck in CKS:
            r = eval_one(arm, ck, args.gpu)
            results.append(r)
            print(f"  {r['arm']} {r['checkpoint']}: status={r['status']} score={r.get('score')}", flush=True)
    write_json(_public_path('experiments/archive/relation_learning/data/entity_hm_hv/entity_eval_results.json'), results)
    
    items = load_entity_items()
    print(f"Entity items: {len(items)} ({len([i for i in items if i['numops']==0])} at numops=0)", flush=True)
    summary = analyze_predictions(results, items)
    
    status = {"status": "ENTITY_HM_HV_V2_DONE", "ok_count": sum(1 for r in results if r["status"] == "ok"),
              "overall_scores": {f"{r['arm']}_{r['checkpoint']}": r.get("score") for r in results}}
    if summary:
        zero_update = {r["arm"]: r["late_mean_accuracy_pct"] for r in summary if r["numops"] == 0}
        status["zero_update_accuracy"] = zero_update
    write_json(_public_path('experiments/archive/relation_learning/data/entity_hm_hv/entity_hm_hv_summary.json'), status)
    print(json.dumps(status, indent=2), flush=True)

if __name__ == "__main__":
    main()
