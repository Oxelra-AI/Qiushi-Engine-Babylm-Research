#!/usr/bin/env python3
"""research matched-horizon adapter evaluation and mechanism readout.

Evaluates the minimum fair residual-adapter experiment requested after research:
  * live adapter128, trained for the first 20.008711M words on the original 100M LR horizon
  * disabled adapter128 path under identical execution settings

The script reuses the official-compatible research harness for the cheap columns and then
measures adapter recruitment and stock-backbone displacement.  It keeps the dynamic-module
cache under the writable output directory to avoid read-only HF cache failures.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WORKSPACE = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
BASELINE_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json')
reference_20M_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M')
OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_collate')
HF_CACHE = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/hf_modules_cache')
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

ARMS = {
    "live128": {
        "label": "Adapter128 live, 100M LR horizon, 20M prefix",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_live_h100M20M_seed43022'),
        "target": "adapter128_live_h100M20M_seed43022",
    },
    "disabled128": {
        "label": "Adapter128 disabled path, 100M LR horizon, 20M prefix",
        "run_dir": _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_disabled_h100M20M_seed43022'),
        "target": "adapter128_disabled_h100M20M_seed43022",
    },
}

reference_20M_SCORES = {
    "BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73,
    "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67,
}
COMPRESSED = {
    "adapter128": {"BLiMP": 53.81, "Supplement": 50.13, "EWoK": 51.08, "Entity": 17.07,
                   "COMPS": 50.34, "GlobalPIQA": 33.75, "Reading": 7.17},
    "adapter64": {"BLiMP": 53.99, "Supplement": 50.12, "EWoK": 49.74, "Entity": 17.20,
                  "COMPS": 49.83, "GlobalPIQA": 32.765, "Reading": 7.12},
}


def extract_scores(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    out = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if rec.get("score") is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if rec.get("score") is not None:
            gp.append(float(rec["score"]))
    out["GlobalPIQA"] = mean(gp) if len(gp) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    return out


def cheap7(scores: dict | None) -> float | None:
    if not scores:
        return None
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def deltas(scores: dict | None, ref: dict | None) -> dict | None:
    if not scores or not ref:
        return None
    out = {}
    for c in CHEAP_COLUMNS:
        out[c] = None if scores.get(c) is None or ref.get(c) is None else float(scores[c] - ref[c])
    c7, r7 = cheap7(scores), cheap7(ref)
    out["cheap7"] = None if c7 is None or r7 is None else float(c7 - r7)
    return out


def load_baseline_scores() -> dict:
    if BASELINE_JSON.exists():
        return extract_scores(json.loads(BASELINE_JSON.read_text(encoding="utf-8")))
    return dict(reference_20M_SCORES)


def run_eval(arm_key: str, arm: dict, gpu: int, force: bool = False) -> Path | None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/per_target')).mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model/chck_20M"
    if not (model_path / "model.safetensors").exists():
        print(json.dumps({"event": "skip_missing_checkpoint", "arm": arm_key, "model_path": str(model_path)}), flush=True)
        return None
    per_target = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/per_target') / f"{arm['target']}.json"
    if per_target.exists() and not force:
        print(json.dumps({"event": "skip_existing_eval", "arm": arm_key, "per_target": str(per_target)}), flush=True)
        return per_target
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--arm", "reinvest",
        "--run-dir", str(run_dir),
        "--target", arm["target"],
        "--endpoint", "chck_20M",
        "--out-root", str(OUT_ROOT),
        "--collate-root", str(COLLATE_ROOT),
        "--gpu", str(gpu),
        "--columns", *EVAL_COLUMNS,
    ]
    if force:
        cmd.append("--force")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(json.dumps({"event": "eval_start", "arm": arm_key, "cmd": cmd}), flush=True)
    proc = subprocess.run(cmd, cwd=str(USER_ROOT), env=env, capture_output=True, text=True, timeout=2400)
    log_dir = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{arm_key}_eval_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{arm_key}_eval_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        print(json.dumps({"event": "eval_failed", "arm": arm_key, "returncode": proc.returncode,
                          "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-3000:]}), flush=True)
        return None
    return per_target if per_target.exists() else None


def tensor_cosine(a, b) -> float:
    import torch
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    denom = float(af.norm() * bf.norm())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(af, bf) / denom)


def measure_mechanism(arm_key: str, arm: dict, gpu: int, reference_arm: str | None = None) -> dict:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer, DebertaV2ForMaskedLM
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model_path = Path(arm["run_dir"]) / "hf_model/chck_20M"
    if not (model_path / "model.safetensors").exists():
        return {"status": "missing_checkpoint", "model_path": str(model_path)}
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    sents = [
        "The cat sat on the mat and looked out the window.",
        "Scientists discovered a new species of fish in the deep ocean.",
        "She gave him the book that she had borrowed from the library.",
        "After the rain stopped, the children went outside to play in the puddles.",
        "The key that opened the cabinet was lying under the red cup.",
        "If the block pushes the ball, the ball rolls across the table.",
    ]
    enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=80).to(device)
    with torch.no_grad():
        out = model(**enc)
    rec: dict = {
        "status": "ok",
        "logits_rms": float(out.logits.detach().float().square().mean().sqrt().cpu()),
        "adapter_rms": model.adapter_rms() if hasattr(model, "adapter_rms") else [],
    }
    if rec["adapter_rms"]:
        rec["adapter_rms_mean"] = float(mean(rec["adapter_rms"]))
        rec["adapter_rms_max"] = float(max(rec["adapter_rms"]))
    # Adapter parameter norms and gradients are not available post-training, but norms expose recruitment.
    adapter_stats = []
    for name, p in model.named_parameters():
        if ".adapter." in name:
            adapter_stats.append({"name": name, "norm": float(p.detach().float().norm().cpu()), "rms": float(p.detach().float().square().mean().sqrt().cpu())})
    rec["adapter_param_stats"] = adapter_stats
    rec["adapter_up_norm_sum"] = float(sum(x["norm"] for x in adapter_stats if ".adapter.up." in x["name"]))
    rec["adapter_down_norm_sum"] = float(sum(x["norm"] for x in adapter_stats if ".adapter.down." in x["name"]))
    rec["adapter_ln_norm_sum"] = float(sum(x["norm"] for x in adapter_stats if ".adapter.layer_norm." in x["name"]))

    def compare_to_stock(ref_path: Path) -> dict:
        if not (ref_path / "model.safetensors").exists():
            return {"status": "missing_reference", "path": str(ref_path)}
        ref = DebertaV2ForMaskedLM.from_pretrained(ref_path).to(device).eval()
        by_name = dict(ref.named_parameters())
        cosines, rel_l2s, l2s = [], [], []
        missing = []
        for n, p in model.named_parameters():
            if ".adapter." in n:
                continue
            q = by_name.get(n)
            if q is None:
                missing.append(n)
                continue
            diff = p.detach().float() - q.detach().float()
            l2 = float(diff.norm().cpu())
            base = float(q.detach().float().norm().cpu())
            l2s.append(l2)
            rel_l2s.append(l2 / base if base > 0 else math.nan)
            cosines.append(tensor_cosine(p, q))
        del ref
        return {
            "status": "ok" if not missing else "name_mismatch",
            "reference": str(ref_path),
            "n_tensors": len(cosines),
            "missing": missing[:10],
            "mean_cosine": float(mean(cosines)) if cosines else None,
            "min_cosine": float(min(cosines)) if cosines else None,
            "mean_rel_l2": float(mean(rel_l2s)) if rel_l2s else None,
            "max_rel_l2": float(max(rel_l2s)) if rel_l2s else None,
            "total_l2_sum": float(sum(l2s)),
        }

    rec["stock_displacement_vs_step35_20M"] = compare_to_stock(reference_20M_MODEL)
    if reference_arm and reference_arm in ARMS:
        ref_path = Path(ARMS[reference_arm]["run_dir"]) / "hf_model/chck_20M"
        # This may be a custom model; compare stock named tensors by loading through AutoModel.
        if (ref_path / "model.safetensors").exists():
            ref_model = AutoModelForMaskedLM.from_pretrained(ref_path, trust_remote_code=True).to(device).eval()
            by_name = {n: p for n, p in ref_model.named_parameters() if ".adapter." not in n}
            cosines, rel_l2s, l2s, missing = [], [], [], []
            for n, p in model.named_parameters():
                if ".adapter." in n:
                    continue
                q = by_name.get(n)
                if q is None:
                    missing.append(n); continue
                diff = p.detach().float() - q.detach().float()
                l2 = float(diff.norm().cpu())
                base = float(q.detach().float().norm().cpu())
                l2s.append(l2); rel_l2s.append(l2 / base if base > 0 else math.nan); cosines.append(tensor_cosine(p, q))
            rec[f"stock_displacement_vs_{reference_arm}"] = {
                "status": "ok" if not missing else "name_mismatch",
                "reference": str(ref_path), "n_tensors": len(cosines), "missing": missing[:10],
                "mean_cosine": float(mean(cosines)) if cosines else None,
                "min_cosine": float(min(cosines)) if cosines else None,
                "mean_rel_l2": float(mean(rel_l2s)) if rel_l2s else None,
                "max_rel_l2": float(max(rel_l2s)) if rel_l2s else None,
                "total_l2_sum": float(sum(l2s)),
            }
            del ref_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


def summarize(force: bool = False) -> dict:
    baseline = load_baseline_scores()
    rows = {
        "reference_20M": {"label": "research legal 20M, original 100M horizon", "scores": baseline,
                         "cheap7": cheap7(baseline), "delta_vs_step35": None},
        "adapter128_compressed20M": {"label": "research adapter128, compressed 20M horizon", "scores": COMPRESSED["adapter128"],
                                             "cheap7": cheap7(COMPRESSED["adapter128"]),
                                             "delta_vs_step35": deltas(COMPRESSED["adapter128"], baseline)},
    }
    for k, arm in ARMS.items():
        p = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/per_target') / f"{arm['target']}.json"
        if p.exists():
            scores = extract_scores(json.loads(p.read_text(encoding="utf-8")))
            rows[k] = {"label": arm["label"], "scores": scores, "cheap7": cheap7(scores),
                       "delta_vs_step35": deltas(scores, baseline)}
        else:
            rows[k] = {"label": arm["label"], "scores": None, "cheap7": None, "delta_vs_step35": None}
        mp = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/mechanism') / f"{k}_mechanism.json"
        if mp.exists():
            rows[k]["mechanism"] = json.loads(mp.read_text(encoding="utf-8"))
    if rows.get("live128", {}).get("scores") and rows.get("disabled128", {}).get("scores"):
        rows["live128"]["delta_vs_disabled128"] = deltas(rows["live128"]["scores"], rows["disabled128"]["scores"])
    out = {"status": "ADAPTER_MATCHED_HORIZON_SUMMARY", "rows": rows}
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/adapter_matched_horizon_summary.json')).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    lines = ["# research Adapter matched-horizon 20M-prefix comparison", "",
             "All research arms use the original research 100M LR horizon (`lr_total_steps=2529`) and stop at the exact research 20M checkpoint exposure (20,008,711 words / 506 batches).", "",
             "| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δ vs research |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in rows.items():
        s = r.get("scores") or {}
        c7 = r.get("cheap7")
        dc = (r.get("delta_vs_step35") or {}).get("cheap7") if r.get("delta_vs_step35") else None
        def f(c):
            return f"{s[c]:.2f}" if s.get(c) is not None else "—"
        dc_str = "" if dc is None else f"{dc:+.4f}"
        lines.append(f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7:.4f} | {dc_str} |" if c7 is not None else f"| {r['label']} | — | — | — | — | — | — | — | — | — |")
    for key in ["live128", "disabled128"]:
        mech = rows.get(key, {}).get("mechanism")
        if mech:
            lines += ["", f"## Mechanism: {rows[key]['label']}",
                      f"- Adapter output RMS mean/max: {mech.get('adapter_rms_mean', 0):.8f} / {mech.get('adapter_rms_max', 0):.8f}",
                      f"- Adapter up/down norm sums: {mech.get('adapter_up_norm_sum', 0):.6f} / {mech.get('adapter_down_norm_sum', 0):.6f}"]
            disp = mech.get("stock_displacement_vs_step35_20M", {})
            if disp:
                lines += [f"- Stock displacement vs research 20M: mean cosine {disp.get('mean_cosine')}, mean relative L2 {disp.get('mean_rel_l2')}, max relative L2 {disp.get('max_rel_l2')}"]
            disp2 = mech.get("stock_displacement_vs_disabled128", {})
            if disp2:
                lines += [f"- Live stock displacement vs disabled128: mean cosine {disp2.get('mean_cosine')}, mean relative L2 {disp2.get('mean_rel_l2')}, max relative L2 {disp2.get('max_rel_l2')}"]
    lines.append("")
    (_public_path('research/documents/frontier_consolidation/data/adapter_matched_horizon_eval/adapter_matched_horizon_summary.md')).write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--mechanism", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/mechanism')).mkdir(parents=True, exist_ok=True)
    if args.arm:
        arm = ARMS[args.arm]
        if args.eval:
            run_eval(args.arm, arm, args.gpu, args.force)
        if args.mechanism:
            ref = "disabled128" if args.arm == "live128" else None
            rec = measure_mechanism(args.arm, arm, args.gpu, reference_arm=ref)
            mp = _public_path('experiments/archive/frontier_consolidation/data/adapter_matched_horizon_eval/mechanism') / f"{args.arm}_mechanism.json"
            mp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "mechanism_done", "arm": args.arm, "path": str(mp),
                              "adapter_rms_mean": rec.get("adapter_rms_mean"),
                              "adapter_up_norm_sum": rec.get("adapter_up_norm_sum")}, indent=2), flush=True)
    if args.summarize or (not args.eval and not args.mechanism):
        out = summarize(args.force)
        print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
