#!/usr/bin/env python3
"""research: evaluate adapter 20M screens + adapter recruitment metrics.

Runs cheap official-compatible columns (BLiMP, Supplement, EWoK, Entity, COMPS,
GlobalPIQA, Reading) for each adapter arm, then measures adapter branch
recruitment (output RMS) and trunk displacement from research 20M.

Usage:
  python eval_adapter_20m.py --gpu 0 --arm adapter64
  python eval_adapter_20m.py --gpu 0                   # both arms
  python eval_adapter_20m.py --summarize               # print table
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, os, subprocess, sys
from pathlib import Path
from statistics import mean

# Force writable HF dynamic-module cache for custom model loading
os.environ["HF_MODULES_CACHE"] = str(
    _public_path('data/external/hf_modules_cache'))

USER_ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/frontier_consolidation')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py')
BASELINE_JSON = _public_path('experiments/archive/frontier_consolidation/data/legal20m_treatment_effect_eval/per_target/complianttok_reinvest_seed43022_20M.json')
reference_20M_MODEL = _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_20M')

ARMS = {
    "adapter64": {
        "label": "Adapter b=64",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter64_seed43022_20M')),
        "target": "adapter64_seed43022_20M",
        "bottleneck": 64,
    },
    "adapter128": {
        "label": "Adapter b=128",
        "run_dir": str(_public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_seed43022_20M')),
        "target": "adapter128_seed43022_20M",
        "bottleneck": 128,
    },
}

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
EVAL_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
                "GlobalPIQA_parallel", "GlobalPIQA_nonparallel", "Reading"]

reference_20M_SCORES = {
    "BLiMP": 59.69, "Supplement": 55.45, "EWoK": 50.73,
    "Entity": 18.65, "COMPS": 50.26, "GlobalPIQA": 34.195, "Reading": 8.67,
}
MUON_20M_SCORES = {
    "BLiMP": 61.15, "Supplement": 58.10, "EWoK": 50.70,
    "Entity": 20.90, "COMPS": 50.45, "GlobalPIQA": 38.605, "Reading": 7.26,
}

OUT_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_20M_eval')
COLLATE_ROOT = _public_path('experiments/archive/frontier_consolidation/data/adapter_20M_collate')


def extract_scores(payload: dict) -> dict:
    tasks = payload.get("tasks", {})
    out = {}
    for c in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if "score" in rec and rec["score"] is not None else None
    gp = []
    for c in ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
        rec = tasks.get(c, {})
        if "score" in rec and rec["score"] is not None:
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


def cheap7(scores):
    vals = [scores.get(c) for c in CHEAP_COLUMNS]
    if any(v is None for v in vals):
        return None
    return float(mean(float(v) for v in vals))


def run_eval(arm_key, arm, gpu):
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model" / "chck_20M"
    if not model_path.exists():
        model_path = run_dir / "hf_model"
    if not (model_path / "model.safetensors").exists():
        print(f"SKIP {arm_key}: no checkpoint at {model_path}", flush=True)
        return None
    target = arm["target"]
    per_target = _public_path('experiments/archive/frontier_consolidation/data/adapter_20M_eval/per_target') / f"{target}.json"
    if per_target.exists():
        print(f"EXISTS {arm_key}: {per_target}", flush=True)
        return per_target
    cmd = [sys.executable, "-B", str(EVALUATOR), "--arm", "reinvest",
           "--run-dir", str(run_dir), "--target", target, "--endpoint", "chck_20M",
           "--out-root", str(OUT_ROOT), "--collate-root", str(COLLATE_ROOT),
           "--gpu", str(gpu), "--columns", *EVAL_COLUMNS]
    print(f"RUN {arm_key}: {target} gpu={gpu}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        print(f"FAIL {arm_key}:\nSTDOUT {proc.stdout[-2000:]}\nSTDERR {proc.stderr[-3000:]}", flush=True)
        return None
    return per_target if per_target.exists() else None


def measure_adapter_recruitment(arm_key, arm, gpu):
    """Measure adapter output RMS and trunk displacement on a fixed test batch."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    run_dir = Path(arm["run_dir"])
    model_path = run_dir / "hf_model" / "chck_20M"
    if not (model_path / "model.safetensors").exists():
        return {"status": "no_checkpoint"}
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    # Fixed test sentences
    sents = [
        "The cat sat on the mat and looked out the window.",
        "Scientists discovered a new species of fish in the deep ocean.",
        "She gave him the book that she had borrowed from the library.",
        "After the rain stopped, the children went outside to play in the puddles.",
    ]
    enc = tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        _ = model(**enc)
    rms = model.adapter_rms() if hasattr(model, "adapter_rms") else []
    # Trunk displacement from research 20M
    disp = {}
    if reference_20M_MODEL.exists():
        from transformers import DebertaV2ForMaskedLM
        base_model = DebertaV2ForMaskedLM.from_pretrained(reference_20M_MODEL).to(device).eval()
        cos_sums, l2_sums, n = 0.0, 0.0, 0
        for (n1, p1), (n2, p2) in zip(
            ((n, p) for n, p in model.named_parameters() if ".adapter." not in n),
            base_model.named_parameters()
        ):
            if n1 != n2:
                disp["name_mismatch"] = f"{n1} != {n2}"
                break
            flat1 = p1.detach().float().flatten()
            flat2 = p2.detach().float().flatten()
            cos_sums += float(torch.nn.functional.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0)))
            l2_sums += float((flat1 - flat2).square().sum().sqrt())
            n += 1
        if n > 0:
            disp["mean_cosine"] = cos_sums / n
            disp["total_l2"] = l2_sums
            disp["n_tensors"] = n
        del base_model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"adapter_rms": rms, "adapter_rms_mean": mean(rms) if rms else 0,
            "adapter_rms_max": max(rms) if rms else 0, "trunk_displacement": disp}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--arm", type=str, default=None, choices=list(ARMS.keys()))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--skip-recruitment", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_20M_eval/per_target')).mkdir(parents=True, exist_ok=True)
    COLLATE_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(BASELINE_JSON.read_text()) if BASELINE_JSON.exists() else None
    b_scores = extract_scores(baseline) if baseline else reference_20M_SCORES
    b_c7 = cheap7(b_scores)

    rows = {}
    rows["reference_20M"] = {"label": "research legal 20M", "scores": b_scores, "cheap7": b_c7, "deltas": None}
    rows["muon_20M"] = {"label": "Muon wd-matched 20M", "scores": MUON_20M_SCORES,
                        "cheap7": cheap7(MUON_20M_SCORES),
                        "deltas": {c: MUON_20M_SCORES[c] - b_scores[c] for c in CHEAP_COLUMNS
                                   if b_scores.get(c) is not None}}

    arms_to_eval = {args.arm: ARMS[args.arm]} if args.arm else ARMS
    for k, arm in arms_to_eval.items():
        if not args.summarize:
            rp = run_eval(k, arm, args.gpu)
        else:
            rp = _public_path('experiments/archive/frontier_consolidation/data/adapter_20M_eval/per_target') / f"{arm['target']}.json"
            if not rp.exists():
                rp = None
        if rp and rp.exists():
            sc = extract_scores(json.loads(rp.read_text()))
            c7 = cheap7(sc)
            d = {c: (sc[c] - b_scores[c]) if sc.get(c) is not None and b_scores.get(c) is not None else None
                 for c in CHEAP_COLUMNS}
            d["cheap7"] = (c7 - b_c7) if c7 and b_c7 else None
            entry = {"label": arm["label"], "scores": sc, "cheap7": c7, "deltas": d}
        else:
            entry = {"label": arm["label"], "scores": None, "cheap7": None, "deltas": None}
        # Adapter recruitment metrics
        if not args.summarize and not args.skip_recruitment:
            try:
                recruit = measure_adapter_recruitment(k, arm, args.gpu)
                entry["recruitment"] = recruit
            except Exception as e:
                entry["recruitment"] = {"error": str(e)}
        rows[k] = entry

    out = {"status": "ADAPTER_20M_EVAL", "baseline_cheap7": b_c7, "rows": rows}
    (_public_path('experiments/archive/frontier_consolidation/data/adapter_20M_eval/adapter_20M_comparison.json')).write_text(json.dumps(out, indent=2) + "\n")

    lines = ["# research Adapter 20M comparison", "",
             f"Baseline (research legal 20M) cheap7: **{b_c7:.4f}**" if b_c7 else "Baseline: N/A", "",
             "| arm | BLiMP | Supp | EWoK | Entity | COMPS | GP | Read | cheap7 | Δcheap7 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rows.items():
        s = r.get("scores") or {}
        c7v = r.get("cheap7")
        dc = (r.get("deltas") or {}).get("cheap7") if r.get("deltas") else None
        def f(c): return f"{s[c]:.2f}" if s.get(c) is not None else "—"
        if c7v is not None:
            dc_str = f"{dc:+.4f}" if dc is not None else ""
            lines.append(f"| {r['label']} | {f('BLiMP')} | {f('Supplement')} | {f('EWoK')} | {f('Entity')} | {f('COMPS')} | {f('GlobalPIQA')} | {f('Reading')} | {c7v:.4f} | {dc_str} |")
        else:
            lines.append(f"| {r['label']} | — | — | — | — | — | — | — | — | — |")
    # Adapter recruitment section
    for k in arms_to_eval:
        r = rows.get(k, {})
        rec = r.get("recruitment", {})
        if rec and "adapter_rms" in rec:
            rms = rec["adapter_rms"]
            lines.extend(["", f"### {r.get('label', k)} recruitment",
                          f"- Adapter RMS mean: {rec.get('adapter_rms_mean', 0):.6f}",
                          f"- Adapter RMS max:  {rec.get('adapter_rms_max', 0):.6f}",
                          f"- Per-layer RMS: {['%.6f' % x for x in rms]}"])
            td = rec.get("trunk_displacement", {})
            if td.get("mean_cosine") is not None:
                lines.extend([f"- Trunk mean cosine to research: {td['mean_cosine']:.6f}",
                              f"- Trunk total L2 displacement: {td['total_l2']:.4f}",
                              f"- Stock tensors compared: {td['n_tensors']}"])
    lines.append("")
    (_public_path('research/documents/frontier_consolidation/data/adapter_20M_eval/adapter_20M_comparison.md')).write_text("\n".join(lines) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
