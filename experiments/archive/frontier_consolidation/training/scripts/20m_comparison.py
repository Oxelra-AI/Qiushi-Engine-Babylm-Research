#!/usr/bin/env python3
"""research: Compare spatial-repair vs original reinvest at 20M checkpoint.

Runs the full no-AoA fast screen + official EWoK domain evaluation on both:
  - Original compact_view_reinvest 20M (seed43022)
  - Spatial-repair compact_view_reinvest 20M (seed43022)

Produces a structured comparison with per-task deltas and EWoK domain breakdown.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional

def discover_root() -> Path:
    root = Path.cwd()
    if (root / "experiments").exists():
        return root
    here = _public_path('experiments/archive/frontier_consolidation/training/scripts/20m_comparison.py')
    for parent in [here] + list(here.parents):
        if (parent / "experiments").exists():
            return parent
    raise RuntimeError("Could not discover user root")

ROOT = discover_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
STRICT = ROOT / "experiments/archive" / 'initial_model_studies' / "repos" / "babylm-eval" / "strict"
PY = sys.executable

# ── Models to compare ───────────────────────────────────────────────────────
MODELS = {
    "original_19M": {
        "path": ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_19M",
        "label": "compact_view_reinvest_original_19M",
    },
    "spatial_repair_19M": {
        "path": ROOT / "experiments/archive/frontier_consolidation/training/runs/spatial_repair_reinvest_2pass_seed43022/hf_model/chck_19M",
        "label": "spatial_repair_reinvest_19M",
    },
}

# ── Standard fast no-AoA tasks ─────────────────────────────────────────────
TASKS = [
    ("BLiMP",                   "blimp",                 "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement",              "blimp",                 "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK",                    "ewok",                  "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity",                  "entity_tracking",       "evaluation_data/fast_eval/entity_tracking_fast", 128),
    ("Entity_full",             "entity_tracking",       "evaluation_data/full_eval/entity_tracking", 128),
    ("COMPS",                   "comps",                 "evaluation_data/full_eval/comps", 128),
    ("GlobalPIQA_parallel",     "global_piqa_parallel",  "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GlobalPIQA_nonparallel",  "global_piqa_nonparallel","evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]
TASK_MAP = {c: (t, d, b) for c, t, d, b in TASKS}

# ── Official EWoK (domain-level) ───────────────────────────────────────────
OFFICIAL_EWOK = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/ewok/ewok_filtered"

OUT_ROOT = STUDY / "data" / "20m_comparison"
WORK_TAG = "20m_comparison"


def setup_env(gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT_ROOT / "hf_cache"
    for sub in ["home", "hub", "datasets", "transformers", "modules"]:
        (hf / sub).mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf / "home")
    env["HF_HUB_CACHE"] = str(hf / "hub")
    env["HF_DATASETS_CACHE"] = str(hf / "datasets")
    env["TRANSFORMERS_CACHE"] = str(hf / "transformers")
    env["HF_MODULES_CACHE"] = str(hf / "modules")
    env["NLTK_DATA"] = str((ROOT / "experiments/archive/initial_model_studies/data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tmp = OUT_ROOT / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    return env


def parse_score(text: str) -> Optional[float]:
    for pat in [r"### AVERAGE [A-Z_ '\\-]*\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
                r"AVERAGE ACCURACY\s*\n\s*([+-]?[0-9]+(?:\.[0-9]+)?)"]:
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if math.isfinite(v) and -1000 < v < 1000:
                return v
    for line in reversed(text.splitlines()):
        m = re.match(r"^\s*([+-]?[0-9]+\.[0-9]+)\s*$", line.strip())
        if m:
            v = float(m.group(1))
            if math.isfinite(v) and -1000 < v < 1000:
                return v
    return None


def parse_reading(text: str) -> Dict[str, float]:
    out = {}
    for label, key in [("EYE TRACKING SCORE", "Reading_eye"), ("SELF-PACED READING SCORE", "Reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
        if m:
            out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"] + out["Reading_self_paced"]) / 2.0
    return out


def eval_task(model_key: str, model_path: Path, column: str, env: Dict[str, str]) -> Dict[str, Any]:
    task, data_path, batch = TASK_MAP[column]
    task_out = OUT_ROOT / "outputs" / model_key / column
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / f"{model_key}_{column}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "evaluation_pipeline.sentence_zero_shot.run",
           "--model_path_or_name", str(model_path.resolve()),
           "--backend", "mlm", "--task", task,
           "--data_path", data_path,
           "--revision_name", f"step035_{model_key}_{column}",
           "--save_predictions", "--batch_size", str(batch),
           "--output_dir", str(task_out.resolve())]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                       capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    with log.open("a") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] {' '.join(cmd[:8])} ...\n")
        f.write(p.stdout[-2000:] if len(p.stdout) > 2000 else p.stdout)
        if p.stderr: f.write(f"\n--- STDERR ---\n{p.stderr[-500:]}\n")
        f.write(f"[rc={p.returncode} elapsed={elapsed:.1f}s]\n")
    score = parse_score(p.stdout)
    # Also try report file
    if score is None:
        for rpt in sorted(task_out.rglob("best_temperature_report.txt")):
            score = parse_score(rpt.read_text())
            if score is not None:
                break
    return {"column": column, "score": score, "returncode": p.returncode, "elapsed": round(elapsed, 1)}


def eval_reading(model_key: str, model_path: Path, env: Dict[str, str]) -> Dict[str, Any]:
    task_out = OUT_ROOT / "outputs" / model_key / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT_ROOT / "logs" / f"{model_key}_Reading.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "evaluation_pipeline.reading.run",
           "--model_path_or_name", str(model_path.resolve()),
           "--backend", "mlm",
           "--data_path", "evaluation_data/fast_eval/reading/reading_data.csv",
           "--revision_name", f"step035_{model_key}_Reading",
           "--output_dir", str(task_out.resolve())]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                       capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    with log.open("a") as f:
        f.write(f"\n[{time.strftime('%H:%M:%S')}] Reading\n{p.stdout[-1000:]}\n")
        f.write(f"[rc={p.returncode} elapsed={elapsed:.1f}s]\n")
    scores = parse_reading(p.stdout)
    if not scores:
        for rpt in sorted(task_out.rglob("report.txt")):
            scores = parse_reading(rpt.read_text())
            if scores: break
    return {"column": "Reading", "scores": scores, "returncode": p.returncode, "elapsed": round(elapsed, 1)}


def eval_ewok_domains(model_key: str, model_path: Path, env: Dict[str, str]) -> Dict[str, Any]:
    """Run official-EWoK domain evaluation using the margin scorer approach."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    
    ewok_path = OFFICIAL_EWOK
    items_path = sorted(ewok_path.glob("*.jsonl"))
    if not items_path:
        return {"error": "No EWoK JSONL files found"}
    
    # Load EWoK data
    ewok_items = []
    for fp in items_path:
        with open(fp) as f:
            for line in f:
                ewok_items.append(json.loads(line))
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path.resolve()))
    model = AutoModelForMaskedLM.from_pretrained(str(model_path.resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Score each item
    domain_correct = {}
    domain_total = {}
    
    with torch.no_grad():
        for item in ewok_items:
            s1 = item.get("sentence1", item.get("target", ""))
            s2 = item.get("sentence2", item.get("distractor", ""))
            uid = item.get("uid", item.get("domain", "unknown"))
            domain = uid.split("_")[0] if "_" in uid else uid
            # Full domain from UID
            parts = uid.split("_")
            if len(parts) >= 2:
                domain = parts[0] + "-" + parts[1]
            
            # Compute pseudo-log-likelihood for both sentences
            def pll(text):
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                input_ids = tokens["input_ids"].to(device)
                n = input_ids.shape[1]
                total_ll = 0.0
                for i in range(1, n - 1):  # skip [CLS] and [SEP]
                    masked = input_ids.clone()
                    masked[0, i] = tokenizer.mask_token_id
                    logits = model(**{k: v for k, v in tokens.items() if k != "input_ids"}, 
                                   input_ids=masked).logits
                    log_probs = torch.log_softmax(logits[0, i], dim=-1)
                    total_ll += log_probs[input_ids[0, i]].item()
                return total_ll
            
            pll1 = pll(s1)
            pll2 = pll(s2)
            correct = 1 if pll1 > pll2 else 0
            
            domain_correct[domain] = domain_correct.get(domain, 0) + correct
            domain_total[domain] = domain_total.get(domain, 0) + 1
    
    domain_acc = {d: domain_correct[d] / domain_total[d] * 100 for d in sorted(domain_total)}
    macro = sum(domain_acc.values()) / len(domain_acc) if domain_acc else 0
    
    return {
        "model": model_key,
        "total_items": len(ewok_items),
        "macro_accuracy": round(macro, 4),
        "domain_accuracy": {d: round(v, 4) for d, v in sorted(domain_acc.items())},
        "domain_counts": dict(sorted(domain_total.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-ewok-domains", action="store_true")
    args = ap.parse_args()
    
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    env = setup_env(args.gpu)
    
    # Check both models exist
    for mk, m in MODELS.items():
        if not m["path"].exists():
            print(f"ERROR: model not found: {mk} -> {m['path']}")
            sys.exit(1)
    
    results = {}
    
    for mk, m in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {mk} ({m['label']})")
        print(f"  Path: {m['path']}")
        print(f"{'='*60}")
        
        model_results = {}
        
        # Standard fast tasks
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
                     "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]:
            print(f"  {col}...", end=" ", flush=True)
            r = eval_task(mk, m["path"], col, env)
            model_results[col] = r["score"]
            print(f"{r['score']:.2f}" if r["score"] is not None else "FAILED", f"({r['elapsed']:.0f}s)")
        
        # Reading
        print(f"  Reading...", end=" ", flush=True)
        r = eval_reading(mk, m["path"], env)
        model_results.update(r["scores"])
        print(f"{r['scores'].get('Reading', 'FAILED')}", f"({r['elapsed']:.0f}s)")
        
        # Derived
        gp = model_results.get("GlobalPIQA_parallel")
        gnp = model_results.get("GlobalPIQA_nonparallel")
        if gp is not None and gnp is not None:
            model_results["GlobalPIQA_mean"] = (gp + gnp) / 2.0
        
        # equal7 (BLiMP, Supplement, EWoK, COMPS, GlobalPIQA_mean, Reading, Entity_full)
        eq7_cols = ["BLiMP", "Supplement", "EWoK", "COMPS", "GlobalPIQA_mean", "Reading", "Entity_full"]
        eq7_vals = [model_results.get(c) for c in eq7_cols]
        if all(v is not None for v in eq7_vals):
            model_results["equal7_full_entity"] = sum(eq7_vals) / len(eq7_vals)
        
        results[mk] = model_results
    
    # ── EWoK domain evaluation using margin scorer ──────────────────────────
    if not args.skip_ewok_domains:
        print(f"\n{'='*60}")
        print("EWoK domain-level evaluation (official 7618 items)")
        print(f"{'='*60}")
        
        # Use the research margin scorer for domain-level accuracy
        scorer_path = STUDY / "training/scripts/ewok_margin_scorer.py"
        if scorer_path.exists():
            for mk, m in MODELS.items():
                print(f"  {mk}...", end=" ", flush=True)
                ewok_out = OUT_ROOT / "ewok_domains" / mk
                ewok_out.mkdir(parents=True, exist_ok=True)
                cmd = [PY, str(scorer_path),
                       "--models", f"{mk}={m['path']}",
                       "--output-dir", str(ewok_out),
                       "--gpu", args.gpu]
                t0 = time.time()
                p = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                   capture_output=True, text=True, timeout=600)
                elapsed = time.time() - t0
                if p.returncode == 0:
                    # Try to read the summary
                    summary_files = sorted(ewok_out.rglob("*summary*.json"))
                    if summary_files:
                        ewok_data = json.loads(summary_files[0].read_text())
                        results[mk]["ewok_domains"] = ewok_data
                        print(f"OK ({elapsed:.0f}s)")
                    else:
                        print(f"No summary found ({elapsed:.0f}s)")
                else:
                    print(f"FAILED rc={p.returncode} ({elapsed:.0f}s)")
                    print(f"  stderr: {p.stderr[-500:]}")
    
    # ── Comparison ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARISON: spatial_repair_19M vs original_19M")
    print(f"{'='*60}")
    
    compare_cols = ["BLiMP", "Supplement", "EWoK", "Entity", "Entity_full", "COMPS",
                    "GlobalPIQA_mean", "Reading", "equal7_full_entity"]
    
    orig = results.get("original_19M", {})
    repair = results.get("spatial_repair_19M", {})
    
    comparison = {}
    for col in compare_cols:
        o = orig.get(col)
        r = repair.get(col)
        delta = (r - o) if (o is not None and r is not None) else None
        comparison[col] = {"original": o, "spatial_repair": r, "delta": delta}
        sign = "+" if delta and delta > 0 else ""
        print(f"  {col:30s}: orig={o:7.3f}  repair={r:7.3f}  delta={sign}{delta:7.3f}" if (o is not None and r is not None) else f"  {col:30s}: incomplete")
    
    # Save
    output = {
        "status": "reference_20M_COMPARISON",
        "models": {mk: str(m["path"]) for mk, m in MODELS.items()},
        "results": results,
        "comparison": comparison,
    }
    
    out_path = OUT_ROOT / "reference_20m_comparison.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    
    # Save markdown summary
    md_path = OUT_ROOT / "reference_20m_comparison.md"
    with md_path.open("w") as f:
        f.write("# research — 20M Spatial Repair Comparison\n\n")
        f.write("| Column | Original 20M | Spatial Repair 20M | Delta |\n")
        f.write("|--------|-------------|-------------------|-------|\n")
        for col in compare_cols:
            c = comparison.get(col, {})
            o = c.get("original")
            r = c.get("spatial_repair")
            d = c.get("delta")
            f.write(f"| {col} | {o:.3f if o is not None else 'N/A':>8s} | {r:.3f if r is not None else 'N/A':>8s} | {'+' if d and d > 0 else ''}{d:.3f if d is not None else 'N/A':>8s} |\n")
        f.write(f"\nJSON: `{out_path}`\n")
    
    print(f"\nSaved: {out_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
