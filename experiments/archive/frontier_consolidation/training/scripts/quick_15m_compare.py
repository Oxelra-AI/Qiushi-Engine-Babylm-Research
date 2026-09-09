#!/usr/bin/env python3
"""research: Quick 15M comparison — spatial repair vs original.

Runs EWoK + key columns on both 15M checkpoints for a fast diagnostic.
Uses two GPUs in parallel for speed.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json, math, os, re, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, Optional

def discover_root() -> Path:
    root = Path.cwd()
    if (root / "experiments").exists():
        return root
    here = _public_path('experiments/archive/frontier_consolidation/training/scripts/quick_15m_compare.py')
    for parent in [here] + list(here.parents):
        if (parent / "experiments").exists():
            return parent
    raise RuntimeError("Could not discover user root")

ROOT = discover_root()
STRICT = ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
PY = sys.executable
OUT = ROOT / "experiments/archive/frontier_consolidation/data/15m_quick_comparison"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "original_15M": ROOT / "experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_15M",
    "spatial_15M": ROOT / "experiments/archive/frontier_consolidation/training/runs/spatial_repair_reinvest_2pass_seed43022/hf_model/chck_15M",
}

# Key diagnostic columns
TASKS = [
    ("BLiMP",      "blimp",      "evaluation_data/fast_eval/blimp_fast", 128),
    ("Supplement",  "blimp",      "evaluation_data/fast_eval/supplement_fast", 128),
    ("EWoK",       "ewok",       "evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast", 64),
    ("Entity_full","entity_tracking","evaluation_data/full_eval/entity_tracking", 128),
    ("COMPS",      "comps",       "evaluation_data/full_eval/comps", 128),
    ("GP_par",     "global_piqa_parallel",    "evaluation_data/fast_eval/global_piqa_parallel", 128),
    ("GP_npar",    "global_piqa_nonparallel", "evaluation_data/fast_eval/global_piqa_nonparallel", 128),
]

def setup_env(gpu: str) -> Dict[str, str]:
    env = os.environ.copy()
    hf = OUT / "hf_cache"
    for s in ["home","hub","datasets","transformers","modules"]:
        (hf/s).mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf/"home")
    env["HF_HUB_CACHE"] = str(hf/"hub")
    env["HF_DATASETS_CACHE"] = str(hf/"datasets")
    env["TRANSFORMERS_CACHE"] = str(hf/"transformers")
    env["HF_MODULES_CACHE"] = str(hf/"modules")
    env["NLTK_DATA"] = str((ROOT/"experiments/archive/initial_model_studies/data/nltk_data").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    (OUT/"tmp").mkdir(exist_ok=True)
    env["TMPDIR"] = str(OUT/"tmp")
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

def eval_col(mk, model_path, col, task, data_path, batch, env):
    task_out = OUT / "outputs" / mk / col
    task_out.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{mk}_{col}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "evaluation_pipeline.sentence_zero_shot.run",
           "--model_path_or_name", str(model_path.resolve()),
           "--backend", "mlm", "--task", task,
           "--data_path", data_path,
           "--revision_name", f"s35_{mk}_{col}",
           "--save_predictions", "--batch_size", str(batch),
           "--output_dir", str(task_out.resolve())]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                       capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    with log.open("a") as f:
        f.write(p.stdout[-2000:]); f.write(f"\n[rc={p.returncode} t={elapsed:.0f}s]\n")
    score = parse_score(p.stdout)
    if score is None:
        for rpt in sorted(task_out.rglob("best_temperature_report.txt")):
            score = parse_score(rpt.read_text())
            if score is not None: break
    return score

def parse_reading(text):
    out = {}
    for label, key in [("EYE TRACKING SCORE","Reading_eye"),("SELF-PACED READING SCORE","Reading_self_paced")]:
        m = re.search(re.escape(label)+r":\s*([+-]?[0-9.]+)", text)
        if m: out[key] = float(m.group(1))
    if "Reading_eye" in out and "Reading_self_paced" in out:
        out["Reading"] = (out["Reading_eye"]+out["Reading_self_paced"])/2
    return out

def eval_reading(mk, model_path, env):
    task_out = OUT / "outputs" / mk / "Reading"
    task_out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "evaluation_pipeline.reading.run",
           "--model_path_or_name", str(model_path.resolve()),
           "--backend","mlm",
           "--data_path","evaluation_data/fast_eval/reading/reading_data.csv",
           "--revision_name",f"s35_{mk}_Reading",
           "--output_dir", str(task_out.resolve())]
    p = subprocess.run(cmd, cwd=str(STRICT.resolve()), env=env,
                       capture_output=True, text=True, timeout=1800)
    scores = parse_reading(p.stdout)
    if not scores:
        for rpt in sorted(task_out.rglob("report.txt")):
            scores = parse_reading(rpt.read_text())
            if scores: break
    return scores

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    
    env = setup_env(args.gpu)
    results = {}
    
    for mk, mp in MODELS.items():
        if not mp.exists():
            print(f"SKIP {mk}: {mp} not found"); continue
        print(f"\n{'='*50}\n  {mk}\n{'='*50}")
        r = {}
        for col, task, dp, bs in TASKS:
            print(f"  {col}...", end=" ", flush=True)
            s = eval_col(mk, mp, col, task, dp, bs, env)
            r[col] = s
            print(f"{s:.2f}" if s is not None else "FAIL")
        
        # Reading
        print(f"  Reading...", end=" ", flush=True)
        rd = eval_reading(mk, mp, env)
        r.update(rd)
        print(f"{rd.get('Reading','FAIL')}")
        
        # Derived
        gp = r.get("GP_par"); gnp = r.get("GP_npar")
        if gp is not None and gnp is not None:
            r["GP_mean"] = (gp + gnp) / 2
        
        eq7 = ["BLiMP","Supplement","EWoK","COMPS","GP_mean","Reading","Entity_full"]
        vals = [r.get(c) for c in eq7]
        if all(v is not None for v in vals):
            r["equal7"] = sum(vals)/len(vals)
        
        results[mk] = r
    
    # Comparison
    print(f"\n{'='*50}")
    print("  COMPARISON: spatial_15M vs original_15M")
    print(f"{'='*50}")
    cols = ["BLiMP","Supplement","EWoK","Entity_full","COMPS","GP_mean","Reading","equal7"]
    comp = {}
    for c in cols:
        o = results.get("original_15M",{}).get(c)
        s = results.get("spatial_15M",{}).get(c)
        d = (s-o) if o is not None and s is not None else None
        comp[c] = {"orig": o, "spatial": s, "delta": d}
        if o is not None and s is not None:
            sign = "+" if d > 0 else ""
            print(f"  {c:15s}  orig={o:7.3f}  spatial={s:7.3f}  Δ={sign}{d:.3f}")
        else:
            print(f"  {c:15s}  incomplete")
    
    out = {"status": "reference_15M_QUICK_COMPARISON", "results": results, "comparison": comp}
    p = OUT / "quick_15m_comparison.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {p}")

if __name__ == "__main__":
    main()
