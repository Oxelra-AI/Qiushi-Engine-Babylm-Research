#!/usr/bin/env python3
"""research: evaluate format-alignment replay endpoints via proven research pipeline.

Creates the directory structure that research's evaluator expects from the trainer output,
then runs full cheap7 evaluation and generates per_target payloads for item comparison.

Usage:
  python eval_format_replay.py --train-dir <path_to_trained_arm> \
    --label <arm_label> --gpu <gpu>
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, shutil, subprocess, sys, time

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
WS = _public_path('experiments/archive/relation_learning')

REPRESENTATION_FRONTIER_STUDIES_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')
MATERIALIZER = _public_path('experiments/archive/frontier_consolidation/scripts/materialize_private_scale.py')

def rel(p): 
    try: return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except: return str(p)

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def run_cmd(cmd, timeout=7200, gpu=0):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"RUN: {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run([str(c) for c in cmd], env=env, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"Command failed: {result.returncode}")
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, help="Trainer output dir (e.g. isolated_all_replay)")
    ap.add_argument("--label", required=True, help="Evaluation label (e.g. isolated_all_alpha0p75)")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-root", default="")
    args = ap.parse_args()

    train_dir = pathlib.Path(args.train_dir)
    if not train_dir.is_absolute():
        train_dir = ROOT / train_dir

    # Find the checkpoint: trainer saves at checkpoint/ and alpha_X.XX/
    alpha_label = f"alpha_{args.alpha:.2f}"
    alpha_src = train_dir / alpha_label
    ckpt_src = train_dir / "checkpoint"
    
    if alpha_src.exists() and (alpha_src / "model.safetensors").exists():
        model_src = alpha_src
        print(f"Using pre-materialized alpha endpoint: {rel(model_src)}", flush=True)
    elif ckpt_src.exists() and (ckpt_src / "model.safetensors").exists():
        # Materialize from checkpoint
        model_src = train_dir / f"materialized_{alpha_label}"
        model_src.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", str(MATERIALIZER),
               "--source", str(ckpt_src), "--output", str(model_src),
               "--scale", str(args.alpha), "--force"]
        run_cmd(cmd, timeout=300, gpu=args.gpu)
        print(f"Materialized alpha={args.alpha} at {rel(model_src)}", flush=True)
    else:
        raise FileNotFoundError(f"No checkpoint found in {train_dir}")

    # Create run-like directory structure for research evaluator
    run_like = train_dir / f"runlike_{alpha_label}"
    hf_final = run_like / "hf_model" / "final"
    hf_final.mkdir(parents=True, exist_ok=True)
    
    # Copy model files (symlinks don't work well across sandbox mounts)
    for f in model_src.iterdir():
        dst = hf_final / f.name
        if not dst.exists():
            if f.is_file():
                shutil.copy2(str(f), str(dst))
            elif f.is_dir():
                shutil.copytree(str(f), str(dst))
    
    # Create minimal scientific_metrics.json
    train_config = {}
    tc_path = train_dir / "train_config.json"
    if tc_path.exists():
        train_config = json.loads(tc_path.read_text())
    
    metrics = {
        "status": "FORMAT_REPLAY_ENDPOINT",
        "arm_label": train_config.get("arm_label", args.label),
        "alpha": args.alpha,
        "total_words": train_config.get("total_words"),
        "macro_updates": train_config.get("macro_updates"),
        "total_params": train_config.get("total_params", 36458592),
        "private_params": train_config.get("trainable_params", 995584),
        "initial_consumed_words": 82012495,
        "tail_main_word_exposure": train_config.get("total_words", 3992800),
        "endpoint": "final",
        "created_utc": now(),
    }
    (run_like / "scientific_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Run research evaluator
    eval_root = pathlib.Path(args.eval_root) if args.eval_root else (
        WS / f"data/eval_{args.label}")
    
    cmd = [
        sys.executable, "-B", str(EVALUATOR),
        "--run-dir", str(run_like),
        "--target", args.label,
        "--endpoint", "final",
        "--out-root", str(eval_root / "eval"),
        "--collate-root", str(eval_root / "collate"),
        "--summary-root", str(eval_root / "summary"),
        "--gpu", str(args.gpu),
    ]
    result = run_cmd(cmd, timeout=7200, gpu=args.gpu)
    
    # Collect results
    summary_path = eval_root / "summary" / f"{args.label}_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        out = {
            "status": "FORMAT_REPLAY_EVAL_DONE",
            "arm_label": args.label,
            "alpha": args.alpha,
            "scores": summary.get("scores"),
            "cheap7": summary.get("cheap7"),
            "cheap7_delta_vs_chck82": summary.get("cheap7_delta_vs_chck82"),
            "eval_summary_path": rel(summary_path),
            "per_target_path": rel(eval_root / "eval" / "per_target" / f"{args.label}.json"),
            "created_utc": now(),
        }
        (eval_root / "eval_result.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2), flush=True)
    else:
        # Check for partial results
        print(f"WARNING: Summary not found at {summary_path}", flush=True)
        print(f"Eval stdout tail: {result.stdout[-2000:]}", flush=True)

if __name__ == "__main__":
    main()
