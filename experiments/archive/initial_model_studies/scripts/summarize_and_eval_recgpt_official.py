#!/usr/bin/env python3
"""research utilities for official-corpus public RecGPT run.

Modes:
  summarize: parse training stdout into JSON loss/checkpoint summary and smoke-load final model.
  eval: run local official-compatible causal evaluation on key BabyLM columns.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
RUNTIME = ROOT / "data/public_recgpt_runtime"
MODEL = RUNTIME / "models/recgpt_official_public_full_triton"
CKPTS = RUNTIME / "models/recgpt_official_public_full_triton-checkpoints"
SUMMARY = ROOT / "data/recgpt_official_training_summary.json"
STRICT = ROOT / "repos/babylm-eval/strict"
FULL = STRICT / "evaluation_data/full_eval"
EVAL_OUT_JSON = ROOT / "data/recgpt_official_final_causal_scores.json"
EVAL_LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/recgpt_official_final_causal_eval.log')
EVAL_OUTDIR = ROOT / "training/runs/recgpt_official_final_causal_eval"

TASKS = [
    ("blimp", "blimp", FULL / "blimp_filtered", "blimp_filtered", 8),
    ("supplement", "blimp", FULL / "supplement_filtered", "supplement_filtered", 8),
    ("entity_tracking", "entity_tracking", FULL / "entity_tracking", "entity_tracking", 8),
    ("comps", "comps", FULL / "comps", "comps", 8),
    ("global_piqa_parallel", "global_piqa_parallel", FULL / "global_piqa_parallel", "global_piqa_parallel", 8),
    ("global_piqa_nonparallel", "global_piqa_nonparallel", FULL / "global_piqa_nonparallel", "global_piqa_nonparallel", 8),
    ("ewok", "ewok", FULL / "ewok_filtered_word_tokenize", "ewok_filtered_word_tokenize", 8),
]
LOCAL_PUBLIC_REF = {
    "blimp": 73.20,
    "supplement": 63.24,
    "ewok": 52.82,
    "entity_tracking": 16.73,
    "comps": 55.47,
    "GlobalPIQA_mean": 38.71,
    "Reading_mean_spacefix": 2.03,
}
PROTECTED_DEBERTA = {
    "blimp": 66.76,
    "supplement": 59.88,
    "ewok": 52.19,
    "entity_tracking": 22.62,
    "comps": 52.19,
    "GlobalPIQA_mean": 35.635,
    "Reading_mean": 7.62,
}


def summarize(training_stdout: pathlib.Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    stdout = training_stdout.read_text(encoding="utf-8", errors="replace")
    rows = []
    pat = re.compile(r"Epoch (\d+)/(\d+) Step (\d+) training loss: ([0-9.eE+-]+) nextlat loss: ([0-9.eE+-]+) lr_embed ([0-9.eE+-]+) lr_block ([0-9.eE+-]+) lr_nl ([0-9.eE+-]+) step_time ([0-9.eE+-]+)s tok/s ([0-9.eE+-]+)")
    for m in pat.finditer(stdout):
        rows.append({
            "epoch": int(m.group(1)), "total_epochs": int(m.group(2)), "step": int(m.group(3)),
            "train_loss": float(m.group(4)), "nextlat_loss": float(m.group(5)),
            "lr_embed": float(m.group(6)), "lr_block": float(m.group(7)), "lr_nl": float(m.group(8)),
            "step_time": float(m.group(9)), "tok_per_sec": float(m.group(10)),
        })
    if not rows:
        raise ValueError("Training stdout contains no matching RecGPT training records")
    saves = re.findall(r"saved model to (.*)", stdout)
    ckpt_dirs = sorted([p.name for p in CKPTS.iterdir() if p.is_dir()]) if CKPTS.exists() else []
    # smoke-load final model
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True, torch_dtype=torch.float32).to("cuda").eval()
    enc = tok("The cat sat on the mat.", return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits
    by_epoch = {}
    for r in rows:
        by_epoch.setdefault(r["epoch"], []).append(r)
    epoch_summary = {str(e): {
        "n_logs": len(v),
        "mean_loss": sum(x["train_loss"] for x in v)/len(v),
        "last_loss": v[-1]["train_loss"],
        "mean_nextlat": sum(x["nextlat_loss"] for x in v)/len(v),
        "last_nextlat": v[-1]["nextlat_loss"],
    } for e, v in by_epoch.items()}
    payload = {
        "status": "RECGPT_OFFICIAL_TRAINING_COMPLETE_SUMMARY",
        "run_model": str(MODEL),
        "checkpoint_dir": str(CKPTS),
        "training_stdout": str(training_stdout),
        "n_log_rows": len(rows),
        "first_log": rows[0] if rows else None,
        "last_log": rows[-1] if rows else None,
        "epoch_summary": epoch_summary,
        "saved_model_paths_tail": saves[-8:],
        "checkpoint_dirs": ckpt_dirs,
        "checkpoint_count": len(ckpt_dirs),
        "smoke": {
            "tokenizer_class": tok.__class__.__name__, "vocab_size": len(tok), "pad_token_id": tok.pad_token_id,
            "model_class": model.__class__.__name__, "num_parameters": sum(p.numel() for p in model.parameters()),
            "logits_shape": list(logits.shape), "logits_finite": bool(torch.isfinite(logits).all().item()),
        },
        "interpretation": "Full public RecGPT learning system ran on official 10M-word corpus through 10 epochs. Ability evaluation is still required; training loss alone is not a BabyLM result.",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "last_log": payload["last_log"], "checkpoint_count": len(ckpt_dirs), "smoke": payload["smoke"]}, indent=2))


def env_setup():
    env = os.environ.copy()
    root_abs = pathlib.Path.cwd().resolve()
    env["NLTK_DATA"] = str((root_abs / ROOT / "data/nltk_data").resolve())
    hf = root_abs / ROOT / "training/hf_home"
    env["HF_HOME"] = str(hf.resolve())
    env["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    env["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    env["HF_MODULES_CACHE"] = str((root_abs / ROOT / "training/hf_modules_cache").resolve())
    env["PYTHONPATH"] = str((root_abs / STRICT).resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_cmd(cmd, env, logf):
    line = "$ " + " ".join(cmd)
    print(line, flush=True); logf.write("\n" + line + "\n"); logf.flush()
    p = subprocess.run(cmd, cwd=str(pathlib.Path.cwd()), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True); logf.write(p.stdout + f"\n[rc={p.returncode}]\n"); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {line}\n{p.stdout[-8000:]}")


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"AVERAGE ACCURACY\s*\n([0-9.\-]+)", txt)
    if m: return float(m.group(1))
    vals = re.findall(r"(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)", txt, flags=re.I)
    if vals:
        v = float(vals[-1]); return v * (100 if v <= 1 else 1)
    raise RuntimeError(f"Could not parse score from {report}")


def read_reading(report: pathlib.Path):
    txt = report.read_text(encoding="utf-8", errors="replace")
    out = {}
    for label, key in [("EYE TRACKING SCORE", "reading_eye_tracking"), ("SELF-PACED READING SCORE", "reading_self_paced")]:
        m = re.search(re.escape(label) + r":\s*([0-9.\-]+)", txt)
        if not m: raise RuntimeError(f"Could not parse {label}")
        out[key] = float(m.group(1))
    return out


def evaluate() -> None:
    t0 = time.time(); env = env_setup(); root_abs = pathlib.Path.cwd().resolve()
    model_abs = root_abs / MODEL; out_abs = root_abs / EVAL_OUTDIR
    out_abs.mkdir(parents=True, exist_ok=True); EVAL_OUT_JSON.parent.mkdir(parents=True, exist_ok=True); EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "RUNNING", "model_path": str(MODEL), "scores": {}, "reports": {}, "local_public_reference": LOCAL_PUBLIC_REF, "protected_deberta_reference": PROTECTED_DEBERTA,
               "notes": "Reading here uses the unmodified local strict evaluator; RecGPT Reading and GlobalPIQA coordinates have known public-vs-local mismatches that must be carried in interpretation."}
    EVAL_OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with EVAL_LOG.open("w", encoding="utf-8") as logf:
        for col, task, data, dataset, bs in TASKS:
            run_cmd([sys.executable, "-m", "evaluation_pipeline.sentence_zero_shot.run", "--model_path_or_name", str(model_abs), "--backend", "causal", "--task", task, "--data_path", str((root_abs / data).resolve()), "--save_predictions", "--revision_name", "recgpt_official_final", "--batch_size", str(bs), "--output_dir", str(out_abs)], env, logf)
            reports = sorted(out_abs.rglob(f"zero_shot/causal/{task}/{dataset}/best_temperature_report.txt"), key=lambda p: str(p))
            if not reports and col == "ewok": reports = sorted(out_abs.rglob("zero_shot/causal/ewok/*/best_temperature_report.txt"), key=lambda p: str(p))
            report = reports[-1]
            payload["reports"][col] = str(report); payload["scores"][col] = read_avg(report)
            if "global_piqa_parallel" in payload["scores"] and "global_piqa_nonparallel" in payload["scores"]:
                payload["scores"]["GlobalPIQA_mean"] = (payload["scores"]["global_piqa_parallel"] + payload["scores"]["global_piqa_nonparallel"]) / 2
            EVAL_OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        run_cmd([sys.executable, "-m", "evaluation_pipeline.reading.run", "--model_path_or_name", str(model_abs), "--backend", "causal", "--data_path", str((root_abs / FULL / "reading/reading_data.csv").resolve()), "--revision_name", "recgpt_official_final", "--output_dir", str(out_abs)], env, logf)
        reports = sorted(out_abs.rglob("zero_shot/causal/reading/report.txt"), key=lambda p: str(p)); report = reports[-1]
        payload["reports"]["reading"] = str(report); payload["scores"].update(read_reading(report)); payload["scores"]["Reading_mean"] = (payload["scores"]["reading_eye_tracking"] + payload["scores"]["reading_self_paced"]) / 2
    if all(k in payload["scores"] for k in ["blimp","supplement","ewok","entity_tracking","comps","GlobalPIQA_mean","Reading_mean"]):
        payload["scores"]["NLP_mean_no_superglue_aoa"] = sum(payload["scores"][k] for k in ["blimp","supplement","ewok","entity_tracking","comps","GlobalPIQA_mean","Reading_mean"]) / 7
    payload["deltas_vs_local_public_ref"] = {k: payload["scores"][k] - v for k, v in LOCAL_PUBLIC_REF.items() if k in payload["scores"]}
    payload["deltas_vs_protected_deberta"] = {k: payload["scores"][k] - v for k, v in PROTECTED_DEBERTA.items() if k in payload["scores"]}
    payload["elapsed_sec"] = round(time.time() - t0, 1); payload["status"] = "COMPLETE"
    EVAL_OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(EVAL_OUT_JSON), "scores": payload["scores"], "deltas_vs_local_public_ref": payload["deltas_vs_local_public_ref"]}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["summarize", "eval"])
    ap.add_argument("--training-stdout", type=pathlib.Path, help="Scientific stdout from the original RecGPT training run.")
    args = ap.parse_args()
    if args.mode == "summarize":
        if args.training_stdout is None or not args.training_stdout.is_file():
            ap.error("summarize requires an existing --training-stdout file")
        summarize(args.training_stdout)
    else:
        evaluate()

if __name__ == "__main__":
    main()
