#!/usr/bin/env python3
"""research SCORE-FREEZE: complete missing AoA and (Super)GLUE for official-corpus RecGPT.

No new training route or model modification. This script evaluates the already trained
candidate and aggregates the full 9-column Overall once AoA and SuperGLUE are done.
AoA and SuperGLUE are launched concurrently by `run_all` on separate GPUs when available.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
FINAL_MODEL = ROOT / "data/public_recgpt_runtime/models/recgpt_official_public_full_triton"
CKPTS = ROOT / "data/public_recgpt_runtime/models/recgpt_official_public_full_triton-checkpoints"
BASE7 = ROOT / "data/recgpt_official_final_causal_scores.json"
AOA_OUTDIR = ROOT / "training/runs/recgpt_official_aoa"
SUPER_OUT_BASE = ROOT / "training/runs/recgpt_official_superglue"
SUPER_RESULTS = SUPER_OUT_BASE / "results"
SUPER_MODELS = SUPER_OUT_BASE / "models"
AOA_JSON = ROOT / "data/recgpt_official_aoa_result.json"
SUPER_JSON = ROOT / "data/recgpt_official_superglue_result.json"
OVERALL_JSON = ROOT / "data/recgpt_official_full_overall.json"
LOGDIR = (_PUBLIC_ROOT / 'research/notes/initial_model_studies')
AOA_LOG = (_PUBLIC_ROOT / 'research/notes/initial_model_studies/recgpt_official_aoa.log')
SUPER_LOG = (_PUBLIC_ROOT / 'research/notes/initial_model_studies/recgpt_official_superglue.log')
WORD_PATH = STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json"
GLUE_DATA = STRICT / "evaluation_data/full_eval/glue_filtered"

TASKS = [
    {"task":"boolq", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"multirc", "num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"rte", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"wsc", "num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":30},
    {"task":"mrpc", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"qqp", "num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    {"task":"mnli", "num_labels":3, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy"], "epochs":10},
]


def env(cuda: str | None = None) -> dict[str, str]:
    e = os.environ.copy()
    hf = ROOT / "training/hf_home"
    e["HF_HOME"] = str(hf.resolve())
    e["HF_HUB_CACHE"] = str((hf / "hub").resolve())
    e["TRANSFORMERS_CACHE"] = str((hf / "transformers").resolve())
    e["HF_MODULES_CACHE"] = str((ROOT / "training/hf_modules_cache").resolve())
    e.setdefault("TOKENIZERS_PARALLELISM", "false")
    if cuda is not None:
        e["CUDA_VISIBLE_DEVICES"] = cuda
    return e


def run_aoa() -> None:
    """Run AoA using official evaluation logic with a local checkpoint-path loader repair."""
    sys.path.insert(0, str(STRICT.resolve()))
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    from evaluation_pipeline.AoA_word.eval_util import StepConfig, load_eval, JsonProcessor
    from evaluation_pipeline.AoA_word.evaluation_functions import StepSurprisalExtractor
    from evaluation_pipeline.utils import AoAEvaluator

    class LocalCheckpointExtractor(StepSurprisalExtractor):
        def load_model_for_step(self, step):
            model = AutoModelForCausalLM.from_pretrained(str(CKPTS / step), trust_remote_code=True).to(self.device)
            model.eval(); return model
        def load_tokenizer_for_step(self, step):
            processor = AutoProcessor.from_pretrained(str(CKPTS / step), trust_remote_code=True, padding_side="right")
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            return processor, tokenizer

    AOA_OUTDIR.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    target_words, contexts = load_eval(WORD_PATH.resolve(), min_context=20, debug=False)
    result_dir = AOA_OUTDIR / FINAL_MODEL.name / "main" / "zero_shot" / "causal" / "AoA_word"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "surprisal.json"
    cfg = StepConfig(resume=False, track="strict-small", debug=False, file_path=None)
    extractor = LocalCheckpointExtractor(cfg, str(FINAL_MODEL.resolve()), "causal", "cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    results_data = extractor.analyze_steps(contexts=contexts, target_words=target_words, resume_path=None)
    JsonProcessor.save_json(results_data, result_file)
    cdi_human = WORD_PATH.resolve().parent / "cdi_human.csv"
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL.resolve()), trust_remote_code=True)
    score = AoAEvaluator(cdi_human).compute_curve_fitness(results_data, tokenizer)["curve_fitness"] if cdi_human.is_file() else 0.0
    score_file = result_dir / "aoa_score.json"
    JsonProcessor.save_json({"aoa": score}, score_file)
    payload = {"status":"AOA_COMPLETE", "model_path":str(FINAL_MODEL), "checkpoint_dir":str(CKPTS), "backend":"causal", "track_name":"strict-small", "word_path":str(WORD_PATH), "output_dir":str(AOA_OUTDIR), "surprisal_path":str(result_file), "score_path":str(score_file), "aoa":float(score), "elapsed_sec":round(time.time()-t0,1), "note":"Official AoA scoring logic with local checkpoint path loader repair; checkpoint names chck_1M..chck_100M preserved."}
    AOA_JSON.parent.mkdir(parents=True, exist_ok=True); AOA_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    AOA_LOG.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"status":"AOA_DONE","aoa":score,"out":str(AOA_JSON)}, indent=2))


def run_one_super(spec, logf):
    t = spec["task"]
    cmd = [sys.executable, "-m", "evaluation_pipeline.finetune.run",
           "--model_name_or_path", str(FINAL_MODEL.resolve()),
           "--train_data", str((GLUE_DATA / f"{t}.train.jsonl").resolve()),
           "--valid_data", str((GLUE_DATA / f"{t}.valid.jsonl").resolve()),
           "--predict_data", str((GLUE_DATA / f"{t}.valid.jsonl").resolve()),
           "--task", t, "--num_labels", str(spec["num_labels"]), "--batch_size", str(spec["batch_size"]),
           "--learning_rate", "3e-5", "--num_epochs", str(spec["epochs"]), "--sequence_length", "512",
           "--results_dir", str(SUPER_RESULTS.resolve()), "--save", "--save_dir", str(SUPER_MODELS.resolve()),
           "--metrics", *spec["metrics"], "--metric_for_valid", spec["metric_for_valid"], "--seed", "42",
           "--verbose", "--padding_side", "left", "--take_final"]
    logf.write("\n$ "+" ".join(cmd)+"\n"); logf.flush()
    p = subprocess.run(cmd, cwd=str(STRICT), env=env(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logf.write(p.stdout + f"\n[returncode={p.returncode}]\n"); logf.flush(); print(p.stdout[-3000:])
    if p.returncode != 0: raise RuntimeError(f"{t} failed rc={p.returncode}")
    pred = SUPER_RESULTS / FINAL_MODEL.name / "main" / "finetune" / t / "predictions.json"
    if not pred.exists():
        hits = list(SUPER_RESULTS.rglob(f"finetune/{t}/predictions.json"))
        if not hits: raise FileNotFoundError(f"missing predictions for {t}")
        pred = hits[-1]
    return {"task":t, "predictions":str(pred), "num_predictions":len(json.loads(pred.read_text())[t]["predictions"])}


def run_superglue() -> None:
    SUPER_OUT_BASE.mkdir(parents=True, exist_ok=True); SUPER_RESULTS.mkdir(parents=True, exist_ok=True); SUPER_MODELS.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    rows=[]; t0=time.time()
    with SUPER_LOG.open("w", encoding="utf-8") as logf:
        for spec in TASKS:
            rows.append(run_one_super(spec, logf))
    # Aggregate validation accuracy exactly as previous coordinate scripts did.
    subtasks=[]
    for row in rows:
        task=row["task"]; pred=json.loads(pathlib.Path(row["predictions"]).read_text())[task]["predictions"]
        labels=[json.loads(line)["label"] for line in (GLUE_DATA/f"{task}.valid.jsonl").read_text().splitlines() if line.strip()]
        correct=sum(1 for p,y in zip(pred, labels) if int(p["pred"])==int(y)); acc=correct/len(labels)*100.0
        subtasks.append({"task":task,"predictions":row["predictions"],"num_examples":len(labels),"correct":correct,"accuracy":acc})
    score=sum(r["accuracy"] for r in subtasks)/len(subtasks)
    payload={"status":"SUPERGLUE_COMPLETE","model_path":str(FINAL_MODEL),"results_dir":str(SUPER_RESULTS),"models_dir":str(SUPER_MODELS),"tasks":rows,"subtasks":subtasks,"superglue":score,"elapsed_sec":round(time.time()-t0,1)}
    SUPER_JSON.parent.mkdir(parents=True, exist_ok=True); SUPER_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"status":"SUPERGLUE_DONE","superglue":score,"out":str(SUPER_JSON)}, indent=2))


def aggregate_if_ready() -> None:
    if not (AOA_JSON.exists() and SUPER_JSON.exists() and BASE7.exists()):
        return
    base=json.loads(BASE7.read_text()); aoa=json.loads(AOA_JSON.read_text()); sup=json.loads(SUPER_JSON.read_text())
    s=base["scores"]
    columns={"BLiMP":s["blimp"],"Supplement":s["supplement"],"EWoK":s["ewok"],"Entity Tracking":s["entity_tracking"],"COMPS":s["comps"],"(Super)GLUE":sup["superglue"],"GlobalPIQA":s["GlobalPIQA_mean"],"Reading":s["Reading_mean"],"AoA":aoa["aoa"]}
    overall=sum(columns.values())/len(columns)
    payload={"status":"FULL_9_COLUMN_OVERALL_COMPLETE","model_path":str(FINAL_MODEL),"columns":columns,"overall":overall,"nlp_mean_six_with_superglue":sum(columns[k] for k in ["BLiMP","Supplement","EWoK","Entity Tracking","COMPS","(Super)GLUE","GlobalPIQA"])/7.0,"human_like_mean":(columns["Reading"]+columns["AoA"])/2.0,"input_files":{"base7":str(BASE7),"aoa":str(AOA_JSON),"superglue":str(SUPER_JSON)},"known_coordinate_caveat":"Reading and GlobalPIQA local-vs-public evaluator offsets from RecGPT reference remain documented; this JSON is the local official-compatible full coordinate for the trained official-corpus candidate."}
    OVERALL_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"status":"OVERALL_DONE","overall":overall,"out":str(OVERALL_JSON)}, indent=2))


def run_all() -> None:
    # Concurrent AoA and SuperGLUE on separate visible GPUs; if only one GPU is visible, the caller can run modes separately.
    py=sys.executable; script=_public_path('experiments/archive/initial_model_studies/scripts/score_freeze_missing_columns.py')
    p_aoa=subprocess.Popen([py, str(script), "aoa"], env=env("1"), cwd=str(pathlib.Path.cwd()))
    p_sup=subprocess.Popen([py, str(script), "superglue"], env=env("0"), cwd=str(pathlib.Path.cwd()))
    rc1=p_aoa.wait(); rc2=p_sup.wait()
    if rc1!=0 or rc2!=0: raise SystemExit(f"parallel eval failed: aoa={rc1} superglue={rc2}")
    aggregate_if_ready()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode", choices=["aoa","superglue","aggregate","run_all"]); args=ap.parse_args()
    if args.mode=="aoa": run_aoa()
    elif args.mode=="superglue": run_superglue()
    elif args.mode=="aggregate": aggregate_if_ready()
    else: run_all()

if __name__=="__main__": main()
