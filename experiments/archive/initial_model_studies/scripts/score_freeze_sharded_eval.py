#!/usr/bin/env python3
"""research SCORE-FREEZE sharded evaluation for the frozen official-corpus RecGPT candidate.

Only evaluation/runtime repair. No model training or research route changes.

Modes:
  aoa_shard --steps chck_1M,... --batch_size N
      Batched implementation of the official causal AoA phrase-mask surprisal for selected checkpoints.
      Writes one durable JSON per checkpoint immediately.
  super_task --task boolq|... 
      Runs one official SuperGLUE fine-tuning task and writes one durable JSON.
  aggregate
      Combines existing 7/9 scores, all AoA checkpoint JSONs, all SuperGLUE task JSONs, and writes 9/9 Overall.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, re, subprocess, sys, time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
sys.path.insert(0, str(STRICT.resolve()))
from evaluation_pipeline.AoA_word.eval_util import StepConfig, load_eval, JsonProcessor
from evaluation_pipeline.utils import AoAEvaluator

FINAL_MODEL = ROOT / "data/public_recgpt_runtime/models/recgpt_official_public_full_triton"
CKPTS = ROOT / "data/public_recgpt_runtime/models/recgpt_official_public_full_triton-checkpoints"
BASE7 = ROOT / "data/recgpt_official_final_causal_scores.json"
WORD_PATH = STRICT / "evaluation_data/full_eval/aoa/cdi_childes.json"
GLUE_DATA = STRICT / "evaluation_data/full_eval/glue_filtered"
AOA_DIR = ROOT / "training/runs/recgpt_official_aoa_sharded"
AOA_PARTS = AOA_DIR / "parts"
SUPER_BASE = ROOT / "training/runs/recgpt_official_superglue_sharded"
SUPER_RESULTS = SUPER_BASE / "results"
SUPER_MODELS = SUPER_BASE / "models"
SUPER_TASK_JSON_DIR = SUPER_BASE / "task_json"
DATA_DIR = ROOT / "data"
AOA_JSON = DATA_DIR / "recgpt_official_aoa_result.json"
SUPER_JSON = DATA_DIR / "recgpt_official_superglue_result.json"
OVERALL_JSON = DATA_DIR / "recgpt_official_full_overall.json"
LOGDIR = (ROOT.parents[2] / 'research/notes/initial_model_studies')

TASK_SPECS = {
    "boolq": {"num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    "multirc": {"num_labels":2, "batch_size":16, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    "rte": {"num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":10},
    "wsc": {"num_labels":2, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy","f1","mcc"], "epochs":30},
    "mrpc": {"num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    "qqp": {"num_labels":2, "batch_size":32, "metric_for_valid":"f1", "metrics":["accuracy","f1","mcc"], "epochs":10},
    "mnli": {"num_labels":3, "batch_size":32, "metric_for_valid":"accuracy", "metrics":["accuracy"], "epochs":10},
}


def set_env_defaults(cuda: str | None = None) -> dict[str, str]:
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


def steps_and_counts() -> tuple[list[str], dict[str, int]]:
    cfg = StepConfig(track="strict-small", debug=False)
    return list(cfg.steps), {s: int(w) for s, w in zip(cfg.steps, cfg.word_counts)}


def _processor_tokenizer(path: pathlib.Path):
    try:
        processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, padding_side="right")
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, padding_side="right")
        processor = tokenizer
    return processor, tokenizer


def build_aoa_examples():
    target_words, contexts = load_eval(WORD_PATH.resolve(), min_context=20, debug=False)
    examples = []
    for word_contexts, target_word in zip(contexts, target_words, strict=False):
        for context_idx, context in enumerate(word_contexts):
            examples.append((context, target_word, context_idx))
    return examples


def encode_example(processor, context: str, target_word: str):
    # Matches official StepSurprisalExtractor.compute_surprisal -> process_causal_input with use_bos_only=False.
    context2 = context.strip() + " "
    target2 = target_word.strip()
    input_text = context2 + target2
    tok_out = processor(text=input_text, return_offsets_mapping=True, add_special_tokens=True)
    ids = tok_out["input_ids"]
    att = tok_out["attention_mask"]
    start_char_idx = len(input_text) - len(target2)
    phrase_mask_full = [0] * len(ids)
    for i, (start, end) in enumerate(tok_out["offset_mapping"]):
        if end > start_char_idx:
            phrase_mask_full[i] = 1
    return ids[:-1], ids[1:], att[:-1], phrase_mask_full[1:]


def batched_aoa_for_step(step: str, batch_size: int = 64, max_examples: int = -1) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = CKPTS / step
    word_counts = steps_and_counts()[1]
    processor, tokenizer = _processor_tokenizer(ckpt)
    model = AutoModelForCausalLM.from_pretrained(str(ckpt), trust_remote_code=True).to(device).eval()
    examples = build_aoa_examples()
    if max_examples > 0:
        examples = examples[:max_examples]
    results = []
    t0 = time.time()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start+batch_size]
            encs = [encode_example(processor, c, w) for c, w, _ in chunk]
            max_len = max(len(e[0]) for e in encs)
            input_ids=[]; target_ids=[]; attn=[]; pm=[]
            for inp, tgt, att, mask in encs:
                pad = max_len - len(inp)
                input_ids.append(inp + [pad_id]*pad)
                target_ids.append(tgt + [pad_id]*pad)
                attn.append(att + [0]*pad)
                pm.append(mask + [0]*pad)
            input_ids_t=torch.tensor(input_ids, dtype=torch.long, device=device)
            target_ids_t=torch.tensor(target_ids, dtype=torch.long, device=device)
            attn_t=torch.tensor(attn, dtype=torch.long, device=device)
            pm_t=torch.tensor(pm, dtype=torch.float32, device=device)
            out=model(input_ids=input_ids_t, attention_mask=attn_t)
            logits=out[0] if isinstance(out, tuple) else out.logits
            log_probs=F.log_softmax(logits, dim=-1)
            target_lp=torch.gather(log_probs, -1, target_ids_t.unsqueeze(-1)).squeeze(-1)
            surprisal=(-torch.sum(target_lp * pm_t, dim=1)).detach().cpu().tolist()
            for (context, target_word, context_idx), sp in zip(chunk, surprisal):
                results.append({"step": step, "word_count": word_counts[step], "target_word": target_word, "context_id": context_idx, "context": context, "surprisal": float(sp)})
    payload={"metadata":{"model_name":str(FINAL_MODEL),"checkpoint":step,"backend":"causal","use_bos_only":False,"total_steps":19,"completed_steps":1,"num_results":len(results),"batch_size":batch_size,"elapsed_sec":round(time.time()-t0,1)},"results":results}
    AOA_PARTS.mkdir(parents=True, exist_ok=True)
    part=AOA_PARTS / f"surprisal_{step}.json"
    JsonProcessor.save_json(payload, part)
    print(json.dumps({"status":"AOA_STEP_DONE","step":step,"n":len(results),"elapsed_sec":payload["metadata"]["elapsed_sec"],"out":str(part)}, indent=2))
    return payload


def run_aoa_shard(steps: list[str], batch_size: int, max_examples: int = -1) -> None:
    for step in steps:
        part=AOA_PARTS / f"surprisal_{step}.json"
        if part.exists() and max_examples < 0:
            print(json.dumps({"status":"AOA_STEP_SKIP_EXISTS","step":step,"out":str(part)}))
            continue
        batched_aoa_for_step(step, batch_size=batch_size, max_examples=max_examples)


def run_super_task(task: str) -> None:
    spec = TASK_SPECS[task]
    SUPER_BASE.mkdir(parents=True, exist_ok=True); SUPER_RESULTS.mkdir(parents=True, exist_ok=True); SUPER_MODELS.mkdir(parents=True, exist_ok=True); SUPER_TASK_JSON_DIR.mkdir(parents=True, exist_ok=True); LOGDIR.mkdir(parents=True, exist_ok=True)
    out_json = SUPER_TASK_JSON_DIR / f"{task}.json"
    if out_json.exists():
        print(json.dumps({"status":"SUPER_TASK_SKIP_EXISTS","task":task,"out":str(out_json)})); return
    env = set_env_defaults(os.environ.get("CUDA_VISIBLE_DEVICES"))
    cmd = [sys.executable, "-m", "evaluation_pipeline.finetune.run",
           "--model_name_or_path", str(FINAL_MODEL.resolve()),
           "--train_data", str((GLUE_DATA / f"{task}.train.jsonl").resolve()),
           "--valid_data", str((GLUE_DATA / f"{task}.valid.jsonl").resolve()),
           "--predict_data", str((GLUE_DATA / f"{task}.valid.jsonl").resolve()),
           "--task", task, "--num_labels", str(spec["num_labels"]), "--batch_size", str(spec["batch_size"]),
           "--learning_rate", "3e-5", "--num_epochs", str(spec["epochs"]), "--sequence_length", "512",
           "--results_dir", str(SUPER_RESULTS.resolve()), "--save", "--save_dir", str(SUPER_MODELS.resolve()),
           "--metrics", *spec["metrics"], "--metric_for_valid", spec["metric_for_valid"], "--seed", "42", "--verbose", "--padding_side", "left", "--take_final"]
    log = LOGDIR / f"superglue_{task}.log"
    t0=time.time()
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n"); f.flush()
        p=subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        f.write(p.stdout + f"\n[returncode={p.returncode}]\n")
    if p.returncode != 0:
        raise SystemExit(f"{task} failed rc={p.returncode}; see {log}")
    pred = SUPER_RESULTS / FINAL_MODEL.name / "main" / "finetune" / task / "predictions.json"
    if not pred.exists():
        hits=list(SUPER_RESULTS.rglob(f"finetune/{task}/predictions.json"))
        if not hits: raise FileNotFoundError(f"missing predictions for {task}")
        pred=hits[-1]
    preds=json.loads(pred.read_text())[task]["predictions"]
    labels=[json.loads(line)["label"] for line in (GLUE_DATA/f"{task}.valid.jsonl").read_text().splitlines() if line.strip()]
    correct=sum(1 for p2,y in zip(preds, labels) if int(p2["pred"]) == int(y))
    acc=correct/len(labels)*100.0
    payload={"status":"SUPER_TASK_COMPLETE","task":task,"predictions":str(pred),"num_examples":len(labels),"correct":correct,"accuracy":acc,"elapsed_sec":round(time.time()-t0,1),"log":str(log)}
    out_json.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload, indent=2))


def aggregate() -> None:
    all_steps, word_counts = steps_and_counts()
    results=[]; missing=[]
    for step in all_steps:
        part=AOA_PARTS / f"surprisal_{step}.json"
        if not part.exists():
            missing.append(step); continue
        data=json.loads(part.read_text())
        results.extend(data.get("results", []))
    if missing:
        print(json.dumps({"status":"AOA_INCOMPLETE","missing_steps":missing,"have":len(all_steps)-len(missing)}))
    else:
        aoa_data={"metadata":{"model_name":str(FINAL_MODEL),"use_bos_only":False,"total_steps":19,"completed_steps":19,"sharded":True},"results":results}
        cdi_human = WORD_PATH.parent / "cdi_human.csv"
        tok = AutoTokenizer.from_pretrained(str(FINAL_MODEL.resolve()), trust_remote_code=True)
        score = AoAEvaluator(cdi_human).compute_curve_fitness(aoa_data, tok)["curve_fitness"]
        AOA_JSON.write_text(json.dumps({"status":"AOA_COMPLETE","aoa":float(score),"surprisal_parts":str(AOA_PARTS),"num_results":len(results),"backend":"causal","track_name":"strict-small"}, indent=2)+"\n")
    tasks=[]; missing_tasks=[]
    for task in TASK_SPECS:
        path=SUPER_TASK_JSON_DIR / f"{task}.json"
        if not path.exists(): missing_tasks.append(task)
        else: tasks.append(json.loads(path.read_text()))
    if missing_tasks:
        print(json.dumps({"status":"SUPERGLUE_INCOMPLETE","missing_tasks":missing_tasks,"have":len(tasks)}))
    else:
        score=sum(t["accuracy"] for t in tasks)/len(tasks)
        SUPER_JSON.write_text(json.dumps({"status":"SUPERGLUE_COMPLETE","superglue":score,"subtasks":tasks,"results_dir":str(SUPER_RESULTS)}, indent=2)+"\n")
    if not (AOA_JSON.exists() and SUPER_JSON.exists() and BASE7.exists()):
        return
    base=json.loads(BASE7.read_text()); aoa=json.loads(AOA_JSON.read_text()); sup=json.loads(SUPER_JSON.read_text())
    s=base["scores"]
    columns={"BLiMP":s["blimp"],"Supplement":s["supplement"],"EWoK":s["ewok"],"Entity Tracking":s["entity_tracking"],"COMPS":s["comps"],"(Super)GLUE":sup["superglue"],"GlobalPIQA":s["GlobalPIQA_mean"],"Reading":s["Reading_mean"],"AoA":aoa["aoa"]}
    overall=sum(columns.values())/9.0
    payload={"status":"FULL_9_COLUMN_OVERALL_COMPLETE","model_path":str(FINAL_MODEL),"columns":columns,"overall":overall,"nlp_mean_seven":sum(columns[k] for k in ["BLiMP","Supplement","EWoK","Entity Tracking","COMPS","(Super)GLUE","GlobalPIQA"])/7.0,"human_like_mean":(columns["Reading"]+columns["AoA"])/2.0,"input_files":{"base7":str(BASE7),"aoa":str(AOA_JSON),"superglue":str(SUPER_JSON)},"reference":{"protected_deberta_overall":40.5269,"public_leader_overall":41.8011},"known_coordinate_caveat":"Reading and GlobalPIQA local-vs-public evaluator offsets from RecGPT reference remain documented; this is local official-compatible full coordinate for the trained official-corpus candidate."}
    OVERALL_JSON.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps({"status":"OVERALL_DONE","overall":overall,"out":str(OVERALL_JSON)}, indent=2))


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="mode", required=True)
    p=sub.add_parser("aoa_shard"); p.add_argument("--steps", required=True); p.add_argument("--batch_size", type=int, default=32); p.add_argument("--max_examples", type=int, default=-1)
    p=sub.add_parser("super_task"); p.add_argument("--task", required=True, choices=list(TASK_SPECS))
    sub.add_parser("aggregate")
    args=ap.parse_args()
    if args.mode == "aoa_shard":
        run_aoa_shard([s for s in args.steps.split(',') if s], args.batch_size, args.max_examples)
    elif args.mode == "super_task":
        run_super_task(args.task)
    else:
        aggregate()

if __name__ == "__main__":
    main()
