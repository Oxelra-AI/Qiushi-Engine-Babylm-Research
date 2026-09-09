#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
MODEL = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model"
STRICT = ROOT / "repos/babylm-eval/strict"
DATA_ROOT = STRICT / "evaluation_data/full_eval"
OFFICIAL_PRED_ROOT = MODEL.parent / "eval_results_available/hf_model/chck_100M/zero_shot/mlm"
OUT_JSON = ROOT / "data/debertav2_globalpiqa_score_variants.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_globalpiqa_score_variants.md')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def batch_forward(model, rows, batch_size=64):
    vals = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        pad = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        input_ids = []
        attn = []
        idxs = []
        targets = []
        for r in batch:
            ids = r["input_ids"]
            input_ids.append(ids + [pad] * (max_len - len(ids)))
            attn.append([1] * len(ids) + [0] * (max_len - len(ids)))
            idxs.append(r["mask_idx"])
            targets.append(r["target"])
        input_ids_t = torch.tensor(input_ids, device=DEVICE)
        attn_t = torch.tensor(attn, device=DEVICE)
        idxs_t = torch.tensor(idxs, device=DEVICE)
        targets_t = torch.tensor(targets, device=DEVICE)
        with torch.no_grad():
            out = model(input_ids=input_ids_t, attention_mask=attn_t)
            logits = out.logits
            mb = torch.arange(logits.shape[0], device=DEVICE)
            masked = logits[mb, idxs_t]
            lp = torch.log_softmax(masked, dim=-1)
            vals.extend(torch.gather(lp, -1, targets_t[:, None]).squeeze(-1).detach().cpu().tolist())
    return vals


def score_candidate(tokenizer, model, prompt: str, solution: str):
    completion = " " + solution
    sentence = " ".join([prompt, solution])
    tok = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    ids = tok["input_ids"]
    offsets = tok["offset_mapping"]
    start_char_idx = len(sentence) - len(completion)
    phrase_positions = []
    target_tokens = []
    for i, (start, end) in enumerate(offsets):
        if end > start_char_idx:
            phrase_positions.append(i)
            target_tokens.append(ids[i])
    if not phrase_positions:
        return {"sum": float("-inf"), "mean": float("-inf"), "first": float("-inf"), "n_tokens": 0, "target_tokens": []}
    rows = []
    mask_id = tokenizer.mask_token_id
    for pos, target in zip(phrase_positions, target_tokens):
        cur = list(ids)
        cur[pos] = mask_id
        rows.append({"input_ids": cur, "mask_idx": pos, "target": target})
    vals = batch_forward(model, rows, batch_size=64)
    return {"sum": float(sum(vals)), "mean": float(sum(vals) / max(len(vals), 1)), "first": float(vals[0]), "n_tokens": len(vals), "target_tokens": target_tokens}


def summarize(records, key):
    n = len(records)
    correct = sum(1 for r in records if r[f"pred_{key}"] == r["label"])
    by_cat = defaultdict(lambda: [0, 0])
    by_label = defaultdict(lambda: [0, 0])
    for r in records:
        ok = int(r[f"pred_{key}"] == r["label"])
        by_label[r["label"]][0] += ok; by_label[r["label"]][1] += 1
        for cat in str(r["categories"]).split(","):
            cat = cat.strip()
            if cat:
                by_cat[cat][0] += ok; by_cat[cat][1] += 1
    return {
        "accuracy": correct / n * 100.0,
        "correct": correct,
        "n": n,
        "pred_distribution": dict(Counter(r[f"pred_{key}"] for r in records)),
        "accuracy_by_label": {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_label.items())},
        "accuracy_by_category": {k: {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_cat.items(), key=lambda kv: (-kv[1][1], kv[0]))},
    }


def load_official_pred(split, uid):
    path = OFFICIAL_PRED_ROOT / split / split / "predictions.json"
    preds = json.loads(path.read_text(encoding="utf-8"))
    return preds[uid]["predictions"][0]["pred"].strip()


def process_split(split, tokenizer, model):
    rows = read_jsonl(DATA_ROOT / split / "eng_latn.jsonl")
    pred_json = json.loads((OFFICIAL_PRED_ROOT / split / split / "predictions.json").read_text(encoding="utf-8"))
    records = []
    for raw in rows:
        sol_keys = sorted([k for k in raw if k.startswith("solution") and k[len("solution"):].isdigit()], key=lambda x: int(x[len("solution"):]))
        sols = [raw[k] for k in sol_keys]
        scores = [score_candidate(tokenizer, model, raw["prompt"], s) for s in sols]
        pred_mean = max(range(len(sols)), key=lambda i: scores[i]["mean"])
        pred_sum = max(range(len(sols)), key=lambda i: scores[i]["sum"])
        pred_first = max(range(len(sols)), key=lambda i: scores[i]["first"])
        official_text = pred_json[raw["example_id"]]["predictions"][0]["pred"].strip()
        official_matches = [i for i, s in enumerate(sols) if s.strip() == official_text]
        rec = {
            "uid": raw["example_id"],
            "prompt": raw["prompt"],
            "categories": raw.get("categories", ""),
            "label": int(raw["label"]),
            "solutions": sols,
            "n_tokens": [s["n_tokens"] for s in scores],
            "score_mean": [s["mean"] for s in scores],
            "score_sum": [s["sum"] for s in scores],
            "score_first": [s["first"] for s in scores],
            "pred_mean": pred_mean,
            "pred_sum": pred_sum,
            "pred_first": pred_first,
            "pred_official_saved": official_matches[0] if official_matches else None,
            "official_saved_matches_mean": (official_matches and official_matches[0] == pred_mean),
        }
        records.append(rec)
    summary = {k: summarize(records, k) for k in ["mean", "sum", "first"]}
    summary["official_saved_mean_mismatches"] = sum(1 for r in records if not r["official_saved_matches_mean"])
    summary["mean_vs_sum_changed"] = sum(1 for r in records if r["pred_mean"] != r["pred_sum"])
    summary["mean_correct_sum_wrong"] = sum(1 for r in records if r["pred_mean"] == r["label"] and r["pred_sum"] != r["label"])
    summary["sum_correct_mean_wrong"] = sum(1 for r in records if r["pred_sum"] == r["label"] and r["pred_mean"] != r["label"])
    summary["examples_mean_wrong_sum_correct"] = [r for r in records if r["pred_sum"] == r["label"] and r["pred_mean"] != r["label"]][:12]
    summary["examples_mean_correct_sum_wrong"] = [r for r in records if r["pred_mean"] == r["label"] and r["pred_sum"] != r["label"]][:12]
    return {"summary": summary, "records_first20": records[:20], "records": records}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL / "chck_100M").to(DEVICE)
    model.eval()
    payload = {split: process_split(split, tokenizer, model) for split in ["global_piqa_parallel", "global_piqa_nonparallel"]}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Records are small enough; keep full per-item scores for later paired analysis.
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — DeBERTa GlobalPIQA scoring variants", "", f"Evidence JSON: `{OUT_JSON}`", "", "| split | official mean acc | raw-sum acc | first-token acc | mean-vs-sum changed | sum-correct/mean-wrong | mean-correct/sum-wrong |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, d in payload.items():
        s = d["summary"]
        lines.append(f"| {split} | {s['mean']['accuracy']:.2f} | {s['sum']['accuracy']:.2f} | {s['first']['accuracy']:.2f} | {s['mean_vs_sum_changed']} | {s['sum_correct_mean_wrong']} | {s['mean_correct_sum_wrong']} |")
    lines += ["", "Official GlobalPIQA uses length-normalized completion log-probability; raw-sum and first-token are probes only, not leaderboard scores."]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "GPIQA_SCORING_VARIANTS_DONE", "out": str(OUT_JSON), "summary": {k: v["summary"] for k, v in payload.items()}}, indent=2))

if __name__ == "__main__":
    main()
