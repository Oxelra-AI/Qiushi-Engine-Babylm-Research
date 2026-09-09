#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
MODEL_RUN = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256"
GPIQA_ROOT = MODEL_RUN / "eval_results_available/hf_model/chck_100M/zero_shot/mlm"
GPIQA_DATA = STRICT / "evaluation_data/full_eval"
OUT_JSON = ROOT / "data/debertav2_b256_globalpiqa_diagnostics.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_b256_globalpiqa_diagnostics.md')


def read_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def load_split(split: str):
    data_path = GPIQA_DATA / split / "eng_latn.jsonl"
    pred_path = GPIQA_ROOT / split / split / "predictions.json"
    rows = read_jsonl(data_path)
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    joined = []
    for r in rows:
        uid = r["example_id"]
        pred_text = preds[uid]["predictions"][0]["pred"].strip()
        sol_keys = sorted([k for k in r.keys() if k.startswith("solution") and k[len("solution"):].isdigit()], key=lambda x: int(x[len("solution"):]))
        sols = [r[k] for k in sol_keys]
        matches = [i for i, s in enumerate(sols) if s.strip() == pred_text]
        pred_idx = matches[0] if matches else None
        label = int(r["label"])
        joined.append({
            "uid": uid,
            "prompt": r["prompt"],
            "category": r.get("categories", ""),
            "num_solutions": len(sols),
            "label": label,
            "pred_idx": pred_idx,
            "pred_text": pred_text,
            "correct": pred_idx == label,
            "label_length_words": len(sols[label].split()),
            "pred_length_words": len(sols[pred_idx].split()) if pred_idx is not None else len(pred_text.split()),
            "solutions": sols,
        })
    return joined


def summarize(rows):
    by_label = defaultdict(lambda: [0, 0])
    by_pred = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_label[r["label"]][0] += int(r["correct"]); by_label[r["label"]][1] += 1
        by_pred[r["pred_idx"]][0] += int(r["correct"]); by_pred[r["pred_idx"]][1] += 1
        for cat in str(r["category"]).split(","):
            c = cat.strip()
            if c:
                by_cat[c][0] += int(r["correct"]); by_cat[c][1] += 1
    total = len(rows); correct = sum(r["correct"] for r in rows)
    length_delta = [r["pred_length_words"] - r["label_length_words"] for r in rows if r["pred_idx"] is not None]
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total * 100.0,
        "pred_missing": sum(1 for r in rows if r["pred_idx"] is None),
        "label_distribution": dict(Counter(r["label"] for r in rows)),
        "pred_distribution": {str(k): v for k, v in Counter(r["pred_idx"] for r in rows).items()},
        "accuracy_by_label": {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_label.items())},
        "accuracy_by_pred": {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_pred.items(), key=lambda kv: str(kv[0]))},
        "accuracy_by_category": {k: {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_cat.items(), key=lambda kv: (-kv[1][1], kv[0]))},
        "length_delta_pred_minus_label_mean": mean(length_delta),
        "examples_wrong_first20": [r for r in rows if not r["correct"]][:20],
        "examples_correct_first10": [r for r in rows if r["correct"]][:10],
    }


def main():
    payload = {}
    for split in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        payload[split] = summarize(load_split(split))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — DeBERTa-v2 b256 GlobalPIQA diagnostics", "", f"Evidence JSON: `{OUT_JSON}`", "", "| split | accuracy | correct/n | pred distribution | length delta |", "|---|---:|---:|---|---:|"]
    for split, s in payload.items():
        lines.append(f"| {split} | {s['accuracy']:.2f} | {s['correct']}/{s['n']} | {s['pred_distribution']} | {s['length_delta_pred_minus_label_mean']:.2f} |")
    lines += ["", "## Parallel categories", "", "| category | accuracy | correct/n |", "|---|---:|---:|"]
    for cat, v in payload["global_piqa_parallel"]["accuracy_by_category"].items():
        lines.append(f"| {cat} | {v['accuracy']:.2f} | {v['correct']}/{v['n']} |")
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status":"DEBERTA_GPIQA_DIAG_DONE","out":str(OUT_JSON),"parallel_acc":payload['global_piqa_parallel']['accuracy'],"nonparallel_acc":payload['global_piqa_nonparallel']['accuracy'],"parallel_categories":payload['global_piqa_parallel']['accuracy_by_category']}, indent=2))

if __name__ == "__main__":
    main()
