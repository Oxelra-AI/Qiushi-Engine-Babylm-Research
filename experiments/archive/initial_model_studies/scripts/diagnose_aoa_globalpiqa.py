#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import curve_fit
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
STRICT = ROOT / "repos/babylm-eval/strict"
sys.path.insert(0, str(STRICT))
from evaluation_pipeline.utils import AoAEvaluator, sigmoid_function  # noqa: E402

MODEL = ROOT / "training/runs/babylm_fullcycle_wwm_seed42_100M/hf_model"
AOA_SURPRISAL = MODEL.parent / "eval_results_aoa/hf_model/main/zero_shot/mlm/AoA_word/surprisal.json"
CDI = STRICT / "evaluation_data/full_eval/aoa/cdi_human.csv"
GPIQA_ROOT = MODEL.parent / "eval_results_available_official/hf_model/chck_100M/zero_shot/mlm"
GPIQA_DATA = STRICT / "evaluation_data/full_eval"
OUT_JSON = ROOT / "data/aoa_globalpiqa_diagnostics.json"
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/aoa_globalpiqa_diagnostics.md')


def read_jsonl(path: pathlib.Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def diagnose_aoa():
    data = json.loads(AOA_SURPRISAL.read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    evaluator = AoAEvaluator(CDI)
    vocab_size = tok.vocab_size
    prefix_ids = tok("The", add_special_tokens=False)["input_ids"]

    def subword_len(word: str) -> int:
        ids = tok("The " + word, add_special_tokens=False)["input_ids"]
        return max(1, len(ids) - len(prefix_ids))

    word_data = defaultdict(lambda: {"steps": [], "surprisals": []})
    for r in data["results"]:
        step = evaluator.extract_step_number(r["step"])
        if step is None:
            continue
        word_data[r["target_word"]]["steps"].append(step)
        word_data[r["target_word"]]["surprisals"].append(float(r["surprisal"]))

    cdi_words = set(evaluator.cdi_data["word"].astype(str).tolist())
    counts = Counter()
    examples = defaultdict(list)
    model_none_words = []
    success_records = []

    for word, wd in sorted(word_data.items()):
        if word not in cdi_words:
            counts["no_exact_cdi_word"] += 1
            if len(examples["no_exact_cdi_word"]) < 12:
                examples["no_exact_cdi_word"].append(word)
            continue
        word_idx = np.where(evaluator.cdi_data["word"] == word)[0][0]
        child_aoa = evaluator.compute_child_aoa(word_idx)
        if child_aoa is None:
            counts["no_child_aoa"] += 1
            if len(examples["no_child_aoa"]) < 12:
                examples["no_child_aoa"].append(word)
            continue
        steps_arr = np.array(wd["steps"])
        surp_arr = np.array(wd["surprisals"])
        uniq_steps = np.unique(steps_arr)
        if len(uniq_steps) < 3:
            counts["fewer_than_3_steps"] += 1
            continue
        mean_surps = np.array([float(surp_arr[steps_arr == s].mean()) for s in uniq_steps])
        # First call official function.
        model_aoa = evaluator.compute_model_aoa(mean_surps.tolist(), uniq_steps.tolist(), vocab_size, n_subword_tokens=subword_len(word))
        if model_aoa is not None:
            counts["valid"] += 1
            success_records.append({"word": word, "model_aoa": float(model_aoa), "child_aoa": float(child_aoa), "first_surprisal": float(mean_surps[0]), "last_surprisal": float(mean_surps[-1])})
            continue
        # Reason approximation mirroring compute_model_aoa.
        valid_mask = ~np.isnan(mean_surps)
        if not np.any(valid_mask):
            reason = "all_nan_surprisal"
        else:
            valid_steps = uniq_steps[valid_mask]
            valid_surps = mean_surps[valid_mask]
            random_chance_surprisal = subword_len(word) * np.log(vocab_size)
            min_surprisal = np.min(valid_surps)
            threshold = random_chance_surprisal - 0.5 * (random_chance_surprisal - min_surprisal)
            try:
                neg = -valid_surps
                log_steps = np.log10(valid_steps + 1)
                rng = np.max(neg) - np.min(neg)
                p0 = [rng, 1.0, np.mean(log_steps), np.min(neg)]
                lower = [0.0, 0.0, np.min(log_steps) - 1, np.min(neg) - 2 * rng - 1]
                upper = [10 * rng + 1, 100.0, np.max(log_steps) + 1, np.max(neg) + 1]
                popt, _ = curve_fit(sigmoid_function, log_steps, neg, p0=p0, bounds=(lower, upper), maxfev=20000)
                a, b, c, d = popt
                neg_threshold = -threshold
                if b <= 1e-6 or a <= 1e-6:
                    reason = "flat_or_zero_fit_amplitude"
                elif neg_threshold <= d or neg_threshold >= a + d:
                    reason = "threshold_outside_fit_range"
                else:
                    log_aoa_step = c - np.log((a / (neg_threshold - d)) - 1) / b
                    aoa_step = 10**log_aoa_step - 1
                    if aoa_step < valid_steps[0] or aoa_step > valid_steps[-1]:
                        reason = "aoa_step_outside_training_range"
                    else:
                        reason = "unknown_official_none"
            except Exception as e:
                reason = "curve_fit_exception:" + e.__class__.__name__
        counts[reason] += 1
        model_none_words.append({"word": word, "reason": reason, "n_steps": int(len(uniq_steps)), "subword_len": subword_len(word), "first_surprisal": float(mean_surps[0]), "last_surprisal": float(mean_surps[-1]), "min_surprisal": float(np.min(mean_surps)), "max_surprisal": float(np.max(mean_surps))})
        if len(examples[reason]) < 12:
            examples[reason].append(word)

    return {
        "surprisal_path": str(AOA_SURPRISAL),
        "cdi_path": str(CDI),
        "metadata": data.get("metadata", {}),
        "num_result_rows": len(data["results"]),
        "unique_target_words": len(word_data),
        "num_cdi_words": len(cdi_words),
        "counts": dict(counts),
        "examples": dict(examples),
        "valid_records_first20": success_records[:20],
        "model_none_words_first40": model_none_words[:40],
    }


def load_gpiqa(split: str):
    data_path = GPIQA_DATA / split / "eng_latn.jsonl"
    pred_path = GPIQA_ROOT / split / split / "predictions.json"
    rows = read_jsonl(data_path)
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    joined = []
    for r in rows:
        uid = r["example_id"]
        pred_text = preds[uid]["predictions"][0]["pred"]
        stripped = pred_text.strip()
        sol_keys = sorted([k for k in r.keys() if k.startswith("solution") and k[len("solution"):].isdigit()], key=lambda x: int(x[len("solution"):]))
        sols = [r[k] for k in sol_keys]
        matches = [i for i, s in enumerate(sols) if s.strip() == stripped]
        pred_idx = matches[0] if matches else None
        label = int(r["label"])
        if label < 0 or label >= len(sols):
            raise RuntimeError(f"label {label} outside {len(sols)} solutions for {uid}")
        joined.append({
            "uid": uid,
            "prompt": r["prompt"],
            "category": r.get("categories", ""),
            "num_solutions": len(sols),
            "label": label,
            "pred_idx": pred_idx,
            "pred_text": stripped,
            "correct": pred_idx == label,
            "solution_lengths_words": [len(s.split()) for s in sols],
            "label_length_words": len(sols[label].split()),
            "pred_length_words": len(sols[pred_idx].split()) if pred_idx is not None else len(stripped.split()),
            "solutions": sols,
        })
    return joined


def summarize_gpiqa_rows(rows):
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    by_label = defaultdict(lambda: [0, 0])
    by_pred = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    pred_missing = sum(1 for r in rows if r["pred_idx"] is None)
    for r in rows:
        by_label[r["label"]][0] += int(r["correct"]); by_label[r["label"]][1] += 1
        by_pred[r["pred_idx"]][0] += int(r["correct"]); by_pred[r["pred_idx"]][1] += 1
        for cat in str(r["category"]).split(","):
            c = cat.strip()
            if c:
                by_cat[c][0] += int(r["correct"]); by_cat[c][1] += 1
    # Length tendency: chosen minus correct length; negative means the model tends to choose shorter than correct.
    length_delta = [r["pred_length_words"] - r["label_length_words"] for r in rows if r["pred_idx"] is not None]
    correct_length_delta = [r["pred_length_words"] - r["label_length_words"] for r in rows if r["pred_idx"] is not None and r["correct"]]
    wrong_length_delta = [r["pred_length_words"] - r["label_length_words"] for r in rows if r["pred_idx"] is not None and not r["correct"]]
    confusion = Counter((r["label"], r["pred_idx"]) for r in rows)
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total * 100,
        "pred_missing": pred_missing,
        "label_distribution": dict(Counter(r["label"] for r in rows)),
        "pred_distribution": {str(k): v for k, v in Counter(r["pred_idx"] for r in rows).items()},
        "accuracy_by_label": {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_label.items())},
        "accuracy_by_pred": {str(k): {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_pred.items(), key=lambda kv: str(kv[0]))},
        "accuracy_by_category": {k: {"correct": v[0], "n": v[1], "accuracy": v[0]/v[1]*100} for k, v in sorted(by_cat.items(), key=lambda kv: (-kv[1][1], kv[0]))},
        "confusion_label_pred": {f"{a}->{b}": n for (a,b), n in confusion.items()},
        "length_delta_pred_minus_label_mean": mean(length_delta),
        "length_delta_correct_mean": mean(correct_length_delta),
        "length_delta_wrong_mean": mean(wrong_length_delta),
        "examples_wrong_first20": [r for r in rows if not r["correct"]][:20],
        "examples_correct_first10": [r for r in rows if r["correct"]][:10],
    }


def diagnose_gpiqa():
    out = {}
    for split in ["global_piqa_parallel", "global_piqa_nonparallel"]:
        rows = load_gpiqa(split)
        out[split] = summarize_gpiqa_rows(rows)
    return out


def main():
    payload = {"aoa": diagnose_aoa(), "global_piqa": diagnose_gpiqa()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aoa_counts = payload["aoa"]["counts"]
    gp = payload["global_piqa"]
    lines = [
        "# research — AoA and GlobalPIQA diagnostics", "",
        f"Evidence JSON: `{OUT_JSON}`", "",
        "## AoA valid-word pipeline", "",
        f"Surprisal rows: {payload['aoa']['num_result_rows']}; unique target words: {payload['aoa']['unique_target_words']}; CDI words: {payload['aoa']['num_cdi_words']}.", "",
        "| drop/success category | count | examples |", "|---|---:|---|",
    ]
    for k, v in sorted(aoa_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        ex = ", ".join(payload["aoa"].get("examples", {}).get(k, [])[:8])
        lines.append(f"| {k} | {v} | {ex} |")
    lines += ["", "## GlobalPIQA", "", "| split | accuracy | correct/n | pred distribution | mean chosen-minus-correct answer length |", "|---|---:|---:|---|---:|"]
    for split, s in gp.items():
        lines.append(f"| {split} | {s['accuracy']:.2f} | {s['correct']}/{s['n']} | {s['pred_distribution']} | {s['length_delta_pred_minus_label_mean']:.2f} |")
    lines += ["", "Detailed by-label/by-category/error examples are in the JSON."]
    OUT_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "DIAGNOSTICS_DONE",
        "out": str(OUT_JSON),
        "aoa_counts": aoa_counts,
        "global_piqa": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "n": v["n"], "pred_distribution": v["pred_distribution"], "length_delta_mean": v["length_delta_pred_minus_label_mean"]} for k, v in gp.items()},
    }, indent=2))

if __name__ == "__main__":
    main()
