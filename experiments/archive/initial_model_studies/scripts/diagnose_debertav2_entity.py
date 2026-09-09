#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
DATA_DIR = ROOT / "repos/babylm-eval/strict/evaluation_data/full_eval/entity_tracking"
PRED = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_available/hf_model/chck_100M/zero_shot/mlm/entity_tracking/entity_tracking/predictions.json"
TOK16 = ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model"
TOK40 = ROOT / "training/tokenizers/official40k"
OUT = ROOT / "data/debertav2_entity_diagnostics.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/debertav2_entity_diagnostics.md')

SPLITS = ["regular", "ambiref", "move_contents"]


def read_jsonl(p: pathlib.Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def option_token_len(tok, prefix: str, option: str) -> int:
    base = tok(prefix, add_special_tokens=True)["input_ids"]
    full = tok(prefix + option, add_special_tokens=True)["input_ids"]
    # Because special-token behavior can make exact subtraction imperfect, use completion alone as robust proxy.
    comp = tok(option, add_special_tokens=False)["input_ids"]
    return len(comp) if comp else max(0, len(full) - len(base))


def summarize_records(records):
    def init(): return {"correct": 0, "n": 0}
    by_split = defaultdict(init); by_subset = defaultdict(init); by_numops = defaultdict(init)
    pred_idx_dist = Counter(); missing = 0
    wrong_kind = Counter()
    len_delta_correct_chosen_16 = []
    len_delta_correct_chosen_40 = []
    for r in records:
        ok = r["correct"]
        for d, key in [(by_split, r["split"]), (by_subset, r["subset"]), (by_numops, str(r["numops"]))]:
            d[key]["correct"] += int(ok); d[key]["n"] += 1
        pred_idx_dist[str(r["pred_idx"])] += 1
        missing += int(r["pred_idx"] is None)
        if not ok:
            if r["pred_idx"] is None:
                wrong_kind["prediction_not_matching_any_option"] += 1
            elif r["pred_idx"] == 1:
                wrong_kind["wrong_option_1_other_box_or_first_distractor"] += 1
            else:
                wrong_kind["wrong_option_2plus_corrupted_or_other"] += 1
        if r["pred_idx"] is not None:
            len_delta_correct_chosen_16.append(r["option_token_lengths_16k"][r["pred_idx"]] - r["option_token_lengths_16k"][0])
            len_delta_correct_chosen_40.append(r["option_token_lengths_40k"][r["pred_idx"]] - r["option_token_lengths_40k"][0])
    def finalize(d):
        return {k: {"correct": v["correct"], "n": v["n"], "accuracy": 100.0 * v["correct"] / v["n"] if v["n"] else None} for k, v in sorted(d.items())}
    return {
        "n": len(records),
        "correct": sum(r["correct"] for r in records),
        "accuracy": 100.0 * sum(r["correct"] for r in records) / len(records),
        "by_split": finalize(by_split),
        "by_subset": finalize(by_subset),
        "by_numops": finalize(by_numops),
        "pred_idx_distribution": dict(pred_idx_dist),
        "missing_option_matches": missing,
        "wrong_kind": dict(wrong_kind),
        "mean_chosen_minus_correct_tokens_16k": sum(len_delta_correct_chosen_16)/len(len_delta_correct_chosen_16),
        "mean_chosen_minus_correct_tokens_40k": sum(len_delta_correct_chosen_40)/len(len_delta_correct_chosen_40),
    }


def main():
    preds = json.loads(PRED.read_text(encoding="utf-8"))
    tok16 = AutoTokenizer.from_pretrained(TOK16)
    tok40 = AutoTokenizer.from_pretrained(TOK40)
    subset_cursors = defaultdict(int)
    records = []
    skipped_nothing = 0
    option_len_all16 = []
    option_len_all40 = []
    correct_len16 = []
    correct_len40 = []
    for split in SPLITS:
        for raw in read_jsonl(DATA_DIR / f"{split}.jsonl"):
            if any("nothing" in opt for opt in raw["options"]):
                skipped_nothing += 1
                continue
            subset = f"{split}_{raw['numops']}_ops"
            idx = subset_cursors[subset]
            subset_cursors[subset] += 1
            pred_list = preds[subset]["predictions"]
            if idx >= len(pred_list):
                raise RuntimeError(f"prediction index overflow for {subset}: {idx} >= {len(pred_list)}")
            pred_text = pred_list[idx]["pred"]
            opts = raw["options"]
            matches = [i for i, opt in enumerate(opts) if norm(opt) == norm(pred_text)]
            pred_idx = matches[0] if matches else None
            lens16 = [option_token_len(tok16, raw["input_prefix"], opt) for opt in opts]
            lens40 = [option_token_len(tok40, raw["input_prefix"], opt) for opt in opts]
            option_len_all16.extend(lens16); option_len_all40.extend(lens40)
            correct_len16.append(lens16[0]); correct_len40.append(lens40[0])
            records.append({
                "split": split,
                "subset": subset,
                "numops": int(raw["numops"]),
                "example_id": raw["example_id"],
                "sample_id": raw.get("sample_id"),
                "pred_idx": pred_idx,
                "pred_text": pred_text,
                "correct": pred_idx == 0,
                "options": opts,
                "option_token_lengths_16k": lens16,
                "option_token_lengths_40k": lens40,
                "correct_option_tokens_16k": lens16[0],
                "correct_option_tokens_40k": lens40[0],
                "input_prefix_len_words": len(raw["input_prefix"].split()),
                "num_options": len(opts),
            })
    # Check all prediction lists consumed.
    pred_consumption = {k: {"used": subset_cursors[k], "available": len(v["predictions"])} for k, v in preds.items()}
    for k, v in pred_consumption.items():
        if v["used"] != v["available"]:
            raise RuntimeError(f"not all predictions consumed for {k}: {v}")
    summary = summarize_records(records)
    token_summary = {
        "all_option_mean_tokens_16k": sum(option_len_all16)/len(option_len_all16),
        "all_option_mean_tokens_40k": sum(option_len_all40)/len(option_len_all40),
        "correct_option_mean_tokens_16k": sum(correct_len16)/len(correct_len16),
        "correct_option_mean_tokens_40k": sum(correct_len40)/len(correct_len40),
        "all_option_token_delta_40k_minus_16k": sum(option_len_all40)/len(option_len_all40) - sum(option_len_all16)/len(option_len_all16),
        "correct_option_token_delta_40k_minus_16k": sum(correct_len40)/len(correct_len40) - sum(correct_len16)/len(correct_len16),
        "options_shorter_under_40k": sum(1 for a, b in zip(option_len_all16, option_len_all40) if b < a),
        "options_equal_under_40k": sum(1 for a, b in zip(option_len_all16, option_len_all40) if b == a),
        "options_longer_under_40k": sum(1 for a, b in zip(option_len_all16, option_len_all40) if b > a),
        "num_option_strings": len(option_len_all16),
    }
    wrong_examples = [r for r in records if not r["correct"]][:40]
    correct_examples = [r for r in records if r["correct"]][:20]
    payload = {
        "prediction_path": str(PRED),
        "data_dir": str(DATA_DIR),
        "tokenizers": {"baseline16k": str(TOK16), "official40k": str(TOK40)},
        "skipped_examples_with_nothing_option": skipped_nothing,
        "prediction_consumption": pred_consumption,
        "summary": summary,
        "token_summary": token_summary,
        "wrong_examples_first40": wrong_examples,
        "correct_examples_first20": correct_examples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — DeBERTa-v2 b256 Entity diagnostics", "", f"Evidence JSON: `{OUT}`", "",
        f"Overall exact-option accuracy: {summary['accuracy']:.2f} ({summary['correct']}/{summary['n']}).", "",
        "## By split", "", "| split | accuracy | correct/n |", "|---|---:|---:|",
    ]
    for k, v in summary["by_split"].items():
        lines.append(f"| {k} | {v['accuracy']:.2f} | {v['correct']}/{v['n']} |")
    lines += ["", "## By numops", "", "| numops | accuracy | correct/n |", "|---|---:|---:|"]
    for k, v in summary["by_numops"].items():
        lines.append(f"| {k} | {v['accuracy']:.2f} | {v['correct']}/{v['n']} |")
    lines += [
        "", "## Tokenizer effect on Entity options", "", "| quantity | baseline16k | official40k | delta |", "|---|---:|---:|---:|",
        f"| all option mean tokens | {token_summary['all_option_mean_tokens_16k']:.3f} | {token_summary['all_option_mean_tokens_40k']:.3f} | {token_summary['all_option_token_delta_40k_minus_16k']:.3f} |",
        f"| correct option mean tokens | {token_summary['correct_option_mean_tokens_16k']:.3f} | {token_summary['correct_option_mean_tokens_40k']:.3f} | {token_summary['correct_option_token_delta_40k_minus_16k']:.3f} |",
        f"| option strings shorter/equal/longer under 40k | {token_summary['options_shorter_under_40k']} | {token_summary['options_equal_under_40k']} | {token_summary['options_longer_under_40k']} |",
        "", "Subset details and examples are in the JSON.",
    ]
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ENTITY_DIAGNOSTICS_DONE", "out": str(OUT), "accuracy": summary["accuracy"], "by_split": summary["by_split"], "token_summary": token_summary}, indent=2))

if __name__ == "__main__":
    main()
