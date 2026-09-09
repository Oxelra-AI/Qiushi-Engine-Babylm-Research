#!/usr/bin/env python3
"""research: audit official causal-eval token lengths under the neutral tokenizer.

The research trainer used 256-token training chunks, but GPT-style models need
`n_positions` at least as large as the official evaluation candidate sequences.
This script measures the actual candidate sentence lengths used by the official
zero-shot readers plus reading word contexts, before any expensive causal runs.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Any

USER_ROOT = pathlib.Path(".").resolve()
EVAL_PIPE = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_pipeline"
CURRENT_FULL = USER_ROOT / "experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval"
LEGACY_FULL = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval"
if str(EVAL_PIPE.parent.resolve()) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPE.parent.resolve()))

from evaluation_pipeline.sentence_zero_shot.read_files import read_files  # noqa: E402

TASKS = [
    {"column": "BLiMP", "task": "blimp", "data_path": CURRENT_FULL / "blimp_filtered"},
    {"column": "Supplement", "task": "blimp", "data_path": CURRENT_FULL / "supplement_filtered"},
    {"column": "EWoK", "task": "ewok", "data_path": CURRENT_FULL / "ewok_filtered"},
    {"column": "Entity", "task": "entity_tracking", "data_path": CURRENT_FULL / "entity_tracking"},
    {"column": "COMPS", "task": "comps", "data_path": CURRENT_FULL / "comps"},
    {"column": "GlobalPIQA_parallel", "task": "global_piqa_parallel", "data_path": LEGACY_FULL / "global_piqa_parallel"},
    {"column": "GlobalPIQA_nonparallel", "task": "global_piqa_nonparallel", "data_path": LEGACY_FULL / "global_piqa_nonparallel"},
]

class Args:
    def __init__(self, task: str, data_path: pathlib.Path):
        self.task = task
        self.data_path = data_path
        self.full_sentence_scores = False
        self.images_path = None
        self.image_split = None


def q(xs: list[int], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = p * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1 - frac) + ys[hi] * frac


def summarize(xs: list[int]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "min": min(xs) if xs else None,
        "max": max(xs) if xs else None,
        "mean": statistics.fmean(xs) if xs else None,
        "p50": q(xs, 0.5),
        "p90": q(xs, 0.9),
        "p95": q(xs, 0.95),
        "p99": q(xs, 0.99),
        "n_gt_256": sum(1 for x in xs if x > 256),
        "n_gt_384": sum(1 for x in xs if x > 384),
        "n_gt_512": sum(1 for x in xs if x > 512),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer")
    ap.add_argument("--out-dir", default="experiments/archive/frontier_consolidation/data/causal_eval_length_audit")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    results: dict[str, Any] = {}
    all_lengths: list[int] = []
    examples_gt256: list[dict[str, Any]] = []
    for spec in TASKS:
        decoded = read_files(Args(spec["task"], spec["data_path"]))
        lengths: list[int] = []
        phrase_lengths: list[int] = []
        for item_i, item in enumerate(decoded):
            for cand_i, (sent, comp) in enumerate(zip(item["sentences"], item["completions"])):
                enc = tok(sent, return_offsets_mapping=True)
                ids = enc["input_ids"]
                lengths.append(len(ids))
                all_lengths.append(len(ids))
                # mimic official phrase mask for causal backend
                offset_mapping = enc["offset_mapping"]
                sent_tokens = tok(sent, return_offsets_mapping=True)["input_ids"]
                start_idx = len(ids) - len(sent_tokens)
                start_char_idx = len(sent) - len(comp) + offset_mapping[start_idx][0]
                phrase_len = sum(1 for (start, end) in offset_mapping[start_idx:] if end > start_char_idx)
                phrase_lengths.append(phrase_len)
                if len(ids) > 256 and len(examples_gt256) < 50:
                    examples_gt256.append({
                        "column": spec["column"],
                        "item_index": item_i,
                        "candidate_index": cand_i,
                        "length": len(ids),
                        "phrase_len": phrase_len,
                        "uid": item.get("UID"),
                        "sentence_preview": sent[:240],
                        "completion_preview": str(comp)[:120],
                    })
        results[spec["column"]] = {"sentence_lengths": summarize(lengths), "phrase_lengths": summarize(phrase_lengths)}

    # Reading contexts are evaluated by reading/evaluation_functions.py rather than read_files.
    # Measure the item and previous-item strings as they are passed to get_p2.
    import pandas as pd
    reading_path = CURRENT_FULL / "reading/reading_data.csv"
    df = pd.read_csv(reading_path, dtype={"item": str})
    rd_lengths: list[int] = []
    for _, row in df.iterrows():
        for text_key, word_key in [("item", "word"), ("prev_item", "prev_word")]:
            text = row.get(text_key)
            word = row.get(word_key)
            if isinstance(text, str) and isinstance(word, str):
                # get_p2 scores word after context.  A conservative max is context + word.
                length = len(tok(str(text) + " " + str(word))["input_ids"])
                rd_lengths.append(length)
                all_lengths.append(length)
    results["Reading_context_plus_word"] = {"sentence_lengths": summarize(rd_lengths)}

    summary = {
        "status": "CAUSAL_EVAL_LENGTH_AUDIT_DONE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t0, 1),
        "tokenizer": args.tokenizer,
        "tasks": results,
        "all_candidate_lengths": summarize(all_lengths),
        "examples_gt256": examples_gt256,
        "interpretation": "If any official candidate exceeds 256 tokens, a GPT model with n_positions=256 is not official-eval compatible; use larger n_positions while keeping training chunks fixed if needed.",
    }
    json_path = out_dir / "causal_eval_length_audit.json"
    md_path = out_dir / "causal_eval_length_audit.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# research causal official-eval length audit",
        "",
        f"Tokenizer: `{args.tokenizer}`",
        "",
        "| column | n | max | p99 | >256 | >384 | >512 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for col, rec in results.items():
        s = rec["sentence_lengths"]
        lines.append(f"| {col} | {s['n']} | {s['max']} | {s['p99']:.1f} | {s['n_gt_256']} | {s['n_gt_384']} | {s['n_gt_512']} |")
    s = summary["all_candidate_lengths"]
    lines.extend(["", f"Overall max={s['max']}, >256={s['n_gt_256']}, >384={s['n_gt_384']}, >512={s['n_gt_512']}.", "", f"JSON: `{json_path}`"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "md": str(md_path), "all": summary["all_candidate_lengths"]}, indent=2))


if __name__ == "__main__":
    main()
