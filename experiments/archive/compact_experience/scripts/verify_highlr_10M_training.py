#!/usr/bin/env python3
"""research: verify research high-LR 10M optimizer×data training outputs.

This is a cheap pre-evaluation check for the four 10M runs:
AdamW/LAMB × official/Qwen at LR=7e-3. It reads scientific_metrics.json and
checkpoint directories, ensuring the runs are comparable before launching the
expensive no-AoA trajectory evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import sys
from pathlib import Path
from typing import Any

WORKSPACE = _public_path('experiments/archive/compact_experience')
RUN_ROOT = _public_path('experiments/archive/compact_experience/training/runs')
OUT_ROOT = _public_path('experiments/archive/compact_experience/data/optimizer_10M_training_verify')

ARMS = [
    ("adamw", "official", "adamw_lr0.007_official_10M_seed43022"),
    ("lamb", "official", "lamb_lr0.007_official_10M_seed43022"),
    ("adamw", "qwen", "adamw_lr0.007_qwen_10M_seed43022"),
    ("lamb", "qwen", "lamb_lr0.007_qwen_10M_seed43022"),
]
EXPECTED_CKPTS = [f"chck_{i}M" for i in range(1, 11)]
EXPECTED = {
    "learning_rate": 0.007,
    "beta1": 0.9,
    "beta2": 0.98,
    "eps": 1e-8,
    "weight_decay": 0.01,
    "batch_size": 128,
    "seq_length": 256,
    "mask_prob": 0.15,
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "param_count": 34467424,
    "vocab_size": 16384,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
    "selected_words": 10000000,
}


def approx(a: Any, b: float, tol: float = 1e-12) -> bool:
    return isinstance(a, (int, float)) and math.isfinite(float(a)) and abs(float(a) - b) <= tol


def check_arm(opt: str, corpus: str, name: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / name
    metrics_path = run_dir / "scientific_metrics.json"
    rec: dict[str, Any] = {"arm": name, "optimizer_expected": opt, "corpus": corpus, "run_dir": str(run_dir), "errors": [], "warnings": []}
    if not run_dir.exists():
        rec["errors"].append("missing_run_dir")
        return rec
    if not metrics_path.exists():
        rec["errors"].append("missing_scientific_metrics_json")
        return rec

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    rec["metrics"] = {k: m.get(k) for k in [
        "variant", "optimizer", "learning_rate", "beta1", "beta2", "eps", "weight_decay",
        "batch_size", "seq_length", "mask_prob", "hidden_size", "n_layer", "n_head",
        "param_count", "vocab_size", "selected_words", "total_steps", "warmup_steps",
        "cumulative_words", "loss_last", "n_pair_rows", "pair_words", "extra_init_seed",
        "train_rng_seed", "jsonl_path", "jsonl_sha256"
    ] if k in m}

    if m.get("optimizer") != opt:
        rec["errors"].append(f"optimizer {m.get('optimizer')} != {opt}")
    for k, v in EXPECTED.items():
        if isinstance(v, float):
            if not approx(m.get(k), v, tol=1e-10):
                rec["errors"].append(f"{k} {m.get(k)} != {v}")
        else:
            if m.get(k) != v:
                rec["errors"].append(f"{k} {m.get(k)} != {v}")

    # The trainer consumes complete rows only; actual cumulative words should be
    # at least the target once checkpoint chck_10M exists, and not wildly over.
    cw = m.get("cumulative_words")
    if not isinstance(cw, int):
        rec["errors"].append(f"cumulative_words_not_int {cw}")
    elif cw < 10_000_000:
        rec["errors"].append(f"cumulative_words_below_10M {cw}")
    elif cw > 10_040_000:
        rec["warnings"].append(f"cumulative_words_more_than_40k_over_target {cw}")

    saved = m.get("saved_checkpoints", [])
    saved_names = [x.get("name") for x in saved if isinstance(x, dict)]
    rec["saved_checkpoint_names"] = saved_names
    missing_ckpts = [c for c in EXPECTED_CKPTS if c not in saved_names]
    if missing_ckpts:
        rec["errors"].append(f"missing_saved_checkpoint_names {missing_ckpts}")

    missing_dirs = [c for c in EXPECTED_CKPTS if not (run_dir / "hf_model" / c).is_dir()]
    if missing_dirs:
        rec["errors"].append(f"missing_checkpoint_dirs {missing_dirs}")

    # Basic corpus sanity.
    jsonl = str(m.get("jsonl_path", ""))
    if corpus == "official" and "official_only_100M.jsonl" not in jsonl:
        rec["errors"].append(f"wrong_official_jsonl {jsonl}")
    if corpus == "qwen" and "qwen_aligned_100M.jsonl" not in jsonl:
        rec["errors"].append(f"wrong_qwen_jsonl {jsonl}")

    if corpus == "official":
        if m.get("n_pair_rows") not in (0, None):
            rec["errors"].append(f"official_has_pair_rows {m.get('n_pair_rows')}")
        if m.get("pair_words") not in (0, None):
            rec["errors"].append(f"official_has_pair_words {m.get('pair_words')}")
    else:
        if not isinstance(m.get("n_pair_rows"), int) or m.get("n_pair_rows") <= 0:
            rec["errors"].append(f"qwen_pair_rows_nonpositive {m.get('n_pair_rows')}")
        if not isinstance(m.get("pair_words"), int) or m.get("pair_words") <= 0:
            rec["errors"].append(f"qwen_pair_words_nonpositive {m.get('pair_words')}")

    rec["status"] = "pass" if not rec["errors"] else "fail"
    return rec


def main() -> None:
    out_root = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    records = [check_arm(opt, corpus, name) for opt, corpus, name in ARMS]

    # Cross-arm comparability checks.
    cross_errors: list[str] = []
    metrics = [r.get("metrics", {}) for r in records if r.get("status") == "pass"]
    if len(metrics) == 4:
        comparable_keys = [
            "learning_rate", "beta1", "beta2", "eps", "weight_decay", "batch_size",
            "seq_length", "mask_prob", "hidden_size", "n_layer", "n_head", "param_count",
            "vocab_size", "extra_init_seed", "train_rng_seed", "selected_words"
        ]
        for k in comparable_keys:
            vals = {m.get(k) for m in metrics}
            if len(vals) != 1:
                cross_errors.append(f"cross_arm_mismatch_{k}: {sorted(map(str, vals))}")
        # Corpus data-hash identity: the two official arms must share one jsonl hash,
        # and the two Qwen arms must share one jsonl hash. This is the load-bearing
        # comparability check for the data factor.
        by_key = {r["arm"]: r for r in records}
        off_hashes = {by_key[n].get("metrics", {}).get("jsonl_sha256") for _, corpus, n in ARMS if corpus == "official"}
        qwn_hashes = {by_key[n].get("metrics", {}).get("jsonl_sha256") for _, corpus, n in ARMS if corpus == "qwen"}
        if len(off_hashes) != 1 or None in off_hashes:
            cross_errors.append(f"official_arms_data_hash_mismatch: {sorted(map(str, off_hashes))}")
        if len(qwn_hashes) != 1 or None in qwn_hashes:
            cross_errors.append(f"qwen_arms_data_hash_mismatch: {sorted(map(str, qwn_hashes))}")
        if off_hashes == qwn_hashes:
            cross_errors.append("official_and_qwen_share_identical_data_hash (data factor collapsed)")
        # Tokenizer label identity across all four arms.
        tok_labels = {m.get("tokenizer_label") for m in metrics}
        if len(tok_labels) != 1:
            cross_errors.append(f"tokenizer_label_mismatch: {sorted(map(str, tok_labels))}")
    else:
        cross_errors.append(f"not_all_arms_pass_individual_checks {sum(1 for r in records if r.get('status') == 'pass')}/4")

    # Checkpoint overshoot comparability: report actual cumulative words at each
    # saved checkpoint per arm so the evaluator does not treat name-equal checkpoints
    # with different actual exposure as identical.
    overshoot = {}
    for opt, corpus, name in ARMS:
        run_dir = RUN_ROOT / name
        metrics_path = run_dir / "scientific_metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text(encoding="utf-8"))
        overshoot[name] = {
            c.get("name"): c.get("cumulative_words")
            for c in m.get("saved_checkpoints", []) if isinstance(c, dict)
        }

    summary = {
        "status": "HIGHLR_10M_TRAINING_VERIFY",
        "overall_pass": all(r.get("status") == "pass" for r in records) and not cross_errors,
        "records": records,
        "cross_errors": cross_errors,
        "checkpoint_cumulative_words_by_arm": overshoot,
        "next_if_pass": "run bash scripts/eval_highlr_10M_interaction.sh for no-AoA 3M/5M/10M trajectory evaluation",
        "next_if_fail": "repair or rerun only the mismatched arm before spending evaluator time",
    }
    out_path = out_root / "highlr_10M_training_verify.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
