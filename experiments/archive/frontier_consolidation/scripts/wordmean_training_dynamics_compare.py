#!/usr/bin/env python3
"""research: CPU-only word-mean vs token-mean training trajectory comparison.

The word-mean 80M screen has finished, but official-compatible cheap-column
evaluation remains pending. This script does not run
or poll evaluation. It reads only training artifacts to verify matched exposure
geometry and summarize training dynamics. The result is not downstream evidence;
it helps interpret the upcoming eval and catches hidden trajectory mismatches.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
TOKEN_RUN = WORKSPACE / "training" / "runs" / "complianttok_reinvest_seed43022_r2"
WORD_RUN = WORKSPACE / "training" / "runs" / "wordmean_mlm_complianttok_reinvest_seed43022_80M"
OUT_DIR = WORKSPACE / "data" / "wordmean_training_dynamics"
NOTE = (USER_ROOT / 'research/notes/frontier_consolidation/wordmean_training_dynamics.md')
MAX_EXPOSURE = 80_000_000


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(USER_ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def nearest(rows: list[dict[str, Any]], exposure: int) -> dict[str, Any]:
    return min(rows, key=lambda r: abs(int(r["cumulative_word_exposure"]) - exposure))


def window_stats(rows: list[dict[str, Any]], lo: int, hi: int) -> dict[str, Any]:
    win = [r for r in rows if lo <= int(r["cumulative_word_exposure"]) <= hi]
    if not win:
        return {"n": 0}
    losses = [float(r["loss"]) for r in win]
    masked = [float(r.get("masked_tokens", 0)) for r in win]
    eff = [float(r.get("effective_mask_rate", 0.0)) for r in win]
    return {
        "n": len(win),
        "loss_mean": statistics.fmean(losses),
        "loss_last": losses[-1],
        "masked_tokens_mean": statistics.fmean(masked),
        "effective_mask_rate_mean": statistics.fmean(eff),
        "first_step": int(win[0]["step"]),
        "last_step": int(win[-1]["step"]),
    }


def summarize_checkpoint_alignment(token_metrics: dict[str, Any], word_metrics: dict[str, Any]) -> dict[str, Any]:
    token_ck = {c["name"]: c for c in token_metrics["saved_checkpoints"] if int(c["target_word_exposure"]) <= MAX_EXPOSURE}
    word_ck = {c["name"]: c for c in word_metrics["saved_checkpoints"]}
    names = sorted(set(token_ck) & set(word_ck), key=lambda x: int(x.split("_")[1][:-1]))
    deltas = []
    for n in names:
        deltas.append({
            "name": n,
            "target": int(word_ck[n]["target_word_exposure"]),
            "token_actual": int(token_ck[n]["actual_cumulative_word_exposure"]),
            "word_actual": int(word_ck[n]["actual_cumulative_word_exposure"]),
            "delta_word_actual_minus_token_actual": int(word_ck[n]["actual_cumulative_word_exposure"]) - int(token_ck[n]["actual_cumulative_word_exposure"]),
        })
    return {
        "common_checkpoint_count": len(names),
        "first_delta": deltas[0] if deltas else None,
        "last_delta": deltas[-1] if deltas else None,
        "max_abs_actual_exposure_delta": max((abs(d["delta_word_actual_minus_token_actual"]) for d in deltas), default=None),
        "all_actual_exposures_identical": all(d["delta_word_actual_minus_token_actual"] == 0 for d in deltas),
    }


def write_note(result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research word-mean training dynamics\n")
    lines.append("CPU-only reading of training artifacts after the 80M word-mean screen finished. No official-compatible evaluation was run or polled here.\n")
    lines.append("\n## Matched run facts\n")
    lines.append(f"- Token-mean reference: `{result['runs']['token_mean']['run_dir']}`.\n")
    lines.append(f"- Word-mean screen: `{result['runs']['word_mean']['run_dir']}`.\n")
    lines.append(f"- Checkpoint exposure alignment identical through 80M: `{result['checkpoint_alignment']['all_actual_exposures_identical']}`; common checkpoints `{result['checkpoint_alignment']['common_checkpoint_count']}`.\n")
    lines.append(f"- Word-mean finished: exposure `{result['runs']['word_mean']['metrics']['word_exposure']}`, steps `{result['runs']['word_mean']['metrics']['actual_training_steps']}`, last loss `{result['runs']['word_mean']['metrics']['loss_last']:.6f}`, mean tokens per selected group trace `{result['runs']['word_mean']['metrics'].get('mean_tokens_per_selected_group_trace_mean'):.6f}`.\n")
    lines.append("\n## Loss windows (not directly comparable as competence because objectives differ)\n")
    lines.append("| window | token loss mean/last | word loss mean/last | token eff mask | word eff mask |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for w, rec in result["windows"].items():
        t = rec["token_mean"]
        wd = rec["word_mean"]
        lines.append(f"| {w} | {t.get('loss_mean', 0):.4f}/{t.get('loss_last', 0):.4f} | {wd.get('loss_mean', 0):.4f}/{wd.get('loss_last', 0):.4f} | {t.get('effective_mask_rate_mean', 0):.4f} | {wd.get('effective_mask_rate_mean', 0):.4f} |\n")
    lines.append("\n## Scientific reading\n")
    lines.append("- The word-mean screen is a real matched 80M trajectory on the same corpus/tokenizer/architecture/seeds/checkpoint exposures, so the pending cheap-column evaluation will be interpretable as an objective-normalization screen.\n")
    lines.append("- The scalar loss cannot decide the route because token-mean and word-mean optimize different reductions. The previous research credit profile remains more informative about mechanism: word-mean shifts relative credit toward one-piece function words and away from multi-piece content.\n")
    lines.append("- Do not continue to 100M or start the minfreq50 fallback from training loss alone; wait for the managed 70M/80M official-compatible columns.\n")
    lines.append(f"\nFull JSON: `{rel(OUT_DIR / 'wordmean_training_dynamics.json')}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    token_metrics = read_json(TOKEN_RUN / "scientific_metrics.json")
    word_metrics = read_json(WORD_RUN / "scientific_metrics.json")
    token_log = [r for r in read_jsonl(TOKEN_RUN / "training_log.jsonl") if int(r["cumulative_word_exposure"]) <= MAX_EXPOSURE]
    word_log = read_jsonl(WORD_RUN / "training_log.jsonl")
    exposures = [20_000_000, 70_000_000, 80_000_000]
    exposure_snapshots = {}
    for e in exposures:
        t = nearest(token_log, e)
        w = nearest(word_log, e)
        exposure_snapshots[str(e)] = {
            "token_mean": {"step": t["step"], "actual_exposure": t["cumulative_word_exposure"], "loss": t["loss"], "masked_tokens": t.get("masked_tokens"), "effective_mask_rate": t.get("effective_mask_rate")},
            "word_mean": {"step": w["step"], "actual_exposure": w["cumulative_word_exposure"], "loss": w["loss"], "masked_tokens": w.get("masked_tokens"), "effective_mask_rate": w.get("effective_mask_rate")},
        }
    windows = {
        "0-5M": (0, 5_000_000),
        "15-20M": (15_000_000, 20_100_000),
        "65-70M": (65_000_000, 70_100_000),
        "75-80M": (75_000_000, 80_100_000),
    }
    window_payload = {name: {"token_mean": window_stats(token_log, *bounds), "word_mean": window_stats(word_log, *bounds)} for name, bounds in windows.items()}
    result = {
        "status": "WORDMEAN_TRAINING_DYNAMICS",
        "purpose": "Verify matched training geometry and summarize dynamics after word-mean training, without running or polling official-compatible evaluation.",
        "runs": {
            "token_mean": {"run_dir": rel(TOKEN_RUN), "metrics": {k: token_metrics.get(k) for k in ["variant", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "parameter_count", "vocab_size", "tokenizer_label", "seed", "seq_length", "max_seq_length", "seq_len_schedule"]}},
            "word_mean": {"run_dir": rel(WORD_RUN), "metrics": {k: word_metrics.get(k) for k in ["variant", "word_exposure", "actual_training_steps", "loss_first", "loss_last", "parameter_count", "vocab_size", "tokenizer_label", "loss_normalization", "mean_tokens_per_selected_group_trace_mean", "seed", "extra_init_seed", "train_rng_seed", "seq_length", "max_seq_length", "seq_len_schedule"]}},
        },
        "checkpoint_alignment": summarize_checkpoint_alignment(token_metrics, word_metrics),
        "exposure_snapshots": exposure_snapshots,
        "windows": window_payload,
        "interpretation": {
            "matched_geometry": "The word-mean screen has identical corpus/tokenizer/architecture/seeds/exposure checkpoint targets through 80M; objective reduction is the intended changed factor.",
            "loss_caution": "Loss values are objective-coordinate scalars and cannot be read as downstream BabyLM competence or as route success.",
            "next_evidence": "Read the completed word-mean cheap-column results; then run wordmean_subtask_interpreter.py before any 100M continuation or fallback launch.",
        },
    }
    out_json = OUT_DIR / "wordmean_training_dynamics.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_note(result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(NOTE), "checkpoint_alignment": result["checkpoint_alignment"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
