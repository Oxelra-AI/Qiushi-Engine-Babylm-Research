#!/usr/bin/env python3
"""research: register-removal late-rate commitment and reference-rate spread.

Scientific role
---------------
Before interpreting MAX-register official scores, measure the clean model's own
late loss trajectory on the two word-matched clean blocks that the register arms
remove.  The register arms admit the identical MAX FineWeb block; they differ only
in whether the sacrificed clean words are developmental/speech rows or
Gutenberg/SimpleWiki adult-prose rows.  If the clean model was still reducing
loss on one removed block at 60M->100M, removing that block should be costlier
under the active-error/rate account.

The script also recomputes the existing repeat/view/breadth 60M->100M changed-
block rates under three independent row/mask resamples.  The previously used
reference rates (repeat 0.148, breadth 0.223, view 0.296) came from one 300-row
sample and one mask seed; this spread is needed before using those values as a
mechanistic scale for the final in-corpus cell.

This is CPU-only forward MLM loss.  It performs no training, official benchmark
evaluation, upload, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import time
from typing import Any, Iterable

# CPU-only before torch/transformers import.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = WS / "data" / "register_removal_rate_and_reference_spread"
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"

BASE_POOL = ROOT / "experiments/archive" / 'compact_experience' / "data" / "qwen_clean_aligned" / "training_corpora" / "qwen_aligned_10M.jsonl"
SELECTION_RECORDS = WS / "data" / "register_max_rowholdout_pools" / "regmax_selection_records.jsonl"
REGISTER_META = WS / "data" / "register_max_rowholdout_pools" / "register_max_rowholdout_metadata.json"

CLEAN_RUN = WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022"
MODEL_RUNS = {
    "view": WS / "training" / "runs" / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "repeat": WS / "training" / "runs" / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "breadth": WS / "training" / "runs" / "full_p2c_c2p_abs_breadth_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "clean": CLEAN_RUN,
}
POOL_FILES = {
    "view": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_10M.jsonl",
    "repeat": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_10M.jsonl",
    "breadth": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_10M.jsonl",
}
CHANGED_META = {
    "view": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
    "breadth": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_changed_block_rows_meta.jsonl",
}

# Historical single-sample anchors from research/287, preserved for comparison.
HISTORICAL_REFERENCE_RATES_60_100 = {
    "repeat": 0.148226,
    "breadth": 0.222660,
    "view": 0.295710,
}
HISTORICAL_LATE_EXENTITY5 = {
    "repeat": 0.0343,
    "breadth": 0.3350,
    "view": 0.3853,
}
REGISTER_PROFILE_PREDICTION = {
    "contrast": "childspeech_removed - adultprose_removed",
    "meaning_positive": "the arm removing child/speech rows scores higher, so removing adult prose was costlier",
    "profile_pred_exEntity5": 0.748,
    "profile_pred_cheap6": 0.817,
    "strict_profile_pred_exEntity5": 0.7697,
    "strict_profile_pred_cheap6": 0.8441,
    "word_js_pred_exEntity5": -0.068,
    "word_js_pred_cheap6": -0.085,
}

MASK_PROB = 0.15
MAX_SEQ_LEN = 256
CHECKPOINTS = ["chck_60M", "chck_80M", "chck_100M"]
REFERENCE_ARMS = ["repeat", "breadth", "view"]
REPLICATE_SEEDS = [29501, 29502, 29503]
MASK_SEEDS = [295101, 295102, 295103]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_source(source: Any) -> str:
    s = str(source or "")
    return s.split("::", 1)[1] if "::" in s else s


def nwords(row: dict[str, Any]) -> int:
    if row.get("words") is not None:
        return int(row.get("words"))
    return len(str(row.get("text", "")).split())


def model_ready(run: pathlib.Path, ck: str) -> bool:
    p = run / "hf_model" / ck
    if not (p / "config.json").exists():
        return False
    return any((p / name).exists() for name in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"])


def load_changed_indices(meta_path: pathlib.Path) -> list[int]:
    idx: list[int] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx.append(int(row["row_index"]))
    return sorted(set(idx))


def select_rows_by_indices(rows: list[dict[str, Any]], indices: Iterable[int], *, label: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in indices:
        r = rows[int(i)]
        text = str(r.get("text", ""))
        if not text.strip():
            continue
        out.append({
            "set": label,
            "row_index": int(i),
            "text": text,
            "words": nwords(r),
            "source": normalize_source(r.get("source", r.get("source_name", "unknown"))),
            "sha12": sha12(text),
        })
    return out


def sample_records(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(records) <= n:
        selected = list(records)
    else:
        selected = rng.sample(records, n)
    return sorted(selected, key=lambda r: (int(r["row_index"]), r["sha12"]))


def source_word_summary(records: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r["source"]] = out.get(r["source"], 0) + int(r["words"])
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def summarize_values(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "sd": round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0,
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def deterministic_positions(input_ids, special_ids: set[int], text: str, mask_seed: int) -> list[int]:
    maskable = [i for i in range(input_ids.shape[1]) if int(input_ids[0, i]) not in special_ids]
    if not maskable:
        return []
    n_mask = max(1, int(len(maskable) * MASK_PROB))
    h = hashlib.sha256((str(mask_seed) + "\n" + text).encode("utf-8", errors="ignore")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return rng.sample(maskable, min(n_mask, len(maskable)))


def mean_loss(model, tokenizer, records: list[dict[str, Any]], mask_seed: int) -> dict[str, Any]:
    import torch
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    total_loss = 0.0
    total_masked = 0
    n_rows_used = 0
    n_subword_tokens = 0
    row_losses: list[float] = []
    with torch.no_grad():
        for r in records:
            text = r["text"]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=False)
            input_ids = enc["input_ids"].clone()
            labels = torch.full_like(input_ids, -100)
            positions = deterministic_positions(input_ids, special_ids, text, mask_seed)
            if not positions:
                continue
            for pos in positions:
                labels[0, pos] = input_ids[0, pos]
                input_ids[0, pos] = tokenizer.mask_token_id
            outputs = model(input_ids=input_ids, attention_mask=enc["attention_mask"], labels=labels)
            nm = int((labels != -100).sum().item())
            loss = float(outputs.loss.item())
            total_loss += loss * nm
            total_masked += nm
            n_rows_used += 1
            n_subword_tokens += int(enc["attention_mask"].sum().item())
            row_losses.append(loss)
    return {
        "mean_loss": round(total_loss / total_masked, 6) if total_masked else None,
        "row_loss_mean_unweighted": round(statistics.mean(row_losses), 6) if row_losses else None,
        "row_loss_sd_unweighted": round(statistics.stdev(row_losses), 6) if len(row_losses) >= 2 else 0.0 if row_losses else None,
        "n_rows": n_rows_used,
        "n_tokens_masked": total_masked,
        "n_subword_tokens": n_subword_tokens,
    }


def build_register_sets(n_register: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    base_rows = load_jsonl(BASE_POOL)
    sel = load_jsonl(SELECTION_RECORDS)
    child_idx = [int(r["childspeech_base_index"]) for r in sel]
    adult_idx = [int(r["adultprose_base_index"]) for r in sel]
    if len(child_idx) != len(set(child_idx)) or len(adult_idx) != len(set(adult_idx)):
        raise RuntimeError("register selection records contain duplicate base indices")
    child_all = select_rows_by_indices(base_rows, child_idx, label="register_childspeech_removed_block")
    adult_all = select_rows_by_indices(base_rows, adult_idx, label="register_adultprose_removed_block")
    all_sets: dict[str, list[dict[str, Any]]] = {}
    sample_preview: dict[str, Any] = {}
    for rep_i, (row_seed, mask_seed) in enumerate(zip(REPLICATE_SEEDS, MASK_SEEDS), start=1):
        all_sets[f"register_childspeech_removed__rep{rep_i}"] = sample_records(child_all, n_register, row_seed)
        all_sets[f"register_adultprose_removed__rep{rep_i}"] = sample_records(adult_all, n_register, row_seed + 100)
        sample_preview[f"rep{rep_i}"] = {
            "row_seed_child": row_seed,
            "row_seed_adult": row_seed + 100,
            "mask_seed": mask_seed,
            "child_first20": [{k: r[k] for k in ["row_index", "source", "words", "sha12"]} for r in all_sets[f"register_childspeech_removed__rep{rep_i}"][:20]],
            "adult_first20": [{k: r[k] for k in ["row_index", "source", "words", "sha12"]} for r in all_sets[f"register_adultprose_removed__rep{rep_i}"][:20]],
        }
    audit = {
        "base_pool": rel(BASE_POOL),
        "selection_records": rel(SELECTION_RECORDS),
        "n_selection_records": len(sel),
        "childspeech_all_rows": len(child_all),
        "adultprose_all_rows": len(adult_all),
        "childspeech_all_words": sum(int(r["words"]) for r in child_all),
        "adultprose_all_words": sum(int(r["words"]) for r in adult_all),
        "childspeech_source_words": source_word_summary(child_all),
        "adultprose_source_words": source_word_summary(adult_all),
        "sample_preview": sample_preview,
    }
    return all_sets, audit


def build_reference_sets(n_reference: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    all_sets: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {"arms": {}, "sample_preview": {}}
    for arm in REFERENCE_ARMS:
        rows = load_jsonl(POOL_FILES[arm])
        idx = load_changed_indices(CHANGED_META[arm])
        all_records = select_rows_by_indices(rows, idx, label=f"reference_{arm}_changed_block")
        audit["arms"][arm] = {
            "pool": rel(POOL_FILES[arm]),
            "changed_meta": rel(CHANGED_META[arm]),
            "n_changed_indices": len(idx),
            "all_rows": len(all_records),
            "all_words": sum(int(r["words"]) for r in all_records),
            "source_words": source_word_summary(all_records),
        }
        for rep_i, (row_seed, mask_seed) in enumerate(zip(REPLICATE_SEEDS, MASK_SEEDS), start=1):
            key = f"reference_{arm}_changed__rep{rep_i}"
            all_sets[key] = sample_records(all_records, n_reference, row_seed + 1000 + 17 * rep_i + 101 * REFERENCE_ARMS.index(arm))
            audit["sample_preview"][key] = {
                "row_seed": row_seed + 1000 + 17 * rep_i + 101 * REFERENCE_ARMS.index(arm),
                "mask_seed": mask_seed,
                "first20": [{k: r[k] for k in ["row_index", "source", "words", "sha12"]} for r in all_sets[key][:20]],
            }
    return all_sets, audit


def group_measurement_plan(register_sets: dict[str, list[dict[str, Any]]], reference_sets: dict[str, list[dict[str, Any]]]) -> list[tuple[str, str, pathlib.Path, list[str]]]:
    jobs: list[tuple[str, str, pathlib.Path, list[str]]] = []
    # Clean model: removal-side trajectory on both removed blocks.
    for ck in CHECKPOINTS:
        keys = []
        for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
            keys.extend([f"register_childspeech_removed__rep{rep_i}", f"register_adultprose_removed__rep{rep_i}"])
        jobs.append(("clean", ck, CLEAN_RUN / "hf_model" / ck, keys))
    # Reference arms: each trained model on its own changed block, resampled.
    for arm in REFERENCE_ARMS:
        for ck in ["chck_60M", "chck_100M"]:
            keys = [f"reference_{arm}_changed__rep{rep_i}" for rep_i in range(1, len(REPLICATE_SEEDS) + 1)]
            jobs.append((arm, ck, MODEL_RUNS[arm] / "hf_model" / ck, keys))
    return jobs


def rates_from_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, int, str], dict[str, float]] = {}
    for r in measurements:
        if r.get("mean_loss") is None:
            continue
        by_key.setdefault((r["set_base"], int(r["replicate"]), r["model_arm"]), {})[r["checkpoint"]] = float(r["mean_loss"])

    rate_rows: list[dict[str, Any]] = []
    for (set_base, rep, model_arm), vals in sorted(by_key.items()):
        rec: dict[str, Any] = {"set_base": set_base, "replicate": rep, "model_arm": model_arm}
        for ck in CHECKPOINTS:
            if ck in vals:
                rec[f"loss_{ck}"] = round(vals[ck], 6)
        if "chck_60M" in vals and "chck_100M" in vals:
            rec["loss_reduction_60_to_100"] = round(vals["chck_60M"] - vals["chck_100M"], 6)
        if "chck_80M" in vals and "chck_100M" in vals:
            rec["loss_reduction_80_to_100"] = round(vals["chck_80M"] - vals["chck_100M"], 6)
        if "chck_60M" in vals and "chck_80M" in vals:
            rec["loss_reduction_60_to_80"] = round(vals["chck_60M"] - vals["chck_80M"], 6)
        rate_rows.append(rec)

    summary: dict[str, Any] = {"rate_rows": rate_rows, "by_set": {}}
    for set_base in sorted({r["set_base"] for r in rate_rows}):
        rows = [r for r in rate_rows if r["set_base"] == set_base]
        summary["by_set"][set_base] = {
            "loss_reduction_60_to_100": summarize_values([r.get("loss_reduction_60_to_100") for r in rows if r.get("loss_reduction_60_to_100") is not None]),
            "loss_reduction_80_to_100": summarize_values([r.get("loss_reduction_80_to_100") for r in rows if r.get("loss_reduction_80_to_100") is not None]),
            "loss_reduction_60_to_80": summarize_values([r.get("loss_reduction_60_to_80") for r in rows if r.get("loss_reduction_60_to_80") is not None]),
        }

    # Register removal direct prediction.
    child_vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == "register_childspeech_removed" and r.get("loss_reduction_60_to_100") is not None]
    adult_vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == "register_adultprose_removed" and r.get("loss_reduction_60_to_100") is not None]
    paired_diffs = []
    by_rep_child = {r["replicate"]: r for r in rate_rows if r["set_base"] == "register_childspeech_removed"}
    by_rep_adult = {r["replicate"]: r for r in rate_rows if r["set_base"] == "register_adultprose_removed"}
    for rep in sorted(set(by_rep_child) & set(by_rep_adult)):
        c = by_rep_child[rep].get("loss_reduction_60_to_100")
        a = by_rep_adult[rep].get("loss_reduction_60_to_100")
        if c is not None and a is not None:
            # Positive means adult-prose removed block was still being learned more late.
            paired_diffs.append(float(a) - float(c))
    child_mean = statistics.mean(child_vals) if child_vals else None
    adult_mean = statistics.mean(adult_vals) if adult_vals else None
    diff_mean = statistics.mean(paired_diffs) if paired_diffs else None
    if diff_mean is None:
        sign_text = "unavailable"
    elif diff_mean > 0.0:
        sign_text = "positive_register_contrast_predicted_by_rate: adult-prose block has larger clean late loss reduction, so removing adult prose should be costlier and childspeech_removed - adultprose_removed should be positive"
    elif diff_mean < 0.0:
        sign_text = "negative_register_contrast_predicted_by_rate: child/speech block has larger clean late loss reduction, so removing child/speech should be costlier and childspeech_removed - adultprose_removed should be negative"
    else:
        sign_text = "near_zero_register_contrast_predicted_by_rate: removed blocks have matched clean late loss reductions"

    summary["register_rate_prediction"] = {
        "contrast": REGISTER_PROFILE_PREDICTION["contrast"],
        "meaning_positive": REGISTER_PROFILE_PREDICTION["meaning_positive"],
        "childspeech_removed_block_rate_60_to_100": summarize_values(child_vals),
        "adultprose_removed_block_rate_60_to_100": summarize_values(adult_vals),
        "adult_minus_child_removed_block_rate": summarize_values(paired_diffs),
        "rate_based_sign_commitment": sign_text,
        "comparison_to_step288_static_profile": REGISTER_PROFILE_PREDICTION,
        "interpretation": "If this rate sign and the static profile sign coincide, a positive register score cannot by itself distinguish target-profile proximity from removal-side active-error opportunity cost. If they diverge, the register score adjudicates between them. In either case, Entity must remain separated from broad ex-Entity interpretation.",
    }

    # Reference spread and midpoint uncertainty.
    ref_means = {}
    for arm in REFERENCE_ARMS:
        vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == f"reference_{arm}_changed" and r.get("loss_reduction_60_to_100") is not None]
        ref_means[arm] = statistics.mean(vals) if vals else None
    if all(ref_means.get(a) is not None for a in ["repeat", "breadth", "view"]):
        rep_vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == "reference_repeat_changed" and r.get("loss_reduction_60_to_100") is not None]
        br_vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == "reference_breadth_changed" and r.get("loss_reduction_60_to_100") is not None]
        vw_vals = [r["loss_reduction_60_to_100"] for r in rate_rows if r["set_base"] == "reference_view_changed" and r.get("loss_reduction_60_to_100") is not None]
        midpoint_repeat_breadth = [(r + b) / 2.0 for r, b in zip(rep_vals, br_vals)]
        midpoint_repeat_view = [(r + v) / 2.0 for r, v in zip(rep_vals, vw_vals)]
        summary["reference_rate_spread"] = {
            "historical_single_sample_rates_60_to_100": HISTORICAL_REFERENCE_RATES_60_100,
            "historical_late_exEntity5": HISTORICAL_LATE_EXENTITY5,
            "resampled_rates_60_to_100": {arm: summary["by_set"].get(f"reference_{arm}_changed", {}).get("loss_reduction_60_to_100") for arm in REFERENCE_ARMS},
            "repeat_breadth_midpoint_spread": summarize_values(midpoint_repeat_breadth),
            "repeat_view_midpoint_spread": summarize_values(midpoint_repeat_view),
            "note": "Use this empirical spread when placing future in-corpus rates; the repeat/breadth/view scale is narrow and sample-sensitive, so midpoint rules should be treated as measured bands rather than sharp constants.",
        }
    return summary


def write_markdown(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    summary = payload["rate_summary"]
    reg = summary.get("register_rate_prediction", {})
    ref = summary.get("reference_rate_spread", {})
    lines: list[str] = []
    lines.append("# research register-removal rate and reference-rate spread")
    lines.append("")
    lines.append("This CPU-only forward MLM loss measurement was made before inspecting any new childspeech official scorer result. It freezes how the active-error/rate account predicts the MAX-register contrast and quantifies how much the repeat/view/breadth rate scale moves under row/mask resampling.")
    lines.append("")
    lines.append("## Register removal-side rate commitment")
    lines.append(f"- Contrast: `{reg.get('contrast')}`")
    lines.append(f"- Positive contrast means: {reg.get('meaning_positive')}")
    lines.append(f"- Child/speech removed block 60M→100M clean loss reduction: {reg.get('childspeech_removed_block_rate_60_to_100')}")
    lines.append(f"- Adult-prose removed block 60M→100M clean loss reduction: {reg.get('adultprose_removed_block_rate_60_to_100')}")
    lines.append(f"- Adult-minus-child removed-block rate: {reg.get('adult_minus_child_removed_block_rate')}")
    lines.append(f"- Rate sign: {reg.get('rate_based_sign_commitment')}")
    lines.append("")
    lines.append("## Static-profile comparison frozen earlier")
    profile = reg.get("comparison_to_step288_static_profile", {})
    for k in ["profile_pred_exEntity5", "profile_pred_cheap6", "strict_profile_pred_exEntity5", "strict_profile_pred_cheap6", "word_js_pred_exEntity5", "word_js_pred_cheap6"]:
        lines.append(f"- {k}: {profile.get(k)}")
    lines.append("")
    lines.append("## Reference rate spread")
    lines.append(f"- Historical single-sample rates: {ref.get('historical_single_sample_rates_60_to_100')}")
    lines.append(f"- Resampled rates: {ref.get('resampled_rates_60_to_100')}")
    lines.append(f"- Repeat/breadth midpoint spread: {ref.get('repeat_breadth_midpoint_spread')}")
    lines.append(f"- Repeat/view midpoint spread: {ref.get('repeat_view_midpoint_spread')}")
    lines.append("")
    lines.append("## Row-block audit")
    ra = payload.get("register_audit", {})
    lines.append(f"- Register selected rows: child/speech {ra.get('childspeech_all_rows')} rows / {ra.get('childspeech_all_words')} words; adult prose {ra.get('adultprose_all_rows')} rows / {ra.get('adultprose_all_words')} words.")
    lines.append(f"- Child/speech sources: {ra.get('childspeech_source_words')}")
    lines.append(f"- Adult-prose sources: {ra.get('adultprose_source_words')}")
    lines.append("")
    lines.append("## Files")
    for name, path in payload.get("files", {}).items():
        lines.append(f"- {name}: `{path}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--n-register", type=int, default=300)
    parser.add_argument("--n-reference", type=int, default=300)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for p in [TOKENIZER_DIR, BASE_POOL, SELECTION_RECORDS, REGISTER_META, CLEAN_RUN]:
        if not p.exists():
            missing.append(rel(p))
    for arm in REFERENCE_ARMS:
        for p in [POOL_FILES[arm], CHANGED_META[arm], MODEL_RUNS[arm]]:
            if not p.exists():
                missing.append(rel(p))
    for ck in CHECKPOINTS:
        if not model_ready(CLEAN_RUN, ck):
            missing.append(f"clean_model:{ck}")
    for arm in REFERENCE_ARMS:
        for ck in ["chck_60M", "chck_100M"]:
            if not model_ready(MODEL_RUNS[arm], ck):
                missing.append(f"{arm}_model:{ck}")

    register_sets: dict[str, list[dict[str, Any]]] = {}
    reference_sets: dict[str, list[dict[str, Any]]] = {}
    register_audit: dict[str, Any] = {}
    reference_audit: dict[str, Any] = {}
    if not missing:
        register_sets, register_audit = build_register_sets(args.n_register)
        reference_sets, reference_audit = build_reference_sets(args.n_reference)
    jobs = group_measurement_plan(register_sets, reference_sets) if not missing else []
    plan = {
        "status": "REGISTER_REMOVAL_RATE_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "cpu_only": True,
        "no_training_no_official_eval_no_upload_no_leaderboard": True,
        "n_register_per_replicate": args.n_register,
        "n_reference_per_replicate": args.n_reference,
        "replicate_seeds": REPLICATE_SEEDS,
        "mask_seeds": MASK_SEEDS,
        "checkpoints": CHECKPOINTS,
        "reference_arms": REFERENCE_ARMS,
        "register_audit_short": {k: register_audit.get(k) for k in ["n_selection_records", "childspeech_all_rows", "childspeech_all_words", "childspeech_source_words", "adultprose_all_rows", "adultprose_all_words", "adultprose_source_words"]},
        "n_model_load_jobs": len(jobs),
        "n_measurements": sum(len(keys) for _, _, _, keys in jobs),
        "missing": missing,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if missing:
        raise SystemExit("Missing required inputs: " + "; ".join(missing[:30]))

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    all_sets: dict[str, list[dict[str, Any]]] = {}
    all_sets.update(register_sets)
    all_sets.update(reference_sets)
    set_meta: dict[str, dict[str, Any]] = {}
    for key in all_sets:
        if key.startswith("register_childspeech_removed"):
            set_base = "register_childspeech_removed"
            rep = int(key.rsplit("rep", 1)[1])
        elif key.startswith("register_adultprose_removed"):
            set_base = "register_adultprose_removed"
            rep = int(key.rsplit("rep", 1)[1])
        elif key.startswith("reference_"):
            parts = key.split("__rep")
            set_base = parts[0]
            rep = int(parts[1])
        else:
            set_base = key
            rep = 0
        set_meta[key] = {"set_base": set_base, "replicate": rep, "mask_seed": MASK_SEEDS[rep - 1] if rep else MASK_SEEDS[0]}

    measurements: list[dict[str, Any]] = []
    for job_i, (arm, ck, model_path, keys) in enumerate(jobs, start=1):
        print(f"\n[{job_i}/{len(jobs)}] loading {arm} {ck}: {rel(model_path)}", flush=True)
        t0 = time.time()
        model = AutoModelForMaskedLM.from_pretrained(str(model_path))
        model.eval()
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        for key in keys:
            meta = set_meta[key]
            t1 = time.time()
            info = mean_loss(model, tokenizer, all_sets[key], int(meta["mask_seed"]))
            rec = {
                "model_arm": arm,
                "checkpoint": ck,
                "set_key": key,
                "set_base": meta["set_base"],
                "replicate": int(meta["replicate"]),
                "mask_seed": int(meta["mask_seed"]),
                **info,
                "eval_time_sec": round(time.time() - t1, 1),
            }
            measurements.append(rec)
            print(f"  {key:<42} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
        del model
        import gc
        gc.collect()

    rate_summary = rates_from_measurements(measurements)
    files = {
        "summary_json": rel(out_dir / "register_removal_rate_and_reference_spread.json"),
        "measurements_csv": rel(out_dir / "loss_measurements.csv"),
        "rates_csv": rel(out_dir / "rate_rows.csv"),
        "commitment_json": rel(out_dir / "prediction_commitment.json"),
        "summary_md": rel(out_dir / "register_removal_rate_and_reference_spread.md"),
    }
    payload = {
        "status": "REGISTER_REMOVAL_RATE_DONE",
        "finished_utc": now(),
        "purpose": "pre-score register-removal rate commitment plus reference rate spread for active-error account",
        "parameters": {
            "n_register_per_replicate": args.n_register,
            "n_reference_per_replicate": args.n_reference,
            "replicate_seeds": REPLICATE_SEEDS,
            "mask_seeds": MASK_SEEDS,
            "mask_prob": MASK_PROB,
            "max_seq_len": MAX_SEQ_LEN,
            "checkpoints": CHECKPOINTS,
            "reference_arms": REFERENCE_ARMS,
            "cpu_only": True,
            "no_training_no_official_eval_no_upload_no_leaderboard": True,
        },
        "register_audit": register_audit,
        "reference_audit": reference_audit,
        "measurements": measurements,
        "rate_summary": rate_summary,
        "files": files,
    }

    (out_dir / "register_removal_rate_and_reference_spread.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "prediction_commitment.json").write_text(json.dumps(rate_summary.get("register_rate_prediction", {}), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (out_dir / "loss_measurements.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["model_arm", "checkpoint", "set_key", "set_base", "replicate", "mask_seed", "mean_loss", "row_loss_mean_unweighted", "row_loss_sd_unweighted", "n_rows", "n_tokens_masked", "n_subword_tokens", "eval_time_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in measurements:
            w.writerow({k: r.get(k) for k in fieldnames})
    with (out_dir / "rate_rows.csv").open("w", encoding="utf-8", newline="") as f:
        rate_rows = rate_summary.get("rate_rows", [])
        fieldnames = sorted({k for r in rate_rows for k in r.keys()}) if rate_rows else ["set_base", "replicate", "model_arm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rate_rows:
            w.writerow(r)
    write_markdown(payload, out_dir / "register_removal_rate_and_reference_spread.md")

    print("\n=== research REGISTER RATE COMMITMENT ===", flush=True)
    print(json.dumps(rate_summary.get("register_rate_prediction", {}), indent=2, ensure_ascii=False), flush=True)
    print("\n=== research REFERENCE RATE SPREAD ===", flush=True)
    print(json.dumps(rate_summary.get("reference_rate_spread", {}), indent=2, ensure_ascii=False), flush=True)
    print(json.dumps({"status": payload["status"], "files": files}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
