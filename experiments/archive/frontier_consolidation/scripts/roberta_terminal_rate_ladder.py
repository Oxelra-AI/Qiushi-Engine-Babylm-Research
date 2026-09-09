#!/usr/bin/env python3
"""research: RoBERTa terminal active-error ladder for the fixed-budget principle.

Scientific role
---------------
No new training and no official benchmark evaluation.  Existing RoBERTa MAX-dose
view/repeat/clean checkpoints are used to test whether the DeBERTa active-error
pattern survives in an architecture whose downstream broad transfer reverses.

Two readouts are recorded:
  1. own-arm terminal rates: each trained RoBERTa arm's loss reduction on its
     own changed block and matched unchanged rows over 60->100M and 80->100M;
  2. clean-prior terminal rates: the clean RoBERTa model's mature 80->100M loss
     reduction on the view/repeat admitted blocks and the displaced clean block,
     using the research-style three row/mask replicates.

Interpretation:
  - If terminal residual-error rates are DeBERTa-like but downstream RoBERTa
    view-clean is negative, the principle requires an architecture/conversion
    term: residual error is an available learning source but not automatically
    transferred into selected competence.
  - If the rates themselves collapse or reverse, the active-error component is
    not architecture-invariant.
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

# CPU-only before torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = WS / "data" / "roberta_terminal_rate_ladder"
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"

MODEL_RUNS = {
    "view": WS / "training" / "runs" / "roberta_view_dose2p64x_matched_rowholdout_100M_seed43022",
    "repeat": WS / "training" / "runs" / "roberta_repeat_dose2p64x_matched_rowholdout_100M_seed43022",
    "clean": WS / "training" / "runs" / "roberta_clean_dose2p64x_matched_rowholdout_100M_seed43022",
}
POOL_FILES = {
    "view": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_10M.jsonl",
    "repeat": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_10M.jsonl",
    "clean": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
}
CHANGED_META = {
    "view": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
    "clean": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl",
}

# Own-arm ladder uses a compact terminal trajectory; clean-prior reproduces the
# research mature window exactly.
CHECKPOINTS_OWN = ["chck_60M", "chck_80M", "chck_100M"]
CHECKPOINTS_CLEAN_PRIOR = ["chck_80M", "chck_100M"]
ARMS = ["view", "repeat", "clean"]
REPLICATE_SEEDS = [29501, 29502, 29503]
MASK_SEEDS = [295101, 295102, 295103]
MASK_PROB = 0.15
MAX_SEQ_LEN = 256

# Downstream anchors already established in earlier files. These are not refit.
DOWNSTREAM_ANCHORS = {
    "deberta_late_exEntity5": {"view_minus_clean": 0.3853, "repeat_minus_clean": 0.0343},
    "roberta_late_exEntity5": {"view_minus_clean": -0.6873},
    "roberta_late_cheap6_no_GlobalPIQA": {"view_minus_clean": -0.5283},
    "same_coordinate_mature_exEntity5_resolution_floor": 0.1765,
}
DEBERTA_REFERENCE_RATES = {
    "own_60_to_100_changed": {"view": 0.295710, "repeat": 0.148226, "clean": 0.176365},  # clean from research changed 3.154680->2.978315
    "own_80_to_100_changed": {"view": 0.051544, "repeat": 0.036072, "clean": 0.039342},
    "clean_prior_80_to_100": {"view": 0.057070, "repeat": 0.037811, "clean": 0.042389},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def nwords(row: dict[str, Any]) -> int:
    if row.get("words") is not None:
        return int(row["words"])
    return len(str(row.get("text", "")).split())


def normalize_source(source: Any) -> str:
    s = str(source or "unknown")
    return s.split("::", 1)[1] if "::" in s else s


def load_changed_indices(meta_path: pathlib.Path) -> set[int]:
    out: set[int] = set()
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(int(json.loads(line)["row_index"]))
    return out


def load_records(pool_path: pathlib.Path, meta_path: pathlib.Path, label: str) -> dict[str, list[dict[str, Any]]]:
    changed_idx = load_changed_indices(meta_path)
    changed: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    with pool_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            rec = {
                "arm": label,
                "row_index": idx,
                "text": text,
                "words": nwords(row),
                "source": normalize_source(row.get("source", row.get("source_name", "unknown"))),
                "sha12": sha12(text),
            }
            if idx in changed_idx:
                changed.append(rec)
            else:
                unchanged.append(rec)
    return {"changed": changed, "unchanged": unchanged}


def source_word_summary(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r["source"]] = out.get(r["source"], 0) + int(r["words"])
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def sample_records(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(records) if len(records) <= n else rng.sample(records, n)
    return sorted(selected, key=lambda r: (int(r["row_index"]), r["sha12"]))


def model_ready(arm: str, ck: str) -> bool:
    p = MODEL_RUNS[arm] / "hf_model" / ck
    if not (p / "config.json").exists():
        return False
    return any((p / name).exists() for name in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"])


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
    # RoBERTa/BPE uses bos/eos in addition to cls/sep aliases; include if present.
    for tok_id in [getattr(tokenizer, "bos_token_id", None), getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "mask_token_id", None)]:
        if tok_id is not None and tok_id != tokenizer.mask_token_id:
            special_ids.add(int(tok_id))
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
        "row_loss_sd_unweighted": round(statistics.stdev(row_losses), 6) if len(row_losses) >= 2 else (0.0 if row_losses else None),
        "n_rows": n_rows_used,
        "n_tokens_masked": total_masked,
        "n_subword_tokens": n_subword_tokens,
    }


def summarize(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "values": []}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 6),
        "sd": round(statistics.stdev(vals), 6) if len(vals) >= 2 else 0.0,
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_outputs(payload: dict[str, Any], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "summary_json": out_dir / "roberta_terminal_rate_ladder.json",
        "measurements_csv": out_dir / "loss_measurements.csv",
        "rate_rows_csv": out_dir / "rate_rows.csv",
        "summary_md": out_dir / "roberta_terminal_rate_ladder.md",
    }
    write_json(files["summary_json"], payload)
    if payload.get("measurements"):
        with files["measurements_csv"].open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "readout", "model_arm", "checkpoint", "text_arm", "block_type", "replicate", "row_seed", "mask_seed",
                "mean_loss", "row_loss_mean_unweighted", "row_loss_sd_unweighted", "n_rows", "n_tokens_masked", "n_subword_tokens", "eval_time_sec",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in payload["measurements"]:
                w.writerow({k: r.get(k) for k in fieldnames})
    with files["rate_rows_csv"].open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["readout", "model_arm", "text_arm", "block_type", "replicate", "rate_60_to_100", "rate_80_to_100", "loss_60M", "loss_80M", "loss_100M"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in payload.get("rate_rows", []):
            w.writerow({k: r.get(k) for k in fieldnames})

    lines: list[str] = []
    lines.append("# research RoBERTa terminal active-error ladder")
    lines.append("")
    lines.append("CPU-only forward MLM measurement on existing RoBERTa MAX-dose view/repeat/clean checkpoints. No training, no official benchmark evaluation, no upload, no leaderboard action.")
    lines.append("")
    lines.append("## Why this measurement matters")
    lines.append("RoBERTa already has a negative broad late MAX view-clean downstream contrast (exEntity5 -0.6873, cheap6 -0.5283), while DeBERTa retains positive late value for distinct admitted content. This readout asks whether RoBERTa lacks the terminal active-error source itself, or has the source but does not convert it into broad competence.")
    lines.append("")
    lines.append("## Clean-prior mature 80M→100M rates")
    for text_arm, summ in payload.get("clean_prior_rate_summary", {}).items():
        lines.append(f"- {text_arm}: {summ}")
    lines.append("")
    lines.append("## Own-arm changed-block terminal rates")
    for arm, by_block in payload.get("own_rate_summary", {}).items():
        lines.append(f"### {arm}")
        for block, summ in by_block.items():
            lines.append(f"- {block}: {summ}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(payload.get("scientific_interpretation", ""))
    lines.append("")
    lines.append("## Files")
    for k, p in files.items():
        lines.append(f"- {k}: `{rel(p)}`")
    files["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["files"] = {k: rel(v) for k, v in files.items()}
    write_json(files["summary_json"], payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    if not TOKENIZER_DIR.exists():
        missing.append(rel(TOKENIZER_DIR))
    for arm in ARMS:
        if not POOL_FILES[arm].exists():
            missing.append(rel(POOL_FILES[arm]))
        if not CHANGED_META[arm].exists():
            missing.append(rel(CHANGED_META[arm]))
        for ck in sorted(set(CHECKPOINTS_OWN + CHECKPOINTS_CLEAN_PRIOR)):
            if not model_ready(arm, ck):
                missing.append(f"model:{arm}/{ck}")

    all_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    audits: dict[str, Any] = {}
    if not missing:
        for arm in ARMS:
            recs = load_records(POOL_FILES[arm], CHANGED_META[arm], arm)
            all_records[arm] = recs
            audits[arm] = {
                "pool": rel(POOL_FILES[arm]),
                "meta": rel(CHANGED_META[arm]),
                "n_changed_rows": len(recs["changed"]),
                "n_unchanged_rows": len(recs["unchanged"]),
                "changed_words": sum(int(r["words"]) for r in recs["changed"]),
                "unchanged_words_total": sum(int(r["words"]) for r in recs["unchanged"]),
                "changed_source_words": source_word_summary(recs["changed"]),
            }

    sample_sets: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    sample_meta: dict[tuple[str, str, int], dict[str, int]] = {}
    for rep_i, (row_seed, mask_seed) in enumerate(zip(REPLICATE_SEEDS, MASK_SEEDS), start=1):
        for arm_index, arm in enumerate(ARMS):
            changed_seed = row_seed + 3000 + 17 * rep_i + 101 * arm_index
            unchanged_seed = row_seed + 4000 + 17 * rep_i + 101 * arm_index
            sample_meta[(arm, "changed", rep_i)] = {"row_seed": changed_seed, "mask_seed": mask_seed}
            sample_meta[(arm, "unchanged", rep_i)] = {"row_seed": unchanged_seed, "mask_seed": mask_seed}
            if not missing:
                sample_sets[(arm, "changed", rep_i)] = sample_records(all_records[arm]["changed"], args.n, changed_seed)
                sample_sets[(arm, "unchanged", rep_i)] = sample_records(all_records[arm]["unchanged"], args.n, unchanged_seed)

    plan = {
        "status": "ROBERTA_TERMINAL_RATE_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "cpu_only": True,
        "no_training_no_official_eval_no_upload_no_leaderboard": True,
        "n_per_replicate": args.n,
        "arms": ARMS,
        "checkpoints_own": CHECKPOINTS_OWN,
        "checkpoints_clean_prior": CHECKPOINTS_CLEAN_PRIOR,
        "replicate_seeds": REPLICATE_SEEDS,
        "mask_seeds": MASK_SEEDS,
        "n_loss_measurements": len(ARMS) * len(REPLICATE_SEEDS) * (len(CHECKPOINTS_OWN) * 2 + len(CHECKPOINTS_CLEAN_PRIOR)),
        "audits_short": audits,
        "missing": missing,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    write_json(out_dir / "last_plan.json", plan)
    if args.plan_only:
        return
    if missing:
        raise SystemExit("Missing inputs: " + "; ".join(missing))

    import gc
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    try:
        torch.set_num_threads(min(24, max(1, os.cpu_count() or 1)))
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    measurements: list[dict[str, Any]] = []

    # Own-arm ladder.
    total_loads = len(ARMS) * len(CHECKPOINTS_OWN) + len(CHECKPOINTS_CLEAN_PRIOR)
    load_i = 0
    for model_arm in ARMS:
        for ck in CHECKPOINTS_OWN:
            load_i += 1
            model_path = MODEL_RUNS[model_arm] / "hf_model" / ck
            print(f"\n[{load_i}/{total_loads}] loading own {model_arm} {ck}: {rel(model_path)}", flush=True)
            t0 = time.time()
            model = AutoModelForMaskedLM.from_pretrained(str(model_path))
            model.eval()
            print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
            for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
                for block in ["changed", "unchanged"]:
                    meta = sample_meta[(model_arm, block, rep_i)]
                    t1 = time.time()
                    info = mean_loss(model, tokenizer, sample_sets[(model_arm, block, rep_i)], int(meta["mask_seed"]))
                    rec = {
                        "readout": "own_arm",
                        "model_arm": model_arm,
                        "checkpoint": ck,
                        "text_arm": model_arm,
                        "block_type": block,
                        "replicate": rep_i,
                        "row_seed": int(meta["row_seed"]),
                        "mask_seed": int(meta["mask_seed"]),
                        **info,
                        "eval_time_sec": round(time.time() - t1, 1),
                    }
                    measurements.append(rec)
                    print(f"  rep{rep_i} {block:<9} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
            del model
            gc.collect()

    # Clean-prior mature window: clean RoBERTa model on each changed/admitted block.
    for ck in CHECKPOINTS_CLEAN_PRIOR:
        load_i += 1
        model_path = MODEL_RUNS["clean"] / "hf_model" / ck
        print(f"\n[{load_i}/{total_loads}] loading clean-prior RoBERTa {ck}: {rel(model_path)}", flush=True)
        t0 = time.time()
        model = AutoModelForMaskedLM.from_pretrained(str(model_path))
        model.eval()
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
            for text_arm in ARMS:
                meta = sample_meta[(text_arm, "changed", rep_i)]
                t1 = time.time()
                info = mean_loss(model, tokenizer, sample_sets[(text_arm, "changed", rep_i)], int(meta["mask_seed"]))
                rec = {
                    "readout": "clean_prior",
                    "model_arm": "clean",
                    "checkpoint": ck,
                    "text_arm": text_arm,
                    "block_type": "changed",
                    "replicate": rep_i,
                    "row_seed": int(meta["row_seed"]),
                    "mask_seed": int(meta["mask_seed"]),
                    **info,
                    "eval_time_sec": round(time.time() - t1, 1),
                }
                measurements.append(rec)
                print(f"  rep{rep_i} clean_prior->{text_arm:<6} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
        del model
        gc.collect()

    by_key: dict[tuple[str, str, str, str, int], dict[str, float]] = {}
    for rec in measurements:
        if rec.get("mean_loss") is None:
            continue
        key = (rec["readout"], rec["model_arm"], rec["text_arm"], rec["block_type"], int(rec["replicate"]))
        by_key.setdefault(key, {})[rec["checkpoint"]] = float(rec["mean_loss"])

    rate_rows: list[dict[str, Any]] = []
    # Own-arm rates.
    for arm in ARMS:
        for block in ["changed", "unchanged"]:
            for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
                losses = by_key.get(("own_arm", arm, arm, block, rep_i), {})
                r60 = (losses.get("chck_60M") - losses.get("chck_100M")) if "chck_60M" in losses and "chck_100M" in losses else None
                r80 = (losses.get("chck_80M") - losses.get("chck_100M")) if "chck_80M" in losses and "chck_100M" in losses else None
                rate_rows.append({
                    "readout": "own_arm",
                    "model_arm": arm,
                    "text_arm": arm,
                    "block_type": block,
                    "replicate": rep_i,
                    "rate_60_to_100": round(r60, 6) if r60 is not None else None,
                    "rate_80_to_100": round(r80, 6) if r80 is not None else None,
                    "loss_60M": round(losses.get("chck_60M"), 6) if "chck_60M" in losses else None,
                    "loss_80M": round(losses.get("chck_80M"), 6) if "chck_80M" in losses else None,
                    "loss_100M": round(losses.get("chck_100M"), 6) if "chck_100M" in losses else None,
                })
    # Clean-prior rates.
    for text_arm in ARMS:
        for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
            losses = by_key.get(("clean_prior", "clean", text_arm, "changed", rep_i), {})
            r80 = (losses.get("chck_80M") - losses.get("chck_100M")) if "chck_80M" in losses and "chck_100M" in losses else None
            rate_rows.append({
                "readout": "clean_prior",
                "model_arm": "clean",
                "text_arm": text_arm,
                "block_type": "changed",
                "replicate": rep_i,
                "rate_60_to_100": None,
                "rate_80_to_100": round(r80, 6) if r80 is not None else None,
                "loss_60M": None,
                "loss_80M": round(losses.get("chck_80M"), 6) if "chck_80M" in losses else None,
                "loss_100M": round(losses.get("chck_100M"), 6) if "chck_100M" in losses else None,
            })

    clean_prior_rate_summary = {
        arm: summarize([r["rate_80_to_100"] for r in rate_rows if r["readout"] == "clean_prior" and r["text_arm"] == arm and r["rate_80_to_100"] is not None])
        for arm in ARMS
    }
    own_rate_summary: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        own_rate_summary[arm] = {}
        for block in ["changed", "unchanged"]:
            rows = [r for r in rate_rows if r["readout"] == "own_arm" and r["model_arm"] == arm and r["block_type"] == block]
            own_rate_summary[arm][block] = {
                "rate_60_to_100": summarize([r["rate_60_to_100"] for r in rows if r["rate_60_to_100"] is not None]),
                "rate_80_to_100": summarize([r["rate_80_to_100"] for r in rows if r["rate_80_to_100"] is not None]),
                "loss_100M": summarize([r["loss_100M"] for r in rows if r["loss_100M"] is not None]),
            }

    clean_means = {arm: clean_prior_rate_summary[arm]["mean"] for arm in ARMS}
    own_changed_60_means = {arm: own_rate_summary[arm]["changed"]["rate_60_to_100"]["mean"] for arm in ARMS}
    own_changed_80_means = {arm: own_rate_summary[arm]["changed"]["rate_80_to_100"]["mean"] for arm in ARMS}
    view_active = (clean_means.get("view") is not None and clean_means.get("repeat") is not None and clean_means["view"] > clean_means["repeat"])
    view_vs_clean_active = (clean_means.get("view") is not None and clean_means.get("clean") is not None and clean_means["view"] > clean_means["clean"])
    if view_active and view_vs_clean_active:
        interp = (
            "RoBERTa preserves a mature clean-prior active-error signal for the view block despite its negative broad late downstream view-clean contrast. "
            "This makes terminal residual error an available source rather than a sufficient cause of transferable competence; the missing term is architecture/objective/representation conversion."
        )
    elif clean_means.get("view") is not None and clean_means.get("repeat") is not None and clean_means["view"] <= clean_means["repeat"]:
        interp = (
            "RoBERTa does not preserve the DeBERTa-like view>repeat clean-prior terminal-rate ordering. "
            "The active-error component itself is architecture/coordinate-sensitive, so conversion cannot be the only boundary."
        )
    else:
        interp = (
            "RoBERTa clean-prior terminal rates are weak or near the clean-displaced drift; the result supports a bounded active-error component but does not by itself establish architecture conversion."
        )
    interp += (
        f" Downstream anchors remain: DeBERTa late view-clean exEntity5 {DOWNSTREAM_ANCHORS['deberta_late_exEntity5']['view_minus_clean']:+.4f}, "
        f"DeBERTa repeat-clean {DOWNSTREAM_ANCHORS['deberta_late_exEntity5']['repeat_minus_clean']:+.4f}, RoBERTa late view-clean exEntity5 {DOWNSTREAM_ANCHORS['roberta_late_exEntity5']['view_minus_clean']:+.4f}, "
        f"with same-coordinate mature exEntity5 resolution floor {DOWNSTREAM_ANCHORS['same_coordinate_mature_exEntity5_resolution_floor']:.4f}."
    )

    payload = {
        "status": "ROBERTA_TERMINAL_RATE_DONE",
        "finished_utc": now(),
        "purpose": "RoBERTa out-of-sample terminal active-error test for fixed-budget substitution principle",
        "parameters": {
            "n_per_replicate": args.n,
            "arms": ARMS,
            "checkpoints_own": CHECKPOINTS_OWN,
            "checkpoints_clean_prior": CHECKPOINTS_CLEAN_PRIOR,
            "replicate_seeds": REPLICATE_SEEDS,
            "mask_seeds": MASK_SEEDS,
            "mask_prob": MASK_PROB,
            "max_seq_len": MAX_SEQ_LEN,
            "cpu_only": True,
            "no_training_no_official_eval_no_upload_no_leaderboard": True,
        },
        "audits": audits,
        "measurements": measurements,
        "rate_rows": rate_rows,
        "clean_prior_rate_summary": clean_prior_rate_summary,
        "own_rate_summary": own_rate_summary,
        "roberta_rate_means": {
            "clean_prior_80_to_100_changed": clean_means,
            "own_changed_60_to_100": own_changed_60_means,
            "own_changed_80_to_100": own_changed_80_means,
        },
        "deberta_reference_rates": DEBERTA_REFERENCE_RATES,
        "downstream_anchors": DOWNSTREAM_ANCHORS,
        "scientific_interpretation": interp,
    }
    write_outputs(payload, out_dir)
    print(json.dumps({
        "status": payload["status"],
        "clean_prior_rate_summary": clean_prior_rate_summary,
        "own_changed_60_to_100": own_changed_60_means,
        "own_changed_80_to_100": own_changed_80_means,
        "scientific_interpretation": interp,
        "files": payload["files"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
