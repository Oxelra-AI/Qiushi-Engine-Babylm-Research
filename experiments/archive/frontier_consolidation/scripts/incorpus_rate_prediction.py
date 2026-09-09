#!/usr/bin/env python3
"""research: pre-score rate prediction for the in-corpus adult-prose final cell.

Purpose
-------
Before reading official family scores for the in-corpus adult-prose arm, measure
whether its admitted rows are already easy for the clean DeBERTa model and
whether the in-corpus-trained model (for checkpoints that already exist) treats
those rows like an exhausted duplicate block or like an active distinct-content
error source. This is a CPU-only forward MLM loss probe, not official benchmark
scoring and not new training.

It implements the rate correction: the existing view/repeat/breadth
loss ladder is most informative through late loss-reduction rate, especially
60M->100M, not absolute loss level. Repeat collapses early and has only a small
late loss drop; view and breadth keep reducing admitted-block loss later and are
the arms whose broad downstream advantage persisted late.

The script writes a pre-official-inspection prediction commitment from these
losses. If in-corpus 60M/100M losses are not available yet, the commitment is a
frozen conditional prediction keyed to the future same-script rate measurement,
with the current clean-prior measurement recorded now.
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
from typing import Any

# CPU-only before torch/transformers import.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"
OUT_DIR = WS / "data" / "incorpus_rate_prediction"

CLEAN_POOL = WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl"
INCORPUS_POOL = WS / "data" / "incorpus_adultprose_arm" / "incorpus_adultprose_rho0p042_10M.jsonl"
INCORPUS_META = WS / "data" / "incorpus_adultprose_arm" / "incorpus_adultprose_metadata.json"

CLEAN_RUN = WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022"
INCORPUS_RUN = WS / "training" / "runs" / "incorpus_adultprose_deberta100M_seed43022"

PERSISTENCE_JSON = WS / "data" / "persistence_loss_analysis" / "persistence_loss_results.json"
CROSSLOSS_JSON = WS / "data" / "persistence_crossloss_analysis" / "crossloss_results.json"

DEFAULT_CLEAN_CKS = ["chck_60M", "chck_80M", "chck_100M"]
ALL_CKS = [f"chck_{i}M" for i in range(10, 101, 10)]
# Match the research deterministic text-hash masking so clean-prior comparisons
# are against the recorded repeat/view/breadth cross-loss values.
BASE_MASK_SEED = 293293
MASK_PROB = 0.15
MAX_SEQ_LEN = 256

# Late downstream retained broad effect from research/287 for the MAX arms. These
# are not recomputed here; they anchor the rate-to-score comparison.
LATE_EXENTITY5_FROM_STEP287 = {
    "repeat": 0.0343,
    "view": 0.3853,
    "breadth": 0.3350,
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_src(s: str) -> str:
    return s.split("::", 1)[1] if "::" in s else s


def text_sha(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def detect_changed_rows(clean_rows: list[dict[str, Any]], inc_rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    changed: list[int] = []
    unchanged: list[int] = []
    if len(clean_rows) != len(inc_rows):
        raise ValueError(f"row count mismatch clean={len(clean_rows)} inc={len(inc_rows)}")
    for i, (c, r) in enumerate(zip(clean_rows, inc_rows)):
        if c.get("text") != r.get("text"):
            changed.append(i)
        else:
            unchanged.append(i)
    return changed, unchanged


def sample_rows(rows: list[dict[str, Any]], indices: list[int], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(indices)
    if len(selected) > n:
        selected = rng.sample(selected, n)
    selected.sort()
    out: list[dict[str, Any]] = []
    for i in selected:
        r = rows[i]
        text = r.get("text", "")
        if not text.strip():
            continue
        out.append({
            "row_index": i,
            "text": text,
            "words": int(r.get("words", 0)),
            "source": str(r.get("source", "")),
            "sha12": text_sha(text),
        })
    return out


def model_ready(run: pathlib.Path, ck: str) -> bool:
    p = run / "hf_model" / ck
    if not p.exists():
        return False
    if not (p / "config.json").exists():
        return False
    if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
        return True
    # Some HF saves use shards with an index file.
    if (p / "model.safetensors.index.json").exists() or (p / "pytorch_model.bin.index.json").exists():
        return True
    return False


def available_checkpoints(run: pathlib.Path, requested: list[str]) -> list[str]:
    return [ck for ck in requested if model_ready(run, ck)]


def deterministic_positions(input_ids, special_ids: set[int], text: str, mask_prob: float) -> list[int]:
    maskable = [i for i in range(input_ids.shape[1]) if int(input_ids[0, i]) not in special_ids]
    if not maskable:
        return []
    n_mask = max(1, int(len(maskable) * mask_prob))
    h = hashlib.sha256((str(BASE_MASK_SEED) + "\n" + text).encode("utf-8", errors="ignore")).hexdigest()
    rng = random.Random(int(h[:16], 16))
    return rng.sample(maskable, min(n_mask, len(maskable)))


def mean_loss(model, tokenizer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    total_loss = 0.0
    total_masked = 0
    n_rows_used = 0
    n_subword_tokens = 0
    with torch.no_grad():
        for r in rows:
            text = r["text"]
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding=False)
            input_ids = enc["input_ids"].clone()
            labels = torch.full_like(input_ids, -100)
            positions = deterministic_positions(input_ids, special_ids, text, MASK_PROB)
            if not positions:
                continue
            for pos in positions:
                labels[0, pos] = input_ids[0, pos]
                input_ids[0, pos] = tokenizer.mask_token_id
            outputs = model(input_ids=input_ids, attention_mask=enc["attention_mask"], labels=labels)
            nm = int((labels != -100).sum().item())
            total_loss += float(outputs.loss.item()) * nm
            total_masked += nm
            n_rows_used += 1
            n_subword_tokens += int(enc["attention_mask"].sum().item())
    return {
        "mean_loss": round(total_loss / total_masked, 6) if total_masked else None,
        "n_rows": n_rows_used,
        "n_tokens_masked": total_masked,
        "n_subword_tokens": n_subword_tokens,
    }


def load_existing_rate_anchors() -> dict[str, Any]:
    anchors: dict[str, Any] = {"source": rel(PERSISTENCE_JSON), "arms": {}, "late_exEntity5": LATE_EXENTITY5_FROM_STEP287}
    if not PERSISTENCE_JSON.exists():
        anchors["missing"] = True
        return anchors
    payload = json.loads(PERSISTENCE_JSON.read_text(encoding="utf-8"))
    idx: dict[tuple[str, str], float] = {}
    for r in payload.get("results", []):
        if r.get("block_type") == "changed" and r.get("mean_loss") is not None:
            idx[(r["arm"], r["checkpoint"])] = float(r["mean_loss"])
    for arm in ["repeat", "view", "breadth", "clean"]:
        vals = {ck: idx.get((arm, ck)) for ck in ["chck_20M", "chck_40M", "chck_60M", "chck_80M", "chck_100M"]}
        def red(a: str, b: str) -> float | None:
            if vals.get(a) is None or vals.get(b) is None:
                return None
            return round(float(vals[a]) - float(vals[b]), 6)
        anchors["arms"][arm] = {
            "changed_loss": vals,
            "loss_reduction_40_to_100": red("chck_40M", "chck_100M"),
            "loss_reduction_60_to_100": red("chck_60M", "chck_100M"),
            "loss_reduction_80_to_100": red("chck_80M", "chck_100M"),
            "late_exEntity5_from_step287": LATE_EXENTITY5_FROM_STEP287.get(arm),
        }
    return anchors


def load_clean_prior_anchors() -> dict[str, Any]:
    out: dict[str, Any] = {"source": rel(CROSSLOSS_JSON), "clean_100M_prior_loss": {}}
    if not CROSSLOSS_JSON.exists():
        out["missing"] = True
        return out
    payload = json.loads(CROSSLOSS_JSON.read_text(encoding="utf-8"))
    for r in payload.get("results", []):
        if r.get("model_arm") == "clean" and r.get("checkpoint") == "chck_100M" and r.get("text_set") in {"repeat_changed", "view_changed", "breadth_changed"}:
            out["clean_100M_prior_loss"][r["text_set"].replace("_changed", "")] = float(r["mean_loss"])
    return out


def interval_reductions(loss_by_ck: dict[str, float]) -> dict[str, float | None]:
    def red(a: str, b: str) -> float | None:
        if a not in loss_by_ck or b not in loss_by_ck:
            return None
        return round(float(loss_by_ck[a]) - float(loss_by_ck[b]), 6)
    return {
        "loss_reduction_40_to_100": red("chck_40M", "chck_100M"),
        "loss_reduction_60_to_100": red("chck_60M", "chck_100M"),
        "loss_reduction_80_to_100": red("chck_80M", "chck_100M"),
        "loss_reduction_40_to_60": red("chck_40M", "chck_60M"),
        "loss_reduction_60_to_80": red("chck_60M", "chck_80M"),
        "loss_reduction_latest20M": None,
    }


def fit_rate_to_downstream(anchors: dict[str, Any]) -> dict[str, Any]:
    pts = []
    for arm in ["repeat", "view", "breadth"]:
        x = anchors.get("arms", {}).get(arm, {}).get("loss_reduction_60_to_100")
        y = LATE_EXENTITY5_FROM_STEP287.get(arm)
        if x is not None and y is not None:
            pts.append((float(x), float(y), arm))
    if len(pts) < 2:
        return {"status": "insufficient"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xm = statistics.mean(xs)
    ym = statistics.mean(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return {"status": "degenerate", "points": pts}
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom
    intercept = ym - slope * xm
    pred_at_repeat = intercept + slope * anchors["arms"]["repeat"]["loss_reduction_60_to_100"]
    return {
        "status": "ok",
        "points": [{"arm": a, "loss_reduction_60_to_100": round(x, 6), "late_exEntity5": y} for x, y, a in pts],
        "slope": round(slope, 6),
        "intercept": round(intercept, 6),
        "repeat_fit_check": round(pred_at_repeat, 6),
        "warning": "three-arm descriptive calibration only; use as a frozen quantitative lens, not as proof of causality",
    }


def predict_from_measurement(clean_prior: dict[str, Any], own_losses: dict[str, float], anchors: dict[str, Any]) -> dict[str, Any]:
    prior_100 = clean_prior.get("chck_100M")
    prior_refs = load_clean_prior_anchors().get("clean_100M_prior_loss", {})
    inc_rates = interval_reductions(own_losses)
    rate_fit = fit_rate_to_downstream(anchors)

    # Distances to known clean-prior anchors.
    prior_position: dict[str, Any] = {}
    if prior_100 is not None and prior_refs:
        dists = {arm: abs(float(prior_100) - float(val)) for arm, val in prior_refs.items()}
        nearest = min(dists, key=dists.get)
        prior_position = {
            "incorpus_clean_prior_100M": round(float(prior_100), 6),
            "reference_clean_prior_100M": {k: round(float(v), 6) for k, v in prior_refs.items()},
            "nearest_reference_by_level": nearest,
            "distance_to_nearest": round(dists[nearest], 6),
            "distance_to_breadth_prior": round(dists.get("breadth", float("nan")), 6),
        }

    rate_60_100 = inc_rates.get("loss_reduction_60_to_100")
    rate_based_numeric = None
    if rate_60_100 is not None and rate_fit.get("status") == "ok":
        rate_based_numeric = round(rate_fit["intercept"] + rate_fit["slope"] * float(rate_60_100), 6)

    # Primary commitment. If late in-corpus rate is not yet available, freeze a
    # conditional test rather than silently substituting a level account.
    repeat_rate = anchors.get("arms", {}).get("repeat", {}).get("loss_reduction_60_to_100")
    breadth_rate = anchors.get("arms", {}).get("breadth", {}).get("loss_reduction_60_to_100")
    view_rate = anchors.get("arms", {}).get("view", {}).get("loss_reduction_60_to_100")
    distinct_floor = None
    if repeat_rate is not None and breadth_rate is not None:
        distinct_floor = round((float(repeat_rate) + float(breadth_rate)) / 2.0, 6)
    view_like_floor = None
    if repeat_rate is not None and view_rate is not None:
        view_like_floor = round((float(repeat_rate) + float(view_rate)) / 2.0, 6)

    if rate_60_100 is None:
        primary = (
            "conditional_rate_commitment: official in-corpus broad late gain should be small/repeat-like if the same-script "
            "in-corpus admitted-block loss reduction from 60M to 100M is at or below the repeat-distinct midpoint; it should "
            "approach the view/breadth late-retention regime only if that 60M->100M reduction remains distinct-like. The current "
            "clean-prior level is recorded now and must not be used alone as the mechanism."
        )
        predicted_exentity5_late = None
    else:
        if distinct_floor is not None and float(rate_60_100) <= distinct_floor:
            primary = (
                "repeat_like_rate_prediction: in-corpus adult-prose rows collapse early; predict little broad late ex-Entity gain "
                "over matched clean and a clearly smaller late effect than MAX view/breadth."
            )
        elif view_like_floor is not None and float(rate_60_100) >= view_like_floor:
            primary = (
                "view_breadth_like_rate_prediction: in-corpus adult-prose rows remain an active late error source; predict a "
                "positive broad late ex-Entity gain comparable to the persistent view/breadth regime."
            )
        else:
            primary = (
                "intermediate_rate_prediction: in-corpus late gain should be positive but below the persistent view/breadth regime, "
                "with official scores expected to fall between repeat and view/breadth."
            )
        predicted_exentity5_late = rate_based_numeric

    return {
        "status": "INCORPUS_RATE_PRE_SCORE_COMMITMENT",
        "created_utc": now(),
        "made_before_reading_step293_decisive_official_scores_in_this_step": True,
        "primary_commitment": primary,
        "predicted_late_exEntity5_from_rate_fit_if_available": predicted_exentity5_late,
        "incorpus_clean_prior_position": prior_position,
        "incorpus_owner_loss_by_checkpoint": {k: round(float(v), 6) for k, v in own_losses.items()},
        "incorpus_owner_rate_metrics": inc_rates,
        "rate_classification_thresholds": {
            "repeat_rate_60_to_100": repeat_rate,
            "breadth_rate_60_to_100": breadth_rate,
            "view_rate_60_to_100": view_rate,
            "repeat_distinct_midpoint": distinct_floor,
            "repeat_view_midpoint": view_like_floor,
        },
        "rate_to_downstream_descriptive_fit": rate_fit,
        "interpretation_rule": {
            "static_profile_proximity": "adult-prose official scores high even if in-corpus admitted-block loss collapsed early; this would identify distribution target proximity more than active late error",
            "out_of_corpus_novelty": "in-corpus scores flat while FineWeb view/breadth stay positive; this would favor novelty/out-of-corpus content not official adult prose alone",
            "rate_account": "official late movement tracks in-corpus 60M->100M admitted-block loss reduction; rows that stay active late support transfer, rows exhausted early do not",
        },
    }


def write_markdown(payload: dict[str, Any], losses: list[dict[str, Any]], out_md: pathlib.Path) -> None:
    commit = payload["prediction_commitment"]
    lines: list[str] = []
    lines.append("# research in-corpus adult-prose rate prediction")
    lines.append("")
    lines.append("This file was written before inspecting any new research decisive official score output. It is a CPU-only forward MLM loss readout and prediction commitment for the final in-corpus adult-prose cell.")
    lines.append("")
    lines.append("## Measured in-corpus block")
    lines.append(f"- Changed rows: {payload['changed_rows']['count']} / {payload['pool_rows']} rows")
    lines.append(f"- Changed words: {payload['changed_rows']['words']}")
    lines.append(f"- Changed sources: {payload['changed_rows']['source_words']}")
    lines.append(f"- Sampled changed rows: {payload['sample_sizes']['changed']} ; sampled unchanged rows: {payload['sample_sizes']['unchanged']}")
    lines.append("")
    lines.append("## Loss table")
    lines.append("")
    lines.append("| model_arm | checkpoint | block | mean_loss | masked tokens |")
    lines.append("|---|---:|---|---:|---:|")
    for r in losses:
        lines.append(f"| {r['model_arm']} | {r['checkpoint']} | {r['block_type']} | {r['mean_loss']} | {r['n_tokens_masked']} |")
    lines.append("")
    lines.append("## Pre-score commitment")
    lines.append("")
    lines.append(commit["primary_commitment"])
    lines.append("")
    lines.append("### Clean-prior position")
    pos = commit.get("incorpus_clean_prior_position", {})
    for k, v in pos.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### In-corpus owner rate")
    for k, v in commit.get("incorpus_owner_rate_metrics", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("### Existing MAX-arm rate anchors")
    for arm, rec in payload.get("existing_rate_anchors", {}).get("arms", {}).items():
        if arm in {"repeat", "view", "breadth"}:
            lines.append(f"- {arm}: changed_loss_60M={rec['changed_loss'].get('chck_60M')}, changed_loss_100M={rec['changed_loss'].get('chck_100M')}, loss_reduction_60_to_100={rec.get('loss_reduction_60_to_100')}, late_exEntity5={rec.get('late_exEntity5_from_step287')}")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- JSON: `{rel(OUT_DIR / 'incorpus_rate_prediction.json')}`")
    lines.append(f"- CSV: `{rel(OUT_DIR / 'incorpus_rate_losses.csv')}`")
    lines.append(f"- Commitment JSON: `{rel(OUT_DIR / 'prediction_commitment.json')}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for the loss/readout files")
    ap.add_argument("--n", type=int, default=300, help="sample size for changed and unchanged row sets")
    ap.add_argument("--clean-checkpoints", nargs="*", default=DEFAULT_CLEAN_CKS)
    ap.add_argument("--incorpus-checkpoints", nargs="*", default=ALL_CKS)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--skip-incorpus-model", action="store_true")
    args = ap.parse_args()

    OUT_DIR = pathlib.Path(args.out_dir)
    if not OUT_DIR.is_absolute():
        OUT_DIR = ROOT / OUT_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for p in [TOKENIZER_DIR, CLEAN_POOL, INCORPUS_POOL, CLEAN_RUN]:
        if not p.exists():
            missing.append(rel(p))
    clean_cks = available_checkpoints(CLEAN_RUN, args.clean_checkpoints)
    inc_cks = [] if args.skip_incorpus_model else available_checkpoints(INCORPUS_RUN, args.incorpus_checkpoints)
    for ck in args.clean_checkpoints:
        if ck not in clean_cks:
            missing.append(f"clean_model:{ck}")

    clean_rows = load_jsonl(CLEAN_POOL) if CLEAN_POOL.exists() else []
    inc_rows = load_jsonl(INCORPUS_POOL) if INCORPUS_POOL.exists() else []
    changed_idx: list[int] = []
    unchanged_idx: list[int] = []
    if clean_rows and inc_rows:
        changed_idx, unchanged_idx = detect_changed_rows(clean_rows, inc_rows)

    plan = {
        "status": "INCORPUS_RATE_PLAN",
        "created_utc": now(),
        "out_dir": rel(OUT_DIR),
        "cpu_only": True,
        "no_training_no_official_eval_no_upload_no_leaderboard": True,
        "clean_checkpoints_requested": args.clean_checkpoints,
        "clean_checkpoints_available": clean_cks,
        "incorpus_checkpoints_available": inc_cks,
        "n_sample": args.n,
        "changed_rows_detected": len(changed_idx),
        "unchanged_rows_detected": len(unchanged_idx),
        "missing": missing,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if missing:
        raise SystemExit("Missing required clean inputs: " + "; ".join(missing[:20]))
    if not changed_idx:
        raise SystemExit("No changed in-corpus rows detected")

    changed_new = sample_rows(inc_rows, changed_idx, args.n, seed=29401)
    unchanged_new = sample_rows(inc_rows, unchanged_idx, args.n, seed=29402)
    changed_old = sample_rows(clean_rows, changed_idx, args.n, seed=29401)  # displaced original rows at the same positions

    source_words: dict[str, int] = {}
    for i in changed_idx:
        src = normalize_src(str(inc_rows[i].get("source", "")))
        source_words[src] = source_words.get(src, 0) + int(inc_rows[i].get("words", 0))

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    import torch
    try:
        torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    losses: list[dict[str, Any]] = []
    model_jobs: list[tuple[str, pathlib.Path, list[str]]] = [("clean", CLEAN_RUN, clean_cks)]
    if inc_cks:
        model_jobs.append(("incorpus", INCORPUS_RUN, inc_cks))

    block_rows = {
        "incorpus_admitted_changed": changed_new,
        "incorpus_unchanged": unchanged_new,
        "clean_displaced_same_positions": changed_old,
    }

    for arm, run, cks in model_jobs:
        for ck in cks:
            mp = run / "hf_model" / ck
            print(f"\nLoading {arm} {ck}: {rel(mp)}", flush=True)
            t0 = time.time()
            try:
                model = AutoModelForMaskedLM.from_pretrained(str(mp))
            except Exception as e:
                print(f"  LOAD_FAIL {arm} {ck}: {type(e).__name__}: {e}", flush=True)
                losses.append({"model_arm": arm, "checkpoint": ck, "block_type": "LOAD_FAIL", "error": repr(e)})
                continue
            model.eval()
            print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
            for block, rows in block_rows.items():
                t1 = time.time()
                info = mean_loss(model, tokenizer, rows)
                rec = {
                    "model_arm": arm,
                    "checkpoint": ck,
                    "block_type": block,
                    **info,
                    "eval_time_sec": round(time.time() - t1, 1),
                }
                losses.append(rec)
                print(f"  {block:<32} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
            del model
            import gc
            gc.collect()

    # Index relevant clean prior and in-corpus owner admitted-block losses.
    clean_prior_changed: dict[str, float] = {}
    inc_owner_changed: dict[str, float] = {}
    inc_owner_unchanged: dict[str, float] = {}
    for r in losses:
        if r.get("mean_loss") is None:
            continue
        if r.get("model_arm") == "clean" and r.get("block_type") == "incorpus_admitted_changed":
            clean_prior_changed[r["checkpoint"]] = float(r["mean_loss"])
        if r.get("model_arm") == "incorpus" and r.get("block_type") == "incorpus_admitted_changed":
            inc_owner_changed[r["checkpoint"]] = float(r["mean_loss"])
        if r.get("model_arm") == "incorpus" and r.get("block_type") == "incorpus_unchanged":
            inc_owner_unchanged[r["checkpoint"]] = float(r["mean_loss"])

    existing_anchors = load_existing_rate_anchors()
    prediction_commitment = predict_from_measurement(clean_prior_changed, inc_owner_changed, existing_anchors)

    meta = json.loads(INCORPUS_META.read_text(encoding="utf-8")) if INCORPUS_META.exists() else {}
    output = {
        "status": "INCORPUS_RATE_DONE",
        "finished_utc": now(),
        "purpose": "pre-official-score in-corpus adult-prose clean-prior and rate-account commitment",
        "parameters": {
            "n_sample": args.n,
            "mask_prob": MASK_PROB,
            "base_mask_seed": BASE_MASK_SEED,
            "max_seq_len": MAX_SEQ_LEN,
            "clean_checkpoints": clean_cks,
            "incorpus_checkpoints": inc_cks,
            "cpu_only": True,
            "no_training_no_official_eval_no_upload_no_leaderboard": True,
        },
        "pool_rows": len(inc_rows),
        "changed_rows": {
            "count": len(changed_idx),
            "words": sum(int(inc_rows[i].get("words", 0)) for i in changed_idx),
            "source_words": source_words,
            "metadata_changed_rows": meta.get("changed_positions"),
            "metadata_changed_words": meta.get("changed_words"),
        },
        "sample_sizes": {"changed": len(changed_new), "unchanged": len(unchanged_new), "displaced_old": len(changed_old)},
        "samples": {
            "changed": [{"row_index": r["row_index"], "words": r["words"], "source": r["source"], "sha12": r["sha12"]} for r in changed_new],
            "unchanged": [{"row_index": r["row_index"], "words": r["words"], "source": r["source"], "sha12": r["sha12"]} for r in unchanged_new[:50]],
        },
        "losses": losses,
        "clean_prior_loss_on_incorpus_admitted": {k: round(v, 6) for k, v in clean_prior_changed.items()},
        "incorpus_owner_loss_on_admitted": {k: round(v, 6) for k, v in inc_owner_changed.items()},
        "incorpus_owner_loss_on_unchanged": {k: round(v, 6) for k, v in inc_owner_unchanged.items()},
        "existing_rate_anchors": existing_anchors,
        "existing_clean_prior_anchors": load_clean_prior_anchors(),
        "prediction_commitment": prediction_commitment,
    }

    out_json = OUT_DIR / "incorpus_rate_prediction.json"
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = OUT_DIR / "incorpus_rate_losses.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fields = ["model_arm", "checkpoint", "block_type", "mean_loss", "n_rows", "n_tokens_masked", "n_subword_tokens", "eval_time_sec", "error"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in losses:
            w.writerow({k: r.get(k) for k in fields})

    commit_path = OUT_DIR / "prediction_commitment.json"
    commit_path.write_text(json.dumps(prediction_commitment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_path = OUT_DIR / "incorpus_rate_prediction.md"
    write_markdown(output, losses, md_path)

    print(f"\nSaved {rel(out_json)}", flush=True)
    print(f"Saved {rel(csv_path)}", flush=True)
    print(f"Saved {rel(commit_path)}", flush=True)
    print(f"Saved {rel(md_path)}", flush=True)
    print(json.dumps({
        "status": output["status"],
        "clean_prior_loss_on_incorpus_admitted": output["clean_prior_loss_on_incorpus_admitted"],
        "incorpus_owner_loss_on_admitted": output["incorpus_owner_loss_on_admitted"],
        "prediction_commitment": prediction_commitment,
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
