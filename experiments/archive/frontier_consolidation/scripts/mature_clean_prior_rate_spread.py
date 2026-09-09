#!/usr/bin/env python3
"""research: mature-window clean-prior active-error spread.

Scientific role
---------------
The fixed-budget active-error mechanism should be an ex-ante quantity, not an
own-trained-arm description.  research already measured the clean DeBERTa model
on the admitted MAX view/repeat/breadth blocks at 80M and 100M; this script
puts row/mask spread on exactly that mature-window quantity:

    r_clean(B; 80->100M) = L_clean_80M(B) - L_clean_100M(B)

for view_changed, breadth_changed, repeat_changed, and the matched clean
sacrificed rows.  This is CPU-only forward MLM loss: no training, no official
benchmark evaluation, no upload, no leaderboard action.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_DIR = WS / "data" / "mature_clean_prior_rate_spread"
TOKENIZER_DIR = WS / "data" / "compliant_tokenizer"
CLEAN_RUN = WS / "training" / "runs" / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022"

POOL_FILES = {
    "view_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_10M.jsonl",
    "repeat_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_10M.jsonl",
    "breadth_changed": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_10M.jsonl",
    "clean_displaced": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_10M.jsonl",
}
META_FILES = {
    "view_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_view_dose2p64x_changed_block_rows_meta.jsonl",
    "repeat_changed": WS / "data" / "dose_2p64x_rowholdout_pools" / "compact_repeat_dose2p64x_changed_block_rows_meta.jsonl",
    "breadth_changed": WS / "data" / "dose_2p64x_breadth_rowholdout_pools" / "compact_breadth_dose2p64x_changed_block_rows_meta.jsonl",
    "clean_displaced": WS / "data" / "dose_2p64x_rowholdout_pools" / "cleanqwen_lengthmatched_dose2p64x_changed_block_rows_meta.jsonl",
}

CHECKPOINTS = ["chck_80M", "chck_100M"]
TEXT_SETS = ["view_changed", "breadth_changed", "repeat_changed", "clean_displaced"]
REPLICATE_SEEDS = [29501, 29502, 29503]
MASK_SEEDS = [295101, 295102, 295103]
MASK_PROB = 0.15
MAX_SEQ_LEN = 256

# Existing downstream late-retention anchors from research/287; held here only for
# alignment reading, not refit.
LATE_DOWNSTREAM_EXENTITY5 = {
    "view_changed": 0.3853,
    "breadth_changed": 0.3350,
    "repeat_changed": 0.0343,
}
SINGLE_SAMPLE_RATE_80_100 = {
    "view_changed": 3.330201 - 3.265860,
    "breadth_changed": 5.309294 - 5.266488,
    "repeat_changed": 2.983797 - 2.949448,
    "clean_displaced": 3.026670 - 2.988545,
}
SINGLE_SAMPLE_RATE_40_100 = {
    "view_changed": 4.303809 - 3.265860,
    "breadth_changed": 5.697204 - 5.266488,
    "repeat_changed": 3.851267 - 2.949448,
    "clean_displaced": 3.441430 - 2.988545,
}
INCORPUS_CLEAN_PRIOR = {
    "loss_80M": 3.541737,
    "loss_100M": 3.495350,
    "rate_80_to_100": 3.541737 - 3.495350,
    "distinct_words": 442987,
    "source_words": {"gutenberg": 272048, "simple_wiki": 170939},
}


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
    s = str(source or "unknown")
    return s.split("::", 1)[1] if "::" in s else s


def nwords(row: dict[str, Any]) -> int:
    if row.get("words") is not None:
        return int(row["words"])
    return len(str(row.get("text", "")).split())


def load_changed_indices(meta_path: pathlib.Path) -> list[int]:
    idx: list[int] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx.append(int(json.loads(line)["row_index"]))
    return sorted(set(idx))


def select_records(pool_path: pathlib.Path, meta_path: pathlib.Path, label: str) -> list[dict[str, Any]]:
    rows = load_jsonl(pool_path)
    idx = load_changed_indices(meta_path)
    out: list[dict[str, Any]] = []
    for i in idx:
        row = rows[i]
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        out.append({
            "text_set": label,
            "row_index": int(i),
            "text": text,
            "words": nwords(row),
            "source": normalize_source(row.get("source", row.get("source_name", "unknown"))),
            "sha12": sha12(text),
        })
    return out


def source_word_summary(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[r["source"]] = out.get(r["source"], 0) + int(r["words"])
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def sample_records(records: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected = list(records) if len(records) <= n else rng.sample(records, n)
    return sorted(selected, key=lambda r: (int(r["row_index"]), r["sha12"]))


def model_ready(ck: str) -> bool:
    p = CLEAN_RUN / "hf_model" / ck
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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rank_order(values: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))]


def write_outputs(payload: dict[str, Any], out_dir: pathlib.Path) -> None:
    summary_json = out_dir / "mature_clean_prior_rate_spread.json"
    measurements_csv = out_dir / "loss_measurements.csv"
    rates_csv = out_dir / "mature_rate_rows.csv"
    commitment_json = out_dir / "incorpus_mature_rate_commitment.json"
    summary_md = out_dir / "mature_clean_prior_rate_spread.md"

    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    commitment_json.write_text(json.dumps(payload["incorpus_commitment"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with measurements_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["checkpoint", "text_set", "replicate", "mask_seed", "row_seed", "mean_loss", "row_loss_mean_unweighted", "row_loss_sd_unweighted", "n_rows", "n_tokens_masked", "n_subword_tokens", "eval_time_sec"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in payload["measurements"]:
            w.writerow({k: r.get(k) for k in fieldnames})

    with rates_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["text_set", "replicate", "loss_80M", "loss_100M", "rate_80_to_100", "mass_distinct_or_changed_words", "mass_rate_product"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in payload["rate_rows"]:
            w.writerow({k: r.get(k) for k in fieldnames})

    lines: list[str] = []
    lines.append("# research mature-window clean-prior active-error spread")
    lines.append("")
    lines.append("CPU-only forward MLM measurement using the clean DeBERTa MAX-geometry model at `chck_80M` and `chck_100M`. The script measures an ex-ante quantity on text the clean model did not train on for the intervention admitted blocks: `L_clean(80M) - L_clean(100M)`. No training, official benchmark scoring, upload, or leaderboard action is performed.")
    lines.append("")
    lines.append("## Single-sample anchor from research")
    for ts, val in payload["single_sample_rate_80_to_100"].items():
        lines.append(f"- {ts}: {val:.6f}")
    lines.append(f"- In the broader 40M→100M window, repeat exceeds breadth (`repeat={payload['single_sample_rate_40_to_100']['repeat_changed']:.6f}`, `breadth={payload['single_sample_rate_40_to_100']['breadth_changed']:.6f}`), so the mature terminal window is the mechanism-bearing coordinate here.")
    lines.append("")
    lines.append("## Three-replicate mature rates")
    for ts, summ in payload["rate_summary"].items():
        lines.append(f"- {ts}: {summ}")
    lines.append("")
    lines.append("## In-corpus adult-prose mature-window commitment")
    inc = payload["incorpus_commitment"]
    for k in ["rate_80_to_100", "distinct_words", "mass_rate_product", "position_vs_reference", "prediction"]:
        lines.append(f"- {k}: {inc.get(k)}")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- summary_json: `{rel(summary_json)}`")
    lines.append(f"- measurements_csv: `{rel(measurements_csv)}`")
    lines.append(f"- rates_csv: `{rel(rates_csv)}`")
    lines.append(f"- commitment_json: `{rel(commitment_json)}`")
    lines.append(f"- summary_md: `{rel(summary_md)}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload["files"] = {
        "summary_json": rel(summary_json),
        "measurements_csv": rel(measurements_csv),
        "rates_csv": rel(rates_csv),
        "commitment_json": rel(commitment_json),
        "summary_md": rel(summary_md),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    for p in [TOKENIZER_DIR, CLEAN_RUN]:
        if not p.exists():
            missing.append(rel(p))
    for ck in CHECKPOINTS:
        if not model_ready(ck):
            missing.append(f"clean_model:{ck}")
    for ts in TEXT_SETS:
        if not POOL_FILES[ts].exists():
            missing.append(rel(POOL_FILES[ts]))
        if not META_FILES[ts].exists():
            missing.append(rel(META_FILES[ts]))

    all_records: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    if not missing:
        for ts in TEXT_SETS:
            recs = select_records(POOL_FILES[ts], META_FILES[ts], ts)
            all_records[ts] = recs
            audits[ts] = {
                "pool": rel(POOL_FILES[ts]),
                "meta": rel(META_FILES[ts]),
                "n_changed_rows": len(recs),
                "changed_words": sum(int(r["words"]) for r in recs),
                "source_words": source_word_summary(recs),
            }

    # Mirror research row/mask design.  For the three reference intervention sets,
    # use the exact research offsets; clean_displaced gets the same replicate seeds
    # with a separate fixed offset.
    sample_sets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    sample_meta: dict[tuple[str, int], dict[str, Any]] = {}
    for rep_i, (row_seed, mask_seed) in enumerate(zip(REPLICATE_SEEDS, MASK_SEEDS), start=1):
        for ts in TEXT_SETS:
            if ts == "repeat_changed":
                ts_index = 0
                seed = row_seed + 1000 + 17 * rep_i + 101 * ts_index
            elif ts == "breadth_changed":
                ts_index = 1
                seed = row_seed + 1000 + 17 * rep_i + 101 * ts_index
            elif ts == "view_changed":
                ts_index = 2
                seed = row_seed + 1000 + 17 * rep_i + 101 * ts_index
            else:
                seed = row_seed + 2000 + 17 * rep_i
            if not missing:
                sample_sets[(ts, rep_i)] = sample_records(all_records[ts], args.n, seed)
            sample_meta[(ts, rep_i)] = {"row_seed": seed, "mask_seed": mask_seed}

    plan = {
        "status": "MATURE_CLEAN_PRIOR_RATE_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "cpu_only": True,
        "no_training_no_official_eval_no_upload_no_leaderboard": True,
        "n_per_replicate": args.n,
        "checkpoints": CHECKPOINTS,
        "text_sets": TEXT_SETS,
        "replicate_seeds": REPLICATE_SEEDS,
        "mask_seeds": MASK_SEEDS,
        "n_model_loads": len(CHECKPOINTS),
        "n_loss_measurements": len(CHECKPOINTS) * len(TEXT_SETS) * len(REPLICATE_SEEDS),
        "audits_short": audits,
        "missing": missing,
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    if missing:
        raise SystemExit("Missing inputs: " + "; ".join(missing))

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    try:
        torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    measurements: list[dict[str, Any]] = []
    for ck_i, ck in enumerate(CHECKPOINTS, start=1):
        model_path = CLEAN_RUN / "hf_model" / ck
        print(f"\n[{ck_i}/{len(CHECKPOINTS)}] loading clean {ck}: {rel(model_path)}", flush=True)
        t0 = time.time()
        model = AutoModelForMaskedLM.from_pretrained(str(model_path))
        model.eval()
        print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
            for ts in TEXT_SETS:
                meta = sample_meta[(ts, rep_i)]
                t1 = time.time()
                info = mean_loss(model, tokenizer, sample_sets[(ts, rep_i)], int(meta["mask_seed"]))
                rec = {
                    "checkpoint": ck,
                    "text_set": ts,
                    "replicate": rep_i,
                    "mask_seed": int(meta["mask_seed"]),
                    "row_seed": int(meta["row_seed"]),
                    **info,
                    "eval_time_sec": round(time.time() - t1, 1),
                }
                measurements.append(rec)
                print(f"  rep{rep_i} {ts:<16} loss={rec['mean_loss']} masks={rec['n_tokens_masked']} time={rec['eval_time_sec']}s", flush=True)
        del model
        import gc
        gc.collect()

    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for rec in measurements:
        if rec.get("mean_loss") is None:
            continue
        by_key.setdefault((rec["text_set"], int(rec["replicate"])), {})[rec["checkpoint"]] = float(rec["mean_loss"])

    rate_rows: list[dict[str, Any]] = []
    for ts in TEXT_SETS:
        mass = int(audits[ts]["changed_words"])
        for rep_i in range(1, len(REPLICATE_SEEDS) + 1):
            losses = by_key.get((ts, rep_i), {})
            rate = None
            if "chck_80M" in losses and "chck_100M" in losses:
                rate = losses["chck_80M"] - losses["chck_100M"]
            rate_rows.append({
                "text_set": ts,
                "replicate": rep_i,
                "loss_80M": round(losses.get("chck_80M"), 6) if "chck_80M" in losses else None,
                "loss_100M": round(losses.get("chck_100M"), 6) if "chck_100M" in losses else None,
                "rate_80_to_100": round(rate, 6) if rate is not None else None,
                "mass_distinct_or_changed_words": mass,
                "mass_rate_product": round(mass * rate, 3) if rate is not None else None,
            })

    rate_summary = {
        ts: summarize([r["rate_80_to_100"] for r in rate_rows if r["text_set"] == ts and r.get("rate_80_to_100") is not None])
        for ts in TEXT_SETS
    }
    mass_rate_summary = {
        ts: summarize([r["mass_rate_product"] for r in rate_rows if r["text_set"] == ts and r.get("mass_rate_product") is not None])
        for ts in TEXT_SETS
    }

    # Compare reference ranking against late downstream anchors over the three intervention sets.
    intervention_sets = ["view_changed", "breadth_changed", "repeat_changed"]
    rate_means = {ts: float(rate_summary[ts]["mean"]) for ts in intervention_sets}
    downstream = {ts: LATE_DOWNSTREAM_EXENTITY5[ts] for ts in intervention_sets}
    rate_order = rank_order(rate_means)
    downstream_order = rank_order(downstream)
    xs = [rate_means[ts] for ts in intervention_sets]
    ys = [downstream[ts] for ts in intervention_sets]

    clean_drift_mean = float(rate_summary["clean_displaced"]["mean"])
    inc_rate = float(INCORPUS_CLEAN_PRIOR["rate_80_to_100"])
    inc_mass_product = inc_rate * int(INCORPUS_CLEAN_PRIOR["distinct_words"])
    breadth_mean = float(rate_summary["breadth_changed"]["mean"])
    repeat_mean = float(rate_summary["repeat_changed"]["mean"])
    view_mean = float(rate_summary["view_changed"]["mean"])
    if inc_rate > clean_drift_mean and inc_rate >= breadth_mean:
        position = "above clean-displaced mature drift and at/above breadth-scale clean-prior mature rate"
        pred = "predict non-flat persistence for in-corpus adult prose; if downstream is flat, active error alone is insufficient"
    elif inc_rate > clean_drift_mean and inc_rate > repeat_mean:
        position = "above clean-displaced mature drift and repeat-scale rate, below breadth/view scale"
        pred = "predict weak but non-flat persistence, weaker than MAX view/breadth"
    else:
        position = "at or below generic clean-displaced mature drift"
        pred = "predict flat or exhausted downstream value"

    payload: dict[str, Any] = {
        "status": "MATURE_CLEAN_PRIOR_RATE_DONE",
        "finished_utc": now(),
        "purpose": "pre-score mature-window clean-prior active-error calibration for fixed-budget substitution principle",
        "parameters": {
            "n_per_replicate": args.n,
            "checkpoints": CHECKPOINTS,
            "text_sets": TEXT_SETS,
            "replicate_seeds": REPLICATE_SEEDS,
            "mask_seeds": MASK_SEEDS,
            "mask_prob": MASK_PROB,
            "max_seq_len": MAX_SEQ_LEN,
            "cpu_only": True,
            "no_training_no_official_eval_no_upload_no_leaderboard": True,
        },
        "audits": audits,
        "sample_preview": {
            f"{ts}__rep{rep_i}": {
                "row_seed": sample_meta[(ts, rep_i)]["row_seed"],
                "mask_seed": sample_meta[(ts, rep_i)]["mask_seed"],
                "first20": [{k: r[k] for k in ["row_index", "source", "words", "sha12"]} for r in sample_sets[(ts, rep_i)][:20]],
            }
            for ts in TEXT_SETS for rep_i in range(1, len(REPLICATE_SEEDS) + 1)
        },
        "measurements": measurements,
        "rate_rows": rate_rows,
        "rate_summary": rate_summary,
        "mass_rate_summary": mass_rate_summary,
        "single_sample_rate_80_to_100": {k: round(v, 6) for k, v in SINGLE_SAMPLE_RATE_80_100.items()},
        "single_sample_rate_40_to_100": {k: round(v, 6) for k, v in SINGLE_SAMPLE_RATE_40_100.items()},
        "downstream_alignment": {
            "late_downstream_exEntity5_from_step275_287": downstream,
            "clean_prior_rate_means_80_to_100": rate_means,
            "rate_order": rate_order,
            "downstream_order": downstream_order,
            "pearson_three_points": round(pearson(xs, ys), 6) if pearson(xs, ys) is not None else None,
            "interpretation": "Mature clean-prior rates reproduce the late retention ordering for DeBERTa MAX view/breadth/repeat; the broader 40M->100M clean-prior window does not, so the terminal-window condition is load-bearing.",
        },
        "incorpus_commitment": {
            "status": "INCORPUS_MATURE_RATE_PRE_SCORE_COMMITMENT",
            "source": "experiments/archive/frontier_consolidation/data/incorpus_rate_prediction/incorpus_rate_prediction.json",
            "loss_80M": INCORPUS_CLEAN_PRIOR["loss_80M"],
            "loss_100M": INCORPUS_CLEAN_PRIOR["loss_100M"],
            "rate_80_to_100": round(inc_rate, 6),
            "distinct_words": INCORPUS_CLEAN_PRIOR["distinct_words"],
            "source_words": INCORPUS_CLEAN_PRIOR["source_words"],
            "mass_rate_product": round(inc_mass_product, 3),
            "reference_rate_summary_80_to_100": rate_summary,
            "position_vs_reference": position,
            "prediction": pred,
            "frozen_before_official_incorpus_scores_used_in_this_step": True,
        },
    }
    write_outputs(payload, out_dir)
    print(json.dumps({
        "status": payload["status"],
        "rate_summary": payload["rate_summary"],
        "downstream_alignment": payload["downstream_alignment"],
        "incorpus_commitment": payload["incorpus_commitment"],
        "files": payload["files"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
