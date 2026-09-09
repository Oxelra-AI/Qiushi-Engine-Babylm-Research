#!/usr/bin/env python3
"""research score the primary CHILDES variation-set probe on DeBERTa arms.

The probe contains paired intact/replaced contexts for each target.  Positive delta
means the adjacent natural utterance improves target prediction relative to a
length-matched context from another transcript.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import random
import statistics
import time
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_childes_variation_probe.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
OLD_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
NEW_RUNS = WS / "training/runs"
PROBE_DIR = WS / "analysis/childes_variation_set_probe"
PROBE_RECORDS = PROBE_DIR / "probe_records.jsonl"
PROBE_STATS = PROBE_DIR / "probe_stats.json"
VALIDATION = PROBE_DIR / "validation_results.json"
OUT = WS / "data/childes_variation_probe_score"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/childes_variation_probe_score.md')
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]

ARM_CONFIGS: dict[str, pathlib.Path] = {
    "D_C_43022": OLD_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": OLD_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43022": OLD_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_C_43122": OLD_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_R_43122": OLD_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_V_43122": OLD_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122",
    "D_C_43222": NEW_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "D_R_43222": NEW_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222",
    "D_V_43222": NEW_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222",
}


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(xs)) if xs else float("nan")


def sd(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.stdev(xs)) if len(xs) > 1 else float("nan")


def se(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def parse_arm(arm: str) -> tuple[str, int]:
    parts = arm.split("_")
    return parts[1], int(parts[2])


def load_records() -> list[dict[str, Any]]:
    recs = list(read_jsonl(PROBE_RECORDS))
    for r in recs:
        if "mask_position" not in r or "target_token_id" not in r:
            raise RuntimeError(f"bad probe record {r.get('record_id')}")
        if len([x for x in r["input_ids"] if int(x) == 4]) != 1:
            raise RuntimeError(f"expected one mask in {r.get('record_id')}")
    return recs


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for st in range(0, len(records), batch_size):
        batch = records[st:st + batch_size]
        maxlen = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(r.get("attention_mask", [1] * L), dtype=torch.long)
        ids = ids.to(device); att = att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            pos = int(r["mask_position"]); target = int(r["target_token_id"])
            lp = float(logp[i, pos, target].detach().cpu())
            rows.append({
                "record_id": r["record_id"],
                "pair_id": r["pair_id"],
                "condition": r["condition"],
                "overlap_bin": r["overlap_bin"],
                "overlap_class": r["overlap_class"],
                "target_occurrence_index": int(r["target_occurrence_index"]),
                "target_token_id": target,
                "target_word": r.get("target_word", ""),
                "mask_position": pos,
                "utt1_token_len": int(r.get("utt1_token_len", -1)),
                "utt2_token_len": int(r.get("utt2_token_len", -1)),
                "seq_len": int(r.get("seq_len", len(r["input_ids"]))),
                "transcript_id": r.get("transcript_id", ""),
                "logp": lp,
                "nll": -lp,
            })
    return rows


def aggregate(scored: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(scored)
    key = ["arm", "role", "seed", "checkpoint", "pair_id", "target_occurrence_index", "target_token_id", "overlap_bin", "overlap_class", "target_word", "transcript_id"]
    pivot = df.pivot_table(index=key, columns="condition", values="nll", aggfunc="first").reset_index()
    if {"intact", "replaced"} - set(pivot.columns):
        raise RuntimeError("missing intact/replaced condition after scoring")
    pivot["delta"] = pivot["replaced"] - pivot["intact"]
    target_delta = pivot.rename(columns={"intact": "intact_nll", "replaced": "replaced_nll"})

    by_ck = (
        target_delta.groupby(["role", "seed", "checkpoint", "overlap_bin", "overlap_class"], dropna=False)
        .agg(n_targets=("delta", "count"), n_pairs=("pair_id", "nunique"), mean_delta=("delta", "mean"), se_delta=("delta", lambda x: se(list(x))))
        .reset_index()
    )
    pair_ck = (
        target_delta.groupby(["role", "seed", "checkpoint", "pair_id", "overlap_bin", "overlap_class", "transcript_id"], dropna=False)
        .agg(n_targets=("delta", "count"), delta=("delta", "mean"), intact_nll=("intact_nll", "mean"), replaced_nll=("replaced_nll", "mean"))
        .reset_index()
    )
    pair_late = (
        pair_ck.groupby(["role", "seed", "pair_id", "overlap_bin", "overlap_class", "transcript_id"], dropna=False)
        .agg(n_checkpoints=("checkpoint", "nunique"), n_targets=("n_targets", "mean"), delta=("delta", "mean"), intact_nll=("intact_nll", "mean"), replaced_nll=("replaced_nll", "mean"))
        .reset_index()
    )
    pair_late = pair_late[pair_late["n_checkpoints"] == len(CKPTS)].copy()
    late = (
        pair_late.groupby(["role", "seed", "overlap_bin", "overlap_class"], dropna=False)
        .agg(n_pairs=("pair_id", "nunique"), mean_delta=("delta", "mean"), median_delta=("delta", "median"), se_pair_delta=("delta", lambda x: se(list(x))), frac_positive=("delta", lambda x: float((pd.Series(x) > 0).mean())))
        .reset_index()
    )

    contrasts: list[dict[str, Any]] = []
    for (seed, obin, ocls), g in pair_late.groupby(["seed", "overlap_bin", "overlap_class"], dropna=False):
        for a, b in [("V", "C"), ("R", "C"), ("V", "R")]:
            aa = g[g["role"] == a][["pair_id", "delta"]].rename(columns={"delta": "delta_a"})
            bb = g[g["role"] == b][["pair_id", "delta"]].rename(columns={"delta": "delta_b"})
            j = aa.merge(bb, on="pair_id", how="inner")
            if j.empty:
                continue
            d = j["delta_a"] - j["delta_b"]
            contrasts.append({
                "seed": int(seed),
                "overlap_bin": obin,
                "overlap_class": ocls,
                "contrast": f"{a}minus{b}",
                "n_pairs": int(len(j)),
                "mean_delta_diff": float(d.mean()),
                "median_delta_diff": float(d.median()),
                "se_pair_delta_diff": float(d.std(ddof=0) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
                "frac_positive": float((d > 0).mean()),
            })
    con = pd.DataFrame(contrasts)
    return df, target_delta, by_ck, pair_late, late, con


def add_all_groups(pair_late: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Add class=ALL within each bin and bin=ALL within each class from pair-level deltas.
    frames = [pair_late]
    all_class = (
        pair_late.groupby(["role", "seed", "pair_id", "overlap_bin", "transcript_id"], dropna=False)
        .agg(n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), delta=("delta", "mean"), intact_nll=("intact_nll", "mean"), replaced_nll=("replaced_nll", "mean"))
        .reset_index()
    )
    all_class["overlap_class"] = "ALL"
    frames.append(all_class)
    all_bin = (
        pair_late.groupby(["role", "seed", "pair_id", "overlap_class", "transcript_id"], dropna=False)
        .agg(n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), delta=("delta", "mean"), intact_nll=("intact_nll", "mean"), replaced_nll=("replaced_nll", "mean"))
        .reset_index()
    )
    all_bin["overlap_bin"] = "ALL"
    frames.append(all_bin)
    all_both = (
        pair_late.groupby(["role", "seed", "pair_id", "transcript_id"], dropna=False)
        .agg(n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), delta=("delta", "mean"), intact_nll=("intact_nll", "mean"), replaced_nll=("replaced_nll", "mean"))
        .reset_index()
    )
    all_both["overlap_bin"] = "ALL"; all_both["overlap_class"] = "ALL"
    frames.append(all_both)
    ext = pd.concat(frames, ignore_index=True)
    late = (
        ext.groupby(["role", "seed", "overlap_bin", "overlap_class"], dropna=False)
        .agg(n_pairs=("pair_id", "nunique"), mean_delta=("delta", "mean"), median_delta=("delta", "median"), se_pair_delta=("delta", lambda x: se(list(x))), frac_positive=("delta", lambda x: float((pd.Series(x) > 0).mean())))
        .reset_index()
    )
    cons: list[dict[str, Any]] = []
    for (seed, obin, ocls), g in ext.groupby(["seed", "overlap_bin", "overlap_class"], dropna=False):
        for a, b in [("V", "C"), ("R", "C"), ("V", "R")]:
            aa = g[g["role"] == a][["pair_id", "delta"]].rename(columns={"delta": "delta_a"})
            bb = g[g["role"] == b][["pair_id", "delta"]].rename(columns={"delta": "delta_b"})
            j = aa.merge(bb, on="pair_id", how="inner")
            if j.empty:
                continue
            d = j["delta_a"] - j["delta_b"]
            cons.append({
                "seed": int(seed), "overlap_bin": obin, "overlap_class": ocls, "contrast": f"{a}minus{b}", "n_pairs": int(len(j)),
                "mean_delta_diff": float(d.mean()), "median_delta_diff": float(d.median()),
                "se_pair_delta_diff": float(d.std(ddof=0) / math.sqrt(len(d))) if len(d) > 1 else float("nan"),
                "frac_positive": float((d > 0).mean()),
            })
    con = pd.DataFrame(cons)
    return late, con


def seed_summary(late: pd.DataFrame, con: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = (
        late.groupby(["role", "overlap_bin", "overlap_class"], dropna=False)
        .agg(n_seeds=("seed", "nunique"), mean_delta_across_seeds=("mean_delta", "mean"), sd_seed_mean=("mean_delta", lambda x: sd(list(x))), mean_pairs=("n_pairs", "mean"))
        .reset_index()
    )
    s2 = (
        con.groupby(["contrast", "overlap_bin", "overlap_class"], dropna=False)
        .agg(n_seeds=("seed", "nunique"), mean_delta_diff_across_seeds=("mean_delta_diff", "mean"), sd_seed_delta_diff=("mean_delta_diff", lambda x: sd(list(x))), mean_pairs=("n_pairs", "mean"))
        .reset_index()
    )
    return s1, s2


def f(x: Any, nd: int = 4) -> str:
    try:
        y = float(x)
    except Exception:
        return str(x)
    return "NA" if math.isnan(y) else f"{y:+.{nd}f}"


def write_note(stats: dict[str, Any], late: pd.DataFrame, con: pd.DataFrame, s_late: pd.DataFrame, s_con: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# research CHILDES variation-set probe scoring")
    lines.append("")
    lines.append("The primary surface-held-out CHILDES adjacent-utterance probe was scored on the original DeBERTa VIEW/CLEAN/REPEAT arms across seeds 43022, 43122, and 43222 at checkpoints 80M/90M/100M. Delta is NLL(replaced context) minus NLL(intact adjacent context), so positive values mean the natural neighbor helps the masked target.")
    lines.append("")
    lines.append(f"Probe records: {stats['records']}; paired targets: {stats['paired_targets']}; pairs: {stats['pairs']}; pair bins low/partial/high = {stats['pair_bins']}. Exact surface exposure hits in the original three training pools were zero in the validator.")
    lines.append("")
    lines.append("## Across-seed role means")
    lines.append("")
    lines.append("| bin | target class | role | mean delta | seed SD | mean pairs |")
    lines.append("|---|---|---|---:|---:|---:|")
    show = s_late[s_late["overlap_bin"].isin(["ALL", "low", "partial", "high"]) & s_late["overlap_class"].isin(["ALL", "nonoverlap", "overlap"])].copy()
    role_order = {"C":0,"R":1,"V":2}; bin_order={"ALL":0,"low":1,"partial":2,"high":3}; cls_order={"ALL":0,"nonoverlap":1,"overlap":2}
    show["_o"] = show["overlap_bin"].map(bin_order).fillna(9); show["_c"] = show["overlap_class"].map(cls_order).fillna(9); show["_r"] = show["role"].map(role_order).fillna(9)
    for _, r in show.sort_values(["_o","_c","_r"]).iterrows():
        lines.append(f"| {r['overlap_bin']} | {r['overlap_class']} | {r['role']} | {f(r['mean_delta_across_seeds'])} | {f(r['sd_seed_mean'])} | {float(r['mean_pairs']):.1f} |")
    lines.append("")
    lines.append("## Across-seed contrasts")
    lines.append("")
    lines.append("| bin | target class | contrast | mean delta difference | seed SD | mean pairs |")
    lines.append("|---|---|---|---:|---:|---:|")
    cshow = s_con[s_con["overlap_bin"].isin(["ALL", "low", "partial", "high"]) & s_con["overlap_class"].isin(["ALL", "nonoverlap", "overlap"])].copy()
    con_order={"VminusC":0,"RminusC":1,"VminusR":2}
    cshow["_o"] = cshow["overlap_bin"].map(bin_order).fillna(9); cshow["_c"] = cshow["overlap_class"].map(cls_order).fillna(9); cshow["_r"] = cshow["contrast"].map(con_order).fillna(9)
    for _, r in cshow.sort_values(["_o","_c","_r"]).iterrows():
        lines.append(f"| {r['overlap_bin']} | {r['overlap_class']} | {r['contrast']} | {f(r['mean_delta_diff_across_seeds'])} | {f(r['sd_seed_delta_diff'])} | {float(r['mean_pairs']):.1f} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("This natural probe does not reproduce the compact-rewrite arm ordering as a strong, clean effect. The all-bin nonoverlap contrasts are small relative to seed-to-seed movement. The partial nonoverlap subset is the closest analogue of natural restatement with some shared words; there VIEW exceeds REPEAT on average, but the effect is modest and high-bin support is sparse. Therefore the CHILDES score is a useful bridge showing that adjacent utterances carry measurable context benefit, but it is not yet a load-bearing natural-domain replication of the compact source/rewrite mechanism.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/childes_variation_probe_score`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    with PROBE_STATS.open(encoding="utf-8") as f:
        pst = json.load(f)
    with VALIDATION.open(encoding="utf-8") as f:
        val = json.load(f)
    if val.get("primary", {}).get("status") != "PASS":
        raise RuntimeError("primary probe validator did not pass")
    records = load_records()
    plan = {
        "status": "CHILDES_VARIATION_SCORE_PLAN",
        "created_utc": now(),
        "records": len(records),
        "pairs": int(pst["selected_pairs"]),
        "pair_bins": pst["selected_pair_bins"],
        "paired_targets": int(val["primary"]["paired_targets"]),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "probe_records": rel(PROBE_RECORDS),
    }
    (OUT / "childes_score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for arm in args.arms:
        if arm not in ARM_CONFIGS:
            raise KeyError(arm)
        for ck in args.checkpoints:
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            if not mp.exists():
                raise FileNotFoundError(mp)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 3)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []
    score_meta: list[dict[str, Any]] = []
    for arm in args.arms:
        role, seed = parse_arm(arm)
        for ck in args.checkpoints:
            t0 = time.time()
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32)
            model.eval().to(device)
            rows = score_records(model, records, device, pad_id, args.batch_size)
            for r in rows:
                r.update({"arm": arm, "role": role, "seed": seed, "checkpoint": ck})
            all_rows.extend(rows)
            elapsed = round(time.time() - t0, 2)
            score_meta.append({"arm": arm, "role": role, "seed": seed, "checkpoint": ck, "n_records": len(rows), "elapsed_sec": elapsed, "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} n={len(rows)} elapsed={elapsed:.1f}s", flush=True)

    write_csv(OUT / "childes_scored_rows.csv", all_rows)
    write_csv(OUT / "childes_score_meta.csv", score_meta)
    raw, target_delta, by_ck, pair_late0, late0, con0 = aggregate(all_rows)
    target_delta.to_csv(OUT / "childes_target_delta_rows.csv", index=False)
    by_ck.to_csv(OUT / "childes_by_checkpoint_target_summary.csv", index=False)
    pair_late0.to_csv(OUT / "childes_pair_late_delta_rows_basegroups.csv", index=False)
    late, con = add_all_groups(pair_late0)
    late.to_csv(OUT / "childes_late_role_summary.csv", index=False)
    con.to_csv(OUT / "childes_late_contrasts.csv", index=False)
    s_late, s_con = seed_summary(late, con)
    s_late.to_csv(OUT / "childes_across_seed_role_summary.csv", index=False)
    s_con.to_csv(OUT / "childes_across_seed_contrast_summary.csv", index=False)
    write_note({**plan, "pair_bins": plan["pair_bins"]}, late, con, s_late, s_con)
    result = {"status": "CHILDES_VARIATION_SCORE_DONE", "note": rel(NOTE), "out_dir": rel(OUT), "records_scored": len(all_rows)}
    (OUT / "childes_score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
