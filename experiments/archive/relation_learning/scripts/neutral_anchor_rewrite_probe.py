#!/usr/bin/env python3
"""research neutral-source anchor for the compact rewrite probe.

The existing rewrite probe has T=true related source and U=unrelated compact source.
This script adds N=length-matched ordinary held-out text in the source slot for the
same compact rewrite targets and the same seed43022 arms.  The three NLLs separate
source use, unrelated-neighbor interference, and neutral target fit.
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
import sys
import time
from collections import defaultdict
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/neutral_anchor_rewrite_probe.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
RUNS_OLD = ROOT / "experiments/archive/frontier_consolidation/training/runs"
RUNS_NEW = WS / "training/runs"
OUT = WS / "data/neutral_anchor_rewrite_probe"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/neutral_anchor_rewrite_probe.md')
T_U_ROWS = WS / "data/view_split_integration/rewrite_input_rows.csv"
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]

ARM_CONFIGS: dict[str, pathlib.Path] = {
    "D_C_43022": RUNS_OLD / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": RUNS_OLD / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43022": RUNS_OLD / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_RS_43022": RUNS_NEW / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022",
    "D_VS_43022": RUNS_NEW / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022",
}
ROLE = {
    "D_C_43022": "C",
    "D_R_43022": "R",
    "D_V_43022": "V",
    "D_RS_43022": "RS",
    "D_VS_43022": "VS",
}
CONTRASTS = [("R", "RS"), ("V", "VS"), ("R", "C"), ("RS", "C"), ("V", "C"), ("VS", "C"), ("V", "R"), ("VS", "RS")]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


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


def se(xs: list[float]) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.pstdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")


def ordinary_source_pool(tokenizer, rng: random.Random, scan_rows: int = 6992) -> list[list[int]]:
    pool: list[list[int]] = []
    for obj in base.read_jsonl(base.HELDOUT_ROWS, scan_rows):
        txt = base.sentence_or_window(str(obj.get("text", "")), rng)
        if not txt:
            continue
        ids = tokenizer(" ".join(txt.split()), add_special_tokens=False)["input_ids"]
        if ids:
            pool.append([int(x) for x in ids])
    if len(pool) < 100:
        raise RuntimeError(f"too few ordinary held-out source windows: {len(pool)}")
    return pool


def length_match(pool: list[list[int]], idx: int, target_len: int, rng: random.Random, forbid: set[int], vocab_ids: list[int]) -> list[int]:
    if target_len <= 0:
        return []
    out: list[int] = []
    j = (idx * 15485863 + 2909) % len(pool)
    attempts = 0
    while len(out) < target_len and attempts < len(pool) + 5:
        cand = pool[j % len(pool)]
        if cand:
            need = target_len - len(out)
            if len(cand) > need:
                start = rng.randrange(0, len(cand) - need + 1)
                out.extend(cand[start:start + need])
            else:
                out.extend(cand[:need])
        j += 104729
        attempts += 1
    if len(out) < target_len:
        out.extend(base.rand_ids(rng, vocab_ids, target_len - len(out), forbid=forbid))
    out = out[:target_len]
    for k, x in enumerate(out):
        if x in forbid:
            out[k] = base.rand_ids(rng, vocab_ids, 1, forbid=forbid)[0]
    if len(out) != target_len:
        raise RuntimeError("neutral source length mismatch")
    return [int(x) for x in out]


def build_neutral_records(tokenizer, vocab_ids: list[int], rng: random.Random, max_len: int, max_pairs: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = base.load_unselected_rewrite_pairs(max_pairs)
    ordinary_pool = ordinary_source_pool(tokenizer, rng)
    records: list[dict[str, Any]] = []
    stats: defaultdict[str, int] = defaultdict(int)
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = [int(x) for x in tokenizer(src_text, add_special_tokens=False)["input_ids"]]
        rew_enc = tokenizer(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = [int(x) for x in rew_enc["input_ids"]]
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > max_len:
            stats["too_long_or_empty"] += 1
            continue
        src_set = set(src_ids)
        by_cls: dict[str, list[int]] = {"overlap": [], "nonoverlap": []}
        for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a:
                continue
            piece = rew_text[a:b]
            if not base.token_content_class(piece):
                continue
            cls = "overlap" if int(rew_ids[j]) in src_set else "nonoverlap"
            by_cls[cls].append(j)
        chosen: list[tuple[int, str]] = []
        for cls in ["nonoverlap", "overlap"]:
            vals = by_cls[cls]
            if not vals:
                continue
            if len(vals) <= 2:
                picks = vals
            else:
                picks = [vals[round(k * (len(vals) - 1) / 1)] for k in range(2)]
            chosen.extend((j, cls) for j in picks)
        if not chosen:
            stats["no_content_tokens"] += 1
            continue
        for j, cls in chosen:
            target = int(rew_ids[j])
            neutral_src = length_match(ordinary_pool, i * 17 + j, len(src_ids), rng, forbid={target}, vocab_ids=vocab_ids)
            body_neutral = neutral_src + rew_ids
            rec = base.wrap_body(tokenizer, body_neutral, [len(neutral_src) + j], [target], {
                "probe_family": "heldout_rewrite_neutral_anchor",
                "probe_id": f"rewritecond:{i}:{j}:{p['pair_id']}",
                "condition": "neutral_ordinary",
                "token_class": cls,
                "pair_id": p["pair_id"],
                "sentence_id": p.get("sentence_id"),
                "doc_id": p.get("doc_id"),
                "source_words": int(p.get("source_words", base.wc(src_text))),
                "rewrite_words": int(p.get("rewrite_words", base.wc(rew_text))),
                "source_token_len": len(src_ids),
                "rewrite_token_len": len(rew_ids),
                "body_len": len(body_neutral),
                "target_token_id": target,
                "neutral_source_kind": "heldout_ordinary_text",
            }, max_len=max_len)
            if rec is not None:
                records.append(rec)
        stats["pairs_used"] += 1
        stats["pairs_with_nonoverlap"] += int(bool(by_cls["nonoverlap"]))
        stats["pairs_with_overlap"] += int(bool(by_cls["overlap"]))
    stats["unselected_pairs_loaded"] = len(pairs)
    stats["ordinary_source_pool"] = len(ordinary_pool)
    return records, dict(stats)


def score_neutral_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for arm in args.arms:
        for ck in args.checkpoints:
            t0 = time.time()
            model_path = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {model_path}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval().to(device)
            scored = base.score_records(model, records, device, pad_id, args.batch_size)
            for r in scored:
                rows.append({
                    "arm": arm,
                    "checkpoint": ck,
                    "seed": 43022,
                    "role2": ROLE[arm],
                    "probe_id": r["probe_id"],
                    "pair_id": r["pair_id"],
                    "token_class": r["token_class"],
                    "neutral_nll": r["nll_per_token"],
                    "target_token_id": r["target_token_id"],
                    "source_token_len": r["source_token_len"],
                    "rewrite_token_len": r["rewrite_token_len"],
                    "source_words": r["source_words"],
                    "rewrite_words": r["rewrite_words"],
                })
            meta.append({"arm": arm, "checkpoint": ck, "n_records": len(scored), "elapsed_sec": round(time.time() - t0, 2), "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} n={len(scored)} elapsed={time.time()-t0:.1f}s", flush=True)
    return rows, meta


def load_tu_rows() -> pd.DataFrame:
    df = pd.read_csv(T_U_ROWS)
    df = df[(df["seed"] == 43022) & (df["checkpoint"].isin(CKPTS))].copy()
    # Keep one row per arm/checkpoint/probe. The input already unifies original and split arms.
    keep = ["arm", "checkpoint", "seed", "role2", "probe_id", "pair_id", "token_class", "gain", "true_source_nll", "unrelated_source_nll"]
    return df[keep].drop_duplicates(subset=["arm", "checkpoint", "probe_id"])


def integrate(neutral_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ndf = pd.DataFrame(neutral_rows)
    tu = load_tu_rows()
    merged = tu.merge(ndf[["arm", "checkpoint", "probe_id", "neutral_nll"]], on=["arm", "checkpoint", "probe_id"], how="inner")
    if len(merged) != len(tu):
        raise RuntimeError(f"N/TU merge lost rows: neutral merge {len(merged)} vs TU {len(tu)}")
    merged["T_minus_N"] = merged["true_source_nll"] - merged["neutral_nll"]
    merged["U_minus_N"] = merged["unrelated_source_nll"] - merged["neutral_nll"]
    merged["true_help_N_minus_T"] = merged["neutral_nll"] - merged["true_source_nll"]
    merged["unrel_help_N_minus_U"] = merged["neutral_nll"] - merged["unrelated_source_nll"]
    merged["rewrite_gain_U_minus_T"] = merged["unrelated_source_nll"] - merged["true_source_nll"]

    group = (
        merged.groupby(["role2", "checkpoint", "token_class"], dropna=False)
        .agg(
            n=("probe_id", "count"),
            T=("true_source_nll", "mean"),
            U=("unrelated_source_nll", "mean"),
            N=("neutral_nll", "mean"),
            U_minus_T=("rewrite_gain_U_minus_T", "mean"),
            T_minus_N=("T_minus_N", "mean"),
            U_minus_N=("U_minus_N", "mean"),
            N_minus_T=("true_help_N_minus_T", "mean"),
            N_minus_U=("unrel_help_N_minus_U", "mean"),
        )
        .reset_index()
    )
    late = (
        group.groupby(["role2", "token_class"], dropna=False)
        .agg(
            n_min=("n", "min"),
            T=("T", "mean"),
            U=("U", "mean"),
            N=("N", "mean"),
            U_minus_T=("U_minus_T", "mean"),
            T_minus_N=("T_minus_N", "mean"),
            U_minus_N=("U_minus_N", "mean"),
            N_minus_T=("N_minus_T", "mean"),
            N_minus_U=("N_minus_U", "mean"),
        )
        .reset_index()
    )
    crows: list[dict[str, Any]] = []
    for token_class, g in late.groupby("token_class", dropna=False):
        gd = {r["role2"]: r for _, r in g.iterrows()}
        for a, b in CONTRASTS:
            if a not in gd or b not in gd:
                continue
            ra, rb = gd[a], gd[b]
            row = {"contrast": f"{a}minus{b}", "token_class": token_class, "n_min": int(min(ra["n_min"], rb["n_min"]))}
            for k in ["T", "U", "N", "U_minus_T", "T_minus_N", "U_minus_N", "N_minus_T", "N_minus_U"]:
                row[f"delta_{k}"] = float(ra[k]) - float(rb[k])
            crows.append(row)
    con = pd.DataFrame(crows)

    # Pair-averaged late contrasts for the main nonoverlap readout.
    rows = []
    pa = (
        merged[merged["token_class"] == "nonoverlap"]
        .groupby(["pair_id", "role2", "checkpoint"], dropna=False)
        .agg(T=("true_source_nll", "mean"), U=("unrelated_source_nll", "mean"), N=("neutral_nll", "mean"), U_minus_T=("rewrite_gain_U_minus_T", "mean"), T_minus_N=("T_minus_N", "mean"), U_minus_N=("U_minus_N", "mean"))
        .reset_index()
    )
    late_pa = (
        pa.groupby(["pair_id", "role2"], dropna=False)
        .agg(n_ck=("checkpoint", "nunique"), T=("T", "mean"), U=("U", "mean"), N=("N", "mean"), U_minus_T=("U_minus_T", "mean"), T_minus_N=("T_minus_N", "mean"), U_minus_N=("U_minus_N", "mean"))
        .reset_index()
    )
    late_pa = late_pa[late_pa["n_ck"] == len(CKPTS)]
    for a, b in CONTRASTS:
        aa = late_pa[late_pa["role2"] == a].rename(columns={k: f"{k}_a" for k in ["T", "U", "N", "U_minus_T", "T_minus_N", "U_minus_N"]})
        bb = late_pa[late_pa["role2"] == b].rename(columns={k: f"{k}_b" for k in ["T", "U", "N", "U_minus_T", "T_minus_N", "U_minus_N"]})
        j = aa[["pair_id", "T_a", "U_a", "N_a", "U_minus_T_a", "T_minus_N_a", "U_minus_N_a"]].merge(bb[["pair_id", "T_b", "U_b", "N_b", "U_minus_T_b", "T_minus_N_b", "U_minus_N_b"]], on="pair_id", how="inner")
        if j.empty:
            continue
        out: dict[str, Any] = {"contrast": f"{a}minus{b}", "token_class": "nonoverlap", "n_pairs": int(len(j))}
        for k in ["T", "U", "N", "U_minus_T", "T_minus_N", "U_minus_N"]:
            d = j[f"{k}_a"] - j[f"{k}_b"]
            out[f"delta_{k}_mean"] = float(d.mean())
            out[f"delta_{k}_median"] = float(d.median())
            out[f"delta_{k}_se"] = float(d.std(ddof=0) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
            out[f"delta_{k}_frac_positive"] = float((d > 0).mean())
        rows.append(out)
    pair_con = pd.DataFrame(rows)
    return merged, group, late, con, pair_con


def fmt(x: Any, nd: int = 4) -> str:
    try:
        y = float(x)
    except Exception:
        return str(x)
    return "NA" if math.isnan(y) else f"{y:+.{nd}f}"


def make_note(stats: dict[str, Any], late: pd.DataFrame, con: pd.DataFrame, pair_con: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# research neutral-source anchor for rewrite conditioning")
    lines.append("")
    lines.append("This run adds a third context to the existing compact-rewrite probe. T is the true source, U is an unrelated compact source, and N is length-matched ordinary held-out text in the same source slot. N separates broad target fit from dependence on having another compact source-like neighbor.")
    lines.append("")
    lines.append(f"Records: {stats['records']} neutral rows over {stats['pairs_used']} pairs; ordinary held-out source windows: {stats['ordinary_source_pool']}. The T/U merge recovered all matching rows.")
    lines.append("")
    lines.append("## Late token-nonoverlap terms")
    lines.append("")
    lines.append("| role | n | T | U | N | U-T | T-N | U-N |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    show = late[late["token_class"] == "nonoverlap"].copy()
    order = {"C":0,"R":1,"RS":2,"V":3,"VS":4}
    show["_ord"] = show["role2"].map(order).fillna(9)
    for _, r in show.sort_values("_ord").iterrows():
        lines.append(f"| {r['role2']} | {int(r['n_min'])} | {fmt(r['T'])} | {fmt(r['U'])} | {fmt(r['N'])} | {fmt(r['U_minus_T'])} | {fmt(r['T_minus_N'])} | {fmt(r['U_minus_N'])} |")
    lines.append("")
    lines.append("Here T-N < 0 means the related source helps compared with ordinary text; U-N > 0 means the unrelated compact source hurts compared with ordinary text.")
    lines.append("")
    lines.append("## Local-versus-split reading")
    lines.append("")
    lines.append("| contrast | ΔN | ΔU | ΔT | Δ(U-T) | Δ(T-N) | Δ(U-N) | pair ΔN | pair Δ(U-N) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    main = con[(con["token_class"] == "nonoverlap") & (con["contrast"].isin(["RminusRS", "VminusVS"]))]
    for _, r in main.sort_values("contrast").iterrows():
        pc = pair_con[pair_con["contrast"] == r["contrast"]]
        pair_n = float(pc.iloc[0]["delta_N_mean"]) if len(pc) else float("nan")
        pair_un = float(pc.iloc[0]["delta_U_minus_N_mean"]) if len(pc) else float("nan")
        lines.append(f"| {r['contrast']} | {fmt(r['delta_N'])} | {fmt(r['delta_U'])} | {fmt(r['delta_T'])} | {fmt(r['delta_U_minus_T'])} | {fmt(r['delta_T_minus_N'])} | {fmt(r['delta_U_minus_N'])} | {fmt(pair_n)} | {fmt(pair_un)} |")
    lines.append("")
    lines.append("If the earlier U gap came from weaker learning of the companion content, local-minus-split should remain visible on N. If it came mainly from unrelated-neighbor interference, ΔN should be small while Δ(U-N) grows for the local arm. The result above decides which reading better fits each relation.")
    lines.append("")
    lines.append("## Arm-versus-CLEAN anchor terms")
    lines.append("")
    lines.append("| contrast | ΔN | ΔT | ΔU | Δ(U-T) | Δ(T-N) | Δ(U-N) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    clean_show = con[(con["token_class"] == "nonoverlap") & (con["contrast"].isin(["RminusC", "RSminusC", "VminusC", "VSminusC"]))]
    for _, r in clean_show.sort_values("contrast").iterrows():
        lines.append(f"| {r['contrast']} | {fmt(r['delta_N'])} | {fmt(r['delta_T'])} | {fmt(r['delta_U'])} | {fmt(r['delta_U_minus_T'])} | {fmt(r['delta_T_minus_N'])} | {fmt(r['delta_U_minus_N'])} |")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/neutral_anchor_rewrite_probe`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=920020)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ck in args.checkpoints:
            p = ARM_CONFIGS[arm] / "hf_model" / ck
            if not p.exists():
                raise FileNotFoundError(p)

    tokenizer = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    vocab_ids = base.safe_vocab_ids(tokenizer)
    rng = random.Random(args.seed)
    max_pairs = None if args.rewrite_max_pairs <= 0 else args.rewrite_max_pairs
    records, stats = build_neutral_records(tokenizer, vocab_ids, rng, args.max_len, max_pairs)
    stats.update({"records": len(records), "created_utc": now(), "arms": args.arms, "checkpoints": args.checkpoints})
    plan = {"status": "NEUTRAL_ANCHOR_PLAN", **stats, "out_dir": rel(OUT)}
    (OUT / "neutral_anchor_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return

    neutral_rows, meta = score_neutral_records(records, args)
    write_csv(OUT / "neutral_anchor_rows.csv", neutral_rows)
    write_csv(OUT / "neutral_anchor_score_meta.csv", meta)
    merged, group, late, con, pair_con = integrate(neutral_rows)
    merged.to_csv(OUT / "rewrite_TUN_rows.csv", index=False)
    group.to_csv(OUT / "rewrite_TUN_by_checkpoint.csv", index=False)
    late.to_csv(OUT / "rewrite_TUN_late_roles.csv", index=False)
    con.to_csv(OUT / "rewrite_TUN_late_contrasts.csv", index=False)
    pair_con.to_csv(OUT / "rewrite_TUN_pair_late_contrasts.csv", index=False)
    make_note({**stats, "records": len(records)}, late, con, pair_con)
    result = {
        "status": "NEUTRAL_ANCHOR_DONE",
        "note": rel(NOTE),
        "out_dir": rel(OUT),
        "records": len(records),
        "pairs_used": stats.get("pairs_used"),
    }
    (OUT / "neutral_anchor_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
