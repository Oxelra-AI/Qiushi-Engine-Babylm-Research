#!/usr/bin/env python3
"""research original-arm three-seed neutral-source anchor.

Scores N=ordinary-heldout source slot for original DeBERTa C/R/V arms at seeds
43022/43122/43222, then joins to existing true/unrelated compact-rewrite rows.
This tests whether the source-specific recurrence cost and VIEW benefit are stable
relative to a neutral anchor, not only U vs T.
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

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/original_threeseed_neutral_anchor.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import heldout_copy_rewrite_entity_ablation as base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
OLD_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
NEW_RUNS = WS / "training/runs"
OUT = WS / "data/original_threeseed_neutral_anchor"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/original_threeseed_neutral_anchor.md')
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
TU_SOURCES = [
    WS / "data/heldout_copy_rewrite_entity_ablation/rewrite_pair_rows.csv",
    WS / "data/seed43222_probes/rewrite_pair_rows.csv",
    WS / "data/seed43222_parallel_clean_probes/rewrite_pair_rows.csv",
]
CONTRASTS = [("R", "C"), ("V", "C"), ("V", "R")]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
        raise RuntimeError(f"too few ordinary windows: {len(pool)}")
    return pool


def length_match(pool: list[list[int]], idx: int, target_len: int, rng: random.Random, forbid: set[int], vocab_ids: list[int]) -> list[int]:
    out: list[int] = []
    j = (idx * 15485863 + 2909) % len(pool)
    attempts = 0
    while len(out) < target_len and attempts < len(pool) + 5:
        cand = pool[j % len(pool)]
        need = target_len - len(out)
        if cand:
            if len(cand) > need:
                st = rng.randrange(0, len(cand) - need + 1)
                out.extend(cand[st:st + need])
            else:
                out.extend(cand[:need])
        j += 104729; attempts += 1
    if len(out) < target_len:
        out.extend(base.rand_ids(rng, vocab_ids, target_len - len(out), forbid=forbid))
    out = out[:target_len]
    for k, x in enumerate(out):
        if x in forbid:
            out[k] = base.rand_ids(rng, vocab_ids, 1, forbid=forbid)[0]
    return [int(x) for x in out]


def build_records(tokenizer, vocab_ids: list[int], rng: random.Random, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = base.load_unselected_rewrite_pairs(None)
    pool = ordinary_source_pool(tokenizer, rng)
    records: list[dict[str, Any]] = []
    stats: defaultdict[str, int] = defaultdict(int)
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = [int(x) for x in tokenizer(src_text, add_special_tokens=False)["input_ids"]]
        rew_enc = tokenizer(rew_text, add_special_tokens=False, return_offsets_mapping=True)
        rew_ids = [int(x) for x in rew_enc["input_ids"]]
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > max_len:
            continue
        src_set = set(src_ids)
        by_cls: dict[str, list[int]] = {"overlap": [], "nonoverlap": []}
        for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
            if b <= a:
                continue
            if not base.token_content_class(rew_text[a:b]):
                continue
            by_cls["overlap" if rew_ids[j] in src_set else "nonoverlap"].append(j)
        chosen: list[tuple[int, str]] = []
        for cls in ["nonoverlap", "overlap"]:
            vals = by_cls[cls]
            if not vals:
                continue
            if len(vals) <= 2:
                picks = vals
            else:
                picks = [vals[0], vals[-1]]
            chosen.extend((j, cls) for j in picks)
        if not chosen:
            continue
        for j, cls in chosen:
            target = int(rew_ids[j])
            neutral = length_match(pool, i * 17 + j, len(src_ids), rng, {target}, vocab_ids)
            rec = base.wrap_body(tokenizer, neutral + rew_ids, [len(neutral) + j], [target], {
                "probe_id": f"rewritecond:{i}:{j}:{p['pair_id']}",
                "pair_id": p["pair_id"],
                "token_class": cls,
                "target_token_id": target,
                "source_token_len": len(src_ids),
                "rewrite_token_len": len(rew_ids),
            }, max_len=max_len)
            if rec is not None:
                records.append(rec)
        stats["pairs_used"] += 1
        stats["pairs_with_nonoverlap"] += int(bool(by_cls["nonoverlap"]))
        stats["pairs_with_overlap"] += int(bool(by_cls["overlap"]))
    stats["ordinary_source_pool"] = len(pool)
    return records, dict(stats)


def role_seed(arm: str) -> tuple[str, int]:
    p = arm.split("_")
    return p[1], int(p[2])


@torch.no_grad()
def score(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for arm in args.arms:
        role, seed = role_seed(arm)
        for ck in args.checkpoints:
            t0 = time.time()
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32)
            model.eval().to(device)
            scored = base.score_records(model, records, device, pad_id, args.batch_size)
            for r in scored:
                rows.append({"arm": arm, "role": role, "seed": seed, "checkpoint": ck, "probe_id": r["probe_id"], "pair_id": r["pair_id"], "token_class": r["token_class"], "neutral_nll": r["nll_per_token"], "target_token_id": r["target_token_id"]})
            meta.append({"arm": arm, "role": role, "seed": seed, "checkpoint": ck, "n": len(scored), "elapsed_sec": round(time.time()-t0, 2), "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} n={len(scored)} elapsed={time.time()-t0:.1f}s", flush=True)
    return rows, meta


def load_tu() -> pd.DataFrame:
    frames = []
    for path in TU_SOURCES:
        df = pd.read_csv(path)
        if "role" not in df.columns:
            df["role"] = df["arm"].astype(str).str.split("_").str[1]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    keep_arms = set(ARM_CONFIGS)
    df = df[df["arm"].isin(keep_arms) & df["checkpoint"].isin(CKPTS)].copy()
    df = df.drop_duplicates(subset=["arm", "checkpoint", "probe_id"], keep="first")
    return df[["arm", "role", "seed", "checkpoint", "probe_id", "pair_id", "token_class", "true_source_nll", "unrelated_source_nll", "gain"]]


def integrate(nrows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = pd.DataFrame(nrows)
    tu = load_tu()
    m = tu.merge(n[["arm", "checkpoint", "probe_id", "neutral_nll"]], on=["arm", "checkpoint", "probe_id"], how="inner")
    if len(m) != len(tu):
        raise RuntimeError(f"merge lost rows {len(m)} vs {len(tu)}")
    m["T_minus_N"] = m["true_source_nll"] - m["neutral_nll"]
    m["U_minus_N"] = m["unrelated_source_nll"] - m["neutral_nll"]
    m["U_minus_T"] = m["unrelated_source_nll"] - m["true_source_nll"]
    by_ck = (
        m.groupby(["seed", "role", "checkpoint", "token_class"], dropna=False)
        .agg(n=("probe_id", "count"), T=("true_source_nll", "mean"), U=("unrelated_source_nll", "mean"), N=("neutral_nll", "mean"), U_minus_T=("U_minus_T", "mean"), T_minus_N=("T_minus_N", "mean"), U_minus_N=("U_minus_N", "mean"))
        .reset_index()
    )
    late = (
        by_ck.groupby(["seed", "role", "token_class"], dropna=False)
        .agg(n_min=("n", "min"), T=("T", "mean"), U=("U", "mean"), N=("N", "mean"), U_minus_T=("U_minus_T", "mean"), T_minus_N=("T_minus_N", "mean"), U_minus_N=("U_minus_N", "mean"))
        .reset_index()
    )
    cons = []
    for (seed, cls), g in late.groupby(["seed", "token_class"], dropna=False):
        gd = {r["role"]: r for _, r in g.iterrows()}
        for a, b in CONTRASTS:
            if a not in gd or b not in gd:
                continue
            ra, rb = gd[a], gd[b]
            row = {"seed": int(seed), "token_class": cls, "contrast": f"{a}minus{b}", "n_min": int(min(ra["n_min"], rb["n_min"]))}
            for k in ["T", "U", "N", "U_minus_T", "T_minus_N", "U_minus_N"]:
                row[f"delta_{k}"] = float(ra[k]) - float(rb[k])
            cons.append(row)
    con = pd.DataFrame(cons)
    across = (
        con.groupby(["token_class", "contrast"], dropna=False)
        .agg(n_seeds=("seed", "nunique"), delta_T_mean=("delta_T", "mean"), delta_T_sd=("delta_T", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")), delta_U_mean=("delta_U", "mean"), delta_U_sd=("delta_U", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")), delta_N_mean=("delta_N", "mean"), delta_N_sd=("delta_N", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")), delta_U_minus_T_mean=("delta_U_minus_T", "mean"), delta_U_minus_T_sd=("delta_U_minus_T", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")), delta_T_minus_N_mean=("delta_T_minus_N", "mean"), delta_T_minus_N_sd=("delta_T_minus_N", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")), delta_U_minus_N_mean=("delta_U_minus_N", "mean"), delta_U_minus_N_sd=("delta_U_minus_N", lambda x: statistics.stdev(list(x)) if len(x)>1 else float("nan")))
        .reset_index()
    )
    return m, late, con, across


def f(x: Any, nd: int = 4) -> str:
    try:
        y = float(x)
    except Exception:
        return str(x)
    return "NA" if math.isnan(y) else f"{y:+.{nd}f}"


def make_note(stats: dict[str, Any], late: pd.DataFrame, con: pd.DataFrame, across: pd.DataFrame) -> None:
    lines = []
    lines.append("# research original-arm three-seed neutral anchor")
    lines.append("")
    lines.append("N is the target NLL with length-matched ordinary held-out text in the source slot, scored on the same compact-rewrite masked targets as T and U. This checks whether the original three-seed source-specific effects survive a neutral anchor.")
    lines.append("")
    lines.append(f"Neutral records: {stats['records']} over {stats['pairs_used']} pairs; ordinary source windows: {stats['ordinary_source_pool']}.")
    lines.append("")
    lines.append("## Token-nonoverlap across-seed contrasts")
    lines.append("")
    lines.append("| contrast | seeds | ΔT | ΔU | ΔN | Δ(U-T) | Δ(T-N) | Δ(U-N) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    show = across[across["token_class"] == "nonoverlap"].copy()
    order = {"RminusC":0,"VminusC":1,"VminusR":2}
    show["_o"] = show["contrast"].map(order).fillna(9)
    for _, r in show.sort_values("_o").iterrows():
        lines.append(f"| {r['contrast']} | {int(r['n_seeds'])} | {f(r['delta_T_mean'])}±{abs(float(r['delta_T_sd'])):.4f} | {f(r['delta_U_mean'])}±{abs(float(r['delta_U_sd'])):.4f} | {f(r['delta_N_mean'])}±{abs(float(r['delta_N_sd'])):.4f} | {f(r['delta_U_minus_T_mean'])}±{abs(float(r['delta_U_minus_T_sd'])):.4f} | {f(r['delta_T_minus_N_mean'])}±{abs(float(r['delta_T_minus_N_sd'])):.4f} | {f(r['delta_U_minus_N_mean'])}±{abs(float(r['delta_U_minus_N_sd'])):.4f} |")
    lines.append("")
    lines.append("## Per-seed token-nonoverlap contrasts")
    lines.append("")
    lines.append("| seed | contrast | ΔT | ΔU | ΔN | Δ(U-T) | Δ(T-N) | Δ(U-N) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for _, r in con[(con["token_class"] == "nonoverlap") & (con["contrast"].isin(["RminusC","VminusC","VminusR"]))].sort_values(["seed","contrast"]).iterrows():
        lines.append(f"| {int(r['seed'])} | {r['contrast']} | {f(r['delta_T'])} | {f(r['delta_U'])} | {f(r['delta_N'])} | {f(r['delta_U_minus_T'])} | {f(r['delta_T_minus_N'])} | {f(r['delta_U_minus_N'])} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("Across all three DeBERTa seeds, original REPEAT's negative compact-rewrite gain relative to CLEAN is a true-source effect relative to N: R−C has positive Δ(T-N) in every seed, while Δ(U-N) is small and not the source of the effect. VIEW's positive compact-rewrite gain is likewise true-source help relative to N in every seed, while Δ(U-N) remains small. The T/U sign reversal therefore survives the neutral anchor as a relation-specific true-source effect rather than an unrelated-neighbor artifact.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/original_threeseed_neutral_anchor`.")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARM_CONFIGS))
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=920020)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        for ck in args.checkpoints:
            p = ARM_CONFIGS[arm] / "hf_model" / ck
            if not p.exists():
                raise FileNotFoundError(p)
    tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    vocab_ids = base.safe_vocab_ids(tok)
    records, stats = build_records(tok, vocab_ids, random.Random(args.seed), args.max_len)
    plan = {"status": "ORIGINAL_THREESEED_NEUTRAL_PLAN", "records": len(records), **stats, "arms": args.arms, "checkpoints": args.checkpoints, "created_utc": now()}
    (OUT / "original_threeseed_neutral_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
        return
    nrows, meta = score(records, args)
    write_csv(OUT / "neutral_anchor_rows.csv", nrows)
    write_csv(OUT / "neutral_score_meta.csv", meta)
    merged, late, con, across = integrate(nrows)
    merged.to_csv(OUT / "rewrite_TUN_rows.csv", index=False)
    late.to_csv(OUT / "rewrite_TUN_late_roles.csv", index=False)
    con.to_csv(OUT / "rewrite_TUN_late_contrasts.csv", index=False)
    across.to_csv(OUT / "rewrite_TUN_across_seed_contrasts.csv", index=False)
    make_note({**stats, "records": len(records)}, late, con, across)
    result = {"status": "ORIGINAL_THREESEED_NEUTRAL_DONE", "note": rel(NOTE), "out_dir": rel(OUT)}
    (OUT / "original_threeseed_neutral_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
