#!/usr/bin/env python3
"""research: score the completed causal-GPT C/R/RS objective-boundary runs.

The causal GPT models were trained on the same clean/repeat/repeat-split 100M
streams as the DeBERTa relation-locality test.  This script scores the current
study's held-out source-conditioning probes with a causal next-token readout:
for each masked probe position, score p(target_token | source context + target
prefix before that position).  It deliberately does not use right context, so the
numbers are an objective-boundary test rather than a direct MLM-equivalent score.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_causal_gpt_relation_boundary.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
SCRIPTS = WS / "scripts"
sys.path.insert(0, str(SCRIPTS))
import heldout_copy_rewrite_entity_ablation as base009  # noqa: E402
import neutral_anchor_rewrite_probe as base020  # noqa: E402

WIKI_DIR = WS / "analysis/wikipedia_simplification_restatement_probe"
WIKI_RECORDS = WIKI_DIR / "wikipedia_simplification_probe_records.jsonl"
WIKI_VALIDATION = WIKI_DIR / "validation_results.json"
DEFAULT_OUT = WS / "data/causal_gpt_relation_boundary"
DEFAULT_NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/causal_gpt_relation_boundary.md')

CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ARMS = {
    "C": WS / "training/runs/causal_gpt_c_dose2p64x_seed43022",
    "R": WS / "training/runs/causal_gpt_r_dose2p64x_seed43022",
    "RS": WS / "training/runs/causal_gpt_rs_dose2p64x_seed43022",
}
CONTRASTS = [("RminusC", "R", "C"), ("RSminusC", "RS", "C"), ("RminusRS", "R", "RS")]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def se(xs) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.pstdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    if preferred:
        fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def model_path(role: str, ck: str) -> pathlib.Path:
    return ARMS[role] / "hf_model" / ck


def build_probe_records(tokenizer, args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    vocab_ids = base009.safe_vocab_ids(tokenizer)
    copy_records, copy_stats = base009.build_heldout_copy_records(
        tokenizer, vocab_ids, rng, args.copy_n_per_span, args.copy_span_lengths,
        args.max_len_copy_rewrite, args.copy_scan_rows,
    )
    rewrite_limit = None if args.rewrite_max_pairs <= 0 else args.rewrite_max_pairs
    rewrite_records, rewrite_stats = base009.build_rewrite_conditioning_records(
        tokenizer, vocab_ids, rng, rewrite_limit, args.rewrite_tokens_per_class,
        args.max_len_copy_rewrite,
    )
    neutral_records, neutral_stats = base020.build_neutral_records(
        tokenizer, vocab_ids, random.Random(args.seed + 17), args.max_len_copy_rewrite, rewrite_limit,
    )
    wiki_validation = read_json(WIKI_VALIDATION)
    if wiki_validation.get("status") != "PASS":
        raise RuntimeError(f"Wiki validation is not PASS: {wiki_validation}")
    wiki_records = read_jsonl(WIKI_RECORDS)
    if args.wiki_max_records and args.wiki_max_records > 0:
        wiki_records = wiki_records[:args.wiki_max_records]
    return copy_records + rewrite_records + neutral_records + wiki_records, {
        "copy_records": len(copy_records),
        "rewrite_TU_records": len(rewrite_records),
        "rewrite_N_records": len(neutral_records),
        "wiki_records": len(wiki_records),
        "copy_stats": copy_stats,
        "rewrite_stats": rewrite_stats,
        "neutral_stats": neutral_stats,
    }


def score_instances_from_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metas: list[dict[str, Any]] = []
    inst: list[dict[str, Any]] = []
    for ridx, r in enumerate(records):
        if "positions" in r and "targets" in r:
            positions = list(r["positions"])
            targets = list(r["targets"])
            hidden = {"input_ids", "attention_mask", "positions", "targets"}
        else:
            positions = [int(r["mask_position"])]
            targets = [int(r["target_token_id"])]
            hidden = {"input_ids", "attention_mask", "mask_position"}
        metas.append({k: v for k, v in r.items() if k not in hidden})
        for pos, target in zip(positions, targets):
            pos = int(pos)
            if pos <= 0:
                continue
            inst.append({"record_index": ridx, "prefix_ids": list(r["input_ids"][:pos]), "target": int(target)})
    return metas, inst


@torch.no_grad()
def score_records_causal(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    metas, inst = score_instances_from_records(records)
    sums = [0.0 for _ in metas]
    counts = [0 for _ in metas]
    for start in range(0, len(inst), batch_size):
        batch = inst[start:start + batch_size]
        max_len = max(len(x["prefix_ids"]) for x in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        last = []
        targets = []
        ridxs = []
        for i, x in enumerate(batch):
            pref = x["prefix_ids"]
            L = len(pref)
            ids[i, :L] = torch.tensor(pref, dtype=torch.long)
            att[i, :L] = 1
            last.append(L - 1)
            targets.append(int(x["target"]))
            ridxs.append(int(x["record_index"]))
        out = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.float()
        logp = torch.nn.functional.log_softmax(out, dim=-1)
        for i, ridx in enumerate(ridxs):
            lp = float(logp[i, int(last[i]), int(targets[i])].detach().cpu())
            sums[ridx] += lp
            counts[ridx] += 1
    rows: list[dict[str, Any]] = []
    for meta, s, n in zip(metas, sums, counts):
        if n <= 0:
            continue
        row = dict(meta)
        row.update({"logprob_sum": s, "nll_per_token": -s / n, "n_masked_tokens": n})
        rows.append(row)
    return rows


def split_scored(scored: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    copy_like: list[dict[str, Any]] = []
    rewrite_tu_like: list[dict[str, Any]] = []
    neutral_like: list[dict[str, Any]] = []
    wiki_like: list[dict[str, Any]] = []
    for r in scored:
        fam = r.get("probe_family")
        if fam == "heldout_natural_source_repeat":
            copy_like.append(r)
        elif fam == "heldout_rewrite_content_conditioning":
            rewrite_tu_like.append(r)
        elif fam == "heldout_rewrite_neutral_anchor":
            neutral_like.append(r)
        elif "condition_code" in r and "record_id" in r:
            wiki_like.append(r)
    return copy_like, rewrite_tu_like, neutral_like, wiki_like


def aggregate_all(scored_rows: list[dict[str, Any]], out: pathlib.Path) -> dict[str, pd.DataFrame]:
    copy_rows: list[dict[str, Any]] = []
    rewrite_tu_rows: list[dict[str, Any]] = []
    neutral_rows: list[dict[str, Any]] = []
    wiki_rows: list[dict[str, Any]] = []
    for role in ARMS:
        for ck in CKPTS:
            subset = [r for r in scored_rows if r["role"] == role and r["checkpoint"] == ck]
            csub, tusub, nsub, wsub = split_scored(subset)
            for q in base009.aggregate_copy(csub, role, ck):
                q["role"] = role; q["seed"] = 43022; copy_rows.append(q)
            for q in base009.aggregate_rewrite(tusub, role, ck):
                q["role"] = role; q["seed"] = 43022; rewrite_tu_rows.append(q)
            for r in nsub:
                neutral_rows.append({
                    "arm": role, "role": role, "checkpoint": ck, "probe_id": r["probe_id"],
                    "pair_id": r["pair_id"], "token_class": r["token_class"],
                    "neutral_nll": r["nll_per_token"], "target_token_id": r["target_token_id"],
                })
            for r in wsub:
                wiki_rows.append({
                    "arm": role, "role": role, "seed": 43022, "checkpoint": ck,
                    "record_id": r["record_id"], "pair_id": r["pair_id"], "target_key": r["target_key"],
                    "condition_code": r["condition_code"], "overlap_bin": r["overlap_bin"],
                    "token_class": r["token_class"], "surface_word_class": r.get("surface_word_class"),
                    "target_word": r.get("target_word"), "target_token_id": r["target_token_id"],
                    "nll": r["nll_per_token"], "logp": -r["nll_per_token"],
                })
    write_csv(out / "copy_pair_rows.csv", copy_rows)
    write_csv(out / "rewrite_TU_pair_rows.csv", rewrite_tu_rows)
    write_csv(out / "rewrite_neutral_rows.csv", neutral_rows)
    write_csv(out / "wikipedia_scored_rows.csv", wiki_rows)
    return {"copy": pd.DataFrame(copy_rows), "rewrite": pd.DataFrame(rewrite_tu_rows), "neutral": pd.DataFrame(neutral_rows), "wiki": pd.DataFrame(wiki_rows)}


def integrate_compact(dfs: dict[str, pd.DataFrame], out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tu = dfs["rewrite"].copy()
    n = dfs["neutral"].copy()
    m = tu.merge(n[["arm", "checkpoint", "probe_id", "neutral_nll"]], on=["arm", "checkpoint", "probe_id"], how="inner")
    if len(m) != len(tu):
        raise RuntimeError(f"compact T/U/N merge lost rows: {len(m)} vs {len(tu)}")
    m["role"] = m["arm"]
    m["A_T"] = m["neutral_nll"] - m["true_source_nll"]
    m["A_U"] = m["neutral_nll"] - m["unrelated_source_nll"]
    m["G"] = m["unrelated_source_nll"] - m["true_source_nll"]
    m.to_csv(out / "compact_TUN_rows.csv", index=False)
    by_ck = m.groupby(["role", "checkpoint", "token_class"], dropna=False).agg(
        n=("probe_id", "count"), T=("true_source_nll", "mean"), U=("unrelated_source_nll", "mean"), N=("neutral_nll", "mean"),
        A_T=("A_T", "mean"), A_U=("A_U", "mean"), G=("G", "mean"),
    ).reset_index()
    by_ck.to_csv(out / "compact_TUN_by_checkpoint.csv", index=False)
    late = by_ck.groupby(["role", "token_class"], dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"), n=("n", "mean"), T=("T", "mean"), U=("U", "mean"), N=("N", "mean"),
        A_T=("A_T", "mean"), A_U=("A_U", "mean"), G=("G", "mean"),
    ).reset_index()
    late.to_csv(out / "compact_TUN_late_roles.csv", index=False)
    idx = {(r.role, r.token_class): r for r in late.itertuples(index=False)}
    rows = []
    for token_class in sorted(late["token_class"].unique().tolist()):
        for cname, a, b in CONTRASTS:
            if (a, token_class) not in idx or (b, token_class) not in idx:
                continue
            ra, rb = idx[(a, token_class)], idx[(b, token_class)]
            rec = {"token_class": token_class, "contrast": cname, "role_a": a, "role_b": b, "n": int(ra.n)}
            for col in ["T", "U", "N", "A_T", "A_U", "G"]:
                rec[f"delta_{col}"] = float(getattr(ra, col)) - float(getattr(rb, col))
                rec[f"{col}_a"] = float(getattr(ra, col)); rec[f"{col}_b"] = float(getattr(rb, col))
            rows.append(rec)
    con = pd.DataFrame(rows)
    con.to_csv(out / "compact_TUN_late_contrasts.csv", index=False)
    return m, late, con


def integrate_copy(dfs: dict[str, pd.DataFrame], out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    copy = dfs["copy"].copy()
    by_ck = copy.groupby(["role", "checkpoint"], dropna=False).agg(
        n=("probe_id", "count"), gain=("gain", "mean"), gain_se=("gain", lambda x: se(x)),
        repeated_nll=("repeated_nll", "mean"), unrepeated_nll=("unrepeated_nll", "mean"),
    ).reset_index()
    by_ck.to_csv(out / "copy_by_checkpoint.csv", index=False)
    late = by_ck.groupby("role", dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"), n=("n", "mean"), gain=("gain", "mean"),
        repeated_nll=("repeated_nll", "mean"), unrepeated_nll=("unrepeated_nll", "mean"),
    ).reset_index()
    late.to_csv(out / "copy_late_roles.csv", index=False)
    idx = {r.role: r for r in late.itertuples(index=False)}
    rows = []
    for cname, a, b in CONTRASTS:
        if a in idx and b in idx:
            ra, rb = idx[a], idx[b]
            rows.append({"contrast": cname, "role_a": a, "role_b": b, "n": int(ra.n),
                         "delta_gain": float(ra.gain) - float(rb.gain),
                         "gain_a": float(ra.gain), "gain_b": float(rb.gain),
                         "delta_repeated_nll": float(ra.repeated_nll) - float(rb.repeated_nll),
                         "delta_unrepeated_nll": float(ra.unrepeated_nll) - float(rb.unrepeated_nll)})
    con = pd.DataFrame(rows)
    con.to_csv(out / "copy_late_contrasts.csv", index=False)
    return late, con


def integrate_wikipedia(dfs: dict[str, pd.DataFrame], out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dfs["wiki"].copy()
    index = ["role", "checkpoint", "pair_id", "target_key", "overlap_bin", "token_class", "surface_word_class", "target_word", "target_token_id"]
    target = frame.pivot(index=index, columns="condition_code", values="nll").reset_index()
    if {"T", "U", "N"} - set(target.columns):
        raise RuntimeError("wikipedia T/U/N condition missing")
    target["gain_T_vs_U"] = target["U"] - target["T"]
    target["gain_T_vs_N"] = target["N"] - target["T"]
    target["gain_U_vs_N"] = target["N"] - target["U"]
    target.to_csv(out / "wikipedia_target_TUN_rows.csv", index=False)
    gains = ["gain_T_vs_U", "gain_T_vs_N", "gain_U_vs_N"]
    pair_ck = target.groupby(["role", "checkpoint", "pair_id", "overlap_bin", "token_class"], dropna=False).agg(
        n_targets=("target_key", "nunique"), **{g: (g, "mean") for g in gains},
    ).reset_index()
    pair_late = pair_ck.groupby(["role", "pair_id", "overlap_bin", "token_class"], dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"), n_targets=("n_targets", "mean"), **{g: (g, "mean") for g in gains},
    ).reset_index()
    pair_late = pair_late[pair_late["n_checkpoints"] == len(CKPTS)].copy()
    frames = [pair_late]
    all_class = pair_late.groupby(["role", "pair_id", "overlap_bin"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_class["token_class"] = "ALL"; frames.append(all_class)
    all_bin = pair_late.groupby(["role", "pair_id", "token_class"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_bin["overlap_bin"] = "ALL"; frames.append(all_bin)
    all_both = pair_late.groupby(["role", "pair_id"], dropna=False).agg(
        n_checkpoints=("n_checkpoints", "first"), n_targets=("n_targets", "sum"), **{g: (g, "mean") for g in gains},
    ).reset_index(); all_both["overlap_bin"] = "ALL"; all_both["token_class"] = "ALL"; frames.append(all_both)
    ext = pd.concat(frames, ignore_index=True)
    late = ext.groupby(["role", "overlap_bin", "token_class"], dropna=False).agg(
        n_pairs=("pair_id", "nunique"), **{f"mean_{g}": (g, "mean") for g in gains},
        **{f"se_pair_{g}": (g, lambda x: se(x)) for g in gains},
    ).reset_index()
    late.to_csv(out / "wikipedia_late_roles.csv", index=False)
    rows = []
    for (overlap_bin, token_class), group in ext.groupby(["overlap_bin", "token_class"], dropna=False):
        for cname, a, b in CONTRASTS:
            aa = group[group.role == a][["pair_id"] + gains]
            bb = group[group.role == b][["pair_id"] + gains]
            joined = aa.merge(bb, on="pair_id", suffixes=("_a", "_b"))
            if joined.empty:
                continue
            for gain in gains:
                diff = joined[f"{gain}_a"] - joined[f"{gain}_b"]
                rows.append({"overlap_bin": overlap_bin, "token_class": token_class, "contrast": cname, "estimand": gain,
                             "n_pairs": len(diff), "mean_difference": float(diff.mean()),
                             "se_pair_difference": se(diff), "fraction_positive": float((diff > 0).mean())})
    con = pd.DataFrame(rows)
    con.to_csv(out / "wikipedia_late_contrasts.csv", index=False)
    return target, late, con


def write_note(out: pathlib.Path, note: pathlib.Path, plan: dict[str, Any], compact_con: pd.DataFrame, wiki_con: pd.DataFrame, copy_con: pd.DataFrame) -> None:
    def cval(token_class: str, contrast: str, col: str) -> str:
        sub = compact_con[(compact_con.token_class == token_class) & (compact_con.contrast == contrast)]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0][col]):+.4f}"
    def wval(token_class: str, contrast: str, estimand: str = "gain_T_vs_N") -> str:
        sub = wiki_con[(wiki_con.overlap_bin == "ALL") & (wiki_con.token_class == token_class) & (wiki_con.contrast == contrast) & (wiki_con.estimand == estimand)]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0].mean_difference):+.4f} ± {float(sub.iloc[0].se_pair_difference):.4f}"
    def cp(contrast: str) -> str:
        sub = copy_con[copy_con.contrast == contrast]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0].delta_gain):+.4f}"
    lines = [
        "# research causal-GPT objective-boundary readout",
        "",
        f"Created: {now()}",
        "",
        "This scores the completed GPT2LMHead C/R/RS runs with a left-context next-token readout. It is not directly comparable in level to MLM T/U/N because the causal model cannot use right context; only signs and relative C/R/RS structure are used as objective-boundary evidence.",
        "",
        f"Records per model: copy {plan['records']['copy_records']}, compact T/U {plan['records']['rewrite_TU_records']}, compact N {plan['records']['rewrite_N_records']}, Wikipedia {plan['records']['wiki_records']}.",
        "",
        "## Compact FineWeb-register T/U/N, token-nonoverlap",
        "",
        "| contrast | ΔA_T | ΔA_U | ΔG |",
        "|---|---:|---:|---:|",
    ]
    for con in ["RminusC", "RSminusC", "RminusRS"]:
        lines.append(f"| {con} | {cval('nonoverlap', con, 'delta_A_T')} | {cval('nonoverlap', con, 'delta_A_U')} | {cval('nonoverlap', con, 'delta_G')} |")
    lines += [
        "",
        "## Wikipedia/Simple-English source use by target class",
        "",
        "| contrast | overlap Δ(T vs N) | nonoverlap Δ(T vs N) |",
        "|---|---:|---:|",
    ]
    for con in ["RminusC", "RSminusC", "RminusRS"]:
        lines.append(f"| {con} | {wval('overlap', con)} | {wval('nonoverlap', con)} |")
    lines += [
        "",
        "## Held-out natural-copy gain",
        "",
        "| contrast | Δ copy gain |",
        "|---|---:|",
    ]
    for con in ["RminusC", "RSminusC", "RminusRS"]:
        lines.append(f"| {con} | {cp(con)} |")
    lines += [
        "",
        "## Reading",
        "",
        "If the causal RminusC compact changed-form cost and RminusRS locality contrast match the MLM direction, exact-recurrence locality is not limited to bidirectional masking. If not, the current principle must keep objective/left-context deployment as an explicit boundary.",
        "",
        "Output CSVs are in `" + rel(out) + "`.",
    ]
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global CKPTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS.keys()))
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--copy-n-per-span", type=int, default=1000)
    ap.add_argument("--copy-span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--copy-scan-rows", type=int, default=6992)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0)
    ap.add_argument("--rewrite-tokens-per-class", type=int, default=2)
    ap.add_argument("--max-len-copy-rewrite", type=int, default=256)
    ap.add_argument("--wiki-max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=930031)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--note", type=pathlib.Path, default=DEFAULT_NOTE)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    CKPTS = list(args.checkpoints)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms = list(args.arms)
    for role in arms:
        if role not in ARMS:
            raise KeyError(role)
        for ck in CKPTS:
            p = model_path(role, ck)
            if not p.exists():
                raise FileNotFoundError(p)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path(arms[0], CKPTS[-1])), use_fast=True)
    records, rec_stats = build_probe_records(tokenizer, args)
    manifests = {role: rel(ARMS[role] / "training_manifest.json") for role in arms}
    plan = {
        "status": "CAUSAL_GPT_RELATION_BOUNDARY_PLAN",
        "created_utc": now(),
        "arms": {k: rel(ARMS[k]) for k in arms},
        "manifests": manifests,
        "checkpoints": CKPTS,
        "records": rec_stats,
        "total_records_per_model": len(records),
        "readout": "causal next-token probability at the masked target position, using only tokens before that position",
        "out_dir": rel(args.out_dir),
    }
    (args.out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id or 0)
    scored_all: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    for role in arms:
        for ck in CKPTS:
            t0 = time.time()
            mp = model_path(role, ck)
            print(f"[LOAD] {role} {ck} {mp}", flush=True)
            model = AutoModelForCausalLM.from_pretrained(str(mp), torch_dtype=torch.float32).eval().to(device)
            rows = score_records_causal(model, records, device, pad_id, args.batch_size)
            for r in rows:
                r["role"] = role; r["arm"] = role; r["checkpoint"] = ck
            scored_all.extend(rows)
            elapsed = time.time() - t0
            meta.append({"role": role, "checkpoint": ck, "records": len(rows), "elapsed_sec": round(elapsed, 2), "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {role} {ck}: {len(rows)} records in {elapsed:.1f}s", flush=True)
    write_csv(args.out_dir / "score_meta.csv", meta)
    dfs = aggregate_all(scored_all, args.out_dir)
    _m, _cl, compact_con = integrate_compact(dfs, args.out_dir)
    _copy_late, copy_con = integrate_copy(dfs, args.out_dir)
    _wt, _wl, wiki_con = integrate_wikipedia(dfs, args.out_dir)
    write_note(args.out_dir, args.note, plan, compact_con, wiki_con, copy_con)
    result = {"status": "CAUSAL_GPT_RELATION_BOUNDARY_DONE", "created_utc": now(), "out_dir": rel(args.out_dir), "note": rel(args.note), "score_rows": len(scored_all)}
    (args.out_dir / "score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
