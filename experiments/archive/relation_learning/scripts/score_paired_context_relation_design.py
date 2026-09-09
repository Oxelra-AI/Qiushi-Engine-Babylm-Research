#!/usr/bin/env python3
"""Score the inherited Qwen relation-typed composition design with source-conditioning probes.

The COMPACT_EXPERIENCE seed43022 family already contains five trained 100M DeBERTa arms:
  official_lengthmatched: no qwen-pair practice at this dose
  selected_original_dup_all: local selected original+original duplication
  qwen_clean_aligned: local selected original+own Qwen rewrite
  qwen_shuffled_control: local selected original+wrong Qwen rewrite
  qwen_separated_pair: same originals/rewrites but split across rows/windows

This script applies the source-conditioning readouts at zero training cost:
compact held-out rewrite T/U/N, WikiLarge/Simple-English T/U/N by target class,
held-out natural copy, and official Entity predictions by relevant updates.  It
uses only seed43022 and late checkpoints by default because the full five-arm
COMPACT_EXPERIENCE design is available at that seed.
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
import os
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

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_paired_context_relation_design.py')
ROOT = _PUBLIC_ROOT

WS = ROOT / "experiments/archive/relation_learning"
COMPACT_EXPERIENCE = ROOT / "experiments/archive/compact_experience"
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive/frontier_consolidation"
SCRIPTS = WS / "scripts"
sys.path.insert(0, str(SCRIPTS))
import heldout_copy_rewrite_entity_ablation as base009  # noqa: E402
import neutral_anchor_rewrite_probe as base020  # noqa: E402

WIKI_DIR = WS / "analysis/wikipedia_simplification_restatement_probe"
WIKI_RECORDS = WIKI_DIR / "wikipedia_simplification_probe_records.jsonl"
WIKI_STATS = WIKI_DIR / "probe_stats.json"
WIKI_VALIDATION = WIKI_DIR / "validation_results.json"
ENTITY_META = WS / "data/entity_relevant_update_analysis/entity_item_metadata.csv"
DEFAULT_OUT = WS / "data/paired_context_relation_design_probe"
DEFAULT_NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/paired_context_relation_design_probe.md')

CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
ARMS = {
    "OFF": {
        "role": "OFF",
        "description": "official_lengthmatched no qwen-pair practice",
        "run": COMPACT_EXPERIENCE / "training/runs/official_lengthmatched_16k_seed43022",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/official_lengthmatched.json",
    },
    "DUP": {
        "role": "DUP",
        "description": "selected_original_dup_all local selected original+original duplication",
        "run": COMPACT_EXPERIENCE / "training/runs/selected_original_dup_all_16k_seed43022",
        "per_target": COMPACT_EXPERIENCE / "data/mechanism_eval/per_target/selected_original_dup_all.json",
    },
    "SHUF": {
        "role": "SHUF",
        "description": "qwen_shuffled_control local original+wrong rewrite, correspondence broken",
        "run": COMPACT_EXPERIENCE / "training/runs/qwen_shuffled_control_16k_seed43022",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/qwen_shuffled_control.json",
    },
    "SEP": {
        "role": "SEP",
        "description": "qwen_separated_pair same originals/rewrites in separate rows/windows",
        "run": COMPACT_EXPERIENCE / "training/runs/qwen_separated_pair_16k_seed43022",
        "per_target": COMPACT_EXPERIENCE / "data/mechanism_eval/per_target/qwen_separated_pair.json",
    },
    "ALN": {
        "role": "ALN",
        "description": "qwen_clean_aligned local original+own Qwen rewrite",
        "run": COMPACT_EXPERIENCE / "training/runs/qwen_clean_aligned_16k_seed43022",
        "per_target": COMPACT_EXPERIENCE / "data/full_eval/per_target/qwen_clean_aligned.json",
    },
}
SCORE_SEED = 43022

CONTRASTS = [
    ("ALNminusOFF", "ALN", "OFF"),
    ("ALNminusSEP", "ALN", "SEP"),
    ("ALNminusSHUF", "ALN", "SHUF"),
    ("SHUFminusOFF", "SHUF", "OFF"),
    ("SEPminusOFF", "SEP", "OFF"),
    ("DUPminusOFF", "DUP", "OFF"),
    ("ALNminusDUP", "ALN", "DUP"),
    ("DUPminusALN", "DUP", "ALN"),
    ("SHUFminusSEP", "SHUF", "SEP"),
    ("SEPminusSHUF", "SEP", "SHUF"),
]


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


def fnum(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(xs) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(statistics.mean(vals)) if vals else float("nan")


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
    return ARMS[role]["run"] / "hf_model" / ck


def configure_hf_cache(cache_dir: pathlib.Path | None) -> None:
    """Set writable HF/dynamic-module caches before trusted local checkpoint loading."""
    if cache_dir is None:
        return
    cache_map = {
        "HF_HOME": cache_dir / "hf_home",
        "HF_HUB_CACHE": cache_dir / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_dir / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_dir / "transformers",
        "HF_MODULES_CACHE": cache_dir / "modules",
        "HF_DATASETS_CACHE": cache_dir / "datasets",
        "TMPDIR": cache_dir / "tmp",
    }
    for key, path in cache_map.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read_config_for_identity(mp: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads((mp / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_identity(mp: pathlib.Path, model: Any, trust_remote_code: bool, local_files_only: bool) -> dict[str, Any]:
    cfg = read_config_for_identity(mp)
    dyn_files = sorted(p.name for p in mp.glob("*modeling*.py"))
    total_params = int(sum(p.numel() for p in model.parameters()))
    adapter_params = int(sum(p.numel() for n, p in model.named_parameters() if "adapter" in n.lower()))
    return {
        "checkpoint_path": rel(mp),
        "loaded_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "total_params_loaded": total_params,
        "adapter_params_loaded": adapter_params,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "auto_map_present": bool(cfg.get("auto_map")),
        "adapter_enabled_config": cfg.get("adapter_enabled"),
        "adapter_scale_config": cfg.get("adapter_scale"),
        "adapter_bottleneck_config": cfg.get("adapter_bottleneck"),
        "dynamic_modeling_files": dyn_files,
        "trust_remote_code": bool(trust_remote_code),
        "local_files_only": bool(local_files_only),
    }


def load_masked_lm_for_scoring(mp: pathlib.Path, args: argparse.Namespace, device: torch.device) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.local_files_only:
        kwargs["local_files_only"] = True
    model = AutoModelForMaskedLM.from_pretrained(str(mp), **kwargs).eval().to(device)
    return model, model_identity(mp, model, args.trust_remote_code, args.local_files_only)


def set_dose43022_preset() -> None:
    """Replace COMPACT_EXPERIENCE arms with the adapter-scaled seed43022 base -> dose21 -> dose25 triple."""
    global ARMS, CONTRASTS, SCORE_SEED
    SCORE_SEED = 43022
    ARMS = {
        "BASE0": {
            "role": "BASE0",
            "description": "compact-view-reinvest adapter-scaled base seed43022",
            "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder",
        },
        "DOSE21": {
            "role": "DOSE21",
            "description": "probe-clean nested aligned-restatement dose21 seed43022",
            "run": WS / "training/runs/probe_clean_restatement_dose21_seed43022",
        },
        "DOSE25": {
            "role": "DOSE25",
            "description": "probe-clean nested aligned-restatement dose25 seed43022",
            "run": WS / "training/runs/probe_clean_restatement_dose25_seed43022",
        },
    }
    CONTRASTS = [
        ("DOSE21minusBASE0", "DOSE21", "BASE0"),
        ("DOSE25minusBASE0", "DOSE25", "BASE0"),
        ("DOSE25minusDOSE21", "DOSE25", "DOSE21"),
    ]


def set_dose43122_preset() -> None:
    """Replace COMPACT_EXPERIENCE arms with the adapter-scaled seed43122 base -> dose21 -> dose25 triple."""
    global ARMS, CONTRASTS, SCORE_SEED
    SCORE_SEED = 43122
    ARMS = {
        "BASE0": {
            "role": "BASE0",
            "description": "compact-view-reinvest adapter-scaled base seed43122",
            "run": REPRESENTATION_FRONTIER_STUDIES / "training/runs/adapter128_scale1p75_seed43122_dense100M",
        },
        "DOSE21": {
            "role": "DOSE21",
            "description": "probe-clean nested aligned-restatement dose21 seed43122",
            "run": WS / "training/runs/probe_clean_restatement_dose21_seed43122",
        },
        "DOSE25": {
            "role": "DOSE25",
            "description": "probe-clean nested aligned-restatement dose25 seed43122",
            "run": WS / "training/runs/probe_clean_restatement_dose25_seed43122",
        },
    }
    CONTRASTS = [
        ("DOSE21minusBASE0", "DOSE21", "BASE0"),
        ("DOSE25minusBASE0", "DOSE25", "BASE0"),
        ("DOSE25minusDOSE21", "DOSE25", "DOSE21"),
    ]


@torch.no_grad()
def score_records(model, records: list[dict[str, Any]], device: torch.device, pad_id: int, batch_size: int) -> list[dict[str, Any]]:
    """Score mixed MLM probe schemas.

    research/020 compact and copy records mask one or more positions and store
    them as `positions`/`targets`.  The Wikipedia T/U/N records mask
    exactly one token and store it as `mask_position`/`target_token_id`.  The
    first failed full run exposed this schema mismatch; keeping a local scorer
    avoids rewriting the validated record builders.
    """
    rows: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, r in enumerate(batch):
            L = len(r["input_ids"])
            ids[i, :L] = torch.tensor(r["input_ids"], dtype=torch.long)
            if "attention_mask" in r:
                att[i, :L] = torch.tensor(r["attention_mask"], dtype=torch.long)
            else:
                att[i, :L] = 1
        ids = ids.to(device)
        att = att.to(device)
        logits = model(input_ids=ids, attention_mask=att).logits.float()
        logp = torch.nn.functional.log_softmax(logits, dim=-1)
        for i, r in enumerate(batch):
            if "positions" in r and "targets" in r:
                positions = list(r["positions"])
                targets = list(r["targets"])
                hidden_keys = {"input_ids", "attention_mask", "positions", "targets"}
            else:
                positions = [int(r["mask_position"])]
                targets = [int(r["target_token_id"])]
                hidden_keys = {"input_ids", "attention_mask", "mask_position"}
            lp_sum = 0.0
            for pos, target in zip(positions, targets):
                lp_sum += float(logp[i, int(pos), int(target)].detach().cpu())
            meta = {k: v for k, v in r.items() if k not in hidden_keys}
            meta.update({
                "logprob_sum": lp_sum,
                "nll_per_token": -lp_sum / max(1, len(positions)),
                "n_masked_tokens": len(positions),
                "seq_len": len(r["input_ids"]),
            })
            rows.append(meta)
    return rows


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
    # Tag families are already distinct in aggregation by field names; keep raw schemas.
    return copy_records + rewrite_records + neutral_records + wiki_records, {
        "copy_records": len(copy_records),
        "rewrite_TU_records": len(rewrite_records),
        "rewrite_N_records": len(neutral_records),
        "wiki_records": len(wiki_records),
        "copy_stats": copy_stats,
        "rewrite_stats": rewrite_stats,
        "neutral_stats": neutral_stats,
    }


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
            copy_rows.extend(base009.aggregate_copy(csub, role, ck))
            rewrite_tu_rows.extend(base009.aggregate_rewrite(tusub, role, ck))
            for r in nsub:
                neutral_rows.append({
                    "arm": role, "role": role, "checkpoint": ck, "probe_id": r["probe_id"],
                    "pair_id": r["pair_id"], "token_class": r["token_class"],
                    "neutral_nll": r["nll_per_token"], "target_token_id": r["target_token_id"],
                    "source_token_len": r.get("source_token_len"), "rewrite_token_len": r.get("rewrite_token_len"),
                })
            for r in wsub:
                wiki_rows.append({
                    "arm": role, "role": role, "seed": SCORE_SEED, "checkpoint": ck,
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

    copy_df = pd.DataFrame(copy_rows)
    rewrite_df = pd.DataFrame(rewrite_tu_rows)
    neutral_df = pd.DataFrame(neutral_rows)
    wiki_df = pd.DataFrame(wiki_rows)
    return {"copy": copy_df, "rewrite": rewrite_df, "neutral": neutral_df, "wiki": wiki_df}


def integrate_compact(dfs: dict[str, pd.DataFrame], out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tu = dfs["rewrite"].copy()
    n = dfs["neutral"].copy()
    m = tu.merge(n[["arm", "checkpoint", "probe_id", "neutral_nll"]], on=["arm", "checkpoint", "probe_id"], how="inner")
    if len(m) != len(tu):
        raise RuntimeError(f"compact T/U/N merge lost rows: {len(m)} vs {len(tu)}")
    m["role"] = m["arm"]
    m["seed"] = SCORE_SEED
    m["A_T_N_minus_T"] = m["neutral_nll"] - m["true_source_nll"]
    m["A_U_N_minus_U"] = m["neutral_nll"] - m["unrelated_source_nll"]
    m["G_U_minus_T"] = m["unrelated_source_nll"] - m["true_source_nll"]
    m.to_csv(out / "compact_TUN_rows.csv", index=False)
    by_ck = m.groupby(["role", "checkpoint", "token_class"], dropna=False).agg(
        n=("probe_id", "count"),
        T=("true_source_nll", "mean"), U=("unrelated_source_nll", "mean"), N=("neutral_nll", "mean"),
        A_T=("A_T_N_minus_T", "mean"), A_U=("A_U_N_minus_U", "mean"), G=("G_U_minus_T", "mean"),
    ).reset_index()
    by_ck.to_csv(out / "compact_TUN_by_checkpoint.csv", index=False)
    late = by_ck[by_ck["checkpoint"].isin(CKPTS)].groupby(["role", "token_class"], dropna=False).agg(
        n_checkpoints=("checkpoint", "nunique"), n=("n", "mean"),
        T=("T", "mean"), U=("U", "mean"), N=("N", "mean"), A_T=("A_T", "mean"), A_U=("A_U", "mean"), G=("G", "mean"),
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
    if copy.empty:
        late = pd.DataFrame(columns=["role", "n_checkpoints", "n", "gain", "repeated_nll", "unrepeated_nll"])
        con = pd.DataFrame(columns=["contrast", "role_a", "role_b", "n", "delta_gain", "gain_a", "gain_b", "delta_repeated_nll", "delta_unrepeated_nll"])
        late.to_csv(out / "copy_late_roles.csv", index=False)
        con.to_csv(out / "copy_late_contrasts.csv", index=False)
        return late, con
    # base009.arm_meta cannot parse COMPACT_EXPERIENCE short arm names and writes blank role;
    # the experimental condition is reliably stored in `arm`.
    copy["role"] = copy["arm"]
    by_ck = copy.groupby(["arm", "checkpoint"], dropna=False).agg(
        n=("probe_id", "count"), gain=("gain", "mean"), gain_se=("gain", lambda x: se(x)),
        repeated_nll=("repeated_nll", "mean"), unrepeated_nll=("unrepeated_nll", "mean"),
    ).reset_index().rename(columns={"arm": "role"})
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


def entity_meta_map() -> dict[tuple[str, int], dict[str, str]]:
    with ENTITY_META.open(encoding="utf-8") as f:
        return {(r["uid"], int(r["item_index"])): r for r in csv.DictReader(f)}


def norm(s: str) -> str:
    import re
    s = str(s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def load_entity_predictions(out: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = entity_meta_map()
    rows = []
    for role, cfg in ARMS.items():
        pt = read_json(cfg["per_target"])
        pred_path = ROOT / pt["tasks"]["Entity"]["predictions"]
        obj = read_json(pred_path)
        for uid, block in obj.items():
            for i, pr in enumerate(block.get("predictions", [])):
                key = (uid, i)
                if key not in meta:
                    raise RuntimeError(f"missing entity meta {key}")
                m = meta[key]
                pred = str(pr.get("pred", ""))
                gold = str(m.get("gold", ""))
                stale = str(m.get("stale_initial", ""))
                rows.append({
                    "role": role, "uid": uid, "item_index": i, "correct": int(norm(pred) == norm(gold)),
                    "reported_numops": int(m["reported_numops"]), "relevant_updates": int(m["relevant_updates"]),
                    "total_ops": int(m["total_ops"]), "prefix_words": int(m["prefix_words"]),
                    "stale_available": int(m.get("stale_available", 0)), "stale_is_gold": int(m.get("stale_is_gold", 0)),
                    "pred_is_stale_initial": int(norm(pred) == norm(stale) and stale != ""),
                    "prediction_path": rel(pred_path),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / "entity_prediction_rows.csv", index=False)
    def groups(row):
        relu = int(row.relevant_updates); total = int(row.total_ops)
        gs = ["ALL", f"rel_updates_{relu}", f"total_ops_{total}"]
        if relu == 0:
            gs += ["rel_eq0", f"rel0_total_ops_{total}"]
        if relu >= 1: gs.append("rel_ge1")
        if relu >= 2: gs.append("rel_ge2")
        if relu >= 3: gs.append("rel_ge3")
        if relu >= 4: gs.append("rel_ge4")
        return gs
    expanded = []
    for r in df.itertuples(index=False):
        for g in groups(r):
            d = r._asdict(); d["group"] = g; expanded.append(d)
    ex = pd.DataFrame(expanded)
    summ = ex.groupby(["role", "group"], dropna=False).agg(
        n=("correct", "count"), accuracy_pct=("correct", lambda x: 100.0 * float(x.mean())),
        mean_relevant_updates=("relevant_updates", "mean"), mean_total_ops=("total_ops", "mean"), mean_prefix_words=("prefix_words", "mean"),
    ).reset_index()
    summ.to_csv(out / "entity_accuracy_by_group.csv", index=False)
    idx = {(r.role, r.group): r for r in summ.itertuples(index=False)}
    cons = []
    for g in sorted(summ["group"].unique().tolist()):
        for cname, a, b in CONTRASTS:
            if (a, g) not in idx or (b, g) not in idx:
                continue
            ra, rb = idx[(a, g)], idx[(b, g)]
            cons.append({"group": g, "contrast": cname, "role_a": a, "role_b": b, "n": int(ra.n),
                         "delta_accuracy_pct": float(ra.accuracy_pct) - float(rb.accuracy_pct),
                         "acc_a": float(ra.accuracy_pct), "acc_b": float(rb.accuracy_pct),
                         "mean_relevant_updates": float(ra.mean_relevant_updates),
                         "mean_total_ops": float(ra.mean_total_ops), "mean_prefix_words": float(ra.mean_prefix_words)})
    con = pd.DataFrame(cons)
    con.to_csv(out / "entity_contrasts_by_group.csv", index=False)
    return df, summ, con


def pick(df: pd.DataFrame, **kw) -> float:
    if df.empty:
        return float("nan")
    mask = pd.Series([True] * len(df))
    for k, v in kw.items():
        mask &= (df[k] == v)
    sub = df[mask]
    if len(sub) != 1:
        return float("nan")
    valcols = [c for c in sub.columns if c.startswith("delta_") or c == "mean_difference"]
    return float(sub.iloc[0][valcols[0]]) if valcols else float("nan")


def write_note(out: pathlib.Path, note: pathlib.Path, plan: dict[str, Any], compact_con: pd.DataFrame, wiki_con: pd.DataFrame, copy_con: pd.DataFrame, ent_con: pd.DataFrame) -> None:
    def cval(token_class: str, contrast: str, col: str) -> str:
        sub = compact_con[(compact_con.token_class == token_class) & (compact_con.contrast == contrast)]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0][col]):+.4f}"
    def wval(token_class: str, contrast: str, estimand: str = "gain_T_vs_N") -> str:
        sub = wiki_con[(wiki_con.overlap_bin == "ALL") & (wiki_con.token_class == token_class) & (wiki_con.contrast == contrast) & (wiki_con.estimand == estimand)]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0].mean_difference):+.4f} ± {float(sub.iloc[0].se_pair_difference):.4f}"
    def cp(contrast: str) -> str:
        sub = copy_con[copy_con.contrast == contrast]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0].delta_gain):+.4f}"
    def ev(group: str, contrast: str) -> str:
        sub = ent_con[(ent_con.group == group) & (ent_con.contrast == contrast)]
        return "NA" if len(sub) != 1 else f"{float(sub.iloc[0].delta_accuracy_pct):+.2f}"
    lines = [
        "# research COMPACT_EXPERIENCE qwen relation design: current-session probe readout",
        "",
        f"Created: {now()}",
        "",
        "This note scores the existing COMPACT_EXPERIENCE seed43022 relation-typed arms with the relation_learning mechanism probes. It deliberately ignores COMPACT_EXPERIENCE official-style Overall/Human-like aggregates for mechanism interpretation because AoA accounting differs across the five arms.",
        "",
        "## Arms and records",
        "",
        f"- Arms: {', '.join(ARMS.keys())}.",
        f"- Per-model records: copy {plan['records']['copy_records']}, compact T/U {plan['records']['rewrite_TU_records']}, compact N {plan['records']['rewrite_N_records']}, Wikipedia {plan['records']['wiki_records']}.",
        "- `DUP` was verified from metadata and source script as local original+original duplication: 37,594 selected pairs, 12,550 packed duplicate rows, no truncation, `cur_segments.extend([p.original, p.original])`.",
        "- `SHUF` preserves original and rewrite multisets and same-window rewrite adjacency while breaking pair correspondence for 37,594/37,594 pairs.",
        "- `SEP` preserves original/rewrite coexistence but moves sides to separate rows; its row-count/row-length mismatch means ambiguous middle outcomes need N/ordinary-heldout follow-up.",
        "",
        "## Compact FineWeb-register T/U/N, token-nonoverlap",
        "",
        "Positive `delta_A_T` means the first arm gains more from the true compact source relative to neutral N. Positive `delta_G` means larger U−T source-conditioned gain.",
        "",
        "| contrast | ΔA_T | ΔA_U | ΔG |",
        "|---|---:|---:|---:|",
    ]
    for con in ["ALNminusOFF", "ALNminusSEP", "ALNminusSHUF", "SHUFminusOFF", "SEPminusOFF", "DUPminusOFF", "ALNminusDUP"]:
        lines.append(f"| {con} | {cval('nonoverlap', con, 'delta_A_T')} | {cval('nonoverlap', con, 'delta_A_U')} | {cval('nonoverlap', con, 'delta_G')} |")
    lines += [
        "",
        "## Wikipedia/Simple-English source use by target class",
        "",
        "Positive values mean the first arm extracts more true-source benefit than the second arm. Overlap targets recur as tokenizer IDs in the English-Wikipedia source; nonoverlap targets are absent from it.",
        "",
        "| contrast | overlap Δ(T vs N) | nonoverlap Δ(T vs N) |",
        "|---|---:|---:|",
    ]
    for con in ["ALNminusOFF", "ALNminusSEP", "ALNminusSHUF", "SHUFminusOFF", "SEPminusOFF", "DUPminusOFF", "ALNminusDUP"]:
        lines.append(f"| {con} | {wval('overlap', con)} | {wval('nonoverlap', con)} |")
    lines += [
        "",
        "## Held-out natural-copy gain",
        "",
        "| contrast | Δ copy gain |",
        "|---|---:|",
    ]
    for con in ["DUPminusOFF", "DUPminusALN", "ALNminusOFF", "ALNminusSHUF", "ALNminusSEP"]:
        lines.append(f"| {con} | {cp(con)} |")
    lines += [
        "",
        "## Entity by relevant queried-state updates",
        "",
        "These are computed from existing per-target official Entity predictions, not from aggregate Entity scores. Values are accuracy-point differences at the 100M checkpoint.",
        "",
        "| contrast | rel_eq0 | rel_ge3 |",
        "|---|---:|---:|",
    ]
    for con in ["ALNminusOFF", "ALNminusSEP", "ALNminusSHUF", "DUPminusOFF", "DUPminusALN"]:
        lines.append(f"| {con} | {ev('rel_eq0', con)} | {ev('rel_ge3', con)} |")
    lines += [
        "",
        "## Scientific reading from this run",
        "",
        "The compact readout is the cross-register test of whether the COMPACT_EXPERIENCE qwen relation practice installs the same FineWeb-compact source-use routine as the designed compact VIEW/REPEAT family. The Wikipedia readout is nearer to the qwen substrate because the inherited block includes many SimpleWiki pairs. The shuffled arm is the crucial new correspondence control: aligned-minus-shuffled isolates own-source correspondence while keeping same-window rewrite-register adjacency. If SHUF falls below OFF on `A_T=N-T` while `A_U=N-U` is flat or higher, that is not inert adjacency: it is a practiced non-correspondence/discounting routine, separated from general fit by N and ordinary held-out loss.",
        "",
        "Use the CSV files for exact rows and additional overlap bins before making stronger statements.",
        "",
        "## Output files",
        "",
        f"- `{rel(out / 'compact_TUN_late_contrasts.csv')}`",
        f"- `{rel(out / 'wikipedia_late_contrasts.csv')}`",
        f"- `{rel(out / 'copy_late_contrasts.csv')}`",
        f"- `{rel(out / 'entity_contrasts_by_group.csv')}`",
    ]
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global CKPTS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=["compact_experience", "dose43022", "dose43122"], default="compact_experience")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--checkpoints", nargs="+", default=CKPTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--copy-n-per-span", type=int, default=1000)
    ap.add_argument("--copy-span-lengths", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--copy-scan-rows", type=int, default=6992)
    ap.add_argument("--rewrite-max-pairs", type=int, default=0)
    ap.add_argument("--rewrite-tokens-per-class", type=int, default=2)
    ap.add_argument("--max-len-copy-rewrite", type=int, default=256)
    ap.add_argument("--wiki-max-records", type=int, default=0)
    ap.add_argument("--seed", type=int, default=930030)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--note", type=pathlib.Path, default=DEFAULT_NOTE)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true", help="Use trusted custom checkpoint code; required for adapter-scaled compact-view lineage.")
    ap.add_argument("--local-files-only", action="store_true", help="Load checkpoints without network access.")
    ap.add_argument("--hf-cache-dir", type=pathlib.Path, default=None, help="Writable HF cache/modules directory used before trusted loading.")
    ap.add_argument("--skip-entity", action="store_true", help="Skip official Entity prediction aggregation for arms without per-target JSON.")
    ap.add_argument("--skip-note", action="store_true", help="Write CSV/JSON outputs without the COMPACT_EXPERIENCE-specific note.")
    args = ap.parse_args()

    if args.preset == "dose43022":
        set_dose43022_preset()
    elif args.preset == "dose43122":
        set_dose43122_preset()
    configure_hf_cache(args.hf_cache_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arms = list(args.arms) if args.arms else list(ARMS.keys())
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise KeyError(unknown)
    CKPTS = list(args.checkpoints)
    for role in arms:
        for ck in CKPTS:
            p = model_path(role, ck)
            if not p.exists():
                raise FileNotFoundError(p)
        if not args.skip_entity and ARMS[role].get("per_target") is not None and not ARMS[role]["per_target"].exists():
            raise FileNotFoundError(ARMS[role]["per_target"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_path(arms[0], CKPTS[-1]).parent), use_fast=True)
    records, rec_stats = build_probe_records(tokenizer, args)
    plan = {
        "status": "RELATION_PROBE_SCORE_PLAN",
        "created_utc": now(),
        "preset": args.preset,
        "arms": {k: {"description": ARMS[k]["description"], "run": rel(ARMS[k]["run"]), "per_target": (rel(ARMS[k]["per_target"]) if ARMS[k].get("per_target") is not None else None)} for k in arms},
        "checkpoints": CKPTS,
        "records": rec_stats,
        "total_records_per_model": len(records),
        "out_dir": rel(args.out_dir),
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "hf_cache_dir": None if args.hf_cache_dir is None else rel(args.hf_cache_dir),
    }
    (args.out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False)); return

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    scored_all: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for role in arms:
        for ck in CKPTS:
            t0 = time.time()
            mp = model_path(role, ck)
            print(f"[LOAD] {role} {ck} {mp}", flush=True)
            model, ident = load_masked_lm_for_scoring(mp, args, device)
            ident.update({"role": role, "checkpoint": ck})
            identities.append(ident)
            print(json.dumps({"event": "model_identity", **ident}, ensure_ascii=False), flush=True)
            rows = score_records(model, records, device, pad_id, args.batch_size)
            for r in rows:
                r["role"] = role; r["arm"] = role; r["checkpoint"] = ck
            scored_all.extend(rows)
            elapsed = time.time() - t0
            meta.append({"role": role, "checkpoint": ck, "records": len(rows), "elapsed_sec": round(elapsed, 2), "device": str(device), "loaded_class": ident.get("loaded_class"), "total_params_loaded": ident.get("total_params_loaded"), "adapter_params_loaded": ident.get("adapter_params_loaded"), "adapter_scale_config": ident.get("adapter_scale_config")})
            del model
            if device.type == "cuda": torch.cuda.empty_cache()
            print(f"[DONE] {role} {ck}: {len(rows)} records in {elapsed:.1f}s", flush=True)
    write_csv(args.out_dir / "score_meta.csv", meta)
    write_csv(args.out_dir / "model_identity_preamble.csv", identities)
    with (args.out_dir / "model_identity_preamble.jsonl").open("w", encoding="utf-8") as f:
        for ident in identities:
            f.write(json.dumps(ident, ensure_ascii=False) + "\n")
    dfs = aggregate_all(scored_all, args.out_dir)
    _m, compact_late, compact_con = integrate_compact(dfs, args.out_dir)
    copy_late, copy_con = integrate_copy(dfs, args.out_dir)
    _wt, wiki_late, wiki_con = integrate_wikipedia(dfs, args.out_dir)
    if args.skip_entity:
        entity_summary = pd.DataFrame()
        entity_con = pd.DataFrame(columns=["group", "contrast", "delta_accuracy_pct"])
    else:
        _er, entity_summary, entity_con = load_entity_predictions(args.out_dir)
    if not args.skip_note:
        write_note(args.out_dir, args.note, plan, compact_con, wiki_con, copy_con, entity_con)
    result = {"status": "RELATION_PROBE_SCORE_DONE", "created_utc": now(), "preset": args.preset, "out_dir": rel(args.out_dir), "note": (None if args.skip_note else rel(args.note)), "score_rows": len(scored_all), "identity_preamble": rel(args.out_dir / "model_identity_preamble.jsonl")}
    (args.out_dir / "score_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
