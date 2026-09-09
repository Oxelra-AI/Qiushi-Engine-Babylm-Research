#!/usr/bin/env python3
"""research: score ordinary-fit axes for the cleaned nested restatement dose curve.

The scorer extends the research base scorer in three ways that matter for the
post-leakage dose experiment:

1. it reads base -> dose21 -> dose25 ordered triples at chck_80M, chck_90M,
   and chck_100M rather than using a single endpoint;
2. it separates the research 2,647-row axis into the 1,743 no-inherited-ALN-text
   rows and the complementary 904 hard/selectable-register rows;
3. it adds the research Strict-complement n-gram axis, a same-source broad-fit
   axis sampled from the larger 2026 Strict corpus after same-source 16-token
   Strict-Small exclusion and exact stream-row equality scanning.

The output is an ordinary-language fit measurement under deterministic masking.
It is not the practiced-class source-conditioned readout; those faces must be
scored separately.
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
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_dose_ordinary_fit_axes.py')
ROOT = _PUBLIC_ROOT

REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
WS = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = WS / "data/dose_ordinary_fit_axes"
MAX_LEN = 256
MASK_PROB = 0.15

SUBSETS: dict[str, pathlib.Path] = {
    "heldout2647_full": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl",
    "heldout2647_no_inherited_aln_text": WS / "data/current_dose_stream_heldout2647_exposure/heldout2647_no_inherited_aln_pair_text_hits.jsonl",
    "heldout2647_inherited_aln_text_hit": WS / "data/current_dose_stream_heldout2647_exposure/heldout2647_inherited_aln_pair_text_hit_rows.jsonl",
    "strict_complement_ngram_3000": WS / "data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl",
}

ARM_CONFIGS: dict[str, dict[str, Any]] = {
    "base43022": {
        "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_h100M100M_seed43022_official_ladder",
        "seed": 43022,
        "dose": "base0",
        "family": "compact_view_reinvest",
    },
    "base43122": {
        "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_seed43122_dense100M",
        "seed": 43122,
        "dose": "base0",
        "family": "compact_view_reinvest",
    },
    "dose21_43022": {
        "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose21_seed43022",
        "seed": 43022,
        "dose": "dose21",
        "family": "probe_clean_restatement_dose",
        "expected_stream_sha256": "80f8151764cc0a4865a0a8dfc80e092cfba56186eeb41997a96c3bc7db057779",
    },
    "dose25_43022": {
        "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43022",
        "seed": 43022,
        "dose": "dose25",
        "family": "probe_clean_restatement_dose",
        "expected_stream_sha256": "3949e70367f9455a382f5e2e48a8500c8eb8f08cb5ad9454a34c6bfc93af5924",
    },
    "dose21_43122": {
        "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose21_seed43122",
        "seed": 43122,
        "dose": "dose21",
        "family": "probe_clean_restatement_dose",
        "expected_stream_sha256": "80f8151764cc0a4865a0a8dfc80e092cfba56186eeb41997a96c3bc7db057779",
    },
    "dose25_43122": {
        "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43122",
        "seed": 43122,
        "dose": "dose25",
        "family": "probe_clean_restatement_dose",
        "expected_stream_sha256": "3949e70367f9455a382f5e2e48a8500c8eb8f08cb5ad9454a34c6bfc93af5924",
    },
}
DEFAULT_CKPTS = ["chck_80M", "chck_90M", "chck_100M"]


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


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    for i, obj in enumerate(read_jsonl(path)):
        # research/061 rows and research rows both carry stable example_id values.
        # Keep a fallback only to make dry-run inspection robust.
        exid = obj.get("example_id", obj.get("axis_row_index", i))
        rows.append({
            "example_id": int(exid),
            "text": str(obj["text"]),
            "source": str(obj.get("source", obj.get("origin_file", ""))),
            "words": int(obj.get("words", len(str(obj.get("text", "")).split()))),
        })
    return rows


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def configure_hf_cache(cache_dir: pathlib.Path | None) -> dict[str, str]:
    if cache_dir is None:
        return {}
    mapping = {
        "HF_HOME": cache_dir / "hf_home",
        "HF_HUB_CACHE": cache_dir / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache_dir / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache_dir / "transformers",
        "HF_MODULES_CACHE": cache_dir / "modules",
        "HF_DATASETS_CACHE": cache_dir / "datasets",
        "TMPDIR": cache_dir / "tmp",
    }
    out: dict[str, str] = {}
    for key, value in mapping.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value.resolve())
        out[key] = str(value.resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return out


def model_identity(mp: pathlib.Path, model: Any, trust_remote_code: bool, local_files_only: bool) -> dict[str, Any]:
    cfg = load_json(mp / "config.json") or {}
    total_params = int(sum(p.numel() for p in model.parameters()))
    adapter_params = int(sum(p.numel() for name, p in model.named_parameters() if "adapter" in name.lower()))
    dyn_files = sorted(p.name for p in mp.glob("*modeling*.py"))
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


def load_model_for_scoring(mp: pathlib.Path, args: argparse.Namespace, device: torch.device) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32}
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.local_files_only:
        kwargs["local_files_only"] = True
    model = AutoModelForMaskedLM.from_pretrained(str(mp), **kwargs).eval().to(device)
    return model, model_identity(mp, model, bool(args.trust_remote_code), bool(args.local_files_only))


def model_file_ready(path: pathlib.Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def arm_complete_enough(arm: str, checkpoints: list[str], require_complete_run: bool) -> tuple[bool, list[str], dict[str, Any]]:
    cfg = ARM_CONFIGS[arm]
    run = pathlib.Path(cfg["run_dir"])
    missing: list[str] = []
    for ck in checkpoints:
        d = run / "hf_model" / ck
        if not model_file_ready(d):
            missing.append(rel(d))
    hf_root = run / "hf_model"
    if require_complete_run and not model_file_ready(hf_root):
        missing.append(rel(hf_root) + " (final model)")
    sci = load_json(run / "scientific_metrics.json")
    launch = load_json(run / "launch_command.json")
    manifest = load_json(run / "example_order_manifest.json")
    info = {
        "arm": arm,
        "run_dir": rel(run),
        "has_scientific_metrics": sci is not None,
        "word_exposure": None if sci is None else sci.get("word_exposure"),
        "loss_last": None if sci is None else sci.get("loss_last"),
        "saved_checkpoints": None if sci is None else len(sci.get("saved_checkpoints", [])),
        "manifest_words": None if manifest is None else manifest.get("selected_for_training_words"),
        "manifest_rows": None if manifest is None else manifest.get("num_consumed_examples"),
        "launch_stream_sha256": None if launch is None else launch.get("stream_sha256"),
        "expected_stream_sha256": cfg.get("expected_stream_sha256"),
    }
    if require_complete_run:
        if sci is None:
            missing.append(rel(run / "scientific_metrics.json"))
        else:
            if int(sci.get("word_exposure") or 0) < 100_000_000:
                missing.append(f"{arm} word_exposure={sci.get('word_exposure')}")
        if manifest is not None:
            if int(manifest.get("selected_for_training_words") or 0) != 100_000_000:
                missing.append(f"{arm} manifest words={manifest.get('selected_for_training_words')}")
            if int(manifest.get("num_consumed_examples") or 0) != 647_400:
                missing.append(f"{arm} manifest rows={manifest.get('num_consumed_examples')}")
        if cfg.get("expected_stream_sha256") and launch is not None and launch.get("stream_sha256") != cfg.get("expected_stream_sha256"):
            missing.append(f"{arm} stream sha mismatch {launch.get('stream_sha256')}")
    return len(missing) == 0, missing, info


def wait_for_arms(arms: list[str], checkpoints: list[str], timeout_sec: int, poll_sec: int, require_complete_run: bool, out_dir: pathlib.Path) -> None:
    start = time.time()
    last_missing: dict[str, list[str]] | None = None
    events = out_dir / "wait_events.jsonl"
    while True:
        missing_map: dict[str, list[str]] = {}
        info_rows: list[dict[str, Any]] = []
        for arm in arms:
            ok, miss, info = arm_complete_enough(arm, checkpoints, require_complete_run=require_complete_run)
            info_rows.append({"ok": ok, **info})
            if not ok:
                missing_map[arm] = miss
        if missing_map != last_missing:
            ev = {"created_utc": now(), "event": "wait_status", "missing": missing_map, "arm_info": info_rows}
            with events.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            print(json.dumps(ev, ensure_ascii=False), flush=True)
            last_missing = missing_map
        if not missing_map:
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError(json.dumps({"missing": missing_map, "arm_info": info_rows}, ensure_ascii=False))
        time.sleep(max(5, int(poll_sec)))


def apply_fixed_mask(input_ids: torch.Tensor, example_id: int, mask_token_id: int, special_ids: set[int], pad_id: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.RandomState(seed=int(example_id) % (2**31))
    maskable = torch.ones(input_ids.shape[0], dtype=torch.bool)
    for sid in special_ids:
        maskable &= input_ids != int(sid)
    maskable &= input_ids != int(pad_id)
    probs = rng.random(input_ids.shape[0])
    mask = torch.tensor(probs < MASK_PROB, dtype=torch.bool) & maskable
    if mask.sum() == 0 and maskable.sum() > 0:
        mask[maskable.nonzero(as_tuple=True)[0][0].item()] = True
    labels = input_ids.clone()
    labels[~mask] = -100
    masked = input_ids.clone()
    decisions = rng.random(input_ids.shape[0])
    replace_mask = mask & torch.tensor(decisions < 0.8, dtype=torch.bool)
    random_mask = mask & torch.tensor((decisions >= 0.8) & (decisions < 0.9), dtype=torch.bool)
    masked[replace_mask] = int(mask_token_id)
    if random_mask.sum() > 0:
        repl = torch.tensor(rng.randint(0, vocab_size, size=int(random_mask.sum())), dtype=torch.long)
        masked[random_mask] = repl
    return masked, labels, int(mask.sum().item())


def special_id_set(tok) -> set[int]:
    out: set[int] = set(int(x) for x in tok.all_special_ids if x is not None)
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"]:
        x = getattr(tok, attr, None)
        if x is not None:
            out.add(int(x))
    return out


@torch.no_grad()
def score_rows(model, tok, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    mask_id = int(tok.mask_token_id)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    special = special_id_set(tok)
    vocab_size = int(getattr(tok, "vocab_size", len(tok)))
    out: list[dict[str, Any]] = []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    for st in range(0, len(rows), batch_size):
        batch = rows[st:st+batch_size]
        enc = tok([r["text"] for r in batch], max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        ids0 = enc["input_ids"]
        att = enc["attention_mask"].to(device)
        mids = []
        labs = []
        nm = []
        for i, r in enumerate(batch):
            mi, lab, n = apply_fixed_mask(ids0[i].clone(), int(r["example_id"]), mask_id, special, pad_id, vocab_size)
            mids.append(mi); labs.append(lab); nm.append(n)
        mids_t = torch.stack(mids).to(device)
        labs_t = torch.stack(labs).to(device)
        logits = model(input_ids=mids_t, attention_mask=att).logits.float()
        B, L, V = logits.shape
        losses = loss_fn(logits.view(B*L, V), labs_t.view(B*L)).view(B, L)
        for i, r in enumerate(batch):
            mpos = labs_t[i] != -100
            loss = float(losses[i][mpos].mean().detach().cpu()) if bool(mpos.any()) else float("nan")
            out.append({"example_id": int(r["example_id"]), "source": r["source"], "words": int(r["words"]), "loss": loss, "n_masked": int(nm[i])})
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame(rows)
    by_ck = df.groupby(["subset", "arm", "dose", "seed", "checkpoint"], dropna=False).agg(
        n=("example_id", "count"), mean_loss=("loss", "mean"), sd_loss=("loss", "std"), mean_masked=("n_masked", "mean")
    ).reset_index()
    late = by_ck.groupby(["subset", "arm", "dose", "seed"], dropna=False).agg(
        n_min=("n", "min"), mean_loss=("mean_loss", "mean"), sd_across_checkpoints=("mean_loss", "std"), mean_masked=("mean_masked", "mean")
    ).reset_index()

    # Ordered base -> dose21 -> dose25 triples at each checkpoint and in late average.
    triple_rows: list[dict[str, Any]] = []
    for (subset, seed, ck), g in by_ck.groupby(["subset", "seed", "checkpoint"], dropna=False):
        idx = g.set_index("dose")
        row = {"subset": subset, "seed": int(seed), "checkpoint": ck}
        for dose in ["base0", "dose21", "dose25"]:
            row[f"{dose}_loss"] = float(idx.loc[dose, "mean_loss"]) if dose in idx.index else float("nan")
        row["dose21_minus_base0"] = row["dose21_loss"] - row["base0_loss"] if math.isfinite(row["dose21_loss"]) and math.isfinite(row["base0_loss"]) else float("nan")
        row["dose25_minus_base0"] = row["dose25_loss"] - row["base0_loss"] if math.isfinite(row["dose25_loss"]) and math.isfinite(row["base0_loss"]) else float("nan")
        row["dose25_minus_dose21"] = row["dose25_loss"] - row["dose21_loss"] if math.isfinite(row["dose25_loss"]) and math.isfinite(row["dose21_loss"]) else float("nan")
        triple_rows.append(row)
    triples = pd.DataFrame(triple_rows)

    late_triple_rows: list[dict[str, Any]] = []
    for (subset, seed), g in late.groupby(["subset", "seed"], dropna=False):
        idx = g.set_index("dose")
        row = {"subset": subset, "seed": int(seed), "checkpoint": "late_mean_80_90_100"}
        for dose in ["base0", "dose21", "dose25"]:
            row[f"{dose}_loss"] = float(idx.loc[dose, "mean_loss"]) if dose in idx.index else float("nan")
            row[f"{dose}_sd_across_checkpoints"] = float(idx.loc[dose, "sd_across_checkpoints"]) if dose in idx.index and pd.notna(idx.loc[dose, "sd_across_checkpoints"]) else float("nan")
        row["dose21_minus_base0"] = row["dose21_loss"] - row["base0_loss"] if math.isfinite(row["dose21_loss"]) and math.isfinite(row["base0_loss"]) else float("nan")
        row["dose25_minus_base0"] = row["dose25_loss"] - row["base0_loss"] if math.isfinite(row["dose25_loss"]) and math.isfinite(row["base0_loss"]) else float("nan")
        row["dose25_minus_dose21"] = row["dose25_loss"] - row["dose21_loss"] if math.isfinite(row["dose25_loss"]) and math.isfinite(row["dose21_loss"]) else float("nan")
        late_triple_rows.append(row)
    late_triples = pd.DataFrame(late_triple_rows)

    rowpair_rows: list[dict[str, Any]] = []
    for subset, g in df.groupby("subset"):
        # checkpoint-averaged row losses, then paired contrasts within seed.
        wide = g.groupby(["example_id", "source", "words", "seed", "dose"], dropna=False)["loss"].mean().reset_index()
        for seed, gs in wide.groupby("seed"):
            piv = gs.pivot(index=["example_id", "source", "words"], columns="dose", values="loss")
            if "base0" not in piv.columns:
                continue
            for dose in ["dose21", "dose25"]:
                if dose in piv.columns:
                    vals = (piv[dose] - piv["base0"]).dropna().to_numpy(dtype=float)
                    if len(vals):
                        rowpair_rows.append({
                            "subset": subset,
                            "seed": int(seed),
                            "contrast": f"{dose}-base0",
                            "n_rows": int(len(vals)),
                            "mean_delta_loss": float(vals.mean()),
                            "median_delta_loss": float(np.median(vals)),
                            "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                        })
            if "dose21" in piv.columns and "dose25" in piv.columns:
                vals = (piv["dose25"] - piv["dose21"]).dropna().to_numpy(dtype=float)
                if len(vals):
                    rowpair_rows.append({
                        "subset": subset,
                        "seed": int(seed),
                        "contrast": "dose25-dose21",
                        "n_rows": int(len(vals)),
                        "mean_delta_loss": float(vals.mean()),
                        "median_delta_loss": float(np.median(vals)),
                        "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else float("nan"),
                    })
    pair = pd.DataFrame(rowpair_rows)
    return {"by_checkpoint": by_ck, "late_summary": late, "ordered_triples_by_checkpoint": triples, "ordered_triples_late": late_triples, "rowpaired_contrasts": pair}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arms", nargs="*", default=["base43022", "dose21_43022", "dose25_43022"])
    ap.add_argument("--subsets", nargs="*", default=[
        "heldout2647_full",
        "heldout2647_no_inherited_aln_text",
        "heldout2647_inherited_aln_text_hit",
        "strict_complement_ngram_3000",
    ])
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKPTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--trust-remote-code", action="store_true", help="Load adapter-scaled checkpoints through their local custom modeling file.")
    ap.add_argument("--local-files-only", action="store_true", help="Do not fetch remote model code or weights during scoring.")
    ap.add_argument("--hf-cache-dir", default="", help="Writable Hugging Face and dynamic-module cache directory; set this in the shell before Python import as well for adapter models.")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--wait-timeout-sec", type=int, default=8 * 3600)
    ap.add_argument("--wait-poll-sec", type=int, default=60)
    ap.add_argument("--wait-complete-runs", action="store_true", help="When waiting, require 100M scientific_metrics and final model in addition to requested checkpoints.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_env = configure_hf_cache(pathlib.Path(args.hf_cache_dir) if args.hf_cache_dir else None)

    unknown_arms = [a for a in args.arms if a not in ARM_CONFIGS]
    unknown_subsets = [s for s in args.subsets if s not in SUBSETS]
    if unknown_arms or unknown_subsets:
        raise SystemExit(f"unknown arms={unknown_arms} subsets={unknown_subsets}")

    row_cache = {name: load_rows(SUBSETS[name]) for name in args.subsets}
    if args.wait:
        wait_for_arms(args.arms, list(args.checkpoints), args.wait_timeout_sec, args.wait_poll_sec, args.wait_complete_runs, out_dir)

    missing = []
    arm_infos = []
    for arm in args.arms:
        ok, miss, info = arm_complete_enough(arm, list(args.checkpoints), require_complete_run=False)
        arm_infos.append({"ok": ok, **info})
        missing.extend(miss)
    plan = {
        "status": "DOSE_ORDINARY_FIT_AXES_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "arm_info": arm_infos,
        "checkpoints": args.checkpoints,
        "subsets": {k: {"path": rel(SUBSETS[k]), "rows": len(v), "words": int(sum(r["words"] for r in v))} for k, v in row_cache.items()},
        "missing": missing,
        "out_dir": rel(out_dir),
        "device": args.device,
        "max_len": MAX_LEN,
        "mask_prob": MASK_PROB,
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "hf_cache_env": cache_env,
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]]["run_dir"] / "hf_model"), use_fast=True)
    all_rows: list[dict[str, Any]] = []
    meta: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for arm in args.arms:
        cfg = ARM_CONFIGS[arm]
        run = pathlib.Path(cfg["run_dir"])
        for ck in args.checkpoints:
            mp = run / "hf_model" / ck
            t0 = time.time()
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model, ident = load_model_for_scoring(mp, args, device)
            ident.update({"arm": arm, "dose": cfg["dose"], "seed": cfg["seed"], "checkpoint": ck})
            identity_rows.append(ident)
            print(json.dumps({"event": "model_identity", **ident}, ensure_ascii=False), flush=True)
            for subset_name in args.subsets:
                rows = row_cache[subset_name]
                scored = score_rows(model, tok, rows, device, args.batch_size)
                mean_loss = statistics.mean([x["loss"] for x in scored])
                for r in scored:
                    q = dict(r)
                    q.update({
                        "subset": subset_name,
                        "subset_path": rel(SUBSETS[subset_name]),
                        "arm": arm,
                        "dose": cfg["dose"],
                        "seed": cfg["seed"],
                        "family": cfg["family"],
                        "checkpoint": ck,
                    })
                    all_rows.append(q)
                meta.append({
                    "arm": arm,
                    "dose": cfg["dose"],
                    "seed": cfg["seed"],
                    "checkpoint": ck,
                    "subset": subset_name,
                    "n": len(scored),
                    "mean_loss": mean_loss,
                    "elapsed_sec_since_model_load": round(time.time() - t0, 2),
                    "device": str(device),
                    "loaded_class": ident.get("loaded_class"),
                    "total_params_loaded": ident.get("total_params_loaded"),
                    "adapter_params_loaded": ident.get("adapter_params_loaded"),
                    "adapter_scale_config": ident.get("adapter_scale_config"),
                })
                print(f"[DONE] {arm} {ck} {subset_name} mean={mean_loss:.6f} n={len(scored)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(out_dir / "ordinary_fit_rows.csv", all_rows)
    write_csv(out_dir / "ordinary_fit_score_meta.csv", meta)
    with (out_dir / "model_identity_preamble.jsonl").open("w", encoding="utf-8") as f:
        for row in identity_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summaries = summarize(all_rows)
    for name, df in summaries.items():
        df.to_csv(out_dir / f"ordinary_fit_{name}.csv", index=False)
    result = {
        "status": "DOSE_ORDINARY_FIT_AXES_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "rows_scored": len(all_rows),
        "model_identity_preamble": rel(out_dir / "model_identity_preamble.jsonl"),
        "trust_remote_code": bool(args.trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "late_summary": json.loads(summaries["late_summary"].to_json(orient="records")),
        "ordered_triples_late": json.loads(summaries["ordered_triples_late"].to_json(orient="records")) if not summaries["ordered_triples_late"].empty else [],
        "rowpaired_contrasts": json.loads(summaries["rowpaired_contrasts"].to_json(orient="records")) if not summaries["rowpaired_contrasts"].empty else [],
        "interpretation_note": (
            "Read each subset within seed as base0 -> dose21 -> dose25 at chck_80M, chck_90M, chck_100M and in the late mean. "
            "The 1,743 rows are outside inherited ALN pair-text exposure; the 904 rows are the harder restatement-selectable complement; "
            "the Strict-complement axis is source-matched larger-Strict text after same-source 16-token Strict-Small exclusion and exact stream-row equality scanning."
        ),
    }
    (out_dir / "ordinary_fit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
