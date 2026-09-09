#!/usr/bin/env python3
"""research: score compact-view-reinvest ordinary-fit subsets.

This scorer uses the same deterministic-mask MLM measurement as research, but on
subsets that are meaningful for the current compact-view-reinvest dose curve.
The research 2,647-row set is absent as exact stream rows from the base and the
repaired up-dose streams; the 1,743-row subset additionally removes rows whose
text contains inherited COMPACT_EXPERIENCE ALN source/rewrite material.  These rows are the
ordinary-fit ordinate for the dose curve, while the later 6,992-row research set
is not a clean heldout set for this substrate.
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
import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/score_compactview_ordinary_fit_subsets.py')
ROOT = _PUBLIC_ROOT

REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
WS = ROOT / "experiments/archive/relation_learning"
OUT_DEFAULT = WS / "data/compactview_ordinary_fit_subsets"
MAX_LEN = 256
MASK_PROB = 0.15

SUBSETS = {
    "heldout2647_full": ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/heldout_cleanqwen_rows.jsonl",
    "heldout2647_no_inherited_aln_text": WS / "data/current_dose_stream_heldout2647_exposure/heldout2647_no_inherited_aln_pair_text_hits.jsonl",
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
    },
    "dose25_43022": {
        "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43022",
        "seed": 43022,
        "dose": "dose25",
        "family": "probe_clean_restatement_dose",
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
    for obj in read_jsonl(path):
        rows.append({
            "example_id": int(obj["example_id"]),
            "text": str(obj["text"]),
            "source": str(obj.get("source", "")),
            "words": int(obj.get("words", len(str(obj.get("text", "")).split()))),
        })
    return rows


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
    vocab_size = int(getattr(tok, "vocab_size", 16384))
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


def summarize(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows)
    by_ck = df.groupby(["subset", "arm", "dose", "seed", "checkpoint"], dropna=False).agg(
        n=("example_id", "count"), mean_loss=("loss", "mean"), sd_loss=("loss", "std"), mean_masked=("n_masked", "mean")
    ).reset_index()
    late = by_ck.groupby(["subset", "arm", "dose", "seed"], dropna=False).agg(
        n_min=("n", "min"), mean_loss=("mean_loss", "mean"), mean_masked=("mean_masked", "mean")
    ).reset_index()

    contrasts = []
    for subset, g in late.groupby("subset"):
        for seed, gs in g.groupby("seed"):
            idx = gs.set_index("dose")
            if "base0" not in idx.index:
                continue
            base = idx.loc["base0"]
            for dose in ["dose21", "dose25"]:
                if dose in idx.index:
                    r = idx.loc[dose]
                    contrasts.append({
                        "subset": subset,
                        "seed": int(seed),
                        "contrast": f"{dose}-base0",
                        "delta_loss": float(r["mean_loss"] - base["mean_loss"]),
                        "loss_a": float(r["mean_loss"]),
                        "loss_b": float(base["mean_loss"]),
                        "n_min": int(min(r["n_min"], base["n_min"])),
                    })
            if "dose21" in idx.index and "dose25" in idx.index:
                a = idx.loc["dose25"]; b = idx.loc["dose21"]
                contrasts.append({
                    "subset": subset,
                    "seed": int(seed),
                    "contrast": "dose25-dose21",
                    "delta_loss": float(a["mean_loss"] - b["mean_loss"]),
                    "loss_a": float(a["mean_loss"]),
                    "loss_b": float(b["mean_loss"]),
                    "n_min": int(min(a["n_min"], b["n_min"])),
                })
    con = pd.DataFrame(contrasts)

    rowpair_rows = []
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
    return by_ck, late, con, pair


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arms", nargs="*", default=["base43022", "base43122"])
    ap.add_argument("--subsets", nargs="*", default=["heldout2647_full", "heldout2647_no_inherited_aln_text"])
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT_CKPTS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_cache = {name: load_rows(SUBSETS[name]) for name in args.subsets}
    missing = []
    for arm in args.arms:
        run = ARM_CONFIGS[arm]["run_dir"]
        for ck in args.checkpoints:
            d = run / "hf_model" / ck
            if not ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
                missing.append(rel(d))
    plan = {
        "status": "COMPACTVIEW_ORDINARY_FIT_SUBSETS_PLAN",
        "created_utc": now(),
        "arms": args.arms,
        "checkpoints": args.checkpoints,
        "subsets": {k: {"path": rel(SUBSETS[k]), "rows": len(v)} for k, v in row_cache.items()},
        "missing": missing,
        "out_dir": rel(out_dir),
        "device": args.device,
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
    for arm in args.arms:
        cfg = ARM_CONFIGS[arm]
        run = cfg["run_dir"]
        for ck in args.checkpoints:
            mp = run / "hf_model" / ck
            t0 = time.time()
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32)
            model.eval().to(device)
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
                })
                print(f"[DONE] {arm} {ck} {subset_name} mean={mean_loss:.6f} n={len(scored)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(out_dir / "ordinary_fit_rows.csv", all_rows)
    write_csv(out_dir / "ordinary_fit_score_meta.csv", meta)
    by_ck, late, con, pair = summarize(all_rows)
    by_ck.to_csv(out_dir / "ordinary_fit_by_checkpoint.csv", index=False)
    late.to_csv(out_dir / "ordinary_fit_late_summary.csv", index=False)
    con.to_csv(out_dir / "ordinary_fit_late_contrasts.csv", index=False)
    pair.to_csv(out_dir / "ordinary_fit_rowpaired_contrasts.csv", index=False)
    result = {
        "status": "COMPACTVIEW_ORDINARY_FIT_SUBSETS_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "rows_scored": len(all_rows),
        "late_summary": json.loads(late.to_json(orient="records")),
        "contrasts": json.loads(con.to_json(orient="records")) if not con.empty else [],
        "rowpaired_contrasts": json.loads(pair.to_json(orient="records")) if not pair.empty else [],
    }
    (out_dir / "ordinary_fit_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
