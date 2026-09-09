#!/usr/bin/env python3
"""research ordinary-heldout MLM price probe.

Score deterministic-mask MLM loss on the 6,992 ordinary held-out rows for the
seed43022 local and split arms: C/R/V/RS/VS.  This tests whether the ~0.3 nat
local-versus-split N gap from compact-rewrite targets also appears on ordinary
held-out text, or whether it is confined to the compact companion target family.
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
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/ordinary_heldout_price_probe.py')
ROOT = _PUBLIC_ROOT

REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
DATA = ROOT / "experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools"
WS = ROOT / "experiments/archive/relation_learning"
OUT = WS / "data/ordinary_heldout_price_probe"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/ordinary_heldout_price_probe.md')

HELDOUT = DATA / "heldout_cleanqwen_rows.jsonl"
CKPTS = ["chck_80M", "chck_90M", "chck_100M"]
MAX_LEN = 256
MASK_PROB = 0.15

ARM_CONFIGS: dict[str, pathlib.Path] = {
    "D_C_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_R_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_V_43022": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022",
    "D_RS_43022": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022",
    "D_VS_43022": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022",
}
ROLE = {"D_C_43022": "C", "D_R_43022": "R", "D_V_43022": "V", "D_RS_43022": "RS", "D_VS_43022": "VS"}
CONTRASTS = [("R", "RS"), ("V", "VS"), ("R", "C"), ("RS", "C"), ("V", "C"), ("VS", "C"), ("V", "R"), ("VS", "RS")]


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


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for obj in read_jsonl(HELDOUT):
        rows.append({
            "example_id": int(obj["example_id"]),
            "text": str(obj["text"]),
            "source": str(obj.get("source", "")),
            "words": int(obj.get("words", len(str(obj.get("text", "")).split()))),
        })
    if len(rows) != 6992:
        raise RuntimeError(f"expected 6992 held-out rows, found {len(rows)}")
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
def score_arm(model, tok, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
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
        path.write_text("", encoding="utf-8"); return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def summarize(all_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(all_rows)
    by_ck = (
        df.groupby(["role", "arm", "checkpoint"], dropna=False)
        .agg(n=("example_id", "count"), mean_loss=("loss", "mean"), sd_loss=("loss", "std"), mean_masked=("n_masked", "mean"))
        .reset_index()
    )
    late = (
        by_ck.groupby(["role", "arm"], dropna=False)
        .agg(n_min=("n", "min"), mean_loss=("mean_loss", "mean"), mean_masked=("mean_masked", "mean"))
        .reset_index()
    )
    con = []
    idx = late.set_index("role")
    for a, b in CONTRASTS:
        if a in idx.index and b in idx.index:
            ra = idx.loc[a]; rb = idx.loc[b]
            con.append({"contrast": f"{a}minus{b}", "n_min": int(min(ra["n_min"], rb["n_min"])), "delta_loss": float(ra["mean_loss"] - rb["mean_loss"]), "loss_a": float(ra["mean_loss"]), "loss_b": float(rb["mean_loss"])})
    con_df = pd.DataFrame(con)

    pair_rows = []
    wide = df.groupby(["example_id", "source", "words", "role"], dropna=False)["loss"].mean().reset_index().pivot(index=["example_id", "source", "words"], columns="role", values="loss").dropna()
    for a, b in CONTRASTS:
        if a in wide.columns and b in wide.columns:
            vals = (wide[a] - wide[b]).to_numpy(dtype=float)
            pair_rows.append({"contrast": f"{a}minus{b}", "n_rows": int(len(vals)), "mean_delta_loss": float(vals.mean()), "median_delta_loss": float(np.median(vals)), "se_delta_loss": float(vals.std(ddof=1) / math.sqrt(len(vals)))})
    pair_df = pd.DataFrame(pair_rows)
    return by_ck, late, con_df, pair_df


def write_note(late: pd.DataFrame, con: pd.DataFrame, pair: pd.DataFrame) -> None:
    def val(c: str) -> float:
        r = con[con["contrast"] == c]
        return float(r.iloc[0]["delta_loss"]) if not r.empty else float("nan")
    def pval(c: str) -> float:
        r = pair[pair["contrast"] == c]
        return float(r.iloc[0]["mean_delta_loss"]) if not r.empty else float("nan")
    lines = []
    lines.append("# research ordinary held-out MLM price probe")
    lines.append("")
    lines.append("This readout scores deterministic-mask MLM loss on the same 6,992 ordinary held-out rows for seed43022 CLEAN, REPEAT, VIEW, REPEAT_SPLIT, and VIEW_SPLIT. It tests whether the local-vs-split price seen on compact-rewrite N targets also appears on ordinary held-out text.")
    lines.append("")
    lines.append("## Late mean loss")
    lines.append("")
    lines.append("| role | arm | mean loss | n rows |")
    lines.append("|---|---|---:|---:|")
    for _, r in late.sort_values("role").iterrows():
        lines.append(f"| {r['role']} | {r['arm']} | {float(r['mean_loss']):.4f} | {int(r['n_min'])} |")
    lines.append("")
    lines.append("## Arm contrasts")
    lines.append("")
    lines.append("| contrast | late Δloss | row-paired Δloss | reading |")
    lines.append("|---|---:|---:|---|")
    readings = {
        "RminusRS": "local exact recurrence versus split exact recurrence",
        "VminusVS": "local restatement versus split restatement",
        "RminusC": "local exact recurrence versus CLEAN",
        "RSminusC": "split exact recurrence versus CLEAN",
        "VminusC": "local restatement versus CLEAN",
        "VSminusC": "split restatement versus CLEAN",
    }
    for c in ["RminusRS", "VminusVS", "RminusC", "RSminusC", "VminusC", "VSminusC", "VminusR", "VSminusRS"]:
        if c in set(con["contrast"]):
            lines.append(f"| {c} | {val(c):+.4f} | {pval(c):+.4f} | {readings.get(c, '')} |")
    lines.append("")
    rrs = val("RminusRS")
    vvs = val("VminusVS")
    lines.append(f"The compact-rewrite N price was about +0.3027 for R−RS and +0.2688 for V−VS. On ordinary held-out rows the corresponding deltas are R−RS {rrs:+.4f} and V−VS {vvs:+.4f}. If these ordinary deltas are near zero while compact N is large, the price is concentrated on the compact companion target family rather than broad held-out language fit.")
    lines.append("")
    lines.append("Data outputs: `experiments/archive/relation_learning/data/ordinary_heldout_price_probe`.")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--arms", nargs="*", default=list(ARM_CONFIGS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = load_rows()
    missing = []
    for arm in args.arms:
        for ck in CKPTS:
            d = ARM_CONFIGS[arm] / "hf_model" / ck
            if not ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()):
                missing.append(rel(d))
    plan = {"status": "ORDINARY_HELDOUT_PRICE_PLAN", "created_utc": now(), "arms": args.arms, "checkpoints": CKPTS, "heldout_rows": len(rows), "missing": missing, "out_dir": rel(OUT)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ordinary_price_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(missing[0])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(ARM_CONFIGS[args.arms[0]] / "hf_model"), use_fast=True)
    all_rows: list[dict[str, Any]] = []
    meta = []
    for arm in args.arms:
        for ck in CKPTS:
            t0 = time.time()
            mp = ARM_CONFIGS[arm] / "hf_model" / ck
            print(f"[LOAD] {arm} {ck} {mp}", flush=True)
            model = AutoModelForMaskedLM.from_pretrained(str(mp), torch_dtype=torch.float32)
            model.eval().to(device)
            scored = score_arm(model, tok, rows, device, args.batch_size)
            for r in scored:
                q = dict(r)
                q.update({"arm": arm, "role": ROLE[arm], "checkpoint": ck, "seed": 43022})
                all_rows.append(q)
            meta.append({"arm": arm, "role": ROLE[arm], "checkpoint": ck, "n": len(scored), "elapsed_sec": round(time.time()-t0, 2), "device": str(device)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[DONE] {arm} {ck} mean={statistics.mean([x['loss'] for x in scored]):.4f} elapsed={time.time()-t0:.1f}s", flush=True)
    write_csv(OUT / "ordinary_heldout_rows.csv", all_rows)
    write_csv(OUT / "ordinary_heldout_score_meta.csv", meta)
    by_ck, late, con, pair = summarize(all_rows)
    by_ck.to_csv(OUT / "ordinary_heldout_by_checkpoint.csv", index=False)
    late.to_csv(OUT / "ordinary_heldout_late_roles.csv", index=False)
    con.to_csv(OUT / "ordinary_heldout_late_contrasts.csv", index=False)
    pair.to_csv(OUT / "ordinary_heldout_rowpaired_contrasts.csv", index=False)
    write_note(late, con, pair)
    result = {"status": "ORDINARY_HELDOUT_PRICE_DONE", "note": rel(NOTE), "out_dir": rel(OUT), "main": {r["contrast"]: r["delta_loss"] for _, r in con.iterrows()}}
    (OUT / "ordinary_heldout_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
