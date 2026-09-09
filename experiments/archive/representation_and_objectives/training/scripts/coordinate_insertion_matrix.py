#!/usr/bin/env python3
"""research: train a working unique-tag address model, then insert context variants at evaluation.

The canonical research unique-tag direct arm fits only with the research support
geometry.  A separately trained filler arm collapsed, but this does not tell us
whether added sentences disrupt inference or disrupt acquisition.  This script
trains the known working supplied-coordinate model and evaluates the same fitted
model on direct-tag queries with extra filler / duplicate / declaration material
inserted into the context.

If insertion immediately degrades margins, the address readout is inference-time
fragile.  If insertion is harmless, the failed filler arm is an acquisition or
optimization problem, not a readout interference result.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
PATH = STUDY / "training/scripts/role_coordinate_collision_probe.py"
spec = importlib.util.spec_from_file_location("collision_for_insertion", PATH)
s268 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = s268
spec.loader.exec_module(s268)
base = s268.base
DEFAULT_MODEL = s268.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/coordinate_insertion_matrix"

CTX_MODES = ["unique", "filler", "labeldup", "entrydup", "slotdecl", "roledecl"]


def build(args):
    train_base, train_updates, _, cons_train = s268.build_arm(args, "unique_tag_direct")
    selected, inventory = base.select_splits(args)
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    evals: dict[str, list[dict[str, Any]]] = {}
    pairings = [
        ("heldChanged_heldStable", selected["eval_held_changed"], selected["eval_held_stable"]),
        ("trainChanged_heldStable", selected["eval_train_changed"], selected["eval_held_stable"]),
        ("heldStable_heldStable", selected["eval_held_stable"], selected["eval_held_stable2"]),
    ]
    for ctx_mode in CTX_MODES:
        for pname, focals, seconds in pairings:
            evals[f"insert_{ctx_mode}_{pname}_directTag_heldHyp"] = s268.relation_rows(
                focals, seconds, ns=f"eval268_insert_{ctx_mode}_{pname}", ctx_mode=ctx_mode,
                split="held", eval_set=pname, arm="eval", queries=base_queries,
                train_kind="eval_temporal", query_mode="direct_tag", hyp_style="held")
            if ctx_mode == "roledecl" and pname == "heldChanged_heldStable":
                evals[f"insert_{ctx_mode}_{pname}_directTag_roleSwap_heldHyp"] = s268.relation_rows(
                    focals, seconds, ns=f"eval268_insert_{ctx_mode}_{pname}_swap", ctx_mode=ctx_mode,
                    split="held", eval_set=pname, arm="eval", queries=base_queries,
                    train_kind="eval_temporal", query_mode="direct_tag", hyp_style="held",
                    swap_role_decls=True)
    construction = {
        "train_arm": "unique_tag_direct",
        "train_construction": cons_train,
        "eval_ctx_modes": CTX_MODES,
        "eval_inventory": inventory,
        "eval_selection": {k: base.world_summary(v) for k, v in selected.items()},
        "eval_counts": {k: len(v) for k, v in sorted(evals.items())},
        "boundary": "Evaluation-time insertion matrix after fitting supplied unique-tag address model.",
    }
    return list(train_base) + list(train_updates), evals, construction


def summarize_result(r: dict[str, Any]) -> dict[str, Any]:
    out = {"best_train_acc": r["best_train_acc"], "train_fit": r.get("train_fit", {}), "evals": {}}
    for en, ev in sorted(r["evals"].items()):
        c = ev["contrastive"]
        out["evals"][en] = {"contrastive_acc": c.get("contrastive_acc"),
                             "by_query": c.get("by_query", {}),
                             "by_query_margin": c.get("by_query_margin", {}),
                             "standard_acc": ev["standard"].get("acc")}
    return out


def write_md(summary: dict[str, Any], out: Path):
    s = summary.get("summary", {})
    lines = ["# research coordinate insertion matrix\n\n",
             "Train condition: unique entry tags only. Evaluation inserts extra context material while keeping direct tag queries.\n\n"]
    if s:
        lines.append(f"Best train acc: {s['best_train_acc']:.3f}\n\n")
        lines.append(f"Train fit by kind: `{json.dumps(s.get('train_fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")
        lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in s["evals"].items():
            if "heldChanged_heldStable" not in en:
                continue
            bq, bm = ev["by_query"], ev["by_query_margin"]
            def v(q):
                x = bq.get(q)
                return float("nan") if x is None else float(x)
            def m(q):
                x = bm.get(q)
                return float("nan") if x is None else float(x)
            lines.append(f"| {en} | {ev['contrastive_acc']:.3f} | {v('focal_before'):.3f} | {v('focal_after'):.3f} | {v('secondary_before'):.3f} | {v('secondary_after'):.3f} | {m('focal_after'):.2f} | {m('secondary_after'):.2f} |\n")
    (out / "coordinate_insertion_matrix.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=26700)
    ap.add_argument("--base_stable", type=int, default=80)
    ap.add_argument("--base_train_wording", action="store_true")
    ap.add_argument("--sparse_changed", type=int, default=16)
    ap.add_argument("--sparse_stable", type=int, default=16)
    ap.add_argument("--eval_held_changed", type=int, default=40)
    ap.add_argument("--eval_held_stable", type=int, default=40)
    ap.add_argument("--eval_train_changed", type=int, default=40)
    ap.add_argument("--eval_train_stable", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    train_rows, evals, construction = build(args)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    audit = s268.token_audit(tok, {"train": train_rows, "eval_all": [r for rows in evals.values() for r in rows]}, args.max_len)
    print(json.dumps({"status": "BUILT", "train_rows": len(train_rows), "eval_sets": len(evals), "token_audit": audit}), flush=True)
    if args.dry_build:
        dry = {"status": "INSERTION_MATRIX_DRY", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "token_audit": audit}
        (out / "coordinate_insertion_matrix.json").write_text(json.dumps(dry, indent=2, ensure_ascii=False) + "\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "coordinate_insertion_matrix.json")}), flush=True)
        return
    r = base.train_one(Path(args.model_path), tok, "unique_tag_direct", args.seed, train_rows, evals, args, args.device)
    summary = {"status": "INSERTION_MATRIX", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "token_audit": audit,
               "summary": summarize_result(r), "per_seed": [r],
               "boundary": "Same fitted unique-tag coordinate model evaluated with inserted contexts."}
    (out / "coordinate_insertion_matrix.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(summary, out)
    for en, ev in sorted(r["evals"].items()):
        if "heldChanged_heldStable" in en:
            c = ev["contrastive"]
            print(json.dumps({"event": "eval", "eval_set": en, "train_acc": r["best_train_acc"],
                              "train_fit": r.get("train_fit"), "by_query": c.get("by_query"),
                              "by_query_margin": c.get("by_query_margin"), "con_acc": c.get("contrastive_acc")}), flush=True)
    print(json.dumps({"status": "DONE", "json": str(out / "coordinate_insertion_matrix.json"),
                      "md": str(out / "coordinate_insertion_matrix.md")}), flush=True)


if __name__ == "__main__":
    main()
