#!/usr/bin/env python3
"""research: component fit for the failed indirect-address construction.

Tests whether direct tag queries can fit when the context also contains role-to-tag
declarations.  In the successful arbitrary-tag bridge, each tag occurs only on
its entry.  In the indirect construction, each tag occurs in a declaration and an
entry; this can make even direct tag retrieval harder.  This component run keeps
that harder context but trains only direct-tag queries (or only role queries) to
localize why the full two-hop run stayed at chance.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
IND_PATH = STUDY / "training/scripts/temporal_indirect_address_probe.py"
spec = importlib.util.spec_from_file_location("ind267_component", IND_PATH)
ind = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = ind
spec.loader.exec_module(ind)
base = ind.base
DEFAULT_OUT = STUDY / "data/indirect_component_fit"


def build_component(args):
    selected, inventory = base.select_splits(args)
    fam = args.family
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    sparse_queries = ["focal_before", "focal_after"]
    qmode = "direct_tag" if args.component == "direct" else "role"
    base_rows = ind.relation_rows(selected["base_stable"], selected["base_stable2"], ns=f"comp267_base_{fam}_{qmode}", family=fam,
                                  split="train", eval_set="base", arm="base", queries=base_queries,
                                  train_kind=f"base_{args.component}", query_mode=qmode, hyp_style="train")
    update_rows = []
    seconds = selected["sparse_secondary_stable"]
    update_rows += ind.relation_rows(selected["sparse_changed"], seconds, ns=f"comp267_chg_{fam}_{qmode}", family=fam,
                                     split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                     train_kind=f"sparse_changed_{args.component}", query_mode=qmode, hyp_style="train")
    update_rows += ind.relation_rows(selected["sparse_stable"], seconds, ns=f"comp267_stb_{fam}_{qmode}", family=fam,
                                     split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                     train_kind=f"sparse_stable_{args.component}", query_mode=qmode, hyp_style="train")
    evals = {}
    for mode in [qmode, "direct_tag", "role"]:
        evals[f"{fam}_heldChanged_heldStable_{mode}_heldHyp"] = ind.relation_rows(
            selected["eval_held_changed"], selected["eval_held_stable"], ns=f"comp267_eval_{fam}_{mode}", family=fam,
            split="held", eval_set="heldChanged_heldStable", arm="eval", queries=base_queries,
            train_kind="eval_temporal", query_mode=mode, hyp_style="held")
    construction = {"family": fam, "component": args.component, "base": base.describe_rows(base_rows),
                    "updates": {"balanced_temporal": base.describe_rows(update_rows)},
                    "eval_counts": {k: len(v) for k, v in evals.items()},
                    "selection": {k: base.world_summary(v) for k, v in selected.items()},
                    "boundary": "Component fit for the indirect-address construction."}
    return base_rows, update_rows, evals, construction


def write_md(summary, out: Path):
    lines = ["# research indirect component fit\n\n"]
    cons = summary.get("construction", {})
    if cons:
        lines.append(f"Family: {cons.get('family')}; component: {cons.get('component')}\n\n")
        lines.append(f"Base rows: {cons.get('base',{}).get('n')}; update rows: {cons.get('updates',{}).get('balanced_temporal',{}).get('n')}\n\n")
    agg = summary.get("aggregate", {})
    for arm, item in agg.items():
        lines.append(f"Best train mean: {item['train_acc_mean']:.3f}\n\n")
        lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in item["evals"].items():
            bq = ev["contrastive_by_query"]; bm = ev["contrastive_by_query_margin"]
            def v(q): return bq.get(q, {}).get("mean", float("nan"))
            def m(q): return bm.get(q, {}).get("mean", float("nan"))
            lines.append(f"| {en} | {ev['contrastive_acc']['mean']:.3f} | {v('focal_before'):.3f} | {v('focal_after'):.3f} | {v('secondary_before'):.3f} | {v('secondary_after'):.3f} | {m('focal_after'):.3f} | {m('secondary_after'):.3f} |\n")
    (out / "component_fit_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(ind.DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--family", type=str, default="background_update")
    ap.add_argument("--component", choices=["direct", "role"], default="direct")
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--seed_base", type=int, default=26790)
    ap.add_argument("--base_stable", type=int, default=20)
    ap.add_argument("--sparse_changed", type=int, default=8)
    ap.add_argument("--sparse_stable", type=int, default=8)
    ap.add_argument("--eval_held_changed", type=int, default=10)
    ap.add_argument("--eval_held_stable", type=int, default=10)
    ap.add_argument("--eval_train_changed", type=int, default=10)
    ap.add_argument("--eval_train_stable", type=int, default=10)
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
    base_rows, update_rows, evals, construction = build_component(args)
    print(json.dumps({"status": "BUILT", "component": args.component, "base_rows": len(base_rows), "update_rows": len(update_rows), "eval_sets": len(evals)}), flush=True)
    if args.dry_build or args.n_seeds <= 0:
        (out / "component_fit_summary.json").write_text(json.dumps({"status": "DRY", "args": vars(args), "construction": construction}, indent=2) + "\n", "utf-8")
        return
    tok = AutoTokenizer.from_pretrained(args.model_path)
    results = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = base.train_one(Path(args.model_path), tok, "balanced_temporal", seed, list(base_rows) + list(update_rows), evals, args, args.device)
        results.append(r)
        for en, ev in sorted(r["evals"].items()):
            c = ev["contrastive"]
            print(json.dumps({"event": "eval", "component": args.component, "seed": seed, "eval_set": en,
                              "train_acc": r["best_train_acc"], "train_fit": r.get("train_fit"),
                              "by_query": c.get("by_query"), "by_query_margin": c.get("by_query_margin"),
                              "con_acc": c.get("contrastive_acc")}), flush=True)
    final = {"status": "INDIRECT_COMPONENT_FIT", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "args": vars(args), "construction": construction, "aggregate": base.aggregate(results), "per_seed": results,
             "boundary": "Direct/role component fit for failed two-hop construction."}
    (out / "component_fit_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(final, out)
    print(json.dumps({"status": "DONE", "json": str(out / "component_fit_summary.json"), "md": str(out / "component_fit_summary.md")}), flush=True)

if __name__ == "__main__":
    main()
