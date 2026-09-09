#!/usr/bin/env python3
"""research: no-tag query ablation for arbitrary tag temporal bridge.

Imports the address-dissociation builder and reuses the arbitrary-tag training
setup.  Held evaluation adds a no_tag_query surface: the context still contains
per-example arbitrary tags for focal/secondary before/after records, but the
hypothesis does not name a tag ("Using the relevant record...").  If accuracy
stays high, the arbitrary-tag result was not true address use.  If it collapses
while normal tag-query evaluation succeeds, the previous result is a key-value
addressing ceiling.
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
ADDR_PATH = STUDY / "training/scripts/temporal_address_dissociation.py"
spec = importlib.util.spec_from_file_location("addr267", ADDR_PATH)
addr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = addr
spec.loader.exec_module(addr)
base = addr.base
DEFAULT_OUT = STUDY / "data/temporal_tag_notag_ablation"


def relation_rows_notag(focals, seconds, *, ns: str, split: str, eval_set: str, arm: str,
                        queries: list[str], train_kind: str):
    rows = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                for hd in ["AB", "BA"]:
                    for qf in queries:
                        ctx, _hyp, tpl = addr.build_context_and_hyp(
                            mode="arbitrary_tag", f=f, s=s, am=am, qf=qf, hd=hd, vi=vi, vl=vl,
                            ns=ns, hyp_style="held", shuffle_records=True, swap_focal_tags=False)
                        qw = f if qf.startswith("focal") else s
                        x, y = addr.hyp_xy(qw, am, hd)
                        # No address tag and no before/after lexical cue.  This should be underspecified.
                        hyp = f"Using the relevant record, {x} outranked {y}."
                        lab = addr.labels_for(qf, f, s, hd)
                        rows.append(base.mk(
                            row_id=f"{ns}|notag|{f.world_id}|{s.world_id}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=hyp, label=lab, split=split, eval_set=eval_set,
                            arm=arm, query_family=qf, focal_world=f.world_id, secondary_world=s.world_id,
                            changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                            template_group=tpl + "|notag_query", train_kind=train_kind, hyp_dir=hd,
                            pair_key=f"{ns}|notag|{f.world_id}|{s.world_id}|{tpl}|{qf}",
                            hyp_wording="no_tag", ctx_wording="arbitrary_tag",
                            label_mode="no_tag_query_ablation"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(addr.DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--base_stable", type=int, default=80)
    ap.add_argument("--sparse_changed", type=int, default=16)
    ap.add_argument("--sparse_stable", type=int, default=16)
    ap.add_argument("--eval_held_changed", type=int, default=40)
    ap.add_argument("--eval_held_stable", type=int, default=40)
    ap.add_argument("--eval_train_changed", type=int, default=40)
    ap.add_argument("--eval_train_stable", type=int, default=40)
    ap.add_argument("--base_train_wording", action="store_true")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed_base", type=int, default=26750)
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    base_rows, update_rows, evals, construction = addr.build_for_mode(args, "arbitrary_tag")
    selected, inventory = base.select_splits(args)
    all_q = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    evals["arbitrary_tag_heldChanged_heldStable_noTagHyp"] = relation_rows_notag(
        selected["eval_held_changed"], selected["eval_held_stable"], ns="eval267_arbitrary_tag_notag_hc_hs",
        split="held", eval_set="heldChanged_heldStable_noTagHyp", arm="eval", queries=all_q, train_kind="eval_temporal")
    evals["arbitrary_tag_heldStable_heldStable_noTagHyp"] = relation_rows_notag(
        selected["eval_held_stable"], selected["eval_held_stable2"], ns="eval267_arbitrary_tag_notag_hs_hs",
        split="held", eval_set="heldStable_heldStable_noTagHyp", arm="eval", queries=all_q, train_kind="eval_temporal")
    construction["eval_counts"] = {k: len(v) for k, v in sorted(evals.items())}
    construction["no_tag_ablation"] = "Context keeps arbitrary tags but hypothesis names no tag."
    print(json.dumps({"status": "BUILT", "base_rows": len(base_rows), "update_rows": len(update_rows), "eval_sets": len(evals)}), flush=True)
    if args.dry_build or args.n_seeds <= 0:
        (out / "tag_notag_ablation_summary.json").write_text(json.dumps({"status": "DRY", "args": vars(args), "construction": construction}, indent=2) + "\n", "utf-8")
        return
    tok = AutoTokenizer.from_pretrained(args.model_path)
    results = []
    for si in range(args.n_seeds):
        seed = args.seed_base + si
        r = base.train_one(Path(args.model_path), tok, "balanced_temporal", seed, list(base_rows) + list(update_rows), evals, args, args.device)
        results.append(r)
        for en, ev in sorted(r["evals"].items()):
            if "heldChanged_heldStable" in en:
                c = ev["contrastive"]
                print(json.dumps({"event": "eval", "seed": seed, "eval_set": en, "train_acc": r["best_train_acc"],
                                  "train_fit": r.get("train_fit"), "by_query": c.get("by_query"),
                                  "by_query_margin": c.get("by_query_margin"), "con_acc": c.get("contrastive_acc")}), flush=True)
    final = {"status": "TAG_NOTAG_ABLATION", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "args": vars(args), "construction": construction, "aggregate": base.aggregate(results), "per_seed": results,
             "boundary": "No-tag ablation should be underspecified; collapse supports address use in arbitrary-tag result."}
    (out / "tag_notag_ablation_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    # Reuse base md writer for compact eval tables.
    base.write_md({"construction": {"base": base.describe_rows(base_rows), "updates": {"balanced_temporal": base.describe_rows(update_rows)}},
                   "aggregate": final["aggregate"]}, out)
    print(json.dumps({"status": "DONE", "json": str(out / "tag_notag_ablation_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
