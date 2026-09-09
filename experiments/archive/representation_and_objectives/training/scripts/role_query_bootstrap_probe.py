#!/usr/bin/env python3
"""research: Can an existing address route bootstrap role-query composition?

research prefix warm-start showed that exact-chance direct-prefix failures should
not be promoted: after a tag address route is fitted, the same tags in prefixed
or inline-role entries can be read and adapted.  The remaining scientific
question is harder: can sparse training teach a query such as "the focal update
record" to select the entry whose context says "Focal update entry TAG:" without
putting TAG in the hypothesis?

This script compares:
  warm_chain: tag_only direct -> inline_role direct -> inline_role role queries
  scratch_role: inline_role role queries from the pretrained model only

Both use identical worlds, labels, role contexts, and address namespaces.  The
purpose is to separate address-route bootstrapping from raw role-to-address
composition.  Small pretrained bridge only.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
ST270_PATH = STUDY / "training/scripts/prefix_warmstart_probe.py"
spec = importlib.util.spec_from_file_location("st270", ST270_PATH)
st = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = st
spec.loader.exec_module(st)

s269 = st.s269
base = st.base
DEFAULT_MODEL = st.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/role_query_bootstrap_probe"


def role_eval_suite(args, mode="inline_role"):
    sel, _ = base.select_splits(args)
    foc, sec_e = sel["eval_held_changed"], sel["eval_held_stable"]
    train_foc = sel["eval_train_changed"]
    out = {}
    for pn, fs in [("hC_hS", foc), ("tC_hS", train_foc)]:
        ns = st._eval_ns(args, mode, pn)
        out[f"{mode}_{pn}_direct_hH"] = s269.mk_rows(
            fs, sec_e, ns=ns, mode=mode, split="held", eset=pn, arm="eval",
            queries=s269.ALL_Q, tkind="eval", qmode="direct_tag", style="held")
        out[f"{mode}_{pn}_role_hH"] = s269.mk_rows(
            fs, sec_e, ns=ns, mode=mode, split="held", eset=pn, arm="eval",
            queries=s269.ALL_Q, tkind="eval", qmode="role_inline", style="held")
        out[f"{mode}_{pn}_rolePara_hH"] = s269.mk_rows(
            fs, sec_e, ns=ns, mode=mode, split="held", eset=pn, arm="eval",
            queries=s269.ALL_Q, tkind="eval", qmode="role_para", style="held")
        out[f"{mode}_{pn}_roleSwap_role_hH"] = s269.mk_rows(
            fs, sec_e, ns=ns, mode=mode, split="held", eset=pn, arm="eval",
            queries=s269.ALL_Q, tkind="eval", qmode="role_inline", style="held",
            swap=True, swap_labels=True)
        out[f"{mode}_{pn}_roleSwap_direct_hH"] = s269.mk_rows(
            fs, sec_e, ns=ns, mode=mode, split="held", eset=pn, arm="eval",
            queries=s269.ALL_Q, tkind="eval", qmode="direct_tag", style="held",
            swap=True)
    return out


def short_evals(model, tok, evals, args, dev):
    return st.summarize_eval(st.eval_sets(model, tok, evals, args, dev))


def write_md(summary: dict[str, Any], out: Path):
    lines = ["# research role-query bootstrap probe\n\n",
             "Question: after direct address retrieval is fitted, can sparse role-query training learn to select records by context role words rather than by a tag in the hypothesis?\n\n",
             f"Mode: `{summary['args']['mode']}`; shared namespace: `{summary['args'].get('shared_address_ns')}` suffix `{summary['args'].get('shared_address_suffix')}`.\n\n"]
    lines.append("## Fits\n\n")
    for k, r in summary["results"].items():
        if r is None:
            continue
        lines.append(f"### {k}\n\nBest train acc: {r.get('best_train_acc')}\n\n")
        lines.append(f"Fit: `{json.dumps(r.get('fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")
    lines.append("## Evaluations\n\n")
    for block, evs in summary.get("evals", {}).items():
        lines.append(f"### {block}\n\n")
        lines.append("| eval | con | fb | fa | sb | sa | fa_m | sa_m |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in sorted(evs.items()):
            bq, bm = ev.get("by_q", {}), ev.get("by_m", {})
            def gv(q):
                v = bq.get(q); return float("nan") if v is None else float(v)
            def gm(q):
                v = bm.get(q); return float("nan") if v is None else float(v)
            lines.append(f"| {en} | {ev.get('con_acc', float('nan')):.3f} | {gv('focal_before'):.3f} | {gv('focal_after'):.3f} | {gv('secondary_before'):.3f} | {gv('secondary_after'):.3f} | {gm('focal_after'):.2f} | {gm('secondary_after'):.2f} |\n")
        lines.append("\n")
    (out / "role_query_bootstrap_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=27000)
    ap.add_argument("--mode", default="inline_role", choices=["inline_role", "postfix_role", "suffix_role"])
    ap.add_argument("--shared_address_ns", default="s269_inline_direct")
    ap.add_argument("--shared_address_suffix", default="_direct_tag")
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
    ap.add_argument("--tag_epochs", type=int, default=10)
    ap.add_argument("--direct_epochs", type=int, default=6)
    ap.add_argument("--role_epochs", type=int, default=8)
    ap.add_argument("--scratch_epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--prefix_head_lr", type=float, default=1e-3)
    ap.add_argument("--prefix_encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    ap.add_argument("--run", default="warm,scratch", help="comma list from warm,scratch")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tag_tr, tag_ev, tag_con = st.build_train_eval(args, "tag_only", "direct_tag")
    direct_tr, direct_ev, direct_con = st.build_train_eval(args, args.mode, "direct_tag")
    role_tr, role_ev_basic, role_con = st.build_train_eval(args, args.mode, "role_inline")
    evals = role_eval_suite(args, args.mode)
    all_rows = tag_tr + direct_tr + role_tr
    for ev in [tag_ev, direct_ev, role_ev_basic, evals]:
        for rows in ev.values():
            all_rows += rows
    lens = [len(tok.encode(r["text"], add_special_tokens=True)) for r in all_rows]
    arr = np.array(lens, dtype=float)
    ta = {"n": len(lens), "min": int(arr.min()), "p50": float(np.percentile(arr, 50)),
          "p90": float(np.percentile(arr, 90)), "max": int(arr.max()),
          "frac_gt_max_len": float(np.mean(arr > args.max_len))}
    construction = {"tag_train": tag_con, "direct_train": direct_con, "role_train": role_con,
                    "eval_counts": {k: len(v) for k, v in evals.items()}, "token_audit": ta}
    print(json.dumps({"status": "BUILT", "tag_train": len(tag_tr), "direct_train": len(direct_tr),
                      "role_train": len(role_tr), "eval_sets": len(evals), "token_audit": ta}), flush=True)
    if args.dry_build:
        summary = {"status": "ROLE_BOOTSTRAP_DRY", "args": vars(args), "construction": construction}
        (out / "role_query_bootstrap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "role_query_bootstrap_summary.json")}), flush=True)
        return

    dev = args.device
    runset = {x.strip() for x in args.run.split(",") if x.strip()}
    results: dict[str, Any] = {}
    eval_blocks: dict[str, Any] = {}

    if "scratch" in runset:
        scratch = st.make_model(Path(args.model_path), dev, seed=args.seed + 11)
        results["scratch_role"] = st.train_loop(
            scratch, tok, role_tr, args, dev, seed=args.seed + 11,
            epochs=args.scratch_epochs, label="scratch_role",
            init_best_from_current=False, track_eval_rows=evals)
        eval_blocks["scratch_after_role"] = short_evals(scratch, tok, evals, args, dev)
        del scratch
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    if "warm" in runset:
        model = st.make_model(Path(args.model_path), dev, seed=args.seed)
        results["warm_tag_pretrain"] = st.train_loop(
            model, tok, tag_tr, args, dev, seed=args.seed, epochs=args.tag_epochs,
            label="warm_tag_pretrain", init_best_from_current=False,
            track_eval_rows=tag_ev)
        eval_blocks["warm_after_tag_on_role_suite"] = short_evals(model, tok, evals, args, dev)
        results["warm_inline_direct"] = st.train_loop(
            model, tok, direct_tr, args, dev, seed=args.seed + 1,
            epochs=args.direct_epochs, label="warm_inline_direct",
            head_lr=args.prefix_head_lr, encoder_lr=args.prefix_encoder_lr,
            init_best_from_current=True, track_eval_rows=direct_ev)
        eval_blocks["warm_after_direct_on_role_suite"] = short_evals(model, tok, evals, args, dev)
        results["warm_role_query"] = st.train_loop(
            model, tok, role_tr, args, dev, seed=args.seed + 2,
            epochs=args.role_epochs, label="warm_role_query",
            head_lr=args.prefix_head_lr, encoder_lr=args.prefix_encoder_lr,
            init_best_from_current=True, track_eval_rows=evals)
        eval_blocks["warm_after_role_on_role_suite"] = short_evals(model, tok, evals, args, dev)
        eval_blocks["warm_after_role_direct_retention"] = short_evals(model, tok, direct_ev, args, dev)
        results["warm_role_tag_fit_retention"] = {"best_train_acc": None,
            "fit": st.train_breakdown(model, tok, tag_tr, args, dev), "history": []}

    summary = {"status": "ROLE_QUERY_BOOTSTRAP", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "results": results,
               "evals": eval_blocks,
               "boundary": "Small pretrained bridge; tests sequential address-to-role composition, not BabyLM-scale training."}
    (out / "role_query_bootstrap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "json": str(out / "role_query_bootstrap_summary.json"),
                      "md": str(out / "role_query_bootstrap_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
