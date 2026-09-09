#!/usr/bin/env python3
"""research: Does a learned address route bootstrap a minimally prefixed format?

research showed exact-chance training from scratch for `The entry TAG:` and
`Focal background entry TAG:` formats.  Those are not interpreted here as a
substantive boundary until optimization access is resolved.  This probe uses the
fitted `Entry TAG:` address reader as a lever:

  1. Train a tag-only direct-address model on exactly the canonical research/268
     supervision and world selections.
  2. Continue the same model on the same labels/worlds rendered as a matched
     prefix format (`The entry TAG:` by default).
  3. Compare against scratch prefix training under the same code path and, after
     continuation, evaluate both tag-only and prefix contexts.

A warm-start rescue would support a bootstrapping principle: once address
coordinates are formed, sparse continuation can extend them to a new surface
format. Persistent failure after verified tag-only fitting would localize a more
specific interface incompatibility.  This is a small bridge probe only, not
BabyLM-scale training.
"""
from __future__ import annotations

import argparse, copy, json, time, importlib.util, sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
PATH = STUDY / "training/scripts/inline_role_bridge.py"
spec = importlib.util.spec_from_file_location("inline", PATH)
s269 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = s269
spec.loader.exec_module(s269)

base = s269.base
DEFAULT_MODEL = s269.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/prefix_warmstart_probe"


def _addr_ns(args, mode: str, kind: str) -> str:
    shared = getattr(args, "shared_address_ns", "") or ""
    suffix = getattr(args, "shared_address_suffix", "") or ""
    if shared:
        return f"{shared}_{kind}{suffix}"
    return f"s270_{mode}_{kind}"


def _eval_ns(args, mode: str, kind: str) -> str:
    shared = getattr(args, "shared_address_ns", "") or ""
    if shared:
        return f"e{shared}_{kind}"
    return f"e270_{mode}_{kind}"


def build_train_eval(args, mode: str, qmode: str = "direct_tag"):
    # Reuse exactly research/267/268 selection and supervision geometry.  When
    # --shared_address_ns is set, tag-only and transformed-format rows use the
    # same arbitrary address tokens for each world/vi/vl record; only the context
    # rendering changes.
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]
    tr = []
    tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                       ns=_addr_ns(args, mode, "b"), mode=mode, split="train",
                       eset="base", arm="base", queries=s269.ALL_Q,
                       tkind="base_stable_anchor", qmode=qmode)
    if getattr(args, "base_train_wording", False):
        tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                           ns=_addr_ns(args, mode, "bw"), mode=mode, split="train",
                           eset="base", arm="base", queries=s269.ALL_Q,
                           tkind="base_stable_train", qmode=qmode)
    tr += s269.mk_rows(sel["sparse_changed"], secs,
                       ns=_addr_ns(args, mode, "c"), mode=mode, split="train",
                       eset="sparse", arm="balanced_temporal",
                       queries=s269.SPARSE_Q, tkind="sparse_changed_focal", qmode=qmode)
    tr += s269.mk_rows(sel["sparse_stable"], secs,
                       ns=_addr_ns(args, mode, "s"), mode=mode, split="train",
                       eset="sparse", arm="balanced_temporal",
                       queries=s269.SPARSE_Q, tkind="sparse_stable_focal", qmode=qmode)

    foc, sec_e = sel["eval_held_changed"], sel["eval_held_stable"]
    ev = {
        f"{mode}_heldChanged_heldStable_direct_hH": s269.mk_rows(
            foc, sec_e, ns=_eval_ns(args, mode, "hC_hS_dt"), mode=mode, split="held",
            eset="hC_hS", arm="eval", queries=s269.ALL_Q, tkind="eval",
            qmode=qmode, style="held"),
        f"{mode}_trainChanged_heldStable_direct_hH": s269.mk_rows(
            sel["eval_train_changed"], sec_e, ns=_eval_ns(args, mode, "tC_hS_dt"), mode=mode, split="held",
            eset="tC_hS", arm="eval", queries=s269.ALL_Q, tkind="eval",
            qmode=qmode, style="held"),
    }
    con = {"mode": mode, "qmode": qmode, "inv": inv,
           "selection": {k: base.world_summary(v) for k, v in sel.items()},
           "train": base.describe_rows(tr),
           "eval_counts": {k: len(v) for k, v in ev.items()}}
    return tr, ev, con


def encode(tok, rows, max_len, dev):
    return base.encode_rows(tok, rows, max_len, dev)


def train_breakdown(model, tok, rows, args, dev):
    logits = base.get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size)
    return base.train_breakdown_from_logits(logits, rows)


def eval_sets(model, tok, evals, args, dev):
    return base.full_eval(model, tok, evals, args.max_len, dev, args.eval_batch_size)


def make_model(model_path: Path, dev: str, seed: int | None = None):
    # Same full fine-tuning model class as research/267/269.  Seed before
    # construction because the entailment head is randomly initialized.
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed)
    return base.m.DebertaEntailment(model_path, "full", "pretrained").to(dev)


def make_opt(model, args, *, head_lr=None, encoder_lr=None):
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    return torch.optim.AdamW([
        {"params": head_p, "lr": args.head_lr if head_lr is None else head_lr},
        {"params": enc_p, "lr": args.encoder_lr if encoder_lr is None else encoder_lr},
    ], weight_decay=args.weight_decay)


def train_loop(model, tok, rows, args, dev, *, seed: int, epochs: int, label: str,
               head_lr=None, encoder_lr=None, init_best_from_current: bool = True,
               track_eval_rows: dict[str, list[dict[str, Any]]] | None = None):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = make_opt(model, args, head_lr=head_lr, encoder_lr=encoder_lr)
    x, mask, y = encode(tok, rows, args.max_len, dev)
    ds = TensorDataset(x, mask, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}) if init_best_from_current else None
    best_train = -1.0
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss.item()))
        fit = train_breakdown(model, tok, rows, args, dev)
        ev_track = {}
        if track_eval_rows:
            eout = eval_sets(model, tok, track_eval_rows, args, dev)
            for en, ev in eout.items():
                ev_track[en] = ev["contrastive"].get("by_query", {})
        ta = float(fit.get("overall", float("nan")))
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "train_acc": ta,
                     "fit_by_train_kind": fit.get("by_train_kind", {}),
                     "track_by_query": ev_track})
        print(json.dumps({"event": "epoch", "phase": label, "epoch": ep,
                          "loss": hist[-1]["loss"], "train_acc": ta,
                          "fit": fit.get("by_train_kind", {}),
                          "track": ev_track}), flush=True)
        if ta > best_train:
            best_train = ta
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    fit = train_breakdown(model, tok, rows, args, dev)
    return {"phase": label, "best_train_acc": float(best_train), "fit": fit, "history": hist}


def summarize_eval(evals_out):
    out = {}
    for en, ev in sorted(evals_out.items()):
        c = ev["contrastive"]
        out[en] = {"con_acc": c.get("contrastive_acc"),
                   "by_q": c.get("by_query", {}),
                   "by_m": c.get("by_query_margin", {}),
                   "std_acc": ev["standard"].get("acc")}
    return out


def write_md(summary: dict[str, Any], out: Path):
    lines = ["# research prefix warm-start probe\n\n",
             "Question: does a previously learned `Entry TAG:` address route make a minimally prefixed `The entry TAG:` format learnable under identical supervision?\n\n"]
    lines.append(f"Prefix mode: `{summary['args']['prefix_mode']}`; tag pretrain mode: `tag_only`.\n\n")
    lines.append("## Construction\n\n")
    for k in ["tag_train", "prefix_train"]:
        lines.append(f"- {k}: {summary['construction'][k]['train']}\n")
    lines.append("\n## Main fits\n\n")
    for k in ["scratch_prefix", "tag_pretrain", "warm_prefix", "optional_tag_refresh"]:
        if k in summary["results"] and summary["results"][k] is not None:
            r = summary["results"][k]
            lines.append(f"### {k}\n\nBest train acc: {r.get('best_train_acc')}\n\n")
            lines.append(f"Fit: `{json.dumps(r.get('fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")
    lines.append("## Evaluation summary\n\n")
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
    (out / "prefix_warmstart_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=27000)
    ap.add_argument("--prefix_mode", default="prefix_one", choices=["prefix_one", "inline_role", "postfix_role", "suffix_role"])
    ap.add_argument("--shared_address_ns", default="", help="if nonempty, use the same address-token namespace across tag_only and prefix_mode rows")
    ap.add_argument("--shared_address_suffix", default="", help="optional suffix appended after each shared kind, e.g. _direct_tag to reproduce research namespaces")
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
    ap.add_argument("--prefix_epochs", type=int, default=10)
    ap.add_argument("--scratch_epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--prefix_head_lr", type=float, default=1e-3)
    ap.add_argument("--prefix_encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    ap.add_argument("--skip_scratch", action="store_true")
    ap.add_argument("--only_scratch", action="store_true", help="run only direct scratch training on the prefix_mode rows, for matched direct-learning controls")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    tag_tr, tag_ev, tag_con = build_train_eval(args, "tag_only")
    pref_tr, pref_ev, pref_con = build_train_eval(args, args.prefix_mode)
    all_rows = tag_tr + pref_tr
    for ev in [tag_ev, pref_ev]:
        for rows in ev.values():
            all_rows += rows
    lens = [len(tok.encode(r["text"], add_special_tokens=True)) for r in all_rows]
    arr = np.array(lens, dtype=float)
    token_audit = {"n": len(lens), "min": int(arr.min()), "p50": float(np.percentile(arr, 50)),
                   "p90": float(np.percentile(arr, 90)), "max": int(arr.max()),
                   "frac_gt_max_len": float(np.mean(arr > args.max_len))}
    construction = {"tag_train": tag_con, "prefix_train": pref_con, "token_audit": token_audit}
    print(json.dumps({"status": "BUILT", "tag_train": len(tag_tr), "prefix_train": len(pref_tr),
                      "tag_evals": {k: len(v) for k, v in tag_ev.items()},
                      "prefix_evals": {k: len(v) for k, v in pref_ev.items()},
                      "token_audit": token_audit}), flush=True)
    if args.dry_build:
        summary = {"status": "PREFIX_WARMSTART_DRY", "args": vars(args), "construction": construction}
        (out / "prefix_warmstart_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "prefix_warmstart_summary.json")}), flush=True)
        return

    dev = args.device
    results: dict[str, Any] = {}
    eval_blocks: dict[str, Any] = {}

    if not args.skip_scratch:
        scratch = make_model(Path(args.model_path), dev, seed=args.seed + 1)
        results["scratch_prefix"] = train_loop(scratch, tok, pref_tr, args, dev,
                                                seed=args.seed + 1, epochs=args.scratch_epochs,
                                                label="scratch_prefix", init_best_from_current=False,
                                                track_eval_rows=pref_ev)
        eval_blocks["scratch_prefix_on_prefix"] = summarize_eval(eval_sets(scratch, tok, pref_ev, args, dev))
        del scratch
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    else:
        results["scratch_prefix"] = None

    if args.only_scratch:
        summary = {"status": "PREFIX_SCRATCH_ONLY", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "args": vars(args), "construction": construction, "results": results,
                   "evals": eval_blocks,
                   "boundary": "Small pretrained bridge direct-learning control; no continuation run."}
        (out / "prefix_warmstart_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
        write_md(summary, out)
        print(json.dumps({"status": "DONE", "json": str(out / "prefix_warmstart_summary.json"),
                          "md": str(out / "prefix_warmstart_summary.md")}), flush=True)
        return

    model = make_model(Path(args.model_path), dev, seed=args.seed)
    results["tag_pretrain"] = train_loop(model, tok, tag_tr, args, dev,
                                          seed=args.seed, epochs=args.tag_epochs,
                                          label="tag_pretrain", init_best_from_current=False,
                                          track_eval_rows=tag_ev)
    eval_blocks["after_tag_on_tag"] = summarize_eval(eval_sets(model, tok, tag_ev, args, dev))
    eval_blocks["after_tag_on_prefix_before_continuation"] = summarize_eval(eval_sets(model, tok, pref_ev, args, dev))

    # Continue the exact fitted address reader into the matched prefix surface.
    results["warm_prefix"] = train_loop(model, tok, pref_tr, args, dev,
                                         seed=args.seed + 2, epochs=args.prefix_epochs,
                                         label="warm_prefix", head_lr=args.prefix_head_lr,
                                         encoder_lr=args.prefix_encoder_lr,
                                         init_best_from_current=True,
                                         track_eval_rows=pref_ev)
    eval_blocks["after_warm_on_prefix"] = summarize_eval(eval_sets(model, tok, pref_ev, args, dev))
    eval_blocks["after_warm_on_tag_retention"] = summarize_eval(eval_sets(model, tok, tag_ev, args, dev))
    # Also measure training fit on the old tag rows after continuation.
    results["optional_tag_refresh"] = {"best_train_acc": None, "fit": train_breakdown(model, tok, tag_tr, args, dev), "history": []}

    summary = {"status": "PREFIX_WARMSTART", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "results": results,
               "evals": eval_blocks,
               "boundary": "Small pretrained bridge; continuation test of optimization access, not BabyLM-scale training."}
    (out / "prefix_warmstart_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "json": str(out / "prefix_warmstart_summary.json"),
                      "md": str(out / "prefix_warmstart_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
