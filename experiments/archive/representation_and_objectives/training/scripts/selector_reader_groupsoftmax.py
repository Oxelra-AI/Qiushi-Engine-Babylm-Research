#!/usr/bin/env python3
"""research repair: selector-reader bridge with group-softmax selector.

The first research selector run used a 3:1 binary candidate objective and, because
selector seed 27100 is a known bad head seed from research, collapsed to predicting
all candidates negative.  This wrapper keeps the same construction but trains the
selector M with a four-way group softmax over the candidate tags for each role.
That directly optimizes role->tag top-1 and removes the all-negative shortcut.

R is still trained first and then held fixed; M never sees state labels; composed
evaluation uses M-selected tags, not oracle routing.
"""
from __future__ import annotations

import argparse, copy, importlib.util, json, sys, time
from pathlib import Path
from typing import Any
import collections

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
ST272_PATH = STUDY / "training/scripts/selector_reader_bridge.py"
spec = importlib.util.spec_from_file_location("st272", ST272_PATH)
st272 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = st272
spec.loader.exec_module(st272)

st270 = st272.st270
base = st272.base
DEFAULT_MODEL = st272.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/selector_reader_groupsoftmax"


def group_tensors(tok, rows: list[dict[str, Any]], args, dev):
    x, mask, _y = base.encode_rows(tok, rows, args.max_len, dev)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["select_group"]].append(i)
    group_keys = []
    idxs = []
    labels = []
    for g, inds in groups.items():
        # Preserve original candidate order inside the group; exactly one positive.
        inds = list(inds)
        labs = [int(rows[i]["label"]) for i in inds]
        assert len(inds) == 4, (g, len(inds))
        assert sum(labs) == 1, (g, labs)
        group_keys.append(g)
        idxs.append(inds)
        labels.append(labs.index(1))
    return x, mask, torch.tensor(idxs, dtype=torch.long, device=dev), torch.tensor(labels, dtype=torch.long, device=dev), group_keys


def train_selector_group(model, tok, rows, args, dev, *, seed: int, epochs: int,
                         label: str, track_evals: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    torch.manual_seed(seed); np.random.seed(seed)
    opt = st272.make_opt(model, args)
    x, mask, group_idx, y, group_keys = group_tensors(tok, rows, args, dev)
    ds = TensorDataset(torch.arange(group_idx.shape[0], device=dev), y)
    loader = DataLoader(ds, batch_size=args.selector_group_batch_size, shuffle=True)
    best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    best_top1 = -1.0
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for gb, yb in loader:
            inds = group_idx[gb]  # (B,4)
            xb = x[inds.reshape(-1)]
            mb = mask[inds.reshape(-1)]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mb)
            scores = (logits[:, 1] - logits[:, 0]).reshape(inds.shape[0], inds.shape[1])
            loss = F.cross_entropy(scores, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss.item()))
        logits_all = st272.get_logits(model, tok, rows, args, dev)
        top = st272.selector_top1(logits_all, rows)
        binb = st272.selector_binary_breakdown(logits_all, rows)
        ev_snap = {}
        if track_evals:
            for en, evrows in track_evals.items():
                evlog = st272.get_logits(model, tok, evrows, args, dev)
                ev_snap[en] = st272.selector_top1(evlog, evrows)
        hist.append({"epoch": ep, "loss": float(np.mean(losses)),
                     "train_top1": top, "train_binary": binb,
                     "eval_top1": ev_snap})
        print(json.dumps({"event": "epoch", "phase": label, "epoch": ep,
                          "loss": hist[-1]["loss"], "train_top1": top.get("top1"),
                          "row_acc": binb.get("row_acc"),
                          "pos_rate_pred": binb.get("pos_rate_pred"),
                          "eval_top1": {k: v.get("top1") for k, v in ev_snap.items()}}), flush=True)
        if float(top.get("top1", -1.0)) > best_top1:
            best_top1 = float(top.get("top1", -1.0))
            best_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    final_logits = st272.get_logits(model, tok, rows, args, dev)
    return {"phase": label, "objective": "group_softmax_4way",
            "best_train_top1": float(best_top1),
            "final_top1": st272.selector_top1(final_logits, rows),
            "final_binary": st272.selector_binary_breakdown(final_logits, rows),
            "history": hist}


def write_md(summary: dict[str, Any], out: Path) -> None:
    # Reuse the original writer; it reports the same fields and objective appears in JSON.
    st272.write_md(summary, out)
    p = out / "selector_reader_summary.md"
    txt = p.read_text("utf-8")
    txt = txt.replace("# research selector-reader bridge\n",
                      "# research selector-reader bridge — group-softmax selector repair\n")
    txt += "\n## Selector objective repair\n\nThis run replaces the 3:1 binary candidate objective with a four-way group softmax over candidate tags. It directly optimizes selector top-1 and removes the all-negative row-accuracy shortcut observed in the first research run.\n"
    p.write_text(txt, "utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=27000)
    ap.add_argument("--selector_seed", type=int, default=27000)
    ap.add_argument("--mode", default="inline_role", choices=["inline_role"])
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
    ap.add_argument("--selector_group_batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--reader_tag_epochs", type=int, default=5)
    ap.add_argument("--reader_direct_epochs", type=int, default=6)
    ap.add_argument("--selector_epochs", type=int, default=8)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    reader_tag_tr, reader_tag_ev, reader_tag_con = st270.build_train_eval(args, "tag_only", "direct_tag")
    reader_direct_tr, reader_direct_ev, reader_direct_con = st270.build_train_eval(args, args.mode, "direct_tag")
    sel_tr, sel_con = st272.selector_train_rows(args)
    sel_evals = st272.selector_eval_suite(args)
    all_rows = reader_tag_tr + reader_direct_tr + sel_tr
    for d in [reader_tag_ev, reader_direct_ev, sel_evals]:
        for rows in d.values():
            all_rows += rows
    ta = st272.token_audit(tok, all_rows, args.max_len)
    construction = {
        "reader_tag_train_rows": len(reader_tag_tr),
        "reader_train_rows": len(reader_direct_tr),
        "selector_train_rows": len(sel_tr),
        "reader_tag_train": reader_tag_con,
        "reader_direct_train": reader_direct_con,
        "selector_train": sel_con,
        "selector_eval_counts": {k: len(v) for k, v in sel_evals.items()},
        "selector_group_audit": {"train": st272.count_groups(sel_tr),
                                  **{k: st272.count_groups(v) for k, v in sel_evals.items()}},
        "token_audit": ta,
        "selector_objective": "group_softmax_4way",
    }
    print(json.dumps({"status": "BUILT", "reader_tag_rows": len(reader_tag_tr),
                      "reader_direct_rows": len(reader_direct_tr),
                      "selector_rows": len(sel_tr), "selector_groups": st272.count_groups(sel_tr),
                      "selector_evals": {k: len(v) for k, v in sel_evals.items()},
                      "token_audit": ta}), flush=True)
    if args.dry_build:
        summary = {"status": "DRY", "args": vars(args), "construction": construction,
                   "boundary": "Dry group-softmax construction only; no model training."}
        (out / "selector_reader_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "selector_reader_summary.json")}), flush=True)
        return

    dev = args.device
    results: dict[str, Any] = {}
    reader = st270.make_model(Path(args.model_path), dev, seed=args.seed)
    results["reader_tag_pretrain"] = st270.train_loop(
        reader, tok, reader_tag_tr, args, dev, seed=args.seed,
        epochs=args.reader_tag_epochs, label="reader_tag_pretrain",
        init_best_from_current=False, track_eval_rows=reader_tag_ev)
    results["reader_inline_direct"] = st270.train_loop(
        reader, tok, reader_direct_tr, args, dev, seed=args.seed + 1,
        epochs=args.reader_direct_epochs, label="reader_inline_direct",
        init_best_from_current=True, track_eval_rows=reader_direct_ev)
    reader_direct_eval = st270.summarize_eval(st270.eval_sets(reader, tok, reader_direct_ev, args, dev))

    selector = st270.make_model(Path(args.model_path), dev, seed=args.selector_seed)
    results["selector_train"] = train_selector_group(
        selector, tok, sel_tr, args, dev, seed=args.selector_seed,
        epochs=args.selector_epochs, label="selector_group_softmax", track_evals=sel_evals)
    selector_evals = st272.selector_eval(selector, tok, sel_evals, args, dev)
    comp = st272.composition_eval(selector, reader, tok, sel_evals, args, dev)

    summary = {"status": "DONE", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction, "results": results,
               "reader_direct_eval": reader_direct_eval,
               "selector_evals": selector_evals, "composition": comp,
               "boundary": "Small pretrained selector-reader bridge with group-softmax selector. R is trained then held fixed; M is separate and sees no state labels; composition uses M-selected tags, with oracle-reader ceiling reported only as a reference."}
    (out / "selector_reader_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "json": str(out / "selector_reader_summary.json"),
                      "md": str(out / "selector_reader_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
