#!/usr/bin/env python3
"""research: Prephase alignment probe — aligned vs permuted vs disjoint warm-start.

research showed that tag-only pretraining can rescue inline-role direct acquisition
for changed focal records that scratch training fails on.  This probe tests whether
the rescue is:
  (a) genuine aligned coordinate reuse (only aligned pretraining helps)
  (b) tag-token familiarity (aligned + permuted help, disjoint doesn't)
  (c) generic warm-up / format learning (all prephase conditions help)

Arms (all continue on identical inline-role direct training with correct aligned mapping):
  aligned:  tag-only prephase with correct tag→record mapping (same as research)
  permuted: tag-only prephase with swapped focal/secondary tag→record mapping
  disjoint: tag-only prephase with different-namespace tags (correct mapping for those)
  scratch:  no prephase; direct inline-role training only

Prediction table:
  Only aligned rescues changed focal → genuine coordinate reuse
  Aligned + permuted rescue, disjoint fails → tag-token familiarity
  All rescue → generic warm-up / format learning
  None rescues → result is seed/path-specific, not a reliable mechanism

Scientific boundary: small pretrained bridge only; not BabyLM-scale training.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

# ── imports from the established research/270 chain ───────────────────────────
import importlib.util
STUDY = Path("experiments/archive/representation_and_objectives")

ST270_PATH = STUDY / "training/scripts/prefix_warmstart_probe.py"
spec270 = importlib.util.spec_from_file_location("st270", ST270_PATH)
st270 = importlib.util.module_from_spec(spec270)
assert spec270.loader is not None
sys.modules[spec270.name] = st270
spec270.loader.exec_module(st270)

s269 = st270.s269
base = st270.base
DEFAULT_MODEL = st270.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/prephase_alignment"

# ── permuted tag-only row builder ────────────────────────────────────────────
def build_permuted_tag_rows(focals, seconds, *, ns, split, eset, arm,
                            queries, tkind, qmode="direct_tag", style="train"):
    """Build tag-only rows with swapped focal and secondary tag assignments.

    Uses the SAME namespace as aligned, generating the SAME 4 tag tokens per
    world, but assigns tf_b ↔ tf_a and ts_b ↔ ts_a.  This means a hypothesis
    about the before-state references what was the after-tag in aligned, and
    vice versa.  Labels remain correct for the ACTUAL content under each tag.

    After this prephase, TAG_1 is associated with the after-state; in the
    aligned continuation, TAG_1 maps to the before-state.  The model must
    UNLEARN its associations during continuation if it uses them.
    """
    rows = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                # Generate the SAME tags as aligned (same namespace)
                tags = s269.tag4(f"{ns}|{f.world_id}|{s.world_id}|{vi}|{vl}")
                orig_fb, orig_fa, orig_sb, orig_sa = tags
                # SWAP: focal before ↔ after, secondary before ↔ after
                tf_b, tf_a = orig_fa, orig_fb
                ts_b, ts_a = orig_sa, orig_sb
                tmap = {"focal_before": tf_b, "focal_after": tf_a,
                        "secondary_before": ts_b, "secondary_after": ts_a}
                # Build tag_only entries with the swapped tags
                entries = [
                    f"Entry {tf_b}: " + s269.sent_i(f, am, vi),
                    f"Entry {tf_a}: " + s269.sent_l(f, am, vl),
                    f"Entry {ts_b}: " + s269.sent_i(s, am, vi + 1),
                    f"Entry {ts_a}: " + s269.sent_l(s, am, vl + 1)]
                sk = f"{f.world_id}|{s.world_id}|{vi}|{vl}"
                ctx = " ".join(s269.shuf(entries, sk))
                tpl = f"ctx=tag_only|vi={vi}|vl={vl}|sw=False"

                for hd in ["AB", "BA"]:
                    for qf in queries:
                        qw = f if qf.startswith("focal") else s
                        h = s269.hyp_for(qf=qf, qw=qw, am=am, hd=hd,
                                         qmode=qmode, tag=tmap[qf], style=style)
                        lab = s269.lab_normal(qf, f, s, hd)
                        rows.append(base.mk(
                            row_id=f"{ns}|permuted|{qmode}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=h, label=lab,
                            split=split, eval_set=eset, arm=arm,
                            query_family=qf, focal_world=f.world_id,
                            secondary_world=s.world_id,
                            changed_focal=bool(f.changed),
                            stable_secondary=not bool(s.changed),
                            template_group=tpl,
                            train_kind=f"{tkind}|ctx=tag_only|q={qmode}",
                            hyp_dir=hd,
                            pair_key=f"{ns}|perm|{qmode}|{tpl}|{qf}",
                            hyp_wording=style, ctx_wording="tag_only",
                            label_mode="permuted"))
    return rows


# ── construction helpers ─────────────────────────────────────────────────────
def _ns(args, kind: str) -> str:
    """Shared address namespace for aligned/permuted/continuation (same tags)."""
    shared = args.shared_address_ns
    suffix = args.shared_address_suffix
    return f"{shared}_{kind}{suffix}"


def _ns_disjoint(args, kind: str) -> str:
    """Separate address namespace for the disjoint arm (different tags)."""
    return f"s271_disjoint_{kind}"


def build_aligned_prephase(args):
    """Standard tag-only training rows with correct tag→record mapping."""
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]
    tr = []
    ns = _ns(args, "b")
    tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                       ns=ns, mode="tag_only", split="train", eset="base",
                       arm="base", queries=s269.ALL_Q, tkind="base_stable_anchor",
                       qmode="direct_tag")
    if args.base_train_wording:
        ns_w = _ns(args, "bw")
        tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                           ns=ns_w, mode="tag_only", split="train", eset="base",
                           arm="base", queries=s269.ALL_Q, tkind="base_stable_train",
                           qmode="direct_tag")
    ns_c = _ns(args, "c")
    tr += s269.mk_rows(sel["sparse_changed"], secs,
                       ns=ns_c, mode="tag_only", split="train", eset="sparse",
                       arm="balanced_temporal", queries=s269.SPARSE_Q,
                       tkind="sparse_changed_focal", qmode="direct_tag")
    ns_s = _ns(args, "s")
    tr += s269.mk_rows(sel["sparse_stable"], secs,
                       ns=ns_s, mode="tag_only", split="train", eset="sparse",
                       arm="balanced_temporal", queries=s269.SPARSE_Q,
                       tkind="sparse_stable_focal", qmode="direct_tag")
    return tr, {"type": "aligned", "n": len(tr), "desc": base.describe_rows(tr)}


def build_permuted_prephase(args):
    """Tag-only rows with swapped focal/secondary tag assignments."""
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]
    tr = []
    ns = _ns(args, "b")
    tr += build_permuted_tag_rows(sel["base_stable"], sel["base_stable2"],
                                  ns=ns, split="train", eset="base", arm="base",
                                  queries=s269.ALL_Q, tkind="base_stable_anchor")
    if args.base_train_wording:
        ns_w = _ns(args, "bw")
        tr += build_permuted_tag_rows(sel["base_stable"], sel["base_stable2"],
                                      ns=ns_w, split="train", eset="base",
                                      arm="base", queries=s269.ALL_Q,
                                      tkind="base_stable_train")
    ns_c = _ns(args, "c")
    tr += build_permuted_tag_rows(sel["sparse_changed"], secs,
                                  ns=ns_c, split="train", eset="sparse",
                                  arm="balanced_temporal", queries=s269.SPARSE_Q,
                                  tkind="sparse_changed_focal")
    ns_s = _ns(args, "s")
    tr += build_permuted_tag_rows(sel["sparse_stable"], secs,
                                  ns=ns_s, split="train", eset="sparse",
                                  arm="balanced_temporal", queries=s269.SPARSE_Q,
                                  tkind="sparse_stable_focal")
    return tr, {"type": "permuted", "n": len(tr), "desc": base.describe_rows(tr)}


def build_disjoint_prephase(args):
    """Tag-only rows with correct mapping but a DIFFERENT tag namespace."""
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]
    tr = []
    ns = _ns_disjoint(args, "b")
    tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                       ns=ns, mode="tag_only", split="train", eset="base",
                       arm="base", queries=s269.ALL_Q, tkind="base_stable_anchor",
                       qmode="direct_tag")
    if args.base_train_wording:
        ns_w = _ns_disjoint(args, "bw")
        tr += s269.mk_rows(sel["base_stable"], sel["base_stable2"],
                           ns=ns_w, mode="tag_only", split="train", eset="base",
                           arm="base", queries=s269.ALL_Q, tkind="base_stable_train",
                           qmode="direct_tag")
    ns_c = _ns_disjoint(args, "c")
    tr += s269.mk_rows(sel["sparse_changed"], secs,
                       ns=ns_c, mode="tag_only", split="train", eset="sparse",
                       arm="balanced_temporal", queries=s269.SPARSE_Q,
                       tkind="sparse_changed_focal", qmode="direct_tag")
    ns_s = _ns_disjoint(args, "s")
    tr += s269.mk_rows(sel["sparse_stable"], secs,
                       ns=ns_s, mode="tag_only", split="train", eset="sparse",
                       arm="balanced_temporal", queries=s269.SPARSE_Q,
                       tkind="sparse_stable_focal", qmode="direct_tag")
    return tr, {"type": "disjoint", "n": len(tr), "desc": base.describe_rows(tr)}


def build_inline_continuation(args):
    """Inline-role direct rows with ALIGNED (correct) namespace — identical for all arms."""
    return st270.build_train_eval(args, "inline_role", "direct_tag")


def build_eval_suite(args):
    """Evaluation on aligned inline-role direct and tag-only direct."""
    _, tag_ev, _ = st270.build_train_eval(args, "tag_only", "direct_tag")
    _, inline_ev, _ = st270.build_train_eval(args, "inline_role", "direct_tag")
    return {"tag_evals": tag_ev, "inline_evals": inline_ev}


# ── training and evaluation ──────────────────────────────────────────────────
def train_loop(model, tok, rows, args, dev, *, seed, epochs, label,
               head_lr=None, encoder_lr=None, track_evals=None):
    """Train for a fixed number of epochs, tracking best accuracy."""
    torch.manual_seed(seed); np.random.seed(seed)
    h_lr = head_lr or args.head_lr
    e_lr = encoder_lr or args.encoder_lr
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": head_p, "lr": h_lr},
        {"params": enc_p, "lr": e_lr},
    ], weight_decay=args.weight_decay)
    x, mask, y = base.encode_rows(tok, rows, args.max_len, dev)
    loader = DataLoader(TensorDataset(x, mask, y), batch_size=args.batch_size, shuffle=True)
    best_state = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})
    best_acc = -1.0
    hist = []
    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss))
        fit = base.train_breakdown_from_logits(
            base.get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size), rows)
        ta = float(fit.get("overall", float("nan")))
        ev_snap = {}
        if track_evals:
            for en, ev_rows in track_evals.items():
                ev_out = base.full_eval(model, tok, {en: ev_rows}, args.max_len, dev, args.eval_batch_size)
                c = ev_out[en]["contrastive"]
                ev_snap[en] = {"con": c.get("contrastive_acc"),
                               "bq": c.get("by_query", {})}
        hist.append({"ep": ep, "loss": float(np.mean(losses)), "ta": ta,
                     "fit": fit.get("by_train_kind", {}), "ev": ev_snap})
        print(json.dumps({"event": "epoch", "phase": label, "epoch": ep,
                          "train_acc": ta, "fit": fit.get("by_train_kind", {}),
                          "ev_snap": {k: v.get("bq", {}) for k, v in ev_snap.items()}}),
              flush=True)
        if ta > best_acc:
            best_acc = ta
            best_state = copy.deepcopy({k: v.cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    final_fit = base.train_breakdown_from_logits(
        base.get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size), rows)
    return {"phase": label, "best_train_acc": float(best_acc),
            "fit": final_fit, "history": hist}


def full_eval(model, tok, eval_dict, args, dev):
    out = {}
    for en, rows in eval_dict.items():
        ev = base.full_eval(model, tok, {en: rows}, args.max_len, dev, args.eval_batch_size)
        c = ev[en]["contrastive"]
        out[en] = {"con_acc": c.get("contrastive_acc"),
                   "by_q": c.get("by_query", {}),
                   "by_m": c.get("by_query_margin", {})}
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["aligned", "permuted", "disjoint", "scratch"])
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--seed", type=int, default=27100)
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
    ap.add_argument("--prephase_epochs", type=int, default=10)
    ap.add_argument("--continuation_epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    arm = args.arm
    if not args.out_dir:
        args.out_dir = str(DEFAULT_OUT / f"{arm}_seed{args.seed}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    # Build continuation rows (same for all arms): inline-role direct with aligned namespace
    cont_tr, cont_ev, cont_con = build_inline_continuation(args)
    # Build eval suite (same for all arms)
    evsuite = build_eval_suite(args)
    all_inline_evals = evsuite["inline_evals"]
    all_tag_evals = evsuite["tag_evals"]

    # Build prephase rows
    if arm == "aligned":
        pre_tr, pre_con = build_aligned_prephase(args)
    elif arm == "permuted":
        pre_tr, pre_con = build_permuted_prephase(args)
    elif arm == "disjoint":
        pre_tr, pre_con = build_disjoint_prephase(args)
    else:  # scratch
        pre_tr, pre_con = [], {"type": "scratch", "n": 0}

    # Token audit
    all_rows = pre_tr + cont_tr
    for ev in [all_inline_evals, all_tag_evals]:
        for rows in ev.values():
            all_rows += rows
    lens = [len(tok.encode(r["text"], add_special_tokens=True)) for r in all_rows]
    arr = np.array(lens, dtype=float)
    token_audit = {"n": len(lens), "min": int(arr.min()), "p50": float(np.percentile(arr, 50)),
                   "p90": float(np.percentile(arr, 90)), "max": int(arr.max()),
                   "frac_gt_max": float(np.mean(arr > args.max_len))}

    construction = {"arm": arm, "prephase": pre_con, "continuation": cont_con,
                    "pre_rows": len(pre_tr), "cont_rows": len(cont_tr), "token_audit": token_audit}
    print(json.dumps({"status": "BUILT", "arm": arm, "pre_rows": len(pre_tr),
                      "cont_rows": len(cont_tr), "token_audit": token_audit}), flush=True)

    if args.dry_build:
        summary = {"status": "DRY", "arm": arm, "args": vars(args), "construction": construction}
        (out / "alignment_summary.json").write_text(json.dumps(summary, indent=2)+"\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE", "arm": arm, "json": str(out / "alignment_summary.json")}), flush=True)
        return

    dev = args.device
    results = {}
    evals = {}

    # ── PREPHASE ─────────────────────────────────────────────────────────────
    model = st270.make_model(Path(args.model_path), dev, seed=args.seed)

    if arm != "scratch" and pre_tr:
        results["prephase"] = train_loop(
            model, tok, pre_tr, args, dev, seed=args.seed,
            epochs=args.prephase_epochs, label=f"prephase_{arm}",
            track_evals={**all_inline_evals, **all_tag_evals})
        evals["after_prephase_tag"] = full_eval(model, tok, all_tag_evals, args, dev)
        evals["after_prephase_inline"] = full_eval(model, tok, all_inline_evals, args, dev)
    else:
        results["prephase"] = None
        evals["after_prephase_tag"] = full_eval(model, tok, all_tag_evals, args, dev)
        evals["after_prephase_inline"] = full_eval(model, tok, all_inline_evals, args, dev)

    # ── CONTINUATION (identical for all arms) ────────────────────────────────
    results["continuation"] = train_loop(
        model, tok, cont_tr, args, dev, seed=args.seed + 2,
        epochs=args.continuation_epochs, label=f"continuation_{arm}",
        head_lr=args.head_lr, encoder_lr=args.encoder_lr,
        track_evals={**all_inline_evals, **all_tag_evals})
    evals["after_continuation_inline"] = full_eval(model, tok, all_inline_evals, args, dev)
    evals["after_continuation_tag"] = full_eval(model, tok, all_tag_evals, args, dev)

    # ── Also check prephase data fit after continuation ──────────────────────
    if pre_tr:
        pre_fit = base.train_breakdown_from_logits(
            base.get_all_logits(model, tok, pre_tr, args.max_len, dev, args.eval_batch_size), pre_tr)
        results["prephase_retention"] = {"fit": pre_fit}

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {"status": "DONE", "arm": arm, "seed": args.seed,
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": construction,
               "results": results, "evals": evals,
               "boundary": "Small pretrained bridge; prephase alignment test, not BabyLM training."}
    (out / "alignment_summary.json").write_text(json.dumps(summary, indent=2)+"\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "arm": arm,
                      "json": str(out / "alignment_summary.json"),
                      "md": str(out / "alignment_summary.md")}), flush=True)


def write_md(summary, out):
    arm = summary["arm"]
    lines = [f"# research prephase alignment — {arm}\n\n"]
    lines.append(f"Seed: {summary['seed']}. Prephase rows: {summary['construction']['pre_rows']}. "
                 f"Continuation rows: {summary['construction']['cont_rows']}.\n\n")

    for phase in ["prephase", "continuation"]:
        r = summary["results"].get(phase)
        if r is None:
            lines.append(f"## {phase}\n\nSkipped (scratch arm).\n\n")
            continue
        lines.append(f"## {phase}\n\n")
        lines.append(f"Best train acc: {r['best_train_acc']:.6f}\n\n")
        fit = r.get("fit", {}).get("by_train_kind", {})
        lines.append(f"Fit: `{json.dumps(fit, sort_keys=True)}`\n\n")

    lines.append("## Evaluations\n\n")
    for block, evdict in sorted(summary.get("evals", {}).items()):
        lines.append(f"### {block}\n\n")
        lines.append("| eval | con | fb | fa | sb | sa | fa_m | sa_m |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in sorted(evdict.items()):
            bq, bm = ev.get("by_q", {}), ev.get("by_m", {})
            gv = lambda q: float(bq.get(q, float("nan")))
            gm = lambda q: float(bm.get(q, float("nan")))
            lines.append(f"| {en} | {ev.get('con_acc', float('nan')):.3f} "
                         f"| {gv('focal_before'):.3f} | {gv('focal_after'):.3f} "
                         f"| {gv('secondary_before'):.3f} | {gv('secondary_after'):.3f} "
                         f"| {gm('focal_after'):.2f} | {gm('secondary_after'):.2f} |\n")
        lines.append("\n")

    if "prephase_retention" in summary["results"]:
        pr = summary["results"]["prephase_retention"]
        lines.append("## Prephase retention after continuation\n\n")
        lines.append(f"Fit: `{json.dumps(pr.get('fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")

    (out / "alignment_summary.md").write_text("".join(lines), "utf-8")


if __name__ == "__main__":
    main()
