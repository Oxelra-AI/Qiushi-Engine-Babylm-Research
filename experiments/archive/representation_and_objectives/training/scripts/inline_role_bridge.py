#!/usr/bin/env python3
"""research: inline-role bridge — do co-located role annotations avoid collision?

research established that supplied unique entry tags enable sparse changed-focal
state selection but repeating those tags in *separate* sentences prevents changed-
focal acquisition.  This script tests whether placing role annotations INSIDE each
entry sentence (each tag appears exactly once) avoids that collision, and if so,
whether the model can learn role-based composition C(r,v).

Arms:
  tag_only       — entries without role info; R(k,v) baseline
  inline_direct  — inline role prefix + direct-tag queries; R with role context
  inline_role    — inline role prefix + role-phrase queries; C composition
  inline_both    — both query types jointly
  compact_direct — compact preamble role=tag + entries + direct-tag queries;
                   tag appears twice (preamble + entry), testing collision

For inline arms, evaluation includes role-swap (swap focal annotations, should
invert focal answers if following roles) and held paraphrases.

Scientific boundary: small pretrained bridge only; not BabyLM-scale training.
"""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
_spec = importlib.util.spec_from_file_location(
    "addr267", STUDY / "training/scripts/temporal_address_dissociation.py")
_addr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _addr
_spec.loader.exec_module(_addr)
base = _addr.base
DEFAULT_MODEL = _addr.DEFAULT_MODEL
TAG_POOL = _addr.TAG_POOL
DEFAULT_OUT = STUDY / "data/inline_role_bridge"

# ── deterministic helpers ─────────────────────────────────────────────────────
def _si(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)

def tag4(key: str) -> list[str]:
    return random.Random(_si("s269t4|" + key)).sample(TAG_POOL, 4)

def shuf(parts: list[str], key: str) -> list[str]:
    """Shuffle entries deterministically; key should NOT include swap/mode so that
    physical order is consistent across conditions for the same worlds."""
    p = list(parts)
    random.Random(_si("s269s|" + key)).shuffle(p)
    return p

def sent_i(w, am, v):
    t = ["{H} was ranked above {Lo} in the ATP list.",
         "{Lo} was ranked below {H} in the ATP list."]
    return t[v % 2].format(H=base.an(am, w.higher_initial), Lo=base.an(am, w.lower_initial))

def sent_l(w, am, v):
    t = ["{H} was ranked above {Lo} in the ATP list.",
         "{Lo} was ranked below {H} in the ATP list."]
    return t[v % 2].format(H=base.an(am, w.higher_later), Lo=base.an(am, w.lower_later))

# scope/time maps
SCOPE = {"focal_before": "focal", "focal_after": "focal",
         "secondary_before": "separate", "secondary_after": "separate"}
TNAME = {"focal_before": "background", "focal_after": "update",
         "secondary_before": "background", "secondary_after": "update"}
TPARA = {"background": "prior", "update": "revised"}

# ── context builders ──────────────────────────────────────────────────────────
def build_ctx(*, f, s, am, mode, ns, vi, vl, swap=False):
    """Returns (context_str, tag_map, template_descriptor).

    swap: if True, focal role annotations are exchanged (background ↔ update).
    Physical shuffle order is world-deterministic and independent of swap.
    """
    tags = tag4(f"{ns}|{f.world_id}|{s.world_id}|{vi}|{vl}")
    tf_b, tf_a, ts_b, ts_a = tags
    tmap = {"focal_before": tf_b, "focal_after": tf_a,
            "secondary_before": ts_b, "secondary_after": ts_a}
    # role labels
    fb_r, fa_r = ("update", "background") if swap else ("background", "update")
    sb_r, sa_r = ("update", "background") if swap else ("background", "update")
    # world-only shuffle key for consistent physical order
    sk = f"{f.world_id}|{s.world_id}|{vi}|{vl}"

    if mode == "tag_only":
        entries = [
            f"Entry {tf_b}: " + sent_i(f, am, vi),
            f"Entry {tf_a}: " + sent_l(f, am, vl),
            f"Entry {ts_b}: " + sent_i(s, am, vi + 1),
            f"Entry {ts_a}: " + sent_l(s, am, vl + 1)]
        ctx = " ".join(shuf(entries, sk))

    elif mode == "inline_role":
        entries = [
            f"Focal {fb_r} entry {tf_b}: " + sent_i(f, am, vi),
            f"Focal {fa_r} entry {tf_a}: " + sent_l(f, am, vl),
            f"Separate {sb_r} entry {ts_b}: " + sent_i(s, am, vi + 1),
            f"Separate {sa_r} entry {ts_a}: " + sent_l(s, am, vl + 1)]
        ctx = " ".join(shuf(entries, sk))

    elif mode == "postfix_role":
        entries = [
            f"Entry {tf_b} (focal {fb_r}): " + sent_i(f, am, vi),
            f"Entry {tf_a} (focal {fa_r}): " + sent_l(f, am, vl),
            f"Entry {ts_b} (separate {sb_r}): " + sent_i(s, am, vi + 1),
            f"Entry {ts_a} (separate {sa_r}): " + sent_l(s, am, vl + 1)]
        ctx = " ".join(shuf(entries, sk))

    elif mode == "suffix_role":
        entries = [
            f"Entry {tf_b}: " + sent_i(f, am, vi) + f" This is the focal {fb_r}.",
            f"Entry {tf_a}: " + sent_l(f, am, vl) + f" This is the focal {fa_r}.",
            f"Entry {ts_b}: " + sent_i(s, am, vi + 1) + f" This is the separate {sb_r}.",
            f"Entry {ts_a}: " + sent_l(s, am, vl + 1) + f" This is the separate {sa_r}."]
        ctx = " ".join(shuf(entries, sk))

    elif mode == "prefix_one":
        entries = [
            f"The entry {tf_b}: " + sent_i(f, am, vi),
            f"The entry {tf_a}: " + sent_l(f, am, vl),
            f"The entry {ts_b}: " + sent_i(s, am, vi + 1),
            f"The entry {ts_a}: " + sent_l(s, am, vl + 1)]
        ctx = " ".join(shuf(entries, sk))

    elif mode == "compact_map":
        pre = (f"Role map: focal-bg={tf_b}, focal-up={tf_a}, "
               f"sep-bg={ts_b}, sep-up={ts_a}.")
        entries = [
            f"Entry {tf_b}: " + sent_i(f, am, vi),
            f"Entry {tf_a}: " + sent_l(f, am, vl),
            f"Entry {ts_b}: " + sent_i(s, am, vi + 1),
            f"Entry {ts_a}: " + sent_l(s, am, vl + 1)]
        ctx = pre + " " + " ".join(shuf(entries, sk))

    else:
        raise ValueError(mode)

    tpl = f"ctx={mode}|vi={vi}|vl={vl}|sw={swap}"
    return ctx, tmap, tpl


# ── hypothesis builders ───────────────────────────────────────────────────────
def hyp_for(*, qf, qw, am, hd, qmode, tag, style):
    x, y = _addr.hyp_xy(qw, am, hd)
    if qmode == "direct_tag":
        if style == "train":
            return f"According to entry {tag}, {x} was ranked higher than {y}."
        return f"Using the record named {tag}, {x} outranked {y}."
    if qmode == "role_inline":
        role = f"{SCOPE[qf]} {TNAME[qf]}"
        if style == "train":
            return f"According to the {role} entry, {x} was ranked higher than {y}."
        return f"Using the {role} record, {x} outranked {y}."
    if qmode == "role_para":
        role = f"{SCOPE[qf]} {TPARA[TNAME[qf]]}"
        return f"Using the {role} record, {x} outranked {y}."
    raise ValueError(qmode)


# ── labels ────────────────────────────────────────────────────────────────────
def lab_normal(qf, f, s, hd):
    return _addr.labels_for(qf, f, s, hd)

def lab_swap(qf, f, s, hd):
    """Labels when focal role annotations are swapped: background→later, update→initial."""
    if qf == "focal_before":     return base.later_label(f, hd)
    if qf == "focal_after":      return base.initial_label(f, hd)
    if qf == "secondary_before": return base.initial_label(s, hd)
    if qf == "secondary_after":  return base.later_label(s, hd)
    raise ValueError(qf)


# ── row generation ────────────────────────────────────────────────────────────
ALL_Q = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
SPARSE_Q = ["focal_before", "focal_after"]

def mk_rows(focals, seconds, *, ns, mode, split, eset, arm, queries, tkind,
            qmode, style="train", swap=False, swap_labels=False):
    rows = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                ctx, tm, tpl = build_ctx(f=f, s=s, am=am, mode=mode, ns=ns,
                                         vi=vi, vl=vl, swap=swap)
                for hd in ["AB", "BA"]:
                    for qf in queries:
                        qw = f if qf.startswith("focal") else s
                        h = hyp_for(qf=qf, qw=qw, am=am, hd=hd, qmode=qmode,
                                    tag=tm[qf], style=style)
                        # For role queries in swap condition, use swapped labels
                        if swap_labels and qmode != "direct_tag":
                            lab = lab_swap(qf, f, s, hd)
                        else:
                            lab = lab_normal(qf, f, s, hd)
                        rows.append(base.mk(
                            row_id=f"{ns}|{mode}|{qmode}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=h, label=lab,
                            split=split, eval_set=eset, arm=arm,
                            query_family=qf, focal_world=f.world_id,
                            secondary_world=s.world_id,
                            changed_focal=bool(f.changed),
                            stable_secondary=not bool(s.changed),
                            template_group=tpl,
                            train_kind=f"{tkind}|ctx={mode}|q={qmode}",
                            hyp_dir=hd,
                            pair_key=f"{ns}|{mode}|{qmode}|{tpl}|{qf}",
                            hyp_wording=style, ctx_wording=mode,
                            label_mode="inline"))
    return rows


# ── arm configuration ─────────────────────────────────────────────────────────
ARM_CFGS = {
    "tag_only":        {"ctx": "tag_only",      "tq": ["direct_tag"]},
    "inline_direct":   {"ctx": "inline_role",   "tq": ["direct_tag"]},
    "inline_role":     {"ctx": "inline_role",   "tq": ["role_inline"]},
    "inline_both":     {"ctx": "inline_role",   "tq": ["direct_tag", "role_inline"]},
    "compact_direct":  {"ctx": "compact_map",   "tq": ["direct_tag"]},
    "postfix_direct":  {"ctx": "postfix_role",  "tq": ["direct_tag"]},
    "postfix_role":    {"ctx": "postfix_role",  "tq": ["role_inline"]},
    "suffix_direct":   {"ctx": "suffix_role",   "tq": ["direct_tag"]},
    "prefix1_direct":  {"ctx": "prefix_one",    "tq": ["direct_tag"]},
}


def build_arm(args, arm):
    c = ARM_CFGS[arm]
    ctx = c["ctx"]
    tqs = c["tq"]
    sel, inv = base.select_splits(args)
    secs = sel["sparse_secondary_stable"]

    # Training rows
    tr = []
    for qm in tqs:
        tr += mk_rows(sel["base_stable"], sel["base_stable2"],
                       ns=f"s269_{arm}_b_{qm}", mode=ctx, split="train",
                       eset="base", arm="base", queries=ALL_Q,
                       tkind="base_stable_anchor", qmode=qm)
        if getattr(args, "base_train_wording", False):
            tr += mk_rows(sel["base_stable"], sel["base_stable2"],
                           ns=f"s269_{arm}_bw_{qm}", mode=ctx, split="train",
                           eset="base", arm="base", queries=ALL_Q,
                           tkind="base_stable_train", qmode=qm)
        tr += mk_rows(sel["sparse_changed"], secs,
                       ns=f"s269_{arm}_c_{qm}", mode=ctx, split="train",
                       eset="sparse", arm="balanced_temporal",
                       queries=SPARSE_Q, tkind="sparse_changed_focal", qmode=qm)
        tr += mk_rows(sel["sparse_stable"], secs,
                       ns=f"s269_{arm}_s_{qm}", mode=ctx, split="train",
                       eset="sparse", arm="balanced_temporal",
                       queries=SPARSE_Q, tkind="sparse_stable_focal", qmode=qm)

    # Evaluation sets
    ev = {}
    foc, sec_e = sel["eval_held_changed"], sel["eval_held_stable"]
    pn = "hC_hS"

    # All arms: direct-tag held
    ev[f"{arm}_{pn}_direct_hH"] = mk_rows(
        foc, sec_e, ns=f"e269_{arm}_{pn}_dt", mode=ctx, split="held",
        eset=pn, arm="eval", queries=ALL_Q, tkind="eval",
        qmode="direct_tag", style="held")

    # Inline arms: role query + role-swap + paraphrase evaluations
    if ctx in ("inline_role", "postfix_role", "suffix_role"):
        # Role inline held
        ev[f"{arm}_{pn}_role_hH"] = mk_rows(
            foc, sec_e, ns=f"e269_{arm}_{pn}_ri", mode=ctx, split="held",
            eset=pn, arm="eval", queries=ALL_Q, tkind="eval",
            qmode="role_inline", style="held")
        # Role-swap with role query (swap labels reflect that background→later)
        ev[f"{arm}_{pn}_roleSwap_role_hH"] = mk_rows(
            foc, sec_e, ns=f"e269_{arm}_{pn}_rs_ri", mode=ctx, split="held",
            eset=pn, arm="eval", queries=ALL_Q, tkind="eval",
            qmode="role_inline", style="held", swap=True, swap_labels=True)
        # Role-swap with direct-tag (should be unchanged)
        ev[f"{arm}_{pn}_roleSwap_direct_hH"] = mk_rows(
            foc, sec_e, ns=f"e269_{arm}_{pn}_rs_dt", mode=ctx, split="held",
            eset=pn, arm="eval", queries=ALL_Q, tkind="eval",
            qmode="direct_tag", style="held", swap=True)
        # Held role paraphrase (background→prior, update→revised)
        ev[f"{arm}_{pn}_rolePara_hH"] = mk_rows(
            foc, sec_e, ns=f"e269_{arm}_{pn}_rp", mode=ctx, split="held",
            eset=pn, arm="eval", queries=ALL_Q, tkind="eval",
            qmode="role_para", style="held")

    con = {
        "arm": arm, "ctx": ctx, "tq": tqs, "inv": inv,
        "sel": {k: base.world_summary(v) for k, v in sel.items()},
        "tr": base.describe_rows(tr),
        "ev": {k: len(v) for k, v in sorted(ev.items())},
    }
    return tr, ev, con


# ── summary/markdown ──────────────────────────────────────────────────────────
def summarize(r):
    return {
        "best_train_acc": r["best_train_acc"],
        "train_fit": r.get("train_fit", {}),
        "evals": {
            en: {
                "con_acc": ev["contrastive"].get("contrastive_acc"),
                "by_q": ev["contrastive"].get("by_query", {}),
                "by_m": ev["contrastive"].get("by_query_margin", {}),
                "std_acc": ev["standard"].get("acc"),
            }
            for en, ev in sorted(r["evals"].items())
        },
    }


def write_md(summary, out):
    lines = [
        "# research inline-role bridge\n\n",
        "Tests whether co-located role annotations avoid the coordinate collision\n"
        "that research established for separated declaration sentences.\n"
        "Held readouts meaningful only after sparse changed-focal rows fit.\n\n",
    ]
    for arm, item in summary.items():
        lines.append(f"## {arm}\n\nBest train: {item['best_train_acc']:.4f}\n\n")
        tf = item.get("train_fit", {})
        if tf:
            lines.append(f"Fit by kind: `{json.dumps(tf.get('by_train_kind', {}), sort_keys=True)}`\n\n")
        lines.append("| eval | con | fb | fa | sb | sa | fa_m | sa_m |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in item.get("evals", {}).items():
            bq, bm = ev["by_q"], ev["by_m"]
            def v(q): x = bq.get(q); return float("nan") if x is None else float(x)
            def m(q): x = bm.get(q); return float("nan") if x is None else float(x)
            lines.append(
                f"| {en} | {ev['con_acc']:.3f} | {v('focal_before'):.3f} "
                f"| {v('focal_after'):.3f} | {v('secondary_before'):.3f} "
                f"| {v('secondary_after'):.3f} | {m('focal_after'):.2f} "
                f"| {m('secondary_after'):.2f} |\n")
        lines.append("\n")
    (out / "inline_role_bridge_summary.md").write_text("".join(lines), "utf-8")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--arms", default="tag_only,inline_direct")
    ap.add_argument("--seed_base", type=int, default=26900)
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_CFGS:
            raise ValueError(f"unknown arm: {a}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    built = {}
    cons = {}
    for arm in arms:
        tr, ev, con = build_arm(args, arm)
        built[arm] = (tr, ev)
        cons[arm] = con
        print(json.dumps({"status": "BUILT", "arm": arm, "train": len(tr),
                          "evals": len(ev), "sel": con["sel"]}), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    all_tr, all_ev = [], []
    for tr, ev in built.values():
        all_tr.extend(tr)
        for rows in ev.values():
            all_ev.extend(rows)

    def audit(rows):
        lens = [len(tok.encode(r["text"], add_special_tokens=True)) for r in rows]
        a = np.array(lens, dtype=float)
        return {"n": len(lens), "min": int(a.min()) if len(a) else 0,
                "p50": float(np.percentile(a, 50)) if len(a) else 0,
                "p90": float(np.percentile(a, 90)) if len(a) else 0,
                "max": int(a.max()) if len(a) else 0,
                "gt_max": float(np.mean(a > args.max_len)) if len(a) else 0}

    ta = {"train": audit(all_tr), "eval": audit(all_ev)}
    print(json.dumps({"status": "TOKEN_AUDIT", "audit": ta}), flush=True)

    if args.dry_build:
        dry = {"status": "DRY", "args": vars(args), "construction": cons,
               "token_audit": ta}
        (out / "inline_role_bridge_summary.json").write_text(
            json.dumps(dry, indent=2) + "\n", "utf-8")
        print(json.dumps({"status": "DRY_DONE",
                          "json": str(out / "inline_role_bridge_summary.json")}),
              flush=True)
        return

    results = {}
    for ai, arm in enumerate(arms):
        tr, ev = built[arm]
        seed = args.seed_base + 100 * ai
        r = base.train_one(Path(args.model_path), tok, "balanced_temporal",
                           seed, list(tr), ev, args, args.device)
        s = summarize(r)
        results[arm] = s
        for en, evr in sorted(r["evals"].items()):
            c = evr["contrastive"]
            print(json.dumps({
                "event": "eval", "arm": arm, "seed": seed, "set": en,
                "train_acc": r["best_train_acc"],
                "fit": r.get("train_fit", {}).get("by_train_kind", {}),
                "by_q": c.get("by_query"),
                "by_m": c.get("by_query_margin"),
                "con": c.get("contrastive_acc"),
            }), flush=True)

    final = {
        "status": "INLINE_ROLE_BRIDGE",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args), "construction": cons, "token_audit": ta,
        "summary": results,
        "boundary": "Small pretrained bridge; not BabyLM-scale training.",
    }
    (out / "inline_role_bridge_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(results, out)
    print(json.dumps({
        "status": "DONE",
        "json": str(out / "inline_role_bridge_summary.json"),
        "md": str(out / "inline_role_bridge_summary.md"),
    }), flush=True)


if __name__ == "__main__":
    main()
