#!/usr/bin/env python3
"""research: crossed event/ranking orientation with fixed secondary state.

Scientific purpose
------------------
research showed that flipping event, focal-state, and secondary-state labels
together is not enough to tell whether the learner has a reusable relation
coordinate, a head-wide polarity convention, or a template-specific mapping.
This script keeps the filler interface, pretrained encoder, paired AB-vs-BA
scoring, ATP conflict worlds, and held wording surfaces from research, but varies
sparse evidence for event and focal ranking independently while the secondary
ranking query is always trained with the true label.

Sparse relation arms:
  exposure        : no relation labels beyond the shared anchor base
  Etrue_Rtrue     : event true, focal ranking true, secondary ranking true
  Eflip_Rtrue     : event flipped, focal ranking true, secondary ranking true
  Etrue_Rflip     : event true, focal ranking flipped, secondary ranking true
  Eflip_Rflip     : event flipped, focal ranking flipped, secondary ranking true

Evaluation labels are always the real ATP / paired-world facts.  The key readout
is whether flipping one sparse orientation selectively changes only its own
query family, or whether event, focal ranking, and secondary ranking move
together as a broad output-polarity convention.  Non-equivalent context rows are
included as controls for context-side extraction: they mention the same aliases
but do not encode winner or ranking direction, so relational accuracy should
fall toward chance rather than remain high.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Model

PATH = Path("experiments/archive/representation_and_objectives/training/scripts/contrastive_wording_cross_probe.py")
spec = importlib.util.spec_from_file_location("mod", PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = m
spec.loader.exec_module(m)

DEFAULT_MODEL = m.DEFAULT_MODEL
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/independent_orientation_cross")

# Directional paraphrases and non-equivalent mention-only contexts.  The latter
# intentionally balance alias order across variants, so a first-mentioned rule
# cannot reliably recover the true relation.
m.EVENT_CTX.update({
    "dirpara": [
        "{W} beat {L} during the match.",
        "{L} fell to {W} during the match.",
    ],
    "nonpara": [
        "The match report mentioned {W} alongside {L}.",
        "The match report mentioned {L} alongside {W}.",
    ],
})
m.STATE_CTX.update({
    "dirpara": [
        "{H} was ahead of {Lo} in the ATP rankings.",
        "{Lo} followed {H} in the ATP rankings.",
    ],
    "nonpara": [
        "The ranking note mentioned {H} near {Lo}.",
        "The ranking note mentioned {Lo} near {H}.",
    ],
})

ARM_MODES: dict[str, tuple[str, str] | None] = {
    "exposure": None,
    "Etrue_Rtrue": ("true", "true"),
    "Eflip_Rtrue": ("flip", "true"),
    "Etrue_Rflip": ("true", "flip"),
    "Eflip_Rflip": ("flip", "flip"),
}


def apply_mode(label: int, mode: Literal["true", "flip"]) -> int:
    return int(label) if mode == "true" else 1 - int(label)


def compound_rows_indep(
    prims,
    secs,
    *,
    ev_grp: str,
    st_grp: str,
    ns: str,
    split: str,
    ev_set: str,
    arm: str,
    event_mode: Literal["true", "flip"] = "true",
    rank_mode: Literal["true", "flip"] = "true",
    untouched_mode: Literal["true", "flip"] = "true",
    tk: str = "compound",
    hyp_w: str = "train",
):
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prims):
        s = secs[(i * 7 + 3) % len(secs)]
        if s.world_id == p.world_id:
            s = secs[(i * 7 + 4) % len(secs)]
        am = m.amap4(p, s, ns)
        for ev in range(len(m.EVENT_CTX[ev_grp])):
            for sv in range(len(m.STATE_CTX[st_grp])):
                ctx = (
                    f"Match record: {m.ev_sent(p, am, ev_grp, ev)} "
                    f"Ranking record: {m.st_sent(p, am, st_grp, sv)} "
                    f"Separate ranking record: {m.st_sent(s, am, st_grp, (sv + 1))}"
                )
                tpl = f"e={ev_grp}:{ev}|s={st_grp}:{sv}"
                base_pk = f"{ns}|{p.world_id}|{s.world_id}|{tpl}"
                conflict = not bool(p.winner_higher_at_time)
                for hd in ["AB", "BA"]:
                    ev_lab = apply_mode(m.ev_label(p, hd), event_mode)
                    r = m.mk(
                        row_id=f"{base_pk}|event|{hd}|h={hyp_w}",
                        context=ctx,
                        hypothesis=m.hyp_ev(am, p, hd, hyp_w),
                        label=ev_lab,
                        split=split,
                        eval_set=ev_set,
                        arm=arm,
                        source=p.source,
                        sport=p.sport,
                        query_family="event_role",
                        primary_world=p.world_id,
                        secondary_world=s.world_id,
                        template_group=tpl,
                        label_mode=f"event={event_mode}|rank={rank_mode}|secondary={untouched_mode}",
                        conflict=conflict,
                        train_kind=tk,
                        hyp_dir=hd,
                        pair_key=f"{base_pk}|event|h={hyp_w}",
                        hyp_wording=hyp_w,
                        ctx_wording=f"event={ev_grp}|state={st_grp}",
                    )
                    r.update({"event_mode": event_mode, "rank_mode": rank_mode,
                              "untouched_mode": untouched_mode,
                              "event_ctx_group": ev_grp, "state_ctx_group": st_grp})
                    rows.append(r)

                    st_lab = apply_mode(m.st_label(p, hd), rank_mode)
                    r = m.mk(
                        row_id=f"{base_pk}|focal|{hd}|h={hyp_w}",
                        context=ctx,
                        hypothesis=m.hyp_st(am, p, hd, hyp_w),
                        label=st_lab,
                        split=split,
                        eval_set=ev_set,
                        arm=arm,
                        source=p.source,
                        sport=p.sport,
                        query_family="focal_state",
                        primary_world=p.world_id,
                        secondary_world=s.world_id,
                        template_group=tpl,
                        label_mode=f"event={event_mode}|rank={rank_mode}|secondary={untouched_mode}",
                        conflict=conflict,
                        train_kind=tk,
                        hyp_dir=hd,
                        pair_key=f"{base_pk}|focal|h={hyp_w}",
                        hyp_wording=hyp_w,
                        ctx_wording=f"event={ev_grp}|state={st_grp}",
                    )
                    r.update({"event_mode": event_mode, "rank_mode": rank_mode,
                              "untouched_mode": untouched_mode,
                              "event_ctx_group": ev_grp, "state_ctx_group": st_grp})
                    rows.append(r)

                    un_lab = apply_mode(m.st_label(s, hd), untouched_mode)
                    r = m.mk(
                        row_id=f"{base_pk}|untouched|{hd}|h={hyp_w}",
                        context=ctx,
                        hypothesis=m.hyp_st(am, s, hd, hyp_w),
                        label=un_lab,
                        split=split,
                        eval_set=ev_set,
                        arm=arm,
                        source=s.source,
                        sport=s.sport,
                        query_family="untouched_state",
                        primary_world=p.world_id,
                        secondary_world=s.world_id,
                        template_group=tpl,
                        label_mode=f"event={event_mode}|rank={rank_mode}|secondary={untouched_mode}",
                        conflict=conflict,
                        train_kind=tk,
                        hyp_dir=hd,
                        pair_key=f"{base_pk}|untouched|h={hyp_w}",
                        hyp_wording=hyp_w,
                        ctx_wording=f"event={ev_grp}|state={st_grp}",
                    )
                    r.update({"event_mode": event_mode, "rank_mode": rank_mode,
                              "untouched_mode": untouched_mode,
                              "event_ctx_group": ev_grp, "state_ctx_group": st_grp})
                    rows.append(r)
    return rows


def select_worlds(args):
    atp_tr, atp_he = m.load_atp()
    pair_tr, pair_he = m.load_pairs()
    return {
        "atp_train": m.select(atp_tr, args.train_atp, 26301, True),
        "atp_train2": m.select(atp_tr, args.train_atp + 17, 26302, True),
        "atp_sparse": m.select(atp_tr, args.sparse_atp, 26303, True),
        "atp_sparse2": m.select(atp_tr, args.sparse_atp + 17, 26304, True),
        "atp_eval": m.select(atp_he, args.eval_atp, 26305, True),
        "atp_eval2": m.select(atp_he, args.eval_atp + 17, 26306, True),
        "pair_train": m.select(pair_tr, args.train_pair, 26307),
        "pair_sparse": m.select(pair_tr, args.sparse_pair, 26308),
        "pair_eval": m.select(pair_he, args.eval_pair, 26309),
    }


def build_datasets(args):
    w = select_worlds(args)
    base = []
    base += compound_rows_indep(
        w["atp_train"], w["atp_train2"], ev_grp="anchor", st_grp="anchor",
        ns="base_anchor263", split="train", ev_set="base", arm="base",
        event_mode="true", rank_mode="true", untouched_mode="true", tk="base_compound")
    base += m.event_only_rows(
        w["pair_train"], ev_grp="anchor", ns="base_pair263", split="train",
        ev_set="base", arm="base", lm="true", tk="base_pair_event")

    ctx_conditions = [
        ("trainTrain", "train", "train"),
        ("heldHeld", "held", "held"),
        ("dirDir", "dirpara", "dirpara"),
        ("trainEvent_dirState", "train", "dirpara"),
        ("dirEvent_trainState", "dirpara", "train"),
        ("trainEvent_nonState", "train", "nonpara"),
        ("nonEvent_trainState", "nonpara", "train"),
    ]

    def make_evals(arm_name: str):
        ev: dict[str, list[dict[str, Any]]] = {}
        for cname, eg, sg in ctx_conditions:
            for hw in ["train", "held"]:
                ev[f"atp_{cname}_{hw}Hyp"] = compound_rows_indep(
                    w["atp_eval"], w["atp_eval2"], ev_grp=eg, st_grp=sg,
                    ns=f"eval263_{cname}_{hw}", split="held",
                    ev_set=f"atp_{cname}_{hw}Hyp", arm=arm_name,
                    event_mode="true", rank_mode="true", untouched_mode="true",
                    tk="eval_compound", hyp_w=hw)
        for eg in ["train", "held", "dirpara", "nonpara"]:
            for hw in ["train", "held"]:
                ev[f"pair_{eg}Ctx_{hw}Hyp"] = m.event_only_rows(
                    w["pair_eval"], ev_grp=eg, ns=f"eval263_pair_{eg}_{hw}",
                    split="held", ev_set=f"pair_{eg}Ctx_{hw}Hyp",
                    arm=arm_name, lm="true", tk="eval_pair_event", hyp_w=hw)
        return ev

    arms: dict[str, dict[str, Any]] = {}
    for arm, modes in ARM_MODES.items():
        train = list(base)
        if modes is None:
            train += m.exposure_rows(
                w["atp_sparse"], w["atp_sparse2"], ev_grp="train", st_grp="train",
                ns="sp263_exp", split="train", ev_set="sparse", arm=arm)
            train += m.paired_exposure_event(
                w["pair_sparse"], ev_grp="train", ns="sp263_pair_exp",
                split="train", ev_set="sparse", arm=arm)
        else:
            event_mode, rank_mode = modes
            train += compound_rows_indep(
                w["atp_sparse"], w["atp_sparse2"], ev_grp="train", st_grp="train",
                ns="sp263_rel", split="train", ev_set="sparse", arm=arm,
                event_mode=event_mode, rank_mode=rank_mode,
                untouched_mode="true", tk=f"sparse_{arm}")
            train += m.event_only_rows(
                w["pair_sparse"], ev_grp="train", ns="sp263_pair", split="train",
                ev_set="sparse", arm=arm, lm=event_mode, tk=f"sparse_{arm}_pair")
        arms[arm] = {"train": train, "evals": make_evals(arm)}
    return arms, w


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = collections.Counter(r.get("train_kind", "") for r in rows)
    by_query = collections.Counter(r.get("query_family", "") for r in rows)
    by_label = collections.Counter(str(r.get("label")) for r in rows)
    by_modes = collections.Counter(r.get("label_mode", "") for r in rows)
    conflicts = collections.Counter(str(r.get("conflict")) for r in rows if r.get("conflict") is not None)
    return {
        "n": len(rows),
        "by_kind": dict(sorted(by_kind.items())),
        "by_query": dict(sorted(by_query.items())),
        "by_label": dict(sorted(by_label.items())),
        "by_modes": dict(sorted(by_modes.items())),
        "by_conflict": dict(sorted(conflicts.items())),
    }


def construction_summary(arms: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    s = {"arms": {}}
    for arm, obj in arms.items():
        s["arms"][arm] = {
            "train": describe_rows(obj["train"]),
            "eval_counts": {k: len(v) for k, v in obj["evals"].items()},
        }
    (out / "construction_summary.json").write_text(json.dumps(s, indent=2) + "\n", "utf-8")
    return s


def pool_text(enc, tokenizer, text: str, device: str, max_len: int):
    tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        h = enc(tok["input_ids"].to(device), tok["attention_mask"].to(device)).last_hidden_state
        mask = tok["attention_mask"].to(device).unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp_min(1)


def semantic_calibration(model_path: Path, tokenizer, worlds, device: str, max_len: int, n: int = 30):
    enc = DebertaV2Model.from_pretrained(str(model_path)).to(device).eval()
    pairs: dict[str, list[float]] = collections.defaultdict(list)
    sample = worlds[:min(n, len(worlds))]
    for w in sample:
        am = m.amap2(w, "ground263")
        for grp in ["held", "dirpara", "nonpara"]:
            for tv in range(len(m.EVENT_CTX["train"])):
                for gv in range(len(m.EVENT_CTX[grp])):
                    a = pool_text(enc, tokenizer, m.ev_sent(w, am, "train", tv), device, max_len)
                    b = pool_text(enc, tokenizer, m.ev_sent(w, am, grp, gv), device, max_len)
                    pairs[f"event_train_vs_{grp}"].append(float(F.cosine_similarity(a, b).item()))
        if w.higher_name:
            for grp in ["held", "dirpara", "nonpara"]:
                for tv in range(len(m.STATE_CTX["train"])):
                    for gv in range(len(m.STATE_CTX[grp])):
                        a = pool_text(enc, tokenizer, m.st_sent(w, am, "train", tv), device, max_len)
                        b = pool_text(enc, tokenizer, m.st_sent(w, am, grp, gv), device, max_len)
                        pairs[f"state_train_vs_{grp}"].append(float(F.cosine_similarity(a, b).item()))
    del enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    def stats(vals):
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n": len(vals)}
    return {k: stats(v) for k, v in sorted(pairs.items()) if v}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
    agg: dict[str, Any] = {}
    for arm, rs in sorted(by_arm.items()):
        item = {
            "n_seeds": len(rs),
            "train_acc": stat([r["best_train_acc"] for r in rs]),
            "evals": {},
        }
        eval_names = sorted(rs[0]["evals"].keys())
        for en in eval_names:
            ed: dict[str, Any] = {}
            ed["standard_acc"] = stat([r["evals"][en]["standard"].get("acc") for r in rs])
            ed["standard_by_query"] = {}
            ed["contrastive_acc"] = stat([r["evals"][en]["contrastive"].get("contrastive_acc") for r in rs])
            ed["contrastive_by_query"] = {}
            ed["contrastive_by_conflict"] = {}
            for q in ["event_role", "focal_state", "untouched_state"]:
                ed["standard_by_query"][q] = stat([
                    r["evals"][en]["standard"].get("key_subsets", {}).get(q) for r in rs])
                ed["contrastive_by_query"][q] = stat([
                    r["evals"][en]["contrastive"].get("by_query", {}).get(q) for r in rs])
            conflict_keys = set()
            for r in rs:
                conflict_keys.update(r["evals"][en]["contrastive"].get("by_conflict", {}).keys())
            for ck in sorted(conflict_keys):
                ed["contrastive_by_conflict"][ck] = stat([
                    r["evals"][en]["contrastive"].get("by_conflict", {}).get(ck) for r in rs])
            item["evals"][en] = ed
        agg[arm] = item
    agg["cross_arm_effects"] = cross_arm_effects(agg)
    return agg


def stat(vals) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs)}


def metric(agg: dict[str, Any], arm: str, ev: str, q: str, conflict: str | None = None):
    try:
        if conflict is None:
            return agg[arm]["evals"][ev]["contrastive_by_query"][q]["mean"]
        return agg[arm]["evals"][ev]["contrastive_by_conflict"][f"{q}_{conflict}"]["mean"]
    except KeyError:
        return None


def cross_arm_effects(agg: dict[str, Any]) -> dict[str, Any]:
    arms = [a for a in agg.keys() if a != "cross_arm_effects"]
    if not arms:
        return {}
    eval_names = sorted(agg[arms[0]]["evals"].keys())
    out: dict[str, Any] = {}
    for ev in eval_names:
        item: dict[str, Any] = {}
        for q in ["event_role", "focal_state", "untouched_state"]:
            t = metric(agg, "Etrue_Rtrue", ev, q)
            ef = metric(agg, "Eflip_Rtrue", ev, q)
            rf = metric(agg, "Etrue_Rflip", ev, q)
            ff = metric(agg, "Eflip_Rflip", ev, q)
            if t is not None and ef is not None:
                item[f"event_flip_effect_on_{q}"] = t - ef
            if t is not None and rf is not None:
                item[f"rank_flip_effect_on_{q}"] = t - rf
            if t is not None and ff is not None:
                item[f"joint_flip_effect_on_{q}"] = t - ff
        out[ev] = item
    return out


def write_md(summary: dict[str, Any], out: Path):
    agg = summary.get("aggregate", {})
    lines = ["# research independent event/ranking orientation cross\n\n"]
    lines.append("Event and focal-ranking sparse labels vary independently; secondary ranking labels remain true. Evaluation labels are always true facts.\n\n")
    if summary.get("semantic_calibration"):
        lines.append("## Context wording calibration\n\n")
        for k, v in summary["semantic_calibration"].items():
            lines.append(f"- {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, min={v['min']:.4f}, n={v['n']}\n")
        lines.append("\n")
    key_evals = [
        "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
        "atp_heldHeld_trainHyp", "atp_heldHeld_heldHyp",
        "atp_dirDir_trainHyp", "atp_trainEvent_dirState_trainHyp",
        "atp_dirEvent_trainState_trainHyp", "atp_trainEvent_nonState_trainHyp",
        "atp_nonEvent_trainState_trainHyp",
    ]
    for arm in ["exposure", "Etrue_Rtrue", "Eflip_Rtrue", "Etrue_Rflip", "Eflip_Rflip"]:
        if arm not in agg:
            continue
        tr = agg[arm]["train_acc"]
        lines.append(f"## Arm {arm} (train acc mean={tr['mean']:.3f}, std={tr['std']:.3f}, seeds={tr['n']})\n\n")
        lines.append("| eval_set | con_event | con_focal | con_untouched | focal_conflict | untouched_conflict |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for ev in key_evals:
            if ev not in agg[arm]["evals"]:
                continue
            e = agg[arm]["evals"][ev]
            cq = e["contrastive_by_query"]
            cf = e["contrastive_by_conflict"]
            def get(d, k):
                return d.get(k, {}).get("mean", float("nan"))
            lines.append(
                f"| {ev} | {get(cq,'event_role'):.3f} | {get(cq,'focal_state'):.3f} | {get(cq,'untouched_state'):.3f} | "
                f"{get(cf,'focal_state_conflict'):.3f} | {get(cf,'untouched_state_conflict'):.3f} |\n")
        lines.append("\n")
    if agg.get("cross_arm_effects"):
        lines.append("## Selective movement relative to Etrue_Rtrue\n\n")
        lines.append("Positive values mean accuracy against true facts dropped when the named sparse orientation was flipped.\n\n")
        lines.append("| eval_set | event flip on event | event flip on focal | event flip on untouched | rank flip on event | rank flip on focal | rank flip on untouched |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for ev in key_evals:
            item = agg["cross_arm_effects"].get(ev, {})
            lines.append(
                f"| {ev} | {item.get('event_flip_effect_on_event_role', float('nan')):.3f} | "
                f"{item.get('event_flip_effect_on_focal_state', float('nan')):.3f} | "
                f"{item.get('event_flip_effect_on_untouched_state', float('nan')):.3f} | "
                f"{item.get('rank_flip_effect_on_event_role', float('nan')):.3f} | "
                f"{item.get('rank_flip_effect_on_focal_state', float('nan')):.3f} | "
                f"{item.get('rank_flip_effect_on_untouched_state', float('nan')):.3f} |\n")
    (out / "independent_orientation_cross_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--train_atp", type=int, default=80)
    ap.add_argument("--sparse_atp", type=int, default=16)
    ap.add_argument("--eval_atp", type=int, default=80)
    ap.add_argument("--train_pair", type=int, default=80)
    ap.add_argument("--sparse_pair", type=int, default=16)
    ap.add_argument("--eval_pair", type=int, default=100)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--skip_calibration", action="store_true")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": "START", "args": vars(args),
                      "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)

    arms, worlds = build_datasets(args)
    csum = construction_summary(arms, out)
    for arm, obj in arms.items():
        print(json.dumps({"event": "arm_built", "arm": arm,
                          "train_rows": len(obj["train"]),
                          "labels": csum["arms"][arm]["train"]["by_label"],
                          "eval_sets": len(obj["evals"])}), flush=True)
    if args.dry_build:
        (out / "dry_build_done.json").write_text(json.dumps({"status": "dry_build_done", "construction": csum}, indent=2) + "\n", "utf-8")
        print(json.dumps({"status": "DRY_BUILD_DONE", "out": str(out)}), flush=True)
        return

    dev = args.device
    tok = AutoTokenizer.from_pretrained(args.model_path)
    cal = {}
    if not args.skip_calibration:
        cal = semantic_calibration(Path(args.model_path), tok, worlds["atp_eval"], dev, args.max_len, n=min(30, args.eval_atp))
        (out / "context_wording_calibration.json").write_text(json.dumps(cal, indent=2) + "\n", "utf-8")
        print(json.dumps({"event": "context_calibration", **cal}), flush=True)

    results = []
    for arm, obj in arms.items():
        for si in range(args.n_seeds):
            seed = 26300 + si
            r = m.train_one(model_path=Path(args.model_path), tok=tok, arm=arm,
                            seed=seed, train_rows=obj["train"], evals=obj["evals"],
                            args=args, dev=dev)
            results.append(r)
            for en in ["atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
                       "atp_heldHeld_trainHyp", "atp_dirDir_trainHyp",
                       "atp_trainEvent_nonState_trainHyp", "atp_nonEvent_trainState_trainHyp"]:
                ev = r["evals"].get(en, {})
                con = ev.get("contrastive", {})
                print(json.dumps({"event": "eval", "arm": arm, "seed": seed,
                                  "train_acc": r["best_train_acc"], "eval_set": en,
                                  "con_event": con.get("by_query", {}).get("event_role"),
                                  "con_focal": con.get("by_query", {}).get("focal_state"),
                                  "con_untouched": con.get("by_query", {}).get("untouched_state"),
                                  "con_conflict": con.get("by_conflict", {})}), flush=True)

    agg = aggregate(results)
    summary = {"status": "INDEPENDENT_ORIENTATION_CROSS",
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": csum,
               "semantic_calibration": cal, "aggregate": agg, "per_seed": results}
    (out / "independent_orientation_cross_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "summary_json": str(out / "independent_orientation_cross_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
