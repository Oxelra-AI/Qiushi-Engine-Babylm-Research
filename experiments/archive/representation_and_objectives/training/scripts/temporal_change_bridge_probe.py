#!/usr/bin/env python3
"""research: fixed-meaning temporal-change bridge for entity-state updating.

Scientific purpose
------------------
Steps263-264 showed that sparse evidence can orient a familiar ranking coordinate,
but sign-flip updates are not the right proxy for untouched-state retention: a
flipped ranking relation globally contradicts the predicate.  This script builds
the next source-attested test with fixed relation meaning and fixed evaluation
truths.

The substrate is the research ATP ranking family.  Each world has an independently
recorded ranking relation at match time and another ranking snapshot roughly 26
weeks later.  Some worlds change which participant is higher ranked; other worlds
remain stable.  The bridge pairs a focal world with a separate stable secondary
world.  The context contains before/later ranking records for both pairs.  Sparse
training labels query only the focal pair; secondary before/after labels are held
out from sparse changed examples.  Evaluation asks whether the learner updates the
changed focal pair from the later record while preserving the stable secondary
pair across held entities and held wording.

Important boundary: this tests source-attested temporal state update/readout, not
causal attribution from the tennis match outcome.  Event/match text is deliberately
omitted from the default temporal context; event-outcome shortcut audits are still
saved from source fields.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
ATP_DIR = WS / "data/upset_balanced_state_substrate"
PATH = WS / "training/scripts/contrastive_wording_cross_probe.py"
spec = importlib.util.spec_from_file_location("temporal_mod", PATH)
m = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = m
spec.loader.exec_module(m)

DEFAULT_MODEL = m.DEFAULT_MODEL
DEFAULT_OUT = WS / "data/temporal_change_bridge"

ALIAS_POOL = m.ALIAS_POOL

INIT_CTX = {
    "anchor": [
        "In the earlier ATP ranking, {H} was above {Lo}.",
        "Before the later ranking update, {Lo} was below {H}.",
    ],
    "train": [
        "At the first ranking date, {H} stood higher than {Lo}.",
        "The initial ATP list placed {Lo} under {H}.",
    ],
    "held": [
        "In the prior ranking snapshot, {H} had the better position than {Lo}.",
        "At the earlier snapshot, {Lo} trailed {H} in the rankings.",
    ],
    "dirpara": [
        "Initially, {H} led {Lo} in the ATP ranking order.",
        "Before the update, {Lo} followed {H} on the ranking list.",
    ],
    "nonpara": [
        "The earlier ranking note mentioned {H} and {Lo}.",
        "The earlier ranking note listed {Lo} near {H}.",
    ],
}

LATER_CTX = {
    "anchor": [
        "In the later ATP ranking, {H} was above {Lo}.",
        "After the later ranking update, {Lo} was below {H}.",
    ],
    "train": [
        "At the later ranking date, {H} stood higher than {Lo}.",
        "The later ATP list placed {Lo} under {H}.",
    ],
    "held": [
        "In the subsequent ranking snapshot, {H} had the better position than {Lo}.",
        "At the later snapshot, {Lo} trailed {H} in the rankings.",
    ],
    "dirpara": [
        "Later, {H} led {Lo} in the ATP ranking order.",
        "After the update, {Lo} followed {H} on the ranking list.",
    ],
    "nonpara": [
        "The later ranking note mentioned {H} and {Lo}.",
        "The later ranking note listed {Lo} near {H}.",
    ],
}

HYP_BEFORE = {
    "train": "Before the later ranking update, {X} held the higher ranking than {Y}.",
    "held": "Initially, {X} outranked {Y}.",
}
HYP_AFTER = {
    "train": "After the later ranking update, {X} held the higher ranking than {Y}.",
    "held": "Later, {X} outranked {Y}.",
}

ARM_SPECS = ["exposure", "stable_only", "changed_only", "balanced_temporal", "oracle_secondary"]


@dataclass(frozen=True)
class TWorld:
    source: str
    family_id: str
    world_id: str
    split: str
    participant_a: str
    participant_b: str
    winner_name: str
    loser_name: str
    higher_initial: str
    lower_initial: str
    higher_later: str
    lower_later: str
    winner_higher_initial: bool
    winner_higher_later: bool
    changed: bool
    event_text: str
    state_at_time_text: str
    state_later_text: str


def _si(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def aliases(key: str, n: int) -> list[str]:
    return random.Random(_si("aliases265|" + key)).sample(ALIAS_POOL, n)


def other_aliases(used: set[str], key: str, n: int) -> list[str]:
    cands = [x for x in ALIAS_POOL if x not in used]
    return random.Random(_si("other265|" + key)).sample(cands, n)


def load_temporal_worlds() -> tuple[list[TWorld], list[TWorld]]:
    out: dict[str, list[TWorld]] = {"train": [], "held": []}
    for sp in ["train", "held"]:
        p = ATP_DIR / f"families_{sp}.jsonl"
        for line in p.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            f = json.loads(line)
            a, b = f["participant_a"], f["participant_b"]
            for wk in ["context1", "context2"]:
                c = f[wk]
                w, l = c["winner"], c["loser"]
                hi0, lo0 = (w, l) if c["winner_higher_at_time"] else (l, w)
                hi1, lo1 = (w, l) if c["winner_higher_later"] else (l, w)
                out[sp].append(TWorld(
                    source="atp_temporal", family_id=f["family_id"], world_id=f"{f['family_id']}::{wk}", split=sp,
                    participant_a=a, participant_b=b, winner_name=w, loser_name=l,
                    higher_initial=hi0, lower_initial=lo0, higher_later=hi1, lower_later=lo1,
                    winner_higher_initial=bool(c["winner_higher_at_time"]),
                    winner_higher_later=bool(c["winner_higher_later"]),
                    changed=bool(c["state_changed_between_snapshots"]),
                    event_text=c["event_text"], state_at_time_text=c["state_at_time_text"],
                    state_later_text=c["state_later_text"],
                ))
    return out["train"], out["held"]


def amap4(f: TWorld, s: TWorld, ns: str) -> dict[str, str]:
    """Distinct aliases for four distinct real participants.

    A collision would silently merge a focal and a secondary participant into one
    alias and create an unintended identity bridge between the two records, so the
    caller must guarantee four distinct real names.
    """
    names = [f.participant_a, f.participant_b, s.participant_a, s.participant_b]
    if len(set(names)) != 4:
        raise ValueError(f"participant overlap between focal {f.world_id} and secondary {s.world_id}: {names}")
    al = aliases(f"four|{ns}|{f.world_id}|{s.world_id}", 4)
    return {f.participant_a: al[0], f.participant_b: al[1], s.participant_a: al[2], s.participant_b: al[3]}


def pick_secondary(focal: TWorld, seconds: list[TWorld], i: int) -> TWorld | None:
    """Deterministically pick a secondary world with no shared family or participant."""
    if not seconds:
        return None
    n = len(seconds)
    fnames = {focal.participant_a, focal.participant_b}
    for off in range(n):
        s = seconds[(i * 7 + 3 + off) % n]
        if s.world_id == focal.world_id or s.family_id == focal.family_id:
            continue
        if fnames & {s.participant_a, s.participant_b}:
            continue
        return s
    return None


def an(am: dict[str, str], name: str) -> str:
    return am[name]


def initial_label(w: TWorld, hd: str) -> int:
    hi = "A" if w.higher_initial == w.participant_a else "B"
    return int((hd == "AB" and hi == "A") or (hd == "BA" and hi == "B"))


def later_label(w: TWorld, hd: str) -> int:
    hi = "A" if w.higher_later == w.participant_a else "B"
    return int((hd == "AB" and hi == "A") or (hd == "BA" and hi == "B"))


def sent_initial(w: TWorld, am: dict[str, str], grp: str, var: int) -> str:
    if grp == "source":
        txt = w.state_at_time_text
        for real, alias in am.items():
            txt = txt.replace(real, alias)
        return txt
    return INIT_CTX[grp][var % len(INIT_CTX[grp])].format(H=an(am, w.higher_initial), Lo=an(am, w.lower_initial))


def sent_later(w: TWorld, am: dict[str, str], grp: str, var: int) -> str:
    if grp == "source":
        txt = w.state_later_text
        for real, alias in am.items():
            txt = txt.replace(real, alias)
        return txt
    return LATER_CTX[grp][var % len(LATER_CTX[grp])].format(H=an(am, w.higher_later), Lo=an(am, w.lower_later))


def hyp_state(w: TWorld, am: dict[str, str], hd: str, when: Literal["before", "after"], wording: str) -> str:
    x = an(am, w.participant_a if hd == "AB" else w.participant_b)
    y = an(am, w.participant_b if hd == "AB" else w.participant_a)
    return (HYP_BEFORE if when == "before" else HYP_AFTER)[wording].format(X=x, Y=y)


def mk(*, row_id: str, context: str, hypothesis: str, label: int, split: str, eval_set: str,
       arm: str, query_family: str, focal_world: str, secondary_world: str | None,
       changed_focal: bool | None, stable_secondary: bool | None, template_group: str,
       train_kind: str, hyp_dir: str = "", pair_key: str = "", hyp_wording: str = "train",
       ctx_wording: str = "train", label_mode: str = "true") -> dict[str, Any]:
    return dict(id=row_id, text=context + " [SEP] " + hypothesis, context=context,
                hypothesis=hypothesis, label=int(label), split=split, eval_set=eval_set,
                arm=arm, source="atp_temporal", sport="tennis_atp",
                query_family=query_family, focal_world=focal_world, secondary_world=secondary_world,
                changed_focal=changed_focal, stable_secondary=stable_secondary,
                template_group=template_group, train_kind=train_kind, hyp_dir=hyp_dir,
                pair_key=pair_key, hyp_wording=hyp_wording, ctx_wording=ctx_wording,
                label_mode=label_mode)


def temporal_context(f: TWorld, s: TWorld, am: dict[str, str], ctx_grp: str, vi: int, vl: int,
                     order: str = "standard", include_initial: bool = True,
                     include_later: bool = True, include_secondary: bool = True) -> tuple[str, str]:
    parts = []
    focal_init = f"Focal initial ranking: {sent_initial(f, am, ctx_grp, vi)}"
    focal_later = f"Focal later ranking: {sent_later(f, am, ctx_grp, vl)}"
    sec_init = f"Separate initial ranking: {sent_initial(s, am, ctx_grp, vi + 1)}"
    sec_later = f"Separate later ranking: {sent_later(s, am, ctx_grp, vl + 1)}"
    if order == "secondary_first":
        cand = [(include_secondary and include_initial, sec_init), (include_secondary and include_later, sec_later),
                (include_initial, focal_init), (include_later, focal_later)]
    elif order == "later_first":
        cand = [(include_later, focal_later), (include_initial, focal_init),
                (include_secondary and include_later, sec_later), (include_secondary and include_initial, sec_init)]
    else:
        cand = [(include_initial, focal_init), (include_later, focal_later),
                (include_secondary and include_initial, sec_init), (include_secondary and include_later, sec_later)]
    for ok, txt in cand:
        if ok:
            parts.append(txt)
    tpl = f"ctx={ctx_grp}|vi={vi}|vl={vl}|order={order}|init={include_initial}|later={include_later}|sec={include_secondary}"
    return " ".join(parts), tpl


def temporal_relation_rows(focals: list[TWorld], seconds: list[TWorld], *, ns: str, ctx_grp: str,
                           split: str, eval_set: str, arm: str, queries: list[str],
                           train_kind: str, hyp_w: str = "train", order: str = "standard",
                           include_secondary_context: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seconds:
        return rows
    for i, f in enumerate(focals):
        s = pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = amap4(f, s, ns)
        for vi in range(len(INIT_CTX[ctx_grp])):
            for vl in range(len(LATER_CTX[ctx_grp])):
                ctx, tpl = temporal_context(f, s, am, ctx_grp, vi, vl, order=order,
                                            include_secondary=include_secondary_context)
                base_pk = f"{ns}|{f.world_id}|{s.world_id}|{tpl}|h={hyp_w}"
                for hd in ["AB", "BA"]:
                    qspec = []
                    if "focal_before" in queries:
                        qspec.append(("focal_before", f, "before", initial_label(f, hd)))
                    if "focal_after" in queries:
                        qspec.append(("focal_after", f, "after", later_label(f, hd)))
                    if "secondary_before" in queries:
                        qspec.append(("secondary_before", s, "before", initial_label(s, hd)))
                    if "secondary_after" in queries:
                        qspec.append(("secondary_after", s, "after", later_label(s, hd)))
                    for qf, qw, when, lab in qspec:
                        rows.append(mk(
                            row_id=f"{base_pk}|{qf}|{hd}", context=ctx,
                            hypothesis=hyp_state(qw, am, hd, when, hyp_w), label=lab,
                            split=split, eval_set=eval_set, arm=arm, query_family=qf,
                            focal_world=f.world_id, secondary_world=s.world_id,
                            changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                            template_group=tpl, train_kind=train_kind, hyp_dir=hd,
                            pair_key=f"{base_pk}|{qf}", hyp_wording=hyp_w, ctx_wording=ctx_grp,
                            label_mode="true_fixed_meaning"))
    return rows


def neutral_rows(focals: list[TWorld], seconds: list[TWorld], *, ns: str, ctx_grp: str, split: str,
                 eval_set: str, arm: str, neutral_per_template: int, train_kind: str = "mention_exposure") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not seconds or neutral_per_template <= 0:
        return rows
    assert neutral_per_template % 2 == 0
    for i, f in enumerate(focals):
        s = pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = amap4(f, s, ns)
        used = set(am.values())
        outsiders = other_aliases(used, f"neutral|{ns}|{f.world_id}|{s.world_id}", max(2, neutral_per_template // 2))
        true_aliases = [am[f.participant_a], am[f.participant_b], am[s.participant_a], am[s.participant_b]]
        for vi in range(len(INIT_CTX[ctx_grp])):
            for vl in range(len(LATER_CTX[ctx_grp])):
                ctx, tpl = temporal_context(f, s, am, ctx_grp, vi, vl)
                n_true = neutral_per_template // 2
                n_false = neutral_per_template - n_true
                start = (vi + vl) % len(true_aliases)
                trues = (true_aliases[start:] + true_aliases[:start])[:n_true]
                items = [(a, 1) for a in trues] + [(a, 0) for a in outsiders[:n_false]]
                for j, (alias, lab) in enumerate(items):
                    rows.append(mk(row_id=f"{ns}|{f.world_id}|{s.world_id}|{tpl}|neutral|{j}",
                                   context=ctx, hypothesis=f"The passage mentions {alias}.", label=lab,
                                   split=split, eval_set=eval_set, arm=arm, query_family="neutral_mention",
                                   focal_world=f.world_id, secondary_world=s.world_id,
                                   changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                                   template_group=tpl, train_kind=train_kind, label_mode="neutral_mention"))
    return rows


def key_changed_dir(w: TWorld) -> str:
    hi0 = "A" if w.higher_initial == w.participant_a else "B"
    hi1 = "A" if w.higher_later == w.participant_a else "B"
    return f"{hi0}->{hi1}"


def key_stable_dir(w: TWorld) -> str:
    hi0 = "A" if w.higher_initial == w.participant_a else "B"
    hi1 = "A" if w.higher_later == w.participant_a else "B"
    return f"{hi0}->{hi1}"


def balanced_sample(pool: list[TWorld], n: int, seed: int, *, changed: bool | None = None) -> list[TWorld]:
    if changed is not None:
        pool = [w for w in pool if w.changed == changed]
    groups: dict[str, list[TWorld]] = collections.defaultdict(list)
    for w in pool:
        groups[key_changed_dir(w)].append(w)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    if not keys:
        return []
    # Draw round-robin over available direction groups.  For changed worlds this
    # balances A->B and B->A; for stable worlds this balances A->A and B->B.
    out: list[TWorld] = []
    while len(out) < n:
        made = False
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
                made = True
        if not made:
            break
    rng.shuffle(out)
    return out


def disjoint(exclude: set[str], pool: list[TWorld]) -> list[TWorld]:
    return [w for w in pool if w.world_id not in exclude and w.family_id not in exclude]


def select_splits(args):
    tr, he = load_temporal_worlds()
    selected: dict[str, list[TWorld]] = {}
    selected["base_stable"] = balanced_sample(tr, args.base_stable, 26501, changed=False)
    used = {w.world_id for w in selected["base_stable"]}
    tr2 = [w for w in tr if w.world_id not in used]
    selected["base_stable2"] = balanced_sample(tr2, args.base_stable + 17, 26502, changed=False)
    used |= {w.world_id for w in selected["base_stable2"]}
    tr3 = [w for w in tr if w.world_id not in used]
    selected["sparse_changed"] = balanced_sample(tr3, args.sparse_changed, 26503, changed=True)
    used |= {w.world_id for w in selected["sparse_changed"]}
    tr4 = [w for w in tr if w.world_id not in used]
    selected["sparse_stable"] = balanced_sample(tr4, args.sparse_stable, 26504, changed=False)
    used |= {w.world_id for w in selected["sparse_stable"]}
    tr5 = [w for w in tr if w.world_id not in used]
    selected["sparse_secondary_stable"] = balanced_sample(tr5, max(args.sparse_changed, args.sparse_stable) + 17, 26505, changed=False)
    used |= {w.world_id for w in selected["sparse_secondary_stable"]}
    tr6 = [w for w in tr if w.world_id not in used]
    selected["eval_train_changed"] = balanced_sample(tr6, args.eval_train_changed, 26506, changed=True)
    used |= {w.world_id for w in selected["eval_train_changed"]}
    tr7 = [w for w in tr if w.world_id not in used]
    selected["eval_train_stable"] = balanced_sample(tr7, args.eval_train_stable, 26507, changed=False)

    selected["eval_held_changed"] = balanced_sample(he, args.eval_held_changed, 26508, changed=True)
    hused = {x.world_id for x in selected["eval_held_changed"]}
    he2 = [x for x in he if x.world_id not in hused]
    selected["eval_held_stable"] = balanced_sample(he2, args.eval_held_stable, 26509, changed=False)
    hused |= {x.world_id for x in selected["eval_held_stable"]}
    he3 = [x for x in he if x.world_id not in hused]
    selected["eval_held_stable2"] = balanced_sample(he3, args.eval_held_stable, 26510, changed=False)
    return selected, {"train_total": len(tr), "held_total": len(he)}


def build_base_updates_evals(args, arms: list[str]):
    w, inventory = select_splits(args)
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    base = temporal_relation_rows(w["base_stable"], w["base_stable2"], ns="base265_stable", ctx_grp="anchor",
                                  split="train", eval_set="base", arm="base", queries=base_queries,
                                  train_kind="base_stable_anchor", hyp_w="train")
    if args.base_train_wording:
        base += temporal_relation_rows(w["base_stable"], w["base_stable2"], ns="base265_stable_train", ctx_grp="train",
                                       split="train", eval_set="base", arm="base", queries=base_queries,
                                       train_kind="base_stable_train", hyp_w="train")

    # A common set of sparse contexts: changed focal + stable secondary, and stable focal + stable secondary.
    sparse_changed_seconds = w["sparse_secondary_stable"]
    sparse_stable_seconds = w["sparse_secondary_stable"]
    updates: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        rows: list[dict[str, Any]] = []
        if arm == "exposure":
            rows += neutral_rows(w["sparse_changed"], sparse_changed_seconds, ns="sp265_changed_exp", ctx_grp="train",
                                 split="train", eval_set="sparse", arm=arm, neutral_per_template=4)
            rows += neutral_rows(w["sparse_stable"], sparse_stable_seconds, ns="sp265_stable_exp", ctx_grp="train",
                                 split="train", eval_set="sparse", arm=arm, neutral_per_template=4)
        elif arm == "stable_only":
            rows += temporal_relation_rows(w["sparse_stable"], sparse_stable_seconds, ns="sp265_stable_only", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after"], train_kind="sparse_stable_focal", hyp_w="train")
            rows += neutral_rows(w["sparse_changed"], sparse_changed_seconds, ns="sp265_changed_exp_stableonly", ctx_grp="train",
                                 split="train", eval_set="sparse", arm=arm, neutral_per_template=4)
        elif arm == "changed_only":
            rows += temporal_relation_rows(w["sparse_changed"], sparse_changed_seconds, ns="sp265_changed_only", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after"], train_kind="sparse_changed_focal", hyp_w="train")
            rows += neutral_rows(w["sparse_stable"], sparse_stable_seconds, ns="sp265_stable_exp_changedonly", ctx_grp="train",
                                 split="train", eval_set="sparse", arm=arm, neutral_per_template=4)
        elif arm == "balanced_temporal":
            rows += temporal_relation_rows(w["sparse_changed"], sparse_changed_seconds, ns="sp265_bal_changed", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after"], train_kind="sparse_changed_focal", hyp_w="train")
            rows += temporal_relation_rows(w["sparse_stable"], sparse_stable_seconds, ns="sp265_bal_stable", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after"], train_kind="sparse_stable_focal", hyp_w="train")
        elif arm == "oracle_secondary":
            rows += temporal_relation_rows(w["sparse_changed"], sparse_changed_seconds, ns="sp265_oracle_changed", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after", "secondary_before", "secondary_after"],
                                           train_kind="sparse_changed_oracle", hyp_w="train")
            rows += temporal_relation_rows(w["sparse_stable"], sparse_stable_seconds, ns="sp265_oracle_stable", ctx_grp="train",
                                           split="train", eval_set="sparse", arm=arm,
                                           queries=["focal_before", "focal_after", "secondary_before", "secondary_after"],
                                           train_kind="sparse_stable_oracle", hyp_w="train")
        else:
            raise ValueError(f"unknown arm {arm}")
        updates[arm] = rows

    evals: dict[str, list[dict[str, Any]]] = {}
    eval_pairings = [
        ("heldChanged_heldStable", w["eval_held_changed"], w["eval_held_stable"]),
        ("heldChanged_trainStable", w["eval_held_changed"], w["eval_train_stable"]),
        ("trainChanged_heldStable", w["eval_train_changed"], w["eval_held_stable"]),
        # Stable focal control uses a disjoint stable secondary pool so the focal
        # and secondary records never come from the same worlds.
        ("heldStable_heldStable", w["eval_held_stable"], w["eval_held_stable2"]),
    ]
    ctxs = [("trainCtx", "train", "standard"), ("heldCtx", "held", "standard"),
            ("dirCtx", "dirpara", "standard"), ("secondaryFirst", "train", "secondary_first"),
            ("laterFirst", "train", "later_first"), ("nonDirCtx", "nonpara", "standard")]
    all_eval_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    for pname, focals, seconds in eval_pairings:
        for cname, cg, order in ctxs:
            for hw in ["train", "held"]:
                if cg == "nonpara" and hw == "held":
                    continue
                en = f"{pname}_{cname}_{hw}Hyp"
                evals[en] = temporal_relation_rows(focals, seconds, ns=f"eval265_{pname}_{cname}_{hw}",
                                                    ctx_grp=cg, split="held", eval_set=en, arm="eval",
                                                    queries=all_eval_queries, train_kind="eval_temporal",
                                                    hyp_w=hw, order=order)
    return base, updates, evals, w, inventory


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = lambda key: dict(sorted(collections.Counter(str(r.get(key, "")) for r in rows).items()))
    return {"n": len(rows), "by_query": c("query_family"), "by_label": c("label"),
            "by_kind": c("train_kind"), "by_changed_focal": c("changed_focal"),
            "by_stable_secondary": c("stable_secondary")}


def world_summary(ws: list[TWorld]) -> dict[str, Any]:
    return {"n": len(ws), "changed": dict(collections.Counter(str(w.changed) for w in ws)),
            "dir": dict(collections.Counter(key_changed_dir(w) for w in ws)),
            "winner_initial_later": dict(collections.Counter(f"{int(w.winner_higher_initial)}->{int(w.winner_higher_later)}" for w in ws))}


def shortcut_audits(selection: dict[str, list[TWorld]], evals: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    audits: dict[str, Any] = {"selection": {k: world_summary(v) for k, v in selection.items()}}
    # Heuristics over eval rows.  initial-as-after: answer after queries using before relation; universal flip: answer after as opposite before.
    by_eval = {}
    for name, rows in evals.items():
        pairs = collections.defaultdict(dict)
        for r in rows:
            if r.get("hyp_dir"):
                pairs[(r["pair_key"], r["query_family"])][r["hyp_dir"]] = r
        # For contrastive pair, true AB label is stored on AB row.  For focal/secondary after we can infer initial-label heuristic by matching before row.
        counts = {"initial_as_after": [0, 0], "flip_initial_as_after": [0, 0], "winner_as_after": [0, 0], "always_A_after": [0, 0]}
        before_label_by_pair = {}
        for (pk, qf), d in pairs.items():
            if "AB" in d and qf.endswith("before"):
                # normalize key by replacing query suffix in pair_key
                before_label_by_pair[pk.replace(qf, qf.replace("before", "after"))] = int(d["AB"]["label"])
        for (pk, qf), d in pairs.items():
            if "AB" not in d or not qf.endswith("after"):
                continue
            true_lab = int(d["AB"]["label"])
            init_lab = before_label_by_pair.get(pk)
            if init_lab is not None:
                counts["initial_as_after"][0] += int(init_lab == true_lab); counts["initial_as_after"][1] += 1
                counts["flip_initial_as_after"][0] += int((1 - init_lab) == true_lab); counts["flip_initial_as_after"][1] += 1
            counts["always_A_after"][0] += int(true_lab == 1); counts["always_A_after"][1] += 1
        by_eval[name] = {k: (v[0] / v[1] if v[1] else None) for k, v in counts.items()}
    audits["eval_heuristics"] = by_eval
    return audits


def construction_summary(base, updates, evals, selection, inventory, out: Path):
    s = {"status": "TEMPORAL_CHANGE_CONSTRUCTION",
         "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "inventory": inventory,
         "selection": {k: world_summary(v) for k, v in selection.items()},
         "base": describe_rows(base),
         "updates": {a: describe_rows(r) for a, r in updates.items()},
         "eval_counts": {k: len(v) for k, v in sorted(evals.items())},
         "shortcut_audits": shortcut_audits(selection, evals),
         "boundary": "Fixed-meaning source-attested temporal relation test; no sign flips and no BabyLM-scale training."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "construction_summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return s


def encode_rows(tok, rows, ml, dev):
    texts = [r["text"] for r in rows]
    enc = tok(texts, padding=True, truncation=True, max_length=ml, return_tensors="pt")
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev)


def get_all_logits(model, tok, rows, ml, dev, bs):
    model.eval(); outs = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            x, mask, _ = encode_rows(tok, rows[i:i+bs], ml, dev)
            outs.append(model(x, mask).detach().cpu())
    return torch.cat(outs, 0) if outs else torch.zeros(0, 2)


def standard_eval(logits, rows):
    if not rows:
        return {}
    pred = logits.argmax(1).numpy(); lab = np.array([r["label"] for r in rows])
    out = {"n": len(rows), "acc": float(np.mean(pred == lab))}
    for key in ["query_family", "changed_focal", "stable_secondary"]:
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows): groups[str(r.get(key, ""))].append(i)
        out[f"by_{key}"] = {g: float(np.mean(pred[idx] == lab[idx])) for g, idx in groups.items() if idx}
    return out


def contrastive_eval_with_margins(logits, rows):
    if not rows:
        return {}
    pairs = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        if r.get("pair_key") and r.get("hyp_dir"):
            pairs[r["pair_key"]][r["hyp_dir"]] = (i, r)
    correct = [0, 0]
    by_q = collections.defaultdict(lambda: [0, 0])
    by_q_margin = collections.defaultdict(list)
    by_changed = collections.defaultdict(lambda: [0, 0])
    margins = []
    for pk, d in pairs.items():
        if "AB" not in d or "BA" not in d:
            continue
        iab, rab = d["AB"]; iba, rba = d["BA"]
        b_ab = float(logits[iab, 1] - logits[iab, 0])
        b_ba = float(logits[iba, 1] - logits[iba, 0])
        true_ab = int(rab["label"]) == 1
        signed = (b_ab - b_ba) if true_ab else (b_ba - b_ab)
        ok = int(signed > 0)
        correct[0] += ok; correct[1] += 1
        qf = rab.get("query_family", "")
        by_q[qf][0] += ok; by_q[qf][1] += 1
        by_q_margin[qf].append(signed)
        ch = str(rab.get("changed_focal"))
        by_changed[ch][0] += ok; by_changed[ch][1] += 1
        margins.append(signed)
    return {"contrastive_acc": correct[0] / correct[1] if correct[1] else float("nan"),
            "n_pairs": correct[1],
            "by_query": {k: v[0] / v[1] for k, v in by_q.items() if v[1]},
            "by_query_margin": {k: float(np.mean(v)) for k, v in by_q_margin.items() if v},
            "by_changed_focal": {k: v[0] / v[1] for k, v in by_changed.items() if v[1]},
            "margin_mean": float(np.mean(margins)) if margins else float("nan")}


def full_eval(model, tok, evals, ml, dev, bs):
    out = {}
    for name, rows in evals.items():
        logits = get_all_logits(model, tok, rows, ml, dev, bs)
        out[name] = {"standard": standard_eval(logits, rows),
                     "contrastive": contrastive_eval_with_margins(logits, rows)}
    return out


def train_breakdown_from_logits(logits, rows) -> dict[str, Any]:
    if not rows:
        return {}
    pred = logits.argmax(1).numpy()
    lab = np.array([int(r["label"]) for r in rows])
    out: dict[str, Any] = {"overall": float(np.mean(pred == lab))}
    for key in ["train_kind", "query_family", "changed_focal"]:
        groups = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r.get(key, ""))].append(i)
        out[f"by_{key}"] = {g: float(np.mean(pred[idx] == lab[idx])) for g, idx in sorted(groups.items()) if idx}
    return out


def train_one(model_path: Path, tok, arm: str, seed: int, train_rows, evals, args, dev):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = m.DebertaEntailment(model_path, "full", "pretrained").to(dev)
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_p, "lr": args.head_lr}, {"params": enc_p, "lr": args.encoder_lr}],
                            weight_decay=args.weight_decay)
    x, mask, y = encode_rows(tok, train_rows, args.max_len, dev)
    ds = TensorDataset(x, mask, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_state = None; best_train = -1.0; hist = []
    for ep in range(1, args.epochs + 1):
        model.train(); losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); losses.append(float(loss.item()))
        model.eval(); preds = []
        with torch.no_grad():
            for i in range(0, len(train_rows), args.eval_batch_size):
                preds.append(model(x[i:i+args.eval_batch_size], mask[i:i+args.eval_batch_size]).detach())
            ta = float((torch.cat(preds).argmax(1) == y).float().mean())
        hist.append({"epoch": ep, "loss": float(np.mean(losses)), "train_acc": ta})
        if ta > best_train:
            best_train = ta
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    train_logits = get_all_logits(model, tok, train_rows, args.max_len, dev, args.eval_batch_size)
    fit = train_breakdown_from_logits(train_logits, train_rows)
    eval_out = full_eval(model, tok, evals, args.max_len, dev, args.eval_batch_size)
    return {"arm": arm, "seed": seed, "best_train_acc": float(best_train), "train_fit": fit, "history": hist, "evals": eval_out}


def aggregate(results):
    out = {}
    for arm, rs in collections.defaultdict(list, {}).items():
        pass
    by_arm = collections.defaultdict(list)
    for r in results:
        by_arm[r["arm"]].append(r)
    for arm, rs in sorted(by_arm.items()):
        item = {"n_seeds": len(rs), "train_acc_mean": float(np.mean([r["best_train_acc"] for r in rs])), "evals": {}}
        eval_names = sorted(rs[0]["evals"].keys())
        for en in eval_names:
            evitem = {"contrastive_by_query": {}, "contrastive_by_query_margin": {}, "contrastive_acc": {}, "standard_acc": {}}
            evitem["contrastive_acc"] = {"mean": float(np.mean([r["evals"][en]["contrastive"].get("contrastive_acc", float("nan")) for r in rs])),
                                          "std": float(np.std([r["evals"][en]["contrastive"].get("contrastive_acc", float("nan")) for r in rs]))}
            evitem["standard_acc"] = {"mean": float(np.mean([r["evals"][en]["standard"].get("acc", float("nan")) for r in rs])),
                                       "std": float(np.std([r["evals"][en]["standard"].get("acc", float("nan")) for r in rs]))}
            for q in ["focal_before", "focal_after", "secondary_before", "secondary_after"]:
                vals = [r["evals"][en]["contrastive"].get("by_query", {}).get(q) for r in rs]
                vals = [v for v in vals if v is not None]
                if vals:
                    evitem["contrastive_by_query"][q] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                valsm = [r["evals"][en]["contrastive"].get("by_query_margin", {}).get(q) for r in rs]
                valsm = [v for v in valsm if v is not None]
                if valsm:
                    evitem["contrastive_by_query_margin"][q] = {"mean": float(np.mean(valsm)), "std": float(np.std(valsm))}
            item["evals"][en] = evitem
        out[arm] = item
    return out


def write_md(summary, out: Path):
    lines = ["# research fixed-meaning temporal-change bridge\n\n",
             "The experiment uses source-attested ATP ranking snapshots. Focal worlds can change higher-ranked participant between the first and later snapshots; secondary worlds are stable. Sparse training labels query only focal before/after state except in the oracle arm. Evaluation asks for focal update and secondary preservation under train/held wording.\n\n"]
    cons = summary.get("construction", {})
    if cons:
        lines.append("## Construction\n\n")
        lines.append(f"Base rows: {cons['base']['n']} with labels {cons['base']['by_label']}\n\n")
        for arm, d in cons["updates"].items():
            lines.append(f"- {arm}: rows={d['n']} queries={d['by_query']} labels={d['by_label']} changed={d['by_changed_focal']}\n")
        lines.append("\n")
    if summary.get("aggregate"):
        for arm, item in sorted(summary["aggregate"].items()):
            lines.append(f"## Arm: {arm} (train_acc {item['train_acc_mean']:.3f}, {item['n_seeds']} seeds)\n\n")
            lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
            for en, ev in sorted(item["evals"].items()):
                cq = ev["contrastive_by_query"]; cm = ev["contrastive_by_query_margin"]
                def gv(q): return cq.get(q, {}).get("mean", float("nan"))
                def gm(q): return cm.get(q, {}).get("mean", float("nan"))
                lines.append(f"| {en} | {ev['contrastive_acc']['mean']:.3f} | {gv('focal_before'):.3f} | {gv('focal_after'):.3f} | {gv('secondary_before'):.3f} | {gv('secondary_after'):.3f} | {gm('focal_after'):.2f} | {gm('secondary_after'):.2f} |\n")
            lines.append("\n")
    (out / "temporal_change_bridge_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--arms", type=str, default="exposure,stable_only,changed_only,balanced_temporal")
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
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    ap.add_argument("--core_eval_only", action="store_true", help="evaluate only central temporal-change surfaces for cheap pilots")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_SPECS:
            raise ValueError(f"Unknown arm {a}; choose from {ARM_SPECS}")
    base, updates, evals, selection, inventory = build_base_updates_evals(args, arms)
    if args.core_eval_only:
        keep = {"heldChanged_heldStable_trainCtx_trainHyp", "heldChanged_heldStable_trainCtx_heldHyp",
                "heldChanged_heldStable_heldCtx_trainHyp", "heldChanged_heldStable_heldCtx_heldHyp",
                "heldChanged_heldStable_laterFirst_trainHyp", "heldChanged_heldStable_secondaryFirst_trainHyp",
                "heldStable_heldStable_trainCtx_trainHyp", "heldStable_heldStable_trainCtx_heldHyp",
                "heldStable_heldStable_heldCtx_trainHyp", "trainChanged_heldStable_trainCtx_trainHyp"}
        evals = {k: v for k, v in evals.items() if k in keep}
    cons = construction_summary(base, updates, evals, selection, inventory, out)
    print(json.dumps({"status": "BUILT", "out": str(out), "base_rows": len(base),
                      "updates": {a: len(r) for a, r in updates.items()},
                      "eval_sets": len(evals), "selection": cons["selection"]}), flush=True)
    if args.dry_build or args.n_seeds <= 0:
        summary = {"status": "TEMPORAL_CHANGE_BRIDGE_DRY", "args": vars(args), "construction": cons}
        (out / "temporal_change_bridge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
        write_md(summary, out)
        print(json.dumps({"status": "DRY_DONE", "summary_json": str(out / "temporal_change_bridge_summary.json")}), flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = args.device
    results = []
    for arm in arms:
        train_rows = list(base) + updates[arm]
        for si in range(args.n_seeds):
            seed = 26500 + si
            r = train_one(Path(args.model_path), tok, arm, seed, train_rows, evals, args, dev)
            results.append(r)
            partial = {"status": "TEMPORAL_CHANGE_BRIDGE_PARTIAL",
                       "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "args": vars(args), "construction": cons,
                       "aggregate": aggregate(results), "per_seed": results,
                       "boundary": "Incrementally saved partial evidence from a small bridge run."}
            (out / "temporal_change_bridge_partial.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", "utf-8")
            write_md(partial, out)
            # Print the central held/train surfaces immediately.
            for en in ["heldChanged_heldStable_trainCtx_trainHyp", "heldChanged_heldStable_trainCtx_heldHyp",
                       "heldStable_heldStable_trainCtx_trainHyp", "heldChanged_heldStable_heldCtx_trainHyp"]:
                ev = r["evals"].get(en, {}).get("contrastive", {})
                print(json.dumps({"event": "eval", "arm": arm, "seed": seed,
                                  "train_acc": r["best_train_acc"], "train_fit": r.get("train_fit"),
                                  "eval_set": en,
                                  "by_query": ev.get("by_query"),
                                  "by_query_margin": ev.get("by_query_margin"),
                                  "con_acc": ev.get("contrastive_acc")}), flush=True)
    agg = aggregate(results)
    summary = {"status": "TEMPORAL_CHANGE_BRIDGE", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": cons, "aggregate": agg, "per_seed": results,
               "boundary": "Small pretrained bridge; not BabyLM-scale training or final principle by itself."}
    (out / "temporal_change_bridge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "summary_json": str(out / "temporal_change_bridge_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
