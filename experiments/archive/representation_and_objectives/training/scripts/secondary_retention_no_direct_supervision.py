#!/usr/bin/env python3
"""research: no-direct-secondary-supervision retention after sparse orientation updates.

Scientific purpose
------------------
research showed a fit-repaired one-seed factorial dissociation when sparse event
and focal-ranking labels were varied independently while secondary-ranking labels
remained true in the same sparse compound rows.  That result rules out a simple
head-wide polarity on the familiar interface, but it does not show untouched
secondary-state conservation: the secondary query was directly supervised in the
update rows.

This script turns the limitation into a direct measurement.  It first trains a
pretrained DeBERTa entailment head on anchor-wording compound worlds where event,
focal ranking, and secondary ranking are all true.  It records eval margins
BEFORE any sparse train-wording update.  Then it clones that fitted base and
applies sparse updates in which secondary-ranking query labels are deliberately
withheld.  Update arms manipulate only event and/or focal-ranking labels plus
neutral mention rows matched to the same row count.  The readout is whether the
secondary ranking signed margin and contrastive accuracy survive from before to
after update without direct secondary labels on the update interface.

Evaluation labels are always true facts.  Positive signed contrastive margin
means the model gives higher belief to the true AB/BA hypothesis than its swapped
paired alternative.  Accuracy alone can hide margin weakening, so both are saved.
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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

PATH = Path("experiments/archive/representation_and_objectives/training/scripts/independent_orientation_cross_probe.py")
spec = importlib.util.spec_from_file_location("mod", PATH)
s263 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = s263
spec.loader.exec_module(s263)
m = s263.m

DEFAULT_MODEL = s263.DEFAULT_MODEL
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/secondary_retention_no_direct_supervision")

# event_mode, rank_mode, include_event_update, include_rank_update
ARM_SPECS: dict[str, tuple[str, str, bool, bool] | None] = {
    "exposure": None,
    "Etrue_Rtrue": ("true", "true", True, True),
    "Eflip_Rtrue": ("flip", "true", True, True),
    "Etrue_Rflip": ("true", "flip", True, True),
    "Eflip_Rflip": ("flip", "flip", True, True),
    "Etrue_only": ("true", "true", True, False),
    "Eflip_only": ("flip", "true", True, False),
    "Rtrue_only": ("true", "true", False, True),
    "Rflip_only": ("true", "flip", False, True),
}


def apply_mode(label: int, mode: Literal["true", "flip"]) -> int:
    return int(label) if mode == "true" else 1 - int(label)


def compound_update_rows_nosec(
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
    include_event: bool = True,
    include_rank: bool = True,
    tk: str = "sparse_update_nosec",
    hyp_w: str = "train",
) -> list[dict[str, Any]]:
    """Sparse ATP update rows with secondary-state labels withheld.

    Event and/or focal-ranking AB-vs-BA rows are added according to the include
    flags.  Secondary query rows are never added by this function.
    """
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
                    if include_event:
                        ev_lab = apply_mode(m.ev_label(p, hd), event_mode)  # type: ignore[arg-type]
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
                            label_mode=f"event={event_mode}|rank={rank_mode}|secondary=withheld",
                            conflict=conflict,
                            train_kind=tk,
                            hyp_dir=hd,
                            pair_key=f"{base_pk}|event|h={hyp_w}",
                            hyp_wording=hyp_w,
                            ctx_wording=f"event={ev_grp}|state={st_grp}",
                        )
                        r.update({"event_mode": event_mode, "rank_mode": rank_mode,
                                  "untouched_mode": "withheld", "event_ctx_group": ev_grp,
                                  "state_ctx_group": st_grp})
                        rows.append(r)
                    if include_rank:
                        st_lab = apply_mode(m.st_label(p, hd), rank_mode)  # type: ignore[arg-type]
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
                            label_mode=f"event={event_mode}|rank={rank_mode}|secondary=withheld",
                            conflict=conflict,
                            train_kind=tk,
                            hyp_dir=hd,
                            pair_key=f"{base_pk}|focal|h={hyp_w}",
                            hyp_wording=hyp_w,
                            ctx_wording=f"event={ev_grp}|state={st_grp}",
                        )
                        r.update({"event_mode": event_mode, "rank_mode": rank_mode,
                                  "untouched_mode": "withheld", "event_ctx_group": ev_grp,
                                  "state_ctx_group": st_grp})
                        rows.append(r)
    return rows


def neutral_update_rows(
    prims,
    secs,
    *,
    ev_grp: str,
    st_grp: str,
    ns: str,
    split: str,
    ev_set: str,
    arm: str,
    neutral_per_template: int,
) -> list[dict[str, Any]]:
    """Neutral mention rows matched to omitted relation rows.

    They expose the same compound contexts and aliases without giving event,
    focal-ranking, or secondary-ranking directional labels.  For the core arms,
    neutral_per_template=2 exactly replaces the two omitted secondary AB/BA rows
    in each (event-template, state-template) cell.  Event-only/rank-only arms use
    neutral_per_template=4, and exposure uses the research mention exposure helper.
    """
    assert neutral_per_template in {0, 2, 4, 6}
    if neutral_per_template == 0:
        return []
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prims):
        s = secs[(i * 7 + 3) % len(secs)]
        if s.world_id == p.world_id:
            s = secs[(i * 7 + 4) % len(secs)]
        am = m.amap4(p, s, ns)
        used = set(am.values())
        outsiders = m.other_aliases(used, f"neutral264|{ns}|{p.world_id}|{s.world_id}", 4)
        true_aliases = [am[p.participant_a], am[p.participant_b], am[s.participant_a], am[s.participant_b]]
        false_aliases = outsiders
        for ev in range(len(m.EVENT_CTX[ev_grp])):
            for sv in range(len(m.STATE_CTX[st_grp])):
                ctx = (
                    f"Match record: {m.ev_sent(p, am, ev_grp, ev)} "
                    f"Ranking record: {m.st_sent(p, am, st_grp, sv)} "
                    f"Separate ranking record: {m.st_sent(s, am, st_grp, (sv + 1))}"
                )
                tpl = f"e={ev_grp}:{ev}|s={st_grp}:{sv}"
                base_pk = f"{ns}|{p.world_id}|{s.world_id}|{tpl}"
                # Balanced true/false rows, with true aliases rotated so the
                # secondary aliases are exposed but not queried directionally.
                n_true = neutral_per_template // 2
                n_false = neutral_per_template - n_true
                start = (ev * 2 + sv) % len(true_aliases)
                ordered_true = true_aliases[start:] + true_aliases[:start]
                items = [(a, 1) for a in ordered_true[:n_true]] + [(a, 0) for a in false_aliases[:n_false]]
                for j, (alias, lab) in enumerate(items):
                    rows.append(m.mk(
                        row_id=f"{base_pk}|neutral|{j}",
                        context=ctx,
                        hypothesis=f"The passage mentions {alias}.",
                        label=lab,
                        split=split,
                        eval_set=ev_set,
                        arm=arm,
                        source=p.source,
                        sport=p.sport,
                        query_family="neutral_mention",
                        primary_world=p.world_id,
                        secondary_world=s.world_id,
                        template_group=tpl,
                        label_mode="neutral_mention",
                        conflict=not bool(p.winner_higher_at_time),
                        train_kind="neutral_update",
                    ))
    return rows


def compound_eval_rows(
    prims,
    secs,
    *,
    ev_grp: str,
    st_grp: str,
    ns: str,
    split: str,
    ev_set: str,
    arm: str,
    hyp_w: str = "train",
    order: str = "standard",
    include_event_record: bool = True,
    include_focal_state_record: bool = True,
    include_secondary_state_record: bool = True,
) -> list[dict[str, Any]]:
    """True-label ATP evaluation rows with optional record deletion/order change."""
    rows: list[dict[str, Any]] = []
    for i, p in enumerate(prims):
        s = secs[(i * 7 + 3) % len(secs)]
        if s.world_id == p.world_id:
            s = secs[(i * 7 + 4) % len(secs)]
        am = m.amap4(p, s, ns)
        for ev in range(len(m.EVENT_CTX[ev_grp])):
            for sv in range(len(m.STATE_CTX[st_grp])):
                rec_event = f"Match record: {m.ev_sent(p, am, ev_grp, ev)}"
                rec_focal = f"Ranking record: {m.st_sent(p, am, st_grp, sv)}"
                rec_secondary = f"Separate ranking record: {m.st_sent(s, am, st_grp, (sv + 1))}"
                parts = []
                if order == "secondary_first":
                    cand = [(include_secondary_state_record, rec_secondary), (include_event_record, rec_event), (include_focal_state_record, rec_focal)]
                elif order == "state_first":
                    cand = [(include_focal_state_record, rec_focal), (include_secondary_state_record, rec_secondary), (include_event_record, rec_event)]
                else:
                    cand = [(include_event_record, rec_event), (include_focal_state_record, rec_focal), (include_secondary_state_record, rec_secondary)]
                for ok, txt in cand:
                    if ok:
                        parts.append(txt)
                ctx = " ".join(parts)
                tpl = f"e={ev_grp}:{ev}|s={st_grp}:{sv}|order={order}|hasE={include_event_record}|hasF={include_focal_state_record}|hasS={include_secondary_state_record}"
                base_pk = f"{ns}|{p.world_id}|{s.world_id}|{tpl}"
                conflict = not bool(p.winner_higher_at_time)
                for hd in ["AB", "BA"]:
                    for qf, hyp, lab, src, pw in [
                        ("event_role", m.hyp_ev(am, p, hd, hyp_w), m.ev_label(p, hd), p.source, p.world_id),
                        ("focal_state", m.hyp_st(am, p, hd, hyp_w), m.st_label(p, hd), p.source, p.world_id),
                        ("untouched_state", m.hyp_st(am, s, hd, hyp_w), m.st_label(s, hd), s.source, p.world_id),
                    ]:
                        rows.append(m.mk(
                            row_id=f"{base_pk}|{qf}|{hd}|h={hyp_w}",
                            context=ctx,
                            hypothesis=hyp,
                            label=int(lab),
                            split=split,
                            eval_set=ev_set,
                            arm=arm,
                            source=src,
                            sport=p.sport,
                            query_family=qf,
                            primary_world=pw,
                            secondary_world=s.world_id,
                            template_group=tpl,
                            label_mode="true_eval",
                            conflict=conflict,
                            train_kind="eval_compound",
                            hyp_dir=hd,
                            pair_key=f"{base_pk}|{qf}|h={hyp_w}",
                            hyp_wording=hyp_w,
                            ctx_wording=f"event={ev_grp}|state={st_grp}|order={order}",
                        ))
    return rows


def select_worlds(args):
    atp_tr, atp_he = m.load_atp()
    pair_tr, pair_he = m.load_pairs()
    return {
        "atp_train": m.select(atp_tr, args.train_atp, 26401, True),
        "atp_train2": m.select(atp_tr, args.train_atp + 17, 26402, True),
        "atp_sparse": m.select(atp_tr, args.sparse_atp, 26403, True),
        "atp_sparse2": m.select(atp_tr, args.sparse_atp + 17, 26404, True),
        "atp_eval": m.select(atp_he, args.eval_atp, 26405, True),
        "atp_eval2": m.select(atp_he, args.eval_atp + 17, 26406, True),
        "pair_train": m.select(pair_tr, args.train_pair, 26407),
        "pair_sparse": m.select(pair_tr, args.sparse_pair, 26408),
        "pair_eval": m.select(pair_he, args.eval_pair, 26409),
    }


def build_base_and_evals(args, arms: list[str]):
    w = select_worlds(args)
    base: list[dict[str, Any]] = []
    base_ctx_groups = [x.strip() for x in getattr(args, "base_ctx_groups", "anchor").split(",") if x.strip()]
    for bg in base_ctx_groups:
        if bg not in {"anchor", "train"}:
            raise ValueError(f"base_ctx_groups may contain only anchor/train, got {bg!r}")
        base += s263.compound_rows_indep(
            w["atp_train"], w["atp_train2"], ev_grp=bg, st_grp=bg,
            ns=f"base_{bg}264", split="train", ev_set="base", arm="base",
            event_mode="true", rank_mode="true", untouched_mode="true", tk=f"base_compound_{bg}")
        base += m.event_only_rows(
            w["pair_train"], ev_grp=bg, ns=f"base_pair_{bg}264", split="train",
            ev_set="base", arm="base", lm="true", tk=f"base_pair_event_{bg}")

    ctx_conditions = [
        ("trainTrain", "train", "train", "standard", True, True, True),
        ("heldHeld", "held", "held", "standard", True, True, True),
        ("dirDir", "dirpara", "dirpara", "standard", True, True, True),
        ("trainEvent_dirState", "train", "dirpara", "standard", True, True, True),
        ("dirEvent_trainState", "dirpara", "train", "standard", True, True, True),
        ("trainEvent_nonState", "train", "nonpara", "standard", True, True, True),
        ("nonEvent_trainState", "nonpara", "train", "standard", True, True, True),
        ("trainTrain_stateFirst", "train", "train", "state_first", True, True, True),
        ("trainTrain_secondaryFirst", "train", "train", "secondary_first", True, True, True),
        ("stateOnly_train", "train", "train", "state_first", False, True, True),
        ("eventOnly_train", "train", "train", "standard", True, False, False),
    ]

    evals: dict[str, list[dict[str, Any]]] = {}
    for cname, eg, sg, order, has_e, has_f, has_s in ctx_conditions:
        for hw in ["train", "held"]:
            evals[f"atp_{cname}_{hw}Hyp"] = compound_eval_rows(
                w["atp_eval"], w["atp_eval2"], ev_grp=eg, st_grp=sg,
                ns=f"eval264_{cname}_{hw}", split="held", ev_set=f"atp_{cname}_{hw}Hyp",
                arm="eval", hyp_w=hw, order=order,
                include_event_record=has_e, include_focal_state_record=has_f,
                include_secondary_state_record=has_s)
    for eg in ["train", "held", "dirpara", "nonpara"]:
        for hw in ["train", "held"]:
            evals[f"pair_{eg}Ctx_{hw}Hyp"] = m.event_only_rows(
                w["pair_eval"], ev_grp=eg, ns=f"eval264_pair_{eg}_{hw}",
                split="held", ev_set=f"pair_{eg}Ctx_{hw}Hyp",
                arm="eval", lm="true", tk="eval_pair_event", hyp_w=hw)

    updates: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        spec = ARM_SPECS[arm]
        if spec is None:
            rows = []
            rows += m.exposure_rows(w["atp_sparse"], w["atp_sparse2"], ev_grp="train", st_grp="train",
                                    ns="sp264_exp", split="train", ev_set="update", arm=arm)
            rows += m.paired_exposure_event(w["pair_sparse"], ev_grp="train", ns="sp264_pair_exp",
                                            split="train", ev_set="update", arm=arm)
        else:
            event_mode, rank_mode, include_event, include_rank = spec
            rows = []
            rows += compound_update_rows_nosec(
                w["atp_sparse"], w["atp_sparse2"], ev_grp="train", st_grp="train",
                ns=f"sp264_rel_{arm}", split="train", ev_set="update", arm=arm,
                event_mode=event_mode, rank_mode=rank_mode,
                include_event=include_event, include_rank=include_rank,
                tk=f"sparse_{arm}_nosec")
            relation_per_world = (8 if include_event else 0) + (8 if include_rank else 0)
            neutral_per_template = (24 - relation_per_world) // 4
            rows += neutral_update_rows(
                w["atp_sparse"], w["atp_sparse2"], ev_grp="train", st_grp="train",
                ns=f"sp264_rel_{arm}", split="train", ev_set="update", arm=arm,
                neutral_per_template=neutral_per_template)
            if include_event:
                rows += m.event_only_rows(w["pair_sparse"], ev_grp="train", ns=f"sp264_pair_{arm}",
                                          split="train", ev_set="update", arm=arm,
                                          lm=event_mode, tk=f"sparse_{arm}_pair")
            else:
                rows += m.paired_exposure_event(w["pair_sparse"], ev_grp="train", ns=f"sp264_pair_exp_{arm}",
                                                split="train", ev_set="update", arm=arm)
        updates[arm] = rows
    return base, updates, evals, w


def describe_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind = collections.Counter(r.get("train_kind", "") for r in rows)
    by_query = collections.Counter(r.get("query_family", "") for r in rows)
    by_label = collections.Counter(str(r.get("label")) for r in rows)
    by_mode = collections.Counter(r.get("label_mode", "") for r in rows)
    return {"n": len(rows), "by_kind": dict(sorted(by_kind.items())),
            "by_query": dict(sorted(by_query.items())), "by_label": dict(sorted(by_label.items())),
            "by_mode": dict(sorted(by_mode.items()))}


def select_replay_rows(base: list[dict[str, Any]], *, rows_per_query: int,
                       query_families: list[str], seed: int = 26499) -> list[dict[str, Any]]:
    """Deterministically select balanced true-label replay rows from base.

    Replay rows are old/base facts, not sparse-update-world labels.  They are
    copied with new ids and train_kind so later analyses can distinguish them.
    rows_per_query is the target total per query family, split as evenly as
    possible over labels 0/1.  If the requested number exceeds available rows,
    the largest balanced subset is used.
    """
    rng = random.Random(seed)
    replay: list[dict[str, Any]] = []
    for qf in query_families:
        qrows = [r for r in base if r.get("query_family") == qf]
        by_lab = {0: [], 1: []}
        for r in qrows:
            by_lab[int(r["label"])].append(r)
        for lab in [0, 1]:
            rng.shuffle(by_lab[lab])
        target_each = rows_per_query // 2
        n_each = min(target_each, len(by_lab[0]), len(by_lab[1]))
        chosen = by_lab[0][:n_each] + by_lab[1][:n_each]
        rng.shuffle(chosen)
        for j, r in enumerate(chosen):
            rr = dict(r)
            rr["id"] = f"replay264|{qf}|{j}|" + str(r.get("id", ""))
            rr["train_kind"] = "state_replay_update"
            rr["eval_set"] = "update_replay"
            rr["label_mode"] = "replay_true"
            replay.append(rr)
    rng.shuffle(replay)
    return replay


def construction_summary(base, updates, evals, out: Path, replay_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    s = {"base": describe_rows(base),
         "replay": describe_rows(replay_rows or []),
         "updates": {a: describe_rows(r) for a, r in updates.items()},
         "eval_counts": {k: len(v) for k, v in sorted(evals.items())}}
    (out / "construction_summary.json").write_text(json.dumps(s, indent=2) + "\n", "utf-8")
    return s


def encode_rows(tok, rows, ml, dev):
    texts = [r["text"] for r in rows]
    enc = tok(texts, padding=True, truncation=True, max_length=ml, return_tensors="pt")
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev), labels.to(dev)


def get_all_logits(model, tok, rows, ml, dev, bs):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            batch = rows[i:i + bs]
            x, mask, _ = encode_rows(tok, batch, ml, dev)
            logits_list.append(model(x, mask).detach().cpu())
    return torch.cat(logits_list, 0) if logits_list else torch.zeros(0, 2)


def standard_eval(logits, rows):
    return m.standard_eval(logits, rows)


def contrastive_eval_with_margins(logits, rows):
    if not rows:
        return {}
    pairs: dict[str, dict[str, tuple[int, dict[str, Any]]]] = collections.defaultdict(dict)
    for i, r in enumerate(rows):
        pk = r.get("pair_key", "")
        hd = r.get("hyp_dir", "")
        if pk and hd:
            pairs[pk][hd] = (i, r)
    correct = [0, 0]
    by_query: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    by_query_margin: dict[str, list[float]] = collections.defaultdict(list)
    by_conflict: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0.0])
    by_conflict_margin: dict[str, list[float]] = collections.defaultdict(list)
    margins = []
    raw_margins = []
    for pk, dirs in pairs.items():
        if "AB" not in dirs or "BA" not in dirs:
            continue
        i_ab, r_ab = dirs["AB"]
        i_ba, r_ba = dirs["BA"]
        belief_ab = float(logits[i_ab, 1] - logits[i_ab, 0])
        belief_ba = float(logits[i_ba, 1] - logits[i_ba, 0])
        raw = belief_ab - belief_ba
        true_ab = bool(r_ab["label"] == 1)
        signed = raw if true_ab else -raw
        ok = int(signed > 0.0)
        correct[0] += ok
        correct[1] += 1
        margins.append(signed)
        raw_margins.append(raw)
        qf = r_ab.get("query_family", "")
        by_query[qf][0] += ok
        by_query[qf][1] += 1
        by_query_margin[qf].append(signed)
        cf = r_ab.get("conflict")
        if cf is not None:
            ck = f"{qf}_{'conflict' if cf else 'nonconflict'}"
            by_conflict[ck][0] += ok
            by_conflict[ck][1] += 1
            by_conflict_margin[ck].append(signed)
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    def std(xs):
        return float(np.std(xs)) if xs else float("nan")
    return {
        "contrastive_acc": correct[0] / correct[1] if correct[1] else float("nan"),
        "n_pairs": correct[1],
        "by_query": {k: v[0] / v[1] if v[1] else float("nan") for k, v in by_query.items()},
        "by_query_signed_margin": {k: mean(v) for k, v in by_query_margin.items()},
        "by_query_signed_margin_std": {k: std(v) for k, v in by_query_margin.items()},
        "by_conflict": {k: v[0] / v[1] if v[1] else float("nan") for k, v in by_conflict.items()},
        "by_conflict_signed_margin": {k: mean(v) for k, v in by_conflict_margin.items()},
        "overall_signed_margin": mean(margins),
        "overall_raw_ab_minus_ba_margin": mean(raw_margins),
    }


def eval_model(model, tok, evals, args, dev, names: list[str] | None = None):
    out = {}
    use_names = names if names is not None else sorted(evals.keys())
    for name in use_names:
        rows = evals[name]
        logits = get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size)
        out[name] = {"standard": standard_eval(logits, rows),
                     "contrastive": contrastive_eval_with_margins(logits, rows)}
    return out


def train_acc(model, tok, rows, args, dev) -> float:
    if not rows:
        return float("nan")
    logits = get_all_logits(model, tok, rows, args.max_len, dev, args.eval_batch_size)
    pred = logits.argmax(1).numpy()
    lab = np.array([int(r["label"]) for r in rows])
    return float(np.mean(pred == lab))


def train_rows(model, tok, rows, args, dev, epochs: int, head_lr: float, encoder_lr: float, tag: str):
    x, mask, y = encode_rows(tok, rows, args.max_len, dev)
    ds = TensorDataset(x, mask, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    head_p = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_p = [p for n, p in model.named_parameters() if not n.startswith("head") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head_p, "lr": head_lr}, {"params": enc_p, "lr": encoder_lr}],
                            weight_decay=args.weight_decay)
    best_acc = -1.0
    best_state = None
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb, mb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss.item()))
        acc = train_acc(model, tok, rows, args, dev)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "train_acc": acc, "tag": tag})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_train_acc": float(best_acc), "history": history}


def stat(vals):
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs)}


def metric_eval(evals, name, q, field="by_query"):
    try:
        return evals[name]["contrastive"][field][q]
    except KeyError:
        return float("nan")


def aggregate(per_seed: list[dict[str, Any]], eval_names: list[str], arms: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"base": {}, "arms": {}, "retention_delta": {}}
    out["base"]["train_acc"] = stat([s["base_train_acc"] for s in per_seed])
    for en in eval_names:
        out["base"][en] = {}
        for q in ["event_role", "focal_state", "untouched_state"]:
            out["base"][en][q] = {
                "acc": stat([metric_eval(s["before_evals"], en, q, "by_query") for s in per_seed]),
                "margin": stat([metric_eval(s["before_evals"], en, q, "by_query_signed_margin") for s in per_seed]),
            }
    for arm in arms:
        ars = [s["arms"][arm] for s in per_seed]
        out["arms"][arm] = {
            "update_train_acc": stat([a["update_train_acc"] for a in ars]),
            "base_anchor_acc_after": stat([a["base_anchor_acc_after"] for a in ars]),
            "evals": {},
        }
        for en in eval_names:
            out["arms"][arm]["evals"][en] = {}
            for q in ["event_role", "focal_state", "untouched_state"]:
                out["arms"][arm]["evals"][en][q] = {
                    "acc": stat([metric_eval(a["after_evals"], en, q, "by_query") for a in ars]),
                    "margin": stat([metric_eval(a["after_evals"], en, q, "by_query_signed_margin") for a in ars]),
                }
                before = [metric_eval(s["before_evals"], en, q, "by_query_signed_margin") for s in per_seed]
                after = [metric_eval(a["after_evals"], en, q, "by_query_signed_margin") for a in ars]
                out["retention_delta"].setdefault(arm, {}).setdefault(en, {})[q] = stat([af - bf for af, bf in zip(after, before)])
    return out


def write_md(summary: dict[str, Any], out: Path):
    agg = summary.get("aggregate", {})
    key_evals = summary.get("key_eval_names", [])
    arms = summary.get("arms", [])
    lines = ["# research secondary retention without direct sparse secondary supervision\n\n"]
    lines.append("Base training uses anchor-wording event/focal/secondary labels. Sparse update rows omit secondary-state directional labels and replace them with neutral mention rows so update row counts stay matched. Positive contrastive margin means the true AB/BA fact has higher belief than the swapped alternative.\n\n")
    if summary.get("construction"):
        lines.append("## Construction\n\n")
        lines.append(f"- base rows: {summary['construction']['base']['n']} {summary['construction']['base']['by_query']} labels={summary['construction']['base']['by_label']}\n")
        if summary['construction'].get('replay', {}).get('n', 0):
            rep = summary['construction']['replay']
            lines.append(f"- replay rows appended to every update arm: {rep['n']} {rep['by_query']} labels={rep['by_label']}\n")
        for a, d in summary['construction']['updates'].items():
            lines.append(f"- update {a}: rows={d['n']} queries={d['by_query']} labels={d['by_label']}\n")
        lines.append("\n")
    if agg:
        btr = agg['base']['train_acc']
        lines.append(f"## Base before sparse update (train acc mean={btr['mean']:.3f}, std={btr['std']:.3f}, seeds={btr['n']})\n\n")
        lines.append("| eval | event acc/margin | focal acc/margin | secondary acc/margin |\n|---|---:|---:|---:|\n")
        for en in key_evals:
            be = agg['base'].get(en, {})
            def fmt(q):
                return f"{be.get(q,{}).get('acc',{}).get('mean', float('nan')):.3f}/{be.get(q,{}).get('margin',{}).get('mean', float('nan')):.3f}"
            lines.append(f"| {en} | {fmt('event_role')} | {fmt('focal_state')} | {fmt('untouched_state')} |\n")
        lines.append("\n")
        for arm in arms:
            ar = agg['arms'][arm]
            utr = ar['update_train_acc']; bar = ar['base_anchor_acc_after']
            lines.append(f"## After update arm {arm} (update acc mean={utr['mean']:.3f}, base-anchor-after={bar['mean']:.3f}, seeds={utr['n']})\n\n")
            lines.append("| eval | event acc/margin | focal acc/margin | secondary acc/margin | secondary margin delta |\n|---|---:|---:|---:|---:|\n")
            for en in key_evals:
                ee = ar['evals'].get(en, {})
                def fmt(q):
                    return f"{ee.get(q,{}).get('acc',{}).get('mean', float('nan')):.3f}/{ee.get(q,{}).get('margin',{}).get('mean', float('nan')):.3f}"
                dm = agg['retention_delta'].get(arm, {}).get(en, {}).get('untouched_state', {}).get('mean', float('nan'))
                lines.append(f"| {en} | {fmt('event_role')} | {fmt('focal_state')} | {fmt('untouched_state')} | {dm:.3f} |\n")
            lines.append("\n")
    (out / "secondary_retention_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--arms", type=str, default="exposure,Etrue_Rtrue,Eflip_Rtrue,Etrue_Rflip,Eflip_Rflip")
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--train_atp", type=int, default=80)
    ap.add_argument("--sparse_atp", type=int, default=16)
    ap.add_argument("--eval_atp", type=int, default=80)
    ap.add_argument("--train_pair", type=int, default=80)
    ap.add_argument("--sparse_pair", type=int, default=16)
    ap.add_argument("--eval_pair", type=int, default=100)
    ap.add_argument("--base_epochs", type=int, default=6)
    ap.add_argument("--base_ctx_groups", type=str, default="anchor",
                    help="comma-separated base context groups to train before sparse update; anchor or anchor,train")
    ap.add_argument("--update_epochs", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--base_head_lr", type=float, default=1e-3)
    ap.add_argument("--base_encoder_lr", type=float, default=8e-5)
    ap.add_argument("--update_head_lr", type=float, default=8e-4)
    ap.add_argument("--update_encoder_lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    ap.add_argument("--replay_rows_per_query", type=int, default=0,
                    help="append this many balanced base replay rows per listed query family to every sparse update arm")
    ap.add_argument("--replay_queries", type=str, default="focal_state,untouched_state",
                    help="comma-separated base query families to replay during sparse update")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for a in arms:
        if a not in ARM_SPECS:
            raise ValueError(f"unknown arm {a}; allowed {sorted(ARM_SPECS)}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"status": "START", "args": vars(args), "arms": arms,
                      "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)

    base, updates, evals, worlds = build_base_and_evals(args, arms)
    replay_rows: list[dict[str, Any]] = []
    if args.replay_rows_per_query > 0:
        replay_queries = [q.strip() for q in args.replay_queries.split(',') if q.strip()]
        replay_rows = select_replay_rows(base, rows_per_query=args.replay_rows_per_query,
                                         query_families=replay_queries)
        for arm in list(updates.keys()):
            updates[arm] = list(updates[arm]) + [dict(r) for r in replay_rows]
    csum = construction_summary(base, updates, evals, out, replay_rows)
    print(json.dumps({"event": "construction", "base_rows": len(base),
                      "replay_rows": len(replay_rows),
                      "updates": {a: len(r) for a, r in updates.items()},
                      "eval_sets": len(evals)}), flush=True)
    if args.dry_build:
        (out / "dry_build_done.json").write_text(json.dumps({"status": "dry_build_done", "construction": csum}, indent=2) + "\n", "utf-8")
        print(json.dumps({"status": "DRY_BUILD_DONE", "out": str(out)}), flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = args.device
    key_eval_names = [
        "atp_trainTrain_trainHyp", "atp_trainTrain_heldHyp",
        "atp_heldHeld_trainHyp", "atp_dirDir_trainHyp",
        "atp_trainTrain_stateFirst_trainHyp", "atp_trainTrain_secondaryFirst_trainHyp",
        "atp_stateOnly_train_trainHyp", "atp_eventOnly_train_trainHyp",
        "atp_trainEvent_nonState_trainHyp", "atp_nonEvent_trainState_trainHyp",
    ]
    # Keep full evals in JSON, but report these in stdout/markdown.
    per_seed = []
    for si in range(args.n_seeds):
        seed = 26400 + si
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = m.DebertaEntailment(args.model_path, "full", "pretrained").to(dev)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        base_train = train_rows(model, tok, base, args, dev, args.base_epochs,
                                args.base_head_lr, args.base_encoder_lr, tag="base")
        base_train_acc = base_train["best_train_acc"]
        before_evals = eval_model(model, tok, evals, args, dev)
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        seed_rec: dict[str, Any] = {"seed": seed, "trainable_params": int(trainable),
                                    "base_train": base_train, "base_train_acc": float(base_train_acc),
                                    "before_evals": before_evals, "arms": {}}
        print(json.dumps({"event": "base_done", "seed": seed, "base_train_acc": base_train_acc,
                          "before_trainTrain_secondary_acc": before_evals["atp_trainTrain_trainHyp"]["contrastive"]["by_query"].get("untouched_state"),
                          "before_trainTrain_secondary_margin": before_evals["atp_trainTrain_trainHyp"]["contrastive"]["by_query_signed_margin"].get("untouched_state")}), flush=True)
        for arm in arms:
            model.load_state_dict(base_state)
            upd = train_rows(model, tok, updates[arm], args, dev, args.update_epochs,
                             args.update_head_lr, args.update_encoder_lr, tag=f"update_{arm}")
            update_train_acc = upd["best_train_acc"]
            base_after = train_acc(model, tok, base, args, dev)
            after_evals = eval_model(model, tok, evals, args, dev)
            seed_rec["arms"][arm] = {"update_train": upd, "update_train_acc": float(update_train_acc),
                                      "base_anchor_acc_after": float(base_after), "after_evals": after_evals}
            con = after_evals["atp_trainTrain_trainHyp"]["contrastive"]
            print(json.dumps({"event": "arm_done", "seed": seed, "arm": arm,
                              "update_train_acc": update_train_acc,
                              "base_anchor_acc_after": base_after,
                              "trainTrain_event": con["by_query"].get("event_role"),
                              "trainTrain_focal": con["by_query"].get("focal_state"),
                              "trainTrain_secondary": con["by_query"].get("untouched_state"),
                              "trainTrain_secondary_margin": con["by_query_signed_margin"].get("untouched_state")}), flush=True)
        per_seed.append(seed_rec)
        del model
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    agg = aggregate(per_seed, key_eval_names, arms)
    summary = {"status": "SECONDARY_RETENTION_NO_DIRECT_SUPERVISION",
               "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "arms": arms, "key_eval_names": key_eval_names,
               "construction": csum, "aggregate": agg, "per_seed": per_seed}
    (out / "secondary_retention_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(summary, out)
    print(json.dumps({"status": "DONE", "summary_json": str(out / "secondary_retention_summary.json")}), flush=True)


if __name__ == "__main__":
    main()
