#!/usr/bin/env python3
"""research: pretrained-language probe for reusable filler interface + sparse orientation.

Scientific purpose
------------------
research showed in controlled BiGRUs that the useful mechanism is not identity
randomization by itself, but a reusable argument-slot interface plus reliable
sparse evidence that orients a relation coordinate.  The next proposed comparison is a cheaper pretrained-language test before any 100M LM trajectory:
use the repaired ATP state substrate and paired-world records, hold the filler
interface fixed, vary aligned/flipped/exposure-matched sparse relation evidence,
and require the same learner to answer affected event-role queries while preserving
independently attested state for the focal and untouched pair.

This script is deliberately small.  It initializes a sequence classifier from an
existing BabyLM DeBERTa MLM checkpoint and trains only a tiny entailment head
(frozen regime) or the head plus top two encoder layers (top2 regime).  It does
not continue pretraining or create a BabyLM submission model.

Rows are NLI-style: <compound context> [SEP] <hypothesis>.  Every participant name
is replaced by a deterministic alias from one shared in-distribution name pool,
independent of labels.  The same alias interface is used in all arms.

Arms
----
base training: anchor event and rank-state context templates on train ATP worlds,
plus anchor event examples from the research paired-world train material.

sparse arms add matched probe-context rows:
  exposure : same probe contexts and aliases, but only mention/same-token labels;
             no win/rank orientation labels.
  aligned  : correct probe event/rank relation labels.
  flipped  : exactly the same probe relation rows with complemented labels.

Evaluation asks whether aligned vs flipped sparse evidence changes a pretrained
learner on:
  * ATP held-family compound contexts with probe wording;
  * ATP held-family compound contexts with held wording;
  * conflict state rows where match winner and higher-ranked player disagree;
  * untouched-state rows for a separate pair in the same context;
  * research held paired-world event-role transfer, split by sport/domain.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2Model

STUDY = Path("experiments/archive/representation_and_objectives")
WORKSPACE = STUDY
ATP_DIR = WORKSPACE / "data/upset_balanced_state_substrate"
PAIR_DIR = WORKSPACE / "data/paired_world_pilot"
DEFAULT_MODEL = Path("experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M")
DEFAULT_OUT = WORKSPACE / "data/pretrained_atp_slot_orientation_probe"

# Common, ordinary name tokens: all arms use the same finite filler interface.
# These are not label-bearing and are sampled deterministically from row keys.
ALIAS_POOL = [
    "Alice", "Ben", "Clara", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Aaron", "Bella", "Caleb", "Diana", "Ethan", "Fiona",
    "Gavin", "Hannah", "Isaac", "Julia", "Kevin", "Laura", "Mason", "Nora",
    "Owen", "Paula", "Riley", "Sarah", "Thomas", "Vera", "Will", "Nina",
]

EVENT_TEMPLATES = {
    "anchor": [
        "{W} defeated {L} in the match.",
        "{L} lost to {W} in the match.",
    ],
    "probe": [
        "{W} prevailed over {L} in the match.",
        "{L} was beaten by {W} in the match.",
    ],
    "held": [
        "{W} edged out {L} in the match.",
        "{L} went down to {W} in the match.",
    ],
}

STATE_TEMPLATES = {
    "anchor": [
        "The weekly ranking record put {H} above {Lo}.",
        "That week, {Lo} was ranked below {H}.",
    ],
    "probe": [
        "According to the rankings, {H} stood higher than {Lo}.",
        "The ATP list placed {Lo} under {H}.",
    ],
    "held": [
        "{H} held the better ranking position than {Lo}.",
        "{Lo} trailed {H} in the weekly rankings.",
    ],
}

MENTION_TEMPLATES = [
    "The passage mentions {A}.",
    "The passage mentions {B}.",
    "The passage mentions {C}.",
    "The passage mentions {D}.",
    "The passage mentions {X}.",
    "The passage mentions {Y}.",
]


@dataclass(frozen=True)
class World:
    source: str
    family_id: str
    world_id: str
    split: str
    sport: str
    participant_a: str
    participant_b: str
    winner_name: str
    loser_name: str
    winner_label: str  # A or B
    higher_name: str | None = None
    lower_name: str | None = None
    winner_higher_at_time: bool | None = None
    original_event_text: str = ""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def choose_aliases(key: str, n: int) -> list[str]:
    rng = random.Random(stable_int("aliases|" + key))
    return rng.sample(ALIAS_POOL, n)


def other_aliases(used: set[str], key: str, n: int = 2) -> list[str]:
    candidates = [x for x in ALIAS_POOL if x not in used]
    rng = random.Random(stable_int("other|" + key))
    return rng.sample(candidates, n)


def load_atp_worlds() -> tuple[list[World], list[World]]:
    out: dict[str, list[World]] = {"train": [], "held": []}
    for split in ["train", "held"]:
        for fam in read_jsonl(ATP_DIR / f"families_{split}.jsonl"):
            for wk in ["context1", "context2"]:
                cx = fam[wk]
                winner = cx["winner"]
                loser = cx["loser"]
                a = fam["participant_a"]
                b = fam["participant_b"]
                winner_label = "A" if winner == a else "B"
                if cx["winner_higher_at_time"]:
                    higher, lower = winner, loser
                else:
                    higher, lower = loser, winner
                out[split].append(World(
                    source="atp_rank_event_state",
                    family_id=fam["family_id"],
                    world_id=f"{fam['family_id']}::{wk}",
                    split=split,
                    sport="tennis_atp",
                    participant_a=a,
                    participant_b=b,
                    winner_name=winner,
                    loser_name=loser,
                    winner_label=winner_label,
                    higher_name=higher,
                    lower_name=lower,
                    winner_higher_at_time=bool(cx["winner_higher_at_time"]),
                    original_event_text=cx.get("event_text", ""),
                ))
    return out["train"], out["held"]


def load_pair_worlds() -> tuple[list[World], list[World]]:
    out: dict[str, list[World]] = {"train": [], "held": []}
    for split in ["train", "held"]:
        for fam in read_jsonl(PAIR_DIR / f"families_{split}.jsonl"):
            for wk in ["context1", "context2"]:
                cx = fam[wk]
                winner = cx["winner_name"]
                loser = cx["loser_name"]
                a = fam["participant_a"]
                b = fam["participant_b"]
                winner_label = "A" if winner == a else "B"
                out[split].append(World(
                    source="paired_world_event",
                    family_id=fam["family_id"],
                    world_id=f"{fam['family_id']}::{wk}",
                    split=split,
                    sport=str(fam.get("sport") or fam.get("source_type") or "sport"),
                    participant_a=a,
                    participant_b=b,
                    winner_name=winner,
                    loser_name=loser,
                    winner_label=winner_label,
                    original_event_text=cx.get("text_score_ablated", ""),
                ))
    return out["train"], out["held"]


def select_worlds(worlds: list[World], limit: int, seed: int, require_state: bool = False) -> list[World]:
    pool = [w for w in worlds if (not require_state or w.higher_name is not None)]
    rng = random.Random(seed)
    pool = pool[:]
    rng.shuffle(pool)
    return pool[: min(limit, len(pool))]


def alias_map_two(w: World, namespace: str) -> dict[str, str]:
    a_alias, b_alias = choose_aliases(f"two|{namespace}|{w.world_id}", 2)
    return {w.participant_a: a_alias, w.participant_b: b_alias}


def alias_map_four(primary: World, secondary: World, namespace: str) -> dict[str, str]:
    aliases = choose_aliases(f"four|{namespace}|{primary.world_id}|{secondary.world_id}", 4)
    return {
        primary.participant_a: aliases[0], primary.participant_b: aliases[1],
        secondary.participant_a: aliases[2], secondary.participant_b: aliases[3],
    }


def amap_name(amap: dict[str, str], name: str) -> str:
    try:
        return amap[name]
    except KeyError as e:
        raise KeyError(f"name {name!r} absent from alias map {amap}") from e


def event_sentence(w: World, amap: dict[str, str], group: str, variant: int) -> str:
    W = amap_name(amap, w.winner_name)
    L = amap_name(amap, w.loser_name)
    return EVENT_TEMPLATES[group][variant % len(EVENT_TEMPLATES[group])].format(W=W, L=L)


def state_sentence(w: World, amap: dict[str, str], group: str, variant: int) -> str:
    assert w.higher_name is not None and w.lower_name is not None
    H = amap_name(amap, w.higher_name)
    Lo = amap_name(amap, w.lower_name)
    return STATE_TEMPLATES[group][variant % len(STATE_TEMPLATES[group])].format(H=H, Lo=Lo)


def event_label_for(w: World, hyp_dir: str) -> int:
    return int((hyp_dir == "AB" and w.winner_label == "A") or (hyp_dir == "BA" and w.winner_label == "B"))


def state_label_for(w: World, hyp_dir: str) -> int:
    assert w.higher_name is not None
    higher_label = "A" if w.higher_name == w.participant_a else "B"
    return int((hyp_dir == "AB" and higher_label == "A") or (hyp_dir == "BA" and higher_label == "B"))


def hyp_event(amap: dict[str, str], w: World, hyp_dir: str) -> str:
    x = amap_name(amap, w.participant_a if hyp_dir == "AB" else w.participant_b)
    y = amap_name(amap, w.participant_b if hyp_dir == "AB" else w.participant_a)
    return f"{x} won the match against {y}."


def hyp_state(amap: dict[str, str], w: World, hyp_dir: str) -> str:
    x = amap_name(amap, w.participant_a if hyp_dir == "AB" else w.participant_b)
    y = amap_name(amap, w.participant_b if hyp_dir == "AB" else w.participant_a)
    return f"{x} held the higher ranking than {y}."


def make_row(
    *, row_id: str, context: str, hypothesis: str, label: int,
    split: str, eval_set: str, arm: str, source: str, sport: str, query_family: str,
    primary_world: str, secondary_world: str | None = None, template_group: str = "",
    label_mode: str = "true", conflict: bool | None = None, train_kind: str = "",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "text": context + " [SEP] " + hypothesis,
        "context": context,
        "hypothesis": hypothesis,
        "label": int(label),
        "split": split,
        "eval_set": eval_set,
        "arm": arm,
        "source": source,
        "sport": sport,
        "query_family": query_family,
        "primary_world": primary_world,
        "secondary_world": secondary_world,
        "template_group": template_group,
        "label_mode": label_mode,
        "conflict": bool(conflict) if conflict is not None else None,
        "train_kind": train_kind,
    }


def compound_rows(
    primaries: list[World], secondaries: list[World], *, event_group: str, state_group: str,
    namespace: str, split: str, eval_set: str, arm: str, label_mode: Literal["true", "flip"] = "true",
    train_kind: str = "compound",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    assert primaries and secondaries
    for i, p in enumerate(primaries):
        s = secondaries[(i * 7 + 3) % len(secondaries)]
        if s.world_id == p.world_id:
            s = secondaries[(i * 7 + 4) % len(secondaries)]
        amap = alias_map_four(p, s, namespace)
        for ev_var in range(len(EVENT_TEMPLATES[event_group])):
            for st_var in range(len(STATE_TEMPLATES[state_group])):
                ctxt = (
                    "Match record: " + event_sentence(p, amap, event_group, ev_var) + " "
                    "Ranking record: " + state_sentence(p, amap, state_group, st_var) + " "
                    "Separate ranking record: " + state_sentence(s, amap, state_group, (st_var + 1))
                )
                tpl = f"event={event_group}:{ev_var}|state={state_group}:{st_var}"
                for hd in ["AB", "BA"]:
                    lab = event_label_for(p, hd)
                    if label_mode == "flip": lab = 1 - lab
                    rows.append(make_row(
                        row_id=f"{namespace}|{p.world_id}|{s.world_id}|{tpl}|event|{hd}",
                        context=ctxt, hypothesis=hyp_event(amap, p, hd), label=lab,
                        split=split, eval_set=eval_set, arm=arm, source=p.source, sport=p.sport,
                        query_family="event_role", primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=label_mode, conflict=not bool(p.winner_higher_at_time),
                        train_kind=train_kind,
                    ))
                    lab = state_label_for(p, hd)
                    if label_mode == "flip": lab = 1 - lab
                    rows.append(make_row(
                        row_id=f"{namespace}|{p.world_id}|{s.world_id}|{tpl}|focal_state|{hd}",
                        context=ctxt, hypothesis=hyp_state(amap, p, hd), label=lab,
                        split=split, eval_set=eval_set, arm=arm, source=p.source, sport=p.sport,
                        query_family="focal_state", primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=label_mode, conflict=not bool(p.winner_higher_at_time),
                        train_kind=train_kind,
                    ))
                    lab = state_label_for(s, hd)
                    if label_mode == "flip": lab = 1 - lab
                    rows.append(make_row(
                        row_id=f"{namespace}|{p.world_id}|{s.world_id}|{tpl}|untouched_state|{hd}",
                        context=ctxt, hypothesis=hyp_state(amap, s, hd), label=lab,
                        split=split, eval_set=eval_set, arm=arm, source=s.source, sport=s.sport,
                        query_family="untouched_state", primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode=label_mode, conflict=not bool(p.winner_higher_at_time),
                        train_kind=train_kind,
                    ))
    return rows


def exposure_rows(
    primaries: list[World], secondaries: list[World], *, event_group: str, state_group: str,
    namespace: str, split: str, eval_set: str, arm: str,
) -> list[dict[str, Any]]:
    """Mention-only rows with the same probe relation contexts and the same row count as compound_rows.

    Six mention hypotheses are produced per compound context, matching the six
    relation hypotheses (event AB/BA, focal state AB/BA, untouched state AB/BA).
    Three are true for aliases in the context and three are false for aliases absent
    from the context, so the arm is label-balanced like aligned/flipped relation rows.
    They expose context words and filler tokens without orienting win/rank relation labels.
    """
    rows: list[dict[str, Any]] = []
    assert primaries and secondaries
    for i, p in enumerate(primaries):
        s = secondaries[(i * 7 + 3) % len(secondaries)]
        if s.world_id == p.world_id:
            s = secondaries[(i * 7 + 4) % len(secondaries)]
        amap = alias_map_four(p, s, namespace)
        used = set(amap.values())
        x, y, z = other_aliases(used, f"{namespace}|{p.world_id}|{s.world_id}", 3)
        vals = {
            "A": amap[p.participant_a], "B": amap[p.participant_b],
            "C": amap[s.participant_a], "D": amap[s.participant_b], "X": x, "Y": y, "Z": z,
        }
        for ev_var in range(len(EVENT_TEMPLATES[event_group])):
            for st_var in range(len(STATE_TEMPLATES[state_group])):
                ctxt = (
                    "Match record: " + event_sentence(p, amap, event_group, ev_var) + " "
                    "Ranking record: " + state_sentence(p, amap, state_group, st_var) + " "
                    "Separate ranking record: " + state_sentence(s, amap, state_group, (st_var + 1))
                )
                tpl = f"event={event_group}:{ev_var}|state={state_group}:{st_var}"
                # 3 true mentions (A/B/C) and 3 false mentions (X/Y/Z), matching relation-row balance.
                mention_pairs = [
                    (f"The passage mentions {vals['A']}.", 1),
                    (f"The passage mentions {vals['B']}.", 1),
                    (f"The passage mentions {vals['C']}.", 1),
                    (f"The passage mentions {vals['X']}.", 0),
                    (f"The passage mentions {vals['Y']}.", 0),
                    (f"The passage mentions {vals['Z']}.", 0),
                ]
                for j, (hyp_text, label) in enumerate(mention_pairs):
                    rows.append(make_row(
                        row_id=f"{namespace}|{p.world_id}|{s.world_id}|{tpl}|mention|{j}",
                        context=ctxt, hypothesis=hyp_text, label=label,
                        split=split, eval_set=eval_set, arm=arm, source=p.source, sport=p.sport,
                        query_family="mention_exposure", primary_world=p.world_id, secondary_world=s.world_id,
                        template_group=tpl, label_mode="mention", conflict=not bool(p.winner_higher_at_time),
                        train_kind="sparse_exposure",
                    ))
    return rows


def event_only_rows(
    worlds: list[World], *, event_group: str, namespace: str, split: str, eval_set: str,
    arm: str, label_mode: Literal["true", "flip"] = "true", train_kind: str = "event_only",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in worlds:
        amap = alias_map_two(w, namespace)
        for ev_var in range(len(EVENT_TEMPLATES[event_group])):
            ctxt = "Match record: " + event_sentence(w, amap, event_group, ev_var)
            tpl = f"event={event_group}:{ev_var}"
            for hd in ["AB", "BA"]:
                lab = event_label_for(w, hd)
                if label_mode == "flip": lab = 1 - lab
                rows.append(make_row(
                    row_id=f"{namespace}|{w.world_id}|{tpl}|event|{hd}",
                    context=ctxt, hypothesis=hyp_event(amap, w, hd), label=lab,
                    split=split, eval_set=eval_set, arm=arm, source=w.source, sport=w.sport,
                    query_family="event_role", primary_world=w.world_id,
                    template_group=tpl, label_mode=label_mode, conflict=None, train_kind=train_kind,
                ))
    return rows


def paired_exposure_rows(worlds: list[World], *, event_group: str, namespace: str, split: str, eval_set: str, arm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in worlds:
        amap = alias_map_two(w, namespace)
        used = set(amap.values())
        x, y = other_aliases(used, f"paired_exp|{namespace}|{w.world_id}", 2)
        vals = {"A": amap[w.participant_a], "B": amap[w.participant_b], "C": x, "D": y, "X": x, "Y": y}
        for ev_var in range(len(EVENT_TEMPLATES[event_group])):
            ctxt = "Match record: " + event_sentence(w, amap, event_group, ev_var)
            # two rows per event variant match event-only relation row count and label balance.
            templates = [
                (f"The passage mentions {vals['A']}.", 1),
                (f"The passage mentions {vals['X']}.", 0),
            ]
            for j, (hyp, label) in enumerate(templates):
                rows.append(make_row(
                    row_id=f"{namespace}|{w.world_id}|event={event_group}:{ev_var}|mention|{j}",
                    context=ctxt, hypothesis=hyp, label=label,
                    split=split, eval_set=eval_set, arm=arm, source=w.source, sport=w.sport,
                    query_family="mention_exposure", primary_world=w.world_id,
                    template_group=f"event={event_group}:{ev_var}", label_mode="mention",
                    train_kind="sparse_exposure",
                ))
    return rows


def build_datasets(args: argparse.Namespace) -> dict[str, Any]:
    atp_train_all, atp_held_all = load_atp_worlds()
    pair_train_all, pair_held_all = load_pair_worlds()
    atp_train = select_worlds(atp_train_all, args.train_atp_worlds, 26101, require_state=True)
    atp_train_secondary = select_worlds(atp_train_all, args.train_atp_worlds + 17, 26102, require_state=True)
    atp_sparse = select_worlds(atp_train_all, args.sparse_atp_worlds, 26103, require_state=True)
    atp_sparse_secondary = select_worlds(atp_train_all, args.sparse_atp_worlds + 17, 26104, require_state=True)
    atp_eval = select_worlds(atp_held_all, args.eval_atp_worlds, 26105, require_state=True)
    atp_eval_secondary = select_worlds(atp_held_all, args.eval_atp_worlds + 17, 26106, require_state=True)

    pair_train = select_worlds(pair_train_all, args.train_pair_worlds, 26107)
    pair_sparse = select_worlds(pair_train_all, args.sparse_pair_worlds, 26108)
    pair_eval = select_worlds(pair_held_all, args.eval_pair_worlds, 26109)

    # Base rows: anchor surfaces only, same for all arms.
    base_rows = []
    base_rows += compound_rows(atp_train, atp_train_secondary, event_group="anchor", state_group="anchor",
                               namespace="base_atp_anchor", split="train", eval_set="train_base", arm="base",
                               label_mode="true", train_kind="base_anchor_compound")
    base_rows += event_only_rows(pair_train, event_group="anchor", namespace="base_pair_anchor", split="train",
                                 eval_set="train_base", arm="base", label_mode="true", train_kind="base_anchor_pair_event")

    arms: dict[str, dict[str, Any]] = {}
    for arm in ["exposure", "aligned", "flipped"]:
        train = list(base_rows)
        if arm == "exposure":
            train += exposure_rows(atp_sparse, atp_sparse_secondary, event_group="probe", state_group="probe",
                                   namespace="sparse_atp_probe", split="train", eval_set="train_sparse_exposure", arm=arm)
            train += paired_exposure_rows(pair_sparse, event_group="probe", namespace="sparse_pair_probe", split="train",
                                          eval_set="train_sparse_exposure", arm=arm)
        else:
            mode: Literal["true", "flip"] = "true" if arm == "aligned" else "flip"
            train += compound_rows(atp_sparse, atp_sparse_secondary, event_group="probe", state_group="probe",
                                   namespace="sparse_atp_probe", split="train", eval_set=f"train_sparse_{arm}", arm=arm,
                                   label_mode=mode, train_kind=f"sparse_{arm}_compound")
            train += event_only_rows(pair_sparse, event_group="probe", namespace="sparse_pair_probe", split="train",
                                     eval_set=f"train_sparse_{arm}", arm=arm, label_mode=mode,
                                     train_kind=f"sparse_{arm}_pair_event")
        evals = {
            "atp_anchor_heldfam": compound_rows(atp_eval, atp_eval_secondary, event_group="anchor", state_group="anchor",
                                                 namespace="eval_atp_anchor", split="held", eval_set="atp_anchor_heldfam", arm=arm),
            "atp_probe_heldfam": compound_rows(atp_eval, atp_eval_secondary, event_group="probe", state_group="probe",
                                                namespace="eval_atp_probe", split="held", eval_set="atp_probe_heldfam", arm=arm),
            "atp_heldword_heldfam": compound_rows(atp_eval, atp_eval_secondary, event_group="held", state_group="held",
                                                   namespace="eval_atp_held", split="held", eval_set="atp_heldword_heldfam", arm=arm),
            "atp_probe_trainfam": compound_rows(atp_train[: args.eval_train_atp_worlds], atp_train_secondary[: args.eval_train_atp_worlds + 17],
                                                 event_group="probe", state_group="probe", namespace="eval_atp_probe_trainfam",
                                                 split="train_eval", eval_set="atp_probe_trainfam", arm=arm),
            "pair_probe_heldfam": event_only_rows(pair_eval, event_group="probe", namespace="eval_pair_probe", split="held",
                                                   eval_set="pair_probe_heldfam", arm=arm),
            "pair_heldword_heldfam": event_only_rows(pair_eval, event_group="held", namespace="eval_pair_held", split="held",
                                                      eval_set="pair_heldword_heldfam", arm=arm),
            "pair_anchor_heldfam": event_only_rows(pair_eval, event_group="anchor", namespace="eval_pair_anchor", split="held",
                                                    eval_set="pair_anchor_heldfam", arm=arm),
        }
        arms[arm] = {"train": train, "evals": evals}

    all_rows_by_arm = {arm: rows["train"] + [r for ev in rows["evals"].values() for r in ev] for arm, rows in arms.items()}
    audits = audit_dataset(arms, atp_train, atp_eval, pair_train, pair_eval)
    return {"arms": arms, "all_rows_by_arm": all_rows_by_arm, "audit": audits}


def audit_dataset(arms: dict[str, dict[str, Any]], atp_train: list[World], atp_eval: list[World], pair_train: list[World], pair_eval: list[World]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "world_counts": {
            "atp_train_primary": len(atp_train), "atp_eval_primary": len(atp_eval),
            "pair_train": len(pair_train), "pair_eval": len(pair_eval),
            "atp_eval_conflict_worlds": sum(1 for w in atp_eval if w.winner_higher_at_time is False),
        },
        "arms": {},
        "alias_pool_size": len(ALIAS_POOL),
    }
    for arm, obj in arms.items():
        train = obj["train"]
        arm_audit: dict[str, Any] = {"train_rows": len(train), "train_label_balance": dict(collections.Counter(r["label"] for r in train))}
        arm_audit["train_by_kind"] = {k: len(v) for k, v in group_rows(train, "train_kind").items()}
        arm_audit["train_by_query_family"] = {k: label_balance(v) for k, v in group_rows(train, "query_family").items()}
        alias_counts = collections.Counter()
        for r in train:
            for a in ALIAS_POOL:
                if re.search(r"\\b" + re.escape(a) + r"\\b", r["text"]):
                    alias_counts[a] += 1
        arm_audit["alias_coverage"] = {"covered": sum(1 for a in ALIAS_POOL if alias_counts[a] > 0), "min": min(alias_counts.values()) if alias_counts else 0, "max": max(alias_counts.values()) if alias_counts else 0, "counts": dict(alias_counts)}
        ev_aud = {}
        for name, rows in obj["evals"].items():
            ev_aud[name] = {
                "rows": len(rows),
                "label_balance": label_balance(rows),
                "query_family_counts": {k: len(v) for k, v in group_rows(rows, "query_family").items()},
                "conflict_rows": sum(1 for r in rows if r.get("conflict") is True),
                "sport_counts": dict(collections.Counter(r.get("sport", "") for r in rows)),
            }
        arm_audit["evals"] = ev_aud
        out["arms"][arm] = arm_audit
    # alias-label independence scan: for train/eval text, whether first alias in hypothesis is over-associated with positive.
    for arm, obj in arms.items():
        vals = []
        for r in obj["train"]:
            m = re.match(r"([A-Za-z]+) ", r["hypothesis"])
            if m and r["query_family"] in {"event_role", "focal_state", "untouched_state"}:
                vals.append((m.group(1), int(r["label"])))
        by = collections.defaultdict(list)
        for a, y in vals:
            by[a].append(y)
        out["arms"][arm]["alias_positive_rate_extremes"] = sorted(
            [(a, len(v), float(np.mean(v))) for a, v in by.items() if len(v) >= 10], key=lambda x: abs(x[2] - 0.5), reverse=True
        )[:10]
    return out


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    d: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        d[str(r.get(key, ""))].append(r)
    return d


def label_balance(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = collections.Counter(int(r["label"]) for r in rows)
    return {str(k): int(v) for k, v in sorted(c.items())}


class DebertaEntailment(nn.Module):
    def __init__(self, model_path: str | Path, regime: str, init: str):
        super().__init__()
        if init == "pretrained":
            self.encoder = DebertaV2Model.from_pretrained(str(model_path))
        elif init == "random":
            cfg = DebertaV2Config.from_pretrained(str(model_path))
            self.encoder = DebertaV2Model(cfg)
        else:
            raise ValueError(init)
        hidden = int(self.encoder.config.hidden_size)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden // 2, 2))
        if regime == "frozen":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif regime == "top2":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # DeBERTaV2Model has encoder.layer as a ModuleList.
            for layer in self.encoder.encoder.layer[-2:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            # keep final rel embeddings/LayerNorm trainable if present? no, top layers suffice.
        elif regime == "full":
            for p in self.encoder.parameters():
                p.requires_grad_(True)
        else:
            raise ValueError(regime)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)


def encode_rows(tokenizer, rows: list[dict[str, Any]], max_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [r["text"] for r in rows]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), labels.to(device)


def acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def evaluate_model(model: nn.Module, tokenizer, rows: list[dict[str, Any]], max_len: int, device: str, batch_size: int) -> dict[str, Any]:
    model.eval()
    all_pred: list[int] = []
    all_lab: list[int] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            x, m, y = encode_rows(tokenizer, batch, max_len, device)
            logits = model(x, m)
            pred = logits.argmax(dim=-1).detach().cpu().tolist()
            all_pred.extend(pred)
            all_lab.extend(y.detach().cpu().tolist())
    arr_p = np.array(all_pred, dtype=np.int64)
    arr_y = np.array(all_lab, dtype=np.int64)
    out: dict[str, Any] = {"n": len(rows), "acc": float(np.mean(arr_p == arr_y)) if len(rows) else float("nan")}
    for key in ["query_family", "sport", "template_group", "conflict"]:
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for i, r in enumerate(rows):
            groups[str(r.get(key))].append(i)
        out[f"by_{key}"] = {g: float(np.mean(arr_p[idx] == arr_y[idx])) for g, idx in groups.items() if len(idx) > 0}
    # Key scientific subsets for compound ATP contexts.
    def filt(fn) -> float | None:
        idx = [i for i, r in enumerate(rows) if fn(r)]
        if not idx:
            return None
        return float(np.mean(arr_p[idx] == arr_y[idx]))
    out["key_subsets"] = {
        "event_role": filt(lambda r: r.get("query_family") == "event_role"),
        "focal_state": filt(lambda r: r.get("query_family") == "focal_state"),
        "focal_state_conflict": filt(lambda r: r.get("query_family") == "focal_state" and r.get("conflict") is True),
        "focal_state_nonconflict": filt(lambda r: r.get("query_family") == "focal_state" and r.get("conflict") is False),
        "untouched_state": filt(lambda r: r.get("query_family") == "untouched_state"),
        "untouched_state_conflict_context": filt(lambda r: r.get("query_family") == "untouched_state" and r.get("conflict") is True),
    }
    return out


def train_one(
    *, model_path: Path, tokenizer, arm: str, regime: str, init: str, seed: int, train_rows: list[dict[str, Any]], evals: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace, device: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = DebertaEntailment(model_path, regime=regime, init=init).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    head_params = [p for n, p in model.named_parameters() if n.startswith("head") and p.requires_grad]
    enc_params = [p for n, p in model.named_parameters() if (not n.startswith("head")) and p.requires_grad]
    if enc_params:
        opt = torch.optim.AdamW([
            {"params": head_params, "lr": args.head_lr},
            {"params": enc_params, "lr": args.encoder_lr},
        ], weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(head_params, lr=args.head_lr, weight_decay=args.weight_decay)
    epochs = args.epochs_frozen if regime == "frozen" else args.epochs_tune

    x, m, y = encode_rows(tokenizer, train_rows, args.max_len, device)
    ds = TensorDataset(x, m, y)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    history = []
    best_state = None
    best_train = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, mb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb, mb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        model.eval()
        with torch.no_grad():
            pred_logits = []
            for start in range(0, len(train_rows), args.eval_batch_size):
                xb = x[start:start + args.eval_batch_size]
                mb = m[start:start + args.eval_batch_size]
                pred_logits.append(model(xb, mb).detach())
            logits_all = torch.cat(pred_logits, dim=0)
            train_acc = acc_from_logits(logits_all, y)
        history.append({"epoch": ep, "loss": float(np.mean(losses)) if losses else None, "train_acc": train_acc})
        if train_acc > best_train:
            best_train = train_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    eval_out = {name: evaluate_model(model, tokenizer, rows, args.max_len, device, args.eval_batch_size) for name, rows in evals.items()}
    train_eval = evaluate_model(model, tokenizer, train_rows, args.max_len, device, args.eval_batch_size)
    return {
        "arm": arm, "regime": regime, "init": init, "seed": seed, "epochs": epochs,
        "trainable_params": int(trainable), "total_params": int(total),
        "history": history, "best_train_acc": float(best_train), "train_eval": train_eval, "evals": eval_out,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in results:
        grouped[(r["init"], r["regime"], r["arm"])].append(r)
    agg: dict[str, Any] = {}
    for (init, regime, arm), rs in grouped.items():
        key = f"{init}/{regime}/{arm}"
        item: dict[str, Any] = {
            "n_seeds": len(rs),
            "best_train_acc_mean": float(np.mean([r["best_train_acc"] for r in rs])),
            "best_train_acc_min": float(np.min([r["best_train_acc"] for r in rs])),
            "evals": {},
        }
        eval_names = sorted(rs[0]["evals"].keys())
        for en in eval_names:
            mets = [r["evals"][en] for r in rs]
            item["evals"][en] = {
                "acc_mean": mean_field(mets, ["acc"]),
                "acc_std": std_field(mets, ["acc"]),
                "key_subsets_mean": {},
                "key_subsets_std": {},
                "by_query_family_mean": {},
                "by_sport_mean": {},
            }
            subset_keys = sorted(set(k for m in mets for k in m.get("key_subsets", {}).keys()))
            for sk in subset_keys:
                vals = [m.get("key_subsets", {}).get(sk) for m in mets]
                vals2 = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
                item["evals"][en]["key_subsets_mean"][sk] = float(np.mean(vals2)) if vals2 else None
                item["evals"][en]["key_subsets_std"][sk] = float(np.std(vals2)) if vals2 else None
            for qk in sorted(set(k for m in mets for k in m.get("by_query_family", {}).keys())):
                vals = [m.get("by_query_family", {}).get(qk) for m in mets]
                vals2 = [v for v in vals if v is not None]
                item["evals"][en]["by_query_family_mean"][qk] = float(np.mean(vals2)) if vals2 else None
            for sk in sorted(set(k for m in mets for k in m.get("by_sport", {}).keys())):
                vals = [m.get("by_sport", {}).get(sk) for m in mets]
                vals2 = [v for v in vals if v is not None]
                item["evals"][en]["by_sport_mean"][sk] = float(np.mean(vals2)) if vals2 else None
        agg[key] = item
    # contrasts within each regime.
    contrasts: dict[str, Any] = {}
    for init in sorted(set(r["init"] for r in results)):
        for regime in sorted(set(r["regime"] for r in results if r["init"] == init)):
            contrast_key = f"{init}/{regime}"
            if not all(f"{init}/{regime}/{a}" in agg for a in ["aligned", "flipped", "exposure"]):
                continue
            contrasts[contrast_key] = {}
            for eval_name in agg[f"{init}/{regime}/aligned"]["evals"].keys():
                contrasts[contrast_key][eval_name] = {}
                for metric in ["event_role", "focal_state", "focal_state_conflict", "untouched_state", "untouched_state_conflict_context"]:
                    av = agg[f"{init}/{regime}/aligned"]["evals"][eval_name]["key_subsets_mean"].get(metric)
                    fv = agg[f"{init}/{regime}/flipped"]["evals"][eval_name]["key_subsets_mean"].get(metric)
                    ev = agg[f"{init}/{regime}/exposure"]["evals"][eval_name]["key_subsets_mean"].get(metric)
                    if av is not None and fv is not None:
                        contrasts[contrast_key][eval_name][f"aligned_minus_flipped_{metric}"] = av - fv
                    if av is not None and ev is not None:
                        contrasts[contrast_key][eval_name][f"aligned_minus_exposure_{metric}"] = av - ev
                    if fv is not None and ev is not None:
                        contrasts[contrast_key][eval_name][f"flipped_minus_exposure_{metric}"] = fv - ev
    return {"by_regime_arm": agg, "contrasts": contrasts}


def mean_field(items: list[dict[str, Any]], path: list[str]) -> float | None:
    vals = []
    for it in items:
        cur: Any = it
        for p in path:
            cur = cur[p]
        if cur is not None:
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else None


def std_field(items: list[dict[str, Any]], path: list[str]) -> float | None:
    vals = []
    for it in items:
        cur: Any = it
        for p in path:
            cur = cur[p]
        if cur is not None:
            vals.append(float(cur))
    return float(np.std(vals)) if vals else None


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines += ["# research pretrained ATP slot-orientation probe", ""]
    lines += [summary["scientific_purpose"], ""]
    lines += [f"Model: `{summary['model_path']}`", f"Device: `{summary['device']}`, elapsed {summary['elapsed_seconds']:.2f}s", ""]
    aud = summary["audit"]
    lines += ["## Data construction", "", "```json", json.dumps(aud["world_counts"], indent=2), "```", ""]
    for arm, a in aud["arms"].items():
        lines += [f"### Arm `{arm}`", "", f"Train rows: {a['train_rows']}; label balance {a['train_label_balance']}", ""]
        lines += ["Train row kinds:", "", "```json", json.dumps(a["train_by_kind"], indent=2), "```", ""]
    agg = summary["aggregate"]["by_regime_arm"]
    key_evals = ["atp_probe_heldfam", "atp_heldword_heldfam", "pair_probe_heldfam", "pair_heldword_heldfam"]
    lines += ["## Mean accuracy over seeds", ""]
    for eval_name in key_evals:
        lines += [f"### {eval_name}", "", "| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for key in sorted(agg.keys()):
            item = agg[key]
            ev = item["evals"].get(eval_name)
            if not ev:
                continue
            ks = ev["key_subsets_mean"]
            sports = ev.get("by_sport_mean", {})
            sport_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(sports.items())[:5] if v is not None)
            lines.append(
                f"| {key} | {item['best_train_acc_mean']:.3f} | {fmt(ev['acc_mean'])} | {fmt(ks.get('event_role'))} | "
                f"{fmt(ks.get('focal_state'))} | {fmt(ks.get('focal_state_conflict'))} | {fmt(ks.get('untouched_state'))} | "
                f"{fmt(ks.get('untouched_state_conflict_context'))} | {sport_str} |"
            )
        lines.append("")
    lines += ["## Orientation contrasts", ""]
    for regime, evs in summary["aggregate"]["contrasts"].items():
        lines += [f"### {regime}", "", "| eval | metric | value |", "|---|---|---:|"]
        for en in key_evals:
            d = evs.get(en, {})
            for k, v in sorted(d.items()):
                if any(x in k for x in ["aligned_minus_flipped", "aligned_minus_exposure"]):
                    lines.append(f"| {en} | {k} | {v:.3f} |")
        lines.append("")
    lines += ["## Scientific reading", "", summary["scientific_reading"], "", f"JSON: `{summary['summary_json']}`"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def make_scientific_reading(aggregate: dict[str, Any]) -> str:
    lines: list[str] = []
    contrasts = aggregate.get("contrasts", {})
    for regime in sorted(contrasts):
        d = contrasts[regime].get("atp_probe_heldfam", {})
        af_event = d.get("aligned_minus_flipped_event_role")
        af_state = d.get("aligned_minus_flipped_focal_state_conflict")
        af_unt = d.get("aligned_minus_flipped_untouched_state")
        ae_event = d.get("aligned_minus_exposure_event_role")
        ae_state = d.get("aligned_minus_exposure_focal_state_conflict")
        if af_event is not None:
            lines.append(
                f"In `{regime}` on ATP probe held-family contexts, aligned-minus-flipped was {af_event:+.3f} for event role, "
                f"{af_state:+.3f} for conflict focal-state rows, and {af_unt:+.3f} for untouched-state rows. "
                f"Aligned-minus-exposure was {ae_event:+.3f} for event and {ae_state:+.3f} for conflict focal state."
            )
    if not lines:
        return "No complete aligned/flipped/exposure contrast was available."
    lines.append(
        "The experiment is a small classifier/probe over an existing BabyLM DeBERTa checkpoint, not evidence that a Strict-Small LM trajectory will improve. "
        "A useful positive signal would be simultaneous aligned-over-flipped movement for event_role, conflict focal_state, and untouched_state under held wording/domain; failure or poor train fit means the present bridge is still too weak for a full corpus intervention."
    )
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--regimes", nargs="+", default=["frozen", "top2"], choices=["frozen", "top2", "full"])
    ap.add_argument("--inits", nargs="+", default=["pretrained"], choices=["pretrained", "random"])
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--train_atp_worlds", type=int, default=220)
    ap.add_argument("--sparse_atp_worlds", type=int, default=24)
    ap.add_argument("--eval_atp_worlds", type=int, default=160)
    ap.add_argument("--eval_train_atp_worlds", type=int, default=96)
    ap.add_argument("--train_pair_worlds", type=int, default=180)
    ap.add_argument("--sparse_pair_worlds", type=int, default=24)
    ap.add_argument("--eval_pair_worlds", type=int, default=160)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=128)
    ap.add_argument("--epochs_frozen", type=int, default=10)
    ap.add_argument("--epochs_tune", type=int, default=4)
    ap.add_argument("--head_lr", type=float, default=2e-3)
    ap.add_argument("--encoder_lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "cuda":
        torch.cuda.set_device(0)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.set_num_threads(min(32, max(1, torch.get_num_threads())))

    model_path = Path(args.model_path)
    print(json.dumps({"status": "PRETRAINED_ATP_SLOT_ORIENTATION_START", "model": str(model_path), "device": args.device, "regimes": args.regimes, "inits": args.inits, "n_seeds": args.n_seeds}), flush=True)
    data = build_datasets(args)
    # Save exact row material for independent audits and later reuse.
    for arm, obj in data["arms"].items():
        write_jsonl(out_dir / f"rows_train_{arm}.jsonl", obj["train"])
        for name, rows in obj["evals"].items():
            if arm == "aligned":  # eval rows are identical across arms except metadata; avoid redundant large output.
                write_jsonl(out_dir / f"rows_eval_{name}.jsonl", rows)
    write_json(out_dir / "construction_audit.json", data["audit"])

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    results: list[dict[str, Any]] = []
    seed_values = [261000 + i * 97 for i in range(args.n_seeds)]
    for init in args.inits:
        for regime in args.regimes:
            if init == "random" and regime != "full":
                continue
            for arm in ["exposure", "aligned", "flipped"]:
                for seed in seed_values:
                    print(json.dumps({"event": "train_start", "init": init, "regime": regime, "arm": arm, "seed": seed, "rows": len(data["arms"][arm]["train"])}), flush=True)
                    res = train_one(model_path=model_path, tokenizer=tokenizer, arm=arm, regime=regime, init=init, seed=seed,
                                    train_rows=data["arms"][arm]["train"], evals=data["arms"][arm]["evals"], args=args, device=args.device)
                    results.append(res)
                    key = res["evals"]["atp_probe_heldfam"]["key_subsets"]
                    print(json.dumps({"event": "train_done", "init": init, "regime": regime, "arm": arm, "seed": seed,
                                      "best_train": res["best_train_acc"], "atp_probe": key}, ensure_ascii=False), flush=True)
                    # Release memory before next arm.
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
    aggregate = aggregate_results(results)
    summary = {
        "status": "PRETRAINED_ATP_SLOT_ORIENTATION_PROBE",
        "created_utc": now(),
        "model_path": str(model_path),
        "device": args.device,
        "args": vars(args),
        "scientific_purpose": "Small pretrained-language bridge test for reusable filler interface plus sparse relation-coordinate orientation on independently attested ATP event/state and research paired-world material.",
        "audit": data["audit"],
        "results": results,
        "aggregate": aggregate,
        "scientific_reading": make_scientific_reading(aggregate),
        "elapsed_seconds": time.time() - t0,
        "summary_json": str(out_dir / "pretrained_atp_slot_orientation_summary.json"),
    }
    write_json(out_dir / "pretrained_atp_slot_orientation_summary.json", summary)
    write_markdown(out_dir / "pretrained_atp_slot_orientation_summary.md", summary)
    print(json.dumps({
        "status": summary["status"],
        "elapsed_seconds": round(summary["elapsed_seconds"], 2),
        "summary_json": summary["summary_json"],
        "key_contrasts": aggregate.get("contrasts", {}),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
