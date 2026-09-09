#!/usr/bin/env python3
"""Step008b: non-overlap rewrite source-specific contrast.

This script reuses the research matched-support experiment but changes the
attribute vocabulary so the content/rewrite target token is disjoint from the
source token. The previous "content" probe used the same attribute token under a
different verb/template, so exact repetition could still help by copying the
source attribute. relation_learning's defining probe is tokenizer-nonoverlap rewrite: the
target token is not present in the source. This variant is the minimal synthetic
analogue.

Source concept i is token s{i}; rewrite concept i is token r{i}. Source events
in the context/base slots use s{i}. REPEAT trains matched final events as the
same s{i}; VARIED trains matched final events as the paired r{i}; COPY eval
asks for s{i}; CONTENT/REWRITE eval asks for r{i}. Other support, target/source
positions, match/no-match balance, and full-vs-masked attention interventions are
identical to research.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


BASE_PATH = _public_path('experiments/archive/functional_learning/scripts/revision_008b_nonoverlap_source_contrast.py').with_name("matched_support_source_contrast.py")
spec = importlib.util.spec_from_file_location("step008base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


# Patch the vocabulary before any Vocab object is created.
N_CONCEPT = 32
SRC_ATTRS = [f"s{i}" for i in range(N_CONCEPT)]
REW_ATTRS = [f"r{i}" for i in range(N_CONCEPT)]
SRC_TO_REW = {s: r for s, r in zip(SRC_ATTRS, REW_ATTRS)}
REW_TO_SRC = {r: s for s, r in SRC_TO_REW.items()}
base.N_ATTR = 2 * N_CONCEPT
base.ATTRIBUTES = SRC_ATTRS + REW_ATTRS
base.ALL_TOKENS = base.SPECIALS + base.VERBS + ["."] + base.ENTITIES + base.ATTRIBUTES


def concept_index(src_attr: str) -> int:
    assert src_attr.startswith("s")
    return int(src_attr[1:])


def random_src_attr(rng: random.Random) -> str:
    return rng.choice(SRC_ATTRS)


def random_any_attr(rng: random.Random) -> str:
    return rng.choice(base.ATTRIBUTES)


def _pick_unused(rng: random.Random, used: set[str]) -> str:
    choices = [e for e in base.ENTITIES if e not in used]
    e = rng.choice(choices)
    used.add(e)
    return e


BlockTuple = Tuple[int, int, int, int]


def gen_sequence_nonoverlap(condition: str, vocab: base.Vocab, rng: random.Random):
    """Generate one train sequence with disjoint source/rewrite attributes."""
    fam = base.condition_family(condition)
    used: set[str] = set()
    base_events: List[Tuple[str, str, str]] = []
    for _ in range(base.BASE_N):
        # Source contexts always use source-side tokens s_i.
        base_events.append((_pick_unused(rng, used), rng.choice(base.VERBS), random_src_attr(rng)))

    extras: List[Optional[Tuple[str, str, str]]] = [None, None]
    meta: Dict[str, Optional[int | str]] = {
        "family": fam,
        "matched_extra_local": None,
        "unmatched_extra_local": None,
        "source_slot": None,
        "target_slot": None,
    }

    if fam == "unique":
        for j in range(base.EXTRA_N):
            extras[j] = (_pick_unused(rng, used), rng.choice(base.VERBS), random_any_attr(rng))
    else:
        matched_local = rng.randrange(base.EXTRA_N)
        unmatched_local = 1 - matched_local
        source_slot = rng.randrange(base.BASE_N)
        e, v, s_attr = base_events[source_slot]
        r_attr = SRC_TO_REW[s_attr]

        if fam == "repeat":
            matched = (e, v, s_attr)  # exact in-window recurrence: source token repeats
        elif fam == "varied":
            matched = (e, base.flip(v), r_attr)  # non-overlap restatement/rewrite
        elif fam == "wrong":
            # Same entity and template, wrong source-side concept token.
            wrong_s = rng.choice([x for x in SRC_ATTRS if x != s_attr])
            matched = (e, v, wrong_s)
        elif fam == "support_control":
            # Same entity-match and source/target support, but independent target attribute.
            matched = (e, rng.choice(base.VERBS), random_any_attr(rng))
        else:
            raise AssertionError(fam)

        extras[matched_local] = matched
        extras[unmatched_local] = (_pick_unused(rng, used), rng.choice(base.VERBS), random_any_attr(rng))
        meta.update({
            "matched_extra_local": matched_local,
            "unmatched_extra_local": unmatched_local,
            "source_slot": source_slot,
            "target_slot": base.BASE_N + matched_local,
        })

    toks = ["<bos>"]
    for e, v, a in base_events:
        toks.extend(base.ev_tok(e, v, a))
    for ex in extras:
        assert ex is not None
        toks.extend(base.ev_tok(*ex))
    toks.append("<eos>")
    assert len(toks) == base.SEQ_LEN

    blocks: List[BlockTuple] = []
    if base.is_masked_condition(condition) and fam in ("repeat", "varied", "wrong"):
        assert meta["source_slot"] is not None and meta["target_slot"] is not None
        ss = int(meta["source_slot"])
        ts = int(meta["target_slot"])
        blocks.append((base.ev_start(ss), base.ev_end(ss), base.ev_start(ts), base.ev_end(ts)))

    return vocab.encode(toks), blocks, meta


def filler_event_nonoverlap(rng: random.Random, excluded: set[str]) -> List[str]:
    e = _pick_unused(rng, excluded)
    # Expose both s and r tokens in filler/final support.
    return base.ev_tok(e, rng.choice(base.VERBS), random_any_attr(rng))


def build_probe_sequence_nonoverlap(
    *,
    rng: random.Random,
    ent: str,
    target_attr: str,
    source_slot: int,
    target_slot: int,
    source_event: List[str],
    target_verb: str,
) -> List[str]:
    excluded = {ent}
    if source_event[0] in base.ENTITIES:
        excluded.add(source_event[0])
    slots: List[Optional[List[str]]] = [None for _ in range(base.EV_TOTAL)]
    slots[source_slot] = source_event
    slots[target_slot] = base.ev_tok(ent, target_verb, target_attr)
    for s in range(base.EV_TOTAL):
        if slots[s] is None:
            slots[s] = filler_event_nonoverlap(rng, excluded)
    toks = ["<bos>"]
    for s in range(base.EV_TOTAL):
        toks.extend(slots[s])
    toks.append("<eos>")
    assert len(toks) == base.SEQ_LEN
    return toks


def gen_eval_probes_nonoverlap(n: int, vocab: base.Vocab, rng: random.Random):
    probes = []
    used_pairs: set[Tuple[str, str]] = set()
    for _ in range(n):
        while True:
            ent = rng.choice(base.ENTITIES)
            s_attr = random_src_attr(rng)
            if (ent, s_attr) not in used_pairs:
                used_pairs.add((ent, s_attr))
                break
        r_attr = SRC_TO_REW[s_attr]
        source_slot = rng.randrange(base.BASE_N)
        target_slot = base.BASE_N + rng.randrange(base.EXTRA_N)
        wrong_s = rng.choice([x for x in SRC_ATTRS if x != s_attr])
        other_ent = rng.choice([e for e in base.ENTITIES if e != ent])
        other_s = random_src_attr(rng)

        true_source = base.ev_tok(ent, "has", s_attr)
        unrelated_source = base.ev_tok(other_ent, rng.choice(base.VERBS), other_s)
        wrong_match_source = base.ev_tok(ent, "has", wrong_s)

        state = rng.getstate()
        seqs: Dict[str, List[int]] = {}
        # Copy target token overlaps with source (s_i). Content/rewrite target token
        # is disjoint (r_i), matching the tokenizer-nonoverlap BabyLM probe.
        specs = (
            ("copy", "has", s_attr),
            ("content", "is", r_attr),
        )
        for relation, target_verb, target_attr in specs:
            for source_kind, src_ev in (
                ("T", true_source),
                ("U_nomatch", unrelated_source),
                ("U_wrongmatch", wrong_match_source),
            ):
                rng.setstate(state)
                toks = build_probe_sequence_nonoverlap(
                    rng=rng,
                    ent=ent,
                    target_attr=target_attr,
                    source_slot=source_slot,
                    target_slot=target_slot,
                    source_event=src_ev,
                    target_verb=target_verb,
                )
                seqs[f"{relation}_{source_kind}"] = vocab.encode(toks)
        rng.random()
        probes.append({
            "entity": ent,
            "src_attr": s_attr,
            "rewrite_attr": r_attr,
            "source_slot": source_slot,
            "target_slot": target_slot,
            "target_pos": base.attr_pos(target_slot),
            "copy_target_id": vocab.stoi[s_attr],
            "content_target_id": vocab.stoi[r_attr],
            **seqs,
        })
    return probes


@torch.no_grad()
def eval_probes_nonoverlap(model, probes, vocab: base.Vocab, dev: torch.device):
    model.eval()
    cmask = base.causal_mask(base.SEQ_LEN, dev)
    keys = [
        "copy_T", "copy_U_nomatch", "copy_U_wrongmatch",
        "content_T", "content_U_nomatch", "content_U_wrongmatch",
    ]
    all_seqs = []
    for p in probes:
        for k in keys:
            all_seqs.append(p[k])
    batch = torch.tensor(all_seqs, dtype=torch.long, device=dev)
    all_logits = []
    for s in range(0, len(batch), 256):
        all_logits.append(model(batch[s:s + 256], mask=cmask))
    logits = torch.cat(all_logits, dim=0)
    vals = {k: [] for k in keys}
    for idx, p in enumerate(probes):
        base_idx = idx * len(keys)
        tpos = p["target_pos"]
        # Relation-specific target IDs are the only difference from research eval.
        tids = {
            "copy_T": p["copy_target_id"],
            "copy_U_nomatch": p["copy_target_id"],
            "copy_U_wrongmatch": p["copy_target_id"],
            "content_T": p["content_target_id"],
            "content_U_nomatch": p["content_target_id"],
            "content_U_wrongmatch": p["content_target_id"],
        }
        lp = F.log_softmax(logits[base_idx:base_idx + len(keys), tpos - 1], dim=-1)
        for j, k in enumerate(keys):
            vals[k].append(-float(lp[j, tids[k]].item()))
    out = {f"{k}_nll": round(sum(v) / len(v), 5) for k, v in vals.items()}
    out["copy_gain_nomatch"] = round(out["copy_U_nomatch_nll"] - out["copy_T_nll"], 5)
    out["content_gain_nomatch"] = round(out["content_U_nomatch_nll"] - out["content_T_nll"], 5)
    out["copy_gain_wrongmatch"] = round(out["copy_U_wrongmatch_nll"] - out["copy_T_nll"], 5)
    out["content_gain_wrongmatch"] = round(out["content_U_wrongmatch_nll"] - out["content_T_nll"], 5)
    return out


# Monkey-patch the research base module.
base.gen_sequence = gen_sequence_nonoverlap
base.gen_eval_probes = gen_eval_probes_nonoverlap
base.eval_probes_fn = eval_probes_nonoverlap


if __name__ == "__main__":
    base.main()
