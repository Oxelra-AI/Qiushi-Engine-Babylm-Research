#!/usr/bin/env python3
"""Step234b: novelty x relational interaction on the completed 20M target-selective losses.

Tests whether the drop_abs - drop_copied_word gap requires BOTH source-novelty AND
relational/event semantics, using the research event_losses (source_absent vs retained
copied content, rel/event vs other). Cluster (pair_id) bootstrap.
"""
from __future__ import annotations
import collections
import json
import pathlib
import random
import sys

sys.path.insert(0, "experiments/archive/representation_and_objectives/scripts")
from source_absent_lexical_profile import flags as lex  # noqa: E402
from annotate_packed_pool import norm_word  # noqa: E402

PATH = pathlib.Path("experiments/archive/representation_and_objectives/data/wholeword_control_readout/event_losses.jsonl")
OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification/novelty_x_relational_interaction.json")


def relev(w: str) -> bool:
    fs = set(lex(w, norm_word(w)))
    return bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)


def pw(cl):
    s = sum(d for evs in cl.values() for d, _ in evs)
    p = sum(pp for evs in cl.values() for _, pp in evs)
    return (s / p) if p else None, p


def boot(cl, rng, nb=3000):
    items = list(cl.items())
    if len(items) < 2:
        return None, None
    out = []
    for _ in range(nb):
        s = 0.0
        p = 0
        for _ in items:
            _, evs = items[rng.randrange(len(items))]
            for d, pp in evs:
                s += d
                p += pp
        if p:
            out.append(s / p)
    out.sort()
    return round(out[int(0.025 * len(out))], 6), round(out[int(0.975 * len(out))], 6)


def main():
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in PATH.open():
        o = json.loads(line)
        ck = o.get("chck_20M", {})
        if "drop_abs" not in ck or "drop_copied_word" not in ck:
            continue
        da, dc = ck["drop_abs"], ck["drop_copied_word"]
        pc = da.get("pieces", 0)
        if pc <= 0 or o["category"] not in ("source_absent_content", "retained_content"):
            continue
        key = (o["category"], "rel_event" if relev(o.get("word", "")) else "other")
        buckets[key][o["pair_id"]].append((da["loss_sum"] - dc["loss_sum"], pc))
    rng = random.Random(7)
    cells = {}
    for k in [("source_absent_content", "rel_event"), ("source_absent_content", "other"),
              ("retained_content", "rel_event"), ("retained_content", "other")]:
        delta, pieces = pw(buckets[k])
        lo, hi = boot(buckets[k], rng)
        cells["|".join(k)] = {"piece_weighted_delta": None if delta is None else round(delta, 6),
                              "pieces": pieces, "boot95": [lo, hi]}
    sa_rel = cells["source_absent_content|rel_event"]["piece_weighted_delta"]
    sa_oth = cells["source_absent_content|other"]["piece_weighted_delta"]
    rc_rel = cells["retained_content|rel_event"]["piece_weighted_delta"]
    rc_oth = cells["retained_content|other"]["piece_weighted_delta"]
    result = {
        "status": "STEP234B_NOVELTY_X_RELATIONAL_INTERACTION",
        "contrast": "drop_abs_minus_drop_copied_word at chck_20M",
        "cells": cells,
        "interaction_I": round((sa_rel - sa_oth) - (rc_rel - rc_oth), 6),
        "reading": "Large positive I means the drop_abs gap needs BOTH source-novelty and relational/event semantics: copied relational/event content shows a near-zero gap, novel relational/event content shows a large gap.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
