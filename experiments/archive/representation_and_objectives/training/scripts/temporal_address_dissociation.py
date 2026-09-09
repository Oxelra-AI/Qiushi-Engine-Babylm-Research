#!/usr/bin/env python3
"""research: dissociate temporal address mechanisms in the full temporal bridge.

The research section wrapper showed that Background/Update labels can rescue the
full mixed stable+changed temporal bridge.  This script asks what mechanism that
success uses.  It deliberately keeps the research source-attested world selection,
labels, stable base, sparse changed/stable focal rows, and paired AB-vs-BA
scoring, but changes the address interface:

1. section_rolepara: context sections carry Background/Update roles, all four
   records are deterministically shuffled, and held queries use prior/revised
   paraphrases that do not repeat the exact words "background" or "update".
   This attacks fixed position and exact section-key matching while preserving a
   temporal/discourse role.

2. arbitrary_tag: every focal/secondary before/after record receives arbitrary
   per-example tags sampled from ordinary words; the four records are shuffled;
   hypotheses address a record by its tag.  Tags have no global before/after
   meaning across examples, so success is an address/key-value ceiling rather
   than a temporal-semantics result.  A swapped-tag readout keeps labels fixed but
   queries the opposite focal tag; changed focal accuracy should collapse if the
   model truly follows the tag address.

Scientific boundary: all runs are small pretrained-bridge probes, not BabyLM-scale
training or final evidence of a general data-efficient principle.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
BASE_PATH = STUDY / "training/scripts/temporal_change_bridge_probe.py"
spec = importlib.util.spec_from_file_location("temporal_base_for_dissociation", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

DEFAULT_OUT = STUDY / "data/temporal_address_dissociation"
DEFAULT_MODEL = base.DEFAULT_MODEL

TAG_POOL = [
    "amber", "basil", "cedar", "dawn", "ember", "frost", "garden", "harbor", "iris", "juniper",
    "kernel", "lantern", "meadow", "nectar", "olive", "pebble", "quartz", "river", "saffron", "timber",
    "umber", "velvet", "willow", "xenia", "yarrow", "zephyr", "atlas", "beacon", "cobalt", "drift",
    "elm", "field", "grove", "hazel", "ivory", "jasper", "kiwi", "lagoon", "marble", "nova",
    "onyx", "prairie", "raven", "silver", "topaz", "violet", "walnut", "yonder", "zenith", "apricot",
]


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def tag4(key: str) -> list[str]:
    return random.Random(stable_int("tag4|" + key)).sample(TAG_POOL, 4)


def sent_plain_initial(w, am, variant: int) -> str:
    tmpls = [
        "{H} was ranked above {Lo} in the ATP list.",
        "{Lo} was ranked below {H} in the ATP list.",
    ]
    return tmpls[variant % len(tmpls)].format(H=base.an(am, w.higher_initial), Lo=base.an(am, w.lower_initial))


def sent_plain_later(w, am, variant: int) -> str:
    tmpls = [
        "{H} was ranked above {Lo} in the ATP list.",
        "{Lo} was ranked below {H} in the ATP list.",
    ]
    return tmpls[variant % len(tmpls)].format(H=base.an(am, w.higher_later), Lo=base.an(am, w.lower_later))


def shuffle_parts(parts: list[str], key: str, enabled: bool = True) -> list[str]:
    parts = list(parts)
    if enabled:
        random.Random(stable_int("shuffle|" + key)).shuffle(parts)
    return parts


def hyp_xy(w, am, hd: str) -> tuple[str, str]:
    x = base.an(am, w.participant_a if hd == "AB" else w.participant_b)
    y = base.an(am, w.participant_b if hd == "AB" else w.participant_a)
    return x, y


def labels_for(qf: str, f, s, hd: str) -> int:
    if qf == "focal_before":
        return base.initial_label(f, hd)
    if qf == "focal_after":
        return base.later_label(f, hd)
    if qf == "secondary_before":
        return base.initial_label(s, hd)
    if qf == "secondary_after":
        return base.later_label(s, hd)
    raise ValueError(qf)


def build_context_and_hyp(*, mode: str, f, s, am: dict[str, str], qf: str, hd: str,
                          vi: int, vl: int, ns: str, hyp_style: str,
                          shuffle_records: bool = True, swap_focal_tags: bool = False) -> tuple[str, str, str]:
    """Return context, hypothesis, template descriptor."""
    if mode == "section_rolepara":
        parts = [
            "Focal background record: " + sent_plain_initial(f, am, vi),
            "Focal update record: " + sent_plain_later(f, am, vl),
            "Separate background record: " + sent_plain_initial(s, am, vi + 1),
            "Separate update record: " + sent_plain_later(s, am, vl + 1),
        ]
        parts = shuffle_parts(parts, f"{mode}|{ns}|{f.world_id}|{s.world_id}|{vi}|{vl}", shuffle_records)
        context = " ".join(parts)
        qw = f if qf.startswith("focal") else s
        x, y = hyp_xy(qw, am, hd)
        when = "before" if qf.endswith("before") else "after"
        if hyp_style == "train":
            # Exact keys during training, like the supplied-address ceiling.
            if when == "before":
                hyp = f"Based on the background, {x} was ranked higher than {y}."
            else:
                hyp = f"Based on the update, {x} was ranked higher than {y}."
        else:
            # No exact recurrence of background/update.
            if when == "before":
                hyp = f"Using the prior ranking record, {x} outranked {y}."
            else:
                hyp = f"Using the revised ranking record, {x} outranked {y}."
        tpl = f"mode={mode}|hyp={hyp_style}|shuffle={shuffle_records}|vi={vi}|vl={vl}"
        return context, hyp, tpl

    if mode == "arbitrary_tag":
        tags = tag4(f"{ns}|{f.world_id}|{s.world_id}|{vi}|{vl}")
        # Tags are assigned per context; no tag has a stable before/after meaning across examples.
        tf_b, tf_a, ts_b, ts_a = tags
        parts = [
            f"Entry {tf_b}: " + sent_plain_initial(f, am, vi),
            f"Entry {tf_a}: " + sent_plain_later(f, am, vl),
            f"Entry {ts_b}: " + sent_plain_initial(s, am, vi + 1),
            f"Entry {ts_a}: " + sent_plain_later(s, am, vl + 1),
        ]
        parts = shuffle_parts(parts, f"{mode}|{ns}|{f.world_id}|{s.world_id}|{vi}|{vl}", shuffle_records)
        context = " ".join(parts)
        qw = f if qf.startswith("focal") else s
        x, y = hyp_xy(qw, am, hd)
        tag_map = {
            "focal_before": tf_b,
            "focal_after": tf_a,
            "secondary_before": ts_b,
            "secondary_after": ts_a,
        }
        if swap_focal_tags and qf == "focal_before":
            tag = tf_a
        elif swap_focal_tags and qf == "focal_after":
            tag = tf_b
        else:
            tag = tag_map[qf]
        if hyp_style == "train":
            hyp = f"According to entry {tag}, {x} was ranked higher than {y}."
        else:
            hyp = f"Using the record named {tag}, {x} outranked {y}."
        tpl = f"mode={mode}|hyp={hyp_style}|shuffle={shuffle_records}|swap={swap_focal_tags}|vi={vi}|vl={vl}"
        return context, hyp, tpl

    raise ValueError(mode)


def relation_rows(focals, seconds, *, ns: str, mode: str, split: str, eval_set: str, arm: str,
                  queries: list[str], train_kind: str, hyp_style: str = "train",
                  shuffle_records: bool = True, swap_focal_tags: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        # Two wording variants for each temporal record -> four contexts per pair, matching research/267 count.
        for vi in range(2):
            for vl in range(2):
                for hd in ["AB", "BA"]:
                    for qf in queries:
                        ctx, hyp, tpl = build_context_and_hyp(
                            mode=mode, f=f, s=s, am=am, qf=qf, hd=hd, vi=vi, vl=vl, ns=ns,
                            hyp_style=hyp_style, shuffle_records=shuffle_records,
                            swap_focal_tags=swap_focal_tags)
                        lab = labels_for(qf, f, s, hd)
                        rows.append(base.mk(
                            row_id=f"{ns}|{f.world_id}|{s.world_id}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=hyp, label=lab, split=split, eval_set=eval_set,
                            arm=arm, query_family=qf, focal_world=f.world_id, secondary_world=s.world_id,
                            changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                            template_group=tpl, train_kind=train_kind, hyp_dir=hd,
                            pair_key=f"{ns}|{f.world_id}|{s.world_id}|{tpl}|{qf}",
                            hyp_wording=hyp_style, ctx_wording=mode,
                            label_mode="address_dissociation"))
    return rows


def build_for_mode(args, mode: str):
    selected, inventory = base.select_splits(args)
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    base_rows = relation_rows(selected["base_stable"], selected["base_stable2"],
                              ns=f"s267_{mode}_base_anchor", mode=mode, split="train", eval_set="base",
                              arm="base", queries=base_queries, train_kind="base_stable_anchor",
                              hyp_style="train", shuffle_records=True)
    if args.base_train_wording:
        base_rows += relation_rows(selected["base_stable"], selected["base_stable2"],
                                   ns=f"s267_{mode}_base_train", mode=mode, split="train", eval_set="base",
                                   arm="base", queries=base_queries, train_kind="base_stable_train",
                                   hyp_style="train", shuffle_records=True)
    seconds = selected["sparse_secondary_stable"]
    update_rows = []
    update_rows += relation_rows(selected["sparse_changed"], seconds,
                                 ns=f"s267_{mode}_changed", mode=mode, split="train", eval_set="sparse",
                                 arm="balanced_temporal", queries=["focal_before", "focal_after"],
                                 train_kind="sparse_changed_focal", hyp_style="train", shuffle_records=True)
    update_rows += relation_rows(selected["sparse_stable"], seconds,
                                 ns=f"s267_{mode}_stable", mode=mode, split="train", eval_set="sparse",
                                 arm="balanced_temporal", queries=["focal_before", "focal_after"],
                                 train_kind="sparse_stable_focal", hyp_style="train", shuffle_records=True)

    evals: dict[str, list[dict[str, Any]]] = {}
    all_q = base_queries
    pairings = [
        ("heldChanged_heldStable", selected["eval_held_changed"], selected["eval_held_stable"]),
        ("heldStable_heldStable", selected["eval_held_stable"], selected["eval_held_stable2"]),
        ("trainChanged_heldStable", selected["eval_train_changed"], selected["eval_held_stable"]),
    ]
    for pname, focals, seconds_eval in pairings:
        evals[f"{mode}_{pname}_trainHyp"] = relation_rows(
            focals, seconds_eval, ns=f"eval267_{mode}_{pname}_train", mode=mode, split="held", eval_set=pname,
            arm="eval", queries=all_q, train_kind="eval_temporal", hyp_style="train", shuffle_records=True)
        evals[f"{mode}_{pname}_heldHyp"] = relation_rows(
            focals, seconds_eval, ns=f"eval267_{mode}_{pname}_held", mode=mode, split="held", eval_set=pname,
            arm="eval", queries=all_q, train_kind="eval_temporal", hyp_style="held", shuffle_records=True)
        if mode == "arbitrary_tag" and pname == "heldChanged_heldStable":
            evals[f"{mode}_{pname}_heldHyp_swappedFocalTags"] = relation_rows(
                focals, seconds_eval, ns=f"eval267_{mode}_{pname}_heldswap", mode=mode, split="held", eval_set=pname,
                arm="eval", queries=all_q, train_kind="eval_temporal", hyp_style="held", shuffle_records=True,
                swap_focal_tags=True)
    construction = {
        "inventory": inventory,
        "selection": {k: base.world_summary(v) for k, v in selected.items()},
        "base": base.describe_rows(base_rows),
        "updates": {"balanced_temporal": base.describe_rows(update_rows)},
        "eval_counts": {k: len(v) for k, v in sorted(evals.items())},
        "mode": mode,
        "randomized_record_order": True,
        "boundary": "Small one-seed balanced-temporal bridge dissociation; not BabyLM-scale training.",
    }
    return base_rows, update_rows, evals, construction


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for r in results:
        mode = r["mode"]
        item = {"best_train_acc": r["result"]["best_train_acc"],
                "train_fit": r["result"].get("train_fit", {}),
                "evals": {}}
        for en, ev in sorted(r["result"]["evals"].items()):
            c = ev["contrastive"]
            item["evals"][en] = {
                "contrastive_acc": c.get("contrastive_acc"),
                "by_query": c.get("by_query", {}),
                "by_query_margin": c.get("by_query_margin", {}),
                "standard_acc": ev["standard"].get("acc"),
            }
        out[mode] = item
    return out


def write_md(summary: dict[str, Any], out: Path):
    lines = ["# research temporal address dissociation\n\n",
             "This run uses the research fixed-meaning ATP temporal-change bridge with one seed and the balanced-temporal non-oracle arm. It tests whether section success survives randomized record order, held query paraphrases without exact Background/Update repetition, and arbitrary per-example tag addresses.\n\n"]
    for mode, item in summary["summary"].items():
        lines.append(f"## Mode: {mode}\n\n")
        lines.append(f"Best train accuracy: {item['best_train_acc']:.3f}\n\n")
        tf = item.get("train_fit", {})
        if tf:
            lines.append(f"Train fit by kind: `{json.dumps(tf.get('by_train_kind', {}), sort_keys=True)}`\n\n")
        lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in item["evals"].items():
            bq = ev["by_query"]; bm = ev["by_query_margin"]
            def v(q):
                x = bq.get(q)
                return float("nan") if x is None else x
            def m(q):
                x = bm.get(q)
                return float("nan") if x is None else x
            lines.append(f"| {en} | {ev['contrastive_acc']:.3f} | {v('focal_before'):.3f} | {v('focal_after'):.3f} | {v('secondary_before'):.3f} | {v('secondary_after'):.3f} | {m('focal_after'):.2f} | {m('secondary_after'):.2f} |\n")
        lines.append("\n")
    (out / "address_dissociation_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--modes", type=str, default="section_rolepara,arbitrary_tag")
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
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--encoder_lr", type=float, default=8e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_construction = {}
    built = {}
    for mode in modes:
        if mode not in {"section_rolepara", "arbitrary_tag"}:
            raise ValueError(f"unknown mode {mode}")
        base_rows, update_rows, evals, construction = build_for_mode(args, mode)
        all_construction[mode] = construction
        built[mode] = (base_rows, update_rows, evals)
        print(json.dumps({"status": "BUILT_MODE", "mode": mode, "base_rows": len(base_rows),
                          "update_rows": len(update_rows), "eval_sets": len(evals),
                          "selection": construction["selection"]}), flush=True)

    if args.dry_build or args.n_seeds <= 0:
        summary = {"status": "ADDRESS_DISSOCIATION_DRY", "args": vars(args),
                   "construction": all_construction}
        (out / "address_dissociation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
        write_md({"summary": {}}, out)
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "address_dissociation_summary.json")}), flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    dev = args.device
    results = []
    for mi, mode in enumerate(modes):
        base_rows, update_rows, evals = built[mode]
        for si in range(args.n_seeds):
            seed = 26700 + 100 * mi + si
            train_rows = list(base_rows) + list(update_rows)
            r = base.train_one(Path(args.model_path), tok, "balanced_temporal", seed, train_rows, evals, args, dev)
            r["mode"] = mode
            results.append({"mode": mode, "seed": seed, "result": r})
            partial = {"status": "ADDRESS_DISSOCIATION_PARTIAL", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "args": vars(args), "construction": all_construction, "summary": summarize(results), "per_seed": results,
                       "boundary": "Small bridge dissociation; arbitrary tags are address ceilings, not semantic temporal proof."}
            (out / "address_dissociation_partial.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", "utf-8")
            write_md(partial, out)
            # Print central held changed surfaces immediately.
            for en, ev in sorted(r["evals"].items()):
                if "heldChanged_heldStable" in en:
                    c = ev["contrastive"]
                    print(json.dumps({"event": "eval", "mode": mode, "seed": seed, "eval_set": en,
                                      "train_acc": r["best_train_acc"], "train_fit": r.get("train_fit"),
                                      "by_query": c.get("by_query"),
                                      "by_query_margin": c.get("by_query_margin"),
                                      "con_acc": c.get("contrastive_acc")}), flush=True)
    final = {"status": "ADDRESS_DISSOCIATION", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "args": vars(args), "construction": all_construction, "summary": summarize(results), "per_seed": results,
             "boundary": "One-seed small pretrained bridge; if positive, still requires replication and stronger semantic dissociation."}
    (out / "address_dissociation_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(final, out)
    print(json.dumps({"status": "DONE", "json": str(out / "address_dissociation_summary.json"),
                      "md": str(out / "address_dissociation_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
