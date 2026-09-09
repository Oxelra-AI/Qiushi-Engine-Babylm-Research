#!/usr/bin/env python3
"""research: two-hop temporal role -> tag -> record probe.

research established that arbitrary per-example tags let a pretrained bridge read
one of two conflicting focal records.  This probe asks whether temporal/discourse
language can assign such an address without repeating the tag in the query.

Context example:
  Focal prior snapshot is entry amber.
  Focal revised snapshot is entry cobalt.
  Entry cobalt: B was ranked above A.
  Entry amber: A was ranked above B.
  [SEP] According to the original snapshot, does A rank above B?

If the model transfers original~prior and current~revised, it must compose:
  query temporal expression -> declared role -> random tag -> record value.

The run remains a small pretrained bridge, not BabyLM-scale training.
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
ADDR_PATH = STUDY / "training/scripts/temporal_address_dissociation.py"
spec = importlib.util.spec_from_file_location("addr267_indirect", ADDR_PATH)
addr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = addr
spec.loader.exec_module(addr)
base = addr.base
DEFAULT_MODEL = addr.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/temporal_indirect_address_probe"

TAG_POOL = addr.TAG_POOL

ROLE_FAMILIES = {
    "background_update": {"bind_b": "background", "bind_a": "update", "query_b": "background", "query_a": "update"},
    "initial_latest": {"bind_b": "initial", "bind_a": "latest", "query_b": "first", "query_a": "latest"},
    "old_new": {"bind_b": "old", "bind_a": "new", "query_b": "former", "query_a": "new"},
    # Held exact: unseen role words, but query repeats binding word.
    "original_current_exact": {"bind_b": "original", "bind_a": "current", "query_b": "original", "query_a": "current"},
    # Held paraphrase: query does not repeat the binding role word.
    "prior_revised_to_original_current": {"bind_b": "prior", "bind_a": "revised", "query_b": "original", "query_a": "current"},
    "earlier_later_to_original_current": {"bind_b": "earlier", "bind_a": "later", "query_b": "original", "query_a": "current"},
}


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def tag4(key: str) -> list[str]:
    return random.Random(stable_int("indirect_tag4|" + key)).sample(TAG_POOL, 4)


def shuffle(items: list[str], key: str) -> list[str]:
    items = list(items)
    random.Random(stable_int("indirect_shuffle|" + key)).shuffle(items)
    return items


def sent_initial(w, am, variant: int) -> str:
    tmpls = ["{H} was ranked above {Lo} in the ATP list.", "{Lo} was ranked below {H} in the ATP list."]
    return tmpls[variant % 2].format(H=base.an(am, w.higher_initial), Lo=base.an(am, w.lower_initial))


def sent_later(w, am, variant: int) -> str:
    tmpls = ["{H} was ranked above {Lo} in the ATP list.", "{Lo} was ranked below {H} in the ATP list."]
    return tmpls[variant % 2].format(H=base.an(am, w.higher_later), Lo=base.an(am, w.lower_later))


def xy(w, am, hd: str):
    return addr.hyp_xy(w, am, hd)


def label_for(qf: str, f, s, hd: str) -> int:
    return addr.labels_for(qf, f, s, hd)


def build_context(f, s, am, family: str, ns: str, vi: int, vl: int, *, swap_roles: bool = False) -> tuple[str, dict[str, str], str]:
    rf = ROLE_FAMILIES[family]
    tf_b, tf_a, ts_b, ts_a = tag4(f"{ns}|{family}|{f.world_id}|{s.world_id}|{vi}|{vl}")
    # Entry tags always point to their true temporal slice.  swap_roles changes only the declarations.
    decl_tf_b, decl_tf_a = (tf_a, tf_b) if swap_roles else (tf_b, tf_a)
    decl_ts_b, decl_ts_a = (ts_a, ts_b) if swap_roles else (ts_b, ts_a)
    parts = [
        f"Focal {rf['bind_b']} snapshot is entry {decl_tf_b}.",
        f"Focal {rf['bind_a']} snapshot is entry {decl_tf_a}.",
        f"Separate {rf['bind_b']} snapshot is entry {decl_ts_b}.",
        f"Separate {rf['bind_a']} snapshot is entry {decl_ts_a}.",
        f"Entry {tf_b}: " + sent_initial(f, am, vi),
        f"Entry {tf_a}: " + sent_later(f, am, vl),
        f"Entry {ts_b}: " + sent_initial(s, am, vi + 1),
        f"Entry {ts_a}: " + sent_later(s, am, vl + 1),
    ]
    parts = shuffle(parts, f"{ns}|{family}|{f.world_id}|{s.world_id}|{vi}|{vl}|swap={swap_roles}")
    tag_map = {"focal_before": tf_b, "focal_after": tf_a, "secondary_before": ts_b, "secondary_after": ts_a}
    tpl = f"family={family}|vi={vi}|vl={vl}|swap={swap_roles}"
    return " ".join(parts), tag_map, tpl


def hyp_role(qf: str, qw, am, hd: str, family: str, style: str) -> str:
    rf = ROLE_FAMILIES[family]
    role = rf["query_b"] if qf.endswith("before") else rf["query_a"]
    x, y = xy(qw, am, hd)
    if style == "train":
        return f"According to the {role} snapshot, {x} was ranked higher than {y}."
    return f"Using the {role} ranking record, {x} outranked {y}."


def hyp_direct(qf: str, qw, am, hd: str, tag: str, style: str) -> str:
    x, y = xy(qw, am, hd)
    if style == "train":
        return f"According to entry {tag}, {x} was ranked higher than {y}."
    return f"Using the record named {tag}, {x} outranked {y}."


def relation_rows(focals, seconds, *, ns: str, family: str, split: str, eval_set: str, arm: str,
                  queries: list[str], train_kind: str, query_mode: str,
                  hyp_style: str = "train", swap_roles: bool = False) -> list[dict[str, Any]]:
    rows = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                ctx, tag_map, tpl = build_context(f, s, am, family, ns, vi, vl, swap_roles=swap_roles)
                for hd in ["AB", "BA"]:
                    for qf in queries:
                        qw = f if qf.startswith("focal") else s
                        if query_mode == "role":
                            hyp = hyp_role(qf, qw, am, hd, family, hyp_style)
                            # Labels keep intended temporal truth even when declarations are swapped;
                            # changed focal rows should invert if the model follows swapped declarations.
                            lab = label_for(qf, f, s, hd)
                            mode_label = "role_query"
                        elif query_mode == "direct_tag":
                            hyp = hyp_direct(qf, qw, am, hd, tag_map[qf], hyp_style)
                            # Direct tag labels are the content of that tag, unaffected by role declarations.
                            lab = label_for(qf, f, s, hd)
                            mode_label = "direct_tag_query"
                        else:
                            raise ValueError(query_mode)
                        rows.append(base.mk(
                            row_id=f"{ns}|{family}|{query_mode}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=hyp, label=lab, split=split, eval_set=eval_set,
                            arm=arm, query_family=qf, focal_world=f.world_id, secondary_world=s.world_id,
                            changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                            template_group=tpl, train_kind=f"{train_kind}|{family}|{mode_label}", hyp_dir=hd,
                            pair_key=f"{ns}|{family}|{query_mode}|{tpl}|{qf}", hyp_wording=hyp_style,
                            ctx_wording=family, label_mode="indirect_address"))
    return rows


def parse_list(s: str) -> list[str]:
    xs = [x.strip() for x in s.split(",") if x.strip()]
    for x in xs:
        if x not in ROLE_FAMILIES:
            raise ValueError(f"unknown role family {x}")
    return xs


def build(args):
    selected, inventory = base.select_splits(args)
    train_fams = parse_list(args.train_families)
    eval_fams = parse_list(args.eval_families)
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    sparse_queries = ["focal_before", "focal_after"]
    base_rows = []
    update_rows = []
    seconds = selected["sparse_secondary_stable"]
    for fam in train_fams:
        base_rows += relation_rows(selected["base_stable"], selected["base_stable2"], ns=f"ind267_base_{fam}", family=fam,
                                   split="train", eval_set="base", arm="base", queries=base_queries,
                                   train_kind="base_stable", query_mode="role", hyp_style="train")
        update_rows += relation_rows(selected["sparse_changed"], seconds, ns=f"ind267_chg_{fam}", family=fam,
                                      split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                      train_kind="sparse_changed_focal", query_mode="role", hyp_style="train")
        update_rows += relation_rows(selected["sparse_stable"], seconds, ns=f"ind267_stb_{fam}", family=fam,
                                      split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                      train_kind="sparse_stable_focal", query_mode="role", hyp_style="train")
        if args.train_direct_tag:
            base_rows += relation_rows(selected["base_stable"], selected["base_stable2"], ns=f"ind267_baseD_{fam}", family=fam,
                                       split="train", eval_set="base", arm="base", queries=base_queries,
                                       train_kind="base_stable_direct", query_mode="direct_tag", hyp_style="train")
            update_rows += relation_rows(selected["sparse_changed"], seconds, ns=f"ind267_chgD_{fam}", family=fam,
                                          split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                          train_kind="sparse_changed_direct", query_mode="direct_tag", hyp_style="train")
            update_rows += relation_rows(selected["sparse_stable"], seconds, ns=f"ind267_stbD_{fam}", family=fam,
                                          split="train", eval_set="sparse", arm="balanced_temporal", queries=sparse_queries,
                                          train_kind="sparse_stable_direct", query_mode="direct_tag", hyp_style="train")
    evals = {}
    pairings = [
        ("heldChanged_heldStable", selected["eval_held_changed"], selected["eval_held_stable"]),
        ("heldStable_heldStable", selected["eval_held_stable"], selected["eval_held_stable2"]),
        ("trainChanged_heldStable", selected["eval_train_changed"], selected["eval_held_stable"]),
    ]
    for fam in eval_fams:
        for pname, focals, seconds_eval in pairings:
            for mode in ["role", "direct_tag"]:
                evals[f"{fam}_{pname}_{mode}_heldHyp"] = relation_rows(
                    focals, seconds_eval, ns=f"eval267_ind_{fam}_{pname}_{mode}", family=fam,
                    split="held", eval_set=pname, arm="eval", queries=base_queries,
                    train_kind="eval_temporal", query_mode=mode, hyp_style="held")
            if pname == "heldChanged_heldStable":
                evals[f"{fam}_{pname}_roleSwap_heldHyp"] = relation_rows(
                    focals, seconds_eval, ns=f"eval267_ind_{fam}_{pname}_roleswap", family=fam,
                    split="held", eval_set=pname, arm="eval", queries=base_queries,
                    train_kind="eval_temporal", query_mode="role", hyp_style="held", swap_roles=True)
                evals[f"{fam}_{pname}_directTag_roleSwap_heldHyp"] = relation_rows(
                    focals, seconds_eval, ns=f"eval267_ind_{fam}_{pname}_direct_roleswap", family=fam,
                    split="held", eval_set=pname, arm="eval", queries=base_queries,
                    train_kind="eval_temporal", query_mode="direct_tag", hyp_style="held", swap_roles=True)
    construction = {
        "inventory": inventory,
        "selection": {k: base.world_summary(v) for k, v in selected.items()},
        "train_families": train_fams,
        "eval_families": eval_fams,
        "train_direct_tag": args.train_direct_tag,
        "base": base.describe_rows(base_rows),
        "updates": {"balanced_temporal": base.describe_rows(update_rows)},
        "eval_counts": {k: len(v) for k, v in sorted(evals.items())},
        "boundary": "Two-hop role-to-tag probe with shuffled context sentences; one-seed small pretrained bridge.",
    }
    return base_rows, update_rows, evals, construction


def summarize_result(r: dict[str, Any]) -> dict[str, Any]:
    out = {"best_train_acc": r["best_train_acc"], "train_fit": r.get("train_fit", {}), "evals": {}}
    for en, ev in sorted(r["evals"].items()):
        c = ev["contrastive"]
        out["evals"][en] = {"contrastive_acc": c.get("contrastive_acc"),
                             "by_query": c.get("by_query", {}),
                             "by_query_margin": c.get("by_query_margin", {}),
                             "standard_acc": ev["standard"].get("acc")}
    return out


def write_md(summary: dict[str, Any], out: Path):
    lines = ["# research two-hop temporal indirect address probe\n\n",
             "Context binds a temporal/discourse role to a random entry tag; the queried relation is stored under that tag. Role-query rows ask via the role, not the tag. Direct-tag rows name the tag. Role-swap rows swap the declarations while labels retain intended temporal truth.\n\n"]
    cons = summary.get("construction", {})
    if cons:
        lines.append(f"Train role families: {cons.get('train_families')}\n\n")
        lines.append(f"Eval role families: {cons.get('eval_families')}\n\n")
        lines.append(f"Train rows: base={cons.get('base',{}).get('n')} update={cons.get('updates',{}).get('balanced_temporal',{}).get('n')} direct_tag_support={cons.get('train_direct_tag')}\n\n")
    s = summary.get("summary", {})
    if s:
        lines.append(f"Best train accuracy: {s['best_train_acc']:.3f}\n\n")
        lines.append(f"Train fit by kind: `{json.dumps(s.get('train_fit', {}).get('by_train_kind', {}), sort_keys=True)}`\n\n")
        lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in s["evals"].items():
            if "heldChanged_heldStable" not in en:
                continue
            bq, bm = ev["by_query"], ev["by_query_margin"]
            def v(q):
                return bq.get(q, float("nan"))
            def m(q):
                return bm.get(q, float("nan"))
            lines.append(f"| {en} | {ev['contrastive_acc']:.3f} | {v('focal_before'):.3f} | {v('focal_after'):.3f} | {v('secondary_before'):.3f} | {v('secondary_after'):.3f} | {m('focal_after'):.2f} | {m('secondary_after'):.2f} |\n")
    (out / "indirect_address_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--train_families", type=str, default="background_update,initial_latest,old_new")
    ap.add_argument("--eval_families", type=str, default="original_current_exact,prior_revised_to_original_current,earlier_later_to_original_current")
    ap.add_argument("--train_direct_tag", action="store_true")
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--base_stable", type=int, default=40)
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
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dry_build", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    base_rows, update_rows, evals, construction = build(args)
    print(json.dumps({"status": "BUILT", "base_rows": len(base_rows), "update_rows": len(update_rows),
                      "train_rows": len(base_rows) + len(update_rows), "eval_sets": len(evals),
                      "construction": {"train_families": construction["train_families"], "eval_families": construction["eval_families"], "train_direct_tag": construction["train_direct_tag"]}}), flush=True)
    if args.dry_build or args.n_seeds <= 0:
        summary = {"status": "INDIRECT_ADDRESS_DRY", "args": vars(args), "construction": construction}
        (out / "indirect_address_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
        write_md(summary, out)
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "indirect_address_summary.json")}), flush=True)
        return

    tok = AutoTokenizer.from_pretrained(args.model_path)
    results = []
    for si in range(args.n_seeds):
        seed = 26780 + si
        r = base.train_one(Path(args.model_path), tok, "balanced_temporal", seed, list(base_rows) + list(update_rows), evals, args, args.device)
        results.append(r)
        partial = {"status": "INDIRECT_ADDRESS_PARTIAL", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "args": vars(args), "construction": construction, "summary": summarize_result(r), "per_seed": results,
                   "boundary": "Small two-hop role-to-tag bridge; not BabyLM-scale training."}
        (out / "indirect_address_partial.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", "utf-8")
        write_md(partial, out)
        for en, ev in sorted(r["evals"].items()):
            if "heldChanged_heldStable" in en:
                c = ev["contrastive"]
                print(json.dumps({"event": "eval", "seed": seed, "eval_set": en,
                                  "train_acc": r["best_train_acc"], "train_fit": r.get("train_fit"),
                                  "by_query": c.get("by_query"), "by_query_margin": c.get("by_query_margin"),
                                  "con_acc": c.get("contrastive_acc")}), flush=True)
    # One seed is intended; if multiple are requested, keep the first for compact summary and aggregate separately.
    final = {"status": "INDIRECT_ADDRESS", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "args": vars(args), "construction": construction, "summary": summarize_result(results[0]),
             "aggregate": base.aggregate(results), "per_seed": results,
             "boundary": "Small pretrained bridge; positive transfer would still need replication."}
    (out / "indirect_address_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(final, out)
    print(json.dumps({"status": "DONE", "json": str(out / "indirect_address_summary.json"), "md": str(out / "indirect_address_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
