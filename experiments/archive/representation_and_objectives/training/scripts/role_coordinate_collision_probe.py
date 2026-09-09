#!/usr/bin/env python3
"""research: matched collision/indirection probe for temporal role -> coordinate learning.

research showed that arbitrary entry tags can rescue changed-focal state readout,
but the first role->tag indirect construction changed duplicated tags, declaration
sentences, and compositional indirection at once.  This script separates those
factors while keeping the research/267 ATP worlds, balanced changed/stable focal
updates, shuffled record order, and paired contrastive scoring.

Scientific boundary: small pretrained bridge only; not BabyLM-scale training.
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
from transformers import AutoTokenizer

STUDY = Path("experiments/archive/representation_and_objectives")
ADDR_PATH = STUDY / "training/scripts/temporal_address_dissociation.py"
spec = importlib.util.spec_from_file_location("addr267_for_step268", ADDR_PATH)
addr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = addr
spec.loader.exec_module(addr)
base = addr.base
DEFAULT_MODEL = addr.DEFAULT_MODEL
DEFAULT_OUT = STUDY / "data/role_coordinate_collision_probe"
TAG_POOL = addr.TAG_POOL

ARM_CONFIGS = {
    # one value-bearing entry occurrence per tag; recreates research arbitrary-tag ceiling
    "unique_tag_direct": {"ctx_mode": "unique", "train_modes": ["direct_tag"]},
    # four extra tag-free sentences: length/attention burden without lexical collision
    "filler_direct": {"ctx_mode": "filler", "train_modes": ["direct_tag"]},
    # extra neutral tag mentions without entry-shaped syntax
    "labeldup_direct": {"ctx_mode": "labeldup", "train_modes": ["direct_tag"]},
    # extra neutral entry TAG mentions without relation content
    "entrydup_direct": {"ctx_mode": "entrydup", "train_modes": ["direct_tag"]},
    # non-temporal role-like binding sentences using entry TAG
    "slotdecl_direct": {"ctx_mode": "slotdecl", "train_modes": ["direct_tag"]},
    # temporal role declarations are present, but query still names the entry tag directly
    "roledecl_direct": {"ctx_mode": "roledecl", "train_modes": ["direct_tag"]},
    # temporal role declarations and query uses the role phrase; direct-tag readout is eval-only
    "roledecl_indirect": {"ctx_mode": "roledecl", "train_modes": ["role_exact"]},
    # train both direct tag retrieval and role query composition in the same model
    "roledecl_mixed": {"ctx_mode": "roledecl", "train_modes": ["direct_tag", "role_exact"]},
}

FILLER_SENTENCES = [
    "The archive includes an auxiliary catalog note for balance.",
    "The collection stores a plain auxiliary sentence for context.",
    "The dataset contains an extra neutral note without a record.",
    "The file keeps a balancing sentence in the same passage.",
]
SLOT_NAMES = ["northern marker", "eastern marker", "southern marker", "western marker"]


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def tag4(key: str) -> list[str]:
    return random.Random(stable_int("tag4|" + key)).sample(TAG_POOL, 4)


def shuffle(parts: list[str], key: str) -> list[str]:
    parts = list(parts)
    random.Random(stable_int("shuffle|" + key)).shuffle(parts)
    return parts


def sent_initial(w, am, variant: int) -> str:
    tmpls = ["{H} was ranked above {Lo} in the ATP list.", "{Lo} was ranked below {H} in the ATP list."]
    return tmpls[variant % 2].format(H=base.an(am, w.higher_initial), Lo=base.an(am, w.lower_initial))


def sent_later(w, am, variant: int) -> str:
    tmpls = ["{H} was ranked above {Lo} in the ATP list.", "{Lo} was ranked below {H} in the ATP list."]
    return tmpls[variant % 2].format(H=base.an(am, w.higher_later), Lo=base.an(am, w.lower_later))


def q_scope(qf: str) -> str:
    return "focal" if qf.startswith("focal") else "separate"


def q_time(qf: str) -> str:
    return "background" if qf.endswith("before") else "update"


def q_time_sem(qf: str) -> str:
    return "prior" if qf.endswith("before") else "revised"


def tags_for(qf_tags: tuple[str, str, str, str]) -> dict[str, str]:
    tf_b, tf_a, ts_b, ts_a = qf_tags
    return {"focal_before": tf_b, "focal_after": tf_a, "secondary_before": ts_b, "secondary_after": ts_a}


def build_context(*, f, s, am: dict[str, str], ctx_mode: str, ns: str, vi: int, vl: int,
                  swap_role_decls: bool = False) -> tuple[str, dict[str, str], str]:
    tags = tag4(f"{ns}|{ctx_mode}|{f.world_id}|{s.world_id}|{vi}|{vl}")
    tf_b, tf_a, ts_b, ts_a = tags
    tmap = tags_for((tf_b, tf_a, ts_b, ts_a))
    entries = [
        f"Entry {tf_b}: " + sent_initial(f, am, vi),
        f"Entry {tf_a}: " + sent_later(f, am, vl),
        f"Entry {ts_b}: " + sent_initial(s, am, vi + 1),
        f"Entry {ts_a}: " + sent_later(s, am, vl + 1),
    ]
    extras: list[str] = []
    if ctx_mode == "unique":
        extras = []
    elif ctx_mode == "filler":
        extras = list(FILLER_SENTENCES)
    elif ctx_mode == "labeldup":
        extras = [f"The catalog includes the label {t}." for t in tags]
    elif ctx_mode == "entrydup":
        extras = [f"The catalog mentions entry {t}." for t in tags]
    elif ctx_mode == "slotdecl":
        extras = [f"The {slot} is entry {t}." for slot, t in zip(SLOT_NAMES, tags)]
    elif ctx_mode == "roledecl":
        decl_tf_b, decl_tf_a = (tf_a, tf_b) if swap_role_decls else (tf_b, tf_a)
        decl_ts_b, decl_ts_a = (ts_a, ts_b) if swap_role_decls else (ts_b, ts_a)
        extras = [
            f"The focal background snapshot is entry {decl_tf_b}.",
            f"The focal update snapshot is entry {decl_tf_a}.",
            f"The separate background snapshot is entry {decl_ts_b}.",
            f"The separate update snapshot is entry {decl_ts_a}.",
        ]
    else:
        raise ValueError(ctx_mode)
    parts = shuffle(entries + extras, f"{ns}|{ctx_mode}|{f.world_id}|{s.world_id}|{vi}|{vl}|swap={swap_role_decls}")
    tpl = f"ctx={ctx_mode}|vi={vi}|vl={vl}|swapDecl={swap_role_decls}"
    return " ".join(parts), tmap, tpl


def xy(w, am, hd: str) -> tuple[str, str]:
    return addr.hyp_xy(w, am, hd)


def label_for(qf: str, f, s, hd: str) -> int:
    return addr.labels_for(qf, f, s, hd)


def hyp_for(*, qf: str, qw, am: dict[str, str], hd: str, query_mode: str, tag: str, hyp_style: str) -> str:
    x, y = xy(qw, am, hd)
    if query_mode == "direct_tag":
        if hyp_style == "train":
            return f"According to entry {tag}, {x} was ranked higher than {y}."
        return f"Using the record named {tag}, {x} outranked {y}."
    if query_mode == "role_exact":
        role = f"{q_scope(qf)} {q_time(qf)} snapshot"
        if hyp_style == "train":
            return f"According to the {role}, {x} was ranked higher than {y}."
        return f"Using the {role}, {x} outranked {y}."
    if query_mode == "role_sem":
        role = f"{q_scope(qf)} {q_time_sem(qf)} record"
        return f"Using the {role}, {x} outranked {y}."
    raise ValueError(query_mode)


def relation_rows(focals, seconds, *, ns: str, ctx_mode: str, split: str, eval_set: str, arm: str,
                  queries: list[str], train_kind: str, query_mode: str, hyp_style: str = "train",
                  swap_role_decls: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, f in enumerate(focals):
        s = base.pick_secondary(f, seconds, i)
        if s is None:
            continue
        am = base.amap4(f, s, ns)
        for vi in range(2):
            for vl in range(2):
                ctx, tmap, tpl = build_context(f=f, s=s, am=am, ctx_mode=ctx_mode, ns=ns, vi=vi, vl=vl,
                                                swap_role_decls=swap_role_decls)
                for hd in ["AB", "BA"]:
                    for qf in queries:
                        qw = f if qf.startswith("focal") else s
                        hyp = hyp_for(qf=qf, qw=qw, am=am, hd=hd, query_mode=query_mode,
                                      tag=tmap[qf], hyp_style=hyp_style)
                        lab = label_for(qf, f, s, hd)
                        rows.append(base.mk(
                            row_id=f"{ns}|{ctx_mode}|{query_mode}|{tpl}|{qf}|{hd}",
                            context=ctx, hypothesis=hyp, label=lab, split=split, eval_set=eval_set,
                            arm=arm, query_family=qf, focal_world=f.world_id, secondary_world=s.world_id,
                            changed_focal=bool(f.changed), stable_secondary=not bool(s.changed),
                            template_group=tpl, train_kind=f"{train_kind}|ctx={ctx_mode}|q={query_mode}",
                            hyp_dir=hd, pair_key=f"{ns}|{ctx_mode}|{query_mode}|{tpl}|{qf}",
                            hyp_wording=hyp_style, ctx_wording=ctx_mode, label_mode="collision"))
    return rows


def build_arm(args, arm_name: str):
    cfg = ARM_CONFIGS[arm_name]
    ctx_mode = cfg["ctx_mode"]
    train_modes = cfg["train_modes"]
    selected, inventory = base.select_splits(args)
    base_queries = ["focal_before", "focal_after", "secondary_before", "secondary_after"]
    sparse_queries = ["focal_before", "focal_after"]
    seconds = selected["sparse_secondary_stable"]
    base_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    for qm in train_modes:
        base_rows += relation_rows(selected["base_stable"], selected["base_stable2"],
                                   ns=f"s268_{arm_name}_base_{qm}", ctx_mode=ctx_mode, split="train", eval_set="base",
                                   arm="base", queries=base_queries, train_kind="base_stable_anchor", query_mode=qm,
                                   hyp_style="train")
        if getattr(args, "base_train_wording", False):
            base_rows += relation_rows(selected["base_stable"], selected["base_stable2"],
                                       ns=f"s268_{arm_name}_base_train_{qm}", ctx_mode=ctx_mode, split="train", eval_set="base",
                                       arm="base", queries=base_queries, train_kind="base_stable_train", query_mode=qm,
                                       hyp_style="train")
        update_rows += relation_rows(selected["sparse_changed"], seconds,
                                     ns=f"s268_{arm_name}_chg_{qm}", ctx_mode=ctx_mode, split="train", eval_set="sparse",
                                     arm="balanced_temporal", queries=sparse_queries, train_kind="sparse_changed_focal",
                                     query_mode=qm, hyp_style="train")
        update_rows += relation_rows(selected["sparse_stable"], seconds,
                                     ns=f"s268_{arm_name}_stb_{qm}", ctx_mode=ctx_mode, split="train", eval_set="sparse",
                                     arm="balanced_temporal", queries=sparse_queries, train_kind="sparse_stable_focal",
                                     query_mode=qm, hyp_style="train")
    evals: dict[str, list[dict[str, Any]]] = {}
    pairings = [
        ("heldChanged_heldStable", selected["eval_held_changed"], selected["eval_held_stable"]),
        ("trainChanged_heldStable", selected["eval_train_changed"], selected["eval_held_stable"]),
        ("heldStable_heldStable", selected["eval_held_stable"], selected["eval_held_stable2"]),
    ]
    eval_modes = ["direct_tag"]
    if ctx_mode == "roledecl":
        eval_modes += ["role_exact", "role_sem"]
    for pname, focals, seconds_eval in pairings:
        for qm in eval_modes:
            evals[f"{arm_name}_{pname}_{qm}_heldHyp"] = relation_rows(
                focals, seconds_eval, ns=f"eval268_{arm_name}_{pname}_{qm}", ctx_mode=ctx_mode,
                split="held", eval_set=pname, arm="eval", queries=base_queries, train_kind="eval_temporal",
                query_mode=qm, hyp_style="held")
            if ctx_mode == "roledecl" and pname == "heldChanged_heldStable":
                evals[f"{arm_name}_{pname}_{qm}_roleSwap_heldHyp"] = relation_rows(
                    focals, seconds_eval, ns=f"eval268_{arm_name}_{pname}_{qm}_swap", ctx_mode=ctx_mode,
                    split="held", eval_set=pname, arm="eval", queries=base_queries, train_kind="eval_temporal",
                    query_mode=qm, hyp_style="held", swap_role_decls=True)
    construction = {
        "arm_name": arm_name,
        "ctx_mode": ctx_mode,
        "train_modes": train_modes,
        "inventory": inventory,
        "selection": {k: base.world_summary(v) for k, v in selected.items()},
        "base": base.describe_rows(base_rows),
        "updates": {"balanced_temporal": base.describe_rows(update_rows)},
        "eval_counts": {k: len(v) for k, v in sorted(evals.items())},
        "boundary": "Matched research collision/indirection arm; small pretrained bridge only.",
    }
    return base_rows, update_rows, evals, construction


def summarize_one(r: dict[str, Any]) -> dict[str, Any]:
    out = {"best_train_acc": r["best_train_acc"], "train_fit": r.get("train_fit", {}), "evals": {}}
    for en, ev in sorted(r["evals"].items()):
        c = ev["contrastive"]
        out["evals"][en] = {"contrastive_acc": c.get("contrastive_acc"),
                             "by_query": c.get("by_query", {}),
                             "by_query_margin": c.get("by_query_margin", {}),
                             "standard_acc": ev["standard"].get("acc")}
    return out


def token_audit(tok, groups: dict[str, list[dict[str, Any]]], max_len: int) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name, rows in groups.items():
        texts = [r["text"] for r in rows]
        lens = [len(tok.encode(t, add_special_tokens=True)) for t in texts]
        arr = np.array(lens, dtype=float)
        audit[name] = {"n": len(lens), "min": int(arr.min()) if len(arr) else 0,
                       "p50": float(np.percentile(arr, 50)) if len(arr) else 0.0,
                       "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
                       "p99": float(np.percentile(arr, 99)) if len(arr) else 0.0,
                       "max": int(arr.max()) if len(arr) else 0,
                       "frac_gt_max_len": float(np.mean(arr > max_len)) if len(arr) else 0.0}
    return audit


def write_md(summary: dict[str, Any], out: Path):
    lines = ["# research role-coordinate collision probe\n\n",
             "Small matched bridge separating context burden, duplicated tag mentions, declaration language, and role-to-coordinate indirection. Held readouts are meaningful only when sparse changed-focal rows fit.\n\n"]
    for arm, item in summary.get("summary", {}).items():
        lines.append(f"## {arm}\n\n")
        lines.append(f"Best train acc: {item['best_train_acc']:.3f}\n\n")
        tf = item.get("train_fit", {})
        if tf:
            lines.append(f"Train fit by kind: `{json.dumps(tf.get('by_train_kind', {}), sort_keys=True)}`\n\n")
        lines.append("| eval_set | con_acc | focal_before | focal_after | secondary_before | secondary_after | focal_after_margin | secondary_after_margin |\n")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for en, ev in item.get("evals", {}).items():
            if "heldChanged_heldStable" not in en:
                continue
            bq, bm = ev["by_query"], ev["by_query_margin"]
            def v(q):
                x = bq.get(q)
                return float("nan") if x is None else float(x)
            def m(q):
                x = bm.get(q)
                return float("nan") if x is None else float(x)
            lines.append(f"| {en} | {ev['contrastive_acc']:.3f} | {v('focal_before'):.3f} | {v('focal_after'):.3f} | {v('secondary_before'):.3f} | {v('secondary_after'):.3f} | {m('focal_after'):.2f} | {m('secondary_after'):.2f} |\n")
        lines.append("\n")
    (out / "role_coordinate_collision_summary.md").write_text("".join(lines), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--arms", type=str, default="unique_tag_direct,filler_direct,labeldup_direct,entrydup_direct,slotdecl_direct,roledecl_direct,roledecl_indirect")
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--seed_base", type=int, default=26800)
    ap.add_argument("--base_stable", type=int, default=20)
    ap.add_argument("--base_train_wording", action="store_true")
    ap.add_argument("--sparse_changed", type=int, default=8)
    ap.add_argument("--sparse_stable", type=int, default=8)
    ap.add_argument("--eval_held_changed", type=int, default=20)
    ap.add_argument("--eval_held_stable", type=int, default=20)
    ap.add_argument("--eval_train_changed", type=int, default=20)
    ap.add_argument("--eval_train_stable", type=int, default=20)
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

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_CONFIGS:
            raise ValueError(f"unknown arm {a}")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    built = {}
    constructions = {}
    for arm in arms:
        base_rows, update_rows, evals, construction = build_arm(args, arm)
        built[arm] = (base_rows, update_rows, evals)
        constructions[arm] = construction
        print(json.dumps({"status": "BUILT_ARM", "arm": arm, "base_rows": len(base_rows),
                          "update_rows": len(update_rows), "eval_sets": len(evals),
                          "selection": construction["selection"]}), flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    all_train = []
    all_eval = []
    for base_rows, update_rows, evals in built.values():
        all_train.extend(base_rows); all_train.extend(update_rows)
        for rows in evals.values():
            all_eval.extend(rows)
    audit = token_audit(tok, {"train_all": all_train, "eval_all": all_eval}, args.max_len)
    print(json.dumps({"status": "TOKEN_AUDIT", "audit": audit}), flush=True)

    if args.dry_build or args.n_seeds <= 0:
        dry = {"status": "ROLE_COORDINATE_COLLISION_DRY", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "args": vars(args), "construction": constructions, "token_audit": audit}
        (out / "role_coordinate_collision_summary.json").write_text(json.dumps(dry, indent=2, ensure_ascii=False) + "\n", "utf-8")
        write_md({"summary": {}}, out)
        print(json.dumps({"status": "DRY_DONE", "json": str(out / "role_coordinate_collision_summary.json")}), flush=True)
        return

    results: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    summary: dict[str, Any] = {}
    for ai, arm in enumerate(arms):
        base_rows, update_rows, evals = built[arm]
        train_rows = list(base_rows) + list(update_rows)
        for si in range(args.n_seeds):
            seed = args.seed_base + 100 * ai + si
            r = base.train_one(Path(args.model_path), tok, "balanced_temporal", seed, train_rows, evals, args, args.device)
            r["arm"] = arm
            results[arm].append(r)
            summary[arm] = summarize_one(r)
            partial = {"status": "ROLE_COORDINATE_COLLISION_PARTIAL", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "args": vars(args), "construction": constructions, "token_audit": audit,
                       "summary": summary, "per_seed": {k: v for k, v in results.items()},
                       "boundary": "Small bridge probe; fit-first interpretation."}
            (out / "role_coordinate_collision_partial.json").write_text(json.dumps(partial, indent=2, ensure_ascii=False) + "\n", "utf-8")
            write_md(partial, out)
            for en, ev in sorted(r["evals"].items()):
                if "heldChanged_heldStable" in en:
                    c = ev["contrastive"]
                    print(json.dumps({"event": "eval", "arm": arm, "seed": seed, "eval_set": en,
                                      "train_acc": r["best_train_acc"], "train_fit": r.get("train_fit"),
                                      "by_query": c.get("by_query"), "by_query_margin": c.get("by_query_margin"),
                                      "con_acc": c.get("contrastive_acc")}), flush=True)
    final = {"status": "ROLE_COORDINATE_COLLISION", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "args": vars(args), "construction": constructions, "token_audit": audit,
             "summary": summary, "per_seed": {k: v for k, v in results.items()},
             "boundary": "Small pretrained bridge probe; use as route-local evidence only after changed-row fit checks."}
    (out / "role_coordinate_collision_summary.json").write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", "utf-8")
    write_md(final, out)
    print(json.dumps({"status": "DONE", "json": str(out / "role_coordinate_collision_summary.json"),
                      "md": str(out / "role_coordinate_collision_summary.md")}), flush=True)


if __name__ == "__main__":
    main()
