#!/usr/bin/env python3
"""research refinement: matched-update controls for slot reuse.

This addresses two concerns raised by the research verifier:
  * winner-first/loser-first surface order could drive the apparent coordinate;
  * neutral exposure in the previous script was ordered/coreference training, not
    pure token familiarity, and exposure modes had more rows than absent.

All modes here receive the same number of extra rows.  Anchor/probe template sets
are exactly balanced for winner-first and loser-first forms.

Exposure modes for held-pool fillers:
  absent_control: extra rows use only the train pool (held tokens absent), so row
                  count and update budget match exposure modes.
  pure_match:     held tokens appear only in same/different identity-matching rows;
                  no win/loss relation and no A-before-B convention.
  ordered_neutral:held tokens appear in ordered non-sports pair rows.
  anchor_random: held tokens appear in relation-anchor sentences with balanced
                  uninformative labels.
  anchor_true:   held tokens appear in true relation-anchor sentences.
  anchor_flip:   held tokens appear in complemented relation-anchor sentences.

Sparse probe anchors still use the train pool and are crossed as zero/true/anti.
Evaluation uses held-pool fillers on balanced probe templates.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import identity_orbit_anchor_test as base  # noqa: E402

OUT_DIR = base.WORKSPACE / "data/slot_reuse_control_refinement"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_POOL = base.ALIAS_POOL[:32]
HELD_POOL = base.ALIAS_POOL[32:64]
BAL_ANCHOR = [1, 2, 3, 6, 7, 8]      # 3 winner-first, 3 loser-first
BAL_PROBE = [4, 5, 9, 10]            # 2 winner-first, 2 loser-first
BAL_HELD = [16, 17, 19, 20]          # 2 winner-first, 2 loser-first
EXPOSURE_MODES = ["absent_control", "pure_match", "ordered_neutral", "anchor_random", "anchor_true", "anchor_flip"]
ARMS = [("zero", 0), ("true", 8), ("anti", 8)]
N_SEEDS = 3
TRAIN_EVENT_LIMIT = 200
EVAL_TRAIN_EVENT_LIMIT = 140
EVAL_HELD_EVENT_LIMIT = 140
NEUTRAL_TEMPLATES = [
    "{A} and {B} are listed together in a note.",
    "A note names {A} with {B}.",
    "The record mentions {A} beside {B}.",
    "{A} appears in the same line as {B}.",
    "A sentence contains both {A} and {B}.",
    "The entry includes {A} and also {B}.",
]


def pool_pair(pool: List[str], key: str) -> Tuple[str, str]:
    i = base.stable_int("A|" + key) % len(pool)
    j = base.stable_int("B|" + key) % (len(pool) - 1)
    if j >= i:
        j += 1
    return pool[i], pool[j]


def first_role(tid: int) -> str:
    # Metadata from base templates: w/m contexts start with winner, l contexts with loser.
    return "L" if base.TEMPLATES[tid][1] == "l" else "W"


def render_rel(ev: dict, tid: int, hd: str, aa: str, bb: str, mode: str, namespace: str, label_mode: str = "true", seed: int = 0) -> dict:
    context = base.render_context(tid, ev, aa, bb)
    hyp = f"{aa} defeated {bb}." if hd == "AB" else f"{bb} defeated {aa}."
    label = 1 if (hd == "AB" and ev["winner_label"] == "A") or (hd == "BA" and ev["winner_label"] == "B") else 0
    if label_mode == "flip":
        label = 1 - label
    elif label_mode == "random":
        bit = base.stable_int(f"RANDREL|{seed}|{mode}|{ev['event_id']}|T{tid}") % 2
        label = bit if hd == "AB" else 1 - bit
    elif label_mode == "true":
        pass
    else:
        raise ValueError(label_mode)
    return {
        "row_key": f"{namespace}|{mode}|{ev['event_id']}|T{tid}|{hd}",
        "mode": mode,
        "event_id": ev["event_id"],
        "family_id": ev["family_id"],
        "tid": tid,
        "first_role": first_role(tid),
        "template_group": base.TEMPLATES[tid][2],
        "hyp_dir": hd,
        "text": f"{context} [SEP] {hyp}",
        "label": int(label),
    }


def make_relation_rows(events: List[dict], tids: List[int], pool: List[str], namespace: str, mode: str, label_mode: str = "true", salt: str = "", seed: int = 0) -> List[dict]:
    rows=[]
    for ev in events:
        aa, bb = pool_pair(pool, salt + ev["family_id"])
        for tid in tids:
            for hd in ["AB","BA"]:
                rows.append(render_rel(ev, tid, hd, aa, bb, mode, namespace, label_mode, seed))
    return rows


def make_sparse_rows(events: List[dict], pool: List[str], k: int, arm: str, seed: int, mode: str, salt: str = "train|") -> List[dict]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    rows=[]
    for tid in BAL_PROBE:
        evs=events[:]; rng.shuffle(evs)
        for ev in evs[:min(k,len(evs))]:
            aa, bb = pool_pair(pool, salt + ev["family_id"])
            for hd in ["AB","BA"]:
                r = render_rel(ev, tid, hd, aa, bb, mode, f"sparse_{arm}_{k}", "true", seed)
                if arm == "true":
                    pass
                elif arm == "anti":
                    r["label"] = 1 - r["label"]
                else:
                    raise ValueError(arm)
                r["train_kind"] = f"probe_{arm}"
                rows.append(r)
    return rows


def make_pure_match_rows(events: List[dict], pool: List[str], namespace: str, mode: str, salt: str) -> List[dict]:
    rows=[]
    # 6 pseudo templates x 2 rows per event = relation anchor row count with BAL_ANCHOR.
    for ev in events:
        aa, bb = pool_pair(pool, salt + ev["family_id"])
        for j in range(len(BAL_ANCHOR)):
            rows.append({"row_key":f"{namespace}|{mode}|{ev['event_id']}|M{j}|sameA", "mode":mode, "event_id":ev["event_id"], "family_id":ev["family_id"], "tid":-200-j, "first_role":"NA", "template_group":"pure_match", "hyp_dir":"same", "text":f"The name is {aa}. [SEP] The name is {aa}.", "label":1, "train_kind":"pure_match"})
            rows.append({"row_key":f"{namespace}|{mode}|{ev['event_id']}|M{j}|diff", "mode":mode, "event_id":ev["event_id"], "family_id":ev["family_id"], "tid":-200-j, "first_role":"NA", "template_group":"pure_match", "hyp_dir":"diff", "text":f"The name is {aa}. [SEP] The name is {bb}.", "label":0, "train_kind":"pure_match"})
    return rows


def make_ordered_neutral_rows(events: List[dict], pool: List[str], namespace: str, mode: str, salt: str) -> List[dict]:
    rows=[]
    for ev in events:
        aa, bb = pool_pair(pool, salt + ev["family_id"])
        for j, tpl in enumerate(NEUTRAL_TEMPLATES):
            context = tpl.format(A=aa, B=bb)
            for hd in ["AB","BA"]:
                hyp = f"{aa} is named before {bb}." if hd == "AB" else f"{bb} is named before {aa}."
                rows.append({"row_key":f"{namespace}|{mode}|{ev['event_id']}|N{j}|{hd}", "mode":mode, "event_id":ev["event_id"], "family_id":ev["family_id"], "tid":-100-j, "first_role":"NA", "template_group":"ordered_neutral", "hyp_dir":hd, "text":f"{context} [SEP] {hyp}", "label":1 if hd=="AB" else 0, "train_kind":"ordered_neutral"})
    return rows


def exposure_rows(mode: str, events: List[dict]) -> List[dict]:
    if mode == "absent_control":
        return make_pure_match_rows(events, TRAIN_POOL, "extra_train_pool_match", mode, "extra_train|")
    if mode == "pure_match":
        return make_pure_match_rows(events, HELD_POOL, "held_pure_match", mode, "held_exp|")
    if mode == "ordered_neutral":
        return make_ordered_neutral_rows(events, HELD_POOL, "held_ordered_neutral", mode, "held_exp|")
    if mode == "anchor_random":
        rs = make_relation_rows(events, BAL_ANCHOR, HELD_POOL, "held_anchor_random", mode, "random", "held_exp|", seed=26077)
    elif mode == "anchor_true":
        rs = make_relation_rows(events, BAL_ANCHOR, HELD_POOL, "held_anchor_true", mode, "true", "held_exp|")
    elif mode == "anchor_flip":
        rs = make_relation_rows(events, BAL_ANCHOR, HELD_POOL, "held_anchor_flip", mode, "flip", "held_exp|")
    else:
        raise ValueError(mode)
    for r in rs:
        r["train_kind"] = mode
    return rs


def token_counts(rows: List[dict], pool: List[str]) -> Dict[str, Any]:
    c=Counter(); ps=set(pool)
    for r in rows:
        c.update(base.tokenize(r["text"]))
    vals=[c[t] for t in ps]
    return {"types":len(ps), "covered":sum(v>0 for v in vals), "total":int(sum(vals)), "mean":round(float(np.mean(vals)),2), "min":int(min(vals)), "max":int(max(vals))}


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals=[float(v) for v in vals if not (isinstance(v,float) and math.isnan(v))]
    return {"mean":round(float(np.mean(vals)),4), "std":round(float(np.std(vals)),4), "n":len(vals), "values":[round(v,4) for v in vals]} if vals else {"mean":float("nan"),"std":float("nan"),"n":0,"values":[]}


def main() -> None:
    t0=time.time()
    if base.DEVICE == "cuda": torch.cuda.set_device(0)
    torch.set_num_threads(min(32, max(1, torch.get_num_threads())))
    train_fams, held_fams = base.load_families()
    all_train=base.extract_events(train_fams); all_held=base.extract_events(held_fams)
    train_events=base.select_events(all_train, TRAIN_EVENT_LIMIT, 2601)
    eval_train_events=base.select_events(all_train, EVAL_TRAIN_EVENT_LIMIT, 2602)
    eval_held_events=base.select_events(all_held, EVAL_HELD_EVENT_LIMIT, 2603)
    print(json.dumps({"status":"SLOT_REUSE_CONTROL_REFINEMENT_START","device":base.DEVICE,"train_events":len(train_events),"eval_held":len(eval_held_events),"BAL_ANCHOR":BAL_ANCHOR,"BAL_PROBE":BAL_PROBE,"modes":EXPOSURE_MODES,"arms":ARMS,"n_seeds":N_SEEDS}), flush=True)
    datasets={}; rows_by_mode=defaultdict(list); counts={}
    for mode in EXPOSURE_MODES:
        base_anchor = make_relation_rows(train_events, BAL_ANCHOR, TRAIN_POOL, "base_anchor_trainpool", mode, "true", "train|")
        for r in base_anchor: r["train_kind"]="anchor_trainpool"
        extra = exposure_rows(mode, train_events)
        evals = {
            "probe_heldpool_bal": make_relation_rows(eval_held_events, BAL_PROBE, HELD_POOL, "eval_probe_heldpool", mode, "true", "held_eval|"),
            "probe_heldpool_Wfirst": make_relation_rows(eval_held_events, [t for t in BAL_PROBE if first_role(t)=="W"], HELD_POOL, "eval_probe_heldpool_W", mode, "true", "held_eval|"),
            "probe_heldpool_Lfirst": make_relation_rows(eval_held_events, [t for t in BAL_PROBE if first_role(t)=="L"], HELD_POOL, "eval_probe_heldpool_L", mode, "true", "held_eval|"),
            "anchor_heldpool_bal": make_relation_rows(eval_held_events, BAL_ANCHOR, HELD_POOL, "eval_anchor_heldpool", mode, "true", "held_eval|"),
            "heldtemplate_heldpool_bal": make_relation_rows(eval_held_events, BAL_HELD, HELD_POOL, "eval_heldtemplate_heldpool", mode, "true", "held_eval|"),
            "probe_trainpool_heldfam_bal": make_relation_rows(eval_held_events, BAL_PROBE, TRAIN_POOL, "eval_probe_trainpool_heldfam", mode, "true", "train_eval|"),
            "probe_trainpool_trainfam_bal": make_relation_rows(eval_train_events, BAL_PROBE, TRAIN_POOL, "eval_probe_trainpool_trainfam", mode, "true", "train|"),
        }
        datasets[mode] = {"base_anchor":base_anchor, "extra":extra, **evals}
        rows_by_mode[mode].extend(base_anchor + extra)
        for e in evals.values(): rows_by_mode[mode].extend(e)
        for arm,k in ARMS:
            if k: rows_by_mode[mode].extend(make_sparse_rows(train_events, TRAIN_POOL, k, arm, 2609, mode, "train|"))
        counts[mode] = token_counts(base_anchor+extra, HELD_POOL)
        print(json.dumps({"event":"mode", "mode":mode, "base_anchor_rows":len(base_anchor), "extra_rows":len(extra), "held_pool_counts":counts[mode]}), flush=True)
    vocabs = base.build_vocab(rows_by_mode)
    raw=[]
    for mode in EXPOSURE_MODES:
        print(f"\n=== MODE {mode} vocab={len(vocabs[mode])} ===", flush=True)
        base_anchor=datasets[mode]["base_anchor"]; extra=datasets[mode]["extra"]
        eval_sets={k:v for k,v in datasets[mode].items() if k not in {"base_anchor","extra"}}
        for arm,k in ARMS:
            for si in range(N_SEEDS):
                sparse = make_sparse_rows(train_events, TRAIN_POOL, k, arm, 260500+si*101+base.stable_int(mode+arm)%997, mode, "train|") if k else []
                train_rows = base_anchor + extra + sparse
                seed=260800+si*1213+base.stable_int(mode+arm)%883
                res=base.train_eval(train_rows, eval_sets, vocabs[mode], seed)
                res.update({"mode":mode,"arm":arm,"k":k,"seed_idx":si,"seed":seed,"n_train_rows":len(train_rows),"n_extra_rows":len(extra),"n_sparse_rows":len(sparse),"train_fit":bool(res["train_acc"]>=base.TRAIN_FIT)})
                raw.append(res)
                print(json.dumps({"mode":mode,"arm":arm,"seed_idx":si,"train":round(res["train_acc"],4),"probeH":round(res["probe_heldpool_bal"],4),"W":round(res["probe_heldpool_Wfirst"],4),"L":round(res["probe_heldpool_Lfirst"],4),"anchorH":round(res["anchor_heldpool_bal"],4),"fit":res["train_fit"]}), flush=True)
    grouped=defaultdict(list)
    for r in raw: grouped[(r["mode"],r["arm"])].append(r)
    metrics=["train_acc","probe_heldpool_bal","probe_heldpool_Wfirst","probe_heldpool_Lfirst","anchor_heldpool_bal","heldtemplate_heldpool_bal","probe_trainpool_heldfam_bal","probe_trainpool_trainfam_bal"]
    summary={}
    for (mode,arm), items in grouped.items():
        key=f"{mode}|{arm}"; summary[key]={"mode":mode,"arm":arm,"n_runs":len(items),"fit_count":sum(x["train_fit"] for x in items),"n_train_rows":items[0]["n_train_rows"],"n_extra_rows":items[0]["n_extra_rows"]}
        for m in metrics: summary[key][m]=summarize([x[m] for x in items])
    sep={}
    for mode in EXPOSURE_MODES:
        z=summary[f"{mode}|zero"]["probe_heldpool_bal"]["mean"]; t=summary[f"{mode}|true"]["probe_heldpool_bal"]["mean"]; a=summary[f"{mode}|anti"]["probe_heldpool_bal"]["mean"]
        sep[mode]={"zero":z,"true":t,"anti":a,"true_minus_anti":round(t-a,4),"true_minus_zero":round(t-z,4),"held_count":counts[mode]}
    out={"status":"SLOT_REUSE_CONTROL_REFINEMENT","elapsed_seconds":round(time.time()-t0,2),"purpose":"Matched-update, order-balanced controls for token familiarity, ordered matching, random relation cooccurrence, and oriented relation-slot exposure.","config":{"device":base.DEVICE,"modes":EXPOSURE_MODES,"arms":ARMS,"n_seeds":N_SEEDS,"train_event_limit":TRAIN_EVENT_LIMIT,"eval_held_event_limit":EVAL_HELD_EVENT_LIMIT,"balanced_anchor":BAL_ANCHOR,"balanced_probe":BAL_PROBE,"balanced_held":BAL_HELD,"train_pool":TRAIN_POOL,"held_pool":HELD_POOL,"vocab_sizes":{m:len(v) for m,v in vocabs.items()}},"held_pool_train_counts":counts,"summary":summary,"separation":sep,"raw":raw,"interpretation":"All modes have the same extra row count. Balanced template sets reduce mention-order imbalance. anchor_random separates relation-token cooccurrence from true orientation; pure_match separates token identity matching from two-argument order or relation exposure."}
    out_json=OUT_DIR/"slot_reuse_control_refinement_summary.json"; out_json.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    lines=["# research slot reuse control refinement\n\n",f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s. All modes have matched extra-row count.\n\n","## Held-pool exposure\n\n| mode | held types | total held occ | mean | min | max |\n|---|---:|---:|---:|---:|---:|\n"]
    for m,c in counts.items(): lines.append(f"| {m} | {c['covered']}/{c['types']} | {c['total']} | {c['mean']:.1f} | {c['min']} | {c['max']} |\n")
    lines.append("\n## Balanced held-pool probe accuracy\n\n| mode | arm | fit | train | probe | W-first | L-first | anchor | held-template |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for m in EXPOSURE_MODES:
        for arm,_ in ARMS:
            s=summary[f"{m}|{arm}"]
            lines.append(f"| {m} | {arm} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldpool_bal']['mean']:.3f} | {s['probe_heldpool_Wfirst']['mean']:.3f} | {s['probe_heldpool_Lfirst']['mean']:.3f} | {s['anchor_heldpool_bal']['mean']:.3f} | {s['heldtemplate_heldpool_bal']['mean']:.3f} |\n")
    lines.append("\n## Coordinate separation\n\n| mode | zero | true | anti | true-anti | true-zero |\n|---|---:|---:|---:|---:|---:|\n")
    for m,v in sep.items(): lines.append(f"| {m} | {v['zero']:.3f} | {v['true']:.3f} | {v['anti']:.3f} | {v['true_minus_anti']:+.3f} | {v['true_minus_zero']:+.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR/"slot_reuse_control_refinement_summary.md").write_text("".join(lines),encoding="utf-8")
    print("\n"+"="*100); print("research SLOT REUSE CONTROL REFINEMENT SUMMARY")
    for m,v in sep.items(): print(f"{m:<15s} zero={v['zero']:.3f} true={v['true']:.3f} anti={v['anti']:.3f} true-anti={v['true_minus_anti']:+.3f} held_occ={v['held_count']['total']}")
    print(json.dumps({"status":out["status"],"elapsed_seconds":out["elapsed_seconds"],"summary_json":str(out_json)},indent=2))

if __name__ == "__main__": main()
