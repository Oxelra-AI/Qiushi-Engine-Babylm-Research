#!/usr/bin/env python3
"""research independent verification of the research identity-orbit result.

Two things are checked here, both of which the research note left open or assumed.

Part A — construction audits that could invalidate the interpretation:
  A1 alias generation must not depend on the label. Verified structurally (row_key
     excludes winner information) and empirically: a predictor built only from alias
     surface order/identity must stay at chance on training rows.
  A2 context and hypothesis inside one item must share the same alias assignment
     (coreference preserved), and the two participants must never collide.
  A3 train/held family separation must be exact, and in fixed_names mode the
     participant-name overlap between the two splits is measured rather than assumed.
  A4 probe/held template groups must never appear in the anchor training rows.

Part B — the confound named in the research note: realization diversity was varied
together with training row count. Here total anchor rows, alias-token pool, sparse
anchor count, optimizer, model, and seeds are held fixed while diversity D (number
of distinct anchor predicates per source event) is traded against event breadth:

    rows = events x D x 2   with rows fixed, so events = rows / (2 D)

If the mechanism needs several realizations of the same relation, transfer should
rise with D at fixed rows. If it only needs many labeled rows, D should not matter.
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

OUT_DIR = base.WORKSPACE / "data/identity_orbit_verification"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["fixed_names", "family_alias", "per_item_alias"]
ARMS = [("zero", 0), ("true", 8), ("shuffled", 8)]
DIVERSITY = [1, 2, 5, 10]
FIXED_ANCHOR_ROWS = 1600  # events x D x 2 hypotheses
N_SEEDS = 3


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]}


# ─────────────────────────── Part A: construction audits ───────────────────────────

def audit_alias_label_independence(events: List[dict], mode: str, tids: List[int]) -> Dict[str, Any]:
    """Alias identity must carry no label information."""
    rows = base.make_rows(events, tids, mode, "audit_alias")
    # Feature 1: does the first participant alias sort before the second?
    order_bits = []
    labels = []
    first_alias_tokens = []
    for r in rows:
        aa, bb = base.aliases_for(mode, next(e for e in events if e["event_id"] == r["event_id"]), r["row_key"])
        order_bits.append(1 if aa.lower() < bb.lower() else 0)
        first_alias_tokens.append(aa.lower())
        labels.append(r["label"])
    order_bits = np.array(order_bits)
    labels = np.array(labels)
    agree = float((order_bits == labels).mean())
    # Feature 2: best single-alias majority predictor on the training rows.
    by_alias = defaultdict(list)
    for tok, lab in zip(first_alias_tokens, labels):
        by_alias[tok].append(lab)
    majority_hits = sum(max(sum(v), len(v) - sum(v)) for v in by_alias.values())
    majority_acc = majority_hits / max(1, len(labels))
    return {
        "mode": mode,
        "n_rows": int(len(labels)),
        "label_mean": round(float(labels.mean()), 4),
        "alias_order_predicts_label_acc": round(agree, 4),
        "best_first_alias_majority_acc": round(float(majority_acc), 4),
        "n_distinct_first_aliases": len(by_alias),
    }


def audit_within_item_coreference(events: List[dict], mode: str, tids: List[int]) -> Dict[str, Any]:
    """Context and hypothesis of one item must use the same two aliases, distinct from each other."""
    rows = base.make_rows(events, tids, mode, "audit_coref")
    collisions = 0
    shared_ok = 0
    for r in rows:
        ev = next(e for e in events if e["event_id"] == r["event_id"])
        aa, bb = base.aliases_for(mode, ev, r["row_key"])
        if aa == bb:
            collisions += 1
        ctx, hyp = r["text"].split(" [SEP] ")
        if aa in ctx and bb in ctx and aa in hyp and bb in hyp:
            shared_ok += 1
    return {"mode": mode, "n_rows": len(rows), "alias_collisions": collisions, "context_hypothesis_alias_shared": shared_ok, "shared_fraction": round(shared_ok / max(1, len(rows)), 4)}


def audit_split_separation(train_fams: List[dict], held_fams: List[dict]) -> Dict[str, Any]:
    tr_ids = {f["family_id"] for f in train_fams}
    hd_ids = {f["family_id"] for f in held_fams}
    tr_names = {n for f in train_fams for n in (f["participant_a"], f["participant_b"])}
    hd_names = {n for f in held_fams for n in (f["participant_a"], f["participant_b"])}
    return {
        "train_families": len(tr_ids),
        "held_families": len(hd_ids),
        "family_id_overlap": len(tr_ids & hd_ids),
        "train_participants": len(tr_names),
        "held_participants": len(hd_names),
        "participant_name_overlap": len(tr_names & hd_names),
        "held_names_also_in_train_fraction": round(len(tr_names & hd_names) / max(1, len(hd_names)), 4),
    }


def audit_template_leakage() -> Dict[str, Any]:
    a = set(base.ANCHOR_TEMPLATES); p = set(base.PROBE_TEMPLATES); h = set(base.HELD_TEMPLATES)
    return {
        "anchor": sorted(a), "probe": sorted(p), "held": sorted(h),
        "anchor_probe_overlap": sorted(a & p), "anchor_held_overlap": sorted(a & h), "probe_held_overlap": sorted(p & h),
        "anchor_surface_strings": [base.TEMPLATES[t][0] for t in sorted(a)],
        "probe_surface_strings": [base.TEMPLATES[t][0] for t in sorted(p)],
    }


# ─────────────── Part B: matched-row-count realization diversity ───────────────

def build_diversity_anchor_rows(all_train_events: List[dict], mode: str, D: int, target_rows: int, seed: int) -> Tuple[List[dict], Dict[str, Any]]:
    """Fixed anchor row count; D distinct anchor templates per event; events chosen to hit target."""
    rng = random.Random(seed)
    need_events = target_rows // (2 * D)
    evs = all_train_events[:]
    rng.shuffle(evs)
    if need_events > len(evs):
        raise RuntimeError(f"D={D} needs {need_events} events but only {len(evs)} available")
    chosen = evs[:need_events]
    rows: List[dict] = []
    for ev in chosen:
        tids = base.ANCHOR_TEMPLATES[:]
        rng.shuffle(tids)
        for tid in tids[:D]:
            for hd in ["AB", "BA"]:
                r = base.render_row(ev, tid, hd, mode, f"div{D}_seed{seed}")
                r["train_kind"] = "anchor"
                rows.append(r)
    meta = {
        "D": D, "events_used": len(chosen), "rows": len(rows),
        "template_row_counts": dict(Counter(r["tid"] for r in rows)),
        "label_mean": round(float(np.mean([r["label"] for r in rows])), 4),
    }
    return rows, meta


def main():
    t0 = time.time()
    if base.DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))

    train_fams, held_fams = base.load_families()
    all_train_events = base.extract_events(train_fams)
    all_held_events = base.extract_events(held_fams)
    eval_train_events = base.select_events(all_train_events, base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(all_held_events, base.EVAL_HELD_EVENT_LIMIT, 2257)
    sparse_pool_events = base.select_events(all_train_events, base.TRAIN_EVENT_LIMIT, 257)

    print(json.dumps({"status": "VERIFICATION_START", "device": base.DEVICE,
                      "train_events": len(all_train_events), "held_events": len(all_held_events),
                      "fixed_anchor_rows": FIXED_ANCHOR_ROWS, "diversity": DIVERSITY}), flush=True)

    # ---------- Part A ----------
    audit_events = base.select_events(all_train_events, 120, 999)
    audits = {
        "split_separation": audit_split_separation(train_fams, held_fams),
        "template_groups": audit_template_leakage(),
        "alias_label_independence": {m: audit_alias_label_independence(audit_events, m, base.ANCHOR_TEMPLATES) for m in MODES},
        "within_item_coreference": {m: audit_within_item_coreference(audit_events, m, base.ANCHOR_TEMPLATES) for m in MODES},
    }
    print(json.dumps({"event": "part_a_audits", **{k: audits[k] for k in ["split_separation"]}}), flush=True)
    for m in MODES:
        print(json.dumps({"event": "alias_audit", **audits["alias_label_independence"][m], **{k: v for k, v in audits["within_item_coreference"][m].items() if k != "mode"}}), flush=True)

    # ---------- Part B ----------
    # Shared vocabulary per mode over every row this experiment can produce.
    rows_by_mode = defaultdict(list)
    evals_by_mode = {}
    for mode in MODES:
        evals = {
            "probe_heldfam": base.make_rows(eval_held_events, base.PROBE_TEMPLATES, mode, "v_held_probe"),
            "anchor_heldfam": base.make_rows(eval_held_events, base.ANCHOR_TEMPLATES, mode, "v_held_anchor"),
            "heldtemplate_heldfam": base.make_rows(eval_held_events, base.HELD_TEMPLATES, mode, "v_heldtemplate"),
            "probe_trainfam": base.make_rows(eval_train_events, base.PROBE_TEMPLATES, mode, "v_train_probe"),
        }
        for tid in base.PROBE_TEMPLATES:
            evals[f"probe_T{tid:02d}_heldfam"] = base.make_rows(eval_held_events, [tid], mode, f"v_probeT{tid}")
        evals_by_mode[mode] = evals
        for rs in evals.values():
            rows_by_mode[mode].extend(rs)
        for D in DIVERSITY:
            for s in range(N_SEEDS):
                rs, _ = build_diversity_anchor_rows(all_train_events, mode, D, FIXED_ANCHOR_ROWS, 258000 + s)
                rows_by_mode[mode].extend(rs)
        for arm, k in ARMS:
            if k:
                rows_by_mode[mode].extend(base.make_sparse_probe_rows(sparse_pool_events, mode, k, arm, 999))
    vocabs = base.build_vocab(rows_by_mode)

    raw = []
    diversity_meta = {}
    for mode in MODES:
        print(f"\n=== MODE {mode} vocab={len(vocabs[mode])} ===", flush=True)
        for D in DIVERSITY:
            for arm, k in ARMS:
                for si in range(N_SEEDS):
                    anchor_rows, meta = build_diversity_anchor_rows(all_train_events, mode, D, FIXED_ANCHOR_ROWS, 258000 + si)
                    diversity_meta[f"{mode}|D{D}|s{si}"] = meta
                    seed_sparse = 258500 + si * 97 + k * 31 + base.stable_int(mode + arm) % 401
                    sparse = base.make_sparse_probe_rows(sparse_pool_events, mode, k, arm, seed_sparse) if k else []
                    train_rows = anchor_rows + sparse
                    seed = 258900 + si * 1301 + D * 53 + k * 41 + base.stable_int(mode + arm) % 887
                    res = base.train_eval(train_rows, evals_by_mode[mode], vocabs[mode], seed)
                    res.update({"mode": mode, "D": D, "arm": arm, "k": k, "seed_idx": si, "seed": seed,
                                "events_used": meta["events_used"], "n_anchor_rows": len(anchor_rows),
                                "n_sparse_rows": len(sparse), "n_train_rows": len(train_rows),
                                "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT)})
                    raw.append(res)
                    print(json.dumps({"mode": mode, "D": D, "arm": arm, "seed_idx": si,
                                      "events": meta["events_used"], "rows": len(train_rows),
                                      "train": round(res["train_acc"], 4),
                                      "probeH": round(res["probe_heldfam"], 4),
                                      "anchorH": round(res["anchor_heldfam"], 4),
                                      "heldT": round(res["heldtemplate_heldfam"], 4),
                                      "fit": res["train_fit"]}), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["D"], r["arm"])].append(r)
    metrics = ["train_acc", "anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"]
    metrics += [f"probe_T{tid:02d}_heldfam" for tid in base.PROBE_TEMPLATES]
    summary = {}
    for (mode, D, arm), items in grouped.items():
        key = f"{mode}|D{D}|{arm}"
        summary[key] = {"mode": mode, "D": D, "arm": arm, "n_runs": len(items),
                        "fit_count": sum(x["train_fit"] for x in items),
                        "events_used": items[0]["events_used"], "n_anchor_rows": items[0]["n_anchor_rows"],
                        "n_train_rows": items[0]["n_train_rows"]}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])

    # Aligned-minus-anti separation at each diversity level: the load-bearing contrast.
    separation = {}
    for mode in MODES:
        for D in DIVERSITY:
            t = summary.get(f"{mode}|D{D}|true", {}).get("probe_heldfam", {}).get("mean")
            s = summary.get(f"{mode}|D{D}|shuffled", {}).get("probe_heldfam", {}).get("mean")
            z = summary.get(f"{mode}|D{D}|zero", {}).get("probe_heldfam", {}).get("mean")
            if None not in (t, s, z):
                separation[f"{mode}|D{D}"] = {"zero": z, "true": t, "shuffled": s,
                                              "true_minus_shuffled": round(t - s, 4),
                                              "true_minus_zero": round(t - z, 4)}

    out = {
        "status": "IDENTITY_ORBIT_VERIFICATION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "config": {"modes": MODES, "arms": ARMS, "diversity": DIVERSITY, "fixed_anchor_rows": FIXED_ANCHOR_ROWS,
                   "n_seeds": N_SEEDS, "device": base.DEVICE,
                   "model": {"emb_dim": base.EMB_DIM, "hidden_dim": base.HIDDEN_DIM, "epochs": base.EPOCHS,
                             "batch_size": base.BATCH_SIZE, "lr": base.LR}},
        "part_a_audits": audits,
        "diversity_meta": diversity_meta,
        "summary": summary,
        "aligned_vs_antialigned_separation": separation,
        "raw": raw,
    }
    out_json = OUT_DIR / "identity_orbit_verification_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    lines = ["# research independent verification of the research identity-orbit mechanism\n\n",
             f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s.\n\n",
             "## Part A construction audits\n\n",
             f"- family id overlap: {audits['split_separation']['family_id_overlap']}; participant-name overlap between splits: {audits['split_separation']['participant_name_overlap']} of {audits['split_separation']['held_participants']} held names.\n",
             f"- anchor/probe/held template overlaps: {audits['template_groups']['anchor_probe_overlap']} / {audits['template_groups']['anchor_held_overlap']} / {audits['template_groups']['probe_held_overlap']}.\n\n",
             "| mode | alias-order predicts label | best first-alias majority | alias collisions | ctx/hyp alias shared |\n",
             "|---|---:|---:|---:|---:|\n"]
    for m in MODES:
        a = audits["alias_label_independence"][m]; c = audits["within_item_coreference"][m]
        lines.append(f"| {m} | {a['alias_order_predicts_label_acc']:.3f} | {a['best_first_alias_majority_acc']:.3f} | {c['alias_collisions']} | {c['shared_fraction']:.3f} |\n")
    lines.append("\n## Part B matched-row-count realization diversity\n\n")
    lines.append(f"Total anchor rows fixed at {FIXED_ANCHOR_ROWS}; D distinct anchor predicates per event trades against event breadth.\n\n")
    lines.append("| mode | D | events | rows | arm | fit | train | probe-held | anchor-held | held-template |\n")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|\n")
    for mode in MODES:
        for D in DIVERSITY:
            for arm, k in ARMS:
                s = summary.get(f"{mode}|D{D}|{arm}")
                if not s:
                    continue
                lines.append(f"| {mode} | {D} | {s['events_used']} | {s['n_train_rows']} | {arm} | {s['fit_count']}/{s['n_runs']} | "
                             f"{s['train_acc']['mean']:.3f} | {s['probe_heldfam']['mean']:.3f} | {s['anchor_heldfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append("\n## Aligned minus anti-aligned separation at fixed row count\n\n| mode | D | zero | true | shuffled | true-shuffled |\n|---|---:|---:|---:|---:|---:|\n")
    for key, v in separation.items():
        mode, D = key.split("|D")
        lines.append(f"| {mode} | {D} | {v['zero']:.3f} | {v['true']:.3f} | {v['shuffled']:.3f} | {v['true_minus_shuffled']:+.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "identity_orbit_verification_summary.md").write_text("".join(lines))

    print("\n" + "=" * 96)
    print("research VERIFICATION SUMMARY")
    for key, v in separation.items():
        print(f"{key:<28s} zero={v['zero']:.3f} true={v['true']:.3f} shuf={v['shuffled']:.3f} true-shuf={v['true_minus_shuffled']:+.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
