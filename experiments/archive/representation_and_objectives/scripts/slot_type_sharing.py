#!/usr/bin/env python3
"""research: is the necessary condition identity randomization or argument-slot type sharing?

research's matched-row verification produced a result the research reading does not
explain. At a fixed 1,600 anchor rows:
  fixed_names    aligned-minus-anti  +0.002 / +0.060 / -0.003 / +0.011  (D=1,2,5,10)
  family_alias   aligned-minus-anti  +0.000 / +0.366 / +0.303 / +0.118
  per_item_alias aligned-minus-anti  +0.213 / +0.274 / +0.182 / +0.212
family_alias keeps a persistent identity per family, so per-item resampling is not
necessary. What family_alias and per_item_alias share is that argument-slot tokens
are drawn from one small pool used by both train and held families, while natural
names are almost disjoint across families.

Competing explanations for the transfer failure of natural names:
  H_random  the learner must be unable to associate a label with an identity;
  H_share   the argument slot must be filled by tokens whose distribution is shared
            between training and evaluation, so a relation coordinate learned on
            training rows is applicable at all to held rows;
  H_vocab   fewer distinct identity tokens simply reduce estimation burden.

Arms (all with the same relation grammar, row counts, and anchor structure):
  natural_disjoint     original names; train/held name sets nearly disjoint
  natural_shared_pool  natural names resampled from ONE shared pool of real names,
                       stable per family: identity persists, tokens are shared
  natural_shared_large same construction with a large shared pool: tokens shared but
                       each identity token is rare, separating sharing from vocabulary size
  family_alias         synthetic stable aliases from the shared pool (reference)
  per_item_alias       fresh aliases per item (reference)

If natural_shared_pool recovers aligned-minus-anti separation, the necessary
condition is argument-slot type sharing and H_random is refuted as stated.
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

OUT_DIR = base.WORKSPACE / "data/slot_type_sharing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [("zero", 0), ("true", 8), ("shuffled", 8)]
MODES = ["natural_disjoint", "natural_shared_pool", "natural_shared_large", "family_alias", "per_item_alias"]
N_SEEDS = 4
SHARED_POOL_SIZE = 64          # matches ALIAS_POOL size in the reference modes
LARGE_POOL_SIZE = 420          # shared but each token rare


def natural_name_pool(train_fams: List[dict], held_fams: List[dict], size: int, seed: int) -> List[str]:
    """One pool of real participant surnames used by every family in every split."""
    names = sorted({n.split()[-1] for f in (train_fams + held_fams) for n in (f["participant_a"], f["participant_b"]) if n.split()})
    rng = random.Random(seed)
    rng.shuffle(names)
    return names[: min(size, len(names))]


def install_modes(train_fams: List[dict], held_fams: List[dict]) -> Dict[str, List[str]]:
    """Extend the base module's alias resolution with the shared-natural-pool modes."""
    pools = {
        "natural_shared_pool": natural_name_pool(train_fams, held_fams, SHARED_POOL_SIZE, 4001),
        "natural_shared_large": natural_name_pool(train_fams, held_fams, LARGE_POOL_SIZE, 4002),
    }
    original = base.aliases_for

    def aliases_for(mode: str, ev: dict, row_key: str) -> Tuple[str, str]:
        if mode == "natural_disjoint":
            return ev["participant_a"], ev["participant_b"]
        if mode in pools:
            pool = pools[mode]
            i = base.stable_int(f"A::{mode}::" + ev["family_id"]) % len(pool)
            j = base.stable_int(f"B::{mode}::" + ev["family_id"]) % (len(pool) - 1)
            if j >= i:
                j += 1
            return pool[i], pool[j]
        return original(mode, ev, row_key)

    base.aliases_for = aliases_for
    return pools


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]}


def slot_token_sharing(train_events: List[dict], held_events: List[dict], mode: str) -> Dict[str, Any]:
    """How much of the held-family argument-slot vocabulary is seen in training rows."""
    def slot_tokens(events: List[dict], tids: List[int], ns: str) -> Counter:
        c = Counter()
        for r in base.make_rows(events, tids, mode, ns):
            ev = next(e for e in events if e["event_id"] == r["event_id"])
            aa, bb = base.aliases_for(mode, ev, r["row_key"])
            c.update(base.tokenize(aa))
            c.update(base.tokenize(bb))
        return c
    tr = slot_tokens(train_events, base.ANCHOR_TEMPLATES, "share_train")
    hd = slot_tokens(held_events, base.PROBE_TEMPLATES, "share_held")
    shared = set(tr) & set(hd)
    covered_occ = sum(n for t, n in hd.items() if t in shared)
    return {
        "mode": mode,
        "train_slot_types": len(tr),
        "held_slot_types": len(hd),
        "shared_types": len(shared),
        "held_type_coverage": round(len(shared) / max(1, len(hd)), 4),
        "held_occurrence_coverage": round(covered_occ / max(1, sum(hd.values())), 4),
        "mean_train_occurrences_per_shared_type": round(float(np.mean([tr[t] for t in shared])) if shared else 0.0, 2),
    }


def main() -> None:
    t0 = time.time()
    torch.set_num_threads(min(48, max(1, torch.get_num_threads())))
    train_fams, held_fams = base.load_families()
    pools = install_modes(train_fams, held_fams)

    all_train = base.extract_events(train_fams)
    all_held = base.extract_events(held_fams)
    train_events = base.select_events(all_train, base.TRAIN_EVENT_LIMIT, 257)
    eval_train_events = base.select_events(all_train, base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(all_held, base.EVAL_HELD_EVENT_LIMIT, 2257)

    print(json.dumps({"status": "SLOT_TYPE_SHARING_START", "device": base.DEVICE,
                      "pool_sizes": {k: len(v) for k, v in pools.items()},
                      "train_events": len(train_events), "held_events": len(eval_held_events),
                      "modes": MODES, "arms": ARMS, "n_seeds": N_SEEDS}), flush=True)

    sharing = {m: slot_token_sharing(train_events, eval_held_events, m) for m in MODES}
    for m in MODES:
        print(json.dumps({"event": "slot_sharing", **sharing[m]}), flush=True)

    rows_by_mode: Dict[str, List[dict]] = defaultdict(list)
    datasets: Dict[str, Dict[str, List[dict]]] = {}
    for mode in MODES:
        anchor = base.make_rows(train_events, base.ANCHOR_TEMPLATES, mode, "train_anchor")
        for r in anchor:
            r["train_kind"] = "anchor"
        evals = {
            "probe_heldfam": base.make_rows(eval_held_events, base.PROBE_TEMPLATES, mode, "eval_held_probe"),
            "anchor_heldfam": base.make_rows(eval_held_events, base.ANCHOR_TEMPLATES, mode, "eval_held_anchor"),
            "heldtemplate_heldfam": base.make_rows(eval_held_events, base.HELD_TEMPLATES, mode, "eval_heldtemplate"),
            "probe_trainfam": base.make_rows(eval_train_events, base.PROBE_TEMPLATES, mode, "eval_train_probe"),
        }
        for tid in base.PROBE_TEMPLATES:
            evals[f"probe_T{tid:02d}_heldfam"] = base.make_rows(eval_held_events, [tid], mode, f"eval_probeT{tid}")
        datasets[mode] = {"anchor": anchor, **evals}
        for part in datasets[mode].values():
            rows_by_mode[mode].extend(part)
        for arm, k in ARMS:
            if k:
                rows_by_mode[mode].extend(base.make_sparse_probe_rows(train_events, mode, k, arm, 999))
    vocabs = base.build_vocab(rows_by_mode)

    raw: List[dict] = []
    for mode in MODES:
        print(f"\n=== MODE {mode} vocab={len(vocabs[mode])} ===", flush=True)
        anchor = datasets[mode]["anchor"]
        eval_sets = {k: v for k, v in datasets[mode].items() if k != "anchor"}
        for arm, k in ARMS:
            for si in range(N_SEEDS):
                seed_sparse = 259500 + si * 101 + base.stable_int(mode + arm) % 733
                sparse = base.make_sparse_probe_rows(train_events, mode, k, arm, seed_sparse) if k else []
                train_rows = anchor + sparse
                seed = 259900 + si * 1213 + base.stable_int(mode + arm) % 911
                res = base.train_eval(train_rows, eval_sets, vocabs[mode], seed)
                res.update({"mode": mode, "arm": arm, "k": k, "seed_idx": si, "seed": seed,
                            "n_train_rows": len(train_rows), "n_sparse_rows": len(sparse),
                            "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT)})
                raw.append(res)
                print(json.dumps({"mode": mode, "arm": arm, "seed_idx": si,
                                  "train": round(res["train_acc"], 4),
                                  "probeH": round(res["probe_heldfam"], 4),
                                  "anchorH": round(res["anchor_heldfam"], 4),
                                  "probeTrain": round(res["probe_trainfam"], 4),
                                  "heldT": round(res["heldtemplate_heldfam"], 4),
                                  "fit": res["train_fit"]}), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"])].append(r)
    metrics = ["train_acc", "probe_heldfam", "anchor_heldfam", "probe_trainfam", "heldtemplate_heldfam"]
    metrics += [f"probe_T{tid:02d}_heldfam" for tid in base.PROBE_TEMPLATES]
    summary = {}
    for (mode, arm), items in grouped.items():
        key = f"{mode}|{arm}"
        summary[key] = {"mode": mode, "arm": arm, "n_runs": len(items),
                        "fit_count": sum(x["train_fit"] for x in items),
                        "n_train_rows": items[0]["n_train_rows"]}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])
        fit = [x for x in items if x["train_fit"]]
        if fit:
            summary[key]["probe_heldfam__fit_only"] = summarize([x["probe_heldfam"] for x in fit])

    separation = {}
    for mode in MODES:
        z = summary.get(f"{mode}|zero", {}).get("probe_heldfam", {}).get("mean")
        t = summary.get(f"{mode}|true", {}).get("probe_heldfam", {}).get("mean")
        s = summary.get(f"{mode}|shuffled", {}).get("probe_heldfam", {}).get("mean")
        if None not in (z, t, s):
            separation[mode] = {"zero": z, "true": t, "shuffled": s,
                                "true_minus_shuffled": round(t - s, 4),
                                "true_minus_zero": round(t - z, 4),
                                "held_slot_type_coverage": sharing[mode]["held_type_coverage"],
                                "held_slot_occurrence_coverage": sharing[mode]["held_occurrence_coverage"],
                                "mean_train_occ_per_shared_slot_type": sharing[mode]["mean_train_occurrences_per_shared_type"]}

    out = {
        "status": "SLOT_TYPE_SHARING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "config": {"modes": MODES, "arms": ARMS, "n_seeds": N_SEEDS, "device": base.DEVICE,
                   "shared_pool_size": SHARED_POOL_SIZE, "large_pool_size": LARGE_POOL_SIZE,
                   "pool_sizes": {k: len(v) for k, v in pools.items()},
                   "vocab_sizes": {m: len(v) for m, v in vocabs.items()},
                   "model": {"emb_dim": base.EMB_DIM, "hidden_dim": base.HIDDEN_DIM, "epochs": base.EPOCHS, "lr": base.LR}},
        "slot_token_sharing": sharing,
        "summary": summary,
        "separation": separation,
        "raw": raw,
    }
    out_json = OUT_DIR / "slot_type_sharing_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))

    lines = ["# research argument-slot type sharing versus identity randomization\n\n",
             f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s, {N_SEEDS} seeds per cell.\n\n",
             "## Held-family argument-slot vocabulary coverage\n\n",
             "| mode | held slot types | shared with train | type coverage | occurrence coverage | mean train occ/shared type |\n",
             "|---|---:|---:|---:|---:|---:|\n"]
    for m in MODES:
        s = sharing[m]
        lines.append(f"| {m} | {s['held_slot_types']} | {s['shared_types']} | {s['held_type_coverage']:.3f} | {s['held_occurrence_coverage']:.3f} | {s['mean_train_occurrences_per_shared_type']:.1f} |\n")
    lines.append("\n## Transfer to held-family probe predicates\n\n")
    lines.append("| mode | arm | fit | train | probe-held | anchor-held | probe-trainfam | held-template |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
    for mode in MODES:
        for arm, _ in ARMS:
            s = summary.get(f"{mode}|{arm}")
            if not s:
                continue
            lines.append(f"| {mode} | {arm} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | "
                         f"{s['probe_heldfam']['mean']:.3f} | {s['anchor_heldfam']['mean']:.3f} | "
                         f"{s['probe_trainfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append("\n## Aligned minus anti-aligned, against slot-sharing\n\n| mode | zero | true | shuffled | true-shuffled | held slot type coverage |\n|---|---:|---:|---:|---:|---:|\n")
    for mode, v in separation.items():
        lines.append(f"| {mode} | {v['zero']:.3f} | {v['true']:.3f} | {v['shuffled']:.3f} | {v['true_minus_shuffled']:+.3f} | {v['held_slot_type_coverage']:.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "slot_type_sharing_summary.md").write_text("".join(lines))

    print("\n" + "=" * 100)
    print("research SLOT-TYPE-SHARING SUMMARY")
    for mode, v in separation.items():
        print(f"{mode:<22s} cov={v['held_slot_type_coverage']:.3f} zero={v['zero']:.3f} true={v['true']:.3f} shuf={v['shuffled']:.3f} true-shuf={v['true_minus_shuffled']:+.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
