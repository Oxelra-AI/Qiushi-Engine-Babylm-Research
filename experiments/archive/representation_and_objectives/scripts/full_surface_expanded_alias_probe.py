#!/usr/bin/env python3
"""research full-surface expanded alias probe.

The main identity-orbit run used minimal relation sentences. This probe keeps the
research-style score-ablated event surface (dates, tournaments, rounds) while
expanding each source event across all train/probe/held templates. It isolates
whether per-item alias randomization + sparse aligned anchors survives natural
surface clutter when the template coverage is comparable to the minimal run.
"""

from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
sys.path.insert(0, str(SCRIPT_DIR))
import identity_orbit_anchor_test as base  # noqa: E402

OUT_DIR = base.WORKSPACE / "data/full_surface_expanded_alias_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean unique full-sentence templates from research, preserving the research verbal repertoire.
FULL_TEMPLATES: Dict[int, Tuple[str, str, str]] = {
    1:  ("On {D}, {W} defeated {L} in the {R} of {T}.", "w", "anchor"),
    2:  ("{W} beat {L} at {T} during the {R} on {D}.", "w", "anchor"),
    3:  ("In the {R} at {T} on {D}, {W} won against {L}.", "w", "anchor"),
    4:  ("{W} overcame {L} in the {R} of {T} on {D}.", "w", "probe"),
    5:  ("{W} proved too strong for {L} in the {R} at {T} on {D}.", "w", "probe"),
    6:  ("On {D} at {T}, {L} lost to {W} in the {R}.", "l", "anchor"),
    7:  ("{L} fell to {W} during the {R} at {T} on {D}.", "l", "anchor"),
    8:  ("In the {R} of {T} on {D}, {L} was defeated by {W}.", "l", "anchor"),
    9:  ("{L} was beaten by {W} at {T} in the {R} on {D}.", "l", "probe"),
    10: ("In the {R} at {T} on {D}, {L} was unable to overcome {W}.", "l", "probe"),
    11: ("At {T} on {D}, the {R} saw {W} triumph over {L}.", "m", "anchor"),
    12: ("{W} emerged victorious over {L} in the {R} at {T} on {D}.", "m", "anchor"),
    13: ("During the {R} at {T} on {D}, {W} prevailed against {L}.", "m", "anchor"),
    14: ("It was {W} who came out on top against {L} in the {R} of {T} on {D}.", "m", "anchor"),
    15: ("The {R} of {T} on {D} ended with {W} victorious over {L}.", "m", "probe"),
    16: ("{W} edged out {L} in the {R} of {T} on {D}.", "w", "held"),
    17: ("{L} succumbed to {W} at {T} during the {R} on {D}.", "l", "held"),
    18: ("A {R} contest at {T} on {D} resulted in a victory for {W} over {L}.", "m", "held"),
    19: ("{W} claimed the win against {L} in the {R} of {T} on {D}.", "w", "held"),
    20: ("At {T} on {D}, {L} went down to {W} in the {R}.", "l", "held"),
}
ANCHOR_TEMPLATES = [tid for tid, (_, _, g) in FULL_TEMPLATES.items() if g == "anchor"]
PROBE_TEMPLATES = [tid for tid, (_, _, g) in FULL_TEMPLATES.items() if g == "probe"]
HELD_TEMPLATES = [tid for tid, (_, _, g) in FULL_TEMPLATES.items() if g == "held"]

MODES = ["fixed_names", "per_item_alias"]
ARM_K = [("zero", 0), ("true", 8), ("shuffled", 8), ("exposure", 8)]
N_SEEDS = 5

# Use longer sequences than the minimal run.
base.MAX_LEN = 80
base.EMB_DIM = 48
base.HIDDEN_DIM = 48
base.EPOCHS = 18
base.BATCH_SIZE = 128
base.LR = 0.003


def extract_events(families: List[dict]) -> List[dict]:
    evs = []
    for fam in families:
        for ck in ["context1", "context2"]:
            cx = fam[ck]
            raw = cx.get("event_raw", {})
            evs.append({
                "event_id": f"{fam['family_id']}::{ck}",
                "family_id": fam["family_id"],
                "participant_a": fam["participant_a"],
                "participant_b": fam["participant_b"],
                "winner_label": cx["winner_label"],
                "date": cx.get("date_formatted", raw.get("date_raw", "date")),
                "round": cx.get("round_normalized", raw.get("round", "round")),
                "tournament": raw.get("tournament", "event"),
            })
    return evs


def aliases(mode: str, ev: dict, row_key: str) -> Tuple[str, str]:
    if mode == "fixed_names":
        return ev["participant_a"], ev["participant_b"]
    if mode == "per_item_alias":
        return base.item_alias_map(row_key)
    raise ValueError(mode)


def render_row(ev: dict, tid: int, hyp_dir: str, mode: str, namespace: str) -> dict:
    row_key = f"{namespace}|{ev['event_id']}|T{tid}|{hyp_dir}|{mode}"
    aa, bb = aliases(mode, ev, row_key)
    if ev["winner_label"] == "A":
        W, L = aa, bb
    else:
        W, L = bb, aa
    pat, _, group = FULL_TEMPLATES[tid]
    context = pat.format(W=W, L=L, D=ev["date"], R=ev["round"], T=ev["tournament"])
    hyp = f"{aa} defeated {bb}." if hyp_dir == "AB" else f"{bb} defeated {aa}."
    label = 1 if (hyp_dir == "AB" and ev["winner_label"] == "A") or (hyp_dir == "BA" and ev["winner_label"] == "B") else 0
    return {"row_key": row_key, "mode": mode, "event_id": ev["event_id"], "family_id": ev["family_id"], "tid": tid, "template_group": group, "hyp_dir": hyp_dir, "text": f"{context} [SEP] {hyp}", "label": label}


def make_rows(events: List[dict], tids: List[int], mode: str, namespace: str) -> List[dict]:
    return [render_row(ev, tid, hd, mode, namespace) for ev in events for tid in tids for hd in ["AB", "BA"]]


def make_sparse(events: List[dict], mode: str, k: int, arm: str, seed: int) -> List[dict]:
    if k <= 0:
        return []
    rng = random.Random(seed)
    out = []
    for tid in PROBE_TEMPLATES:
        evs = events[:]
        rng.shuffle(evs)
        for ev in evs[: min(k, len(evs))]:
            for hd in ["AB", "BA"]:
                r = render_row(ev, tid, hd, mode, f"full_sparse_{arm}_{k}")
                if arm == "shuffled":
                    r["label"] = 1 - r["label"]
                elif arm == "exposure":
                    bit = base.stable_int(f"FULL_EXP|{seed}|{ev['event_id']}|T{tid}") % 2
                    r["label"] = bit if hd == "AB" else 1 - bit
                elif arm != "true":
                    raise ValueError(arm)
                r["train_kind"] = f"probe_{arm}"
                out.append(r)
    return out


def summarize(vals):
    vals = [float(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]} if vals else {"mean": float("nan"), "std": float("nan"), "n": 0}


def main():
    t0 = time.time()
    if base.DEVICE == "cuda":
        torch.cuda.set_device(0)
    torch.set_num_threads(min(16, max(1, torch.get_num_threads())))
    train_fams, held_fams = base.load_families()
    train_events = base.select_events(extract_events(train_fams), base.TRAIN_EVENT_LIMIT, 257)
    eval_train_events = base.select_events(extract_events(train_fams), base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(extract_events(held_fams), base.EVAL_HELD_EVENT_LIMIT, 2257)
    print(json.dumps({"status": "FULL_SURFACE_EXPANDED_ALIAS_PROBE_START", "device": base.DEVICE, "modes": MODES, "arm_k": ARM_K, "n_seeds": N_SEEDS}), flush=True)

    datasets = {}
    rows_by_mode = defaultdict(list)
    for mode in MODES:
        base_anchor = make_rows(train_events, ANCHOR_TEMPLATES, mode, "full_train_anchor")
        for r in base_anchor:
            r["train_kind"] = "anchor"
        evals = {
            "anchor_heldfam": make_rows(eval_held_events, ANCHOR_TEMPLATES, mode, "full_eval_anchor"),
            "probe_heldfam": make_rows(eval_held_events, PROBE_TEMPLATES, mode, "full_eval_probe"),
            "probe_trainfam": make_rows(eval_train_events, PROBE_TEMPLATES, mode, "full_eval_probe_trainfam"),
            "heldtemplate_heldfam": make_rows(eval_held_events, HELD_TEMPLATES, mode, "full_eval_heldtemplate"),
        }
        datasets[mode] = {"base_anchor": base_anchor, "evals": evals}
        rows_by_mode[mode].extend(base_anchor)
        for rs in evals.values():
            rows_by_mode[mode].extend(rs)
        for arm, k in ARM_K:
            if k:
                rows_by_mode[mode].extend(make_sparse(train_events, mode, k, arm, 999))
    vocabs = base.build_vocab(rows_by_mode)

    raw = []
    for mode in MODES:
        print(f"\n=== {mode} vocab={len(vocabs[mode])} ===", flush=True)
        for arm, k in ARM_K:
            for seed_idx in range(N_SEEDS):
                seed = 257500 + seed_idx * 1117 + k * 43 + base.stable_int(mode + arm) % 809
                sparse = make_sparse(train_events, mode, k, arm, seed) if k else []
                train_rows = datasets[mode]["base_anchor"] + sparse
                res = base.train_eval(train_rows, datasets[mode]["evals"], vocabs[mode], seed)
                res.update({"mode": mode, "arm": arm, "k_per_template": k, "seed_idx": seed_idx, "seed": seed, "n_train_rows": len(train_rows), "n_sparse_rows": len(sparse), "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT)})
                raw.append(res)
                print(json.dumps({"mode": mode, "arm": arm, "k": k, "seed_idx": seed_idx, "train": round(res["train_acc"], 4), "probeH": round(res["probe_heldfam"], 4), "heldT": round(res["heldtemplate_heldfam"], 4), "fit": res["train_fit"]}), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"], r["k_per_template"])].append(r)
    metrics = ["train_acc", "anchor_heldfam", "probe_heldfam", "probe_trainfam", "heldtemplate_heldfam"]
    summary = {}
    for (mode, arm, k), items in grouped.items():
        key = f"{mode}|{arm}|k{k}"
        summary[key] = {"mode": mode, "arm": arm, "k_per_template": k, "n_runs": len(items), "fit_count": sum(x["train_fit"] for x in items)}
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])
    out = {"status": "FULL_SURFACE_EXPANDED_ALIAS_PROBE", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": round(time.time() - t0, 2), "config": {"modes": MODES, "arm_k": ARM_K, "n_seeds": N_SEEDS, "max_len": base.MAX_LEN, "device": base.DEVICE}, "summary": summary, "raw": raw}
    out_json = OUT_DIR / "full_surface_expanded_alias_probe_summary.json"
    out_json.write_text(json.dumps(out, indent=2, default=str))
    lines = ["# research full-surface expanded alias probe\n\n", f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s. Full research-style sentence templates expanded over the same events as the minimal run.\n\n", "| mode | arm | k/template | fit | train | probe-held | probe-trainfam | held-template |\n", "|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for mode in MODES:
        for arm, k in ARM_K:
            s = summary[f"{mode}|{arm}|k{k}"]
            lines.append(f"| {mode} | {arm} | {k} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldfam']['mean']:.3f} | {s['probe_trainfam']['mean']:.3f} | {s['heldtemplate_heldfam']['mean']:.3f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "full_surface_expanded_alias_probe_summary.md").write_text("".join(lines))
    print("\nFULL-SURFACE EXPANDED ALIAS PROBE SUMMARY")
    for mode in MODES:
        for arm, k in ARM_K:
            s = summary[f"{mode}|{arm}|k{k}"]
            print(f"{mode:<15s} {arm:<9s} k={k:<2d} fit={s['fit_count']}/{s['n_runs']} train={s['train_acc']['mean']:.3f} probeH={s['probe_heldfam']['mean']:.3f} probeTrain={s['probe_trainfam']['mean']:.3f} heldT={s['heldtemplate_heldfam']['mean']:.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
