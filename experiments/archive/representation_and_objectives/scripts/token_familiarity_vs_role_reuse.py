#!/usr/bin/env python3
"""research controlled separation: ordinary token familiarity vs relation-role reuse.

research showed that shared argument-slot tokens recover held-family transfer, while
natural disjoint names suppress it.  A remaining ambiguity is whether this is only
ordinary token familiarity (the eval filler embeddings were trained somewhere) or
whether the fillers must be trained as reusable *argument slots in oriented relation
contexts*.

This cheap raw-text diagnostic keeps the main predicate/anchor protocol from
research/259 but makes evaluation fillers come from a held-only alias pool.  The
base anchor and sparse probe-orientation rows use a disjoint train pool.  We then
add matched exposure rows for the held-only pool:

  absent       : held eval tokens never occur in training;
  neutral      : held eval tokens occur in a balanced non-relational/coreference task;
  anchor_true  : held tokens occur as arguments of the already-oriented anchor
                 predicates with true labels on separate train events;
  anchor_flip  : same surface/quantity as anchor_true but labels flipped.

All modes still cross sparse probe anchors {zero,true,shuffled}.  If neutral
exposure does not recover transfer while anchor_true does, token familiarity alone
is insufficient; the learner needs relation-level slot reuse.  If anchor_flip
systematically hurts or inverts the held probe surface, the slot coordinate is
oriented rather than merely regularized.
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

OUT_DIR = base.WORKSPACE / "data/token_familiarity_vs_role_reuse"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_POOL = base.ALIAS_POOL[:32]
HELD_POOL = base.ALIAS_POOL[32:64]
EXPOSURE_MODES = ["absent", "neutral", "anchor_true", "anchor_flip"]
ARMS = [("zero", 0), ("true", 8), ("shuffled", 8)]
N_SEEDS = 4

NEUTRAL_TEMPLATES = [
    "{A} and {B} are listed together in a note.",
    "A note names {A} with {B}.",
    "The record mentions {A} beside {B}.",
    "{A} appears in the same line as {B}.",
    "A sentence contains both {A} and {B}.",
    "{A} is paired with {B} in the entry.",
    "The entry includes {A} and also {B}.",
    "{A} and {B} are the two names in the entry.",
    "The item lists {A} before {B}.",
    "The text repeats {A} along with {B}.",
]


def pool_pair(pool: List[str], key: str) -> Tuple[str, str]:
    i = base.stable_int("A|" + key) % len(pool)
    j = base.stable_int("B|" + key) % (len(pool) - 1)
    if j >= i:
        j += 1
    return pool[i], pool[j]


def render_with_aliases(ev: dict, tid: int, hyp_dir: str, aa: str, bb: str, mode: str, namespace: str, label_mode: str = "true") -> dict:
    context = base.render_context(tid, ev, aa, bb)
    hyp = f"{aa} defeated {bb}." if hyp_dir == "AB" else f"{bb} defeated {aa}."
    label = 1 if (hyp_dir == "AB" and ev["winner_label"] == "A") or (hyp_dir == "BA" and ev["winner_label"] == "B") else 0
    if label_mode == "flip":
        label = 1 - label
    elif label_mode == "true":
        pass
    else:
        raise ValueError(label_mode)
    return {
        "row_key": f"{namespace}|{mode}|{ev['event_id']}|T{tid}|{hyp_dir}",
        "mode": mode,
        "event_id": ev["event_id"],
        "family_id": ev["family_id"],
        "tid": tid,
        "template_group": base.TEMPLATES[tid][2],
        "hyp_dir": hyp_dir,
        "text": f"{context} [SEP] {hyp}",
        "label": int(label),
    }


def make_relation_rows(events: List[dict], tids: List[int], pool: List[str], namespace: str, mode: str, label_mode: str = "true", salt: str = "") -> List[dict]:
    rows = []
    for ev in events:
        aa, bb = pool_pair(pool, salt + ev["family_id"])
        for tid in tids:
            for hd in ["AB", "BA"]:
                rows.append(render_with_aliases(ev, tid, hd, aa, bb, mode, namespace, label_mode))
    return rows


def make_sparse_probe_rows(events: List[dict], pool: List[str], k_per_template: int, arm: str, seed: int, mode: str, salt: str = "train") -> List[dict]:
    if k_per_template <= 0:
        return []
    rng = random.Random(seed)
    rows = []
    for tid in base.PROBE_TEMPLATES:
        evs = events[:]
        rng.shuffle(evs)
        for ev in evs[: min(k_per_template, len(evs))]:
            aa, bb = pool_pair(pool, salt + ev["family_id"])
            for hd in ["AB", "BA"]:
                r = render_with_aliases(ev, tid, hd, aa, bb, mode, f"sparse_{arm}_{k_per_template}", "true")
                if arm == "true":
                    pass
                elif arm == "shuffled":
                    r["label"] = 1 - r["label"]
                elif arm == "exposure":
                    bit = base.stable_int(f"EXPO|{seed}|{ev['event_id']}|T{tid}") % 2
                    r["label"] = bit if hd == "AB" else 1 - bit
                else:
                    raise ValueError(arm)
                r["train_kind"] = f"probe_{arm}"
                rows.append(r)
    return rows


def make_neutral_exposure_rows(events: List[dict], pool: List[str], namespace: str, mode: str, salt: str = "held_exposure") -> List[dict]:
    """Balanced non-relational rows using the same row count as anchor exposure."""
    rows = []
    for ev in events:
        aa, bb = pool_pair(pool, salt + ev["family_id"])
        for j, template in enumerate(NEUTRAL_TEMPLATES):
            context = template.format(A=aa, B=bb)
            # Balanced identity/order task.  It updates eval-token embeddings and
            # coreference machinery but gives no win/loss role semantics.
            for hd in ["AB", "BA"]:
                hyp = f"{aa} is named before {bb}." if hd == "AB" else f"{bb} is named before {aa}."
                label = 1 if hd == "AB" else 0
                rows.append({
                    "row_key": f"{namespace}|{mode}|{ev['event_id']}|N{j}|{hd}",
                    "mode": mode,
                    "event_id": ev["event_id"],
                    "family_id": ev["family_id"],
                    "tid": -100 - j,
                    "template_group": "neutral_exposure",
                    "hyp_dir": hd,
                    "text": f"{context} [SEP] {hyp}",
                    "label": int(label),
                    "train_kind": "held_token_neutral_exposure",
                })
    return rows


def train_token_counts(rows: List[dict], tokens: List[str]) -> Dict[str, Any]:
    token_set = set(tokens)
    c = Counter()
    for r in rows:
        c.update(base.tokenize(r["text"]))
    vals = [c[t] for t in token_set]
    return {
        "types": len(token_set),
        "covered_types": sum(v > 0 for v in vals),
        "type_coverage": round(sum(v > 0 for v in vals) / max(1, len(vals)), 4),
        "total_occurrences": int(sum(vals)),
        "mean_occurrences": round(float(np.mean(vals)), 2) if vals else 0.0,
        "min_occurrences": int(min(vals)) if vals else 0,
        "max_occurrences": int(max(vals)) if vals else 0,
    }


def summarize(vals: List[float]) -> Dict[str, Any]:
    vals = [float(v) for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "values": []}
    return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "n": len(vals), "values": [round(v, 4) for v in vals]}


def main() -> None:
    t0 = time.time()
    if base.DEVICE == "cuda":
        # Honor CUDA_VISIBLE_DEVICES; inside the process this is device 0.
        torch.cuda.set_device(0)
    torch.set_num_threads(min(32, max(1, torch.get_num_threads())))
    train_fams, held_fams = base.load_families()
    all_train = base.extract_events(train_fams)
    all_held = base.extract_events(held_fams)
    train_events = base.select_events(all_train, base.TRAIN_EVENT_LIMIT, 257)
    eval_train_events = base.select_events(all_train, base.EVAL_TRAIN_EVENT_LIMIT, 1257)
    eval_held_events = base.select_events(all_held, base.EVAL_HELD_EVENT_LIMIT, 2257)

    print(json.dumps({
        "status": "TOKEN_FAMILIARITY_VS_ROLE_REUSE_START",
        "device": base.DEVICE,
        "train_events": len(train_events),
        "eval_held_events": len(eval_held_events),
        "train_pool": TRAIN_POOL,
        "held_pool": HELD_POOL,
        "modes": EXPOSURE_MODES,
        "arms": ARMS,
        "n_seeds": N_SEEDS,
    }), flush=True)

    datasets: Dict[str, Dict[str, List[dict]]] = {}
    rows_by_mode: Dict[str, List[dict]] = defaultdict(list)
    exposure_counts: Dict[str, Dict[str, Any]] = {}
    for mode in EXPOSURE_MODES:
        base_anchor = make_relation_rows(train_events, base.ANCHOR_TEMPLATES, TRAIN_POOL, "train_anchor", mode, "true", salt="train|")
        for r in base_anchor:
            r["train_kind"] = "anchor_trainpool"
        exposure: List[dict] = []
        if mode == "neutral":
            exposure = make_neutral_exposure_rows(train_events, HELD_POOL, "held_token_neutral", mode, salt="held_exp|")
        elif mode == "anchor_true":
            exposure = make_relation_rows(train_events, base.ANCHOR_TEMPLATES, HELD_POOL, "held_token_anchor_true", mode, "true", salt="held_exp|")
            for r in exposure:
                r["train_kind"] = "held_token_anchor_true"
        elif mode == "anchor_flip":
            exposure = make_relation_rows(train_events, base.ANCHOR_TEMPLATES, HELD_POOL, "held_token_anchor_flip", mode, "flip", salt="held_exp|")
            for r in exposure:
                r["train_kind"] = "held_token_anchor_flip"
        elif mode == "absent":
            exposure = []
        else:
            raise ValueError(mode)
        evals = {
            "probe_heldpool_heldfam": make_relation_rows(eval_held_events, base.PROBE_TEMPLATES, HELD_POOL, "eval_held_probe", mode, "true", salt="held_eval|"),
            "anchor_heldpool_heldfam": make_relation_rows(eval_held_events, base.ANCHOR_TEMPLATES, HELD_POOL, "eval_held_anchor", mode, "true", salt="held_eval|"),
            "heldtemplate_heldpool_heldfam": make_relation_rows(eval_held_events, base.HELD_TEMPLATES, HELD_POOL, "eval_heldtemplate", mode, "true", salt="held_eval|"),
            "probe_trainpool_heldfam": make_relation_rows(eval_held_events, base.PROBE_TEMPLATES, TRAIN_POOL, "eval_held_probe_trainpool", mode, "true", salt="train_eval|"),
            "probe_trainpool_trainfam": make_relation_rows(eval_train_events, base.PROBE_TEMPLATES, TRAIN_POOL, "eval_train_probe_trainpool", mode, "true", salt="train|"),
        }
        for tid in base.PROBE_TEMPLATES:
            evals[f"probe_T{tid:02d}_heldpool_heldfam"] = make_relation_rows(eval_held_events, [tid], HELD_POOL, f"eval_held_probeT{tid}", mode, "true", salt="held_eval|")
        datasets[mode] = {"base_anchor": base_anchor, "exposure": exposure, **evals}
        # Include all prospective sparse rows in vocab so OOV is not the measured factor.
        rows_by_mode[mode].extend(base_anchor + exposure)
        for part in evals.values():
            rows_by_mode[mode].extend(part)
        for arm, k in ARMS:
            if k:
                rows_by_mode[mode].extend(make_sparse_probe_rows(train_events, TRAIN_POOL, k, arm, 999, mode, salt="train|"))
        exposure_counts[mode] = train_token_counts(base_anchor + exposure, HELD_POOL)
        print(json.dumps({"event": "mode_constructed", "mode": mode, "base_anchor_rows": len(base_anchor), "exposure_rows": len(exposure), "held_pool_train_counts": exposure_counts[mode]}), flush=True)

    vocabs = base.build_vocab(rows_by_mode)
    raw: List[dict] = []
    for mode in EXPOSURE_MODES:
        print(f"\n=== MODE {mode} vocab={len(vocabs[mode])} ===", flush=True)
        base_anchor = datasets[mode]["base_anchor"]
        exposure = datasets[mode]["exposure"]
        eval_sets = {k: v for k, v in datasets[mode].items() if k not in {"base_anchor", "exposure"}}
        for arm, k in ARMS:
            for si in range(N_SEEDS):
                seed_sparse = 260500 + si * 101 + base.stable_int(mode + arm) % 733
                sparse = make_sparse_probe_rows(train_events, TRAIN_POOL, k, arm, seed_sparse, mode, salt="train|") if k else []
                train_rows = base_anchor + exposure + sparse
                seed = 260900 + si * 1213 + base.stable_int(mode + arm) % 911
                res = base.train_eval(train_rows, eval_sets, vocabs[mode], seed)
                res.update({
                    "mode": mode,
                    "arm": arm,
                    "k": k,
                    "seed_idx": si,
                    "seed": seed,
                    "n_train_rows": len(train_rows),
                    "n_exposure_rows": len(exposure),
                    "n_sparse_rows": len(sparse),
                    "held_pool_train_counts": exposure_counts[mode],
                    "train_fit": bool(res["train_acc"] >= base.TRAIN_FIT),
                })
                raw.append(res)
                print(json.dumps({
                    "mode": mode, "arm": arm, "seed_idx": si,
                    "train": round(res["train_acc"], 4),
                    "probeHeldPool": round(res["probe_heldpool_heldfam"], 4),
                    "anchorHeldPool": round(res["anchor_heldpool_heldfam"], 4),
                    "probeTrainPoolHeldFam": round(res["probe_trainpool_heldfam"], 4),
                    "probeTrainPoolTrainFam": round(res["probe_trainpool_trainfam"], 4),
                    "fit": res["train_fit"],
                }), flush=True)

    grouped = defaultdict(list)
    for r in raw:
        grouped[(r["mode"], r["arm"])].append(r)
    metrics = [
        "train_acc", "probe_heldpool_heldfam", "anchor_heldpool_heldfam", "heldtemplate_heldpool_heldfam",
        "probe_trainpool_heldfam", "probe_trainpool_trainfam",
    ] + [f"probe_T{tid:02d}_heldpool_heldfam" for tid in base.PROBE_TEMPLATES]
    summary: Dict[str, Any] = {}
    for (mode, arm), items in grouped.items():
        key = f"{mode}|{arm}"
        summary[key] = {
            "mode": mode,
            "arm": arm,
            "n_runs": len(items),
            "fit_count": sum(x["train_fit"] for x in items),
            "n_train_rows": items[0]["n_train_rows"],
            "n_exposure_rows": items[0]["n_exposure_rows"],
        }
        for m in metrics:
            summary[key][m] = summarize([x[m] for x in items if m in x])

    separation = {}
    for mode in EXPOSURE_MODES:
        z = summary[f"{mode}|zero"]["probe_heldpool_heldfam"]["mean"]
        t = summary[f"{mode}|true"]["probe_heldpool_heldfam"]["mean"]
        s = summary[f"{mode}|shuffled"]["probe_heldpool_heldfam"]["mean"]
        az = summary[f"{mode}|zero"]["anchor_heldpool_heldfam"]["mean"]
        at = summary[f"{mode}|true"]["anchor_heldpool_heldfam"]["mean"]
        separation[mode] = {
            "zero_probe": z,
            "true_probe": t,
            "shuffled_probe": s,
            "true_minus_shuffled_probe": round(t - s, 4),
            "true_minus_zero_probe": round(t - z, 4),
            "zero_anchor": az,
            "true_anchor": at,
            "held_pool_type_coverage_in_train": exposure_counts[mode]["type_coverage"],
            "held_pool_mean_train_occ": exposure_counts[mode]["mean_occurrences"],
        }

    out = {
        "status": "TOKEN_FAMILIARITY_VS_ROLE_REUSE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t0, 2),
        "purpose": "Separate ordinary eval-token familiarity from relation-level argument-slot reuse in the research/259 role-coordinate diagnostic.",
        "config": {
            "device": base.DEVICE,
            "train_pool": TRAIN_POOL,
            "held_pool": HELD_POOL,
            "modes": EXPOSURE_MODES,
            "arms": ARMS,
            "n_seeds": N_SEEDS,
            "train_event_limit": base.TRAIN_EVENT_LIMIT,
            "eval_held_event_limit": base.EVAL_HELD_EVENT_LIMIT,
            "model": {"type": "BiGRU_attention", "emb_dim": base.EMB_DIM, "hidden_dim": base.HIDDEN_DIM, "epochs": base.EPOCHS, "batch_size": base.BATCH_SIZE, "lr": base.LR},
            "vocab_sizes": {m: len(v) for m, v in vocabs.items()},
        },
        "held_pool_train_counts": exposure_counts,
        "summary": summary,
        "separation": separation,
        "raw": raw,
        "interpretation": "Neutral exposure trains held-pool token embeddings/coreference without win/loss relation roles. Anchor_true/anchor_flip use the same held-pool token support as oriented or inverted argument slots. Probe true-vs-shuffled movement that appears only after anchor_true/anchor_flip supports relation-level slot reuse rather than ordinary token familiarity alone.",
    }
    out_json = OUT_DIR / "token_familiarity_vs_role_reuse_summary.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# research token familiarity versus relation-role reuse\n\n",
             f"Device `{base.DEVICE}`, elapsed {out['elapsed_seconds']} s, {N_SEEDS} seeds per cell.\n\n",
             "## Held-pool token exposure in training\n\n",
             "| mode | covered held-pool types | total occ | mean occ/type | min | max |\n|---|---:|---:|---:|---:|---:|\n"]
    for m in EXPOSURE_MODES:
        c = exposure_counts[m]
        lines.append(f"| {m} | {c['covered_types']}/{c['types']} | {c['total_occurrences']} | {c['mean_occurrences']:.1f} | {c['min_occurrences']} | {c['max_occurrences']} |\n")
    lines.append("\n## Held-family probe accuracy with held-pool argument fillers\n\n")
    lines.append("| mode | arm | fit | train | probe-heldpool | anchor-heldpool | heldtemplate-heldpool | trainpool-heldfam probe | trainpool-trainfam probe |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for mode in EXPOSURE_MODES:
        for arm, _ in ARMS:
            s = summary[f"{mode}|{arm}"]
            lines.append(f"| {mode} | {arm} | {s['fit_count']}/{s['n_runs']} | {s['train_acc']['mean']:.3f} | {s['probe_heldpool_heldfam']['mean']:.3f} | {s['anchor_heldpool_heldfam']['mean']:.3f} | {s['heldtemplate_heldpool_heldfam']['mean']:.3f} | {s['probe_trainpool_heldfam']['mean']:.3f} | {s['probe_trainpool_trainfam']['mean']:.3f} |\n")
    lines.append("\n## Probe-coordinate separation\n\n| mode | zero | true | shuffled | true-shuffled | true-zero | held token train coverage | mean occ/type |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for mode, v in separation.items():
        lines.append(f"| {mode} | {v['zero_probe']:.3f} | {v['true_probe']:.3f} | {v['shuffled_probe']:.3f} | {v['true_minus_shuffled_probe']:+.3f} | {v['true_minus_zero_probe']:+.3f} | {v['held_pool_type_coverage_in_train']:.3f} | {v['held_pool_mean_train_occ']:.1f} |\n")
    lines.append(f"\nSummary JSON: `{out_json}`\n")
    (OUT_DIR / "token_familiarity_vs_role_reuse_summary.md").write_text("".join(lines), encoding="utf-8")

    print("\n" + "=" * 100)
    print("research TOKEN FAMILIARITY VS RELATION-ROLE REUSE SUMMARY")
    for mode, v in separation.items():
        print(f"{mode:<12s} cov={v['held_pool_type_coverage_in_train']:.3f} occ={v['held_pool_mean_train_occ']:.1f} zero={v['zero_probe']:.3f} true={v['true_probe']:.3f} shuf={v['shuffled_probe']:.3f} true-shuf={v['true_minus_shuffled_probe']:+.3f} anchor0={v['zero_anchor']:.3f}")
    print(json.dumps({"status": out["status"], "elapsed_seconds": out["elapsed_seconds"], "summary_json": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
