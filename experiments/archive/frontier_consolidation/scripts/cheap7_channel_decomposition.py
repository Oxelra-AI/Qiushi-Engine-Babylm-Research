#!/usr/bin/env python3
"""research cheap7 channel decomposition and low-count column bootstrap.

independent review (research) flagged two zero-training, payload-only checks that must precede
treating any SuperGLUE/endpoint number as a promotion decision:

  (A) Decompose each arm's cheap7 delta vs the anchor into per-column
      contributions (Δcolumn / 7), and record how many net item flips produced
      each column's movement.  This asks whether the aggregate edge is broad
      discrete-accuracy gain or concentration in a few low-cardinality columns
      (GlobalPIQA n~203 macro-averaged, EWoK, Supplement).

  (B) Bootstrap the low-count columns (GlobalPIQA, EWoK, Entity) over their
      UID/example groups and test whether the arm-vs-anchor cheap7 ordering and
      the +margin over the anchor survive resampling, i.e. whether the endpoint
      ranking is a small-n sampling artifact.

No model inference: this consumes saved official-compatible per_target payloads
only, reusing the validated research PayloadLoader / official_score.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import OrderedDict, defaultdict
from statistics import mean
from typing import Any

import numpy as np

ROOT = pathlib.Path(".")
SCRIPTS = ROOT / "experiments/archive/frontier_consolidation/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pairwise_item_flip_analysis import (  # noqa: E402
    CHEAP_COLS,
    ItemRow,
    PayloadLoader,
    official_score,
)

DEFAULT_ARMS = OrderedDict([
    ("chck82_anchor",
     "experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/staged_full_eval/per_target/scale1p75_chck82_independent.json"),
    ("ordinary86_backbone",
     "experiments/archive/representation_and_objectives/data/scale1p75_chck86_cheap7_eval/per_target/scale1p75_chck86_cheap7.json"),
    ("shuffled86_private",
     "experiments/archive/frontier_consolidation/data/frozen82_tail4M_shuffled_eval/per_target/frozen82_tail4M_shuffled.json"),
    ("coherent86_alpha1",
     "experiments/archive/frontier_consolidation/data/fastpath4M_coherent_eval/per_target/fastpath4M_coherent.json"),
    ("spanbreak86",
     "experiments/archive/frontier_consolidation/data/fastpath4M_spanbreak_eval/per_target/fastpath4M_spanbreak.json"),
    ("coherent86_alpha0p5",
     "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p5/per_target/coherent86_private_scale_0p5.json"),
    ("coherent86_alpha0p75",
     "experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json"),
])
DEFAULT_OUT = ROOT / "experiments/archive/frontier_consolidation/data/cheap7_channel_decomposition"
ANCHOR = "chck82_anchor"
BOOT_COLS = ["GlobalPIQA", "EWoK", "Entity", "Supplement"]
N_BOOT = 2000
SEED = 43022


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def payload_ready(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    tasks = obj.get("tasks", {}) if isinstance(obj, dict) else {}
    needed = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS",
              "GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
    return all(k in tasks for k in needed)


def load_all_rows(loader: PayloadLoader) -> dict[str, list[ItemRow]]:
    out: dict[str, list[ItemRow]] = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]:
        rows, _ = loader.load_column(col)
        out[col] = rows
    return out


def col_score(rows: list[ItemRow], col: str) -> float:
    return float(official_score(rows, col))


def cheap7_from_rowsets(rowsets: dict[str, list[ItemRow]]) -> tuple[float, dict[str, float]]:
    scores = {}
    for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]:
        scores[col] = col_score(rowsets[col], col)
    # Reading is a non-discrete column carried directly from the payload later.
    return scores


def bootstrap_column_delta(
    anchor_rows: list[ItemRow], cand_rows: list[ItemRow], col: str, rng: np.random.Generator, n_boot: int
) -> dict[str, Any]:
    """Bootstrap resample the grouping units of a column and recompute the
    anchor->candidate column-score delta.  Grouping units are the objects the
    official metric averages over (UIDs for BLiMP-like/EWoK/Entity/COMPS; for
    GlobalPIQA, example UIDs within each sub, then subs macro-averaged)."""
    amap = {r.item_id: r for r in anchor_rows}
    cmap = {r.item_id: r for r in cand_rows}
    common = sorted(set(amap) & set(cmap))
    a_common = [amap[i] for i in common]
    c_common = [cmap[i] for i in common]

    if col == "GlobalPIQA":
        # group by (sub, uid); macro over uids within sub, then mean over subs
        def sub_uid_index(rows):
            idx = defaultdict(lambda: defaultdict(list))
            for r in rows:
                idx[str(r.sub)][r.uid].append(r)
            return idx
        a_idx = sub_uid_index(a_common)
        c_idx = sub_uid_index(c_common)
        subs = sorted(a_idx.keys())
        # For bootstrap, resample uids within each sub.
        boot_deltas = []
        # Precompute per-uid accuracy for both arms
        for _ in range(n_boot):
            sub_delta = []
            for sub in subs:
                uids = list(a_idx[sub].keys())
                if not uids:
                    continue
                sample = rng.choice(len(uids), size=len(uids), replace=True)
                a_acc = []
                c_acc = []
                for si in sample:
                    uid = uids[si]
                    a_rows = a_idx[sub][uid]
                    c_rows = c_idx[sub].get(uid, [])
                    a_acc.append(np.mean([int(r.correct) for r in a_rows]))
                    if c_rows:
                        c_acc.append(np.mean([int(r.correct) for r in c_rows]))
                    else:
                        c_acc.append(0.0)
                sub_delta.append(100.0 * (np.mean(c_acc) - np.mean(a_acc)))
            boot_deltas.append(float(np.mean(sub_delta)) if sub_delta else 0.0)
        boot_deltas = np.array(boot_deltas)
    elif col == "Entity":
        # Entity: macro over splits {regular, ambiref, move_contents}, each mean over uids.
        def split_uid_index(rows):
            idx = defaultdict(lambda: defaultdict(list))
            for r in rows:
                for split in ["regular", "ambiref", "move_contents"]:
                    if r.uid.startswith(split):
                        idx[split][r.uid].append(r)
                        break
            return idx
        a_idx = split_uid_index(a_common)
        c_idx = split_uid_index(c_common)
        splits = sorted(a_idx.keys())
        boot_deltas = []
        for _ in range(n_boot):
            sp_delta = []
            for split in splits:
                uids = list(a_idx[split].keys())
                if not uids:
                    continue
                sample = rng.choice(len(uids), size=len(uids), replace=True)
                a_acc = []
                c_acc = []
                for si in sample:
                    uid = uids[si]
                    a_acc.append(100.0 * np.mean([int(r.correct) for r in a_idx[split][uid]]))
                    c_rows = c_idx[split].get(uid, [])
                    c_acc.append(100.0 * np.mean([int(r.correct) for r in c_rows]) if c_rows else 0.0)
                sp_delta.append(np.mean(c_acc) - np.mean(a_acc))
            boot_deltas.append(float(np.mean(sp_delta)) if sp_delta else 0.0)
        boot_deltas = np.array(boot_deltas)
    else:
        # BLiMP-like / EWoK / Supplement / COMPS: mean over uid accuracies
        def uid_index(rows):
            idx = defaultdict(list)
            for r in rows:
                idx[r.uid].append(r)
            return idx
        a_idx = uid_index(a_common)
        c_idx = uid_index(c_common)
        uids = list(a_idx.keys())
        a_acc = {u: 100.0 * np.mean([int(r.correct) for r in a_idx[u]]) for u in uids}
        c_acc = {u: 100.0 * np.mean([int(r.correct) for r in c_idx.get(u, [])]) if c_idx.get(u) else 0.0 for u in uids}
        per_uid_delta = np.array([c_acc[u] - a_acc[u] for u in uids])
        boot_deltas = []
        n = len(uids)
        for _ in range(n_boot):
            sample = rng.choice(n, size=n, replace=True)
            boot_deltas.append(float(np.mean(per_uid_delta[sample])))
        boot_deltas = np.array(boot_deltas)

    point = col_score(c_common, col) - col_score(a_common, col)
    return {
        "point_delta": float(point),
        "boot_mean": float(np.mean(boot_deltas)),
        "boot_std": float(np.std(boot_deltas)),
        "boot_ci95": [float(np.percentile(boot_deltas, 2.5)), float(np.percentile(boot_deltas, 97.5))],
        "boot_p_delta_le_0": float(np.mean(boot_deltas <= 0.0)),
        "n_boot": int(n_boot),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = OrderedDict((k, pathlib.Path(v)) for k, v in DEFAULT_ARMS.items())
    ready = OrderedDict((k, v) for k, v in arms.items() if payload_ready(v))
    pending = {k: str(v) for k, v in arms.items() if k not in ready}

    loaders = {k: PayloadLoader(v) for k, v in ready.items()}
    rowsets = {k: load_all_rows(l) for k, l in loaders.items()}

    # Column scores + cheap7 (discrete cols reconstructed, Reading from payload).
    col_scores = {}
    reading_scores = {}
    cheap7 = {}
    for k in ready:
        sc = cheap7_from_rowsets(rowsets[k])
        rd = loaders[k].payload.get("official_overall", {}).get("scores", {}).get("Reading")
        reading_scores[k] = None if rd is None else float(rd)
        col_scores[k] = sc
        vals = list(sc.values()) + ([reading_scores[k]] if reading_scores[k] is not None else [])
        cheap7[k] = float(mean(vals)) if len(vals) == 7 else None

    # Per-column contribution to cheap7 delta vs anchor: Δcol / 7.
    decomposition = OrderedDict()
    if ANCHOR in ready:
        anchor_sc = col_scores[ANCHOR]
        anchor_rd = reading_scores[ANCHOR]
        for k in ready:
            if k == ANCHOR:
                continue
            contribs = {}
            for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA"]:
                dcol = col_scores[k][col] - anchor_sc[col]
                contribs[col] = {"delta_col": float(dcol), "contrib_cheap7": float(dcol / 7.0)}
            drd = (reading_scores[k] - anchor_rd) if (reading_scores[k] is not None and anchor_rd is not None) else None
            contribs["Reading"] = {"delta_col": drd, "contrib_cheap7": (None if drd is None else float(drd / 7.0))}
            total = cheap7[k] - cheap7[ANCHOR] if (cheap7[k] is not None and cheap7[ANCHOR] is not None) else None
            # concentration: share of positive contribution from top column
            pos = {c: v["contrib_cheap7"] for c, v in contribs.items() if v["contrib_cheap7"] is not None and v["contrib_cheap7"] > 0}
            pos_sum = sum(pos.values())
            top_col = max(pos, key=pos.get) if pos else None
            decomposition[k] = {
                "cheap7": cheap7[k],
                "cheap7_delta_vs_anchor": total,
                "column_contributions": contribs,
                "positive_contribution_sum": float(pos_sum),
                "top_positive_column": top_col,
                "top_positive_share_of_positive": (None if not pos else float(pos[top_col] / pos_sum)),
                "globalpiqa_share_of_positive": (None if pos_sum == 0 else float(pos.get("GlobalPIQA", 0.0) / pos_sum)),
            }

    # cheap7 with GlobalPIQA excluded (cheap6-noGP): does the ordering survive?
    cheap6_noGP = {}
    for k in ready:
        cols = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
        vals = [col_scores[k][c] for c in cols]
        if reading_scores[k] is not None:
            vals.append(reading_scores[k])
        cheap6_noGP[k] = float(mean(vals))

    # Bootstrap low-count columns for candidate-vs-anchor deltas.
    rng = np.random.default_rng(SEED)
    bootstrap = OrderedDict()
    if ANCHOR in ready:
        for k in ready:
            if k == ANCHOR:
                continue
            per_col = {}
            for col in BOOT_COLS:
                per_col[col] = bootstrap_column_delta(
                    rowsets[ANCHOR][col], rowsets[k][col], col, np.random.default_rng(SEED + hash(k + col) % 100000), args.n_boot
                )
            bootstrap[k] = per_col

    # Order arms by cheap7 and by cheap6_noGP to expose GlobalPIQA dependence.
    order_cheap7 = sorted([k for k in ready if cheap7[k] is not None], key=lambda k: cheap7[k], reverse=True)
    order_noGP = sorted(cheap6_noGP, key=lambda k: cheap6_noGP[k], reverse=True)

    out = {
        "status": "COMPLETE" if not pending else "COMPLETE_WITH_PENDING",
        "created_utc": now(),
        "anchor": ANCHOR,
        "n_boot": args.n_boot,
        "ready_arms": {k: str(v) for k, v in ready.items()},
        "pending_arms": pending,
        "cheap7": cheap7,
        "cheap6_no_globalpiqa": cheap6_noGP,
        "column_scores": {k: {**col_scores[k], "Reading": reading_scores[k]} for k in ready},
        "decomposition_vs_anchor": decomposition,
        "bootstrap_low_count_vs_anchor": bootstrap,
        "ordering_cheap7": order_cheap7,
        "ordering_no_globalpiqa": order_noGP,
    }
    out_json = out_dir / "cheap7_channel_decomposition.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research cheap7 channel decomposition and low-count bootstrap",
        "",
        f"Status: **{out['status']}**  ·  anchor `{ANCHOR}`  ·  n_boot {args.n_boot}",
        "",
        "Payload-only (no model inference). Discrete columns reconstructed via the",
        "validated official_score; Reading carried from payload.",
        "",
        "## cheap7 vs cheap7-without-GlobalPIQA",
        "",
        "| arm | cheap7 | Δcheap7 vs anchor | cheap6(no GP) | Δcheap6(no GP) vs anchor |",
        "|---|---:|---:|---:|---:|",
    ]
    a7 = cheap7.get(ANCHOR)
    a6 = cheap6_noGP.get(ANCHOR)
    for k in order_cheap7:
        d7 = None if (cheap7[k] is None or a7 is None) else cheap7[k] - a7
        d6 = cheap6_noGP[k] - a6
        lines.append(f"| {k} | {cheap7[k]:.6f} | {'' if d7 is None else f'{d7:+.6f}'} | {cheap6_noGP[k]:.6f} | {d6:+.6f} |")
    lines += [
        "",
        f"Ordering by cheap7: {order_cheap7}",
        f"Ordering by cheap6 (no GlobalPIQA): {order_noGP}",
        "",
        "## cheap7-delta decomposition vs anchor (Δcol / 7)",
        "",
    ]
    for k, dec in decomposition.items():
        lines += [
            f"### {k}  (Δcheap7 {dec['cheap7_delta_vs_anchor']:+.6f})",
            "",
            "| column | Δcol | contrib to cheap7 |",
            "|---|---:|---:|",
        ]
        for col in ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]:
            c = dec["column_contributions"][col]
            dc = c["delta_col"]
            cc = c["contrib_cheap7"]
            lines.append(f"| {col} | {'' if dc is None else f'{dc:+.4f}'} | {'' if cc is None else f'{cc:+.6f}'} |")
        lines.append("")
        top_share = dec.get('top_positive_share_of_positive')
        gp_share = dec.get('globalpiqa_share_of_positive')
        if top_share is None:
            lines.append(
                f"Positive-contribution sum {dec['positive_contribution_sum']:+.6f}; no positive column contribution."
            )
        else:
            lines.append(
                f"Positive-contribution sum {dec['positive_contribution_sum']:+.6f}; top positive column "
                f"`{dec['top_positive_column']}` = {top_share:.3f} of positive; "
                f"GlobalPIQA share of positive = {gp_share}."
            )
        lines.append("")
    lines += ["## Low-count column bootstrap (candidate − anchor)", ""]
    for k, per_col in bootstrap.items():
        lines += [f"### {k}", "", "| column | point Δ | boot mean | boot std | CI95 | P(Δ<=0) |", "|---|---:|---:|---:|---|---:|"]
        for col in BOOT_COLS:
            b = per_col[col]
            lines.append(
                f"| {col} | {b['point_delta']:+.4f} | {b['boot_mean']:+.4f} | {b['boot_std']:.4f} | "
                f"[{b['boot_ci95'][0]:+.4f}, {b['boot_ci95'][1]:+.4f}] | {b['boot_p_delta_le_0']:.4f} |"
            )
        lines.append("")
    lines += [f"JSON: `{out_json}`"]
    (out_dir / "cheap7_channel_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "ready": list(ready), "pending": list(pending),
                      "ordering_cheap7": order_cheap7, "ordering_no_globalpiqa": order_noGP,
                      "out_json": str(out_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
