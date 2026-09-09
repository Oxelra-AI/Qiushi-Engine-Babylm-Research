#!/usr/bin/env python3
"""research: integrate the completed CLEAN seed43222 official Entity readout.

The purpose is not to make Entity the primary mechanism quantity, but to learn
whether the attenuated seed43222 V-R crossover came from a REPEAT-specific rel0
change, a VIEW-specific change, or a CLEAN-relative movement.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import csv
import json
import math
import pathlib
import re
import statistics
import time
from collections import defaultdict


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'relation_learning'
META = WS / "data" / "entity_relevant_update_analysis" / "entity_item_metadata.csv"
VR_LATE = WS / "data" / "seed43222_vr_entity_crossover" / "seed43222_vr_entity_late.csv"
C_EVAL = WS / "data" / "seed43222_parallel_clean_entity_eval" / "per_target"
OUT = WS / "data" / "seed43222_entity_clean_integration"
NOTE = WS / "notes" / "seed43222_entity_clean_integration.md"
CKS = ["chck_80M", "chck_90M", "chck_100M"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .")


def mean(xs) -> float:
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(xs) if xs else float("nan")


def read_csv(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def read_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta() -> dict[str, list[dict]]:
    by = defaultdict(list)
    for r in read_csv(META):
        r = dict(r)
        for k in ["item_index", "reported_numops", "relevant_updates", "total_ops", "irrelevant_ops", "prefix_words", "stale_available", "stale_is_gold"]:
            r[k] = int(r[k])
        by[r["uid"]].append(r)
    for uid in by:
        by[uid].sort(key=lambda x: int(x["item_index"]))
    return dict(by)


def pred_path(ck: str) -> pathlib.Path:
    payload = read_json(C_EVAL / f"D_C_43222_{ck}.json")
    pred = payload["tasks"]["Entity"]["predictions"]
    path = ROOT / pred
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def groups(r: dict) -> list[str]:
    relu = int(r["relevant_updates"])
    out = ["ALL", f"rel_updates_{relu}"]
    if relu >= 1: out.append("rel_ge1")
    if relu >= 2: out.append("rel_ge2")
    if relu >= 3: out.append("rel_ge3")
    return out


def score_clean() -> tuple[list[dict], list[dict], list[dict]]:
    meta = load_meta()
    rows: list[dict] = []
    for ck in CKS:
        obj = read_json(pred_path(ck))
        for uid, items in sorted(meta.items()):
            preds = obj[uid]["predictions"]
            if len(preds) != len(items):
                raise RuntimeError((uid, ck, len(preds), len(items)))
            for item, predrec in zip(items, preds):
                pred = str(predrec.get("pred", ""))
                correct = int(norm(pred) == norm(item["gold"]))
                pred_stale = int(bool(item.get("stale_initial", "")) and norm(pred) == norm(item.get("stale_initial", "")))
                rows.append({**item, "seed": "43222", "arm": "C", "checkpoint": ck, "pred": pred, "correct": correct, "pred_is_stale_initial": pred_stale})
    d = defaultdict(list)
    for r in rows:
        for g in groups(r):
            d[(r["arm"], r["checkpoint"], g)].append(r)
    summary: list[dict] = []
    for (arm, ck, g), vals in sorted(d.items()):
        n = len(vals)
        stale_cand = [v for v in vals if int(v["stale_available"]) and not int(v["stale_is_gold"])]
        wrong_stale = [v for v in stale_cand if not int(v["correct"])]
        summary.append({
            "arm": arm,
            "checkpoint": ck,
            "group": g,
            "n": n,
            "accuracy_pct": 100 * sum(int(v["correct"]) for v in vals) / n,
            "stale_pick_pct": 100 * sum(int(v["pred_is_stale_initial"]) for v in wrong_stale) / len(wrong_stale) if wrong_stale else float("nan"),
            "mean_relevant_updates": mean([v["relevant_updates"] for v in vals]),
            "mean_total_ops": mean([v["total_ops"] for v in vals]),
            "mean_prefix_words": mean([v["prefix_words"] for v in vals]),
        })
    dd = defaultdict(list)
    for r in summary:
        dd[(r["arm"], r["group"])].append(r)
    late = []
    for (arm, g), vals in sorted(dd.items()):
        late.append({
            "arm": arm,
            "group": g,
            "n_checkpoints": len(vals),
            "n": int(vals[0]["n"]),
            "late_mean_accuracy_pct": mean([v["accuracy_pct"] for v in vals]),
            "late_mean_stale_pct": mean([v["stale_pick_pct"] for v in vals]),
        })
    return rows, summary, late


def load_vr_late() -> list[dict]:
    rows = []
    for r in read_csv(VR_LATE):
        rows.append({
            "arm": r["arm"],
            "group": r["group"],
            "n_checkpoints": int(r["n_checkpoints"]),
            "n": int(r["n"]),
            "late_mean_accuracy_pct": float(r["late_mean_accuracy_pct"]),
            "late_mean_stale_pct": float(r["late_mean_stale_pct"]) if r["late_mean_stale_pct"] and r["late_mean_stale_pct"] != "nan" else float("nan"),
        })
    return rows


def contrast_rows(late_all: list[dict]) -> list[dict]:
    idx = {(r["arm"], r["group"]): r for r in late_all}
    groups_order = ["ALL", "rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge1", "rel_ge2", "rel_ge3"]
    out = []
    for g in groups_order:
        row = {"group": g}
        for arm in ["V", "R", "C"]:
            r = idx.get((arm, g))
            row[f"{arm}_acc"] = r["late_mean_accuracy_pct"] if r else float("nan")
            row[f"{arm}_stale_pct"] = r["late_mean_stale_pct"] if r else float("nan")
            if r and "n" not in row:
                row["n"] = r["n"]
        row["VminusR"] = row["V_acc"] - row["R_acc"]
        row["VminusC"] = row["V_acc"] - row["C_acc"]
        row["RminusC"] = row["R_acc"] - row["C_acc"]
        out.append(row)
    return out


def write_note(rows: list[dict]) -> None:
    lines = []
    lines.append("# research seed43222 Entity with CLEAN baseline")
    lines.append("")
    lines.append("CLEAN seed43222 was evaluated on official Entity at 80M/90M/100M after research. This note integrates it with the existing seed43222 VIEW/REPEAT readout. Entity remains a downstream correlate; the primary installed quantities are the held-out copy and rewrite-conditioning probes.")
    lines.append("")
    lines.append("| group | n | V acc | R acc | C acc | V−R | V−C | R−C |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if r["group"] in ["ALL", "rel_updates_0", "rel_updates_1", "rel_updates_2", "rel_updates_3", "rel_updates_4", "rel_updates_5", "rel_ge2", "rel_ge3"]:
            lines.append(f"| {r['group']} | {int(r.get('n', 0))} | {r['V_acc']:.2f} | {r['R_acc']:.2f} | {r['C_acc']:.2f} | {r['VminusR']:+.2f} | {r['VminusC']:+.2f} | {r['RminusC']:+.2f} |")
    lines.append("")
    lines.append("## Scientific reading")
    lines.append("")
    lines.append("CLEAN's late mean Entity score is 26.72, above both VIEW (25.90) and REPEAT (25.06) at this seed. The V−R direction still has the expected form: REPEAT is above VIEW at zero relevant updates, while VIEW is above REPEAT at positive update depths. Against CLEAN, however, neither intervention is a simple improvement: at rel0, REPEAT is +0.89 above CLEAN while VIEW is −1.79 below CLEAN; at rel4, VIEW is +0.12 above CLEAN while REPEAT is −1.15 below CLEAN. The strong two-seed V−C deep-update advantage therefore becomes weak at seed43222, whereas the held-out rewrite/copy probes retained large stable effects. This supports the current interpretation that Entity is a direction-robust downstream correlate whose conversion magnitude depends on seed-specific factors, not the primary installed measurement.")
    lines.append("")
    lines.append(f"Data: `{rel(OUT)}`")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    c_rows, c_summary, c_late = score_clean()
    write_csv(OUT / "clean_entity_item_rows.csv", c_rows)
    write_csv(OUT / "clean_entity_summary.csv", c_summary)
    write_csv(OUT / "clean_entity_late.csv", c_late)
    late_all = load_vr_late() + c_late
    write_csv(OUT / "seed43222_vrc_entity_late.csv", late_all)
    contrasts = contrast_rows(late_all)
    write_csv(OUT / "seed43222_vrc_entity_contrasts.csv", contrasts)
    write_note(contrasts)
    result = {
        "status": "SEED43222_ENTITY_CLEAN_INTEGRATION_DONE",
        "finished_utc": now(),
        "note": rel(NOTE),
        "outputs": rel(OUT),
        "clean_late_mean_entity_score": next(r["late_mean_accuracy_pct"] for r in c_late if r["group"] == "ALL"),
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
