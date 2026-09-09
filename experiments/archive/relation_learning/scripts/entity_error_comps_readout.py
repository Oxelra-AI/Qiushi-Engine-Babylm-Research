#!/usr/bin/env python3
"""research: CPU readout of Entity depth/error shape and seed43022 COMPS V-R.

Uses existing official prediction/report files. No model forward pass and no
evaluation run. The scientific object is whether the research depth dissociation
has the error shape expected from repetition buying shallow direct recall rather
than multi-step state updating, and whether the same repetition/variation trade
appears outside Entity in COMPS stored-property retrieval.
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
from collections import defaultdict
from typing import Any

ROOT = _public_path('experiments/archive/relation_learning/scripts/entity_error_comps_readout.py')
ROOT = _PUBLIC_ROOT

WS = _public_path('experiments/archive/relation_learning')
OUT = _public_path('experiments/archive/relation_learning/data/entity_error_comps_readout')
OUT.mkdir(parents=True, exist_ok=True)

ENTITY_DIR = _public_path('experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/entity_tracking')
S274 = _public_path('experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout')
S268 = _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval')
S005 = _public_path('experiments/archive/relation_learning/data/vc_seed_replication')
CKS = ["chck_80M", "chck_90M", "chck_100M"]
SEEDS = [43022, 43122]
ROLES = ["V", "C", "R"]


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .")


def read_json(p: pathlib.Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def split_entity_prefix(prefix: str) -> tuple[str, list[str], str]:
    parts = [x.strip() for x in re.split(r"(?<=\.)\s+", prefix.strip()) if x.strip()]
    if len(parts) < 2:
        return prefix.strip(), [], ""
    return parts[0], parts[1:-1], parts[-1]


def parse_initial_contents(initial_sentence: str) -> dict[int, str]:
    s = initial_sentence.strip()
    if s.endswith("."):
        s = s[:-1]
    out: dict[int, str] = {}
    for m in re.finditer(r"Box\s+(\d+)\s+contains\s+(.*?)(?=,\s*Box\s+\d+\s+contains\s+|$)", s):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def parse_query_box(query: str) -> int | None:
    m = re.search(r"Box\s+(\d+)\s+contains\s*$", query.strip())
    return int(m.group(1)) if m else None


def load_entity_items_by_uid() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fn in ["regular.jsonl", "ambiref.jsonl", "move_contents.jsonl"]:
        typ = fn[:-6]
        with (ENTITY_DIR / fn).open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if any("nothing" in str(o).lower() for o in obj.get("options", [])):
                    continue
                depth = int(obj["numops"])
                uid = f"{typ}_{depth}_ops"
                initial, ops, query = split_entity_prefix(obj["input_prefix"])
                qbox = parse_query_box(query)
                init_map = parse_initial_contents(initial)
                stale = init_map.get(qbox) if qbox is not None else None
                options = [str(x) for x in obj["options"]]
                stale_idx = None
                if stale is not None:
                    for i, opt in enumerate(options):
                        if norm(opt) == norm(stale):
                            stale_idx = i
                            break
                out[uid].append({
                    "uid": uid,
                    "type": typ,
                    "numops": depth,
                    "sample_id": obj.get("sample_id"),
                    "example_id": obj.get("example_id"),
                    "options": options,
                    "gold": options[0],
                    "stale_initial": stale,
                    "stale_idx": stale_idx,
                    "stale_is_gold": int(stale_idx == 0) if stale_idx is not None else 0,
                })
    return out


def payload_path_seed43022(role: str, ck: str) -> pathlib.Path:
    role_to_arm = {"V": "deberta_basin1_view", "C": "deberta_basin1_clean_maxgeom", "R": "deberta_basin1_repeat"}
    with (_public_path('experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/score_rows.csv')).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("seed") == "43022" and row.get("arm") == role_to_arm[role] and row.get("checkpoint") == ck and row.get("family") == "Entity":
                return ROOT / row["payload"]
    raise FileNotFoundError((role, ck))


def payload_path_seed43122(role: str, ck: str) -> pathlib.Path:
    if role in {"V", "C"}:
        d = _public_path('experiments/archive/relation_learning/data/vc_seed_replication/per_target') / f"D_{role}_43122_{ck}.json"
        if d.exists():
            return d
        d2 = S005 / f"D_{role}_43122_{ck}.json"
        if d2.exists():
            return d2
    if role == "R":
        d = _public_path('experiments/archive/frontier_consolidation/data/second_basin_entity_ewok_eval/repeat_eval/per_target') / f"second_basin_max_repeat_seed43122_{ck}.json"
        if d.exists():
            return d
    raise FileNotFoundError((role, ck))


def prediction_path_from_payload(p: pathlib.Path, seed: int, role: str, ck: str) -> pathlib.Path:
    obj = read_json(p)
    pred = obj.get("tasks", {}).get("Entity", {}).get("predictions")
    if pred:
        pp = ROOT / pred
        if pp.exists():
            return pp
    # research summary JSON records only logs/scores; predictions live under the output tree.
    if seed == 43122 and role in {"V", "C"}:
        d = _public_path('experiments/archive/relation_learning/data/vc_seed_replication/outputs') / f"D_{role}_43122_{ck}" / "Entity"
        matches = sorted(d.rglob("predictions.json"))
        if matches:
            return matches[-1]
    raise KeyError(f"no Entity predictions in {p}")


def get_prediction_path(seed: int, role: str, ck: str) -> pathlib.Path:
    payload = payload_path_seed43022(role, ck) if seed == 43022 else payload_path_seed43122(role, ck)
    return prediction_path_from_payload(payload, seed, role, ck)


def summarize_entity_predictions() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items_by_uid = load_entity_items_by_uid()
    item_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    path_map: dict[str, str] = {}
    for seed in SEEDS:
        for role in ROLES:
            for ck in CKS:
                pred_path = get_prediction_path(seed, role, ck)
                path_map[f"{seed}_{role}_{ck}"] = rel(pred_path)
                pred_obj = read_json(pred_path)
                for uid, items in sorted(items_by_uid.items()):
                    preds = pred_obj[uid]["predictions"]
                    if len(preds) != len(items):
                        raise RuntimeError({"uid": uid, "pred_len": len(preds), "item_len": len(items), "path": rel(pred_path)})
                    for i, (item, pred_rec) in enumerate(zip(items, preds)):
                        pred = str(pred_rec.get("pred", ""))
                        pred_idx = None
                        for oi, opt in enumerate(item["options"]):
                            if norm(opt) == norm(pred):
                                pred_idx = oi
                                break
                        correct = int(pred_idx == 0)
                        stale_avail = int(item["stale_idx"] is not None)
                        stale_is_gold = int(item["stale_idx"] == 0) if item["stale_idx"] is not None else 0
                        pred_is_stale = int(item["stale_idx"] is not None and pred_idx == item["stale_idx"])
                        item_rows.append({
                            "seed": seed, "arm": role, "checkpoint": ck, "uid": uid,
                            "entity_type": item["type"], "numops": item["numops"], "item_index": i,
                            "correct": correct, "pred_idx": pred_idx if pred_idx is not None else -1,
                            "stale_available": stale_avail, "stale_is_gold": stale_is_gold,
                            "pred_is_stale_initial": pred_is_stale,
                            "wrong_and_stale_initial": int((not correct) and pred_is_stale),
                        })
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in item_rows:
        for group in ["ALL", f"numops_{r['numops']}", f"type_{r['entity_type']}", f"{r['entity_type']}_numops_{r['numops']}"]:
            groups[(r["seed"], r["arm"], r["checkpoint"], group)].append(r)
    for (seed, role, ck, group), vals in sorted(groups.items()):
        wrong = [v for v in vals if not v["correct"]]
        stale_cand = [v for v in vals if v["stale_available"] and not v["stale_is_gold"]]
        wrong_stale_cand = [v for v in stale_cand if not v["correct"]]
        summary_rows.append({
            "seed": seed, "arm": role, "checkpoint": ck, "group": group, "n": len(vals),
            "accuracy_pct": 100.0 * sum(v["correct"] for v in vals) / len(vals),
            "stale_available_not_gold_n": len(stale_cand),
            "stale_pick_pct_among_stale_available_not_gold": 100.0 * sum(v["pred_is_stale_initial"] for v in stale_cand) / len(stale_cand) if stale_cand else float("nan"),
            "stale_pick_pct_among_wrong_stale_available_not_gold": 100.0 * sum(v["pred_is_stale_initial"] for v in wrong_stale_cand) / len(wrong_stale_cand) if wrong_stale_cand else float("nan"),
            "stale_pick_pct_among_all_wrong": 100.0 * sum(v["pred_is_stale_initial"] for v in wrong) / len(wrong) if wrong else float("nan"),
        })
    (_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_paths.json')).write_text(json.dumps(path_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return item_rows, summary_rows


def contrast_entity_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    idx = {(r["seed"], r["arm"], r["checkpoint"], r["group"]): r for r in summary_rows}
    out: list[dict[str, Any]] = []
    for seed in SEEDS:
        for ck in CKS:
            groups = sorted({r["group"] for r in summary_rows if r["seed"] == seed and r["checkpoint"] == ck})
            for group in groups:
                for name, a, b in [("VminusC", "V", "C"), ("VminusR", "V", "R"), ("CminusR", "C", "R")]:
                    ka, kb = (seed, a, ck, group), (seed, b, ck, group)
                    if ka not in idx or kb not in idx:
                        continue
                    ra, rb = idx[ka], idx[kb]
                    out.append({
                        "seed": seed, "checkpoint": ck, "group": group, "contrast": name,
                        "delta_accuracy_pct": ra["accuracy_pct"] - rb["accuracy_pct"],
                        "delta_stale_pick_wrong_pct": ra["stale_pick_pct_among_wrong_stale_available_not_gold"] - rb["stale_pick_pct_among_wrong_stale_available_not_gold"],
                        "delta_stale_pick_all_wrong_pct": ra["stale_pick_pct_among_all_wrong"] - rb["stale_pick_pct_among_all_wrong"],
                        "n": ra["n"],
                    })
    return out


def mean(xs: list[float]) -> float:
    xs = [x for x in xs if math.isfinite(x)]
    return statistics.mean(xs) if xs else float("nan")


def late_entity_contrasts(contrast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in contrast_rows:
        if r["checkpoint"] in CKS:
            groups[(r["seed"], r["group"], r["contrast"])].append(r)
    out = []
    for (seed, group, contrast), vals in sorted(groups.items()):
        out.append({
            "seed": seed, "group": group, "contrast": contrast,
            "late_mean_delta_accuracy_pct": mean([v["delta_accuracy_pct"] for v in vals]),
            "late_mean_delta_stale_pick_wrong_pct": mean([v["delta_stale_pick_wrong_pct"] for v in vals]),
            "late_mean_delta_stale_pick_all_wrong_pct": mean([v["delta_stale_pick_all_wrong_pct"] for v in vals]),
        })
    return out


def comps_vr_seed43022() -> list[dict[str, Any]]:
    rows = []
    with (_public_path('experiments/archive/frontier_consolidation/data/fixed_budget_allocation_readout/contrast_rows.csv')).open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("contrast") == "deberta_basin1_VminusR" and row.get("quantity") == "COMPS" and row.get("checkpoint") in CKS:
                rows.append({k: row[k] for k in row})
    vals = [float(r["delta"]) for r in rows]
    rows.append({
        "contrast": "deberta_basin1_VminusR", "arm_a": "deberta_basin1_view", "arm_b": "deberta_basin1_repeat", "checkpoint": "late_mean_80_90_100M", "words": "", "quantity": "COMPS",
        "delta": mean(vals), "score_a": mean([float(r["score_a"]) for r in rows]), "score_b": mean([float(r["score_b"]) for r in rows])
    })
    return rows


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    preferred = ["seed", "arm", "checkpoint", "group", "contrast", "quantity", "n", "delta", "score_a", "score_b", "accuracy_pct", "delta_accuracy_pct", "stale_available_not_gold_n", "stale_pick_pct_among_wrong_stale_available_not_gold", "delta_stale_pick_wrong_pct"]
    fields = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    item_rows, summary_rows = summarize_entity_predictions()
    contrast_rows = contrast_entity_summary(summary_rows)
    late_rows = late_entity_contrasts(contrast_rows)
    comps_rows = comps_vr_seed43022()
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_item_rows.csv'), item_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_summary.csv'), summary_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_contrasts.csv'), contrast_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_late_contrasts.csv'), late_rows)
    write_csv(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/comps_seed43022_vminusr.csv'), comps_rows)
    result = {
        "status": "ENTITY_ERROR_COMPS_READOUT_DONE",
        "files": {
            "entity_item_rows": rel(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_item_rows.csv')),
            "entity_summary": rel(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_summary.csv')),
            "entity_contrasts": rel(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_contrasts.csv')),
            "entity_late_contrasts": rel(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_prediction_late_contrasts.csv')),
            "comps_vminusr": rel(_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/comps_seed43022_vminusr.csv')),
        },
        "note": "Official predictions confirm the depth accuracy pattern; stale-initial selection is computed only when the queried box's initial content is an explicit non-gold option, so low availability limits this error readout.",
    }
    (_public_path('experiments/archive/relation_learning/data/entity_error_comps_readout/entity_error_comps_summary.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ENTITY LATE CONTRASTS OF INTEREST")
    for r in late_rows:
        if r["group"] in ["numops_0", "numops_3", "numops_4", "numops_5"] and r["contrast"] in ["VminusC", "VminusR", "CminusR"]:
            print(f"seed={r['seed']} {r['group']} {r['contrast']} dAcc={r['late_mean_delta_accuracy_pct']:+.2f} dStaleWrong={r['late_mean_delta_stale_pick_wrong_pct']:+.2f}")
    print("COMPS seed43022 V-R")
    for r in comps_rows:
        print(r)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
