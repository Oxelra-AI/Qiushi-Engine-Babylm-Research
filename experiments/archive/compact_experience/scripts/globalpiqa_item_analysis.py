#!/usr/bin/env python3
"""Item-level GlobalPIQA analysis for the research 2x2 tokenizer×data screen.

Uses already produced official-evaluator prediction files and the local official
GlobalPIQA data. Computes exact per-item correctness, paired effects, and bootstrap
intervals for 10M and 20M endpoints. This does not run models and does not use any
training/evaluation signal for model selection beyond interpreting existing route-screen
outputs.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = _public_path('experiments/archive/compact_experience')
SESSIONS_ROOT = _public_path('experiments/archive')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
DATA = {
    "parallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_parallel/eng_latn.jsonl'),
    "nonparallel": _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/global_piqa_nonparallel/eng_latn.jsonl'),
}
ENDPOINTS = {
    "10M": {
        "out_root": _public_path('experiments/archive/compact_experience/data/tok2x2_noaoa_eval_10M'),
        "targets": {
            "O16": "tok16_official_10M_b128",
            "Q16": "tok16_qwen_10M_b128",
            "O40": "tok40_official_10M_b128",
            "Q40": "tok40_qwen_10M_b128",
        },
    },
    "20M": {
        "out_roots": {
            "O16": _public_path('experiments/archive/compact_experience/data/tok16_b128_noaoa_eval'),
            "Q16": _public_path('experiments/archive/compact_experience/data/tok16_b128_noaoa_eval'),
            "O40": _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval'),
            "Q40": _public_path('experiments/archive/compact_experience/data/tok40_isolation_noaoa_eval'),
        },
        "targets": {
            "O16": "tok16_official_20M_b128",
            "Q16": "tok16_qwen_20M_b128",
            "O40": "tok40_official_20M_b128",
            "Q40": "tok40_qwen_20M_b128",
        },
    },
}
OUT = _public_path('experiments/archive/compact_experience/data/globalpiqa_item_analysis.json')


def norm(s: Any) -> str:
    return " ".join(str(s).strip().split()).lower().rstrip(".")


def load_data(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            exid = str(r["example_id"])
            label = int(r["label"])
            opts = []
            i = 0
            while f"solution{i}" in r:
                opts.append(str(r[f"solution{i}"]))
                i += 1
            rows[exid] = {"label": label, "options": opts, "category": r.get("categories"), "prompt": r.get("prompt")}
    return rows


def payload_path(endpoint: str, cell: str) -> Path:
    cfg = ENDPOINTS[endpoint]
    target = cfg["targets"][cell]
    if endpoint == "20M":
        root = cfg["out_roots"][cell]
    else:
        root = cfg["out_root"]
    return root / "per_target" / f"{target}.json"


def pred_path_from_payload(endpoint: str, cell: str, split: str) -> Path:
    p = json.loads(payload_path(endpoint, cell).read_text())
    col = "GlobalPIQA_parallel" if split == "parallel" else "GlobalPIQA_nonparallel"
    pred = Path(p["tasks"][col]["predictions"])
    if pred.exists():
        return pred
    wp = ROOT / pred
    if wp.exists():
        return wp
    raise FileNotFoundError(pred)


def load_correctness(endpoint: str, cell: str, split: str, data_rows: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, Any]]:
    p = pred_path_from_payload(endpoint, cell, split)
    pred_obj = json.loads(p.read_text())
    correct: Dict[str, int] = {}
    unmatched: List[Dict[str, Any]] = []
    predicted_label_counts: Dict[str, int] = {}
    for exid, info in data_rows.items():
        rec = pred_obj.get(exid)
        if rec is None:
            # Some evaluators key by prediction id prefix, but current files use example_id keys.
            unmatched.append({"example_id": exid, "reason": "missing_prediction"})
            continue
        plist = rec.get("predictions", [])
        if not plist:
            unmatched.append({"example_id": exid, "reason": "empty_predictions"})
            continue
        pred_text = plist[0].get("pred", "")
        pred_norm = norm(pred_text)
        labels = [i for i, opt in enumerate(info["options"]) if norm(opt) == pred_norm]
        if not labels:
            unmatched.append({"example_id": exid, "reason": "unmatched_text", "pred": pred_text, "options": info["options"]})
            continue
        pred_label = labels[0]
        predicted_label_counts[str(pred_label)] = predicted_label_counts.get(str(pred_label), 0) + 1
        correct[exid] = int(pred_label == int(info["label"]))
    return correct, {"prediction_path": str(p), "n": len(correct), "unmatched": unmatched[:12], "unmatched_count": len(unmatched), "predicted_label_counts": predicted_label_counts}


def mean_percent(vals: List[float]) -> float:
    return 100.0 * sum(vals) / max(1, len(vals))


def ci_bootstrap(vals: List[float], n_boot: int = 10000, seed: int = 56043) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_boot):
        boots.append(100.0 * sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {"mean": mean_percent(vals), "lo": boots[int(0.025 * n_boot)], "hi": boots[int(0.975 * n_boot)]}


def paired_effect(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, Any]:
    keys = sorted(set(a) & set(b))
    vals = [a[k] - b[k] for k in keys]
    gained = [k for k in keys if a[k] == 1 and b[k] == 0]
    lost = [k for k in keys if a[k] == 0 and b[k] == 1]
    return {
        "n": len(keys),
        "effect_points": mean_percent(vals),
        "bootstrap_ci_points": ci_bootstrap(vals),
        "gained_items": len(gained),
        "lost_items": len(lost),
        "both_correct": sum(1 for k in keys if a[k] == 1 and b[k] == 1),
        "both_wrong": sum(1 for k in keys if a[k] == 0 and b[k] == 0),
        "net_items": len(gained) - len(lost),
        "gained_examples_head": gained[:10],
        "lost_examples_head": lost[:10],
    }


def paired_interaction(cells: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    keys = sorted(set(cells["Q40"]) & set(cells["O40"]) & set(cells["Q16"]) & set(cells["O16"]))
    vals = [(cells["Q40"][k] - cells["O40"][k]) - (cells["Q16"][k] - cells["O16"][k]) for k in keys]
    hist: Dict[str, int] = {}
    for v in vals:
        hist[str(v)] = hist.get(str(v), 0) + 1
    return {
        "n": len(keys),
        "interaction_points": mean_percent(vals),
        "bootstrap_ci_points": ci_bootstrap(vals, seed=56044),
        "histogram_per_item_values": hist,
        "positive_items": sum(1 for v in vals if v > 0),
        "negative_items": sum(1 for v in vals if v < 0),
        "zero_items": sum(1 for v in vals if v == 0),
    }


def summarize_endpoint(endpoint: str) -> Dict[str, Any]:
    endpoint_report: Dict[str, Any] = {"cells": {}, "splits": {}, "combined": {}}
    all_cells_by_split: Dict[str, Dict[str, Dict[str, int]]] = {}
    for split, data_path in DATA.items():
        data_rows = load_data(data_path)
        cells: Dict[str, Dict[str, int]] = {}
        meta: Dict[str, Any] = {"n_items": len(data_rows), "num_options": sorted(set(len(r["options"]) for r in data_rows.values())), "categories": {}}
        for r in data_rows.values():
            c = str(r.get("category"))
            meta["categories"][c] = meta["categories"].get(c, 0) + 1
        for cell in ["O16", "Q16", "O40", "Q40"]:
            corr, pred_meta = load_correctness(endpoint, cell, split, data_rows)
            cells[cell] = corr
            endpoint_report["cells"].setdefault(cell, {})[split] = {
                "accuracy_points": mean_percent(list(corr.values())),
                "correct_items": sum(corr.values()),
                "n": len(corr),
                "prediction_meta": pred_meta,
            }
        split_effects = {
            "data_effect_16_Q_minus_O": paired_effect(cells["Q16"], cells["O16"]),
            "data_effect_40_Q_minus_O": paired_effect(cells["Q40"], cells["O40"]),
            "tokenizer_effect_official_40_minus_16": paired_effect(cells["O40"], cells["O16"]),
            "tokenizer_effect_qwen_40_minus_16": paired_effect(cells["Q40"], cells["Q16"]),
            "data_x_tokenizer_interaction": paired_interaction(cells),
        }
        endpoint_report["splits"][split] = {"data_meta": meta, "effects": split_effects}
        all_cells_by_split[split] = cells
    # Combined GlobalPIQA treats parallel and nonparallel equally by item count here (103+100),
    # whereas the leaderboard column averages subtask percentages. Both are close but not identical.
    combined_cells: Dict[str, Dict[str, int]] = {}
    for cell in ["O16", "Q16", "O40", "Q40"]:
        d: Dict[str, int] = {}
        for split in DATA:
            for k, v in all_cells_by_split[split][cell].items():
                d[f"{split}:{k}"] = v
        combined_cells[cell] = d
    endpoint_report["combined"] = {
        "item_count_weighted_effects": {
            "data_effect_16_Q_minus_O": paired_effect(combined_cells["Q16"], combined_cells["O16"]),
            "data_effect_40_Q_minus_O": paired_effect(combined_cells["Q40"], combined_cells["O40"]),
            "tokenizer_effect_official_40_minus_16": paired_effect(combined_cells["O40"], combined_cells["O16"]),
            "tokenizer_effect_qwen_40_minus_16": paired_effect(combined_cells["Q40"], combined_cells["Q16"]),
            "data_x_tokenizer_interaction": paired_interaction(combined_cells),
        }
    }
    return endpoint_report


def main() -> None:
    report = {
        "status": "GLOBALPIQA_ITEM_ANALYSIS",
        "purpose": "Use existing GlobalPIQA predictions to test whether the 40k interaction is a robust broad signal or a small GlobalPIQA swing; no model execution.",
        "important_note": "No unsupported chance-level claim is made; num_options are reported from the data. Leaderboard GlobalPIQA averages parallel and nonparallel subtask percentages.",
        "endpoints": {ep: summarize_endpoint(ep) for ep in ["10M", "20M"]},
    }
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {
        "status": report["status"],
        "out": str(OUT),
        "endpoint_effects": {}
    }
    for ep, er in report["endpoints"].items():
        compact["endpoint_effects"][ep] = {
            "parallel_interaction_points": er["splits"]["parallel"]["effects"]["data_x_tokenizer_interaction"]["interaction_points"],
            "parallel_interaction_ci": er["splits"]["parallel"]["effects"]["data_x_tokenizer_interaction"]["bootstrap_ci_points"],
            "nonparallel_interaction_points": er["splits"]["nonparallel"]["effects"]["data_x_tokenizer_interaction"]["interaction_points"],
            "nonparallel_interaction_ci": er["splits"]["nonparallel"]["effects"]["data_x_tokenizer_interaction"]["bootstrap_ci_points"],
            "data_effect_40_parallel": er["splits"]["parallel"]["effects"]["data_effect_40_Q_minus_O"]["effect_points"],
            "data_effect_40_nonparallel": er["splits"]["nonparallel"]["effects"]["data_effect_40_Q_minus_O"]["effect_points"],
        }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
