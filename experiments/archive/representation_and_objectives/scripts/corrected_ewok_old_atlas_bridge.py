#!/usr/bin/env python3
"""research: bridge corrected-tokenizer EWoK predictions to old atlas row classes.

This CPU-only post-delivery script reads the official EWoK predictions produced by
the running corrected-tokenizer full evaluations and maps them onto the old
four-cell EWoK atlas classes from research.  It avoids another model forward pass:
correctness is reconstructed by comparing the selected prediction sentence with
candidate0/candidate1 from the pristine official EWoK rows.

If corrected predictions are not present yet, it writes a WAITING JSON.
"""
from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
EWOK_DIR = WORKSPACE / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
OLD_ATLAS = WORKSPACE / "data/full_old_ewok_atlas_synthesis/full_old_ewok_atlas_synthesis.json"
EXPOSURE = WORKSPACE / "data/ewok_old_atlas_training_exposure_link/ewok_old_atlas_training_exposure_link.json"
OUT_DIR = WORKSPACE / "data/corrected_ewok_old_atlas_bridge"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/corrected_ewok_old_atlas_bridge.md')
PRED_PATHS = {
    "43022": WORKSPACE / "data/strictsmalltok_seed43022_official_ewok/official_outputs/EWoK/chck_100M/official_ewok_strictsmalltok_seed43022/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
    "43122": WORKSPACE / "data/strictsmalltok_seed43122_official_ewok/official_outputs/EWoK/chck_100M/official_ewok_strictsmalltok_seed43122/zero_shot/mlm/ewok/ewok_filtered/predictions.json",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_ewok_candidates() -> dict[str, dict[str, Any]]:
    rows = {}
    for p in sorted(EWOK_DIR.glob("*.jsonl")):
        dom = p.stem
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                raw = json.loads(line)
                uid = f"{dom}_{idx}"
                context1 = raw["Context1"]
                context2 = raw["Context2"]
                target1 = raw["Target1"]
                cand0 = " ".join([context1, target1]).strip()
                cand1 = " ".join([context2, target1]).strip()
                rows[uid] = {
                    "uid": uid,
                    "domain": raw.get("Domain", dom),
                    "file_domain": dom,
                    "idx": idx,
                    "ConceptA": raw.get("ConceptA"),
                    "ConceptB": raw.get("ConceptB"),
                    "ContextType": raw.get("ContextType"),
                    "ContextDiff": raw.get("ContextDiff"),
                    "TargetDiff": raw.get("TargetDiff"),
                    "cand0": cand0,
                    "cand1": cand1,
                }
    return rows


def load_predictions(path: Path, candidates: dict[str, dict[str, Any]]) -> dict[str, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for dom, block in raw.items():
        preds = block.get("predictions", []) if isinstance(block, dict) else []
        for rec in preds:
            uid = rec.get("id")
            pred = (rec.get("pred") or "").strip()
            if uid not in candidates:
                raise KeyError({"prediction_id_not_in_candidates": uid, "path": str(path)})
            cand = candidates[uid]
            if pred == cand["cand0"]:
                out[uid] = 1
            elif pred == cand["cand1"]:
                out[uid] = 0
            else:
                # Sometimes whitespace normalization differs; try a collapsed comparison.
                pred_norm = " ".join(pred.split())
                cand0_norm = " ".join(cand["cand0"].split())
                cand1_norm = " ".join(cand["cand1"].split())
                if pred_norm == cand0_norm:
                    out[uid] = 1
                elif pred_norm == cand1_norm:
                    out[uid] = 0
                else:
                    raise ValueError({"prediction_not_candidate": {"uid": uid, "pred": pred[:200], "cand0": cand["cand0"][:200], "cand1": cand["cand1"][:200]}})
    return out


def load_old_rows() -> dict[str, dict[str, Any]]:
    atlas = json.loads(OLD_ATLAS.read_text(encoding="utf-8"))
    src_csv = USER_ROOT / atlas["source_csv"]
    grouped: dict[str, dict[str, Any]] = {}
    with src_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            uid = rec["uid"]
            mk = rec["model_key"]
            g = grouped.setdefault(uid, {
                "uid": uid,
                "domain": rec.get("domain"),
                "idx": int(rec.get("idx", -1)),
                "ConceptA": rec.get("ConceptA") or "",
                "ConceptB": rec.get("ConceptB") or "",
                "ContextType": rec.get("ContextType") or "",
                "ContextDiff": rec.get("ContextDiff") or "",
                "TargetDiff": rec.get("TargetDiff") or "",
                "old_correct": {},
                "old_margin": {},
            })
            g["old_correct"][mk] = int(float(rec["correct"]))
            g["old_margin"][mk] = float(rec["margin_c0_minus_c1"])
    out = {}
    for uid, r in grouped.items():
        if not all(m in r["old_correct"] for m in ["clean430", "reinv430", "clean431", "reinv431"]):
            continue
        pat = "".join(str(r["old_correct"][m]) for m in ["clean430", "reinv430", "clean431", "reinv431"])
        te430 = r["old_correct"]["reinv430"] - r["old_correct"]["clean430"]
        te431 = r["old_correct"]["reinv431"] - r["old_correct"]["clean431"]
        r.update({"old_pattern": pat, "old_te430": te430, "old_te431": te431, "old_interaction": te431 - te430})
        out[uid] = r
    return out


def acc(rs: list[dict[str, Any]], key: str) -> float | None:
    if not rs:
        return None
    return sum(r[key] for r in rs) / len(rs) * 100.0


def summarize_group(rs: list[dict[str, Any]], label: str) -> dict[str, Any]:
    n = len(rs)
    if n == 0:
        return {"label": label, "n": 0}
    patterns = Counter(f"{r['corrected430']}{r['corrected431']}" for r in rs)
    return {
        "label": label,
        "n": n,
        "corrected_accuracy": {"seed43022": acc(rs, "corrected430"), "seed43122": acc(rs, "corrected431")},
        "corrected_seed_delta_431_minus_430_pp": acc(rs, "corrected431") - acc(rs, "corrected430"),
        "old_inherited_accuracy_reinvest": {
            "seed43022": sum(r["old_correct"]["reinv430"] for r in rs) / n * 100.0,
            "seed43122": sum(r["old_correct"]["reinv431"] for r in rs) / n * 100.0,
        },
        "corrected_minus_old_reinvest_pp": {
            "seed43022": acc(rs, "corrected430") - (sum(r["old_correct"]["reinv430"] for r in rs) / n * 100.0),
            "seed43122": acc(rs, "corrected431") - (sum(r["old_correct"]["reinv431"] for r in rs) / n * 100.0),
        },
        "corrected_pattern_430431_counts": dict(sorted(patterns.items(), key=lambda kv: (-kv[1], kv[0]))),
        "corrected_seed_agreement_frac": sum(1 for r in rs if r["corrected430"] == r["corrected431"]) / n,
    }


def group_table(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[str(r.get(key) or "<missing>")].append(r)
    table = [summarize_group(rs, k) | {"group_key": key} for k, rs in by.items()]
    table.sort(key=lambda d: (d["corrected_seed_delta_431_minus_430_pp"], -d["n"], d["label"]))
    return table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [str(p.relative_to(USER_ROOT)) for p in PRED_PATHS.values() if not p.exists()]
    out_json = OUT_DIR / "corrected_ewok_old_atlas_bridge.json"
    if missing:
        payload = {
            "status": "CORRECTED_EWOK_OLD_ATLAS_BRIDGE_WAITING",
            "created_utc": now_utc(),
            "missing_prediction_files": missing,
            "expected_prediction_files": {k: str(v.relative_to(USER_ROOT)) for k, v in PRED_PATHS.items()},
            "interpretation": "Corrected-tokenizer official EWoK predictions are not both present yet; rerun after both corrected-tokenizer evaluations complete or after repairing any failed EWoK column.",
        }
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "out_json": str(out_json.relative_to(USER_ROOT)), "missing": missing}, indent=2), flush=True)
        return

    candidates = load_ewok_candidates()
    old_rows = load_old_rows()
    pred430 = load_predictions(PRED_PATHS["43022"], candidates)
    pred431 = load_predictions(PRED_PATHS["43122"], candidates)
    if len(pred430) != 7618 or len(pred431) != 7618:
        raise RuntimeError({"expected_7618": True, "n430": len(pred430), "n431": len(pred431)})
    rows = []
    for uid, old in old_rows.items():
        rr = dict(old)
        rr["corrected430"] = pred430[uid]
        rr["corrected431"] = pred431[uid]
        rr["corrected_pattern_430431"] = f"{pred430[uid]}{pred431[uid]}"
        rr["corrected_seed_delta_item"] = pred431[uid] - pred430[uid]
        rows.append(rr)
    if len(rows) != 7618:
        raise RuntimeError({"expected_rows": 7618, "found": len(rows)})

    subsets = {
        "all": rows,
        "old_negative_interaction": [r for r in rows if r["old_interaction"] < 0],
        "old_positive_interaction": [r for r in rows if r["old_interaction"] > 0],
        "old_0110_seed430_help_seed431_hurt": [r for r in rows if r["old_pattern"] == "0110"],
        "old_1001_opposite_seed_pattern": [r for r in rows if r["old_pattern"] == "1001"],
        "old_all_correct_1111": [r for r in rows if r["old_pattern"] == "1111"],
        "old_all_wrong_0000": [r for r in rows if r["old_pattern"] == "0000"],
    }
    subset_summary = {name: summarize_group(rs, name) for name, rs in subsets.items()}
    domain_table = group_table(rows, "domain")
    targetdiff_table = group_table(rows, "TargetDiff")
    contexttype_table = group_table(rows, "ContextType")
    contextdiff_table = group_table(rows, "ContextDiff")

    payload = {
        "status": "CORRECTED_EWOK_OLD_ATLAS_BRIDGE_READY",
        "created_utc": now_utc(),
        "prediction_files": {k: str(v.relative_to(USER_ROOT)) for k, v in PRED_PATHS.items()},
        "n_rows": len(rows),
        "subset_summary": subset_summary,
        "domain_table": domain_table,
        "targetdiff_table": targetdiff_table,
        "contexttype_table": contexttype_table,
        "contextdiff_table": contextdiff_table,
        "interpretation": [
            "This bridge uses official corrected-tokenizer EWoK predictions only; it does not score models again.",
            "Use subset_summary.old_0110 and old_negative_interaction to test whether corrected-tokenizer EWoK preserves or erases the old inherited-tokenizer seed-polarized failure surface.",
        ],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTE.write_text("\n".join([
        "# research corrected EWoK old-atlas bridge",
        "",
        f"Status: ready; rows {len(rows)}.",
        f"Overall corrected EWoK micro accuracy from predictions: seed43022 {subset_summary['all']['corrected_accuracy']['seed43022']:.4f}, seed43122 {subset_summary['all']['corrected_accuracy']['seed43122']:.4f}.",
        f"Old negative-interaction subset corrected accuracy: seed43022 {subset_summary['old_negative_interaction']['corrected_accuracy']['seed43022']:.4f}, seed43122 {subset_summary['old_negative_interaction']['corrected_accuracy']['seed43122']:.4f}.",
        "This file is produced only after both corrected official EWoK predictions exist.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "note": str(NOTE.relative_to(USER_ROOT)),
        "n_rows": len(rows),
        "corrected_micro_430": subset_summary["all"]["corrected_accuracy"]["seed43022"],
        "corrected_micro_431": subset_summary["all"]["corrected_accuracy"]["seed43122"],
        "old_negative_subset_corrected": subset_summary["old_negative_interaction"]["corrected_accuracy"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
