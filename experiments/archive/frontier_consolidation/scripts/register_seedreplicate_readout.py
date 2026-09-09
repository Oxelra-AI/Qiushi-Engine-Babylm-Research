#!/usr/bin/env python3
"""research: file-only readout for the seed43122 register replicate.

Reads the original seed43022 register contrast and any completed research
seed43122 broad-family scoring payloads. It performs no model inference, no
training, no official evaluation, no upload, and no leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import pathlib
import statistics
import time
from typing import Any


def find_root() -> pathlib.Path:
    return _PUBLIC_ROOT


ROOT = find_root()
WS = ROOT / "experiments/archive" / 'frontier_consolidation'
SEED43022_CONTRASTS = WS / "data" / "register_readout_only" / "decisive_contrasts.csv"
SEED43022_ARM_SCORES = WS / "data" / "register_readout_only" / "arm_scores_vs_clean.csv"
SEED43122_ROOT = WS / "data" / "register_seed43122_decisive_eval"
OUT_DIR = WS / "data" / "register_seedreplicate_readout"
FAMILIES = ["BLiMP", "Supplement", "EWoK", "COMPS"]
CHECKPOINTS = ["chck_80M", "chck_100M"]
ARM_KEYS = {
    "childspeech": "regmax_childspeech_seed43122",
    "adultprose": "regmax_adultprose_seed43122",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_score(x: Any) -> float | None:
    try:
        v = float(x)
        if v == v and abs(v) < 1e9:
            return v
    except Exception:
        pass
    return None


def per_target_path(root: pathlib.Path, arm_short: str, ck: str) -> pathlib.Path:
    arm_key = ARM_KEYS[arm_short]
    # research wrapper writes each arm under root/<arm_short>/per_target.
    return root / arm_short / "per_target" / f"{arm_key}_{ck}.json"


def load_seed43122_scores(root: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for arm_short, arm_key in ARM_KEYS.items():
        for ck in CHECKPOINTS:
            p = per_target_path(root, arm_short, ck)
            if not p.exists():
                missing.append(rel(p))
                continue
            payload = read_json(p)
            tasks = payload.get("tasks", {}) if isinstance(payload, dict) else {}
            row: dict[str, Any] = {
                "seed": "43122",
                "arm_short": arm_short,
                "arm": arm_key,
                "checkpoint": ck,
                "per_target": rel(p),
            }
            complete = True
            for fam in FAMILIES:
                score = finite_score((tasks.get(fam) or {}).get("score") if isinstance(tasks, dict) else None)
                row[fam] = score
                if score is None:
                    complete = False
            row["complete_broad4"] = complete
            row["exEntity4"] = round(statistics.mean(float(row[f]) for f in FAMILIES), 6) if complete else None
            rows.append(row)
    return rows, missing


def seed43022_summary() -> list[dict[str, Any]]:
    rows = []
    for r in read_csv_rows(SEED43022_CONTRASTS):
        if r.get("contrast") != "childspeech_minus_adultprose":
            continue
        out: dict[str, Any] = {"seed": "43022", "checkpoint": r.get("checkpoint"), "contrast": "childspeech_minus_adultprose"}
        for fam in FAMILIES + ["exEntity4", "cheap5_noReading"]:
            out[fam] = finite_score(r.get(fam))
        rows.append(out)
    return rows


def compute_contrasts(seed43122_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["arm_short"], r["checkpoint"]): r for r in seed43122_rows}
    out: list[dict[str, Any]] = []
    for ck in CHECKPOINTS:
        child = by_key.get(("childspeech", ck))
        adult = by_key.get(("adultprose", ck))
        row: dict[str, Any] = {"seed": "43122", "checkpoint": ck, "contrast": "childspeech_minus_adultprose"}
        complete = bool(child and adult and child.get("complete_broad4") and adult.get("complete_broad4"))
        for fam in FAMILIES:
            cv = finite_score(child.get(fam)) if child else None
            av = finite_score(adult.get(fam)) if adult else None
            row[fam] = round(cv - av, 6) if cv is not None and av is not None else None
        row["exEntity4"] = round(statistics.mean(float(row[f]) for f in FAMILIES), 6) if complete else None
        row["complete_broad4"] = complete
        out.append(row)
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed43122-root", default=str(SEED43122_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.seed43122_root)
    if not root.is_absolute():
        root = ROOT / root
    out_dir = pathlib.Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows43122, missing = load_seed43122_scores(root)
    contrasts43122 = compute_contrasts(rows43122)
    contrasts43022 = seed43022_summary()
    seed43122_complete = all(r.get("complete_broad4") for r in contrasts43122) and bool(contrasts43122)

    payload = {
        "status": "REGISTER_SEEDREPLICATE_READOUT" if seed43122_complete else "REGISTER_SEEDREPLICATE_NOT_COMPLETE",
        "created_utc": now(),
        "seed43122_root": rel(root),
        "seed43022_contrasts_source": rel(SEED43022_CONTRASTS),
        "families": FAMILIES,
        "checkpoints": CHECKPOINTS,
        "missing_seed43122_payloads": missing,
        "seed43022_child_minus_adult": contrasts43022,
        "seed43122_arm_scores": rows43122,
        "seed43122_child_minus_adult": contrasts43122,
        "scientific_reading": "When complete, compare the sign and magnitude of seed43122 childspeech_removed minus adultprose_removed broad-family contrasts against seed43022. Reproduction of the negative sign at both endpoints makes the displaced developmental/spoken value more durable; sign loss makes seed43022 another basin-specific allocation.",
        "no_model_inference_training_official_eval_upload_or_leaderboard": True,
    }
    write_json(out_dir / "register_seedreplicate_readout.json", payload)
    write_csv(out_dir / "seed43122_arm_scores.csv", rows43122, ["seed", "arm_short", "arm", "checkpoint", *FAMILIES, "exEntity4", "complete_broad4", "per_target"])
    write_csv(out_dir / "register_child_minus_adult_contrasts.csv", contrasts43022 + contrasts43122, ["seed", "checkpoint", "contrast", *FAMILIES, "exEntity4", "cheap5_noReading", "complete_broad4"])

    lines = [
        "# research register seed-replication readout",
        "",
        "File-only reader for the research seed43122 register scorer outputs. It does not run models or official evaluation.",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Seed43022 childspeech_removed − adultprose_removed",
    ]
    for r in contrasts43022:
        lines.append(f"- {r.get('checkpoint')}: BLiMP {r.get('BLiMP')}, Supplement {r.get('Supplement')}, EWoK {r.get('EWoK')}, COMPS {r.get('COMPS')}, exEntity4 {r.get('exEntity4')}, cheap5(no Reading) {r.get('cheap5_noReading')}")
    lines.append("")
    lines.append("## Seed43122 childspeech_removed − adultprose_removed")
    if not seed43122_complete:
        lines.append(f"- Not complete. Missing payload count: {len(missing)}")
    for r in contrasts43122:
        lines.append(f"- {r.get('checkpoint')}: BLiMP {r.get('BLiMP')}, Supplement {r.get('Supplement')}, EWoK {r.get('EWoK')}, COMPS {r.get('COMPS')}, exEntity4 {r.get('exEntity4')}, complete {r.get('complete_broad4')}")
    lines.extend([
        "",
        "## Files",
        f"- JSON: `{rel(out_dir / 'register_seedreplicate_readout.json')}`",
        f"- arm scores: `{rel(out_dir / 'seed43122_arm_scores.csv')}`",
        f"- contrasts: `{rel(out_dir / 'register_child_minus_adult_contrasts.csv')}`",
    ])
    (out_dir / "register_seedreplicate_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["files"] = {
        "json": rel(out_dir / "register_seedreplicate_readout.json"),
        "arm_scores_csv": rel(out_dir / "seed43122_arm_scores.csv"),
        "contrast_csv": rel(out_dir / "register_child_minus_adult_contrasts.csv"),
        "summary_md": rel(out_dir / "register_seedreplicate_readout.md"),
    }
    write_json(out_dir / "register_seedreplicate_readout.json", payload)
    print(json.dumps({"status": payload["status"], "missing": len(missing), "files": payload["files"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
