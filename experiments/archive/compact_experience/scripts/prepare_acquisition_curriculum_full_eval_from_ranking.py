#!/usr/bin/env python3
"""Prepare frozen full-eval targets from acquisition-curriculum no-AoA ranking."""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import argparse
import json
import pathlib
from typing import Any, Dict, List

ROOT = _public_path('experiments/archive/compact_experience')
RANKING = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_trajectory_ranking.json')
OUT = _public_path('experiments/archive/compact_experience/data/acquisition_curriculum_full_eval_targets.json')
NOTE = _public_path('research/notes/compact_experience/acquisition_curriculum_full_eval_targets.md')
FAMILIES = ["acq_A0_baseline_order", "acq_A0p_repeated_random", "acq_A1_frequency_only", "acq_A2_full", "acq_A3_random_label", "acq_A4_reverse"]


def load(p: pathlib.Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def target_name(row: Dict[str, Any]) -> str:
    ckpt = str(row["checkpoint"])
    if not ckpt.startswith("chck_"):
        raise ValueError(ckpt)
    return f"{row['target']}_{ckpt.removeprefix('chck_')}"


def key(row: Dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("target")), str(row.get("checkpoint")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(RANKING))
    ap.add_argument("--top_k_any", type=int, default=2)
    args = ap.parse_args()
    rp = pathlib.Path(args.ranking)
    if not rp.exists():
        raise SystemExit(f"missing ranking: {rp}")
    r = load(rp)
    selected: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    best = r.get("best_by_target") or {}
    for fam in FAMILIES:
        rec = best.get(fam)
        if isinstance(rec, dict) and key(rec) not in seen:
            selected.append({"reason": f"best_{fam}_equal7_no_aoa", "row": rec})
            seen.add(key(rec))
    family_top = [rec for rec in r.get("top_rows", []) if isinstance(rec, dict) and str(rec.get("target")) in FAMILIES]
    for rec in family_top[: max(0, args.top_k_any)]:
        if key(rec) not in seen:
            selected.append({"reason": "global_top_equal7_no_aoa", "row": rec})
            seen.add(key(rec))
    targets = [target_name(x["row"]) for x in selected]
    payload = {"status":"ACQUISITION_CURRICULUM_FULL_EVAL_TARGETS","non_leakage_statement":"Targets selected only from no-AoA BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading ranking. SuperGLUE and AoA are final measurements only.","ranking":str(rp),"targets":targets,"selected":[dict(x, target=target_name(x["row"])) for x in selected],"key_deltas":{k:r.get(k) for k in ["a2_minus_contemporaneous_a0_best","a2_minus_repeated_random_a0p_best","a2_minus_a1_best","a2_minus_a3_best","a2_minus_a4_reverse_best","a2_minus_historical_clean_reference_best"]}}
    _public_path('experiments/archive/compact_experience/data').mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research acquisition-curriculum full-eval target freeze","",payload["non_leakage_statement"],"",f"Ranking source: `{rp}`","","## Targets"]
    for item,t in zip(selected,targets):
        row=item["row"]
        lines.append(f"- `{t}` reason={item['reason']} equal7={row.get('equal7_full_eval')} BLiMP={row.get('BLiMP')} Supplement={row.get('Supplement')} EWoK={row.get('EWoK')} Entity={row.get('Entity')} COMPS={row.get('COMPS')} GlobalPIQA={row.get('GlobalPIQA')} Reading={row.get('Reading')}")
    _public_path('research/notes/compact_experience').mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"out":str(OUT),"note":str(NOTE),"targets":targets}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
