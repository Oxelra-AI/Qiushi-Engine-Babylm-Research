#!/usr/bin/env python3
"""research: compare Supplement row-slice reports across endpoints.

CPU-only.  Takes two or more JSONs produced by supplement_prediction_slices.py
and writes a compact comparison of subtask, affected/unaffected, and structural
slice accuracies.  Used here to validate interpretation on already-existing
old-tokenizer endpoints and later reusable for compliant endpoints.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_OUT = STUDY / "data/supplement_prediction_slices"


def load(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def acc(d: dict[str, Any] | None) -> float | None:
    if not isinstance(d, dict):
        return None
    v = d.get("accuracy")
    return float(v) if v is not None else None


def score_at(payload: dict[str, Any], section: str, key: str) -> dict[str, Any]:
    s = payload["score"]
    d = s.get(section, {}).get(key, {})
    if not isinstance(d, dict):
        d = {}
    return {"n": d.get("n", 0), "correct": d.get("correct", 0), "accuracy": acc(d)}


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return 100.0 * (a - b)


def fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.3f}"


def write_md(payload: dict[str, Any], out_md: pathlib.Path) -> None:
    names = payload["tags"]
    lines = []
    lines.append("# research Supplement row-slice comparison")
    lines.append("")
    lines.append("This comparison uses only existing saved Supplement predictions. It validates the post-evaluation slice tool and gives a reference for interpreting future compliant-tokenizer endpoint differences.")
    lines.append("")
    lines.append("## Overall and affected-row surface")
    lines.append("")
    lines.append("| tag | Supplement macro | affected acc | unaffected acc | affected n | unaffected n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for rec in payload["records"]:
        s = rec["score"]
        aff = s["by_affected"].get("affected", {})
        una = s["by_affected"].get("unaffected", {})
        lines.append(f"| {rec['tag']} | {s['official_supplement_subtask_macro_accuracy_0to100']:.6f} | {100.0*(aff.get('accuracy') or 0.0):.3f} | {100.0*(una.get('accuracy') or 0.0):.3f} | {aff.get('n', 0)} | {una.get('n', 0)} |")
    if payload.get("pairwise_deltas"):
        lines.append("")
        lines.append("## Pairwise deltas in percentage points (first minus second)")
        lines.append("")
        lines.append("| first - second | Supplement macro | affected acc | unaffected acc | qa easy aff | qa tricky aff | turn-taking aff | hypernym | subject-aux |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for d in payload["pairwise_deltas"]:
            lines.append(
                f"| {d['first']} - {d['second']} | {fmt(d['supplement_macro_delta_pp'])} | {fmt(d['affected_delta_pp'])} | {fmt(d['unaffected_delta_pp'])} | {fmt(d['qa_congruence_easy_affected_delta_pp'])} | {fmt(d['qa_congruence_tricky_affected_delta_pp'])} | {fmt(d['turn_taking_affected_delta_pp'])} | {fmt(d['hypernym_delta_pp'])} | {fmt(d['subject_aux_inversion_delta_pp'])} |"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("For the two already-existing old-tokenizer endpoints, the compact-view-reinvest model's Supplement macro advantage over clean-Qwen is not carried by the research newline-affected rows: affected-row accuracy is slightly lower, while unaffected-row accuracy is higher. Thus future research-versus-byte-alphabet compliant endpoint differences should be localized with this script rather than attributed wholesale to newline `<unk>` exposure.")
    lines.append("")
    lines.append(f"Full JSON: `{payload['out_json']}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tag", default="existing_oldtok_reinvest_vs_clean")
    args = ap.parse_args()
    records = [load(pathlib.Path(p)) for p in args.inputs]
    tags = [r["tag"] for r in records]
    pairwise = []
    if len(records) >= 2:
        # Compare every later input to the first?  The default call supplies reinvest first, clean second.
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                sa, sb = a["score"], b["score"]
                rec = {
                    "first": a["tag"],
                    "second": b["tag"],
                    "supplement_macro_delta_pp": sa["official_supplement_subtask_macro_accuracy_0to100"] - sb["official_supplement_subtask_macro_accuracy_0to100"],
                    "affected_delta_pp": delta(acc(sa["by_affected"].get("affected")), acc(sb["by_affected"].get("affected"))),
                    "unaffected_delta_pp": delta(acc(sa["by_affected"].get("unaffected")), acc(sb["by_affected"].get("unaffected"))),
                    "qa_congruence_easy_affected_delta_pp": delta(acc(sa["by_task_and_affected"].get("qa_congruence_easy::affected")), acc(sb["by_task_and_affected"].get("qa_congruence_easy::affected"))),
                    "qa_congruence_tricky_affected_delta_pp": delta(acc(sa["by_task_and_affected"].get("qa_congruence_tricky::affected")), acc(sb["by_task_and_affected"].get("qa_congruence_tricky::affected"))),
                    "turn_taking_affected_delta_pp": delta(acc(sa["by_task_and_affected"].get("turn_taking::affected")), acc(sb["by_task_and_affected"].get("turn_taking::affected"))),
                    "hypernym_delta_pp": delta(acc(sa["by_task"].get("hypernym")), acc(sb["by_task"].get("hypernym"))),
                    "subject_aux_inversion_delta_pp": delta(acc(sa["by_task"].get("subject_aux_inversion")), acc(sb["by_task"].get("subject_aux_inversion"))),
                }
                pairwise.append(rec)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.tag}_comparison.json"
    out_md = out_dir / f"{args.tag}_comparison.md"
    payload = {
        "status": "SUPPLEMENT_SLICE_COMPARISON",
        "tag": args.tag,
        "input_paths": args.inputs,
        "tags": tags,
        "records": records,
        "pairwise_deltas": pairwise,
        "elapsed_sec": round(time.time() - t0, 3),
        "out_json": str(out_json),
        "out_md": str(out_md),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(payload, out_md)
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "out_md": str(out_md),
        "tags": tags,
        "pairwise_deltas": pairwise,
        "elapsed_sec": payload["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
