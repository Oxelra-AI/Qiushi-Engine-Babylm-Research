#!/usr/bin/env python3
"""Summarize research cluster-heldout diagnostic results.

The diagnostic measures MLM loss on selected-cluster held-out sentences and on
seen cluster sentences. Lower E1 heldout loss than E2/E3/E4 would support the
natural complementary-evidence mechanism independently of downstream benchmarks.
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

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_IN = _public_path('experiments/archive/compact_experience/data/cluster_heldout_diagnostic/heldout_diagnostic_results.json')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/cluster_heldout_diagnostic/heldout_diagnostic_summary.json')
DEFAULT_NOTE = _public_path('research/notes/compact_experience/cluster_heldout_diagnostic_result.md')
ARMS = ["E1_true_cluster", "E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--note", default=str(DEFAULT_NOTE))
    args = ap.parse_args()

    in_path = pathlib.Path(args.input)
    out_path = pathlib.Path(args.output)
    note_path = pathlib.Path(args.note)
    if not in_path.exists():
        raise FileNotFoundError(in_path)
    obj = json.loads(in_path.read_text())
    rows = list(obj.get("results", []))
    if not rows:
        raise SystemExit("No heldout diagnostic rows found")

    # Rank by heldout loss per checkpoint and find E1 contrasts.
    by_ckpt: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ckpt.setdefault(str(r["checkpoint"]), []).append(r)

    ckpt_summaries: dict[str, Any] = {}
    for ckpt, entries in sorted(by_ckpt.items()):
        ranked = sorted(entries, key=lambda r: float(r["heldout_loss"]))
        by_arm = {r["arm"]: r for r in entries}
        contrasts = {}
        if "E1_true_cluster" in by_arm:
            e1 = by_arm["E1_true_cluster"]
            for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]:
                if ctrl in by_arm:
                    c = by_arm[ctrl]
                    contrasts[f"E1_minus_{ctrl}"] = {
                        "heldout_loss": float(e1["heldout_loss"]) - float(c["heldout_loss"]),
                        "seen_loss": float(e1["seen_loss"]) - float(c["seen_loss"]),
                        "generalization_gap": float(e1["generalization_gap"]) - float(c["generalization_gap"]),
                    }
        ckpt_summaries[ckpt] = {
            "ranked_by_heldout_loss": ranked,
            "contrasts": contrasts,
            "E1_best_heldout": bool(ranked and ranked[0].get("arm") == "E1_true_cluster"),
        }

    # Overall best row per arm across checkpoints.
    best_by_arm = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r.get("arm") == arm]
        if arm_rows:
            best_by_arm[arm] = min(arm_rows, key=lambda r: float(r["heldout_loss"]))

    e1_best_any = bool(
        "E1_true_cluster" in best_by_arm and all(
            ctrl in best_by_arm and float(best_by_arm["E1_true_cluster"]["heldout_loss"]) < float(best_by_arm[ctrl]["heldout_loss"])
            for ctrl in ["E2_anchor_shuffle", "E3_anchor_repeat", "E4_untouched_tail"]
        )
    )

    payload = {
        "status": "CLUSTER_HELDOUT_DIAGNOSTIC_SUMMARY",
        "created_utc": now(),
        "input": str(in_path),
        "n_heldout_sentences": obj.get("n_heldout_sentences"),
        "n_seen_sentences": obj.get("n_seen_sentences"),
        "best_by_arm": best_by_arm,
        "per_checkpoint": ckpt_summaries,
        "E1_best_across_arm_bests": e1_best_any,
        "non_leakage_statement": "Diagnostic uses only training-corpus cluster heldout/seen sentences, not downstream benchmark labels/items and not AoA/CDI material.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# research cluster held-out diagnostic",
        "",
        f"Created UTC: {payload['created_utc']}",
        f"Held-out sentences: {payload['n_heldout_sentences']}; seen sentences: {payload['n_seen_sentences']}.",
        "",
        "## Best held-out loss per arm",
        "",
        "| arm | checkpoint | heldout loss | seen loss | generalization gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        r = best_by_arm.get(arm)
        if r:
            lines.append(f"| {arm} | {r['checkpoint']} | {float(r['heldout_loss']):.4f} | {float(r['seen_loss']):.4f} | {float(r['generalization_gap']):+.4f} |")
    lines.extend(["", "## E1 contrasts by checkpoint", ""])
    for ckpt, csum in ckpt_summaries.items():
        lines.append(f"### {ckpt}")
        for name, vals in csum["contrasts"].items():
            lines.append(f"- `{name}`: heldout {vals['heldout_loss']:+.4f}; seen {vals['seen_loss']:+.4f}; gap {vals['generalization_gap']:+.4f}")
    lines.extend(["", f"E1 best across arm bests: `{e1_best_any}`.", ""])
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n")

    print(json.dumps({"out": str(out_path), "note": str(note_path), "E1_best_across_arm_bests": e1_best_any}, indent=2), flush=True)


if __name__ == "__main__":
    main()
