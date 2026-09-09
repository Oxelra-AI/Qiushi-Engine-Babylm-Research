#!/usr/bin/env python3
"""research: Full BPE and supervised-mass audit across all changed rows for the factorial.

Measures per-arm BPE pieces, word groups, and expected WWM target mass
across all 3,006 changed rows (not just a sample), plus the interaction
(HS-LS)-(HD-LD) in each dimension. This is the construction-level verification
that the factorial is not confounded by differential supervised mass.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
POOL_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/data/factorial_streams"
TOKENIZER_DIR = USER_ROOT / "experiments/archive/frontier_consolidation/data/compliant_tokenizer"
DEFAULT_OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/full_bpe_audit"

CHANGED_ROWS = 3006
ARMS = ["HS", "LS", "HD", "LD"]


def is_word_start(tok_str: str) -> bool:
    return tok_str.startswith("Ġ") or tok_str.startswith("▁")


def count_word_groups(tok, text: str, max_len: int = 256) -> tuple[int, int, int]:
    """Returns (bpe_pieces, candidate_tokens, word_groups) for text truncated to max_len."""
    enc = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
    enc = enc[:max_len]
    special = set(tok.all_special_ids)
    grp = 0
    candidate = 0
    for i, tid in enumerate(enc):
        if tid in special:
            continue
        candidate += 1
        ts = tok.convert_ids_to_tokens(tid)
        if grp == 0 or is_word_start(str(ts)) or i == 0:
            grp += 1
    return len(enc), candidate, grp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_dir", default=str(POOL_DIR))
    ap.add_argument("--tokenizer", default=str(TOKENIZER_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = Path(args.pool_dir)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(json.dumps({"event": "tokenizer_loaded", "vocab_size": len(tok)}), flush=True)

    arm_stats: dict[str, dict[str, list[int]]] = {arm: {"bpe": [], "cand": [], "wg": []} for arm in ARMS}

    for arm in ARMS:
        pool_path = pool_dir / f"factorial_{arm.lower()}_10M.jsonl"
        with pool_path.open("r", encoding="utf-8") as f:
            for ri, line in enumerate(f):
                if ri >= CHANGED_ROWS:
                    break
                obj = json.loads(line)
                bpe, cand, wg = count_word_groups(tok, obj["text"])
                arm_stats[arm]["bpe"].append(bpe)
                arm_stats[arm]["cand"].append(cand)
                arm_stats[arm]["wg"].append(wg)
        print(json.dumps({"event": "arm_done", "arm": arm, "rows": len(arm_stats[arm]["bpe"])}), flush=True)

    # Compute per-arm totals and deltas
    def total(arm: str, key: str) -> int:
        return sum(arm_stats[arm][key])

    def per_pair_mean(arm: str, key: str) -> float:
        return statistics.mean(arm_stats[arm][key])

    totals = {arm: {k: total(arm, k) for k in ["bpe", "cand", "wg"]} for arm in ARMS}

    # Compute deltas
    def delta(a1: str, a2: str, key: str) -> int:
        return totals[a1][key] - totals[a2][key]

    deltas = {}
    for key in ["bpe", "cand", "wg"]:
        deltas[key] = {
            "HS_minus_LS": delta("HS", "LS", key),
            "HD_minus_LD": delta("HD", "LD", key),
            "HS_minus_HD": delta("HS", "HD", key),
            "LS_minus_LD": delta("LS", "LD", key),
            "interaction": delta("HS", "LS", key) - delta("HD", "LD", key),
        }

    # Estimated WWM targets per pass (p=0.15 of word groups)
    wwm_targets = {arm: {
        "expected_targets_per_pass": round(totals[arm]["wg"] * 0.15),
        "expected_targets_100M": round(totals[arm]["wg"] * 0.15 * 10),
    } for arm in ARMS}

    # Per-row delta distributions for interaction key: bpe
    interaction_bpe_per_row = [
        (arm_stats["HS"]["bpe"][i] - arm_stats["LS"]["bpe"][i]) - (arm_stats["HD"]["bpe"][i] - arm_stats["LD"]["bpe"][i])
        for i in range(CHANGED_ROWS)
    ]
    ib_mean = statistics.mean(interaction_bpe_per_row)
    ib_median = statistics.median(interaction_bpe_per_row)
    ib_stdev = statistics.stdev(interaction_bpe_per_row) if len(interaction_bpe_per_row) > 1 else 0

    elapsed = time.time() - t0
    report = {
        "status": "FULL_BPE_AUDIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "meaning": "Full BPE, candidate token, and word-group audit across all 3,006 changed rows for the HS/LS/HD/LD factorial.",
        "changed_rows": CHANGED_ROWS,
        "totals": totals,
        "deltas": deltas,
        "wwm_targets": wwm_targets,
        "interaction_bpe_per_row": {
            "mean": round(ib_mean, 4),
            "median": ib_median,
            "stdev": round(ib_stdev, 4),
            "min": min(interaction_bpe_per_row),
            "max": max(interaction_bpe_per_row),
        },
        "interpretation": {
            "bpe": f"Compact uses {deltas['bpe']['HS_minus_LS']}/{deltas['bpe']['HD_minus_LD']} more BPE pieces than repeat in same/deranged arms. Interaction {deltas['bpe']['interaction']} is small relative to totals ({totals['HS']['bpe']}).",
            "word_groups": f"Compact generates {deltas['wg']['HS_minus_LS']}/{deltas['wg']['HD_minus_LD']} more word groups. Interaction {deltas['wg']['interaction']} controls WWM target mass balance.",
            "wwm_target_interaction_per_100M": round((deltas['wg']['interaction'] * 0.15 * 10)),
        },
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "full_bpe_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    # Markdown
    md = [
        "# research full BPE/supervised-mass audit (all 3,006 changed rows)\n",
        f"JSON: `{out_dir / 'full_bpe_audit.json'}`\n",
        "## Totals per arm (one pass / changed block only)\n",
        "| arm | BPE pieces | candidate tokens | word groups |",
        "|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        md.append(f"| {arm} | {totals[arm]['bpe']:,} | {totals[arm]['cand']:,} | {totals[arm]['wg']:,} |")
    md.append(f"\n## Deltas (all changed rows)\n")
    md.append("| contrast | BPE delta | word group delta | BPE % | WG % |")
    md.append("|---|---:|---:|---:|---:|")
    for label in ["HS_minus_LS", "HD_minus_LD", "HS_minus_HD", "LS_minus_LD", "interaction"]:
        bpe_d = deltas["bpe"][label]
        wg_d = deltas["wg"][label]
        bpe_pct = round(100 * bpe_d / max(1, totals["HS"]["bpe"]), 3)
        wg_pct = round(100 * wg_d / max(1, totals["HS"]["wg"]), 3)
        md.append(f"| {label} | {bpe_d:+,} | {wg_d:+,} | {bpe_pct:+.3f}% | {wg_pct:+.3f}% |")
    md.append(f"\n## Per-row interaction BPE distribution")
    md.append(f"- mean: {ib_mean:.4f}, median: {ib_median}, stdev: {ib_stdev:.4f}")
    md.append(f"- range: [{min(interaction_bpe_per_row)}, {max(interaction_bpe_per_row)}]")
    md.append(f"\n## Expected WWM target mass over 100M")
    for arm in ARMS:
        md.append(f"- {arm}: ~{wwm_targets[arm]['expected_targets_100M']:,} targets")
    md.append(f"- interaction target difference: ~{round(abs(deltas['wg']['interaction'] * 0.15 * 10))} targets over 100M")
    (out_dir / "full_bpe_audit.md").write_text("\n".join(md) + "\n")

    print(json.dumps({
        "status": report["status"],
        "deltas_bpe": deltas["bpe"],
        "deltas_wg": deltas["wg"],
        "elapsed": report["elapsed_sec"],
    }), flush=True)


if __name__ == "__main__":
    main()
