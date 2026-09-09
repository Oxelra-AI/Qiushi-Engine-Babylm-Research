#!/usr/bin/env python3
"""research: accidental target-token occurrence in compact rewrite source slots.

The T/U/N compact readout treats U and N as source slots that should not supply
the masked rewrite target.  If the target token often appears in U or N, the
source-use interpretation is weakened.  This script rebuilds the same compact
rewrite masked records used by research's neutral anchor and research/018 T/U rows,
then records how often the masked target token occurs in the true source, the
length-matched unrelated compact source, and the length-matched ordinary-text N
source.

This is a data-only audit; it loads no models and performs no benchmark scoring.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

ROOT0 = _public_path('experiments/archive/relation_learning/scripts/accidental_target_context_audit.py')
ROOT = _PUBLIC_ROOT

sys.path.insert(0, str(ROOT / "experiments/archive/relation_learning/scripts"))
import heldout_copy_rewrite_entity_ablation as base  # noqa: E402
import neutral_anchor_rewrite_probe as neutral_base  # noqa: E402

WS = ROOT / "experiments/archive/relation_learning"
REPRESENTATION_FRONTIER_STUDIES = ROOT / "experiments/archive/frontier_consolidation"
TOKENIZER = REPRESENTATION_FRONTIER_STUDIES / "data/compliant_tokenizer"
OUT = WS / "data/accidental_target_context_audit"
NOTE = (_PUBLIC_ROOT / 'research/notes/relation_learning/accidental_target_context_audit.md')
T_U_ROWS = WS / "data/view_split_integration/rewrite_input_rows.csv"
CKS = {"chck_80M", "chck_90M", "chck_100M"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def mean(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return statistics.mean(vals) if vals else float("nan")


def frac(k: int, n: int) -> float:
    return float(k) / float(n) if n else float("nan")


def content_positions(tokenizer, rewrite_text: str, rew_ids: list[int], src_ids: list[int]) -> list[dict[str, Any]]:
    rew_enc = tokenizer(rewrite_text, add_special_tokens=False, return_offsets_mapping=True)
    assert [int(x) for x in rew_enc["input_ids"]] == rew_ids
    src_set = set(int(x) for x in src_ids)
    out: list[dict[str, Any]] = []
    for j, (a, b) in enumerate(rew_enc["offset_mapping"]):
        if b <= a:
            continue
        piece = rewrite_text[a:b]
        if not base.token_content_class(piece):
            continue
        target = int(rew_ids[j])
        out.append({"rewrite_pos": j, "token_class": "overlap" if target in src_set else "nonoverlap", "target_token_id": target, "target_piece": piece})
    return out


def pick_positions(pos_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cls: dict[str, list[dict[str, Any]]] = {"nonoverlap": [], "overlap": []}
    for r in pos_rows:
        by_cls[r["token_class"]].append(r)
    chosen: list[dict[str, Any]] = []
    for cls in ["nonoverlap", "overlap"]:
        vals = by_cls[cls]
        if not vals:
            continue
        if len(vals) <= 2:
            chosen.extend(vals)
        else:
            chosen.extend(vals[round(k * (len(vals) - 1) / 1)] for k in range(2))
    return chosen


def select_unrelated_source(src_ids_all: list[list[int]], i: int, max_len_suffix: int) -> list[int] | None:
    N = len(src_ids_all)
    target_len = len(src_ids_all[i])
    best_j, best_diff = None, float("inf")
    for offset in range(N // 4, N // 4 + N // 2):
        j = (i + offset) % N
        if j == i:
            continue
        diff = abs(len(src_ids_all[j]) - target_len)
        if diff < best_diff:
            best_j, best_diff = j, diff
            if diff == 0:
                break
    if best_j is None:
        return None
    unrel = list(src_ids_all[best_j])
    if len(unrel) > target_len:
        unrel = unrel[:target_len]
    if len(unrel) + max_len_suffix + 2 > 256:
        unrel = unrel[:256 - max_len_suffix - 2]
    return [int(x) for x in unrel]


def build_rows(max_pairs: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(9020022)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    pairs = base.load_unselected_rewrite_pairs(max_pairs)
    ordinary_pool = neutral_base.ordinary_source_pool(tok, rng)
    vocab_ids = [i for i in range(len(tok)) if i not in set(int(x) for x in tok.all_special_ids)]
    src_ids_all = [[int(x) for x in tok(" ".join(str(p["source_text"]).split()), add_special_tokens=False)["input_ids"]] for p in pairs]
    rows: list[dict[str, Any]] = []
    stats: defaultdict[str, int] = defaultdict(int)
    for i, p in enumerate(pairs):
        src_text = " ".join(str(p["source_text"]).split())
        rew_text = " ".join(str(p["rewrite_text"]).split())
        src_ids = src_ids_all[i]
        rew_ids = [int(x) for x in tok(rew_text, add_special_tokens=False)["input_ids"]]
        if not src_ids or not rew_ids or len(src_ids) + len(rew_ids) + 2 > 256:
            stats["too_long_or_empty"] += 1
            continue
        unrel_ids = select_unrelated_source(src_ids_all, i, len(rew_ids))
        if not unrel_ids:
            stats["no_unrelated"] += 1
            continue
        positions = pick_positions(content_positions(tok, rew_text, rew_ids, src_ids))
        if not positions:
            stats["no_content_positions"] += 1
            continue
        for pr in positions:
            target = int(pr["target_token_id"])
            neutral_ids = neutral_base.length_match(ordinary_pool, i * 17 + int(pr["rewrite_pos"]), len(src_ids), rng, forbid={target}, vocab_ids=vocab_ids)
            row = {
                "pair_index": i,
                "pair_id": p.get("pair_id"),
                "probe_id": f"rewritecond:{i}:{pr['rewrite_pos']}:{p.get('pair_id')}",
                "rewrite_pos": int(pr["rewrite_pos"]),
                "token_class": pr["token_class"],
                "target_token_id": target,
                "target_piece": pr["target_piece"],
                "source_token_len": len(src_ids),
                "unrelated_source_token_len": len(unrel_ids),
                "neutral_source_token_len": len(neutral_ids),
                "true_source_target_count": sum(1 for x in src_ids if int(x) == target),
                "unrelated_source_target_count": sum(1 for x in unrel_ids if int(x) == target),
                "neutral_source_target_count": sum(1 for x in neutral_ids if int(x) == target),
                "true_source_contains_target": int(target in set(src_ids)),
                "unrelated_source_contains_target": int(target in set(unrel_ids)),
                "neutral_source_contains_target": int(target in set(neutral_ids)),
            }
            rows.append(row)
        stats["pairs_used"] += 1
    stats["pairs_loaded"] = len(pairs)
    stats["records"] = len(rows)
    return rows, dict(stats)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cls in ["nonoverlap", "overlap", "all"]:
        vals = rows if cls == "all" else [r for r in rows if r["token_class"] == cls]
        n = len(vals)
        out.append({
            "token_class": cls,
            "n_records": n,
            "n_pairs": len({r["pair_id"] for r in vals}),
            "true_contains_frac": frac(sum(int(r["true_source_contains_target"]) for r in vals), n),
            "unrelated_contains_frac": frac(sum(int(r["unrelated_source_contains_target"]) for r in vals), n),
            "neutral_contains_frac": frac(sum(int(r["neutral_source_contains_target"]) for r in vals), n),
            "true_count_mean": mean([r["true_source_target_count"] for r in vals]),
            "unrelated_count_mean": mean([r["unrelated_source_target_count"] for r in vals]),
            "neutral_count_mean": mean([r["neutral_source_target_count"] for r in vals]),
        })
    return out


def compare_to_tu(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not T_U_ROWS.exists():
        return {"status": "missing_tu_rows", "path": rel(T_U_ROWS)}
    df = pd.read_csv(T_U_ROWS)
    df = df[(df["checkpoint"].isin(CKS)) & (df["seed"].isin([43022, 43122]))].copy()
    base_ids = set(df["probe_id"].astype(str).unique())
    own_ids = set(str(r["probe_id"]) for r in rows)
    return {
        "status": "compared",
        "tu_unique_probe_ids": len(base_ids),
        "own_unique_probe_ids": len(own_ids),
        "missing_from_own": len(base_ids - own_ids),
        "extra_in_own": len(own_ids - base_ids),
        "tu_rows_path": rel(T_U_ROWS),
    }


def make_note(summary: list[dict[str, Any]], stats: dict[str, Any], coverage: dict[str, Any]) -> None:
    def fmt(x: float) -> str:
        return "nan" if not math.isfinite(float(x)) else f"{100.0 * float(x):.2f}%"
    lines = [
        "# research accidental target-token context audit",
        "",
        "This data-only readout checks whether compact rewrite masked targets are accidentally visible inside the unrelated compact source slot U or the ordinary-text neutral slot N. The same target positions as the compact T/U/N probe are reconstructed without loading any model.",
        "",
        f"Rows reconstructed: {stats.get('records')} from {stats.get('pairs_used')} pairs. Coverage versus T/U file: {coverage}.",
        "",
        "| token class | records | pairs | target in true source | target in unrelated source | target in neutral source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['token_class']} | {r['n_records']} | {r['n_pairs']} | {fmt(r['true_contains_frac'])} | {fmt(r['unrelated_contains_frac'])} | {fmt(r['neutral_contains_frac'])} |")
    lines += [
        "",
        "For token-nonoverlap rows, true-source target visibility should be zero by construction. If U/N target visibility is also near zero, the T/U/N contrasts are not mainly driven by hidden answer tokens in the alternate source slots. This readout does not distinguish diffuse source-content pull from precise copying; that still requires probability concentration over individual source-content tokens.",
        "",
        f"Data outputs: `{rel(OUT)}`.",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pairs", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, stats = build_rows(args.max_pairs)
    summary = summarize_rows(rows)
    coverage = compare_to_tu(rows)
    write_csv(OUT / "target_context_occurrence_rows.csv", rows)
    write_csv(OUT / "target_context_occurrence_summary.csv", summary)
    payload = {"status": "ACCIDENTAL_TARGET_CONTEXT_AUDIT_DONE", "created_utc": now(), "stats": stats, "coverage": coverage, "summary": summary, "note": rel(NOTE), "rows_csv": rel(OUT / "target_context_occurrence_rows.csv")}
    (OUT / "target_context_occurrence_result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(summary, stats, coverage)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
