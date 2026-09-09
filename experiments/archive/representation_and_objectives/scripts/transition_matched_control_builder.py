#!/usr/bin/env python3
"""research: build matched neutral controls for the strict transition substrate.

This is a CPU-only substrate builder, not a training launch.  It reads the research
strict transition candidates and the same allowed reservoirs, then creates paired
research slices (50k/100k/200k target words) consisting of:
  - transition treatment sentences selected for broad relation/consequence content;
  - neutral control sentences from the same reservoirs with similar length/source
    distribution but low transition score.

No official evaluation item text is read.  The output is a future low-cost probe
asset only; any actual training arm must still be materialized into a full strict
10M-word corpus and tested from shared checkpoints before a 100M commitment.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
MINER_SCRIPT = A01_WS / "scripts/strict_transition_substrate_miner.py"
CANDIDATES = A01_WS / "data/strict_transition_substrate/strict_transition_candidates.jsonl"
BALANCED_200K = A01_WS / "data/strict_transition_substrate/strict_transition_balanced_200k_slice.jsonl"
OUT = A01_WS / "data/transition_matched_controls"
NOTE = A01_WS / "notes/108_transition_matched_controls.md"
BUDGETS = [50_000, 100_000, 200_000]


def load_miner():
    spec = importlib.util.spec_from_file_location("strict_transition_substrate_miner", MINER_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {MINER_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_sort_key(r: dict[str, Any]) -> tuple:
    # High score, high purity, and moderate lengths first; stable by text hash.
    return (not r.get("very_high_purity", False), -float(r.get("score", 0)), abs(int(r.get("words", 0)) - 32), str(r.get("text_sha256", "")))


def select_approx_exact(pool: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Greedy select, then one-row replacement repair for exact or near-exact words."""
    selected = []
    used = set()
    total = 0
    doc_counts = Counter()
    for r in sorted(pool, key=stable_sort_key):
        w = int(r["words"])
        key = r["text_sha256"]
        doc = str(r.get("norm_hash") or r.get("doc_id") or f"{r.get('source_label')}:{r.get('row_index')}")
        if key in used or doc_counts[doc] >= 3:
            continue
        if total + w <= budget:
            selected.append(r); used.add(key); doc_counts[doc] += 1; total += w
        if total == budget:
            return selected
    if total < budget:
        rem = budget - total
        for r in sorted(pool, key=lambda x: (abs(int(x["words"]) - rem), stable_sort_key(x))):
            if r["text_sha256"] not in used and int(r["words"]) == rem:
                selected.append(r); return selected
    # Try replacing one selected row by up to two unused rows to reach exact.
    used = {r["text_sha256"] for r in selected}
    unused_by_word = defaultdict(list)
    for r in pool:
        if r["text_sha256"] not in used:
            unused_by_word[int(r["words"])].append(r)
    total = sum(int(r["words"]) for r in selected)
    for i, old in enumerate(list(selected)):
        need = budget - (total - int(old["words"]))
        if need in unused_by_word:
            repl = unused_by_word[need][0]
            selected[i] = repl
            return selected
        for w1, rows1 in unused_by_word.items():
            w2 = need - w1
            if w2 in unused_by_word:
                r1 = rows1[0]
                r2 = next((x for x in unused_by_word[w2] if x["text_sha256"] != r1["text_sha256"]), None)
                if r2 is not None:
                    selected.pop(i)
                    selected.extend([r1, r2])
                    return selected
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sentences": 0, "words": 0}
    route_counts = Counter(); route_words = Counter(); source_counts = Counter(); source_words = Counter(); channels = Counter(); noise = Counter()
    for r in rows:
        source_counts[r.get("source_label", "")] += 1
        source_words[r.get("source_label", "")] += int(r.get("words", 0))
        for route in r.get("routes", []):
            route_counts[route] += 1; route_words[route] += int(r.get("words", 0))
        for ch in r.get("content_channels", []):
            channels[ch] += 1
        for n in r.get("noise", []):
            noise[n] += 1
    return {
        "sentences": len(rows),
        "words": sum(int(r.get("words", 0)) for r in rows),
        "mean_words": statistics.fmean(int(r.get("words", 0)) for r in rows),
        "median_words": statistics.median(int(r.get("words", 0)) for r in rows),
        "mean_score": statistics.fmean(float(r.get("score", 0)) for r in rows),
        "very_high_purity": sum(bool(r.get("very_high_purity")) for r in rows),
        "route_counts": dict(route_counts),
        "route_words": dict(route_words),
        "source_counts": dict(source_counts),
        "source_words": dict(source_words),
        "channel_counts": dict(channels),
        "noise_counts": dict(noise),
    }


def collect_neutral_pool(miner) -> list[dict[str, Any]]:
    treatment_keys = {r["text_sha256"] for r in read_jsonl(CANDIDATES)}
    pool = []
    for label, spec in miner.RESERVOIRS.items():
        for rec in miner.iter_records(label, spec):
            feat = miner.features(rec["text"])
            key = miner.stable_key(rec["text"])
            if key in treatment_keys:
                continue
            # Low relation/transition score and no high-purity routes.  We permit broad content words so length/source matching is not just empty filler.
            if feat["high_purity"] or feat["score"] > 10:
                continue
            out = dict(rec)
            out.update(feat)
            out["text_sha256"] = key
            pool.append(out)
    # Deduplicate neutral rows.
    best = {}
    for r in pool:
        best.setdefault(r["text_sha256"], r)
    return list(best.values())


def match_neutral(treatment: list[dict[str, Any]], neutral_pool: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    by_source = defaultdict(list)
    for r in neutral_pool:
        by_source[r.get("source_label", "")].append(r)
    for src in by_source:
        by_source[src].sort(key=lambda r: (abs(int(r["words"]) - 30), float(r.get("score", 0)), str(r.get("text_sha256", ""))))
    used = set(); selected = []; total = 0
    # Match each treatment row to same source and nearby length where possible.
    for tr in treatment:
        src = tr.get("source_label", "")
        tw = int(tr["words"])
        candidates = [r for r in by_source.get(src, []) if r["text_sha256"] not in used and abs(int(r["words"]) - tw) <= 8]
        if not candidates:
            candidates = [r for r in by_source.get(src, []) if r["text_sha256"] not in used]
        if not candidates:
            candidates = [r for r in neutral_pool if r["text_sha256"] not in used]
        if not candidates:
            break
        r = min(candidates, key=lambda x: (abs(int(x["words"]) - tw), float(x.get("score", 0)), str(x.get("text_sha256", ""))))
        if total + int(r["words"]) <= budget:
            selected.append(r); used.add(r["text_sha256"]); total += int(r["words"])
        if total >= budget:
            break
    # Fill residual from neutral rows, preferring same source distribution and low score.
    if total < budget:
        leftovers = [r for r in neutral_pool if r["text_sha256"] not in used]
        leftovers.sort(key=lambda r: (float(r.get("score", 0)), abs(int(r["words"]) - 30), str(r.get("text_sha256", ""))))
        for r in leftovers:
            if total + int(r["words"]) <= budget:
                selected.append(r); used.add(r["text_sha256"]); total += int(r["words"])
            if total == budget:
                break
    # One-row/two-row repair to exact if possible.
    if total != budget:
        repaired = select_approx_exact(selected + [r for r in neutral_pool if r["text_sha256"] not in used], budget)
        if abs(sum(int(r["words"]) for r in repaired) - budget) <= abs(total - budget):
            selected = repaired
    return selected


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    miner = load_miner()
    candidates = read_jsonl(CANDIDATES)
    balanced200 = read_jsonl(BALANCED_200K)
    neutral_pool = collect_neutral_pool(miner)
    summary: dict[str, Any] = {
        "status": "TRANSITION_MATCHED_CONTROLS_BUILT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary": "Allowed-reservoir corpus slices only; no official evaluation item text used; not a training launch.",
        "candidate_source": str(CANDIDATES),
        "balanced_200k_source": str(BALANCED_200K),
        "neutral_pool_sentences": len(neutral_pool),
        "neutral_pool_words": sum(int(r["words"]) for r in neutral_pool),
        "budgets": {},
    }
    for budget in BUDGETS:
        if budget == 200_000:
            treatment = balanced200
        else:
            treatment = select_approx_exact(balanced200, budget)
        treatment_words = sum(int(r["words"]) for r in treatment)
        neutral = match_neutral(treatment, neutral_pool, treatment_words)
        neutral_words = sum(int(r["words"]) for r in neutral)
        tpath = OUT / f"transition_treatment_{budget//1000}k.jsonl"
        npath = OUT / f"neutral_control_{budget//1000}k.jsonl"
        write_jsonl(tpath, treatment)
        write_jsonl(npath, neutral)
        summ = {
            "requested_budget": budget,
            "treatment_path": str(tpath),
            "neutral_path": str(npath),
            "treatment": summarize(treatment),
            "neutral": summarize(neutral),
            "word_difference_treatment_minus_neutral": treatment_words - neutral_words,
            "scientific_status": "substrate for future shared-checkpoint low-cost factorial only",
        }
        summary["budgets"][f"{budget//1000}k"] = summ
    # Write a compact CSV for quick checking.
    csv_rows = []
    for b, rec in summary["budgets"].items():
        for arm in ["treatment", "neutral"]:
            s = rec[arm]
            csv_rows.append({
                "budget": b,
                "arm": arm,
                "sentences": s["sentences"],
                "words": s["words"],
                "mean_words": s.get("mean_words"),
                "mean_score": s.get("mean_score"),
                "very_high_purity": s.get("very_high_purity"),
                "source_counts": json.dumps(s.get("source_counts", {}), ensure_ascii=False),
                "route_counts": json.dumps(s.get("route_counts", {}), ensure_ascii=False),
            })
    csv_path = OUT / "transition_matched_control_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    summary["summary_csv"] = str(csv_path)
    summary["elapsed_sec"] = round(time.time() - t0, 2)
    out_json = OUT / "transition_matched_control_builder.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# research — Transition substrate matched controls\n\n",
        "Built matched neutral-control slices from the same allowed reservoirs as the strict transition substrate. No official evaluation text is used and no training is launched.\n\n",
        f"Neutral pool: `{summary['neutral_pool_sentences']}` sentences / `{summary['neutral_pool_words']}` words.\n\n",
        "| budget | treatment words | neutral words | treatment mean score | neutral mean score | treatment sentences | neutral sentences |\n",
        "|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for b, rec in summary["budgets"].items():
        ts = rec["treatment"]; ns = rec["neutral"]
        lines.append(f"| {b} | {ts['words']} | {ns['words']} | {ts.get('mean_score'):.3f} | {ns.get('mean_score'):.3f} | {ts['sentences']} | {ns['sentences']} |\n")
    lines += [
        "\nScientific use: only after the FW compact/breadth endpoint and relational readouts are known, these slices can support a small shared-checkpoint factorial or objective probe. The treatment still needs semantic filtering before a full 10M materialization; many regex-selected rows are useful but not all are pure transition facts.\n\n",
        f"Summary JSON: `{out_json}`\n",
        f"Summary CSV: `{csv_path}`\n",
    ]
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "neutral_pool_sentences": summary["neutral_pool_sentences"],
        "neutral_pool_words": summary["neutral_pool_words"],
        "budgets": {k: {"treatment_words": v["treatment"]["words"], "neutral_words": v["neutral"]["words"], "diff": v["word_difference_treatment_minus_neutral"]} for k, v in summary["budgets"].items()},
        "json": str(out_json),
        "csv": str(csv_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
