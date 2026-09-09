#!/usr/bin/env python3
"""Split research target-selective event losses by whether the probed word was actually trained.

The research fixed probe samples compact-side words from the training pair population.  For
interpretation we need to know whether the deterministic research/225 WWM schedule selected
the same word in epoch 0/1.  The drop_abs arm removes labels only for selected rw_abs_content
words, so an effect on selected words is expected; an effect on unselected words is stronger
evidence for category-level generalization.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
EVENT_LOSSES = ROOT / "data/target_selective_event_bootstrap/event_losses.jsonl"
OUT_DIR = ROOT / "data/event_exposure_split"
MASK_SEED = 430230221
MASK_PROB = 0.15
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]


def stable_u64(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def stable_uniform01(s: str) -> float:
    return ((stable_u64(s) >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def selected_epochs(pair_id: str, word_index: int) -> list[int]:
    eps = []
    for ep in (0, 1):
        key = f"s={MASK_SEED}|ep={ep}|seg=rewrite|id={pair_id}|word={word_index}|select"
        if stable_uniform01(key) < MASK_PROB:
            eps.append(ep)
    return eps


def qstats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    n = len(s)
    def q(p: float):
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"n": n, "mean": round(statistics.mean(s), 6), "median": round(statistics.median(s), 6), "p025": round(q(0.025), 6), "p975": round(q(0.975), 6), "min": round(s[0], 6), "max": round(s[-1], 6)}


def weighted_delta(records: list[dict[str, Any]], ck: str, lhs: str, rhs: str) -> float:
    num = 0.0
    den = 0
    for r in records:
        num += r[ck][lhs]["loss_sum"] - r[ck][rhs]["loss_sum"]
        den += r[ck][lhs]["pieces"]
    return num / den if den else float("nan")


def event_delta(records: list[dict[str, Any]], ck: str, lhs: str, rhs: str) -> float:
    return statistics.mean(r[ck][lhs]["loss_mean"] - r[ck][rhs]["loss_mean"] for r in records) if records else float("nan")


def cluster_boot(records: list[dict[str, Any]], ck: str, lhs: str, rhs: str, n_boot: int, seed: int) -> dict[str, Any]:
    if not records:
        return {"n_events": 0}
    by_pair = collections.defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    pairs = sorted(by_pair)
    rng = random.Random(seed)
    evs = []
    wts = []
    for _ in range(n_boot):
        sample = []
        for _j in range(len(pairs)):
            sample.extend(by_pair[rng.choice(pairs)])
        evs.append(event_delta(sample, ck, lhs, rhs))
        wts.append(weighted_delta(sample, ck, lhs, rhs))
    return {
        "n_events": len(records),
        "n_pair_clusters": len(pairs),
        "event_mean_delta": round(event_delta(records, ck, lhs, rhs), 6),
        "piece_weighted_delta": round(weighted_delta(records, ck, lhs, rhs), 6),
        "event_bootstrap": qstats(evs),
        "piece_bootstrap": qstats(wts),
        "p_piece_gt0": round(sum(v > 0 for v in wts) / len(wts), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    with EVENT_LOSSES.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # event_losses currently omits word_index; recover it from original probe event order.
            records.append(r)
    # Load probe events for word_index/source text metadata.
    probe_events = []
    with (ROOT / "data/existing_trajectory_denoising_probe/probe_events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probe_events.append(json.loads(line))
    if len(probe_events) != len(records):
        raise RuntimeError(f"probe/event loss length mismatch: {len(probe_events)} vs {len(records)}")
    enriched = []
    for r, e in zip(records, probe_events):
        eps = selected_epochs(e["pair_id"], int(e["word_index"]))
        rr = dict(r)
        rr["word_index"] = int(e["word_index"])
        rr["selected_epochs"] = eps
        rr["selected_count_2epochs"] = len(eps)
        rr["selected_any"] = bool(eps)
        enriched.append(rr)
    (OUT_DIR / "event_losses_with_training_exposure.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in enriched), encoding="utf-8")

    summary: dict[str, Any] = {
        "status": "EVENT_EXPOSURE_SPLIT",
        "meaning": "Split fixed-event target-selective losses by whether the probed compact-side word was selected by deterministic WWM in epoch 0/1. Positive drop_abs_minus_drop_copied means the absent-label-drop arm is worse.",
        "inputs": {"event_losses": str(EVENT_LOSSES), "mask_seed": MASK_SEED, "mask_prob": MASK_PROB, "epochs": [0, 1]},
        "counts": {},
        "comparisons": {},
    }
    # counts by category and selection state
    for cat in CATEGORIES:
        cr = [r for r in enriched if r["category"] == cat]
        summary["counts"][cat] = dict(collections.Counter(str(r["selected_count_2epochs"]) for r in cr))
    for ck in CHECKPOINTS:
        summary["comparisons"][ck] = {}
        for cat in CATEGORIES:
            summary["comparisons"][ck][cat] = {}
            for sel_key, pred in [
                ("selected0", lambda r: r["selected_count_2epochs"] == 0),
                ("selected1", lambda r: r["selected_count_2epochs"] == 1),
                ("selected2", lambda r: r["selected_count_2epochs"] == 2),
                ("selected_any", lambda r: r["selected_count_2epochs"] > 0),
                ("all", lambda r: True),
            ]:
                recs = [r for r in enriched if r["category"] == cat and pred(r)]
                summary["comparisons"][ck][cat][sel_key] = {
                    "drop_abs_minus_drop_copied": cluster_boot(recs, ck, "drop_abs", "drop_copied", 500, seed=225600 + len(sel_key) + 11*CATEGORIES.index(cat) + 100*CHECKPOINTS.index(ck)),
                    "drop_abs_minus_full": cluster_boot(recs, ck, "drop_abs", "full", 500, seed=225700 + len(sel_key) + 11*CATEGORIES.index(cat) + 100*CHECKPOINTS.index(ck)),
                    "drop_copied_minus_full": cluster_boot(recs, ck, "drop_copied", "full", 500, seed=225800 + len(sel_key) + 11*CATEGORIES.index(cat) + 100*CHECKPOINTS.index(ck)),
                }
    out = OUT_DIR / "event_exposure_split.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    main = summary["comparisons"]["chck_20M"]["source_absent_content"]
    print(json.dumps({"status": summary["status"], "out": str(out), "counts_source_absent": summary["counts"]["source_absent_content"], "chck20_source_absent_main": {k: v["drop_abs_minus_drop_copied"] for k, v in main.items()}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
