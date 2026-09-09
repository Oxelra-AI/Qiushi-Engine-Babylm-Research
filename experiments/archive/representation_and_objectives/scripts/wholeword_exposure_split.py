#!/usr/bin/env python3
"""research exposure split for whole-word copied-content control.

Splits fixed-event losses by whether the probed compact-side word was selected by the
identity-attached WWM schedule in epoch 0/1.  The central contrast is
`drop_abs_minus_drop_copied_word` on source_absent_content events.  Persistence on
selected0 events means the source-absent target channel transfers within the compact-pair
population beyond the exact words selected during the two training epochs.
"""
from __future__ import annotations

import collections
import hashlib
import json
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/representation_and_objectives")
EVENT_LOSSES = ROOT / "data/wholeword_control_readout/event_losses.jsonl"
OUT_DIR = ROOT / "data/wholeword_exposure_split"
MASK_SEED = 430230221
MASK_PROB = 0.15
CHECKPOINTS = ["chck_10M", "chck_20M"]
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]
CONTRASTS = [
    ("drop_abs_minus_drop_copied_word", "drop_abs", "drop_copied_word"),
    ("drop_abs_minus_drop_copied_tok", "drop_abs", "drop_copied_tok"),
    ("drop_abs_minus_full", "drop_abs", "full"),
    ("drop_copied_word_minus_full", "drop_copied_word", "full"),
    ("drop_copied_tok_minus_full", "drop_copied_tok", "full"),
]


def stable_u64(s: str) -> int:
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
        if lhs not in r[ck] or rhs not in r[ck]:
            continue
        num += r[ck][lhs]["loss_sum"] - r[ck][rhs]["loss_sum"]
        den += r[ck][lhs]["pieces"]
    return num / den if den else float("nan")


def event_delta(records: list[dict[str, Any]], ck: str, lhs: str, rhs: str) -> float:
    vals = []
    for r in records:
        if lhs in r[ck] and rhs in r[ck]:
            vals.append(r[ck][lhs]["loss_mean"] - r[ck][rhs]["loss_mean"])
    return statistics.mean(vals) if vals else float("nan")


def cluster_boot(records: list[dict[str, Any]], ck: str, lhs: str, rhs: str, n_boot: int, seed: int) -> dict[str, Any]:
    records = [r for r in records if lhs in r[ck] and rhs in r[ck]]
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
            eps = selected_epochs(str(r["pair_id"]), int(r["word_index"]))
            r["selected_epochs"] = eps
            r["selected_count_2epochs"] = len(eps)
            r["selected_any"] = bool(eps)
            records.append(r)
    enriched_path = OUT_DIR / "event_losses_with_training_exposure.jsonl"
    enriched_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    summary: dict[str, Any] = {
        "status": "WHOLEWORD_EXPOSURE_SPLIT",
        "meaning": "Exposure split for whole-word copied-content comparator. Positive drop_abs_minus_drop_copied_word means absent-label deletion leaves worse fixed-event denoising than whole matched copied-content deletion.",
        "inputs": {"event_losses": str(EVENT_LOSSES), "mask_seed": MASK_SEED, "mask_prob": MASK_PROB, "epochs": [0, 1]},
        "event_losses_with_training_exposure": str(enriched_path),
        "counts": {},
        "comparisons": {},
    }
    for cat in CATEGORIES:
        cr = [r for r in records if r["category"] == cat]
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
                recs = [r for r in records if r["category"] == cat and pred(r)]
                summary["comparisons"][ck][cat][sel_key] = {}
                for ci, (name, lhs, rhs) in enumerate(CONTRASTS):
                    summary["comparisons"][ck][cat][sel_key][name] = cluster_boot(
                        recs,
                        ck,
                        lhs,
                        rhs,
                        500,
                        seed=226000 + ci * 1009 + len(sel_key) + 31 * CATEGORIES.index(cat) + 101 * CHECKPOINTS.index(ck),
                    )
    out = OUT_DIR / "wholeword_event_exposure_split.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    main = summary["comparisons"]["chck_20M"]["source_absent_content"]
    print(json.dumps({
        "status": summary["status"],
        "out": str(out),
        "counts_source_absent": summary["counts"]["source_absent_content"],
        "chck20_source_absent_drop_abs_minus_drop_copied_word": {k: v["drop_abs_minus_drop_copied_word"] for k, v in main.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
