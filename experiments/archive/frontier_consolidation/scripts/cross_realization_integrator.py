#!/usr/bin/env python3
"""research: integrate split-GPU cross-realization probe outputs.

Reads event-level no-training probe losses from research GPU0/GPU1 outputs and
computes paired arm contrasts with pair-cluster bootstrap intervals. This script
performs no training, official evaluation, upload, AoA, or leaderboard action.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import pathlib
import random
import statistics
from typing import Any

ROOT = pathlib.Path("experiments/archive/frontier_consolidation")
OUT_DEFAULT = ROOT / "data/cross_realization_probe_integrated"
PATHS = {
    "full_compact_100M": ROOT / "data/cross_realization_probe_gpu0/full_compact_100M_event_losses.jsonl",
    "drop_abs_100M": ROOT / "data/cross_realization_probe_gpu0/drop_abs_100M_event_losses.jsonl",
    "drop_copied_word_100M": ROOT / "data/cross_realization_probe_gpu0/drop_copied_word_100M_event_losses.jsonl",
    "repeat_100M": ROOT / "data/cross_realization_probe_gpu1/repeat_100M_event_losses.jsonl",
    "adjbreak_100M": ROOT / "data/cross_realization_probe_gpu1/adjbreak_100M_event_losses.jsonl",
}
CONTRASTS = [
    ("full_minus_drop_abs", "full_compact_100M", "drop_abs_100M"),
    ("full_minus_drop_copied_word", "full_compact_100M", "drop_copied_word_100M"),
    ("drop_abs_minus_drop_copied_word", "drop_abs_100M", "drop_copied_word_100M"),
    ("drop_abs_minus_repeat", "drop_abs_100M", "repeat_100M"),
    ("drop_copied_word_minus_repeat", "drop_copied_word_100M", "repeat_100M"),
    ("full_minus_repeat", "full_compact_100M", "repeat_100M"),
    ("adjbreak_minus_repeat", "adjbreak_100M", "repeat_100M"),
    ("full_minus_adjbreak", "full_compact_100M", "adjbreak_100M"),
]
FIELDS = [
    "compact_context_advantage_nats",
    "source_context_advantage_nats",
    "compact_minus_source_loss_nats",
    "rep_source_compact_advantage",
    "rep_cos_source_compact",
    "rep_cos_source_counterfactual",
]
ARM_ORDER = ["full_compact_100M", "drop_abs_100M", "drop_copied_word_100M", "repeat_100M", "adjbreak_100M"]


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))]


def weighted_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    den = sum(float(r["target_pieces"]) for r in rows)
    if den <= 0:
        return None
    return sum(float(r[field]) * float(r["target_pieces"]) for r in rows) / den


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "n_events": len(rows),
        "n_pairs": len({str(r.get("pair_id")) for r in rows}),
        "target_pieces": int(sum(int(r["target_pieces"]) for r in rows)),
    }
    for f in FIELDS:
        out[f] = weighted_mean(rows, f)
    return out


def stable_seed(*parts: str) -> int:
    s = "||".join(parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:12], 16) % (2**31 - 1)


def contrast_rows(merged_rows: list[dict[str, Any]], arm_a: str, arm_b: str, field: str) -> list[dict[str, Any]]:
    out = []
    for m in merged_rows:
        ra = m["arms"][arm_a]
        rb = m["arms"][arm_b]
        pieces = float(m["target_pieces"])
        out.append({
            "event_uid": m["event_uid"],
            "pair_id": m["pair_id"],
            "eval_set": m["eval_set"],
            "source_name": m.get("source_name"),
            "target_pieces": pieces,
            "diff": (float(ra[field]) - float(rb[field])),
        })
    return out


def summarize_contrast(rows: list[dict[str, Any]], seed: int, n_boot: int) -> dict[str, Any]:
    den = sum(float(r["target_pieces"]) for r in rows)
    mean = sum(float(r["diff"]) * float(r["target_pieces"]) for r in rows) / den if den else None
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_pair[str(r["pair_id"])].append(r)
    clusters = list(by_pair.items())
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        num = 0.0
        bden = 0.0
        for _j in range(len(clusters)):
            _cid, rs = clusters[rng.randrange(len(clusters))]
            for r in rs:
                w = float(r["target_pieces"])
                num += float(r["diff"]) * w
                bden += w
        if bden > 0:
            boots.append(num / bden)
    return {
        "n_events": len(rows),
        "n_pairs": len(clusters),
        "target_pieces": int(den),
        "mean": mean,
        "bootstrap": {
            "n_boot": len(boots),
            "p025": q(boots, 0.025),
            "p50": q(boots, 0.5),
            "p975": q(boots, 0.975),
            "fraction_gt0": (sum(1 for x in boots if x > 0) / len(boots)) if boots else None,
            "fraction_lt0": (sum(1 for x in boots if x < 0) / len(boots)) if boots else None,
        },
    }


def merge_arms(by_arm: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = {arm: {str(r["event_uid"]): r for r in rows} for arm, rows in by_arm.items()}
    event_sets = {arm: set(idx) for arm, idx in index.items()}
    common = set.intersection(*event_sets.values()) if event_sets else set()
    union = set.union(*event_sets.values()) if event_sets else set()
    merged = []
    mismatches = []
    for uid in sorted(common):
        base = index[ARM_ORDER[0]][uid]
        arms = {arm: index[arm][uid] for arm in ARM_ORDER}
        target_pieces = {arm: int(arms[arm]["target_pieces"]) for arm in ARM_ORDER}
        pair_ids = {arm: str(arms[arm].get("pair_id")) for arm in ARM_ORDER}
        eval_sets = {arm: str(arms[arm].get("eval_set")) for arm in ARM_ORDER}
        if len(set(target_pieces.values())) != 1 or len(set(pair_ids.values())) != 1 or len(set(eval_sets.values())) != 1:
            mismatches.append({"event_uid": uid, "target_pieces": target_pieces, "pair_ids": pair_ids, "eval_sets": eval_sets})
            continue
        merged.append({
            "event_uid": uid,
            "pair_id": str(base.get("pair_id")),
            "eval_set": str(base.get("eval_set")),
            "source_name": str(base.get("source_name")),
            "target_word": base.get("target_word"),
            "target_norm": base.get("target_norm"),
            "target_pieces": int(base["target_pieces"]),
            "arms": arms,
        })
    checks = {
        "arms": ARM_ORDER,
        "event_counts": {arm: len(rows) for arm, rows in by_arm.items()},
        "common_event_count": len(common),
        "union_event_count": len(union),
        "missing_by_arm": {arm: len(union - event_sets[arm]) for arm in ARM_ORDER},
        "metadata_mismatch_count": len(mismatches),
        "metadata_mismatch_examples": mismatches[:10],
        "merged_event_count": len(merged),
        "all_event_sets_identical": all(event_sets[arm] == common for arm in ARM_ORDER),
        "all_metadata_matched": len(mismatches) == 0,
    }
    return merged, checks


def group_values(merged: list[dict[str, Any]]) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    by_eval: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for m in merged:
        by_eval[m["eval_set"]].append(m)
        by_source[str(m.get("source_name"))].append(m)
    return {
        "all": [("all", merged)],
        "eval_set": sorted(by_eval.items()),
        "source_name": sorted(by_source.items()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_arm = {arm: load_jsonl(path) for arm, path in PATHS.items()}
    merged, checks = merge_arms(by_arm)
    arm_means: dict[str, Any] = {}
    contrast_summaries: dict[str, Any] = {}
    for gname, groups in group_values(merged).items():
        arm_means[gname] = {}
        contrast_summaries[gname] = {}
        for gkey, mrows in groups:
            arm_means[gname][gkey] = {arm: arm_summary([m["arms"][arm] for m in mrows]) for arm in ARM_ORDER}
            contrast_summaries[gname][gkey] = {}
            for cname, a, b in CONTRASTS:
                contrast_summaries[gname][gkey][cname] = {}
                for f in FIELDS:
                    rows = contrast_rows(mrows, a, b, f)
                    contrast_summaries[gname][gkey][cname][f] = summarize_contrast(
                        rows, seed=stable_seed(gname, gkey, cname, f), n_boot=args.n_boot
                    )

    all_means = arm_means["all"]["all"]
    all_contrasts = contrast_summaries["all"]["all"]
    compact_adv_by_arm = {arm: all_means[arm]["compact_context_advantage_nats"] for arm in ARM_ORDER}
    rep_adv_by_arm = {arm: all_means[arm]["rep_source_compact_advantage"] for arm in ARM_ORDER}
    source_adv_by_arm = {arm: all_means[arm]["source_context_advantage_nats"] for arm in ARM_ORDER}
    def ci_field(cname: str, field: str) -> dict[str, Any]:
        return all_contrasts[cname][field]
    qualitative = {
        "compact_advantage_by_arm": compact_adv_by_arm,
        "source_advantage_by_arm": source_adv_by_arm,
        "representation_advantage_by_arm": rep_adv_by_arm,
        "drop_abs_minus_repeat_compact_advantage": ci_field("drop_abs_minus_repeat", "compact_context_advantage_nats"),
        "drop_copied_word_minus_repeat_compact_advantage": ci_field("drop_copied_word_minus_repeat", "compact_context_advantage_nats"),
        "full_minus_drop_abs_compact_advantage": ci_field("full_minus_drop_abs", "compact_context_advantage_nats"),
        "full_minus_drop_copied_word_compact_advantage": ci_field("full_minus_drop_copied_word", "compact_context_advantage_nats"),
        "drop_abs_minus_repeat_representation": ci_field("drop_abs_minus_repeat", "rep_source_compact_advantage"),
        "full_minus_drop_abs_representation": ci_field("full_minus_drop_abs", "rep_source_compact_advantage"),
        "interpretive_summary": (
            "The probe supports same-realization context enrichment if deletion arms retaining the compact input distribution remain above repeat "
            "while full_compact adds little beyond them. It supports full same-sequence target complementarity only if full_compact is uniquely above "
            "both deletion arms on compact-context and representation interactions. This inference is local-mechanistic and does not replace selected official-compatible endpoint scores."
        ),
    }

    # Compact row for quick plotting/inspection.
    contrast_csv = out_dir / "all_event_arm_contrasts.csv"
    with contrast_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["contrast", "field", "mean", "p025", "p50", "p975", "fraction_gt0", "n_events", "n_pairs", "target_pieces"])
        for cname, field_map in all_contrasts.items():
            for field, s in field_map.items():
                b = s["bootstrap"]
                wr.writerow([cname, field, s["mean"], b["p025"], b["p50"], b["p975"], b["fraction_gt0"], s["n_events"], s["n_pairs"], s["target_pieces"]])

    out = {
        "status": "CROSS_REALIZATION_PROBE_INTEGRATED",
        "meaning": "File-only integration of split-GPU saved-checkpoint probe outputs; paired arm contrasts are bootstrapped by compact pair/document cluster.",
        "input_paths": {arm: str(path) for arm, path in PATHS.items()},
        "input_sha256": {arm: sha256_file(path) for arm, path in PATHS.items()},
        "checks": checks,
        "arm_means": arm_means,
        "contrast_summaries": contrast_summaries,
        "qualitative_signature_fields": qualitative,
        "contrast_csv": str(contrast_csv),
        "n_boot": args.n_boot,
        "no_training_official_eval_upload_aoa_or_leaderboard": True,
    }
    out_json = out_dir / "cross_realization_probe_integrated.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    def fmt_ci(s: dict[str, Any]) -> str:
        b = s["bootstrap"]
        return f"mean={s['mean']:.6f}, boot95=[{b['p025']:.6f},{b['p975']:.6f}], P>0={b['fraction_gt0']:.3f}"
    md = [
        "# research cross-realization probe integration",
        "",
        f"Merged events: {checks['merged_event_count']} / common {checks['common_event_count']} / union {checks['union_event_count']}",
        f"Event sets identical: {checks['all_event_sets_identical']}; metadata matched: {checks['all_metadata_matched']}",
        "",
        "## All-event arm means",
    ]
    for arm in ARM_ORDER:
        s = all_means[arm]
        md.append(
            f"- {arm}: compact_adv={s['compact_context_advantage_nats']:.6f}, "
            f"source_adv={s['source_context_advantage_nats']:.6f}, "
            f"compact_minus_source={s['compact_minus_source_loss_nats']:.6f}, "
            f"rep_A={s['rep_source_compact_advantage']:.6f}, n={s['n_events']}, pieces={s['target_pieces']}"
        )
    md.extend(["", "## Decision-bearing all-event contrasts"])
    for cname in ["drop_abs_minus_repeat", "drop_copied_word_minus_repeat", "full_minus_drop_abs", "full_minus_drop_copied_word", "full_minus_repeat", "adjbreak_minus_repeat"]:
        md.append(f"- {cname} compact_adv: {fmt_ci(all_contrasts[cname]['compact_context_advantage_nats'])}")
        md.append(f"  - {cname} rep_A: {fmt_ci(all_contrasts[cname]['rep_source_compact_advantage'])}")
    md.extend(["", "## Eval-set compact-advantage contrasts"])
    for gkey in sorted(contrast_summaries["eval_set"]):
        c = contrast_summaries["eval_set"][gkey]
        md.append(f"- {gkey}: drop_abs-repeat {fmt_ci(c['drop_abs_minus_repeat']['compact_context_advantage_nats'])}; full-drop_abs {fmt_ci(c['full_minus_drop_abs']['compact_context_advantage_nats'])}; full-drop_copied {fmt_ci(c['full_minus_drop_copied_word']['compact_context_advantage_nats'])}")
    md.extend([
        "",
        "## Scientific reading",
        "This is local saved-checkpoint inference, not endpoint evidence. A full-compact-unique same-sequence target-complementarity reading would require full_compact to exceed both deletion arms on the cross-realization compact-context and representation interactions. If drop_abs/drop_copied remain above repeat and full adds little, the result instead supports compact input/contextual enrichment around shared/copied content as the local object, while downstream selected scores still decide whether that object matters for BabyLM competence.",
        "",
        f"JSON: `{out_json}`",
        f"CSV: `{contrast_csv}`",
        "",
    ])
    out_md = out_dir / "cross_realization_probe_integrated.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": out["status"], "out_json": str(out_json), "out_md": str(out_md), "merged_events": checks["merged_event_count"], "no_training_official_eval_upload_aoa_or_leaderboard": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
