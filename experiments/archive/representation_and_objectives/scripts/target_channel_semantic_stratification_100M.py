#!/usr/bin/env python3
"""research: semantic stratification of the 100M packed target-selective local readout.

Reads the research/235 local fixed-event losses for the historical packed-geometry
100M endpoints and asks whether the source-absent loss effect is concentrated in
the same novel relational/event content identified at 20M, or whether the mature
packed intervention is broader.

Main contrast: drop_abs_100M - drop_copied_word_100M.  Positive means the model
that did NOT train on source-absent compact-content labels has higher local loss,
so those labels were locally useful for that event subset.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Any

SCRIPT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/scripts")
if str(SCRIPT_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.resolve()))

from source_absent_lexical_profile import flags as lexical_flags  # noqa: E402
from annotate_packed_pool import norm_word, word_class  # noqa: E402

DEFAULT_EVENTS = pathlib.Path("experiments/archive/representation_and_objectives/data/packed_targetselect_denoising_probe_100M/packed_targetselect_denoising_event_losses_100M.jsonl")
DEFAULT_OUT = pathlib.Path("experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification_100M")
CONTRASTS = {
    "drop_abs_minus_drop_copied_word": ("drop_abs_100M", "drop_copied_word_100M"),
    "drop_abs_minus_full": ("drop_abs_100M", "full_compact_100M"),
    "drop_copied_word_minus_full": ("drop_copied_word_100M", "full_compact_100M"),
}
FOCUS_GROUPS = [
    "all",
    "relational_or_event_state",
    "not_relational_or_event_state",
    "ordinary_nonrel_nonentity",
    "capitalized_or_number",
    "relational_or_abstracting_word",
    "event_or_state_word",
    "generic_entity_word",
    "capitalized",
    "number_or_year",
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def event_groups(word: str, category: str) -> set[str]:
    nw = norm_word(word)
    fs = set(lexical_flags(word, nw))
    groups = {"all"}
    groups.update(fs)
    rel_event = bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)
    capnum = bool({"capitalized", "number_or_year"} & fs)
    groups.add("relational_or_event_state" if rel_event else "not_relational_or_event_state")
    groups.add("capitalized_or_number" if capnum else "not_capitalized_or_number")
    if not ({"relational_or_abstracting_word", "event_or_state_word", "capitalized", "number_or_year", "generic_entity_word"} & fs):
        groups.add("ordinary_nonrel_nonentity")
    groups.add(f"class_{word_class(word)}")
    groups.add(f"category_{category}")
    return groups


def load_events(path: pathlib.Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            arms = o.get("arms", {})
            if not all(a in arms for pair in CONTRASTS.values() for a in pair):
                continue
            word = str(o.get("word", ""))
            cat = str(o.get("category", "unknown"))
            rec = {
                "event_uid": str(o.get("event_uid")),
                "eval_set": str(o.get("eval_set", "unknown")),
                "pair_id": str(o.get("pair_id", o.get("event_uid"))),
                "category": cat,
                "word": word,
                "norm": norm_word(word),
                "word_class": word_class(word),
                "groups": sorted(event_groups(word, cat)),
                "pieces": int(o.get("n_pieces_realized", o.get("n_pieces", 0))),
                "arms": arms,
            }
            out.append(rec)
    return out


def summarize_delta(events: list[dict[str, Any]], contrast: str, n_boot: int, seed: int) -> dict[str, Any]:
    a, b = CONTRASTS[contrast]
    by_cluster: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    type_counter = collections.Counter()
    for e in events:
        aa = e["arms"].get(a)
        bb = e["arms"].get(b)
        if not aa or not bb:
            continue
        pa = int(aa.get("pieces", e["pieces"]))
        pb = int(bb.get("pieces", e["pieces"]))
        if pa <= 0 or pb <= 0 or pa != pb:
            continue
        delta = float(aa["loss_sum"]) - float(bb["loss_sum"])
        by_cluster[e["pair_id"]].append((delta, pa))
        type_counter[e["norm"]] += 1
    pairs = list(by_cluster.items())
    total_pieces = sum(p for _, evs in pairs for _, p in evs)
    total_delta = sum(d for _, evs in pairs for d, _ in evs)
    if total_pieces <= 0:
        return {"n_events": 0}
    event_means = [d / p for _, evs in pairs for d, p in evs]
    rng = random.Random(seed)
    boot = []
    if len(pairs) >= 2 and n_boot > 0:
        for _ in range(n_boot):
            bd = 0.0
            bp = 0
            for _ in pairs:
                _, evs = pairs[rng.randrange(len(pairs))]
                for d, p in evs:
                    bd += d
                    bp += p
            if bp:
                boot.append(bd / bp)
    return {
        "n_events": sum(len(evs) for _, evs in pairs),
        "n_pairs": len(pairs),
        "pieces": total_pieces,
        "piece_weighted_delta": round(total_delta / total_pieces, 6),
        "event_mean_unweighted": round(statistics.mean(event_means), 6),
        "event_median_unweighted": round(statistics.median(event_means), 6),
        "bootstrap": {
            "n_boot": len(boot),
            "median": None if not boot else round(statistics.median(boot), 6),
            "p025": None if not boot else round(q(boot, 0.025), 6),
            "p05": None if not boot else round(q(boot, 0.05), 6),
            "p95": None if not boot else round(q(boot, 0.95), 6),
            "p975": None if not boot else round(q(boot, 0.975), 6),
            "fraction_gt0": None if not boot else round(sum(1 for x in boot if x > 0) / len(boot), 6),
        },
        "top_norms": [{"norm": w, "count": int(c)} for w, c in type_counter.most_common(20)],
    }


def summarize_cell(events: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    return {c: summarize_delta(events, c, n_boot, seed + i * 100003) for i, c in enumerate(CONTRASTS)}


def build_tables(events: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"by_eval_set_category_group": {}, "focus": {}, "novelty_x_rel_event": {}}
    eval_sets = sorted(set(e["eval_set"] for e in events))
    categories = sorted(set(e["category"] for e in events))
    for es in eval_sets:
        out["by_eval_set_category_group"].setdefault(es, {})
        for cat in categories:
            cat_events = [e for e in events if e["eval_set"] == es and e["category"] == cat]
            if not cat_events:
                continue
            out["by_eval_set_category_group"][es].setdefault(cat, {})
            for grp in FOCUS_GROUPS:
                subset = [e for e in cat_events if grp in e["groups"]]
                if subset:
                    out["by_eval_set_category_group"][es][cat][grp] = summarize_cell(subset, n_boot, seed + hash((es, cat, grp)) % 1_000_000)
    # Compact focus table for source_absent_content in main contrast.
    for es in eval_sets:
        out["focus"].setdefault(es, {})
        for cat in ["source_absent_content", "retained_content", "function_other"]:
            for grp in ["all", "relational_or_event_state", "not_relational_or_event_state", "ordinary_nonrel_nonentity", "capitalized_or_number"]:
                rec = out["by_eval_set_category_group"].get(es, {}).get(cat, {}).get(grp, {}).get("drop_abs_minus_drop_copied_word")
                if rec:
                    out["focus"][es][f"{cat}|{grp}"] = rec
    # Difference-in-differences: (SA_rel-SA_other)-(retained_rel-retained_other), main contrast.
    for es in eval_sets:
        def val(cat: str, grp: str) -> float | None:
            rec = out["by_eval_set_category_group"].get(es, {}).get(cat, {}).get(grp, {}).get("drop_abs_minus_drop_copied_word")
            return None if not rec or rec.get("n_events", 0) == 0 else float(rec["piece_weighted_delta"])
        sa_rel = val("source_absent_content", "relational_or_event_state")
        sa_other = val("source_absent_content", "not_relational_or_event_state")
        ret_rel = val("retained_content", "relational_or_event_state")
        ret_other = val("retained_content", "not_relational_or_event_state")
        if None not in (sa_rel, sa_other, ret_rel, ret_other):
            out["novelty_x_rel_event"][es] = {
                "source_absent_rel_event": sa_rel,
                "source_absent_other": sa_other,
                "retained_rel_event": ret_rel,
                "retained_other": ret_other,
                "interaction": round((sa_rel - sa_other) - (ret_rel - ret_other), 6),
                "meaning": "Positive means the source-absent-label local effect is especially concentrated in relational/event words beyond retained relational/event movement.",
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_losses", default=str(DEFAULT_EVENTS))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=23543023)
    args = ap.parse_args()
    t0 = time.time()
    events_path = pathlib.Path(args.event_losses)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_path)
    tables = build_tables(events, args.n_boot, args.seed)
    payload = {
        "status": "TARGET_CHANNEL_SEMANTIC_STRATIFICATION_100M",
        "meaning": "Semantic stratification of the mature packed target-selective local denoising readout. Positive drop_abs_minus_drop_copied_word means source-absent compact-content target labels locally reduce loss for that event subset.",
        "inputs": {"event_losses": str(events_path), "event_losses_sha256": sha256_file(events_path)},
        "contrast_orientations": {
            "drop_abs_minus_drop_copied_word": "positive = removing source-absent labels hurts this local event subset more than removing matched copied labels",
            "drop_abs_minus_full": "positive = source-absent deletion hurts versus historical full compact model",
            "drop_copied_word_minus_full": "positive = copied deletion hurts versus historical full compact model",
        },
        "n_events": len(events),
        **tables,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / "target_channel_semantic_stratification_100M.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research target-channel semantic stratification at 100M",
        "",
        f"JSON: `{out_json}`",
        "",
        "Positive `drop_abs_minus_drop_copied_word` means removing source-absent compact-content labels increased local loss relative to removing matched copied labels.",
        "",
        "## Focus table: main contrast",
        "",
    ]
    for es, recs in tables["focus"].items():
        md.append(f"### {es}")
        md.append("| cell | n_events | pieces | delta | boot p025 | boot p975 | frac_gt0 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for cell, rec in recs.items():
            boot = rec.get("bootstrap", {})
            md.append(f"| {cell} | {rec.get('n_events')} | {rec.get('pieces')} | {rec.get('piece_weighted_delta')} | {boot.get('p025')} | {boot.get('p975')} | {boot.get('fraction_gt0')} |")
        if es in tables["novelty_x_rel_event"]:
            md.append("")
            md.append("Novelty×rel/event: `" + json.dumps(tables["novelty_x_rel_event"][es], ensure_ascii=False) + "`")
        md.append("")
    (out_dir / "target_channel_semantic_stratification_100M.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "focus": tables["focus"], "novelty_x_rel_event": tables["novelty_x_rel_event"], "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
