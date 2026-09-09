#!/usr/bin/env python3
"""research: stratify the validated target-channel loss effects by lexical semantics.

This CPU-only readout uses completed 20M target-selective loss files.  It asks
whether the source-absent target-label effect is spread across all novel content
or concentrated in relational/event/state-like rewrite words, which matters for
interpreting the pending 100M packed endpoint.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Iterable

USER_ROOT = pathlib.Path(".").resolve()
SCRIPT_DIR = USER_ROOT / "experiments/archive/representation_and_objectives/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from source_absent_lexical_profile import flags as lexical_flags  # noqa: E402
from annotate_packed_pool import norm_word, word_class  # noqa: E402

EVENT_LOSSES = pathlib.Path("experiments/archive/representation_and_objectives/data/wholeword_control_readout/event_losses.jsonl")
EVENT_LOSSES = pathlib.Path("experiments/archive/representation_and_objectives/data/source_disjoint_target_probe/source_disjoint_event_losses.jsonl")
OUT_DIR = pathlib.Path("experiments/archive/representation_and_objectives/data/target_channel_semantic_stratification")

CHECKPOINTS = ["chck_10M", "chck_20M"]
ARMS = ("drop_abs", "drop_copied_word")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def q(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return float(s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))) )])


def flag_groups(word: str, category: str) -> set[str]:
    nw = norm_word(word)
    fs = set(lexical_flags(word, nw))
    groups = {"all"}
    groups.update(fs)
    rel_event = bool({"relational_or_abstracting_word", "event_or_state_word"} & fs)
    if rel_event:
        groups.add("relational_or_event_state")
    else:
        groups.add("not_relational_or_event_state")
    if {"capitalized", "number_or_year"} & fs:
        groups.add("capitalized_or_number")
    else:
        groups.add("not_capitalized_or_number")
    if not ({"relational_or_abstracting_word", "event_or_state_word", "capitalized", "number_or_year", "generic_entity_word"} & fs):
        groups.add("ordinary_nonrel_nonentity")
    groups.add(f"class_{word_class(word)}")
    groups.add(f"category_{category}")
    return groups


def parse_event_line(obj: dict[str, Any], source_name: str) -> dict[str, Any]:
    word = str(obj.get("word", ""))
    category = str(obj.get("category", "unknown"))
    pair_id = str(obj.get("pair_id", obj.get("event_uid", "unknown")))
    rec = {
        "source_name": source_name,
        "eval_set": str(obj.get("eval_set", "train_fixed")),
        "pair_id": pair_id,
        "doc_id": str(obj.get("doc_id", "")),
        "category": category,
        "word": word,
        "norm": norm_word(word),
        "word_class": word_class(word),
        "flags": sorted(lexical_flags(word, norm_word(word))),
        "groups": sorted(flag_groups(word, category)),
        "n_pieces": int(obj.get("n_pieces", 0)),
        "loss_by_checkpoint": {},
    }
    for ck in CHECKPOINTS:
        by_arm = obj.get(ck, {})
        if not all(a in by_arm for a in ARMS):
            continue
        da = by_arm["drop_abs"]
        dc = by_arm["drop_copied_word"]
        pieces = int(da.get("pieces", obj.get("n_pieces", 0)))
        if pieces <= 0:
            continue
        rec["loss_by_checkpoint"][ck] = {
            "pieces": pieces,
            "drop_abs_loss_sum": float(da["loss_sum"]),
            "drop_copied_word_loss_sum": float(dc["loss_sum"]),
            "delta_loss_sum": float(da["loss_sum"]) - float(dc["loss_sum"]),
            "delta_mean": (float(da["loss_sum"]) - float(dc["loss_sum"])) / pieces,
        }
    return rec


def load_records(path: pathlib.Path, source_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec = parse_event_line(obj, source_name)
            if rec["loss_by_checkpoint"]:
                out.append(rec)
    return out


def summarize(records: list[dict[str, Any]], ck: str, n_boot: int, rng: random.Random) -> dict[str, Any]:
    vals = []
    by_cluster: dict[str, list[tuple[float, int]]] = collections.defaultdict(list)
    flag_counter = collections.Counter()
    type_counter = collections.Counter()
    for r in records:
        v = r["loss_by_checkpoint"].get(ck)
        if not v:
            continue
        delta = float(v["delta_loss_sum"])
        pieces = int(v["pieces"])
        vals.append((delta, pieces))
        by_cluster[str(r["pair_id"])].append((delta, pieces))
        flag_counter.update(r["flags"])
        type_counter[r["norm"]] += 1
    tot_pieces = sum(p for _, p in vals)
    tot_delta = sum(d for d, _ in vals)
    if not vals or tot_pieces <= 0:
        return {"n_events": 0}

    clusters = list(by_cluster.items())
    boot = []
    if n_boot > 0 and len(clusters) >= 2:
        for _ in range(n_boot):
            bd = 0.0
            bp = 0
            for _ in clusters:
                _, evs = clusters[rng.randrange(len(clusters))]
                for d, p in evs:
                    bd += d
                    bp += p
            if bp > 0:
                boot.append(bd / bp)
    means = [d / p for d, p in vals if p > 0]
    return {
        "n_events": len(vals),
        "n_clusters": len(clusters),
        "pieces": int(tot_pieces),
        "piece_weighted_delta_nats": round(tot_delta / tot_pieces, 6),
        "event_delta_mean_unweighted": round(statistics.mean(means), 6),
        "event_delta_median_unweighted": round(statistics.median(means), 6),
        "bootstrap": {
            "n_boot": len(boot),
            "median": None if not boot else round(statistics.median(boot), 6),
            "p025": None if not boot else round(q(boot, 0.025), 6),
            "p975": None if not boot else round(q(boot, 0.975), 6),
            "fraction_gt0": None if not boot else round(sum(1 for x in boot if x > 0) / len(boot), 6),
        },
        "flag_fraction_in_subset": {k: round(v / len(vals), 6) for k, v in sorted(flag_counter.items())},
        "top_norms": [{"norm": w, "count": int(c)} for w, c in type_counter.most_common(25)],
    }


def build_tables(records: list[dict[str, Any]], source_name: str, n_boot: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    # Categories and semantic flag groups are deliberately overlapping: they answer
    # different interpretive questions rather than forming one partition.
    wanted_groups = [
        "all",
        "relational_or_event_state",
        "not_relational_or_event_state",
        "relational_or_abstracting_word",
        "event_or_state_word",
        "ordinary_nonrel_nonentity",
        "capitalized_or_number",
        "not_capitalized_or_number",
        "generic_entity_word",
        "number_or_year",
        "capitalized",
    ]
    out: dict[str, Any] = {"source_name": source_name, "by_eval_set_category_group": {}}
    eval_sets = sorted(set(str(r["eval_set"]) for r in records))
    categories = sorted(set(str(r["category"]) for r in records))
    for eval_set in eval_sets:
        out["by_eval_set_category_group"].setdefault(eval_set, {})
        for cat in categories:
            cat_recs = [r for r in records if r["eval_set"] == eval_set and r["category"] == cat]
            if not cat_recs:
                continue
            out["by_eval_set_category_group"][eval_set].setdefault(cat, {})
            for grp in wanted_groups:
                grp_recs = [r for r in cat_recs if grp in r["groups"]]
                if grp_recs:
                    out["by_eval_set_category_group"][eval_set][cat][grp] = {
                        ck: summarize(grp_recs, ck, n_boot=n_boot, rng=rng) for ck in CHECKPOINTS
                    }
    return out


def extract_focus(payload: dict[str, Any]) -> dict[str, Any]:
    focus = {}
    table = payload["by_eval_set_category_group"]
    for eval_set, cats in table.items():
        if "source_absent_content" not in cats:
            continue
        focus[eval_set] = {}
        for grp in ["all", "relational_or_event_state", "not_relational_or_event_state", "ordinary_nonrel_nonentity", "capitalized_or_number"]:
            if grp in cats["source_absent_content"]:
                focus[eval_set][grp] = cats["source_absent_content"][grp].get("chck_20M", {})
    return focus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_losses", default=str(EVENT_LOSSES))
    ap.add_argument("--event_losses", default=str(EVENT_LOSSES))
    ap.add_argument("--output_dir", default=str(OUT_DIR))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=23443023)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_rows = load_records(pathlib.Path(args.event_losses), "train_fixed")
    source_disjoint_rows = load_records(pathlib.Path(args.event_losses), "source_disjoint")
    p226 = build_tables(training_rows, "train_fixed", args.n_boot, args.seed)
    p227 = build_tables(source_disjoint_rows, "source_disjoint", args.n_boot, args.seed + 1)

    result = {
        "status": "TARGET_CHANNEL_SEMANTIC_STRATIFICATION",
        "meaning": "Positive delta means drop_abs_content has higher compact-side loss than the matched whole-word copied-drop arm; this reads as the removed source-absent labels mattering more for that event subset.",
        "inputs": {
            "event_losses": args.event_losses,
            "event_losses_sha256": sha256_file(pathlib.Path(args.event_losses)),
            "event_losses": args.event_losses,
            "event_losses_sha256": sha256_file(pathlib.Path(args.event_losses)),
        },
        "arm_contrast": "drop_abs_minus_drop_copied_word",
        "semantic_group_definition": "relational_or_event_state = research heuristic relational_or_abstracting_word OR event_or_state_word; groups overlap except the explicit relational_or_event_state/not_relational_or_event_state partition.",
        "research": p226,
        "research": p227,
        "focus_chck20_source_absent_content": {
            "research": extract_focus(p226),
            "research": extract_focus(p227),
        },
        "reading_for_pending_100M": {
            "if_endpoint_moves_with_relational_ewok": "The removed population is not source absence in isolation: research and this stratification should be used to read it as naturally coupled novel-plus-relational/event abstractive compact content unless a later matched separation holds semantics fixed.",
            "if_local_effect_is_broad_across_rel_event_and_other": "Source novelty may be a broad property of compact-side denoising, but endpoint mediation still needs the pending 100M task-surface pattern.",
            "if_local_effect_concentrates_in_rel_event": "The strongest next separation should hold relational/event semantics fixed while varying source novelty, and hold source novelty fixed while varying relational/event semantics.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out_json = out_dir / "target_channel_semantic_stratification.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = [
        "# research target-channel semantic stratification",
        "",
        f"JSON: `{out_json}`",
        "",
        "Positive values mean that removing source-absent labels harms the evaluated compact-side events more than removing the matched copied whole-word labels.",
        "",
        "## Focus: chck_20M source_absent_content",
        "",
    ]
    for src, focus in result["focus_chck20_source_absent_content"].items():
        md.append(f"### {src}")
        for eval_set, groups in focus.items():
            md.append(f"- eval_set `{eval_set}`")
            for grp, rec in groups.items():
                if rec.get("n_events", 0):
                    boot = rec.get("bootstrap", {})
                    md.append(
                        f"  - {grp}: n={rec['n_events']}, pieces={rec['pieces']}, delta={rec['piece_weighted_delta_nats']}, "
                        f"boot[{boot.get('p025')}, {boot.get('p975')}], frac_gt0={boot.get('fraction_gt0')}"
                    )
        md.append("")
    md += [
        "## Interpretation for the pending packed endpoint",
        "",
        result["reading_for_pending_100M"]["if_endpoint_moves_with_relational_ewok"],
        "",
    ]
    (out_dir / "target_channel_semantic_stratification.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "out_json": str(out_json),
        "focus_chck20_source_absent_content": result["focus_chck20_source_absent_content"],
        "elapsed_sec": result["elapsed_sec"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
