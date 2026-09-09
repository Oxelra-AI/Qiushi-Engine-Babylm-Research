#!/usr/bin/env python3
"""research: prepare held-out state-margin probe sets for the state-use arm.

The validator's balanced split used only 200 UPDATED_USE items as held-out and
left 925 accepted UPDATED_USE packets unused.  This script verifies that those
unused UPDATED_USE packets do not overlap the training split and writes a larger
non-training probe set so the UPDATED half has higher statistical power while the
balanced 200/200 set remains available for mixed-half comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

ROOT = pathlib.Path.cwd()
DATA = ROOT / "experiments/archive/relation_learning/data/state_use_generation"
OUT = ROOT / "experiments/archive/relation_learning/data/state_margin_probe"
ALL = DATA / "validated_state_use_packets_plainuse_full_step047_all.jsonl"
TRAIN = DATA / "validated_state_use_packets_plainuse_full_step047_train.jsonl"
HELDOUT = DATA / "validated_state_use_packets_plainuse_full_step047_heldout.jsonl"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\u2019'][A-Za-z0-9]+)?")

LOCATION = {
    "location", "locat", "site", "place", "city", "country", "nation", "state", "province",
    "town", "village", "room", "house", "hotel", "classroom", "kitchen", "office", "school",
    "university", "library", "campus", "building", "porch", "wagon", "bay", "shore", "valley",
    "river", "island", "line", "station", "route", "inside", "outside", "nearby", "north",
    "south", "east", "west", "austin", "sydney", "france", "europe", "maryland", "washington",
}
POSSESSION = {
    "own", "possess", "hold", "holding", "carry", "carries", "wear", "wearing", "badge",
    "coat", "trophy", "title", "championship", "money", "book", "weapon", "gear", "hand",
    "recording", "server", "archive", "document", "proof", "evidence", "key", "sword",
}
ROLE = {
    "king", "queen", "monarch", "director", "lead", "leader", "member", "founder", "advisor",
    "champion", "prisoner", "nobleman", "scientist", "physicist", "theologian", "paleontologist",
    "executive", "colleague", "participant", "judge", "captain", "doctor", "missionary", "athlete",
    "powerhouse", "peacekeeper", "government", "company", "team", "party", "movement", "institution",
    "affiliat", "affiliated", "catholic", "orthodox", "protestant", "policy",
}
PROPERTY = {
    "color", "yellow", "bright", "warm", "cold", "large", "small", "long", "short", "complex",
    "overgrown", "weed", "beautiful", "hilarious", "satirical", "ancient", "stone", "grand",
    "renovated", "digital", "electric", "quantum", "shield", "fierce", "intensity", "cheerful",
    "calm", "energetic", "silent", "confused", "quiet", "modern", "tactical",
}
STATUS = {
    "defeat", "defeated", "lost", "ill", "fever", "recovery", "dead", "alive", "captured",
    "capture", "escape", "retired", "retir", "dissolved", "dissolv", "closed", "clos", "broken",
    "fixed", "transmit", "transmitted", "melt", "melted", "surviv", "extinct", "standing",
    "sleep", "sleeping", "looking", "look", "work", "working", "waiting", "wait", "fight",
    "battle", "victory", "won", "winner", "promoted", "promot", "renamed", "renam",
}
CHANGE_TERMS = {
    "became", "become", "turned", "turn", "transformed", "transform", "shifted", "shift",
    "changed", "change", "moved", "move", "left", "entered", "stopped", "declared", "appointed",
    "converted", "evolved", "transitioned", "replaced", "relocated", "renamed", "stripped",
    "emerged", "reached", "began", "started", "stored", "activated", "equipped", "promoted",
}


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem(word: str) -> str:
    w = word.lower().strip("\u2019'")
    if w.endswith("'s"):
        w = w[:-2]
    if len(w) > 6 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 5 and w.endswith("ied"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 5 and w.endswith("es"):
        return w[:-2]
    if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def terms(text: str) -> list[str]:
    return [stem(m.group(0)) for m in WORD_RE.finditer(str(text)) if len(stem(m.group(0))) >= 3]


def type_scores(text: str, explicit_terms: list[str] | None = None) -> Counter:
    ts = set(explicit_terms or terms(text))
    scores = Counter()
    scores["location_containment"] = len(ts & LOCATION)
    scores["possession_artifact"] = len(ts & POSSESSION)
    scores["role_identity_affiliation"] = len(ts & ROLE)
    scores["property_attribute"] = len(ts & PROPERTY)
    scores["status_condition_event"] = len(ts & STATUS)
    # A few transparent phrase rules that catch common Entity-like states.
    low = str(text).lower()
    if re.search(r"\b(in|inside|outside|at|near|beside|on|from|toward|between)\b", low):
        scores["location_containment"] += 1
    if re.search(r"\b(has|had|holds?|carries?|wears?|owns?)\b", low):
        scores["possession_artifact"] += 1
    if re.search(r"\b(as|role|member|director|leader|king|queen|founder|advisor|champion)\b", low):
        scores["role_identity_affiliation"] += 1
    if re.search(r"\b(defeated|won|lost|ill|retired|captured|dead|alive|closed|dissolved)\b", low):
        scores["status_condition_event"] += 1
    return scores


def classify_state(text: str, explicit_terms: list[str] | None = None) -> str:
    sc = type_scores(text, explicit_terms)
    if not sc or max(sc.values(), default=0) <= 0:
        return "other_relation_state"
    order = [
        "location_containment",
        "possession_artifact",
        "role_identity_affiliation",
        "property_attribute",
        "status_condition_event",
    ]
    best = max(order, key=lambda k: (sc[k], -order.index(k)))
    # If location and role tie, keep location for Entity-aligned interpretation.
    return best


def annotate(pkt: dict[str, Any]) -> dict[str, Any]:
    r = dict(pkt)
    source_terms = list(r.get("source_state_terms") or terms(r.get("source_state", "")))
    new_terms = list(r.get("new_state_terms") or terms(r.get("new_state", "")))
    r["state_type_source"] = classify_state(str(r.get("source_state", "")), source_terms)
    r["state_type_new"] = classify_state(str(r.get("new_state", "")), new_terms)
    if r.get("packet_type") == "UPDATED_USE":
        r["gold_state_type"] = r["state_type_new"]
        r["competitor_state_type"] = r["state_type_source"]
        low_update = " ".join(terms(str(r.get("update_sentence", ""))))
        sc = set(source_terms); nc = set(new_terms)
        overlap = len(sc & nc) / max(1, len(sc | nc))
        change_word = any(t in low_update.split() for t in CHANGE_TERMS)
        r["updated_conflict_heuristic"] = bool(overlap <= 0.20 and (r["state_type_source"] != r["state_type_new"] or change_word))
    else:
        r["gold_state_type"] = r["state_type_source"]
        r["competitor_state_type"] = r["state_type_new"]
        r["updated_conflict_heuristic"] = None
    return r


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(r.get(key)) for r in rows).most_common())


def summarize_surface(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for path_key, label in [
        (("surface", "use_vs_source", "content_jaccard"), "use_source_content_jaccard"),
        (("surface", "use_vs_source", "longest_common_contiguous_words"), "use_source_lcs"),
        (("surface", "use_vs_update", "longest_common_contiguous_words"), "use_update_lcs"),
    ]:
        vals = []
        for r in rows:
            x: Any = r
            try:
                for k in path_key:
                    x = x[k]
                vals.append(float(x))
            except Exception:
                pass
        if vals:
            out[label] = {
                "mean": round(statistics.mean(vals), 4),
                "median": round(statistics.median(vals), 4),
                "max": round(max(vals), 4),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", default=str(ALL))
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--heldout", default=str(HELDOUT))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = [annotate(r) for r in read_jsonl(pathlib.Path(args.all))]
    train_rows = [annotate(r) for r in read_jsonl(pathlib.Path(args.train))]
    heldout_rows = [annotate(r) for r in read_jsonl(pathlib.Path(args.heldout))]

    train_ids = {r["pair_id"] for r in train_rows}
    heldout_ids = {r["pair_id"] for r in heldout_rows}
    all_ids = {r["pair_id"] for r in all_rows}
    unused = [r for r in all_rows if r["pair_id"] not in train_ids and r["pair_id"] not in heldout_ids]
    unused_updated = [r for r in unused if r.get("packet_type") == "UPDATED_USE"]
    unused_distractor = [r for r in unused if r.get("packet_type") == "UNCHANGED_DISTRACTOR_USE"]
    extended = sorted(heldout_rows + unused_updated, key=lambda r: (r.get("packet_type"), int(r.get("original_prompt_index", 10**12)), r.get("pair_id", "")))
    balanced = sorted(heldout_rows, key=lambda r: (r.get("packet_type"), int(r.get("original_prompt_index", 10**12)), r.get("pair_id", "")))

    paths = {
        "balanced_heldout": out / "state_margin_probe_balanced_heldout.jsonl",
        "unused_updated_extra": out / "state_margin_probe_unused_updated_extra.jsonl",
        "extended_nontrain": out / "state_margin_probe_extended_nontrain.jsonl",
        "train_annotated_copy": out / "state_margin_probe_train_annotated_copy.jsonl",
    }
    write_jsonl(paths["balanced_heldout"], balanced)
    write_jsonl(paths["unused_updated_extra"], sorted(unused_updated, key=lambda r: int(r.get("original_prompt_index", 10**12))))
    write_jsonl(paths["extended_nontrain"], extended)
    write_jsonl(paths["train_annotated_copy"], train_rows)

    overlap_report = {
        "train_intersects_heldout": sorted(train_ids & heldout_ids),
        "train_intersects_unused_updated": sorted(train_ids & {r["pair_id"] for r in unused_updated}),
        "heldout_intersects_unused_updated": sorted(heldout_ids & {r["pair_id"] for r in unused_updated}),
        "all_pair_id_count": len(all_ids),
        "all_row_count": len(all_rows),
    }
    meta = {
        "status": "STATE_MARGIN_PROBE_SETS_READY",
        "inputs": {"all": str(args.all), "train": str(args.train), "heldout": str(args.heldout)},
        "input_sha256": {"all": sha256_file(pathlib.Path(args.all)), "train": sha256_file(pathlib.Path(args.train)), "heldout": sha256_file(pathlib.Path(args.heldout))},
        "counts": {
            "all": len(all_rows),
            "train": len(train_rows),
            "heldout_balanced": len(heldout_rows),
            "unused_total": len(unused),
            "unused_updated_extra": len(unused_updated),
            "unused_distractor": len(unused_distractor),
            "extended_nontrain": len(extended),
        },
        "type_counts": {
            "all": count_by(all_rows, "packet_type"),
            "train": count_by(train_rows, "packet_type"),
            "heldout_balanced": count_by(heldout_rows, "packet_type"),
            "unused": count_by(unused, "packet_type"),
            "extended_nontrain": count_by(extended, "packet_type"),
        },
        "gold_state_type_counts": {
            "heldout_balanced": count_by(heldout_rows, "gold_state_type"),
            "extended_nontrain": count_by(extended, "gold_state_type"),
            "updated_heldout_plus_extra": count_by([r for r in extended if r.get("packet_type") == "UPDATED_USE"], "gold_state_type"),
            "distractor_heldout": count_by([r for r in extended if r.get("packet_type") == "UNCHANGED_DISTRACTOR_USE"], "gold_state_type"),
        },
        "updated_conflict_heuristic_counts": count_by([r for r in extended if r.get("packet_type") == "UPDATED_USE"], "updated_conflict_heuristic"),
        "surface": {
            "heldout_balanced": summarize_surface(heldout_rows),
            "unused_updated_extra": summarize_surface(unused_updated),
            "extended_nontrain": summarize_surface(extended),
        },
        "overlap_report": {k: (v[:20] if isinstance(v, list) else v) for k, v in overlap_report.items()},
        "no_train_overlap_for_extended_nontrain": not (train_ids & {r["pair_id"] for r in extended}),
        "paths": {k: str(v) for k, v in paths.items()},
        "sha256": {k: sha256_file(v) for k, v in paths.items()},
        "note": "The extended non-training set keeps the balanced 200/200 held-out items and adds the 925 accepted UPDATED_USE packets that were not in train or held-out; it does not alter the already-running training stream.",
    }
    (out / "state_margin_probe_set_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
