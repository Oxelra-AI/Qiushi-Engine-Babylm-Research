#!/usr/bin/env python3
"""research: shortcut baselines for the paired-world v2 pilot.

Runs CPU-only deterministic baselines on the score-visible and score-ablated
LaLiga paired-world packets.  The goal is to check whether simple non-binding
signals can solve the construction before any neural training is considered.

Predicted label is whether the team_a-over-team_b hypothesis is entailed for a
context packet.

Baselines include:
  * majority / template-majority / canonical bag-of-words upper bounds;
  * relation-erased canonical bag-of-words upper bound;
  * lexical overlap of context with the two directed hypotheses;
  * leave-family-out team-prior and season/year-prior heuristics;
  * score-visible numeric parser positive control;
  * relation-sentence oracle for intended predicate-argument signal.

No model training, no selected evaluation, no SuperGLUE/AoA, no upload/submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
DEFAULT_IN_DIR = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_shortcut_pilot_v2"
DEFAULT_OUT = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_shortcut_baselines"
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def toks(s: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(s)]


def bow_key(tokens: list[str]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(Counter(tokens).items()))


def canonicalize(text: str, team_a: str, team_b: str, *, normalize_numbers: bool = True, erase_relation: bool = False) -> str:
    s = text
    if erase_relation:
        # The last sentence is the relation sentence in the v2 materializer.
        parts = re.split(r"(?<=\.)\s+", s.strip())
        if len(parts) > 1:
            s = " ".join(parts[:-1])
    s = re.sub(r"(?<!\w)" + re.escape(team_a) + r"(?!\w)", "TEAM_A", s)
    s = re.sub(r"(?<!\w)" + re.escape(team_b) + r"(?!\w)", "TEAM_B", s)
    if normalize_numbers:
        s = re.sub(r"\b\d[\d,/.-]*\b", "NUM", s)
    return s


def label_team_a_wins(p: dict[str, Any]) -> int:
    return 1 if p.get("entailed_direction") == "team_a_over_team_b" else 0


def score_accuracy_binary(preds: list[float | None], labels: list[int]) -> dict[str, Any]:
    assert len(preds) == len(labels)
    correct = 0.0
    covered = 0
    ties = 0
    for pred, y in zip(preds, labels):
        if pred is None or (isinstance(pred, float) and math.isnan(pred)):
            continue
        covered += 1
        if pred == 0.5:
            ties += 1
            correct += 0.5
        elif int(pred) == y:
            correct += 1.0
    return {
        "n": len(labels),
        "covered": covered,
        "coverage": covered / len(labels) if labels else None,
        "tie_count": ties,
        "tie_rate": ties / covered if covered else None,
        "accuracy_tie_half": correct / covered if covered else None,
        "accuracy_on_all_with_uncovered_wrong": correct / len(labels) if labels else None,
    }


def group_majority_accuracy(packets: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str], *, leave_family_out: bool = False) -> dict[str, Any]:
    labels = [label_team_a_wins(p) for p in packets]
    preds: list[float | None] = []
    if not leave_family_out:
        groups: dict[str, Counter[int]] = defaultdict(Counter)
        for p, y in zip(packets, labels):
            groups[key_fn(p)][y] += 1
        for p in packets:
            c = groups[key_fn(p)]
            if c[1] > c[0]:
                preds.append(1)
            elif c[0] > c[1]:
                preds.append(0)
            else:
                preds.append(0.5)
    else:
        for i, p in enumerate(packets):
            c: Counter[int] = Counter()
            k = key_fn(p)
            fid = p["family_id"]
            for j, q in enumerate(packets):
                if q["family_id"] != fid and key_fn(q) == k:
                    c[labels[j]] += 1
            if c[1] > c[0]:
                preds.append(1)
            elif c[0] > c[1]:
                preds.append(0)
            elif c[0] or c[1]:
                preds.append(0.5)
            else:
                preds.append(None)
    return score_accuracy_binary(preds, labels)


def lexical_overlap_baseline(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label_team_a_wins(p) for p in packets]
    preds: list[float | None] = []
    for p in packets:
        ctoks = Counter(toks(p["text"]))
        scores = []
        for hyp in p["directed_hypotheses"]:
            htoks = Counter(toks(hyp["hypothesis"]))
            scores.append(sum(min(ctoks[k], v) for k, v in htoks.items()))
        if scores[0] > scores[1]:
            preds.append(1)
        elif scores[1] > scores[0]:
            preds.append(0)
        else:
            preds.append(0.5)
    return score_accuracy_binary(preds, labels)


def score_numeric_parser(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label_team_a_wins(p) for p in packets]
    preds: list[float | None] = []
    for p in packets:
        text = p["text"]
        # Pattern produced by the materializer: score line was WINNER wg, LOSER lg.
        m = re.search(r"The score line was (.*) ([0-9]+), (.*) ([0-9]+)\.", text)
        if not m:
            preds.append(None)
            continue
        name1, g1, name2, g2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if g1 == g2:
            preds.append(0.5)
            continue
        pred_winner = name1 if g1 > g2 else name2
        preds.append(1 if pred_winner == p["team_a"] else 0)
    return score_accuracy_binary(preds, labels)


def relation_oracle(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label_team_a_wins(p) for p in packets]
    preds: list[float | None] = []
    for p in packets:
        t = p["template"]
        text = p["text"]
        ta, tb = p["team_a"], p["team_b"]
        pred_winner = None
        if t == "active_winner_first":
            m = re.search(r"(.+?) prevailed over (.+?) in the match\.", text)
            pred_winner = m.group(1) if m else None
        elif t == "passive_loser_first":
            m = re.search(r"(.+?) was beaten by (.+?) in the match\.", text)
            pred_winner = m.group(2) if m else None
        elif t == "lost_to_loser_first":
            m = re.search(r"(.+?) lost to (.+?) in the match\.", text)
            pred_winner = m.group(2) if m else None
        elif t == "not_loser_winner_first":
            m = re.search(r"(.+?), not (.+?), came away as the winning side\.", text)
            pred_winner = m.group(1) if m else None
        if pred_winner == ta:
            preds.append(1)
        elif pred_winner == tb:
            preds.append(0)
        else:
            preds.append(None)
    return score_accuracy_binary(preds, labels)


def team_prior_lofo(packets: list[dict[str, Any]]) -> dict[str, Any]:
    # Use unique contexts only to avoid 4-template replication, then project preds back to rows.
    labels = [label_team_a_wins(p) for p in packets]
    preds: list[float | None] = []
    for p in packets:
        fid = p["family_id"]
        wins: Counter[str] = Counter()
        losses: Counter[str] = Counter()
        seen_contexts: set[tuple[str, str]] = set()
        for q in packets:
            key = (q["family_id"], q["context_name"])
            if q["family_id"] == fid or key in seen_contexts:
                continue
            seen_contexts.add(key)
            wins[q["winner"]] += 1
            losses[q["loser"]] += 1
        a_score = wins[p["team_a"]] - losses[p["team_a"]]
        b_score = wins[p["team_b"]] - losses[p["team_b"]]
        if a_score > b_score:
            preds.append(1)
        elif b_score > a_score:
            preds.append(0)
        else:
            preds.append(0.5)
    return score_accuracy_binary(preds, labels)


def season_year_key(p: dict[str, Any], kind: str) -> str:
    raw = p.get("raw_record") or {}
    if kind == "season":
        return str(raw.get("Season", ""))
    if kind == "year":
        date = str(raw.get("Date", ""))
        m = re.search(r"(20\d\d|19\d\d)", date)
        if m:
            return m.group(1)
        parts = date.split("/")
        if len(parts) == 3 and len(parts[-1]) == 4:
            return parts[-1]
        return ""
    raise ValueError(kind)


def run_variant(variant: str, packets: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [label_team_a_wins(p) for p in packets]
    return {
        "n_packets": len(packets),
        "n_families": len({p["family_id"] for p in packets}),
        "label_team_a_win_rate": sum(labels) / len(labels) if labels else None,
        "majority": score_accuracy_binary([1 if sum(labels) > len(labels) / 2 else 0.5 for _ in labels], labels),
        "template_majority_upper_bound": group_majority_accuracy(packets, lambda p: p["template"], leave_family_out=False),
        "canonical_bow_full_names_numbers_normalized_upper_bound": group_majority_accuracy(packets, lambda p: bow_key(toks(canonicalize(p["text"], p["team_a"], p["team_b"], normalize_numbers=True, erase_relation=False))), leave_family_out=False),
        "canonical_bow_full_names_only_upper_bound": group_majority_accuracy(packets, lambda p: bow_key(toks(canonicalize(p["text"], p["team_a"], p["team_b"], normalize_numbers=False, erase_relation=False))), leave_family_out=False),
        "relation_erased_canonical_bow_upper_bound": group_majority_accuracy(packets, lambda p: bow_key(toks(canonicalize(p["text"], p["team_a"], p["team_b"], normalize_numbers=True, erase_relation=True))), leave_family_out=False),
        "relation_erased_canonical_bow_leave_family_out": group_majority_accuracy(packets, lambda p: bow_key(toks(canonicalize(p["text"], p["team_a"], p["team_b"], normalize_numbers=True, erase_relation=True))), leave_family_out=True),
        "lexical_overlap_context_hypothesis": lexical_overlap_baseline(packets),
        "team_strength_prior_leave_family_out": team_prior_lofo(packets),
        "season_prior_leave_family_out": group_majority_accuracy(packets, lambda p: season_year_key(p, "season"), leave_family_out=True),
        "year_prior_leave_family_out": group_majority_accuracy(packets, lambda p: season_year_key(p, "year"), leave_family_out=True),
        "score_numeric_parser_positive_control": score_numeric_parser(packets),
        "relation_sentence_oracle": relation_oracle(packets),
        "per_template_relation_oracle": {t: relation_oracle([p for p in packets if p["template"] == t]) for t in sorted({p["template"] for p in packets})},
    }


def write_md(payload: dict[str, Any], out_dir: Path) -> None:
    lines = ["# research paired-world shortcut baselines", "", f"Created UTC: `{payload['created_utc']}`", ""]
    for variant, rec in payload["variants"].items():
        lines.append(f"## {variant}")
        for key in [
            "label_team_a_win_rate",
            "majority",
            "template_majority_upper_bound",
            "canonical_bow_full_names_numbers_normalized_upper_bound",
            "canonical_bow_full_names_only_upper_bound",
            "relation_erased_canonical_bow_upper_bound",
            "relation_erased_canonical_bow_leave_family_out",
            "lexical_overlap_context_hypothesis",
            "team_strength_prior_leave_family_out",
            "season_prior_leave_family_out",
            "year_prior_leave_family_out",
            "score_numeric_parser_positive_control",
            "relation_sentence_oracle",
        ]:
            lines.append(f"- {key}: `{json.dumps(rec.get(key), sort_keys=True)}`")
        lines.append("")
    lines.append("## Interpretation")
    lines.append(payload["interpretation"])
    lines.append("")
    lines.append("## Boundary")
    lines.append(payload["boundary"])
    (out_dir / "paired_world_shortcut_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    variants: dict[str, Any] = {}
    for variant in ["score_ablated", "score_visible"]:
        packets = read_jsonl(args.input_dir / f"laliga_{variant}_packets.jsonl")
        variants[variant] = run_variant(variant, packets)
    payload = {
        "status": "PAIRED_WORLD_SHORTCUT_BASELINES",
        "created_utc": now(),
        "input_dir": str(args.input_dir),
        "variants": variants,
        "interpretation": (
            "A usable score-ablated paired-world substrate should be unsolved by majority, template-majority, canonical bag-of-words, relation-erased, lexical-overlap, entity/team-prior, and date/season heuristics, while being solved by a relation-sentence oracle. The score-visible variant should be solved by the numeric parser, demonstrating the shortcut that the ablated variant removes."
        ),
        "boundary": "CPU deterministic baselines only; no neural model training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paired_world_shortcut_baselines.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, args.out_dir)
    print(json.dumps({
        "status": payload["status"],
        "out_dir": str(args.out_dir),
        "score_ablated_key_acc": {
            k: variants["score_ablated"][k].get("accuracy_tie_half") if isinstance(variants["score_ablated"].get(k), dict) else variants["score_ablated"].get(k)
            for k in ["majority", "template_majority_upper_bound", "canonical_bow_full_names_numbers_normalized_upper_bound", "relation_erased_canonical_bow_leave_family_out", "lexical_overlap_context_hypothesis", "team_strength_prior_leave_family_out", "season_prior_leave_family_out", "year_prior_leave_family_out", "score_numeric_parser_positive_control", "relation_sentence_oracle"]
        },
        "score_visible_numeric_parser_acc": variants["score_visible"]["score_numeric_parser_positive_control"].get("accuracy_tie_half"),
        "boundary": payload["boundary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
