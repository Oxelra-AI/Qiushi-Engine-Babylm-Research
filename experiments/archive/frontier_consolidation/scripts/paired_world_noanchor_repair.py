#!/usr/bin/env python3
"""research: remove numeric/event anchors from the v2 score-ablated paired-world pilot.

The v2 pilot already crosses templates and balances entity/query bags, but still
contains date/season anchors (`any_digit_rate=1.0`).  This CPU-only repair creates
a stricter score_ablated_noanchor variant in which the text contains only a
neutral match sentence and the source-attested role sentence.  It then recomputes
simple shortcut statistics.

No teacher use, neural training, selected evaluation, SuperGLUE/AoA, upload, or
leaderboard submission.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
DEFAULT_IN = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_shortcut_pilot_v2/laliga_score_ablated_packets.jsonl"
DEFAULT_OUT = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_noanchor_pilot"
WORD_RE = re.compile(r"[A-Za-z0-9]+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def norm_tokens(s: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(s)]


def bow_key(tokens: list[str]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(Counter(tokens).items()))


def count_name(text: str, name: str) -> int:
    return len(re.findall(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text))


def extract_relation_sentence(text: str) -> str:
    parts = re.split(r"(?<=\.)\s+", text.strip())
    if not parts:
        return text.strip()
    return parts[-1].strip()


def transform(p: dict[str, Any]) -> dict[str, Any]:
    q = dict(p)
    q["variant"] = "score_ablated_noanchor"
    relation = extract_relation_sentence(str(p["text"]))
    q["text"] = f"The record is about a LaLiga match between {p['team_a']} and {p['team_b']}. {relation}"
    q["anchor_policy"] = "removed_dates_seasons_scores_attendance_and_row_ids_from_training_text"
    return q


def label(p: dict[str, Any]) -> int:
    return 1 if p.get("entailed_direction") == "team_a_over_team_b" else 0


def accuracy_tie_half(preds: list[float | None], labels: list[int]) -> dict[str, Any]:
    correct = 0.0
    covered = 0
    ties = 0
    for pred, y in zip(preds, labels):
        if pred is None:
            continue
        covered += 1
        if pred == 0.5:
            ties += 1
            correct += 0.5
        elif int(pred) == y:
            correct += 1.0
    return {"n": len(labels), "covered": covered, "coverage": covered / len(labels) if labels else None, "tie_rate": ties / covered if covered else None, "accuracy_tie_half": correct / covered if covered else None}


def group_majority(packets: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    labs = [label(p) for p in packets]
    groups: dict[str, Counter[int]] = defaultdict(Counter)
    for p, y in zip(packets, labs):
        groups[key_fn(p)][y] += 1
    preds = []
    for p in packets:
        c = groups[key_fn(p)]
        if c[1] > c[0]:
            preds.append(1)
        elif c[0] > c[1]:
            preds.append(0)
        else:
            preds.append(0.5)
    return accuracy_tie_half(preds, labs)


def canon(text: str, team_a: str, team_b: str, erase_relation: bool = False) -> str:
    s = text
    if erase_relation:
        parts = re.split(r"(?<=\.)\s+", s.strip())
        if len(parts) > 1:
            s = " ".join(parts[:-1])
    s = re.sub(r"(?<!\w)" + re.escape(team_a) + r"(?!\w)", "TEAM_A", s)
    s = re.sub(r"(?<!\w)" + re.escape(team_b) + r"(?!\w)", "TEAM_B", s)
    s = re.sub(r"\b\d[\d,/.-]*\b", "NUM", s)
    return s


def lexical_overlap(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labs = [label(p) for p in packets]
    preds = []
    for p in packets:
        ct = Counter(norm_tokens(p["text"]))
        scores = []
        for h in p["directed_hypotheses"]:
            ht = Counter(norm_tokens(h["hypothesis"]))
            scores.append(sum(min(ct[k], v) for k, v in ht.items()))
        preds.append(1 if scores[0] > scores[1] else 0 if scores[1] > scores[0] else 0.5)
    return accuracy_tie_half(preds, labs)


def relation_oracle(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labs = [label(p) for p in packets]
    preds = []
    for p in packets:
        text = p["text"]
        t = p["template"]
        pred_winner = None
        if t == "active_winner_first":
            m = re.search(r"(.+?) prevailed over (.+?) in the match\.", text)
            pred_winner = m.group(1).split(". ")[-1] if m else None
        elif t == "passive_loser_first":
            m = re.search(r"(.+?) was beaten by (.+?) in the match\.", text)
            pred_winner = m.group(2) if m else None
        elif t == "lost_to_loser_first":
            m = re.search(r"(.+?) lost to (.+?) in the match\.", text)
            pred_winner = m.group(2) if m else None
        elif t == "not_loser_winner_first":
            m = re.search(r"(.+?), not (.+?), came away as the winning side\.", text)
            pred_winner = m.group(1).split(". ")[-1] if m else None
        if pred_winner == p["team_a"]:
            preds.append(1)
        elif pred_winner == p["team_b"]:
            preds.append(0)
        else:
            preds.append(None)
    return accuracy_tie_half(preds, labs)


def summarize(packets: list[dict[str, Any]]) -> dict[str, Any]:
    labs = [label(p) for p in packets]
    template_dir = defaultdict(Counter)
    for p in packets:
        template_dir[p["template"]][p["entailed_direction"]] += 1
    td = {t: {"team_a_over_team_b": c.get("team_a_over_team_b", 0), "team_b_over_team_a": c.get("team_b_over_team_a", 0), "absolute_delta": abs(c.get("team_a_over_team_b", 0) - c.get("team_b_over_team_a", 0))} for t, c in sorted(template_dir.items())}
    return {
        "status": "PAIRED_WORLD_NOANCHOR_PILOT",
        "created_utc": now(),
        "n_packets": len(packets),
        "n_families": len({p["family_id"] for p in packets}),
        "label_team_a_win_rate": sum(labs) / len(labs),
        "template_by_entailed_direction": td,
        "max_template_direction_abs_delta": max(v["absolute_delta"] for v in td.values()) if td else None,
        "winner_first_rate": sum(1 for p in packets if p["winner_first_in_relation_sentence"]) / len(packets),
        "team_a_count_equal_rate": sum(1 for p in packets if count_name(p["text"], p["team_a"]) == count_name(p["text"], p["team_b"])) / len(packets),
        "winner_loser_count_equal_rate": sum(1 for p in packets if count_name(p["text"], p["winner"]) == count_name(p["text"], p["loser"])) / len(packets),
        "exact_query_verb_defeated_rate": sum(1 for p in packets if "defeated" in norm_tokens(p["text"])) / len(packets),
        "any_digit_rate": sum(1 for p in packets if re.search(r"\b\d[\d,/.-]*\b", p["text"])) / len(packets),
        "hypothesis_bow_collision_rate": sum(1 for p in packets if Counter(norm_tokens(p["directed_hypotheses"][0]["hypothesis"])) == Counter(norm_tokens(p["directed_hypotheses"][1]["hypothesis"]))) / len(packets),
        "mean_text_token_count": sum(len(norm_tokens(p["text"])) for p in packets) / len(packets),
        "baselines": {
            "majority": accuracy_tie_half([0.5 for _ in packets], labs),
            "template_majority_upper_bound": group_majority(packets, lambda p: p["template"]),
            "canonical_bow_upper_bound": group_majority(packets, lambda p: bow_key(norm_tokens(canon(p["text"], p["team_a"], p["team_b"], erase_relation=False)))),
            "relation_erased_canonical_bow_upper_bound": group_majority(packets, lambda p: bow_key(norm_tokens(canon(p["text"], p["team_a"], p["team_b"], erase_relation=True)))),
            "lexical_overlap_context_hypothesis": lexical_overlap(packets),
            "relation_sentence_oracle": relation_oracle(packets),
        },
        "boundary": "CPU construction and deterministic shortcut baselines only; no teacher generation, BabyLM training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }


def write_md(summary: dict[str, Any], out_dir: Path) -> None:
    lines = ["# research no-anchor paired-world pilot", "", f"Created UTC: `{summary['created_utc']}`", ""]
    for k, v in summary.items():
        if k in {"baselines", "template_by_entailed_direction"}:
            lines.append(f"- {k}: `{json.dumps(v, sort_keys=True)}`")
        elif k not in {"status", "created_utc", "boundary"}:
            lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Boundary")
    lines.append(summary["boundary"])
    (out_dir / "paired_world_noanchor_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    packets = [transform(p) for p in read_jsonl(args.input)]
    summary = summarize(packets)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "laliga_score_ablated_noanchor_packets.jsonl").open("w", encoding="utf-8") as f:
        for p in packets:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    (args.out_dir / "paired_world_noanchor_pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(summary, args.out_dir)
    print(json.dumps({"status": summary["status"], "out_dir": str(args.out_dir), "n_packets": summary["n_packets"], "shortcut_acc": {k: v.get("accuracy_tie_half") for k, v in summary["baselines"].items()}, "any_digit_rate": summary["any_digit_rate"], "boundary": summary["boundary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
