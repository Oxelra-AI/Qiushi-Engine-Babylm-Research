#!/usr/bin/env python3
"""CPU-only shortcut-resistant paired-world pilot from LaLiga seed pairs.

This script materializes a small source-attested role-flip substrate without
launching BabyLM training.  It is a design/probe asset for the post-common-copy
proposed route: if DeBERTa compact gains really depend on a
positional/binding path, the next experiment should test a joint data-binding
mechanism using independent paired worlds where an asymmetric relation flips.

The pilot produces two versions for the same LaLiga reversed-win families:
  * score_visible: keeps the numeric score line (known shortcut risk).
  * score_ablated: withholds the score line; role must be read from linguistic
    predicate-argument structure, not integer comparison.

The text is deterministic, template-balanced, uses source-attested winner/loser
facts only, avoids the exact query verb "defeated" in contexts, keeps both team
names at matched counts, and emits four-cell hypotheses plus symmetric controls.
No teachers, GPUs, selected evaluation, SuperGLUE/AoA, upload, or submission.
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
DEFAULT_IN = ROOT / "experiments/archive/representation_and_objectives/data/world_pair_source_feasibility/laliga_reversed_win_seed_pairs.jsonl"
DEFAULT_OUT = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_shortcut_pilot"

TEMPLATES = [
    {
        "name": "active_winner_first",
        "winner_first": True,
        "relation": "{winner} prevailed over {loser} in the match.",
    },
    {
        "name": "passive_loser_first",
        "winner_first": False,
        "relation": "{loser} was beaten by {winner} in the match.",
    },
    {
        "name": "lost_to_loser_first",
        "winner_first": False,
        "relation": "{loser} lost to {winner} in the match.",
    },
    {
        "name": "not_loser_winner_first",
        "winner_first": True,
        "relation": "{winner}, not {loser}, came away as the winning side.",
    },
]

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_tokens(s: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(s)]


def token_multiset(s: str) -> Counter[str]:
    return Counter(norm_tokens(s))


def count_name(text: str, name: str) -> int:
    # Use a conservative literal count.  Team names are source strings, not regexes.
    return len(re.findall(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text))


def parse_goals(raw: dict[str, Any], winner: str, loser: str) -> tuple[str, str] | tuple[None, None]:
    home = str(raw.get("HomeTeam", ""))
    away = str(raw.get("AwayTeam", ""))
    hg = str(raw.get("Full Time Home Team Goals", ""))
    ag = str(raw.get("Full Time Away Team Goals", ""))
    if winner == home and loser == away:
        return hg, ag
    if winner == away and loser == home:
        return ag, hg
    return None, None


def entailed_winner_loser(row: dict[str, Any], context_name: str) -> tuple[str, str]:
    for lab in row.get("four_cell_labels", []):
        if lab.get("context") == context_name and lab.get("label") == "ENTAILED":
            hyp = str(lab.get("hypothesis", ""))
            if " defeated " in hyp:
                a, b = hyp.rstrip(".").split(" defeated ", 1)
                return a, b
    raise ValueError(f"No entailed winner/loser for {row.get('family_id')} {context_name}")


def make_context(row: dict[str, Any], context_name: str, variant: str, template: dict[str, Any], winner: str, loser: str, raw: dict[str, Any]) -> dict[str, Any]:
    team_a = row["team_a"]
    team_b = row["team_b"]
    date = str(raw.get("Date", "unknown date"))
    season = str(raw.get("Season", "unknown season"))
    row_index = str(raw.get("row_index", row.get(f"{context_name}_row", "?")))
    relation_sentence = template["relation"].format(winner=winner, loser=loser)
    neutral_sentence = f"The source record concerns {team_a} and {team_b} in LaLiga season {season}."
    anchor_sentence = f"Event anchor: source row {row_index} on date {date}."
    if variant == "score_visible":
        wg, lg = parse_goals(raw, winner, loser)
        if wg is None or lg is None:
            score_sentence = "The numeric score is present in the source but could not be normalized."
        else:
            score_sentence = f"The score line was {winner} {wg}, {loser} {lg}."
    elif variant == "score_ablated":
        score_sentence = "The score line is deliberately withheld in this role-binding version."
    else:
        raise ValueError(variant)
    text = " ".join([neutral_sentence, anchor_sentence, score_sentence, relation_sentence])
    hypotheses = [
        {"hypothesis": f"{team_a} defeated {team_b}.", "label": "ENTAILED" if winner == team_a and loser == team_b else "NOT_ENTAILED"},
        {"hypothesis": f"{team_b} defeated {team_a}.", "label": "ENTAILED" if winner == team_b and loser == team_a else "NOT_ENTAILED"},
    ]
    controls = [
        {"hypothesis": f"{team_a} played {team_b}.", "label": "ENTAILED"},
        {"hypothesis": f"{team_b} played {team_a}.", "label": "ENTAILED"},
    ]
    return {
        "family_id": row["family_id"],
        "variant": variant,
        "context_name": context_name,
        "source_dataset": row.get("source_dataset"),
        "source_sha256": row.get("source_sha256"),
        "team_a": team_a,
        "team_b": team_b,
        "winner": winner,
        "loser": loser,
        "template": template["name"],
        "winner_first_in_relation_sentence": template["winner_first"],
        "text": text,
        "directed_hypotheses": hypotheses,
        "symmetric_controls": controls,
        "raw_record": raw,
    }


def annotate_shortcuts(packet: dict[str, Any]) -> dict[str, Any]:
    text = packet["text"]
    winner = packet["winner"]
    loser = packet["loser"]
    team_a = packet["team_a"]
    team_b = packet["team_b"]
    hyps = packet["directed_hypotheses"]
    return {
        "family_id": packet["family_id"],
        "variant": packet["variant"],
        "context_name": packet["context_name"],
        "template": packet["template"],
        "winner_first": packet["winner_first_in_relation_sentence"],
        "contains_exact_query_verb_defeated": "defeated" in [t.lower() for t in norm_tokens(text)],
        "team_a_count": count_name(text, team_a),
        "team_b_count": count_name(text, team_b),
        "winner_count": count_name(text, winner),
        "loser_count": count_name(text, loser),
        "winner_loser_count_equal": count_name(text, winner) == count_name(text, loser),
        "team_a_team_b_count_equal": count_name(text, team_a) == count_name(text, team_b),
        "goal_score_sentence_visible": packet["variant"] == "score_visible",
        "has_goal_score_digits_after_score_line": bool(re.search(r"score line was .*\b\d+\b.*\b\d+\b", text.lower())),
        "directed_hypothesis_bow_collision": token_multiset(hyps[0]["hypothesis"]) == token_multiset(hyps[1]["hypothesis"]),
        "text_token_count": len(norm_tokens(text)),
    }


def replace_names_for_pair_bow(text: str, team_a: str, team_b: str) -> str:
    # Normalize context anchors/numbers so within-family C1/C2 lexical comparison is not dominated by dates/scores.
    s = re.sub(r"(?<!\w)" + re.escape(team_a) + r"(?!\w)", "TEAM_A", text)
    s = re.sub(r"(?<!\w)" + re.escape(team_b) + r"(?!\w)", "TEAM_B", s)
    s = re.sub(r"\b\d[\d,/.-]*\b", "NUM", s)
    return s


def jaccard_counter(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def summarize(packets: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_ann: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in packets:
        by_variant[p["variant"]].append(p)
    for a in annotations:
        by_variant_ann[a["variant"]].append(a)

    summary: dict[str, Any] = {
        "status": "PAIRED_WORLD_SHORTCUT_PILOT",
        "created_utc": now(),
        "n_context_packets": len(packets),
        "n_families": len({p["family_id"] for p in packets}),
        "variants": {},
        "interpretation": {
            "score_visible": "contains numeric result lines and is retained only as an explicit shortcut-risk control",
            "score_ablated": "withholds score line, avoids exact query verb, balances entity counts, and makes directed hypotheses bag-of-words identical; role must be encoded by predicate-argument order/syntax",
        },
        "boundary": "CPU construction/readout only; no teacher generation, BabyLM training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }
    for variant, anns in sorted(by_variant_ann.items()):
        packets_v = by_variant[variant]
        template_counts = Counter(a["template"] for a in anns)
        # Within each family compare the two contexts after normalizing names and numbers.
        pair_jaccards = []
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in packets_v:
            grouped[p["family_id"]].append(p)
        for fid, ps in grouped.items():
            if len(ps) != 2:
                continue
            p1, p2 = sorted(ps, key=lambda x: x["context_name"])
            t1 = replace_names_for_pair_bow(p1["text"], p1["team_a"], p1["team_b"])
            t2 = replace_names_for_pair_bow(p2["text"], p2["team_a"], p2["team_b"])
            pair_jaccards.append(jaccard_counter(token_multiset(t1), token_multiset(t2)))
        summary["variants"][variant] = {
            "context_packets": len(anns),
            "template_counts": dict(template_counts),
            "winner_first_rate": sum(1 for a in anns if a["winner_first"]) / len(anns) if anns else None,
            "exact_query_verb_defeated_rate": sum(1 for a in anns if a["contains_exact_query_verb_defeated"]) / len(anns) if anns else None,
            "team_count_equal_rate": sum(1 for a in anns if a["team_a_team_b_count_equal"]) / len(anns) if anns else None,
            "winner_loser_count_equal_rate": sum(1 for a in anns if a["winner_loser_count_equal"]) / len(anns) if anns else None,
            "hypothesis_bow_collision_rate": sum(1 for a in anns if a["directed_hypothesis_bow_collision"]) / len(anns) if anns else None,
            "score_digit_sentence_rate": sum(1 for a in anns if a["has_goal_score_digits_after_score_line"]) / len(anns) if anns else None,
            "mean_text_token_count": sum(a["text_token_count"] for a in anns) / len(anns) if anns else None,
            "mean_context_pair_bow_jaccard_names_numbers_normalized": sum(pair_jaccards) / len(pair_jaccards) if pair_jaccards else None,
            "min_context_pair_bow_jaccard_names_numbers_normalized": min(pair_jaccards) if pair_jaccards else None,
        }
    return summary


def write_md(summary: dict[str, Any], out_dir: Path) -> None:
    lines = ["# research paired-world shortcut pilot", "", f"Created UTC: `{summary['created_utc']}`", ""]
    lines.append(f"Families: `{summary['n_families']}`; context packets: `{summary['n_context_packets']}`")
    lines.append("")
    for variant, rec in summary["variants"].items():
        lines.append(f"## {variant}")
        for key, val in rec.items():
            lines.append(f"- {key}: `{val}`")
        lines.append("")
    lines.append("## Scientific use")
    lines.append("This is not a training corpus. It is a CPU construction probe for a later, shortcut-resistant paired-world test after the common-copy DeBERTa adjudication is complete.")
    lines.append("The score-visible version intentionally preserves the numeric shortcut; the score-ablated version removes it and keeps entity counts/directed-query bags matched so order-sensitive role binding is needed.")
    lines.append("")
    lines.append("## Boundary")
    lines.append(summary["boundary"])
    (out_dir / "paired_world_shortcut_pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-families", type=int, default=149)
    args = ap.parse_args()
    rows = read_jsonl(args.input)[: args.max_families]
    packets: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        raws = row.get("raw_records") or []
        if len(raws) != 2:
            continue
        for j, context_name in enumerate(["context_1", "context_2"]):
            winner, loser = entailed_winner_loser(row, context_name)
            # Counterbalance relation sentence across families and contexts.
            template = TEMPLATES[(2 * i + j) % len(TEMPLATES)]
            for variant in ["score_visible", "score_ablated"]:
                packets.append(make_context(row, context_name, variant, template, winner, loser, raws[j]))
    annotations = [annotate_shortcuts(p) for p in packets]
    summary = summarize(packets, annotations)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ["score_visible", "score_ablated"]:
        with (args.out_dir / f"laliga_{variant}_packets.jsonl").open("w", encoding="utf-8") as f:
            for p in packets:
                if p["variant"] == variant:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (args.out_dir / "shortcut_annotations.jsonl").open("w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (args.out_dir / "paired_world_shortcut_pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(summary, args.out_dir)
    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(args.out_dir),
        "families": summary["n_families"],
        "variants": summary["variants"],
        "boundary": summary["boundary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
