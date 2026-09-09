#!/usr/bin/env python3
"""research v2: fully crossed shortcut-resistant paired-world pilot.

Repairs the first research CPU pilot: each paired-world family/context is now
materialized under *all* relation templates, so template identity is not a proxy
for whether team_a or team_b is the winner.  The score-ablated mode omits the
score sentence completely rather than saying that the score was withheld.

This remains a construction/probe asset only.  It uses source-attested
LaLiga reversed-win seed pairs; it does not train models, use teachers, run
selected evaluation, SuperGLUE/AoA, upload, or submit.
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
DEFAULT_OUT = ROOT / "experiments/archive/frontier_consolidation/data/paired_world_shortcut_pilot_v2"

TEMPLATES = [
    {"name": "active_winner_first", "winner_first": True, "relation": "{winner} prevailed over {loser} in the match."},
    {"name": "passive_loser_first", "winner_first": False, "relation": "{loser} was beaten by {winner} in the match."},
    {"name": "lost_to_loser_first", "winner_first": False, "relation": "{loser} lost to {winner} in the match."},
    {"name": "not_loser_winner_first", "winner_first": True, "relation": "{winner}, not {loser}, came away as the winning side."},
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
    return len(re.findall(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text))


def entailed_winner_loser(row: dict[str, Any], context_name: str) -> tuple[str, str]:
    for lab in row.get("four_cell_labels", []):
        if lab.get("context") == context_name and lab.get("label") == "ENTAILED":
            hyp = str(lab.get("hypothesis", ""))
            if " defeated " in hyp:
                a, b = hyp.rstrip(".").split(" defeated ", 1)
                return a, b
    raise ValueError(f"No entailed winner/loser for {row.get('family_id')} {context_name}")


def parse_goals(raw: dict[str, Any], winner: str, loser: str) -> tuple[str | None, str | None]:
    home = str(raw.get("HomeTeam", ""))
    away = str(raw.get("AwayTeam", ""))
    hg = str(raw.get("Full Time Home Team Goals", ""))
    ag = str(raw.get("Full Time Away Team Goals", ""))
    if winner == home and loser == away:
        return hg, ag
    if winner == away and loser == home:
        return ag, hg
    return None, None


def make_context(row: dict[str, Any], context_name: str, variant: str, template: dict[str, Any], winner: str, loser: str, raw: dict[str, Any]) -> dict[str, Any]:
    team_a = row["team_a"]
    team_b = row["team_b"]
    date = str(raw.get("Date", "unknown date"))
    season = str(raw.get("Season", "unknown season"))
    # Keep anchors but make them syntactically neutral; future held-out family split
    # must prevent event-ID memorization from being useful.
    neutral_sentence = f"The source record concerns {team_a} and {team_b} in LaLiga season {season}."
    anchor_sentence = f"The match was listed for date {date}."
    relation_sentence = template["relation"].format(winner=winner, loser=loser)
    if variant == "score_visible":
        wg, lg = parse_goals(raw, winner, loser)
        score_sentence = f"The score line was {winner} {wg}, {loser} {lg}." if wg is not None else "The score line was present but not normalized."
        parts = [neutral_sentence, anchor_sentence, score_sentence, relation_sentence]
    elif variant == "score_ablated":
        # Omit the score sentence entirely; the only asymmetric cue is linguistic
        # predicate-argument structure in relation_sentence.
        parts = [neutral_sentence, anchor_sentence, relation_sentence]
    else:
        raise ValueError(variant)
    text = " ".join(parts)
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
        "entailed_direction": "team_a_over_team_b" if winner == team_a and loser == team_b else "team_b_over_team_a",
        "template": template["name"],
        "winner_first_in_relation_sentence": template["winner_first"],
        "text": text,
        "directed_hypotheses": [
            {"hypothesis": f"{team_a} defeated {team_b}.", "label": "ENTAILED" if winner == team_a and loser == team_b else "NOT_ENTAILED"},
            {"hypothesis": f"{team_b} defeated {team_a}.", "label": "ENTAILED" if winner == team_b and loser == team_a else "NOT_ENTAILED"},
        ],
        "symmetric_controls": [
            {"hypothesis": f"{team_a} played {team_b}.", "label": "ENTAILED"},
            {"hypothesis": f"{team_b} played {team_a}.", "label": "ENTAILED"},
        ],
        "raw_record": raw,
    }


def annotate(packet: dict[str, Any]) -> dict[str, Any]:
    text = packet["text"]
    hyps = packet["directed_hypotheses"]
    return {
        "family_id": packet["family_id"],
        "variant": packet["variant"],
        "context_name": packet["context_name"],
        "template": packet["template"],
        "entailed_direction": packet["entailed_direction"],
        "winner_first": packet["winner_first_in_relation_sentence"],
        "contains_exact_query_verb_defeated": "defeated" in [t.lower() for t in norm_tokens(text)],
        "team_a_count": count_name(text, packet["team_a"]),
        "team_b_count": count_name(text, packet["team_b"]),
        "winner_count": count_name(text, packet["winner"]),
        "loser_count": count_name(text, packet["loser"]),
        "team_a_team_b_count_equal": count_name(text, packet["team_a"]) == count_name(text, packet["team_b"]),
        "winner_loser_count_equal": count_name(text, packet["winner"]) == count_name(text, packet["loser"]),
        "hypothesis_bow_collision": token_multiset(hyps[0]["hypothesis"]) == token_multiset(hyps[1]["hypothesis"]),
        "score_digit_sentence_rate_flag": bool(re.search(r"score line was .*\b\d+\b.*\b\d+\b", text.lower())),
        "any_digits": bool(re.search(r"\b\d[\d,/.-]*\b", text)),
        "text_token_count": len(norm_tokens(text)),
    }


def normalize_context_for_lexical_overlap(text: str, team_a: str, team_b: str) -> str:
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


def summarize(packets: list[dict[str, Any]], anns: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in anns:
        by_variant[a["variant"]].append(a)
    by_packet_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for p in packets:
        by_packet_key[(p["variant"], p["family_id"])].append(p)

    variants: dict[str, Any] = {}
    for variant, xs in sorted(by_variant.items()):
        template_direction = defaultdict(Counter)
        for a in xs:
            template_direction[a["template"]][a["entailed_direction"]] += 1
        template_direction_balance = {
            t: {
                "team_a_over_team_b": c.get("team_a_over_team_b", 0),
                "team_b_over_team_a": c.get("team_b_over_team_a", 0),
                "absolute_delta": abs(c.get("team_a_over_team_b", 0) - c.get("team_b_over_team_a", 0)),
            }
            for t, c in sorted(template_direction.items())
        }
        pair_jaccards = []
        for (v, fid), ps in by_packet_key.items():
            if v != variant:
                continue
            # Compare C1/C2 under the same template.
            by_t = defaultdict(list)
            for p in ps:
                by_t[p["template"]].append(p)
            for t, qs in by_t.items():
                if len(qs) != 2:
                    continue
                q1, q2 = sorted(qs, key=lambda z: z["context_name"])
                s1 = normalize_context_for_lexical_overlap(q1["text"], q1["team_a"], q1["team_b"])
                s2 = normalize_context_for_lexical_overlap(q2["text"], q2["team_a"], q2["team_b"])
                pair_jaccards.append(jaccard_counter(token_multiset(s1), token_multiset(s2)))
        variants[variant] = {
            "context_packets": len(xs),
            "families": len({a["family_id"] for a in xs}),
            "template_counts": dict(Counter(a["template"] for a in xs)),
            "template_by_entailed_direction": template_direction_balance,
            "max_template_direction_abs_delta": max((v["absolute_delta"] for v in template_direction_balance.values()), default=None),
            "winner_first_rate": sum(1 for a in xs if a["winner_first"]) / len(xs) if xs else None,
            "team_a_entailed_rate": sum(1 for a in xs if a["entailed_direction"] == "team_a_over_team_b") / len(xs) if xs else None,
            "exact_query_verb_defeated_rate": sum(1 for a in xs if a["contains_exact_query_verb_defeated"]) / len(xs) if xs else None,
            "team_count_equal_rate": sum(1 for a in xs if a["team_a_team_b_count_equal"]) / len(xs) if xs else None,
            "winner_loser_count_equal_rate": sum(1 for a in xs if a["winner_loser_count_equal"]) / len(xs) if xs else None,
            "hypothesis_bow_collision_rate": sum(1 for a in xs if a["hypothesis_bow_collision"]) / len(xs) if xs else None,
            "score_digit_sentence_rate": sum(1 for a in xs if a["score_digit_sentence_rate_flag"]) / len(xs) if xs else None,
            "any_digit_rate": sum(1 for a in xs if a["any_digits"]) / len(xs) if xs else None,
            "mean_text_token_count": sum(a["text_token_count"] for a in xs) / len(xs) if xs else None,
            "mean_same_template_context_pair_bow_jaccard_names_numbers_normalized": sum(pair_jaccards) / len(pair_jaccards) if pair_jaccards else None,
            "min_same_template_context_pair_bow_jaccard_names_numbers_normalized": min(pair_jaccards) if pair_jaccards else None,
        }
    return {
        "status": "PAIRED_WORLD_SHORTCUT_PILOT_V2",
        "created_utc": now(),
        "n_families": len({p["family_id"] for p in packets}),
        "n_context_packets": len(packets),
        "variants": variants,
        "repair_vs_v1": "Every family/context is crossed with all templates; score_ablated omits numeric score sentence entirely. Template identity is no longer a label proxy because each template appears equally with team_a and team_b as winner.",
        "remaining_limits": "Still one relation family (sports outcome) and deterministic templates; future use needs mixed sources/relations, held-out family splits, paraphrase checks, and a bag-of-words/position-light baseline before model training.",
        "boundary": "CPU construction/readout only; no teacher generation, BabyLM training, selected evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }


def write_md(summary: dict[str, Any], out_dir: Path) -> None:
    lines = ["# research paired-world shortcut pilot v2", "", f"Created UTC: `{summary['created_utc']}`", ""]
    lines.append(summary["repair_vs_v1"])
    lines.append("")
    lines.append(f"Families: `{summary['n_families']}`; context packets: `{summary['n_context_packets']}`")
    lines.append("")
    for variant, rec in summary["variants"].items():
        lines.append(f"## {variant}")
        for k, v in rec.items():
            if k == "template_by_entailed_direction":
                lines.append(f"- {k}: `{json.dumps(v, sort_keys=True)}`")
            else:
                lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines.append("## Remaining scientific limits")
    lines.append(summary["remaining_limits"])
    lines.append("")
    lines.append("## Boundary")
    lines.append(summary["boundary"])
    (out_dir / "paired_world_shortcut_pilot_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-families", type=int, default=149)
    args = ap.parse_args()
    rows = read_jsonl(args.input)[: args.max_families]
    packets: list[dict[str, Any]] = []
    for row in rows:
        raws = row.get("raw_records") or []
        if len(raws) != 2:
            continue
        for j, context_name in enumerate(["context_1", "context_2"]):
            winner, loser = entailed_winner_loser(row, context_name)
            raw = raws[j]
            for template in TEMPLATES:
                for variant in ["score_visible", "score_ablated"]:
                    packets.append(make_context(row, context_name, variant, template, winner, loser, raw))
    anns = [annotate(p) for p in packets]
    summary = summarize(packets, anns)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ["score_visible", "score_ablated"]:
        with (args.out_dir / f"laliga_{variant}_packets.jsonl").open("w", encoding="utf-8") as f:
            for p in packets:
                if p["variant"] == variant:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (args.out_dir / "shortcut_annotations.jsonl").open("w", encoding="utf-8") as f:
        for a in anns:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (args.out_dir / "paired_world_shortcut_pilot_v2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(summary, args.out_dir)
    print(json.dumps({
        "status": summary["status"],
        "out_dir": str(args.out_dir),
        "families": summary["n_families"],
        "context_packets": summary["n_context_packets"],
        "variants": summary["variants"],
        "boundary": summary["boundary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
