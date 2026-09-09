#!/usr/bin/env python3
"""research: feasibility of independently-sourced role-reassignment world pairs.

This is CPU-only Explore work. It analyzes a downloaded structured source, not a BabyLM
training corpus. The scientific question is whether we can obtain context pairs where an
asymmetric relation is true in one independent world and its role-reversed mate is true in
a second independent world, with deterministic source-grounded labels.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
CSV_PATH = Path("data/external/1f3a57899024_analyticsfootballdata.csv")
OUT_DIR = ROOT / "data/world_pair_source_feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_team(x: str) -> str:
    return " ".join((x or "").strip().split())


def parse_int(x: str):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def winner(row: dict):
    hg = parse_int(row.get("Full Time Home Team Goals", ""))
    ag = parse_int(row.get("Full Time Away Team Goals", ""))
    if hg is None or ag is None:
        return None
    if hg == ag:
        return None
    home = norm_team(row.get("HomeTeam", ""))
    away = norm_team(row.get("AwayTeam", ""))
    if not home or not away:
        return None
    return home if hg > ag else away


def loser(row: dict):
    w = winner(row)
    if w is None:
        return None
    home = norm_team(row.get("HomeTeam", ""))
    away = norm_team(row.get("AwayTeam", ""))
    return away if w == home else home


def clean_date(x: str) -> str:
    x = (x or "").strip()
    for fmt in ["%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(x, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return x


def row_context(row: dict) -> str:
    home = norm_team(row.get("HomeTeam", ""))
    away = norm_team(row.get("AwayTeam", ""))
    date = clean_date(row.get("Date", ""))
    season = (row.get("Season", "") or "").strip()
    hg = parse_int(row.get("Full Time Home Team Goals", ""))
    ag = parse_int(row.get("Full Time Away Team Goals", ""))
    att = (row.get("Game attendance", "") or "").strip()
    base = f"On {date} in LaLiga season {season}, {home} hosted {away}. The final score was {home} {hg}, {away} {ag}."
    if att:
        base += f" The recorded attendance was {att}."
    if hg is not None and ag is not None:
        if hg > ag:
            base += f" Therefore, {home} defeated {away}, and {away} lost to {home}."
        elif ag > hg:
            base += f" Therefore, {away} defeated {home}, and {home} lost to {away}."
        else:
            base += " Therefore, the match was a draw and neither team defeated the other."
    return base


def statement_win(a: str, b: str) -> str:
    return f"{a} defeated {b}."


def statement_played(a: str, b: str) -> str:
    return f"{a} played {b}."


def main():
    rows = []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr, start=1):
            row["_row_index"] = i
            row["_home"] = norm_team(row.get("HomeTeam", ""))
            row["_away"] = norm_team(row.get("AwayTeam", ""))
            row["_winner"] = winner(row)
            row["_loser"] = loser(row)
            row["_date_norm"] = clean_date(row.get("Date", ""))
            rows.append(row)

    non_draw = [r for r in rows if r["_winner"]]
    draws = [r for r in rows if not r["_winner"]]
    # Directed outcome rows by ordered winner->loser.
    by_directed = defaultdict(list)
    by_unordered = defaultdict(list)
    for r in non_draw:
        a, b = r["_winner"], r["_loser"]
        by_directed[(a, b)].append(r)
        by_unordered[tuple(sorted([a, b]))].append(r)

    candidates = []
    for pair, prs in by_unordered.items():
        t1, t2 = pair
        d12 = by_directed.get((t1, t2), [])
        d21 = by_directed.get((t2, t1), [])
        if not d12 or not d21:
            continue
        # Build up to 3 pairings per unordered team pair, preferring different dates and close lexical form.
        for i, r12 in enumerate(d12[:3]):
            for j, r21 in enumerate(d21[:3]):
                if r12["_row_index"] == r21["_row_index"]:
                    continue
                a, b = t1, t2
                c1 = row_context(r12)
                c2 = row_context(r21)
                fam = {
                    "family_id": f"laliga_{len(candidates):05d}",
                    "source_dataset": "Zenodo 10.5281/zenodo.18861500 analyticsfootballdata.csv",
                    "source_sha256": file_sha256(CSV_PATH),
                    "team_a": a,
                    "team_b": b,
                    "context_1_row": r12["_row_index"],
                    "context_2_row": r21["_row_index"],
                    "context_1": c1,
                    "context_2": c2,
                    "flip_relation": "defeated",
                    "four_cell_labels": [
                        {"context": "context_1", "hypothesis": statement_win(a, b), "label": "ENTAILED"},
                        {"context": "context_1", "hypothesis": statement_win(b, a), "label": "NOT_ENTAILED"},
                        {"context": "context_2", "hypothesis": statement_win(a, b), "label": "NOT_ENTAILED"},
                        {"context": "context_2", "hypothesis": statement_win(b, a), "label": "ENTAILED"},
                    ],
                    "untouched_or_symmetric_facts": [
                        {"context": "context_1", "hypothesis": statement_played(a, b), "label": "ENTAILED"},
                        {"context": "context_2", "hypothesis": statement_played(a, b), "label": "ENTAILED"},
                    ],
                    "raw_records": [
                        {k: r12.get(k, "") for k in ["Date", "Season", "HomeTeam", "AwayTeam", "Full Time Home Team Goals", "Full Time Away Team Goals", "Full Time Result", "Game attendance"]},
                        {k: r21.get(k, "") for k in ["Date", "Season", "HomeTeam", "AwayTeam", "Full Time Home Team Goals", "Full Time Away Team Goals", "Full Time Result", "Game attendance"]},
                    ],
                }
                candidates.append(fam)

    # Deduplicate to one representative per unordered team pair for seed file.
    seed = []
    seen = set()
    for fam in candidates:
        key = tuple(sorted([fam["team_a"], fam["team_b"]]))
        if key in seen:
            continue
        seen.add(key)
        seed.append(fam)

    teams = sorted({r["_home"] for r in rows if r["_home"]} | {r["_away"] for r in rows if r["_away"]})
    season_counts = Counter(r.get("Season", "") for r in rows)
    summary = {
        "status": "WORLD_PAIR_SOURCE_FEASIBILITY",
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_question": "Can an existing source provide independent paired worlds where an asymmetric event relation flips roles while symmetric/untouched facts persist?",
        "source": {
            "path": str(CSV_PATH),
            "sha256": file_sha256(CSV_PATH),
            "citation": "\\cite{torrecilla2026what}",
            "license_from_source": "CC-BY-4.0",
            "description": "LaLiga match-level records with home/away teams, full-time scores, attendance, and season.",
        },
        "row_counts": {
            "total_rows": len(rows),
            "non_draw_rows": len(non_draw),
            "draw_rows": len(draws),
            "teams": len(teams),
            "seasons": dict(season_counts),
            "directed_winner_loser_pairs": len(by_directed),
            "unordered_team_pairs_with_any_result": len(by_unordered),
            "unordered_team_pairs_with_reversed_wins": len({tuple(sorted([f["team_a"], f["team_b"]])) for f in candidates}),
            "candidate_world_pair_instances_capped": len(candidates),
            "seed_one_per_unordered_pair": len(seed),
        },
        "role_signal_properties": {
            "deterministic_labels_from_source": True,
            "teachers_needed_for_fact_generation": False,
            "teachers_needed_for_language_realization_check_only": True,
            "independent_worlds_not_same_world_paraphrases": True,
            "asymmetric_four_cell_flip": "A defeated B true in context_1 and false in context_2; B defeated A false in context_1 and true in context_2.",
            "untouched_fact_type_available": "both teams played each other; match date/season/home-away/attendance are context-specific anchors, not invariant across pair.",
            "main_weakness": "sports outcome relation is narrow and score-table-generated, so it tests directed role assignment but not rich event-state/condition/outcome semantics by itself.",
        },
        "output_files": {
            "all_candidate_pairs": str(OUT_DIR / "laliga_reversed_win_world_pairs.jsonl"),
            "seed_one_per_pair": str(OUT_DIR / "laliga_reversed_win_seed_pairs.jsonl"),
            "summary": str(OUT_DIR / "world_pair_feasibility_summary.json"),
            "note": str((ROOT.parents[2] / 'research/notes/representation_and_objectives/role_equivariance_world_pair_route.md')),
        },
        "no_babylm_training_eval_upload_submission": True,
    }

    with (OUT_DIR / "laliga_reversed_win_world_pairs.jsonl").open("w", encoding="utf-8") as f:
        for fam in candidates:
            f.write(json.dumps(fam, ensure_ascii=False) + "\n")
    with (OUT_DIR / "laliga_reversed_win_seed_pairs.jsonl").open("w", encoding="utf-8") as f:
        for fam in seed:
            f.write(json.dumps(fam, ensure_ascii=False) + "\n")
    (OUT_DIR / "world_pair_feasibility_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
