#!/usr/bin/env python3
"""research: event-source inventory for independent paired-world role flips.

CPU-only. Opens candidate sports/event archives in-place and estimates how many
independently-sourced reversed outcome pairs can be generated. This does not create
training data; it informs source-route choice.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("experiments/archive/representation_and_objectives")
OUT_DIR = ROOT / "data/world_pair_source_feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BWF_ZIP = Path("data/external/SahilMotyar-bwf-match-data-main.zip")
TENNIS_ZIP = Path("data/external/Aneeshers-tennis-sackmann-archive-main.zip")
EPL_CSV = Path("Knowledge/objects/datasets/Premier-League--55ddc94e5820--7617e33c5e71/originals/E0.csv")
LALIGA_CSV = Path("data/external/1f3a57899024_analyticsfootballdata.csv")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def csv_rows_from_zip(zp: Path, wanted_suffix: str, max_rows: int | None = None):
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        matches = [n for n in names if n.endswith(wanted_suffix)]
        if not matches:
            return None, []
        name = sorted(matches, key=len)[0]
        data = zf.read(name)
        txt = data.decode("utf-8", errors="replace")
        rdr = csv.DictReader(io.StringIO(txt))
        rows = []
        for i, r in enumerate(rdr, start=1):
            r["_row_index"] = i
            rows.append(r)
            if max_rows and i >= max_rows:
                break
        return name, rows


def reversed_pair_stats(events, a_key="winner", b_key="loser", category_key=None, cap_pairings=2):
    # event entries must contain winner/loser and context fields.
    by_directed = defaultdict(list)
    entities = set()
    for e in events:
        a, b = norm(e.get(a_key, "")), norm(e.get(b_key, ""))
        if not a or not b or a == b:
            continue
        by_directed[(a, b)].append(e)
        entities.add(a); entities.add(b)
    unordered = defaultdict(list)
    for (a,b), vals in by_directed.items():
        unordered[tuple(sorted([a,b]))].extend(vals)
    rev_pairs = []
    rev_unordered = []
    for (a,b) in unordered.keys():
        d1 = by_directed.get((a,b), [])
        d2 = by_directed.get((b,a), [])
        if d1 and d2:
            rev_unordered.append((a,b,len(d1),len(d2)))
            for e1 in d1[:cap_pairings]:
                for e2 in d2[:cap_pairings]:
                    rev_pairs.append((e1,e2))
    cat_counts = Counter()
    if category_key:
        for e in events:
            cat_counts[e.get(category_key, "")] += 1
    return {
        "events": len(events),
        "entities": len(entities),
        "directed_pairs": len(by_directed),
        "unordered_pairs": len(unordered),
        "unordered_pairs_with_reversed_wins": len(rev_unordered),
        "capped_reversed_pair_instances": len(rev_pairs),
        "top_categories": cat_counts.most_common(12),
        "top_pair_multiplicities": sorted(rev_unordered, key=lambda x: x[2]+x[3], reverse=True)[:12],
        "sample_reversed_pairs": rev_pairs[:5],
    }


def inv_bwf():
    name, rows = csv_rows_from_zip(BWF_ZIP, "matches.csv")
    events = []
    for r in rows:
        t1, t2 = norm(r.get("team1")), norm(r.get("team2"))
        w = str(r.get("winner", "")).strip()
        if w == "1": winner, loser = t1, t2
        elif w == "2": winner, loser = t2, t1
        else: continue
        if winner and loser and winner != loser:
            events.append({
                "winner": winner,
                "loser": loser,
                "date": r.get("date", ""),
                "discipline": r.get("discipline", ""),
                "tournament": r.get("tournament", ""),
                "round": r.get("round", ""),
                "score": r.get("score", ""),
                "team1": t1,
                "team2": t2,
                "row_index": r["_row_index"],
            })
    stats = reversed_pair_stats(events, category_key="discipline")
    # Keep examples compact.
    samples = []
    for e1, e2 in stats.pop("sample_reversed_pairs"):
        samples.append({"context1_event": e1, "context2_event": e2})
    stats["sample_reversed_pairs"] = samples
    return {
        "source": "SahilMotyar/bwf-match-data",
        "citation": "\\cite{sahilmotyar2026sahilmotyar}",
        "archive": str(BWF_ZIP),
        "archive_sha256": sha(BWF_ZIP),
        "csv_member": name,
        "license_note": "Repository object did not expose a standard license string; README says see licensing/attribution and not official BWF product.",
        **stats,
    }


def inv_tennis():
    # Read ATP/WTA yearly match CSVs. Determine winner/loser columns.
    events = []
    members_used = []
    with zipfile.ZipFile(TENNIS_ZIP) as zf:
        names = zf.namelist()
        match_files = [n for n in names if re.search(r"/(atp|wta)/atp_matches_\d{4}\.csv$", n) or re.search(r"/(atp|wta)/wta_matches_\d{4}\.csv$", n)]
        # If archive prefix changes, accept any basename.
        if not match_files:
            match_files = [n for n in names if re.search(r"/(atp|wta)_matches_\d{4}\.csv$", n)]
        for n in sorted(match_files):
            base = Path(n).name
            if not ("matches_" in base):
                continue
            # Sample all, still cheap.
            try:
                txt = zf.read(n).decode("utf-8", errors="replace")
            except Exception:
                continue
            rdr = csv.DictReader(io.StringIO(txt))
            if not rdr.fieldnames:
                continue
            if not {"winner_name", "loser_name"}.issubset(set(rdr.fieldnames)):
                continue
            members_used.append(n)
            for i, r in enumerate(rdr, start=1):
                w, l = norm(r.get("winner_name")), norm(r.get("loser_name"))
                if not w or not l or w == l:
                    continue
                tourney_date = r.get("tourney_date", "")
                year = base[-8:-4] if base.endswith(".csv") else ""
                events.append({
                    "winner": w,
                    "loser": l,
                    "date": tourney_date,
                    "year": year,
                    "tournament": r.get("tourney_name", ""),
                    "surface": r.get("surface", ""),
                    "round": r.get("round", ""),
                    "score": r.get("score", ""),
                    "row_file": n,
                    "winner_rank": r.get("winner_rank", ""),
                    "loser_rank": r.get("loser_rank", ""),
                })
    stats = reversed_pair_stats(events, category_key="surface")
    samples = []
    for e1, e2 in stats.pop("sample_reversed_pairs"):
        samples.append({"context1_event": e1, "context2_event": e2})
    stats["sample_reversed_pairs"] = samples
    return {
        "source": "Aneeshers/tennis-sackmann-archive mirror of Jeff Sackmann ATP/WTA",
        "citation": "\\cite{aneeshers2026aneeshers}",
        "archive": str(TENNIS_ZIP),
        "archive_sha256": sha(TENNIS_ZIP),
        "members_used_count": len(members_used),
        "members_used_sample": members_used[:10],
        "license_note": "CC BY-NC-SA 4.0 in source object; useful for research probes but noncommercial/sharealike matters for redistribution.",
        **stats,
    }


def inv_epl_single():
    if not EPL_CSV.exists():
        return {"exists": False, "path": str(EPL_CSV)}
    rows = list(csv.DictReader(EPL_CSV.open(newline="", encoding="utf-8", errors="replace")))
    events = []
    for i, r in enumerate(rows, start=1):
        ftr = r.get("FTR", "")
        home, away = norm(r.get("HomeTeam")), norm(r.get("AwayTeam"))
        if ftr == "H": w,l = home, away
        elif ftr == "A": w,l = away, home
        else: continue
        events.append({"winner": w, "loser": l, "date": r.get("Date",""), "round":"", "tournament":"Premier League 2012-13", "score": f"{r.get('FTHG','')}-{r.get('FTAG','')}", "row_index": i})
    stats = reversed_pair_stats(events)
    samples = []
    for e1, e2 in stats.pop("sample_reversed_pairs"):
        samples.append({"context1_event": e1, "context2_event": e2})
    stats["sample_reversed_pairs"] = samples
    return {"source":"football-data.co.uk Premier League E0 2012-13 single season", "citation":"\\cite{data2016premier}", "path":str(EPL_CSV), "sha256":sha(EPL_CSV), **stats}


def main():
    out = {
        "status": "EVENT_PAIR_SOURCE_INVENTORY",
        "bwf": inv_bwf(),
        "tennis": inv_tennis(),
        "premier_league_single_season": inv_epl_single(),
        "interpretation": {
            "event_record_strength": "winner/loser records create deterministic four-cell flips from independent event worlds and avoid teacher-generated facts",
            "event_record_limit": "competitive outcomes remain one relation family; score/result fields can give numeric shortcuts unless contexts are written or ablated to require role parsing",
            "best_use_next": "build a CPU source-construction pilot that mixes BWF/Tennis/football records with templated and score-ablated contexts, then test teacher/language realization and shortcut probes before any H100 model training",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    path = OUT_DIR / "event_pair_source_inventory.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "bwf_reversed_pairs": out["bwf"].get("unordered_pairs_with_reversed_wins"),
        "tennis_reversed_pairs": out["tennis"].get("unordered_pairs_with_reversed_wins"),
        "epl_single_reversed_pairs": out["premier_league_single_season"].get("unordered_pairs_with_reversed_wins"),
        "out": str(path),
    }, indent=2))

if __name__ == "__main__":
    main()
