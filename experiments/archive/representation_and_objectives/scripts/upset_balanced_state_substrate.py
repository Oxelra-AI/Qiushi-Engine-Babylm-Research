#!/usr/bin/env python3
"""research: upset-balanced, temporally-attested entity-state substrate.

Why this exists
---------------
The research ranking pilot retained only families where the match winner was also
the higher-ranked player in both worlds. The research event-only probe showed the
resulting ranking labels are 100% recoverable from event winner identity, so it
cannot test whether a learner binds an independently recorded state to an entity.

This script rebuilds the substrate so that event outcome and independently
recorded state are dissociated by construction:

1. Reversed match pairs are found as before (A beats B in one record, B beats A
   in another), but each match is classified from the independent weekly ranking
   file as `consistent` (winner ranked better) or `upset` (winner ranked worse).
2. Families are drawn so that, within every context slot, the winner is the
   higher-ranked participant in exactly half the cases. The transparent
   winner->higher-rank parser is then at chance by construction.
3. Every family carries three independently grounded query families:
     - event role   : who defeated whom (from the match record)
     - state at time: who was ranked higher at the match date (from ranking file)
     - later state  : who was ranked higher ~26 weeks after the match date
       (from a separate later ranking snapshot), which supplies genuine temporal
       state persistence/overwrite rather than a relabeled event.
4. Untouched-fact rows assert invariant participant facts that must not change
   across worlds.

All labels come from source records. No teacher model is used and no student is
trained here; this is substrate construction plus deterministic shortcut audits.
"""
from __future__ import annotations

import collections, csv, io, json, random, re, time, zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

STUDY = Path("experiments/archive/representation_and_objectives")
ARCHIVE = Path("data/external/Aneeshers-tennis-sackmann-archive-main.zip")
PREFIX = "Aneeshers-tennis-sackmann-archive-8373358"
OUT_DIR = STUDY / "data/upset_balanced_state_substrate"

ROUND_NAMES = {"F": "final", "SF": "semifinal", "QF": "quarterfinal", "R16": "round of 16",
               "R32": "round of 32", "R64": "round of 64", "R128": "round of 128", "RR": "round robin"}

EVENT_TEMPLATES = [
    ("wf", "On {D}, {W} defeated {L} in the {R} at {T}."),
    ("wf", "{W} beat {L} in the {R} at {T} on {D}."),
    ("lf", "On {D}, {L} lost to {W} in the {R} at {T}."),
    ("lf", "{L} was beaten by {W} in the {R} at {T} on {D}."),
    ("wf", "At {T} on {D}, {W} came through against {L} in the {R}."),
    ("lf", "In the {R} at {T}, {L} went out to {W} on {D}."),
]


def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def write_jsonl(p: Path, rows: list[dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_date(s: str):
    try: return datetime.strptime(str(s)[:8], "%Y%m%d")
    except Exception: return None

def fmt_date(s: str) -> str:
    d = to_date(s)
    return d.strftime("%B %d, %Y") if d else str(s)

def norm_round(r: str) -> str:
    return ROUND_NAMES.get(r, (r or "").lower() or "a match")


def load_rankings(z) -> dict[int, list[tuple[datetime, int]]]:
    files = sorted(n for n in z.namelist() if "atp/atp_rankings_" in n and n.endswith(".csv"))
    out: dict[int, list[tuple[datetime, int]]] = collections.defaultdict(list)
    for fn in files:
        with z.open(fn) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                try:
                    d = to_date(row["ranking_date"])
                    if d is None: continue
                    out[int(row["player"])].append((d, int(row["rank"])))
                except (ValueError, KeyError, TypeError):
                    continue
    for pid in out: out[pid].sort()
    return out


def rank_at(entries: list[tuple[datetime, int]], target: datetime, max_days: int = 14) -> tuple[datetime, int] | None:
    if not entries or target is None: return None
    best = None; best_gap = None
    for d, r in entries:
        gap = abs((d - target).days)
        if best_gap is None or gap < best_gap:
            best_gap, best = gap, (d, r)
        if d > target and best_gap is not None and gap > best_gap:
            break
    if best is None or best_gap is None or best_gap > max_days: return None
    return best


def load_matches(z, years=range(2000, 2025)) -> list[dict[str, Any]]:
    ms: list[dict[str, Any]] = []
    for y in years:
        fn = f"{PREFIX}/atp/atp_matches_{y}.csv"
        try:
            with z.open(fn) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                    try:
                        wid, lid = int(row.get("winner_id") or 0), int(row.get("loser_id") or 0)
                        wn = (row.get("winner_name") or "").strip(); ln = (row.get("loser_name") or "").strip()
                        d = to_date(row.get("tourney_date") or "")
                        if not (wid and lid and wn and ln and d): continue
                        ms.append({"winner_id": wid, "loser_id": lid, "winner_name": wn, "loser_name": ln,
                                   "date": d, "date_raw": (row.get("tourney_date") or "").strip(),
                                   "tournament": (row.get("tourney_name") or "").strip() or "an ATP event",
                                   "round": (row.get("round") or "").strip(), "year": y})
                    except (ValueError, KeyError, TypeError):
                        continue
        except KeyError:
            continue
    return ms


def classify_match(m: dict[str, Any], rk: dict[int, list[tuple[datetime, int]]], later_weeks: int) -> dict[str, Any] | None:
    wr = rank_at(rk.get(m["winner_id"], []), m["date"])
    lr = rank_at(rk.get(m["loser_id"], []), m["date"])
    if wr is None or lr is None or wr[1] == lr[1]: return None
    later = m["date"] + timedelta(weeks=later_weeks)
    wr2 = rank_at(rk.get(m["winner_id"], []), later, max_days=21)
    lr2 = rank_at(rk.get(m["loser_id"], []), later, max_days=21)
    if wr2 is None or lr2 is None or wr2[1] == lr2[1]: return None
    return {**m,
            "winner_rank": wr[1], "loser_rank": lr[1],
            "rank_date": wr[0].strftime("%Y%m%d"), "loser_rank_date": lr[0].strftime("%Y%m%d"),
            "winner_rank_later": wr2[1], "loser_rank_later": lr2[1],
            "later_rank_date": wr2[0].strftime("%Y%m%d"), "later_weeks": later_weeks,
            "winner_higher_at_time": wr[1] < lr[1],
            "winner_higher_later": wr2[1] < lr2[1],
            "state_changed_between_snapshots": (wr[1] < lr[1]) != (wr2[1] < lr2[1])}


def find_reversed_pairs(matches: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for m in matches:
        by_pair[tuple(sorted((m["winner_id"], m["loser_id"])))].append(m)
    out = []
    for key, ms in by_pair.items():
        a, b = key
        wins_a = [m for m in ms if m["winner_id"] == a]
        wins_b = [m for m in ms if m["winner_id"] == b]
        if not wins_a or not wins_b: continue
        for m1 in wins_a[:3]:
            for m2 in wins_b[:3]:
                out.append((m1, m2))
    return out


def event_sentence(m: dict[str, Any], tpl_idx: int) -> tuple[str, str]:
    pol, tpl = EVENT_TEMPLATES[tpl_idx % len(EVENT_TEMPLATES)]
    return pol, tpl.format(W=m["winner_name"], L=m["loser_name"], D=fmt_date(m["date_raw"]),
                           R=norm_round(m["round"]), T=m["tournament"])


def state_sentence(m: dict[str, Any], later: bool) -> str:
    if later:
        return (f"In the official ATP ranking published on {fmt_date(m['later_rank_date'])}, "
                f"{m['winner_name']} was ranked #{m['winner_rank_later']} and "
                f"{m['loser_name']} was ranked #{m['loser_rank_later']}.")
    return (f"In the official ATP ranking published on {fmt_date(m['rank_date'])}, "
            f"{m['winner_name']} was ranked #{m['winner_rank']} and "
            f"{m['loser_name']} was ranked #{m['loser_rank']}.")


def build_family(fid: str, m1: dict[str, Any], m2: dict[str, Any], tpl_idx: int) -> dict[str, Any]:
    A, B = m1["winner_name"], m1["loser_name"]
    worlds = {}
    for key, m in [("context1", m1), ("context2", m2)]:
        pol, ev = event_sentence(m, tpl_idx if key == "context1" else tpl_idx + 3)
        worlds[key] = {
            "event_text": ev, "event_template_polarity": pol,
            "state_at_time_text": state_sentence(m, later=False),
            "state_later_text": state_sentence(m, later=True),
            "winner": m["winner_name"], "loser": m["loser_name"],
            "winner_rank": m["winner_rank"], "loser_rank": m["loser_rank"],
            "winner_rank_later": m["winner_rank_later"], "loser_rank_later": m["loser_rank_later"],
            "rank_date": m["rank_date"], "later_rank_date": m["later_rank_date"],
            "match_date": m["date_raw"], "tournament": m["tournament"], "round": m["round"],
            "winner_higher_at_time": m["winner_higher_at_time"],
            "winner_higher_later": m["winner_higher_later"],
            "state_changed_between_snapshots": m["state_changed_between_snapshots"],
        }
    queries = []
    for wk in ["context1", "context2"]:
        w = worlds[wk]
        higher_at_time = w["winner"] if w["winner_higher_at_time"] else w["loser"]
        higher_later = w["winner"] if w["winner_higher_later"] else w["loser"]
        for x, y in [(A, B), (B, A)]:
            queries.append({"world": wk, "family": "event_role", "requires": "event_text",
                            "hypothesis": f"{x} won the match against {y}.",
                            "gold": "ENTAILED" if x == w["winner"] else "NOT_ENTAILED"})
            queries.append({"world": wk, "family": "state_at_time", "requires": "state_at_time_text",
                            "hypothesis": f"{x} was ranked higher than {y} in the ranking published that week.",
                            "gold": "ENTAILED" if x == higher_at_time else "NOT_ENTAILED"})
            queries.append({"world": wk, "family": "state_later", "requires": "state_later_text",
                            "hypothesis": f"{x} was ranked higher than {y} about {m1['later_weeks']} weeks later.",
                            "gold": "ENTAILED" if x == higher_later else "NOT_ENTAILED"})
        queries.append({"world": wk, "family": "untouched_fact", "requires": "invariant",
                        "hypothesis": f"Both {A} and {B} are professional tennis players.", "gold": "ENTAILED"})
    return {"family_id": fid, "source_type": "upset_balanced_ranking_attested",
            "participant_a": A, "participant_b": B, "event_template_index": tpl_idx,
            **worlds, "nli_queries": queries,
            "stratum": f"{'C' if m1['winner_higher_at_time'] else 'U'}{'C' if m2['winner_higher_at_time'] else 'U'}",
            "temporal_change_worlds": sum(worlds[w]["state_changed_between_snapshots"] for w in ["context1", "context2"])}


def audit(families: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic shortcut audits, no learning."""
    counts = collections.defaultdict(lambda: collections.Counter())
    winner_shortcut = collections.defaultdict(list)
    event_state_agreement = []
    for fam in families:
        for q in fam["nli_queries"]:
            w = fam[q["world"]]
            counts[q["family"]][q["gold"]] += 1
            if q["family"] in {"state_at_time", "state_later"}:
                # transparent parser: assume the match winner is the higher-ranked entity
                first = q["hypothesis"].split(" was ranked higher")[0]
                pred = "ENTAILED" if first == w["winner"] else "NOT_ENTAILED"
                winner_shortcut[q["family"]].append(pred == q["gold"])
            if q["family"] == "state_at_time":
                event_state_agreement.append(w["winner_higher_at_time"])
    strata = collections.Counter(f["stratum"] for f in families)
    temporal = collections.Counter(f["temporal_change_worlds"] for f in families)
    return {
        "label_balance": {k: dict(v) for k, v in counts.items()},
        "winner_to_state_shortcut_accuracy": {k: sum(v)/len(v) for k, v in winner_shortcut.items() if v},
        "fraction_worlds_where_winner_is_higher_ranked": sum(event_state_agreement)/max(1, len(event_state_agreement)),
        "stratum_counts": dict(strata),
        "temporal_change_world_counts": {str(k): v for k, v in sorted(temporal.items())},
        "state_text_contains_no_event_verb": all(
            not re.search(r"\b(defeated|beat|lost to|was beaten|came through|went out)\b", fam[w][k], re.I)
            for fam in families for w in ["context1", "context2"] for k in ["state_at_time_text", "state_later_text"]),
        "event_text_contains_no_rank_number": all(
            not re.search(r"#\d+|\branked\b|\branking\b", fam[w]["event_text"], re.I)
            for fam in families for w in ["context1", "context2"]),
    }


def main() -> None:
    random.seed(254)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ARCHIVE)
    print("loading rankings", flush=True)
    rk = load_rankings(z)
    print(f"  players with rankings: {len(rk)}", flush=True)
    print("loading matches", flush=True)
    ms = load_matches(z)
    print(f"  matches: {len(ms)}", flush=True)
    print("finding reversed pairs", flush=True)
    pairs = find_reversed_pairs(ms)
    print(f"  reversed match-pair candidates: {len(pairs)}", flush=True)

    random.shuffle(pairs)
    strata: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    checked = 0
    for m1, m2 in pairs:
        if all(len(v) >= 220 for v in [strata.get("CC", []), strata.get("CU", []), strata.get("UC", []), strata.get("UU", [])]) and len(strata) == 4:
            break
        checked += 1
        if checked > 60000: break
        c1 = classify_match(m1, rk, later_weeks=26)
        if c1 is None: continue
        c2 = classify_match(m2, rk, later_weeks=26)
        if c2 is None: continue
        key = f"{'C' if c1['winner_higher_at_time'] else 'U'}{'C' if c2['winner_higher_at_time'] else 'U'}"
        if len(strata[key]) < 220:
            strata[key].append((c1, c2))
    avail = {k: len(v) for k, v in sorted(strata.items())}
    print(f"  stratum availability: {avail}", flush=True)

    # Balanced draw: equal numbers of CC/CU/UC/UU so the winner->higher-rank parser is at chance.
    n_per = min([len(strata[k]) for k in ["CC", "CU", "UC", "UU"] if k in strata] or [0])
    families: list[dict[str, Any]] = []
    idx = 0
    for k in ["CC", "CU", "UC", "UU"]:
        for m1, m2 in strata.get(k, [])[:n_per]:
            families.append(build_family(f"ub{idx:04d}", m1, m2, tpl_idx=idx))
            idx += 1
    random.shuffle(families)
    n_train = int(len(families) * 0.75)
    for i, fam in enumerate(families):
        fam["family_split"] = "train" if i < n_train else "held"

    write_jsonl(OUT_DIR / "families_train.jsonl", [f for f in families if f["family_split"] == "train"])
    write_jsonl(OUT_DIR / "families_held.jsonl", [f for f in families if f["family_split"] == "held"])
    aud = audit(families)
    summary = {
        "status": "UPSET_BALANCED_STATE_SUBSTRATE",
        "created_utc": now(),
        "archive": str(ARCHIVE),
        "matches_loaded": len(ms),
        "reversed_pair_candidates": len(pairs),
        "pairs_examined": checked,
        "stratum_availability": avail,
        "families_per_stratum": n_per,
        "families_total": len(families),
        "split": {"train": sum(f["family_split"] == "train" for f in families), "held": sum(f["family_split"] == "held" for f in families)},
        "query_families": ["event_role", "state_at_time", "state_later", "untouched_fact"],
        "audits": aud,
        "scientific_purpose": "Provide a source-attested substrate in which the independently recorded entity state cannot be recovered from the event outcome, and in which a later independent snapshot supplies genuine temporal state change. This is the substrate required to test entity-event-state binding rather than event-outcome parsing.",
        "not_done_here": "No teacher labeling, no student training, no BabyLM training, evaluation, upload, or submission.",
    }
    write_json(OUT_DIR / "substrate_summary.json", summary)
    md = ["# research upset-balanced, temporally-attested state substrate", "",
          f"- Matches loaded: {len(ms)}; reversed pair candidates: {len(pairs)}; pairs examined: {checked}",
          f"- Stratum availability: {avail}; families per stratum: {n_per}; total families: {len(families)}",
          f"- Winner→higher-rank transparent parser accuracy: {aud['winner_to_state_shortcut_accuracy']}",
          f"- Fraction of worlds where the match winner is the higher-ranked participant: {aud['fraction_worlds_where_winner_is_higher_ranked']:.3f}",
          f"- Temporal-change world counts: {aud['temporal_change_world_counts']}",
          f"- State text free of event verbs: {aud['state_text_contains_no_event_verb']}; event text free of rank numbers: {aud['event_text_contains_no_rank_number']}",
          "",
          "The research pilot failed because retained families made the ranking label identical to the event winner. Here the four strata (consistent/upset in each world) are drawn in equal numbers, so event outcome carries no information about the ranking relation, and a separate later snapshot supplies attested state change.",
          "",
          f"Summary JSON: `{OUT_DIR / 'substrate_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/upset_balanced_state_substrate/substrate_summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "families": len(families), "per_stratum": n_per,
                      "winner_shortcut": aud["winner_to_state_shortcut_accuracy"],
                      "summary_json": str(OUT_DIR / "substrate_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
