#!/usr/bin/env python3
"""research: source-attested event-to-state paired-world pilot.

Builds a distinct family from final-match records: if W defeated L in a final,
then W is assigned the champion state and L the runner-up state. This is meant
as a CPU source-construction pilot after the sports outcome relation, not as
student training or BabyLM evaluation.
"""
from __future__ import annotations

import collections, csv, datetime, hashlib, io, json, os, random, re, statistics, time, zipfile
from pathlib import Path
from typing import Any, Iterable

STUDY = Path("experiments/archive/representation_and_objectives")
ROOT = STUDY
OUT = ROOT / "data/event_to_state_final_pilot"
TENNIS_ZIP = Path("data/external/Aneeshers-tennis-sackmann-archive-main.zip")
BWF_ZIP = Path("data/external/SahilMotyar-bwf-match-data-main.zip")
SEED = 252147
MAX_FAMILIES = 300
TRAIN_FRAC = 0.8
MN = ["January","February","March","April","May","June","July","August","September","October","November","December"]

TEMPLATES = [
    (1, "In the final of {T} on {D}, {W} defeated {L}.", "w", "train"),
    (2, "{W} beat {L} in the final at {T} on {D}.", "w", "train"),
    (3, "The {T} final on {D} ended with {W} victorious over {L}.", "m", "train"),
    (4, "{L} lost to {W} in the final of {T} on {D}.", "l", "train"),
    (5, "At {T} on {D}, {L} was beaten by {W} in the final.", "l", "train"),
    (6, "A championship final at {T} on {D} resulted in {W} winning over {L}.", "m", "held"),
    (7, "{W} overcame {L} to take the final at {T} on {D}.", "w", "held"),
    (8, "The final match at {T} on {D} saw {L} finish second to {W}.", "l", "held"),
]
TMAP = {t[0]: (t[1], t[2], t[3]) for t in TEMPLATES}
TRAIN_TIDS = [t[0] for t in TEMPLATES if t[3] == "train"]
HELD_TIDS = [t[0] for t in TEMPLATES if t[3] == "held"]
SPORT_INV = {
    "tennis": "Both {A} and {B} competed as tennis players.",
    "badminton": "Both {A} and {B} competed in badminton.",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fmt_date(raw: str, src: str) -> str:
    try:
        if src == "tennis":
            dt = datetime.datetime.strptime(str(raw)[:8], "%Y%m%d")
        else:
            dt = datetime.datetime.strptime(str(raw)[:10], "%Y-%m-%d")
        return f"{MN[dt.month-1]} {dt.day}, {dt.year}"
    except Exception:
        return raw or "an unrecorded date"


def is_final(raw_round: str) -> bool:
    u = str(raw_round or "").strip().upper()
    return u == "F" or u.endswith(" FINAL") or u == "FINAL"


def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def load_tennis_finals() -> list[dict[str, Any]]:
    evs = []
    with zipfile.ZipFile(TENNIS_ZIP) as z:
        for nm in sorted(z.namelist()):
            if not nm.endswith(".csv") or ("atp_matches_" not in nm and "wta_matches_" not in nm):
                continue
            with z.open(nm) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                    if not is_final(row.get("round")):
                        continue
                    w = norm_spaces(row.get("winner_name"))
                    l = norm_spaces(row.get("loser_name"))
                    if not w or not l or w == l:
                        continue
                    evs.append({
                        "sport": "tennis", "src": "tennis", "winner": w, "loser": l,
                        "date_raw": row.get("tourney_date", ""),
                        "tournament": norm_spaces(row.get("tourney_name")) or "a tennis tournament",
                        "score": norm_spaces(row.get("score")),
                        "surface": norm_spaces(row.get("surface")),
                        "round": "F", "source_file": nm,
                    })
    return evs


def load_bwf_finals() -> list[dict[str, Any]]:
    evs = []
    with zipfile.ZipFile(BWF_ZIP) as z:
        for nm in z.namelist():
            if not nm.endswith("matches.csv"):
                continue
            with z.open(nm) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                    if not is_final(row.get("round")):
                        continue
                    t1 = norm_spaces(row.get("team1"))
                    t2 = norm_spaces(row.get("team2"))
                    wc = norm_spaces(row.get("winner"))
                    if not t1 or not t2 or t1 == t2:
                        continue
                    if wc == "1":
                        w, l = t1, t2
                    elif wc == "2":
                        w, l = t2, t1
                    else:
                        continue
                    evs.append({
                        "sport": "badminton", "src": "bwf", "winner": w, "loser": l,
                        "date_raw": row.get("date", ""),
                        "tournament": norm_spaces(row.get("tournament")) or "a badminton tournament",
                        "score": norm_spaces(row.get("score")),
                        "discipline": norm_spaces(row.get("discipline")),
                        "round": "F", "source_file": nm,
                    })
    return evs


def group_reversed(evs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], dict[tuple[str, str], list[dict[str, Any]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in evs:
        pair = tuple(sorted([e["winner"], e["loser"]]))
        by_pair[pair][(e["winner"], e["loser"])].append(e)
    pairs = []
    for pair, dirs in by_pair.items():
        if len(dirs) < 2:
            continue
        a, b = pair
        if (a, b) in dirs and (b, a) in dirs:
            pairs.append({"p1": a, "p2": b, "a_beats_b_events": dirs[(a,b)], "b_beats_a_events": dirs[(b,a)], "n_events": sum(len(v) for v in dirs.values())})
    pairs.sort(key=lambda x: (-min(len(x["a_beats_b_events"]), len(x["b_beats_a_events"])), x["p1"], x["p2"]))
    return pairs


def render(ctx: dict[str, Any], tid: int) -> str:
    pat = TMAP[tid][0]
    return norm_spaces(pat.format(W=ctx["winner_name"], L=ctx["loser_name"], T=ctx["tournament"], D=ctx["date_formatted"]))


def make_family(i: int, pair: dict[str, Any], sport: str, rng: random.Random, split: str) -> dict[str, Any]:
    e_ab = rng.choice(pair["a_beats_b_events"])
    e_ba = rng.choice(pair["b_beats_a_events"])
    # Randomly map sorted pair entities to A/B labels and randomly choose context order.
    names = [pair["p1"], pair["p2"]]
    rng.shuffle(names)
    A, B = names
    # Make context1/context2 opposite winners.
    events = [e_ab, e_ba]
    rng.shuffle(events)
    tids = TRAIN_TIDS if split == "train" else (HELD_TIDS if rng.random() < 0.5 else TRAIN_TIDS)
    cxs = []
    for ev in events:
        winner_label = "A" if ev["winner"] == A else "B"
        loser_label = "A" if ev["loser"] == A else "B"
        tid = rng.choice(tids)
        ctx = {
            "winner_label": winner_label,
            "loser_label": loser_label,
            "winner_name": ev["winner"],
            "loser_name": ev["loser"],
            "tournament": ev["tournament"],
            "date_formatted": fmt_date(ev.get("date_raw", ""), ev["src"]),
            "event_raw": ev,
            "template_id": tid,
            "template_split": TMAP[tid][2],
            "template_first_mention": TMAP[tid][1],
        }
        ctx["text_event_only"] = render(ctx, tid)
        if ev.get("score"):
            ctx["text_score_visible"] = ctx["text_event_only"] + f" The score was {ev['score']}."
        else:
            ctx["text_score_visible"] = ctx["text_event_only"]
        ctx["state_inference_rule"] = "Because this was a final, the winner is the champion and the loser is the runner-up."
        cxs.append(ctx)
    fam = {
        "family_id": f"s{i:04d}", "sport": sport, "source_type": events[0]["src"],
        "participant_a": A, "participant_b": B,
        "context1": cxs[0], "context2": cxs[1],
        "symmetric_invariant": SPORT_INV[sport].format(A=A, B=B),
        "state_rule": "In these event records, a final-match winner is treated as champion of that event and the final-match loser as runner-up.",
        "family_split": split,
    }
    fam["state_cells"] = {
        "champion_a_c1": cxs[0]["winner_label"] == "A",
        "champion_b_c1": cxs[0]["winner_label"] == "B",
        "runnerup_a_c1": cxs[0]["loser_label"] == "A",
        "runnerup_b_c1": cxs[0]["loser_label"] == "B",
        "champion_a_c2": cxs[1]["winner_label"] == "A",
        "champion_b_c2": cxs[1]["winner_label"] == "B",
        "runnerup_a_c2": cxs[1]["loser_label"] == "A",
        "runnerup_b_c2": cxs[1]["loser_label"] == "B",
    }
    q = []
    for cx_name, ctx in [("c1", cxs[0]), ("c2", cxs[1])]:
        for ent_label, ent_name in [("A", A), ("B", B)]:
            q.append({"cx": cx_name, "state": "champion", "text": ctx["text_event_only"], "hyp": f"{ent_name} became the champion at {ctx['tournament']}.", "gold": "ENTAILED" if ctx["winner_label"] == ent_label else "NOT_ENTAILED"})
            q.append({"cx": cx_name, "state": "runnerup", "text": ctx["text_event_only"], "hyp": f"{ent_name} finished as the runner-up at {ctx['tournament']}.", "gold": "ENTAILED" if ctx["loser_label"] == ent_label else "NOT_ENTAILED"})
        q.append({"cx": cx_name, "state": "invariant", "text": ctx["text_event_only"], "hyp": fam["symmetric_invariant"], "gold": "ENTAILED"})
    fam["nli_queries"] = q
    return fam


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_names(text: str, a: str, b: str) -> str:
    out = str(text)
    for name, marker in sorted([(a, "@@A@@"), (b, "@@B@@")], key=lambda x: len(x[0]), reverse=True):
        out = out.replace(name, marker)
    return norm_spaces(out.replace("@@A@@", "ENTITY_A").replace("@@B@@", "ENTITY_B"))


def sequence_probe(fams: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
    except Exception as e:
        return {"status": "sklearn_unavailable", "error": repr(e)}
    rows = []
    for f in fams:
        A, B = f["participant_a"], f["participant_b"]
        for q in f["nli_queries"]:
            if q["state"] == "invariant":
                continue
            rows.append({
                "family_id": f["family_id"], "split": f["family_split"], "state": q["state"],
                "text": q["text"] + " [SEP] " + q["hyp"],
                "canonical": replace_names(q["text"], A, B) + " [SEP] " + replace_names(q["hyp"], A, B),
                "y": 1 if q["gold"] == "ENTAILED" else 0,
            })
    def cv(view: str) -> dict[str, Any]:
        y = [r["y"] for r in rows]
        groups = [r["family_id"] for r in rows]
        vals = []
        for tr, te in GroupKFold(n_splits=5).split(range(len(rows)), y, groups):
            pipe = make_pipeline(TfidfVectorizer(analyzer="char" if view.endswith("char") else "word", ngram_range=(3,5) if view.endswith("char") else (1,2), lowercase=True), LogisticRegression(max_iter=1000, solver="liblinear", random_state=SEED))
            pipe.fit([rows[i]["canonical" if view.startswith("canonical") else "text"] for i in tr], [y[i] for i in tr])
            pred = pipe.predict([rows[i]["canonical" if view.startswith("canonical") else "text"] for i in te])
            vals.append(sum(int(a==b) for a,b in zip([y[i] for i in te], pred)) / len(te))
        return {"mean": statistics.fmean(vals), "sd": statistics.pstdev(vals), "folds": vals}
    return {"status": "completed", "n_rows": len(rows), "raw_word12": cv("raw_word"), "raw_char35": cv("raw_char"), "canonical_word12": cv("canonical_word"), "canonical_char35": cv("canonical_char")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    tennis = load_tennis_finals()
    bwf = load_bwf_finals()
    tpairs = group_reversed(tennis)
    bpairs = group_reversed(bwf)
    rng.shuffle(tpairs); rng.shuffle(bpairs)
    # Use both sports; if one is sparse, fill from the other.
    target_t = min(220, len(tpairs))
    target_b = min(80, len(bpairs))
    selected = [(p, "tennis") for p in tpairs[:target_t]] + [(p, "badminton") for p in bpairs[:target_b]]
    if len(selected) < MAX_FAMILIES:
        rest = [(p, "tennis") for p in tpairs[target_t:]] + [(p, "badminton") for p in bpairs[target_b:]]
        rng.shuffle(rest)
        selected += rest[:MAX_FAMILIES-len(selected)]
    selected = selected[:MAX_FAMILIES]
    rng.shuffle(selected)
    fams = []
    for i, (pair, sport) in enumerate(selected):
        split = "train" if i < int(TRAIN_FRAC*len(selected)) else "held"
        fams.append(make_family(i, pair, sport, rng, split))
    train = [f for f in fams if f["family_split"] == "train"]
    held = [f for f in fams if f["family_split"] == "held"]
    write_jsonl(OUT / "state_families_train.jsonl", train)
    write_jsonl(OUT / "state_families_held.jsonl", held)
    seq = sequence_probe(fams)
    errors = []
    for f in fams:
        c1, c2 = f["context1"], f["context2"]
        if c1["winner_label"] == c2["winner_label"]:
            errors.append({"family_id": f["family_id"], "error": "winner_label_not_flipped"})
        for cx in ["c1", "c2"]:
            if not (f["state_cells"][f"champion_a_{cx}"] != f["state_cells"][f"champion_b_{cx}"]):
                errors.append({"family_id": f["family_id"], "error": f"champion_not_complementary_{cx}"})
            if not (f["state_cells"][f"runnerup_a_{cx}"] != f["state_cells"][f"runnerup_b_{cx}"]):
                errors.append({"family_id": f["family_id"], "error": f"runnerup_not_complementary_{cx}"})
    summary = {
        "status": "EVENT_TO_STATE_FINAL_PILOT",
        "created_utc": now(),
        "scientific_object": "Paired independent final-match worlds where outcome event assigns champion/runner-up states to entities, with the state assignment flipped across contexts.",
        "source_counts": {
            "tennis_final_events": len(tennis), "tennis_reversed_final_pairs": len(tpairs),
            "bwf_final_events": len(bwf), "bwf_reversed_final_pairs": len(bpairs),
        },
        "families": len(fams), "train_families": len(train), "held_families": len(held),
        "sport_distribution": dict(collections.Counter(f["sport"] for f in fams)),
        "unique_participants": len({x for f in fams for x in [f["participant_a"], f["participant_b"]]}),
        "label_consistency_errors": len(errors), "label_errors_sample": errors[:20],
        "nli_state_queries": sum(sum(1 for q in f["nli_queries"] if q["state"] != "invariant") for f in fams),
        "invariant_queries": sum(sum(1 for q in f["nli_queries"] if q["state"] == "invariant") for f in fams),
        "sequence_probe": seq,
        "files": {"train": str(OUT / "state_families_train.jsonl"), "held": str(OUT / "state_families_held.jsonl")},
        "interpretation_boundary": "The champion/runner-up state is derived from the final-match schema and not separately present as an official title field in the raw CSV. This is a stronger event-to-state construction than defeated(A,B), but still sports-derived and must be teacher-checked and transferred beyond outcome predicates before student training.",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT / "event_to_state_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md = ["# research event-to-state final-match pilot", "", "## Result", "", f"- tennis final events: {len(tennis)}; reversed final pairs: {len(tpairs)}", f"- BWF final events: {len(bwf)}; reversed final pairs: {len(bpairs)}", f"- built families: {len(fams)} ({dict(collections.Counter(f['sport'] for f in fams))})", f"- label errors: {len(errors)}", f"- sequence probe: {seq}", "", "## Scientific meaning", "", "This constructs a distinct source-derived operation: final outcome -> entity state (champion vs runner-up). It does not yet show general data-efficient learning; it gives a next substrate to teacher-check and to use for cross-predicate/event-state transfer if language realization is sound.", "", "## Files", "", f"- train: `{OUT / 'state_families_train.jsonl'}`", f"- held: `{OUT / 'state_families_held.jsonl'}`", f"- summary: `{OUT / 'event_to_state_summary.json'}`"]
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/event_to_state_final_pilot/event_to_state_summary.md')).write_text("\n".join(md)+"\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "source_counts": summary["source_counts"], "families": len(fams), "sport_distribution": summary["sport_distribution"], "label_consistency_errors": len(errors), "sequence_probe": seq, "summary": str(OUT / "event_to_state_summary.json")}, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
