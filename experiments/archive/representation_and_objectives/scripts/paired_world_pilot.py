#!/usr/bin/env python3
"""research: Role-equivariant paired-world construction pilot.

CPU-only construction. No BabyLM training, evaluation, upload, or submission.
Builds role-equivariant four-cell families from source-attested competitive
event records (tennis ATP/WTA, BWF badminton, LaLiga football) with:
  - 20 diverse verbalization templates (15 train + 5 held), balanced
    winner-first / loser-first / mixed ordering
  - Score-visible and score-ablated variants
  - Symmetric invariants and context-specific anchors
  - Random A/B participant label assignment
  - Deterministic four-cell labels + NLI-format queries
  - Bag-of-words shortcut-resistance test (CPU sklearn)
"""

import collections, csv, datetime, hashlib, io, json, os, random, sys, zipfile

# ── paths ─────────────────────────────────────────────────────────────────────
STUDY = "experiments/archive/representation_and_objectives"
OUT = os.path.join(STUDY, "workspace", "data", "paired_world_pilot")

TENNIS_ZIP = ("Knowledge/objects/code/"
              "Aneeshers-tennis-sackmann-archive--d300adba5eba--3fe4d6b41983/"
              "originals/Aneeshers-tennis-sackmann-archive-main.zip")
BWF_ZIP = ("Knowledge/objects/code/"
           "SahilMotyar-bwf-match-data--b136c8f057f9--1cf91a16fdb1/"
           "originals/SahilMotyar-bwf-match-data-main.zip")
LALIGA_CSV = os.path.join(STUDY, "staging", "downloads",
                          "1f3a57899024_analyticsfootballdata.csv")

SEED = 251042
N_TENNIS, N_BWF, N_LALIGA = 350, 100, 50
TRAIN_FRAC = 0.80
HELD_TEMPLATE_FRAC = 0.50  # fraction of held families using held-only templates

# ── templates ─────────────────────────────────────────────────────────────────
# (id, pattern, first_mention = "w"|"l"|"m", split)
# {W}=winner, {L}=loser, {R}=round, {T}=tournament, {D}=date
TEMPLATES = [
    # ─ Train: winner-first (5) ─
    (1,  "On {D}, {W} defeated {L} in the {R} of {T}.", "w", "train"),
    (2,  "{W} beat {L} at {T} during the {R} on {D}.", "w", "train"),
    (3,  "{W} won against {L} in the {R} of {T} on {D}.", "w", "train"),
    (4,  "{W} overcame {L} in the {R} at {T} on {D}.", "w", "train"),
    (5,  "{W} proved too strong for {L} in the {R} at {T} on {D}.", "w", "train"),
    # ─ Train: loser-first (5) ─
    (6,  "{L} lost to {W} in the {R} of {T} on {D}.", "l", "train"),
    (7,  "{L} fell to {W} at {T} in the {R} on {D}.", "l", "train"),
    (8,  "{L} was defeated by {W} in the {R} of {T} on {D}.", "l", "train"),
    (9,  "{L} was beaten by {W} at {T} during the {R} on {D}.", "l", "train"),
    (10, "In the {R} at {T} on {D}, {L} was unable to overcome {W}.", "l", "train"),
    # ─ Train: mixed (5) ─
    (11, "At {T} on {D}, the {R} saw {W} triumph over {L}.", "m", "train"),
    (12, "{W} emerged victorious over {L} in the {R} at {T} on {D}.", "m", "train"),
    (13, "During the {R} at {T} on {D}, {W} prevailed against {L}.", "m", "train"),
    (14, "It was {W} who came out on top against {L} in the {R} of {T} on {D}.", "m", "train"),
    (15, "The {R} of {T} on {D} ended with {W} victorious over {L}.", "m", "train"),
    # ─ Held (5) ─
    (16, "{W} edged out {L} in the {R} of {T} on {D}.", "w", "held"),
    (17, "{L} succumbed to {W} at {T} during the {R} on {D}.", "l", "held"),
    (18, "A {R} contest at {T} on {D} resulted in a victory for {W} over {L}.", "m", "held"),
    (19, "{W} claimed the win against {L} in the {R} of {T} on {D}.", "w", "held"),
    (20, "At {T} on {D}, {L} went down to {W} in the {R}.", "l", "held"),
    (18, "A {R} contest at the {T} on {D} resulted in a victory for {W} over {L}.", "m", "held"),
    (19, "{W} claimed the win against {L} in the {R} of the {T} on {D}.", "w", "held"),
    (20, "At the {T} on {D}, {L} went down to {W} in the {R}.", "l", "held"),
]
TMAP = {t[0]: t for t in TEMPLATES}
TRAIN_TID = [t[0] for t in TEMPLATES if t[3] == "train"]
HELD_TID  = [t[0] for t in TEMPLATES if t[3] == "held"]

SPORT_INV = {
    "tennis":    "Both {A} and {B} are professional tennis players.",
    "badminton": "Both {A} and {B} are competitive badminton players.",
    "football":  "Both {A} and {B} are professional football clubs.",
}

# ── helpers ───────────────────────────────────────────────────────────────────
MN = ["January","February","March","April","May","June",
      "July","August","September","October","November","December"]

def fmt_date(raw, src):
    """Parse source-specific date → 'Month D, YYYY'."""
    try:
        if src == "tennis":
            dt = datetime.datetime.strptime(raw[:8], "%Y%m%d")
        elif src == "bwf":
            dt = datetime.datetime.strptime(raw[:10], "%Y-%m-%d")
        elif src == "laliga":
            parts = raw.split("/")
            if len(parts) != 3:
                return raw
            p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
            if p2 >= 1900:                        # M/D/YYYY
                dt = datetime.datetime(p2, p0, p1)
            elif p0 > 12:                         # DD/MM/YY
                dt = datetime.datetime(2000+p2 if p2 < 100 else p2, p1, p0)
            else:                                 # ambiguous, try DD/MM/YY
                yr = 2000+p2 if p2 < 100 else p2
                dt = datetime.datetime(yr, p1, p0)
        else:
            return raw
        return f"{MN[dt.month-1]} {dt.day}, {dt.year}"
    except Exception:
        return raw if raw else "an unrecorded date"

_RD = {"SF":"semifinal","QF":"quarterfinal",
       "R128":"round of 128","R64":"round of 64",
       "R32":"round of 32","R16":"round of 16",
       "RR":"round robin","BR":"bronze medal match","F":"final"}
def norm_round(r):
    if not r or not r.strip():
        return "match"
    u = r.strip().upper()
    if u in _RD:
        return _RD[u]
    for k in sorted(_RD, key=len, reverse=True):
        if u.endswith(k):
            return _RD[k]
    if u in ("1R","R1"): return "first round"
    if u in ("2R","R2"): return "second round"
    if u in ("3R","R3"): return "third round"
    if u in ("4R","R4"): return "fourth round"
    if u.startswith("Q"): return "qualifying round"
    return r.strip().lower() or "match"
    return r.strip().lower() or "match"

# ── loaders ───────────────────────────────────────────────────────────────────
def load_tennis():
    evs = []
    with zipfile.ZipFile(TENNIS_ZIP) as z:
        for nm in sorted(z.namelist()):
            if not nm.endswith(".csv"):
                continue
            if "atp_matches_" not in nm and "wta_matches_" not in nm:
                continue
            try:
                with z.open(nm) as f:
                    for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                        w = (row.get("winner_name") or "").strip()
                        l = (row.get("loser_name") or "").strip()
                        if not w or not l or w == l:
                            continue
                        evs.append({
                            "winner": w, "loser": l,
                            "date_raw": row.get("tourney_date", ""),
                            "tournament": (row.get("tourney_name") or "").strip() or "an ATP/WTA event",
                            "surface": (row.get("surface") or "").strip(),
                            "round": (row.get("round") or "").strip(),
                            "score": (row.get("score") or "").strip(),
                            "sport": "tennis", "src": "tennis",
                        })
            except Exception:
                pass
    return evs

def load_bwf():
    evs = []
    with zipfile.ZipFile(BWF_ZIP) as z:
        for nm in z.namelist():
            if not nm.endswith("matches.csv"):
                continue
            with z.open(nm) as f:
                for row in csv.DictReader(io.TextIOWrapper(f, errors="replace")):
                    wcode = (row.get("winner") or "").strip()
                    t1 = (row.get("team1") or "").strip()
                    t2 = (row.get("team2") or "").strip()
                    if not t1 or not t2 or t1 == t2: continue
                    # winner column is "1" or "2" (team index)
                    if wcode == "1":
                        w, l = t1, t2
                    elif wcode == "2":
                        w, l = t2, t1
                    else:
                        continue
                    evs.append({
                        "winner": w, "loser": l,
                        "date_raw": (row.get("date") or "").strip(),
                        "tournament": (row.get("tournament") or "").strip() or "a BWF event",
                        "discipline": (row.get("discipline") or "").strip(),
                        "round": (row.get("round") or "").strip(),
                        "score": (row.get("score") or "").strip(),
                        "sport": "badminton", "src": "bwf",
                    })
    return evs

def load_laliga():
    evs = []
    with open(LALIGA_CSV, errors="replace") as f:
        for row in csv.DictReader(f):
            h = (row.get("HomeTeam") or "").strip()
            a = (row.get("AwayTeam") or "").strip()
            r = (row.get("Full Time Result") or "").strip()
            if r not in ("H", "A") or not h or not a or h == a:
                continue
            try:
                hg = int(row.get("Full Time Home Team Goals") or 0)
                ag = int(row.get("Full Time Away Team Goals") or 0)
            except ValueError:
                continue
            winner, loser = (h, a) if r == "H" else (a, h)
            sc = f"{hg}-{ag}" if r == "H" else f"{ag}-{hg}"
            evs.append({
                "winner": winner, "loser": loser,
                "date_raw": (row.get("Date") or "").strip(),
                "tournament": f"La Liga {(row.get('Season') or '').strip()}",
                "round": "",
                "score": sc,
                "sport": "football", "src": "laliga",
            })
    return evs

# ── reversed pairs ────────────────────────────────────────────────────────────
def reversed_pairs(evs):
    pm = collections.defaultdict(lambda: {"aw": [], "bw": []})
    for e in evs:
        k = tuple(sorted([e["winner"], e["loser"]]))
        if e["winner"] == k[0]:
            pm[k]["aw"].append(e)
        else:
            pm[k]["bw"].append(e)
    return [{"sa": k[0], "sb": k[1],
             "aw": v["aw"], "bw": v["bw"],
             "sport": v["aw"][0]["sport"]}
            for k, v in pm.items() if v["aw"] and v["bw"]]

# ── verbalize ─────────────────────────────────────────────────────────────────
def verbalize(event, tid, score_ablated=False):
    _, pat, _, _ = TMAP[tid]
    w, l = event["winner"], event["loser"]
    d = fmt_date(event["date_raw"], event["src"])
    r = norm_round(event.get("round", ""))
    t = event.get("tournament", "") or "an international event"
    txt = (pat.replace("{W}", w).replace("{L}", l)
              .replace("{R}", r).replace("{T}", t).replace("{D}", d))
    if not score_ablated and event.get("score"):
        txt += f" The score was {event['score']}."
    return txt

# ── build families ────────────────────────────────────────────────────────────
def build_families(ten_rp, bwf_rp, lla_rp, rng):
    def _sample(rps, n):
        return rng.sample(rps, min(n, len(rps)))

    raw = []
    for rps, n in [(ten_rp, N_TENNIS), (bwf_rp, N_BWF), (lla_rp, N_LALIGA)]:
        for p in _sample(rps, n):
            # pick one event from each direction
            c1_ev = rng.choice(p["aw"])   # sorted_a won
            c2_ev = rng.choice(p["bw"])   # sorted_b won
            # random A/B label assignment
            if rng.random() < 0.5:
                a_name, b_name = p["sa"], p["sb"]
                # sorted_a = A, sorted_b = B → A won in c1, B won in c2
                c1wl, c2wl = "A", "B"
            else:
                a_name, b_name = p["sb"], p["sa"]
                # sorted_b = A, sorted_a = B → B won in c1, A won in c2
                c1wl, c2wl = "B", "A"
            raw.append({"a": a_name, "b": b_name,
                        "c1_ev": c1_ev, "c2_ev": c2_ev,
                        "c1wl": c1wl, "c2wl": c2wl,
                        "sport": p["sport"]})

    rng.shuffle(raw)
    n_train = int(len(raw) * TRAIN_FRAC)

    families = []
    for i, r in enumerate(raw):
        fam_split = "train" if i < n_train else "held"
        # template pool
        if fam_split == "train":
            pool = TRAIN_TID; ts = "train"
        else:
            if rng.random() < HELD_TEMPLATE_FRAC:
                pool = HELD_TID; ts = "held"
            else:
                pool = TRAIN_TID; ts = "train"
        t1, t2 = rng.choice(pool), rng.choice(pool)

        c1_vis = verbalize(r["c1_ev"], t1, score_ablated=False)
        c1_abl = verbalize(r["c1_ev"], t1, score_ablated=True)
        c2_vis = verbalize(r["c2_ev"], t2, score_ablated=False)
        c2_abl = verbalize(r["c2_ev"], t2, score_ablated=True)

        sinv = SPORT_INV[r["sport"]].replace("{A}", r["a"]).replace("{B}", r["b"])
        c1_anch = (f"This match took place at the "
                   f"{r['c1_ev'].get('tournament','an event')} on "
                   f"{fmt_date(r['c1_ev']['date_raw'], r['c1_ev']['src'])}.")
        c2_anch = (f"This match took place at the "
                   f"{r['c2_ev'].get('tournament','an event')} on "
                   f"{fmt_date(r['c2_ev']['date_raw'], r['c2_ev']['src'])}.")

        # four-cell labels
        dab_c1 = (r["c1wl"] == "A")
        dba_c1 = (r["c1wl"] == "B")
        dab_c2 = (r["c2wl"] == "A")
        dba_c2 = (r["c2wl"] == "B")

        # NLI queries (on score-ablated contexts)
        h_ab = f"{r['a']} defeated {r['b']}."
        h_ba = f"{r['b']} defeated {r['a']}."
        nli = [
            {"cx": "c1", "text": c1_abl, "hyp": h_ab,
             "gold": "ENTAILED" if dab_c1 else "NOT_ENTAILED"},
            {"cx": "c1", "text": c1_abl, "hyp": h_ba,
             "gold": "ENTAILED" if dba_c1 else "NOT_ENTAILED"},
            {"cx": "c2", "text": c2_abl, "hyp": h_ab,
             "gold": "ENTAILED" if dab_c2 else "NOT_ENTAILED"},
            {"cx": "c2", "text": c2_abl, "hyp": h_ba,
             "gold": "ENTAILED" if dba_c2 else "NOT_ENTAILED"},
            {"cx": "c1", "text": c1_abl, "hyp": sinv, "gold": "ENTAILED"},
            {"cx": "c2", "text": c2_abl, "hyp": sinv, "gold": "ENTAILED"},
        ]

        c1_wname = r["a"] if r["c1wl"] == "A" else r["b"]
        c1_lname = r["b"] if r["c1wl"] == "A" else r["a"]
        c2_wname = r["a"] if r["c2wl"] == "A" else r["b"]
        c2_lname = r["b"] if r["c2wl"] == "A" else r["a"]

        families.append({
            "family_id": f"f{i:04d}",
            "sport": r["sport"],
            "source_type": r["c1_ev"]["src"],
            "participant_a": r["a"],
            "participant_b": r["b"],
            "context1": {
                "winner_label": r["c1wl"],
                "winner_name": c1_wname, "loser_name": c1_lname,
                "event_raw": {k: v for k, v in r["c1_ev"].items()
                              if k not in ("sport", "src")},
                "date_formatted": fmt_date(r["c1_ev"]["date_raw"], r["c1_ev"]["src"]),
                "round_normalized": norm_round(r["c1_ev"].get("round", "")),
                "text_score_visible": c1_vis,
                "text_score_ablated": c1_abl,
                "template_id": t1,
                "template_split": TMAP[t1][3],
                "template_first_mention": TMAP[t1][2],
            },
            "context2": {
                "winner_label": r["c2wl"],
                "winner_name": c2_wname, "loser_name": c2_lname,
                "event_raw": {k: v for k, v in r["c2_ev"].items()
                              if k not in ("sport", "src")},
                "date_formatted": fmt_date(r["c2_ev"]["date_raw"], r["c2_ev"]["src"]),
                "round_normalized": norm_round(r["c2_ev"].get("round", "")),
                "text_score_visible": c2_vis,
                "text_score_ablated": c2_abl,
                "template_id": t2,
                "template_split": TMAP[t2][3],
                "template_first_mention": TMAP[t2][2],
            },
            "symmetric_invariant": sinv,
            "context1_anchor": c1_anch,
            "context2_anchor": c2_anch,
            "four_cells": {
                "defeated_ab_c1": dab_c1, "defeated_ba_c1": dba_c1,
                "defeated_ab_c2": dab_c2, "defeated_ba_c2": dba_c2,
            },
            "invariant_checks": {
                "both_play_sport_c1": True, "both_play_sport_c2": True,
            },
            "nli_queries": nli,
            "family_split": fam_split,
            "template_split_used": ts,
        })
    return families

# ── BoW shortcut test ─────────────────────────────────────────────────────────
def bow_shortcut_test(families):
    """Test: can bag-of-words predict who won from score-ablated contexts?"""
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, GroupKFold
        import numpy as np
    except ImportError:
        return {"status": "sklearn_not_available"}

    texts, labels, groups = [], [], []
    for i, fam in enumerate(families):
        texts.append(fam["context1"]["text_score_ablated"])
        labels.append(1 if fam["context1"]["winner_label"] == "A" else 0)
        groups.append(i)
        texts.append(fam["context2"]["text_score_ablated"])
        labels.append(1 if fam["context2"]["winner_label"] == "A" else 0)
        groups.append(i)

    vec = CountVectorizer(min_df=2, max_features=5000)
    X = vec.fit_transform(texts)
    y = np.array(labels)
    g = np.array(groups)

    clf = LogisticRegression(max_iter=500, random_state=42)
    # family-level group K-fold so C1/C2 of same family stay together
    gkf = GroupKFold(n_splits=5)
    scores_grp = cross_val_score(clf, X, y,
                                 cv=gkf.split(X, y, g), scoring="accuracy")
    # also plain stratified CV for comparison
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_str = cross_val_score(clf, X, y,
                                 cv=skf.split(X, y), scoring="accuracy")

    # within-family word overlap (Jaccard)
    jaccards = []
    for fam in families:
        w1 = set(fam["context1"]["text_score_ablated"].lower().split())
        w2 = set(fam["context2"]["text_score_ablated"].lower().split())
        union = w1 | w2
        if union:
            jaccards.append(len(w1 & w2) / len(union))

    return {
        "status": "completed",
        "n_contexts": len(texts),
        "n_families": len(families),
        "group_cv_accuracy_mean": round(float(scores_grp.mean()), 4),
        "group_cv_accuracy_std": round(float(scores_grp.std()), 4),
        "group_cv_folds": [round(float(s), 4) for s in scores_grp],
        "stratified_cv_accuracy_mean": round(float(scores_str.mean()), 4),
        "stratified_cv_accuracy_std": round(float(scores_str.std()), 4),
        "stratified_cv_folds": [round(float(s), 4) for s in scores_str],
        "chance_level": 0.5,
        "shortcut_detected": bool(scores_grp.mean() > 0.55),
        "jaccard_mean": round(sum(jaccards)/len(jaccards), 4) if jaccards else None,
        "jaccard_min": round(min(jaccards), 4) if jaccards else None,
        "jaccard_max": round(max(jaccards), 4) if jaccards else None,
        "interpretation": (
            "Group-CV accuracy near 0.50 means BoW features cannot predict "
            "who won from score-ablated text. High within-family Jaccard confirms "
            "C1 and C2 share almost identical word sets; the only discriminating "
            "feature is syntactic role assignment (who is winner vs loser)."
        ),
    }

# ── label-consistency check ───────────────────────────────────────────────────
def validate_families(families):
    errs = 0
    for f in families:
        # four cells must be consistent: exactly 1 ENTAILED per hypothesis across C1/C2
        cells = f["four_cells"]
        if cells["defeated_ab_c1"] == cells["defeated_ab_c2"]:
            errs += 1  # ab should flip
        if cells["defeated_ba_c1"] == cells["defeated_ba_c2"]:
            errs += 1  # ba should flip
        if cells["defeated_ab_c1"] != (not cells["defeated_ba_c1"]):
            errs += 1  # complementary in same context
        if cells["defeated_ab_c2"] != (not cells["defeated_ba_c2"]):
            errs += 1
        # winner labels should differ across contexts
        if f["context1"]["winner_label"] == f["context2"]["winner_label"]:
            errs += 1
    return errs

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    print("Loading tennis...", file=sys.stderr)
    ten = load_tennis()
    print(f"  {len(ten)} events", file=sys.stderr)

    print("Loading BWF...", file=sys.stderr)
    bwf = load_bwf()
    print(f"  {len(bwf)} events", file=sys.stderr)

    print("Loading LaLiga...", file=sys.stderr)
    lla = load_laliga()
    print(f"  {len(lla)} events", file=sys.stderr)

    print("Extracting reversed pairs...", file=sys.stderr)
    ten_rp = reversed_pairs(ten)
    bwf_rp = reversed_pairs(bwf)
    lla_rp = reversed_pairs(lla)
    print(f"  Tennis: {len(ten_rp)}, BWF: {len(bwf_rp)}, LaLiga: {len(lla_rp)}",
          file=sys.stderr)

    print("Building families...", file=sys.stderr)
    fams = build_families(ten_rp, bwf_rp, lla_rp, rng)
    print(f"  {len(fams)} families built", file=sys.stderr)

    # validate label consistency
    errs = validate_families(fams)
    print(f"  Label-consistency errors: {errs}", file=sys.stderr)

    # split and save
    train_fams = [f for f in fams if f["family_split"] == "train"]
    held_fams  = [f for f in fams if f["family_split"] == "held"]

    for fn, data in [("families_train.jsonl", train_fams),
                     ("families_held.jsonl", held_fams)]:
        with open(os.path.join(OUT, fn), "w") as fp:
            for d in data:
                fp.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Running BoW shortcut test...", file=sys.stderr)
    bow = bow_shortcut_test(fams)
    with open(os.path.join(OUT, "shortcut_test_result.json"), "w") as fp:
        json.dump(bow, fp, indent=2)

    # ── summary stats ─────────────────────────────────────────────────────────
    sport_ct = collections.Counter(f["sport"] for f in fams)
    a_c1 = sum(1 for f in fams if f["context1"]["winner_label"] == "A")
    tmpl_ct = (collections.Counter(f["context1"]["template_id"] for f in fams)
             + collections.Counter(f["context2"]["template_id"] for f in fams))
    first_ct = (collections.Counter(f["context1"]["template_first_mention"] for f in fams)
              + collections.Counter(f["context2"]["template_first_mention"] for f in fams))
    ts_ct = collections.Counter(f["template_split_used"] for f in fams)
    names = set()
    for f in fams:
        names.add(f["participant_a"]); names.add(f["participant_b"])

    # first sample (for quick inspection)
    sample = None
    if fams:
        s = fams[0].copy()
        s.pop("nli_queries", None)  # keep summary compact
        sample = s

    summary = {
        "status": "PAIRED_WORLD_PILOT",
        "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_families": len(fams),
        "train_families": len(train_fams),
        "held_families": len(held_fams),
        "label_consistency_errors": errs,
        "sport_distribution": dict(sport_ct),
        "unique_participants": len(names),
        "a_wins_c1_fraction": round(a_c1 / len(fams), 4) if fams else 0,
        "four_cell_items": len(fams) * 4,
        "invariant_items": len(fams) * 2,
        "nli_queries_total": len(fams) * 6,
        "template_usage": dict(tmpl_ct.most_common()),
        "template_first_mention": dict(first_ct),
        "template_split_usage": dict(ts_ct),
        "bow_shortcut_test": bow,
        "sample_family": sample,
        "files": {
            "train": os.path.join(OUT, "families_train.jsonl"),
            "held": os.path.join(OUT, "families_held.jsonl"),
            "shortcut": os.path.join(OUT, "shortcut_test_result.json"),
            "summary": os.path.join(OUT, "pilot_summary.json"),
            "summary_md": os.path.join(OUT, "pilot_summary.md"),
            "teacher_check_design": os.path.join(OUT, "teacher_check_design.md"),
        },
        "no_babylm_training_eval_upload_submission": True,
    }

    with open(os.path.join(OUT, "pilot_summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    # ── markdown summary ──────────────────────────────────────────────────────
    md_lines = [
        "# research — Role-Equivariant Paired-World Pilot\n\n",
        f"**Total families**: {len(fams)} ({len(train_fams)} train, {len(held_fams)} held)  \n",
        f"**Sports**: {dict(sport_ct)}  \n",
        f"**Unique participants**: {len(names)}  \n",
        f"**A-wins-in-C1 fraction**: {a_c1/max(len(fams),1):.4f} (target ~0.50)  \n",
        f"**Label-consistency errors**: {errs}  \n",
        f"**Total labeled items**: {len(fams)*6} "
        f"({len(fams)*4} four-cell + {len(fams)*2} invariant)  \n\n",
        "## Template Balance\n\n",
        f"First-mention distribution across all contexts: {dict(first_ct)}  \n",
        f"(w = winner-first, l = loser-first, m = mixed)  \n\n",
        "## BoW Shortcut Test\n\n",
    ]
    if bow.get("status") == "completed":
        md_lines += [
            f"- Group-CV accuracy: **{bow['group_cv_accuracy_mean']}** "
            f"± {bow['group_cv_accuracy_std']}  \n",
            f"- Stratified-CV accuracy: {bow['stratified_cv_accuracy_mean']} "
            f"± {bow['stratified_cv_accuracy_std']}  \n",
            f"- Chance level: 0.50  \n",
            f"- Shortcut detected: **{bow['shortcut_detected']}**  \n",
            f"- Within-family Jaccard: mean {bow['jaccard_mean']}, "
            f"range [{bow['jaccard_min']}, {bow['jaccard_max']}]  \n\n",
        ]
    else:
        md_lines.append(f"- {bow}\n\n")
    md_lines += [
        "## Sample Context (score-ablated)\n\n",
    ]
    if fams:
        md_lines += [
            f"**C1**: {fams[0]['context1']['text_score_ablated']}  \n",
            f"**C2**: {fams[0]['context2']['text_score_ablated']}  \n",
            f"**Invariant**: {fams[0]['symmetric_invariant']}  \n",
            f"**Four cells**: `{fams[0]['four_cells']}`  \n\n",
        ]
    md_lines += [
        "## Files\n\n",
    ]
    for k, v in summary["files"].items():
        md_lines.append(f"- `{k}`: `{v}`  \n")

    with open(os.path.join(OUT, "pilot_summary.md"), "w") as fp:
        fp.writelines(md_lines)

    # ── teacher-check design ──────────────────────────────────────────────────
    tcheck = [
        "# Teacher-Check Design (research)\n\n",
        "## Purpose\n",
        "Verify that approved teacher models (Qwen3.5-9B, Llama3.1-8B-Instruct)\n",
        "can reliably read role assignment from diverse score-ablated contexts.\n\n",
        "## Procedure\n",
        "1. Sample ~200 families (100 train + 100 held).\n",
        "2. For each family, present 4 score-ablated NLI queries:\n",
        '   - "Context: {c1_ablated}. Hypothesis: {A} defeated {B}. '
        'Answer ENTAILED or NOT_ENTAILED."\n',
        "   - (repeat for B defeated A, and for C2)\n",
        "3. Each query → Qwen3.5-9B and Llama3.1-8B-Instruct independently.\n",
        "4. Score:\n",
        "   - Expected-label accuracy per teacher (target ≥0.95)\n",
        "   - Cross-teacher agreement (target ≥0.95)\n",
        "   - Train-template vs held-template comparison\n\n",
        "## Pass Criteria\n",
        "- Both teachers ≥0.95 expected-label accuracy on score-ablated queries.\n",
        "- Cross-teacher agreement ≥0.95.\n",
        "- No significant accuracy drop on held templates vs train templates.\n",
        "- If any criterion fails, investigate failure modes before teacher-\n",
        "  supported student training.\n\n",
        "## Notes\n",
        "- This is the CHEAPEST reliable test before any model training.\n",
        "- Score-visible queries serve as a verification ceiling (near 1.0).\n",
        "- Do NOT use score-visible-only data as learning signal.\n",
        "- Results feed into the broader route decision: if teachers can read\n",
        "  role assignment from diverse templates, the substrate is viable for\n",
        "  student distillation probes.\n",
    ]
    with open(os.path.join(OUT, "teacher_check_design.md"), "w") as fp:
        fp.writelines(tcheck)

    # print summary to stdout (compact)
    out = {k: v for k, v in summary.items() if k != "sample_family"}
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
