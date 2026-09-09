#!/usr/bin/env python3
"""research: link old EWoK instability to training-text exposure.

This CPU-only analysis asks whether the inherited-tokenizer EWoK rows whose
compact-view effect is seed-specific are explainable by ordinary surface exposure of
ConceptA/ConceptB in the compliant compact_view_reinvest 10M pool or in its compact
changed block.  It does not use evaluation text for training or model selection; it
uses official EWoK fields only to interpret prior evaluation behavior.
"""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

USER_ROOT = Path(".").resolve()
WORKSPACE = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_DENSITY = USER_ROOT / "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard"
TRAIN_10M = A02_DENSITY / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
CHANGED_META = A02_DENSITY / "cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
OLD_ATLAS = WORKSPACE / "data/full_old_ewok_atlas_synthesis/full_old_ewok_atlas_synthesis.json"
OUT_DIR = WORKSPACE / "data/ewok_old_atlas_training_exposure_link"
NOTE = (USER_ROOT / 'research/notes/representation_and_objectives/ewok_old_atlas_training_exposure_link.md')

WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_token(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("_", "-")
    return s


def concept_tokens(s: Any) -> list[str]:
    if s is None:
        return []
    text = str(s).strip().lower()
    if not text:
        return []
    toks = [norm_token(t) for t in WORD_RE.findall(text)]
    return [t for t in toks if t]


def text_tokens(text: str) -> list[str]:
    return [norm_token(t) for t in WORD_RE.findall(text.lower())]


def contains_phrase(tokens: list[str], phrase_toks: list[str]) -> bool:
    if not phrase_toks:
        return False
    n = len(phrase_toks)
    if n == 1:
        return phrase_toks[0] in set(tokens)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i:i+n] == phrase_toks:
            return True
    return False


def safe_mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def safe_median(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def load_atlas_rows() -> list[dict[str, Any]]:
    atlas = json.loads(OLD_ATLAS.read_text(encoding="utf-8"))
    rows = []
    # The synthesis JSON stores examples and summary, not all rows.  Re-read its source CSV
    # through the selection files and original CSV to recover all row fields compactly.
    src_csv = USER_ROOT / atlas["source_csv"]
    grouped: dict[str, dict[str, Any]] = {}
    with src_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            uid = rec["uid"]
            mk = rec["model_key"]
            g = grouped.setdefault(uid, {
                "uid": uid,
                "domain": rec.get("domain"),
                "idx": int(rec.get("idx", -1)),
                "ConceptA": rec.get("ConceptA") or "",
                "ConceptB": rec.get("ConceptB") or "",
                "ContextType": rec.get("ContextType") or "",
                "ContextDiff": rec.get("ContextDiff") or "",
                "TargetDiff": rec.get("TargetDiff") or "",
                "correct": {},
                "margin": {},
            })
            g["correct"][mk] = int(float(rec["correct"]))
            g["margin"][mk] = float(rec["margin_c0_minus_c1"])
    for r in grouped.values():
        if not all(m in r["correct"] for m in ["clean430", "reinv430", "clean431", "reinv431"]):
            continue
        r["pattern"] = "".join(str(r["correct"][m]) for m in ["clean430", "reinv430", "clean431", "reinv431"])
        te430 = r["correct"]["reinv430"] - r["correct"]["clean430"]
        te431 = r["correct"]["reinv431"] - r["correct"]["clean431"]
        r["te430"] = te430
        r["te431"] = te431
        r["accuracy_interaction"] = te431 - te430
        r["margin_effect_430"] = r["margin"]["reinv430"] - r["margin"]["clean430"]
        r["margin_effect_431"] = r["margin"]["reinv431"] - r["margin"]["clean431"]
        r["margin_interaction"] = r["margin_effect_431"] - r["margin_effect_430"]
        rows.append(r)
    return rows


def load_changed_example_ids() -> set[int]:
    ids: set[int] = set()
    with CHANGED_META.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids.add(int(obj["example_id"]))
    return ids


def build_exposure_counts(concepts: dict[str, list[str]], changed_ids: set[int]) -> dict[str, Any]:
    counts = {c: {"all_rows": 0, "all_occurrences": 0, "changed_rows": 0, "changed_occurrences": 0, "heldout_rows": 0, "heldout_occurrences": 0} for c in concepts}
    words_all = words_changed = words_heldout = rows_all = rows_changed = rows_heldout = 0
    # Pre-compile single/multi token concepts.
    concept_sets = {c: set(toks) for c, toks in concepts.items() if len(toks) == 1}
    multi = {c: toks for c, toks in concepts.items() if len(toks) > 1}
    with TRAIN_10M.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", len(text.split())))
            eid = int(obj.get("id", obj.get("example_id", i)))
            is_changed = eid in changed_ids
            toks = text_tokens(text)
            tok_counter = Counter(toks)
            rows_all += 1
            words_all += words
            if is_changed:
                rows_changed += 1
                words_changed += words
            else:
                rows_heldout += 1
                words_heldout += words
            for c, ctoks in concepts.items():
                occ = 0
                present = False
                if len(ctoks) == 1:
                    occ = tok_counter.get(ctoks[0], 0)
                    present = occ > 0
                else:
                    # Consecutive phrase occurrence; cheap enough for EWoK concepts.
                    n = len(ctoks)
                    for j in range(0, len(toks) - n + 1):
                        if toks[j:j+n] == ctoks:
                            occ += 1
                    present = occ > 0
                if present:
                    counts[c]["all_rows"] += 1
                    counts[c]["all_occurrences"] += occ
                    if is_changed:
                        counts[c]["changed_rows"] += 1
                        counts[c]["changed_occurrences"] += occ
                    else:
                        counts[c]["heldout_rows"] += 1
                        counts[c]["heldout_occurrences"] += occ
    return {
        "counts": counts,
        "corpus": {
            "rows_all": rows_all,
            "words_all": words_all,
            "rows_changed": rows_changed,
            "words_changed": words_changed,
            "rows_heldout": rows_heldout,
            "words_heldout": words_heldout,
        },
    }


def enrich_rows(rows: list[dict[str, Any]], exposure: dict[str, Any], concepts: dict[str, list[str]]) -> list[dict[str, Any]]:
    counts = exposure["counts"]
    corpus = exposure["corpus"]
    out = []
    for r in rows:
        ca = r["ConceptA"]
        cb = r["ConceptB"]
        toks_a = concept_tokens(ca)
        toks_b = concept_tokens(cb)
        key_a = " ".join(toks_a)
        key_b = " ".join(toks_b)
        ca_rec = counts.get(key_a, {})
        cb_rec = counts.get(key_b, {})
        pair_rec = {
            "conceptA_key": key_a,
            "conceptB_key": key_b,
            "A_all_occ": ca_rec.get("all_occurrences", 0),
            "B_all_occ": cb_rec.get("all_occurrences", 0),
            "A_changed_occ": ca_rec.get("changed_occurrences", 0),
            "B_changed_occ": cb_rec.get("changed_occurrences", 0),
            "A_all_rows": ca_rec.get("all_rows", 0),
            "B_all_rows": cb_rec.get("all_rows", 0),
            "A_changed_rows": ca_rec.get("changed_rows", 0),
            "B_changed_rows": cb_rec.get("changed_rows", 0),
        }
        pair_rec["min_all_occ"] = min(pair_rec["A_all_occ"], pair_rec["B_all_occ"])
        pair_rec["sum_all_occ"] = pair_rec["A_all_occ"] + pair_rec["B_all_occ"]
        pair_rec["min_changed_occ"] = min(pair_rec["A_changed_occ"], pair_rec["B_changed_occ"])
        pair_rec["sum_changed_occ"] = pair_rec["A_changed_occ"] + pair_rec["B_changed_occ"]
        pair_rec["both_seen_all"] = pair_rec["A_all_occ"] > 0 and pair_rec["B_all_occ"] > 0
        pair_rec["both_seen_changed"] = pair_rec["A_changed_occ"] > 0 and pair_rec["B_changed_occ"] > 0
        rr = dict(r)
        rr["exposure"] = pair_rec
        out.append(rr)
    return out


def summarize_subset(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"name": name, "n": 0}
    exp = [r["exposure"] for r in rows]
    return {
        "name": name,
        "n": n,
        "both_seen_all_frac": sum(1 for e in exp if e["both_seen_all"]) / n,
        "both_seen_changed_frac": sum(1 for e in exp if e["both_seen_changed"]) / n,
        "A_seen_all_frac": sum(1 for e in exp if e["A_all_occ"] > 0) / n,
        "B_seen_all_frac": sum(1 for e in exp if e["B_all_occ"] > 0) / n,
        "min_all_occ_mean": safe_mean([e["min_all_occ"] for e in exp]),
        "min_all_occ_median": safe_median([e["min_all_occ"] for e in exp]),
        "sum_all_occ_mean": safe_mean([e["sum_all_occ"] for e in exp]),
        "min_changed_occ_mean": safe_mean([e["min_changed_occ"] for e in exp]),
        "sum_changed_occ_mean": safe_mean([e["sum_changed_occ"] for e in exp]),
        "zero_both_all_rows": sum(1 for e in exp if not e["both_seen_all"]),
        "zero_both_changed_rows": sum(1 for e in exp if not e["both_seen_changed"]),
        "accuracy_patterns_top": dict(Counter(r["pattern"] for r in rows).most_common(12)),
        "domains_top": dict(Counter(r["domain"] for r in rows).most_common(12)),
    }


def exposure_bins(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    bins = [
        ("0", lambda x: x == 0),
        ("1-2", lambda x: 1 <= x <= 2),
        ("3-9", lambda x: 3 <= x <= 9),
        ("10-49", lambda x: 10 <= x <= 49),
        ("50+", lambda x: x >= 50),
    ]
    out = []
    for label, pred in bins:
        subset = [r for r in rows if pred(r["exposure"][key])]
        if not subset:
            continue
        n = len(subset)
        out.append({
            "bin": label,
            "n": n,
            "negative_interaction_frac": sum(1 for r in subset if r["accuracy_interaction"] < 0) / n,
            "positive_interaction_frac": sum(1 for r in subset if r["accuracy_interaction"] > 0) / n,
            "te430_mean": safe_mean([r["te430"] for r in subset]),
            "te431_mean": safe_mean([r["te431"] for r in subset]),
            "accuracy_interaction_mean": safe_mean([r["accuracy_interaction"] for r in subset]),
            "margin_interaction_mean": safe_mean([r["margin_interaction"] for r in subset]),
        })
    return out


def domain_exposure_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    table = []
    for dom, rs in by.items():
        n = len(rs)
        table.append({
            "domain": dom,
            "n": n,
            "both_seen_all_frac": sum(1 for r in rs if r["exposure"]["both_seen_all"]) / n,
            "both_seen_changed_frac": sum(1 for r in rs if r["exposure"]["both_seen_changed"]) / n,
            "min_all_occ_mean": safe_mean([r["exposure"]["min_all_occ"] for r in rs]),
            "sum_all_occ_mean": safe_mean([r["exposure"]["sum_all_occ"] for r in rs]),
            "accuracy_interaction_mean": safe_mean([r["accuracy_interaction"] for r in rs]),
            "margin_interaction_mean": safe_mean([r["margin_interaction"] for r in rs]),
            "negative_interaction_frac": sum(1 for r in rs if r["accuracy_interaction"] < 0) / n,
            "pattern_0110_frac": sum(1 for r in rs if r["pattern"] == "0110") / n,
        })
    table.sort(key=lambda d: (d["accuracy_interaction_mean"], -d["n"], d["domain"]))
    return table


def write_examples(path: Path, rows: list[dict[str, Any]], limit: int = 80) -> None:
    fields = ["uid", "domain", "idx", "ConceptA", "ConceptB", "ContextType", "ContextDiff", "TargetDiff", "pattern", "te430", "te431", "accuracy_interaction", "margin_interaction", "A_all_occ", "B_all_occ", "min_all_occ", "A_changed_occ", "B_changed_occ", "min_changed_occ"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows[:limit]:
            e = r["exposure"]
            w.writerow({
                "uid": r["uid"], "domain": r["domain"], "idx": r["idx"], "ConceptA": r["ConceptA"], "ConceptB": r["ConceptB"],
                "ContextType": r["ContextType"], "ContextDiff": r["ContextDiff"], "TargetDiff": r["TargetDiff"], "pattern": r["pattern"],
                "te430": r["te430"], "te431": r["te431"], "accuracy_interaction": r["accuracy_interaction"], "margin_interaction": r["margin_interaction"],
                "A_all_occ": e["A_all_occ"], "B_all_occ": e["B_all_occ"], "min_all_occ": e["min_all_occ"],
                "A_changed_occ": e["A_changed_occ"], "B_changed_occ": e["B_changed_occ"], "min_changed_occ": e["min_changed_occ"],
            })


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas_rows = load_atlas_rows()
    concepts: dict[str, list[str]] = {}
    for r in atlas_rows:
        for c in [r["ConceptA"], r["ConceptB"]]:
            toks = concept_tokens(c)
            if toks:
                concepts.setdefault(" ".join(toks), toks)
    changed_ids = load_changed_example_ids()
    exposure = build_exposure_counts(concepts, changed_ids)
    rows = enrich_rows(atlas_rows, exposure, concepts)

    subsets = {
        "all": rows,
        "negative_accuracy_interaction": [r for r in rows if r["accuracy_interaction"] < 0],
        "positive_accuracy_interaction": [r for r in rows if r["accuracy_interaction"] > 0],
        "old_0110_seed430_help_seed431_hurt": [r for r in rows if r["pattern"] == "0110"],
        "old_1001_opposite_seed_pattern": [r for r in rows if r["pattern"] == "1001"],
        "stable_all_correct_1111": [r for r in rows if r["pattern"] == "1111"],
        "stable_all_wrong_0000": [r for r in rows if r["pattern"] == "0000"],
    }
    subset_summary = {name: summarize_subset(rs, name) for name, rs in subsets.items()}

    numeric = {
        "corr_log1p_min_all_occ_vs_accuracy_interaction": pearson([math.log1p(r["exposure"]["min_all_occ"]) for r in rows], [r["accuracy_interaction"] for r in rows]),
        "corr_log1p_sum_all_occ_vs_accuracy_interaction": pearson([math.log1p(r["exposure"]["sum_all_occ"]) for r in rows], [r["accuracy_interaction"] for r in rows]),
        "corr_log1p_min_changed_occ_vs_accuracy_interaction": pearson([math.log1p(r["exposure"]["min_changed_occ"]) for r in rows], [r["accuracy_interaction"] for r in rows]),
        "corr_log1p_sum_changed_occ_vs_accuracy_interaction": pearson([math.log1p(r["exposure"]["sum_changed_occ"]) for r in rows], [r["accuracy_interaction"] for r in rows]),
        "corr_log1p_min_all_occ_vs_margin_interaction": pearson([math.log1p(r["exposure"]["min_all_occ"]) for r in rows], [r["margin_interaction"] for r in rows]),
        "corr_log1p_sum_all_occ_vs_margin_interaction": pearson([math.log1p(r["exposure"]["sum_all_occ"]) for r in rows], [r["margin_interaction"] for r in rows]),
    }
    bins = {
        "min_all_occ": exposure_bins(rows, "min_all_occ"),
        "sum_all_occ": exposure_bins(rows, "sum_all_occ"),
        "min_changed_occ": exposure_bins(rows, "min_changed_occ"),
        "sum_changed_occ": exposure_bins(rows, "sum_changed_occ"),
    }
    domain_table = domain_exposure_table(rows)
    # Examples where old interaction is highly negative despite both concepts occurring many times.
    neg_high_exp = [r for r in rows if r["accuracy_interaction"] < 0 and r["exposure"]["min_all_occ"] >= 10]
    neg_high_exp.sort(key=lambda r: (r["margin_interaction"], -r["exposure"]["min_all_occ"]))
    neg_zero_exp = [r for r in rows if r["accuracy_interaction"] < 0 and not r["exposure"]["both_seen_all"]]
    neg_zero_exp.sort(key=lambda r: (r["margin_interaction"], r["exposure"]["min_all_occ"]))
    write_examples(OUT_DIR / "negative_interaction_high_exposure_examples.csv", neg_high_exp, limit=120)
    write_examples(OUT_DIR / "negative_interaction_zero_pair_exposure_examples.csv", neg_zero_exp, limit=120)

    payload = {
        "status": "EWOK_OLD_ATLAS_TRAINING_EXPOSURE_LINK",
        "created_utc": now_utc(),
        "old_atlas": str(OLD_ATLAS.relative_to(USER_ROOT)),
        "train_10m": str(TRAIN_10M.relative_to(USER_ROOT)),
        "changed_meta": str(CHANGED_META.relative_to(USER_ROOT)),
        "corpus_summary": exposure["corpus"],
        "n_ewok_rows": len(rows),
        "n_unique_concept_keys": len(concepts),
        "subset_summary": subset_summary,
        "correlations": numeric,
        "exposure_bins": bins,
        "domain_exposure_table": domain_table,
        "example_files": {
            "negative_high_exposure": str((OUT_DIR / "negative_interaction_high_exposure_examples.csv").relative_to(USER_ROOT)),
            "negative_zero_pair_exposure": str((OUT_DIR / "negative_interaction_zero_pair_exposure_examples.csv").relative_to(USER_ROOT)),
        },
        "interpretation": [
            "This tests a surface-exposure explanation of old EWoK instability; it does not use official evaluation text as training data or select a submitted checkpoint.",
            "If correlations between concept exposure and treatment-by-seed interaction are small, then relation instability is unlikely to be repaired by naive lexeme-count injection alone.",
            "Rows with negative old interaction despite high concept exposure are especially important because they point toward relational composition/order/context learning rather than mere absence of the words.",
        ],
    }
    out_json = OUT_DIR / "ewok_old_atlas_training_exposure_link.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Write a concise note with the most decision-changing numbers.
    def fmt(v):
        return "None" if v is None else f"{v:.4f}"
    top_neg_domains = "; ".join(f"{d['domain']} int={d['accuracy_interaction_mean']:+.3f} bothSeen={d['both_seen_all_frac']*100:.1f}%" for d in domain_table[:6])
    top_pos_domains = sorted(domain_table, key=lambda d: (-d["accuracy_interaction_mean"], -d["n"]))[:6]
    top_pos = "; ".join(f"{d['domain']} int={d['accuracy_interaction_mean']:+.3f} bothSeen={d['both_seen_all_frac']*100:.1f}%" for d in top_pos_domains)
    note = "\n".join([
        "# research EWoK old-atlas link to compliant reinvest training exposure",
        "",
        "Purpose: use the delivered full old EWoK atlas to test whether the seed-specific old compact-view EWoK effect is mostly ordinary surface exposure of EWoK concept words in the allowed 10M reinvest corpus or changed block. This is interpretation only, not training data construction.",
        "",
        f"Corpus: {exposure['corpus']['rows_all']} rows / {exposure['corpus']['words_all']} words; changed block identified by {len(changed_ids)} example ids / {exposure['corpus']['words_changed']} words.",
        f"Unique EWoK concept keys counted: {len(concepts)} across {len(rows)} official EWoK rows.",
        "",
        "## Exposure summaries by row class",
        f"- All rows: both concepts seen in full 10M for {subset_summary['all']['both_seen_all_frac']*100:.1f}%; both seen in changed block for {subset_summary['all']['both_seen_changed_frac']*100:.1f}%; min-occ median {subset_summary['all']['min_all_occ_median']}.",
        f"- Negative old treatment-by-seed rows: n={subset_summary['negative_accuracy_interaction']['n']}, both seen full {subset_summary['negative_accuracy_interaction']['both_seen_all_frac']*100:.1f}%, both seen changed {subset_summary['negative_accuracy_interaction']['both_seen_changed_frac']*100:.1f}%, min-occ mean {subset_summary['negative_accuracy_interaction']['min_all_occ_mean']:.2f}.",
        f"- Stable all-correct rows: n={subset_summary['stable_all_correct_1111']['n']}, both seen full {subset_summary['stable_all_correct_1111']['both_seen_all_frac']*100:.1f}%, min-occ mean {subset_summary['stable_all_correct_1111']['min_all_occ_mean']:.2f}.",
        "",
        "## Correlations with old treatment-by-seed interaction",
        f"- log1p(min concept occurrences in full 10M) vs accuracy interaction: r={fmt(numeric['corr_log1p_min_all_occ_vs_accuracy_interaction'])}.",
        f"- log1p(sum concept occurrences in full 10M) vs accuracy interaction: r={fmt(numeric['corr_log1p_sum_all_occ_vs_accuracy_interaction'])}.",
        f"- log1p(min changed-block occurrences) vs accuracy interaction: r={fmt(numeric['corr_log1p_min_changed_occ_vs_accuracy_interaction'])}.",
        f"- log1p(sum changed-block occurrences) vs accuracy interaction: r={fmt(numeric['corr_log1p_sum_changed_occ_vs_accuracy_interaction'])}.",
        f"- log1p(min full occurrences) vs margin interaction: r={fmt(numeric['corr_log1p_min_all_occ_vs_margin_interaction'])}.",
        "",
        "## Domain exposure and interaction",
        "- Most negative domains: " + top_neg_domains,
        "- Most positive domains: " + top_pos,
        "",
        "## Reusable outputs",
        f"- JSON: `{out_json.relative_to(USER_ROOT)}`",
        "- High-exposure negative rows: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/negative_interaction_high_exposure_examples.csv`",
        "- Zero-pair-exposure negative rows: `experiments/archive/representation_and_objectives/data/ewok_old_atlas_training_exposure_link/negative_interaction_zero_pair_exposure_examples.csv`",
        "",
        "Scientific reading: if the correlations above are small, the next repair should not be naive evaluation-word counting. The compact-view effect likely depends on relational framing, target geometry, or optimization dynamics; any data repair should be general and learning-scale, using independent source selection and neutral controls rather than benchmark-lexeme injection.",
    ])
    NOTE.write_text(note + "\n", encoding="utf-8")

    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json.relative_to(USER_ROOT)),
        "note": str(NOTE.relative_to(USER_ROOT)),
        "n_rows": len(rows),
        "n_concepts": len(concepts),
        "all_both_seen_full_frac": subset_summary["all"]["both_seen_all_frac"],
        "negative_both_seen_full_frac": subset_summary["negative_accuracy_interaction"]["both_seen_all_frac"],
        "negative_both_seen_changed_frac": subset_summary["negative_accuracy_interaction"]["both_seen_changed_frac"],
        "corr_min_full_vs_acc_interaction": numeric["corr_log1p_min_all_occ_vs_accuracy_interaction"],
        "corr_sum_full_vs_acc_interaction": numeric["corr_log1p_sum_all_occ_vs_accuracy_interaction"],
        "corr_min_changed_vs_acc_interaction": numeric["corr_log1p_min_changed_occ_vs_accuracy_interaction"],
        "top_negative_domains": [(d["domain"], d["accuracy_interaction_mean"], d["both_seen_all_frac"]) for d in domain_table[:5]],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
