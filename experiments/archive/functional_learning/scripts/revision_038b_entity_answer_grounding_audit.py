#!/usr/bin/env python3
"""Step038b: entity-answer grounding audit for research rows.

Raw answer overlap is not enough: the answer phrase must be associated with the
right entity in the raw source. This script adds a cheap, inspectable proximity
probe over exact raw spans. It is not an entailment validator, but it finds rows
where the selected answer is absent as a span, or where the competing source
answer is closer to the entity than the assigned answer.
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter
from typing import Any

ROOT = pathlib.Path("experiments/archive/functional_learning")
PAIRS_PATH = pathlib.Path("experiments/archive/relation_learning/data/contrastive_binding_validated_strict/accepted_contrastive_binding_pairs_strict_pilot512.jsonl")
SEM_AUDIT_PATH = ROOT / "data/semantic_foundation_audit/semantic_audit_pairs.jsonl"
OUT_DIR = ROOT / "data/entity_answer_grounding_audit"
NOTE_APPEND_PATH = (ROOT.parents[2] / 'research/notes/functional_learning/semantic_foundation_for_natural_rows.md')


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_ws(x: str) -> str:
    return " ".join(str(x or "").replace("\u00a0", " ").split())


def spans(text: str, sub: str) -> list[tuple[int, int]]:
    text_l = text.lower()
    sub_l = sub.lower().strip()
    out: list[tuple[int, int]] = []
    if not sub_l:
        return out
    start = 0
    while True:
        i = text_l.find(sub_l, start)
        if i < 0:
            break
        out.append((i, i + len(sub_l)))
        start = i + max(1, len(sub_l))
    return out


def token_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    # Fallback: all content-token spans for phrases that are not exact substrings.
    # We only use this as diagnostic evidence, never as acceptance proof.
    toks = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", phrase.lower())
    text_l = text.lower()
    out = []
    for tok in toks:
        if len(tok) <= 2:
            continue
        out.extend(spans(text_l, tok))
    return out


def interval_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[1] <= b[0]:
        return b[0] - a[1]
    if b[1] <= a[0]:
        return a[0] - b[1]
    return 0


def min_distance(spans_a: list[tuple[int, int]], spans_b: list[tuple[int, int]]) -> float:
    if not spans_a or not spans_b:
        return math.inf
    return min(interval_distance(a, b) for a in spans_a for b in spans_b)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_dist(x: float) -> str:
    return "inf" if math.isinf(x) else str(int(x))


def audit_pair(p: dict[str, Any], base: dict[str, Any] | None) -> dict[str, Any]:
    source = p["source_sentence"]
    target = p["target_entity"]
    distractor = p["distractor_entity"]
    t_ans = p["target_source_answer"]
    d_ans = p["distractor_source_answer"]
    ent_t = spans(source, target)
    ent_d = spans(source, distractor)
    t_exact = spans(source, t_ans)
    d_exact = spans(source, d_ans)
    t_tok = token_spans(source, t_ans)
    d_tok = token_spans(source, d_ans)

    row: dict[str, Any] = {
        "pair_id": p["pair_id"],
        "source": p.get("source"),
        "role_position_relation": p.get("role_position_relation"),
        "source_sentence": source,
        "target_entity": target,
        "distractor_entity": distractor,
        "target_source_answer": t_ans,
        "distractor_source_answer": d_ans,
        "target_entity_spans": ent_t,
        "distractor_entity_spans": ent_d,
        "target_answer_exact_spans": t_exact,
        "distractor_answer_exact_spans": d_exact,
        "target_answer_token_spans": t_tok,
        "distractor_answer_token_spans": d_tok,
    }
    # Exact-span distances.
    row["target_ans_to_target_dist"] = min_distance(t_exact, ent_t)
    row["target_ans_to_distractor_dist"] = min_distance(t_exact, ent_d)
    row["distractor_ans_to_distractor_dist"] = min_distance(d_exact, ent_d)
    row["distractor_ans_to_target_dist"] = min_distance(d_exact, ent_t)
    # Does the other source answer look closer to the entity than its assigned answer?
    row["target_entity_competing_source_answer_dist"] = min_distance(d_exact, ent_t)
    row["distractor_entity_competing_source_answer_dist"] = min_distance(t_exact, ent_d)
    row["target_assigned_exact_closer_than_competing"] = row["target_ans_to_target_dist"] < row["target_entity_competing_source_answer_dist"]
    row["distractor_assigned_exact_closer_than_competing"] = row["distractor_ans_to_distractor_dist"] < row["distractor_entity_competing_source_answer_dist"]

    # Token-overlap fallback distances.
    row["target_ans_token_to_target_dist"] = min_distance(t_tok, ent_t)
    row["distractor_ans_token_to_distractor_dist"] = min_distance(d_tok, ent_d)

    flags: list[str] = []
    if not ent_t:
        flags.append("target_entity_no_exact_source_span")
    if not ent_d:
        flags.append("distractor_entity_no_exact_source_span")
    if not t_exact:
        flags.append("target_source_answer_no_exact_raw_span")
    if not d_exact:
        flags.append("distractor_source_answer_no_exact_raw_span")
    if t_exact and ent_t and ent_d and row["target_ans_to_target_dist"] > row["target_ans_to_distractor_dist"]:
        flags.append("target_source_answer_closer_to_distractor_than_target")
    if d_exact and ent_t and ent_d and row["distractor_ans_to_distractor_dist"] > row["distractor_ans_to_target_dist"]:
        flags.append("distractor_source_answer_closer_to_target_than_distractor")
    if t_exact and d_exact and ent_t and row["target_entity_competing_source_answer_dist"] < row["target_ans_to_target_dist"]:
        flags.append("target_entity_closer_to_competing_source_answer")
    if t_exact and d_exact and ent_d and row["distractor_entity_competing_source_answer_dist"] < row["distractor_ans_to_distractor_dist"]:
        flags.append("distractor_entity_closer_to_competing_source_answer")
    # Warn when source answer is exact but quite far from entity and the source is short enough that relation should be local.
    if t_exact and ent_t and row["target_ans_to_target_dist"] > 80:
        flags.append("target_answer_far_from_target_entity")
    if d_exact and ent_d and row["distractor_ans_to_distractor_dist"] > 80:
        flags.append("distractor_answer_far_from_distractor_entity")

    row["entity_answer_flags"] = flags
    row["n_entity_answer_flags"] = len(flags)
    if base:
        for k in ["auto_flags", "n_auto_flags", "U_mean_new_minus_source", "R_mean_source_minus_new", "beta", "abs_alpha", "gamma", "joint_correct_mean", "update_event_class"]:
            row[k] = base.get(k)
    return row


def finite(xs: list[float]) -> list[float]:
    return [x for x in xs if not math.isinf(x) and not math.isnan(x)]


def m(xs: list[float]) -> float:
    vals = finite(xs)
    return sum(vals) / len(vals) if vals else float("nan")


def med(xs: list[float]) -> float:
    vals = finite(xs)
    return statistics.median(vals) if vals else float("nan")


def score_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("gamma") is not None]
    return {
        "n": len(rows),
        "with_scores": len(scored),
        "joint_correct": sum(bool(r.get("joint_correct_mean")) for r in scored),
        "mean_gamma": sum(float(r["gamma"]) for r in scored) / len(scored) if scored else float("nan"),
        "median_gamma": statistics.median(float(r["gamma"]) for r in scored) if scored else float("nan"),
    }


def main() -> None:
    pairs = read_jsonl(PAIRS_PATH)
    base_by_id = {r["pair_id"]: r for r in read_jsonl(SEM_AUDIT_PATH)} if SEM_AUDIT_PATH.exists() else {}
    rows = [audit_pair(p, base_by_id.get(p["pair_id"])) for p in pairs]
    flag_counts = Counter(flag for r in rows for flag in r["entity_answer_flags"])
    both_exact = [r for r in rows if r["target_answer_exact_spans"] and r["distractor_answer_exact_spans"]]
    clean_prox = [r for r in rows if not r["entity_answer_flags"]]
    both_assigned_closer = [r for r in rows if r["target_assigned_exact_closer_than_competing"] and r["distractor_assigned_exact_closer_than_competing"]]
    summary = {
        "status": "STEP038B_ENTITY_ANSWER_GROUNDING_AUDIT",
        "created_utc": now_utc(),
        "inputs": {"pairs_path": str(PAIRS_PATH), "semantic_audit_path": str(SEM_AUDIT_PATH)},
        "n_pairs": len(rows),
        "exact_span_grounding": {
            "both_source_answers_exact_raw_span": len(both_exact),
            "both_source_answers_exact_raw_span_fraction": len(both_exact) / len(rows),
            "both_entities_have_assigned_answer_closer_than_competing_exact_answer": len(both_assigned_closer),
            "both_entities_have_assigned_answer_closer_than_competing_exact_answer_fraction": len(both_assigned_closer) / len(rows),
            "no_entity_answer_flags": len(clean_prox),
            "no_entity_answer_flags_fraction": len(clean_prox) / len(rows),
            "mean_target_assigned_exact_distance": m([r["target_ans_to_target_dist"] for r in rows]),
            "mean_distractor_assigned_exact_distance": m([r["distractor_ans_to_distractor_dist"] for r in rows]),
            "median_target_assigned_exact_distance": med([r["target_ans_to_target_dist"] for r in rows]),
            "median_distractor_assigned_exact_distance": med([r["distractor_ans_to_distractor_dist"] for r in rows]),
        },
        "flag_counts": dict(flag_counts),
        "baseline_score_stats": {
            "all": score_stats(rows),
            "both_answers_exact_raw_span": score_stats(both_exact),
            "no_entity_answer_flags": score_stats(clean_prox),
            "both_assigned_closer_than_competing_exact_answer": score_stats(both_assigned_closer),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "entity_answer_grounding_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "entity_answer_grounding_pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            # JSON cannot represent inf; stringify distance fields.
            rr = dict(r)
            for k, v in list(rr.items()):
                if isinstance(v, float) and math.isinf(v):
                    rr[k] = "inf"
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    suspicious = sorted(rows, key=lambda r: (-r["n_entity_answer_flags"], float(r.get("gamma", 0.0) or 0.0)))[:25]
    with ((OUT_DIR.parents[4] / 'research/documents/functional_learning/data/entity_answer_grounding_audit/manual_review_entity_answer_suspicious.md')).open("w", encoding="utf-8") as f:
        f.write("# Step038b entity-answer grounding suspicious rows\n\n")
        for i, r in enumerate(suspicious, 1):
            f.write(f"## {i}. {r['pair_id']} ({r.get('source')}; {r.get('role_position_relation')})\n\n")
            f.write(f"flags: {', '.join(r['entity_answer_flags']) or 'none'}\n\n")
            f.write(f"gamma={r.get('gamma')} joint={r.get('joint_correct_mean')} update_event={r.get('update_event_class')}\n\n")
            f.write(f"SOURCE: {r['source_sentence']}\n\n")
            f.write(f"TARGET={r['target_entity']} ans={r['target_source_answer']} | dist_to_target={fmt_dist(r['target_ans_to_target_dist'])} dist_to_distractor={fmt_dist(r['target_ans_to_distractor_dist'])} competing_ans_dist_to_target={fmt_dist(r['target_entity_competing_source_answer_dist'])}\n\n")
            f.write(f"DISTRACTOR={r['distractor_entity']} ans={r['distractor_source_answer']} | dist_to_distractor={fmt_dist(r['distractor_ans_to_distractor_dist'])} dist_to_target={fmt_dist(r['distractor_ans_to_target_dist'])} competing_ans_dist_to_distractor={fmt_dist(r['distractor_entity_competing_source_answer_dist'])}\n\n")
            if r.get("shared_update_sentence_frame"):
                f.write(f"UPDATE: {r['shared_update_sentence_frame']}\n\n")
    # Append concise result to the research note.
    with NOTE_APPEND_PATH.open("a", encoding="utf-8") as f:
        f.write("\n## Step038b entity-answer association addendum\n")
        f.write("Raw source overlap is still insufficient because an answer phrase can occur in the source but be associated with a different entity or event. A cheap exact-span proximity probe therefore measured whether each source answer is closer to its assigned entity than to the competing entity/source answer. This is only a proxy, but it catches association failures such as list/rank rows where the right phrase exists in the sentence but belongs to another item.\n")
        e = summary["exact_span_grounding"]
        f.write(f"- Both source answers exact raw spans: {e['both_source_answers_exact_raw_span']}/{summary['n_pairs']} ({100*e['both_source_answers_exact_raw_span_fraction']:.1f}%).\n")
        f.write(f"- Both entities had their assigned source answer closer than the competing source answer under exact-span distance: {e['both_entities_have_assigned_answer_closer_than_competing_exact_answer']}/{summary['n_pairs']} ({100*e['both_entities_have_assigned_answer_closer_than_competing_exact_answer_fraction']:.1f}%).\n")
        f.write(f"- Rows with no entity-answer proximity flags: {e['no_entity_answer_flags']}/{summary['n_pairs']} ({100*e['no_entity_answer_flags_fraction']:.1f}%).\n")
        f.write(f"- Main entity-answer flags: `{summary['flag_counts']}`.\n")
        f.write("This reinforces the construction change: source-grounded records need raw spans or independent evidence for the relation between each entity and answer, not merely phrase overlap with the source.\n")
        f.write(f"Files: `{OUT_DIR / 'entity_answer_grounding_summary.json'}`, `{OUT_DIR / 'entity_answer_grounding_pairs.jsonl'}`, `{(OUT_DIR.parents[4] / 'research/documents/functional_learning/data/entity_answer_grounding_audit/manual_review_entity_answer_suspicious.md')}`.\n")
    print(json.dumps({
        "status": summary["status"],
        "n_pairs": summary["n_pairs"],
        "both_exact": summary["exact_span_grounding"]["both_source_answers_exact_raw_span"],
        "both_assigned_closer": summary["exact_span_grounding"]["both_entities_have_assigned_answer_closer_than_competing_exact_answer"],
        "no_entity_answer_flags": summary["exact_span_grounding"]["no_entity_answer_flags"],
        "top_flags": summary["flag_counts"],
        "summary_path": str(OUT_DIR / "entity_answer_grounding_summary.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
