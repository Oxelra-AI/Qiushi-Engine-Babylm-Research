#!/usr/bin/env python3
"""research: semantic-foundation audit of research contrastive rows.

This script does not try to decide entailment. It measures where the research
surface validator can certify rows using generator-supplied state descriptions,
rather than evidence in the raw source, and produces a candidate contract for the
next natural-data construction.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Any

import matplotlib.pyplot as plt

ROOT = pathlib.Path("experiments/archive/functional_learning")
A02 = pathlib.Path("experiments/archive/relation_learning")
PAIRS_PATH = A02 / "data/contrastive_binding_validated_strict/accepted_contrastive_binding_pairs_strict_pilot512.jsonl"
SCORES_PATH = ROOT / "data/multitoken_scorer_pilot512/pair_scores.jsonl"
META_PATH = A02 / "data/contrastive_binding_validated_strict/validation_metadata_strict_pilot512.json"
OUT_DIR = ROOT / "data/semantic_foundation_audit"
FIG_DIR = ROOT / "figures"
NOTE_PATH = (ROOT.parents[2] / 'research/notes/functional_learning/semantic_foundation_for_natural_rows.md')

STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at", "by", "from", "as",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "his", "her", "their", "about", "into", "over", "under", "during", "after", "before", "which", "who",
    "whom", "whose", "had", "has", "have", "will", "would", "could", "should", "can", "may", "might",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

# These are not all wrong. They separate update statements likely to make a
# current-state replacement explicit from statements that merely add a compatible
# action/possession/attribute.
STRONG_REPLACEMENT_RE = re.compile(
    r"\b(replaced|replace|replacing|switched|switch|changed|change|converted|convert|renamed|rename|"
    r"reassigned|assigned|reclassified|updated|moved into|transferred to|became|declared|elected|appointed|"
    r"listed as|ranked|reached|went to|marked as|designated as|now)\b",
    re.I,
)
ADDITIVE_RE = re.compile(
    r"\b(adopted|adopt|used|use|uses|using|kept|stored|served|showcased|featured|introduced|"
    r"displayed|praised|admired|found|discovered|revealed|associated with|worked with|taught|studied)\b",
    re.I,
)
ENTITY_SUSPECT_RE = re.compile(r"(\*|\bI'm\b|\bPersonal\b|\bIslamic\b|\bBritish\b|\bNative\b|\bJanuary\b|\bAugust\b|\bLP\b|\btown\b)", re.I)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm_tokens(x: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in TOKEN_RE.finditer(x or "")]


def content_tokens(x: str) -> list[str]:
    return [t for t in norm_tokens(x) if t not in STOP and len(t) > 1]


def phrase_in(needle: str, haystack: str) -> bool:
    n = " ".join(norm_tokens(needle))
    h = " ".join(norm_tokens(haystack))
    return bool(n) and n in h


def overlap_min(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, min(len(A), len(B)))


def jaccard(a: list[str], b: list[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def contiguous_lcs_len(a: list[str], b: list[str]) -> int:
    best = 0
    dp = [0] * (len(b) + 1)
    for x in a:
        ndp = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                ndp[j] = dp[j - 1] + 1
                best = max(best, ndp[j])
        dp = ndp
    return best


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def pct(n: int, d: int) -> float:
    return n / d if d else float("nan")


def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{x:.3f}"


def load_scores() -> dict[str, dict[str, Any]]:
    scores = {}
    for r in read_jsonl(SCORES_PATH):
        U = float(r["U_mean_new_minus_source"])
        R = float(r["R_mean_source_minus_new"])
        beta = (U + R) / 2
        alpha = (U - R) / 2
        scores[r["pair_id"]] = {
            **r,
            "beta": beta,
            "alpha": alpha,
            "abs_alpha": abs(alpha),
            "gamma": beta - abs(alpha),
        }
    return scores


def audit_pair(p: dict[str, Any], score: dict[str, Any] | None) -> dict[str, Any]:
    source = p.get("source_sentence", "")
    up_frame = p.get("shared_update_sentence_frame", "")
    use_frame = p.get("use_sentence_frame", "")
    target = p.get("target_entity", "")
    distractor = p.get("distractor_entity", "")
    target_ans = p.get("target_source_answer", "")
    dist_ans = p.get("distractor_source_answer", "")
    new_ans = p.get("shared_new_answer", "")
    target_state = p.get("target_source_state", "")
    dist_state = p.get("distractor_source_state", "")
    new_state = p.get("shared_new_update_state", "")

    src_ct = content_tokens(source)
    row: dict[str, Any] = {
        "pair_id": p.get("pair_id"),
        "source": p.get("source"),
        "cohort": p.get("cohort"),
        "role_position_relation": p.get("role_position_relation"),
        "target_entity": target,
        "distractor_entity": distractor,
        "target_source_answer": target_ans,
        "distractor_source_answer": dist_ans,
        "shared_new_answer": new_ans,
        "source_sentence": source,
        "target_source_state": target_state,
        "distractor_source_state": dist_state,
        "shared_new_update_state": new_state,
        "shared_update_sentence_frame": up_frame,
        "use_sentence_frame": use_frame,
    }

    for prefix, ans, state in [
        ("target", target_ans, target_state),
        ("distractor", dist_ans, dist_state),
    ]:
        ans_ct = content_tokens(ans)
        state_ct = content_tokens(state)
        row[f"{prefix}_answer_exact_in_raw_source"] = phrase_in(ans, source)
        row[f"{prefix}_answer_overlap_raw_source"] = overlap_min(ans_ct, src_ct)
        row[f"{prefix}_answer_overlap_source_plus_generated_state"] = overlap_min(ans_ct, content_tokens(source + " " + state))
        row[f"{prefix}_state_overlap_raw_source"] = overlap_min(state_ct, src_ct)
        row[f"{prefix}_answer_self_certification_gap"] = (
            row[f"{prefix}_answer_overlap_raw_source"] < 0.34
            and row[f"{prefix}_answer_overlap_source_plus_generated_state"] >= 0.34
        )
        row[f"{prefix}_answer_tokens"] = len(ans_ct)
        row[f"{prefix}_state_tokens"] = len(state_ct)

    new_ct = content_tokens(new_ans)
    up_ct = content_tokens(up_frame)
    use_ct = content_tokens(use_frame.replace("{STATE}", ""))
    row["new_answer_exact_in_update_frame"] = phrase_in(new_ans, up_frame)
    row["new_answer_overlap_update_frame"] = overlap_min(new_ct, up_ct)
    row["new_state_overlap_update_frame"] = overlap_min(content_tokens(new_state), up_ct)
    row["update_has_strong_replacement_cue"] = bool(STRONG_REPLACEMENT_RE.search(up_frame))
    row["update_has_additive_cue"] = bool(ADDITIVE_RE.search(up_frame))
    row["update_event_class"] = (
        "strong_replacement_like" if row["update_has_strong_replacement_cue"] else
        "additive_or_descriptive" if row["update_has_additive_cue"] else
        "weak_or_unclear"
    )
    row["target_entity_suspect_form"] = bool(ENTITY_SUSPECT_RE.search(target))
    row["distractor_entity_suspect_form"] = bool(ENTITY_SUSPECT_RE.search(distractor))
    row["use_frame_new_answer_overlap"] = overlap_min(new_ct, use_ct)
    row["use_frame_source_answer_overlap"] = overlap_min(content_tokens(target_ans), use_ct)
    row["use_frame_vs_update_frame_lcs"] = contiguous_lcs_len(norm_tokens(use_frame), norm_tokens(up_frame))
    row["source_length_tokens"] = len(norm_tokens(source))
    row["target_update_sentence"] = p.get("target_update_sentence")
    row["distractor_update_sentence"] = p.get("distractor_update_sentence")
    row["update_use_sentence"] = p.get("update_use_sentence")
    row["retain_use_sentence"] = p.get("retain_use_sentence")

    flags = []
    if row["target_answer_self_certification_gap"]:
        flags.append("target_answer_certified_by_generated_state")
    if row["distractor_answer_self_certification_gap"]:
        flags.append("distractor_answer_certified_by_generated_state")
    if not row["target_answer_exact_in_raw_source"]:
        flags.append("target_answer_not_exact_raw_span")
    if not row["distractor_answer_exact_in_raw_source"]:
        flags.append("distractor_answer_not_exact_raw_span")
    if row["target_answer_overlap_raw_source"] < 0.34:
        flags.append("target_answer_weak_raw_overlap")
    if row["distractor_answer_overlap_raw_source"] < 0.34:
        flags.append("distractor_answer_weak_raw_overlap")
    if row["target_state_overlap_raw_source"] < 0.34:
        flags.append("target_state_weak_raw_overlap")
    if row["distractor_state_overlap_raw_source"] < 0.34:
        flags.append("distractor_state_weak_raw_overlap")
    if not row["update_has_strong_replacement_cue"]:
        flags.append("update_not_explicit_replacement_like")
    if row["update_has_additive_cue"] and not row["update_has_strong_replacement_cue"]:
        flags.append("update_additive_or_coexistence_risk")
    if row["target_entity_suspect_form"] or row["distractor_entity_suspect_form"]:
        flags.append("suspect_entity_surface_form")
    if row["use_frame_new_answer_overlap"] > row["use_frame_source_answer_overlap"] and row["use_frame_new_answer_overlap"] >= 0.34:
        flags.append("use_frame_lexically_favors_new_answer")
    row["auto_flags"] = flags
    row["n_auto_flags"] = len(flags)

    if score:
        for k in ["U_mean_new_minus_source", "R_mean_source_minus_new", "beta", "alpha", "abs_alpha", "gamma",
                  "update_correct_mean", "retain_correct_mean", "joint_correct_mean", "same_token_length_both_rows",
                  "answer_also_elsewhere_update", "answer_also_elsewhere_retain"]:
            row[k] = score.get(k)
    return row


def summarize(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    flag_counts = Counter(flag for r in rows for flag in r["auto_flags"])
    event_counts = Counter(r["update_event_class"] for r in rows)
    both_exact = sum(r["target_answer_exact_in_raw_source"] and r["distractor_answer_exact_in_raw_source"] for r in rows)
    both_raw_overlap = sum(r["target_answer_overlap_raw_source"] >= 0.34 and r["distractor_answer_overlap_raw_source"] >= 0.34 for r in rows)
    any_self_gap = sum(r["target_answer_self_certification_gap"] or r["distractor_answer_self_certification_gap"] for r in rows)
    both_state_034 = sum(r["target_state_overlap_raw_source"] >= 0.34 and r["distractor_state_overlap_raw_source"] >= 0.34 for r in rows)
    replacement_like = sum(r["update_has_strong_replacement_cue"] for r in rows)
    suspect_entity = sum(r["target_entity_suspect_form"] or r["distractor_entity_suspect_form"] for r in rows)
    strict_proxy = [r for r in rows if (
        r["target_answer_exact_in_raw_source"]
        and r["distractor_answer_exact_in_raw_source"]
        and r["target_answer_overlap_raw_source"] >= 0.34
        and r["distractor_answer_overlap_raw_source"] >= 0.34
        and r["target_state_overlap_raw_source"] >= 0.34
        and r["distractor_state_overlap_raw_source"] >= 0.34
        and r["update_has_strong_replacement_cue"]
        and not (r["target_entity_suspect_form"] or r["distractor_entity_suspect_form"])
    )]
    loose_grounded = [r for r in rows if (
        r["target_answer_overlap_raw_source"] >= 0.34
        and r["distractor_answer_overlap_raw_source"] >= 0.34
        and r["target_state_overlap_raw_source"] >= 0.20
        and r["distractor_state_overlap_raw_source"] >= 0.20
        and not (r["target_entity_suspect_form"] or r["distractor_entity_suspect_form"])
    )]

    def score_stats(sub: list[dict[str, Any]]) -> dict[str, Any]:
        with_scores = [r for r in sub if "gamma" in r]
        return {
            "n": len(sub),
            "with_scores": len(with_scores),
            "joint_correct": sum(bool(r.get("joint_correct_mean")) for r in with_scores),
            "mean_beta": mean([float(r["beta"]) for r in with_scores]),
            "mean_abs_alpha": mean([float(r["abs_alpha"]) for r in with_scores]),
            "mean_gamma": mean([float(r["gamma"]) for r in with_scores]),
            "median_gamma": median([float(r["gamma"]) for r in with_scores]),
        }

    return {
        "status": "SEMANTIC_FOUNDATION_AUDIT",
        "created_utc": now_utc(),
        "inputs": {
            "pairs_path": str(PAIRS_PATH),
            "scores_path": str(SCORES_PATH),
            "metadata_path": str(META_PATH),
            "accepted_pairs": meta.get("accepted_pairs"),
            "acceptance_rate_per_output": meta.get("acceptance_rate_per_output"),
        },
        "n_pairs": n,
        "raw_source_grounding": {
            "both_source_answers_exact_substrings_of_raw_source": both_exact,
            "both_source_answers_exact_fraction": pct(both_exact, n),
            "both_source_answers_overlap_min_ge_0p34_with_raw_source": both_raw_overlap,
            "both_source_answers_raw_overlap_fraction": pct(both_raw_overlap, n),
            "any_source_answer_self_certification_gap": any_self_gap,
            "any_source_answer_self_certification_gap_fraction": pct(any_self_gap, n),
            "both_generated_source_states_overlap_raw_ge_0p34": both_state_034,
            "both_generated_source_states_overlap_raw_ge_0p34_fraction": pct(both_state_034, n),
        },
        "update_event_proxy": {
            "strong_replacement_like_count": replacement_like,
            "strong_replacement_like_fraction": pct(replacement_like, n),
            "event_class_counts": dict(event_counts),
        },
        "entity_surface_proxy": {
            "suspect_entity_surface_form_count": suspect_entity,
            "suspect_entity_surface_form_fraction": pct(suspect_entity, n),
        },
        "flag_counts": dict(flag_counts),
        "baseline_score_stats": {
            "all": score_stats(rows),
            "loose_raw_grounded_no_suspect_entity": score_stats(loose_grounded),
            "strict_automatic_proxy": score_stats(strict_proxy),
            "rows_with_any_self_certification_gap": score_stats([r for r in rows if r["target_answer_self_certification_gap"] or r["distractor_answer_self_certification_gap"]]),
            "rows_without_self_certification_gap": score_stats([r for r in rows if not (r["target_answer_self_certification_gap"] or r["distractor_answer_self_certification_gap"])]),
            "replacement_like": score_stats([r for r in rows if r["update_has_strong_replacement_cue"]]),
            "not_replacement_like": score_stats([r for r in rows if not r["update_has_strong_replacement_cue"]]),
        },
    }


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "semantic_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (OUT_DIR / "semantic_audit_pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    fieldnames = [
        "pair_id", "source", "role_position_relation", "target_entity", "distractor_entity",
        "target_source_answer", "distractor_source_answer", "shared_new_answer",
        "target_answer_exact_in_raw_source", "distractor_answer_exact_in_raw_source",
        "target_answer_overlap_raw_source", "distractor_answer_overlap_raw_source",
        "target_answer_self_certification_gap", "distractor_answer_self_certification_gap",
        "target_state_overlap_raw_source", "distractor_state_overlap_raw_source",
        "update_event_class", "target_entity_suspect_form", "distractor_entity_suspect_form",
        "n_auto_flags", "auto_flags", "U_mean_new_minus_source", "R_mean_source_minus_new",
        "beta", "abs_alpha", "gamma", "joint_correct_mean",
    ]
    with (OUT_DIR / "semantic_audit_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["auto_flags"] = ";".join(r["auto_flags"])
            w.writerow(rr)

    # Save short manual-read queues: severe automatic issues and plausible stricter candidates.
    severe = sorted(rows, key=lambda r: (-r["n_auto_flags"], float(r.get("gamma", 0.0))))[:20]
    plausible = [r for r in rows if (
        r["target_answer_overlap_raw_source"] >= 0.34
        and r["distractor_answer_overlap_raw_source"] >= 0.34
        and r["target_state_overlap_raw_source"] >= 0.20
        and r["distractor_state_overlap_raw_source"] >= 0.20
        and not (r["target_entity_suspect_form"] or r["distractor_entity_suspect_form"])
    )]
    plausible = sorted(plausible, key=lambda r: (float(r.get("gamma", 0.0)), r["n_auto_flags"]))[:20]
    for name, subset in [("manual_review_severe_auto_flags.md", severe), ("manual_review_plausible_hard_rows.md", plausible)]:
        with (OUT_DIR / name).open("w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            for i, r in enumerate(subset, 1):
                f.write(f"## {i}. {r['pair_id']} ({r.get('source')}; {r.get('role_position_relation')})\n\n")
                f.write(f"flags: {', '.join(r['auto_flags']) or 'none'}\n\n")
                if "gamma" in r:
                    f.write(f"baseline U={fmt(float(r['U_mean_new_minus_source']))} R={fmt(float(r['R_mean_source_minus_new']))} beta={fmt(float(r['beta']))} |alpha|={fmt(float(r['abs_alpha']))} gamma={fmt(float(r['gamma']))} joint={r.get('joint_correct_mean')}\n\n")
                f.write(f"SOURCE: {r['source_sentence']}\n\n")
                f.write(f"TARGET={r['target_entity']} DISTRACTOR={r['distractor_entity']}\n\n")
                f.write(f"TARGET_SOURCE_STATE={r['target_source_state']} | ANSWER={r['target_source_answer']} | raw_overlap={fmt(float(r['target_answer_overlap_raw_source']))}, exact={r['target_answer_exact_in_raw_source']}\n\n")
                f.write(f"DISTRACTOR_SOURCE_STATE={r['distractor_source_state']} | ANSWER={r['distractor_source_answer']} | raw_overlap={fmt(float(r['distractor_answer_overlap_raw_source']))}, exact={r['distractor_answer_exact_in_raw_source']}\n\n")
                f.write(f"UPDATE_FRAME: {r['shared_update_sentence_frame']} [{r['update_event_class']}]\n\n")
                f.write(f"USE_FRAME: {r['use_sentence_frame']}\n\n")
                f.write(f"UPDATE_USE: {r.get('update_use_sentence')}\n\n")
                f.write(f"RETAIN_USE: {r.get('retain_use_sentence')}\n\n")

    # Figures.
    scored = [r for r in rows if "gamma" in r]
    fig_path = FIG_DIR / "semantic_flags_vs_gamma.png"
    plt.figure(figsize=(9.5, 5.2))
    xs = [r["n_auto_flags"] for r in scored]
    ys = [float(r["gamma"]) for r in scored]
    colors = ["tab:green" if r.get("joint_correct_mean") else "tab:red" for r in scored]
    plt.axhline(0, color="black", lw=1, alpha=0.6)
    plt.scatter(xs, ys, c=colors, alpha=0.75, edgecolor="white", linewidth=0.4)
    plt.xlabel("automatic semantic/grounding flags per pair")
    plt.ylabel("coherent86 baseline gamma = beta - |alpha| (mean-token scoring)")
    plt.title("research accepted rows: automatic flags do not by themselves define capability")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()

    fig_path2 = FIG_DIR / "raw_grounding_event_counts.png"
    labels = [
        "both answers\nexact raw spans",
        "both answers\nraw overlap≥.34",
        "any self-\ncertification gap",
        "strong replacement-\nlike update",
        "suspect entity\nsurface",
    ]
    vals = [
        summary["raw_source_grounding"]["both_source_answers_exact_substrings_of_raw_source"],
        summary["raw_source_grounding"]["both_source_answers_overlap_min_ge_0p34_with_raw_source"],
        summary["raw_source_grounding"]["any_source_answer_self_certification_gap"],
        summary["update_event_proxy"]["strong_replacement_like_count"],
        summary["entity_surface_proxy"]["suspect_entity_surface_form_count"],
    ]
    plt.figure(figsize=(9.2, 4.8))
    plt.bar(range(len(vals)), vals, color=["tab:blue", "tab:blue", "tab:orange", "tab:purple", "tab:red"])
    plt.xticks(range(len(vals)), labels, rotation=0, ha="center")
    plt.ylim(0, max(55, max(vals) + 3))
    plt.ylabel("pairs out of 55")
    plt.title("research semantic-foundation proxies")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.8, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(fig_path2, dpi=180)
    plt.close()


def write_note(summary: dict[str, Any]) -> None:
    s = summary
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# research semantic foundation for natural recipient rows\n")
    lines.append("## Why this audit was needed\n")
    lines.append("The next natural-data construction cannot be made reliable by balancing entities, symmetric scoring, or stricter surface-copy filters alone. research already enforced same source, assigned entities, a shared update frame, and a shared use frame, but the generated examples can still fail the intended semantics: the update may add a compatible object rather than replace the target's current state, the use frame may independently favor one candidate, and the validator can let a generated `target_source_state` or `distractor_source_state` help certify its own answer.\n")
    lines.append("This audit measures these failure modes as proxies, not as entailment judgments. Raw lexical overlap is still not proof of source grounding; it is a lower-level check that generator metadata has not substituted for evidence in the source.\n")
    rg = s["raw_source_grounding"]
    up = s["update_event_proxy"]
    ent = s["entity_surface_proxy"]
    lines.append("## Quantitative audit of research accepted pilot rows\n")
    lines.append(f"Input: `{PAIRS_PATH}`; baseline scores from `{SCORES_PATH}`. The research validator accepted {s['n_pairs']} pairs from {s['inputs'].get('accepted_pairs')} recorded accepted pairs at acceptance rate {s['inputs'].get('acceptance_rate_per_output')}.\n")
    lines.append(f"- Both source answers were exact substrings of the raw source in {rg['both_source_answers_exact_substrings_of_raw_source']}/{s['n_pairs']} pairs ({100*rg['both_source_answers_exact_fraction']:.1f}%).\n")
    lines.append(f"- Both source answers had overlap_min >= 0.34 with the raw source in {rg['both_source_answers_overlap_min_ge_0p34_with_raw_source']}/{s['n_pairs']} pairs ({100*rg['both_source_answers_raw_overlap_fraction']:.1f}%).\n")
    lines.append(f"- At least one source answer had the self-certification pattern (weak raw-source overlap, but acceptable overlap after concatenating the generated state description) in {rg['any_source_answer_self_certification_gap']}/{s['n_pairs']} pairs ({100*rg['any_source_answer_self_certification_gap_fraction']:.1f}%).\n")
    lines.append(f"- Both generated source-state descriptions had raw-source overlap >= 0.34 in {rg['both_generated_source_states_overlap_raw_ge_0p34']}/{s['n_pairs']} pairs ({100*rg['both_generated_source_states_overlap_raw_ge_0p34_fraction']:.1f}%).\n")
    lines.append(f"- The update frame contained a strong replacement/change/assignment-like cue in {up['strong_replacement_like_count']}/{s['n_pairs']} pairs ({100*up['strong_replacement_like_fraction']:.1f}%). Event-class proxy counts: `{up['event_class_counts']}`.\n")
    lines.append(f"- Suspect entity surface forms (e.g. fragmentary roles, adjectives, dates, transcript tokens) appeared in {ent['suspect_entity_surface_form_count']}/{s['n_pairs']} pairs ({100*ent['suspect_entity_surface_form_fraction']:.1f}%).\n")
    lines.append("\nThese numbers explain why the current 55-pair pilot is useful as a scorer stress test but is not yet a strong BabyLM training substrate. The problem is not merely that examples are too easy or too copy-based; many examples do not clearly entail the opposite answers from the texts a competent reader sees.\n")
    lines.append("## Baseline capability should still be measured on semantically valid hard pairs\n")
    bstats = s["baseline_score_stats"]
    for name, st in bstats.items():
        lines.append(f"- `{name}`: n={st['n']}, joint={st['joint_correct']}/{st['with_scores']}, mean beta={fmt(st['mean_beta'])}, mean |alpha|={fmt(st['mean_abs_alpha'])}, mean gamma={fmt(st['mean_gamma'])}, median gamma={fmt(st['median_gamma'])}.\n")
    lines.append("\nAutomatic flags do not decide scientific validity. Poor coherent86 performance on a semantically valid hard pair remains a useful missing-capability signal. The repair should reject ambiguous/non-entailing rows while preserving rows where both answer phrases are present and the learner still must select the entity to choose between competing states.\n")
    lines.append("## Contract for the next tranche\n")
    lines.append("A stronger construction should first establish independently source-grounded relations, then render recipient counterfactuals. The minimum object should be a source-grounded record with raw character spans or independently verified evidence for `(target_entity, relation, target_source_answer)` and `(distractor_entity, relation, distractor_source_answer)`, plus an explicit update event that changes the named entity's current value to `shared_new_answer`. The rendered pair should then be:\n")
    lines.append("1. `TARGET_UPDATE`: raw source + identical replacement/update event applied to the target + identical use frame asking about the target, answer `shared_new_answer`.\n")
    lines.append("2. `DISTRACTOR_UPDATE/RETAIN`: same raw source + the same update event applied to the distractor + the same use frame asking about the target, answer `target_source_answer`.\n")
    lines.append("The answer strings may be copied from source/update; copying after selecting the correct entity is the desired computation when both competing phrases are available. Temporal or replacement language is also allowed when it makes the current state well-defined, provided it is shared across the two variants and cannot by itself identify whether the target or distractor was updated.\n")
    lines.append("## Validator changes needed\n")
    lines.append("- Do not test source-answer grounding against `source + generated_state_description`; test raw source spans and retain the span positions. If a paraphrase answer is desired, keep both an exact raw support span and a separately scored normalized answer, rather than letting the paraphrase certify itself.\n")
    lines.append("- Add an independent reader/entailment pass, ideally not the same generator, over the fully rendered TARGET_UPDATE and RETAIN examples: the reader must answer the target use question with the intended candidate and mark whether the opposite answer is unsupported.\n")
    lines.append("- Require explicit replacement/change/assignment semantics in the update event, but do not ban all temporal language. A shared phrase such as 'after the revision' or 'now listed as' can define current state; it is not a shortcut unless it differs between variants or points to one entity.\n")
    lines.append("- Keep difficult valid pairs, including cases where both source and update candidate phrases are visible. The readout remains UPDATE, RETAIN, pair-level gamma=beta-|alpha|, and joint correctness; easy parent-solved rows are less informative for the BabyLM bridge.\n")
    lines.append("\n## Files produced\n")
    lines.append(f"- Structured summary: `{OUT_DIR / 'semantic_audit_summary.json'}`\n")
    lines.append(f"- Pair-level JSONL/CSV: `{OUT_DIR / 'semantic_audit_pairs.jsonl'}`, `{OUT_DIR / 'semantic_audit_pairs.csv'}`\n")
    lines.append(f"- Manual queues: `{(OUT_DIR.parents[4] / 'research/documents/functional_learning/data/semantic_foundation_audit/manual_review_severe_auto_flags.md')}`, `{(OUT_DIR.parents[4] / 'research/documents/functional_learning/data/semantic_foundation_audit/manual_review_plausible_hard_rows.md')}`\n")
    lines.append(f"- Figures: `{FIG_DIR / 'semantic_flags_vs_gamma.png'}`, `{FIG_DIR / 'raw_grounding_event_counts.png'}`\n")
    NOTE_PATH.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    pairs = read_jsonl(PAIRS_PATH)
    scores = load_scores()
    rows = [audit_pair(p, scores.get(p.get("pair_id"))) for p in pairs]
    summary = summarize(rows, meta)
    write_outputs(rows, summary)
    write_note(summary)
    print(json.dumps({
        "status": summary["status"],
        "n_pairs": summary["n_pairs"],
        "summary_path": str(OUT_DIR / "semantic_audit_summary.json"),
        "note_path": str(NOTE_PATH),
        "raw_both_exact": summary["raw_source_grounding"]["both_source_answers_exact_substrings_of_raw_source"],
        "self_cert_gap": summary["raw_source_grounding"]["any_source_answer_self_certification_gap"],
        "strong_replacement_like": summary["update_event_proxy"]["strong_replacement_like_count"],
        "all_mean_gamma": summary["baseline_score_stats"]["all"]["mean_gamma"],
    }, indent=2))


if __name__ == "__main__":
    main()
