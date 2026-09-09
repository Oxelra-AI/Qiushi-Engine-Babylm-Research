#!/usr/bin/env python3
"""research: critical review of research role-fact substrate.

This CPU-only analysis inspects the teacher-label substrate before any student
training. It separates parser artifacts, bridge/reference semantic loss,
Qwen fact-generation overreach, ambiguous false negatives, and teacher-specific
label behavior.
"""
from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
AUTO = ROOT / "data/auto_role_fact_substrate"
MANUAL = ROOT / "data/natural_role_transform_panel"
OUT = ROOT / "data/role_substrate_review"
NOTE = (ROOT.parents[2] / 'research/notes/representation_and_objectives/role_substrate_review_and_repair_plan.md')

TEACHERS = ["qwen3.5-9b", "llama3.1-8b-instruct"]
CONTEXTS = ["source", "source_attested_bridge", "natural_compact_reference"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def tolerant_parse(raw: str) -> str | None:
    s = str(raw or "").strip().upper()
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\bNOT[_ ]?ENTAIL(?:ED|LED)?\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAIL(?:ED|LED)?\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize(rows: list[dict[str, Any]], key: str, label_field: str = "tol_label") -> dict[str, Any]:
    out: dict[str, Any] = {}
    vals = sorted({str(r.get(key)) for r in rows})
    for v in vals:
        sub = [r for r in rows if str(r.get(key)) == v]
        valid = [r for r in sub if r.get(label_field)]
        exp = [r for r in valid if r[label_field] == r["gold_label"]]
        out[v] = {"n": len(sub), "valid": len(valid), "expected": len(exp), "expected_rate": rate(len(exp), len(valid)), "invalid": len(sub) - len(valid)}
    return out


def compact_example(r: dict[str, Any], include_context: bool = True) -> dict[str, Any]:
    out = {
        "id": r.get("id"),
        "case_id": r.get("case_id"),
        "teacher": r.get("teacher"),
        "context_type": r.get("context_type"),
        "fact_id": r.get("fact_id"),
        "role_type": r.get("role_type"),
        "gold": r.get("gold_label"),
        "label": r.get("tol_label"),
        "raw": r.get("output_raw"),
        "hypothesis": r.get("hypothesis"),
    }
    if include_context:
        out["context"] = r.get("context")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manual_summary = json.loads((MANUAL / "teacher_panel_summary.json").read_text(encoding="utf-8"))
    auto_summary = json.loads((AUTO / "auto_role_fact_substrate_summary.json").read_text(encoding="utf-8"))
    cases = read_jsonl(AUTO / "generated_fact_cases.jsonl")
    rows = read_jsonl(AUTO / "all_teacher_label_rows.jsonl")

    # Add tolerant parser and expected flags.
    fixed_rows = []
    parser_repairs = []
    for r in rows:
        rr = dict(r)
        rr["tol_label"] = tolerant_parse(r.get("output_raw", ""))
        rr["tol_expected"] = rr["tol_label"] == rr["gold_label"]
        if r.get("parsed_label") is None and rr["tol_label"] is not None:
            parser_repairs.append(rr)
        fixed_rows.append(rr)
    rows = fixed_rows
    valid = [r for r in rows if r.get("tol_label")]
    expected = [r for r in valid if r["tol_expected"]]

    # Pair rows by exact prompt id across teachers.
    by_id: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["teacher"]] = r
    prompt_pairs = []
    for pid, d in sorted(by_id.items()):
        if all(t in d for t in TEACHERS):
            q, l = d[TEACHERS[0]], d[TEACHERS[1]]
            prompt_pairs.append({
                "id": pid,
                "case_id": q["case_id"],
                "context_type": q["context_type"],
                "fact_id": q["fact_id"],
                "role_type": q["role_type"],
                "gold_label": q["gold_label"],
                "qwen_label": q.get("tol_label"),
                "llama_label": l.get("tol_label"),
                "qwen_expected": q.get("tol_expected"),
                "llama_expected": l.get("tol_expected"),
                "same_label": bool(q.get("tol_label") and l.get("tol_label") and q.get("tol_label") == l.get("tol_label")),
                "both_expected": bool(q.get("tol_expected") and l.get("tol_expected")),
                "hypothesis": q.get("hypothesis"),
                "context": q.get("context"),
            })

    agreement_valid = [p for p in prompt_pairs if p["qwen_label"] and p["llama_label"]]
    both_expected = [p for p in agreement_valid if p["both_expected"]]
    same_label = [p for p in agreement_valid if p["same_label"]]

    # Recompute invariance with tolerant labels.
    inv_rows = []
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        groups[(r["teacher"], r["case_id"], r["fact_id"])] [r["context_type"]] = r
    for (teacher, cid, fid), d in sorted(groups.items()):
        labels = {ct: d.get(ct, {}).get("tol_label") for ct in CONTEXTS}
        gold = next(iter(d.values()))["gold_label"]
        all_valid = all(labels.values())
        same_all = bool(all_valid and len(set(labels.values())) == 1)
        expected_all = bool(same_all and next(iter(labels.values())) == gold)
        src_bridge_expected = bool(labels.get("source") == gold and labels.get("source_attested_bridge") == gold)
        inv_rows.append({
            "teacher": teacher,
            "case_id": cid,
            "fact_id": fid,
            "gold_label": gold,
            "labels": labels,
            "expected_all_contexts": expected_all,
            "source_bridge_expected": src_bridge_expected,
            "hypothesis": next(iter(d.values()))["hypothesis"],
        })

    # Case-level strong retained set: all teachers and all contexts correct after parser repair.
    case_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        case_groups[r["case_id"]].append(r)
    retained_cases = []
    weak_cases = []
    for cid, rs in sorted(case_groups.items()):
        total = len(rs)
        valid_n = sum(1 for r in rs if r.get("tol_label"))
        exp_n = sum(1 for r in rs if r.get("tol_expected"))
        row = {
            "case_id": cid,
            "n": total,
            "valid": valid_n,
            "expected": exp_n,
            "expected_rate": rate(exp_n, valid_n),
            "all_expected_after_parser_repair": valid_n == total and exp_n == total,
            "tier": rs[0].get("tier") if rs else None,
            "edit_class": rs[0].get("edit_class") if rs else None,
        }
        (retained_cases if row["all_expected_after_parser_repair"] else weak_cases).append(row)

    # Candidate/fact-generation failure signatures.
    def labels_for(cid: str, fid: str, ctx: str) -> list[str | None]:
        return [r.get("tol_label") for r in rows if r["case_id"] == cid and r["fact_id"] == fid and r["context_type"] == ctx]

    fact_level = []
    for c in cases:
        if not c.get("ok_case"):
            continue
        for f in c.get("facts", []):
            cid = c["case_id"]
            fid = f["fact_id"]
            gold = f["gold_label"]
            lbls = {ctx: labels_for(cid, fid, ctx) for ctx in CONTEXTS}
            ql = {ctx: sorted([x for x in lbls[ctx] if x]) for ctx in CONTEXTS}
            both = {ctx: (len(lbls[ctx]) == 2 and all(x == gold for x in lbls[ctx])) for ctx in CONTEXTS}
            non = {ctx: (len(lbls[ctx]) == 2 and all(x and x != gold for x in lbls[ctx])) for ctx in CONTEXTS}
            categories = []
            if gold == "ENTAILED" and both["source"] and non["source_attested_bridge"]:
                categories.append("bridge_lost_entailed_fact")
            if gold == "ENTAILED" and both["source"] and both["source_attested_bridge"] and non["natural_compact_reference"]:
                categories.append("natural_reference_lost_entailed_fact")
            if gold == "ENTAILED" and non["source"]:
                categories.append("generated_entailed_fact_not_supported_by_source")
            if gold == "NOT_ENTAILED" and all(non[ctx] for ctx in CONTEXTS):
                categories.append("generated_negative_likely_wrong_or_ambiguous")
            if gold == "NOT_ENTAILED" and non["source"]:
                categories.append("negative_entails_in_source")
            if not categories:
                if all(both[ctx] for ctx in CONTEXTS):
                    categories.append("stable")
                else:
                    categories.append("teacher_or_context_disagreement")
            fact_level.append({
                "case_id": cid,
                "fact_id": fid,
                "gold_label": gold,
                "role_type": f.get("role_type"),
                "hypothesis": f.get("hypothesis"),
                "labels_by_context": ql,
                "categories": categories,
                "source_text": c.get("source_text"),
                "source_attested_bridge": c.get("source_attested_bridge"),
                "natural_compact_reference": c.get("natural_compact_reference"),
            })

    cat_counts = collections.Counter(cat for fl in fact_level for cat in fl["categories"])

    # Teacher behavior, especially false positives on NOT_ENTAILED rows.
    by_teacher_label = {}
    for t in TEACHERS:
        for g in ["ENTAILED", "NOT_ENTAILED"]:
            sub = [r for r in rows if r["teacher"] == t and r["gold_label"] == g and r.get("tol_label")]
            by_teacher_label[f"{t}|{g}"] = {
                "n": len(sub),
                "expected": sum(1 for r in sub if r["tol_expected"]),
                "expected_rate": rate(sum(1 for r in sub if r["tol_expected"]), len(sub)),
                "entailed_outputs": sum(1 for r in sub if r["tol_label"] == "ENTAILED"),
                "not_entailed_outputs": sum(1 for r in sub if r["tol_label"] == "NOT_ENTAILED"),
            }

    # Example sets.
    parser_examples = [compact_example(r) for r in parser_repairs[:12]]
    both_wrong = [p for p in prompt_pairs if p["qwen_label"] == p["llama_label"] and p["qwen_label"] and p["qwen_label"] != p["gold_label"]]
    llama_only_wrong = [p for p in prompt_pairs if p["qwen_expected"] and not p["llama_expected"]]
    qwen_only_wrong = [p for p in prompt_pairs if p["llama_expected"] and not p["qwen_expected"]]
    bridge_lost = [x for x in fact_level if "bridge_lost_entailed_fact" in x["categories"]]
    natural_lost = [x for x in fact_level if "natural_reference_lost_entailed_fact" in x["categories"]]
    bad_negative = [x for x in fact_level if "generated_negative_likely_wrong_or_ambiguous" in x["categories"] or "negative_entails_in_source" in x["categories"]]
    source_overreach = [x for x in fact_level if "generated_entailed_fact_not_supported_by_source" in x["categories"]]

    summary = {
        "status": "ROLE_SUBSTRATE_REVIEW",
        "created_utc": now(),
        "input_step247_manual_summary": str(MANUAL / "teacher_panel_summary.json"),
        "input_step247_auto_summary": str(AUTO / "auto_role_fact_substrate_summary.json"),
        "manual_panel": {
            "rows": manual_summary.get("n_rows") or manual_summary.get("rows"),
            "overall_accuracy": manual_summary.get("overall_accuracy"),
            "bridge_accuracy": manual_summary.get("by_context", {}).get("source_attested_bridge", {}).get("accuracy"),
            "source_bridge_invariance": manual_summary.get("source_bridge_gold_consistent_invariance_rate"),
        },
        "original_auto_summary_core": {
            "n_teacher_rows": auto_summary["n_teacher_rows"],
            "valid_teacher_rows": auto_summary["valid_teacher_rows"],
            "overall_expected_consistency": auto_summary["overall_expected_consistency"],
            "cross_teacher_label_agreement": auto_summary["cross_teacher_label_agreement"],
            "cross_teacher_both_expected": auto_summary["cross_teacher_both_expected"],
            "source_bridge_expected_invariance": auto_summary["source_bridge_expected_invariance"],
            "bridge_false_rejection_expected_consistency": auto_summary["bridge_false_rejection_expected_consistency"],
            "case_all_expected_rate": auto_summary["case_all_expected_rate"],
        },
        "after_benign_parser_repair": {
            "valid_teacher_rows": len(valid),
            "parser_repaired_rows": len(parser_repairs),
            "overall_expected_consistency": rate(len(expected), len(valid)),
            "cross_teacher_label_agreement": rate(sum(1 for p in agreement_valid if p["same_label"]), len(agreement_valid)),
            "cross_teacher_both_expected": rate(len(both_expected), len(agreement_valid)),
            "source_bridge_expected_invariance": rate(sum(1 for x in inv_rows if x["source_bridge_expected"]), len([x for x in inv_rows if x["labels"].get("source") and x["labels"].get("source_attested_bridge")])) ,
            "context_all_expected_invariance": rate(sum(1 for x in inv_rows if x["expected_all_contexts"]), len([x for x in inv_rows if all(x["labels"].values())])),
            "cases_all_expected": len(retained_cases),
            "cases_total": len(case_groups),
            "case_all_expected_rate": rate(len(retained_cases), len(case_groups)),
        },
        "by_teacher_after_parser_repair": summarize(rows, "teacher"),
        "by_context_after_parser_repair": summarize(rows, "context_type"),
        "by_label_after_parser_repair": summarize(rows, "gold_label"),
        "by_tier_after_parser_repair": summarize(rows, "tier"),
        "by_teacher_label_after_parser_repair": by_teacher_label,
        "case_sets": {
            "strict_retained_case_ids": [c["case_id"] for c in retained_cases],
            "strict_retained_cases": retained_cases,
            "weak_cases_head": weak_cases[:20],
        },
        "failure_categories_fact_level_counts": dict(cat_counts),
        "failure_interpretation": {
            "parser_artifact": "11 Qwen outputs spell ENTAILED as ENTAILLED; treating that as ENTAILED removes invalid rows but leaves cross-teacher agreement and false-role rejection below desired reliability.",
            "bridge_or_reference_semantic_loss": "Several generated entailed facts are supported by the original source but not by the source-attested bridge or natural compact reference, meaning transformation quality and common-fact selection must be filtered before labeling.",
            "fact_generation_overreach_or_bad_negatives": "Some automatically generated NOT_ENTAILED hypotheses are actually inferable from the context, and some ENTAILED facts are over-specific or missing in one transformed sentence.",
            "teacher_behavior": "Qwen is stricter on positives in some contexts, while Llama over-accepts many false role-swap or entity-swap hypotheses; using either teacher alone would imprint model-specific bias.",
        },
        "examples": {
            "parser_repairs": parser_examples,
            "both_teachers_wrong_same_label": both_wrong[:12],
            "llama_only_wrong": llama_only_wrong[:12],
            "qwen_only_wrong": qwen_only_wrong[:12],
            "bridge_lost_entailed_fact": bridge_lost[:8],
            "natural_reference_lost_entailed_fact": natural_lost[:8],
            "generated_negative_wrong_or_ambiguous": bad_negative[:8],
            "generated_entailed_overreach_source": source_overreach[:8],
        },
        "scientific_decision": "repair_source_attested_role_substrate_before_any_student_or_babylm_training",
        "next_execution_object": "Build a clean automatic v2 substrate: shorter exact two-positive/two-negative fact generation, source/bridge/natural entailment prefilter by two teachers, reject any case where source-attested bridge loses a positive fact or a false fact is accepted by either teacher, normalize harmless label spelling, keep only strict retained cases, then connect retained facts to research EWoK/Entity item panel before a tiny student test.",
    }

    write_jsonl(OUT / "fact_level_failure_categories.jsonl", fact_level)
    write_jsonl(OUT / "prompt_pair_teacher_comparison_tolerant.jsonl", prompt_pairs)
    write_jsonl(OUT / "case_retention_after_parser_repair.jsonl", retained_cases + weak_cases)
    (OUT / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown note for later agents.
    lines: list[str] = []
    lines += [
        "# research — Role-substrate review and repair plan",
        "",
        "## Scientific purpose",
        "",
        "research showed a strong hand-built role-label signal on 12 curated source-attested transformation cases, but the automatic 53-case pipeline was mixed. This review asks whether that mixed result is a harmless parser problem, a candidate-quality problem, a generated-label problem, a teacher-disagreement problem, or a deeper reason to stop the teacher-role route before training.",
        "",
        "## What survives",
        "",
        f"- Manual panel: overall accuracy {manual_summary.get('overall_accuracy'):.3f}, source-attested bridge accuracy {manual_summary.get('by_context', {}).get('source_attested_bridge', {}).get('accuracy'):.3f}, false role-swap rejection {manual_summary.get('false_reject_bridge_accuracy'):.3f}, source↔bridge invariance {manual_summary.get('source_bridge_gold_consistent_invariance_rate'):.3f}.",
        f"- Automatic panel original: {auto_summary['valid_teacher_rows']}/{auto_summary['n_teacher_rows']} parsed; expected-label consistency {auto_summary['overall_expected_consistency']:.3f}; cross-teacher agreement {auto_summary['cross_teacher_label_agreement']:.3f}; both-teachers-expected {auto_summary['cross_teacher_both_expected']:.3f}; source↔bridge expected invariance {auto_summary['source_bridge_expected_invariance']:.3f}; bridge false-role rejection {auto_summary['bridge_false_rejection_expected_consistency']:.3f}; all-row stable cases {auto_summary['cases_all_expected']}/{auto_summary['cases_total']}.",
        f"- Benign parser repair maps 11 `ENTAILLED` outputs to `ENTAILED`: {len(valid)}/{len(rows)} rows valid, expected-label consistency {summary['after_benign_parser_repair']['overall_expected_consistency']:.3f}, cross-teacher agreement {summary['after_benign_parser_repair']['cross_teacher_label_agreement']:.3f}, both-teachers-expected {summary['after_benign_parser_repair']['cross_teacher_both_expected']:.3f}, strict retained cases {len(retained_cases)}/{len(case_groups)}.",
        "",
        "The hand-built result is real: approved teachers can judge role-preserving natural/source-attested transformations sentence-by-sentence. The automatic substrate is not yet reliable enough to train from.",
        "",
        "## Failure-mode separation",
        "",
        "1. **Parser artifact is small and repairable.** All invalid labels are harmless Qwen spellings such as `ENTAILLED`. Repairing them removes invalid rows but does not lift cross-teacher agreement or negative rejection to the level needed for a learning substrate.",
        "2. **Bridge/reference semantic loss is common enough to matter.** Some source-attested bridges drop source facts while the fact-generation prompt required support from all three sentences. Example: B018 generated `The American Bulldog is brave and protective`; the source supports it, but the source-attested bridge says only that the dog is best when trained young. Training on this would punish a student for correctly noticing missing information.",
        "3. **Generated negatives are sometimes not actually false.** B045 labels `Children learn the material because they want to be competitive in the game` as NOT_ENTAILED, but the source says the game lets children be competitive, want to win, and therefore want to learn the material. B059 similarly labels a same-token paraphrase as an agent-patient swap when the context supports it. These are not role-learning examples; they are wrong targets.",
        "4. **Teachers differ systematically.** Qwen is stricter on some positive hypotheses in compressed or altered contexts; Llama over-accepts many false entity/role swaps. The automatic substrate cannot use a single-teacher label stream. Retention must require agreement on the exact one-sentence prompt, not aggregate accuracy.",
        "5. **The usable core is real but small.** Strict retained cases after benign parser repair are: " + ", ".join([c["case_id"] for c in retained_cases]) + ". These are the seed cases for v2 construction, not a training corpus by themselves.",
        "",
        "## Consequence for the BabyLM research route",
        "",
        "This review refines the scientific conclusion: the route should not return to compact-view tuning, score patches, or BabyLM training. The live scientific object is a natural, source-attested role-assignment signal that can teach mention-to-entity, event-to-affected-entity, and query-to-entity roles without hard coordinates. research proves existence in curated cases, but the automatic version currently mixes clean role facts with missing facts, wrong false hypotheses, and teacher-specific biases. Training now would likely reproduce label noise rather than address the research/244/245 binding failure.",
        "",
        "## Smallest Useful Repair",
        "",
        "- Build `auto_role_fact_substrate_v2.py` from research, not a BabyLM trainer.",
        "- Generate exactly two positives and two negatives, but make each positive extractive or near-extractive from the intersection of SOURCE, SOURCE_ATTESTED_BRIDGE, and NATURAL_COMPACT_REFERENCE; reject facts mentioning tokens absent from the target context unless an exact alias table is present.",
        "- Generate negatives only by one controlled swap at a time: actor/patient, cause/effect, entity/state, location/object, or outcome-to-wrong-entity. Reject negatives that either teacher accepts in any context.",
        "- Normalize harmless output variants such as `ENTAILLED` before scoring.",
        "- Keep a case only if Qwen and Llama both give the intended label for every retained fact in SOURCE and SOURCE_ATTESTED_BRIDGE, and preferably NATURAL_COMPACT_REFERENCE; if natural compact fails but source↔bridge is stable, save it as a source-bridge-only split rather than discarding the whole case.",
        "- Preserve row-level reasons: parser spelling, bridge lost positive fact, natural reference lost positive fact, generated negative accepted, generated positive unsupported by source, Qwen-only miss, Llama-only miss.",
        "- Connect retained v2 facts to the research EWoK bridge panel by role type and conditional-reversal family before any student distillation. A tiny student test is only scientifically useful if it predicts held source-attested role facts and moves the same EWoK/Entity-like item groups, not just teacher labels on easy paraphrases.",
        "",
        "## Files produced by this review",
        "",
        f"- summary JSON: `{OUT / 'review_summary.json'}`",
        f"- fact-level categories: `{OUT / 'fact_level_failure_categories.jsonl'}`",
        f"- tolerant teacher-pair rows: `{OUT / 'prompt_pair_teacher_comparison_tolerant.jsonl'}`",
        f"- case retention table: `{OUT / 'case_retention_after_parser_repair.jsonl'}`",
    ]
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "valid_after_parser_repair": summary["after_benign_parser_repair"]["valid_teacher_rows"],
        "overall_after_parser_repair": summary["after_benign_parser_repair"]["overall_expected_consistency"],
        "cross_teacher_after_parser_repair": summary["after_benign_parser_repair"]["cross_teacher_label_agreement"],
        "both_expected_after_parser_repair": summary["after_benign_parser_repair"]["cross_teacher_both_expected"],
        "retained_cases": summary["case_sets"]["strict_retained_case_ids"],
        "category_counts": summary["failure_categories_fact_level_counts"],
        "summary_path": str(OUT / "review_summary.json"),
        "note_path": str(NOTE),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
