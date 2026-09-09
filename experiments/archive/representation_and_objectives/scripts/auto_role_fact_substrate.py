#!/usr/bin/env python3
"""research: automatic role-fact substrate probe for source-attested bridge transformations.

The manual research panel showed that Qwen3.5-9B can label hand-authored
role-sensitive hypotheses on 12 bridge cases when each sentence is encoded
independently. This script tests whether that premise scales beyond manual
hypotheses: use Qwen to propose common role facts over Tier-A/B transformation
candidates, then use independent one-sentence teacher labeling (Qwen and Llama)
to test whether the expected labels are stable across source, source-attested
bridge, and natural compact reference.

No BabyLM model is trained/evaluated/submitted. This is a route-admission test
for a possible teacher-role distillation substrate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

ROOT = Path("experiments/archive/representation_and_objectives")
A02_RECORDS = Path("experiments/archive/frontier_consolidation/data/bridge_high_fidelity_candidates/bridge_high_fidelity_candidates.json")
OUT_DIR = ROOT / "data/auto_role_fact_substrate"
PROMPT_DIR = ROOT / "training/data/auto_role_fact_substrate"
TIERS = {"A_seed_transform", "B_adjudicate_transform"}
CONTEXT_TYPES = [
    ("source", "source_text"),
    ("source_attested_bridge", "source_attested_bridge"),
    ("natural_compact_reference", "natural_compact_reference"),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_a02_records() -> list[dict[str, Any]]:
    obj = json.loads(A02_RECORDS.read_text(encoding="utf-8"))
    return [r for r in obj.get("records", []) if r.get("tier") in TIERS]


def build_fact_prompt(r: dict[str, Any]) -> str:
    return (
        "You are designing a semantic-role learning signal from natural sentences.\n"
        "You will see three sentences intended to express the same main proposition: SOURCE, SOURCE_ATTESTED_BRIDGE, and NATURAL_COMPACT_REFERENCE.\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        "{\"entailed\":[{\"id\":\"t1\",\"hypothesis\":\"...\",\"role_type\":\"...\"},{\"id\":\"t2\",\"hypothesis\":\"...\",\"role_type\":\"...\"}],\"not_entailed\":[{\"id\":\"f1\",\"hypothesis\":\"...\",\"role_type\":\"...\"},{\"id\":\"f2\",\"hypothesis\":\"...\",\"role_type\":\"...\"}]}\n\n"
        "Rules for ENTAILED hypotheses:\n"
        "- Each must be directly supported by ALL THREE sentences.\n"
        "- Each must describe a concrete role relation, event, state, cause, comparison, ownership, action, or entity-specific outcome.\n"
        "- Avoid vague statements like 'the sentence mentions X'.\n\n"
        "Rules for NOT_ENTAILED hypotheses:\n"
        "- Each should be plausible English but NOT supported by any of the three sentences.\n"
        "- Create them by swapping actor/patient, cause/effect, object/instrument, location/entity, or by assigning an outcome to the wrong entity.\n"
        "- Do not make absurd word salad; the false sentence should be a realistic conditional-reversal error a small language model might make.\n\n"
        "Keep each hypothesis short, self-contained, and answerable from a single sentence alone.\n\n"
        f"CASE_ID: {r['case_id']}\n"
        f"SOURCE: {r['source_text']}\n"
        f"SOURCE_ATTESTED_BRIDGE: {r['candidate_text']}\n"
        f"NATURAL_COMPACT_REFERENCE: {r['natural_compact_text']}\n"
        "JSON:"
    )


def prepare_facts(args: argparse.Namespace) -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_a02_records()
    rows = rows[:args.max_cases] if args.max_cases else rows
    prompts = []
    for r in rows:
        prompts.append({
            "id": f"factgen_{r['case_id']}",
            "prompt_id": f"factgen_{r['case_id']}",
            "case_id": r["case_id"],
            "pair_id": r.get("pair_id"),
            "tier": r.get("tier"),
            "edit_class": r.get("edit_class"),
            "bucket": r.get("bucket"),
            "source_text": r.get("source_text"),
            "source_attested_bridge": r.get("candidate_text"),
            "natural_compact_reference": r.get("natural_compact_text"),
            "prompt": build_fact_prompt(r),
        })
    write_jsonl(PROMPT_DIR / "fact_generation_prompts.jsonl", prompts)
    manifest = {
        "status": "AUTO_ROLE_FACT_PROMPTS_PREPARED",
        "created_utc": now(),
        "a02_records": str(A02_RECORDS),
        "a02_records_sha256": sha256_file(A02_RECORDS),
        "selected_tiers": sorted(TIERS),
        "n_cases": len(prompts),
        "fact_generation_prompts": str(PROMPT_DIR / "fact_generation_prompts.jsonl"),
        "command": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {PROMPT_DIR / 'fact_generation_prompts.jsonl'} --output-jsonl {PROMPT_DIR / 'fact_generation_outputs.jsonl'} --batch-size 16 --max-new-tokens 260 --temperature 0.1 --device cuda",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "fact_generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def clean_json_text(raw: str) -> str:
    s = str(raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        s = s[first:last+1]
    return s


def parse_fact_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    s = clean_json_text(raw)
    try:
        obj = json.loads(s)
    except Exception as e:
        return None, f"json_parse_error:{type(e).__name__}:{e}"
    if not isinstance(obj, dict):
        return None, "not_object"
    ent = obj.get("entailed")
    ne = obj.get("not_entailed")
    if not isinstance(ent, list) or not isinstance(ne, list):
        return None, "missing_lists"
    return obj, None


def norm_hyp(h: str) -> str:
    h = re.sub(r"\s+", " ", str(h or "").strip())
    return h.strip(" \t\r\n\"'")


def build_label_prompt(context: str, hypothesis: str) -> str:
    return (
        "Use only the CONTEXT sentence. Decide whether the HYPOTHESIS is directly supported.\n"
        "Track semantic roles carefully: who did what to whom, what caused what, which entity has which state, and what stayed unchanged.\n"
        "If the hypothesis swaps actor/patient, cause/effect, entity/state, location/object, or assigns an outcome to the wrong entity, answer NOT_ENTAILED.\n"
        "Answer exactly one label: ENTAILED or NOT_ENTAILED. No explanation.\n\n"
        f"CONTEXT: {context}\n"
        f"HYPOTHESIS: {hypothesis}\n"
        "LABEL:"
    )


def analyze_facts(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(PROMPT_DIR / "fact_generation_prompts.jsonl")
    outs = read_jsonl(Path(args.fact_outputs))
    if len(prompts) != len(outs):
        raise RuntimeError(f"fact prompt/output count mismatch {len(prompts)} vs {len(outs)}")
    cases = []
    label_rows = []
    parse_errors = []
    for p, o in zip(prompts, outs):
        raw = str(o.get("output") or o.get("generated_text") or o.get("text") or "")
        obj, err = parse_fact_json(raw)
        facts = []
        if err:
            parse_errors.append({"case_id": p["case_id"], "error": err, "raw": raw[:1000]})
        else:
            for gold_key, gold_label in [("entailed", "ENTAILED"), ("not_entailed", "NOT_ENTAILED")]:
                for j, item in enumerate(obj.get(gold_key, [])[:2]):
                    hyp = norm_hyp(item.get("hypothesis", "")) if isinstance(item, dict) else ""
                    role_type = str(item.get("role_type", gold_key)) if isinstance(item, dict) else gold_key
                    if not hyp:
                        continue
                    fid = f"{gold_key[0]}{j+1}"
                    facts.append({"fact_id": fid, "gold_label": gold_label, "role_type": role_type, "hypothesis": hyp})
        ok_case = len([f for f in facts if f["gold_label"] == "ENTAILED"]) == 2 and len([f for f in facts if f["gold_label"] == "NOT_ENTAILED"]) == 2
        cases.append({
            **{k: p[k] for k in ["case_id", "pair_id", "tier", "edit_class", "bucket", "source_text", "source_attested_bridge", "natural_compact_reference"]},
            "fact_generation_raw": raw,
            "parse_error": err,
            "facts": facts,
            "ok_case": ok_case,
        })
        if not ok_case:
            continue
        for ctx_type, field in CONTEXT_TYPES:
            context = p[field]
            for f in facts:
                lid = f"label_{p['case_id']}_{ctx_type}_{f['fact_id']}"
                label_rows.append({
                    "id": lid,
                    "prompt_id": lid,
                    "case_id": p["case_id"],
                    "pair_id": p["pair_id"],
                    "tier": p["tier"],
                    "edit_class": p["edit_class"],
                    "bucket": p["bucket"],
                    "context_type": ctx_type,
                    "fact_id": f["fact_id"],
                    "role_type": f["role_type"],
                    "hypothesis": f["hypothesis"],
                    "gold_label": f["gold_label"],
                    "context": context,
                    "prompt": build_label_prompt(context, f["hypothesis"]),
                })
    write_jsonl(OUT_DIR / "generated_fact_cases.jsonl", cases)
    write_jsonl(PROMPT_DIR / "label_prompts_all.jsonl", label_rows)
    # Same prompts for both teachers. Also produce shards for future use if desired.
    write_jsonl(PROMPT_DIR / "label_prompts_qwen.jsonl", label_rows)
    write_jsonl(PROMPT_DIR / "label_prompts_llama.jsonl", label_rows)
    summary = {
        "status": "AUTO_ROLE_FACTS_ANALYZED_PROMPTS_READY",
        "created_utc": now(),
        "fact_outputs": str(Path(args.fact_outputs)),
        "cases_total": len(cases),
        "cases_parse_errors": len(parse_errors),
        "ok_cases": sum(1 for c in cases if c["ok_case"]),
        "label_prompts": len(label_rows),
        "parse_errors_sample": parse_errors[:10],
        "label_prompt_qwen": str(PROMPT_DIR / "label_prompts_qwen.jsonl"),
        "label_prompt_llama": str(PROMPT_DIR / "label_prompts_llama.jsonl"),
        "commands": {
            "qwen": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {PROMPT_DIR / 'label_prompts_qwen.jsonl'} --output-jsonl {PROMPT_DIR / 'label_outputs_qwen.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
            "llama": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model llama3.1-8b-instruct --prompts-jsonl {PROMPT_DIR / 'label_prompts_llama.jsonl'} --output-jsonl {PROMPT_DIR / 'label_outputs_llama.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "fact_generation_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def parse_label(output: str) -> str | None:
    s = str(output or "").strip().upper()
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\bNOT[_ ]?ENTAILED\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAILED\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def load_teacher_labels(model_name: str, prompt_path: Path, output_path: Path) -> list[dict[str, Any]]:
    prompts = read_jsonl(prompt_path)
    outs = read_jsonl(output_path)
    if len(prompts) != len(outs):
        raise RuntimeError(f"{model_name} prompt/output mismatch {len(prompts)} vs {len(outs)}")
    rows = []
    for i, (p, o) in enumerate(zip(prompts, outs)):
        raw = str(o.get("output") or o.get("generated_text") or o.get("text") or "")
        label = parse_label(raw)
        rows.append({
            **{k: p[k] for k in ["id", "case_id", "pair_id", "tier", "edit_class", "bucket", "context_type", "fact_id", "role_type", "hypothesis", "gold_label", "context"]},
            "teacher": model_name,
            "output_raw": raw,
            "parsed_label": label,
            "expected_consistent": (label == p["gold_label"]),
            "output_index": o.get("index", i),
        })
    return rows


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    for k in sorted({str(r.get(key)) for r in rows}):
        rs = [r for r in rows if str(r.get(key)) == k]
        valid = [r for r in rs if r.get("parsed_label")]
        out[k] = {
            "n": len(rs),
            "valid": len(valid),
            "expected_consistency": sum(1 for r in valid if r["expected_consistent"]) / max(1, len(valid)),
            "invalid": len(rs) - len(valid),
        }
    return out


def analyze_labels(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    rows += load_teacher_labels("qwen3.5-9b", PROMPT_DIR / "label_prompts_qwen.jsonl", Path(args.qwen_outputs))
    rows += load_teacher_labels("llama3.1-8b-instruct", PROMPT_DIR / "label_prompts_llama.jsonl", Path(args.llama_outputs))
    write_jsonl(OUT_DIR / "all_teacher_label_rows.jsonl", rows)

    valid = [r for r in rows if r.get("parsed_label")]
    by_teacher = summarize_by(rows, "teacher")
    by_context = summarize_by(rows, "context_type")
    by_label = summarize_by(rows, "gold_label")
    by_tier = summarize_by(rows, "tier")
    by_role_type = summarize_by(rows, "role_type")

    # Cross-teacher agreement on each exact prompt.
    by_id: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["teacher"]] = r
    agreement_rows = []
    for pid, d in sorted(by_id.items()):
        if "qwen3.5-9b" not in d or "llama3.1-8b-instruct" not in d:
            continue
        q, l = d["qwen3.5-9b"], d["llama3.1-8b-instruct"]
        same = bool(q.get("parsed_label") and l.get("parsed_label") and q["parsed_label"] == l["parsed_label"])
        expected_same = bool(same and q["parsed_label"] == q["gold_label"])
        agreement_rows.append({
            "id": pid,
            "case_id": q["case_id"],
            "context_type": q["context_type"],
            "fact_id": q["fact_id"],
            "role_type": q["role_type"],
            "gold_label": q["gold_label"],
            "qwen_label": q.get("parsed_label"),
            "llama_label": l.get("parsed_label"),
            "same_label": same,
            "both_expected": expected_same,
            "hypothesis": q["hypothesis"],
            "context": q["context"],
        })
    write_jsonl(OUT_DIR / "cross_teacher_agreement_rows.jsonl", agreement_rows)

    # Invariance across source/bridge/natural for each teacher+case+fact.
    inv = []
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        groups[(r["teacher"], r["case_id"], r["fact_id"])] [r["context_type"]] = r
    for (teacher, cid, fid), d in sorted(groups.items()):
        labels = {ct: d.get(ct, {}).get("parsed_label") for ct, _ in CONTEXT_TYPES}
        gold = next(iter(d.values()))["gold_label"]
        all_valid = all(labels[ct] for ct, _ in CONTEXT_TYPES)
        same_all = bool(all_valid and len(set(labels.values())) == 1)
        expected_all = bool(same_all and next(iter(labels.values())) == gold)
        source_bridge = bool(labels.get("source") and labels.get("source_attested_bridge") and labels["source"] == labels["source_attested_bridge"])
        expected_source_bridge = bool(source_bridge and labels["source"] == gold)
        inv.append({
            "teacher": teacher,
            "case_id": cid,
            "fact_id": fid,
            "gold_label": gold,
            "labels": labels,
            "same_all_contexts": same_all,
            "expected_all_contexts": expected_all,
            "source_bridge_same": source_bridge,
            "source_bridge_expected": expected_source_bridge,
            "hypothesis": next(iter(d.values()))["hypothesis"],
        })
    write_jsonl(OUT_DIR / "context_invariance_rows.jsonl", inv)

    # Case-level rates: all 4 facts x 3 contexts x 2 teachers expected-consistent.
    case_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        case_groups[r["case_id"]].append(r)
    case_summary = []
    for cid, rs in sorted(case_groups.items()):
        valid_rs = [r for r in rs if r.get("parsed_label")]
        case_summary.append({
            "case_id": cid,
            "n": len(rs),
            "valid": len(valid_rs),
            "expected_consistency": sum(1 for r in valid_rs if r["expected_consistent"]) / max(1, len(valid_rs)),
            "all_expected": len(valid_rs) == len(rs) and all(r["expected_consistent"] for r in valid_rs),
            "tier": rs[0].get("tier") if rs else None,
            "edit_class": rs[0].get("edit_class") if rs else None,
        })
    write_jsonl(OUT_DIR / "case_summary.jsonl", case_summary)

    agreement_valid = [r for r in agreement_rows if r["qwen_label"] and r["llama_label"]]
    inv_valid = [x for x in inv if all(x["labels"].values())]
    qwen_bridge = [r for r in rows if r["teacher"] == "qwen3.5-9b" and r["context_type"] == "source_attested_bridge" and r.get("parsed_label")]
    llama_bridge = [r for r in rows if r["teacher"] == "llama3.1-8b-instruct" and r["context_type"] == "source_attested_bridge" and r.get("parsed_label")]
    false_bridge = [r for r in rows if r["context_type"] == "source_attested_bridge" and r["gold_label"] == "NOT_ENTAILED" and r.get("parsed_label")]

    summary = {
        "status": "AUTO_ROLE_FACT_SUBSTRATE_ANALYZED",
        "created_utc": now(),
        "n_teacher_rows": len(rows),
        "valid_teacher_rows": len(valid),
        "overall_expected_consistency": sum(1 for r in valid if r["expected_consistent"]) / max(1, len(valid)),
        "by_teacher": by_teacher,
        "by_context": by_context,
        "by_label": by_label,
        "by_tier": by_tier,
        "by_role_type": by_role_type,
        "cross_teacher_label_agreement": sum(1 for r in agreement_valid if r["same_label"]) / max(1, len(agreement_valid)),
        "cross_teacher_both_expected": sum(1 for r in agreement_valid if r["both_expected"]) / max(1, len(agreement_valid)),
        "context_all_expected_invariance": sum(1 for x in inv_valid if x["expected_all_contexts"]) / max(1, len(inv_valid)),
        "source_bridge_expected_invariance": sum(1 for x in inv_valid if x["source_bridge_expected"]) / max(1, len(inv_valid)),
        "qwen_bridge_expected_consistency": sum(1 for r in qwen_bridge if r["expected_consistent"]) / max(1, len(qwen_bridge)),
        "llama_bridge_expected_consistency": sum(1 for r in llama_bridge if r["expected_consistent"]) / max(1, len(llama_bridge)),
        "bridge_false_rejection_expected_consistency": sum(1 for r in false_bridge if r["expected_consistent"]) / max(1, len(false_bridge)),
        "case_all_expected_rate": sum(1 for c in case_summary if c["all_expected"]) / max(1, len(case_summary)),
        "cases_total": len(case_summary),
        "cases_all_expected": sum(1 for c in case_summary if c["all_expected"]),
        "premise_positive_for_student_pilot": False,
        "decision": None,
        "thresholds": {
            "valid_teacher_rows": "all parseable or nearly all; inspect invalids",
            "overall_expected_consistency": ">=0.90",
            "cross_teacher_label_agreement": ">=0.90",
            "cross_teacher_both_expected": ">=0.85",
            "source_bridge_expected_invariance": ">=0.85",
            "qwen_bridge_expected_consistency": ">=0.90",
            "llama_bridge_expected_consistency": ">=0.85",
            "bridge_false_rejection_expected_consistency": ">=0.90",
        },
        "interpretation_boundary": "Expected labels come from Qwen-generated fact sets, so consistency is not final truth. Cross-teacher independent one-sentence agreement and manual-panel positivity together determine whether the substrate is worth a small student pilot; this is not BabyLM endpoint evidence.",
        "no_babylm_training_eval_upload_submission": True,
    }
    summary["premise_positive_for_student_pilot"] = (
        summary["valid_teacher_rows"] >= 0.98 * summary["n_teacher_rows"]
        and summary["overall_expected_consistency"] >= 0.90
        and summary["cross_teacher_label_agreement"] >= 0.90
        and summary["cross_teacher_both_expected"] >= 0.85
        and summary["source_bridge_expected_invariance"] >= 0.85
        and summary["qwen_bridge_expected_consistency"] >= 0.90
        and summary["llama_bridge_expected_consistency"] >= 0.85
        and summary["bridge_false_rejection_expected_consistency"] >= 0.90
    )
    summary["decision"] = "POSITIVE_SCALABLE_ROLE_LABEL_PREMISE_SMALL_STUDENT_PILOT_ALLOWED" if summary["premise_positive_for_student_pilot"] else "BORDERLINE_OR_NEGATIVE_ROLE_LABEL_PREMISE_REPAIR_BEFORE_STUDENT"
    summary["next_action"] = (
        "Build a small counted-word source-attested role-label corpus and run a cheap student distillation pilot against held-out bridge facts plus research EWoK/Entity item panel."
        if summary["premise_positive_for_student_pilot"] else
        "Inspect disagreement/failure modes and repair fact generation or candidate selection before any student or architecture training."
    )
    (OUT_DIR / "auto_role_fact_substrate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    fail_rows = [r for r in rows if (not r.get("parsed_label")) or not r.get("expected_consistent")]
    disagree = [r for r in agreement_rows if not r["both_expected"]]
    md = [
        "# research automatic source-attested role-fact substrate probe", "",
        "## Purpose", "",
        "The manual panel was positive but small. This probe asks whether A02's A/B source-attested bridge transformations can automatically yield common role facts whose expected labels remain stable when each source, bridge, and natural compact sentence is encoded independently. Qwen generated facts; Qwen and Llama separately labelled the resulting one-sentence entailment probes.", "",
        "## Aggregate", "",
        f"- teacher label rows: {len(rows)}, valid: {len(valid)}",
        f"- overall expected-label consistency: {summary['overall_expected_consistency']:.3f}",
        f"- cross-teacher label agreement: {summary['cross_teacher_label_agreement']:.3f}",
        f"- cross-teacher both expected: {summary['cross_teacher_both_expected']:.3f}",
        f"- source↔bridge expected invariance: {summary['source_bridge_expected_invariance']:.3f}",
        f"- qwen bridge expected consistency: {summary['qwen_bridge_expected_consistency']:.3f}",
        f"- llama bridge expected consistency: {summary['llama_bridge_expected_consistency']:.3f}",
        f"- bridge false-role rejection consistency: {summary['bridge_false_rejection_expected_consistency']:.3f}",
        f"- case all-expected rate: {summary['case_all_expected_rate']:.3f} ({summary['cases_all_expected']}/{summary['cases_total']})",
        f"- decision: **{summary['decision']}**", "",
        "## By teacher", "", "| teacher | n | expected consistency | invalid |", "|---|---:|---:|---:|",
    ]
    for k, v in by_teacher.items():
        md.append(f"| {k} | {v['n']} | {v['expected_consistency']:.3f} | {v['invalid']} |")
    md += ["", "## By context", "", "| context | n | expected consistency | invalid |", "|---|---:|---:|---:|"]
    for k, v in by_context.items():
        md.append(f"| {k} | {v['n']} | {v['expected_consistency']:.3f} | {v['invalid']} |")
    md += ["", "## Failure/disagreement sample", ""]
    for r in fail_rows[:24]:
        md += [
            f"### {r['teacher']} {r['id']}",
            f"- gold={r['gold_label']} parsed={r['parsed_label']} raw={r['output_raw']!r} context={r['context_type']} tier={r['tier']} role={r['role_type']}",
            f"- hypothesis: {r['hypothesis']}",
            f"- context: {r['context']}",
            "",
        ]
    if disagree:
        md += ["## Cross-teacher non-both-expected sample", ""]
        for r in disagree[:24]:
            md += [
                f"### {r['id']}",
                f"- gold={r['gold_label']} qwen={r['qwen_label']} llama={r['llama_label']} context={r['context_type']} role={r['role_type']}",
                f"- hypothesis: {r['hypothesis']}",
                "",
            ]
    md += ["## Scientific meaning", "", summary["next_action"], "",
           "Boundary: this establishes only a role-label premise for a small student pilot. It does not reopen compact-view tuning or justify 100M BabyLM training by itself.", "",
           "## Files", "",
           f"- generated fact cases: `{OUT_DIR / 'generated_fact_cases.jsonl'}`",
           f"- all teacher label rows: `{OUT_DIR / 'all_teacher_label_rows.jsonl'}`",
           f"- context invariance rows: `{OUT_DIR / 'context_invariance_rows.jsonl'}`",
           f"- summary JSON: `{OUT_DIR / 'auto_role_fact_substrate_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate/auto_role_fact_substrate_summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "overall_expected_consistency": summary["overall_expected_consistency"],
        "cross_teacher_label_agreement": summary["cross_teacher_label_agreement"],
        "source_bridge_expected_invariance": summary["source_bridge_expected_invariance"],
        "premise_positive_for_student_pilot": summary["premise_positive_for_student_pilot"],
        "decision": summary["decision"],
        "summary_json": str(OUT_DIR / "auto_role_fact_substrate_summary.json"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate/auto_role_fact_substrate_summary.md')),
    }, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("prepare-facts")
    pf.add_argument("--max-cases", type=int, default=0, help="0 means all A/B cases")
    af = sub.add_parser("analyze-facts")
    af.add_argument("--fact-outputs", default=str(PROMPT_DIR / "fact_generation_outputs.jsonl"))
    al = sub.add_parser("analyze-labels")
    al.add_argument("--qwen-outputs", default=str(PROMPT_DIR / "label_outputs_qwen.jsonl"))
    al.add_argument("--llama-outputs", default=str(PROMPT_DIR / "label_outputs_llama.jsonl"))
    args = ap.parse_args()
    if args.cmd == "prepare-facts":
        prepare_facts(args)
    elif args.cmd == "analyze-facts":
        analyze_facts(args)
    elif args.cmd == "analyze-labels":
        analyze_labels(args)


if __name__ == "__main__":
    main()
