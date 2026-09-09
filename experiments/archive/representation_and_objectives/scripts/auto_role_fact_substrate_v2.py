#!/usr/bin/env python3
"""research: v2 source-attested role-exchange substrate repair.

Purpose
-------
research proved that approved teachers can label hand-authored role-preserving
facts on a tiny curated panel, but research showed that automatic labels over the
bridge collection retain only a small core and risk selecting easy lexical
entailments.  This v2 script tests a sharper object before any BabyLM or student
training: counterfactually closed role-exchange families.

Each generated family should contain:
  * p1: a source/bridge supported role relation;
  * p2: a paraphrase or untouched-stability statement that should remain true;
  * n1: a minimal role-exchange/query-change false statement;
  * n2: a second controlled false statement changing one role, relation, state,
        condition, location, or outcome.

The scorer keeps the natural/source-attested distinction explicit and interprets
teacher agreement together with coverage and family structure, not agreement
alone.  No BabyLM model is trained, evaluated, uploaded, or submitted.
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
EWOK_PANEL = ROOT / "data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl"
OUT_DIR = ROOT / "data/auto_role_fact_substrate_v2"
PROMPT_DIR = ROOT / "training/data/auto_role_fact_substrate_v2"
TIERS = {"A_seed_transform", "B_adjudicate_transform"}
CONTEXT_TYPES = [
    ("source", "source_text"),
    ("source_attested_bridge", "source_attested_bridge"),
    ("natural_compact_reference", "natural_compact_reference"),
]
TEACHERS = ["qwen3.5-9b", "llama3.1-8b-instruct"]
GOOD_SWAP_WORDS = {
    "actor", "agent", "patient", "theme", "object", "cause", "effect",
    "entity", "state", "location", "instrument", "beneficiary", "condition",
    "outcome", "comparison", "temporal", "time", "source", "target", "owner",
    "query", "role", "polarity",
}
LOW_VALUE_ROLE_WORDS = {
    "mention", "existence", "entity_existence", "lexical", "topic", "sentence",
    "attribute_only", "name", "definition_only",
}
ROLE_TO_EWOK = {
    "cause": {"physical-dynamics", "material-dynamics", "social-relations", "physical-relations"},
    "effect": {"physical-dynamics", "material-dynamics", "social-relations", "physical-relations"},
    "condition": {"physical-dynamics", "material-dynamics", "agent-properties", "social-properties"},
    "outcome": {"physical-dynamics", "material-dynamics", "social-properties", "physical-relations"},
    "agent": {"agent-properties", "social-relations"},
    "actor": {"agent-properties", "social-relations"},
    "patient": {"physical-relations", "social-relations"},
    "beneficiary": {"social-relations", "social-properties"},
    "instrument": {"physical-relations", "physical-dynamics"},
    "location": {"spatial-relations"},
    "spatial": {"spatial-relations"},
    "state": {"material-properties", "social-properties", "physical-relations"},
    "comparison": {"material-properties", "physical-relations", "social-properties"},
    "temporal": {"physical-dynamics", "social-relations"},
    "time": {"physical-dynamics", "social-relations"},
    "ownership": {"social-relations"},
    "possession": {"social-relations"},
    "recommendation": {"agent-properties", "social-relations"},
}


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
    rows = [r for r in obj.get("records", []) if r.get("tier") in TIERS]
    rows.sort(key=lambda r: str(r.get("case_id")))
    return rows


def compact_space(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def norm_hyp(h: Any) -> str:
    return compact_space(h).strip(" \t\r\n\"'")


def tokenize_words(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?", s.lower()))


def overlap_stats(hyp: str, context: str) -> dict[str, Any]:
    hw = tokenize_words(hyp)
    cw = tokenize_words(context)
    stop = {
        "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by",
        "from", "as", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been",
        "being", "has", "have", "had", "do", "does", "did", "will", "would", "can", "could", "may",
        "might", "must", "should", "it", "its", "they", "their", "them", "he", "she", "his", "her",
    }
    hc = hw - stop
    cc = cw - stop
    inter = hc & cc
    return {
        "hyp_content_words": len(hc),
        "context_content_words": len(cc),
        "content_overlap_words": len(inter),
        "content_overlap_frac": (len(inter) / len(hc)) if hc else 0.0,
        "hyp_only_words_sample": sorted(hc - cc)[:20],
    }


def build_family_prompt(r: dict[str, Any]) -> str:
    return (
        "You are building a small scientific test set for semantic-role assignment in natural sentences.\n"
        "The test should NOT reward easy lexical entailment. It should test whether a model keeps roles fixed when a sentence is paraphrased, and rejects a counterfactual where exactly one role/query/condition/outcome is exchanged.\n\n"
        "You will see SOURCE, SOURCE_ATTESTED_BRIDGE, and NATURAL_COMPACT_REFERENCE. The SOURCE_ATTESTED_BRIDGE was generated under source-attested constraints and is the main transformed sentence.\n\n"
        "Return ONLY valid JSON with this exact shape (one family, or an empty family list if no such role-exchange family exists):\n"
        "{\"families\":[{\"family_id\":\"r1\",\"role_type\":\"cause_effect|agent_patient|entity_state|location_relation|condition_outcome|comparison|beneficiary|instrument|temporal_order|other_role_relation\",\"role_slots\":{\"slot_a\":\"...\",\"relation\":\"...\",\"slot_b\":\"...\",\"outcome_or_state\":\"...\"},\"entailed\":[{\"id\":\"p1\",\"hypothesis\":\"...\",\"purpose\":\"main_relation\"},{\"id\":\"p2\",\"hypothesis\":\"...\",\"purpose\":\"paraphrase_or_untouched_stability\"}],\"not_entailed\":[{\"id\":\"n1\",\"hypothesis\":\"...\",\"swap_type\":\"role_exchange\"},{\"id\":\"n2\",\"hypothesis\":\"...\",\"swap_type\":\"wrong_query_or_single_role_change\"}],\"why_counterfactual_closed\":\"...\"}],\"reject_reason\":\"\"}\n\n"
        "Hard rules:\n"
        "1. Use one source-attested role relation supported by SOURCE and SOURCE_ATTESTED_BRIDGE. Prefer one also supported by NATURAL_COMPACT_REFERENCE, but do not force it if the natural reference dropped material.\n"
        "2. p1 and p2 must both be ENTAILED by SOURCE and SOURCE_ATTESTED_BRIDGE. p2 must either paraphrase p1 with a different surface form or state an untouched role/state that remains stable under the transformation.\n"
        "3. n1 and n2 must be NOT_ENTAILED by SOURCE and SOURCE_ATTESTED_BRIDGE. They must be minimally changed counterfactuals: actor/patient swap, cause/effect swap, entity/state swap, location/object swap, condition/outcome swap, beneficiary swap, comparison reversal, or assigning an outcome to the wrong entity.\n"
        "4. Do not use a false statement that adds an unrelated new fact, an obviously absurd word salad, a generic contradiction, or a mere topic/entity substitution. The false item should be a realistic conditional-reversal mistake.\n"
        "5. Avoid families whose positive is only 'X exists' or 'the sentence mentions X'. The family must involve at least two roles or one condition-to-outcome/state relation.\n"
        "6. Keep all four hypotheses short and answerable from a single context sentence alone.\n"
        "7. If the bridge deletes the role relation needed for p1/p2, return {\"families\":[],\"reject_reason\":\"bridge_lost_role_relation\"}. If no natural role-exchange family exists, return {\"families\":[],\"reject_reason\":\"no_counterfactual_role_family\"}.\n\n"
        f"CASE_ID: {r['case_id']}\n"
        f"SOURCE: {r['source_text']}\n"
        f"SOURCE_ATTESTED_BRIDGE: {r['candidate_text']}\n"
        f"NATURAL_COMPACT_REFERENCE: {r['natural_compact_text']}\n"
        "JSON:"
    )


def prepare(args: argparse.Namespace) -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_a02_records()
    if args.max_cases:
        rows = rows[: args.max_cases]
    prompts: list[dict[str, Any]] = []
    for r in rows:
        prompts.append({
            "id": f"familygen_{r['case_id']}",
            "prompt_id": f"familygen_{r['case_id']}",
            "case_id": r["case_id"],
            "pair_id": r.get("pair_id"),
            "tier": r.get("tier"),
            "edit_class": r.get("edit_class"),
            "bucket": r.get("bucket"),
            "source_text": r.get("source_text"),
            "source_attested_bridge": r.get("candidate_text"),
            "natural_compact_reference": r.get("natural_compact_text"),
            "prompt": build_family_prompt(r),
        })
    write_jsonl(PROMPT_DIR / "family_generation_prompts.jsonl", prompts)
    manifest = {
        "status": "V2_ROLE_FAMILY_PROMPTS_PREPARED",
        "created_utc": now(),
        "a02_records": str(A02_RECORDS),
        "a02_records_sha256": sha256_file(A02_RECORDS),
        "selected_tiers": sorted(TIERS),
        "n_cases": len(prompts),
        "family_generation_prompts": str(PROMPT_DIR / "family_generation_prompts.jsonl"),
        "suggested_generation_command": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {PROMPT_DIR / 'family_generation_prompts.jsonl'} --output-jsonl {PROMPT_DIR / 'family_generation_outputs_qwen.jsonl'} --batch-size 12 --max-new-tokens 420 --temperature 0.1 --device cuda",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "family_generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def clean_json_text(raw: str) -> str:
    s = str(raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        s = s[first : last + 1]
    return s


def parse_json_loose(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    s = clean_json_text(raw)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj, None
        return None, "not_object"
    except Exception as e:
        # Very small repair for trailing commas; do not invent missing content.
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            obj = json.loads(s2)
            if isinstance(obj, dict):
                return obj, "repaired_trailing_comma"
            return None, "not_object_after_repair"
        except Exception as e2:
            return None, f"json_parse_error:{type(e2).__name__}:{e2}"


def first_family(obj: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    fams = obj.get("families")
    if not isinstance(fams, list) or not fams:
        rr = obj.get("reject_reason") or "empty_families"
        return None, str(rr)
    fam = fams[0]
    if not isinstance(fam, dict):
        return None, "family_not_object"
    return fam, ""


def role_structural_score(role_type: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    role_norm = re.sub(r"[^a-z0-9]+", "_", str(role_type or "").lower()).strip("_")
    text = " ".join([role_norm] + [str(f.get("swap_type", "")) for f in facts]).lower()
    good_hits = sorted(w for w in GOOD_SWAP_WORDS if w in text)
    low_hits = sorted(w for w in LOW_VALUE_ROLE_WORDS if w in text)
    has_role_exchange_negative = any(
        f.get("gold_label") == "NOT_ENTAILED" and any(w in str(f.get("swap_type", "")).lower() or w in str(f.get("role_type", "")).lower() for w in GOOD_SWAP_WORDS)
        for f in facts
    )
    return {
        "role_type_norm": role_norm,
        "good_role_words": good_hits,
        "low_value_role_words": low_hits,
        "has_role_exchange_negative": has_role_exchange_negative,
        "structural_not_easy": bool(good_hits and not low_hits and has_role_exchange_negative),
    }


def parse_family_cases(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(PROMPT_DIR / "family_generation_prompts.jsonl")
    outs = read_jsonl(Path(args.family_outputs))
    if len(prompts) != len(outs):
        raise RuntimeError(f"prompt/output mismatch {len(prompts)} vs {len(outs)}")
    cases: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for p, o in zip(prompts, outs):
        raw = str(o.get("output") or o.get("generated_text") or o.get("text") or "")
        obj, err = parse_json_loose(raw)
        facts: list[dict[str, Any]] = []
        role_type = ""
        family_id = "r1"
        reject_reason = ""
        role_slots: dict[str, Any] = {}
        why_closed = ""
        if obj is None:
            parse_errors.append({"case_id": p["case_id"], "error": err, "raw_head": raw[:1200]})
            reject_reason = err or "parse_error"
        else:
            fam, reject_reason = first_family(obj)
            if fam is None:
                rejects.append({"case_id": p["case_id"], "reject_reason": reject_reason, "raw_head": raw[:1000]})
            else:
                role_type = compact_space(fam.get("role_type")) or "other_role_relation"
                family_id = compact_space(fam.get("family_id")) or "r1"
                role_slots = fam.get("role_slots") if isinstance(fam.get("role_slots"), dict) else {}
                why_closed = compact_space(fam.get("why_counterfactual_closed"))
                for gold_key, gold_label in [("entailed", "ENTAILED"), ("not_entailed", "NOT_ENTAILED")]:
                    items = fam.get(gold_key)
                    if not isinstance(items, list):
                        continue
                    for j, item in enumerate(items[:2]):
                        if not isinstance(item, dict):
                            continue
                        hyp = norm_hyp(item.get("hypothesis"))
                        if not hyp:
                            continue
                        if gold_label == "ENTAILED":
                            fid = f"p{j+1}"
                            purpose = compact_space(item.get("purpose")) or ("main_relation" if j == 0 else "paraphrase_or_untouched_stability")
                            facts.append({
                                "fact_id": fid,
                                "gold_label": gold_label,
                                "role_type": role_type,
                                "purpose": purpose,
                                "swap_type": "",
                                "hypothesis": hyp,
                            })
                        else:
                            fid = f"n{j+1}"
                            swap_type = compact_space(item.get("swap_type")) or ("role_exchange" if j == 0 else "single_role_change")
                            facts.append({
                                "fact_id": fid,
                                "gold_label": gold_label,
                                "role_type": role_type,
                                "purpose": "counterfactual",
                                "swap_type": swap_type,
                                "hypothesis": hyp,
                            })
        n_pos = sum(1 for f in facts if f["gold_label"] == "ENTAILED")
        n_neg = sum(1 for f in facts if f["gold_label"] == "NOT_ENTAILED")
        structure = role_structural_score(role_type, facts) if facts else {
            "role_type_norm": "", "good_role_words": [], "low_value_role_words": [], "has_role_exchange_negative": False, "structural_not_easy": False,
        }
        # lexical overlap is descriptive only. Low overlap can indicate aliases or bad facts; high overlap can indicate extraction.
        fact_overlaps = []
        for f in facts:
            fact_overlaps.append({
                "fact_id": f["fact_id"],
                "source_overlap": overlap_stats(f["hypothesis"], p["source_text"]),
                "bridge_overlap": overlap_stats(f["hypothesis"], p["source_attested_bridge"]),
                "natural_overlap": overlap_stats(f["hypothesis"], p["natural_compact_reference"]),
            })
        ok_family = (n_pos == 2 and n_neg == 2 and structure["has_role_exchange_negative"])
        case = {
            **{k: p[k] for k in ["case_id", "pair_id", "tier", "edit_class", "bucket", "source_text", "source_attested_bridge", "natural_compact_reference"]},
            "family_generation_raw": raw,
            "parse_note_or_error": err,
            "reject_reason": reject_reason,
            "family_id": family_id,
            "role_type": role_type,
            "role_slots": role_slots,
            "why_counterfactual_closed": why_closed,
            "facts": facts,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "structure": structure,
            "fact_overlaps": fact_overlaps,
            "ok_family": ok_family,
        }
        cases.append(case)
        if not ok_family:
            continue
        for ctx_type, field in CONTEXT_TYPES:
            context = p[field]
            for f in facts:
                lid = f"label_{p['case_id']}_{family_id}_{ctx_type}_{f['fact_id']}"
                label_rows.append({
                    "id": lid,
                    "prompt_id": lid,
                    "case_id": p["case_id"],
                    "pair_id": p["pair_id"],
                    "tier": p["tier"],
                    "edit_class": p["edit_class"],
                    "bucket": p["bucket"],
                    "family_id": family_id,
                    "context_type": ctx_type,
                    "fact_id": f["fact_id"],
                    "role_type": role_type,
                    "purpose": f.get("purpose", ""),
                    "swap_type": f.get("swap_type", ""),
                    "hypothesis": f["hypothesis"],
                    "gold_label": f["gold_label"],
                    "context": context,
                    "prompt": build_label_prompt(context, f["hypothesis"]),
                })
    write_jsonl(OUT_DIR / "generated_family_cases.jsonl", cases)
    write_jsonl(PROMPT_DIR / "label_prompts_qwen.jsonl", label_rows)
    write_jsonl(PROMPT_DIR / "label_prompts_llama.jsonl", label_rows)
    summary = {
        "status": "V2_ROLE_FAMILIES_PARSED_LABEL_PROMPTS_READY",
        "created_utc": now(),
        "family_outputs": str(Path(args.family_outputs)),
        "cases_total": len(cases),
        "json_parse_errors": len(parse_errors),
        "empty_or_model_rejects": len(rejects),
        "ok_counterfactual_families": sum(1 for c in cases if c["ok_family"]),
        "label_prompts_per_teacher": len(label_rows),
        "role_types_ok": collections.Counter(c["structure"]["role_type_norm"] for c in cases if c["ok_family"]),
        "parse_errors_sample": parse_errors[:10],
        "rejects_sample": rejects[:10],
        "label_prompt_qwen": str(PROMPT_DIR / "label_prompts_qwen.jsonl"),
        "label_prompt_llama": str(PROMPT_DIR / "label_prompts_llama.jsonl"),
        "suggested_label_commands": {
            "qwen": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {PROMPT_DIR / 'label_prompts_qwen.jsonl'} --output-jsonl {PROMPT_DIR / 'label_outputs_qwen.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
            "llama": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model llama3.1-8b-instruct --prompts-jsonl {PROMPT_DIR / 'label_prompts_llama.jsonl'} --output-jsonl {PROMPT_DIR / 'label_outputs_llama.jsonl'} --batch-size 32 --max-new-tokens 8 --temperature 0.0 --device cuda",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "family_generation_analysis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def build_label_prompt(context: str, hypothesis: str) -> str:
    return (
        "Use only the CONTEXT sentence. Decide whether the HYPOTHESIS is directly supported.\n"
        "Track semantic roles precisely: who did what to whom, what caused what, which entity has which state, which condition leads to which outcome, and what stayed unchanged.\n"
        "If the hypothesis swaps actor/patient, cause/effect, entity/state, condition/outcome, location/object, beneficiary, comparison direction, or assigns an outcome to the wrong entity, answer NOT_ENTAILED.\n"
        "Answer exactly one label: ENTAILED or NOT_ENTAILED. No explanation.\n\n"
        f"CONTEXT: {context}\n"
        f"HYPOTHESIS: {hypothesis}\n"
        "LABEL:"
    )


def parse_label(output: Any) -> str | None:
    s = str(output or "").strip().upper()
    s = s.replace("ENTAILLED", "ENTAILED").replace("ENTAILLED", "ENTAILED")
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.search(r"\bNOT[_ ]?ENTAILED\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bUNSUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAILED\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def load_teacher_labels(model_name: str, prompt_path: Path, output_path: Path) -> list[dict[str, Any]]:
    prompts = read_jsonl(prompt_path)
    outs = read_jsonl(output_path)
    if len(prompts) != len(outs):
        raise RuntimeError(f"{model_name} prompt/output mismatch {len(prompts)} vs {len(outs)}")
    rows: list[dict[str, Any]] = []
    for i, (p, o) in enumerate(zip(prompts, outs)):
        raw = str(o.get("output") or o.get("generated_text") or o.get("text") or "")
        label = parse_label(raw)
        rows.append({
            **{k: p[k] for k in ["id", "case_id", "pair_id", "tier", "edit_class", "bucket", "family_id", "context_type", "fact_id", "role_type", "purpose", "swap_type", "hypothesis", "gold_label", "context"]},
            "teacher": model_name,
            "output_raw": raw,
            "parsed_label": label,
            "expected_consistent": bool(label == p["gold_label"]),
            "output_index": o.get("index", i),
        })
    return rows


def rate(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def summarize_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted({str(r.get(key)) for r in rows}):
        rs = [r for r in rows if str(r.get(key)) == k]
        valid = [r for r in rs if r.get("parsed_label")]
        out[k] = {
            "n": len(rs),
            "valid": len(valid),
            "expected_consistency": rate(sum(1 for r in valid if r.get("expected_consistent")), len(valid)),
            "invalid": len(rs) - len(valid),
        }
    return out


def ewok_domain_counts() -> dict[str, Any]:
    if not EWOK_PANEL.exists():
        return {"error": f"missing {EWOK_PANEL}"}
    rows = read_jsonl(EWOK_PANEL)
    total = collections.Counter(r.get("domain", "") for r in rows)
    rel = collections.Counter(r.get("domain", "") for r in rows if r.get("is_step211_relational_domain"))
    return {
        "panel_rows": len(rows),
        "domain_counts": dict(sorted(total.items())),
        "relational_rows": sum(rel.values()),
        "relational_domain_counts": dict(sorted(rel.items())),
    }


def mapped_ewok_domains(role_type: str) -> set[str]:
    rt = str(role_type or "").lower()
    out: set[str] = set()
    for key, vals in ROLE_TO_EWOK.items():
        if key in rt:
            out |= vals
    return out


def analyze_labels(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(OUT_DIR / "generated_family_cases.jsonl")
    rows: list[dict[str, Any]] = []
    rows += load_teacher_labels("qwen3.5-9b", PROMPT_DIR / "label_prompts_qwen.jsonl", Path(args.qwen_outputs))
    rows += load_teacher_labels("llama3.1-8b-instruct", PROMPT_DIR / "label_prompts_llama.jsonl", Path(args.llama_outputs))
    write_jsonl(OUT_DIR / "all_teacher_label_rows.jsonl", rows)
    valid = [r for r in rows if r.get("parsed_label")]

    # Cross-teacher agreement for exact prompt rows.
    by_id: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["teacher"]] = r
    pair_rows: list[dict[str, Any]] = []
    for pid, d in sorted(by_id.items()):
        if not all(t in d for t in TEACHERS):
            continue
        q, l = d["qwen3.5-9b"], d["llama3.1-8b-instruct"]
        qlab, llab = q.get("parsed_label"), l.get("parsed_label")
        same = bool(qlab and llab and qlab == llab)
        both_expected = bool(same and qlab == q["gold_label"])
        pair_rows.append({
            "id": pid,
            "case_id": q["case_id"],
            "family_id": q["family_id"],
            "context_type": q["context_type"],
            "fact_id": q["fact_id"],
            "role_type": q["role_type"],
            "purpose": q["purpose"],
            "swap_type": q["swap_type"],
            "gold_label": q["gold_label"],
            "qwen_label": qlab,
            "llama_label": llab,
            "same_label": same,
            "both_expected": both_expected,
            "hypothesis": q["hypothesis"],
            "context": q["context"],
        })
    write_jsonl(OUT_DIR / "cross_teacher_prompt_pairs.jsonl", pair_rows)

    # Invariance by teacher + family + fact.
    inv_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        groups[(r["teacher"], r["case_id"], r["family_id"], r["fact_id"])][r["context_type"]] = r
    for (teacher, case_id, family_id, fact_id), d in sorted(groups.items()):
        labels = {ctx: d.get(ctx, {}).get("parsed_label") for ctx, _ in CONTEXT_TYPES}
        expected = {ctx: d.get(ctx, {}).get("expected_consistent") for ctx, _ in CONTEXT_TYPES}
        inv_rows.append({
            "teacher": teacher,
            "case_id": case_id,
            "family_id": family_id,
            "fact_id": fact_id,
            "role_type": next((v.get("role_type") for v in d.values()), ""),
            "gold_label": next((v.get("gold_label") for v in d.values()), ""),
            "labels": labels,
            "expected_by_context": expected,
            "source_bridge_same": bool(labels.get("source") and labels.get("source") == labels.get("source_attested_bridge")),
            "source_bridge_expected": bool(expected.get("source") and expected.get("source_attested_bridge")),
            "all_context_same": bool(labels.get("source") and labels.get("source") == labels.get("source_attested_bridge") == labels.get("natural_compact_reference")),
            "all_context_expected": bool(expected.get("source") and expected.get("source_attested_bridge") and expected.get("natural_compact_reference")),
            "hypothesis": next((v.get("hypothesis") for v in d.values()), ""),
        })
    write_jsonl(OUT_DIR / "context_invariance_rows.jsonl", inv_rows)

    # Family-level retention: not just agreement; must include structure and all four facts.
    pair_by_family_context_fact: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for pr in pair_rows:
        pair_by_family_context_fact[(pr["case_id"], pr["family_id"], pr["context_type"], pr["fact_id"])] = pr

    case_by_id = {(c["case_id"], c.get("family_id", "r1")): c for c in cases}
    family_summaries: list[dict[str, Any]] = []
    retained_full: list[dict[str, Any]] = []
    retained_source_bridge: list[dict[str, Any]] = []
    for key, c in sorted(case_by_id.items()):
        if not c.get("ok_family"):
            continue
        fact_ids = [f["fact_id"] for f in c.get("facts", [])]
        checks = {}
        for scope, ctxs in [("source_bridge", ["source", "source_attested_bridge"]), ("full_context", ["source", "source_attested_bridge", "natural_compact_reference"]), ("natural_only", ["natural_compact_reference"]), ("source_only", ["source"]), ("bridge_only", ["source_attested_bridge"])]:
            need = [(ctx, fid) for ctx in ctxs for fid in fact_ids]
            prs = [pair_by_family_context_fact.get((c["case_id"], c["family_id"], ctx, fid)) for ctx, fid in need]
            prs = [p for p in prs if p]
            checks[scope] = {
                "rows": len(prs),
                "needed": len(need),
                "both_expected": sum(1 for p in prs if p.get("both_expected")),
                "all_both_expected": bool(len(prs) == len(need) and all(p.get("both_expected") for p in prs)),
                "teacher_same": sum(1 for p in prs if p.get("same_label")),
            }
        role_domains = sorted(mapped_ewok_domains(c.get("role_type", "")))
        fs = {
            "case_id": c["case_id"],
            "pair_id": c.get("pair_id"),
            "tier": c.get("tier"),
            "edit_class": c.get("edit_class"),
            "bucket": c.get("bucket"),
            "family_id": c.get("family_id"),
            "role_type": c.get("role_type"),
            "role_type_norm": c.get("structure", {}).get("role_type_norm"),
            "structural_not_easy": c.get("structure", {}).get("structural_not_easy"),
            "good_role_words": c.get("structure", {}).get("good_role_words"),
            "low_value_role_words": c.get("structure", {}).get("low_value_role_words"),
            "mapped_ewok_domains": role_domains,
            "facts": c.get("facts", []),
            "checks": checks,
            "source_text": c.get("source_text"),
            "source_attested_bridge": c.get("source_attested_bridge"),
            "natural_compact_reference": c.get("natural_compact_reference"),
        }
        family_summaries.append(fs)
        if checks["source_bridge"]["all_both_expected"] and fs["structural_not_easy"]:
            retained_source_bridge.append({**fs, "stable_scope": "source_bridge"})
        if checks["full_context"]["all_both_expected"] and fs["structural_not_easy"]:
            retained_full.append({**fs, "stable_scope": "full_context"})
    write_jsonl(OUT_DIR / "family_retention_summary.jsonl", family_summaries)
    write_jsonl(OUT_DIR / "retained_source_bridge_role_families.jsonl", retained_source_bridge)
    write_jsonl(OUT_DIR / "retained_full_context_role_families.jsonl", retained_full)

    # Failure reasons by family.
    failure_rows: list[dict[str, Any]] = []
    for fs in family_summaries:
        reasons = []
        if not fs["structural_not_easy"]:
            reasons.append("weak_or_easy_family_structure")
        sb = fs["checks"]["source_bridge"]
        fc = fs["checks"]["full_context"]
        if not sb["all_both_expected"]:
            reasons.append("source_bridge_not_stable_both_teachers")
        if sb["all_both_expected"] and not fc["all_both_expected"]:
            reasons.append("natural_reference_not_stable")
        # collect row-level examples for nonstable source/bridge/full.
        bad = []
        for pr in pair_rows:
            if pr["case_id"] == fs["case_id"] and pr["family_id"] == fs["family_id"] and not pr.get("both_expected"):
                bad.append({k: pr[k] for k in ["context_type", "fact_id", "gold_label", "qwen_label", "llama_label", "role_type", "swap_type", "hypothesis"]})
        failure_rows.append({
            "case_id": fs["case_id"],
            "family_id": fs["family_id"],
            "role_type": fs["role_type"],
            "reasons": reasons,
            "bad_prompt_pairs_sample": bad[:12],
        })
    write_jsonl(OUT_DIR / "family_failure_reasons.jsonl", failure_rows)

    # Diversity and natural bridge mapping summaries.
    def dist(rows_: list[dict[str, Any]], key: str) -> dict[str, int]:
        return dict(collections.Counter(str(r.get(key, "")) for r in rows_).most_common())

    def domain_union(rows_: list[dict[str, Any]]) -> dict[str, Any]:
        counter = collections.Counter()
        for r in rows_:
            for d in r.get("mapped_ewok_domains", []):
                counter[d] += 1
        return dict(counter.most_common())

    ewok_counts = ewok_domain_counts()
    stable_domain_counts = domain_union(retained_source_bridge)
    stable_full_domain_counts = domain_union(retained_full)
    ewok_rel_domains = set((ewok_counts.get("relational_domain_counts") or {}).keys()) if isinstance(ewok_counts, dict) else set()
    stable_rel_overlap = sorted(set(stable_domain_counts) & ewok_rel_domains)
    full_rel_overlap = sorted(set(stable_full_domain_counts) & ewok_rel_domains)

    retained_semantic_words = set()
    for r in retained_source_bridge:
        for f in r.get("facts", []):
            retained_semantic_words |= tokenize_words(f.get("hypothesis", ""))
    low_value_retained = [r for r in retained_source_bridge if r.get("low_value_role_words")]

    summary = {
        "status": "V2_ROLE_SUBSTRATE_ANALYZED",
        "created_utc": now(),
        "inputs": {
            "family_generation_prompts": str(PROMPT_DIR / "family_generation_prompts.jsonl"),
            "family_outputs": str(args.family_outputs if hasattr(args, "family_outputs") else PROMPT_DIR / "family_generation_outputs_qwen.jsonl"),
            "qwen_outputs": str(Path(args.qwen_outputs)),
            "llama_outputs": str(Path(args.llama_outputs)),
            "a02_records_sha256": sha256_file(A02_RECORDS),
        },
        "counts": {
            "candidate_cases_total": len(cases),
            "ok_counterfactual_family_cases": sum(1 for c in cases if c.get("ok_family")),
            "teacher_rows": len(rows),
            "valid_teacher_rows": len(valid),
            "prompt_pairs": len(pair_rows),
            "retained_source_bridge_families": len(retained_source_bridge),
            "retained_full_context_families": len(retained_full),
        },
        "label_quality": {
            "overall_expected_consistency": rate(sum(1 for r in valid if r.get("expected_consistent")), len(valid)),
            "cross_teacher_label_agreement": rate(sum(1 for r in pair_rows if r.get("same_label")), len(pair_rows)),
            "both_teachers_expected": rate(sum(1 for r in pair_rows if r.get("both_expected")), len(pair_rows)),
            "source_bridge_all_both_expected_family_rate_over_ok": rate(len(retained_source_bridge), sum(1 for c in cases if c.get("ok_family"))),
            "full_context_all_both_expected_family_rate_over_ok": rate(len(retained_full), sum(1 for c in cases if c.get("ok_family"))),
        },
        "by_teacher": summarize_by(rows, "teacher"),
        "by_context": summarize_by(rows, "context_type"),
        "by_gold_label": summarize_by(rows, "gold_label"),
        "by_purpose": summarize_by(rows, "purpose"),
        "by_role_type": summarize_by(rows, "role_type"),
        "retained_source_bridge_role_type_norm_distribution": dist(retained_source_bridge, "role_type_norm"),
        "retained_full_role_type_norm_distribution": dist(retained_full, "role_type_norm"),
        "retained_source_bridge_case_ids": [r["case_id"] for r in retained_source_bridge],
        "retained_full_context_case_ids": [r["case_id"] for r in retained_full],
        "semantic_diversity": {
            "retained_source_bridge_distinct_role_types": len(set(r.get("role_type_norm") for r in retained_source_bridge)),
            "retained_full_context_distinct_role_types": len(set(r.get("role_type_norm") for r in retained_full)),
            "retained_source_bridge_distinct_hypothesis_content_words": len(retained_semantic_words),
            "low_value_retained_families": len(low_value_retained),
        },
        "ewok_bridge_panel_mapping": {
            **ewok_counts,
            "retained_source_bridge_mapped_domain_counts": stable_domain_counts,
            "retained_full_context_mapped_domain_counts": stable_full_domain_counts,
            "retained_source_bridge_overlap_step211_relational_domains": stable_rel_overlap,
            "retained_full_context_overlap_step211_relational_domains": full_rel_overlap,
            "interpretation_note": "Mapping is heuristic by role_type; it measures whether retained natural role families cover the same kind of conditional-reversal domains as the EWoK bridge panel, not item-level equivalence.",
        },
        "family_retention_summary": str(OUT_DIR / "family_retention_summary.jsonl"),
        "retained_source_bridge_role_families": str(OUT_DIR / "retained_source_bridge_role_families.jsonl"),
        "retained_full_context_role_families": str(OUT_DIR / "retained_full_context_role_families.jsonl"),
        "family_failure_reasons": str(OUT_DIR / "family_failure_reasons.jsonl"),
        "scientific_interpretation": interpret_summary(len(cases), sum(1 for c in cases if c.get("ok_family")), retained_source_bridge, retained_full, stable_domain_counts, ewok_rel_domains),
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "v2_role_substrate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary, retained_source_bridge, retained_full, failure_rows)
    print(json.dumps({
        "status": summary["status"],
        "ok_counterfactual_family_cases": summary["counts"]["ok_counterfactual_family_cases"],
        "retained_source_bridge_families": len(retained_source_bridge),
        "retained_full_context_families": len(retained_full),
        "cross_teacher_label_agreement": summary["label_quality"]["cross_teacher_label_agreement"],
        "both_teachers_expected": summary["label_quality"]["both_teachers_expected"],
        "retained_role_types": summary["retained_source_bridge_role_type_norm_distribution"],
        "mapped_step211_overlap": stable_rel_overlap,
        "summary_json": str(OUT_DIR / "v2_role_substrate_summary.json"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_role_substrate_summary.md')),
    }, indent=2, ensure_ascii=False), flush=True)


def interpret_summary(n_cases: int, ok: int, sb: list[dict[str, Any]], full: list[dict[str, Any]], domain_counts: dict[str, int], ewok_rel_domains: set[str]) -> str:
    if ok == 0:
        return "V2 generation could not produce structured counterfactual role families from the current collection; seek a broader source-attested transformation source."
    sb_rate = len(sb) / ok
    full_rate = len(full) / ok
    role_div = len(set(r.get("role_type_norm") for r in sb))
    overlap = set(domain_counts) & ewok_rel_domains
    if len(sb) >= 20 and sb_rate >= 0.5 and role_div >= 4 and len(overlap) >= 2:
        return "Current collection contains a nontrivial source-bridge-stable counterfactual role-family substrate; next work should manually inspect retained families and consider a tiny held-structure student pilot before any BabyLM-scale run."
    if len(sb) >= 10 and role_div >= 3 and overlap:
        return "Current collection preserves a real but sparse source-bridge-stable role-family core. It is useful as seed/evaluation material, but too small for distillation as a general principle substrate without broader source-attested transformations."
    return "Agreement can be obtained only on a sparse or narrow subset. Selection collapse remains the main danger; use retained families as probes and seek a broader source-attested transformation source rather than training from this collection."


def write_markdown(summary: dict[str, Any], sb: list[dict[str, Any]], full: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = []
    c = summary["counts"]
    q = summary["label_quality"]
    lines += [
        "# research — v2 counterfactual role-family substrate",
        "",
        "## Purpose",
        "",
        "This run repairs the research/248 automatic role-label object by asking for counterfactually closed role-exchange families rather than independent easy facts. It uses short teacher inference only; no BabyLM model or student model is trained.",
        "",
        "## Main numbers",
        "",
        f"- Candidate A/B cases: {c['candidate_cases_total']}",
        f"- Parsed structurally valid counterfactual-family cases: {c['ok_counterfactual_family_cases']}",
        f"- Teacher rows: {c['teacher_rows']} ({c['valid_teacher_rows']} valid)",
        f"- Cross-teacher agreement on exact one-sentence prompts: {q['cross_teacher_label_agreement']:.4f}",
        f"- Both teachers gave expected label: {q['both_teachers_expected']:.4f}",
        f"- Source↔bridge retained role families: {c['retained_source_bridge_families']} ({q['source_bridge_all_both_expected_family_rate_over_ok']:.4f} of structurally valid)",
        f"- Full-context retained role families: {c['retained_full_context_families']} ({q['full_context_all_both_expected_family_rate_over_ok']:.4f} of structurally valid)",
        "",
        "## Retained source↔bridge role types",
        "",
    ]
    for k, v in summary.get("retained_source_bridge_role_type_norm_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Heuristic EWoK bridge-panel domain mapping", ""]
    for k, v in summary.get("ewok_bridge_panel_mapping", {}).get("retained_source_bridge_mapped_domain_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Scientific interpretation",
        "",
        summary.get("scientific_interpretation", ""),
        "",
        "## Retained source↔bridge families (first 12)",
        "",
    ]
    for r in sb[:12]:
        lines += [
            f"### {r['case_id']} — {r.get('role_type')}",
            f"Source: {r.get('source_text')}",
            f"Bridge: {r.get('source_attested_bridge')}",
            "Facts:",
        ]
        for f in r.get("facts", []):
            lines.append(f"- {f['fact_id']} {f['gold_label']} ({f.get('purpose') or f.get('swap_type')}): {f['hypothesis']}")
        lines.append("")
    lines += ["", "## Representative family failures", ""]
    shown = 0
    for fr in failures:
        if not fr.get("reasons"):
            continue
        lines += [f"### {fr['case_id']} — {fr.get('role_type')}", f"Reasons: {', '.join(fr.get('reasons', []))}"]
        for b in fr.get("bad_prompt_pairs_sample", [])[:4]:
            lines.append(f"- {b['context_type']} {b['fact_id']} gold={b['gold_label']} qwen={b['qwen_label']} llama={b['llama_label']} :: {b['hypothesis']}")
        lines.append("")
        shown += 1
        if shown >= 10:
            break
    lines += [
        "## Files",
        "",
        f"- summary JSON: `{OUT_DIR / 'v2_role_substrate_summary.json'}`",
        f"- family table: `{OUT_DIR / 'family_retention_summary.jsonl'}`",
        f"- retained source↔bridge families: `{OUT_DIR / 'retained_source_bridge_role_families.jsonl'}`",
        f"- retained full-context families: `{OUT_DIR / 'retained_full_context_role_families.jsonl'}`",
        f"- failures: `{OUT_DIR / 'family_failure_reasons.jsonl'}`",
    ]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_role_substrate_summary.md')).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p0 = sub.add_parser("prepare", help="prepare v2 family-generation prompts")
    p0.add_argument("--max_cases", type=int, default=0)
    p1 = sub.add_parser("parse-families", help="parse Qwen family outputs and prepare teacher label prompts")
    p1.add_argument("--family_outputs", default=str(PROMPT_DIR / "family_generation_outputs_qwen.jsonl"))
    p2 = sub.add_parser("analyze-labels", help="score Qwen/Llama label outputs")
    p2.add_argument("--qwen_outputs", default=str(PROMPT_DIR / "label_outputs_qwen.jsonl"))
    p2.add_argument("--llama_outputs", default=str(PROMPT_DIR / "label_outputs_llama.jsonl"))
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "parse-families":
        parse_family_cases(args)
    elif args.cmd == "analyze-labels":
        analyze_labels(args)
    else:
        raise SystemExit(args.cmd)


if __name__ == "__main__":
    main()
