#!/usr/bin/env python3
"""research: Natural source-attested transformation role-preservation premise test.

This is a premise-measurement script, not BabyLM pretraining. It builds a small
held-out semantic-role panel from source-attested bridge-atlas candidates
and tests whether an approved teacher model can provide stable independent-
sentence entailment labels for roles under paraphrase and role-swapped false
hypotheses.

Scientific use: decide whether role-assignment signal exists before any new
architecture run or Strict-Small training. A positive result motivates building a
larger teacher/self-supervised distillation substrate; a negative result stops
that route before expensive training.
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
OUT_DIR = ROOT / "data/natural_role_transform_panel"
PROMPT_DIR = ROOT / "training/data/natural_role_transform_panel"

# Case ids were selected from high-fidelity Tier-A/B source-attested bridge
# candidates. The hypotheses are hand-authored to be true or not supported in
# source, bridge, and natural compact reference, so the teacher is judged against
# a fixed semantic-role expectation, not asked to invent the structure.
PANEL = [
    {
        "case_id": "B006",
        "role_focus": "program_improves_food_security",
        "facts": [
            ("true_actor_effect", "The HFP program improved food security.", "ENTAILED"),
            ("true_scope", "More than 5 million vulnerable people benefited from improved food security.", "ENTAILED"),
            ("actor_swap_false", "Food security improved the HFP program.", "NOT_ENTAILED"),
            ("conservation_false", "More than 5 million vulnerable people improved the HFP program.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B019",
        "role_focus": "wwi_effects_bankrupt_europe",
        "facts": [
            ("true_state", "Most of Europe was bankrupt in the 1920s.", "ENTAILED"),
            ("true_cause", "WWI effects were a reason most of Europe was bankrupt.", "ENTAILED"),
            ("actor_swap_false", "Most of Europe caused WWI effects.", "NOT_ENTAILED"),
            ("conservation_false", "WWI effects were bankrupt because of Europe.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B024",
        "role_focus": "households_adopt_fuel",
        "facts": [
            ("true_actor_action", "Korean households took the new fuel.", "ENTAILED"),
            ("true_comparison", "The new fuel was more convenient than firewood.", "ENTAILED"),
            ("actor_swap_false", "The new fuel took Korean households.", "NOT_ENTAILED"),
            ("conservation_false", "Firewood was more convenient than the new fuel.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B039",
        "role_focus": "frog_allows_photographs",
        "facts": [
            ("true_actor_action", "The frog allowed close-up photographs.", "ENTAILED"),
            ("true_instrument", "The close-up photographs used a 50mm lens.", "ENTAILED"),
            ("actor_swap_false", "The close-up photographs allowed the frog.", "NOT_ENTAILED"),
            ("conservation_false", "The 50mm lens cooperated for the frog.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B067",
        "role_focus": "china_accuses_others",
        "facts": [
            ("true_actor_action", "China accused the Philippines and Vietnam.", "ENTAILED"),
            ("true_embedded_role", "China accused the Philippines and Vietnam of using US support.", "ENTAILED"),
            ("actor_swap_false", "The Philippines and Vietnam accused China of using US support.", "NOT_ENTAILED"),
            ("conservation_false", "US support accused China.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B069",
        "role_focus": "war_instability_causes_deaths",
        "facts": [
            ("true_cause", "War's instability caused most deaths.", "ENTAILED"),
            ("true_theme", "Most deaths were caused by war's instability.", "ENTAILED"),
            ("actor_swap_false", "Most deaths caused war's instability.", "NOT_ENTAILED"),
            ("conservation_false", "War's stability caused most deaths.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B073",
        "role_focus": "anagram_made_by_rearranging_letters",
        "facts": [
            ("true_definition", "An anagram is made by rearranging letters.", "ENTAILED"),
            ("true_object", "Letters are rearranged to make an anagram.", "ENTAILED"),
            ("actor_swap_false", "Letters are made by rearranging an anagram.", "NOT_ENTAILED"),
            ("conservation_false", "An anagram rearranges the collection.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B079",
        "role_focus": "website_claims_palm_valley_remnant",
        "facts": [
            ("true_speaker", "The NT government website claims Palm Valley is a rainforest remnant.", "ENTAILED"),
            ("true_theme", "Palm Valley is described as a remnant of rainforests.", "ENTAILED"),
            ("actor_swap_false", "Palm Valley claims the NT government website is a rainforest remnant.", "NOT_ENTAILED"),
            ("conservation_false", "The NT government website is a remnant of Palm Valley.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B082",
        "role_focus": "mercury_upsets_microorganism_balance",
        "facts": [
            ("true_actor_effect", "Excess mercury upsets the body's microorganism balance.", "ENTAILED"),
            ("true_result", "The imbalance may lead to candida.", "ENTAILED"),
            ("actor_swap_false", "The body's microorganism balance upsets excess mercury.", "NOT_ENTAILED"),
            ("conservation_false", "Candida leads to excess mercury.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B095",
        "role_focus": "movement_causes_two_entity_specific_outcomes",
        "facts": [
            ("true_actor_action", "The movement compelled Chimanbhai Patel to resign.", "ENTAILED"),
            ("true_other_role", "The movement gave Indira Gandhi an excuse to declare emergency.", "ENTAILED"),
            ("actor_swap_false", "Indira Gandhi was compelled by the movement to resign.", "NOT_ENTAILED"),
            ("conservation_false", "Chimanbhai Patel got an excuse to declare emergency.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B096",
        "role_focus": "thoughts_control_us_and_feelings",
        "facts": [
            ("true_actor_action", "These thoughts control us.", "ENTAILED"),
            ("true_effect", "These thoughts affect how we feel.", "ENTAILED"),
            ("actor_swap_false", "We control these thoughts.", "NOT_ENTAILED"),
            ("conservation_false", "How we feel controls these thoughts.", "NOT_ENTAILED"),
        ],
    },
    {
        "case_id": "B050",
        "role_focus": "resurvey_causes_missouri_claim",
        "facts": [
            ("true_cause", "Brown's re-survey caused Missouri to claim its border extended into Iowa.", "ENTAILED"),
            ("true_theme", "Missouri claimed its border extended 13 miles into Iowa.", "ENTAILED"),
            ("actor_swap_false", "Missouri's claim caused Brown's re-survey.", "NOT_ENTAILED"),
            ("conservation_false", "Iowa claimed Brown's border extended into Missouri.", "NOT_ENTAILED"),
        ],
    },
]

CONTEXT_TYPES = [
    ("source", "source_text"),
    ("source_attested_bridge", "candidate_text"),
    ("natural_compact_reference", "natural_compact_text"),
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


def read_a02_records() -> dict[str, dict[str, Any]]:
    obj = json.loads(A02_RECORDS.read_text(encoding="utf-8"))
    rows = obj.get("records", [])
    return {str(r["case_id"]): r for r in rows}


def build_prompt(context: str, hypothesis: str) -> str:
    return (
        "You are checking semantic roles in one English sentence. Use only the CONTEXT sentence.\n"
        "Decide whether the HYPOTHESIS is directly supported by that sentence.\n"
        "Important: do not swap who did what to whom; if an event happened to one entity, do not transfer it to another entity.\n"
        "Answer exactly one label: ENTAILED or NOT_ENTAILED. Do not explain.\n\n"
        f"CONTEXT: {context}\n"
        f"HYPOTHESIS: {hypothesis}\n"
        "LABEL:"
    )


def prepare(_: argparse.Namespace) -> None:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_case = read_a02_records()
    prompt_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case_def in PANEL:
        cid = case_def["case_id"]
        if cid not in by_case:
            raise RuntimeError(f"Missing A02 bridge record for {cid}")
        rec = by_case[cid]
        case_row = {
            "case_id": cid,
            "pair_id": rec.get("pair_id"),
            "tier": rec.get("tier"),
            "edit_class": rec.get("edit_class"),
            "bucket": rec.get("bucket"),
            "role_focus": case_def["role_focus"],
            "source_text": rec.get("source_text"),
            "source_attested_bridge": rec.get("candidate_text"),
            "natural_compact_reference": rec.get("natural_compact_text"),
            "facts": [
                {"fact_id": f"f{i:02d}", "category": cat, "hypothesis": hyp, "gold_label": lab}
                for i, (cat, hyp, lab) in enumerate(case_def["facts"])
            ],
        }
        case_rows.append(case_row)
        for ctype, field in CONTEXT_TYPES:
            context = str(rec.get(field) or "").strip()
            if not context:
                raise RuntimeError(f"Empty {field} for {cid}")
            for i, (cat, hyp, gold) in enumerate(case_def["facts"]):
                pid = f"step247_{cid}_{ctype}_f{i:02d}_{cat}"
                prompt_rows.append({
                    "id": pid,
                    "prompt_id": pid,
                    "case_id": cid,
                    "pair_id": rec.get("pair_id"),
                    "tier": rec.get("tier"),
                    "edit_class": rec.get("edit_class"),
                    "bucket": rec.get("bucket"),
                    "role_focus": case_def["role_focus"],
                    "context_type": ctype,
                    "fact_id": f"f{i:02d}",
                    "fact_category": cat,
                    "context": context,
                    "hypothesis": hyp,
                    "gold_label": gold,
                    "prompt": build_prompt(context, hyp),
                })
    # deterministic split so two H100s can be used without a shared output file.
    shard0 = prompt_rows[0::2]
    shard1 = prompt_rows[1::2]
    write_jsonl(PROMPT_DIR / "teacher_prompts_all.jsonl", prompt_rows)
    write_jsonl(PROMPT_DIR / "teacher_prompts_gpu0.jsonl", shard0)
    write_jsonl(PROMPT_DIR / "teacher_prompts_gpu1.jsonl", shard1)
    write_jsonl(OUT_DIR / "panel_cases.jsonl", case_rows)

    md = [
        "# research natural role-preservation premise panel", "",
        "Purpose: test whether an approved teacher can provide stable independent-sentence semantic-role labels on source-attested bridge transformations before any new architecture or Strict-Small training.", "",
        f"Cases: {len(case_rows)}; contexts per case: {len(CONTEXT_TYPES)}; facts per case: 4; prompts: {len(prompt_rows)}.", "",
        "Each prompt contains exactly one context sentence and one hypothesis; no source/bridge pair is shown in the same prompt.", "",
    ]
    for c in case_rows:
        md += [f"## {c['case_id']} — {c['role_focus']} ({c['tier']}, {c['edit_class']})", "",
               f"Source: {c['source_text']}", "",
               f"Bridge: {c['source_attested_bridge']}", "",
               f"Natural compact reference: {c['natural_compact_reference']}", "", "Facts:"]
        for f in c["facts"]:
            md.append(f"- {f['gold_label']} / {f['category']}: {f['hypothesis']}")
        md.append("")
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/natural_role_transform_panel/panel_readme.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "status": "NATURAL_ROLE_TRANSFORM_PANEL_PREPARED",
        "created_utc": now(),
        "a02_records": str(A02_RECORDS),
        "a02_records_sha256": sha256_file(A02_RECORDS),
        "n_cases": len(case_rows),
        "n_prompts": len(prompt_rows),
        "context_types": [x[0] for x in CONTEXT_TYPES],
        "shard0_prompts": str(PROMPT_DIR / "teacher_prompts_gpu0.jsonl"),
        "shard1_prompts": str(PROMPT_DIR / "teacher_prompts_gpu1.jsonl"),
        "all_prompts": str(PROMPT_DIR / "teacher_prompts_all.jsonl"),
        "panel_cases": str(OUT_DIR / "panel_cases.jsonl"),
        "panel_readme": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/natural_role_transform_panel/panel_readme.md')),
        "teacher_model_for_planned_measurement": "qwen3.5-9b (approved family per BabyLM FAQ already retrieved in session)",
        "decision_purpose": "If independent-sentence teacher labels are unstable on bridge paraphrases or role-swapped false hypotheses, do not distill or train a new architecture on this signal. If stable, next step may construct a larger held-out/source-attested role-label substrate and test student distillation against the EWoK/Entity item signature.",
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def parse_label(output: str) -> str | None:
    s = str(output or "").strip().upper()
    s = re.sub(r"[^A-Z_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Check negated labels first because they contain ENTAILED.
    if re.search(r"\bNOT[_ ]?ENTAILED\b", s) or re.search(r"\bNOT\s+SUPPORTED\b", s) or re.search(r"\bNO\b", s):
        return "NOT_ENTAILED"
    if re.search(r"\bENTAILED\b", s) or re.search(r"\bSUPPORTED\b", s) or re.search(r"\bYES\b", s):
        return "ENTAILED"
    return None


def stat(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {"n": 0}
    xs.sort()
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "min": xs[0], "max": xs[-1]}


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    out = {}
    for k in sorted({str(r.get(key)) for r in rows}):
        rs = [r for r in rows if str(r.get(key)) == k]
        valid = [r for r in rs if r.get("parsed_label")]
        out[k] = {
            "n": len(rs),
            "valid": len(valid),
            "accuracy": sum(1 for r in valid if r["correct"]) / max(1, len(valid)),
            "invalid": len(rs) - len(valid),
            "entailed_gold_n": sum(1 for r in rs if r.get("gold_label") == "ENTAILED"),
            "not_entailed_gold_n": sum(1 for r in rs if r.get("gold_label") == "NOT_ENTAILED"),
        }
    return out


def analyze(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt_shards = [PROMPT_DIR / "teacher_prompts_gpu0.jsonl", PROMPT_DIR / "teacher_prompts_gpu1.jsonl"]
    output_paths = [Path(args.gpu0_outputs), Path(args.gpu1_outputs)]
    rows: list[dict[str, Any]] = []
    for pp, op in zip(prompt_shards, output_paths):
        prompts = read_jsonl(pp)
        outs = read_jsonl(op)
        if len(outs) != len(prompts):
            raise RuntimeError(f"Prompt/output count mismatch for {pp}: prompts={len(prompts)} outputs={len(outs)}")
        for i, (p, o) in enumerate(zip(prompts, outs)):
            raw = str(o.get("output") or o.get("generated_text") or o.get("text") or o.get("completion") or "")
            label = parse_label(raw)
            gold = p["gold_label"]
            rows.append({
                **{k: p[k] for k in ["id", "case_id", "pair_id", "tier", "edit_class", "bucket", "role_focus", "context_type", "fact_id", "fact_category", "context", "hypothesis", "gold_label"]},
                "output_raw": raw,
                "parsed_label": label,
                "correct": (label == gold),
                "shard": pp.name,
                "output_index": o.get("index", i),
            })

    valid = [r for r in rows if r.get("parsed_label")]
    true_rows = [r for r in valid if r["gold_label"] == "ENTAILED"]
    false_rows = [r for r in valid if r["gold_label"] == "NOT_ENTAILED"]
    by_context = summarize(rows, "context_type")
    by_category = summarize(rows, "fact_category")
    by_tier = summarize(rows, "tier")

    # All-facts-correct rates for each independently encoded sentence.
    sentence_groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        sentence_groups[(r["case_id"], r["context_type"])].append(r)
    sentence_group_recs = []
    for (cid, ctype), rs in sorted(sentence_groups.items()):
        sentence_group_recs.append({
            "case_id": cid,
            "context_type": ctype,
            "n": len(rs),
            "valid": sum(1 for r in rs if r.get("parsed_label")),
            "all_correct": all(r.get("parsed_label") and r["correct"] for r in rs),
            "correct_count": sum(1 for r in rs if r.get("correct")),
            "raw_outputs": {r["fact_id"]: r["output_raw"] for r in rs},
        })

    # Source-bridge invariance: do source and source-attested bridge produce the
    # same parsed label, and is that label the gold label?
    fact_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in rows:
        fact_groups[(r["case_id"], r["fact_id"])] [r["context_type"]] = r
    invariance = []
    for (cid, fid), byc in sorted(fact_groups.items()):
        src = byc.get("source")
        br = byc.get("source_attested_bridge")
        nat = byc.get("natural_compact_reference")
        if not src or not br:
            continue
        same_sb = bool(src.get("parsed_label") and br.get("parsed_label") and src["parsed_label"] == br["parsed_label"])
        gold_ok_sb = bool(same_sb and src["parsed_label"] == src["gold_label"])
        same_all = bool(same_sb and nat and nat.get("parsed_label") == src["parsed_label"])
        gold_ok_all = bool(same_all and src["parsed_label"] == src["gold_label"])
        invariance.append({
            "case_id": cid,
            "fact_id": fid,
            "fact_category": src["fact_category"],
            "gold_label": src["gold_label"],
            "source_label": src.get("parsed_label"),
            "bridge_label": br.get("parsed_label"),
            "natural_label": nat.get("parsed_label") if nat else None,
            "same_source_bridge": same_sb,
            "gold_consistent_source_bridge": gold_ok_sb,
            "same_all_contexts": same_all,
            "gold_consistent_all_contexts": gold_ok_all,
            "hypothesis": src["hypothesis"],
        })

    inv_valid_sb = [x for x in invariance if x["source_label"] and x["bridge_label"]]
    inv_valid_all = [x for x in invariance if x["source_label"] and x["bridge_label"] and x["natural_label"]]

    source_acc = by_context.get("source", {}).get("accuracy", 0.0)
    bridge_acc = by_context.get("source_attested_bridge", {}).get("accuracy", 0.0)
    natural_acc = by_context.get("natural_compact_reference", {}).get("accuracy", 0.0)
    false_reject_bridge_rows = [r for r in valid if r["context_type"] == "source_attested_bridge" and r["gold_label"] == "NOT_ENTAILED"]
    false_reject_bridge = sum(1 for r in false_reject_bridge_rows if r["correct"]) / max(1, len(false_reject_bridge_rows))
    bridge_sentence_all = [x for x in sentence_group_recs if x["context_type"] == "source_attested_bridge"]
    bridge_all_rate = sum(1 for x in bridge_sentence_all if x["all_correct"]) / max(1, len(bridge_sentence_all))
    source_bridge_gold_inv = sum(1 for x in inv_valid_sb if x["gold_consistent_source_bridge"]) / max(1, len(inv_valid_sb))
    all_context_gold_inv = sum(1 for x in inv_valid_all if x["gold_consistent_all_contexts"]) / max(1, len(inv_valid_all))

    premise_positive = (
        len(valid) == len(rows)
        and source_acc >= 0.90
        and bridge_acc >= 0.85
        and natural_acc >= 0.85
        and false_reject_bridge >= 0.85
        and bridge_all_rate >= 0.70
        and source_bridge_gold_inv >= 0.85
    )
    # Conservative interpretation: natural reference is a robustness/paraphrase
    # check, but the source-attested bridge itself is the route object.
    decision = "POSITIVE_PREMISE_FOR_TEACHER_ROLE_SIGNAL" if premise_positive else "NEGATIVE_OR_BORDERLINE_PREMISE_DO_NOT_TRAIN"
    next_action = (
        "Construct a larger independent-sentence teacher-label substrate over source-attested transformations, with held-out paraphrase and role-swap controls, then test small-student distillation against the EWoK/Entity item signature."
        if premise_positive else
        "Do not start distillation or architecture training from this panel; inspect failures to decide whether the panel is flawed, teacher prompt is unstable, or source-attested transformations lack enough role-preserving signal."
    )
    summary = {
        "status": "NATURAL_ROLE_TRANSFORM_PANEL_ANALYZED",
        "created_utc": now(),
        "teacher_model": args.teacher_model,
        "n_rows": len(rows),
        "valid_rows": len(valid),
        "overall_accuracy": sum(1 for r in valid if r["correct"]) / max(1, len(valid)),
        "true_accept_accuracy": sum(1 for r in true_rows if r["correct"]) / max(1, len(true_rows)),
        "false_reject_accuracy": sum(1 for r in false_rows if r["correct"]) / max(1, len(false_rows)),
        "by_context": by_context,
        "by_category": by_category,
        "by_tier": by_tier,
        "sentence_all_correct_rate": sum(1 for x in sentence_group_recs if x["all_correct"]) / max(1, len(sentence_group_recs)),
        "bridge_sentence_all_correct_rate": bridge_all_rate,
        "source_bridge_label_invariance_rate": sum(1 for x in inv_valid_sb if x["same_source_bridge"]) / max(1, len(inv_valid_sb)),
        "source_bridge_gold_consistent_invariance_rate": source_bridge_gold_inv,
        "all_context_label_invariance_rate": sum(1 for x in inv_valid_all if x["same_all_contexts"]) / max(1, len(inv_valid_all)),
        "all_context_gold_consistent_invariance_rate": all_context_gold_inv,
        "false_reject_bridge_accuracy": false_reject_bridge,
        "premise_positive": premise_positive,
        "decision": decision,
        "next_action": next_action,
        "thresholds": {
            "source_acc": ">=0.90",
            "bridge_acc": ">=0.85",
            "natural_acc": ">=0.85",
            "false_reject_bridge": ">=0.85",
            "bridge_sentence_all": ">=0.70",
            "source_bridge_gold_inv": ">=0.85",
            "valid_rows": "all parseable",
        },
        "interpretation_boundary": "This measures whether an approved teacher can emit stable role labels on independent natural/source-attested sentences. It is not BabyLM endpoint evidence, not a compact-view reopening, and not evidence that a small student can learn the signal until distillation and EWoK/Entity item tests are performed.",
        "no_babylm_training_eval_upload_submission": True,
    }
    write_jsonl(OUT_DIR / "teacher_labeled_rows.jsonl", rows)
    write_jsonl(OUT_DIR / "sentence_group_results.jsonl", sentence_group_recs)
    write_jsonl(OUT_DIR / "source_bridge_invariance.jsonl", invariance)
    (OUT_DIR / "teacher_panel_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = [r for r in rows if (not r.get("parsed_label")) or not r.get("correct")]
    md = [
        "# research natural source-attested role-preservation teacher premise test", "",
        "## Purpose", "",
        "Before any new architecture or Strict-Small training, test whether the raw-text role-assignment signal exists in a cheap form: an approved teacher receives one natural/source-attested sentence at a time and judges role-sensitive hypotheses. The bridge atlas is used adversarially: source-attested bridge candidates must preserve roles under paraphrase and reject actor/query swaps, otherwise they are not a training premise.", "",
        "## Aggregate result", "",
        f"- rows: {len(rows)}, valid parsed labels: {len(valid)}", 
        f"- overall accuracy: {summary['overall_accuracy']:.3f}",
        f"- true-hypothesis accuracy: {summary['true_accept_accuracy']:.3f}",
        f"- false role-swap rejection accuracy: {summary['false_reject_accuracy']:.3f}",
        f"- source context accuracy: {source_acc:.3f}",
        f"- source-attested bridge context accuracy: {bridge_acc:.3f}",
        f"- natural compact reference context accuracy: {natural_acc:.3f}",
        f"- bridge sentence all-facts-correct rate: {bridge_all_rate:.3f}",
        f"- source↔bridge gold-consistent invariance: {source_bridge_gold_inv:.3f}",
        f"- all-context gold-consistent invariance: {all_context_gold_inv:.3f}",
        f"- decision: **{decision}**", "",
        "## Context accuracy", "",
        "| context | n | acc | invalid |", "|---|---:|---:|---:|",
    ]
    for k, v in by_context.items():
        md.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['invalid']} |")
    md += ["", "## Category accuracy", "", "| category | n | acc | invalid |", "|---|---:|---:|---:|"]
    for k, v in by_category.items():
        md.append(f"| {k} | {v['n']} | {v['accuracy']:.3f} | {v['invalid']} |")
    md += ["", "## Failure sample", ""]
    for r in failures[:30]:
        md += [
            f"### {r['id']}",
            f"- context_type={r['context_type']} fact_category={r['fact_category']} gold={r['gold_label']} parsed={r['parsed_label']} raw={r['output_raw']!r}",
            f"- context: {r['context']}",
            f"- hypothesis: {r['hypothesis']}",
            "",
        ]
    md += ["## Next scientific meaning", "", summary["next_action"], "", "## Files", "",
           f"- prompts: `{PROMPT_DIR / 'teacher_prompts_all.jsonl'}`",
           f"- labeled rows: `{OUT_DIR / 'teacher_labeled_rows.jsonl'}`",
           f"- invariance rows: `{OUT_DIR / 'source_bridge_invariance.jsonl'}`",
           f"- summary JSON: `{OUT_DIR / 'teacher_panel_summary.json'}`"]
    ((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/natural_role_transform_panel/teacher_panel_summary.md')).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "overall_accuracy": summary["overall_accuracy"],
        "bridge_accuracy": bridge_acc,
        "false_reject_bridge_accuracy": false_reject_bridge,
        "source_bridge_gold_invariance": source_bridge_gold_inv,
        "premise_positive": premise_positive,
        "decision": decision,
        "summary_json": str(OUT_DIR / "teacher_panel_summary.json"),
        "summary_md": str((OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/natural_role_transform_panel/teacher_panel_summary.md')),
    }, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    an = sub.add_parser("analyze")
    an.add_argument("--gpu0-outputs", default=str(PROMPT_DIR / "teacher_outputs_gpu0.jsonl"))
    an.add_argument("--gpu1-outputs", default=str(PROMPT_DIR / "teacher_outputs_gpu1.jsonl"))
    an.add_argument("--teacher-model", default="qwen3.5-9b")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
