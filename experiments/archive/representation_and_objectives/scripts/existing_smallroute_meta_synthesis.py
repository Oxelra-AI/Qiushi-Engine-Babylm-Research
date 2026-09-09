#!/usr/bin/env python3
"""research: synthesize older small relation/binding route evidence.

This is a CPU-only evidence-consolidation script.  It reads earlier scientific notes
and JSON summaries about BSM / paired-crossview / relation screens and records why
the new transition substrate should not be scaled before a small matched probe.
"""
from __future__ import annotations
import json, re, csv
from pathlib import Path

USER_ROOT = Path.cwd()
OUT = USER_ROOT / "experiments/archive/representation_and_objectives/data/existing_smallroute_synthesis"
NOTE = USER_ROOT / "research/notes/representation_and_objectives/existing_smallroute_meta_synthesis.md"
SOURCES = {
    "initial_model_studies_step338_bsm_1m_eval_note": USER_ROOT / "research/notes/initial_model_studies/legal_bsm_1m_eval.md",
    "initial_model_studies_step338_bsm_1m_interpretation": USER_ROOT / "research/notes/initial_model_studies/legal_bsm_1m_interpretation.md",
    "initial_model_studies_step345_bsm_4m_trace": USER_ROOT / "research/notes/initial_model_studies/4m_trace_eval.md",
    "initial_model_studies_pair_vs_shuffle_1m": USER_ROOT / "research/notes/initial_model_studies/pair_vs_shuffle_1m_profile.md",
    "initial_model_studies_crossview_adjacent_vs_shuffled_1m": USER_ROOT / "research/notes/initial_model_studies/crossview_adjacent_vs_shuffled_1m_profile.md",
    "initial_model_studies_route_after_crossview_negative": USER_ROOT / "research/notes/initial_model_studies/route_after_crossview_negative.md",
}

MANUAL_FINDINGS = [
    {
        "source": "INITIAL_MODEL_STUDIES research legal BSM 1M",
        "evidence_path": str(SOURCES["initial_model_studies_step338_bsm_1m_eval_note"]),
        "comparison": "bsm_coherent_20pct minus bsm_swapped_20pct at ~1M",
        "positive": "BLiMP +1.94, Entity +0.73, EWoK +0.55",
        "negative": "Supplement -2.80, GlobalPIQA_parallel -3.89, GlobalPIQA_nonparallel -3.00, Reading -0.68",
        "mechanism_read": "binding probes had 0.000 both-correct for all arms, so column movement did not show learned entity-conditioned binding",
        "route_consequence": "do not scale exact BSM replacement without matched-update/reference and mechanism probe",
    },
    {
        "source": "INITIAL_MODEL_STUDIES research paired BSM 4M trace",
        "evidence_path": str(SOURCES["initial_model_studies_step345_bsm_4m_trace"]),
        "comparison": "bsm_paired_coherent minus bsm_paired_swapped at 4M",
        "positive": "EWoK +3.09, BLiMP +0.33",
        "negative": "Entity -0.04, GlobalPIQA_parallel -1.95, GlobalPIQA_nonparallel -2.00, Reading -0.025; coherent minus matched reference GlobalPIQA mean -4.93",
        "mechanism_read": "continuous binding probes remained at 0.000 both-correct across available checkpoints",
        "route_consequence": "corpus-derived relation material can move EWoK, but may directly harm GlobalPIQA unless conditional-world structure is cleaner and matched controls are used",
    },
    {
        "source": "INITIAL_MODEL_STUDIES research WikiAuto pair adjacency",
        "evidence_path": str(SOURCES["initial_model_studies_pair_vs_shuffle_1m"]),
        "comparison": "source-rewrite adjacent minus shuffled at 1M",
        "positive": "mean EWoK +1.27 and Entity +0.69",
        "negative": "mean Supplement -1.40, BLiMP -0.10, COMPS -0.33, Reading nearly flat",
        "mechanism_read": "same-source adjacency has some relation/entity signal but not broad endpoint strength",
        "route_consequence": "supports same-proposition recurrence as a mechanism but not a standalone SOTA route",
    },
    {
        "source": "INITIAL_MODEL_STUDIES research/67 active cross-view",
        "evidence_path": str(SOURCES["initial_model_studies_route_after_crossview_negative"]),
        "comparison": "active cross-view adjacent minus shuffled",
        "positive": "Entity +0.085 only",
        "negative": "mean EWoK -0.69 and broad columns not improved",
        "mechanism_read": "source-side exact-overlap anchor prediction with visible simplification did not produce a reliable useful signal",
        "route_consequence": "avoid route variants that only mask overlap anchors or rely on pair adjacency without richer relation supervision",
    },
]


def main():
    OUT.mkdir(parents=True, exist_ok=True); NOTE.parent.mkdir(parents=True, exist_ok=True)
    source_status = {}
    snippets = {}
    for name, path in SOURCES.items():
        source_status[name] = {"path": str(path), "exists": path.exists(), "size": path.stat().st_size if path.exists() else None}
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            # retain compact source text excerpts around key words for later checking
            lower = text.lower()
            hits = []
            for kw in ["globalpiqa", "ewok", "both-correct", "coherent minus swapped", "route implication", "mean delta", "closes"]:
                idx = lower.find(kw.lower())
                if idx >= 0:
                    hits.append(text[max(0, idx-500): idx+1200])
            snippets[name] = hits[:4]
    csv_path = OUT / "existing_smallroute_findings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["source","comparison","positive","negative","mechanism_read","route_consequence","evidence_path"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in MANUAL_FINDINGS:
            w.writerow(row)
    summary = {
        "status": "EXISTING_SMALLROUTE_SYNTHESIS_DONE",
        "source_status": source_status,
        "findings": MANUAL_FINDINGS,
        "scientific_conclusion": "Previous small relation/binding/proposition-pair routes repeatedly produced EWoK or entity signals while failing the intended binding/interaction mechanism or harming GlobalPIQA/Supplement/Reading. The research transition substrate therefore must first be used only in a cheap matched treatment-vs-control probe, with GlobalPIQA readouts preserved, not directly scaled to full 100M training.",
        "files": {"summary_json": str(OUT / "existing_smallroute_synthesis.json"), "findings_csv": str(csv_path), "note": str(NOTE)},
        "source_snippets": snippets,
    }
    (OUT / "existing_smallroute_synthesis.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# research — existing small-route evidence synthesis",
        "",
        "## Purpose",
        "",
        "Before treating the research transition substrate as a new route, this note consolidates older small relation/binding route evidence from INITIAL_MODEL_STUDIES. The point is to preserve what has already been learned: relation-focused data can move EWoK, but earlier forms often failed their own mechanism probes or harmed GlobalPIQA and other columns.",
        "",
        "## Main evidence",
        "",
    ]
    for r in MANUAL_FINDINGS:
        lines += [
            f"### {r['source']}",
            f"- Evidence: `{r['evidence_path']}`",
            f"- Comparison: {r['comparison']}.",
            f"- Positive movement: {r['positive']}.",
            f"- Negative movement: {r['negative']}.",
            f"- Mechanism reading: {r['mechanism_read']}.",
            f"- Consequence: {r['route_consequence']}.",
            "",
        ]
    lines += [
        "## Scientific conclusion",
        "",
        summary["scientific_conclusion"],
        "",
        "The clean next use of the research transition substrate, if needed after the running FW endpoints, is a small shared-coordinate treatment-vs-anchor-control probe that reads both official fast/compatible scores and the EWoK/GlobalPIQA relational readouts. It should not be a direct 100M commitment.",
        "",
        "## Files",
        "",
        f"- summary JSON: `{OUT / 'existing_smallroute_synthesis.json'}`",
        f"- findings CSV: `{csv_path}`",
    ]
    NOTE.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(OUT / "existing_smallroute_synthesis.json"), "csv": str(csv_path), "note": str(NOTE), "findings": len(MANUAL_FINDINGS)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
