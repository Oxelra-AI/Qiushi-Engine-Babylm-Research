#!/usr/bin/env python3
"""research: CPU feasibility scout for a corpus-derived structural contrastive-margin probe.

This does NOT score any model and does NOT use official evaluation items. It scans the legal
10M compact-view-reinvest pool for high-precision-ish natural text frames that could support
minimal perturbations measuring relation/state/procedure sensitivity rather than target-token NLL.

The output is a feasibility map: source-balanced candidate counts, simple examples, and warnings
about likely sparsity/noise. It is intentionally conservative and meant to guide the next construction
step, not to be the final frozen probe builder.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path("experiments/archive/frontier_consolidation")
POOL = ROOT / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
OUT_DIR = ROOT / "data/contrastive_margin_feasibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:[.,:]\d+)*")
CAP_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# Relation/opposition lexicons are generic English classes; none are benchmark-derived item strings.
COMPARATIVE_CUES = {
    "more", "less", "fewer", "greater", "larger", "smaller", "higher", "lower", "better", "worse",
    "older", "younger", "colder", "warmer", "hotter", "cooler", "closer", "farther", "further",
    "faster", "slower", "heavier", "lighter", "stronger", "weaker", "longer", "shorter",
}
SPATIAL_REL_CUES = {"above", "below", "inside", "outside", "under", "over", "behind", "beside", "between", "near", "from", "to", "into", "onto"}
REPORT_VERBS = {"told", "tell", "said", "say", "asked", "ask", "warned", "warn", "informed", "inform", "lied", "lie", "believed", "believe", "thought", "think", "knew", "know", "reported", "report"}
STATE_VERBS = {
    "put", "puts", "placed", "place", "places", "moved", "move", "moves", "took", "take", "takes",
    "removed", "remove", "removes", "brought", "bring", "brings", "left", "leave", "leaves",
    "gave", "give", "gives", "got", "get", "gets", "kept", "keep", "keeps", "filled", "fill", "fills",
    "emptied", "empty", "empties", "added", "add", "adds", "poured", "pour", "pours", "opened", "open", "opens",
    "closed", "close", "closes", "hid", "hide", "hides", "found", "find", "finds", "lost", "lose", "loses",
}
TEMPORAL_CUES = {"first", "then", "next", "finally", "before", "after", "once", "until", "when", "while", "later", "previously", "subsequently"}
POLARITY_CUES = {"no", "not", "never", "neither", "nor", "nobody", "nothing", "nowhere", "without", "cannot", "can't", "won't", "n't"}
NPI_CUES = {"any", "ever", "at", "all", "yet"}

STOP_ARG = {
    "the", "a", "an", "this", "that", "these", "those", "it", "they", "he", "she", "we", "you", "i", "there",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "has", "have", "had", "will", "would",
    "can", "could", "may", "might", "must", "should", "shall", "of", "in", "on", "to", "from", "with", "for", "by",
}


def toks(text: str) -> List[str]:
    return WORD_RE.findall(text)


def low_tokens(text: str) -> List[str]:
    return [w.lower() for w in toks(text)]


def rough_entities(text: str) -> List[str]:
    caps = [m.group(0) for m in CAP_RE.finditer(text)]
    # remove sentence-initial generic capitalized words that are common source artifacts
    bad = {"The", "This", "That", "These", "Those", "There", "Based", "Safety", "For", "On", "In", "A", "An", "If", "When"}
    caps = [c for c in caps if c not in bad]
    return caps[:10]


def content_candidates(text: str) -> List[str]:
    out = []
    for w in low_tokens(text):
        if len(w) >= 4 and w not in STOP_ARG and not w.endswith("ly"):
            out.append(w)
    return out


def add_sample(samples: Dict[str, List[dict]], fam: str, rec: dict, max_per_family=12):
    if len(samples[fam]) < max_per_family:
        samples[fam].append(rec)


def sentence_windows(sentences: List[str], max_window=5) -> List[Tuple[int, int, str]]:
    windows = []
    n = len(sentences)
    for i in range(n):
        acc = []
        for j in range(i, min(n, i + max_window)):
            acc.append(sentences[j])
            if j > i:
                windows.append((i, j, " ".join(acc)))
    return windows


def main():
    family_counts = Counter()
    source_family_counts: Dict[str, Counter] = defaultdict(Counter)
    family_source_unique_rows: Dict[str, set] = defaultdict(set)
    samples: Dict[str, List[dict]] = defaultdict(list)
    source_counts = Counter()
    total_rows = 0

    with POOL.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            o = json.loads(line)
            source = o.get("source", "unknown")
            text = o.get("text", "")
            if source.startswith("neutral_"):
                continue
            total_rows += 1
            source_counts[source] += 1
            sentences = [s.strip() for s in SENT_SPLIT.split(text) if 6 <= len(s.split()) <= 80]
            row_id = o.get("example_id", line_idx)

            for si, s in enumerate(sentences):
                lt = low_tokens(s)
                ents = rough_entities(s)
                contents = content_candidates(s)
                lt_set = set(lt)

                # A. Comparative/directed relation: contains than plus a reversible comparative cue.
                if "than" in lt_set and (COMPARATIVE_CUES & lt_set) and len(contents) >= 3:
                    fam = "comparative_relation"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_index": si, "text": s[:360], "cues": sorted(COMPARATIVE_CUES & lt_set)})

                # Spatial/directed preposition relation with two candidate content arguments.
                if (SPATIAL_REL_CUES & lt_set) and len(contents) >= 4:
                    fam = "spatial_directional_relation"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_index": si, "text": s[:360], "cues": sorted(SPATIAL_REL_CUES & lt_set)[:6]})

                # Belief/report/perspective-holder: report verb plus that/quote or two names/pronouns.
                if (REPORT_VERBS & lt_set) and (("that" in lt_set) or len(ents) >= 2 or '"' in s or "'" in s):
                    fam = "belief_report_role_binding"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_index": si, "text": s[:360], "verbs": sorted(REPORT_VERBS & lt_set), "entities": ents[:5]})

                # Temporal/procedural: temporal cue plus action/update verb.
                if (TEMPORAL_CUES & lt_set) and (STATE_VERBS & lt_set or len(contents) >= 6):
                    fam = "temporal_procedure_order"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_index": si, "text": s[:360], "cues": sorted(TEMPORAL_CUES & lt_set), "verbs": sorted(STATE_VERBS & lt_set)[:6]})

                # Polarity-composition: negation plus NPI or relation cue, not just a negative sentence.
                if (POLARITY_CUES & lt_set) and ((NPI_CUES & lt_set) or "than" in lt_set or (SPATIAL_REL_CUES & lt_set)):
                    fam = "polarity_relation_composition"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_index": si, "text": s[:360], "polarity": sorted(POLARITY_CUES & lt_set), "relation": sorted((NPI_CUES | SPATIAL_REL_CUES | {"than"}) & lt_set)[:6]})

            # C. Multi-operation entity/state windows: at least two update verbs and repeated candidate content/entity.
            for wi, wj, win in sentence_windows(sentences, max_window=5):
                lt = low_tokens(win)
                verb_hits = [v for v in lt if v in STATE_VERBS]
                if len(verb_hits) < 2:
                    continue
                cont = content_candidates(win)
                cc = Counter(cont)
                repeated = [k for k, v in cc.items() if v >= 2]
                ents = rough_entities(win)
                if len(repeated) >= 1 or len(ents) >= 2:
                    depth = min(len(verb_hits), 5)
                    fam = "multi_operation_entity_state"
                    family_counts[fam] += 1
                    source_family_counts[source][fam] += 1
                    family_source_unique_rows[fam].add(row_id)
                    add_sample(samples, fam, {"source": source, "example_id": row_id, "sentence_span": [wi, wj], "depth_proxy_update_verbs": depth, "text": win[:480], "verbs": verb_hits[:10], "repeated_terms": repeated[:6], "entities": ents[:6]})
                    break  # one candidate row is enough for feasibility counting

    # Summaries
    by_family = {}
    for fam, n in sorted(family_counts.items()):
        by_family[fam] = {
            "candidate_events": n,
            "unique_rows": len(family_source_unique_rows[fam]),
            "by_source": dict(sorted((src, cnts[fam]) for src, cnts in source_family_counts.items() if cnts[fam])),
        }

    feasibility_notes = []
    for fam, info in by_family.items():
        n = info["candidate_events"]
        if n >= 5000:
            status = "ample_natural_material"
        elif n >= 1000:
            status = "likely_enough_with_filtering"
        elif n >= 200:
            status = "sparse_needs_synthetic_or_relaxed_rules"
        else:
            status = "very_sparse_not_primary"
        info["rough_feasibility"] = status
        feasibility_notes.append(f"{fam}: {n} events / {info['unique_rows']} rows -> {status}")

    result = {
        "status": "CPU_FEASIBILITY_SCOUT_COMPLETE",
        "pool": str(POOL),
        "total_non_neutral_rows_scanned": total_rows,
        "source_counts": dict(source_counts),
        "by_family": by_family,
        "samples": samples,
        "warnings": [
            "Counts are heuristic frame-yields, not final validated minimal pairs.",
            "The final probe must freeze transformations, filters, tokenizer-length controls, and aggregation before scoring checkpoints.",
            "Official benchmark text/labels were not read by this script; only legal pool text was scanned.",
            "Natural state/procedure frames may still be too noisy; controlled micro-discourses from corpus-attested frames may be needed if validated natural windows are sparse.",
        ],
    }

    out_json = OUT_DIR / "contrastive_margin_feasibility_scout.json"
    out_md = (OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/contrastive_margin_feasibility/contrastive_margin_feasibility_scout.md')
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# research contrastive-margin feasibility scout")
    lines.append("")
    lines.append("CPU-only scan of the legal 10M compact-view-reinvest pool for corpus-derived frames that could support coherent-vs-perturbed structural margins. No model scoring; no official evaluation items or labels.")
    lines.append("")
    lines.append(f"Rows scanned: {total_rows}")
    lines.append("")
    lines.append("## Family yields")
    lines.append("")
    lines.append("| family | candidate events | unique rows | rough feasibility | top sources |")
    lines.append("|---|---:|---:|---|---|")
    for fam, info in by_family.items():
        top_sources = ", ".join(f"{k}:{v}" for k, v in sorted(info["by_source"].items(), key=lambda kv: -kv[1])[:5])
        lines.append(f"| {fam} | {info['candidate_events']} | {info['unique_rows']} | {info['rough_feasibility']} | {top_sources} |")
    lines.append("")
    lines.append("## Samples")
    for fam in sorted(samples):
        lines.append("")
        lines.append(f"### {fam}")
        for r in samples[fam][:8]:
            cue = {k: v for k, v in r.items() if k not in {"text"}}
            lines.append(f"- `{json.dumps(cue, ensure_ascii=False)}` {r['text']}")
    lines.append("")
    lines.append("## Warnings")
    for w in result["warnings"]:
        lines.append(f"- {w}")
    lines.append("")
    lines.append(f"JSON: `{out_json}`")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": str(out_json), "out_md": str(out_md), "by_family": {k: {"candidate_events": v["candidate_events"], "unique_rows": v["unique_rows"], "rough_feasibility": v["rough_feasibility"]} for k, v in by_family.items()}}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
