#!/usr/bin/env python3
"""research: benchmark-independent relational content preservation in compact views.

Analyzes whether compact training views systematically lose dynamic relational
markers (verbs of change, spatial prepositions, causal connectives, temporal
markers) more than their general word-retention rate.

This is benchmark-independent: it characterizes the TRAINING DATA properties,
not the evaluation data. The hypothesis is that the dynamics-vs-properties
learning asymmetry found in the EWoK margin analysis (research) is caused by
differential preservation of dynamic vs static relational content in compact views.

Uses the full 18,682 accepted compact rewrites from the medium tier.
"""
import json, re, sys, pathlib
from collections import defaultdict, Counter

REWRITES_PATH = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/medium_compact_analysis"
    "medium_compact_ws_accepted_rewrites.jsonl"
)
OUT_DIR = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/relational_preservation"
)

# ── Benchmark-independent relational marker categories ────────────────
# These are general linguistic markers, not derived from any evaluation task.

DYNAMIC_VERBS = {
    # Physical change
    "break", "breaks", "breaking", "broke", "broken",
    "melt", "melts", "melting", "melted",
    "burn", "burns", "burning", "burned", "burnt",
    "freeze", "freezes", "freezing", "froze", "frozen",
    "dissolve", "dissolves", "dissolving", "dissolved",
    "evaporate", "evaporates", "evaporating", "evaporated",
    "crack", "cracks", "cracking", "cracked",
    "shatter", "shatters", "shattering", "shattered",
    "tear", "tears", "tearing", "tore", "torn",
    "collapse", "collapses", "collapsing", "collapsed",
    "erode", "erodes", "eroding", "eroded",
    "decay", "decays", "decaying", "decayed",
    "rust", "rusts", "rusting", "rusted",
    "corrode", "corrodes", "corroding", "corroded",
    # Physical movement
    "fall", "falls", "falling", "fell", "fallen",
    "rise", "rises", "rising", "rose", "risen",
    "flow", "flows", "flowing", "flowed",
    "pour", "pours", "pouring", "poured",
    "drip", "drips", "dripping", "dripped",
    "splash", "splashes", "splashing", "splashed",
    "slide", "slides", "sliding", "slid",
    "roll", "rolls", "rolling", "rolled",
    "bounce", "bounces", "bouncing", "bounced",
    "float", "floats", "floating", "floated",
    "sink", "sinks", "sinking", "sank", "sunk",
    "spin", "spins", "spinning", "spun",
    "rotate", "rotates", "rotating", "rotated",
    # Physical interaction
    "push", "pushes", "pushing", "pushed",
    "pull", "pulls", "pulling", "pulled",
    "throw", "throws", "throwing", "threw", "thrown",
    "catch", "catches", "catching", "caught",
    "hit", "hits", "hitting",
    "crash", "crashes", "crashing", "crashed",
    "collide", "collides", "colliding", "collided",
    "squeeze", "squeezes", "squeezing", "squeezed",
    "stretch", "stretches", "stretching", "stretched",
    "compress", "compresses", "compressing", "compressed",
    "bend", "bends", "bending", "bent",
    "twist", "twists", "twisting", "twisted",
    "fold", "folds", "folding", "folded",
    # Size/state change
    "grow", "grows", "growing", "grew", "grown",
    "shrink", "shrinks", "shrinking", "shrank", "shrunk",
    "expand", "expands", "expanding", "expanded",
    "contract", "contracts", "contracting", "contracted",
    "change", "changes", "changing", "changed",
    "transform", "transforms", "transforming", "transformed",
    "convert", "converts", "converting", "converted",
}

SPATIAL_PREPS = {
    "above", "below", "beneath", "under", "underneath",
    "over", "inside", "outside", "within",
    "between", "among", "behind", "beside", "alongside",
    "near", "nearby", "across", "through", "along",
    "around", "toward", "towards", "upward", "downward",
    "inward", "outward", "onto", "upon",
    "atop", "beyond", "adjacent",
}

CAUSAL_MARKERS = {
    "because", "since", "therefore", "thus", "hence",
    "consequently", "accordingly", "causes", "caused",
    "causing", "leads", "leading", "results",
    "resulting", "produces", "producing", "creates",
    "creating", "generates", "generating", "triggers",
    "triggering", "prevents", "preventing",
    "enables", "enabling", "requires", "requiring",
}

TEMPORAL_MARKERS = {
    "then", "after", "before", "during", "while",
    "when", "once", "until", "already", "still",
    "yet", "later", "earlier", "previously",
    "subsequently", "eventually", "meanwhile",
    "simultaneously", "initially", "finally",
    "gradually", "suddenly", "immediately",
    "recently", "formerly", "afterward",
    "beforehand", "thereafter",
}

# Static property markers (for contrast)
STATIC_PROPERTY = {
    "is", "are", "was", "were", "has", "have", "had",
    "contains", "consists", "includes", "involves",
    "represents", "indicates", "suggests", "shows",
    "means", "refers", "belongs", "relates",
    "heavy", "light", "hard", "soft", "rigid", "flexible",
    "solid", "liquid", "dense", "thin", "thick",
    "large", "small", "big", "tiny", "huge", "massive",
    "long", "short", "wide", "narrow", "tall",
    "hot", "cold", "warm", "cool",
    "strong", "weak", "tough", "brittle", "fragile",
    "smooth", "rough", "sharp", "dull", "flat",
    "round", "circular", "square", "rectangular",
}


CATEGORIES = {
    "dynamic_verbs": DYNAMIC_VERBS,
    "spatial_preps": SPATIAL_PREPS,
    "causal_markers": CAUSAL_MARKERS,
    "temporal_markers": TEMPORAL_MARKERS,
    "static_property": STATIC_PROPERTY,
}


def count_markers(text, marker_set):
    """Count marker word occurrences in lowercased text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return sum(1 for w in words if w in marker_set)


def analyze_preservation():
    """Analyze relational marker preservation across all accepted compact rows."""
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Per-row tracking
    cat_source_counts = defaultdict(list)  # category → [source_count per row]
    cat_rewrite_counts = defaultdict(list)
    cat_preservation = defaultdict(list)   # per-row preservation rates
    
    # Global counters
    total_rows = 0
    total_source_words = 0
    total_rewrite_words = 0
    
    # Also track rows with high dynamic content
    rows_with_dynamics = 0
    rows_dynamic_source_gt3 = 0
    
    with open(REWRITES_PATH) as f:
        for line in f:
            d = json.loads(line)
            src = d["source_text"]
            rew = d["rewrite_text"]
            total_rows += 1
            total_source_words += d["source_words"]
            total_rewrite_words += d["rewrite_words"]
            
            for cat, markers in CATEGORIES.items():
                sc = count_markers(src, markers)
                rc = count_markers(rew, markers)
                cat_source_counts[cat].append(sc)
                cat_rewrite_counts[cat].append(rc)
                # Preservation rate (avoid div by zero)
                if sc > 0:
                    cat_preservation[cat].append(min(rc / sc, 2.0))
                # else: no markers in source, can't compute preservation
            
            dyn_src = count_markers(src, DYNAMIC_VERBS)
            if dyn_src > 0:
                rows_with_dynamics += 1
            if dyn_src > 3:
                rows_dynamic_source_gt3 += 1
    
    # Compute statistics
    general_retention = total_rewrite_words / total_source_words
    
    results = {
        "status": "RELATIONAL_PRESERVATION",
        "total_rows": total_rows,
        "general_word_retention_rate": round(general_retention, 4),
        "total_source_words": total_source_words,
        "total_rewrite_words": total_rewrite_words,
        "rows_with_any_dynamic_verb": rows_with_dynamics,
        "rows_with_dynamic_verbs_gt3": rows_dynamic_source_gt3,
    }
    
    lines = ["# research — Relational Content Preservation in Compact Views\n"]
    lines.append(f"Analyzed {total_rows} accepted compact rewrites.\n")
    lines.append(f"General word retention rate: {general_retention:.4f}\n")
    lines.append(f"Rows with any dynamic verb in source: {rows_with_dynamics} "
                 f"({100*rows_with_dynamics/total_rows:.1f}%)")
    lines.append(f"Rows with >3 dynamic verbs in source: {rows_dynamic_source_gt3} "
                 f"({100*rows_dynamic_source_gt3/total_rows:.1f}%)\n")
    
    lines.append("\n## Per-category marker preservation\n")
    lines.append(f"{'Category':<20s} {'SourceTotal':>12s} {'RewriteTotal':>12s} "
                 f"{'Retention':>10s} {'Excess':>10s} "
                 f"{'MeanPres':>10s} {'MedianPres':>10s} {'RowsWSrc':>10s}")
    
    cat_results = {}
    for cat in CATEGORIES:
        src_total = sum(cat_source_counts[cat])
        rew_total = sum(cat_rewrite_counts[cat])
        
        if src_total > 0:
            retention = rew_total / src_total
        else:
            retention = 0.0
        
        excess = retention - general_retention
        
        pres_rates = cat_preservation[cat]
        n_with_source = len(pres_rates)
        mean_pres = sum(pres_rates) / n_with_source if pres_rates else 0.0
        sorted_pres = sorted(pres_rates)
        median_pres = sorted_pres[len(sorted_pres) // 2] if sorted_pres else 0.0
        
        cat_results[cat] = {
            "source_total": src_total,
            "rewrite_total": rew_total,
            "retention_rate": round(retention, 4),
            "excess_vs_general": round(excess, 4),
            "rows_with_source_markers": n_with_source,
            "mean_preservation": round(mean_pres, 4),
            "median_preservation": round(median_pres, 4),
        }
        
        lines.append(
            f"{cat:<20s} {src_total:>12d} {rew_total:>12d} "
            f"{retention:>10.4f} {excess:>+10.4f} "
            f"{mean_pres:>10.4f} {median_pres:>10.4f} {n_with_source:>10d}"
        )
    
    results["per_category"] = cat_results
    
    # Differential preservation analysis
    lines.append("\n\n## Differential Preservation Interpretation\n")
    
    dyn_ret = cat_results["dynamic_verbs"]["retention_rate"]
    stat_ret = cat_results["static_property"]["retention_rate"]
    gen_ret = general_retention
    
    lines.append(f"Dynamic verb retention: {dyn_ret:.4f} "
                 f"(excess vs general: {dyn_ret - gen_ret:+.4f})")
    lines.append(f"Static property retention: {stat_ret:.4f} "
                 f"(excess vs general: {stat_ret - gen_ret:+.4f})")
    lines.append(f"Spatial prep retention: "
                 f"{cat_results['spatial_preps']['retention_rate']:.4f} "
                 f"(excess: {cat_results['spatial_preps']['excess_vs_general']:+.4f})")
    lines.append(f"Causal marker retention: "
                 f"{cat_results['causal_markers']['retention_rate']:.4f} "
                 f"(excess: {cat_results['causal_markers']['excess_vs_general']:+.4f})")
    lines.append(f"Temporal marker retention: "
                 f"{cat_results['temporal_markers']['retention_rate']:.4f} "
                 f"(excess: {cat_results['temporal_markers']['excess_vs_general']:+.4f})")
    
    dynamic_gap = dyn_ret - stat_ret
    lines.append(f"\nDynamic-vs-static retention gap: {dynamic_gap:+.4f}")
    
    if dynamic_gap < -0.02:
        lines.append("→ Compact views preferentially lose dynamic relational markers.")
    elif dynamic_gap > 0.02:
        lines.append("→ Compact views preferentially lose static property markers.")
    else:
        lines.append("→ No clear differential between dynamic and static markers.")
    
    results["dynamic_static_gap"] = round(dynamic_gap, 4)
    
    # Save
    with open(OUT_DIR / "relational_preservation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    lines.append(f"\n\nMachine-readable: `{OUT_DIR / 'relational_preservation.json'}`\n")
    with open((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/relational_preservation/relational_preservation.md'), "w") as f:
        f.write("\n".join(lines))
    
    print(json.dumps({
        "status": results["status"],
        "general_retention": results["general_word_retention_rate"],
        "dynamic_verb_retention": cat_results["dynamic_verbs"]["retention_rate"],
        "static_property_retention": cat_results["static_property"]["retention_rate"],
        "spatial_prep_retention": cat_results["spatial_preps"]["retention_rate"],
        "causal_marker_retention": cat_results["causal_markers"]["retention_rate"],
        "temporal_marker_retention": cat_results["temporal_markers"]["retention_rate"],
        "dynamic_static_gap": results["dynamic_static_gap"],
        "out_json": str(OUT_DIR / "relational_preservation.json"),
        "out_md": str((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/relational_preservation/relational_preservation.md')),
    }, indent=2))


if __name__ == "__main__":
    analyze_preservation()
