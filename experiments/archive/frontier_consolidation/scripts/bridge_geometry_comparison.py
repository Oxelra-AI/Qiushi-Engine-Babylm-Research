#!/usr/bin/env python3
"""research: Bridge geometry comparison in atlas coordinate.

Measures each of 103 retained bridge candidates in the same surface-
discontinuity coordinate as the research atlas (gap-1 adjacency, skip
fraction, source span, source-absent content), split by edit class
(extraction-like vs transformation-like).  Compares to atlas benchmarks
for compact, extractive_balanced, extractive_wide.

CPU-only, no training, no selected evaluation, no leaderboard submission.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, csv, re, statistics, hashlib
from pathlib import Path
from collections import Counter

BRIDGE_ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/improved_fluent_bridge/final_accepted.jsonl')
STRUCTURAL_CSV  = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation/per_pair_structural.csv')
NATURAL_COMPACT = _public_path('experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl')
OUT_DIR         = _public_path('experiments/archive/frontier_consolidation/data/bridge_geometry_comparison')

# Atlas pooled benchmarks from research
ATLAS = {
    "compact":             {"absent_frac": 0.1729, "gap1": 0.7775, "skip": 0.2225, "source_span": 0.8072},
    "extractive_balanced": {"absent_frac": 0.0000, "gap1": 0.5278, "skip": 0.4722, "source_span": 0.9715},
    "extractive_wide":     {"absent_frac": 0.0000, "gap1": 0.6465, "skip": 0.3535, "source_span": 0.9281},
}

# Simple function-word set for content/function separation
FUNC = set("""
a an the this that these those my your his her its our their some any no every
each all both few many much several such what which whose whom
is am are was were be been being have has had do does did will would shall
should can could may might must need dare
and or but nor for yet so if when while because although though since until
unless after before as than not very also just even still already only
in on at by to from with of for about into through during between among
against without within along across it i me we us you he him she her they them
""".split())


def tokenize(text):
    """Lowercase alpha tokens."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text)]


def is_content(w):
    return w not in FUNC and len(w) > 1


def pair_geometry(src_text, gen_text, natural_text=None):
    """Compute atlas-compatible geometry for one (source, generated) pair."""
    src_toks = tokenize(src_text)
    gen_toks = tokenize(gen_text)

    src_content = [(i, w) for i, w in enumerate(src_toks) if is_content(w)]
    gen_content = [w for w in gen_toks if is_content(w)]
    src_content_set = set(w for _, w in src_content)

    # Source-absent content fraction
    absent = sum(1 for w in gen_content if w not in src_content_set)
    absent_frac = absent / max(1, len(gen_content))

    # Map gen content words to source positions (greedy left-to-right)
    avail = {}
    for pos, w in src_content:
        avail.setdefault(w, []).append(pos)
    used = set()
    covered = []
    for w in gen_content:
        if w in avail:
            cands = [p for p in avail[w] if p not in used]
            if cands:
                p = min(cands)
                covered.append(p)
                used.add(p)
    covered_sorted = sorted(set(covered))

    # Gap-1 adjacency (atlas coordinate)
    if len(covered_sorted) < 2:
        gap1, skip = 1.0, 0.0
    else:
        adj = sum(1 for i in range(len(covered_sorted) - 1)
                  if covered_sorted[i + 1] - covered_sorted[i] == 1)
        gap1 = adj / (len(covered_sorted) - 1)
        skip = 1.0 - gap1

    # Source span
    if len(covered_sorted) < 1:
        src_span = 0.0
    else:
        rng = covered_sorted[-1] - covered_sorted[0] + 1
        src_span = rng / max(1, len(src_toks))

    # Function-word fraction and content density
    func_frac = sum(1 for w in gen_toks if w in FUNC) / max(1, len(gen_toks))
    content_density = 1.0 - func_frac

    # Compression ratio (gen words / src words)
    compression = len(gen_toks) / max(1, len(src_toks))

    # If natural compact available, compute its geometry too
    nat = None
    if natural_text:
        nat_toks = tokenize(natural_text)
        nat_content = [w for w in nat_toks if is_content(w)]
        nat_absent = sum(1 for w in nat_content if w not in src_content_set)
        nat_absent_frac = nat_absent / max(1, len(nat_content))
        nat_func = sum(1 for w in nat_toks if w in FUNC) / max(1, len(nat_toks))

        nat_avail = {}
        for pos, w in src_content:
            nat_avail.setdefault(w, []).append(pos)
        nat_used = set()
        nat_cov = []
        for w in nat_content:
            if w in nat_avail:
                cs = [p for p in nat_avail[w] if p not in nat_used]
                if cs:
                    p = min(cs)
                    nat_cov.append(p)
                    nat_used.add(p)
        nat_sorted = sorted(set(nat_cov))
        if len(nat_sorted) < 2:
            nat_gap1, nat_skip = 1.0, 0.0
        else:
            na = sum(1 for i in range(len(nat_sorted) - 1)
                     if nat_sorted[i + 1] - nat_sorted[i] == 1)
            nat_gap1 = na / (len(nat_sorted) - 1)
            nat_skip = 1.0 - nat_gap1
        nat_span = (nat_sorted[-1] - nat_sorted[0] + 1) / max(1, len(src_toks)) if nat_sorted else 0.0
        nat = {
            "absent_frac": nat_absent_frac,
            "gap1": nat_gap1, "skip": nat_skip,
            "source_span": nat_span,
            "content_density": 1.0 - nat_func,
            "func_frac": nat_func,
            "compression": len(nat_toks) / max(1, len(src_toks)),
        }

    return {
        "absent_frac": absent_frac,
        "gap1": gap1, "skip": skip,
        "source_span": src_span,
        "content_density": content_density,
        "func_frac": func_frac,
        "compression": compression,
        "n_covered": len(covered_sorted),
        "n_src_content": len(src_content),
        "gen_words": len(gen_toks),
    }, nat


def distrib(vals, name):
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p10": round(sorted(vals)[max(0, int(len(vals)*0.1))], 4),
        "p90": round(sorted(vals)[min(len(vals)-1, int(len(vals)*0.9))], 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load bridge candidates
    bridges = {}
    with BRIDGE_ACCEPTED.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            bridges[d["pair_id"]] = d

    # Load structural classifications
    struct = {}
    with STRUCTURAL_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            struct[row["pair_id"]] = row

    # Load natural compact references for matched pairs
    nat_compact = {}
    with NATURAL_COMPACT.open() as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["pair_id"] in bridges:
                nat_compact[d["pair_id"]] = d

    # Classify: extraction-like vs transformation-like
    EXTRACT_CLASSES = {"verbatim_substring", "deletion_reorder"}
    TRANSFORM_CLASSES = {"light_restructure", "substantive_restructure"}

    # Compute per-pair geometry
    results = []
    for pid, bd in bridges.items():
        sd = struct.get(pid, {})
        ec = sd.get("edit_class", "unknown")
        is_transform = ec in TRANSFORM_CLASSES
        is_extract = ec in EXTRACT_CLASSES

        nat_text = bd.get("natural_compact_text") or nat_compact.get(pid, {}).get("view_text")
        geom, nat_geom = pair_geometry(bd["source_text"], bd["generated_text"], nat_text)
        geom["pair_id"] = pid
        geom["edit_class"] = ec
        geom["is_transform"] = is_transform
        geom["bucket"] = bd.get("prototype_bucket", "")
        if nat_geom:
            geom["natural_compact"] = nat_geom
        results.append(geom)

    # Aggregate by group
    groups = {
        "all_bridge": results,
        "extraction_like": [r for r in results if r.get("edit_class") in EXTRACT_CLASSES],
        "transformation_like": [r for r in results if r.get("edit_class") in TRANSFORM_CLASSES],
        "substantive_only": [r for r in results if r.get("edit_class") == "substantive_restructure"],
    }

    summary = {"n_total": len(results), "groups": {}}
    nat_summary = {}
    
    for gname, rows in groups.items():
        if not rows:
            continue
        agg = {}
        for metric in ["absent_frac", "gap1", "skip", "source_span", "content_density", "func_frac", "compression"]:
            vals = [r[metric] for r in rows]
            agg[metric] = distrib(vals, metric)
        summary["groups"][gname] = {"n": len(rows), "geometry": agg}

        # Natural compact geometry for matched pairs
        nat_vals = [r["natural_compact"] for r in rows if "natural_compact" in r]
        if nat_vals:
            nat_agg = {}
            for metric in ["absent_frac", "gap1", "skip", "source_span", "content_density", "func_frac", "compression"]:
                vs = [nv[metric] for nv in nat_vals]
                nat_agg[metric] = distrib(vs, metric)
            nat_summary[gname] = {"n": len(nat_vals), "geometry": nat_agg}

    summary["natural_compact_matched"] = nat_summary

    # Key comparison: bridge groups vs atlas benchmarks
    comparisons = {}
    for gname, gdata in summary["groups"].items():
        comp = {}
        for ref_name, ref in ATLAS.items():
            deltas = {}
            for metric in ["absent_frac", "gap1", "skip", "source_span"]:
                bridge_mean = gdata["geometry"][metric]["mean"]
                deltas[metric] = round(bridge_mean - ref[metric], 4)
            comp[f"vs_{ref_name}"] = deltas
        comparisons[gname] = comp
    summary["atlas_comparisons"] = comparisons

    # Diagnostic: is transformation_like geometry closer to compact or extractive?
    diag = {}
    if "transformation_like" in summary["groups"]:
        tg = summary["groups"]["transformation_like"]["geometry"]
        for metric in ["gap1", "skip", "source_span"]:
            t_val = tg[metric]["mean"]
            d_compact = abs(t_val - ATLAS["compact"][metric])
            d_ext_bal = abs(t_val - ATLAS["extractive_balanced"][metric])
            d_ext_wide = abs(t_val - ATLAS["extractive_wide"][metric])
            diag[metric] = {
                "bridge_transform_mean": t_val,
                "compact_ref": ATLAS["compact"][metric],
                "ext_balanced_ref": ATLAS["extractive_balanced"][metric],
                "ext_wide_ref": ATLAS["extractive_wide"][metric],
                "dist_to_compact": round(d_compact, 4),
                "dist_to_ext_balanced": round(d_ext_bal, 4),
                "dist_to_ext_wide": round(d_ext_wide, 4),
                "closest": min(
                    [("compact", d_compact), ("ext_balanced", d_ext_bal), ("ext_wide", d_ext_wide)],
                    key=lambda x: x[1]
                )[0],
            }
    summary["proximity_diagnostic"] = diag

    # Write outputs
    with (_public_path('experiments/archive/frontier_consolidation/data/bridge_geometry_comparison/bridge_geometry_comparison.json')).open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Write per-pair CSV for inspection
    with (_public_path('experiments/archive/frontier_consolidation/data/bridge_geometry_comparison/per_pair_geometry.csv')).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "edit_class", "is_transform", "bucket",
            "absent_frac", "gap1", "skip", "source_span",
            "content_density", "func_frac", "compression", "gen_words",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})

    # Write readable summary
    md_lines = [
        "# research bridge geometry comparison\n",
        f"Total bridge candidates: {len(results)}\n",
        "## Atlas benchmarks\n",
    ]
    for ref_name, ref in ATLAS.items():
        md_lines.append(f"- **{ref_name}**: gap1={ref['gap1']:.4f}, skip={ref['skip']:.4f}, "
                       f"span={ref['source_span']:.4f}, absent={ref['absent_frac']:.4f}")
    md_lines.append("\n## Bridge group geometry\n")
    md_lines.append("| group | n | gap1 | skip | source_span | absent_frac | content_density | func_frac | compression |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for gname, gdata in summary["groups"].items():
        g = gdata["geometry"]
        md_lines.append(f"| {gname} | {gdata['n']} | "
                       f"{g['gap1']['mean']:.4f} | {g['skip']['mean']:.4f} | "
                       f"{g['source_span']['mean']:.4f} | {g['absent_frac']['mean']:.4f} | "
                       f"{g['content_density']['mean']:.4f} | {g['func_frac']['mean']:.4f} | "
                       f"{g['compression']['mean']:.4f} |")

    if nat_summary:
        md_lines.append("\n## Matched natural compact geometry (same pairs)\n")
        md_lines.append("| group | n | gap1 | skip | source_span | absent_frac | content_density | func_frac | compression |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for gname, ndata in nat_summary.items():
            g = ndata["geometry"]
            md_lines.append(f"| {gname} | {ndata['n']} | "
                           f"{g['gap1']['mean']:.4f} | {g['skip']['mean']:.4f} | "
                           f"{g['source_span']['mean']:.4f} | {g['absent_frac']['mean']:.4f} | "
                           f"{g['content_density']['mean']:.4f} | {g['func_frac']['mean']:.4f} | "
                           f"{g['compression']['mean']:.4f} |")

    md_lines.append("\n## Proximity diagnostic (transformation-like)\n")
    for metric, d in diag.items():
        md_lines.append(f"- **{metric}**: bridge_transform={d['bridge_transform_mean']:.4f}, "
                       f"compact={d['compact_ref']:.4f}, ext_bal={d['ext_balanced_ref']:.4f}, "
                       f"ext_wide={d['ext_wide_ref']:.4f} → closest to **{d['closest']}**")

    md_lines.append("\n## Interpretation\n")
    md_lines.append("If transformation-like bridge is closest to extractive on gap1/skip/span, "
                   "then structural operations under the source-attested constraint produce "
                   "extraction-like geometry despite surface restructuring — the route adds "
                   "no geometric dimension beyond extractive, which already failed.\n")
    md_lines.append("If transformation-like bridge has gap1/skip closer to compact than to "
                   "extractive, fluent restructuring genuinely changes the data geometry "
                   "and a matched training comparison is justified.\n")

    with (_public_path('research/documents/frontier_consolidation/data/bridge_geometry_comparison/bridge_geometry_comparison.md')).open("w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
