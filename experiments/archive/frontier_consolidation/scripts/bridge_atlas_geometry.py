#!/usr/bin/env python3
"""research: Bridge geometry in ATLAS-COMPATIBLE coordinate.

Uses exact same monotone alignment and gap/skip computation as the
research atlas, so results can be compared directly to atlas benchmarks:
  compact:             gap1=0.7775, skip=0.2225, span=0.8072
  extractive_balanced: gap1=0.5278, skip=0.4722, span=0.9715
  extractive_wide:     gap1=0.6465, skip=0.3535, span=0.9281

CPU-only.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, csv, re, statistics
from pathlib import Path
from collections import defaultdict, Counter

BRIDGE_ACCEPTED = _public_path('experiments/archive/frontier_consolidation/data/improved_fluent_bridge/final_accepted.jsonl')
STRUCTURAL_CSV  = _public_path('experiments/archive/frontier_consolidation/data/bridge_structural_transformation/per_pair_structural.csv')
OUT_DIR         = _public_path('experiments/archive/frontier_consolidation/data/bridge_atlas_geometry')

ATLAS = {
    "compact":             {"gap1": 0.7775, "skip": 0.2225, "span": 0.8072, "absent": 0.1729},
    "extractive_balanced": {"gap1": 0.5278, "skip": 0.4722, "span": 0.9715, "absent": 0.0000},
    "extractive_wide":     {"gap1": 0.6465, "skip": 0.3535, "span": 0.9281, "absent": 0.0000},
}


# ===== Atlas-compatible functions (replicated from research) =====

def split_words(text):
    """Simple whitespace tokenizer matching atlas."""
    return text.split()

def norm_word(w):
    """Lowercase + strip punctuation, matching atlas."""
    return re.sub(r"[^\w'-]", "", w.lower()).strip("'-")

CONTENT_EXCLUDE = set("""
a an the this that these those my your his her its our their some any no every
each all both few many much several such what which whose whom
is am are was were be been being have has had do does did will would shall
should can could may might must need dare
and or but nor for yet so if when while because although though since until
unless after before as than not very also just even still already only
in on at by to from with of for about into through during between among
against without within along across it i me we us you he him she her they them
""".split())

def is_content(norm):
    return len(norm) > 1 and norm not in CONTENT_EXCLUDE


def monotone_align(src_norms, view_norms):
    """Atlas-exact monotone alignment over ALL normalized words."""
    positions = defaultdict(list)
    for i, n in enumerate(src_norms):
        if n:
            positions[n].append(i)
    cursor_by_word = defaultdict(int)
    prev = -1
    out = []
    for n in view_norms:
        if not n or n not in positions:
            out.append(None)
            continue
        pos_list = positions[n]
        cur = cursor_by_word[n]
        while cur < len(pos_list) and pos_list[cur] <= prev:
            cur += 1
        if cur < len(pos_list):
            pos = pos_list[cur]
            out.append(pos)
            prev = pos
            cursor_by_word[n] = cur + 1
        else:
            out.append(None)
    return out


def atlas_metrics(src_text, view_text):
    """Compute atlas-compatible gap1/skip/span/absent metrics."""
    src_words = split_words(src_text)
    view_words = split_words(view_text)
    src_norms = [norm_word(w) for w in src_words]
    view_norms = [norm_word(w) for w in view_words]

    # Bag copy/absent counts (content only, for absent_frac)
    src_bag = Counter(n for n in src_norms if n)
    content_total = 0
    absent_content = 0
    for n in view_norms:
        if not n:
            continue
        if is_content(n):
            content_total += 1
            if src_bag.get(n, 0) > 0:
                src_bag[n] -= 1
            else:
                absent_content += 1
    absent_frac = absent_content / content_total if content_total else 0.0

    # Monotone alignment over ALL words
    aligned = monotone_align(src_norms, view_norms)
    aligned_positions = [p for p in aligned if p is not None]

    # Gap1/skip between consecutive aligned positions in view order
    adjacent_aligned = 0
    gap1_count = 0
    skip_count = 0
    for a, b in zip(aligned, aligned[1:]):
        if a is None or b is None:
            continue
        g = b - a
        if g <= 0:
            continue
        adjacent_aligned += 1
        if g == 1:
            gap1_count += 1
        elif g > 1:
            skip_count += 1

    gap1_frac = gap1_count / adjacent_aligned if adjacent_aligned else None
    skip_frac = skip_count / adjacent_aligned if adjacent_aligned else None

    # Source span
    span = 0.0
    if aligned_positions:
        rng = max(aligned_positions) - min(aligned_positions) + 1
        span = rng / len(src_words) if src_words else 0.0

    # Aligned fraction
    aligned_frac = len(aligned_positions) / len(view_norms) if view_norms else 0.0

    return {
        "gap1": gap1_frac,
        "skip": skip_frac,
        "source_span": span,
        "absent_frac": absent_frac,
        "aligned_frac": aligned_frac,
        "aligned_words": len(aligned_positions),
        "adjacent_aligned_pairs": adjacent_aligned,
        "gap1_count": gap1_count,
        "skip_count": skip_count,
        "view_words": len(view_words),
        "src_words": len(src_words),
        "compression": len(view_words) / max(1, len(src_words)),
    }


def distrib(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "p10": round(sorted(vals)[max(0, int(len(vals)*0.1))], 4),
        "p90": round(sorted(vals)[min(len(vals)-1, int(len(vals)*0.9))], 4),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load bridge candidates
    bridges = {}
    with BRIDGE_ACCEPTED.open() as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                bridges[d["pair_id"]] = d

    # Load structural classifications
    struct = {}
    with STRUCTURAL_CSV.open() as f:
        for row in csv.DictReader(f):
            struct[row["pair_id"]] = row

    EXTRACT = {"verbatim_substring", "deletion_reorder"}
    TRANSFORM = {"light_restructure", "substantive_restructure"}

    # Compute per-pair atlas metrics for BOTH bridge and matched natural compact
    records = []
    for pid, bd in bridges.items():
        ec = struct.get(pid, {}).get("edit_class", "unknown")
        
        bridge_m = atlas_metrics(bd["source_text"], bd["generated_text"])
        nat_m = atlas_metrics(bd["source_text"], bd["natural_compact_text"]) if bd.get("natural_compact_text") else None

        records.append({
            "pair_id": pid,
            "edit_class": ec,
            "is_transform": ec in TRANSFORM,
            "is_extract": ec in EXTRACT,
            "bucket": bd.get("prototype_bucket", ""),
            "bridge": bridge_m,
            "natural_compact": nat_m,
        })

    # Group aggregation with POOLED statistics (atlas-compatible)
    groups = {
        "all_bridge": records,
        "extraction_like": [r for r in records if r["is_extract"]],
        "transformation_like": [r for r in records if r["is_transform"]],
        "substantive_only": [r for r in records if r["edit_class"] == "substantive_restructure"],
    }

    summary = {"n_total": len(records), "atlas_benchmarks": ATLAS}

    for gname, rows in groups.items():
        if not rows:
            continue
        # Per-pair mean (atlas-compatible "mean" column)
        bridge_means = {
            m: distrib([r["bridge"][m] for r in rows])
            for m in ["gap1", "skip", "source_span", "absent_frac", "aligned_frac", "compression"]
        }
        # Pooled (atlas-compatible "pooled" column: sum counts across pairs)
        total_gap1 = sum(r["bridge"]["gap1_count"] for r in rows)
        total_skip = sum(r["bridge"]["skip_count"] for r in rows)
        total_adj = sum(r["bridge"]["adjacent_aligned_pairs"] for r in rows)
        total_absent = sum(r["bridge"]["absent_frac"] * r["bridge"]["view_words"] for r in rows)  # approx
        
        pooled_gap1 = total_gap1 / total_adj if total_adj else None
        pooled_skip = total_skip / total_adj if total_adj else None

        # Same for matched natural compact
        nat_rows = [r for r in rows if r["natural_compact"] is not None]
        nat_means = {
            m: distrib([r["natural_compact"][m] for r in nat_rows])
            for m in ["gap1", "skip", "source_span", "absent_frac", "aligned_frac", "compression"]
        } if nat_rows else {}
        nat_total_gap1 = sum(r["natural_compact"]["gap1_count"] for r in nat_rows)
        nat_total_skip = sum(r["natural_compact"]["skip_count"] for r in nat_rows)
        nat_total_adj = sum(r["natural_compact"]["adjacent_aligned_pairs"] for r in nat_rows)
        nat_pooled_gap1 = nat_total_gap1 / nat_total_adj if nat_total_adj else None
        nat_pooled_skip = nat_total_skip / nat_total_adj if nat_total_adj else None

        summary[gname] = {
            "n": len(rows),
            "bridge_per_pair_mean": bridge_means,
            "bridge_pooled": {
                "gap1": round(pooled_gap1, 4) if pooled_gap1 is not None else None,
                "skip": round(pooled_skip, 4) if pooled_skip is not None else None,
            },
            "natural_compact_per_pair_mean": nat_means,
            "natural_compact_pooled": {
                "gap1": round(nat_pooled_gap1, 4) if nat_pooled_gap1 is not None else None,
                "skip": round(nat_pooled_skip, 4) if nat_pooled_skip is not None else None,
            },
        }

    # Proximity analysis
    diag = {}
    for gname in ["all_bridge", "extraction_like", "transformation_like", "substantive_only"]:
        if gname not in summary or not isinstance(summary[gname], dict):
            continue
        bg = summary[gname]["bridge_per_pair_mean"]
        ng = summary[gname].get("natural_compact_per_pair_mean", {})
        bp = summary[gname]["bridge_pooled"]
        np_ = summary[gname]["natural_compact_pooled"]
        
        row = {"bridge_pooled_gap1": bp.get("gap1"), "bridge_pooled_skip": bp.get("skip")}
        if bg.get("gap1") and bg["gap1"].get("mean") is not None:
            row["bridge_mean_gap1"] = bg["gap1"]["mean"]
            row["bridge_mean_skip"] = bg["skip"]["mean"]
        if ng.get("gap1") and ng["gap1"].get("mean") is not None:
            row["natcomp_mean_gap1"] = ng["gap1"]["mean"]
            row["natcomp_mean_skip"] = ng["skip"]["mean"]
        row["natcomp_pooled_gap1"] = np_.get("gap1")
        row["natcomp_pooled_skip"] = np_.get("skip")
        
        # Distance to atlas benchmarks (using pooled, which is atlas-comparable)
        if bp.get("gap1") is not None:
            for ref_name, ref in ATLAS.items():
                row[f"dist_to_{ref_name}_gap1"] = round(abs(bp["gap1"] - ref["gap1"]), 4)
                row[f"dist_to_{ref_name}_skip"] = round(abs(bp["skip"] - ref["skip"]), 4)
        diag[gname] = row

    summary["proximity"] = diag

    # Write outputs
    with (_public_path('experiments/archive/frontier_consolidation/data/bridge_atlas_geometry/bridge_atlas_geometry.json')).open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Readable summary
    lines = [
        "# research bridge atlas-compatible geometry\n",
        "## Atlas pooled benchmarks",
        "| variant | pooled gap1 | pooled skip | mean source span | absent frac |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, ref in ATLAS.items():
        lines.append(f"| {name} | {ref['gap1']:.4f} | {ref['skip']:.4f} | {ref['span']:.4f} | {ref['absent']:.4f} |")

    lines.append("\n## Bridge groups — pooled gap1/skip (atlas-comparable)")
    lines.append("| group | n | bridge pooled gap1 | bridge pooled skip | natcomp pooled gap1 | natcomp pooled skip |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for gname in ["all_bridge", "extraction_like", "transformation_like", "substantive_only"]:
        if gname not in summary or not isinstance(summary[gname], dict):
            continue
        d = summary[gname]
        bp, np_ = d["bridge_pooled"], d["natural_compact_pooled"]
        lines.append(f"| {gname} | {d['n']} | "
                    f"{bp['gap1']:.4f} | {bp['skip']:.4f} | "
                    f"{np_['gap1']:.4f} | {np_['skip']:.4f} |")

    lines.append("\n## Bridge groups — per-pair mean (for distribution reference)")
    lines.append("| group | n | mean gap1 | mean skip | mean span | mean absent | mean compress |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for gname in ["all_bridge", "extraction_like", "transformation_like", "substantive_only"]:
        if gname not in summary or not isinstance(summary[gname], dict):
            continue
        d = summary[gname]
        bg = d["bridge_per_pair_mean"]
        lines.append(f"| {gname} | {d['n']} | "
                    f"{bg['gap1']['mean']:.4f} | {bg['skip']['mean']:.4f} | "
                    f"{bg['source_span']['mean']:.4f} | {bg['absent_frac']['mean']:.4f} | "
                    f"{bg['compression']['mean']:.4f} |")

    lines.append("\n## Matched natural compact — per-pair mean (same 103 pairs)")
    lines.append("| group | n | mean gap1 | mean skip | mean span | mean absent | mean compress |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for gname in ["all_bridge", "extraction_like", "transformation_like", "substantive_only"]:
        if gname not in summary or not isinstance(summary[gname], dict):
            continue
        d = summary[gname]
        ng = d.get("natural_compact_per_pair_mean", {})
        if ng.get("gap1"):
            lines.append(f"| {gname} | {d['n']} | "
                        f"{ng['gap1']['mean']:.4f} | {ng['skip']['mean']:.4f} | "
                        f"{ng['source_span']['mean']:.4f} | {ng['absent_frac']['mean']:.4f} | "
                        f"{ng['compression']['mean']:.4f} |")

    lines.append("\n## Proximity to atlas benchmarks (bridge pooled gap1)")
    for gname, row in diag.items():
        dists = {k.replace("dist_to_","").replace("_gap1",""): v 
                 for k, v in row.items() if k.startswith("dist_to_") and k.endswith("_gap1")}
        if dists:
            closest = min(dists, key=dists.get)
            lines.append(f"- **{gname}**: bridge_pooled_gap1={row.get('bridge_pooled_gap1'):.4f}, "
                        f"closest={closest} (dist {dists[closest]:.4f})")

    with (_public_path('research/documents/frontier_consolidation/data/bridge_atlas_geometry/bridge_atlas_geometry.md')).open("w") as f:
        f.write("\n".join(lines) + "\n")

    # Per-pair CSV
    with (_public_path('experiments/archive/frontier_consolidation/data/bridge_atlas_geometry/per_pair_atlas_geometry.csv')).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "edit_class", "is_transform",
                        "bridge_gap1", "bridge_skip", "bridge_span", "bridge_absent",
                        "natcomp_gap1", "natcomp_skip", "natcomp_span", "natcomp_absent"])
        for r in records:
            bm = r["bridge"]
            nm = r.get("natural_compact") or {}
            writer.writerow([
                r["pair_id"], r["edit_class"], r["is_transform"],
                bm.get("gap1"), bm.get("skip"), bm.get("source_span"), bm.get("absent_frac"),
                nm.get("gap1"), nm.get("skip"), nm.get("source_span"), nm.get("absent_frac"),
            ])

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
