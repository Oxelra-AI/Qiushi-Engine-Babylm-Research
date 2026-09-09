#!/usr/bin/env python3
"""research: analyze per-item EWoK margins across the 2×2 treatment grid.

Consumes the output of ewok_margin_scorer.py to produce:
  1. Per-domain margin distributions and flip analysis
  2. Treatment-effect by item: margin_reinvest − margin_clean
  3. Confidence classification: near-zero flips vs confidently wrong
  4. Seed stability: items that flip between seeds within same treatment
  5. Late-checkpoint dynamics: 90M vs 100M reinvest margins
  6. Cross-seed treatment interaction by domain

All analysis is over existing data only; no model loading or GPU use.
"""
import json, csv, sys, pathlib, math
from collections import defaultdict
import statistics

# ── paths ──────────────────────────────────────────────────────────────
MARGIN_ROOT = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/ewok_margins"
)
OUT_DIR = pathlib.Path(
    "experiments/archive/frontier_consolidation/data/ewok_margin_analysis"
)

# ── load margins ───────────────────────────────────────────────────────
def load_margins(model_name):
    """Load per-item margins from JSON; return list of dicts."""
    fpath = MARGIN_ROOT / model_name / "ewok_margins.json"
    if not fpath.exists():
        return None
    with open(fpath) as f:
        data = json.load(f)
    return data["items"]


def build_item_key(item):
    """Unique key for matching items across models."""
    return (item["domain"], item["line_idx"])


def index_by_key(items):
    """Dict from item_key → item record."""
    return {build_item_key(it): it for it in items}


# ── analysis functions ─────────────────────────────────────────────────
def domain_margin_stats(items):
    """Per-domain margin distribution statistics."""
    by_domain = defaultdict(list)
    for it in items:
        by_domain[it["domain"]].append(it["margin"])
    
    stats = {}
    for d in sorted(by_domain):
        margins = by_domain[d]
        n = len(margins)
        correct = sum(1 for m in margins if m > 0)
        wrong = sum(1 for m in margins if m < 0)
        tied = n - correct - wrong
        
        stats[d] = {
            "n": n,
            "accuracy": round(100.0 * (correct + 0.5 * tied) / n, 4),
            "mean_margin": round(statistics.mean(margins), 6),
            "median_margin": round(statistics.median(margins), 6),
            "std_margin": round(statistics.stdev(margins), 6) if n > 1 else 0.0,
            "frac_correct": round(correct / n, 4),
            "frac_wrong": round(wrong / n, 4),
            "frac_near_zero_0.5": round(
                sum(1 for m in margins if abs(m) < 0.5) / n, 4),
            "frac_near_zero_1.0": round(
                sum(1 for m in margins if abs(m) < 1.0) / n, 4),
            "p10_margin": round(sorted(margins)[max(0, n // 10)], 6),
            "p25_margin": round(sorted(margins)[n // 4], 6),
            "p75_margin": round(sorted(margins)[3 * n // 4], 6),
            "p90_margin": round(sorted(margins)[min(n - 1, 9 * n // 10)], 6),
        }
    return stats


def flip_analysis(items_a, items_b, label_a="A", label_b="B"):
    """Identify items that flip between two models.
    
    Returns per-domain counts:
      a_correct_b_wrong: item correct in A, wrong in B
      a_wrong_b_correct: item wrong in A, correct in B
      both_correct, both_wrong
    """
    idx_a = index_by_key(items_a)
    idx_b = index_by_key(items_b)
    
    by_domain = defaultdict(lambda: {
        "a_correct_b_wrong": 0, "a_wrong_b_correct": 0,
        "both_correct": 0, "both_wrong": 0, "total": 0,
    })
    
    for key in sorted(idx_a):
        if key not in idx_b:
            continue
        a = idx_a[key]
        b = idx_b[key]
        domain = a["domain"]
        ca = a["margin"] > 0
        cb = b["margin"] > 0
        
        by_domain[domain]["total"] += 1
        if ca and cb:
            by_domain[domain]["both_correct"] += 1
        elif ca and not cb:
            by_domain[domain]["a_correct_b_wrong"] += 1
        elif not ca and cb:
            by_domain[domain]["a_wrong_b_correct"] += 1
        else:
            by_domain[domain]["both_wrong"] += 1
    
    # Add net flip and net flip rate
    for d in by_domain:
        r = by_domain[d]
        r["net_flip_to_b"] = r["a_wrong_b_correct"] - r["a_correct_b_wrong"]
        r["net_flip_rate"] = round(r["net_flip_to_b"] / r["total"], 4) if r["total"] else 0
    
    return dict(by_domain)


def treatment_effect_by_item(clean_items, reinvest_items):
    """For each matched item, compute margin_reinvest − margin_clean.
    
    Returns per-domain summary of treatment effects.
    """
    idx_c = index_by_key(clean_items)
    idx_r = index_by_key(reinvest_items)
    
    by_domain = defaultdict(list)
    item_effects = []
    
    for key in sorted(idx_c):
        if key not in idx_r:
            continue
        c = idx_c[key]
        r = idx_r[key]
        te = r["margin"] - c["margin"]
        by_domain[c["domain"]].append(te)
        item_effects.append({
            "domain": c["domain"],
            "line_idx": c["line_idx"],
            "ConceptA": c["ConceptA"],
            "ConceptB": c["ConceptB"],
            "clean_margin": c["margin"],
            "reinvest_margin": r["margin"],
            "treatment_effect": round(te, 6),
        })
    
    domain_te = {}
    for d in sorted(by_domain):
        effects = by_domain[d]
        n = len(effects)
        domain_te[d] = {
            "n": n,
            "mean_te": round(statistics.mean(effects), 6),
            "median_te": round(statistics.median(effects), 6),
            "std_te": round(statistics.stdev(effects), 6) if n > 1 else 0.0,
            "frac_positive": round(sum(1 for e in effects if e > 0) / n, 4),
            "frac_negative": round(sum(1 for e in effects if e < 0) / n, 4),
            "p10_te": round(sorted(effects)[max(0, n // 10)], 6),
            "p90_te": round(sorted(effects)[min(n - 1, 9 * n // 10)], 6),
        }
    
    return domain_te, item_effects


def confidence_classification(items, threshold=1.0):
    """Classify items by margin confidence.
    
    Categories:
      confident_correct: margin > threshold
      fragile_correct: 0 < margin <= threshold
      fragile_wrong: -threshold <= margin < 0
      confident_wrong: margin < -threshold
    """
    by_domain = defaultdict(lambda: {
        "confident_correct": 0, "fragile_correct": 0,
        "fragile_wrong": 0, "confident_wrong": 0, "total": 0,
    })
    
    for it in items:
        d = it["domain"]
        m = it["margin"]
        by_domain[d]["total"] += 1
        if m > threshold:
            by_domain[d]["confident_correct"] += 1
        elif m > 0:
            by_domain[d]["fragile_correct"] += 1
        elif m >= -threshold:
            by_domain[d]["fragile_wrong"] += 1
        else:
            by_domain[d]["confident_wrong"] += 1
    
    # Add fractions
    for d in by_domain:
        r = by_domain[d]
        t = r["total"]
        for cat in ["confident_correct", "fragile_correct",
                     "fragile_wrong", "confident_wrong"]:
            r[f"frac_{cat}"] = round(r[cat] / t, 4) if t else 0
    
    return dict(by_domain)


def late_checkpoint_dynamics(items_100m, items_90m):
    """Compare 90M vs 100M reinvest margins at item level."""
    idx_100 = index_by_key(items_100m)
    idx_90 = index_by_key(items_90m)
    
    by_domain = defaultdict(list)
    
    for key in sorted(idx_100):
        if key not in idx_90:
            continue
        m100 = idx_100[key]["margin"]
        m90 = idx_90[key]["margin"]
        delta = m100 - m90  # positive = 100M better
        by_domain[idx_100[key]["domain"]].append({
            "delta": delta,
            "m100": m100,
            "m90": m90,
            "flip_100_to_wrong": m90 > 0 and m100 <= 0,
            "flip_100_to_correct": m90 <= 0 and m100 > 0,
        })
    
    domain_summary = {}
    for d in sorted(by_domain):
        records = by_domain[d]
        n = len(records)
        deltas = [r["delta"] for r in records]
        domain_summary[d] = {
            "n": n,
            "mean_delta_100_minus_90": round(statistics.mean(deltas), 6),
            "frac_100_better": round(sum(1 for d in deltas if d > 0) / n, 4),
            "frac_90_better": round(sum(1 for d in deltas if d < 0) / n, 4),
            "flips_100_to_wrong": sum(r["flip_100_to_wrong"] for r in records),
            "flips_100_to_correct": sum(r["flip_100_to_correct"] for r in records),
            "net_flips_100_gains": (
                sum(r["flip_100_to_correct"] for r in records) -
                sum(r["flip_100_to_wrong"] for r in records)
            ),
        }
    
    return domain_summary


def treatment_interaction(clean43022, clean43122, reinvest43022, reinvest43122):
    """Full 2×2 treatment interaction at item level.
    
    For each item, compute:
      TE43022 = margin_reinvest43022 - margin_clean43022
      TE43122 = margin_reinvest43122 - margin_clean43122
      DiD = TE43022 - TE43122
    
    Returns per-domain summary.
    """
    ic22 = index_by_key(clean43022)
    ic12 = index_by_key(clean43122)
    ir22 = index_by_key(reinvest43022)
    ir12 = index_by_key(reinvest43122)
    
    by_domain = defaultdict(list)
    
    for key in sorted(ic22):
        if key not in ic12 or key not in ir22 or key not in ir12:
            continue
        mc22 = ic22[key]["margin"]
        mc12 = ic12[key]["margin"]
        mr22 = ir22[key]["margin"]
        mr12 = ir12[key]["margin"]
        
        te22 = mr22 - mc22
        te12 = mr12 - mc12
        did = te22 - te12
        
        by_domain[ic22[key]["domain"]].append({
            "te22": te22, "te12": te12, "did": did,
            "mc22": mc22, "mc12": mc12, "mr22": mr22, "mr12": mr12,
        })
    
    domain_summary = {}
    for d in sorted(by_domain):
        records = by_domain[d]
        n = len(records)
        te22s = [r["te22"] for r in records]
        te12s = [r["te12"] for r in records]
        dids = [r["did"] for r in records]
        
        domain_summary[d] = {
            "n": n,
            "mean_TE43022": round(statistics.mean(te22s), 6),
            "mean_TE43122": round(statistics.mean(te12s), 6),
            "mean_DiD": round(statistics.mean(dids), 6),
            "std_TE43022": round(statistics.stdev(te22s), 6) if n > 1 else 0,
            "std_TE43122": round(statistics.stdev(te12s), 6) if n > 1 else 0,
            "frac_TE43022_positive": round(
                sum(1 for t in te22s if t > 0) / n, 4),
            "frac_TE43122_positive": round(
                sum(1 for t in te12s if t > 0) / n, 4),
            "frac_DiD_positive": round(
                sum(1 for d in dids if d > 0) / n, 4),
        }
    
    return domain_summary


# ── main ───────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all available margin files
    models = ["clean43022", "clean43122", "reinvest43022", "reinvest43122",
              "reinvest43022_90M"]
    data = {}
    for m in models:
        items = load_margins(m)
        if items is not None:
            data[m] = items
            print(f"Loaded {m}: {len(items)} items")
        else:
            print(f"MISSING: {m}")
    
    if not data:
        print("ERROR: no margin data found")
        sys.exit(1)
    
    results = {"status": "EWOK_MARGIN_ANALYSIS"}
    lines = ["# research — EWoK per-item margin analysis\n"]
    
    # 1. Per-model domain margin statistics
    results["domain_margin_stats"] = {}
    for m in data:
        stats = domain_margin_stats(data[m])
        results["domain_margin_stats"][m] = stats
        lines.append(f"\n## {m} — domain margin statistics\n")
        for d, s in sorted(stats.items()):
            lines.append(
                f"  {d}: acc={s['accuracy']:.1f}%, "
                f"mean_margin={s['mean_margin']:.4f}, "
                f"frac_near_zero_1.0={s['frac_near_zero_1.0']:.3f}, "
                f"frac_wrong={s['frac_wrong']:.3f}"
            )
    
    # 2. Treatment-effect by seed (if both clean and reinvest available)
    for seed in ["43022", "43122"]:
        ck = f"clean{seed}"
        rk = f"reinvest{seed}"
        if ck in data and rk in data:
            te_domain, te_items = treatment_effect_by_item(data[ck], data[rk])
            results[f"treatment_effect_{seed}"] = te_domain
            lines.append(f"\n## Treatment effect seed {seed}\n")
            for d, s in sorted(te_domain.items()):
                lines.append(
                    f"  {d}: mean_TE={s['mean_te']:.4f}, "
                    f"median_TE={s['median_te']:.4f}, "
                    f"frac_positive={s['frac_positive']:.3f}"
                )
    
    # 3. Flip analysis: clean vs reinvest, same seed
    for seed in ["43022", "43122"]:
        ck = f"clean{seed}"
        rk = f"reinvest{seed}"
        if ck in data and rk in data:
            flips = flip_analysis(data[ck], data[rk],
                                   f"clean{seed}", f"reinvest{seed}")
            results[f"flips_{seed}"] = flips
            lines.append(f"\n## Flip analysis: clean{seed} → reinvest{seed}\n")
            for d, r in sorted(flips.items()):
                lines.append(
                    f"  {d}: clean✓→reinvest✗={r['a_correct_b_wrong']}, "
                    f"clean✗→reinvest✓={r['a_wrong_b_correct']}, "
                    f"net_flip_rate={r['net_flip_rate']:+.4f}"
                )
    
    # 4. Confidence classification for reinvest models
    for m in ["reinvest43022", "reinvest43122"]:
        if m in data:
            conf = confidence_classification(data[m], threshold=1.0)
            results[f"confidence_{m}"] = conf
            lines.append(f"\n## Confidence classification: {m} (threshold=1.0)\n")
            for d, r in sorted(conf.items()):
                lines.append(
                    f"  {d}: conf_correct={r['frac_confident_correct']:.3f}, "
                    f"frag_correct={r['frac_fragile_correct']:.3f}, "
                    f"frag_wrong={r['frac_fragile_wrong']:.3f}, "
                    f"conf_wrong={r['frac_confident_wrong']:.3f}"
                )
    
    # 5. Late-checkpoint dynamics
    if "reinvest43022" in data and "reinvest43022_90M" in data:
        late = late_checkpoint_dynamics(data["reinvest43022"],
                                         data["reinvest43022_90M"])
        results["late_checkpoint_dynamics"] = late
        lines.append("\n## Late checkpoint dynamics: 100M vs 90M (reinvest43022)\n")
        for d, s in sorted(late.items()):
            lines.append(
                f"  {d}: mean_delta={s['mean_delta_100_minus_90']:.4f}, "
                f"frac_100_better={s['frac_100_better']:.3f}, "
                f"net_flips_100_gains={s['net_flips_100_gains']}"
            )
    
    # 6. Full 2×2 treatment interaction
    if all(k in data for k in ["clean43022", "clean43122",
                                "reinvest43022", "reinvest43122"]):
        interaction = treatment_interaction(
            data["clean43022"], data["clean43122"],
            data["reinvest43022"], data["reinvest43122"]
        )
        results["treatment_interaction_2x2"] = interaction
        lines.append("\n## 2×2 treatment interaction by domain\n")
        for d, s in sorted(interaction.items()):
            lines.append(
                f"  {d}: TE43022={s['mean_TE43022']:.4f}, "
                f"TE43122={s['mean_TE43122']:.4f}, "
                f"DiD={s['mean_DiD']:.4f}, "
                f"frac_DiD+={s['frac_DiD_positive']:.3f}"
            )
    
    # 7. Seed stability within treatment
    for treatment in ["clean", "reinvest"]:
        k22 = f"{treatment}43022"
        k12 = f"{treatment}43122"
        if k22 in data and k12 in data:
            flips = flip_analysis(data[k22], data[k12],
                                   f"{treatment}_seed43022",
                                   f"{treatment}_seed43122")
            results[f"seed_stability_{treatment}"] = flips
            lines.append(f"\n## Seed stability: {treatment} 43022 vs 43122\n")
            for d, r in sorted(flips.items()):
                lines.append(
                    f"  {d}: 43022✓→43122✗={r['a_correct_b_wrong']}, "
                    f"43022✗→43122✓={r['a_wrong_b_correct']}, "
                    f"net_flip_rate={r['net_flip_rate']:+.4f}"
                )
    
    # Save
    with open(OUT_DIR / "ewok_margin_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    lines.append(f"\n\nMachine-readable: `{OUT_DIR / 'ewok_margin_analysis.json'}`\n")
    with open((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/ewok_margin_analysis/ewok_margin_analysis.md'), "w") as f:
        f.write("\n".join(lines))
    
    print(json.dumps({
        "status": results["status"],
        "models_loaded": list(data.keys()),
        "out_json": str(OUT_DIR / "ewok_margin_analysis.json"),
        "out_md": str((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/ewok_margin_analysis/ewok_margin_analysis.md')),
    }, indent=2))


if __name__ == "__main__":
    main()
