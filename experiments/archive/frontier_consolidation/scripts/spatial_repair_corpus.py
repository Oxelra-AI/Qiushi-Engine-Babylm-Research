#!/usr/bin/env python3
"""research: Build spatial-preservation repair variants of compact_view_reinvest.

Identifies compact pairs that lose spatial prepositions, substitutes near-length
rewrites for those pairs (same source, longer rewrite preserving spatial markers),
then drops lowest-value added reinvestment pairs to maintain exact 10M word budget.

Produces two variants:
  A: spatial_repair — near-length substitution for spatial-lossy pairs only
  B: spatial_enrich — same plus: among remaining added pairs, preferentially
     retain science_physical / causal_relational domains and drop no_domain

Both write 10M pool JSONL and 100M (10-pass) training JSONL, ready for training.
"""
import json, hashlib, os, re, sys
from pathlib import Path
from collections import Counter

ROOT = Path("experiments/archive/frontier_consolidation")
DATA = ROOT / "data"
OUT  = DATA / "spatial_repair_corpus"
OUT.mkdir(parents=True, exist_ok=True)

# ── spatial preposition wordlist (same as research) ──────────────────────────
SPATIAL_PREPS = {
    "above", "below", "beneath", "under", "over", "inside", "outside",
    "within", "between", "among", "through", "across", "along", "around",
    "behind", "beside", "beyond", "near", "against", "toward", "towards",
    "onto", "upon", "into", "throughout", "underneath", "alongside", "atop",
    "amid", "amidst",
}

def count_spatial(text):
    """Count spatial preposition tokens in text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return sum(1 for w in words if w in SPATIAL_PREPS)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

# ── 1. Load pair data ───────────────────────────────────────────────────────
print("Loading pair data...")

# Core compact pairs (10094) — keyed by shared `key` field
core_compact = {}
with open(DATA / "density_core_reinvestment_medium_riskhard/selected_compact_core_pairs.jsonl") as f:
    for line in f:
        p = json.loads(line)
        core_compact[p["key"]] = p

# Added compact pairs (2061 reinvestment)
added_compact = {}
with open(DATA / "density_core_reinvestment_medium_riskhard/selected_compact_added_pairs.jsonl") as f:
    for line in f:
        p = json.loads(line)
        added_compact[p["pair_id"]] = p

# Near-length core pairs (10094, same sources as core_compact)
near_core = {}
with open(DATA / "density_core_reinvestment_medium_riskhard/selected_near_core_pairs.jsonl") as f:
    for line in f:
        p = json.loads(line)
        near_core[p["key"]] = p  # same key as core_compact

print(f"  Core compact: {len(core_compact)}, Added compact: {len(added_compact)}, Near core: {len(near_core)}")

# Build key → near pair lookup for substitution (key is shared)
near_by_key = {}
for k, p in near_core.items():
    near_by_key[k] = p

# ── 2. Analyze spatial preservation per core compact pair ───────────────────
print("Analyzing spatial preservation...")

spatial_analysis = []
for k, p in core_compact.items():
    src_spatial = count_spatial(p["source_text"])
    rew_spatial = count_spatial(p["rewrite_text"])
    
    # Find matching near pair by shared key
    near_p = near_by_key.get(k)
    near_spatial = count_spatial(near_p["rewrite_text"]) if near_p else 0
    near_pair_words = (near_p["source_words"] + near_p["rewrite_words"]) if near_p else 0
    
    compact_pair_words = p["source_words"] + p["rewrite_words"]
    word_cost_if_substituted = near_pair_words - compact_pair_words if near_p else 0
    
    spatial_analysis.append({
        "key": k,
        "doc_id": p["doc_id"],
        "src_spatial": src_spatial,
        "compact_rew_spatial": rew_spatial,
        "near_rew_spatial": near_spatial,
        "spatial_lost": max(0, src_spatial - rew_spatial),
        "spatial_recovered_by_near": (near_spatial - rew_spatial) if near_p else 0,
        "compact_pair_words": compact_pair_words,
        "near_pair_words": near_pair_words,
        "word_cost": word_cost_if_substituted,
        "has_near": near_p is not None,
    })

# Sort by spatial loss (most lost first), then by recovery potential
spatial_analysis.sort(key=lambda x: (-x["spatial_lost"], -x["spatial_recovered_by_near"]))

# Stats
total_with_source_spatial = sum(1 for s in spatial_analysis if s["src_spatial"] > 0)
total_with_loss = sum(1 for s in spatial_analysis if s["spatial_lost"] > 0)
total_recovery_possible = sum(1 for s in spatial_analysis if s["spatial_recovered_by_near"] > 0)
total_word_cost_all_losses = sum(s["word_cost"] for s in spatial_analysis if s["spatial_lost"] > 0)

print(f"  Core pairs with source spatial preps: {total_with_source_spatial}")
print(f"  Core pairs with spatial loss: {total_with_loss}")
print(f"  Core pairs where near-length recovers spatial: {total_recovery_possible}")
print(f"  Total word cost if all lossy pairs substituted: {total_word_cost_all_losses}")

# ── 3. Select substitution targets ─────────────────────────────────────────
# Only substitute where: spatial_lost > 0 AND near recovers some AND has_near
substitution_targets = [
    s for s in spatial_analysis
    if s["spatial_lost"] > 0 and s["spatial_recovered_by_near"] > 0 and s["has_near"]
]
# Sort by efficiency: recovery per word cost (word_cost can be 0 or negative for short sources)
for t in substitution_targets:
    t["efficiency"] = t["spatial_recovered_by_near"] / max(1, abs(t["word_cost"]) + 1)

substitution_targets.sort(key=lambda x: -x["efficiency"])

total_sub_word_cost = sum(t["word_cost"] for t in substitution_targets)
total_sub_spatial_recovered = sum(t["spatial_recovered_by_near"] for t in substitution_targets)

print(f"\n  Substitution targets: {len(substitution_targets)}")
print(f"  Total word cost for all substitutions: {total_sub_word_cost}")
print(f"  Total spatial tokens recovered: {total_sub_spatial_recovered}")

# ── 4. Budget analysis ─────────────────────────────────────────────────────
# Current budget: changed block = 423,520 words
# After substituting near-length for spatial-lossy pairs, need to free words 
# by dropping added pairs

BUDGET = 423520
current_core_words = sum(p["source_words"] + p["rewrite_words"] for p in core_compact.values())
current_added_words = sum(p["source_words"] + p["rewrite_words"] for p in added_compact.values())
current_neutral_topup = 9
print(f"\n  Current core compact pair words: {current_core_words}")
print(f"  Current added compact pair words: {current_added_words}")
print(f"  Current neutral topup: {current_neutral_topup}")
print(f"  Total: {current_core_words + current_added_words + current_neutral_topup}")

# After substitution, core words increase by total_sub_word_cost
new_core_words = current_core_words + total_sub_word_cost
words_needed_to_free = max(0, new_core_words + current_added_words + current_neutral_topup - BUDGET)
print(f"\n  New core words after all substitutions: {new_core_words}")
print(f"  Words to free from added pairs: {words_needed_to_free}")

# ── 5. Build Variant A: spatial_repair ──────────────────────────────────────
print("\n=== Building Variant A: spatial_repair ===")

# Apply all spatial substitutions to core pairs
sub_keys = set(t["key"] for t in substitution_targets)

variant_a_core = {}
for k, p in core_compact.items():
    if k in sub_keys:
        # Substitute rewrite with near-length version
        near_p = near_by_key[k]
        new_p = dict(p)
        new_p["rewrite_text"] = near_p["rewrite_text"]
        new_p["rewrite_words"] = near_p["rewrite_words"]
        new_p["pair_words"] = new_p["source_words"] + new_p["rewrite_words"]
        new_p["substituted"] = True
        new_p["original_rewrite_words"] = p["rewrite_words"]
        variant_a_core[k] = new_p
    else:
        variant_a_core[k] = dict(p)

new_core_total = sum(p["source_words"] + p["rewrite_words"] for p in variant_a_core.values())

# Sort added pairs by domain priority for dropping (drop no_domain first)
DOMAIN_PRIORITY = {
    "science_physical": 0,
    "causal_relational": 1,
    "quant_numeric": 2,
    "geography_places": 3,
    "institutions_society": 4,
    "people_history": 5,
    "media_culture": 6,
    "no_domain": 7,
}

def pair_domain_score(p):
    """Lower = more valuable (keep), higher = less valuable (drop first)."""
    domains = p.get("domain_hits", [])
    if not domains:
        return 7
    if isinstance(domains, dict):
        domains = list(domains.keys())
    best = min(DOMAIN_PRIORITY.get(d, 7) for d in domains)
    return best

added_list = list(added_compact.values())
# Sort: drop least valuable first (highest domain score, then fewest words)
added_list.sort(key=lambda p: (-pair_domain_score(p), p["source_words"] + p["rewrite_words"]))

# Greedily drop from the end (worst domain, fewest words) until budget fits
variant_a_added = list(added_list)
words_freed = 0
dropped_a = []
while variant_a_added and (new_core_total + sum(p["source_words"] + p["rewrite_words"] for p in variant_a_added) + current_neutral_topup > BUDGET):
    dropped = variant_a_added.pop()
    words_freed += dropped["source_words"] + dropped["rewrite_words"]
    dropped_a.append(dropped)

variant_a_added_words = sum(p["source_words"] + p["rewrite_words"] for p in variant_a_added)
variant_a_total = new_core_total + variant_a_added_words + current_neutral_topup
variant_a_deficit = BUDGET - variant_a_total

print(f"  Substituted {len(sub_keys)} core pairs (near-length for spatial-lossy)")
print(f"  New core words: {new_core_total}")
print(f"  Dropped {len(dropped_a)} added pairs, freed {words_freed} words")
print(f"  Retained added pairs: {len(variant_a_added)}")
print(f"  Variant A total: {variant_a_total}  (budget: {BUDGET}, deficit: {variant_a_deficit})")

# Domain composition of retained added pairs
retained_domains_a = Counter()
for p in variant_a_added:
    for d in (p.get("domain_hits") or []):
        retained_domains_a[d] += 1
dropped_domains_a = Counter()
for p in dropped_a:
    for d in (p.get("domain_hits") or []):
        dropped_domains_a[d] += 1

print(f"  Retained added domain composition: {dict(retained_domains_a.most_common())}")
print(f"  Dropped added domain composition: {dict(dropped_domains_a.most_common())}")

# ── 6. Build Variant B: spatial_enrich ──────────────────────────────────────
print("\n=== Building Variant B: spatial_enrich ===")

# Same core substitution as variant A
# But for added pairs: preferentially keep science_physical and causal_relational
# Sort added pairs: keep most physical/causal first, drop no_domain/media first
added_list_b = list(added_compact.values())
added_list_b.sort(key=lambda p: (pair_domain_score(p), -(p["source_words"] + p["rewrite_words"])))

variant_b_added = list(added_list_b)
dropped_b = []
while variant_b_added and (new_core_total + sum(p["source_words"] + p["rewrite_words"] for p in variant_b_added) + current_neutral_topup > BUDGET):
    dropped = variant_b_added.pop()
    words_freed_b = dropped["source_words"] + dropped["rewrite_words"]
    dropped_b.append(dropped)

variant_b_added_words = sum(p["source_words"] + p["rewrite_words"] for p in variant_b_added)
variant_b_total = new_core_total + variant_b_added_words + current_neutral_topup
variant_b_deficit = BUDGET - variant_b_total

print(f"  Substituted {len(sub_keys)} core pairs (same as A)")
print(f"  New core words: {new_core_total}")
print(f"  Dropped {len(dropped_b)} added pairs")
print(f"  Retained added pairs: {len(variant_b_added)}")
print(f"  Variant B total: {variant_b_total}  (budget: {BUDGET}, deficit: {variant_b_deficit})")

retained_domains_b = Counter()
for p in variant_b_added:
    for d in (p.get("domain_hits") or []):
        retained_domains_b[d] += 1
dropped_domains_b = Counter()
for p in dropped_b:
    for d in (p.get("domain_hits") or []):
        dropped_domains_b[d] += 1

print(f"  Retained added domain composition: {dict(retained_domains_b.most_common())}")
print(f"  Dropped added domain composition: {dict(dropped_domains_b.most_common())}")

# ── 7. Verify spatial improvement ──────────────────────────────────────────
print("\n=== Spatial preservation verification ===")

# Original compact reinvest spatial stats
orig_src_spatial = sum(count_spatial(p["source_text"]) for p in core_compact.values())
orig_rew_spatial = sum(count_spatial(p["rewrite_text"]) for p in core_compact.values())
orig_added_src = sum(count_spatial(p["source_text"]) for p in added_compact.values())
orig_added_rew = sum(count_spatial(p["rewrite_text"]) for p in added_compact.values())

# Variant A spatial stats
new_rew_spatial = sum(count_spatial(p["rewrite_text"]) for p in variant_a_core.values())
va_added_src = sum(count_spatial(p["source_text"]) for p in variant_a_added)
va_added_rew = sum(count_spatial(p["rewrite_text"]) for p in variant_a_added)

# Variant B spatial stats (same core, different added)
vb_added_src = sum(count_spatial(p["source_text"]) for p in variant_b_added)
vb_added_rew = sum(count_spatial(p["rewrite_text"]) for p in variant_b_added)

print(f"  Original core: src_spatial={orig_src_spatial}, rew_spatial={orig_rew_spatial}, retention={orig_rew_spatial/max(1,orig_src_spatial):.4f}")
print(f"  Variant A core: src_spatial={orig_src_spatial}, rew_spatial={new_rew_spatial}, retention={new_rew_spatial/max(1,orig_src_spatial):.4f}")
print(f"  Original added: src={orig_added_src}, rew={orig_added_rew}")
print(f"  Variant A added: src={va_added_src}, rew={va_added_rew}")
print(f"  Variant B added: src={vb_added_src}, rew={vb_added_rew}")
print(f"  Total original spatial tokens: src={orig_src_spatial+orig_added_src}, rew={orig_rew_spatial+orig_added_rew}")
print(f"  Total Variant A spatial tokens: src={orig_src_spatial+va_added_src}, rew={new_rew_spatial+va_added_rew}")
print(f"  Total Variant B spatial tokens: src={orig_src_spatial+vb_added_src}, rew={new_rew_spatial+vb_added_rew}")

# ── 8. Save analysis before materialization ─────────────────────────────────
analysis = {
    "status": "SPATIAL_REPAIR_ANALYSIS",
    "substitution": {
        "total_core_pairs": len(core_compact),
        "pairs_with_source_spatial": total_with_source_spatial,
        "pairs_with_spatial_loss": total_with_loss,
        "pairs_where_near_recovers": total_recovery_possible,
        "substitution_targets": len(substitution_targets),
        "total_word_cost": total_sub_word_cost,
        "total_spatial_recovered": total_sub_spatial_recovered,
    },
    "variant_a": {
        "name": "spatial_repair",
        "core_pairs": len(variant_a_core),
        "added_pairs_retained": len(variant_a_added),
        "added_pairs_dropped": len(dropped_a),
        "substituted_pairs": len(sub_keys),
        "total_unique_sources": len(set(p["doc_id"] for p in variant_a_core.values())) + len(set(p["doc_id"] for p in variant_a_added)),
        "total_words": variant_a_total,
        "budget_deficit": variant_a_deficit,
        "retained_added_domains": dict(retained_domains_a.most_common()),
        "dropped_added_domains": dict(dropped_domains_a.most_common()),
    },
    "variant_b": {
        "name": "spatial_enrich",
        "core_pairs": len(variant_a_core),
        "added_pairs_retained": len(variant_b_added),
        "added_pairs_dropped": len(dropped_b),
        "substituted_pairs": len(sub_keys),
        "total_unique_sources": len(set(p["doc_id"] for p in variant_a_core.values())) + len(set(p["doc_id"] for p in variant_b_added)),
        "total_words": variant_b_total,
        "budget_deficit": variant_b_deficit,
        "retained_added_domains": dict(retained_domains_b.most_common()),
        "dropped_added_domains": dict(dropped_domains_b.most_common()),
    },
    "spatial_verification": {
        "original_core_spatial_retention": orig_rew_spatial / max(1, orig_src_spatial),
        "variant_a_core_spatial_retention": new_rew_spatial / max(1, orig_src_spatial),
        "improvement": (new_rew_spatial - orig_rew_spatial) / max(1, orig_src_spatial),
    },
}

with open(OUT / "spatial_repair_analysis.json", "w") as f:
    json.dump(analysis, f, indent=2)

print(f"\nAnalysis saved to {OUT / 'spatial_repair_analysis.json'}")
print(json.dumps(analysis, indent=2))
