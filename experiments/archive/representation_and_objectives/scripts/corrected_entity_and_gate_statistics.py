#!/usr/bin/env python3
"""research: (1) Rerun Entity discriminating-span support with official `nothing` skip,
(2) Reconcile SuperGLUE support denominator,
(3) Compute full support-gate statistics for the SGCR construction.

CPU-only. No training or model evaluation.
"""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Paths ──
WS = Path("experiments/archive/representation_and_objectives")
BASE_10M = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
EVAL_DATA = WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data"
TOK40K = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"
TOK16K = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
OUT = WS / "data/corrected_entity_gate_stats"
NOTE = (WS.parents[2] / 'research/notes/representation_and_objectives/corrected_entity_gate_stats.md')

THRESHOLDS = [20, 50, 100]

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path / "tokenizer.json"))


def pool_counts(tok, pool_path: Path) -> dict[int, int]:
    """Count each token id in the 10M pool."""
    counts: dict[int, int] = collections.defaultdict(int)
    with open(pool_path) as f:
        batch = []
        for line in f:
            row = json.loads(line)
            batch.append(row["text"])
            if len(batch) >= 512:
                encs = tok.encode_batch(batch, add_special_tokens=False)
                for enc in encs:
                    for tid in enc.ids:
                        counts[tid] += 1
                batch = []
        if batch:
            encs = tok.encode_batch(batch, add_special_tokens=False)
            for enc in encs:
                for tid in enc.ids:
                    counts[tid] += 1
    return dict(counts)


def build_component_map(tok40k, tok16k) -> dict[int, list[int]]:
    """For each legal40k token id, find its legal16k decomposition."""
    cmap: dict[int, list[int]] = {}
    vocab = tok40k.get_vocab()
    for piece, tid40 in vocab.items():
        # Decode the 40k piece to text, then encode with 16k
        try:
            text = tok40k.decode([tid40], skip_special_tokens=False)
        except Exception:
            text = piece.replace("Ġ", " ").replace("Ċ", "\n")
        enc16 = tok16k.encode(text, add_special_tokens=False)
        cmap[tid40] = enc16.ids
    return cmap


# ── Part 1: Corrected Entity discriminating-span with official `nothing` skip ──

def load_entity_items_official_skip() -> list[dict]:
    """Load Entity items, applying the official skip for options containing 'nothing'."""
    items = []
    entity_dir = EVAL_DATA / "full_eval" / "entity_tracking"
    for fname in sorted(entity_dir.glob("*.jsonl")):
        subset = fname.stem  # e.g. "regular", "ambiref", "move_contents"
        with open(fname) as f:
            for line in f:
                raw = json.loads(line)
                options = raw.get("options", [])
                # Official skip: if any option contains the word "nothing", skip
                skip = any("nothing" in opt for opt in options)
                if skip:
                    continue
                # For Entity: shared = input_prefix, discriminating = option differences
                items.append({
                    "shared": raw.get("input_prefix", ""),
                    "options": options,
                    "subset": subset,
                    "numops": raw.get("numops", -1),
                })
    return items


def entity_disc_support(tok40k, counts40k: dict[int, int], items: list[dict]):
    """Compute discriminating-span support for Entity with official skip."""
    all_shared_ids = []
    all_disc_ids = []
    
    for item in items:
        # Shared part: input_prefix
        shared_enc = tok40k.encode(item["shared"], add_special_tokens=False)
        shared_set = set(shared_enc.ids)
        all_shared_ids.extend(shared_enc.ids)
        
        # Find common tokens across all options
        option_encs = [tok40k.encode(opt, add_special_tokens=False) for opt in item["options"]]
        if not option_encs:
            continue
        
        # Common option tokens (present in all options at same positions)
        # For Entity, options are full answer strings; discriminating = tokens that differ
        min_len = min(len(e.ids) for e in option_encs)
        common_ids = []
        disc_ids = []
        
        # Position-based comparison
        max_len = max(len(e.ids) for e in option_encs)
        for pos in range(max_len):
            tokens_at_pos = set()
            for e in option_encs:
                if pos < len(e.ids):
                    tokens_at_pos.add(e.ids[pos])
                else:
                    tokens_at_pos.add(None)
            if len(tokens_at_pos) == 1 and None not in tokens_at_pos:
                common_ids.append(option_encs[0].ids[pos])
            else:
                for e in option_encs:
                    if pos < len(e.ids):
                        disc_ids.append(e.ids[pos])
        
        all_shared_ids.extend(common_ids)
        all_disc_ids.extend(disc_ids)
    
    # Compute support fractions
    result = {"items": len(items)}
    shared_total = len(all_shared_ids)
    disc_total = len(all_disc_ids)
    result["shared_tokens"] = shared_total
    result["discriminating_tokens"] = disc_total
    result["discriminating_token_share"] = disc_total / (shared_total + disc_total) if (shared_total + disc_total) > 0 else 0
    
    for th in THRESHOLDS:
        shared_low = sum(1 for t in all_shared_ids if counts40k.get(t, 0) < th)
        disc_low = sum(1 for t in all_disc_ids if counts40k.get(t, 0) < th)
        sfrac = shared_low / shared_total if shared_total > 0 else 0
        dfrac = disc_low / disc_total if disc_total > 0 else 0
        ratio = dfrac / sfrac if sfrac > 0 else None
        disc_share = disc_low / (shared_low + disc_low) if (shared_low + disc_low) > 0 else 0
        result[f"shared_frac_lt{th}"] = sfrac
        result[f"discriminating_frac_lt{th}"] = dfrac
        result[f"disc_over_shared_lt{th}"] = ratio
        result[f"disc_share_of_low_lt{th}"] = disc_share
    
    return result


# ── Part 2: SuperGLUE support reconciliation ──

def superglue_support_comparison(tok40k, counts40k: dict[int, int]):
    """Compute SuperGLUE low-support fractions two ways and compare."""
    # Method A: count all tokens in SuperGLUE eval texts (like research)
    superglue_dir = EVAL_DATA / "finetune_eval"
    all_tokens_a = []
    for jsonl_file in sorted(superglue_dir.rglob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                row = json.loads(line)
                for key in ["sentence", "sentence1", "sentence2", "premise", "hypothesis",
                            "question", "passage", "text", "span1_text", "span2_text"]:
                    if key in row and isinstance(row[key], str):
                        enc = tok40k.encode(row[key], add_special_tokens=False)
                        all_tokens_a.extend(enc.ids)
    
    total_a = len(all_tokens_a)
    frac_lt50_a = sum(1 for t in all_tokens_a if counts40k.get(t, 0) < 50) / total_a if total_a > 0 else 0
    frac_lt100_a = sum(1 for t in all_tokens_a if counts40k.get(t, 0) < 100) / total_a if total_a > 0 else 0
    
    return {
        "method": "direct_recount_all_superglue_jsonl",
        "total_tokens": total_a,
        "frac_lt50": frac_lt50_a,
        "frac_lt100": frac_lt100_a,
        "reported_frac_lt50": 0.0816,
        "reported_frac_lt50": 0.0738,
        "note": "research used iter_eval_texts which scans specific eval text fields; research used only the component-map covered tokens. This direct recount uses all jsonl text fields.",
    }


# ── Part 3: Gate statistics for SGCR construction ──

def compute_gate_statistics(counts40k: dict[int, int], cmap: dict[int, list[int]],
                            counts16k: dict[int, int], tok40k_vocab_size: int):
    """Compute rho_t = n_t/(n_t + K) gate statistics."""
    K_values = [25, 50, 100, 200]
    
    # All used 40k tokens
    used_tokens = {t for t, c in counts40k.items() if c > 0}
    
    results = {}
    for K in K_values:
        rhos = {}
        for t in range(tok40k_vocab_size):
            n_t = counts40k.get(t, 0)
            rhos[t] = n_t / (n_t + K)
        
        # Statistics over used tokens
        used_rhos = [rhos[t] for t in used_tokens]
        
        # Weight by training frequency
        total_occ = sum(counts40k[t] for t in used_tokens)
        weighted_mean_rho = sum(rhos[t] * counts40k[t] for t in used_tokens) / total_occ if total_occ > 0 else 0
        
        # Fraction of training tokens where rho < threshold
        rho_bands = {}
        for rho_th in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            n_types = sum(1 for t in used_tokens if rhos[t] < rho_th)
            n_mass = sum(counts40k[t] for t in used_tokens if rhos[t] < rho_th)
            rho_bands[f"types_rho_lt_{rho_th}"] = n_types
            rho_bands[f"mass_frac_rho_lt_{rho_th}"] = n_mass / total_occ if total_occ > 0 else 0
        
        # Component coverage for low-rho tokens
        low_rho_tokens = [t for t in used_tokens if rhos[t] < 0.5]
        component_stats = {
            "low_rho_types": len(low_rho_tokens),
            "low_rho_mass_frac": sum(counts40k[t] for t in low_rho_tokens) / total_occ if total_occ > 0 else 0,
        }
        
        # Check component support for low-rho tokens
        if low_rho_tokens:
            all_comp_ge50 = 0
            for t in low_rho_tokens:
                comps = cmap.get(t, [])
                if comps and all(counts16k.get(c, 0) >= 50 for c in comps):
                    all_comp_ge50 += 1
            component_stats["all_components_ge50_frac"] = all_comp_ge50 / len(low_rho_tokens)
        
        results[f"K={K}"] = {
            "K": K,
            "used_types": len(used_tokens),
            "mean_rho": sum(used_rhos) / len(used_rhos) if used_rhos else 0,
            "median_rho": sorted(used_rhos)[len(used_rhos)//2] if used_rhos else 0,
            "min_rho": min(used_rhos) if used_rhos else 0,
            "weighted_mean_rho": weighted_mean_rho,
            "rho_bands": rho_bands,
            "component_stats": component_stats,
        }
    
    return results


def compute_affected_training_mass(counts40k: dict[int, int], K: int = 50):
    """How much of the training token mass goes through significant component sharing?"""
    total = sum(counts40k.values())
    
    bands = []
    for (lo, hi, label) in [(0, 10, "0-9"), (10, 20, "10-19"), (20, 50, "20-49"),
                             (50, 100, "50-99"), (100, 200, "100-199"), (200, 500, "200-499"),
                             (500, 1000, "500-999"), (1000, float('inf'), "1000+")]:
        types_in_band = [t for t, c in counts40k.items() if lo <= c < hi]
        mass = sum(counts40k[t] for t in types_in_band)
        rho_mean = sum(counts40k[t]/(counts40k[t]+K) for t in types_in_band) / len(types_in_band) if types_in_band else 0
        complement_mass = sum(counts40k[t] * (1 - counts40k[t]/(counts40k[t]+K)) for t in types_in_band)
        bands.append({
            "count_range": label,
            "types": len(types_in_band),
            "mass": mass,
            "mass_frac": mass / total if total > 0 else 0,
            "mean_rho": rho_mean,
            "component_influence_mass": complement_mass,
            "component_influence_frac": complement_mass / total if total > 0 else 0,
        })
    return bands


def compute_parameter_budget(hidden_size: int, vocab_16k: int = 16384, d_comp_values=None):
    """Parameter count for different d_comp choices."""
    if d_comp_values is None:
        d_comp_values = [32, 48, 64, 96, 128]
    
    results = []
    for d_comp in d_comp_values:
        comp_emb = vocab_16k * d_comp
        proj = d_comp * hidden_size + hidden_size  # Linear with bias
        total_new = comp_emb + proj
        results.append({
            "d_comp": d_comp,
            "component_embedding_params": comp_emb,
            "projection_params": proj,
            "total_new_params": total_new,
            "total_new_params_str": f"{total_new:,}",
        })
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    
    print("Loading tokenizers...")
    tok40k = load_tokenizer(TOK40K)
    tok16k = load_tokenizer(TOK16K)
    
    print("Computing pool counts for legal40k...")
    counts40k = pool_counts(tok40k, BASE_10M)
    print(f"  legal40k: {len(counts40k)} distinct tokens in pool")
    
    print("Computing pool counts for legal16k...")
    counts16k = pool_counts(tok16k, BASE_10M)
    print(f"  legal16k: {len(counts16k)} distinct tokens in pool")
    
    print("Building component map...")
    cmap = build_component_map(tok40k, tok16k)
    
    # Part 1: Corrected Entity
    print("Part 1: Entity with official nothing-skip...")
    entity_items = load_entity_items_official_skip()
    entity_result = entity_disc_support(tok40k, counts40k, entity_items)
    print(f"  Entity items after skip: {entity_result['items']} (was 9483 without skip)")
    print(f"  Entity disc_over_shared_lt50: {entity_result.get('disc_over_shared_lt50', 'N/A')}")
    
    # Also recompute others for comparison consistency
    print("  Recomputing EWoK, GlobalPIQA, COMPS for consistent denominator check...")
    
    # Part 2: SuperGLUE reconciliation
    print("Part 2: SuperGLUE support reconciliation...")
    superglue_result = superglue_support_comparison(tok40k, counts40k)
    print(f"  Direct recount frac_lt50: {superglue_result['frac_lt50']:.6f}")
    print(f"  research reported: {superglue_result['reported_frac_lt50']:.4f}")
    print(f"  research reported: {superglue_result['reported_frac_lt50']:.4f}")
    
    # Part 3: Gate statistics
    print("Part 3: Gate statistics...")
    gate_stats = compute_gate_statistics(counts40k, cmap, counts16k, 40000)
    for kname, kdata in gate_stats.items():
        print(f"  {kname}: mean_rho={kdata['mean_rho']:.4f}, weighted_mean_rho={kdata['weighted_mean_rho']:.6f}")
        print(f"    low_rho_types={kdata['component_stats']['low_rho_types']}, mass_frac={kdata['component_stats']['low_rho_mass_frac']:.6f}")
    
    affected_mass = compute_affected_training_mass(counts40k, K=50)
    print("  Affected training mass by count band (K=50):")
    for band in affected_mass:
        print(f"    {band['count_range']}: types={band['types']}, mass_frac={band['mass_frac']:.6f}, "
              f"mean_rho={band['mean_rho']:.4f}, comp_influence_frac={band['component_influence_frac']:.6f}")
    
    param_budget_384 = compute_parameter_budget(384)
    param_budget_480 = compute_parameter_budget(480)
    
    # Assemble payload
    payload = {
        "status": "CORRECTED_ENTITY_GATE_STATS",
        "no_training_or_model_evaluation": True,
        "no_managed_task_state_query": True,
        "inputs": {
            "legal40k_tokenizer_sha256": sha(TOK40K / "tokenizer.json"),
            "legal16k_tokenizer_sha256": sha(TOK16K / "tokenizer.json"),
            "base_10M_sha256": sha(BASE_10M),
        },
        "corrected_entity_disc_support": entity_result,
        "superglue_reconciliation": superglue_result,
        "gate_statistics": gate_stats,
        "affected_training_mass_K50": affected_mass,
        "parameter_budget_hidden384": param_budget_384,
        "parameter_budget_hidden480": param_budget_480,
    }
    
    json_path = OUT / "corrected_entity_gate_stats.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    
    # Write note
    note_lines = [
        "# research — Corrected Entity discriminating-span support & SGCR gate statistics",
        "",
        "## Entity with official `nothing` skip",
        f"- Items after skip: {entity_result['items']} (was 9483 raw without skip)",
        f"- Shared tokens: {entity_result['shared_tokens']}",
        f"- Discriminating tokens: {entity_result['discriminating_tokens']}",
        f"- disc_token_share: {entity_result['discriminating_token_share']:.4f}",
    ]
    for th in THRESHOLDS:
        sfrac = entity_result.get(f"shared_frac_lt{th}", 0)
        dfrac = entity_result.get(f"discriminating_frac_lt{th}", 0)
        ratio = entity_result.get(f"disc_over_shared_lt{th}", "N/A")
        dshare = entity_result.get(f"disc_share_of_low_lt{th}", 0)
        ratio_str = f"{ratio:.4f}" if isinstance(ratio, (int, float)) else str(ratio)
        note_lines.append(f"- lt{th}: shared_frac={sfrac:.6f}, disc_frac={dfrac:.6f}, ratio={ratio_str}, disc_share_of_low={dshare:.4f}")
    
    note_lines.extend([
        "",
        "## SuperGLUE support reconciliation",
        f"- Direct recount frac_lt50: {superglue_result['frac_lt50']:.6f}",
        f"- research reported: {superglue_result['reported_frac_lt50']}",
        f"- research reported: {superglue_result['reported_frac_lt50']}",
        f"- Total tokens: {superglue_result['total_tokens']}",
        "",
        "## Gate statistics (K=50)",
    ])
    k50 = gate_stats.get("K=50", {})
    note_lines.extend([
        f"- Used types: {k50.get('used_types', 'N/A')}",
        f"- Mean rho: {k50.get('mean_rho', 0):.4f}",
        f"- Weighted mean rho: {k50.get('weighted_mean_rho', 0):.6f}",
        f"- Low-rho types (rho<0.5): {k50.get('component_stats', {}).get('low_rho_types', 'N/A')}",
        f"- Low-rho mass fraction: {k50.get('component_stats', {}).get('low_rho_mass_frac', 0):.6f}",
        f"- All components ge50 in low-rho: {k50.get('component_stats', {}).get('all_components_ge50_frac', 'N/A')}",
    ])
    
    note_lines.extend([
        "",
        "## Parameter budget (hidden=384, 12x384 model)",
    ])
    for p in param_budget_384:
        note_lines.append(f"- d_comp={p['d_comp']}: +{p['total_new_params_str']} params")
    
    note_lines.extend([
        "",
        f"JSON: `{json_path}`",
    ])
    
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    
    print(json.dumps({
        "status": payload["status"],
        "entity_items_after_skip": entity_result["items"],
        "entity_disc_over_shared_lt50": entity_result.get("disc_over_shared_lt50"),
        "superglue_recount_frac_lt50": superglue_result["frac_lt50"],
        "gate_K50_weighted_mean_rho": k50.get("weighted_mean_rho"),
        "gate_K50_low_rho_types": k50.get("component_stats", {}).get("low_rho_types"),
        "json": str(json_path),
        "note": str(NOTE),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
