#!/usr/bin/env python3
"""research: Build C/S/G marginal corpora from the 12,155 compact pairs.

Arm C: compact rewrite marginal (the rewrite text itself)
Arm S: rewrite-guided order-preserving source extractive compression
Arm G: generic order-preserving source extractive (not rewrite-guided)

All arms produce exactly one side-text per pair, with the same source.
Output: one JSONL with all three side views per row, plus audit JSON.
"""
import json, re, hashlib, pathlib, statistics, collections, time

PAIR_PATH = "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
OUT_DIR   = pathlib.Path("experiments/archive/representation_and_objectives/data/marginal_corpora")

# ---------- helpers ----------
FUNC_WORDS = set("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())

def norm(w):
    """Lowercase alpha-numeric core of a word for matching."""
    return re.sub(r'[^a-z0-9]', '', w.lower())

def is_content(w):
    return norm(w) not in FUNC_WORDS and len(norm(w)) > 1

def build_arm_s(source_words, rewrite_words):
    """Rewrite-guided order-preserving source extractive compression."""
    target_len = len(rewrite_words)
    if target_len == 0 or len(source_words) == 0:
        return [], []

    # Find source positions matching rewrite words (case/punct insensitive)
    rw_norms = collections.Counter(norm(w) for w in rewrite_words if norm(w))
    
    # Greedy matching: assign source positions to rewrite words
    # Allow repeated matches for multiply-occurring rewrite words
    remaining_rw = dict(rw_norms)  # available rewrite-word budget
    matched_positions = []
    for j, sw in enumerate(source_words):
        sn = norm(sw)
        if sn and sn in remaining_rw and remaining_rw[sn] > 0:
            matched_positions.append(j)
            remaining_rw[sn] -= 1

    if len(matched_positions) >= target_len:
        # More matches than needed: prioritize content words, then by position
        content_pos = [j for j in matched_positions if is_content(source_words[j])]
        func_pos    = [j for j in matched_positions if not is_content(source_words[j])]
        # Take content first, then fill with function, all in source order
        if len(content_pos) >= target_len:
            selected = sorted(content_pos[:target_len])
        else:
            needed_func = target_len - len(content_pos)
            selected = sorted(content_pos + func_pos[:needed_func])
    else:
        # Need gap-filling: add source words adjacent to matched positions
        selected_set = set(matched_positions)
        needed = target_len - len(selected_set)
        
        # Score unmatched positions by distance to nearest matched position
        unmatched = [j for j in range(len(source_words)) if j not in selected_set]
        if matched_positions:
            # Prefer words between/adjacent to matches for fluency
            def dist_score(j):
                return min(abs(j - m) for m in matched_positions)
            unmatched.sort(key=dist_score)
        
        for j in unmatched:
            if needed <= 0:
                break
            selected_set.add(j)
            needed -= 1
        selected = sorted(selected_set)

    selected = selected[:target_len]  # safety trim
    
    # Track which selected positions are content-matched vs gap-filled
    rw_norms_set = set(norm(w) for w in rewrite_words if norm(w))
    labels = []
    for j in selected:
        if norm(source_words[j]) in rw_norms_set:
            labels.append("matched")
        else:
            labels.append("filled")
    
    return selected, labels

def build_arm_g(source_words, target_len, pair_idx):
    """Generic order-preserving source extractive (not rewrite-guided).
    
    Evenly space positions through source with deterministic per-pair jitter.
    """
    n = len(source_words)
    if target_len == 0 or n == 0:
        return []
    
    if target_len >= n:
        return list(range(n))
    
    # Deterministic seed from pair index
    seed = int(hashlib.md5(f"bytes((97, 114, 109, 95, 103, 95, 115, 116, 101, 112, 50, 50, 51, 95)).decode('utf-8'){pair_idx}".encode()).hexdigest()[:8], 16)
    
    # Evenly space target_len positions with jitter
    # base_spacing covers [0, n-1] with target_len points
    spacing = (n - 1) / max(target_len - 1, 1) if target_len > 1 else 0
    positions = []
    for k in range(target_len):
        base = k * spacing if target_len > 1 else n // 2
        # Small deterministic jitter: ±1 word
        jitter_hash = (seed + k * 7919) % 3 - 1  # -1, 0, or 1
        pos = max(0, min(n - 1, round(base) + jitter_hash))
        positions.append(pos)
    
    # Deduplicate while maintaining order and count
    positions = sorted(set(positions))
    
    # If deduplication reduced count, greedily fill gaps
    if len(positions) < target_len:
        used = set(positions)
        available = [j for j in range(n) if j not in used]
        # Fill uniformly: pick from middle of largest gaps
        needed = target_len - len(positions)
        for j in available:
            if needed <= 0:
                break
            positions.append(j)
            needed -= 1
        positions = sorted(positions)
    
    return positions[:target_len]


def compute_stats(values):
    if not values:
        return {"n": 0}
    s = sorted(values)
    n = len(s)
    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p10": round(s[n // 10], 6),
        "p90": round(s[9 * n // 10], 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer for BPE audit
    from transformers import AutoTokenizer
    tok_path = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
    tok = AutoTokenizer.from_pretrained(tok_path)
    
    # Build token support map from training corpus vocabulary
    # Use token id frequency as proxy: count from tokenizer vocab
    # (Real support would need corpus scan; use tokenizer-internal frequency)
    
    print("Loading pairs...", flush=True)
    pairs = []
    with open(PAIR_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"  {len(pairs)} pairs", flush=True)

    # --- Build all three arms ---
    rows_out = []
    
    # Per-arm accumulators for audit
    arm_stats = {arm: {
        "side_words": [], "side_bpe": [], "bpe_per_word": [],
        "matched_frac": [],  # S only: fraction of selected words that match rewrite
        "gap_frac": [],      # S only: fraction that are gap-fills
        "compact_overlap": [],  # fraction of compact rewrite words present in side
        "source_overlap": [],   # fraction of source words present in side
        "position_mean": [], "position_spread": [],
        "content_frac": [],  # fraction of side words that are content words
        "abs_content_mass": [],  # words not from source (for C only)
        "span_count": [],    # number of contiguous spans
    } for arm in ["C", "S", "G"]}
    
    for i, p in enumerate(pairs):
        src_text = p["source_text"]
        rw_text  = p["rewrite_text"]
        src_w = src_text.split()
        rw_w  = rw_text.split()
        target_len = len(rw_w)
        
        # Arm C: compact rewrite itself
        c_text = rw_text
        c_words = rw_w
        
        # Arm S: rewrite-guided extractive
        s_positions, s_labels = build_arm_s(src_w, rw_w)
        s_words = [src_w[j] for j in s_positions]
        s_text = " ".join(s_words)
        
        # Arm G: generic extractive
        g_positions = build_arm_g(src_w, target_len, i)
        g_words = [src_w[j] for j in g_positions]
        g_text = " ".join(g_words)
        
        row = {
            "pair_id": p["pair_id"],
            "source_text": src_text,
            "source_words": len(src_w),
            "rewrite_words": len(rw_w),
            "C_text": c_text,
            "C_words": len(c_words),
            "S_text": s_text,
            "S_words": len(s_words),
            "S_positions": s_positions,
            "S_labels": s_labels,
            "G_text": g_text,
            "G_words": len(g_words),
            "G_positions": g_positions,
        }
        rows_out.append(row)
        
        # --- Compute per-row audit stats ---
        rw_norms = set(norm(w) for w in rw_w if norm(w))
        src_norms = set(norm(w) for w in src_w if norm(w))
        
        for arm_name, arm_text, arm_wlist, arm_positions in [
            ("C", c_text, c_words, None),
            ("S", s_text, s_words, s_positions),
            ("G", g_text, g_words, g_positions),
        ]:
            a = arm_stats[arm_name]
            side_w = len(arm_wlist)
            bpe_ids = tok.encode(arm_text, add_special_tokens=False) if arm_text else []
            side_bpe = len(bpe_ids)
            
            a["side_words"].append(side_w)
            a["side_bpe"].append(side_bpe)
            a["bpe_per_word"].append(side_bpe / side_w if side_w else 0)
            
            # Compact overlap: fraction of compact rewrite words in this side
            side_norms = set(norm(w) for w in arm_wlist if norm(w))
            compact_ol = len(rw_norms & side_norms) / len(rw_norms) if rw_norms else 0
            a["compact_overlap"].append(compact_ol)
            
            # Source overlap: fraction of source words present in side
            source_ol = len(src_norms & side_norms) / len(src_norms) if src_norms else 0
            a["source_overlap"].append(source_ol)
            
            # Content word fraction
            cf = sum(1 for w in arm_wlist if is_content(w)) / side_w if side_w else 0
            a["content_frac"].append(cf)
            
            # Source-absent mass (for C: words not in source; for S/G: always 0 since from source)
            if arm_name == "C":
                abs_mass = sum(1 for w in arm_wlist if norm(w) and norm(w) not in src_norms)
                a["abs_content_mass"].append(abs_mass / side_w if side_w else 0)
            else:
                a["abs_content_mass"].append(0.0)
            
            # Position statistics
            if arm_positions is not None and len(arm_positions) > 0 and len(src_w) > 1:
                normed_pos = [j / (len(src_w) - 1) for j in arm_positions]
                a["position_mean"].append(statistics.mean(normed_pos))
                a["position_spread"].append(max(normed_pos) - min(normed_pos) if len(normed_pos) > 1 else 0)
            elif arm_name == "C":
                a["position_mean"].append(0.5)  # N/A for C
                a["position_spread"].append(1.0)
            else:
                a["position_mean"].append(0.5)
                a["position_spread"].append(0.0)
            
            # Span count: number of contiguous runs in position list
            if arm_positions and len(arm_positions) > 1:
                spans = 1
                for k in range(1, len(arm_positions)):
                    if arm_positions[k] > arm_positions[k-1] + 1:
                        spans += 1
                a["span_count"].append(spans)
            elif arm_positions:
                a["span_count"].append(1)
            else:
                a["span_count"].append(0)
        
        # S-specific: matched vs gap-fill fractions
        if s_labels:
            matched_ct = sum(1 for l in s_labels if l == "matched")
            arm_stats["S"]["matched_frac"].append(matched_ct / len(s_labels))
            arm_stats["S"]["gap_frac"].append(1 - matched_ct / len(s_labels))
        else:
            arm_stats["S"]["matched_frac"].append(0)
            arm_stats["S"]["gap_frac"].append(1)
    
    # --- Save JSONL ---
    out_jsonl = OUT_DIR / "marginal_corpora_csg.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    jsonl_sha = hashlib.sha256(open(out_jsonl, "rb").read()).hexdigest()
    
    # --- Build audit ---
    audit = {
        "status": "MARGINAL_CORPORA_BUILT",
        "pairs": len(pairs),
        "out_jsonl": str(out_jsonl),
        "out_sha256": jsonl_sha,
    }
    
    for arm_name in ["C", "S", "G"]:
        a = arm_stats[arm_name]
        arm_audit = {
            "total_side_words": sum(a["side_words"]),
            "total_side_bpe": sum(a["side_bpe"]),
            "side_words_per_row": compute_stats(a["side_words"]),
            "bpe_per_word": compute_stats(a["bpe_per_word"]),
            "compact_overlap": compute_stats(a["compact_overlap"]),
            "source_overlap": compute_stats(a["source_overlap"]),
            "content_frac": compute_stats(a["content_frac"]),
            "position_mean": compute_stats(a["position_mean"]),
            "position_spread": compute_stats(a["position_spread"]),
            "span_count": compute_stats(a["span_count"]),
        }
        if arm_name == "C":
            arm_audit["abs_content_mass_frac"] = compute_stats(a["abs_content_mass"])
        if arm_name == "S":
            arm_audit["matched_frac"] = compute_stats(a["matched_frac"])
            arm_audit["gap_frac"] = compute_stats(a["gap_frac"])
        audit[f"arm_{arm_name}"] = arm_audit
    
    # Word length matching
    c_lens = arm_stats["C"]["side_words"]
    s_lens = arm_stats["S"]["side_words"]
    g_lens = arm_stats["G"]["side_words"]
    cs_diffs = [abs(c - s) for c, s in zip(c_lens, s_lens)]
    cg_diffs = [abs(c - g) for c, g in zip(c_lens, g_lens)]
    sg_diffs = [abs(s - g) for s, g in zip(s_lens, g_lens)]
    audit["word_len_matching"] = {
        "C_vs_S_abs_diff": compute_stats(cs_diffs),
        "C_vs_G_abs_diff": compute_stats(cg_diffs),
        "S_vs_G_abs_diff": compute_stats(sg_diffs),
        "C_eq_S": sum(1 for d in cs_diffs if d == 0),
        "C_eq_G": sum(1 for d in cg_diffs if d == 0),
        "S_eq_G": sum(1 for d in sg_diffs if d == 0),
    }
    
    # BPE matching
    c_bpe = arm_stats["C"]["bpe_per_word"]
    s_bpe = arm_stats["S"]["bpe_per_word"]
    g_bpe = arm_stats["G"]["bpe_per_word"]
    audit["bpe_per_word_matching"] = {
        "C_vs_S_diff": compute_stats([c - s for c, s in zip(c_bpe, s_bpe)]),
        "C_vs_G_diff": compute_stats([c - g for c, g in zip(c_bpe, g_bpe)]),
        "S_vs_G_diff": compute_stats([s - g for s, g in zip(s_bpe, g_bpe)]),
    }
    
    audit["elapsed_sec"] = round(time.time() - t0, 1)
    
    out_audit = OUT_DIR / "marginal_corpora_audit.json"
    with open(out_audit, "w") as f:
        json.dump(audit, f, indent=2)
    
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
