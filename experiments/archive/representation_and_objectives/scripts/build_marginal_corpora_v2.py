#!/usr/bin/env python3
"""research v2: Build C/S/G marginal corpora with content-balanced Arm G.

Improvement over v1: Arm G matches Arm S's content/function word ratio per row,
addressing the BPE/word and content-fraction confounds identified in v1 audit.

Arms:
  C = compact rewrite marginal (abstractive, source-absent words possible)
  S = rewrite-guided order-preserving source extractive (content-matched to rewrite)
  G = content-balanced generic order-preserving source extractive (NOT rewrite-guided,
      but matched to S's content/function counts from the FULL source)

All arms have identical whitespace word counts per row.
"""
import json, re, hashlib, pathlib, statistics, collections, time, random as _random

PAIR_PATH = "experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl"
OUT_DIR   = pathlib.Path("experiments/archive/representation_and_objectives/data/marginal_corpora_v2")

# ---------- helpers ----------
FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
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
    return re.sub(r'[^a-z0-9]', '', w.lower())

def is_content(w):
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1

def evenly_sample(positions, k, rng):
    """Deterministically sample k positions from the given list, evenly spaced."""
    n = len(positions)
    if k <= 0:
        return []
    if k >= n:
        return list(positions)
    # Even spacing with small jitter
    spacing = n / k
    selected = []
    for i in range(k):
        base_idx = int(i * spacing + spacing / 2)
        # Small jitter: ±1 in index space
        jitter = rng.randint(-1, 1)
        idx = max(0, min(n - 1, base_idx + jitter))
        selected.append(positions[idx])
    # Deduplicate (keep order, then fill gaps)
    seen = set()
    deduped = []
    for p in selected:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    # Fill any gaps from unused positions
    if len(deduped) < k:
        unused = [p for p in positions if p not in seen]
        rng.shuffle(unused)
        for p in unused:
            if len(deduped) >= k:
                break
            deduped.append(p)
    return sorted(deduped[:k])


def build_arm_s(source_words, rewrite_words):
    """Rewrite-guided order-preserving source extractive compression."""
    target_len = len(rewrite_words)
    if target_len == 0 or len(source_words) == 0:
        return [], []

    # Match source words to rewrite words (case/punct insensitive, respect multiplicity)
    rw_budget = collections.Counter(norm(w) for w in rewrite_words if norm(w))
    matched_positions = []
    for j, sw in enumerate(source_words):
        sn = norm(sw)
        if sn and sn in rw_budget and rw_budget[sn] > 0:
            matched_positions.append(j)
            rw_budget[sn] -= 1

    if len(matched_positions) >= target_len:
        # More matches than needed: keep content words first, then function, in order
        content_pos = [j for j in matched_positions if is_content(source_words[j])]
        func_pos    = [j for j in matched_positions if not is_content(source_words[j])]
        if len(content_pos) >= target_len:
            selected = sorted(content_pos[:target_len])
        else:
            needed = target_len - len(content_pos)
            selected = sorted(content_pos + func_pos[:needed])
    else:
        # Gap-fill: add source words near matched positions
        selected_set = set(matched_positions)
        needed = target_len - len(selected_set)
        unmatched = [j for j in range(len(source_words)) if j not in selected_set]
        if matched_positions:
            unmatched.sort(key=lambda j: min(abs(j - m) for m in matched_positions))
        for j in unmatched:
            if needed <= 0:
                break
            selected_set.add(j)
            needed -= 1
        selected = sorted(selected_set)

    selected = selected[:target_len]
    rw_norms_set = set(norm(w) for w in rewrite_words if norm(w))
    labels = ["matched" if norm(source_words[j]) in rw_norms_set else "filled"
              for j in selected]
    return selected, labels


def build_arm_g_balanced(source_words, s_positions, pair_idx):
    """Content-balanced generic extractive. Matches S's content/function count
    from the FULL source, not guided by rewrite content, deterministic per pair."""
    target_len = len(s_positions)
    n_src = len(source_words)
    if target_len == 0 or n_src == 0:
        return [], {"method": "empty"}

    # Content/function counts in S
    s_nc = sum(1 for j in s_positions if is_content(source_words[j]))
    s_nf = target_len - s_nc

    # Partition ALL source positions by content/function
    all_content_pos = [j for j in range(n_src) if is_content(source_words[j])]
    all_func_pos    = [j for j in range(n_src) if not is_content(source_words[j])]

    rng = _random.Random(int(hashlib.md5(f"arm_g_bal_v2_{pair_idx}".encode()).hexdigest()[:8], 16))

    # Try to select s_nc content and s_nf function words from source
    nc_available = len(all_content_pos)
    nf_available = len(all_func_pos)

    nc_sel = min(s_nc, nc_available)
    nf_sel = min(s_nf, nf_available)

    # If one pool is short, compensate from the other
    shortage_c = s_nc - nc_sel
    shortage_f = s_nf - nf_sel
    nc_sel += min(shortage_f, max(0, nc_available - nc_sel))
    nf_sel += min(shortage_c, max(0, nf_available - nf_sel))

    # Evenly sample from each pool
    sel_content = evenly_sample(all_content_pos, nc_sel, rng)
    sel_func    = evenly_sample(all_func_pos, nf_sel, rng)

    selected = sorted(sel_content + sel_func)[:target_len]

    # If still short (shouldn't happen normally), fill from any remaining
    if len(selected) < target_len:
        used = set(selected)
        remaining = [j for j in range(n_src) if j not in used]
        rng.shuffle(remaining)
        for j in remaining:
            if len(selected) >= target_len:
                break
            selected.append(j)
        selected = sorted(selected)[:target_len]

    meta = {
        "method": "content_balanced",
        "s_nc": s_nc, "s_nf": s_nf,
        "g_nc": sum(1 for j in selected if is_content(source_words[j])),
        "g_nf": sum(1 for j in selected if not is_content(source_words[j])),
        "shortage_c": shortage_c, "shortage_f": shortage_f,
    }
    return selected, meta


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

    from transformers import AutoTokenizer
    tok_path = "experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M"
    tok = AutoTokenizer.from_pretrained(tok_path)

    print("Loading pairs...", flush=True)
    pairs = []
    with open(PAIR_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"  {len(pairs)} pairs", flush=True)

    rows_out = []
    arm_stats = {arm: {
        "side_words": [], "side_bpe": [], "bpe_per_word": [],
        "compact_overlap": [], "source_overlap": [],
        "content_frac": [], "abs_content_mass": [],
        "position_mean": [], "position_spread": [],
        "span_count": [],
    } for arm in ["C", "S", "G"]}
    arm_stats["S"]["matched_frac"] = []
    arm_stats["S"]["gap_frac"] = []
    g_balance_meta = {"shortage_c": [], "shortage_f": [],
                      "nc_match": 0, "nf_match": 0, "exact_match": 0}

    for i, p in enumerate(pairs):
        src_text = p["source_text"]
        rw_text  = p["rewrite_text"]
        src_w = src_text.split()
        rw_w  = rw_text.split()
        target_len = len(rw_w)

        # Arm C
        c_text = rw_text
        c_words = rw_w

        # Arm S
        s_positions, s_labels = build_arm_s(src_w, rw_w)
        s_words = [src_w[j] for j in s_positions]
        s_text = " ".join(s_words)

        # Arm G (content-balanced)
        g_positions, g_meta = build_arm_g_balanced(src_w, s_positions, i)
        g_words = [src_w[j] for j in g_positions]
        g_text = " ".join(g_words)

        row = {
            "pair_id": p["pair_id"],
            "source_text": src_text,
            "source_words": len(src_w),
            "rewrite_words": len(rw_w),
            "C_text": c_text, "C_words": len(c_words),
            "S_text": s_text, "S_words": len(s_words),
            "S_positions": s_positions, "S_labels": s_labels,
            "G_text": g_text, "G_words": len(g_words),
            "G_positions": g_positions,
        }
        rows_out.append(row)

        # G balance tracking
        g_balance_meta["shortage_c"].append(g_meta.get("shortage_c", 0))
        g_balance_meta["shortage_f"].append(g_meta.get("shortage_f", 0))
        if g_meta.get("g_nc") == g_meta.get("s_nc"):
            g_balance_meta["nc_match"] += 1
        if g_meta.get("g_nf") == g_meta.get("s_nf"):
            g_balance_meta["nf_match"] += 1
        if g_meta.get("g_nc") == g_meta.get("s_nc") and g_meta.get("g_nf") == g_meta.get("s_nf"):
            g_balance_meta["exact_match"] += 1

        # --- per-row audit ---
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

            side_norms = set(norm(w) for w in arm_wlist if norm(w))
            a["compact_overlap"].append(len(rw_norms & side_norms) / len(rw_norms) if rw_norms else 0)
            a["source_overlap"].append(len(src_norms & side_norms) / len(src_norms) if src_norms else 0)
            a["content_frac"].append(sum(1 for w in arm_wlist if is_content(w)) / side_w if side_w else 0)

            if arm_name == "C":
                abs_m = sum(1 for w in arm_wlist if norm(w) and norm(w) not in src_norms)
                a["abs_content_mass"].append(abs_m / side_w if side_w else 0)
            else:
                a["abs_content_mass"].append(0.0)

            if arm_positions and len(arm_positions) > 0 and len(src_w) > 1:
                normed_pos = [j / (len(src_w) - 1) for j in arm_positions]
                a["position_mean"].append(statistics.mean(normed_pos))
                a["position_spread"].append(max(normed_pos) - min(normed_pos) if len(normed_pos) > 1 else 0)
            elif arm_name == "C":
                a["position_mean"].append(0.5)
                a["position_spread"].append(1.0)
            else:
                a["position_mean"].append(0.5)
                a["position_spread"].append(0.0)

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

        if s_labels:
            m_ct = sum(1 for l in s_labels if l == "matched")
            arm_stats["S"]["matched_frac"].append(m_ct / len(s_labels))
            arm_stats["S"]["gap_frac"].append(1 - m_ct / len(s_labels))
        else:
            arm_stats["S"]["matched_frac"].append(0)
            arm_stats["S"]["gap_frac"].append(1)

    # --- Save ---
    out_jsonl = OUT_DIR / "marginal_corpora_csg_v2.jsonl"
    with open(out_jsonl, "w") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    jsonl_sha = hashlib.sha256(open(out_jsonl, "rb").read()).hexdigest()

    # --- Audit ---
    audit = {
        "status": "MARGINAL_CORPORA_V2_BUILT",
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

    # G balance quality
    audit["g_balance"] = {
        "nc_match_count": g_balance_meta["nc_match"],
        "nf_match_count": g_balance_meta["nf_match"],
        "exact_match_count": g_balance_meta["exact_match"],
        "nc_match_frac": round(g_balance_meta["nc_match"] / len(pairs), 4),
        "exact_match_frac": round(g_balance_meta["exact_match"] / len(pairs), 4),
        "shortage_c": compute_stats(g_balance_meta["shortage_c"]),
        "shortage_f": compute_stats(g_balance_meta["shortage_f"]),
    }

    # Cross-arm matching
    c_bpe = arm_stats["C"]["bpe_per_word"]
    s_bpe = arm_stats["S"]["bpe_per_word"]
    g_bpe = arm_stats["G"]["bpe_per_word"]
    c_cf = arm_stats["C"]["content_frac"]
    s_cf = arm_stats["S"]["content_frac"]
    g_cf = arm_stats["G"]["content_frac"]
    audit["matching"] = {
        "bpe_per_word": {
            "C_vs_S": compute_stats([c - s for c, s in zip(c_bpe, s_bpe)]),
            "C_vs_G": compute_stats([c - g for c, g in zip(c_bpe, g_bpe)]),
            "S_vs_G": compute_stats([s - g for s, g in zip(s_bpe, g_bpe)]),
        },
        "content_frac": {
            "C_vs_S": compute_stats([c - s for c, s in zip(c_cf, s_cf)]),
            "C_vs_G": compute_stats([c - g for c, g in zip(c_cf, g_cf)]),
            "S_vs_G": compute_stats([s - g for s, g in zip(s_cf, g_cf)]),
        },
    }

    audit["elapsed_sec"] = round(time.time() - t0, 1)

    out_audit = OUT_DIR / "marginal_corpora_v2_audit.json"
    with open(out_audit, "w") as f:
        json.dump(audit, f, indent=2)

    # Print key summary
    for arm_name in ["C", "S", "G"]:
        aa = audit[f"arm_{arm_name}"]
        print(f"Arm {arm_name}: words={aa['total_side_words']}  bpe={aa['total_side_bpe']}  "
              f"bpe/w={aa['bpe_per_word']['mean']:.3f}  "
              f"content={aa['content_frac']['mean']:.3f}  "
              f"compact_ol={aa['compact_overlap']['mean']:.3f}  "
              f"spans={aa['span_count']['mean']:.1f}")

    gb = audit["g_balance"]
    print(f"\nG balance: exact_match={gb['exact_match_frac']:.3f}  "
          f"nc_match={gb['nc_match_frac']:.3f}  "
          f"shortage_c_mean={gb['shortage_c']['mean']:.2f}  "
          f"shortage_f_mean={gb['shortage_f']['mean']:.2f}")

    m = audit["matching"]
    print(f"\nS-G matching:")
    print(f"  bpe/w diff: mean={m['bpe_per_word']['S_vs_G']['mean']:.4f}  "
          f"med={m['bpe_per_word']['S_vs_G']['median']:.4f}")
    print(f"  content diff: mean={m['content_frac']['S_vs_G']['mean']:.4f}  "
          f"med={m['content_frac']['S_vs_G']['median']:.4f}")

    print(f"\nSHA: {jsonl_sha}")
    print(f"Elapsed: {audit['elapsed_sec']}s")


if __name__ == "__main__":
    main()
