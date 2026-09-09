#!/usr/bin/env python3
"""research: Leakage-decomposition operator inventory of compact-view edits.

Loads all source-rewrite compact pairs, computes word-level diffs,
identifies 1:1 substitutions and other edit operations, decomposes by
source-absence, full-word, content-vs-function, stem leakage, and
reports bidirectional density and operator diversity.

Predeclared thresholds in 164_predeclared_thresholds.md.
"""
import json, re, sys, difflib, math
from pathlib import Path
from collections import Counter, defaultdict

PAIR_JSONL = Path("experiments/archive/frontier_consolidation/data"
                  "density_core_reinvestment_medium_riskhard/"
                  "selected_compact_reinvest_pairs.jsonl")
OUT_DIR = Path("experiments/archive/representation_and_objectives/data/edit_operator_inventory")

STOP_WORDS = frozenset({
    'the','a','an','is','are','was','were','be','been','being','have','has',
    'had','do','does','did','will','would','could','should','may','might',
    'shall','can','to','of','in','for','on','with','at','by','from','as',
    'into','through','during','before','after','above','below','between',
    'under','up','down','out','off','over','and','but','or','nor','not','no',
    'so','if','then','than','that','this','these','those','it','its','he',
    'she','his','her','they','them','their','we','our','you','your','i','me',
    'my','who','which','what','where','when','how','all','each','every',
    'both','some','any','few','more','most','other','such','also','just',
    'about','very','only','even','still','too','much','many','there','here',
    'now','because','since','while','although','though','yet','however',
})

PUNCT_STRIP = re.compile(r'^[\W_]+|[\W_]+$')

def clean_word(w):
    """Strip leading/trailing punctuation."""
    return PUNCT_STRIP.sub('', w)

def is_full_word(w):
    c = clean_word(w)
    return len(c) >= 3 and c.isalpha()

def is_content(w):
    return clean_word(w).lower() not in STOP_WORDS

def stem4(w):
    """Crude stem: first 4 lowercase alpha chars, or full word if shorter."""
    c = clean_word(w).lower()
    alpha = re.sub(r'[^a-z]', '', c)
    return alpha[:4] if len(alpha) >= 4 else alpha

def word_in_set(w, word_set_lower):
    return clean_word(w).lower() in word_set_lower

def stem_in_set(w, stem_set):
    s = stem4(w)
    return s in stem_set if len(s) >= 4 else False

def char_edit_distance(a, b):
    """Normalized Levenshtein (0–1)."""
    a, b = a.lower(), b.lower()
    if a == b: return 0.0
    m, n = len(a), len(b)
    if m == 0 or n == 0: return 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j-1] + 1, prev + cost), dp[j]
    return dp[n] / max(m, n)

def get_edit_operations(source_text, rewrite_text):
    """Word-level diff returning typed edit operations."""
    s_words = source_text.split()
    r_words = rewrite_text.split()
    s_clean = [clean_word(w).lower() for w in s_words]
    r_clean = [clean_word(w).lower() for w in r_words]
    sm = difflib.SequenceMatcher(None, s_clean, r_clean, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        s_span = s_words[i1:i2]
        r_span = r_words[j1:j2]
        ns, nr = len(s_span), len(r_span)
        if ns == 0:
            ops.append({'type': 'insert', 'n': nr, 'words': r_span})
        elif nr == 0:
            ops.append({'type': 'delete', 'n': ns, 'words': s_span})
        elif ns == 1 and nr == 1:
            ops.append({'type': '1to1', 'source_word': s_span[0],
                        'rewrite_word': r_span[0]})
        else:
            ops.append({'type': f'{ns}to{nr}', 'source_words': s_span,
                        'rewrite_words': r_span})
    return ops


def analyze_pair(pair):
    src = str(pair.get('source_text') or '')
    rew = str(pair.get('rewrite_text') or '')
    if not src.strip() or not rew.strip():
        return None
    ops = get_edit_operations(src, rew)
    src_lower_set = set(clean_word(w).lower() for w in src.split())
    src_stem_set = set(stem4(w) for w in src.split() if len(stem4(w)) >= 4)
    subs_1to1 = []
    type_counts = Counter()
    for op in ops:
        type_counts[op['type']] += 1
        if op['type'] == '1to1':
            sw = op['source_word']
            rw = op['rewrite_word']
            sw_c = clean_word(sw)
            rw_c = clean_word(rw)
            src_absent = not word_in_set(rw, src_lower_set)
            stem_leak = stem_in_set(rw, src_stem_set)
            s_fw = is_full_word(sw)
            r_fw = is_full_word(rw)
            s_cont = is_content(sw)
            r_cont = is_content(rw)
            ced = char_edit_distance(sw_c, rw_c)
            subs_1to1.append({
                'source_word': sw_c.lower(),
                'rewrite_word': rw_c.lower(),
                'source_absent': src_absent,
                'stem_leakage': stem_leak,
                'source_full_word': s_fw,
                'rewrite_full_word': r_fw,
                'source_content': s_cont,
                'rewrite_content': r_cont,
                'both_full_content': (s_fw and r_fw and s_cont and r_cont),
                'char_edit_dist': ced,
                'same_stem4': (stem4(sw) == stem4(rw) and len(stem4(sw)) >= 4),
            })
    return {'pair_id': pair.get('pair_id', ''),
            'type_counts': type_counts,
            'subs_1to1': subs_1to1,
            'n_source_words': len(src.split()),
            'n_rewrite_words': len(rew.split())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert PAIR_JSONL.exists(), f"Pair file not found: {PAIR_JSONL}"

    # ---- Load pairs ----
    pairs = []
    with PAIR_JSONL.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} pairs", flush=True)

    # ---- Analyze all pairs ----
    all_subs = []           # flat list of 1:1 subs
    type_totals = Counter() # edit-type distribution
    pairs_with_subs = 0
    total_source_words = 0
    total_rewrite_words = 0
    pair_sub_counts = []    # how many subs per pair

    for pair in pairs:
        result = analyze_pair(pair)
        if result is None:
            continue
        total_source_words += result['n_source_words']
        total_rewrite_words += result['n_rewrite_words']
        for k, v in result['type_counts'].items():
            type_totals[k] += v
        n_subs = len(result['subs_1to1'])
        pair_sub_counts.append(n_subs)
        if n_subs > 0:
            pairs_with_subs += 1
        for s in result['subs_1to1']:
            s['pair_id'] = result['pair_id']
            all_subs.append(s)

    print(f"Total 1:1 substitutions: {len(all_subs)}", flush=True)

    # ---- Filter by criteria ----
    # Source-absent, both full-word, both content-word
    safw = [s for s in all_subs
            if s['source_absent'] and s['both_full_content']]
    # Source-absent, rewrite full-word (broader)
    sa_rfw = [s for s in all_subs
              if s['source_absent'] and s['rewrite_full_word']]
    # Source-absent, rewrite full-word, rewrite content
    sa_rfw_cont = [s for s in all_subs
                   if s['source_absent'] and s['rewrite_full_word']
                   and s['rewrite_content']]

    print(f"SAFW (both full-word, both content, source-absent): {len(safw)}", flush=True)
    print(f"SA rewrite-full-word: {len(sa_rfw)}", flush=True)
    print(f"SA rewrite-full-word content: {len(sa_rfw_cont)}", flush=True)

    # ---- Stem leakage in SAFW ----
    stem_leaked = sum(1 for s in safw if s['stem_leakage'])
    same_stem4 = sum(1 for s in safw if s['same_stem4'])
    stem_leak_frac = stem_leaked / max(1, len(safw))
    same_stem_frac = same_stem4 / max(1, len(safw))

    # ---- Unique word pairs and bidirectional density ----
    pair_counter = Counter()
    for s in safw:
        pair_counter[(s['source_word'], s['rewrite_word'])] += 1
    unique_pairs = len(pair_counter)

    bidir_pairs = set()
    for (a, b) in pair_counter:
        if (b, a) in pair_counter:
            bidir_pairs.add(tuple(sorted([a, b])))
    n_bidir = len(bidir_pairs)

    # ---- Edit distance distribution in SAFW ----
    ced_vals = [s['char_edit_dist'] for s in safw]
    if ced_vals:
        ced_mean = sum(ced_vals) / len(ced_vals)
        ced_vals_sorted = sorted(ced_vals)
        ced_median = ced_vals_sorted[len(ced_vals_sorted) // 2]
        high_dist = sum(1 for v in ced_vals if v >= 0.5)
        high_dist_frac = high_dist / len(ced_vals)
    else:
        ced_mean = ced_median = high_dist_frac = 0.0

    # ---- Pairs contributing ----
    pairs_with_safw = len(set(s['pair_id'] for s in safw))
    pairs_contributing_frac = pairs_with_safw / max(1, len(pairs))

    # ---- Top substitution pairs ----
    top_pairs = pair_counter.most_common(50)
    top_bidir = sorted(bidir_pairs)[:30]
    top_bidir_with_counts = []
    for (a, b) in top_bidir:
        c_ab = pair_counter.get((a, b), 0)
        c_ba = pair_counter.get((b, a), 0)
        top_bidir_with_counts.append({
            'word_a': a, 'word_b': b,
            'count_a_to_b': c_ab, 'count_b_to_a': c_ba,
        })

    # ---- Route decision ----
    viable = (len(safw) >= 2000 and unique_pairs >= 500
              and pairs_contributing_frac >= 0.10
              and stem_leak_frac < 0.30 and n_bidir >= 50)
    marginal = (not viable and len(safw) >= 500 and unique_pairs >= 100
                and stem_leak_frac < 0.50)
    closed = not viable and not marginal

    if viable:
        decision = "VIABLE"
        decision_text = ("Edit operator inventory meets all viable thresholds. "
                         "Proceed to paired-context equivariant mechanism design.")
    elif marginal:
        decision = "MARGINAL"
        decision_text = ("Edit operator inventory in marginal zone. "
                         "Need model-based plausibility readout before committing.")
    else:
        decision = "CLOSED"
        reasons = []
        if len(safw) < 500:
            reasons.append(f"SAFW count {len(safw)} < 500")
        if unique_pairs < 50:
            reasons.append(f"unique pairs {unique_pairs} < 50")
        if stem_leak_frac > 0.60:
            reasons.append(f"stem leakage {stem_leak_frac:.3f} > 0.60")
        decision_text = ("Edit operator inventory below thresholds. "
                         f"Reasons: {'; '.join(reasons) or 'marginal miss'}. "
                         "Edit equivariant route closed; move to different mechanism.")

    # ---- Build summary ----
    summary = {
        "status": "EDIT_OPERATOR_INVENTORY_DONE",
        "input": {
            "pair_file": str(PAIR_JSONL),
            "n_pairs_loaded": len(pairs),
            "total_source_words": total_source_words,
            "total_rewrite_words": total_rewrite_words,
        },
        "edit_type_distribution": dict(type_totals.most_common()),
        "substitutions_1to1": {
            "total": len(all_subs),
            "source_absent_rewrite_full_word": len(sa_rfw),
            "source_absent_rewrite_full_word_content": len(sa_rfw_cont),
            "SAFW_both_full_content_source_absent": len(safw),
        },
        "SAFW_analysis": {
            "count": len(safw),
            "stem_leakage_count": stem_leaked,
            "stem_leakage_fraction": round(stem_leak_frac, 4),
            "same_stem4_count": same_stem4,
            "same_stem4_fraction": round(same_stem_frac, 4),
            "char_edit_distance_mean": round(ced_mean, 4),
            "char_edit_distance_median": round(ced_median, 4),
            "high_distance_ge05_fraction": round(high_dist_frac, 4),
            "unique_word_pairs": unique_pairs,
            "bidirectional_pairs": n_bidir,
            "bidirectional_density": round(n_bidir / max(1, unique_pairs), 4),
            "pairs_contributing": pairs_with_safw,
            "pairs_contributing_fraction": round(pairs_contributing_frac, 4),
        },
        "threshold_check": {
            "SAFW_ge_2000": len(safw) >= 2000,
            "unique_pairs_ge_500": unique_pairs >= 500,
            "contributing_frac_ge_010": pairs_contributing_frac >= 0.10,
            "stem_leak_lt_030": stem_leak_frac < 0.30,
            "bidir_ge_50": n_bidir >= 50,
        },
        "route_decision": decision,
        "route_decision_text": decision_text,
        "top_SAFW_pairs": [{"pair": f"{a} → {b}", "count": c}
                           for (a, b), c in top_pairs],
        "top_bidirectional": top_bidir_with_counts,
        "pair_sub_count_distribution": {
            "pairs_with_any_1to1": pairs_with_subs,
            "pairs_with_SAFW": pairs_with_safw,
            "mean_subs_per_pair": round(sum(pair_sub_counts) / max(1, len(pair_sub_counts)), 3),
        },
    }

    # ---- Save ----
    summary_path = OUT_DIR / "edit_operator_inventory.json"
    with summary_path.open('w') as f:
        json.dump(summary, f, indent=2)

    # Save SAFW examples for inspection
    examples_path = OUT_DIR / "safw_examples.jsonl"
    with examples_path.open('w') as f:
        for s in safw[:500]:
            f.write(json.dumps(s) + '\n')

    # Markdown note
    note_path = (OUT_DIR.parents[4] / 'research/documents/representation_and_objectives/data/edit_operator_inventory/edit_operator_inventory.md')
    with note_path.open('w') as f:
        f.write("# research: Edit operator inventory results\n\n")
        f.write(f"Pairs loaded: {len(pairs)}\n")
        f.write(f"Total 1:1 substitutions: {len(all_subs)}\n")
        f.write(f"SAFW (source-absent, both full-word, both content): {len(safw)}\n")
        f.write(f"Unique SAFW word pairs: {unique_pairs}\n")
        f.write(f"Bidirectional pairs: {n_bidir}\n")
        f.write(f"Stem leakage fraction: {stem_leak_frac:.4f}\n")
        f.write(f"Same-stem4 fraction: {same_stem_frac:.4f}\n")
        f.write(f"Pairs contributing SAFW: {pairs_with_safw} ({pairs_contributing_frac:.3f})\n")
        f.write(f"Char edit dist (mean/median): {ced_mean:.3f}/{ced_median:.3f}\n")
        f.write(f"High-distance (≥0.5) fraction: {high_dist_frac:.3f}\n\n")
        f.write(f"## Route decision: **{decision}**\n\n{decision_text}\n\n")
        f.write("## Edit type distribution\n")
        for t, c in type_totals.most_common(15):
            f.write(f"  {t}: {c}\n")
        f.write(f"\n## Top 20 SAFW word pairs\n")
        for (a, b), c in top_pairs[:20]:
            f.write(f"  {a} → {b}: {c}\n")
        if top_bidir_with_counts:
            f.write(f"\n## Bidirectional examples (first 15)\n")
            for item in top_bidir_with_counts[:15]:
                f.write(f"  {item['word_a']} ↔ {item['word_b']}: "
                        f"{item['count_a_to_b']}+{item['count_b_to_a']}\n")

    print(json.dumps({"status": summary["status"],
                       "route_decision": decision,
                       "SAFW_count": len(safw),
                       "unique_pairs": unique_pairs,
                       "bidirectional": n_bidir,
                       "stem_leak_frac": round(stem_leak_frac, 4),
                       "contributing_frac": round(pairs_contributing_frac, 4),
                       "summary": str(summary_path),
                       "note": str(note_path)}), flush=True)


if __name__ == "__main__":
    main()
