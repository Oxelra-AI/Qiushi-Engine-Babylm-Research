#!/usr/bin/env python3
"""research: Extract ordered-prefix survivors from research v2 census and make route decision.

Key question: How many targets need BOTH the true prefix specifically (full > same_source_near)
AND ordered prefix content (full > block_shuffled)? These are the only candidates where
a cross-sentence training objective would train a genuinely different signal from topic coherence.
"""
import csv, json, math, pathlib, statistics
ROOT = pathlib.Path('experiments/archive/initial_model_studies')
ROWS = ROOT/'data/prefix_target_census_v2_fixedpos_rows.csv'
OUT = ROOT/'data/ordered_prefix_analysis.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/ordered_prefix_route_decision.md')

def signed_min(a, b):
    return math.copysign(min(abs(a), abs(b)), a+b) if a*b > 0 else 0.0

def main():
    rows = list(csv.DictReader(open(ROWS, newline='', encoding='utf-8')))
    results = {}
    for exp in ['40M', '80M', '100M']:
        # Per-target: compute sign-stable deltas for each variant
        ordered_specific = []  # full > same AND full > block
        ordered_only = []      # full > block but not full > same
        specific_only = []     # full > same but not full > block
        neither = []
        
        for r in rows:
            a42 = float(r[f'wwm42_{exp}_full_minus_same_source_near_fixedpos'])
            a43 = float(r[f'wwm43_{exp}_full_minus_same_source_near_fixedpos'])
            b42 = float(r[f'wwm42_{exp}_full_minus_block_shuffled_fixedpos'])
            b43 = float(r[f'wwm43_{exp}_full_minus_block_shuffled_fixedpos'])
            
            same_stable = (a42 >= 0.05 and a43 >= 0.05)
            block_stable = (b42 >= 0.05 and b43 >= 0.05)
            
            if same_stable and block_stable:
                ordered_specific.append(r)
            elif block_stable and not same_stable:
                ordered_only.append(r)
            elif same_stable and not block_stable:
                specific_only.append(r)
            else:
                neither.append(r)
        
        # Also check at lower threshold 0.02
        ordered_specific_002 = []
        for r in rows:
            a42 = float(r[f'wwm42_{exp}_full_minus_same_source_near_fixedpos'])
            a43 = float(r[f'wwm43_{exp}_full_minus_same_source_near_fixedpos'])
            b42 = float(r[f'wwm42_{exp}_full_minus_block_shuffled_fixedpos'])
            b43 = float(r[f'wwm43_{exp}_full_minus_block_shuffled_fixedpos'])
            if a42>=0.02 and a43>=0.02 and b42>=0.02 and b43>=0.02:
                ordered_specific_002.append(r)
        
        results[exp] = {
            'n': len(rows),
            'ordered_specific_005': len(ordered_specific),
            'ordered_specific_005_frac': len(ordered_specific)/len(rows),
            'ordered_only_005': len(ordered_only),
            'specific_only_005': len(specific_only),
            'neither_005': len(neither),
            'ordered_specific_002': len(ordered_specific_002),
            'ordered_specific_002_frac': len(ordered_specific_002)/len(rows),
        }
        
        # Characterize the ordered_specific targets
        if ordered_specific:
            sources = {}
            for r in ordered_specific:
                s = r['source']
                sources[s] = sources.get(s, 0) + 1
            results[exp]['ordered_specific_by_source'] = sources
            results[exp]['ordered_specific_targets'] = [
                {'case_id': r['case_id'], 'source': r['source'], 'target': r['target_surface'],
                 'freq': r['target_freq_bucket'], 'leak': r['local_leak_10tok'],
                 'stable_same': signed_min(float(r[f'wwm42_{exp}_full_minus_same_source_near_fixedpos']),
                                           float(r[f'wwm43_{exp}_full_minus_same_source_near_fixedpos'])),
                 'stable_block': signed_min(float(r[f'wwm42_{exp}_full_minus_block_shuffled_fixedpos']),
                                            float(r[f'wwm43_{exp}_full_minus_block_shuffled_fixedpos']))}
                for r in ordered_specific
            ]
    
    # Cross-exposure stability: how many 100M ordered_specific also pass at 80M?
    os_100 = set()
    for r in rows:
        a42=float(r['wwm42_100M_full_minus_same_source_near_fixedpos']); a43=float(r['wwm43_100M_full_minus_same_source_near_fixedpos'])
        b42=float(r['wwm42_100M_full_minus_block_shuffled_fixedpos']); b43=float(r['wwm43_100M_full_minus_block_shuffled_fixedpos'])
        if a42>=0.05 and a43>=0.05 and b42>=0.05 and b43>=0.05:
            os_100.add(r['case_id'])
    os_80 = set()
    for r in rows:
        a42=float(r['wwm42_80M_full_minus_same_source_near_fixedpos']); a43=float(r['wwm43_80M_full_minus_same_source_near_fixedpos'])
        b42=float(r['wwm42_80M_full_minus_block_shuffled_fixedpos']); b43=float(r['wwm43_80M_full_minus_block_shuffled_fixedpos'])
        if a42>=0.05 and a43>=0.05 and b42>=0.05 and b43>=0.05:
            os_80.add(r['case_id'])
    results['cross_exposure_stability'] = {
        'ordered_specific_at_100M': len(os_100),
        'ordered_specific_at_80M': len(os_80),
        'overlap_100M_and_80M': len(os_100 & os_80),
    }
    
    payload = {'status': 'ORDERED_PREFIX_ANALYSIS', 'results': results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    
    # Write route decision note
    r100 = results['100M']
    lines = [
        '# research — Ordered-prefix route decision',
        '',
        f'Source: `{OUT}`',
        '',
        '## Decisive measurement',
        '',
        'Targets needing BOTH the true prefix specifically (full > same_source_near >= 0.05, sign-stable)',
        'AND ordered prefix content (full > block_shuffled >= 0.05, sign-stable):',
        '',
        '| exposure | ordered+specific | fraction | at threshold 0.02 | fraction |',
        '|---|---:|---:|---:|---:|',
    ]
    for exp in ['40M', '80M', '100M']:
        r = results[exp]
        lines.append(f"| {exp} | {r['ordered_specific_005']} | {r['ordered_specific_005_frac']:.4f} | {r['ordered_specific_002']} | {r['ordered_specific_002_frac']:.4f} |")
    
    cs = results['cross_exposure_stability']
    lines += [
        '',
        f"Cross-exposure stability: {cs['ordered_specific_at_100M']} at 100M, {cs['ordered_specific_at_80M']} at 80M, overlap {cs['overlap_100M_and_80M']}.",
        '',
        '## Route interpretation',
        '',
        'The ordered-prefix filter is the key discriminator between topic/lexical coherence',
        'and genuine ordered discourse dependence. With the block-shuffled requirement:',
        '',
        f"- At 100M only {r100['ordered_specific_005']}/600 targets ({r100['ordered_specific_005_frac']:.1%}) survive both filters at threshold 0.05.",
        f"- At threshold 0.02, {r100['ordered_specific_002']}/600 ({r100['ordered_specific_002_frac']:.1%}) survive.",
        f"- The remaining {r100['specific_only_005']} same-source-positive targets that fail block-shuffled",
        '  are primarily topic/lexical continuity rather than ordered-state dependence.',
        '',
        '## Route decision',
        '',
    ]
    
    if r100['ordered_specific_005'] < 30:
        lines += [
            'The ordered-discourse target population on official text is too sparse (<5% of sampled targets)',
            'to support a reliable training objective. Any CPC, continuation, or contrastive mechanism',
            'that trains on broad suffix targets will be dominated by the much larger locally-predictable',
            'and topic-continuity populations, explaining why RTD, CPC, and prefix-continuation all',
            'learned source/register contrasts instead of ordered-state effects.',
            '',
            '**Cross-sentence ordered-state objective route is closed for official BabyLM text.**',
            '',
            'The next training routes should attack the Entity/EWoK/GlobalPIQA deficit through',
            'channels that do NOT require ordered prefix-to-suffix credit assignment:',
            '',
            '1. **Factorized WWM cadence** (sequence-length schedule, mask-rate decay, smaller effective batch)',
            '   — directly supported by BabyLM 2025 findings; targets context integration and optimization.',
            '2. **Word/morphology-level representation anchoring** — targets low-frequency entity/property',
            '   parameter sharing without requiring a new training interface.',
            '3. **Architecture/tokenizer changes** — more radical but with strong prior evidence.',
        ]
    else:
        lines += [
            f"A nontrivial ordered-discourse target population exists ({r100['ordered_specific_005']}/600).",
            'A high-precision training objective that selects only ordered-specific targets',
            'is worth constructing. Use these targets as a mandatory evaluation surface.',
        ]
    
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(OUT), 'note': str(NOTE), 'summary_100M': results['100M'],
                      'cross_exposure': results['cross_exposure_stability']}, indent=2))

if __name__ == '__main__':
    main()
