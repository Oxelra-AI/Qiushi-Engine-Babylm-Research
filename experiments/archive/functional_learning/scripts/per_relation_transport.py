"""
Analyze h1 vs h3 transport separately in existing research full-alpha data.
Tests whether graph distance affects transport quality.
h1 is distance-1 from h0 anchor; h3 is distance-2 (through h2).
"""
import json, sys
from pathlib import Path
import numpy as np

DATA_ROOT = Path('experiments/archive/representation_and_objectives/data')
OUT = Path('experiments/archive/functional_learning/data/per_relation_transport')

def load_state_preds(run_dir):
    rows = []
    sp = list(run_dir.glob('*/state_predictions.jsonl'))
    if not sp:
        return []
    with open(sp[0]) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_index(rows):
    idx = {}
    for r in rows:
        key = (r.get('row_id', ''), r.get('relation'), tuple(r.get('names', [])), r.get('object'))
        idx[key] = r
    return idx

def analyze_pair(bp_dir, bm_dir, label):
    bp_rows = load_state_preds(bp_dir)
    bm_rows = load_state_preds(bm_dir)
    
    if not bp_rows or not bm_rows:
        print(f'[{label}] Missing data: bp={len(bp_rows)} bm={len(bm_rows)}')
        return None
    
    print(f'[{label}] Loaded {len(bp_rows)} bs+1 rows, {len(bm_rows)} bs-1 rows')
    
    bp_idx = build_index(bp_rows)
    bm_idx = build_index(bm_rows)
    
    results = {}
    for rel in ['h0_dax', 'h2_norp', 'h1_mep', 'h3_ziv']:
        rel_bp = {k: v for k, v in bp_idx.items() if v.get('relation') == rel and v.get('is_changed')}
        
        same_sign = 0
        opposite_sign = 0
        de_plus = []
        de_minus = []
        matched = 0
        
        for key, rp in rel_bp.items():
            if key in bm_idx:
                rm = bm_idx[key]
                dp = rp['d_e']
                dm = rm['d_e']
                de_plus.append(dp)
                de_minus.append(dm)
                if dp * dm > 0:
                    same_sign += 1
                elif dp * dm < 0:
                    opposite_sign += 1
                matched += 1
        
        de_plus = np.array(de_plus) if de_plus else np.array([0.0])
        de_minus = np.array(de_minus) if de_minus else np.array([0.0])
        
        r = {
            'matched': matched,
            'same_sign': same_sign,
            'opposite_sign': opposite_sign,
            'opposite_frac': opposite_sign / matched if matched > 0 else 0,
            'mean_abs_de_plus': float(np.mean(np.abs(de_plus))),
            'mean_abs_de_minus': float(np.mean(np.abs(de_minus))),
            'mean_margin_plus': float(np.mean(de_plus)),
            'mean_margin_minus': float(np.mean(de_minus)),
        }
        if matched > 1:
            r['anti_corr'] = float(np.corrcoef(de_plus, -de_minus)[0, 1])
        results[rel] = r
    
    return results

# Analyze research dual-pos full-alpha (learned equality)
res_299_full = analyze_pair(
    DATA_ROOT / 'dualpos_fullalpha_shared_bsplus_seed29930',
    DATA_ROOT / 'dualpos_fullalpha_shared_bsminus_seed29930',
    'research learned-equality full-alpha'
)

# Also analyze research repaired posalign (frozen equality)
res_299_frozen = analyze_pair(
    DATA_ROOT / 'posalign_repaired_shared_bsplus_seed29920',
    DATA_ROOT / 'posalign_repaired_shared_bsminus_seed29920',
    'research frozen posalign'
)

# Also check research full_graph
res_300 = analyze_pair(
    DATA_ROOT / 'full_graph_shared' / 'full_graph',
    DATA_ROOT / 'full_graph_shared' / 'full_graph',
    'research full_graph (self-comparison, not bridge-sign pair)'
)

# Print table
def print_table(label, results):
    if results is None:
        print(f'\n{label}: NO DATA\n')
        return
    
    print(f'\n=== {label} ===')
    print(f'{"Relation":<12} {"GraphDist":<15} {"Matched":<8} {"Opp":<6} {"Same":<6} '
          f'{"OppFrac":<8} {"AntiCorr":<10} {"MeanMarg+":<12} {"MeanMarg-":<12} {"AbsMarg+":<10} {"AbsMarg-":<10}')
    print('-' * 115)
    for rel in ['h0_dax', 'h2_norp', 'h1_mep', 'h3_ziv']:
        r = results[rel]
        dist = {'h0_dax': 'anchor', 'h2_norp': 'anchor', 'h1_mep': 'dist=1', 'h3_ziv': 'dist=2'}[rel]
        print(f'{rel:<12} {dist:<15} {r["matched"]:<8} {r["opposite_sign"]:<6} {r["same_sign"]:<6} '
              f'{r["opposite_frac"]:<8.3f} {r.get("anti_corr", 0):<10.4f} '
              f'{r["mean_margin_plus"]:<12.3f} {r["mean_margin_minus"]:<12.3f} '
              f'{r["mean_abs_de_plus"]:<10.3f} {r["mean_abs_de_minus"]:<10.3f}')

print_table('research learned-equality full-alpha, shared_trunk, seed29930', res_299_full)
print_table('research frozen posalign, shared_trunk, seed29920', res_299_frozen)

# Save results
OUT.mkdir(parents=True, exist_ok=True)
output = {
    'learned_equality': res_299_full,
    'frozen_posalign': res_299_frozen,
}
(OUT / 'per_relation_transport.json').write_text(json.dumps(output, indent=2) + '\n')

# Generate markdown summary
lines = ['# Per-Relation Transport Analysis: h1 (dist=1) vs h3 (dist=2)', '',
         'Separates h1 and h3 in existing saved outputs to test whether',
         'transport quality varies with graph distance from anchored nodes.', '',
         'Graph: h0--h1, h0--h2, h1--h2, h2--h3',
         'Anchors: h0, h2 (state training rows with bridge_sign flip)',
         'h1: distance 1 from h0 (direct edge), distance 1 from h2 (direct edge)',
         'h3: distance 1 from h2, distance 2 from h0 (via h2)', '']

for label, results in [
    ('research learned-equality full-alpha, shared_trunk, seed29930', res_299_full),
    ('research frozen posalign, shared_trunk, seed29920', res_299_frozen),
]:
    if results is None:
        lines.append(f'## {label}: NO DATA\n')
        continue
    lines.append(f'## {label}')
    lines.append('')
    lines.append('| Relation | GraphDist | Matched | OppFrac | AntiCorr | MeanMarg+ | MeanMarg- | AbsMarg+ | AbsMarg- |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for rel in ['h0_dax', 'h2_norp', 'h1_mep', 'h3_ziv']:
        r = results[rel]
        dist = {'h0_dax': 'anchor', 'h2_norp': 'anchor', 'h1_mep': 'dist=1', 'h3_ziv': 'dist=2'}[rel]
        lines.append(f'| {rel} | {dist} | {r["matched"]} | {r["opposite_frac"]:.3f} | {r.get("anti_corr", 0):.4f} | '
                      f'{r["mean_margin_plus"]:.3f} | {r["mean_margin_minus"]:.3f} | '
                      f'{r["mean_abs_de_plus"]:.3f} | {r["mean_abs_de_minus"]:.3f} |')
    lines.append('')

((OUT.parents[4] / 'research/documents/functional_learning/data/per_relation_transport/per_relation_transport.md')).write_text('\n'.join(lines) + '\n')
print(f'\nSaved to {OUT}')
