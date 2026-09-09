#!/usr/bin/env python3
"""Post-analysis for research v2 role-family substrate.

Separates retained diversity from selection collapse and teacher/counterfactual
failure modes.  CPU-only; uses research outputs.
"""
from __future__ import annotations
import collections, json, re, time
from pathlib import Path
from typing import Any

ROOT = Path('experiments/archive/representation_and_objectives')
OUT = ROOT / 'data/auto_role_fact_substrate_v2'


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

def read_jsonl(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def rate(n,d): return n/d if d else 0.0

def norm_role(r): return re.sub(r'[^a-z0-9]+','_',str(r or '').lower()).strip('_')

def main():
    summary = json.loads((OUT/'v2_role_substrate_summary.json').read_text(encoding='utf-8'))
    cases = read_jsonl(OUT/'generated_family_cases.jsonl')
    pair_rows = read_jsonl(OUT/'cross_teacher_prompt_pairs.jsonl')
    failures = read_jsonl(OUT/'family_failure_reasons.jsonl')
    retained = read_jsonl(OUT/'retained_source_bridge_role_families.jsonl')
    retained_full = read_jsonl(OUT/'retained_full_context_role_families.jsonl')

    ok_cases = [c for c in cases if c.get('ok_family')]
    malformed = [c for c in cases if c.get('parse_note_or_error') and 'json_parse_error' in str(c.get('parse_note_or_error'))]
    empty_reject = [c for c in cases if not c.get('facts') and not c.get('ok_family') and not c.get('parse_note_or_error')]
    structurally_bad = [c for c in cases if c.get('facts') and not c.get('ok_family')]

    # Row-level failure categories.
    fail_pairs = [r for r in pair_rows if not r.get('both_expected')]
    cats = collections.Counter()
    by_context = collections.Counter()
    by_teacher_pattern = collections.Counter()
    by_gold = collections.Counter()
    by_role = collections.Counter()
    examples = []
    for r in fail_pairs:
        q, l, g = r.get('qwen_label'), r.get('llama_label'), r.get('gold_label')
        if q == g and l != g:
            pat = 'llama_not_expected'
        elif l == g and q != g:
            pat = 'qwen_not_expected'
        elif q != g and l != g and q == l:
            pat = 'both_same_wrong'
        elif q != g and l != g:
            pat = 'both_not_expected_different_or_invalid'
        else:
            pat = 'other'
        by_teacher_pattern[pat] += 1
        by_context[r.get('context_type')] += 1
        by_gold[g] += 1
        by_role[norm_role(r.get('role_type'))] += 1
        if g == 'NOT_ENTAILED' and (q == 'ENTAILED' or l == 'ENTAILED'):
            cats['negative_overaccepted_by_at_least_one_teacher'] += 1
        if g == 'NOT_ENTAILED' and q == 'ENTAILED' and l == 'ENTAILED':
            cats['negative_accepted_by_both_teachers'] += 1
        if g == 'ENTAILED' and (q == 'NOT_ENTAILED' or l == 'NOT_ENTAILED'):
            cats['positive_rejected_by_at_least_one_teacher'] += 1
        if len(examples) < 40:
            examples.append(r)

    # Family-level failure reasons aggregated.
    fam_reason_counter = collections.Counter()
    for fr in failures:
        for reason in fr.get('reasons', []): fam_reason_counter[reason] += 1
    retained_ids = {r['case_id'] for r in retained}
    retained_full_ids = {r['case_id'] for r in retained_full}
    strict_ids = []
    seed_summary = ROOT/'data/role_substrate_review/seed_export_summary.json'
    if seed_summary.exists():
        strict_ids = json.loads(seed_summary.read_text(encoding='utf-8')).get('strict_case_ids', [])
    seed_set = set(strict_ids)

    comp = {
        'strict_case_ids': strict_ids,
        'source_bridge_retained_case_ids': sorted(retained_ids),
        'full_context_retained_case_ids': sorted(retained_full_ids),
        'intersection_step248_step249_source_bridge': sorted(seed_set & retained_ids),
        'lost_in_step249_source_bridge': sorted(seed_set - retained_ids),
        'new_vs_step248_source_bridge': sorted(retained_ids - seed_set),
        'intersection_rate_of_step248_strict': rate(len(seed_set & retained_ids), len(seed_set)),
    }

    # Check if positives and negatives survive together per retained family.
    retained_family_shapes = []
    for r in retained:
        pos = [f for f in r.get('facts', []) if f.get('gold_label') == 'ENTAILED']
        neg = [f for f in r.get('facts', []) if f.get('gold_label') == 'NOT_ENTAILED']
        retained_family_shapes.append({
            'case_id': r['case_id'],
            'role_type_norm': r.get('role_type_norm'),
            'n_pos': len(pos), 'n_neg': len(neg),
            'neg_swap_types': [f.get('swap_type') for f in neg],
            'pos_purposes': [f.get('purpose') for f in pos],
            'mapped_ewok_domains': r.get('mapped_ewok_domains', []),
        })

    # Conservative judgment flags.
    ok_n = len(ok_cases); sb_n = len(retained); full_n = len(retained_full)
    role_div = len(set(r.get('role_type_norm') for r in retained))
    rel_overlap = summary.get('ewok_bridge_panel_mapping',{}).get('retained_source_bridge_overlap_step211_relational_domains', [])
    sparse = (sb_n < 20 or rate(sb_n, ok_n) < 0.5 or role_div < 4)
    diverse_but_sparse = (sb_n >= 10 and role_div >= 4 and len(rel_overlap) >= 2)
    finding = 'selection_collapse'
    if diverse_but_sparse:
        finding = 'real_diverse_seed_core_but_sparse'
    if sb_n >= 20 and rate(sb_n, ok_n) >= 0.5 and role_div >= 4:
        finding = 'usable_nontrivial_substrate'

    post = {
        'status': 'V2_ROLE_SUBSTRATE_POSTANALYSIS',
        'created_utc': now(),
        'main_summary': {
            'candidate_cases': len(cases),
            'malformed_json_cases': len(malformed),
            'model_empty_reject_cases': len(empty_reject),
            'structurally_bad_cases': len(structurally_bad),
            'ok_counterfactual_family_cases': ok_n,
            'retained_source_bridge_families': sb_n,
            'retained_full_context_families': full_n,
            'retained_source_bridge_over_candidates': rate(sb_n, len(cases)),
            'retained_source_bridge_over_ok': rate(sb_n, ok_n),
            'retained_full_context_over_candidates': rate(full_n, len(cases)),
            'retained_full_context_over_ok': rate(full_n, ok_n),
            'distinct_retained_role_types': role_div,
            'mapped_step211_relational_overlap_domains': rel_overlap,
            'finding': finding,
        },
        'row_failure_counts': {
            'n_prompt_pairs': len(pair_rows),
            'n_failed_prompt_pairs': len(fail_pairs),
            'failure_rate': rate(len(fail_pairs), len(pair_rows)),
            'semantic_failure_categories': dict(cats),
            'teacher_failure_pattern_counts': dict(by_teacher_pattern),
            'context_counts_among_failures': dict(by_context),
            'gold_label_counts_among_failures': dict(by_gold),
            'role_type_counts_among_failures': dict(by_role.most_common()),
        },
        'family_reason_counts': dict(fam_reason_counter),
        'retained_family_shapes': retained_family_shapes,
        'step249_retention_comparison': comp,
        'interpretation': (
            'The v2 prompt produced many nominal counterfactual families and retained more structural variety than research independent facts, but strict source-bridge retention is only '
            f'{sb_n}/{ok_n} structurally valid families ({rate(sb_n, ok_n):.3f}) and {sb_n}/{len(cases)} original A/B cases ({rate(sb_n, len(cases)):.3f}); full-context retention is {full_n}/{ok_n}. '
            'Most row failures are counterfactual negatives accepted by at least one teacher, especially Llama, not parser noise. The retained core spans several role types and maps heuristically onto research relational EWoK domains, so it is useful as seed/evaluation material, but it is too sparse for distillation from the present collection alone.'
        ),
        'next_research_action': 'Use retained families as probes and either construct a broader source-attested transformation source or search existing corpora/resources for naturally paired role-exchange transformations; do not train a student or BabyLM model from the current 11-family core.',
    }
    (OUT/'v2_postanalysis_summary.json').write_text(json.dumps(post, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    write_jsonl(OUT/'failed_prompt_pair_examples.jsonl', examples)
    md = [
        '# research v2 post-analysis', '',
        '## Main finding', '', post['interpretation'], '',
        '## Counts', '',
    ]
    for k,v in post['main_summary'].items(): md.append(f'- {k}: {v}')
    md += ['', '## Row failure categories', '']
    for k,v in post['row_failure_counts']['semantic_failure_categories'].items(): md.append(f'- {k}: {v}')
    md += ['', '## Teacher failure patterns', '']
    for k,v in post['row_failure_counts']['teacher_failure_pattern_counts'].items(): md.append(f'- {k}: {v}')
    md += ['', '## research/research retained-case relation', '']
    for k,v in comp.items(): md.append(f'- {k}: {v}')
    md += ['', '## Consequence', '', post['next_research_action'], '']
    ((OUT.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_postanalysis_summary.md')).write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps({
        'status': post['status'],
        **post['main_summary'],
        'failure_categories': post['row_failure_counts']['semantic_failure_categories'],
        'teacher_failure_patterns': post['row_failure_counts']['teacher_failure_pattern_counts'],
        'summary_json': str(OUT/'v2_postanalysis_summary.json'),
        'summary_md': str((OUT.parents[4] / 'research/documents/representation_and_objectives/data/auto_role_fact_substrate_v2/v2_postanalysis_summary.md')),
    }, indent=2, ensure_ascii=False), flush=True)

if __name__ == '__main__': main()
