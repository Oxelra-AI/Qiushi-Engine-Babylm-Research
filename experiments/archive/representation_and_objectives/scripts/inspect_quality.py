import json, pathlib, collections
base='experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery'
s=json.load(open(base+'/qwen35_generation_quality_summary.json'))
print('core_accept', round(s['core_accept_rate'],3), 'struct_strict', round(s['structure_strict_accept_rate'],3))
print('fractions:')
for k,v in s['fractions'].items(): print('  ',k,round(v,3))
print('stats_all:')
for k,v in s['stats_all'].items():
    print('  ',k, {kk:(round(vv,3) if isinstance(vv,float) else vv) for kk,vv in v.items()} if v else v)
m=[json.loads(l) for l in open(base+'/qwen35_generation_merged_quality.jsonl')]
core=[r for r in m if r['accepted_core']]
sf=[r for r in core if not r['accepted_structure_strict']]
print('core-accepted:',len(core),'core-but-strict-fail:',len(sf))
reasons=collections.Counter()
for r in sf:
    if r['negation_marker_recall']<1.0: reasons['neg<1']+=1
    if r['modality_marker_recall']<1.0: reasons['modal<1']+=1
    if r['causal_marker_recall']<0.5: reasons['causal<0.5']+=1
    if r['comparison_marker_recall']<0.5: reasons['comp<0.5']+=1
print('strict-fail reasons among core:', dict(reasons))
print('--- 6 core-accepted samples ---')
for r in core[:6]:
    print('SRC:', r['source_text'][:150])
    print('RW :', r['rewrite_text'][:150])
    print('  ratio %.2f cont %.2f ent %.2f num %.2f neg %.2f mod %.2f caus %.2f comp %.2f'%(r['compression_ratio'],r['content_recall'],r['entity_recall'],r['number_recall'],r['negation_marker_recall'],r['modality_marker_recall'],r['causal_marker_recall'],r['comparison_marker_recall']))
print('--- 4 rejected(copy/malformed/short) ---')
bad=[r for r in m if r['copy_like'] or r['malformed'] or r['too_short']]
for r in bad[:4]:
    print('SRC:', r['source_text'][:130]); print('RW :', r['rewrite_text'][:130]); print('  copy',r['copy_like'],'mal',r['malformed'],'short',r['too_short'],'ratio %.2f'%r['compression_ratio'])
