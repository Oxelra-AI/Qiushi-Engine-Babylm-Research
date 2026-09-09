import json, pathlib
base='experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery'
m=[json.loads(l) for l in open(base+'/qwen35_generation_merged_quality.jsonl')]
# Entity loss cases: is it systematic (dropping real named entities) or heuristic (case/possessive/partial match)?
ent_loss=[r for r in m if r['source_entities'] and r['entity_recall']<1.0]
print('rows with any entity:', sum(1 for r in m if r['source_entities']), 'entity-loss rows:', len(ent_loss))
print('--- 8 entity-loss samples ---')
for r in ent_loss[:8]:
    print('ENT:', r['source_entities'])
    print('SRC:', r['source_text'][:150])
    print('RW :', r['rewrite_text'][:150])
    print('  ent_recall %.2f ratio %.2f'%(r['entity_recall'],r['compression_ratio']))
# How many entity-loss cases still contain the entity substring case-insensitively at token level (dropped only leading/full name part)?
import re
partial=0
for r in ent_loss:
    rw=r['rewrite_text'].lower()
    for e in r['source_entities']:
        toks=[t for t in re.findall(r'[a-z0-9]+', e.lower()) if len(t)>2]
        if toks and any(t in rw for t in toks):
            partial+=1; break
print('\nentity-loss rows where >=1 entity token still present:', partial, 'of', len(ent_loss))
