#!/usr/bin/env python3
"""research: Analyze template vocabulary for anchor transfer experiment design."""
import json, collections
from pathlib import Path

WS = Path("experiments/archive/representation_and_objectives")

# Analyze research pilot templates
templates = {}
template_examples = {}
for split in ['train', 'held']:
    p = WS / f"data/paired_world_pilot/families_{split}.jsonl"
    for line in p.read_text().strip().split('\n'):
        fam = json.loads(line)
        for cx in ['context1', 'context2']:
            tid = fam[cx]['template_id']
            tsplit = fam[cx]['template_split']
            polarity = fam[cx].get('template_first_mention', '?')
            text = fam[cx]['text_score_ablated']
            if tid not in templates:
                templates[tid] = {'split': tsplit, 'first_mention': polarity, 'count': 0}
                template_examples[tid] = text
            templates[tid]['count'] += 1

print(f"Total unique templates: {len(templates)}")
print(f"\nTemplate details:")
for tid in sorted(templates.keys()):
    t = templates[tid]
    ex = template_examples[tid]
    print(f"  T{tid:02d} split={t['split']:5s} first={t['first_mention']} n={t['count']:3d} | {ex[:150]}")

# Count train vs held splits
train_tids = [tid for tid, t in templates.items() if t['split'] == 'train']
held_tids = [tid for tid, t in templates.items() if t['split'] == 'held']
print(f"\nTrain templates: {sorted(train_tids)}")
print(f"Held templates: {sorted(held_tids)}")

# Analyze first-mention distribution (winner-first vs loser-first)
fm_dist = collections.Counter(t['first_mention'] for t in templates.values())
print(f"First mention distribution: {dict(fm_dist)}")

# ATP upset substrate
atp_templates = {}
atp_examples = {}
for split in ['train', 'held']:
    p = WS / f"data/upset_balanced_state_substrate/families_{split}.jsonl"
    for line in p.read_text().strip().split('\n'):
        fam = json.loads(line)
        for cx in ['context1', 'context2']:
            pol = fam[cx].get('event_template_polarity', '?')
            etext = fam[cx]['event_text']
            # Extract predicate phrase by removing names
            winner = fam[cx]['winner']
            loser = fam[cx]['loser']
            key = (pol, etext[:80].replace(winner, 'W').replace(loser, 'L'))
            if key not in atp_templates:
                atp_templates[key] = {'count': 0}
                atp_examples[key] = etext
            atp_templates[key]['count'] += 1

# Extract unique predicate verbs from ATP
print(f"\nATP substrate unique event text patterns: {len(atp_templates)}")
# Collect unique predicate phrases
pred_phrases = set()
for split in ['train', 'held']:
    p = WS / f"data/upset_balanced_state_substrate/families_{split}.jsonl"
    for line in p.read_text().strip().split('\n'):
        fam = json.loads(line)
        for cx in ['context1', 'context2']:
            etext = fam[cx]['event_text']
            winner = fam[cx]['winner']
            loser = fam[cx]['loser']
            # Replace names with placeholders
            clean = etext.replace(winner, 'WINNER').replace(loser, 'LOSER')
            pred_phrases.add(clean[:100])

print(f"Unique event text patterns (first 100 chars): {len(pred_phrases)}")
for i, pp in enumerate(sorted(pred_phrases)[:30]):
    print(f"  {i}: {pp}")

# Count polarities in ATP
pol_counts = collections.Counter()
for split in ['train', 'held']:
    p = WS / f"data/upset_balanced_state_substrate/families_{split}.jsonl"
    for line in p.read_text().strip().split('\n'):
        fam = json.loads(line)
        for cx in ['context1', 'context2']:
            pol_counts[fam[cx].get('event_template_polarity', '?')] += 1
print(f"\nATP polarity distribution: {dict(pol_counts)}")
