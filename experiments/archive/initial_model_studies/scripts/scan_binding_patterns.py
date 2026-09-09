#!/usr/bin/env python3
"""research binding-pattern scanner.

Scans high_entity_state windows for multi-entity binding structures:
windows where ≥2 distinct entities each have distinct associated predicates,
locations, or objects that could be swapped to create a binding-broken control.

The key question: does the official BabyLM corpus contain enough natural
multi-entity binding structures to materialize the stronger objective?
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import random
from typing import Any

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT / 'data/structure_density_revision_157/high_entity_state.jsonl'
OUT = ROOT / 'data/binding_pattern_scan.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/binding_pattern_scan.md')

CAP_RE = re.compile(r"^[A-Z][a-z]+$")
STOP_ENTITIES = {
    'The', 'A', 'An', 'And', 'But', 'Or', 'If', 'When', 'While', 'He', 'She',
    'It', 'They', 'We', 'You', 'I', 'His', 'Her', 'Their', 'This', 'That',
    'These', 'Those', 'So', 'As', 'At', 'In', 'On', 'Of', 'To', 'For', 'With',
    'My', 'Your', 'Its', 'What', 'Who', 'Where', 'How', 'Why', 'Then', 'Now',
    'Here', 'There', 'After', 'Before', 'Not', 'No', 'Yes', 'Well', 'Oh',
    'Mr', 'Mrs', 'Miss', 'Dr', 'Sir', 'Just', 'Even', 'Still', 'Yet', 'Once'
}

# Verbs that create entity-state bindings (put X in Y, move X to Y, give X to Y)
BINDING_VERBS = {
    'put', 'puts', 'placed', 'place', 'moved', 'move', 'took', 'take',
    'brought', 'bring', 'gave', 'give', 'sent', 'send', 'left', 'leave',
    'dropped', 'drop', 'picked', 'pick', 'carried', 'carry', 'held', 'hold',
    'kept', 'keep', 'stored', 'store', 'hid', 'hide', 'found', 'find',
    'opened', 'open', 'closed', 'close', 'filled', 'fill', 'poured', 'pour',
    'set', 'sat', 'sit', 'stood', 'stand', 'lay', 'laid', 'hung', 'hang',
    'wore', 'wear', 'went', 'go', 'came', 'come', 'lived', 'live',
    'returned', 'return', 'entered', 'enter', 'pushed', 'push', 'pulled', 'pull',
    'threw', 'throw', 'caught', 'catch', 'grabbed', 'grab', 'lifted', 'lift'
}

SPATIAL_PREPS = {
    'in', 'on', 'at', 'under', 'over', 'above', 'below', 'inside', 'outside',
    'near', 'between', 'behind', 'beside', 'around', 'through', 'across',
    'into', 'onto', 'from', 'to', 'toward', 'towards', 'beneath'
}

POSSESSIVE_PATTERNS = re.compile(r"(\w+)'s\b")


def extract_entities(words: list[str]) -> list[tuple[int, str]]:
    """Extract (position, entity) pairs for capitalized non-stop words."""
    entities = []
    for i, w in enumerate(words):
        clean = re.sub(r"[^A-Za-z]", "", w)
        if CAP_RE.match(clean) and clean not in STOP_ENTITIES and len(clean) >= 2:
            entities.append((i, clean))
    return entities


def extract_bindings(words: list[str], entities: list[tuple[int, str]]) -> list[dict]:
    """Find entity-state/location bindings: entity followed by verb+prep+object patterns."""
    lower_words = [w.lower().strip('.,!?;:\'"()[]') for w in words]
    n = len(words)
    bindings = []

    for ent_pos, ent_name in entities:
        # Look in a window after the entity for binding verbs or spatial constructions
        search_start = ent_pos + 1
        search_end = min(n, ent_pos + 8)  # look up to 8 words ahead

        for i in range(search_start, search_end):
            lw = lower_words[i]

            # Pattern 1: entity ... verb ... prep ... object
            if lw in BINDING_VERBS:
                # Look for spatial prep after verb
                for j in range(i + 1, min(n, i + 4)):
                    if lower_words[j] in SPATIAL_PREPS:
                        # Object is next content word(s) after prep
                        obj_words = []
                        for k in range(j + 1, min(n, j + 4)):
                            cw = lower_words[k]
                            if cw in SPATIAL_PREPS or cw in BINDING_VERBS or cw in ('.', ',', '!', '?'):
                                break
                            if len(cw) >= 2:
                                obj_words.append(cw)
                        if obj_words:
                            bindings.append({
                                'entity': ent_name,
                                'entity_pos': ent_pos,
                                'verb': lw,
                                'verb_pos': i,
                                'prep': lower_words[j],
                                'prep_pos': j,
                                'object': ' '.join(obj_words),
                                'object_positions': list(range(j + 1, j + 1 + len(obj_words))),
                                'type': 'verb_prep_object'
                            })
                        break
                break  # only first verb after entity

            # Pattern 2: entity ... "is/was" ... "in/on/at" ... location
            if lw in ('is', 'was', 'are', 'were', 'lives', 'lived', 'stays', 'stayed'):
                for j in range(i + 1, min(n, i + 3)):
                    if lower_words[j] in SPATIAL_PREPS:
                        obj_words = []
                        for k in range(j + 1, min(n, j + 4)):
                            cw = lower_words[k]
                            if cw in SPATIAL_PREPS or cw in ('.', ',', '!', '?', 'and', 'but', 'or'):
                                break
                            if len(cw) >= 2:
                                obj_words.append(cw)
                        if obj_words:
                            bindings.append({
                                'entity': ent_name,
                                'entity_pos': ent_pos,
                                'verb': lw,
                                'verb_pos': i,
                                'prep': lower_words[j],
                                'prep_pos': j,
                                'object': ' '.join(obj_words),
                                'object_positions': list(range(j + 1, j + 1 + len(obj_words))),
                                'type': 'copula_prep_location'
                            })
                        break
                break

    return bindings


def find_swappable_bindings(bindings: list[dict]) -> list[tuple[dict, dict]]:
    """Find pairs of bindings from different entities that could be swapped."""
    # Group by entity
    by_entity: dict[str, list[dict]] = collections.defaultdict(list)
    for b in bindings:
        by_entity[b['entity']].append(b)

    entities = list(by_entity.keys())
    pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1_bindings = by_entity[entities[i]]
            e2_bindings = by_entity[entities[j]]
            for b1 in e1_bindings:
                for b2 in e2_bindings:
                    # Bindings are swappable if they have different objects
                    if b1['object'] != b2['object']:
                        pairs.append((b1, b2))
    return pairs


def has_later_reference(words: list[str], entity: str, after_pos: int) -> list[int]:
    """Check if the entity is mentioned again after a given position."""
    positions = []
    for i in range(after_pos + 1, len(words)):
        clean = re.sub(r"[^A-Za-z]", "", words[i])
        if clean == entity:
            positions.append(i)
    return positions


def analyze_window(text: str) -> dict[str, Any]:
    """Analyze a single window for binding structures."""
    words = text.split()
    entities = extract_entities(words)
    if len(set(e for _, e in entities)) < 2:
        return {'has_binding': False, 'reason': 'fewer than 2 distinct entities'}

    bindings = extract_bindings(words, entities)
    if len(bindings) < 2:
        return {'has_binding': False, 'reason': f'fewer than 2 bindings found ({len(bindings)})',
                'entities_found': len(set(e for _, e in entities)), 'bindings_found': len(bindings)}

    swappable = find_swappable_bindings(bindings)
    if not swappable:
        return {'has_binding': False, 'reason': 'no swappable binding pairs',
                'entities_found': len(set(e for _, e in entities)), 'bindings_found': len(bindings)}

    # Check for later reference that creates a binding-dependent target
    best_pair = None
    for b1, b2 in swappable:
        # Look for later mention of either entity after both bindings are established
        max_bind_pos = max(max(b1['object_positions']), max(b2['object_positions']))
        later1 = has_later_reference(words, b1['entity'], max_bind_pos)
        later2 = has_later_reference(words, b2['entity'], max_bind_pos)
        if later1 or later2:
            best_pair = (b1, b2, later1, later2)
            break

    has_downstream = best_pair is not None
    return {
        'has_binding': True,
        'has_downstream_reference': has_downstream,
        'entities_found': len(set(e for _, e in entities)),
        'bindings_found': len(bindings),
        'swappable_pairs': len(swappable),
        'bindings': bindings[:6],
        'best_swappable_pair': (best_pair[0], best_pair[1]) if best_pair else None,
        'later_references': (best_pair[2], best_pair[3]) if best_pair else None,
    }


def main():
    # Load unique epoch-0 windows
    windows = []
    seen = set()
    with DATA.open('r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('epoch') != 0:
                continue
            text = r['text']
            key = (r.get('source_file'), r.get('start_line'))
            if key in seen:
                continue
            seen.add(key)
            windows.append(text)

    print(f'Total unique epoch-0 windows: {len(windows)}')

    # Analyze all windows
    results = []
    has_binding_count = 0
    has_downstream_count = 0
    binding_counts = []
    swappable_counts = []
    examples_binding = []
    examples_downstream = []

    for i, text in enumerate(windows):
        analysis = analyze_window(text)
        results.append(analysis)
        if analysis['has_binding']:
            has_binding_count += 1
            binding_counts.append(analysis['bindings_found'])
            swappable_counts.append(analysis['swappable_pairs'])
            if analysis.get('has_downstream_reference'):
                has_downstream_count += 1
                if len(examples_downstream) < 15:
                    examples_downstream.append({
                        'window_idx': i,
                        'text_preview': text[:350],
                        'analysis': analysis,
                    })
            elif len(examples_binding) < 10:
                examples_binding.append({
                    'window_idx': i,
                    'text_preview': text[:300],
                    'analysis': analysis,
                })

    n = len(windows)
    payload = {
        'status': 'BINDING_PATTERN_SCAN',
        'data': str(DATA),
        'total_windows': n,
        'has_multi_entity_binding': has_binding_count,
        'has_multi_entity_binding_frac': has_binding_count / max(1, n),
        'has_downstream_reference': has_downstream_count,
        'has_downstream_reference_frac': has_downstream_count / max(1, n),
        'binding_count_mean': sum(binding_counts) / max(1, len(binding_counts)) if binding_counts else 0,
        'swappable_pair_mean': sum(swappable_counts) / max(1, len(swappable_counts)) if swappable_counts else 0,
        'examples_with_downstream_ref': examples_downstream,
        'examples_binding_only': examples_binding[:5],
        'interpretation': (
            f'{has_binding_count}/{n} windows ({100*has_binding_count/max(1,n):.1f}%) have >=2 entities '
            f'with swappable bindings. {has_downstream_count}/{n} ({100*has_downstream_count/max(1,n):.1f}%) '
            f'also have a later entity reference creating a binding-dependent prediction target. '
            f'If downstream fraction is >5%, a binding-dependent objective is feasible at ~10M word scale.'
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    lines = [
        '# research — binding-pattern scan of high_entity_state windows',
        '',
        f'Data: `{DATA}` (unique epoch-0 windows: {n})',
        '',
        f'| metric | count | fraction |',
        f'|---|---:|---:|',
        f'| ≥2 entities with swappable bindings | {has_binding_count} | {100*has_binding_count/max(1,n):.1f}% |',
        f'| + downstream entity reference (binding-dependent target) | {has_downstream_count} | {100*has_downstream_count/max(1,n):.1f}% |',
        '',
        f'Mean bindings per qualifying window: {payload["binding_count_mean"]:.2f}',
        f'Mean swappable pairs per qualifying window: {payload["swappable_pair_mean"]:.2f}',
        '',
        '## Feasibility assessment',
        '',
        f'If {has_downstream_count} windows × ~64 words/window ≈ {has_downstream_count * 64 // 1000}k words '
        f'of binding-dependent training examples exist, repeating to 10M needs '
        f'{10_000_000 // max(1, has_downstream_count * 64):.0f}× repetition. '
        f'This is feasible if the fraction is >5% (>~{n*5//100} windows).',
        '',
        'The scanner uses conservative heuristics; manual inspection of examples should verify quality.',
    ]
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'status': payload['status'],
        'total_windows': n,
        'has_binding': has_binding_count,
        'has_binding_pct': f'{100*has_binding_count/max(1,n):.1f}%',
        'has_downstream': has_downstream_count,
        'has_downstream_pct': f'{100*has_downstream_count/max(1,n):.1f}%',
    }, indent=2))


if __name__ == '__main__':
    main()
