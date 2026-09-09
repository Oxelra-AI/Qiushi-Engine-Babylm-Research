#!/usr/bin/env python3
"""research Same-Entity Cross-Clause pair decomposition materializer.

Hard-negative design: instead of swapping entities within a simplification
(too sparse, Steps 285-287), use SAME-ENTITY CROSS-CLAUSE negatives:
  true_pair: original_A + simplification_A (same fact restated)
  hard_neg:  original_A + simplification_B (different fact about same entity E)

Both contain entity E, same topic, same vocabulary, naturally readable.
Only the specific predicate/attribute/event binding differs.
"""
from __future__ import annotations
import argparse, collections, json, os, pathlib, random, re, sys, time
from dataclasses import dataclass, asdict, field
from typing import Optional

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR_DEFAULT = ROOT / 'data/same_entity_cross_clause'
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){0,3})\b")
BAD_SUBSTR = ['privacy policy', 'terms of use', 'subscribe', 'copyright',
              'all rights reserved', 'click here', '<script', 'cookie policy',
              'amazon.com', 'purchase through']
LEAD_BAD = {'The', 'This', 'That', 'These', 'Those', 'There', 'When', 'Where',
            'What', 'How', 'Why', 'Because', 'For', 'And', 'But', 'New', 'All',
            'Most', 'Some', 'Many', 'First', 'After', 'Before', 'During', 'Then',
            'Each', 'Every', 'Page', 'Pages'}


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TRANSFORMERS_CACHE', 'HF_MODULES_CACHE']:
        pathlib.Path(os.environ[k]).mkdir(parents=True, exist_ok=True)


def wc(text: str) -> int:
    return len(text.split())


def norm(w: str) -> str:
    return w.strip("'\".,!?;:()[]{}""''").lower()


def entities_in(text: str) -> set[str]:
    out = set()
    for m in ENTITY_RE.finditer(text):
        e = ' '.join(m.group(0).split())
        first = e.split()[0]
        if first in LEAD_BAD:
            continue
        if len(e) >= 4:
            out.add(e.lower())
    return out


def content_words(text: str) -> set[str]:
    stop = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'be', 'been',
            'has', 'had', 'have', 'do', 'did', 'does', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'not', 'no', 'it', 'its',
            'this', 'that', 'these', 'those', 'he', 'she', 'they', 'we', 'you',
            'i', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
            'their', 'who', 'which', 'what', 'where', 'when', 'how', 'why'}
    return {norm(w) for w in WORD_RE.findall(text) if norm(w) not in stop and len(norm(w)) >= 3}


def is_listish(text: str) -> bool:
    if ':' in text or ' - ' in text or ' | ' in text or '\t' in text:
        return True
    if len(re.findall(r'\b\d{4}\b', text)) >= 2:
        return True
    if text.count(',') >= 5 and wc(text) < 35:
        return True
    return False


# ---------------------------------------------------------------------------
# Simplification rules (from research strict version)
# ---------------------------------------------------------------------------

def load_spacy():
    import spacy
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')


def try_relative_clause_split(doc, sent):
    for tok in sent:
        if tok.dep_ == 'relcl':
            subtree_toks = list(tok.subtree)
            subtree_ids = set(t.i for t in subtree_toks)
            rel_text = ' '.join(t.text for t in subtree_toks).strip()
            head = tok.head
            main_parts = [t.text_with_ws for t in sent if t.i not in subtree_ids]
            main_without_rel = ''.join(main_parts).strip().rstrip(',')
            if not main_without_rel or wc(main_without_rel) < 3:
                continue
            rel_clean = rel_text.lstrip(',').strip()
            if rel_clean.lower().startswith(('who ', 'which ', 'that ')):
                ant = head.text
                rel_clean = ant + ' ' + ' '.join(rel_clean.split()[1:])
            simplified = f"{rel_clean.rstrip('.')} . {main_without_rel.rstrip('.')}"
            first_clause = simplified.split(' . ')[0].strip()
            if wc(first_clause) < 4:
                continue
            fw = first_clause.split()[0]
            if not (fw[:1].isupper() or fw.lower() in ('the', 'a', 'an')):
                continue
            if wc(simplified) >= 5 and wc(simplified) <= wc(sent.text) * 1.5:
                return (sent.text.strip(), simplified.strip())
    return None


def try_appositive_extraction(doc, sent):
    if is_listish(sent.text):
        return None
    for tok in sent:
        if tok.dep_ == 'appos':
            # Full antecedent phrase
            ant_parts = []
            for left in tok.head.lefts:
                if left.dep_ in ('compound', 'amod', 'poss', 'nummod', 'flat', 'name'):
                    ant_parts.extend([t.text for t in left.subtree])
            ant_parts.append(tok.head.text)
            ant = ' '.join(ant_parts).strip()
            subtree_toks = list(tok.subtree)
            if not any(t.pos_ == 'NOUN' for t in subtree_toks):
                continue
            propn_num = sum(1 for t in subtree_toks if t.pos_ in ('PROPN', 'NUM'))
            alpha_toks = [t for t in subtree_toks if t.is_alpha]
            if alpha_toks and propn_num / max(1, len(alpha_toks)) > 0.55:
                continue
            appos_text = ' '.join(t.text for t in subtree_toks).strip().strip(' ,()[]')
            if wc(appos_text) < 2 or wc(appos_text) > 10:
                continue
            if wc(ant) < 1 or norm(ant) in {'me', 'him', 'her', 'them', 'something'}:
                continue
            definition = f"{ant} is {appos_text}"
            skip_ids = set(t.i for t in subtree_toks)
            rest_parts = [t.text_with_ws for t in sent if t.i not in skip_ids]
            rest = ''.join(rest_parts).strip().strip(' ,')
            if wc(rest) >= 4:
                simplified = f"{definition} . {rest.rstrip('.')}"
                if wc(simplified) <= wc(sent.text) * 1.5:
                    return (sent.text.strip(), simplified.strip())
    return None


def try_passive_to_active(doc, sent):
    for tok in sent:
        if tok.dep_ == 'agent':
            verb = tok.head
            if not any(c.dep_ == 'auxpass' for c in verb.children):
                continue
            agent_toks = [c for c in tok.children if c.dep_ == 'pobj']
            if not agent_toks:
                continue
            agent = ' '.join(t.text for t in agent_toks[0].subtree)
            patient = None
            for child in verb.children:
                if child.dep_ == 'nsubjpass':
                    patient = ' '.join(t.text for t in child.subtree)
                    break
            if not patient or not agent:
                continue
            active = f"{agent} {verb.lemma_} {patient}"
            simplified = f"{active} ."
            if wc(simplified) >= 3:
                return (sent.text.strip(), simplified.strip())
    return None


RULES = [
    ('relative_clause', try_relative_clause_split),
    ('appositive', try_appositive_extraction),
    ('passive_active', try_passive_to_active),
]


@dataclass
class SimplItem:
    doc_id: int
    sent_idx: int
    original: str
    simplified: str
    rule_type: str
    entities: list[str]   # named entities present in BOTH original and simplified
    content_words_shared: list[str]
    orig_words: int
    simp_words: int


@dataclass
class PairSet:
    """A group of SimplItems sharing entity E within one document."""
    doc_id: int
    entity: str
    items: list[SimplItem]


# ---------------------------------------------------------------------------
# Streaming and extraction
# ---------------------------------------------------------------------------

def stream_fineweb(max_docs):
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        yield i, row.get('text', '')


def clean_doc(text):
    text = text.replace('\r', ' ').replace('\t', ' ')
    lines = [' '.join(line.strip().split()) for line in text.split('\n') if len(line.strip()) >= 25]
    return ' '.join(lines)


def basic_quality(text):
    if any(b in text.lower() for b in BAD_SUBSTR):
        return False
    w = wc(text)
    if w < 80 or w > 900:
        return False
    return sum(ch.isalpha() for ch in text) / max(1, len(text)) > 0.65


def extract_simplifications(nlp, doc_id, text):
    """Extract all simplifiable sentences from one document."""
    items = []
    doc = nlp(text)
    for sent_idx, sent in enumerate(doc.sents):
        st = sent.text.strip()
        if wc(st) < 8 or wc(st) > 50:
            continue
        for rule_name, rule_fn in RULES:
            result = rule_fn(doc, sent)
            if result is None:
                continue
            orig, simp = result
            orig_ents = entities_in(orig)
            simp_ents = entities_in(simp)
            shared_ents = sorted(orig_ents & simp_ents)
            orig_cw = content_words(orig)
            simp_cw = content_words(simp)
            shared_cw = sorted(orig_cw & simp_cw)
            if len(shared_cw) < 2:
                continue
            if wc(simp) > wc(orig) * 1.6 or wc(simp) < 4:
                continue
            items.append(SimplItem(
                doc_id=doc_id, sent_idx=sent_idx,
                original=orig, simplified=simp, rule_type=rule_name,
                entities=shared_ents, content_words_shared=shared_cw,
                orig_words=wc(orig), simp_words=wc(simp),
            ))
            break  # One simplification per sentence
    return items


def group_by_entity(items: list[SimplItem]) -> list[PairSet]:
    """Group items by shared entity within same document. Only groups with ≥2 items
    from DIFFERENT sentences qualify for hard-negative construction."""
    by_doc_ent = collections.defaultdict(list)
    for item in items:
        for ent in item.entities:
            by_doc_ent[(item.doc_id, ent)].append(item)

    groups = []
    for (doc_id, ent), group_items in by_doc_ent.items():
        # Must have items from ≥2 different sentences
        sent_ids = set(it.sent_idx for it in group_items)
        if len(sent_ids) >= 2:
            groups.append(PairSet(doc_id=doc_id, entity=ent, items=group_items))
    return groups


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

@dataclass
class PackedRow:
    example_id: int
    arm: str
    text: str
    words: int


def pack_rows(chunks: list[str], arm: str, target_words: int, max_row_words: int = 160) -> list[PackedRow]:
    rows = []
    buf = []
    consumed = 0
    ex_id = 0
    for chunk in chunks:
        toks = chunk.split()
        pos = 0
        while pos < len(toks) and consumed < target_words:
            need = min(max_row_words - len(buf), target_words - consumed, len(toks) - pos)
            if need <= 0:
                break
            buf.extend(toks[pos:pos + need])
            consumed += need
            pos += need
            if len(buf) >= max_row_words or consumed >= target_words:
                rows.append(PackedRow(ex_id, arm, ' '.join(buf), len(buf)))
                ex_id += 1
                buf = []
    if consumed < target_words:
        raise RuntimeError(f"Insufficient material for {arm}: got {consumed}, need {target_words}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_stream_docs', type=int, default=30000)
    ap.add_argument('--target_words', type=int, default=1_000_000)
    ap.add_argument('--max_row_words', type=int, default=160)
    ap.add_argument('--seed', type=int, default=288)
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    setup_env()
    t0 = time.time()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.target_words = 100_000
        args.max_stream_docs = 5000

    rng = random.Random(args.seed)
    nlp = load_spacy()

    # Phase 1: Extract all simplifications
    print(f"Streaming up to {args.max_stream_docs} FineWeb-Edu docs...", flush=True)
    all_items = []
    docs_streamed = 0
    docs_basic_pass = 0
    rule_counts = collections.Counter()

    for doc_id, raw_text in stream_fineweb(args.max_stream_docs):
        docs_streamed += 1
        text = clean_doc(raw_text)
        if not basic_quality(text):
            continue
        docs_basic_pass += 1
        items = extract_simplifications(nlp, doc_id, text)
        for it in items:
            rule_counts[it.rule_type] += 1
        all_items.extend(items)

        # Early stop based on ENTITY-GROUPED yield (only ~16% of items form groups)
        # Need ~6x total item words to get enough entity-grouped material
        total_pair_words = sum(it.orig_words + it.simp_words for it in all_items)
        if total_pair_words >= args.target_words * 8.0:
            break

        if docs_streamed % 2000 == 0:
            print(f"  streamed {docs_streamed}, basic_pass {docs_basic_pass}, "
                  f"items {len(all_items)}, rules {dict(rule_counts)}", flush=True)

    print(f"Extraction: {len(all_items)} items from {docs_streamed} docs "
          f"({docs_basic_pass} basic-pass), rules: {dict(rule_counts)}", flush=True)

    # Phase 2: Group by entity for cross-clause hard negatives
    groups = group_by_entity(all_items)
    print(f"Entity groups with ≥2 cross-sentence items: {len(groups)}", flush=True)

    # Build true pairs and same-multiset hard negatives from document-entity groups.
    # Within each group, use a fixed-point-free permutation of the SAME selected
    # simplifications. Then true and hard arms have identical originals and
    # identical simplification multisets; only adjacency correspondence differs.
    true_pair_chunks = []
    hard_neg_chunks = []
    orig_chunks = [it.original for it in all_items]
    all_true_pairs = []
    all_hard_negs = []
    deranged_group_records = []

    for grp in groups:
        # Deduplicate to one item per sentence so a cyclic shift never pairs a
        # sentence with another simplification from the same sentence.
        by_sent = {}
        for it in grp.items:
            by_sent.setdefault(it.sent_idx, it)
        items = list(by_sent.values())
        if len(items) < 2:
            continue
        rng.shuffle(items)
        perm = items[1:] + items[:1]  # fixed-point-free cyclic derangement
        records = []
        total_words = 0
        for item_a, item_b in zip(items, perm):
            # Skip near-identical simplification swaps; removing both item_a and
            # its paired item_b would be complex, so keep them but record overlap.
            cw_a = set(item_a.content_words_shared)
            cw_b = set(item_b.content_words_shared)
            jaccard = len(cw_a & cw_b) / max(1, len(cw_a | cw_b))
            rec = (grp, item_a, item_b, jaccard)
            records.append(rec)
            total_words += wc(item_a.original) + wc(item_a.simplified)
        deranged_group_records.append((total_words, records))

    rng.shuffle(deranged_group_records)
    structured_words = 0
    used_groups = 0
    for group_words, records in deranged_group_records:
        if structured_words + group_words > args.target_words:
            continue
        used_groups += 1
        structured_words += group_words
        for grp, item_a, item_b, jaccard in records:
            true_pair_chunks.append(f"{item_a.original} {item_a.simplified}")
            hard_neg_chunks.append(f"{item_a.original} {item_b.simplified}")
            all_true_pairs.append({
                'entity': grp.entity, 'doc_id': grp.doc_id,
                'original': item_a.original, 'simplified': item_a.simplified,
                'rule': item_a.rule_type, 'sent_idx': item_a.sent_idx,
            })
            all_hard_negs.append({
                'entity': grp.entity, 'doc_id': grp.doc_id,
                'original_A': item_a.original, 'simplification_B': item_b.simplified,
                'rule_A': item_a.rule_type, 'rule_B': item_b.rule_type,
                'sent_A': item_a.sent_idx, 'sent_B': item_b.sent_idx,
                'content_jaccard': jaccard,
                'derangement': 'same_doc_entity_fixed_point_free_cycle',
            })

    # Fill any remaining words with identical original-only text in both primary
    # arms. This preserves exact target length without changing their difference:
    # the structured portion alone carries the adjacency-correspondence contrast.
    same_filler_words = args.target_words - structured_words
    if same_filler_words > 0:
        filler = []
        fw = 0
        for chunk in orig_chunks:
            if fw >= same_filler_words:
                break
            toks = chunk.split()
            take = min(len(toks), same_filler_words - fw)
            if take > 0:
                filler.append(' '.join(toks[:take]))
                fw += take
        if fw != same_filler_words:
            raise RuntimeError(f"Could not create same filler: got {fw}, need {same_filler_words}")
        true_pair_chunks.extend(filler)
        hard_neg_chunks.extend(filler)

    # Also build shuffled arm from all items (cross-document, weak negative)
    all_simps = [it.simplified for it in all_items]
    rng.shuffle(all_simps)
    shift = max(1, len(all_simps) // 2)
    shuffled_simps = all_simps[shift:] + all_simps[:shift]
    shuffled_chunks = [f"{all_items[i].original} {shuffled_simps[i]}"
                       for i in range(len(all_items))]
    rng.shuffle(shuffled_chunks)

    rng.shuffle(true_pair_chunks)
    rng.shuffle(hard_neg_chunks)
    rng.shuffle(orig_chunks)

    print(f"True pairs: {len(true_pair_chunks)}, Hard negatives: {len(hard_neg_chunks)}, "
          f"Originals: {len(orig_chunks)}", flush=True)

    # Phase 3: Pack arms
    arms_data = {}
    try:
        arms_data['true_pair_adjacent'] = pack_rows(true_pair_chunks, 'true_pair_adjacent',
                                                     args.target_words, args.max_row_words)
        arms_data['hard_negative_same_entity'] = pack_rows(hard_neg_chunks, 'hard_negative_same_entity',
                                                            args.target_words, args.max_row_words)
        arms_data['orig_only'] = pack_rows(orig_chunks, 'orig_only',
                                            args.target_words, args.max_row_words)
        arms_data['shuffled_pair_adjacent'] = pack_rows(shuffled_chunks, 'shuffled_pair_adjacent',
                                                         args.target_words, args.max_row_words)
    except RuntimeError as e:
        print(f"PACKING FAILED: {e}", flush=True)
        # Save partial results for diagnosis
        meta = {
            'status': 'PACKING_FAILED',
            'error': str(e),
            'total_items': len(all_items),
            'entity_groups': len(groups),
            'true_pair_words': sum(wc(c) for c in true_pair_chunks),
            'hard_neg_words': sum(wc(c) for c in hard_neg_chunks),
            'orig_words': sum(wc(c) for c in orig_chunks),
            'structured_pair_words': structured_words,
            'same_filler_words': same_filler_words,
            'deranged_groups_used': used_groups,
            'target_words': args.target_words,
            'docs_streamed': docs_streamed,
            'docs_basic_pass': docs_basic_pass,
            'rule_counts': dict(rule_counts),
        }
        (out_dir / 'materialization_meta.json').write_text(
            json.dumps(meta, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(meta, indent=2))
        sys.exit(1)

    # Phase 4: Write JSONL
    jsonl_paths = {}
    for arm, rows in arms_data.items():
        path = out_dir / f'{arm}_{args.target_words}w.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps({'example_id': r.example_id, 'source': arm,
                                    'text': r.text, 'words': r.words},
                                   ensure_ascii=False) + '\n')
        jsonl_paths[arm] = str(path)
        actual_words = sum(r.words for r in rows)
        print(f"  {arm}: {len(rows)} rows, {actual_words} words", flush=True)

    # Phase 5: Metadata and samples
    meta = {
        'status': 'SAME_ENTITY_CROSS_CLAUSE_MATERIALIZED',
        'target_words_per_arm': args.target_words,
        'max_row_words': args.max_row_words,
        'seed': args.seed,
        'docs_streamed': docs_streamed,
        'docs_basic_pass': docs_basic_pass,
        'total_simpl_items': len(all_items),
        'entity_groups_with_cross_sentence': len(groups),
        'true_pairs_generated': len(true_pair_chunks),
        'hard_negatives_generated': len(hard_neg_chunks),
        'structured_pair_words': structured_words,
        'same_filler_words': same_filler_words,
        'deranged_groups_used': used_groups,
        'derangement_control': 'within each document-entity group, fixed-point-free cyclic permutation; true and hard arms have identical original and simplification multisets in structured portion; remaining words are identical filler',
        'rule_counts': dict(rule_counts),
        'arms': {arm: {'rows': len(rows), 'words': sum(r.words for r in rows)}
                 for arm, rows in arms_data.items()},
        'jsonl_paths': jsonl_paths,
        'elapsed_sec': time.time() - t0,
    }
    (out_dir / 'materialization_meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    # Save manual reading samples
    samples = {
        'true_pairs': all_true_pairs[:30],
        'hard_negatives': all_hard_negs[:30],
    }
    (out_dir / 'samples_for_manual_reading.json').write_text(
        json.dumps(samples, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps({
        'meta': str(out_dir / 'materialization_meta.json'),
        'samples': str(out_dir / 'samples_for_manual_reading.json'),
        'entity_groups': len(groups),
        'true_pairs': len(true_pair_chunks),
        'hard_negatives': len(hard_neg_chunks),
        'structured_pair_words': structured_words,
        'same_filler_words': same_filler_words,
        'deranged_groups_used': used_groups,
        'elapsed_sec': meta['elapsed_sec'],
    }, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
