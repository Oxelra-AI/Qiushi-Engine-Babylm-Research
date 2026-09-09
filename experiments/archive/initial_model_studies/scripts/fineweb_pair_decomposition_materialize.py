#!/usr/bin/env python3
"""research FineWeb paired-restatement decomposition materializer.

Purpose: test whether aligned FineWeb simplification pairing (same entity-value
binding in two adjacent surface forms) produces Entity/EWoK gains beyond source
distribution, static relation density, or broken-binding controls.

Four arms (exact 1M whitespace words each):
  orig_only:             original FineWeb sentences only
  simp_only:             deterministic simplified/restated sentences only
  true_pair_adjacent:    original immediately followed by its own simplification
  shuffled_pair_adjacent: original followed by simplification from another item

Key contrast: true_pair - shuffled_pair tests whether aligned restatement of
the same binding matters beyond style/token statistics/two-sentence exposure.

Simplification rules are deterministic and reproducible (no external LLM).
All text from publicly available FineWeb-Edu; no evaluation contamination.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, random, re, sys, time
from dataclasses import dataclass, asdict, field
from typing import Optional

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
OUT_DIR_DEFAULT = ROOT / 'data/fineweb_pair_decomposition'
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


def is_listish_sentence(text: str) -> bool:
    """Reject catalog/metadata/list fragments that produce fake appositives."""
    if ':' in text or ' - ' in text or ' | ' in text or '\t' in text:
        return True
    if len(re.findall(r'\b\d{4}\b', text)) >= 2:
        return True
    if text.count(',') >= 5 and wc(text) < 35:
        return True
    return False


def noun_phrase_text(tok) -> str:
    """Small deterministic antecedent phrase: compounds/amods + head token."""
    parts = []
    for left in tok.lefts:
        if left.dep_ in ('compound', 'amod', 'poss', 'nummod', 'flat', 'name'):
            parts.extend([t.text for t in left.subtree])
    parts.append(tok.text)
    return ' '.join(parts).strip()


@dataclass
class PairItem:
    """One original sentence and its deterministic simplification(s)."""
    doc_id: int
    pair_id: int
    original: str
    simplified: str
    rule_type: str
    shared_entities: list[str]
    shared_content_words: list[str]
    orig_words: int
    simp_words: int
    orig_tokens_approx: int  # rough estimate before actual tokenization


@dataclass
class PackedRow:
    example_id: int
    arm: str
    text: str
    words: int
    pair_ids: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deterministic simplification rules using spaCy
# ---------------------------------------------------------------------------

def load_spacy():
    import spacy
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        print("WARNING: en_core_web_sm not found; downloading...")
        os.system('python -m spacy download en_core_web_sm')
        return spacy.load('en_core_web_sm')


def try_relative_clause_split(doc, sent) -> Optional[tuple[str, str]]:
    """X, who/which/that V..., ... → X V... . [main clause]"""
    for tok in sent:
        if tok.dep_ == 'relcl':
            subtree_toks = list(tok.subtree)
            subtree_ids = set(t.i for t in subtree_toks)
            rel_text = ' '.join(t.text for t in subtree_toks).strip()
            head = tok.head
            # Main clause without the relative clause tokens
            main_parts = [t.text_with_ws for t in sent if t.i not in subtree_ids]
            main_without_rel = ''.join(main_parts).strip().rstrip(',')
            if not main_without_rel or wc(main_without_rel) < 3:
                continue
            # Replace relative pronoun with antecedent
            rel_clean = rel_text.lstrip(',').strip()
            if rel_clean.lower().startswith(('who ', 'which ', 'that ')):
                ant = head.text
                rel_clean = ant + ' ' + ' '.join(rel_clean.split()[1:])
            simplified = f"{rel_clean.rstrip('.')} . {main_without_rel.rstrip('.')}"
            # Quality filter: reject fragment-reordering failures. The first clause
            # (before the period) must begin with a capitalized word or a proper
            # subject-like token and contain at least 4 words, otherwise the split
            # produced a dangling fragment rather than a clean restatement.
            first_clause = simplified.split(' . ')[0].strip()
            if wc(first_clause) < 4:
                continue
            fw = first_clause.split()[0]
            if not (fw[:1].isupper() or fw.lower() in ('the', 'a', 'an')):
                continue
            if wc(simplified) >= 5 and wc(simplified) <= wc(sent.text) * 1.5:
                return (sent.text.strip(), simplified.strip())
    return None


def try_coordination_split(doc, sent) -> Optional[tuple[str, str]]:
    """X V1 ... and V2 ... → X V1 ... . X V2 ..."""
    for tok in sent:
        if tok.dep_ == 'conj' and tok.pos_ == 'VERB':
            subj = None
            for child in tok.head.children:
                if child.dep_ in ('nsubj', 'nsubjpass'):
                    subj = child.text
                    break
            if not subj:
                continue
            subtree_toks = list(tok.subtree)
            conj_text = ' '.join(t.text for t in subtree_toks)
            second = f"{subj} {conj_text}".strip()
            skip_ids = set(t.i for t in subtree_toks)
            for child in tok.head.children:
                if child.dep_ == 'cc':
                    skip_ids.add(child.i)
            first_parts = [t.text_with_ws for t in sent if t.i not in skip_ids]
            first = ''.join(first_parts).strip()
            if wc(first) >= 4 and wc(second) >= 4:
                simplified = f"{first.rstrip('.')} . {second.rstrip('.')}"
                if wc(simplified) <= wc(sent.text) * 1.5:
                    return (sent.text.strip(), simplified.strip())
    return None


def try_appositive_extraction(doc, sent) -> Optional[tuple[str, str]]:
    """X, a Y, ... → X is a Y. [rest]

    Conservative high-precision version: require a role/property-like appositive
    containing a common noun, not a pure proper-name alias such as "Johnson is
    Naveen Pawar". Reject catalog/list fragments with colons and metadata.
    """
    if is_listish_sentence(sent.text):
        return None
    for tok in sent:
        if tok.dep_ == 'appos':
            ant = noun_phrase_text(tok.head)
            subtree_toks = list(tok.subtree)
            # Require role/common-noun content; pure PROPN/NUM aliases are noisy.
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


def try_passive_to_active(doc, sent) -> Optional[tuple[str, str]]:
    """X was V-ed by Y → Y V-ed X (only when explicit 'by' agent)"""
    for tok in sent:
        if tok.dep_ == 'agent':
            verb = tok.head
            if not any(c.dep_ == 'auxpass' for c in verb.children):
                continue
            # Agent is the object of 'by'
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


SIMPLIFICATION_RULES = [
    ('relative_clause', try_relative_clause_split),
    # The simple coordination splitter generated too many fragment continuations in research smoke.
    # Keep it disabled until a higher-precision subject/argument reconstruction is implemented.
    # ('coordination', try_coordination_split),
    ('appositive', try_appositive_extraction),
    ('passive_active', try_passive_to_active),
]


def simplify_sentence(nlp, text: str) -> Optional[tuple[str, str, str]]:
    """Apply deterministic simplification rules. Returns (original, simplified, rule_type) or None."""
    doc = nlp(text)
    for sent in doc.sents:
        if wc(sent.text) < 8 or wc(sent.text) > 50:
            continue
        for rule_name, rule_fn in SIMPLIFICATION_RULES:
            result = rule_fn(doc, sent)
            if result is not None:
                orig, simp = result
                # Quality check: shared entities and content words
                orig_ents = entities_in(orig)
                simp_ents = entities_in(simp)
                shared_ents = list(orig_ents & simp_ents)
                orig_cw = content_words(orig)
                simp_cw = content_words(simp)
                shared_cw = list(orig_cw & simp_cw)
                # Require at least some shared content
                if len(shared_cw) < 2:
                    continue
                # Simplified should not be much longer
                if wc(simp) > wc(orig) * 1.6:
                    continue
                # Simplified should not be trivially short
                if wc(simp) < 4:
                    continue
                return (orig, simp, rule_name)
    return None


# ---------------------------------------------------------------------------
# Streaming and filtering
# ---------------------------------------------------------------------------

def stream_fineweb(max_docs: int):
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT',
                      split='train', streaming=True)
    for i, row in enumerate(ds):
        if i >= max_docs:
            break
        yield i, row.get('text', '')


def clean_doc(text: str) -> str:
    text = text.replace('\r', ' ').replace('\t', ' ')
    lines = []
    for line in text.split('\n'):
        s = ' '.join(line.strip().split())
        if len(s) >= 25:
            lines.append(s)
    return ' '.join(lines)


def basic_quality(text: str) -> bool:
    if any(b in text.lower() for b in BAD_SUBSTR):
        return False
    w = wc(text)
    if w < 80 or w > 900:
        return False
    alpha = sum(ch.isalpha() for ch in text) / max(1, len(text))
    return alpha > 0.65


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def pack_arm(items: list[PairItem], target_words: int, arm: str,
             rng: random.Random, max_row_words: int = 160) -> list[PackedRow]:
    """Pack pair items into training rows for one arm."""
    pool = list(items)
    rng.shuffle(pool)
    rows = []
    buf_words = []
    buf_pairs = []
    consumed = 0
    ex_id = 0

    for item in pool:
        if arm == 'orig_only':
            chunk = item.original
        elif arm == 'simp_only':
            chunk = item.simplified
        elif arm == 'true_pair_adjacent':
            chunk = f"{item.original} {item.simplified}"
        elif arm == 'shuffled_pair_adjacent':
            # Will be handled separately; this function handles non-shuffled arms
            chunk = item.original  # placeholder
        else:
            raise ValueError(f"Unknown arm: {arm}")

        toks = chunk.split()
        pos = 0
        while pos < len(toks) and consumed < target_words:
            need = min(max_row_words - len(buf_words), target_words - consumed, len(toks) - pos)
            if need <= 0:
                break
            buf_words.extend(toks[pos:pos + need])
            consumed += need
            pos += need
            if item.pair_id not in buf_pairs:
                buf_pairs.append(item.pair_id)
            if len(buf_words) >= max_row_words or consumed >= target_words:
                rows.append(PackedRow(ex_id, arm, ' '.join(buf_words), len(buf_words), list(buf_pairs)))
                ex_id += 1
                buf_words = []
                buf_pairs = []

    if consumed < target_words:
        raise RuntimeError(f"Insufficient material for {arm}: got {consumed}, need {target_words}")
    return rows


def _coarse_role(tok) -> str:
    """Collapse dependency labels into roles used to prevent anomaly swaps."""
    d = tok.dep_
    if d in ('nsubj', 'nsubjpass', 'csubj'):
        return 'subject'
    if d in ('dobj', 'obj', 'iobj'):
        return 'object'
    if d in ('pobj', 'obl'):
        return 'prep_object'
    if d in ('attr', 'oprd', 'acomp'):
        return 'predicate'
    if d in ('appos',):
        return 'appositive'
    if d in ('conj',):
        return _coarse_role(tok.head)
    return d or 'other'


def _number(tok, text: str) -> str:
    if ' and ' in text.lower() or '&' in text:
        return 'plural'
    if tok.tag_ in ('NNS', 'NNPS'):
        return 'plural'
    return 'singular'


def _gender_hint(text: str) -> str:
    t = text.lower().strip()
    male_titles = ('mr.', 'mr ', 'sir ', 'king ', 'prince ', 'lord ', 'father ')
    female_titles = ('mrs.', 'mrs ', 'ms.', 'ms ', 'miss ', 'queen ', 'princess ', 'lady ', 'mother ')
    if t.startswith(male_titles):
        return 'male'
    if t.startswith(female_titles):
        return 'female'
    # Small high-precision name list; unknown is allowed only with unknown.
    male = {'john','james','william','george','charles','thomas','joseph','robert','michael','david','richard','henry'}
    female = {'mary','jane','elizabeth','anne','anna','maria','sarah','emily','alice','margaret','catherine'}
    first = re.sub(r'[^a-z]', '', t.split()[0]) if t.split() else ''
    if first in male:
        return 'male'
    if first in female:
        return 'female'
    return 'unknown'


def _compatible_entity_candidates(nlp, simplified: str):
    """Named-entity candidates with type/number/role attributes.

    Uses spaCy NER rather than regex capitalization so common words such as
    `Standing`, `March`, or sentence-initial adjectives are not swapped as
    entities. The compatibility constraints deliberately reduce yield to keep the
    hard negative from becoming anomaly detection.
    """
    doc = nlp(simplified)
    allowed = {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'NORP', 'EVENT', 'WORK_OF_ART', 'PRODUCT'}
    out = []
    seen = set()
    for ent in doc.ents:
        txt = ent.text.strip()
        if ent.label_ not in allowed:
            continue
        if len(txt) < 4 or txt.split()[0] in LEAD_BAD:
            continue
        if txt.lower() in seen:
            continue
        root = ent.root
        cand = {
            'start': ent.start_char,
            'end': ent.end_char,
            'text': txt,
            'label': ent.label_,
            'role': _coarse_role(root),
            'number': _number(root, txt),
            'gender': _gender_hint(txt),
        }
        seen.add(txt.lower())
        out.append(cand)
    return out


def corrupt_binding(simplified: str, shared_entities: list[str],
                    rng: random.Random, nlp=None) -> Optional[str]:
    """Create a hard negative by swapping compatible named entities inside the
    simplification.

    Compatibility requirements for the corrected control:
    same NER type, same coarse syntactic role, same number, and no visible
    gender/title conflict. This preserves topic/entity/vocabulary while breaking
    a local binding without making the sentence obviously anomalous.
    """
    if nlp is None:
        return None
    cands = _compatible_entity_candidates(nlp, simplified)
    pairs = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a['text'].lower() == b['text'].lower():
                continue
            if a['label'] != b['label'] or a['role'] != b['role'] or a['number'] != b['number']:
                continue
            if a['gender'] != b['gender'] and 'unknown' not in (a['gender'], b['gender']):
                continue
            # Avoid swapping nested/overlapping spans or entities with very different surface length.
            if not (a['end'] <= b['start'] or b['end'] <= a['start']):
                continue
            if max(len(a['text'].split()), len(b['text'].split())) > 3 + min(len(a['text'].split()), len(b['text'].split())):
                continue
            pairs.append((a, b))
    if not pairs:
        return None
    a, b = rng.choice(pairs)
    (sa, ea, ta), (sb, eb, tb) = sorted([(a['start'], a['end'], a['text']), (b['start'], b['end'], b['text'])], key=lambda x: x[0])
    corrupted = simplified[:sa] + tb + simplified[ea:sb] + ta + simplified[eb:]
    if corrupted == simplified:
        return None
    return corrupted


def pack_hard_negative_arm(items: list[PairItem], target_words: int,
                           rng: random.Random, nlp, max_row_words: int = 160) -> tuple[list[PackedRow], dict]:
    """Pack hard-negative pairs: original + a binding-corrupted version of its OWN
    simplification (same topic, entity set, vocabulary; only entity-role/attribute
    correspondence is broken). Falls back to a same-document donor simplification
    with entity swap when intra-simplification swap is impossible.
    """
    pool = list(items)
    rng.shuffle(pool)
    # Index items by document for same-topic donor fallback
    by_doc = collections.defaultdict(list)
    for it in pool:
        by_doc[it.doc_id].append(it)

    rows = []
    buf_words = []
    buf_pairs = []
    consumed = 0
    ex_id = 0
    n_intra = 0
    n_donor = 0
    n_skipped = 0

    for item in pool:
        corrupted = corrupt_binding(item.simplified, item.shared_entities, rng, nlp)
        if corrupted is not None:
            n_intra += 1
        else:
            # No donor fallback for the hard negative: the matched control must
            # preserve the simplification's own topic/entity/lexical material. If an
            # intra-simplification entity swap is impossible, skip this item.
            n_skipped += 1
            continue
        chunk = f"{item.original} {corrupted}"
        toks = chunk.split()
        pos = 0
        while pos < len(toks) and consumed < target_words:
            need = min(max_row_words - len(buf_words), target_words - consumed, len(toks) - pos)
            if need <= 0:
                break
            buf_words.extend(toks[pos:pos + need])
            consumed += need
            pos += need
            if item.pair_id not in buf_pairs:
                buf_pairs.append(item.pair_id)
            if len(buf_words) >= max_row_words or consumed >= target_words:
                rows.append(PackedRow(ex_id, 'hard_negative_pair', ' '.join(buf_words),
                                      len(buf_words), list(buf_pairs)))
                ex_id += 1
                buf_words = []
                buf_pairs = []

    stats = {'n_compatible_intra_swap': n_intra, 'n_donor_fallback': n_donor, 'n_skipped': n_skipped}
    if consumed < target_words:
        raise RuntimeError(f"Insufficient material for hard_negative_pair: got {consumed}, need {target_words}; stats={stats}")
    return rows, stats


def pack_shuffled_arm(items: list[PairItem], target_words: int,
                      rng: random.Random, max_row_words: int = 160) -> list[PackedRow]:
    """Pack shuffled pairs: each original followed by a DIFFERENT item's simplification."""
    pool = list(items)
    rng.shuffle(pool)
    # Create deranged simplifications
    simps = [item.simplified for item in pool]
    # Derange: shift by half the pool size to ensure no item gets its own simplification
    n = len(simps)
    shift = max(1, n // 2)
    deranged = simps[shift:] + simps[:shift]

    rows = []
    buf_words = []
    buf_pairs = []
    consumed = 0
    ex_id = 0

    for i, item in enumerate(pool):
        chunk = f"{item.original} {deranged[i]}"
        toks = chunk.split()
        pos = 0
        while pos < len(toks) and consumed < target_words:
            need = min(max_row_words - len(buf_words), target_words - consumed, len(toks) - pos)
            if need <= 0:
                break
            buf_words.extend(toks[pos:pos + need])
            consumed += need
            pos += need
            if item.pair_id not in buf_pairs:
                buf_pairs.append(item.pair_id)
            if len(buf_words) >= max_row_words or consumed >= target_words:
                rows.append(PackedRow(ex_id, 'shuffled_pair_adjacent', ' '.join(buf_words),
                                      len(buf_words), list(buf_pairs)))
                ex_id += 1
                buf_words = []
                buf_pairs = []

    if consumed < target_words:
        raise RuntimeError(f"Insufficient material for shuffled_pair_adjacent: got {consumed}, need {target_words}")
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_stream_docs', type=int, default=30000)
    ap.add_argument('--target_words', type=int, default=1_000_000)
    ap.add_argument('--max_row_words', type=int, default=160)
    ap.add_argument('--seed', type=int, default=284)
    ap.add_argument('--out_dir', default=str(OUT_DIR_DEFAULT))
    ap.add_argument('--smoke', action='store_true', help='Run small smoke test (100k words)')
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

    # Phase 1: Stream and extract pair items
    print(f"Streaming up to {args.max_stream_docs} FineWeb-Edu docs...", flush=True)
    pair_items = []
    pair_id = 0
    docs_streamed = 0
    docs_basic_pass = 0
    rule_counts = collections.Counter()

    for doc_id, raw_text in stream_fineweb(args.max_stream_docs):
        docs_streamed += 1
        text = clean_doc(raw_text)
        if not basic_quality(text):
            continue
        docs_basic_pass += 1

        # Process sentences
        doc_nlp = nlp(text)
        for sent in doc_nlp.sents:
            sent_text = sent.text.strip()
            if wc(sent_text) < 8 or wc(sent_text) > 50:
                continue
            result = simplify_sentence(nlp, sent_text)
            if result is None:
                continue
            orig, simp, rule_type = result
            shared_e = list(entities_in(orig) & entities_in(simp))
            shared_c = list(content_words(orig) & content_words(simp))
            pair_items.append(PairItem(
                doc_id=doc_id, pair_id=pair_id,
                original=orig, simplified=simp, rule_type=rule_type,
                shared_entities=shared_e, shared_content_words=shared_c,
                orig_words=wc(orig), simp_words=wc(simp),
                orig_tokens_approx=int(wc(orig) * 1.3)
            ))
            rule_counts[rule_type] += 1
            pair_id += 1

        # Check if we have enough material
        total_orig_words = sum(p.orig_words for p in pair_items)
        total_simp_words = sum(p.simp_words for p in pair_items)
        total_pair_words = sum(p.orig_words + p.simp_words for p in pair_items)
        # For shuffled arm, we need orig+simp words too
        if min(total_orig_words, total_simp_words, total_pair_words) >= args.target_words * 1.3:
            break

        if docs_streamed % 5000 == 0:
            print(f"  streamed {docs_streamed}, basic_pass {docs_basic_pass}, "
                  f"pairs {len(pair_items)}, rules {dict(rule_counts)}", flush=True)

    print(f"Extraction complete: {len(pair_items)} pairs from {docs_streamed} docs "
          f"({docs_basic_pass} basic-pass), rules: {dict(rule_counts)}", flush=True)

    if len(pair_items) < 100:
        raise RuntimeError(f"Too few pair items: {len(pair_items)}")

    # Phase 2: Pack four arms
    print("Packing arms...", flush=True)
    arms_data = {}
    for arm in ['orig_only', 'simp_only', 'true_pair_adjacent']:
        arm_rng = random.Random(args.seed + hash(arm) % 10000)
        arms_data[arm] = pack_arm(pair_items, args.target_words, arm, arm_rng, args.max_row_words)
    # Shuffled uses same items but deranged
    shuf_rng = random.Random(args.seed + 9999)
    arms_data['shuffled_pair_adjacent'] = pack_shuffled_arm(
        pair_items, args.target_words, shuf_rng, args.max_row_words)
    # Hard-negative: same topic/entities/vocabulary, binding correspondence broken
    hn_rng = random.Random(args.seed + 7777)
    arms_data['hard_negative_pair'], hard_negative_stats = pack_hard_negative_arm(
        pair_items, args.target_words, hn_rng, nlp, args.max_row_words)
    print(f"  hard_negative stats: {hard_negative_stats}", flush=True)

    # Phase 3: Write JSONL files
    jsonl_paths = {}
    for arm, rows in arms_data.items():
        path = out_dir / f'{arm}_{args.target_words}w.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps({'example_id': r.example_id, 'source': arm,
                                    'text': r.text, 'words': r.words,
                                    'pair_ids': r.pair_ids[:20]},
                                   ensure_ascii=False) + '\n')
        jsonl_paths[arm] = str(path)
        actual_words = sum(r.words for r in rows)
        print(f"  {arm}: {len(rows)} rows, {actual_words} words", flush=True)

    # Phase 4: Metadata and samples
    sample_true = [asdict(p) for p in pair_items[:30]]
    sample_shuf_indices = list(range(len(pair_items)))
    rng.shuffle(sample_shuf_indices)
    sample_shuf = []
    for idx in sample_shuf_indices[:20]:
        p = pair_items[idx]
        other_idx = (idx + max(1, len(pair_items) // 2)) % len(pair_items)
        other = pair_items[other_idx]
        sample_shuf.append({
            'original': p.original,
            'shuffled_simplified': other.simplified,
            'original_pair_id': p.pair_id,
            'shuffled_from_pair_id': other.pair_id,
            'original_rule': p.rule_type,
        })

    # Hard-negative manual-reading samples: show original, true simp, corrupted simp
    sample_hard_neg = []
    hn_sample_rng = random.Random(args.seed + 4242)
    hn_indices = list(range(len(pair_items)))
    hn_sample_rng.shuffle(hn_indices)
    for idx in hn_indices:
        if len(sample_hard_neg) >= 20:
            break
        p = pair_items[idx]
        corrupted = corrupt_binding(p.simplified, p.shared_entities, hn_sample_rng, nlp)
        if corrupted is None:
            continue
        sample_hard_neg.append({
            'original': p.original,
            'true_simplified': p.simplified,
            'corrupted_simplified': corrupted,
            'shared_entities': p.shared_entities,
            'pair_id': p.pair_id,
            'rule': p.rule_type,
        })

    meta = {
        'status': 'FINEWEB_PAIR_DECOMPOSITION_MATERIALIZED',
        'target_words_per_arm': args.target_words,
        'max_row_words': args.max_row_words,
        'seed': args.seed,
        'docs_streamed': docs_streamed,
        'docs_basic_pass': docs_basic_pass,
        'total_pair_items': len(pair_items),
        'rule_counts': dict(rule_counts),
        'hard_negative_stats': hard_negative_stats,
        'arms': {arm: {'rows': len(rows), 'words': sum(r.words for r in rows)}
                 for arm, rows in arms_data.items()},
        'jsonl_paths': jsonl_paths,
        'elapsed_sec': time.time() - t0,
    }
    meta_path = out_dir / 'materialization_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    samples_path = out_dir / 'samples_for_manual_reading.json'
    samples_path.write_text(json.dumps({
        'true_pairs': sample_true,
        'shuffled_examples': sample_shuf,
        'hard_negative_examples': sample_hard_neg,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(json.dumps({
        'meta': str(meta_path),
        'samples': str(samples_path),
        'jsonl_paths': jsonl_paths,
        'num_pairs': len(pair_items),
        'elapsed_sec': meta['elapsed_sec'],
    }, indent=2))
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
