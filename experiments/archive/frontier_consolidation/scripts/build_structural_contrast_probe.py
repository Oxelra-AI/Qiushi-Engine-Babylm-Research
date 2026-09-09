#!/usr/bin/env python3
"""research: build structural contrast-margin probe from the legal 10M corpus.

Produces minimal pairs where coherent and perturbed versions differ in exactly
one structural dimension. Each pair includes char-level focus spans for
targeted PLL scoring. No official benchmark text, labels, or IDs are used.

Families (ordered by plan weight):
 1. entity_state       (w=0.30) — controlled micro-discourses from corpus vocabulary
 2. temporal_order     (w=0.25) — order-sensitive event/procedure frames
 3. directed_relation  (w=0.20) — comparative argument swap + directed preposition
 4. belief_report      (w=0.15) — speaker/addressee/role binding
 5. polarity_relation  (w=0.10) — negation + relation interaction

Plus surface_control pairs matched to each family (used to compute L_t baseline).

Output:
  data/structural_contrast_probe/structural_contrast_probe.json
  data/structural_contrast_probe/structural_contrast_probe.md
"""
from __future__ import annotations
import json, re, random, hashlib, os, sys, time, itertools
from pathlib import Path
from collections import defaultdict

SEED = 42
POOL = Path("experiments/archive/frontier_consolidation/data"
            "density_cleanqwen_overlay_medium_riskhard/"
            "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
OUT = Path("experiments/archive/frontier_consolidation/data/structural_contrast_probe")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- helpers ---------------------------------------------------------

def sent_split(text: str) -> list[str]:
    """Conservative sentence split."""
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"])', text)
    return [s.strip() for s in sents if len(s.strip()) >= 12]

def word_tokens(s: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer for extraction."""
    return re.findall(r"\w+(?:'\w+)?", s)

def char_span_of_substring(text: str, sub: str, start: int = 0) -> tuple[int, int] | None:
    idx = text.find(sub, start)
    if idx < 0:
        return None
    return (idx, idx + len(sub))

def get_source(row: dict) -> str:
    return row.get("source", "unknown")

# ---------- Family 1: Entity State (controlled micro-discourses) -----------

# Vocabulary drawn from common English words that appear in the BabyLM corpus
# (CHILDES, Gutenberg, OpenSubtitles). No official benchmark text.
ES_OBJECTS = ["ball", "book", "cup", "hat", "key", "pen", "ring", "bag", "toy", "sock",
              "apple", "coin", "doll", "card", "stone", "scarf", "bottle", "shoe"]
# Separate containers (use "in") from surfaces (use "on")
ES_CONTAINERS = ["basket", "drawer", "closet", "box", "bag", "bucket", "cupboard"]
ES_SURFACES = ["table", "shelf", "desk", "floor", "bed", "chair", "bench", "counter"]
ES_ALL_LOCS = ES_CONTAINERS + ES_SURFACES

def loc_prep(loc: str) -> str:
    return "in" if loc in ES_CONTAINERS else "on"
ES_PERSONS = ["Alice", "Bob", "Tom", "Mary", "Sam", "Emma", "Jack", "Kate", "Dan", "Amy"]
ES_VERBS = ["puts", "moves", "places", "takes"]

def build_entity_state_pairs(rng: random.Random, n_depth2=100, n_depth3=100,
                             n_depth4=60) -> list[dict]:
    """Build controlled entity-state tracking micro-discourses."""
    pairs = []
    pid = 0

    def make_depth2(obj, loc1, loc2, person, verb):
        nonlocal pid
        p1, p2 = loc_prep(loc1), loc_prep(loc2)
        coherent = (f"The {obj} is {p1} the {loc1}. "
                    f"{person} {verb} the {obj} to the {loc2}. "
                    f"The {obj} is now {p2} the {loc2}.")
        perturbed = (f"The {obj} is {p1} the {loc1}. "
                     f"{person} {verb} the {obj} to the {loc2}. "
                     f"The {obj} is now {p1} the {loc1}.")
        # Focus: the final location word (where coherent and perturbed differ)
        coh_span = char_span_of_substring(coherent, loc2, len(coherent) - len(loc2) - 5)
        pert_span = char_span_of_substring(perturbed, loc1, len(perturbed) - len(loc1) - 5)
        if not coh_span or not pert_span:
            return None
        pid += 1
        return {
            "pair_id": f"es_d2_{pid:04d}",
            "family": "entity_state", "sub_type": "depth2_stale",
            "depth": 2, "corpus_source": "controlled_template",
            "coherent_text": coherent, "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_span)],
            "focus_spans_perturbed": [list(pert_span)],
        }

    def make_depth3(obj1, obj2, loc1, loc2, loc3, p1, v1, v2):
        nonlocal pid
        pp1, pp2, pp3 = loc_prep(loc1), loc_prep(loc2), loc_prep(loc3)
        coherent = (f"The {obj1} is {pp1} the {loc1}. The {obj2} is {pp2} the {loc2}. "
                    f"{p1} {v1} the {obj1} to the {loc3}. "
                    f"{p1} {v2} the {obj2} to the {loc1}. "
                    f"The {obj1} is now {pp3} the {loc3}.")
        perturbed_stale = (f"The {obj1} is {pp1} the {loc1}. The {obj2} is {pp2} the {loc2}. "
                           f"{p1} {v1} the {obj1} to the {loc3}. "
                           f"{p1} {v2} the {obj2} to the {loc1}. "
                           f"The {obj1} is now {pp1} the {loc1}.")
        coh_span = char_span_of_substring(coherent, loc3, len(coherent) - len(loc3) - 5)
        pert_span = char_span_of_substring(perturbed_stale, loc1,
                                           len(perturbed_stale) - len(loc1) - 5)
        if not coh_span or not pert_span:
            return None
        pid += 1
        return {
            "pair_id": f"es_d3_{pid:04d}",
            "family": "entity_state", "sub_type": "depth3_stale",
            "depth": 3, "corpus_source": "controlled_template",
            "coherent_text": coherent, "perturbed_text": perturbed_stale,
            "focus_spans_coherent": [list(coh_span)],
            "focus_spans_perturbed": [list(pert_span)],
        }

    def make_depth4(obj1, obj2, loc1, loc2, loc3, loc4, p1, p2, v1, v2, v3):
        nonlocal pid
        pp1, pp2, pp3, pp4 = loc_prep(loc1), loc_prep(loc2), loc_prep(loc3), loc_prep(loc4)
        coherent = (f"The {obj1} is {pp1} the {loc1}. The {obj2} is {pp2} the {loc2}. "
                    f"{p1} {v1} the {obj1} to the {loc3}. "
                    f"{p2} {v2} the {obj2} to the {loc4}. "
                    f"{p1} {v3} the {obj1} to the {loc2}. "
                    f"The {obj1} is now {pp2} the {loc2}.")
        perturbed_stale = (f"The {obj1} is {pp1} the {loc1}. The {obj2} is {pp2} the {loc2}. "
                           f"{p1} {v1} the {obj1} to the {loc3}. "
                           f"{p2} {v2} the {obj2} to the {loc4}. "
                           f"{p1} {v3} the {obj1} to the {loc2}. "
                           f"The {obj1} is now {pp3} the {loc3}.")
        coh_span = char_span_of_substring(coherent, loc2, len(coherent) - len(loc2) - 5)
        pert_span = char_span_of_substring(perturbed_stale, loc3,
                                           len(perturbed_stale) - len(loc3) - 5)
        if not coh_span or not pert_span:
            return None
        pid += 1
        return {
            "pair_id": f"es_d4_{pid:04d}",
            "family": "entity_state", "sub_type": "depth4_stale",
            "depth": 4, "corpus_source": "controlled_template",
            "coherent_text": coherent, "perturbed_text": perturbed_stale,
            "focus_spans_coherent": [list(coh_span)],
            "focus_spans_perturbed": [list(pert_span)],
        }

    used = set()
    for _ in range(n_depth2 * 5):
        if len([p for p in pairs if p["sub_type"] == "depth2_stale"]) >= n_depth2:
            break
        obj = rng.choice(ES_OBJECTS)
        loc1, loc2 = rng.sample(ES_ALL_LOCS, 2)
        person = rng.choice(ES_PERSONS)
        verb = rng.choice(ES_VERBS)
        key = (obj, loc1, loc2, person, verb)
        if key in used:
            continue
        used.add(key)
        p = make_depth2(obj, loc1, loc2, person, verb)
        if p:
            pairs.append(p)

    for _ in range(n_depth3 * 5):
        if len([p for p in pairs if p["sub_type"] == "depth3_stale"]) >= n_depth3:
            break
        obj1, obj2 = rng.sample(ES_OBJECTS, 2)
        loc1, loc2, loc3 = rng.sample(ES_ALL_LOCS, 3)
        p1 = rng.choice(ES_PERSONS)
        v1, v2 = rng.choices(ES_VERBS, k=2)
        key = (obj1, obj2, loc1, loc2, loc3, p1)
        if key in used:
            continue
        used.add(key)
        p = make_depth3(obj1, obj2, loc1, loc2, loc3, p1, v1, v2)
        if p:
            pairs.append(p)

    for _ in range(n_depth4 * 5):
        if len([p for p in pairs if p["sub_type"] == "depth4_stale"]) >= n_depth4:
            break
        obj1, obj2 = rng.sample(ES_OBJECTS, 2)
        loc1, loc2, loc3, loc4 = rng.sample(ES_ALL_LOCS, 4)
        p1, p2 = rng.sample(ES_PERSONS, 2)
        v1, v2, v3 = rng.choices(ES_VERBS, k=3)
        key = (obj1, obj2, loc1, loc2, loc3, loc4, p1, p2)
        if key in used:
            continue
        used.add(key)
        p = make_depth4(obj1, obj2, loc1, loc2, loc3, loc4, p1, p2, v1, v2, v3)
        if p:
            pairs.append(p)

    return pairs


# ---------- Family 2: Temporal/Procedure Order (from corpus) ---------------

_TEMPORAL_RE = re.compile(
    r'^([A-Z][^,;]{10,60}),?\s+(then|and then)\s+([a-z][^,;]{10,60})\.?\s*$'
)
_FIRST_THEN_RE = re.compile(
    r'^[Ff]irst,?\s+([a-z][^,;]{8,55}),?\s+then\s+([a-z][^,;]{8,55})\.?\s*$'
)
_BEFORE_AFTER_RE = re.compile(
    r'^([A-Z][^,;]{8,55})\s+(before|after)\s+([a-z][^,;]{8,55})\.?\s*$'
)

def _clause_ok(c: str) -> bool:
    """Quick quality check: clause must have a verb-like word, no internal quotes."""
    toks = word_tokens(c)
    return (3 <= len(toks) <= 12
            and '"' not in c and "'" not in c
            and not c.startswith("If ") and not c.startswith("if "))

def build_temporal_pairs(rows: list[dict], rng: random.Random, max_pairs=250) -> list[dict]:
    """Extract temporal order pairs from corpus sentences."""
    candidates = []
    for row in rows:
        src = get_source(row)
        for sent in sent_split(row.get("text", "")):
            # "first A, then B" pattern
            m = _FIRST_THEN_RE.match(sent)
            if m:
                a, b = m.group(1).strip(), m.group(2).strip()
                if _clause_ok(a) and _clause_ok(b):
                    candidates.append({
                        "sent": sent, "clause_a": a, "clause_b": b,
                        "connector": "first_then", "source": src,
                        "row_id": row.get("example_id", -1)
                    })
                continue
            # "A, then B" pattern
            m = _TEMPORAL_RE.match(sent)
            if m:
                a, conn, b = m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip()
                if _clause_ok(a) and _clause_ok(b):
                    candidates.append({
                        "sent": sent, "clause_a": a, "clause_b": b,
                        "connector": conn, "source": src,
                        "row_id": row.get("example_id", -1)
                    })
            # "A before/after B" pattern
            m = _BEFORE_AFTER_RE.match(sent)
            if m:
                a, conn, b = m.group(1).strip(), m.group(2).strip().lower(), m.group(3).strip()
                if _clause_ok(a) and _clause_ok(b):
                    candidates.append({
                        "sent": sent, "clause_a": a, "clause_b": b,
                        "connector": conn, "source": src,
                        "row_id": row.get("example_id", -1)
                    })

    rng.shuffle(candidates)
    pairs = []
    seen_texts = set()
    for c in candidates:
        if len(pairs) >= max_pairs:
            break
        a, b, conn = c["clause_a"], c["clause_b"], c["connector"]
        # Perturbation: swap clause order
        if conn == "first_then":
            coherent = f"First, {a}, then {b}."
            perturbed = f"First, {b}, then {a}."
        else:
            coherent = f"{a}, {conn} {b}."
            perturbed = f"{b}, {conn} {a}."
        if coherent in seen_texts:
            continue
        seen_texts.add(coherent)
        # Focus spans: both clauses (the swapped content)
        coh_a_span = char_span_of_substring(coherent, a)
        coh_b_span = char_span_of_substring(coherent, b)
        pert_a_span = char_span_of_substring(perturbed, a)
        pert_b_span = char_span_of_substring(perturbed, b)
        if not all([coh_a_span, coh_b_span, pert_a_span, pert_b_span]):
            continue
        pairs.append({
            "pair_id": f"temp_{len(pairs)+1:04d}",
            "family": "temporal_order",
            "sub_type": f"clause_swap_{conn.replace(' ','_')}",
            "corpus_source": c["source"],
            "source_row_id": c["row_id"],
            "coherent_text": coherent,
            "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_a_span), list(coh_b_span)],
            "focus_spans_perturbed": [list(pert_a_span), list(pert_b_span)],
        })
    return pairs


# ---------- Family 3: Directed Relation / Comparative ----------------------

_COMP_RE = re.compile(
    r'((?:[A-Z]\w+|[Tt]he\s+\w+)(?:\s+\w+){0,3})\s+(?:is|are|was|were)\s+(?:much\s+)?(more|less|fewer)\s+'
    r'(\w+)\s+than\s+((?:[A-Z]\w+|[a-z]\w+)(?:\s+\w+){0,3})(?:\.|,|;)',
    re.IGNORECASE
)
_DIR_FROM_TO_RE = re.compile(
    r'(\bfrom)\s+([\w\s]{2,20}?)\s+(to)\s+([\w\s]{2,20}?)(?:\.|,|;|\s+and)',
    re.IGNORECASE
)

def build_directed_relation_pairs(rows: list[dict], rng: random.Random,
                                  max_comp=150, max_dir=150) -> list[dict]:
    """Extract comparative argument-swap and directed-preposition pairs."""
    comp_cands = []
    dir_cands = []

    for row in rows:
        src = get_source(row)
        for sent in sent_split(row.get("text", "")):
            # Comparative: "X is more ADJ than Y"
            for m in _COMP_RE.finditer(sent):
                x = m.group(1).strip()
                rel_word = m.group(2).strip()
                adj = m.group(3).strip()
                y = m.group(4).strip()
                if (2 <= len(word_tokens(x)) <= 5 and 1 <= len(word_tokens(y)) <= 5
                        and len(x) < 30 and len(y) < 30):
                    comp_cands.append({
                        "sent": sent, "x": x, "y": y, "rel": rel_word,
                        "adj": adj, "source": src,
                        "row_id": row.get("example_id", -1),
                        "match_span": (m.start(), m.end()),
                    })

            # Directed: "from X to Y"
            for m in _DIR_FROM_TO_RE.finditer(sent):
                x = m.group(2).strip()
                y = m.group(4).strip()
                if (1 <= len(word_tokens(x)) <= 4 and 1 <= len(word_tokens(y)) <= 4
                        and len(x) < 25 and len(y) < 25 and x.lower() != y.lower()):
                    dir_cands.append({
                        "sent": sent, "x": x, "y": y, "source": src,
                        "row_id": row.get("example_id", -1),
                        "match_span": (m.start(), m.end()),
                    })

    pairs = []
    seen = set()
    rng.shuffle(comp_cands)
    for c in comp_cands:
        if len([p for p in pairs if p["sub_type"].startswith("comp")]) >= max_comp:
            break
        x, y, rel_w, adj = c["x"], c["y"], c["rel"], c["adj"]
        # Perturbation: swap X and Y
        original_frag = f"{x} is {rel_w} {adj} than {y}"
        swapped_frag = f"{y} is {rel_w} {adj} than {x}"
        # Skip if already processed this fragment
        key = original_frag.lower()
        if key in seen:
            continue
        seen.add(key)
        coherent = f"{original_frag}."
        perturbed = f"{swapped_frag}."
        # Focus: X and Y in both versions
        coh_x = char_span_of_substring(coherent, x)
        coh_y = char_span_of_substring(coherent, y, len(coherent) - len(y) - 3)
        pert_x = char_span_of_substring(perturbed, x, len(perturbed) - len(x) - 3)
        pert_y = char_span_of_substring(perturbed, y)
        if not all([coh_x, coh_y, pert_x, pert_y]):
            continue
        pairs.append({
            "pair_id": f"dr_comp_{len(pairs)+1:04d}",
            "family": "directed_relation",
            "sub_type": "comp_arg_swap",
            "corpus_source": c["source"],
            "source_row_id": c["row_id"],
            "coherent_text": coherent,
            "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_x), list(coh_y)],
            "focus_spans_perturbed": [list(pert_x), list(pert_y)],
        })

    rng.shuffle(dir_cands)
    for c in dir_cands:
        if len([p for p in pairs if p["sub_type"].startswith("dir")]) >= max_dir:
            break
        x, y = c["x"], c["y"]
        # Perturbation: swap from↔to (reverse direction)
        original_frag = f"from {x} to {y}"
        swapped_frag = f"from {y} to {x}"
        key = original_frag.lower()
        if key in seen:
            continue
        seen.add(key)
        coherent = f"It goes {original_frag}."
        perturbed = f"It goes {swapped_frag}."
        coh_x = char_span_of_substring(coherent, x)
        coh_y = char_span_of_substring(coherent, y, len(coherent) - len(y) - 3)
        pert_x = char_span_of_substring(perturbed, x, len(perturbed) - len(x) - 3)
        pert_y = char_span_of_substring(perturbed, y)
        if not all([coh_x, coh_y, pert_x, pert_y]):
            continue
        pairs.append({
            "pair_id": f"dr_dir_{len(pairs)+1:04d}",
            "family": "directed_relation",
            "sub_type": "dir_arg_swap",
            "corpus_source": c["source"],
            "source_row_id": c["row_id"],
            "coherent_text": coherent,
            "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_x), list(coh_y)],
            "focus_spans_perturbed": [list(pert_x), list(pert_y)],
        })

    return pairs


# ---------- Family 4: Belief/Report Role Binding ---------------------------

# Multiple report patterns with broader verb coverage
_REPORT_PATTERNS = [
    # X told/asked/warned Y that P
    re.compile(
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s+'
        r'(told|asked|warned|reminded|informed|promised|convinced|assured|showed)\s+'
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s+'
        r'((?:that|to|about)\s+.{8,80}?)(?:\.|$)',
        re.DOTALL
    ),
    # X said to Y that P / X said to Y, P
    re.compile(
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s+'
        r'(said\s+to|called\s+to|shouted\s+to|whispered\s+to|wrote\s+to)\s+'
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s*[,:]?\s*'
        r'(.{8,80}?)(?:\.|$)',
        re.DOTALL
    ),
    # X gave/sent/brought Y something
    re.compile(
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s+'
        r'(gave|sent|brought|handed|passed|offered|showed|taught|lent)\s+'
        r'(\b[A-Z]\w+(?:\s+[A-Z]\w+)?)\s+'
        r'(\w.{5,60}?)(?:\.|$)',
        re.DOTALL
    ),
]

def build_belief_report_pairs(rows: list[dict], rng: random.Random,
                              max_pairs=180) -> list[dict]:
    """Extract speaker/addressee swap pairs from reported speech and transfer."""
    candidates = []
    for row in rows:
        src = get_source(row)
        for sent in sent_split(row.get("text", "")):
            for pat in _REPORT_PATTERNS:
                for m in pat.finditer(sent):
                    speaker = m.group(1).strip()
                    verb = m.group(2).strip()
                    addressee = m.group(3).strip()
                    complement = m.group(4).strip()
                    if (speaker.lower() != addressee.lower()
                            and len(word_tokens(speaker)) <= 3
                            and len(word_tokens(addressee)) <= 3
                            and len(speaker) >= 2 and len(addressee) >= 2):
                        candidates.append({
                            "speaker": speaker, "verb": verb,
                            "addressee": addressee, "complement": complement,
                            "source": src, "row_id": row.get("example_id", -1),
                        })

    pairs = []
    seen = set()
    rng.shuffle(candidates)
    for c in candidates:
        if len(pairs) >= max_pairs:
            break
        sp, v, ad, comp = c["speaker"], c["verb"], c["addressee"], c["complement"]
        coherent = f"{sp} {v} {ad} {comp}."
        perturbed = f"{ad} {v} {sp} {comp}."
        key = coherent.lower()
        if key in seen:
            continue
        seen.add(key)
        # Focus: speaker and addressee positions in both versions
        coh_sp = char_span_of_substring(coherent, sp)
        coh_ad = char_span_of_substring(coherent, ad, len(sp) + 1)
        pert_sp = char_span_of_substring(perturbed, sp, len(ad) + 1)
        pert_ad = char_span_of_substring(perturbed, ad)
        if not all([coh_sp, coh_ad, pert_sp, pert_ad]):
            continue
        pairs.append({
            "pair_id": f"br_{len(pairs)+1:04d}",
            "family": "belief_report",
            "sub_type": "role_swap",
            "corpus_source": c["source"],
            "source_row_id": c["row_id"],
            "coherent_text": coherent,
            "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_sp), list(coh_ad)],
            "focus_spans_perturbed": [list(pert_sp), list(pert_ad)],
        })
    return pairs


# ---------- Family 5: Polarity-Relation Composition -----------------------

_POLARITY_RE = re.compile(
    r'(\b\w[\w\s]{5,50}?)\s+(not|never|no longer)\s+(\w[\w\s]{5,50}?)(?:\.|$)',
    re.IGNORECASE
)

def build_polarity_pairs(rows: list[dict], rng: random.Random,
                         max_pairs=120) -> list[dict]:
    """Extract polarity flip pairs."""
    candidates = []
    for row in rows:
        src = get_source(row)
        for sent in sent_split(row.get("text", "")):
            for m in _POLARITY_RE.finditer(sent):
                pre = m.group(1).strip()
                neg = m.group(2).strip()
                post = m.group(3).strip()
                if 3 <= len(word_tokens(pre)) <= 12 and 3 <= len(word_tokens(post)) <= 12:
                    candidates.append({
                        "pre": pre, "neg": neg, "post": post,
                        "source": src, "row_id": row.get("example_id", -1),
                    })

    pairs = []
    seen = set()
    rng.shuffle(candidates)
    for c in candidates:
        if len(pairs) >= max_pairs:
            break
        pre, neg, post = c["pre"], c["neg"], c["post"]
        coherent = f"{pre} {neg} {post}."
        perturbed = f"{pre} {post}."  # Remove negation
        key = coherent.lower()
        if key in seen:
            continue
        seen.add(key)
        # Focus: the negation word position (present in coherent, absent in perturbed)
        # and surrounding context
        coh_neg_span = char_span_of_substring(coherent, neg, len(pre))
        coh_post_span = char_span_of_substring(coherent, post, len(pre) + len(neg) + 1)
        pert_post_span = char_span_of_substring(perturbed, post, len(pre))
        if not all([coh_neg_span, coh_post_span, pert_post_span]):
            continue
        pairs.append({
            "pair_id": f"pol_{len(pairs)+1:04d}",
            "family": "polarity_relation",
            "sub_type": "negation_removal",
            "corpus_source": c["source"],
            "source_row_id": c["row_id"],
            "coherent_text": coherent,
            "perturbed_text": perturbed,
            "focus_spans_coherent": [list(coh_neg_span), list(coh_post_span)],
            "focus_spans_perturbed": [list(pert_post_span)],
        })
    return pairs


# ---------- Surface Controls -----------------------------------------------

def build_surface_controls(all_pairs: list[dict], rng: random.Random,
                           n_per_family=40) -> list[dict]:
    """Build surface-only perturbation controls by random content-word replacement.

    For each structural family, take clean pairs and replace a non-focus content
    word with a random corpus content word. This measures surface prediction
    sensitivity without structural binding.
    """
    # Collect a vocabulary of content words from existing pairs
    vocab = set()
    for p in all_pairs:
        for w in word_tokens(p["coherent_text"]):
            if len(w) >= 4 and w[0].islower():
                vocab.add(w.lower())
    vocab = sorted(vocab)
    if len(vocab) < 50:
        return []

    controls = []
    by_fam = defaultdict(list)
    for p in all_pairs:
        by_fam[p["family"]].append(p)

    for fam, fam_pairs in by_fam.items():
        rng.shuffle(fam_pairs)
        for p in fam_pairs[:n_per_family * 3]:
            if len([c for c in controls if c.get("control_family") == fam]) >= n_per_family:
                break
            text = p["coherent_text"]
            words_in_text = word_tokens(text)
            # Find a non-focus content word to replace
            focus_chars = set()
            for sp in p.get("focus_spans_coherent", []):
                focus_chars.update(range(sp[0], sp[1]))
            replaceable = []
            for m in re.finditer(r'\b(\w{4,})\b', text):
                if m.group(1)[0].islower() and not any(
                        c in focus_chars for c in range(m.start(), m.end())):
                    replaceable.append(m)
            if not replaceable:
                continue
            target_m = rng.choice(replaceable)
            replacement = rng.choice(vocab)
            while replacement == target_m.group(1).lower():
                replacement = rng.choice(vocab)
            perturbed = text[:target_m.start()] + replacement + text[target_m.end():]
            # Focus: the replaced word position
            coh_span = [target_m.start(), target_m.end()]
            pert_span = [target_m.start(), target_m.start() + len(replacement)]
            controls.append({
                "pair_id": f"surf_{fam[:4]}_{len(controls)+1:04d}",
                "family": "surface_control",
                "sub_type": f"word_replace_{fam}",
                "control_family": fam,
                "corpus_source": p.get("corpus_source", "controlled_template"),
                "coherent_text": text,
                "perturbed_text": perturbed,
                "focus_spans_coherent": [coh_span],
                "focus_spans_perturbed": [pert_span],
            })
    return controls


# ---------- Main -----------------------------------------------------------

def main():
    t0 = time.time()
    rng = random.Random(SEED)
    print(json.dumps({"event": "load_pool", "path": str(POOL)}), flush=True)
    rows = []
    with open(POOL) as f:
        for line in f:
            rows.append(json.loads(line))
    print(json.dumps({"event": "pool_loaded", "rows": len(rows)}), flush=True)

    # Pool hash for provenance
    with open(POOL, "rb") as f:
        pool_sha = hashlib.sha256(f.read()).hexdigest()

    # Build all families
    es_pairs = build_entity_state_pairs(rng)
    print(json.dumps({"event": "entity_state_done", "count": len(es_pairs)}), flush=True)

    temp_pairs = build_temporal_pairs(rows, rng)
    print(json.dumps({"event": "temporal_done", "count": len(temp_pairs)}), flush=True)

    dr_pairs = build_directed_relation_pairs(rows, rng)
    print(json.dumps({"event": "directed_relation_done", "count": len(dr_pairs)}), flush=True)

    br_pairs = build_belief_report_pairs(rows, rng)
    print(json.dumps({"event": "belief_report_done", "count": len(br_pairs)}), flush=True)

    pol_pairs = build_polarity_pairs(rows, rng)
    print(json.dumps({"event": "polarity_done", "count": len(pol_pairs)}), flush=True)

    structural_pairs = es_pairs + temp_pairs + dr_pairs + br_pairs + pol_pairs
    surface_pairs = build_surface_controls(structural_pairs, rng)
    print(json.dumps({"event": "surface_controls_done", "count": len(surface_pairs)}), flush=True)

    all_pairs = structural_pairs + surface_pairs

    # Family counts
    fam_counts = defaultdict(int)
    for p in all_pairs:
        fam_counts[p["family"]] += 1
    # Sub-type counts
    sub_counts = defaultdict(int)
    for p in all_pairs:
        sub_counts[f"{p['family']}/{p['sub_type']}"] += 1
    # Source balance
    src_counts = defaultdict(lambda: defaultdict(int))
    for p in all_pairs:
        src_counts[p["family"]][p.get("corpus_source", "unknown")] += 1

    # Content hash
    content_str = json.dumps([p["coherent_text"] + "|" + p["perturbed_text"]
                              for p in all_pairs], sort_keys=True)
    content_sha = hashlib.sha256(content_str.encode()).hexdigest()

    result = {
        "probe_id": "structural_contrast_margin_v1",
        "created_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "seed": SEED,
        "pool_path": str(POOL),
        "pool_sha256": pool_sha,
        "content_sha256": content_sha,
        "total_pairs": len(all_pairs),
        "structural_pairs": len(structural_pairs),
        "surface_control_pairs": len(surface_pairs),
        "family_counts": dict(fam_counts),
        "sub_type_counts": dict(sub_counts),
        "source_balance": {k: dict(v) for k, v in src_counts.items()},
        "scoring_weights": {
            "entity_state": 0.30,
            "temporal_order": 0.25,
            "directed_relation": 0.20,
            "belief_report": 0.15,
            "polarity_relation": 0.10,
        },
        "pairs": all_pairs,
    }

    out_json = OUT / "structural_contrast_probe.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary
    md = [
        "# research structural contrast-margin probe",
        "",
        f"Created: {result['created_utc']}",
        f"Seed: {SEED}",
        f"Pool SHA256: `{pool_sha[:16]}...`",
        f"Content SHA256: `{content_sha[:16]}...`",
        f"Total pairs: {len(all_pairs)} ({len(structural_pairs)} structural + {len(surface_pairs)} surface controls)",
        "",
        "## Family counts",
        "",
        "| Family | Count | Plan weight |",
        "|---|---:|---:|",
    ]
    weights = result["scoring_weights"]
    for fam in sorted(fam_counts):
        w = weights.get(fam, 0.0)
        md.append(f"| {fam} | {fam_counts[fam]} | {w:.2f} |")
    md.append(f"| surface_control | {len(surface_pairs)} | -0.25 (L_t) |")
    md.append("")
    md.append("## Sub-type counts")
    md.append("")
    for k in sorted(sub_counts):
        md.append(f"- {k}: {sub_counts[k]}")
    md.append("")
    md.append("## Source balance")
    md.append("")
    for fam in sorted(src_counts):
        md.append(f"### {fam}")
        for src, cnt in sorted(src_counts[fam].items(), key=lambda x: -x[1]):
            md.append(f"  - {src}: {cnt}")
    md.append("")
    md.append("## Sample pairs")
    md.append("")
    for fam in ["entity_state", "temporal_order", "directed_relation",
                "belief_report", "polarity_relation", "surface_control"]:
        fam_p = [p for p in all_pairs if p["family"] == fam]
        if not fam_p:
            continue
        md.append(f"### {fam}")
        for p in fam_p[:3]:
            md.append(f"  - **{p['pair_id']}** ({p['sub_type']})")
            md.append(f"    Coherent: {p['coherent_text'][:120]}")
            md.append(f"    Perturbed: {p['perturbed_text'][:120]}")
        md.append("")

    out_md = (OUT.parents[4] / 'research/documents/frontier_consolidation/data/structural_contrast_probe/structural_contrast_probe.md')
    out_md.write_text("\n".join(md), encoding="utf-8")

    elapsed = round(time.time() - t0, 2)
    summary = {
        "status": "PROBE_BUILT",
        "out_json": str(out_json),
        "out_md": str(out_md),
        "total_pairs": len(all_pairs),
        "family_counts": dict(fam_counts),
        "content_sha256": content_sha,
        "elapsed_sec": elapsed,
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
