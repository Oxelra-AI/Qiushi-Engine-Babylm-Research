#!/usr/bin/env python3
"""research: row-aware exact-length source-attested bridge pilot.

Purpose: before any 100M BabyLM training, test whether the source-attested
bridge can be constructed as a CLEAN matched-subset object rather than a
compact/bridge blend.

The pilot selects whole row units from the compact changed block (all 2-pair
rows plus a small deterministic set of 3-pair rows), generates exact-length
source-attested candidates for every pair in those rows, validates proposition
and surface constraints heuristically, classifies structural transformation vs
extraction, and reports:
  - exact-length transformation-like pair yield
  - whole original row fillability
  - estimated full-pool changed-block word fraction if yield scaled
  - whether a later full-pool generation or 100M training run is justified

CPU phases: prepare/analyze. GPU generation is external via the printed command.
No BabyLM training/evaluation/upload/submission occurs here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

# ---------------- paths ----------------

def find_user_root() -> Path:
    return _PUBLIC_ROOT

ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PAIR_PATH = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
ROW_META = WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_changed_block_rows_meta.jsonl"
OUT_DIR = WS / "data/row_aware_exact_bridge_pilot"
TOKENIZER_PATH = WS / "data/compliant_tokenizer"
CHANGED_MIN = 950000
CHANGED_MAX = 953004
EXPECTED_WORDS_10M = 10_000_000

# ---------------- text helpers ----------------
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019\-\.][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|[−\-]?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
WORD_SIMPLE_RE = re.compile(r"[a-z0-9][a-z0-9\-\.']*")
NUM_RE = re.compile(r"(?:[£$€]\s*)?[−\-]?\d+(?:[.,:/\-]\d+)*(?:\s*(?:%|percent|°\s*[CF]|degrees?\s*[CF]|/\s*°?\s*[CF]|per\s+cent|st|nd|rd|th))?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'\-]+(?:\s+|$)){1,8}")
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","so","because","as","than","to","of","in","on","for","with","without","by","from","at","into","onto","over","under","about","between","among","through","during","before","after","above","below","is","am","are","was","were","be","been","being","do","does","did","done","doing","have","has","had","having","can","could","may","might","must","shall","should","will","would","this","that","these","those","there","here","it","its","they","them","their","theirs","he","him","his","she","her","hers","we","us","our","ours","you","your","yours","i","me","my","mine","who","whom","whose","which","what","where","when","why","how","not","no","nor","only","just","also","very","more","most","less","least","much","many","some","any","all","each","every","other","another","such","own","same","too","again","still","already","yet","up","down","out","off","back","away","within","across","per","via","using","used","use","uses","become","became","becomes","based","while","although","however","therefore","both","either","neither","one","two","three","four","five","six","seven","eight","nine","ten",
}
FINITE_AUX = {"is","are","was","were","am","has","have","had","do","does","did","can","could","will","would","shall","should","may","might","must"}
RELATION_WORDS = {"because","caused","causes","cause","led","leads","leading","result","results","resulting","due","based","between","among","during","before","after","while","when","created","creates","using","uses","used","show","shows","showed","found","finds","compared","than","through","include","includes","made","help","helps","against","became","become","becomes","without","improved","increase","decrease","increasing","decreasing","allowed","allows","prevent","prevents"}
POLARITY_WORDS = {"only","not","never","neither","nor","no","none","nothing","cannot","can't","won't","don't","doesn't","didn't","isn't","aren't","wasn't","weren't","hasn't","haven't","hadn't"}
ENTITY_STOP = {"The","A","An","This","That","These","Those","In","On","For","At","By","From","To","And","But","Or","If","When","While","Because","Source","Sentence","Output","Example"}
BAD_PATTERNS = [re.compile(p, re.I) for p in [r"^\s*(sure|here('| i)s|certainly|of course)\b", r"as an ai", r"please provide", r"output only", r"source sentence", r"allowed content", r"target length", r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]"]]
GRAMMAR_PATTERNS = [
    ("bare_fragment_colon", re.compile(r"^\s*[A-Za-z]+(?:\s+[A-Za-z]+){0,5}\s*:\s*")),
    ("missing_article_from_tray", re.compile(r"\bfrom\s+tray\b", re.I)),
    ("bare_passive_no_aux", re.compile(r"\b(?:oxygen|blood|data|information|water|air)\s+(?:carried|transported|measured|collected|processed)\s+by\b", re.I)),
    ("dangling_participle_end", re.compile(r"\b(?:found|shown|known|seen|made|caused|allowed|considered|created|designed|built)\s*[.!?]?\s*$", re.I)),
]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_words(text: str) -> list[str]:
    return [w for w in str(text or "").split() if w]


def wc(text: str) -> int:
    return len(split_words(text))


def lex_tokens(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(str(text or ""))]


def norm(tok: str) -> str:
    return "".join(re.findall(r"[a-z0-9]", str(tok or "").lower()))


def stem(n: str) -> str:
    s = norm(n)
    if not s:
        return s
    if s.endswith("ies") and len(s) > 5:
        return s[:-3] + "y"
    for suf in ("ingly", "edly"):
        if s.endswith(suf) and len(s) > len(suf) + 3:
            return s[:-len(suf)]
    for suf in ("ing", "ers", "er", "ed"):
        if s.endswith(suf) and len(s) > len(suf) + 3:
            base = s[:-len(suf)]
            if len(base) > 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    if s.endswith("es") and len(s) > 4:
        return s[:-2]
    if s.endswith("s") and len(s) > 4 and not s.endswith("ss"):
        return s[:-1]
    return s


def is_content_norm(n: str) -> bool:
    return bool(n) and (n.isdigit() or (n not in STOPWORDS and len(n) >= 4))


def content_norms(text: str) -> list[str]:
    out = []
    for tok in lex_tokens(text):
        n = norm(tok)
        if is_content_norm(n) or NUM_RE.fullmatch(tok):
            out.append(n)
    return out


def content_stems(text: str) -> list[str]:
    return [stem(n) for n in content_norms(text)]


def signed_number_surfaces(text: str) -> list[str]:
    vals = []
    for m in NUM_RE.finditer(str(text or "")):
        x = m.group(0).lower().replace("−", "-").strip()
        x = re.sub(r"\s+", "", x).replace("percent", "%").replace("degrees", "°").replace("degree", "°")
        x = x.strip(".,;:!?()[]{}\"'")
        if x:
            vals.append(x)
    return vals


def capital_entities(text: str) -> set[str]:
    ents = set()
    for m in CAP_SEQ_RE.finditer(str(text or "")):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e or e in ENTITY_STOP:
            continue
        ents.add(" ".join(re.findall(r"[a-z0-9]+", e.lower())))
    return ents


def multiset_missing(required: list[str], observed: list[str]) -> list[str]:
    c = collections.Counter(observed)
    miss = []
    for x in required:
        if c[x] > 0:
            c[x] -= 1
        else:
            miss.append(x)
    return miss


def relation_count(text: str) -> int:
    toks = {norm(x) for x in lex_tokens(text)}
    return sum(1 for x in toks if x in RELATION_WORDS)


def source_content_list(source: str, max_items: int = 70) -> list[str]:
    seen = set(); out = []
    for tok in lex_tokens(source):
        n = norm(tok)
        if not n:
            continue
        if not (is_content_norm(n) or NUM_RE.fullmatch(tok)):
            continue
        st = stem(n)
        if st in seen:
            continue
        seen.add(st)
        out.append(tok.strip(".,;:!?()[]{}\"'"))
        if len(out) >= max_items:
            break
    return out


def has_finite_verb(text: str) -> bool:
    words = [norm(w) for w in split_words(text)]
    if any(w in FINITE_AUX for w in words):
        return True
    for i, w in enumerate(words):
        if i == 0:
            continue
        if w.endswith("ed") and len(w) > 3:
            return True
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss") and w not in STOPWORDS:
            return True
    return False


def clean_output(raw: str) -> str:
    s = re.sub(r"\s+", " ", str(raw or "").strip())
    s = re.sub(r"^(sentence|answer|output)\s*:\s*", "", s, flags=re.I)
    return s.strip(" \t\r\n\"'")


def longest_contiguous_run(gen: list[str], src: list[str]) -> int:
    n, m = len(gen), len(src)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        gi = gen[i - 1]
        for j in range(1, m + 1):
            if gi == src[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ai in a:
        cur = [0] * (len(b) + 1)
        for j, bj in enumerate(b, 1):
            if ai == bj:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def order_agreement(gen_c: list[str], src_c: list[str]) -> float:
    if not gen_c:
        return 1.0
    src_set = set(src_c)
    shared = [w for w in gen_c if w in src_set]
    if not shared:
        return 0.0
    return lcs_len(shared, src_c) / len(shared)


def is_verbatim_substring(gen: list[str], src: list[str]) -> bool:
    if not gen or len(gen) > len(src):
        return False
    return any(src[i:i+len(gen)] == gen for i in range(0, len(src)-len(gen)+1))


def is_source_suffix(gen: list[str], src: list[str]) -> bool:
    return bool(gen) and len(gen) <= len(src) and src[len(src)-len(gen):] == gen


def stopword_delta(gen: list[str], src: list[str]) -> float:
    gs = collections.Counter(w for w in gen if w in STOPWORDS)
    ss = collections.Counter(w for w in src if w in STOPWORDS)
    keys = set(gs) | set(ss)
    if not keys:
        return 0.0
    return sum(abs(gs[k] - ss[k]) for k in keys) / max(1, sum(ss.values()))


def classify_structure(source: str, generated: str) -> dict[str, Any]:
    gen = re.findall(r"[a-z0-9][a-z0-9\-\.']*", generated.lower())
    src = re.findall(r"[a-z0-9][a-z0-9\-\.']*", source.lower())
    gen_c = [w for w in gen if w not in STOPWORDS and len(w) >= 2]
    src_c = [w for w in src if w not in STOPWORDS and len(w) >= 2]
    run = longest_contiguous_run(gen, src)
    long_frac = run / max(1, len(gen))
    ord_agree = order_agreement(gen_c, src_c)
    novel = [w for w in gen if w not in set(src)]
    novel_frac = len(novel) / max(1, len(gen))
    lead = is_source_suffix(gen, src)
    verbatim = is_verbatim_substring(gen, src)
    func_delta = stopword_delta(gen, src)
    if verbatim:
        edit_class = "verbatim_substring"
    elif lead:
        edit_class = "prefix_trim_only"
    elif ord_agree >= 0.9 and long_frac >= 0.5 and novel_frac <= 0.12:
        edit_class = "deletion_reorder"
    elif ord_agree >= 0.75 and novel_frac <= 0.2:
        edit_class = "light_restructure"
    else:
        edit_class = "substantive_restructure"
    return {
        "edit_class": edit_class,
        "longest_run_frac": round(long_frac, 4),
        "order_agreement": round(ord_agree, 4),
        "reordering_index": round(1.0 - ord_agree, 4),
        "novel_word_fraction": round(novel_frac, 4),
        "function_word_edit_rate": round(func_delta, 4),
        "is_verbatim_source_substring": verbatim,
        "leading_deletion_only": lead,
    }


def monotone_align(src_norms: list[str], view_norms: list[str]) -> list[int | None]:
    pos_by_norm: dict[str, list[int]] = collections.defaultdict(list)
    for i, n in enumerate(src_norms):
        if n:
            pos_by_norm[n].append(i)
    cursors: dict[str, int] = collections.defaultdict(int)
    last = -1
    aligned = []
    for n in view_norms:
        if not n or n not in pos_by_norm:
            aligned.append(None); continue
        poss = pos_by_norm[n]
        chosen = None
        for j in range(cursors[n], len(poss)):
            if poss[j] > last:
                chosen = poss[j]
                cursors[n] = j + 1
                break
        if chosen is None:
            aligned.append(None)
        else:
            aligned.append(chosen); last = chosen
    return aligned


def atlas_geometry(source: str, view: str) -> dict[str, Any]:
    src_w = split_words(source); view_w = split_words(view)
    src_n = [norm(w) for w in src_w]; view_n = [norm(w) for w in view_w]
    aligned = monotone_align(src_n, view_n)
    adjacent = gap1 = skip = 0
    pos = [a for a in aligned if a is not None]
    for a, b in zip(aligned, aligned[1:]):
        if a is None or b is None:
            continue
        adjacent += 1
        if b == a + 1: gap1 += 1
        else: skip += 1
    cwords = [n for n in view_n if is_content_norm(n)]
    src_bag = collections.Counter(n for n in src_n if is_content_norm(n))
    rem = dict(src_bag)
    absent = 0
    for n in cwords:
        if rem.get(n, 0) > 0: rem[n] -= 1
        else: absent += 1
    return {
        "source_words": len(src_w), "view_words": len(view_w),
        "compression": len(view_w)/max(1, len(src_w)),
        "adjacent_aligned_pairs": adjacent, "gap1_adjacent_pairs": gap1, "skip_adjacent_pairs": skip,
        "gap1_frac": gap1/adjacent if adjacent else None,
        "skip_frac": skip/adjacent if adjacent else None,
        "source_span": (max(pos)-min(pos)+1)/len(src_w) if pos and src_w else None,
        "view_content_words": len(cwords), "source_absent_content_words": absent,
        "source_absent_content_frac": absent/max(1, len(cwords)),
        "aligned_view_fraction": sum(a is not None for a in aligned)/max(1, len(view_w)),
    }


def validate_candidate(prompt: dict[str, Any], generated: str, tokenizer=None) -> dict[str, Any]:
    src = prompt["source_text"]
    nat = prompt["natural_compact_text"]
    target_words = int(prompt["target_words"])
    reasons: list[str] = []
    if not generated:
        reasons.append("empty")
    if any(rx.search(generated) for rx in BAD_PATTERNS):
        reasons.append("bad_pattern")
    if generated and generated[-1] not in ".!?\"'":
        reasons.append("bad_end")
    if wc(generated) != target_words:
        reasons.append("not_exact_word_count")
    if generated and not has_finite_verb(generated):
        reasons.append("no_finite_verb")
    grammar_hits = [name for name, rx in GRAMMAR_PATTERNS if rx.search(generated)]
    if grammar_hits:
        reasons.append("grammar_pattern")

    # Source-attested content closure by stems.
    src_stems = collections.Counter(content_stems(src))
    unsupported = []
    rem = dict(src_stems)
    for n in content_norms(generated):
        st = stem(n)
        if rem.get(st, 0) > 0:
            rem[st] -= 1
        elif st not in src_stems:
            unsupported.append(n)
    if unsupported:
        reasons.append("unsupported_content_lemma")

    # Numbers: generated numbers must be source-attested; numbers present in natural compact must remain.
    src_nums = signed_number_surfaces(src)
    nat_nums = signed_number_surfaces(nat)
    gen_nums = signed_number_surfaces(generated)
    missing_nat_nums = multiset_missing(nat_nums, gen_nums)
    extra_gen_nums = [x for x in gen_nums if x not in collections.Counter(src_nums)]
    # More robust: multiset extra relative to source.
    extra_gen_nums = multiset_missing(gen_nums, src_nums)
    if missing_nat_nums:
        reasons.append("missing_natural_number_surface")
    if extra_gen_nums:
        reasons.append("extra_number_surface")

    # Polarity: preserve polarity if it is present in natural compact, or if compact-source both indicate it.
    src_low, nat_low, gen_low = f" {src.lower()} ", f" {nat.lower()} ", f" {generated.lower()} "
    for pw in POLARITY_WORDS:
        if f" {pw} " in nat_low and f" {pw} " not in gen_low:
            reasons.append(f"dropped_natural_polarity_{pw}"); break
    # Do not introduce negation absent from source.
    for pw in ["not", "never", "only", "no"]:
        if f" {pw} " in gen_low and f" {pw} " not in src_low:
            reasons.append(f"extra_polarity_{pw}"); break

    # Named entities: generated capital sequences should be compatible with source.
    src_caps = capital_entities(src)
    gen_caps = capital_entities(generated)
    new_caps = [e for e in gen_caps if e not in src_caps and not any(e in s or s in e for s in src_caps)]
    if len(new_caps) > 1:
        reasons.append("new_entity_like")

    # Relation proxy: if natural compact retains a relation word, generated should too; otherwise source relation is enough only if generator retained central shared content.
    nat_rel = relation_count(nat)
    if nat_rel > 0 and relation_count(generated) == 0:
        reasons.append("lost_natural_relation_proxy")

    # Natural-content overlap after excluding natural source-absent terms that cannot be produced under source-attested constraint.
    src_stem_set = set(content_stems(src))
    nat_shared = [st for st in content_stems(nat) if st in src_stem_set]
    gen_stem_set = set(content_stems(generated))
    nat_overlap = len(set(nat_shared) & gen_stem_set) / max(1, len(set(nat_shared))) if nat_shared else 1.0
    if nat_overlap < 0.55:
        reasons.append("low_natural_shared_content_overlap")

    struct = classify_structure(src, generated) if generated else {"edit_class": "empty"}
    transform_like = struct.get("edit_class") in {"light_restructure", "substantive_restructure"}
    if not transform_like:
        reasons.append("not_transformation_like")

    geom = atlas_geometry(src, generated) if generated else {}
    return {
        "accept_exact_transform": len(reasons) == 0,
        "reasons": reasons,
        "unsupported_lemmas": unsupported[:8],
        "missing_natural_number_surfaces": missing_nat_nums,
        "extra_number_surfaces": extra_gen_nums,
        "grammar_hits": grammar_hits,
        "generated_words": wc(generated),
        "target_words": target_words,
        "word_delta": wc(generated) - target_words,
        "natural_shared_content_overlap": round(nat_overlap, 4),
        "structure": struct,
        "geometry": geom,
    }

# ---------------- prompts ----------------
FEWSHOT = """Examples of the required transformation style:
SOURCE: In the 1920s, most of Europe was bankrupt due to after effects of WWI.
TARGET WORDS: 10
GOOD OUTPUT: Most of Europe was bankrupt in the 1920s due to WWI effects.

SOURCE: The frog was very cooperative, allowing for close-up photographs with a 50mm lens coupled with a set of macro tubes.
TARGET WORDS: 12
GOOD OUTPUT: The cooperative frog allowed close-up photographs with a 50mm lens coupled with macro tubes.

SOURCE: Furthermore, a natural balance of microorganisms in the body is upset by excess mercury in the gastrointestinal tract, and this imbalance may lead to candida.
TARGET WORDS: 10
GOOD OUTPUT: Excess mercury upsets the body's natural microorganism balance, leading to candida.
"""

BASE_RULES = """You are making legal BabyLM training text. Write exactly ONE complete fluent English sentence.

Hard rules:
1. Use EXACTLY {target_words} whitespace-separated words.
2. Every CONTENT word in the output must be a lemma, inflection, number, or name already present in the SOURCE. Do not use outside synonyms or new factual terms.
3. Function words and punctuation are allowed freely.
4. Keep the main proposition: who/what did/is what, with central cause/time/comparison/result if present.
5. Keep finite grammar: subject + finite verb + needed objects/complements. No bare keyword lists.
6. Preserve numbers/names/polarity from the NATURAL COMPACT reference when they appear there; do not invent numbers, names, or polarity.
7. Do not copy a contiguous source substring. Reorder or recast the clause structure while staying faithful.
8. Output only the sentence, no explanation.
"""

REGIME_INSTRUCTIONS = {
    "exact_predicate_front": "Prefer moving the main predicate early and removing framing phrases while preserving arguments.",
    "exact_clause_recast": "Prefer changing passive/active voice or clause order using only source-attested content words.",
    "exact_anchor_gapfill": "Use the anchor content words as ingredients, but add function words to make a new complete sentence rather than a list.",
    "exact_compact_shadow": "Imitate the NATURAL COMPACT's information choice and length, but replace source-absent content words with source-attested alternatives.",
}


def anchor_words(source: str, natural: str, max_items: int = 16) -> str:
    src_tokens = lex_tokens(source)
    nat_stems = set(content_stems(natural))
    chosen = []
    seen = set()
    for tok in src_tokens:
        st = stem(norm(tok))
        if st in nat_stems and st not in seen:
            chosen.append(tok.strip(".,;:!?()[]{}\"'")); seen.add(st)
    if len(chosen) < 5:
        for tok in source_content_list(source, max_items=max_items):
            st = stem(norm(tok))
            if st not in seen:
                chosen.append(tok); seen.add(st)
            if len(chosen) >= max_items:
                break
    return ", ".join(chosen[:max_items])


def make_prompt(row: dict[str, Any], regime: str) -> str:
    src = row["source_text"]
    nat = row["view_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    anchors = anchor_words(src, nat)
    return (
        FEWSHOT + "\n" + BASE_RULES.format(target_words=target) + "\n" +
        f"Specific instruction: {REGIME_INSTRUCTIONS[regime]}\n\n" +
        f"SOURCE:\n{src}\n\n" +
        f"NATURAL COMPACT reference (do not copy source-absent words; use it only for proposition/length):\n{nat}\n\n" +
        f"TARGET WORDS: {target}\n" +
        f"SOURCE-ATTESTED CONTENT WORDS:\n{allowed}\n\n" +
        f"ANCHOR CONTENT WORDS:\n{anchors}\n\n" +
        "OUTPUT:"
    )

# ---------------- selection ----------------
def compact_absent_count(source: str, compact: str) -> int:
    src = collections.Counter(content_stems(source))
    absent = 0
    for st in content_stems(compact):
        if src.get(st, 0) > 0: src[st] -= 1
        else: absent += 1
    return absent


def prepare(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(PAIR_PATH)
    pair_by_id = {p["pair_id"]: p for p in pairs}
    rows = [r for r in read_jsonl(ROW_META) if CHANGED_MIN <= int(r.get("example_id", -1)) <= CHANGED_MAX and r.get("pair_ids")]
    two_rows = [r for r in rows if len(r.get("pair_ids", [])) == 2]
    three_rows = [r for r in rows if len(r.get("pair_ids", [])) == 3]
    rng = random.Random(args.seed)
    # All two-pair rows are scarce and maximally favorable for row fill; sample a modest set of three-pair rows for scale realism.
    selected_two = list(two_rows)
    rng.shuffle(selected_two)
    selected_two = selected_two[:min(args.two_rows, len(selected_two))]
    selected_three = list(three_rows)
    rng.shuffle(selected_three)
    selected_three = selected_three[:min(args.three_rows, len(selected_three))]
    selected_rows = selected_two + selected_three
    selected_pair_ids = []
    for r in selected_rows:
        for pid in r["pair_ids"]:
            if pid not in selected_pair_ids:
                selected_pair_ids.append(pid)
    enriched_pairs = []
    for pid in selected_pair_ids:
        p = dict(pair_by_id[pid])
        p["source_text"] = " ".join(str(p["source_text"]).split())
        p["view_text"] = " ".join(str(p.get("view_text") or p.get("original_compact_text") or "").split())
        p["source_words"] = wc(p["source_text"])
        p["view_words"] = wc(p["view_text"])
        p["compact_absent_content_lemma_count"] = compact_absent_count(p["source_text"], p["view_text"])
        p["relation_count_proxy"] = relation_count(p["source_text"])
        enriched_pairs.append(p)
    regimes = list(REGIME_INSTRUCTIONS)
    prompts = []
    for p in enriched_pairs:
        for regime in regimes:
            prompts.append({
                "id": f"step239_{regime}_{p['pair_id']}",
                "prompt_id": f"step239_{regime}_{p['pair_id']}",
                "regime": regime,
                "pair_id": p["pair_id"],
                "source_text": p["source_text"],
                "natural_compact_text": p["view_text"],
                "target_words": p["view_words"],
                "source_words": p["source_words"],
                "compact_absent_content_lemma_count": p["compact_absent_content_lemma_count"],
                "relation_count_proxy": p["relation_count_proxy"],
                "prompt": make_prompt(p, regime),
            })
    # deterministic split into two shards.
    shard0 = prompts[0::2]
    shard1 = prompts[1::2]
    write_jsonl(OUT_DIR / "selected_rows.jsonl", selected_rows)
    write_jsonl(OUT_DIR / "selected_pairs.jsonl", enriched_pairs)
    write_jsonl(OUT_DIR / "prompts_all.jsonl", prompts)
    write_jsonl(OUT_DIR / "prompts_gpu0.jsonl", shard0)
    write_jsonl(OUT_DIR / "prompts_gpu1.jsonl", shard1)
    changed_words = sum(int(r["words"]) for r in rows)
    manifest = {
        "status": "ROW_AWARE_EXACT_BRIDGE_PROMPTS_PREPARED",
        "created_utc": now(),
        "seed": args.seed,
        "selected_rows": len(selected_rows),
        "selected_two_pair_rows": len(selected_two),
        "selected_three_pair_rows": len(selected_three),
        "selected_pairs": len(enriched_pairs),
        "prompts": len(prompts),
        "regimes": regimes,
        "changed_rows_total": len(rows),
        "changed_words_per_10m": changed_words,
        "row_pair_count_distribution_total": dict(collections.Counter(len(r.get("pair_ids", [])) for r in rows)),
        "selected_rows_path": str(OUT_DIR / "selected_rows.jsonl"),
        "selected_pairs_path": str(OUT_DIR / "selected_pairs.jsonl"),
        "prompts_gpu0": str(OUT_DIR / "prompts_gpu0.jsonl"),
        "prompts_gpu1": str(OUT_DIR / "prompts_gpu1.jsonl"),
        "commands": {
            "gpu0": f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {OUT_DIR / 'prompts_gpu0.jsonl'} --output-jsonl {OUT_DIR / 'gen_outputs_gpu0.jsonl'} --batch-size 32 --max-new-tokens 80 --temperature {args.temperature} --device cuda",
            "gpu1": f"CUDA_VISIBLE_DEVICES=1 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {OUT_DIR / 'prompts_gpu1.jsonl'} --output-jsonl {OUT_DIR / 'gen_outputs_gpu1.jsonl'} --batch-size 32 --max-new-tokens 80 --temperature {args.temperature} --device cuda",
        },
        "scientific_decision": {
            "continue_full_pool_generation_if": "exact-length transformation-like pair yield is high enough to plausibly fill at least ~60% of changed-block words in a repacked matched subset, and direct review of accepted outputs does not reveal systematic proposition/grammar damage.",
            "stop_or_redesign_if": "yield remains near research (~20-30%) or whole/pair word mass is too small; do not launch 100M training from a compact/bridge blend.",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def read_outputs_for_shard(prompt_path: Path, output_path: Path) -> list[tuple[dict[str, Any], str]]:
    prompts = read_jsonl(prompt_path)
    outs = read_jsonl(output_path)
    rows = []
    for i, p in enumerate(prompts):
        raw = ""
        if i < len(outs):
            o = outs[i]
            raw = str(o.get("output") or o.get("generated_text") or o.get("text") or o.get("completion") or "")
        rows.append((p, raw))
    return rows


def stat(vals: list[float | int | None]) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0}
    xs.sort()
    def q(p: float) -> float:
        if len(xs) == 1: return xs[0]
        z = p*(len(xs)-1); lo = math.floor(z); hi = math.ceil(z)
        return xs[lo] if lo == hi else xs[lo]*(hi-z) + xs[hi]*(z-lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": min(xs), "max": max(xs)}


def summarize_geom(records: list[dict[str, Any]]) -> dict[str, Any]:
    geoms = [r["geometry"] for r in records if r.get("geometry")]
    totals = {
        "adjacent_aligned_pairs": sum(int(g.get("adjacent_aligned_pairs", 0)) for g in geoms),
        "gap1_adjacent_pairs": sum(int(g.get("gap1_adjacent_pairs", 0)) for g in geoms),
        "skip_adjacent_pairs": sum(int(g.get("skip_adjacent_pairs", 0)) for g in geoms),
        "view_content_words": sum(int(g.get("view_content_words", 0)) for g in geoms),
        "source_absent_content_words": sum(int(g.get("source_absent_content_words", 0)) for g in geoms),
        "view_words": sum(int(g.get("view_words", 0)) for g in geoms),
        "source_words": sum(int(g.get("source_words", 0)) for g in geoms),
    }
    totals["pooled_gap1"] = totals["gap1_adjacent_pairs"] / totals["adjacent_aligned_pairs"] if totals["adjacent_aligned_pairs"] else None
    totals["pooled_skip"] = totals["skip_adjacent_pairs"] / totals["adjacent_aligned_pairs"] if totals["adjacent_aligned_pairs"] else None
    totals["pooled_absent_content_frac"] = totals["source_absent_content_words"] / totals["view_content_words"] if totals["view_content_words"] else 0.0
    totals["pooled_compression"] = totals["view_words"] / totals["source_words"] if totals["source_words"] else None
    return {
        "n": len(geoms),
        "totals": totals,
        "stats": {
            "gap1": stat([g.get("gap1_frac") for g in geoms]),
            "source_span": stat([g.get("source_span") for g in geoms]),
            "aligned_view_fraction": stat([g.get("aligned_view_fraction") for g in geoms]),
            "compression": stat([g.get("compression") for g in geoms]),
        }
    }


def analyze(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    selected_rows = read_jsonl(OUT_DIR / "selected_rows.jsonl")
    selected_pairs = read_jsonl(OUT_DIR / "selected_pairs.jsonl")
    pair_by_id = {p["pair_id"]: p for p in selected_pairs}

    pairs_and_raw = []
    pairs_and_raw += read_outputs_for_shard(OUT_DIR / "prompts_gpu0.jsonl", OUT_DIR / "gen_outputs_gpu0.jsonl")
    pairs_and_raw += read_outputs_for_shard(OUT_DIR / "prompts_gpu1.jsonl", OUT_DIR / "gen_outputs_gpu1.jsonl")

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    except Exception:
        tokenizer = None

    analyzed = []
    for p, raw in pairs_and_raw:
        gen = clean_output(raw)
        val = validate_candidate(p, gen, tokenizer)
        analyzed.append({
            "prompt_id": p["prompt_id"],
            "regime": p["regime"],
            "pair_id": p["pair_id"],
            "source_text": p["source_text"],
            "natural_compact_text": p["natural_compact_text"],
            "generated_text": gen,
            "raw_output": raw,
            "source_words": p["source_words"],
            "target_words": p["target_words"],
            "compact_absent_content_lemma_count": p.get("compact_absent_content_lemma_count"),
            "relation_count_proxy": p.get("relation_count_proxy"),
            **val,
        })

    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in analyzed:
        by_pair[r["pair_id"]].append(r)
    best_by_pair: dict[str, dict[str, Any]] = {}
    for pid, rs in by_pair.items():
        acc = [r for r in rs if r["accept_exact_transform"]]
        if not acc:
            continue
        # Prefer substantive, lower longest run, higher natural overlap, compact-like gap1.
        def key(r: dict[str, Any]) -> tuple:
            struct = r["structure"]
            geom = r["geometry"]
            cls_rank = 0 if struct.get("edit_class") == "substantive_restructure" else 1
            gap = geom.get("gap1_frac")
            return (cls_rank, struct.get("longest_run_frac", 1), -r.get("natural_shared_content_overlap", 0), abs((gap if gap is not None else 0.78) - 0.78), r["regime"])
        best_by_pair[pid] = sorted(acc, key=key)[0]

    # Row fillability under original rows, and pair-subset mass under repacking.
    selected_row_recs = []
    full_rows = []
    partial_rows = []
    for row in selected_rows:
        pids = row["pair_ids"]
        have = [pid for pid in pids if pid in best_by_pair]
        rec = {**row, "n_pairs": len(pids), "n_exact_transform_pairs": len(have), "exact_transform_pair_ids": have, "missing_pair_ids": [pid for pid in pids if pid not in best_by_pair]}
        selected_row_recs.append(rec)
        if pids and len(have) == len(pids):
            full_rows.append(rec)
        elif have:
            partial_rows.append(rec)

    selected_pair_words = sum(pair_by_id[pid]["source_words"] + pair_by_id[pid]["view_words"] for pid in pair_by_id)
    accepted_pair_words = sum(pair_by_id[pid]["source_words"] + pair_by_id[pid]["view_words"] for pid in best_by_pair)
    row_full_words = sum(int(r["words"]) for r in full_rows)
    changed_words_per_10m = int(manifest["changed_words_per_10m"])
    pair_yield = len(best_by_pair) / max(1, len(pair_by_id))
    word_yield = accepted_pair_words / max(1, selected_pair_words)
    # Extrapolate to full changed block only as a rough feasibility number, not a training result.
    estimated_full_changed_words = word_yield * changed_words_per_10m

    reason_counts = collections.Counter()
    for r in analyzed:
        reason_counts.update(r.get("reasons", []))
    by_regime = {}
    for regime in sorted({r["regime"] for r in analyzed}):
        rs = [r for r in analyzed if r["regime"] == regime]
        acc = [r for r in rs if r["accept_exact_transform"]]
        by_regime[regime] = {
            "prompts": len(rs),
            "accepted_prompts": len(acc),
            "accepted_prompt_rate": len(acc)/max(1, len(rs)),
            "pairs_with_accept": len({r["pair_id"] for r in acc}),
            "geometry_accepted": summarize_geom(acc),
        }

    best_records = list(best_by_pair.values())
    summary = {
        "status": "ROW_AWARE_EXACT_BRIDGE_PILOT_ANALYZED",
        "created_utc": now(),
        "selected_rows": len(selected_rows),
        "selected_pairs": len(pair_by_id),
        "total_prompts": len(analyzed),
        "accepted_prompt_count": sum(1 for r in analyzed if r["accept_exact_transform"]),
        "pairs_with_exact_transform": len(best_by_pair),
        "pair_yield": pair_yield,
        "selected_pair_words": selected_pair_words,
        "accepted_pair_words": accepted_pair_words,
        "word_yield": word_yield,
        "estimated_full_pool_changed_words_if_scaled": estimated_full_changed_words,
        "estimated_full_pool_changed_word_fraction_if_scaled": estimated_full_changed_words / changed_words_per_10m if changed_words_per_10m else None,
        "whole_original_rows_filled": len(full_rows),
        "whole_original_row_fill_rate": len(full_rows)/max(1, len(selected_rows)),
        "whole_original_row_words": row_full_words,
        "whole_original_row_word_fraction_of_changed_block_sample_scale": row_full_words/max(1, sum(int(r["words"]) for r in selected_rows)),
        "partial_rows": len(partial_rows),
        "reason_counts": dict(reason_counts.most_common()),
        "by_regime": by_regime,
        "best_pair_geometry": summarize_geom(best_records),
        "decision_readout": {
            "full_generation_continuation_signal": (word_yield >= 0.60 and len(best_by_pair) >= 0.55*len(pair_by_id)),
            "borderline_signal": (0.40 <= word_yield < 0.60),
            "close_or_redesign_signal": (word_yield < 0.40),
            "note": "These thresholds protect against spending 100M training on a tiny matched subset. A future full-pool run must report matched subset word mass and direct quality review; this pilot alone cannot authorize BabyLM pretraining.",
        },
        "interpretation_boundaries": {
            "null_or_negative_future_training": "would implicate the coupled natural-compact realization (source-absent content, stronger compression, lower source retention, surface distribution/quality), not novel vocabulary alone.",
            "positive_future_training": "would show source-attested compact-like structural re-expression can recover stable selected competence on the matched subset, not that source-absent lexical material is unnecessary globally.",
            "current_pilot": "construction feasibility only; no BabyLM model was trained or evaluated.",
        },
        "no_babylm_training_eval_upload_submission": True,
    }
    write_jsonl(OUT_DIR / "analyzed_prompts.jsonl", analyzed)
    write_jsonl(OUT_DIR / "best_exact_transform_by_pair.jsonl", best_records)
    write_jsonl(OUT_DIR / "row_fillability.jsonl", selected_row_recs)
    (OUT_DIR / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown with examples.
    md = [
        "# research row-aware exact bridge pilot",
        "",
        "## Result",
        f"- selected rows: {len(selected_rows)}",
        f"- selected pairs: {len(pair_by_id)}",
        f"- prompts: {len(analyzed)}",
        f"- pairs with exact-length transformation-like output: {len(best_by_pair)} ({pair_yield:.3f})",
        f"- accepted pair word yield: {accepted_pair_words}/{selected_pair_words} ({word_yield:.3f})",
        f"- estimated changed-block word fraction if scaled: {summary['estimated_full_pool_changed_word_fraction_if_scaled']:.3f}",
        f"- whole original rows filled: {len(full_rows)}/{len(selected_rows)} ({summary['whole_original_row_fill_rate']:.3f})",
        "",
        "## Decision meaning",
        "This is construction feasibility only. A 100M training contrast remains unjustified unless a full-pool generation can build a large matched subset with fully bridge rows and shared compact/extractive controls; no compact fallback may be used in the measured block.",
        "",
        "## Top reasons for rejection",
        "| reason | count |",
        "|---|---:|",
    ]
    for k, v in reason_counts.most_common(12):
        md.append(f"| {k} | {v} |")
    md += ["", "## Accepted examples", ""]
    for i, r in enumerate(best_records[:20], 1):
        md += [
            f"### {i}. {r['pair_id']} / {r['regime']} / {r['structure']['edit_class']}",
            f"Source: {r['source_text']}",
            f"Natural compact: {r['natural_compact_text']}",
            f"Bridge: {r['generated_text']}",
            f"Metrics: run={r['structure'].get('longest_run_frac')} order={r['structure'].get('order_agreement')} gap1={r['geometry'].get('gap1_frac')} absent={r['geometry'].get('source_absent_content_frac')}",
            "",
        ]
    (OUT_DIR / "pilot_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "pairs_with_exact_transform": len(best_by_pair),
        "pair_yield": pair_yield,
        "word_yield": word_yield,
        "whole_rows_filled": len(full_rows),
        "decision_readout": summary["decision_readout"],
        "summary": str(OUT_DIR / "pilot_summary.json"),
        "md": str(OUT_DIR / "pilot_summary.md"),
        "no_babylm_training_eval_upload_submission": True,
    }, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--two-rows", type=int, default=40)
    p.add_argument("--three-rows", type=int, default=15)
    p.add_argument("--seed", type=int, default=239003)
    p.add_argument("--temperature", type=float, default=0.1)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
