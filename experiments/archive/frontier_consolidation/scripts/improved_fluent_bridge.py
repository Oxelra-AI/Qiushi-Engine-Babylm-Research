#!/usr/bin/env python3
"""research: Improved source-attested fluent bridge prototype.

Addresses research review defects:
  - Larger sample: 250 pairs from 12,155 (up from 48)
  - Three first-pass prompt regimes with explicit completeness requirements
  - Stricter structural validation (finite verbs, protected complements, polarity)
  - Source-copy degree tracking
  - Repair prompts for repairable failures
  - Blinded stratified review sample

Usage:
  python improved_fluent_bridge.py prepare
  # Then run generation (750 prompts, ~3 min on one H100):
  # CUDA_VISIBLE_DEVICES=0 \"${BABYLM_GENERATOR:?configure-an-external-generator}\" --model qwen3.5-9b ...
  python improved_fluent_bridge.py analyze --outputs-jsonl <path>
  # Generates repair prompts for repairable failures:
  python improved_fluent_bridge.py repair-prompts
  # After repair generation:
  python improved_fluent_bridge.py finalize --repair-outputs-jsonl <path>

Does NOT train, evaluate BabyLM tasks, upload, or submit anything.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, statistics, time
from pathlib import Path
from typing import Any, Iterable

# ── text tools ──────────────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[''\-][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
SIGNED_NUM_RE = re.compile(
    r"[−\-]?\d+(?:[.,:/\-]\d+)*(?:\s*(?:%|percent|°\s*[CF]|degrees?\s*[CF]|/\s*°?\s*[CF]|per\s+cent))?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&''\-]+(?:\s+|$)){1,8}")

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","so","because","as","than",
    "to","of","in","on","for","with","without","by","from","at","into","onto","over",
    "under","about","between","among","through","during","before","after","above","below",
    "is","am","are","was","were","be","been","being","do","does","did","done","doing",
    "have","has","had","having","can","could","may","might","must","shall","should",
    "will","would","this","that","these","those","there","here","it","its","they","them",
    "their","theirs","he","him","his","she","her","hers","we","us","our","ours","you",
    "your","yours","i","me","my","mine","who","whom","whose","which","what","where",
    "when","why","how","not","no","nor","only","just","also","very","more","most","less",
    "least","much","many","some","any","all","each","every","other","another","such","own",
    "same","too","again","still","already","yet","up","down","out","off","back","away",
    "within","across","per","via","using","used","use","uses","become","became","becomes",
    "based","while","although","however","therefore","also","both","either","neither",
    "one","two","three","four","five","six","seven","eight","nine","ten",
}
ENTITY_STOP = {"The","A","An","This","That","These","Those","In","On","For","At","By",
               "From","To","And","But","Or","If","When","While","Because","SOURCE","Source","Sentence"}
RELATION_WORDS = {
    "because","caused","causes","cause","led","leads","leading","result","results","resulting",
    "due","based","between","among","during","before","after","while","when","created","creates",
    "using","uses","used","show","shows","showed","found","finds","compared","than","through",
    "include","includes","made","help","helps","against","became","become","becomes","without",
}
POLARITY_WORDS = {"only","not","never","neither","nor","no","none","nothing","nobody",
                  "nowhere","cannot","can't","won't","don't","doesn't","didn't","isn't",
                  "aren't","wasn't","weren't","hasn't","haven't","hadn't"}
FINITE_AUX = {"is","are","was","were","am","has","have","had","do","does","did",
              "can","could","will","would","shall","should","may","might","must"}
BAD_OUTPUT_PATTERNS = [re.compile(p, re.I) for p in [
    r"^\s*(sure|here('| i)s|certainly|of course)\b", r"as an ai", r"please provide",
    r"output only", r"source sentence", r"allowed content", r"target length",
    r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]",
]]

ROOT = Path("experiments/archive/frontier_consolidation")
PAIRS_PATH = ROOT / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
OUT_DIR = ROOT / "data/improved_fluent_bridge"
TOKENIZER_PATH = ROOT / "data/compliant_tokenizer"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def read_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def wc(t: str) -> int:
    return len(t.split()) if t else 0

def norm(t: str) -> str:
    return "".join(re.findall(r"[a-z0-9]", (t or "").lower()))

def stem(n: str) -> str:
    s = norm(n)
    if not s: return s
    if s.endswith("ies") and len(s) > 5: return s[:-3] + "y"
    for suf in ("ingly","edly"):
        if s.endswith(suf) and len(s) > len(suf)+3: return s[:-len(suf)]
    for suf in ("ing","ers","er","ed"):
        if s.endswith(suf) and len(s) > len(suf)+3:
            base = s[:-len(suf)]
            if len(base)>3 and base[-1]==base[-2]: base = base[:-1]
            return base
    if s.endswith("es") and len(s) > 4: return s[:-2]
    if s.endswith("s") and len(s) > 4 and not s.endswith("ss"): return s[:-1]
    return s

def lex_tokens(t: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(t or "")]

def content_norms(t: str) -> list[str]:
    out = []
    for tok in lex_tokens(t):
        n = norm(tok)
        if not n: continue
        if NUM_RE.fullmatch(tok): out.append(n)
        elif n not in STOPWORDS and len(n) >= 4: out.append(n)
    return out

def content_stems(t: str) -> list[str]:
    return [stem(n) for n in content_norms(t)]

def numbers(t: str) -> set[str]:
    return {norm(m.group(0)) for m in NUM_RE.finditer(t or "")}

def signed_number_surfaces(t: str) -> list[str]:
    vals = []
    for m in SIGNED_NUM_RE.finditer(t or ""):
        x = m.group(0).lower().replace("−","-").strip()
        x = re.sub(r"\s+","",x).replace("percent","%").replace("degrees","°").replace("degree","°")
        x = x.strip(".,;:!?()[]{}\"'")
        if x: vals.append(x)
    return vals

def capital_entities(t: str) -> set[str]:
    ents = set()
    for m in CAP_SEQ_RE.finditer(t or ""):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e or e in ENTITY_STOP: continue
        ents.add(" ".join(re.findall(r"[a-z0-9]+", e.lower())))
    return ents

def clean_output(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    s = re.sub(r"^(sentence|answer|output)\s*:\s*", "", s, flags=re.I)
    return s.strip(" \t\r\n\"'")

def compact_absent_count(source: str, compact: str) -> int:
    src = collections.Counter(content_stems(source))
    absent = 0
    for st in content_stems(compact):
        if src.get(st, 0) > 0: src[st] -= 1
        else: absent += 1
    return absent

def relation_count(t: str) -> int:
    toks = {norm(x) for x in lex_tokens(t)}
    return sum(1 for x in toks if x in RELATION_WORDS)

def source_content_list(source: str, max_items: int = 60) -> list[str]:
    seen = set()
    out = []
    for t in lex_tokens(source):
        n = norm(t)
        if not n: continue
        if n in STOPWORDS and not NUM_RE.fullmatch(t): continue
        if len(n) < 4 and not NUM_RE.fullmatch(t): continue
        st = stem(n)
        if st in seen: continue
        seen.add(st)
        out.append(t.strip(".,;:!?()[]{}\"'"))
        if len(out) >= max_items: break
    return out

def make_anchor_sequence(source: str, target_words: int) -> str:
    toks = lex_tokens(source)
    content = [(i,t) for i,t in enumerate(toks) if stem(norm(t)) in set(content_stems(t))]
    if not content: return " ".join(source.split()[:max(4,min(target_words,12))])
    keep = {content[0][0], content[-1][0]}
    for i,t in content:
        if norm(t) in RELATION_WORDS or NUM_RE.fullmatch(t) or (t[:1].isupper() and i>0):
            keep.add(i)
    k = max(4, min(len(content), math.ceil(0.58*target_words)))
    for j in range(k):
        idx = min(int((j+0.5)*len(content)/k), len(content)-1)
        keep.add(content[idx][0])
    return " ".join(toks[i] for i in sorted(keep))

def extract_protected_phrases(source: str) -> list[str]:
    """Extract multi-word phrases that must be preserved in the output."""
    protected = []
    # Number+unit phrases
    for m in SIGNED_NUM_RE.finditer(source):
        phrase = m.group(0).strip()
        if phrase: protected.append(phrase)
    # Detect polarity-marked phrases: "only X", "not X"
    lower = source.lower()
    for pw in ["only", "not", "never"]:
        for m in re.finditer(rf"\b{pw}\b", lower):
            start = m.start()
            # get the next 3-5 words after the polarity word
            rest = source[start:].split()[:5]
            if len(rest) >= 2:
                protected.append(" ".join(rest[:3]))
    return protected

def source_copy_degree(source: str, generated: str) -> float:
    """Fraction of generated content words that are exact source copies (not just stems)."""
    src_norms = set(content_norms(source))
    gen_norms = content_norms(generated)
    if not gen_norms: return 0.0
    copied = sum(1 for n in gen_norms if n in src_norms)
    return copied / len(gen_norms)

# ── stratified sampling ─────────────────────────────────────────────
def select_pairs(pairs: list[dict], n_pairs: int, seed: int) -> list[dict]:
    enriched = []
    for p in pairs:
        src = " ".join(str(p.get("source_text") or "").split())
        comp = " ".join(str(p.get("view_text") or p.get("original_compact_text") or "").split())
        sw, vw = wc(src), wc(comp)
        if sw < 12 or sw > 55 or vw < 7 or vw > 35: continue
        ac = compact_absent_count(src, comp)
        rc = relation_count(src)
        enriched.append({**p, "source_text": src, "view_text": comp,
                         "source_words": sw, "view_words": vw,
                         "compact_absent_content_lemma_count": ac,
                         "relation_count_proxy": rc,
                         "length_ratio": vw / max(1, sw)})
    rng = random.Random(seed)
    rng.shuffle(enriched)
    # Larger stratified sample with emphasis on hard cases
    buckets = [
        ("hard_absent_relation", lambda r: r["compact_absent_content_lemma_count"]>=1 and r["relation_count_proxy"]>=1, 80),
        ("hard_absent_norelation", lambda r: r["compact_absent_content_lemma_count"]>=2 and r["relation_count_proxy"]==0, 40),
        ("relation_zero_absent", lambda r: r["compact_absent_content_lemma_count"]==0 and r["relation_count_proxy"]>=1, 70),
        ("ordinary", lambda r: True, 60),
    ]
    chosen, seen = [], set()
    for bname, pred, quota in buckets:
        for r in enriched:
            pid = r.get("pair_id")
            if pid in seen or not pred(r): continue
            rr = dict(r); rr["prototype_bucket"] = bname
            chosen.append(rr); seen.add(pid)
            if sum(1 for x in chosen if x["prototype_bucket"]==bname) >= quota: break
            if len(chosen) >= n_pairs: break
        if len(chosen) >= n_pairs: break
    return chosen[:n_pairs]


# ── improved prompts ────────────────────────────────────────────────
COMPRESS_PROMPT = (
    "Task: write ONE complete, fluent English sentence that compresses the source while preserving its main proposition and all central relations.\n\n"
    "HARD RULES:\n"
    "1. Every content word in your output must be a lemma/inflection already present in the source. No new synonyms or factual terms.\n"
    "2. Your sentence MUST have a finite verb (not a bare participle). Include auxiliaries/copulas when needed.\n"
    "3. Preserve ALL numbers exactly including signs, units, ranges, and currencies.\n"
    "4. Preserve polarity markers: only, not, never, neither. Do not drop them.\n"
    "5. Preserve all names and entities exactly.\n"
    "6. Keep who did what to whom — do not drop central complements.\n"
    "7. Function words (the, of, to, by, because, etc.) and punctuation are always allowed.\n"
    "8. If unsure, copy a short source clause rather than inventing a word or dropping a complement.\n\n"
    "BAD example: 'Coefficient found increasing with oxygen.' (missing auxiliary 'was', missing units)\n"
    "GOOD example: 'The coefficient was found to increase with oxygen content from −0.1 to −3.5%/°C.'\n\n"
    "Output ONLY the sentence, nothing else.\n\n"
)

GAPFILL_PROMPT = (
    "Task: turn the source-attested anchor words into ONE complete, fluent, grammatical English sentence that preserves the source proposition.\n\n"
    "HARD RULES:\n"
    "1. Do not add any content lemma absent from the source. Only add function words, punctuation, and source-attested content words.\n"
    "2. Your sentence MUST have a finite verb. Include auxiliaries/copulas when needed.\n"
    "3. Keep names, numbers (with signs/units/ranges), negation, comparisons, and causal/temporal relations.\n"
    "4. Do NOT drop central complements — if the source says X 'were the same' or 'becomes repulsive', keep that.\n"
    "5. If unsure, use a longer clause from the source rather than a bare fragment.\n\n"
    "Output ONLY the sentence, nothing else.\n\n"
)

CLAUSE_BRIDGE_PROMPT = (
    "Task: write ONE complete, fluent English sentence that captures the source's main claim using ONLY words and phrases from the source.\n\n"
    "HARD RULES:\n"
    "1. Every content word must appear in the source sentence. You may add function words and punctuation freely.\n"
    "2. Your sentence must be a complete, grammatical English sentence with subject + finite verb + any needed complements.\n"
    "3. Preserve the source's main claim: who/what did/is what, under what condition, with what result.\n"
    "4. Keep all numbers exactly (with signs, units, ranges). Keep all names exactly.\n"
    "5. Keep polarity (only/not/never) and comparisons (more/less/than) if present in the source.\n"
    "6. Do NOT merely list source words. Write a real sentence.\n\n"
    "Output ONLY the sentence, nothing else.\n\n"
)

def build_compress_prompt(row: dict) -> str:
    src = row["source_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    return COMPRESS_PROMPT + f"TARGET LENGTH: about {target} words (±3 words is fine).\n\nSOURCE:\n{src}\n\nSOURCE-ATTESTED CONTENT WORDS:\n{allowed}"

def build_gapfill_prompt(row: dict) -> str:
    src = row["source_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    anchors = make_anchor_sequence(src, target)
    return GAPFILL_PROMPT + f"TARGET LENGTH: about {target} words (±3 words is fine).\n\nSOURCE:\n{src}\n\nANCHOR WORDS IN SOURCE ORDER:\n{anchors}\n\nSOURCE-ATTESTED CONTENT WORDS:\n{allowed}"

def build_clause_bridge_prompt(row: dict) -> str:
    src = row["source_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    protected = extract_protected_phrases(src)
    prot_str = ", ".join(f'"{p}"' for p in protected[:8]) if protected else "(none detected)"
    return CLAUSE_BRIDGE_PROMPT + (
        f"TARGET LENGTH: about {target} words (±3 words is fine).\n\n"
        f"PROTECTED PHRASES (must appear if relevant to main claim): {prot_str}\n\n"
        f"SOURCE:\n{src}\n\nSOURCE-ATTESTED CONTENT WORDS:\n{allowed}"
    )


# ── validation ──────────────────────────────────────────────────────
def has_finite_verb(text: str) -> bool:
    """Check for at least one plausible finite verb form."""
    words_lower = [w.lower().strip(".,;:!?()[]{}\"'") for w in (text or "").split()]
    # Auxiliaries/modals are always finite
    if any(w in FINITE_AUX for w in words_lower):
        return True
    # Past tense -ed verbs are finite (but also participles — heuristic)
    # Check for verb-like words that aren't at the start after a noun without aux
    for i, w in enumerate(words_lower):
        if w.endswith("ed") and len(w) > 3 and i > 0:
            # If preceded by a subject-like word (not another verb), likely finite
            prev = words_lower[i-1]
            if prev not in FINITE_AUX and prev not in {"and","or","but","then"}:
                return True
        # Present tense -s verbs
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss") and w not in STOPWORDS:
            if i > 0: return True
    return False

def multiset_missing(src_list: list[str], out_list: list[str]) -> list[str]:
    c = collections.Counter(out_list)
    missing = []
    for x in src_list:
        if c[x] > 0: c[x] -= 1
        else: missing.append(x)
    return missing

GRAMMAR_REJECT_PATTERNS = [
    ("bare_advised", re.compile(r"\b(allergies|individuals|people|patients)\s+advised\b", re.I)),
    ("missing_article_from_tray", re.compile(r"\bfrom\s+tray\b", re.I)),
    ("says_plural_subject", re.compile(r"\b(and\s+\w+)\s+says\b", re.I)),
    ("bare_participle_start", re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,3}\s+(?:found|seen|known|considered|observed|created|discovered|designed|built|made)\s+(?:increasing|decreasing|growing|showing|being|having|getting)\b")),
    ("run_on_fragments", re.compile(r",\s+is\s+complete\s+", re.I)),
    ("bare_passive_no_aux", re.compile(r"\b(?:oxygen|blood|data|information|water|air)\s+(?:carried|transported|measured|collected|processed)\s+by\b", re.I)),
    ("dangling_burned", re.compile(r"\bhours?\s+burned\s*[.!?]?\s*$", re.I)),
    ("dedicated_missing_to", re.compile(r"\bdedicated\s+(?:nurtur|creat|build|develop)", re.I)),
]

def validate_row(row: dict, source: str, natural: str, generated: str, tokenizer=None) -> dict:
    """Comprehensive validation of one generated row."""
    reasons = []
    
    # Basic output quality
    if not generated: reasons.append("empty")
    if any(rx.search(generated or "") for rx in BAD_OUTPUT_PATTERNS): reasons.append("bad_pattern")
    if "\n" in (generated or "").strip(): reasons.append("multiline")
    if generated and generated[-1] not in ".!?\"'": reasons.append("bad_end")
    
    # Content lemma closure
    src_stems = collections.Counter(content_stems(source))
    gen_content = content_norms(generated)
    unsupported = []
    rem = dict(src_stems)
    for n in gen_content:
        st = stem(n)
        if rem.get(st, 0) > 0: rem[st] -= 1
        elif st not in src_stems: unsupported.append(n)
    if unsupported: reasons.append("unsupported_content_lemma")
    
    # Number preservation (signed/unit-preserving)
    src_nums = signed_number_surfaces(source)
    gen_nums = signed_number_surfaces(generated)
    missing_nums = multiset_missing(src_nums, gen_nums)
    extra_nums = multiset_missing(gen_nums, src_nums)
    if missing_nums: reasons.append("missing_number_surface")
    if extra_nums: reasons.append("extra_number_surface")
    
    # Entity preservation
    src_caps = capital_entities(source)
    gen_caps = capital_entities(generated)
    new_caps = [e for e in gen_caps if e not in src_caps and not any(e in s or s in e for s in src_caps)]
    if len(new_caps) > 2: reasons.append("new_entity_like")
    
    # Geometry
    nat_words = wc(natural)
    gen_words = wc(generated)
    word_ratio = gen_words / max(1, nat_words) if nat_words else None
    if word_ratio is not None and abs(word_ratio - 1.0) > 0.30:
        reasons.append("geometry_word_mismatch")
    
    # Token geometry
    gen_toks = len(tokenizer.encode(generated, add_special_tokens=False)) if tokenizer and generated else None
    nat_toks = len(tokenizer.encode(natural, add_special_tokens=False)) if tokenizer and natural else None
    tok_ratio = gen_toks / max(1, nat_toks) if gen_toks and nat_toks else None
    if tok_ratio is not None and abs(tok_ratio - 1.0) > 0.35:
        reasons.append("geometry_token_mismatch")
    
    # Grammar patterns
    grammar_hits = [name for name, rx in GRAMMAR_REJECT_PATTERNS if rx.search(generated or "")]
    if grammar_hits: reasons.append("grammar_pattern")
    
    # Finite verb check
    if generated and not has_finite_verb(generated):
        reasons.append("no_finite_verb")
    
    # Polarity preservation
    src_lower = (source or "").lower()
    gen_lower = (generated or "").lower()
    for pw in ["only", "not", "never", "neither"]:
        if f" {pw} " in f" {src_lower} " and f" {pw} " not in f" {gen_lower} ":
            reasons.append(f"dropped_polarity_{pw}")
            break
    
    # Relation proxy
    if relation_count(source) > 0:
        gen_rels = {norm(x) for x in lex_tokens(generated)} & {norm(x) for x in RELATION_WORDS}
        if not gen_rels: reasons.append("no_relation_proxy")
    
    # Source-copy degree
    copy_deg = source_copy_degree(source, generated)
    
    # Content recall
    src_stem_set = set(content_stems(source))
    gen_stem_set = set(stem(n) for n in gen_content)
    nat_stem_set = set(content_stems(natural))
    recall_source = len(src_stem_set & gen_stem_set) / max(1, len(src_stem_set))
    recall_natural = len(nat_stem_set & gen_stem_set) / max(1, len(nat_stem_set)) if nat_stem_set else None
    
    return {
        "accept": len(reasons) == 0,
        "reasons": reasons,
        "unsupported_lemmas": unsupported[:5],
        "missing_number_surfaces": missing_nums,
        "extra_number_surfaces": extra_nums,
        "grammar_hits": grammar_hits,
        "word_ratio": word_ratio,
        "token_ratio": tok_ratio,
        "source_copy_degree": round(copy_deg, 4),
        "source_content_recall": round(recall_source, 4),
        "natural_content_overlap": round(recall_natural, 4) if recall_natural is not None else None,
        "generated_words": gen_words,
        "generated_tokens": gen_toks,
        "natural_words": nat_words,
        "natural_tokens": nat_toks,
    }


# ── repair prompt ───────────────────────────────────────────────────
def build_repair_prompt(row: dict, prev_output: str, fail_reasons: list[str]) -> str:
    src = row["source_text"]
    target = int(row.get("target_words") or row.get("view_words") or 20)
    allowed = ", ".join(source_content_list(src))
    
    issue_desc = []
    if "no_finite_verb" in fail_reasons: issue_desc.append("missing finite verb (needs auxiliary/copula)")
    if "grammar_pattern" in fail_reasons: issue_desc.append("grammatical error")
    if any("dropped_polarity" in r for r in fail_reasons): issue_desc.append("dropped polarity word (only/not/never)")
    if "missing_number_surface" in fail_reasons: issue_desc.append("missing numbers/units")
    if "unsupported_content_lemma" in fail_reasons: issue_desc.append("uses words not in the source")
    if "geometry_word_mismatch" in fail_reasons: issue_desc.append("too long or too short")
    if "no_relation_proxy" in fail_reasons: issue_desc.append("lost causal/temporal relation")
    if not issue_desc: issue_desc.append("quality issue")
    
    return (
        "Task: fix the FLAWED ATTEMPT below into ONE complete, fluent English sentence that preserves the source proposition.\n\n"
        f"ISSUES WITH THE ATTEMPT: {'; '.join(issue_desc)}\n\n"
        "HARD RULES:\n"
        "1. Every content word must be from the source. No new synonyms or factual terms.\n"
        "2. Must be a complete sentence with a finite verb.\n"
        "3. Preserve ALL numbers with signs/units, ALL names, polarity (only/not/never), and central complements.\n"
        "4. Function words and punctuation are always allowed.\n\n"
        f"TARGET LENGTH: about {target} words (±3 words).\n\n"
        f"SOURCE:\n{src}\n\n"
        f"FLAWED ATTEMPT:\n{prev_output}\n\n"
        f"SOURCE-ATTESTED CONTENT WORDS:\n{allowed}\n\n"
        "Output ONLY the corrected sentence, nothing else."
    )


# ── stats helper ────────────────────────────────────────────────────
def stat(vals):
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs: return {"n":0}
    xs.sort()
    def q(p):
        if len(xs)==1: return xs[0]
        z=p*(len(xs)-1); lo=math.floor(z); hi=math.ceil(z)
        return xs[lo] if lo==hi else xs[lo]*(hi-z)+xs[hi]*(z-lo)
    return {"n":len(xs),"mean":round(statistics.fmean(xs),4),"median":round(statistics.median(xs),4),
            "p10":round(q(0.1),4),"p25":round(q(0.25),4),"p75":round(q(0.75),4),
            "p90":round(q(0.9),4),"min":round(xs[0],4),"max":round(xs[-1],4)}


# ── PHASE: prepare ──────────────────────────────────────────────────
def cmd_prepare(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(PAIRS_PATH)
    selected = select_pairs(pairs, args.n_pairs, args.seed)
    
    prompt_rows = []
    regimes = ["lexclosed_compress", "anchor_gapfill", "clause_bridge"]
    for r in selected:
        for regime in regimes:
            if regime == "lexclosed_compress":
                prompt = build_compress_prompt(r)
            elif regime == "anchor_gapfill":
                prompt = build_gapfill_prompt(r)
            elif regime == "clause_bridge":
                prompt = build_clause_bridge_prompt(r)
            else:
                continue
            prompt_rows.append({
                "id": f"step236_{regime}_{r['pair_id']}",
                "prompt_id": f"step236_{regime}_{r['pair_id']}",
                "regime": regime,
                "pair_id": r["pair_id"],
                "prototype_bucket": r.get("prototype_bucket"),
                "source_text": r["source_text"],
                "natural_compact_text": r["view_text"],
                "source_words": r["source_words"],
                "target_words": r["view_words"],
                "compact_absent_content_lemma_count": r["compact_absent_content_lemma_count"],
                "relation_count_proxy": r["relation_count_proxy"],
                "prompt": prompt,
            })
    
    prompts_path = OUT_DIR / "prompts.jsonl"
    selected_path = OUT_DIR / "selected_pairs.jsonl"
    write_jsonl(prompts_path, prompt_rows)
    write_jsonl(selected_path, selected)
    
    bucket_counts = dict(collections.Counter(r.get("prototype_bucket") for r in selected))
    manifest = {
        "status": "PROMPTS_PREPARED",
        "created_utc": now(),
        "n_pairs": len(selected),
        "n_prompts": len(prompt_rows),
        "regimes": regimes,
        "bucket_counts": bucket_counts,
        "pairs_sha256": sha256_file(PAIRS_PATH),
        "prompts_path": str(prompts_path),
        "selected_pairs_path": str(selected_path),
        "generation_command": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b "
            f"--prompts-jsonl {prompts_path} --output-jsonl {OUT_DIR / 'gen_outputs.jsonl'} "
            f"--batch-size 32 --max-new-tokens 80 --temperature 0.1 --device cuda"
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


# ── PHASE: analyze ──────────────────────────────────────────────────
def cmd_analyze(args):
    prompts = read_jsonl(OUT_DIR / "prompts.jsonl")
    outputs_path = Path(args.outputs_jsonl)
    outputs = read_jsonl(outputs_path)
    
    # Load tokenizer
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    except Exception:
        pass
    
    # Match outputs to prompts by index
    analyzed = []
    for i, p in enumerate(prompts):
        raw = ""
        if i < len(outputs):
            o = outputs[i]
            raw = str(o.get("output") or o.get("generated_text") or o.get("text") or "")
        gen = clean_output(raw)
        
        v = validate_row(p, p["source_text"], p.get("natural_compact_text",""), gen, tokenizer)
        
        analyzed.append({
            "index": i,
            "prompt_id": p["prompt_id"],
            "regime": p["regime"],
            "pair_id": p["pair_id"],
            "prototype_bucket": p.get("prototype_bucket"),
            "source_text": p["source_text"],
            "natural_compact_text": p.get("natural_compact_text",""),
            "generated_text": gen,
            "raw_output": raw,
            "source_words": p["source_words"],
            "target_words": p["target_words"],
            "compact_absent_content_lemma_count": p.get("compact_absent_content_lemma_count"),
            "relation_count_proxy": p.get("relation_count_proxy"),
            **v,
        })
    
    # Per-pair best selection
    by_pair = collections.defaultdict(list)
    for r in analyzed:
        by_pair[r["pair_id"]].append(r)
    
    selected_best = []
    for pid, rs in by_pair.items():
        acc = [r for r in rs if r["accept"]]
        if acc:
            best = min(acc, key=lambda r: (
                abs((r.get("word_ratio") or 99) - 1.0),
                abs((r.get("token_ratio") or 99) - 1.0),
                -(r.get("source_content_recall") or 0),
            ))
            selected_best.append(best)
    
    # Identify repairable failures (not accepted, but could be fixed)
    repair_candidates = []
    for pid, rs in by_pair.items():
        acc = [r for r in rs if r["accept"]]
        if acc: continue  # Already has an accepted output
        # Pick the least-bad failure
        fails = sorted(rs, key=lambda r: len(r.get("reasons",[])))
        if fails:
            best_fail = fails[0]
            repairable_reasons = {"no_finite_verb","grammar_pattern","dropped_polarity_only",
                                  "dropped_polarity_not","dropped_polarity_never",
                                  "missing_number_surface","geometry_word_mismatch",
                                  "no_relation_proxy","unsupported_content_lemma"}
            if set(best_fail.get("reasons",[])) & repairable_reasons:
                repair_candidates.append(best_fail)
    
    # Summary
    reason_counts = collections.Counter()
    for r in analyzed:
        reason_counts.update(r.get("reasons",[]))
    
    by_regime = collections.defaultdict(list)
    by_bucket = collections.defaultdict(list)
    for r in analyzed:
        by_regime[r["regime"]].append(r)
        by_bucket[r.get("prototype_bucket","")].append(r)
    
    def small(rs):
        return {
            "n": len(rs), "accepted": sum(1 for r in rs if r["accept"]),
            "accept_rate": round(sum(1 for r in rs if r["accept"])/max(1,len(rs)), 4),
            "word_ratio": stat([r.get("word_ratio") for r in rs if r["accept"]]),
            "token_ratio": stat([r.get("token_ratio") for r in rs if r["accept"]]),
            "source_copy_degree": stat([r.get("source_copy_degree") for r in rs if r["accept"]]),
            "source_recall": stat([r.get("source_content_recall") for r in rs if r["accept"]]),
            "natural_overlap": stat([r.get("natural_content_overlap") for r in rs if r["accept"]]),
        }
    
    summary = {
        "status": "ANALYZE_COMPLETE",
        "total_prompts": len(analyzed),
        "total_accepted": sum(1 for r in analyzed if r["accept"]),
        "accept_rate": round(sum(1 for r in analyzed if r["accept"])/max(1,len(analyzed)), 4),
        "total_pairs": len(by_pair),
        "pairs_with_accepted": len(selected_best),
        "pair_accept_rate": round(len(selected_best)/max(1,len(by_pair)), 4),
        "repair_candidates": len(repair_candidates),
        "reason_counts": dict(reason_counts.most_common()),
        "by_regime": {k: small(v) for k,v in sorted(by_regime.items())},
        "by_bucket": {k: small(v) for k,v in sorted(by_bucket.items())},
        "overall_accepted": small([r for r in analyzed if r["accept"]]),
        "selected_best_per_pair": small(selected_best),
    }
    
    # Write outputs
    write_jsonl(OUT_DIR / "analyzed_rows.jsonl", analyzed)
    write_jsonl(OUT_DIR / "selected_best.jsonl", selected_best)
    write_jsonl(OUT_DIR / "repair_candidates.jsonl", repair_candidates)
    (OUT_DIR / "analyze_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    
    # Review sample: stratified accepted + failures
    review = []
    for r in selected_best:
        if int(r.get("compact_absent_content_lemma_count") or 0) > 0:
            review.append({"sample_kind": "accepted_hard", **r})
    for r in selected_best:
        if int(r.get("compact_absent_content_lemma_count") or 0) == 0 and r.get("prototype_bucket") != "ordinary":
            review.append({"sample_kind": "accepted_relation", **r})
    for r in repair_candidates[:30]:
        review.append({"sample_kind": "repair_candidate", **r})
    write_jsonl(OUT_DIR / "review_sample.jsonl", review[:80])
    
    print(json.dumps(summary, indent=2))


# ── PHASE: repair-prompts ───────────────────────────────────────────
def cmd_repair_prompts(args):
    candidates = read_jsonl(OUT_DIR / "repair_candidates.jsonl")
    if not candidates:
        print(json.dumps({"status": "NO_REPAIR_CANDIDATES", "n": 0}))
        return
    
    repair_rows = []
    for r in candidates:
        prompt = build_repair_prompt(r, r.get("generated_text",""), r.get("reasons",[]))
        repair_rows.append({
            "id": f"repair_{r['pair_id']}_{r['regime']}",
            "prompt_id": f"repair_{r['pair_id']}_{r['regime']}",
            "regime": "repair",
            "original_regime": r["regime"],
            "pair_id": r["pair_id"],
            "prototype_bucket": r.get("prototype_bucket"),
            "source_text": r["source_text"],
            "natural_compact_text": r.get("natural_compact_text",""),
            "source_words": r.get("source_words"),
            "target_words": r.get("target_words"),
            "compact_absent_content_lemma_count": r.get("compact_absent_content_lemma_count"),
            "relation_count_proxy": r.get("relation_count_proxy"),
            "original_fail_reasons": r.get("reasons",[]),
            "original_output": r.get("generated_text",""),
            "prompt": prompt,
        })
    
    repair_path = OUT_DIR / "repair_prompts.jsonl"
    write_jsonl(repair_path, repair_rows)
    
    manifest = {
        "status": "REPAIR_PROMPTS_READY",
        "n_repair": len(repair_rows),
        "repair_path": str(repair_path),
        "generation_command": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b "
            f"--prompts-jsonl {repair_path} --output-jsonl {OUT_DIR / 'repair_outputs.jsonl'} "
            f"--batch-size 32 --max-new-tokens 80 --temperature 0.1 --device cuda"
        ),
    }
    (OUT_DIR / "repair_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


# ── PHASE: finalize ─────────────────────────────────────────────────
def cmd_finalize(args):
    # Load first-pass results
    analyzed = read_jsonl(OUT_DIR / "analyzed_rows.jsonl")
    first_best = read_jsonl(OUT_DIR / "selected_best.jsonl")
    first_best_pids = {r["pair_id"] for r in first_best}
    
    # Load repair results
    repair_prompts = read_jsonl(OUT_DIR / "repair_prompts.jsonl")
    repair_outputs = read_jsonl(Path(args.repair_outputs_jsonl)) if args.repair_outputs_jsonl else []
    
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH))
    except Exception:
        pass
    
    # Validate repair outputs
    repair_accepted = []
    for i, p in enumerate(repair_prompts):
        raw = ""
        if i < len(repair_outputs):
            o = repair_outputs[i]
            raw = str(o.get("output") or o.get("generated_text") or "")
        gen = clean_output(raw)
        v = validate_row(p, p["source_text"], p.get("natural_compact_text",""), gen, tokenizer)
        
        row = {
            "index": len(analyzed) + i,
            "prompt_id": p["prompt_id"],
            "regime": "repair",
            "original_regime": p.get("original_regime"),
            "pair_id": p["pair_id"],
            "prototype_bucket": p.get("prototype_bucket"),
            "source_text": p["source_text"],
            "natural_compact_text": p.get("natural_compact_text",""),
            "generated_text": gen,
            "source_words": p.get("source_words"),
            "target_words": p.get("target_words"),
            "compact_absent_content_lemma_count": p.get("compact_absent_content_lemma_count"),
            "relation_count_proxy": p.get("relation_count_proxy"),
            "original_fail_reasons": p.get("original_fail_reasons"),
            **v,
        }
        if row["accept"] and row["pair_id"] not in first_best_pids:
            repair_accepted.append(row)
    
    # Combine
    combined = list(first_best) + repair_accepted
    
    # Summary by bucket
    by_bucket = collections.defaultdict(list)
    for r in combined:
        by_bucket[r.get("prototype_bucket","")].append(r)
    
    def small(rs):
        return {
            "n": len(rs),
            "word_ratio": stat([r.get("word_ratio") for r in rs]),
            "token_ratio": stat([r.get("token_ratio") for r in rs]),
            "source_copy_degree": stat([r.get("source_copy_degree") for r in rs]),
            "source_recall": stat([r.get("source_content_recall") for r in rs]),
            "natural_overlap": stat([r.get("natural_content_overlap") for r in rs]),
        }
    
    pairs_selected = read_jsonl(OUT_DIR / "selected_pairs.jsonl")
    total_pairs = len(set(r.get("pair_id") for r in pairs_selected))
    
    final_summary = {
        "status": "FINALIZE_COMPLETE",
        "total_pairs_sampled": total_pairs,
        "first_pass_accepted_pairs": len(first_best),
        "repair_new_accepted_pairs": len(repair_accepted),
        "total_accepted_pairs": len(combined),
        "final_pair_accept_rate": round(len(combined)/max(1, total_pairs), 4),
        "by_bucket": {k: small(v) for k,v in sorted(by_bucket.items())},
        "overall": small(combined),
        "repair_attempted": len(repair_prompts),
        "repair_succeeded": len(repair_accepted),
        "repair_success_rate": round(len(repair_accepted)/max(1,len(repair_prompts)), 4),
    }
    
    write_jsonl(OUT_DIR / "final_accepted.jsonl", combined)
    write_jsonl(OUT_DIR / "repair_analyzed.jsonl", 
                [r for r in [repair_accepted] if False])  # placeholder
    (OUT_DIR / "final_summary.json").write_text(json.dumps(final_summary, indent=2)+"\n")
    
    # Final review: stratified sample for semantic review
    review = []
    for r in combined:
        bucket = r.get("prototype_bucket","")
        if bucket == "hard_absent_relation":
            review.append({"review_kind": "hard_relation_accepted", **r})
        elif bucket == "hard_absent_norelation":
            review.append({"review_kind": "hard_absent_accepted", **r})
    for r in combined:
        if r.get("prototype_bucket") == "relation_zero_absent":
            review.append({"review_kind": "relation_accepted", **r})
    for r in combined:
        if r.get("prototype_bucket") == "ordinary":
            review.append({"review_kind": "ordinary_accepted", **r})
    write_jsonl(OUT_DIR / "final_review_sample.jsonl", review[:60])
    
    print(json.dumps(final_summary, indent=2))


# ── main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--n-pairs", type=int, default=250)
    p_prep.add_argument("--seed", type=int, default=236042)
    
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--outputs-jsonl", required=True)
    
    p_repair = sub.add_parser("repair-prompts")
    
    p_final = sub.add_parser("finalize")
    p_final.add_argument("--repair-outputs-jsonl", default="")
    
    args = parser.parse_args()
    if args.cmd == "prepare": cmd_prepare(args)
    elif args.cmd == "analyze": cmd_analyze(args)
    elif args.cmd == "repair-prompts": cmd_repair_prompts(args)
    elif args.cmd == "finalize": cmd_finalize(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
