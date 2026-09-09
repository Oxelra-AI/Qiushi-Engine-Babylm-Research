#!/usr/bin/env python3
"""research: source-attested fluent bridge prototype.

The research extractive result showed that telegraphic source-only selectors do not
recover natural compact's stable selected competence.  This script prepares and
audits a low-cost prototype for the missing quadrant: fluent compact-like views
whose *content lemmas* are all attested in the source sentence.

It does not train, evaluate BabyLM tasks, upload, or submit anything.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/\-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_SEQ_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&'’\-]+(?:\s+|$)){1,8}")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where",
    "when", "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less",
    "least", "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own",
    "same", "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
    "within", "across", "per", "via", "using", "used", "use", "uses", "become", "became", "becomes",
    "based", "while", "although", "however", "therefore", "also", "both", "either", "neither", "nor",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}
ENTITY_STOP = {"The", "A", "An", "This", "That", "These", "Those", "In", "On", "For", "At", "By", "From", "To", "And", "But", "Or", "If", "When", "While", "Because", "SOURCE", "Source", "Sentence"}
BAD_OUTPUT_PATTERNS = [re.compile(p, re.I) for p in [
    r"^\s*(sure|here('| i)s|certainly|of course)\b", r"as an ai", r"please provide", r"output only",
    r"source sentence", r"allowed content", r"target length", r"\[.*\]", r"^\s*[-*•]", r"\n\s*[-*•]",
]]
RELATION_WORDS = {
    "because", "caused", "causes", "cause", "led", "leads", "leading", "result", "results", "resulting", "due", "from",
    "based", "between", "among", "during", "before", "after", "while", "when", "created", "creates", "using", "uses",
    "used", "show", "shows", "showed", "found", "finds", "suspect", "emerge", "compared", "than", "through",
    "include", "includes", "made", "made", "help", "helps", "against", "with", "without", "became", "become",
}

ROOT = Path("experiments/archive/frontier_consolidation")
DEFAULT_PAIRS = ROOT / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
DEFAULT_OUT = ROOT / "data/source_attested_fluent_bridge_prototype"
DEFAULT_RUN = ROOT / "training/runs/source_attested_fluent_bridge_qwen96/outputs.jsonl"
DEFAULT_TOKENIZER = ROOT / "data/compliant_tokenizer"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def words(text: str) -> list[str]:
    return [w for w in (text or "").split() if w]


def wc(text: str) -> int:
    return len(words(text))


def norm(tok: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", (tok or "").lower())
    return "".join(parts)


def stem(n: str) -> str:
    """Small deterministic stemmer for source-attested lemma checks.

    Conservative enough to allow ordinary plural/past/gerund variants without
    importing external linguistic resources.  Exact-form checks are also reported.
    """
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


def lexical_tokens(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text or "")]


def content_norms(text: str) -> list[str]:
    out = []
    for t in lexical_tokens(text):
        n = norm(t)
        if not n:
            continue
        if NUM_RE.fullmatch(t):
            out.append(n)
        elif n not in STOPWORDS and len(n) >= 4:
            out.append(n)
    return out


def content_stems(text: str) -> list[str]:
    return [stem(n) for n in content_norms(text)]


def content_fraction(text: str) -> float:
    ws = words(text)
    if not ws:
        return 0.0
    return sum(1 for w in ws if stem(norm(w)) in {stem(x) for x in content_norms(w)}) / len(ws)


def normalized_phrase(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def numbers(text: str) -> set[str]:
    return {norm(m.group(0)) for m in NUM_RE.finditer(text or "")}


def capital_entities(text: str) -> set[str]:
    ents = set()
    for m in CAP_SEQ_RE.finditer(text or ""):
        e = " ".join(m.group(0).split()).strip(" .,:;!?()[]{}\"'")
        if not e or e in ENTITY_STOP:
            continue
        ents.add(normalized_phrase(e))
    return ents


def clean_output(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    # Remove common label wrappers, but keep the sentence content.
    s = re.sub(r"^(sentence|answer|output)\s*:\s*", "", s, flags=re.I)
    return s.strip(" \t\r\n\"'")


def output_for_row(outputs: list[dict[str, Any]], idx: int) -> str:
    if idx >= len(outputs):
        return ""
    row = outputs[idx]
    return str(row.get("output") or row.get("generated_text") or row.get("text") or row.get("completion") or "")


def compact_absent_count(source: str, compact: str) -> int:
    src = collections.Counter(content_stems(source))
    absent = 0
    for st in content_stems(compact):
        if src.get(st, 0) > 0:
            src[st] -= 1
        else:
            absent += 1
    return absent


def relation_count(text: str) -> int:
    toks = [norm(t) for t in lexical_tokens(text)]
    return sum(1 for t in toks if t in RELATION_WORDS)


def source_content_list(source: str, max_items: int = 60) -> list[str]:
    seen = set()
    out = []
    for t in lexical_tokens(source):
        n = norm(t)
        if not n:
            continue
        if n in STOPWORDS and not NUM_RE.fullmatch(t):
            continue
        if len(n) < 4 and not NUM_RE.fullmatch(t):
            continue
        st = stem(n)
        if st in seen:
            continue
        seen.add(st)
        out.append(t.strip(".,;:!?()[]{}\"'"))
        if len(out) >= max_items:
            break
    return out


def make_anchor_sequence(source: str, target_words: int) -> str:
    """A sparse source-attested content skeleton for gap-fill prompts.

    The skeleton is deliberately incomplete; it supplies relation/content anchors
    while the generator may add function words and additional source content words.
    """
    toks = lexical_tokens(source)
    content = [(i, t) for i, t in enumerate(toks) if stem(norm(t)) in set(content_stems(t))]
    if not content:
        return " ".join(words(source)[:max(4, min(target_words, 12))])
    # Select a compact but source-wide set: endpoints, relation words, and evenly spaced content.
    keep: set[int] = set()
    keep.add(content[0][0]); keep.add(content[-1][0])
    for i, t in content:
        if norm(t) in RELATION_WORDS or NUM_RE.fullmatch(t) or (t[:1].isupper() and i > 0):
            keep.add(i)
    k = max(4, min(len(content), math.ceil(0.58 * target_words)))
    for j in range(k):
        idx = int((j + 0.5) * len(content) / k)
        idx = min(idx, len(content) - 1)
        keep.add(content[idx][0])
    return " ".join(toks[i] for i in sorted(keep))


def select_pairs(pairs: list[dict[str, Any]], n_pairs: int, seed: int) -> list[dict[str, Any]]:
    enriched = []
    for p in pairs:
        src = " ".join(str(p.get("source_text") or "").split())
        comp = " ".join(str(p.get("view_text") or p.get("original_compact_text") or "").split())
        sw, vw = wc(src), wc(comp)
        if sw < 12 or sw > 55 or vw < 7 or vw > 35:
            continue
        ac = compact_absent_count(src, comp)
        rc = relation_count(src)
        enriched.append({**p, "source_text": src, "view_text": comp, "source_words": sw, "view_words": vw,
                         "compact_absent_content_lemma_count": ac, "relation_count_proxy": rc,
                         "length_ratio": vw / max(1, sw)})
    rng = random.Random(seed)
    rng.shuffle(enriched)
    # Balanced prototype: hard lexical-closure cases, relation-bearing cases, and ordinary cases.
    buckets = [
        ("hard_absent_relation", lambda r: r["compact_absent_content_lemma_count"] >= 1 and r["relation_count_proxy"] >= 1, n_pairs // 3),
        ("hard_absent_any", lambda r: r["compact_absent_content_lemma_count"] >= 2, n_pairs // 6),
        ("relation_zero_absent", lambda r: r["compact_absent_content_lemma_count"] == 0 and r["relation_count_proxy"] >= 1, n_pairs // 4),
        ("ordinary", lambda r: True, n_pairs),
    ]
    chosen: list[dict[str, Any]] = []
    seen = set()
    for bname, pred, quota in buckets:
        for r in enriched:
            pid = r.get("pair_id")
            if pid in seen or not pred(r):
                continue
            rr = dict(r); rr["prototype_bucket"] = bname
            chosen.append(rr); seen.add(pid)
            if sum(1 for x in chosen if x.get("prototype_bucket") == bname) >= quota:
                break
            if len(chosen) >= n_pairs:
                break
        if len(chosen) >= n_pairs:
            break
    return chosen[:n_pairs]


def prompt_compress(row: dict[str, Any]) -> str:
    src = row["source_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    return (
        "Task: write ONE fluent English sentence that compresses the source while preserving its main proposition and relation.\n"
        f"Target length: about {target} words (±2 words is fine).\n"
        "Hard constraint: every CONTENT word in your output must be a lemma/inflection already present in the source. "
        "Do not use new synonyms or new factual terms. Function words such as the, of, to, by, with, because, and punctuation are allowed. "
        "Keep all names, numbers, negation, comparisons, and causal/temporal relations that the sentence needs. "
        "If unsure, copy a short source clause rather than inventing a new word. Output only the sentence.\n\n"
        f"SOURCE:\n{src}\n\n"
        f"SOURCE-ATTESTED CONTENT WORDS (use only these as content words):\n{allowed}"
    )


def prompt_gapfill(row: dict[str, Any]) -> str:
    src = row["source_text"]
    target = int(row["view_words"])
    allowed = ", ".join(source_content_list(src))
    anchors = make_anchor_sequence(src, target)
    return (
        "Task: turn the source-attested anchor words into ONE fluent, grammatical sentence preserving the source proposition.\n"
        f"Target length: about {target} words (±2 words is fine).\n"
        "Hard constraint: do not add any content lemma that is absent from the source. You may add only function words, punctuation, and source-attested content words. "
        "You may drop some anchors if needed for grammar, but keep names, numbers, negation, comparisons, and causal/temporal relations. "
        "Do not explain; output only the sentence.\n\n"
        f"SOURCE:\n{src}\n\n"
        f"ANCHOR WORDS IN SOURCE ORDER:\n{anchors}\n\n"
        f"SOURCE-ATTESTED CONTENT WORDS:\n{allowed}"
    )


def prepare(args: argparse.Namespace) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(Path(args.pairs))
    selected = select_pairs(pairs, args.n_pairs, args.seed)
    prompt_rows = []
    for r in selected:
        for regime in args.regimes.split(","):
            regime = regime.strip()
            if not regime:
                continue
            if regime == "lexclosed_compress":
                prompt = prompt_compress(r)
            elif regime == "anchor_gapfill":
                prompt = prompt_gapfill(r)
            else:
                raise ValueError(f"unknown regime {regime}")
            prompt_rows.append({
                "id": f"step234_{regime}_{r['pair_id']}",
                "prompt_id": f"step234_{regime}_{r['pair_id']}",
                "regime": regime,
                "pair_id": r["pair_id"],
                "prototype_bucket": r.get("prototype_bucket"),
                "source_text": r["source_text"],
                "natural_compact_text": r["view_text"],
                "source_words": r["source_words"],
                "target_words": r["view_words"],
                "natural_compact_words": r["view_words"],
                "natural_compact_absent_content_lemma_count": r["compact_absent_content_lemma_count"],
                "relation_count_proxy": r["relation_count_proxy"],
                "allowed_content_words": source_content_list(r["source_text"]),
                "anchor_sequence": make_anchor_sequence(r["source_text"], int(r["view_words"])),
                "prompt": prompt,
            })
    prompts_path = out / "source_attested_fluent_bridge_prompts.jsonl"
    selected_path = out / "source_attested_fluent_bridge_selected_pairs.jsonl"
    write_jsonl(prompts_path, prompt_rows)
    write_jsonl(selected_path, selected)
    payload = {
        "status": "SOURCE_ATTESTED_FLUENT_BRIDGE_PROMPTS_PREPARED",
        "created_utc": now(),
        "purpose": "Minimal generation prototype to test whether fluent compact-like views can be generated with zero source-unattested content lemmas before any BabyLM pretraining.",
        "pairs_input": str(Path(args.pairs)),
        "pairs_input_sha256": sha256_file(Path(args.pairs)),
        "n_pairs": len(selected),
        "n_prompts": len(prompt_rows),
        "regimes": sorted(set(r["regime"] for r in prompt_rows)),
        "bucket_counts": dict(collections.Counter(r.get("prototype_bucket") for r in selected)),
        "prompt_file": str(prompts_path),
        "selected_pairs_file": str(selected_path),
        "recommended_generation_command": (
            f"CUDA_VISIBLE_DEVICES=0 ${{BABYLM_GENERATOR:?configure-an-external-generator}} --model qwen3.5-9b --prompts-jsonl {prompts_path} "
            f"--output-jsonl {Path(args.run_outputs)} --batch-size 32 --max-new-tokens 80 --temperature 0.1 --device cuda"
        ),
        "no_training_eval_upload_aoa_or_leaderboard": True,
    }
    meta = out / "source_attested_fluent_bridge_prompt_manifest.json"
    meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "prompts": str(prompts_path), "prompts_count": len(prompt_rows), "manifest": str(meta)}, indent=2))


def load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(str(path))
    except Exception:
        return None


def nlless_stats(vals: Iterable[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"n": 0}
    xs.sort()
    def q(p: float) -> float:
        if len(xs) == 1: return xs[0]
        z = p * (len(xs)-1); lo = math.floor(z); hi = math.ceil(z)
        return xs[lo] if lo == hi else xs[lo]*(hi-z)+xs[hi]*(z-lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.1), "p25": q(0.25), "p75": q(0.75), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


def analyze_one(prompt: dict[str, Any], raw: str, idx: int, tokenizer=None) -> dict[str, Any]:
    src = prompt["source_text"]
    nat = prompt.get("natural_compact_text") or ""
    out = clean_output(raw)
    source_forms = collections.Counter(content_norms(src))
    source_stems = collections.Counter(content_stems(src))
    out_content = content_norms(out)
    unsupported_exact = []
    unsupported_lemma = []
    rem_exact = dict(source_forms)
    rem_stem = dict(source_stems)
    for n in out_content:
        st = stem(n)
        if rem_exact.get(n, 0) > 0:
            rem_exact[n] -= 1
        elif n not in source_forms:
            unsupported_exact.append(n)
        if rem_stem.get(st, 0) > 0:
            rem_stem[st] -= 1
        elif st not in source_stems:
            unsupported_lemma.append(n)
    src_nums = numbers(src); out_nums = numbers(out)
    src_caps = capital_entities(src); out_caps = capital_entities(out)
    new_caps = sorted(e for e in out_caps if e not in src_caps and not any(e in s or s in e for s in src_caps))
    target = int(prompt.get("target_words") or wc(nat) or 1)
    out_words = wc(out)
    nat_words = wc(nat)
    source_words = int(prompt.get("source_words") or wc(src))
    hard = []
    if not out:
        hard.append("empty")
    if any(rx.search(raw or "") for rx in BAD_OUTPUT_PATTERNS):
        hard.append("bad_pattern")
    if "\n" in (raw or "").strip():
        hard.append("multiline")
    if out and out[-1] not in ".!?\"'":
        hard.append("bad_end")
    if len(unsupported_lemma) > 0:
        hard.append("unsupported_content_lemma")
    if len(out_nums - src_nums) > 0:
        hard.append("new_number")
    if src_nums and len(src_nums & out_nums) < len(src_nums):
        hard.append("missing_number")
    if len(new_caps) > 2:
        hard.append("new_entity_like")
    if out_words < max(5, target - 4) or out_words > target + 5:
        hard.append("length_far_from_target")
    src_content_set = set(content_stems(src))
    out_content_set = set(stem(n) for n in out_content)
    nat_content_set = set(content_stems(nat))
    recall_source = len(src_content_set & out_content_set) / max(1, len(src_content_set))
    recall_natural = len(nat_content_set & out_content_set) / max(1, len(nat_content_set)) if nat_content_set else None
    relation_retained = bool(set(norm(t) for t in lexical_tokens(out)) & {norm(x) for x in RELATION_WORDS}) or relation_count(src) == 0
    if relation_count(src) > 0 and not relation_retained:
        hard.append("no_relation_proxy")
    tok_out = len(tokenizer.encode(out, add_special_tokens=False)) if tokenizer is not None and out else None
    tok_nat = len(tokenizer.encode(nat, add_special_tokens=False)) if tokenizer is not None and nat else None
    return {
        "index": idx,
        "prompt_id": prompt.get("prompt_id") or prompt.get("id"),
        "regime": prompt.get("regime"),
        "pair_id": prompt.get("pair_id"),
        "prototype_bucket": prompt.get("prototype_bucket"),
        "source_text": src,
        "natural_compact_text": nat,
        "generated_text": out,
        "raw_output": raw,
        "source_words": source_words,
        "target_words": target,
        "natural_compact_words": nat_words,
        "generated_words": out_words,
        "generated_to_source_ratio": out_words / max(1, source_words),
        "generated_to_natural_words_ratio": out_words / max(1, nat_words),
        "natural_compact_absent_content_lemma_count": prompt.get("natural_compact_absent_content_lemma_count"),
        "generated_unsupported_content_exact": unsupported_exact,
        "generated_unsupported_content_lemma": unsupported_lemma,
        "unsupported_content_lemma_count": len(unsupported_lemma),
        "unsupported_content_exact_count": len(unsupported_exact),
        "generated_content_fraction": len(out_content) / max(1, out_words),
        "natural_content_fraction": len(content_norms(nat)) / max(1, nat_words),
        "source_content_recall": recall_source,
        "natural_content_overlap_recall": recall_natural,
        "source_numbers": sorted(src_nums),
        "generated_numbers": sorted(out_nums),
        "source_entities_proxy": sorted(src_caps),
        "new_entity_like": new_caps,
        "source_relation_count_proxy": relation_count(src),
        "relation_proxy_retained": relation_retained,
        "generated_active_tokens": tok_out,
        "natural_active_tokens": tok_nat,
        "active_token_ratio_to_natural": (tok_out / tok_nat) if tok_out is not None and tok_nat else None,
        "hard_reasons": hard,
        "prototype_accept": len(hard) == 0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_regime[str(r.get("regime"))].append(r)
    def small(rs: list[dict[str, Any]]) -> dict[str, Any]:
        acc = [r for r in rs if r.get("prototype_accept")]
        unsupported_zero = [r for r in rs if int(r.get("unsupported_content_lemma_count") or 0) == 0]
        hard = collections.Counter(x for r in rs for x in r.get("hard_reasons") or [])
        return {
            "n": len(rs),
            "accepted": len(acc),
            "accepted_rate": len(acc) / max(1, len(rs)),
            "zero_unsupported_content_lemma": len(unsupported_zero),
            "zero_unsupported_content_lemma_rate": len(unsupported_zero) / max(1, len(rs)),
            "mean_unsupported_content_lemma_count": statistics.fmean([int(r.get("unsupported_content_lemma_count") or 0) for r in rs]) if rs else None,
            "generated_words": nlless_stats([r.get("generated_words") for r in rs]),
            "generated_to_natural_words_ratio": nlless_stats([r.get("generated_to_natural_words_ratio") for r in rs]),
            "active_token_ratio_to_natural": nlless_stats([r.get("active_token_ratio_to_natural") for r in rs if r.get("active_token_ratio_to_natural") is not None]),
            "generated_content_fraction": nlless_stats([r.get("generated_content_fraction") for r in rs]),
            "source_content_recall": nlless_stats([r.get("source_content_recall") for r in rs]),
            "natural_content_overlap_recall": nlless_stats([r.get("natural_content_overlap_recall") for r in rs if r.get("natural_content_overlap_recall") is not None]),
            "relation_proxy_retained_rate": sum(1 for r in rs if r.get("relation_proxy_retained")) / max(1, len(rs)),
            "top_hard_reasons": hard.most_common(20),
            "accepted_examples": acc[:8],
            "unsupported_examples": [r for r in rs if r.get("unsupported_content_lemma_count")][:8],
        }
    return {"overall": small(rows), "by_regime": {k: small(v) for k, v in sorted(by_regime.items())}}


def analyze(args: argparse.Namespace) -> None:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(Path(args.prompts))
    outputs = read_jsonl(Path(args.outputs))
    if len(outputs) < len(prompts):
        raise RuntimeError(f"outputs {len(outputs)} shorter than prompts {len(prompts)}")
    tok = load_tokenizer(Path(args.tokenizer))
    rows = [analyze_one(p, output_for_row(outputs, i), i, tok) for i, p in enumerate(prompts)]
    summary = summarize(rows)
    all_path = out / "source_attested_fluent_bridge_generation_rows.jsonl"
    accepted_path = out / "source_attested_fluent_bridge_accepted.jsonl"
    summary_path = out / "source_attested_fluent_bridge_generation_summary.json"
    review_path = out / "source_attested_fluent_bridge_review_sample.jsonl"
    write_jsonl(all_path, rows)
    write_jsonl(accepted_path, [r for r in rows if r.get("prototype_accept")])
    rng = random.Random(args.seed)
    # Review sample focuses on accepted hard cases plus failures; suitable for semantic review.
    accepted_hard = [r for r in rows if r.get("prototype_accept") and int(r.get("natural_compact_absent_content_lemma_count") or 0) > 0]
    failures = [r for r in rows if not r.get("prototype_accept")]
    review = []
    for r in rng.sample(accepted_hard, min(18, len(accepted_hard))) if accepted_hard else []:
        review.append({"sample_kind": "accepted_hard", **r})
    for r in rng.sample(failures, min(18, len(failures))) if failures else []:
        review.append({"sample_kind": "failed", **r})
    for r in rng.sample(rows, min(12, len(rows))):
        review.append({"sample_kind": "random", **r})
    write_jsonl(review_path, review)
    payload = {
        "status": "SOURCE_ATTESTED_FLUENT_BRIDGE_GENERATION_ANALYZED",
        "created_utc": now(),
        "purpose": "Constructibility test for fluent source-attested compact-like views before any pretraining.",
        "prompts": str(Path(args.prompts)),
        "outputs": str(Path(args.outputs)),
        "prompts_sha256": sha256_file(Path(args.prompts)),
        "outputs_sha256": sha256_file(Path(args.outputs)),
        "tokenizer_loaded": tok is not None,
        "acceptance_definition": "single-sentence output, zero unsupported content lemmas by conservative source stem set, no new/missing numbers, no large new entity-like phrase set, target-near length, relation proxy retained when source has relation markers.",
        "summary": summary,
        "files": {"rows": str(all_path), "accepted": str(accepted_path), "review_sample": str(review_path)},
        "interpretation_boundary": "Automatic checks establish lexical closure and geometry only; proposition preservation and fluency require scientist/independent_review reading before any full training.",
        "no_training_eval_upload_aoa_or_leaderboard": True,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Concise note
    note_path = out / "source_attested_fluent_bridge_generation_note.md"
    ov = summary["overall"]
    lines = [
        "# research source-attested fluent bridge prototype\n\n",
        "This is a constructibility probe, not a BabyLM training result. It tests whether Qwen3.5 can produce fluent compact-like views whose content lemmas are all attested in the source.\n\n",
        f"Prompts: {len(prompts)}; outputs: {len(outputs)}. Prototype accepted {ov['accepted']}/{ov['n']} ({ov['accepted_rate']:.3f}); zero unsupported content lemmas {ov['zero_unsupported_content_lemma']}/{ov['n']} ({ov['zero_unsupported_content_lemma_rate']:.3f}).\n\n",
        "| regime | n | accepted | accept rate | zero unsupported lemma rate | gen/natural words mean | token/natural mean | relation retained |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for reg, s in summary["by_regime"].items():
        lines.append(f"| {reg} | {s['n']} | {s['accepted']} | {s['accepted_rate']:.3f} | {s['zero_unsupported_content_lemma_rate']:.3f} | {s['generated_to_natural_words_ratio'].get('mean', float('nan')):.3f} | {s['active_token_ratio_to_natural'].get('mean', float('nan')):.3f} | {s['relation_proxy_retained_rate']:.3f} |\n")
    lines += [
        "\nAutomatic lexical closure is not semantic truth. Review sample: `source_attested_fluent_bridge_review_sample.jsonl`.\n",
        f"\nSummary JSON: `{summary_path}`\n",
    ]
    note_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "summary": str(summary_path), "accepted": ov["accepted"], "accepted_rate": ov["accepted_rate"], "zero_unsupported_rate": ov["zero_unsupported_content_lemma_rate"], "review_sample": str(review_path)}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--run-outputs", default=str(DEFAULT_RUN))
    p.add_argument("--n-pairs", type=int, default=48)
    p.add_argument("--regimes", default="lexclosed_compress,anchor_gapfill")
    p.add_argument("--seed", type=int, default=234031)
    a = sub.add_parser("analyze")
    a.add_argument("--prompts", default=str(DEFAULT_OUT / "source_attested_fluent_bridge_prompts.jsonl"))
    a.add_argument("--outputs", default=str(DEFAULT_RUN))
    a.add_argument("--out-dir", default=str(DEFAULT_OUT))
    a.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    a.add_argument("--seed", type=int, default=234041)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "analyze":
        analyze(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
