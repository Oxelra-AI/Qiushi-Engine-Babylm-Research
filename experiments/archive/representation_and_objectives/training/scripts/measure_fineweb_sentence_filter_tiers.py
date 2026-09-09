#!/usr/bin/env python3
"""Measure FineWeb complete-sentence source screens for the future rewrite test.

No generation/training/evaluation is launched.  The purpose is to choose a source
screen for the later source-by-rewrite experiment that is neither too noisy
(fragment/dialogue/navigation/list artifacts) nor so narrow that it loses the
broader factual-web character needed to challenge the current SimpleWiki route.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random
import re
import statistics
import unicodedata
from typing import Any, Iterable

SRC_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl")
OUT_DIR_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_screens")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_sentence_screen_measurement.md")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z'’.-]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
INTERNAL_SENT_RE = re.compile(r"[.!?][\"”’)]?\s+[\"“‘']?[A-Z0-9]")
BAD_TRAILING_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "no.", "fig.", "etc."}
BAD_START_WORDS = {"i", "you", "we", "he", "she", "they", "it", "me", "my", "your", "our", "his", "her", "their", "of", "and", "but", "or", "so", "then", "because", "though", "although", "while", "which", "who", "whom", "whose"}
BAD_WEB_SUBSTR = [
    "click here", "follow this link", "read more", "press release", "privacy policy", "terms of use", "all rights reserved", "subscribe", "download pdf", "cookie policy", "advertisement", "view all", "story comments", "comments.view", "web site", "doi:", " using doi", "cite or link", "news section does not provide", "related persons", "names of squares", "example :", "example:", "multiple choice", "(a)", "(b)", "(c)", "(d)", "source:", "figure :", "table ",
]
DIALOGUE_MARKERS = [" cried ", " replied ", " asked ", " shouted ", " exclaimed ", " dear,", " good god", " says "]
RELATION_CUES = {
    " is ", " are ", " was ", " were ", " became ", " becomes ", " born ", " died ", " founded ", " located ", " called ", " known ", " used ", " uses ", " includes ", " contains ", " because ", " therefore ", " caused ", " led ", " replaced ", " developed ", " published ", " built ", " served ", " won ", " established ", " created ", " part of ", " member of ", " consists ", " belongs ", " formed ", " opened ", " invented ", " discovered ", " produced ", " released ", " reported ", " found ", " showed ", " according to ",
}
DOMAIN_TERMS = {
    "science_physical": {"volcanic","lava","glacier","molecular","cloud","matter","galaxy","energy","electric","chemical","virus","species","weather","planet","star","biology","engineering","water","geologic","hydrologic","climate","emissions","protein","cell","disease","medical","research","study"},
    "geography_places": {"city","province","county","district","region","capital","river","mountain","island","country","state","community","town","village","located","basin","border","population"},
    "people_history": {"born","died","served","mayor","governor","senate","war","leader","president","minister","writer","battle","settled","incorporated","dynasty","king","queen","election"},
    "institutions_society": {"company","university","government","council","education","law","organization","competition","factory","school","system","church","library","policy","court","military","program"},
    "media_culture": {"film","album","band","song","movie","game","book","published","released","television","novel","music","series"},
    "quant_numeric": {"january","february","march","april","may","june","july","august","september","october","november","december","km","metres","feet","century","year","million","percent"},
    "causal_relational": {"because","therefore","caused","effect","result","led","formed","became","created","due","reason","replaced","allows","requires","helps","influenced","responsible"},
}
GENERIC_SINGLE_ENTS = {"The", "This", "That", "These", "Those", "There", "When", "Where", "What", "How", "Why", "Because", "For", "And", "But", "New", "All", "Most", "Some", "Many", "First", "After", "Before", "During", "Then", "Each", "Every", "Through", "Early", "Research", "Example", "Names", "Related", "Various", "Clinical", "Water", "Skin", "Blood", "Food", "Lungs", "Entire", "Body", "Volume", "Issue", "Page", "Link", "Using", "Cite", "View", "Here"}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def wc(text: str) -> int:
    return len(norm_text(text).split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = sorted(list(vals))
    if not xs:
        return {"n": 0}
    def q(p: float):
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.mean(xs), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": xs[-1], "sum": sum(xs)}


def nonlatin_letter_frac(text: str) -> float:
    letters = 0; non = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                non += 1
    return non / letters if letters else 0.0


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars)


def numbers(text: str) -> list[str]:
    return sorted(set(re.sub(r"\s+", "", m.group(0)) for m in NUM_RE.finditer(text)))


def entities(text: str) -> list[str]:
    toks = text.split(); spans=[]; pos=0
    for tok in toks:
        start = text.find(tok, pos)
        if start < 0: start = pos
        spans.append((start, start+len(tok), tok)); pos = start+len(tok)
    out=[]
    for m in ACRONYM_RE.finditer(text):
        out.append(m.group(0))
    caps=[]
    for m in CAP_WORD_RE.finditer(text):
        ti=0
        for j,(a,b,_) in enumerate(spans):
            if a <= m.start() < b:
                ti=j; break
        caps.append((ti, m.group(0).strip(".,;:()[]{}\"“”‘’")))
    i=0
    while i < len(caps):
        ti,val = caps[i]; seq=[val]; j=i+1; prev=ti
        while j < len(caps) and caps[j][0] <= prev+1:
            seq.append(caps[j][1]); prev=caps[j][0]; j+=1
        if len(seq) >= 2:
            out.append(" ".join(seq))
        elif ti != 0 and len(seq[0]) >= 4:
            out.append(seq[0])
        i=j
    cleaned=[]
    for e in out:
        e=" ".join(e.split()).strip(" .,;:()[]{}\"“”‘’")
        if not e: continue
        if len(e.split()) == 1 and e in GENERIC_SINGLE_ENTS: continue
        cleaned.append(e)
    return sorted(set(cleaned))


def domain_hits(text: str) -> list[str]:
    toks = {m.group(0).lower().strip("'’.-") for m in WORD_RE.finditer(text) if len(m.group(0)) >= 3}
    return sorted(k for k, v in DOMAIN_TERMS.items() if toks & v)


def relation_count(text: str) -> int:
    low = " " + text.lower() + " "
    return sum(1 for c in RELATION_CUES if c in low)


def balance_ok(text: str) -> bool:
    return text.count("(") == text.count(")") and text.count("[") == text.count("]") and text.count("{") == text.count("}") and text.count('"') % 2 == 0 and text.count("“") == text.count("”") and text.count("‘") == text.count("’")


def flags_for_row(text: str, source_flags: list[str]) -> list[str]:
    text = norm_text(text); low = text.lower(); toks = text.split(); n=len(toks)
    ents = entities(text); nums = numbers(text); doms = domain_hits(text); rel = relation_count(text)
    flags=[]
    if source_flags: flags.append("source_row_flag")
    if n < 10: flags.append("short")
    if n > 55: flags.append("long")
    if n > 48: flags.append("long_for_generation")
    if toks:
        first = re.sub(r"[^A-Za-z]+", "", toks[0]).lower()
        if first in BAD_START_WORDS: flags.append("context_start")
        last = toks[-1].lower().strip("\"“”‘’()[]{}")
        if last in BAD_TRAILING_ABBR: flags.append("trailing_abbreviation")
    if not text.endswith(('.', '!', '?', '."', '.“', '."', '.”', '?"', '?”', '!"', '!”')):
        flags.append("bad_terminal")
    if not balance_ok(text): flags.append("unbalanced_quote_or_bracket")
    if any(s in low for s in BAD_WEB_SUBSTR): flags.append("web_or_list_artifact")
    if any(s in low for s in DIALOGUE_MARKERS) and ('"' in text or '“' in text or '”' in text): flags.append("dialogue_or_reported_speech")
    if INTERNAL_SENT_RE.search(text.rstrip('.!?')): flags.append("multi_sentence_or_bad_split")
    if re.search(r"[a-z0-9][.!?][A-Z]", text): flags.append("missing_space_after_period")
    if "..." in text or " …" in text or "…" in text: flags.append("ellipsis")
    if text.count(';') >= 2: flags.append("semicolon_list")
    if text.count(':') >= 2: flags.append("colon_list")
    if text.count(',') / max(1,n) > 0.22: flags.append("comma_dense")
    if len(ents) >= 9 and n < 45: flags.append("entity_list_like")
    if len(nums) >= 5 and n < 45: flags.append("number_list_like")
    if sum(any(c.isdigit() for c in w) for w in toks) / max(1,n) > 0.22: flags.append("digit_dense")
    if alpha_frac(text) < 0.62 or nonlatin_letter_frac(text) > 0.02: flags.append("character_quality")
    if not (ents or nums or doms): flags.append("low_substance")
    if rel < 1: flags.append("low_relation")
    return flags


def load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o=json.loads(line); text=norm_text(o["text"]); w=int(o.get("words", wc(text)))
            if w != wc(text):
                raise RuntimeError(f"word mismatch {o.get('sentence_id')}")
            ents=entities(text); nums=numbers(text); doms=domain_hits(text)
            row={
                "sentence_id": int(o["sentence_id"]), "doc_id": str(o.get("doc_id", "")), "source_row": o.get("source_row"), "sent_index_in_row": o.get("sent_index_in_row"), "text": text, "words": w,
                "entities": ents, "numbers": nums, "domain_hits": doms, "relation_count": relation_count(text), "source_row_quality_flags": list(o.get("source_row_quality_flags") or []),
            }
            row["flags"] = flags_for_row(text, row["source_row_quality_flags"])
            row["content_score"] = round(0.65*len(ents)+0.35*len(nums)+0.50*row["relation_count"]+0.25*len(doms)+min(2.0,w/30.0),6)
            rows.append(row)
    return rows


def keep_for_tier(row: dict[str, Any], tier: str) -> bool:
    f=set(row["flags"])
    if tier == "broad_clean_row":
        return "source_row_flag" not in f
    if tier == "complete_nonfragment":
        bad={"source_row_flag","short","long","context_start","trailing_abbreviation","bad_terminal","unbalanced_quote_or_bracket","multi_sentence_or_bad_split","missing_space_after_period","ellipsis","character_quality"}
        return not (f & bad)
    if tier == "web_artifact_removed":
        bad={"source_row_flag","short","long_for_generation","context_start","trailing_abbreviation","bad_terminal","unbalanced_quote_or_bracket","multi_sentence_or_bad_split","missing_space_after_period","ellipsis","character_quality","web_or_list_artifact","semicolon_list","colon_list","comma_dense","entity_list_like","number_list_like","digit_dense"}
        return not (f & bad)
    if tier == "balanced_source_by_rewrite":
        bad={"source_row_flag","short","long_for_generation","context_start","trailing_abbreviation","bad_terminal","unbalanced_quote_or_bracket","multi_sentence_or_bad_split","missing_space_after_period","ellipsis","character_quality","web_or_list_artifact","semicolon_list","colon_list","comma_dense","entity_list_like","number_list_like","digit_dense"}
        return not (f & bad) and not ({"low_substance","low_relation"} <= f)
    if tier == "strict_factual_expository":
        bad={"source_row_flag","short","long_for_generation","context_start","trailing_abbreviation","bad_terminal","unbalanced_quote_or_bracket","multi_sentence_or_bad_split","missing_space_after_period","ellipsis","character_quality","web_or_list_artifact","semicolon_list","colon_list","comma_dense","entity_list_like","number_list_like","digit_dense","low_substance","low_relation","dialogue_or_reported_speech"}
        return not (f & bad)
    raise ValueError(tier)


def summarize_tier(rows: list[dict[str, Any]], kept: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    rng=random.Random(seed)
    doc_counter=collections.Counter(r["doc_id"] for r in kept)
    dom_counter=collections.Counter(h for r in kept for h in r["domain_hits"])
    flag_counter=collections.Counter(fl for r in rows for fl in r["flags"])
    out={
        "rows": len(kept), "words": sum(r["words"] for r in kept), "unique_docs": len(doc_counter),
        "row_fraction": len(kept)/max(1,len(rows)), "word_fraction": sum(r["words"] for r in kept)/max(1,sum(r["words"] for r in rows)),
        "word_stats": stats([r["words"] for r in kept]), "entity_count_stats": stats([len(r["entities"]) for r in kept]), "number_count_stats": stats([len(r["numbers"]) for r in kept]), "relation_count_stats": stats([r["relation_count"] for r in kept]),
        "domain_hit_counts": dict(dom_counter), "top_docs": doc_counter.most_common(8),
        "first_kept": kept[:8],
        "random_kept": rng.sample(kept, min(8, len(kept))) if kept else [],
        "top_score_kept": sorted(kept, key=lambda r:(-r["content_score"], r["sentence_id"]))[:8],
    }
    return out


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--source", default=str(SRC_DEFAULT)); ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT)); ap.add_argument("--note", default=str(NOTE_DEFAULT)); ap.add_argument("--seed", type=int, default=82910); args=ap.parse_args()
    rows=load_rows(pathlib.Path(args.source))
    tiers=["broad_clean_row","complete_nonfragment","web_artifact_removed","balanced_source_by_rewrite","strict_factual_expository"]
    out_dir=pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results={}
    for tier in tiers:
        kept=[r for r in rows if keep_for_tier(r,tier)]
        kept=sorted(kept, key=lambda r:(r["doc_id"], r["source_row"], r["sent_index_in_row"], r["sentence_id"]))
        results[tier]=summarize_tier(rows, kept, args.seed)
        # Write only metadata-light selected row files for the two plausible future tiers.
        if tier in {"web_artifact_removed","balanced_source_by_rewrite","strict_factual_expository"}:
            write_jsonl(out_dir / f"{tier}_sources.jsonl", kept)
    flag_counts=collections.Counter(fl for r in rows for fl in r["flags"])
    meta={
        "status":"FINEWEB_SENTENCE_SCREEN_TIERS_MEASURED",
        "source":str(args.source), "source_sha256":sha256_file(pathlib.Path(args.source)),
        "total_rows":len(rows), "total_words":sum(r["words"] for r in rows),
        "flag_counts":dict(flag_counts.most_common()),
        "tiers":results,
        "interpretation":"balanced_source_by_rewrite is the intended source screen if the next route needs enough mass while removing obvious fragments, lists, web navigation, and malformed rows; strict_factual_expository is cleaner but may be too small for a broad-source contrast.",
    }
    meta_path=out_dir/"fineweb_sentence_screen_tiers.json"
    sample_path=out_dir/"fineweb_sentence_screen_samples.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    sample_path.write_text(json.dumps({tier: {k: results[tier][k] for k in ["first_kept","random_kept","top_score_kept"]} for tier in tiers}, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    note=pathlib.Path(args.note); note.parent.mkdir(parents=True, exist_ok=True)
    lines=["# research FineWeb sentence source screen measurement\n\n", "No generation, training, or evaluation was launched. The active H100 pair remains untouched.\n\n", "| tier | rows | words | unique docs | word fraction | mean words | p95 words |\n|---|---:|---:|---:|---:|---:|---:|\n"]
    for tier in tiers:
        r=results[tier]; ws=r["word_stats"]
        lines.append(f"| {tier} | {r['rows']:,} | {r['words']:,} | {r['unique_docs']:,} | {r['word_fraction']:.3f} | {ws.get('mean',0):.2f} | {ws.get('p95',0)} |\n")
    lines += ["\nInterpretation: the previous broad prompt file preserves mass but includes dialogue/fragments/artifacts; the previous strict prompt file is cleaner but too narrow. The balanced source-by-rewrite tier is the intended CPU-side source asset if a future Qwen slice is needed, because it removes obvious malformed/web/list rows while preserving enough FineWeb sentence mass to test the leader-like source+rewrite structure.\n\n", f"JSON: `{meta_path}`\n\nSamples: `{sample_path}`\n"]
    note.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status":meta["status"], "out":str(meta_path), "note":str(note), "tier_words":{t:results[t]["words"] for t in tiers}, "tier_rows":{t:results[t]["rows"] for t in tiers}}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
