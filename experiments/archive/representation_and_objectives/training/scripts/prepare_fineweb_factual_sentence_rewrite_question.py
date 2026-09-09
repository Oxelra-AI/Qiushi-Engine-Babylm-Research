#!/usr/bin/env python3
"""Prepare the refined FineWeb source-by-rewrite question from factual complete sentences.

This supersedes research's broad all-sentence prompt asset.  The broad asset used
complete-looking sentences, but inspection showed dialogue/literary fragments,
trailing abbreviations, and list rows.  For the next H100 allocation, if needed,
the source-by-rewrite question should use factual/expository sentences only:

    same cached FineWeb factual sentence sources alone
    versus
    the same sources plus faithful Qwen simplifications.

This script is CPU-only. It writes source/prompt/metadata/note files and does not
launch generation, training, or evaluation.
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
OUT_DIR_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_factual_sentence_rewrite_source_by_rewrite.md")
BROAD_NOTE = pathlib.Path("research/notes/representation_and_objectives/fineweb_sentence_rewrite_source_by_rewrite.md")
LEADER_README = pathlib.Path("research/documents/initial_model_studies/data/leader_package_revision_124/dataset_readme__README.md")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
CAP_WORD_RE = re.compile(r"\b[A-Z][A-Za-z'’.-]{2,}\b")
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")

BAD_TRAILING_ABBR = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "no.", "fig.", "etc."}
BAD_START_WORDS = {
    "i", "you", "we", "he", "she", "they", "it", "me", "my", "your", "our", "his", "her", "their",
    "of", "and", "but", "or", "so", "then", "because", "though", "although", "while", "which", "who",
    "whom", "whose", "this", "that", "these", "those", "there",
}
BAD_SUBSTR = [
    "click here", "follow this link", "read more", "press release", "privacy policy", "terms of use",
    "all rights reserved", "subscribe", "download pdf", "example :", "example:", "related persons",
    "names of", "multiple choice", "(a)", "(b)", "lorem ipsum", "cookie policy", "advertisement",
]
DIALOGUE_MARKERS = [" cried ", " said ", " replied ", " asked ", " shouted ", " exclaimed "]

RELATION_CUES = {
    " is ", " are ", " was ", " were ", " became ", " becomes ", " born ", " died ", " founded ",
    " located ", " called ", " known ", " used ", " uses ", " includes ", " contains ", " because ",
    " therefore ", " caused ", " led ", " replaced ", " developed ", " published ", " built ",
    " served ", " won ", " established ", " created ", " part of ", " member of ", " consists ",
    " belongs ", " formed ", " opened ", " invented ", " discovered ", " produced ", " released ",
}
DOMAIN_TERMS = {
    "science_physical": {"volcanic","lava","glacier","molecular","cloud","matter","galaxy","energy","electric","chemical","virus","species","weather","planet","star","biology","engineering","water","geologic","hydrologic","climate","emissions","protein","cell","disease","medical","research"},
    "geography_places": {"city","province","county","district","region","capital","river","mountain","island","country","state","community","town","village","located","basin","border","population"},
    "people_history": {"born","died","served","mayor","governor","senate","war","leader","president","minister","writer","battle","settled","incorporated","dynasty","king","queen","election"},
    "institutions_society": {"company","university","government","council","education","law","organization","competition","factory","school","system","church","library","policy","court","military"},
    "media_culture": {"film","album","band","song","movie","game","book","published","released","television","novel","music","series"},
    "quant_numeric": {"january","february","march","april","may","june","july","august","september","october","november","december","km","metres","feet","century","year","million","percent"},
    "causal_relational": {"because","therefore","caused","effect","result","led","formed","became","created","due","reason","replaced","allows","requires","helps","influenced","responsible"},
}

SYSTEM = (
    "You simplify factual English sentences for a small masked language model. Preserve exactly the same meaning, "
    "including every named entity, number, date, quantity, and relation. Do not add facts. Output only the simplified sentence."
)


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def wc(text: str) -> int:
    return len(norm_text(text).split())


def words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = sorted(list(vals))
    if not xs:
        return {"n": 0}
    def q(p: float):
        return xs[min(len(xs)-1, max(0, round((len(xs)-1)*p)))]
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.mean(xs), "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": xs[-1], "sum": sum(xs)}


def balanced_quotes(text: str) -> bool:
    # Do not require all quote styles perfectly balanced in web data, but reject clear dangling dialogue.
    for a, b in [("“", "”"), ("‘", "’")]:
        if text.count(a) != text.count(b):
            return False
    if text.count('"') % 2:
        return False
    return True


def balanced_brackets(text: str) -> bool:
    return text.count("(") == text.count(")") and text.count("[") == text.count("]") and text.count("{") == text.count("}")


def nonlatin_letter_frac(text: str) -> float:
    letters = 0
    nonlatin = 0
    for ch in text:
        if unicodedata.category(ch).startswith("L"):
            letters += 1
            if "LATIN" not in unicodedata.name(ch, ""):
                nonlatin += 1
    return nonlatin / letters if letters else 0.0


def alpha_frac(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(unicodedata.category(c).startswith("L") for c in chars) / len(chars)


def numbers(text: str) -> list[str]:
    return sorted(set(re.sub(r"\s+", "", m.group(0)) for m in NUM_RE.finditer(text)))


def conservative_entities(text: str) -> list[str]:
    toks = text.split()
    out: list[str] = []
    # Capture acronyms and multiword names; capture single capitalized words only away from sentence start.
    for m in ACRONYM_RE.finditer(text):
        out.append(m.group(0))
    cap_positions: list[tuple[int, str]] = []
    char_to_tok: list[int] = []
    pos = 0
    spans = []
    for i, tok in enumerate(toks):
        start = text.find(tok, pos)
        if start < 0:
            start = pos
        spans.append((start, start + len(tok), tok))
        pos = start + len(tok)
    for m in CAP_WORD_RE.finditer(text):
        tok_i = 0
        for j, (a, b, _) in enumerate(spans):
            if a <= m.start() < b:
                tok_i = j; break
        cap_positions.append((tok_i, m.group(0).strip(".,;:()[]{}\"“”‘’")))
    i = 0
    while i < len(cap_positions):
        tok_i, val = cap_positions[i]
        seq = [val]
        j = i + 1
        prev_tok_i = tok_i
        while j < len(cap_positions) and cap_positions[j][0] <= prev_tok_i + 1:
            seq.append(cap_positions[j][1]); prev_tok_i = cap_positions[j][0]; j += 1
        if len(seq) >= 2:
            out.append(" ".join(seq))
        else:
            # Single-word entity if not just sentence-initial capitalization, or if acronym-like/internal-cap already captured.
            if tok_i != 0 and len(seq[0]) >= 4:
                out.append(seq[0])
        i = j
    cleaned = []
    bad_single = {"The", "This", "That", "These", "Those", "There", "When", "Where", "What", "How", "Why", "Because", "For", "And", "But", "New", "All", "Most", "Some", "Many", "First", "After", "Before", "During", "Then", "Each", "Every", "Through", "Early", "Research", "Example", "Names", "Related", "Various", "Clinical", "Water", "Skin", "Blood", "Food", "Lungs", "Entire", "Body"}
    for e in out:
        e = " ".join(e.split()).strip(" .,;:()[]{}\"“”‘’")
        if not e:
            continue
        if len(e.split()) == 1 and e in bad_single:
            continue
        cleaned.append(e)
    return sorted(set(cleaned))


def domain_hits(text: str) -> list[str]:
    toks = {w.lower().strip("'’.-") for w in words(text) if len(w) >= 3}
    return sorted(k for k, terms in DOMAIN_TERMS.items() if toks & terms)


def relation_count(text: str) -> int:
    low = " " + text.lower() + " "
    return sum(1 for c in RELATION_CUES if c in low)


def content_score(text: str, ents: list[str], nums: list[str], doms: list[str]) -> float:
    return 0.75 * len(ents) + 0.40 * len(nums) + 0.55 * relation_count(text) + 0.30 * len(doms) + min(2.0, wc(text) / 28.0)


def rejection_reasons(text: str, source_flags: list[str], ents: list[str], nums: list[str], doms: list[str]) -> list[str]:
    text = norm_text(text)
    low = text.lower()
    toks = text.split()
    reasons: list[str] = []
    if source_flags:
        reasons.append("source_row_flag")
    n = len(toks)
    if n < 10:
        reasons.append("too_short_for_rewrite_test")
    if n > 48:
        reasons.append("too_long_or_multi_clause")
    if not text.endswith(('.', '!', '?', '.”', '."', '?”', '?"')):
        reasons.append("bad_terminal_punctuation")
    last = toks[-1].lower().strip("\"“”‘’()[]{}") if toks else ""
    if last in BAD_TRAILING_ABBR:
        reasons.append("trailing_abbreviation_fragment")
    if toks:
        first = re.sub(r"[^A-Za-z]+", "", toks[0]).lower()
        if first in BAD_START_WORDS:
            reasons.append("context_dependent_start")
    if not balanced_quotes(text):
        reasons.append("unbalanced_or_dialogue_quote")
    if not balanced_brackets(text):
        reasons.append("unbalanced_bracket")
    if any(s in low for s in BAD_SUBSTR):
        reasons.append("web_list_or_navigation_artifact")
    if any(s in low for s in DIALOGUE_MARKERS) and ('"' in text or '“' in text or '”' in text):
        reasons.append("dialogue_like")
    if text.count(';') >= 2:
        reasons.append("semicolon_list")
    if text.count(':') >= 2:
        reasons.append("colon_list")
    if text.count(',') / max(1, n) > 0.22:
        reasons.append("comma_dense_list")
    if len(ents) >= 9 and n < 45:
        reasons.append("entity_list_like")
    if len(nums) >= 5 and n < 45:
        reasons.append("number_list_like")
    if sum(any(c.isdigit() for c in w) for w in toks) / max(1, n) > 0.22:
        reasons.append("digit_dense")
    if alpha_frac(text) < 0.62 or nonlatin_letter_frac(text) > 0.02:
        reasons.append("character_quality")
    # Need factual/expository signal: enough self-contained named/numeric/domain material plus a relation cue.
    has_substance = len(ents) >= 1 or len(nums) >= 1 or bool(doms)
    has_relation = relation_count(text) >= 1
    if not has_substance:
        reasons.append("low_factual_substance")
    if not has_relation:
        reasons.append("low_relation_signal")
    return reasons


def load_and_filter(path: pathlib.Path) -> tuple[list[dict[str, Any]], collections.Counter, list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    reject = collections.Counter()
    reject_examples: list[dict[str, Any]] = []
    totals = {"input_rows": 0, "input_words": 0, "clean_source_rows": 0, "clean_source_words": 0}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line); totals["input_rows"] += 1
            text = norm_text(o["text"]); w = int(o.get("words", wc(text)))
            if w != wc(text):
                raise RuntimeError(f"word mismatch sentence_id={o.get('sentence_id')} field={w} actual={wc(text)}")
            totals["input_words"] += w
            flags = list(o.get("source_row_quality_flags") or [])
            if not flags:
                totals["clean_source_rows"] += 1; totals["clean_source_words"] += w
            ents = conservative_entities(text); nums = numbers(text); doms = domain_hits(text)
            reasons = rejection_reasons(text, flags, ents, nums, doms)
            if reasons:
                for r in reasons:
                    reject[r] += 1
                if len(reject_examples) < 16:
                    reject_examples.append({"sentence_id": o.get("sentence_id"), "doc_id": str(o.get("doc_id", "")), "words": w, "reasons": reasons, "text": text})
                continue
            kept.append({
                "sentence_id": int(o["sentence_id"]),
                "source": "fineweb_factual_complete_sentence_cached_initial_model_studies",
                "doc_id": str(o.get("doc_id", "")),
                "source_row": o.get("source_row"),
                "sent_index_in_row": o.get("sent_index_in_row"),
                "text": text,
                "words": w,
                "entities": ents,
                "numbers": nums,
                "domain_hits": doms,
                "relation_count": relation_count(text),
                "content_score": round(content_score(text, ents, nums, doms), 6),
            })
    return kept, reject, reject_examples, totals


def stratified_subset(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    selected: list[dict[str, Any]] = []
    docs = list(by_doc); rng.shuffle(docs)
    for d in docs:
        selected.append(max(by_doc[d], key=lambda r: (r["content_score"], r["words"])))
        if len(selected) >= n:
            break
    if len(selected) < n:
        seen = {r["sentence_id"] for r in selected}
        ordered = sorted(rows, key=lambda r: (-r["content_score"], r["doc_id"], r["sentence_id"]))
        for r in ordered:
            if r["sentence_id"] in seen:
                continue
            selected.append(r); seen.add(r["sentence_id"])
            if len(selected) >= n:
                break
    rng.shuffle(selected)
    return selected[:n]


def make_prompt(row: dict[str, Any]) -> dict[str, Any]:
    text = row["text"]
    return {
        "prompt_id": f"fwfactsimp_sent_{row['sentence_id']:06d}",
        "typ": "simplification",
        "source": "cached_fineweb_factual_sentence_qwen_simplification",
        "sentence_id": row["sentence_id"],
        "doc_id": row["doc_id"],
        "source_row": row.get("source_row"),
        "sent_index_in_row": row.get("sent_index_in_row"),
        "source_text": text,
        "source_words": row["words"],
        "source_entities": row["entities"],
        "source_numbers": row["numbers"],
        "domain_hits": row["domain_hits"],
        "relation_count": row["relation_count"],
        "content_score": row["content_score"],
        "system": SYSTEM,
        "prompt": (
            "Simplify the factual sentence below into clear plain English while preserving exactly the same facts. "
            "Keep every named entity, number, date, quantity, and relation from the source. "
            "Do not add examples, explanations, headings, bullet points, background, or new facts. "
            "Output one grammatical English sentence only.\n\nSOURCE SENTENCE:\n" + text
        ),
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(SRC_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--note", default=str(NOTE_DEFAULT))
    ap.add_argument("--pilot-size", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=82910)
    args = ap.parse_args()

    sources = pathlib.Path(args.sources)
    out_dir = pathlib.Path(args.out_dir)
    kept, reject, reject_examples, totals = load_and_filter(sources)
    kept_sorted = sorted(kept, key=lambda r: (-r["content_score"], r["doc_id"], r["sentence_id"]))
    prompts = [make_prompt(r) for r in kept_sorted]
    pilot = stratified_subset(kept_sorted, min(args.pilot_size, len(kept_sorted)), args.seed)
    pilot_prompts = [make_prompt(r) for r in pilot]

    src_path = out_dir / "fineweb_factual_complete_sentence_sources_selected.jsonl"
    full_prompt_path = out_dir / "fineweb_factual_complete_sentence_simplification_prompts_all.jsonl"
    pilot_prompt_path = out_dir / f"fineweb_factual_complete_sentence_simplification_prompts_pilot{len(pilot_prompts)}.jsonl"
    meta_path = out_dir / "factual_source_by_rewrite_prompt_metadata.json"
    sample_path = out_dir / "factual_prompt_samples.json"
    write_jsonl(src_path, kept_sorted)
    write_jsonl(full_prompt_path, prompts)
    write_jsonl(pilot_prompt_path, pilot_prompts)

    doc_counter = collections.Counter(r["doc_id"] for r in kept_sorted)
    domain_counter = collections.Counter(h for r in kept_sorted for h in r["domain_hits"])
    word_sum = sum(r["words"] for r in kept_sorted)
    pair_low = word_sum + round(0.75 * word_sum)
    pair_mid = word_sum + round(0.90 * word_sum)
    pair_high = word_sum + round(1.05 * word_sum)
    meta: dict[str, Any] = {
        "status": "FINEWEB_FACTUAL_SENTENCE_REWRITE_PROMPTS_PREPARED",
        "purpose": "refined future source-by-rewrite test: broad cached FineWeb factual sentences alone versus same sentences plus faithful Qwen simplifications; no GPU job launched",
        "supersedes_broad_step010_asset": "experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite",
        "source_jsonl": str(sources),
        "source_sha256": sha256_file(sources),
        "leader_readme_reference": str(LEADER_README) if LEADER_README.exists() else "missing",
        "input_totals": totals,
        "selected_sources": len(kept_sorted),
        "selected_source_words": word_sum,
        "selected_fraction_of_step009_sources": len(kept_sorted) / max(1, totals["input_rows"]),
        "selected_word_fraction_of_step009_sources": word_sum / max(1, totals["input_words"]),
        "unique_docs": len(doc_counter),
        "top_docs_by_sentence_count": doc_counter.most_common(10),
        "word_stats": stats([r["words"] for r in kept_sorted]),
        "entity_count_stats": stats([len(r["entities"]) for r in kept_sorted]),
        "number_count_stats": stats([len(r["numbers"]) for r in kept_sorted]),
        "relation_count_stats": stats([r["relation_count"] for r in kept_sorted]),
        "domain_hit_counts": dict(domain_counter),
        "rejection_reason_counts": dict(reject.most_common()),
        "selected_sources_path": str(src_path),
        "selected_sources_sha256": sha256_file(src_path),
        "full_prompts": str(full_prompt_path),
        "full_prompts_count": len(prompts),
        "full_prompt_sha256": sha256_file(full_prompt_path),
        "pilot_prompts": str(pilot_prompt_path),
        "pilot_prompts_count": len(pilot_prompts),
        "pilot_prompt_sha256": sha256_file(pilot_prompt_path),
        "expected_pair_words_if_accept_all": {
            "source_words": word_sum,
            "rewrite_words_low_ratio_0p75": round(0.75 * word_sum),
            "rewrite_words_mid_ratio_0p90": round(0.90 * word_sum),
            "rewrite_words_high_ratio_1p05": round(1.05 * word_sum),
            "pair_words_low": pair_low,
            "pair_words_mid": pair_mid,
            "pair_words_high": pair_high,
        },
        "intended_generation_not_launched": {
            "model": "qwen3.5-9b or current local Qwen allowed under official accounting",
            "prompt_file": str(full_prompt_path),
            "pilot_prompt_file": str(pilot_prompt_path),
            "max_new_tokens_hint": 72,
            "temperature_hint": 0.1,
            "batch_size_hint": 64,
            "reason_not_launched": "active semantic-view treatment/control pair is using H100s; next allocation should be based on paired trajectory and frontier_consolidation evidence",
        },
        "future_matched_contrast": {
            "treatment": "original factual FineWeb sentence plus accepted Qwen simplification for the same sentence",
            "control": "same original factual FineWeb sentence, exact row length matched by same-source repetition; identical official filler after source/rewrite prefix",
            "interpretation": "tests whether rewrite-coupled factual web sentences improve sample efficiency beyond broader factual source exposure alone",
            "required_before_training": "screen generated rewrites, then verify exact word count, row-length identity, source reconstruction, identical filler, and seq256 visibility",
        },
        "not_a_training_result": True,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sample = {
        "first_prompts": prompts[:8],
        "pilot_prompts_first": pilot_prompts[:8],
        "high_score_sources": kept_sorted[:12],
        "reject_examples": reject_examples,
    }
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# research refined FineWeb factual sentence source-by-rewrite question\n\n"
        "The active H100 work remains the SimpleWiki semantic-view treatment/control pair. This analysis did not start Qwen generation, training, or evaluation.\n\n"
        "## Repair over the broad sentence prompt asset\n"
        "Inspection of the first broad prompts found literary dialogue, context-dependent pronoun starts, trailing abbreviation fragments, and list/table-like rows. Those would make the next experiment less about leader-like factual sentence simplification. This refined asset keeps only factual/expository, self-contained sentences with cleaner punctuation and relation signal.\n\n"
        "## Prepared source\n"
        f"- Selected factual sentence sources: {len(kept_sorted):,} / {totals['input_rows']:,}.\n"
        f"- Selected source words: {word_sum:,} / {totals['input_words']:,}.\n"
        f"- Unique docs: {len(doc_counter):,}.\n"
        f"- Mean words/source: {meta['word_stats']['mean']:.2f}; p95 {meta['word_stats']['p95']}; max {meta['word_stats']['max']}.\n"
        f"- Full prompt file: `{full_prompt_path}`.\n"
        f"- Pilot prompt file: `{pilot_prompt_path}`.\n"
        f"- Metadata: `{meta_path}`. Samples: `{sample_path}`.\n\n"
        "## Next scientific use\n"
        "If the active same-source SimpleWiki contrast is weak, use these prompts for a small Qwen faithfulness slice first. If source/output screening shows high acceptance, materialize a matched FineWeb source-by-rewrite contrast: treatment is original factual sentence plus accepted simplification; control is the same original sentence with row length matched by same-source repetition; remaining official filler is identical. This tests whether faithful simplification coupled to broad factual web experience is useful beyond broader factual source exposure alone.\n\n"
        f"At the observed SimpleWiki simplification length scale, accepting all refined sources would provide roughly {pair_mid:,} paired source+rewrite words, with planning range {pair_low:,}--{pair_high:,}.\n",
        encoding="utf-8",
    )
    # Append a warning to the broad note if it exists, without rewriting old evidence.
    if BROAD_NOTE.exists():
        with BROAD_NOTE.open("a", encoding="utf-8") as f:
            f.write("\n\n## research refinement note\nThe broad prompt asset in this note is superseded for future H100 use by the stricter factual/expository prompt asset at `experiments/archive/representation_and_objectives/training/data/fineweb_factual_sentence_rewrite`, because sample inspection found dialogue/fragments/list-like sources.\n")

    print(json.dumps({
        "status": meta["status"],
        "selected_sources": len(kept_sorted),
        "selected_source_words": word_sum,
        "unique_docs": len(doc_counter),
        "full_prompts": str(full_prompt_path),
        "pilot_prompts": str(pilot_prompt_path),
        "metadata": str(meta_path),
        "note": str(note),
        "expected_pair_words_mid": pair_mid,
        "top_rejections": reject.most_common(10),
        "not_launched": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
