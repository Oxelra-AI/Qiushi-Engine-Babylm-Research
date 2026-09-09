#!/usr/bin/env python3
"""Prepare the source-by-rewrite FineWeb sentence question without launching GPU work.

The active H100 work is the research/007 matched SimpleWiki semantic-view pair.
This script only prepares the next source asset requested by the scientific route:
if the same-source SimpleWiki semantic variation effect is weak, test a stronger
leader-like question on cached FineWeb complete sentences:

    broader factual FineWeb source sentences alone
    versus
    the same FineWeb source sentences coupled to faithful Qwen simplifications.

It uses the complete sentence sources extracted in research rather than the earlier
seqsafe96 fragments.  It writes prompts and a design note, but no generation,
training or evaluation.
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
from typing import Any, Iterable

SRC_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_sources/fineweb_sentence_sources.jsonl")
OUT_DIR_DEFAULT = pathlib.Path("experiments/archive/representation_and_objectives/training/data/fineweb_sentence_rewrite")
NOTE_DEFAULT = pathlib.Path("research/notes/representation_and_objectives/fineweb_sentence_rewrite_source_by_rewrite.md")
LEADER_README = pathlib.Path("research/documents/initial_model_studies/data/leader_package_revision_124/dataset_readme__README.md")

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?|[£$€]\s*\d+(?:\.\d+)?|\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?")
NUM_RE = re.compile(r"(?:[£$€]\s*)?\d+(?:[.,:/-]\d+)*(?:%|st|nd|rd|th)?", re.I)
ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z'’.-]{2,}(?:\s+[A-Z][A-Za-z'’.-]{2,}){0,3})\b")

STOP = {
    "The", "This", "That", "These", "Those", "There", "When", "Where", "What", "How", "Why",
    "Because", "For", "And", "But", "New", "All", "Most", "Some", "Many", "First", "After",
    "Before", "During", "Then", "Each", "Every", "Through", "Early", "Originally", "General",
}
RELATION_CUES = {
    " is ", " are ", " was ", " were ", " became ", " becomes ", " born ", " died ", " founded ",
    " located ", " called ", " known ", " used ", " includes ", " contains ", " because ",
    " therefore ", " caused ", " led ", " replaced ", " developed ", " published ", " built ",
    " served ", " won ", " established ", " created ", " part of ", " member of ",
}
DOMAIN_TERMS = {
    "science_physical": {"volcanic","lava","glacier","molecular","cloud","matter","galaxy","energy","electric","chemical","virus","species","weather","planet","star","biology","engineering","water","geologic","hydrologic"},
    "geography_places": {"city","province","county","district","region","capital","river","mountain","island","country","state","community","town","village","located"},
    "people_history": {"born","died","served","mayor","governor","senate","war","leader","president","minister","writer","battle","settled","incorporated"},
    "institutions_society": {"company","university","government","council","education","law","organization","competition","factory","school","system","church","library"},
    "media_culture": {"film","album","band","song","movie","game","book","published","released","television","novel","music"},
    "quant_numeric": {"january","february","march","april","may","june","july","august","september","october","november","december","km","metres","feet","century","year","million","percent"},
    "causal_relational": {"because","therefore","caused","effect","result","led","formed","became","created","due","reason","replaced","allows","requires","helps","influenced"},
}

SYSTEM = (
    "You rewrite English sentences for a small masked language model. Preserve exactly the same meaning, "
    "including every named entity, number, date, quantity, and relation. Do not add facts. Output only the simplified sentence."
)


def norm_text(text: str) -> str:
    return " ".join(str(text).replace("\u00a0", " ").split())


def wc(text: str) -> int:
    return len(norm_text(text).split())


def stats(vals: Iterable[int | float]) -> dict[str, Any]:
    xs = sorted(list(vals))
    if not xs:
        return {"n": 0}
    def q(p: float):
        return xs[min(len(xs) - 1, max(0, round((len(xs) - 1) * p)))]
    return {
        "n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.mean(xs),
        "median": statistics.median(xs), "p95": q(0.95), "p99": q(0.99), "max": xs[-1], "sum": sum(xs),
    }


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entities(text: str) -> list[str]:
    out: list[str] = []
    for m in ENTITY_RE.finditer(text):
        e = " ".join(m.group(0).split()).strip(" .,;:()[]{}\"“”‘’")
        if not e:
            continue
        if e.split()[0] in STOP and len(e.split()) == 1:
            continue
        out.append(e)
    return sorted(set(out))


def numbers(text: str) -> list[str]:
    return sorted(set(re.sub(r"\s+", "", m.group(0)) for m in NUM_RE.finditer(text)))


def domain_hits(text: str) -> list[str]:
    toks = {m.group(0).lower().strip("'’.-") for m in WORD_RE.finditer(text) if len(m.group(0)) >= 3}
    return sorted(k for k, terms in DOMAIN_TERMS.items() if toks & terms)


def content_score(text: str, ent_count: int, num_count: int) -> float:
    low = " " + text.lower() + " "
    rel = sum(1 for c in RELATION_CUES if c in low)
    dom = len(domain_hits(text))
    return 0.60 * ent_count + 0.35 * num_count + 0.45 * rel + 0.25 * dom + min(2.0, wc(text) / 30.0)


def load_sources(path: pathlib.Path, require_clean_row: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            text = norm_text(o["text"])
            words = int(o.get("words", wc(text)))
            if words != wc(text):
                raise RuntimeError(f"word mismatch sentence_id={o.get('sentence_id')} field={words} actual={wc(text)}")
            flags = list(o.get("source_row_quality_flags") or [])
            if require_clean_row and flags:
                continue
            ent = entities(text)
            num = numbers(text)
            row = {
                "sentence_id": int(o["sentence_id"]),
                "source": "fineweb_complete_sentence_cached_initial_model_studies",
                "doc_id": str(o.get("doc_id", "")),
                "source_row": o.get("source_row"),
                "sent_index_in_row": o.get("sent_index_in_row"),
                "text": text,
                "words": words,
                "entities": ent,
                "numbers": num,
                "domain_hits": domain_hits(text),
                "content_score": content_score(text, len(ent), len(num)),
                "source_row_quality_flags": flags,
            }
            rows.append(row)
    return rows


def stratified_pilot(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_doc[r["doc_id"]].append(r)
    # One high-scoring sentence per doc first, then fill by score diversity.
    docs = list(by_doc)
    rng.shuffle(docs)
    selected: list[dict[str, Any]] = []
    for d in docs:
        selected.append(max(by_doc[d], key=lambda x: (x["content_score"], x["words"])))
        if len(selected) >= n:
            break
    if len(selected) < n:
        seen = {r["sentence_id"] for r in selected}
        ordered = sorted(rows, key=lambda x: (-x["content_score"], x["doc_id"], x["sentence_id"]))
        # Add a shuffled tail from every score band to avoid only entity-heavy biography/geography.
        bands: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for rank, r in enumerate(ordered):
            bands[int(10 * rank / max(1, len(ordered)))].append(r)
        band_ids = list(sorted(bands))
        while len(selected) < n and band_ids:
            progressed = False
            for b in band_ids:
                pool = bands[b]
                if not pool:
                    continue
                r = pool.pop(0)
                if r["sentence_id"] in seen:
                    continue
                selected.append(r); seen.add(r["sentence_id"]); progressed = True
                if len(selected) >= n:
                    break
            if not progressed:
                break
    rng.shuffle(selected)
    return selected[:n]


def make_prompt(row: dict[str, Any]) -> dict[str, Any]:
    sid = row["sentence_id"]
    text = row["text"]
    prompt = (
        "Simplify the sentence below into clear plain English while preserving exactly the same facts. "
        "Keep every named entity, number, date, quantity, and relation from the source. "
        "Do not add examples, explanations, headings, bullet points, or background. "
        "Output one grammatical English sentence only.\n\nSOURCE SENTENCE:\n" + text
    )
    return {
        "prompt_id": f"fwsimp_sent_{sid:06d}",
        "typ": "simplification",
        "source": "cached_fineweb_complete_sentence_qwen_simplification",
        "sentence_id": sid,
        "doc_id": row["doc_id"],
        "source_row": row.get("source_row"),
        "sent_index_in_row": row.get("sent_index_in_row"),
        "source_text": text,
        "source_words": row["words"],
        "source_entities": row["entities"],
        "source_numbers": row["numbers"],
        "domain_hits": row["domain_hits"],
        "content_score": round(float(row["content_score"]), 6),
        "system": SYSTEM,
        "prompt": prompt,
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
    ap.add_argument("--allow-flagged-source-rows", action="store_true")
    args = ap.parse_args()

    src = pathlib.Path(args.sources)
    out_dir = pathlib.Path(args.out_dir)
    require_clean = not args.allow_flagged_source_rows
    rows = load_sources(src, require_clean_row=require_clean)
    if not rows:
        raise RuntimeError("no source sentences selected")
    rows_sorted = sorted(rows, key=lambda r: (r["doc_id"], int(r["source_row"] or -1), int(r["sent_index_in_row"] or -1), r["sentence_id"]))
    prompts = [make_prompt(r) for r in rows_sorted]
    pilot_rows = stratified_pilot(rows_sorted, min(args.pilot_size, len(rows_sorted)), args.seed)
    pilot_prompts = [make_prompt(r) for r in pilot_rows]

    selected_path = out_dir / "fineweb_complete_sentence_sources_selected.jsonl"
    full_prompt_path = out_dir / "fineweb_complete_sentence_simplification_prompts_all.jsonl"
    pilot_prompt_path = out_dir / f"fineweb_complete_sentence_simplification_prompts_pilot{len(pilot_prompts)}.jsonl"
    metadata_path = out_dir / "source_by_rewrite_prompt_metadata.json"
    sample_path = out_dir / "prompt_samples.json"
    write_jsonl(selected_path, rows_sorted)
    write_jsonl(full_prompt_path, prompts)
    write_jsonl(pilot_prompt_path, pilot_prompts)

    ent_counts = [len(r["entities"]) for r in rows_sorted]
    num_counts = [len(r["numbers"]) for r in rows_sorted]
    doc_counter = collections.Counter(r["doc_id"] for r in rows_sorted)
    domain_counter = collections.Counter(h for r in rows_sorted for h in r["domain_hits"])
    source_words = sum(r["words"] for r in rows_sorted)
    # Use the already-measured accepted SimpleWiki simplification length ratio (mean ~0.90) only as a planning prior.
    expected_rewrite_words_low = round(0.75 * source_words)
    expected_rewrite_words_mid = round(0.90 * source_words)
    expected_rewrite_words_high = round(1.05 * source_words)
    meta: dict[str, Any] = {
        "status": "FINEWEB_SENTENCE_REWRITE_PROMPTS_PREPARED",
        "purpose": "future matched test of broad cached FineWeb source sentences alone versus the same sentences coupled to faithful Qwen simplifications; no GPU job launched",
        "source_jsonl": str(src),
        "source_sha256": sha256_file(src),
        "leader_readme_reference": str(LEADER_README) if LEADER_README.exists() else "missing",
        "require_clean_source_rows": require_clean,
        "selected_sources": len(rows_sorted),
        "selected_source_words": source_words,
        "unique_docs": len(doc_counter),
        "top_docs_by_sentence_count": doc_counter.most_common(10),
        "word_stats": stats([r["words"] for r in rows_sorted]),
        "entity_count_stats": stats(ent_counts),
        "number_count_stats": stats(num_counts),
        "domain_hit_counts": dict(domain_counter),
        "full_prompts": str(full_prompt_path),
        "full_prompts_count": len(prompts),
        "full_prompt_sha256": sha256_file(full_prompt_path),
        "pilot_prompts": str(pilot_prompt_path),
        "pilot_prompts_count": len(pilot_prompts),
        "pilot_prompt_sha256": sha256_file(pilot_prompt_path),
        "selected_sources_path": str(selected_path),
        "selected_sources_sha256": sha256_file(selected_path),
        "expected_pair_words_if_accept_all": {
            "source_words": source_words,
            "rewrite_words_low_ratio_0p75": expected_rewrite_words_low,
            "rewrite_words_mid_ratio_0p90": expected_rewrite_words_mid,
            "rewrite_words_high_ratio_1p05": expected_rewrite_words_high,
            "pair_words_low": source_words + expected_rewrite_words_low,
            "pair_words_mid": source_words + expected_rewrite_words_mid,
            "pair_words_high": source_words + expected_rewrite_words_high,
        },
        "intended_generation_not_launched": {
            "model": "qwen3.5-9b or the current local Qwen model allowed by official accounting",
            "prompt_file": str(full_prompt_path),
            "max_new_tokens_hint": 72,
            "temperature_hint": 0.1,
            "batch_size_hint": 64,
            "reason_not_launched": "both H100s are occupied by the active semantic-view treatment/control pair; next allocation should wait for paired trajectory and frontier_consolidation evidence",
        },
        "future_source_by_rewrite_contrast": {
            "treatment": "for each accepted source sentence, concatenate original sentence + accepted simplification",
            "control": "use the identical source-sentence set and exact row-length sequence, filling extra words by same-source repetition rather than adding different facts",
            "shared_filler": "remaining words drawn from the same official pool rows in identical order after the source/rewrite prefix",
            "screening": "reject outputs with added numbers, missing source numbers, low entity retention, more than two new named-entity tokens, newline/disclaimer/artifact strings, fragment endings, extreme length ratios, or near-copy outputs",
            "required_before_training": "materialized arms must pass exact word count, exact row-length matching, identical filler, source-string reconstruction, and seq256 visibility checks",
        },
        "not_a_training_result": True,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sample = {
        "first_prompts": prompts[:5],
        "pilot_prompts_first": pilot_prompts[:5],
        "high_score_sources": sorted(rows_sorted, key=lambda r: (-r["content_score"], -len(r["entities"]), -len(r["numbers"])))[:8],
    }
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note = pathlib.Path(args.note)
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "# research FineWeb complete-sentence source-by-rewrite question\n\n"
        "The active H100 work remains the matched SimpleWiki semantic-view treatment/control pair. No new GPU generation, training, or evaluation job was started here.\n\n"
        "## Why this asset exists\n"
        "The public leader's README describes sentence-level FineWeb originals followed by Qwen-generated simplifications. The exact train file remains unavailable locally, so the legal route is to reconstruct the mechanism with cached public FineWeb sentences and count every generated token as training data if used. The earlier research fragment prompts are superseded: this file uses complete sentence-like FineWeb spans.\n\n"
        "## Prepared sources and prompts\n"
        f"- Source sentences selected: {len(rows_sorted):,} from {len(doc_counter):,} docs.\n"
        f"- Source word mass: {source_words:,}.\n"
        f"- Mean sentence length: {meta['word_stats']['mean']:.2f} words; p95 {meta['word_stats']['p95']} words; max {meta['word_stats']['max']} words.\n"
        f"- Full simplification prompts: `{full_prompt_path}` ({len(prompts):,}).\n"
        f"- Pilot prompt subset for a future faithfulness slice: `{pilot_prompt_path}` ({len(pilot_prompts):,}).\n"
        f"- Metadata: `{metadata_path}`. Samples: `{sample_path}`.\n\n"
        "## Scientific comparison to run only after current evidence arrives\n"
        "If the SimpleWiki same-source semantic-view contrast is weak, the next stronger question is not a raw source swap. It is: does broad factual FineWeb experience become more useful when the same source sentences are coupled to faithful simpler rewrites? The matched contrast should use the identical sentence set in both arms. Treatment rows concatenate original sentence plus accepted simplification. Control rows use the same original sentence and match the treatment row length by same-source repetition. The remaining official filler must be identical.\n\n"
        "## Expected scale\n"
        f"Using the observed SimpleWiki simplification length ratio as only a planning prior, accepting all selected sentence rewrites would give roughly {source_words + expected_rewrite_words_mid:,} pair words at a 0.90 rewrite/source ratio, with a plausible range {source_words + expected_rewrite_words_low:,}--{source_words + expected_rewrite_words_high:,}. This is substantially broader FineWeb factual coverage than the current SimpleWiki semantic packet mass and directly tests a leader-like data structure.\n\n"
        "## Before any H100 training\n"
        "Generate only after the active pair and frontier_consolidation evidence make this the best next allocation. Then screen source/output pairs for number preservation, entity retention, no added facts by automated proxies, complete sentence form, length range, non-copy transformation, and no prompt artifacts. Materialize matched rows only from accepted rewrites; verify exact 10M words, matched row lengths, identical filler, source reconstruction, and seq256 visibility before training.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": meta["status"],
        "selected_sources": len(rows_sorted),
        "selected_source_words": source_words,
        "unique_docs": len(doc_counter),
        "full_prompts": str(full_prompt_path),
        "pilot_prompts": str(pilot_prompt_path),
        "metadata": str(metadata_path),
        "note": str(note),
        "expected_pair_words_mid": source_words + expected_rewrite_words_mid,
        "not_launched": True,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
