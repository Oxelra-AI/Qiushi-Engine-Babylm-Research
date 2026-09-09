#!/usr/bin/env python3
"""research v2: high-precision entity-orbit corpus construction.

Compared with v1, this version narrows the resampling interface so a language-scale
Strict-Small test is interpretable as an entity-identity intervention rather than
semantic corruption:
  * replace PERSON, ORG, GPE/LOC/FAC, NORP, EVENT/WORK_OF_ART/LAW only;
  * exclude numeric/time/quantity/product/language labels;
  * require capitalized/entity-like surface form and reject lowercase common nouns;
  * preserve exact whitespace word count for every span and every row;
  * preserve within-row exact mention identity, plus simple PERSON surname links;
  * resample independently per row, so cross-document entity identity is uninformative.

The row is the available document-occurrence unit in the existing packed 10M
Strict-Small JSONL. The unchanged compact_view_reinvest stream remains the matched
reference; this script only materializes the treatment corpus and its 10x exposure
stream with exact provenance and hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import spacy

DEFAULT_INPUT_10M = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2")
# This first language-scale leg targets identity-bearing proper names only.
# PERSON, GPE/LOC/FAC and NORP have comparatively stable named-entity semantics;
# ORG is excluded because the pilot exposed cross-sentence type inconsistency
# (e.g. a person name marked ORG once and PERSON elsewhere), which would corrupt
# coreference and conflate identity randomization with semantic damage.
REPLACE_LABELS = {"PERSON", "GPE", "LOC", "FAC", "NORP"}
EXCLUDE_LABELS = {"DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT", "PRODUCT", "LANGUAGE", "ORG", "EVENT", "WORK_OF_ART", "LAW"}

FIRST = "Alice Ben Clara David Elena Felix Grace Henry Iris Jacob Kara Liam Maya Noah Olivia Peter Quinn Rosa Simon Talia Uma Victor Wendy Xavier Yara Zoe Adrian Bella Caleb Diana Ethan Freya Gavin Helena Isaac Julia Kiran Laura Milo Nina Oscar Paula Rowan Sara Theo Vera".split()
LAST = "Arden Brooks Carter Dalton Ellis Foster Green Harper Ives Jensen Keller Lewis Morgan Nolan Owens Parker Reed Stone Turner Vale Wells Young Zane".split()
PLACE = "Alden Beechport Cedarvale Dover Elmford Fairview Glenhaven Harborton Ivoryton Juniper Kestrel Linden Meadowbrook Norwood Oakridge Pinehaven Quartzford Riverbend Summit Trenton Umber Valley Weston Yorkfield".split()
ORG_HEAD = "Atlas Beacon Cobalt Ember Falcon Granite Horizon Jupiter Keystone Lumina Meridian Nimbus Oracle Pioneer Quantum Redwood Sterling Triton Union Vector Willow Zenith".split()
ORG_TYPE = "Group Institute Systems Foundation Network Council University Agency Company Project Center Laboratory".split()
NORP = "Alden Boreal Cedarian Dorian Eldan Fenian Galen Helian Ivorian Junian Kestrelian Linden Noric Orlan Rhenish Solan Vesperian".split()
MISC = "Alpha Beta Gamma Delta Epsilon Kappa Lambda Sigma Orion Nova Lumen Vesta Aria Ceres Elara Fenix Helio Ionia".split()

# Reject labels that spaCy sometimes assigns to ordinary lower-case concepts in this corpus.
LOWERCASE_ALLOW = set()  # keep empty: high precision is preferred over replacement volume.
BAD_SURFACE = {
    "marijuana", "heroin", "kratom", "bacteria", "viruses", "virus", "protein", "oxygen", "carbon",
    "earthquake", "temperature", "foundation", "the foundation", "home", "media", "internet",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"['’]s$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def ws_count(s: str) -> int:
    return len(s.split())


def title_like(s: str) -> bool:
    words = s.split()
    if not words:
        return False
    alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]
    if not alpha_words:
        return False
    # Any internal capital or all-caps abbreviation is accepted. Sentence-initial
    # one-word common nouns are rejected by BAD_SURFACE/lowercase rules below.
    return any(re.search(r"[A-Z]", w) for w in alpha_words)


def surface_ok(text: str, label: str) -> Tuple[bool, str]:
    stripped = text.strip()
    nrm = norm_text(stripped)
    if not stripped or not nrm:
        return False, "empty"
    if nrm in BAD_SURFACE:
        return False, "bad_surface"
    if re.search(r"\d", stripped):
        return False, "contains_digit"
    if ws_count(stripped) > 8:
        return False, "too_many_words"
    if not title_like(stripped):
        return False, "not_title_like"
    # Reject a single all-lower token even if spaCy marks it as PERSON/ORG.
    if len(stripped.split()) == 1 and stripped.islower() and nrm not in LOWERCASE_ALLOW:
        return False, "single_lowercase"
    # Very short one-letter aliases such as V in Virginia V are not useful identity evidence.
    if len(nrm) <= 1:
        return False, "too_short"
    return True, "ok"


def alias_words(label: str, key: str, row_idx: int, n: int) -> List[str]:
    rng = random.Random(stable_int(f"S258V2|{row_idx}|{label}|{key}|{n}"))
    n = max(1, n)
    if label == "PERSON":
        words = [rng.choice(FIRST), rng.choice(LAST)]
        while len(words) < n:
            words.insert(-1, rng.choice(FIRST + MISC))
        return words[:n]
    if label in {"GPE", "LOC", "FAC"}:
        words = [rng.choice(PLACE)] if n == 1 else [rng.choice(MISC), rng.choice(PLACE)]
        while len(words) < n:
            words.insert(-1, rng.choice(MISC + PLACE))
        return words[:n]
    if label == "ORG":
        words = [rng.choice(ORG_HEAD)] if n == 1 else [rng.choice(ORG_HEAD), rng.choice(ORG_TYPE)]
        while len(words) < n:
            words.insert(-1, rng.choice(MISC + ORG_HEAD))
        return words[:n]
    if label == "NORP":
        words = [rng.choice(NORP)]
        while len(words) < n:
            words.append(rng.choice(MISC))
        return words[:n]
    # EVENT / WORK_OF_ART / LAW: proper-name placeholders, not semantic class labels.
    words = [rng.choice(MISC)] if n == 1 else [rng.choice(MISC), rng.choice(MISC)]
    while len(words) < n:
        words.append(rng.choice(MISC + ORG_HEAD))
    return words[:n]


def build_person_links(spans: List[Tuple[int, int, str, str]]) -> Dict[Tuple[str, str], str]:
    full_by_last: Dict[str, List[str]] = defaultdict(list)
    for _, _, txt, label in spans:
        if label != "PERSON":
            continue
        toks = norm_text(txt).split()
        if len(toks) >= 2:
            key = "PERSON:full:" + " ".join(toks)
            full_by_last[toks[-1]].append(key)
    unique_last = {last: keys[0] for last, keys in full_by_last.items() if len(set(keys)) == 1}
    links: Dict[Tuple[str, str], str] = {}
    for _, _, txt, label in spans:
        if label != "PERSON":
            continue
        toks = norm_text(txt).split()
        if len(toks) == 1 and toks[0] in unique_last:
            links[(label, norm_text(txt))] = unique_last[toks[0]]
        else:
            links[(label, norm_text(txt))] = "PERSON:full:" + " ".join(toks)
    return links


def transform_doc(doc, row_idx: int):
    text = doc.text
    spans: List[Tuple[int, int, str, str]] = []
    rejected = Counter()
    last_end = -1
    for ent in doc.ents:
        label = ent.label_
        if label in EXCLUDE_LABELS:
            rejected[f"excluded_label:{label}"] += 1
            continue
        if label not in REPLACE_LABELS:
            rejected[f"unreplaced_label:{label}"] += 1
            continue
        if ent.start_char < last_end:
            rejected["overlap"] += 1
            continue
        ok, why = surface_ok(ent.text, label)
        if not ok:
            rejected[why] += 1
            continue
        spans.append((ent.start_char, ent.end_char, ent.text, label))
        last_end = ent.end_char

    person_links = build_person_links(spans)
    span_keys: List[Tuple[int, int, str, str, int, bool]] = []
    max_len_by_key: Dict[str, int] = defaultdict(int)
    label_by_key: Dict[str, str] = {}
    for s, e, txt, label in spans:
        nrm = norm_text(txt)
        wc = ws_count(txt)
        surname_only = False
        if label == "PERSON":
            key = person_links.get((label, nrm), "PERSON:full:" + nrm)
            surname_only = (len(nrm.split()) == 1 and key != "PERSON:full:" + nrm)
        else:
            key = f"{label}:exact:{nrm}"
        span_keys.append((s, e, label, key, wc, surname_only))
        max_len_by_key[key] = max(max_len_by_key[key], wc)
        label_by_key[key] = label

    # The intervention targets actual within-row identity orbits, not isolated
    # named facts. Requiring at least two mentions of the linked key both reduces
    # NER false-positive damage and preserves singleton world-knowledge coverage.
    # Full-name/surname pairs survive because build_person_links gives them one key.
    key_occ = Counter(x[3] for x in span_keys)
    kept_span_keys = [x for x in span_keys if key_occ[x[3]] >= 2]
    rejected["singleton_identity_key"] += len(span_keys) - len(kept_span_keys)
    max_len_by_key = {key: max_len for key, max_len in max_len_by_key.items() if key_occ[key] >= 2}
    label_by_key = {key: label for key, label in label_by_key.items() if key_occ[key] >= 2}

    alias_vecs = {key: alias_words(label_by_key[key], key, row_idx, max_len) for key, max_len in max_len_by_key.items()}
    pieces: List[str] = []
    cur = 0
    replacements = []
    for s, e, label, key, wc, surname_only in kept_span_keys:
        pieces.append(text[cur:s])
        vec = alias_vecs[key]
        repl_words = [vec[-1]] if (label == "PERSON" and surname_only) else vec[:wc]
        while len(repl_words) < wc:
            repl_words.append(vec[-1])
        repl = " ".join(repl_words)
        pieces.append(repl)
        replacements.append({"label": label, "original": text[s:e], "alias": repl, "words": wc, "key": key})
        cur = e
    pieces.append(text[cur:])
    new_text = "".join(pieces)
    key_counts = Counter(r["key"] for r in replacements)
    stats = {
        "entities": len(replacements),
        "labels": dict(Counter(r["label"] for r in replacements)),
        "words_replaced": sum(r["words"] for r in replacements),
        "unique_keys": len(key_counts),
        "repeated_keys": sum(1 for c in key_counts.values() if c > 1),
        "rejected": dict(rejected),
    }
    return new_text, stats, replacements


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_repeated_100m(src10: Path, dst100: Path, passes: int = 10) -> Tuple[int, int]:
    rows = 0
    words = 0
    with dst100.open("w", encoding="utf-8") as out:
        for _ in range(passes):
            with src10.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    rows += 1
                    words += int(obj["words"])
    return rows, words


def process(args):
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input10 = Path(args.input10)
    suffix = "" if args.limit_rows <= 0 else f"_pilot_{args.limit_rows}"
    output10 = out_dir / (f"entity_orbit_alias_compact_reinvest_10M{suffix}.jsonl")
    output100 = out_dir / "entity_orbit_alias_compact_reinvest_100M.jsonl"
    sample_path = out_dir / (f"entity_orbit_alias_samples{suffix}.jsonl")
    manifest_path = out_dir / (f"entity_orbit_alias_manifest{suffix}.json")
    md_path = out_dir / (f"entity_orbit_alias_manifest{suffix}.md")

    print(json.dumps({"event": "load_spacy", "model": "en_core_web_sm"}), flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    for p in [output10, sample_path]:
        if p.exists():
            p.unlink()

    counters = Counter()
    label_counts = Counter()
    reject_counts = Counter()
    samples_written = 0
    batch_objs: List[dict] = []
    batch_texts: List[str] = []
    row_idx = 0

    def flush_batch():
        nonlocal row_idx, samples_written
        if not batch_objs:
            return
        docs = list(nlp.pipe(batch_texts, batch_size=args.spacy_batch_size))
        with output10.open("a", encoding="utf-8") as out, sample_path.open("a", encoding="utf-8") as sf:
            for obj, doc in zip(batch_objs, docs):
                row_idx += 1
                orig_text = str(obj["text"])
                orig_words = int(obj.get("words", len(orig_text.split())))
                if orig_words != len(orig_text.split()):
                    raise RuntimeError(f"input word mismatch row {row_idx}: field={orig_words} actual={len(orig_text.split())}")
                new_text, st, repls = transform_doc(doc, row_idx)
                if len(new_text.split()) != orig_words:
                    counters["word_mismatch_reverted_rows"] += 1
                    new_text = orig_text
                    st = {"entities": 0, "labels": {}, "words_replaced": 0, "unique_keys": 0, "repeated_keys": 0, "rejected": st.get("rejected", {})}
                    repls = []
                new_obj = {
                    "text": new_text,
                    "words": orig_words,
                    "example_id": int(obj.get("example_id", row_idx - 1)),
                    "source": str(obj.get("source", "example_jsonl")) + "|entity_orbit_alias_v2",
                }
                out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                counters["rows"] += 1
                counters["words"] += orig_words
                counters["changed_rows"] += int(new_text != orig_text)
                counters["entity_spans"] += st["entities"]
                counters["entity_words"] += st["words_replaced"]
                counters["repeated_entity_keys_rows"] += int(st["repeated_keys"] > 0)
                label_counts.update(st["labels"])
                reject_counts.update(st.get("rejected", {}))
                if repls and samples_written < args.sample_limit:
                    sf.write(json.dumps({"row": row_idx, "example_id": new_obj["example_id"], "original": orig_text, "aliased": new_text, "replacements": repls[:24]}, ensure_ascii=False) + "\n")
                    samples_written += 1
        batch_objs.clear(); batch_texts.clear()
        if counters["rows"] % max(1, args.progress_every) == 0:
            print(json.dumps({"event": "progress", "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "entity_spans": counters["entity_spans"], "elapsed": round(time.time() - t0, 1)}), flush=True)

    for obj in iter_jsonl(input10):
        if args.limit_rows > 0 and counters["rows"] + len(batch_objs) >= args.limit_rows:
            break
        batch_objs.append(obj)
        batch_texts.append(str(obj["text"]))
        if len(batch_objs) >= args.batch_size:
            flush_batch()
    flush_batch()

    rows100 = words100 = 0
    if args.limit_rows <= 0:
        if counters["words"] != 10_000_000:
            raise RuntimeError(f"alias 10M corpus word total {counters['words']} != 10000000")
        rows100, words100 = write_repeated_100m(output10, output100, 10)
        if words100 != 100_000_000:
            raise RuntimeError(f"alias 100M stream word total {words100} != 100000000")

    manifest = {
        "status": "ENTITY_ORBIT_ALIAS_CORPUS_V2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "High-precision per-row named-entity orbit resampling for matched Strict-Small masked-LM training.",
        "document_unit": "training JSONL row; exact repeated mentions and simple PERSON surname links are kept consistent inside the row; aliases are independent across rows",
        "input10": str(input10),
        "input10_sha256": sha256_file(input10),
        "output10": str(output10),
        "output10_sha256": sha256_file(output10),
        "output100": str(output100) if args.limit_rows <= 0 else None,
        "output100_sha256": sha256_file(output100) if args.limit_rows <= 0 else None,
        "samples": str(sample_path),
        "spacy_model": "en_core_web_sm",
        "labels_replaced": sorted(REPLACE_LABELS),
        "labels_excluded": sorted(EXCLUDE_LABELS),
        "surface_filters": ["title_like", "no_digits", "not_bad_surface", "max_8_words", "no_single_lowercase"],
        "counts": dict(counters),
        "label_counts": dict(label_counts),
        "reject_counts": dict(reject_counts),
        "changed_row_fraction": counters["changed_rows"] / max(1, counters["rows"]),
        "entity_words_per_10M_words": counters["entity_words"] / max(1, counters["words"]) * 10_000_000,
        "rows100": rows100,
        "words100": words100,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text("\n".join([
        "# research v2 entity-orbit alias corpus",
        "",
        f"Input 10M: `{input10}`",
        f"Output 10M: `{output10}`",
        f"Rows/words: {counters['rows']} / {counters['words']}",
        f"Changed rows: {counters['changed_rows']} ({manifest['changed_row_fraction']:.3f})",
        f"Entity spans replaced: {counters['entity_spans']}; entity words per 10M words: {manifest['entity_words_per_10M_words']:.0f}",
        f"Rows with repeated entity keys: {counters['repeated_entity_keys_rows']}",
        f"Word-mismatch reverted rows: {counters['word_mismatch_reverted_rows']}",
        f"Label counts: `{dict(label_counts)}`",
        f"Top rejections: `{dict(reject_counts.most_common(12))}`",
        "",
        "This construction preserves exact whitespace word counts and existing tokenizer lineage. It is intentionally high precision: replacement volume is lower than v1, but semantic corruption from false labels is reduced.",
        f"Manifest JSON: `{manifest_path}`",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "entity_spans": counters["entity_spans"], "output10_sha256": manifest["output10_sha256"], "output100_sha256": manifest["output100_sha256"]}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input10", default=str(DEFAULT_INPUT_10M))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--spacy_batch_size", type=int, default=128)
    ap.add_argument("--sample_limit", type=int, default=100)
    ap.add_argument("--progress_every", type=int, default=5000)
    args = ap.parse_args()
    process(args)


if __name__ == "__main__":
    main()
