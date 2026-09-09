#!/usr/bin/env python3
"""research language-scale entity-orbit corpus construction.

Builds a Strict-Small matched corpus from the existing compact_view_reinvest 10M
training pool by resampling named-entity mentions independently for each row
(used here as the available document occurrence unit) while preserving exact
within-row mention identity for repeated exact mentions and simple PERSON
surname links. Each replacement uses the same whitespace word count as the
original span; rows that would change word count are left unchanged and counted.

Scientific purpose: test whether the research identity-orbit mechanism changes
real masked-LM learning under the same tokenizer, exposure, architecture, seed,
and schedule as the unchanged compact_view_reinvest reference. The evidence is
not a leaderboard score; it is the movement of relational EWoK/Entity and broad
unchanged tasks relative to the exact unchanged-corpus reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import spacy

USER_ROOT = Path(".").resolve()
DEFAULT_INPUT_10M = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus")
ALIAS_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE"}
NUMERIC_LABELS = {"DATE", "TIME", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "PERCENT"}

FIRST = "Alice Ben Clara David Elena Felix Grace Henry Iris Jacob Kara Liam Maya Noah Olivia Peter Quinn Rosa Simon Talia Uma Victor Wendy Xavier Yara Zoe Adrian Bella Caleb Diana Ethan Freya Gavin Helena Isaac Julia Kiran Laura Milo Nina Oscar Paula Rowan Sara Theo Vera".split()
LAST = "Arden Brooks Carter Dalton Ellis Foster Green Harper Ives Jensen Keller Lewis Morgan Nolan Owens Parker Quinn Reed Stone Turner Vale Wells Young Zane".split()
PLACE = "Amber Basin Cedar Dover Elm Fairview Grove Harbor Ivory Juniper Kestrel Linden Meadow Norwood Oakridge Pine Quartz Riverbend Summit Trenton Umber Valley Weston York".split()
ORG = "Atlas Beacon Cobalt Delta Ember Falcon Granite Horizon Ion Jupiter Keystone Lumina Meridian Nimbus Oracle Pioneer Quantum Redwood Sterling Triton Union Vector Willow Zenith".split()
TYPE = "Group Institute Systems Foundation Network Council University Agency Company Project Center Laboratory".split()
MISC = "Alpha Beta Gamma Delta Epsilon Kappa Lambda Sigma Orion Nova Lumen Vesta Aria Ceres Dorian Elara Fenix Galen Helio Ionia".split()


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


def make_alias_vector(label: str, key: str, row_idx: int, n: int) -> List[str]:
    rng = random.Random(stable_int(f"S258|{row_idx}|{label}|{key}|{n}"))
    n = max(1, n)
    if label == "PERSON":
        words = [rng.choice(FIRST), rng.choice(LAST)]
        while len(words) < n:
            words.insert(-1, rng.choice(FIRST + MISC))
        return words[:n]
    if label in {"GPE", "LOC", "FAC"}:
        if n == 1:
            return [rng.choice(PLACE)]
        words = [rng.choice(MISC), rng.choice(PLACE)]
        while len(words) < n:
            words.insert(-1, rng.choice(MISC + PLACE))
        return words[:n]
    if label == "ORG":
        if n == 1:
            return [rng.choice(ORG)]
        words = [rng.choice(ORG)]
        while len(words) < n - 1:
            words.append(rng.choice(MISC + ORG))
        words.append(rng.choice(TYPE))
        return words[:n]
    if label == "NORP":
        base = rng.choice(["Arden", "Boreal", "Cedar", "Dorian", "Eldan", "Fenian", "Galen", "Helian"])
        return ([base] + [rng.choice(MISC) for _ in range(n - 1)])[:n]
    # Products, events, works, laws, languages: category-neutral aliases.
    words = [rng.choice(MISC) for _ in range(n)]
    return words


def build_person_links(spans: List[Tuple[int, int, str, str]]) -> Dict[Tuple[str, str], str]:
    """Return normalized PERSON mention -> canonical key with simple surname links."""
    full_by_last: Dict[str, List[str]] = defaultdict(list)
    for _, _, txt, label in spans:
        if label != "PERSON":
            continue
        toks = norm_text(txt).split()
        if len(toks) >= 2:
            key = "PERSON:full:" + " ".join(toks)
            full_by_last[toks[-1]].append(key)
    unique_last = {last: keys[0] for last, keys in full_by_last.items() if len(set(keys)) == 1}
    out: Dict[Tuple[str, str], str] = {}
    for _, _, txt, label in spans:
        if label != "PERSON":
            continue
        toks = norm_text(txt).split()
        if len(toks) == 1 and toks[0] in unique_last:
            out[(label, norm_text(txt))] = unique_last[toks[0]]
        else:
            out[(label, norm_text(txt))] = "PERSON:full:" + " ".join(toks)
    return out


def transform_doc(doc, row_idx: int) -> Tuple[str, dict, list]:
    text = doc.text
    spans: List[Tuple[int, int, str, str]] = []
    last_end = -1
    for ent in doc.ents:
        label = ent.label_
        if label in NUMERIC_LABELS or label not in ALIAS_LABELS:
            continue
        if ent.start_char < last_end:
            continue
        if not ent.text.strip() or ws_count(ent.text) <= 0 or ws_count(ent.text) > 8:
            continue
        spans.append((ent.start_char, ent.end_char, ent.text, label))
        last_end = ent.end_char
    person_links = build_person_links(spans)

    max_len_by_key: Dict[str, int] = defaultdict(int)
    span_keys: List[Tuple[int, int, str, str, int, bool]] = []
    for s, e, txt, label in spans:
        nrm = norm_text(txt)
        wc = ws_count(txt)
        surname_only = False
        if label == "PERSON":
            key = person_links.get((label, nrm), "PERSON:full:" + nrm)
            surname_only = (len(nrm.split()) == 1 and key != "PERSON:full:" + nrm)
        else:
            key = f"{label}:exact:{nrm}"
        max_len_by_key[key] = max(max_len_by_key[key], wc)
        span_keys.append((s, e, label, key, wc, surname_only))

    alias_vecs = {key: make_alias_vector(label if key.startswith("PERSON") else key.split(":", 1)[0], key, row_idx, max_len)
                  for key, max_len in max_len_by_key.items()
                  for label in ["PERSON" if key.startswith("PERSON") else key.split(":", 1)[0]]}

    pieces: List[str] = []
    cur = 0
    replacements = []
    for s, e, label, key, wc, surname_only in span_keys:
        pieces.append(text[cur:s])
        vec = alias_vecs[key]
        if label == "PERSON" and surname_only:
            repl_words = [vec[-1]]
        else:
            repl_words = vec[:wc]
            while len(repl_words) < wc:
                repl_words.append(vec[-1])
        repl = " ".join(repl_words)
        pieces.append(repl)
        replacements.append({"label": label, "original": text[s:e], "alias": repl, "words": wc, "key": key})
        cur = e
    pieces.append(text[cur:])
    new_text = "".join(pieces)
    stats = {
        "entities": len(replacements),
        "labels": dict(Counter(r["label"] for r in replacements)),
        "words_replaced": sum(r["words"] for r in replacements),
        "unique_keys": len(set(r["key"] for r in replacements)),
        "repeated_keys": sum(1 for _, c in Counter(r["key"] for r in replacements).items() if c > 1),
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
        for p in range(passes):
            with src10.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    # Keep the unique example_id; add no per-pass field because the historical
                    # repeated stream also trains by repeated text exposure.
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    rows += 1
                    words += int(obj["words"])
    return rows, words


def process(args: argparse.Namespace) -> None:
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input10 = Path(args.input10)
    output10 = out_dir / ("entity_orbit_alias_compact_reinvest_10M.jsonl" if args.limit_rows <= 0 else f"entity_orbit_alias_pilot_{args.limit_rows}.jsonl")
    output100 = out_dir / "entity_orbit_alias_compact_reinvest_100M.jsonl"
    sample_path = out_dir / ("entity_orbit_alias_samples.jsonl" if args.limit_rows <= 0 else f"entity_orbit_alias_samples_pilot_{args.limit_rows}.jsonl")
    manifest_path = out_dir / ("entity_orbit_alias_manifest.json" if args.limit_rows <= 0 else f"entity_orbit_alias_manifest_pilot_{args.limit_rows}.json")

    print(json.dumps({"event": "load_spacy", "model": "en_core_web_sm"}), flush=True)
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])

    counters = Counter()
    label_counts = Counter()
    examples = []
    batch_objs: List[dict] = []
    batch_texts: List[str] = []
    row_idx = 0

    def flush_batch() -> None:
        nonlocal row_idx
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
                    st = {"entities": 0, "labels": {}, "words_replaced": 0, "unique_keys": 0, "repeated_keys": 0}
                    repls = []
                new_obj = {
                    "text": new_text,
                    "words": orig_words,
                    "example_id": int(obj.get("example_id", row_idx - 1)),
                    "source": str(obj.get("source", "example_jsonl")) + "|entity_orbit_alias",
                }
                out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                counters["rows"] += 1
                counters["words"] += orig_words
                counters["changed_rows"] += int(new_text != orig_text)
                counters["entity_spans"] += st["entities"]
                counters["entity_words"] += st["words_replaced"]
                counters["repeated_entity_keys_rows"] += int(st["repeated_keys"] > 0)
                label_counts.update(st["labels"])
                if repls and len(examples) < args.sample_limit:
                    rec = {"row": row_idx, "example_id": new_obj["example_id"], "original": orig_text, "aliased": new_text, "replacements": repls[:20]}
                    sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    examples.append(rec)
        batch_objs.clear(); batch_texts.clear()
        if counters["rows"] % max(1, args.progress_every) == 0:
            print(json.dumps({"event": "progress", "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "entity_spans": counters["entity_spans"], "elapsed": round(time.time() - t0, 1)}), flush=True)

    if output10.exists():
        output10.unlink()
    if sample_path.exists():
        sample_path.unlink()

    for obj in iter_jsonl(input10):
        if args.limit_rows > 0 and counters["rows"] + len(batch_objs) >= args.limit_rows:
            break
        batch_objs.append(obj)
        batch_texts.append(str(obj["text"]))
        if len(batch_objs) >= args.batch_size:
            flush_batch()
    flush_batch()

    if args.limit_rows <= 0:
        if counters["words"] != 10_000_000:
            raise RuntimeError(f"alias 10M corpus word total {counters['words']} != 10000000")
        rows100, words100 = write_repeated_100m(output10, output100, passes=10)
        if words100 != 100_000_000:
            raise RuntimeError(f"alias 100M stream word total {words100} != 100000000")
    else:
        rows100 = words100 = 0

    manifest = {
        "status": "ENTITY_ORBIT_ALIAS_CORPUS",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Per-row named-entity orbit resampling for matched Strict-Small masked-LM training.",
        "document_unit": "training JSONL row; exact repeated mentions and simple PERSON surname links are kept consistent inside the row",
        "input10": str(input10),
        "input10_sha256": sha256_file(input10),
        "output10": str(output10),
        "output10_sha256": sha256_file(output10),
        "output100": str(output100) if args.limit_rows <= 0 else None,
        "output100_sha256": sha256_file(output100) if args.limit_rows <= 0 else None,
        "samples": str(sample_path),
        "spacy_model": "en_core_web_sm",
        "labels_replaced": sorted(ALIAS_LABELS),
        "labels_excluded_numeric": sorted(NUMERIC_LABELS),
        "counts": dict(counters),
        "label_counts": dict(label_counts),
        "changed_row_fraction": counters["changed_rows"] / max(1, counters["rows"]),
        "entity_words_per_10M_words": counters["entity_words"] / max(1, counters["words"]) * 10_000_000,
        "rows100": rows100,
        "words100": words100,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = out_dir / ("entity_orbit_alias_manifest.md" if args.limit_rows <= 0 else f"entity_orbit_alias_manifest_pilot_{args.limit_rows}.md")
    md_path.write_text("\n".join([
        "# research entity-orbit alias corpus",
        "",
        f"Input 10M: `{input10}`",
        f"Output 10M: `{output10}`",
        f"Rows/words: {counters['rows']} / {counters['words']}",
        f"Changed rows: {counters['changed_rows']} ({manifest['changed_row_fraction']:.3f})",
        f"Entity spans replaced: {counters['entity_spans']}; entity words replaced per 10M words: {manifest['entity_words_per_10M_words']:.0f}",
        f"Word-mismatch reverted rows: {counters['word_mismatch_reverted_rows']}",
        f"Label counts: `{dict(label_counts)}`",
        "",
        "This construction preserves exact whitespace word counts and the existing tokenizer lineage; it deliberately changes entity identity statistics while keeping within-row coreference for exact mentions.",
        f"Manifest JSON: `{manifest_path}`",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "output10_sha256": manifest["output10_sha256"], "output100_sha256": manifest["output100_sha256"]}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input10", default=str(DEFAULT_INPUT_10M))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--spacy_batch_size", type=int, default=128)
    ap.add_argument("--sample_limit", type=int, default=80)
    ap.add_argument("--progress_every", type=int, default=5000)
    args = ap.parse_args()
    process(args)


if __name__ == "__main__":
    main()
