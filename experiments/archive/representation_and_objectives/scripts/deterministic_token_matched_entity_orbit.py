#!/usr/bin/env python3
"""research: deterministic, tokenizer-geometry-matched entity-orbit intervention.

This is the compliant companion to the research external-NER oracle arm.
It uses no learned entity detector. Candidate identity orbits are repeated
capitalized lexical atoms found by a fixed regex and stop lexicon inside each
10M-corpus row. Aliases are mined only from the same legal 10M corpus, are
resampled deterministically per source row, and are injective across distinct
identity keys inside a row.

Every accepted substitution is checked against the fixed legal16k tokenizer.
The transformed row must preserve exactly:
  * the full token-sequence length;
  * the token positions at which the trainer starts a new WWM group;
  * the whitespace word count.
Therefore attention length, truncation location, WWM group sizes, RNG-selected
mask positions, batch word exposure, and update schedule remain matched. The
100M stream is generated in the exact row order of the historical reference
100M stream (ten independent permutations, not ten in-order copies).

The detector deliberately favors precision over coverage. It does not assert
that every selected capitalized atom is a real-world entity; it asks whether
removing persistent lexical identity from unambiguous within-row repeated
proper-name-like orbits changes limited-corpus learning when the token/masking
interface is held fixed.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

DEFAULT_INPUT10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_REFERENCE100 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
DEFAULT_TOKENIZER = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/deterministic_token_matched_entity_orbit")

# Function words, sentence discourse markers, months/days, titles, and common
# capitalized corpus headings are excluded. The rule is fixed and inspectable.
STOP = set("""
a an and are as at be been being but by can could did do does doing for from had has have he her hers him his how i if in into is it its may might more most must my no nor not of on one or our ours she should so than that the their theirs them then there these they this those to too under up us was we were what when where which while who whom whose why will with would you your yours
about after again against all also among any around before between both during each either enough every few first following further here however last less many meanwhile much neither next now once only other otherwise over same second several since some still such through thus together toward until upon very via within without
january february march april may june july august september october november december monday tuesday wednesday thursday friday saturday sunday
chapter section part figure table episode volume book article page source references introduction conclusion history early later modern new old home north south east west central general national international official original final current main local public private american english british european world earth
mr mrs ms miss dr prof professor president king queen lord lady saint st sir dame pope governor general captain chief judge doctor university college school institute company corporation group project foundation council committee department association club team government state city county country
""".split())

TOKEN_RE = re.compile(r"\S+")
# One capitalized alphabetic atom, optionally containing internal apostrophe or
# hyphen. Possessive suffix is kept outside the replaced base.
CORE_RE = re.compile(r"^(?P<lead>[^A-Za-z]*)(?P<base>[A-Z][A-Za-z]*(?:[-'’][A-Za-z]+)*?)(?P<poss>['’]s)?(?P<trail>[^A-Za-z]*)$")
ANY_CORE_RE = re.compile(r"^(?P<lead>[^A-Za-z]*)(?P<base>[A-Za-z][A-Za-z]*(?:[-'’][A-Za-z]+)*?)(?P<poss>['’]s)?(?P<trail>[^A-Za-z]*)$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def source_key(obj: dict) -> str:
    text = str(obj["text"])
    eid = str(obj.get("example_id", ""))
    return eid + "|" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_atom(raw: str) -> Optional[Tuple[str, int, int]]:
    """Return (base, start-within-raw, end-within-raw), or None."""
    m = CORE_RE.match(raw)
    if not m:
        return None
    base = m.group("base")
    low = base.lower()
    if len(base) < 2 or low in STOP or len(base) > 24:
        return None
    # Exclude mixed all-caps abbreviations of length <=2 and strings with odd
    # repeated punctuation. Longer acronyms remain eligible if repeated.
    if base.isupper() and len(base) <= 2:
        return None
    return base, len(m.group("lead")), len(m.group("lead")) + len(base)


def any_core(raw: str) -> Optional[str]:
    m = ANY_CORE_RE.match(raw)
    return m.group("base") if m else None


def atom_occurrences(text: str) -> Dict[str, List[Tuple[int, int, str]]]:
    out: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    toks = list(TOKEN_RE.finditer(text))
    caplike = []
    for m in toks:
        core = any_core(m.group(0))
        caplike.append(bool(core and core[0].isupper()))
    for i, m in enumerate(toks):
        parsed = parse_atom(m.group(0))
        if parsed is None:
            continue
        # Exclude components of multi-token capitalized phrases. Replacing only
        # York in New York or Way in Milky Way would destroy a compound rather
        # than randomize one lexical identity slot.
        if (i > 0 and caplike[i - 1]) or (i + 1 < len(toks) and caplike[i + 1]):
            continue
        base, a, b = parsed
        out[base.lower()].append((m.start() + a, m.start() + b, base))
    return out


def candidate_orbits(text: str, eligible_keys: set) -> Dict[str, List[Tuple[int, int, str]]]:
    occ = atom_occurrences(text)
    ans = {}
    for key, spans in occ.items():
        if key not in eligible_keys:
            continue
        # At least two exact case-sensitive realizations of the same atom.
        by_surface = Counter(s[2] for s in spans)
        surface, n = by_surface.most_common(1)[0]
        exact = [s for s in spans if s[2] == surface]
        if n >= 2:
            ans[key] = exact
    return ans


def is_word_start(token: str) -> bool:
    return token.startswith("Ġ") or token.startswith("▁")


def tokenizer_geometry(tokenizer, text: str) -> Tuple[int, Tuple[int, ...]]:
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    starts = []
    specials = set(tokenizer.all_special_ids)
    for i, tid in enumerate(ids):
        if int(tid) in specials:
            starts.append(-1)
        else:
            ts = str(tokenizer.convert_ids_to_tokens(int(tid)))
            starts.append(1 if i == 0 or is_word_start(ts) else 0)
    return len(ids), tuple(starts)


def alias_profile(tokenizer, word: str) -> Tuple[int, int, int]:
    # Cheap pre-bucket only; final acceptance uses the whole-row geometry.
    a = len(tokenizer(word, add_special_tokens=False)["input_ids"])
    b = len(tokenizer("x " + word, add_special_tokens=False)["input_ids"])
    x = len(tokenizer("x ", add_special_tokens=False)["input_ids"])
    c = len(tokenizer(word + ".", add_special_tokens=False)["input_ids"])
    return a, b - x, c


def mine_alias_bank(input10: Path, tokenizer, min_freq: int, max_per_profile: int, max_candidate_freq: int):
    counts = Counter()
    lower_counts = Counter()
    noninitial = Counter()
    canonical = {}
    rows = 0
    for obj in iter_jsonl(input10):
        rows += 1
        text = str(obj["text"])
        for m in TOKEN_RE.finditer(text):
            core = any_core(m.group(0))
            if core is None:
                continue
            key = core.lower()
            if core[0].islower():
                lower_counts[key] += 1
                continue
            parsed = parse_atom(m.group(0))
            if parsed is None:
                continue
            base = parsed[0]
            counts[key] += 1
            canonical.setdefault(key, base)
            prev = text[:m.start()].rstrip()
            if prev and prev[-1] not in ".!?":
                noninitial[key] += 1
    # A candidate surface must never occur lower-case anywhere in the complete
    # legal 10M corpus. This removes ordinary concepts capitalized only by
    # sentence position or headings (Energy, Safety, Way, Bacteria, Research).
    eligible_keys = {k for k, c in counts.items() if c >= 2 and c <= max_candidate_freq and lower_counts[k] == 0 and noninitial[k] >= 1}
    eligible_aliases = [canonical[k] for k in eligible_keys if counts[k] >= min_freq and canonical[k].isalpha() and not canonical[k].isupper() and 3 <= len(canonical[k]) <= 14]
    eligible_aliases.sort(key=lambda w: (-counts[w.lower()], w))
    bank: Dict[Tuple[int, int, int], List[str]] = defaultdict(list)
    for w in eligible_aliases:
        p = alias_profile(tokenizer, w)
        if len(bank[p]) < max_per_profile:
            bank[p].append(w)
    meta = {
        "source_rows": rows,
        "capitalized_surface_types": len(counts),
        "eligible_candidate_keys": len(eligible_keys),
        "eligible_alias_types": len(eligible_aliases),
        "profiles": {str(k): len(v) for k, v in sorted(bank.items())},
        "top_aliases": eligible_aliases[:100],
        "min_frequency": min_freq,
        "max_candidate_global_frequency": max_candidate_freq,
        "max_per_profile": max_per_profile,
        "candidate_rule": "capitalized >=2 occurrences, >=1 non-sentence-initial, zero lower-case occurrences corpus-wide, global frequency <= max_candidate_freq, not part of a capitalized multiword phrase",
    }
    return dict(bank), eligible_keys, meta


def apply_mapping(text: str, orbit_map: Dict[str, List[Tuple[int, int, str]]], assignments: Dict[str, str]) -> str:
    edits = []
    for key, alias in assignments.items():
        for s, e, old in orbit_map[key]:
            edits.append((s, e, alias, old, key))
    out = text
    for s, e, alias, _, _ in sorted(edits, reverse=True):
        out = out[:s] + alias + out[e:]
    return out


def transform_row(obj: dict, tokenizer, bank: Dict[Tuple[int, int, int], List[str]], eligible_keys: set, max_tries: int, assignment_mode: str) -> Tuple[dict, dict, List[dict]]:
    text = str(obj["text"])
    words = int(obj.get("words", len(text.split())))
    if words != len(text.split()):
        raise RuntimeError("input words field disagrees with text")
    orbits = candidate_orbits(text, eligible_keys)
    if not orbits:
        new = dict(obj)
        return new, {"candidate_keys": 0, "assigned_keys": 0, "replaced_spans": 0, "word_ok": True, "geometry_ok": True, "injective": True}, []

    orig_geom = tokenizer_geometry(tokenizer, text)
    original_atoms = set(atom_occurrences(text))
    assignments: Dict[str, str] = {}
    used = set()
    records: List[dict] = []
    row_seed = source_key(obj)

    # Harder/more frequent orbits first. Tie-breaking is deterministic.
    keys = sorted(orbits, key=lambda k: (-len(orbits[k]), k))
    for key in keys:
        original = orbits[key][0][2]
        prof = alias_profile(tokenizer, original)
        candidates = list(bank.get(prof, []))
        seed_scope = row_seed if assignment_mode == "per_row" else "GLOBAL_STABLE"
        rng = random.Random(stable_int("S259_ALIAS|" + seed_scope + "|" + key))
        rng.shuffle(candidates)
        accepted = None
        for cand in candidates[:max_tries]:
            cl = cand.lower()
            if cl == key or cl in used or cl in original_atoms:
                continue
            trial = dict(assignments)
            trial[key] = cand
            trial_text = apply_mapping(text, orbits, trial)
            if len(trial_text.split()) != words:
                continue
            if tokenizer_geometry(tokenizer, trial_text) != orig_geom:
                continue
            accepted = cand
            break
        if accepted is None:
            continue
        assignments[key] = accepted
        used.add(accepted.lower())
        records.append({"key": key, "original": original, "alias": accepted, "occurrences": len(orbits[key]), "profile": list(prof)})

    new_text = apply_mapping(text, orbits, assignments)
    new_obj = dict(obj)
    new_obj["text"] = new_text
    new_obj["words"] = words
    new_obj["source"] = str(obj.get("source", "example_jsonl")) + f"|det_tokenmatched_orbit_{assignment_mode}"
    geometry_ok = tokenizer_geometry(tokenizer, new_text) == orig_geom
    injective = len(set(assignments.values())) == len(assignments)
    stats = {
        "candidate_keys": len(orbits),
        "assigned_keys": len(assignments),
        "replaced_spans": sum(len(orbits[k]) for k in assignments),
        "word_ok": len(new_text.split()) == words,
        "geometry_ok": geometry_ok,
        "injective": injective,
        "original_tokens": orig_geom[0],
    }
    if not stats["word_ok"] or not geometry_ok or not injective:
        raise RuntimeError(f"accepted row invariant failed: {stats}")
    return new_obj, stats, records


def sequence_hash_update(h, obj: dict) -> None:
    h.update((str(obj.get("example_id", "")) + "\t" + str(obj["words"]) + "\n").encode("utf-8"))


def process(args) -> None:
    t0 = time.time()
    input10 = Path(args.input10)
    ref100 = Path(args.reference100)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit_rows <= 0 else f"_pilot_{args.limit_rows}"
    tag = args.assignment_mode
    output10 = out_dir / f"det_tokenmatched_orbit_{tag}_10M{suffix}.jsonl"
    output100 = out_dir / f"det_tokenmatched_orbit_{tag}_100M.jsonl"
    samples_path = out_dir / f"det_tokenmatched_orbit_{tag}_samples{suffix}.jsonl"
    manifest_path = out_dir / f"det_tokenmatched_orbit_{tag}_manifest{suffix}.json"
    md_path = out_dir / f"det_tokenmatched_orbit_{tag}_manifest{suffix}.md"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(json.dumps({"event": "mine_alias_bank", "input10": str(input10), "tokenizer": str(args.tokenizer)}), flush=True)
    bank, eligible_keys, bank_meta = mine_alias_bank(input10, tokenizer, args.alias_min_freq, args.max_aliases_per_profile, args.max_candidate_global_frequency)
    print(json.dumps({"event": "alias_bank_done", "eligible_candidates": bank_meta["eligible_candidate_keys"], "eligible_aliases": bank_meta["eligible_alias_types"], "profiles": bank_meta["profiles"]}), flush=True)

    for p in [output10, samples_path]:
        if p.exists():
            p.unlink()
    counters = Counter()
    key_to_transformed: Dict[str, dict] = {}
    reference10_order_hash = hashlib.sha256()
    output10_order_hash = hashlib.sha256()
    sample_count = 0

    with output10.open("w", encoding="utf-8") as out, samples_path.open("w", encoding="utf-8") as sf:
        for idx, obj in enumerate(iter_jsonl(input10)):
            if args.limit_rows > 0 and idx >= args.limit_rows:
                break
            new_obj, st, recs = transform_row(obj, tokenizer, bank, eligible_keys, args.max_alias_tries, args.assignment_mode)
            out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            key = source_key(obj)
            if key in key_to_transformed:
                counters["duplicate_source_keys"] += 1
            key_to_transformed[key] = new_obj
            sequence_hash_update(reference10_order_hash, obj)
            sequence_hash_update(output10_order_hash, new_obj)
            counters["rows"] += 1
            counters["words"] += int(obj["words"])
            counters["candidate_rows"] += int(st["candidate_keys"] > 0)
            counters["changed_rows"] += int(new_obj["text"] != obj["text"])
            counters["candidate_keys"] += st["candidate_keys"]
            counters["assigned_keys"] += st["assigned_keys"]
            counters["replaced_spans"] += st["replaced_spans"]
            counters["geometry_failures"] += int(not st["geometry_ok"])
            counters["word_failures"] += int(not st["word_ok"])
            counters["injectivity_failures"] += int(not st["injective"])
            counters["original_tokens"] += int(st.get("original_tokens", 0))
            if recs and sample_count < args.sample_limit:
                sf.write(json.dumps({"row": idx + 1, "example_id": obj.get("example_id"), "original": obj["text"], "aliased": new_obj["text"], "assignments": recs}, ensure_ascii=False) + "\n")
                sample_count += 1
            if counters["rows"] % args.progress_every == 0:
                print(json.dumps({"event": "progress10", "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "assigned_keys": counters["assigned_keys"], "elapsed": round(time.time() - t0, 1)}), flush=True)

    rows100 = words100 = misses100 = 0
    ref100_order_hash = hashlib.sha256()
    out100_order_hash = hashlib.sha256()
    if args.limit_rows <= 0:
        if counters["words"] != 10_000_000:
            raise RuntimeError(f"10M output words={counters['words']}")
        with output100.open("w", encoding="utf-8") as out:
            for obj in iter_jsonl(ref100):
                rows100 += 1
                words100 += int(obj["words"])
                key = source_key(obj)
                tr = key_to_transformed.get(key)
                if tr is None:
                    misses100 += 1
                    raise RuntimeError(f"reference100 row missing from 10M map at row {rows100}")
                new_obj = dict(tr)
                # Retain the exact reference row metadata/order while replacing text.
                new_obj["example_id"] = obj.get("example_id", new_obj.get("example_id"))
                new_obj["words"] = int(obj["words"])
                new_obj["source"] = str(obj.get("source", "example_jsonl")) + f"|det_tokenmatched_orbit_{args.assignment_mode}"
                out.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                sequence_hash_update(ref100_order_hash, obj)
                sequence_hash_update(out100_order_hash, new_obj)
                if rows100 % (args.progress_every * 5) == 0:
                    print(json.dumps({"event": "progress100", "rows": rows100, "words": words100}), flush=True)
        if rows100 != 647400 or words100 != 100_000_000 or misses100:
            raise RuntimeError(f"100M invariant rows={rows100} words={words100} misses={misses100}")
        if ref100_order_hash.hexdigest() != out100_order_hash.hexdigest():
            raise RuntimeError("reference/output 100M example_id+word sequence differs")

    manifest = {
        "status": "DETERMINISTIC_TOKEN_MATCHED_ENTITY_ORBIT",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Strict-Small-compliant deterministic repeated-capitalized-orbit resampling with exact legal16k token/WWM geometry.",
        "detector": {"type": "fixed_regex_repeated_capitalized_atom", "learned_model": None, "stop_lexicon_size": len(STOP), "minimum_exact_occurrences": 2},
        "alias_source": "surfaces mined only from the same legal 10M input corpus",
        "assignment_mode": args.assignment_mode,
        "alias_assignment": "deterministic per source row" if args.assignment_mode == "per_row" else "deterministic corpus-stable mapping by lexical key",
        "injective_across_distinct_keys_within_row": True,
        "input10": str(input10), "input10_sha256": sha256_file(input10),
        "reference100": str(ref100), "reference100_sha256": sha256_file(ref100),
        "tokenizer": str(args.tokenizer),
        "output10": str(output10), "output10_sha256": sha256_file(output10),
        "output100": str(output100) if args.limit_rows <= 0 else None,
        "output100_sha256": sha256_file(output100) if args.limit_rows <= 0 else None,
        "samples": str(samples_path),
        "counts": dict(counters),
        "alias_bank": bank_meta,
        "geometry_contract": {"full_token_count_equal_each_changed_row": counters["geometry_failures"] == 0, "wwm_start_position_vector_equal_each_changed_row": counters["geometry_failures"] == 0, "whitespace_word_count_equal_each_row": counters["word_failures"] == 0, "injective_within_row": counters["injectivity_failures"] == 0},
        "order_contract": {"reference10_exampleid_word_sha256": reference10_order_hash.hexdigest(), "output10_exampleid_word_sha256": output10_order_hash.hexdigest(), "reference100_exampleid_word_sha256": ref100_order_hash.hexdigest() if args.limit_rows <= 0 else None, "output100_exampleid_word_sha256": out100_order_hash.hexdigest() if args.limit_rows <= 0 else None},
        "rows100": rows100, "words100": words100, "reference100_map_misses": misses100,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# research deterministic token-matched entity-orbit corpus\n\n",
        f"Rows/words: {counters['rows']} / {counters['words']}\n\n",
        f"Candidate/changed rows: {counters['candidate_rows']} / {counters['changed_rows']}\n\n",
        f"Candidate/assigned keys; replaced spans: {counters['candidate_keys']} / {counters['assigned_keys']}; {counters['replaced_spans']}\n\n",
        f"Geometry, word-count, injectivity failures: {counters['geometry_failures']} / {counters['word_failures']} / {counters['injectivity_failures']}\n\n",
        f"100M rows/words/order misses: {rows100} / {words100} / {misses100}\n\n",
        "The selector is deterministic and uses no learned external model. Every accepted row preserves the complete tokenizer word-start vector, so the historical WWM trainer samples the same mask groups and token positions under the same RNG.\n\n",
        f"Manifest: `{manifest_path}`\n",
    ]
    md_path.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"event": "done", "manifest": str(manifest_path), "rows": counters["rows"], "words": counters["words"], "changed_rows": counters["changed_rows"], "assigned_keys": counters["assigned_keys"], "geometry_failures": counters["geometry_failures"], "rows100": rows100, "words100": words100, "output10_sha256": manifest["output10_sha256"], "output100_sha256": manifest["output100_sha256"]}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input10", default=str(DEFAULT_INPUT10))
    ap.add_argument("--reference100", default=str(DEFAULT_REFERENCE100))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--sample_limit", type=int, default=100)
    ap.add_argument("--alias_min_freq", type=int, default=2)
    ap.add_argument("--max_aliases_per_profile", type=int, default=512)
    ap.add_argument("--max_alias_tries", type=int, default=200)
    ap.add_argument("--max_candidate_global_frequency", type=int, default=2000)
    ap.add_argument("--assignment_mode", choices=["per_row", "stable"], default="per_row")
    ap.add_argument("--progress_every", type=int, default=5000)
    args = ap.parse_args()
    process(args)


if __name__ == "__main__":
    main()
