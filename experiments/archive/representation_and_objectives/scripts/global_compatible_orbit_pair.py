#!/usr/bin/env python3
"""research: rebuild deterministic orbit pair with a true global stable map.

research's `stable` arm used a stable candidate ordering, but whole-row acceptance
was still row-local.  The same lexical key could receive different aliases when a
row-specific token geometry check or alias exclusion rejected the first candidate.
That makes the per_row--stable contrast scientifically ambiguous.

This builder first chooses one corpus-global alias for each lexical key and an
explicit supported row-key set.  If no single alias is compatible with all rows of
a key, it chooses the alias with maximal compatible row support and drops the
other row-key occurrences from both arms.  It then builds two corpora on exactly
the same final row-key support:

  * stable_global: key -> one alias across the complete corpus;
  * per_row_global_support: same transformed row-key support, but row-specific
    aliases whenever a compatible alternative exists.

Both arms preserve per-row word counts, legal16k token sequence length, and WWM
word-start geometry; the 100M streams follow the historical reference row order.
The output is a corpus construction, not a language-scale result.  It is meant to
make any later H100 training interpretable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path("experiments/archive/representation_and_objectives")
DEFAULT_STEP259_SCRIPT = ROOT / "scripts/deterministic_token_matched_entity_orbit.py"
DEFAULT_INPUT10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_REFERENCE100 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
DEFAULT_TOKENIZER = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_OUT = ROOT / "data/global_compatible_orbit_pair"


def load_step259(path: Path):
    spec = importlib.util.spec_from_file_location("orbit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load research script at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orbit"] = mod
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def sequence_hash_update(h, obj: dict) -> None:
    h.update((str(obj.get("example_id", "")) + "\t" + str(obj["words"]) + "\n").encode("utf-8"))


@dataclass
class RowRec:
    idx: int
    obj: dict
    key: str
    text: str
    words: int
    orbits: Dict[str, list]
    original_atoms: Set[str]
    geom: Tuple[int, Tuple[int, ...]]


def candidate_aliases_for_key(mod, bank: Dict[Tuple[int, int, int], List[str]], row_recs: List[RowRec], key: str) -> List[str]:
    """Deterministic candidate list from all tokenizer profiles seen for key."""
    profiles = []
    seen_profiles = set()
    for rr in row_recs:
        original = rr.orbits[key][0][2]
        prof = tuple(mod.alias_profile(mod._STEP260_TOKENIZER, original))  # set in main after tokenizer creation
        if prof not in seen_profiles:
            seen_profiles.add(prof)
            profiles.append(prof)
    candidates: List[str] = []
    seen = set()
    for prof in sorted(profiles):
        for cand in bank.get(prof, []):
            if cand not in seen:
                seen.add(cand)
                candidates.append(cand)
    # Stable but key-specific order; coverage, not alphabetical accident, decides the selected alias.
    rng = random.Random(stable_int("GLOBAL_ALIAS_ORDER|" + key))
    rng.shuffle(candidates)
    return candidates


def single_key_compatible(mod, tok, rr: RowRec, key: str, cand: str) -> bool:
    cl = cand.lower()
    if cl == key or cl in rr.original_atoms:
        return False
    text2 = mod.apply_mapping(rr.text, {key: rr.orbits[key]}, {key: cand})
    if len(text2.split()) != rr.words:
        return False
    return mod.tokenizer_geometry(tok, text2) == rr.geom


def choose_global_aliases(mod, tok, key_to_rows: Dict[str, List[int]], rows: List[RowRec], bank: Dict[Tuple[int, int, int], List[str]], max_candidates: int, progress_every: int):
    key_to_alias: Dict[str, str] = {}
    key_to_supported_rows: Dict[str, Set[int]] = {}
    key_stats: Dict[str, dict] = {}
    # Process high-coverage keys first. Alias equality is allowed for non-cooccurring keys; row collisions are handled later.
    keys = sorted(key_to_rows, key=lambda k: (-len(key_to_rows[k]), k))
    t0 = time.time()
    for n, key in enumerate(keys, start=1):
        refs = [rows[i] for i in key_to_rows[key]]
        candidates = candidate_aliases_for_key(mod, bank, refs, key)
        best_alias = None
        best_support: List[int] = []
        tried = 0
        for cand in candidates[:max_candidates]:
            tried += 1
            support = []
            for rr in refs:
                if single_key_compatible(mod, tok, rr, key, cand):
                    support.append(rr.idx)
            if len(support) > len(best_support):
                best_alias = cand
                best_support = support
                if len(best_support) == len(refs):
                    break
        if best_alias is not None and best_support:
            key_to_alias[key] = best_alias
            key_to_supported_rows[key] = set(best_support)
        key_stats[key] = {"rows_candidate": len(refs), "rows_supported": len(best_support), "coverage": len(best_support) / len(refs) if refs else 0.0, "alias": best_alias, "aliases_tried": tried}
        if progress_every and n % progress_every == 0:
            print(json.dumps({"event": "global_alias_progress", "keys": n, "of": len(keys), "selected": len(key_to_alias), "elapsed": round(time.time() - t0, 1)}), flush=True)
    return key_to_alias, key_to_supported_rows, key_stats


def choose_per_row_aliases(mod, tok, rr: RowRec, keys: List[str], bank: Dict[Tuple[int, int, int], List[str]], stable_aliases: Dict[str, str], max_tries: int) -> Optional[Dict[str, str]]:
    assignments: Dict[str, str] = {}
    used: Set[str] = set()
    # More constrained keys first: frequent occurrences then key name.
    for key in sorted(keys, key=lambda k: (-len(rr.orbits[k]), k)):
        original = rr.orbits[key][0][2]
        prof = tuple(mod.alias_profile(tok, original))
        candidates = list(bank.get(prof, []))
        rng = random.Random(stable_int("PER_ROW_ALIAS|" + rr.key + "|" + key))
        rng.shuffle(candidates)
        # Try to avoid the stable alias first, but keep it as fallback so support need not drop solely because of a rare bucket.
        stable = stable_aliases[key]
        candidates = [c for c in candidates if c != stable] + [stable]
        accepted = None
        for cand in candidates[:max_tries] + ([stable] if stable not in candidates[:max_tries] else []):
            cl = cand.lower()
            if cl == key or cl in rr.original_atoms or cl in used:
                continue
            trial = dict(assignments)
            trial[key] = cand
            trial_text = mod.apply_mapping(rr.text, {k: rr.orbits[k] for k in trial}, trial)
            if len(trial_text.split()) != rr.words:
                continue
            if mod.tokenizer_geometry(tok, trial_text) != rr.geom:
                continue
            accepted = cand
            break
        if accepted is None:
            return None
        assignments[key] = accepted
        used.add(accepted.lower())
    return assignments


def support_signature(row_idx: int, assignments: Dict[str, str], rr: RowRec) -> str:
    parts = []
    for key in sorted(assignments):
        parts.append(f"{key}:{len(rr.orbits[key])}")
    return f"{row_idx}|" + ",".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(DEFAULT_STEP259_SCRIPT))
    ap.add_argument("--input10", default=str(DEFAULT_INPUT10))
    ap.add_argument("--reference100", default=str(DEFAULT_REFERENCE100))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--alias_min_freq", type=int, default=2)
    ap.add_argument("--max_aliases_per_profile", type=int, default=512)
    ap.add_argument("--max_candidate_global_frequency", type=int, default=2000)
    ap.add_argument("--max_global_alias_candidates", type=int, default=512)
    ap.add_argument("--max_per_row_alias_tries", type=int, default=200)
    ap.add_argument("--sample_limit", type=int, default=80)
    ap.add_argument("--progress_every", type=int, default=10000)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit_rows <= 0 else f"_pilot_{args.limit_rows}"
    out10_per = out_dir / f"det_orbit_per_row_global_support_10M{suffix}.jsonl"
    out10_stable = out_dir / f"det_orbit_stable_global_10M{suffix}.jsonl"
    out100_per = out_dir / "det_orbit_per_row_global_support_100M.jsonl"
    out100_stable = out_dir / "det_orbit_stable_global_100M.jsonl"
    samples_path = out_dir / f"global_orbit_pair_samples{suffix}.jsonl"
    manifest_path = out_dir / f"global_orbit_pair_manifest{suffix}.json"
    md_path = out_dir / f"global_orbit_pair_manifest{suffix}.md"

    mod = load_step259(Path(args.script))
    tok = mod.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # Avoid threading tokenizer through a helper signature used only here.
    mod._STEP260_TOKENIZER = tok
    print(json.dumps({"event": "mine_alias_bank", "input10": args.input10, "tokenizer": args.tokenizer}), flush=True)
    bank, eligible_keys, bank_meta = mod.mine_alias_bank(Path(args.input10), tok, args.alias_min_freq, args.max_aliases_per_profile, args.max_candidate_global_frequency)
    print(json.dumps({"event": "alias_bank_done", "eligible_candidates": bank_meta["eligible_candidate_keys"], "eligible_aliases": bank_meta["eligible_alias_types"], "profiles": bank_meta["profiles"]}), flush=True)

    rows: List[RowRec] = []
    source_to_rowidx: Dict[str, int] = {}
    key_to_rows: Dict[str, List[int]] = defaultdict(list)
    input_counters = Counter()
    h_input10 = hashlib.sha256()
    for idx, obj in enumerate(iter_jsonl(Path(args.input10)), start=1):
        if args.limit_rows > 0 and idx > args.limit_rows:
            break
        text = str(obj["text"])
        words = int(obj.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"input words mismatch at row {idx}")
        key = mod.source_key(obj)
        if key in source_to_rowidx:
            raise RuntimeError(f"duplicate source key at row {idx}")
        source_to_rowidx[key] = len(rows)
        sequence_hash_update(h_input10, obj)
        orbits = mod.candidate_orbits(text, eligible_keys)
        input_counters["rows"] += 1
        input_counters["words"] += words
        input_counters["candidate_rows"] += int(bool(orbits))
        input_counters["candidate_keyevents"] += len(orbits)
        if not orbits:
            rows.append(RowRec(idx=idx, obj=obj, key=key, text=text, words=words, orbits={}, original_atoms=set(), geom=(0, ())))
            continue
        geom = mod.tokenizer_geometry(tok, text)
        atoms = set(mod.atom_occurrences(text))
        rr = RowRec(idx=idx, obj=obj, key=key, text=text, words=words, orbits=orbits, original_atoms=atoms, geom=geom)
        row_pos = len(rows)
        rows.append(rr)
        for k in orbits:
            key_to_rows[k].append(row_pos)
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"event": "enumerate_progress", "rows": idx, "candidate_rows": input_counters["candidate_rows"], "keyevents": input_counters["candidate_keyevents"], "elapsed": round(time.time() - t0, 1)}), flush=True)

    print(json.dumps({"event": "enumerate_done", "rows": input_counters["rows"], "words": input_counters["words"], "candidate_rows": input_counters["candidate_rows"], "candidate_keys": len(key_to_rows), "candidate_keyevents": input_counters["candidate_keyevents"]}), flush=True)
    print(json.dumps({"event": "choose_global_aliases_start", "keys": len(key_to_rows), "max_candidates": args.max_global_alias_candidates}), flush=True)
    key_to_alias, key_to_supported_rows, key_stats = choose_global_aliases(mod, tok, key_to_rows, rows, bank, args.max_global_alias_candidates, max(1, args.progress_every // 10))
    print(json.dumps({"event": "choose_global_aliases_done", "selected_keys": len(key_to_alias), "elapsed": round(time.time() - t0, 1)}), flush=True)

    for p in [out10_per, out10_stable, samples_path]:
        if p.exists():
            p.unlink()
    counters = Counter()
    support_hash = hashlib.sha256()
    stable_alias_by_key = defaultdict(Counter)
    per_alias_by_key = defaultdict(Counter)
    transformed_by_source_per: Dict[str, dict] = {}
    transformed_by_source_stable: Dict[str, dict] = {}
    sample_count = 0
    dropped_examples = []

    with out10_per.open("w", encoding="utf-8") as fp, out10_stable.open("w", encoding="utf-8") as fs, samples_path.open("w", encoding="utf-8") as sf:
        for pos, rr in enumerate(rows):
            orig = rr.obj
            keys = []
            if rr.orbits:
                for k in sorted(rr.orbits):
                    if k in key_to_alias and rr.idx in key_to_supported_rows[k]:
                        keys.append(k)
            stable_assign = {k: key_to_alias[k] for k in keys}
            accept = bool(stable_assign)
            reason = ""
            if accept and len(set(a.lower() for a in stable_assign.values())) != len(stable_assign):
                accept = False
                reason = "stable_alias_collision"
            if accept:
                stable_text = mod.apply_mapping(rr.text, rr.orbits, stable_assign)
                if len(stable_text.split()) != rr.words or mod.tokenizer_geometry(tok, stable_text) != rr.geom:
                    accept = False
                    reason = "stable_whole_row_geometry"
            else:
                stable_text = rr.text
            per_assign: Optional[Dict[str, str]] = None
            per_text = rr.text
            if accept:
                per_assign = choose_per_row_aliases(mod, tok, rr, keys, bank, key_to_alias, args.max_per_row_alias_tries)
                if per_assign is None:
                    accept = False
                    reason = "per_row_alias_assignment"
                else:
                    per_text = mod.apply_mapping(rr.text, rr.orbits, per_assign)
                    if len(per_text.split()) != rr.words or mod.tokenizer_geometry(tok, per_text) != rr.geom:
                        accept = False
                        reason = "per_row_whole_row_geometry"
            if not accept:
                if stable_assign:
                    counters["dropped_rows_after_global_support"] += 1
                    counters[f"drop_reason_{reason}"] += 1
                    if len(dropped_examples) < 20:
                        dropped_examples.append({"row": rr.idx, "example_id": orig.get("example_id"), "reason": reason, "keys": keys, "text": rr.text[:300]})
                stable_assign = {}
                per_assign = {}
                stable_text = rr.text
                per_text = rr.text

            new_per = dict(orig)
            new_st = dict(orig)
            if per_assign:
                new_per["text"] = per_text
                new_per["source"] = str(orig.get("source", "example_jsonl")) + "|det_orbit_per_row_global_support"
            if stable_assign:
                new_st["text"] = stable_text
                new_st["source"] = str(orig.get("source", "example_jsonl")) + "|det_orbit_stable_global"
            new_per["words"] = rr.words
            new_st["words"] = rr.words
            fp.write(json.dumps(new_per, ensure_ascii=False) + "\n")
            fs.write(json.dumps(new_st, ensure_ascii=False) + "\n")
            transformed_by_source_per[rr.key] = new_per
            transformed_by_source_stable[rr.key] = new_st

            counters["rows"] += 1
            counters["words"] += rr.words
            counters["changed_rows"] += int(bool(per_assign))
            counters["assigned_keyevents"] += len(per_assign)
            counters["replaced_spans"] += sum(len(rr.orbits[k]) for k in per_assign)
            if per_assign:
                sig = support_signature(rr.idx, per_assign, rr)
                support_hash.update((sig + "\n").encode("utf-8"))
            for k, a in stable_assign.items():
                stable_alias_by_key[k][a] += 1
            for k, a in per_assign.items():
                per_alias_by_key[k][a] += 1
            if per_assign and sample_count < args.sample_limit:
                recs = []
                for k in sorted(per_assign):
                    recs.append({"key": k, "occurrences": len(rr.orbits[k]), "stable_alias": stable_assign[k], "per_row_alias": per_assign[k], "original": rr.orbits[k][0][2]})
                sf.write(json.dumps({"row": rr.idx, "example_id": orig.get("example_id"), "original": rr.text, "stable": stable_text, "per_row": per_text, "assignments": recs}, ensure_ascii=False) + "\n")
                sample_count += 1
            if args.progress_every and rr.idx % args.progress_every == 0:
                print(json.dumps({"event": "write10_progress", "rows": rr.idx, "changed_rows": counters["changed_rows"], "assigned_keyevents": counters["assigned_keyevents"], "elapsed": round(time.time() - t0, 1)}), flush=True)

    # Full 100M stream in reference order.
    rows100 = words100 = misses100 = 0
    ref100_order_hash = hashlib.sha256()
    out100_per_order_hash = hashlib.sha256()
    out100_stable_order_hash = hashlib.sha256()
    if args.limit_rows <= 0:
        if counters["words"] != 10_000_000 or counters["rows"] != 64740:
            raise RuntimeError(f"10M invariant failed rows={counters['rows']} words={counters['words']}")
        for p in [out100_per, out100_stable]:
            if p.exists():
                p.unlink()
        with out100_per.open("w", encoding="utf-8") as fp100, out100_stable.open("w", encoding="utf-8") as fs100:
            for obj in iter_jsonl(Path(args.reference100)):
                rows100 += 1
                words100 += int(obj["words"])
                key = mod.source_key(obj)
                trp = transformed_by_source_per.get(key)
                trs = transformed_by_source_stable.get(key)
                if trp is None or trs is None:
                    misses100 += 1
                    raise RuntimeError(f"reference100 row missing from 10M map at row {rows100}")
                newp = dict(trp)
                news = dict(trs)
                newp["example_id"] = obj.get("example_id", newp.get("example_id"))
                news["example_id"] = obj.get("example_id", news.get("example_id"))
                newp["words"] = int(obj["words"])
                news["words"] = int(obj["words"])
                if trp.get("text") != obj.get("text"):
                    newp["source"] = str(obj.get("source", "example_jsonl")) + "|det_orbit_per_row_global_support"
                else:
                    newp["source"] = obj.get("source", newp.get("source", "example_jsonl"))
                if trs.get("text") != obj.get("text"):
                    news["source"] = str(obj.get("source", "example_jsonl")) + "|det_orbit_stable_global"
                else:
                    news["source"] = obj.get("source", news.get("source", "example_jsonl"))
                fp100.write(json.dumps(newp, ensure_ascii=False) + "\n")
                fs100.write(json.dumps(news, ensure_ascii=False) + "\n")
                sequence_hash_update(ref100_order_hash, obj)
                sequence_hash_update(out100_per_order_hash, newp)
                sequence_hash_update(out100_stable_order_hash, news)
                if args.progress_every and rows100 % (args.progress_every * 5) == 0:
                    print(json.dumps({"event": "write100_progress", "rows": rows100, "words": words100}), flush=True)
        if rows100 != 647400 or words100 != 100_000_000 or misses100:
            raise RuntimeError(f"100M invariant rows={rows100} words={words100} misses={misses100}")
        if ref100_order_hash.hexdigest() != out100_per_order_hash.hexdigest() or ref100_order_hash.hexdigest() != out100_stable_order_hash.hexdigest():
            raise RuntimeError("100M example_id+word order differs from reference")

    stable_multi_alias_keys = {k: c for k, c in stable_alias_by_key.items() if len(c) > 1}
    per_multi_alias_keys = {k: c for k, c in per_alias_by_key.items() if len(c) > 1}
    key_coverage = Counter()
    for k, st in key_stats.items():
        if st["rows_supported"] == 0:
            key_coverage["zero"] += 1
        elif st["rows_supported"] == st["rows_candidate"]:
            key_coverage["full"] += 1
        else:
            key_coverage["partial"] += 1
    partial_keys = sorted([{"key": k, **v} for k, v in key_stats.items() if 0 < v["rows_supported"] < v["rows_candidate"]], key=lambda r: (-(r["rows_candidate"] - r["rows_supported"]), -r["rows_candidate"], r["key"]))[:100]

    manifest = {
        "status": "GLOBAL_COMPATIBLE_ORBIT_PAIR",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Correct research stable-arm row-local alias selection by building identical row-key support with a true corpus-global key->alias mapping.",
        "inputs": {"input10": args.input10, "input10_sha256": sha256_file(Path(args.input10)), "reference100": args.reference100, "reference100_sha256": sha256_file(Path(args.reference100)), "tokenizer": args.tokenizer, "limit_rows": args.limit_rows},
        "alias_bank": bank_meta,
        "input_candidate_counts": dict(input_counters),
        "global_alias_selection": {"keys_candidate": len(key_to_rows), "keys_with_global_alias": len(key_to_alias), "coverage_classes": dict(key_coverage), "max_global_alias_candidates": args.max_global_alias_candidates, "partial_key_examples": partial_keys},
        "final_counts": dict(counters),
        "stable_global_alias_check": {"changed_keys": len(stable_alias_by_key), "keys_with_multiple_aliases": len(stable_multi_alias_keys), "assigned_keyevents_on_multi_alias_keys": sum(sum(c.values()) for c in stable_multi_alias_keys.values())},
        "per_row_alias_variation": {"changed_keys": len(per_alias_by_key), "keys_with_multiple_aliases": len(per_multi_alias_keys), "assigned_keyevents_on_multi_alias_keys": sum(sum(c.values()) for c in per_multi_alias_keys.values()), "top_multi_alias_keys": [{"key": k, "rows": sum(c.values()), "n_aliases": len(c), "top_aliases": c.most_common(8)} for k, c in sorted(per_multi_alias_keys.items(), key=lambda kv: (-sum(kv[1].values()), -len(kv[1]), kv[0]))[:50]]},
        "geometry_contract": {"whitespace_word_count_equal": True, "full_token_count_and_wwm_vector_equal": True, "support_identical_by_construction": True, "support_sha256": support_hash.hexdigest()},
        "outputs": {"per_row_10M": str(out10_per), "per_row_10M_sha256": sha256_file(out10_per), "stable_10M": str(out10_stable), "stable_10M_sha256": sha256_file(out10_stable), "per_row_100M": str(out100_per) if args.limit_rows <= 0 else None, "per_row_100M_sha256": sha256_file(out100_per) if args.limit_rows <= 0 else None, "stable_100M": str(out100_stable) if args.limit_rows <= 0 else None, "stable_100M_sha256": sha256_file(out100_stable) if args.limit_rows <= 0 else None, "samples": str(samples_path)},
        "order_contract": {"input10_exampleid_word_sha256": h_input10.hexdigest(), "reference100_exampleid_word_sha256": ref100_order_hash.hexdigest() if args.limit_rows <= 0 else None, "per_row100_exampleid_word_sha256": out100_per_order_hash.hexdigest() if args.limit_rows <= 0 else None, "stable100_exampleid_word_sha256": out100_stable_order_hash.hexdigest() if args.limit_rows <= 0 else None, "rows100": rows100, "words100": words100, "reference100_map_misses": misses100},
        "dropped_examples": dropped_examples,
        "interpretation": "This file validates corpus construction only. Stable_global has exact zero key-to-alias entropy; per_row_global_support uses the same row-key support but destroys most cross-row alias persistence. Later model results must still separate ordinary token familiarity from relation-level reuse using token exposure summaries, broad unaffected tasks, Entity/world-knowledge movement, and item-level relational EWoK rather than Overall alone.",
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = []
    md.append("# research global-compatible orbit pair\n\n")
    md.append(f"Input rows/words: {input_counters['rows']} / {input_counters['words']}\n\n")
    md.append(f"Candidate rows/key-events/keys: {input_counters['candidate_rows']} / {input_counters['candidate_keyevents']} / {len(key_to_rows)}\n\n")
    md.append(f"Global aliases selected: {len(key_to_alias)} keys; coverage classes {dict(key_coverage)}\n\n")
    md.append(f"Final changed rows/key-events/spans: {counters['changed_rows']} / {counters['assigned_keyevents']} / {counters['replaced_spans']}\n\n")
    md.append(f"Rows dropped after row-level paired geometry/injectivity checks: {counters['dropped_rows_after_global_support']}\n\n")
    md.append(f"Stable keys with multiple aliases: {len(stable_multi_alias_keys)} (must be 0)\n\n")
    md.append(f"Per-row keys with multiple aliases: {len(per_multi_alias_keys)} / {len(per_alias_by_key)}\n\n")
    md.append(f"100M rows/words: {rows100} / {words100}\n\n")
    md.append(f"Support SHA256: `{support_hash.hexdigest()}`\n\n")
    md.append("The two arms transform exactly the same row-key occurrences. `stable_global` uses one corpus-global alias per lexical key; `per_row_global_support` uses row-specific aliases on the same support. Every transformed row preserves whitespace word count and the complete legal16k WWM word-start geometry.\n\n")
    md.append(f"Manifest: `{manifest_path}`\n")
    md_path.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "rows": counters["rows"],
        "words": counters["words"],
        "changed_rows": counters["changed_rows"],
        "assigned_keyevents": counters["assigned_keyevents"],
        "replaced_spans": counters["replaced_spans"],
        "stable_multi_alias_keys": len(stable_multi_alias_keys),
        "per_row_multi_alias_keys": len(per_multi_alias_keys),
        "rows100": rows100,
        "words100": words100,
        "manifest": str(manifest_path),
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
