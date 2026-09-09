#!/usr/bin/env python3
"""research: globally injective token-matched orbit pair.

The prior global-compatible pair enforces one alias per source key but still lets
many source keys collapse onto the same stable alias.  This builder uses a global
used-alias set while selecting stable aliases.  Keys without any compatible unused
alias are dropped from both arms.  The output pair is therefore:

  stable_injective: each transformed source key maps to one unique alias;
  per_row_injective_support: the same transformed row-key support, but the alias
    can vary by row while preserving row-local injectivity and token/WWM geometry.

This is still only a corpus substrate.  It is not a model result and does not by
itself show language-scale transfer.
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
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path("experiments/archive/representation_and_objectives")
DEFAULT_STEP259_SCRIPT = ROOT / "scripts/deterministic_token_matched_entity_orbit.py"
DEFAULT_BASE260_SCRIPT = ROOT / "scripts/global_compatible_orbit_pair.py"
DEFAULT_INPUT10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_REFERENCE100 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
DEFAULT_TOKENIZER = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_OUT = ROOT / "data/global_injective_orbit_pair"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path):
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


def choose_injective_aliases(research, base260, tok, key_to_rows, rows, bank, max_candidates: int, progress_every: int):
    key_to_alias = {}
    key_to_supported_rows = {}
    key_stats = {}
    used_aliases: Set[str] = set()
    keys = sorted(key_to_rows, key=lambda k: (-len(key_to_rows[k]), k))
    t0 = time.time()
    for n, key in enumerate(keys, start=1):
        refs = [rows[i] for i in key_to_rows[key]]
        candidates = base260.candidate_aliases_for_key(research, bank, refs, key)
        # Deterministic key-specific shuffle is already inside candidate_aliases; keep high coverage.
        best_alias = None
        best_support: List[int] = []
        tried = 0
        for cand in candidates[:max_candidates]:
            tried += 1
            cl = cand.lower()
            if cl in used_aliases:
                continue
            support = []
            for rr in refs:
                if base260.single_key_compatible(research, tok, rr, key, cand):
                    support.append(rr.idx)
            if len(support) > len(best_support):
                best_alias = cand
                best_support = support
                if len(best_support) == len(refs):
                    break
        if best_alias is not None and best_support:
            key_to_alias[key] = best_alias
            key_to_supported_rows[key] = set(best_support)
            used_aliases.add(best_alias.lower())
        key_stats[key] = {"rows_candidate": len(refs), "rows_supported": len(best_support), "coverage": len(best_support)/len(refs) if refs else 0.0, "alias": best_alias, "aliases_tried": tried}
        if progress_every and n % progress_every == 0:
            print(json.dumps({"event": "injective_alias_progress", "keys": n, "of": len(keys), "selected": len(key_to_alias), "used_aliases": len(used_aliases), "elapsed": round(time.time()-t0, 1)}), flush=True)
    return key_to_alias, key_to_supported_rows, key_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(DEFAULT_STEP259_SCRIPT))
    ap.add_argument("--base260_script", default=str(DEFAULT_BASE260_SCRIPT))
    ap.add_argument("--input10", default=str(DEFAULT_INPUT10))
    ap.add_argument("--reference100", default=str(DEFAULT_REFERENCE100))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--limit_rows", type=int, default=0)
    ap.add_argument("--alias_min_freq", type=int, default=2)
    ap.add_argument("--max_aliases_per_profile", type=int, default=1000000)
    ap.add_argument("--max_candidate_global_frequency", type=int, default=2000)
    ap.add_argument("--max_global_alias_candidates", type=int, default=1000)
    ap.add_argument("--max_per_row_alias_tries", type=int, default=300)
    ap.add_argument("--sample_limit", type=int, default=80)
    ap.add_argument("--progress_every", type=int, default=10000)
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit_rows <= 0 else f"_pilot_{args.limit_rows}"
    out10_per = out_dir / f"det_orbit_per_row_injective_support_10M{suffix}.jsonl"
    out10_st = out_dir / f"det_orbit_stable_injective_10M{suffix}.jsonl"
    out100_per = out_dir / "det_orbit_per_row_injective_support_100M.jsonl"
    out100_st = out_dir / "det_orbit_stable_injective_100M.jsonl"
    samples_path = out_dir / f"injective_orbit_pair_samples{suffix}.jsonl"
    manifest_path = out_dir / f"injective_orbit_pair_manifest{suffix}.json"
    md_path = out_dir / f"injective_orbit_pair_manifest{suffix}.md"

    research = load_module("orbit_for_injective", Path(args.script))
    base260 = load_module("global_compatible_base", Path(args.base260_script))
    tok = research.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    research._STEP260_TOKENIZER = tok
    print(json.dumps({"event":"mine_alias_bank", "input10":args.input10, "tokenizer":args.tokenizer, "max_per_profile":args.max_aliases_per_profile}), flush=True)
    bank, eligible_keys, bank_meta = research.mine_alias_bank(Path(args.input10), tok, args.alias_min_freq, args.max_aliases_per_profile, args.max_candidate_global_frequency)
    print(json.dumps({"event":"alias_bank_done", "eligible_candidates":bank_meta["eligible_candidate_keys"], "eligible_aliases":bank_meta["eligible_alias_types"], "profiles":bank_meta["profiles"]}), flush=True)

    rows = []
    source_to_rowidx = {}
    key_to_rows = defaultdict(list)
    input_counters = Counter()
    h_input10 = hashlib.sha256()
    for idx, obj in enumerate(iter_jsonl(Path(args.input10)), start=1):
        if args.limit_rows > 0 and idx > args.limit_rows:
            break
        text = str(obj["text"]); words = int(obj.get("words", len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f"input word mismatch at row {idx}")
        skey = research.source_key(obj)
        if skey in source_to_rowidx:
            raise RuntimeError(f"duplicate source key at row {idx}")
        source_to_rowidx[skey] = len(rows)
        sequence_hash_update(h_input10, obj)
        orbits = research.candidate_orbits(text, eligible_keys)
        input_counters["rows"] += 1; input_counters["words"] += words
        input_counters["candidate_rows"] += int(bool(orbits)); input_counters["candidate_keyevents"] += len(orbits)
        if not orbits:
            rows.append(base260.RowRec(idx=idx, obj=obj, key=skey, text=text, words=words, orbits={}, original_atoms=set(), geom=(0,())))
            continue
        rr = base260.RowRec(idx=idx, obj=obj, key=skey, text=text, words=words, orbits=orbits, original_atoms=set(research.atom_occurrences(text)), geom=research.tokenizer_geometry(tok, text))
        pos = len(rows); rows.append(rr)
        for k in orbits:
            key_to_rows[k].append(pos)
        if args.progress_every and idx % args.progress_every == 0:
            print(json.dumps({"event":"enumerate_progress", "rows":idx, "candidate_rows":input_counters["candidate_rows"], "keyevents":input_counters["candidate_keyevents"], "elapsed":round(time.time()-t0,1)}), flush=True)
    print(json.dumps({"event":"enumerate_done", "rows":input_counters["rows"], "words":input_counters["words"], "candidate_rows":input_counters["candidate_rows"], "candidate_keys":len(key_to_rows), "candidate_keyevents":input_counters["candidate_keyevents"]}), flush=True)

    key_to_alias, key_to_supported_rows, key_stats = choose_injective_aliases(research, base260, tok, key_to_rows, rows, bank, args.max_global_alias_candidates, max(1, args.progress_every//10))
    print(json.dumps({"event":"choose_injective_aliases_done", "selected_keys":len(key_to_alias), "elapsed":round(time.time()-t0,1)}), flush=True)

    for p in [out10_per, out10_st, samples_path]:
        if p.exists(): p.unlink()
    counters = Counter(); support_hash = hashlib.sha256()
    stable_alias_by_key = defaultdict(Counter); inverse_alias_key = defaultdict(Counter); per_alias_by_key = defaultdict(Counter)
    transformed_per = {}; transformed_st = {}; dropped_examples=[]; sample_count=0
    with out10_per.open("w",encoding="utf-8") as fp, out10_st.open("w",encoding="utf-8") as fs, samples_path.open("w",encoding="utf-8") as sf:
        for rr in rows:
            orig = rr.obj
            keys = []
            if rr.orbits:
                for k in sorted(rr.orbits):
                    if k in key_to_alias and rr.idx in key_to_supported_rows[k]:
                        keys.append(k)
            stable_assign = {k:key_to_alias[k] for k in keys}
            accept = bool(stable_assign); reason=""
            stable_text = rr.text; per_text = rr.text; per_assign = {}
            if accept:
                stable_text = research.apply_mapping(rr.text, rr.orbits, stable_assign)
                if len(stable_text.split()) != rr.words or research.tokenizer_geometry(tok, stable_text) != rr.geom:
                    accept=False; reason="stable_geometry"
            if accept:
                per_assign0 = base260.choose_per_row_aliases(research, tok, rr, keys, bank, key_to_alias, args.max_per_row_alias_tries)
                if per_assign0 is None:
                    accept=False; reason="per_assignment"
                else:
                    per_assign = per_assign0
                    per_text = research.apply_mapping(rr.text, rr.orbits, per_assign)
                    if len(per_text.split()) != rr.words or research.tokenizer_geometry(tok, per_text) != rr.geom:
                        accept=False; reason="per_geometry"
            if not accept:
                if stable_assign:
                    counters["dropped_rows_after_injective_support"] += 1
                    counters[f"drop_reason_{reason}"] += 1
                    if len(dropped_examples) < 30:
                        dropped_examples.append({"row":rr.idx, "example_id":orig.get("example_id"), "reason":reason, "keys":keys, "text":rr.text[:350]})
                stable_assign={}; per_assign={}; stable_text=rr.text; per_text=rr.text
            newp=dict(orig); news=dict(orig)
            newp["words"]=rr.words; news["words"]=rr.words
            if per_assign:
                newp["text"]=per_text; newp["source"]=str(orig.get("source","example_jsonl"))+"|det_orbit_per_row_injective_support"
            if stable_assign:
                news["text"]=stable_text; news["source"]=str(orig.get("source","example_jsonl"))+"|det_orbit_stable_injective"
            fp.write(json.dumps(newp, ensure_ascii=False)+"\n"); fs.write(json.dumps(news, ensure_ascii=False)+"\n")
            transformed_per[rr.key]=newp; transformed_st[rr.key]=news
            counters["rows"] += 1; counters["words"] += rr.words
            counters["changed_rows"] += int(bool(per_assign)); counters["assigned_keyevents"] += len(per_assign)
            counters["replaced_spans"] += sum(len(rr.orbits[k]) for k in per_assign)
            if per_assign:
                support_hash.update((base260.support_signature(rr.idx, per_assign, rr)+"\n").encode("utf-8"))
            for k,a in stable_assign.items():
                stable_alias_by_key[k][a] += 1; inverse_alias_key[a][k] += 1
            for k,a in per_assign.items():
                per_alias_by_key[k][a] += 1
            if per_assign and sample_count < args.sample_limit:
                sf.write(json.dumps({"row":rr.idx, "example_id":orig.get("example_id"), "original":rr.text, "stable":stable_text, "per_row":per_text, "assignments":[{"key":k,"original":rr.orbits[k][0][2],"occurrences":len(rr.orbits[k]),"stable_alias":stable_assign[k],"per_row_alias":per_assign[k]} for k in sorted(per_assign)]}, ensure_ascii=False)+"\n")
                sample_count += 1
            if args.progress_every and rr.idx % args.progress_every == 0:
                print(json.dumps({"event":"write10_progress", "rows":rr.idx, "changed_rows":counters["changed_rows"], "assigned_keyevents":counters["assigned_keyevents"], "elapsed":round(time.time()-t0,1)}), flush=True)

    rows100=words100=misses100=0
    h_ref100=hashlib.sha256(); h_per100=hashlib.sha256(); h_st100=hashlib.sha256()
    if args.limit_rows <= 0:
        if counters["rows"] != 64740 or counters["words"] != 10_000_000:
            raise RuntimeError(f"10M invariant rows={counters['rows']} words={counters['words']}")
        for p in [out100_per, out100_st]:
            if p.exists(): p.unlink()
        with out100_per.open("w",encoding="utf-8") as fp100, out100_st.open("w",encoding="utf-8") as fs100:
            for obj in iter_jsonl(Path(args.reference100)):
                rows100 += 1; words100 += int(obj["words"])
                skey = research.source_key(obj)
                trp = transformed_per.get(skey); trs = transformed_st.get(skey)
                if trp is None or trs is None:
                    misses100 += 1; raise RuntimeError(f"reference row missing {rows100}")
                newp=dict(trp); news=dict(trs)
                newp["example_id"]=obj.get("example_id", newp.get("example_id")); news["example_id"]=obj.get("example_id", news.get("example_id"))
                newp["words"]=int(obj["words"]); news["words"]=int(obj["words"])
                if trp.get("text") != obj.get("text"):
                    newp["source"]=str(obj.get("source","example_jsonl"))+"|det_orbit_per_row_injective_support"
                else:
                    newp["source"]=obj.get("source", newp.get("source","example_jsonl"))
                if trs.get("text") != obj.get("text"):
                    news["source"]=str(obj.get("source","example_jsonl"))+"|det_orbit_stable_injective"
                else:
                    news["source"]=obj.get("source", news.get("source","example_jsonl"))
                fp100.write(json.dumps(newp,ensure_ascii=False)+"\n"); fs100.write(json.dumps(news,ensure_ascii=False)+"\n")
                sequence_hash_update(h_ref100,obj); sequence_hash_update(h_per100,newp); sequence_hash_update(h_st100,news)
                if args.progress_every and rows100 % (args.progress_every*5)==0:
                    print(json.dumps({"event":"write100_progress", "rows":rows100, "words":words100}), flush=True)
        if rows100 != 647400 or words100 != 100_000_000 or misses100:
            raise RuntimeError(f"100M invariant rows={rows100} words={words100} misses={misses100}")
        if h_ref100.hexdigest() != h_per100.hexdigest() or h_ref100.hexdigest() != h_st100.hexdigest():
            raise RuntimeError("100M example_id+word order differs")

    stable_multi = {k:c for k,c in stable_alias_by_key.items() if len(c)>1}
    inverse_multi = {a:c for a,c in inverse_alias_key.items() if len(c)>1}
    per_multi = {k:c for k,c in per_alias_by_key.items() if len(c)>1}
    coverage = Counter()
    for k, st in key_stats.items():
        if st["rows_supported"] == 0: coverage["zero"] += 1
        elif st["rows_supported"] == st["rows_candidate"]: coverage["full"] += 1
        else: coverage["partial"] += 1
    partial = sorted([{"key":k, **v} for k,v in key_stats.items() if 0 < v["rows_supported"] < v["rows_candidate"]], key=lambda r: (-(r["rows_candidate"]-r["rows_supported"]), -r["rows_candidate"], r["key"]))[:100]
    failed = sorted([{"key":k, **v} for k,v in key_stats.items() if v["rows_supported"] == 0], key=lambda r: (-r["rows_candidate"], r["key"]))[:100]
    manifest = {
        "status":"GLOBAL_INJECTIVE_ORBIT_PAIR",
        "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose":"Build identical-support token-matched orbit pair with stable arm globally one-to-one over transformed keys.",
        "inputs":{"input10":args.input10,"input10_sha256":sha256_file(Path(args.input10)),"reference100":args.reference100,"reference100_sha256":sha256_file(Path(args.reference100)),"tokenizer":args.tokenizer,"limit_rows":args.limit_rows},
        "alias_bank":bank_meta,
        "input_candidate_counts":dict(input_counters),
        "injective_alias_selection":{"keys_candidate":len(key_to_rows),"keys_with_alias":len(key_to_alias),"coverage_classes":dict(coverage),"partial_key_examples":partial,"zero_key_examples":failed,"max_global_alias_candidates":args.max_global_alias_candidates},
        "final_counts":dict(counters),
        "stable_alias_check":{"changed_keys":len(stable_alias_by_key),"keys_with_multiple_aliases":len(stable_multi),"aliases_used":len(inverse_alias_key),"aliases_with_multiple_source_keys":len(inverse_multi),"events_on_multi_key_aliases":sum(sum(c.values()) for c in inverse_multi.values())},
        "per_row_alias_variation":{"changed_keys":len(per_alias_by_key),"keys_with_multiple_aliases":len(per_multi),"events_on_multi_alias_keys":sum(sum(c.values()) for c in per_multi.values())},
        "geometry_contract":{"support_identical_by_construction":True,"support_sha256":support_hash.hexdigest(),"whitespace_word_count_equal":True,"full_token_count_and_wwm_vector_equal":True},
        "outputs":{"per_row_10M":str(out10_per),"per_row_10M_sha256":sha256_file(out10_per),"stable_10M":str(out10_st),"stable_10M_sha256":sha256_file(out10_st),"per_row_100M":str(out100_per) if args.limit_rows<=0 else None,"per_row_100M_sha256":sha256_file(out100_per) if args.limit_rows<=0 else None,"stable_100M":str(out100_st) if args.limit_rows<=0 else None,"stable_100M_sha256":sha256_file(out100_st) if args.limit_rows<=0 else None,"samples":str(samples_path)},
        "order_contract":{"input10_exampleid_word_sha256":h_input10.hexdigest(),"reference100_exampleid_word_sha256":h_ref100.hexdigest() if args.limit_rows<=0 else None,"per_row100_exampleid_word_sha256":h_per100.hexdigest() if args.limit_rows<=0 else None,"stable100_exampleid_word_sha256":h_st100.hexdigest() if args.limit_rows<=0 else None,"rows100":rows100,"words100":words100,"reference100_map_misses":misses100},
        "dropped_examples":dropped_examples,
        "interpretation":"Stable arm has zero H(alias|key) and zero H(key|alias) on transformed keys. Per-row arm shares the same row-key support but varies aliases across rows. Model interpretation must still account for alias marginal differences and the fact that this intervention targets repeated capitalized atoms, not all relational arguments.",
        "elapsed_seconds":round(time.time()-t0,2),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    md=["# research globally injective orbit pair\n\n",
        f"Input rows/words: {input_counters['rows']} / {input_counters['words']}\n\n",
        f"Candidate rows/key-events/keys: {input_counters['candidate_rows']} / {input_counters['candidate_keyevents']} / {len(key_to_rows)}\n\n",
        f"Stable unique aliases selected: {len(key_to_alias)} keys; coverage classes {dict(coverage)}\n\n",
        f"Final changed rows/key-events/spans: {counters['changed_rows']} / {counters['assigned_keyevents']} / {counters['replaced_spans']}\n\n",
        f"Rows dropped after paired checks: {counters['dropped_rows_after_injective_support']}\n\n",
        f"Stable multi-alias keys: {len(stable_multi)}; stable aliases shared by multiple source keys: {len(inverse_multi)}\n\n",
        f"Per-row multi-alias keys: {len(per_multi)} / {len(per_alias_by_key)}\n\n",
        f"100M rows/words: {rows100} / {words100}\n\n",
        f"Support SHA256: `{support_hash.hexdigest()}`\n\n",
        f"Manifest: `{manifest_path}`\n"]
    md_path.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"rows":counters["rows"],"words":counters["words"],"changed_rows":counters["changed_rows"],"assigned_keyevents":counters["assigned_keyevents"],"replaced_spans":counters["replaced_spans"],"stable_multi_alias_keys":len(stable_multi),"stable_inverse_multi_aliases":len(inverse_multi),"per_row_multi_alias_keys":len(per_multi),"rows100":rows100,"words100":words100,"manifest":str(manifest_path),"elapsed_seconds":manifest["elapsed_seconds"]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
