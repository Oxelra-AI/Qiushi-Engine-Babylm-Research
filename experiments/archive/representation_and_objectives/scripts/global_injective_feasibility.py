#!/usr/bin/env python3
"""research feasibility check for globally injective token-matched aliases.

The research global-compatible pair fixed H(alias|key)=0 but allowed many keys to
share one alias.  This script checks whether there are enough geometry-compatible
same-profile aliases to assign unique aliases to transformed lexical keys.
It is a cheap precondition for deciding whether the language-scale corpus should
be rebuilt before any expensive training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_STEP259_SCRIPT = Path("experiments/archive/representation_and_objectives/scripts/deterministic_token_matched_entity_orbit.py")
DEFAULT_INPUT10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_TOKENIZER = Path("experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model")
DEFAULT_MANIFEST = Path("experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_manifest.json")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/global_injective_feasibility")


def load_mod(path: Path):
    spec = importlib.util.spec_from_file_location("orbit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orbit"] = mod
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_int(s: str) -> int:
    import hashlib
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default=str(DEFAULT_STEP259_SCRIPT))
    ap.add_argument("--input10", default=str(DEFAULT_INPUT10))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max_aliases_per_profile", type=int, default=1000000)
    ap.add_argument("--max_candidate_global_frequency", type=int, default=2000)
    ap.add_argument("--max_compat_check", type=int, default=200)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_mod(Path(args.script))
    tok = mod.AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bank, eligible_keys, bank_meta = mod.mine_alias_bank(Path(args.input10), tok, min_freq=2, max_per_profile=args.max_aliases_per_profile, max_candidate_freq=args.max_candidate_global_frequency)
    key_profile_counts = Counter()
    key_rows = Counter()
    key_profile_examples = defaultdict(list)
    key_sample_row = {}
    for idx, obj in enumerate(iter_jsonl(Path(args.input10)), start=1):
        text = str(obj["text"])
        orbits = mod.candidate_orbits(text, eligible_keys)
        for key, spans in orbits.items():
            original = spans[0][2]
            prof = tuple(mod.alias_profile(tok, original))
            key_profile_counts[prof] += 1
            key_rows[key] += 1
            if len(key_profile_examples[prof]) < 8:
                key_profile_examples[prof].append({"key": key, "original": original, "row": idx})
            key_sample_row.setdefault(key, (obj, orbits, prof))
    unique_key_profiles = Counter()
    for key, (obj, orbits, prof) in key_sample_row.items():
        unique_key_profiles[prof] += 1
    alias_profile_counts = {str(k): len(v) for k, v in sorted(bank.items())}
    profile_balance = []
    for prof, nkeys in sorted(unique_key_profiles.items(), key=lambda kv: (-kv[1], kv[0])):
        aliases = len(bank.get(prof, []))
        profile_balance.append({"profile": list(prof), "unique_candidate_keys": nkeys, "candidate_keyevents": key_profile_counts[prof], "aliases_available": aliases, "surplus_aliases": aliases - nkeys, "examples": key_profile_examples[prof]})

    # Greedy unique-alias compatibility trial with individual row compatibility on up to max_compat_check aliases.
    used = set()
    assigned = {}
    failed = []
    partial = []
    keys_order = sorted(key_sample_row.keys(), key=lambda k: (-key_rows[k], k))
    for key in keys_order:
        obj, orbits, prof = key_sample_row[key]
        text = str(obj["text"]); words = int(obj["words"])
        geom = mod.tokenizer_geometry(tok, text)
        atoms = set(mod.atom_occurrences(text))
        candidates = list(bank.get(prof, []))
        rng = random.Random(stable_int("INJECTIVE_TRIAL|" + key))
        rng.shuffle(candidates)
        picked = None; tried = 0
        for cand in candidates[:args.max_compat_check]:
            tried += 1
            if cand.lower() in used or cand.lower() == key or cand.lower() in atoms:
                continue
            trial_text = mod.apply_mapping(text, {key: orbits[key]}, {key: cand})
            if len(trial_text.split()) != words:
                continue
            if mod.tokenizer_geometry(tok, trial_text) != geom:
                continue
            picked = cand
            break
        if picked is None:
            failed.append({"key": key, "rows": key_rows[key], "profile": list(prof), "tried": tried})
        else:
            assigned[key] = picked
            used.add(picked.lower())
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if Path(args.manifest).exists() else {}
    out = {
        "status": "GLOBAL_INJECTIVE_FEASIBILITY",
        "elapsed_seconds": round(time.time()-t0,2),
        "alias_bank_meta": bank_meta,
        "profile_balance": profile_balance,
        "profile_balance_has_deficit": any(x["surplus_aliases"] < 0 for x in profile_balance),
        "unique_candidate_keys": len(key_sample_row),
        "candidate_keyevents": sum(key_profile_counts.values()),
        "greedy_trial": {"assigned_keys": len(assigned), "failed_keys": len(failed), "max_compat_check": args.max_compat_check, "failed_examples": failed[:200]},
        "current_global_pair_counts": manifest.get("final_counts"),
        "interpretation": "If profile_balance has no deficit and greedy failures are small or repairable, a globally injective alias map is feasible and should be preferred before any expensive per_row/stable training. If impossible, the stable-vs-per-row contrast must report inverse alias collisions as part of the intervention.",
    }
    out_json = out_dir / "global_injective_feasibility.json"
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    lines=["# research globally injective alias feasibility\n\n",
           f"Unique candidate keys: {len(key_sample_row)}; candidate key-events: {sum(key_profile_counts.values())}\n\n",
           f"Profile deficit exists: {out['profile_balance_has_deficit']}\n\n",
           "| profile | keys | key-events | aliases | surplus | examples |\n|---|---:|---:|---:|---:|---|\n"]
    for x in profile_balance:
        lines.append(f"| {tuple(x['profile'])} | {x['unique_candidate_keys']} | {x['candidate_keyevents']} | {x['aliases_available']} | {x['surplus_aliases']} | {x['examples'][:3]} |\n")
    lines.append(f"\nGreedy unique-alias trial: assigned {len(assigned)}, failed {len(failed)} with max_compat_check={args.max_compat_check}.\n\nJSON: `{out_json}`\n")
    (out_dir/"global_injective_feasibility.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": out["status"], "unique_candidate_keys": len(key_sample_row), "profile_deficit": out["profile_balance_has_deficit"], "greedy_assigned": len(assigned), "greedy_failed": len(failed), "out": str(out_json), "elapsed_seconds": out["elapsed_seconds"]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
