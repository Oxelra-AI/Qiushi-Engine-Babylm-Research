#!/usr/bin/env python3
"""research post-build audit for the global-compatible orbit corpora.

This independent reread checks the written research 10M corpora, not just the
builder's internal manifest.  It measures invariants needed before deciding on
expensive H100 training:

  * row/word/order equality and changed support equality;
  * stable H(alias|key)=0 and per-row persistence distance;
  * inverse alias collisions H(key|alias) and alias marginal concentration;
  * repeated-key dose and P_same for stable versus per-row;
  * rough overlap between transformed lexical keys/contexts and the research EWoK
    relational bridge lexicon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_ORIG10 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
DEFAULT_PER10 = Path("experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/det_orbit_per_row_global_support_10M.jsonl")
DEFAULT_STABLE10 = Path("experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/det_orbit_stable_global_10M.jsonl")
DEFAULT_MANIFEST = Path("experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_manifest.json")
DEFAULT_EWOK_PANEL = Path("experiments/archive/representation_and_objectives/data/ewok_natural_bridge_panel/ewok_bridge_panel.jsonl")
DEFAULT_OUT = Path("experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def toks(s: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(s)]


def content(s: str) -> List[str]:
    stop = {"the","a","an","is","are","was","were","to","of","in","on","at","and","or","but","not","with","than","as","it","this","that","then","after","before","both","now","nearby","from","for","by","into","over","under","there","their","they","them","he","she","his","her"}
    return [t for t in toks(s) if t not in stop and len(t) > 1]


def changed_words(a: str, b: str) -> List[Tuple[str, str]]:
    wa = a.split(); wb = b.split()
    if len(wa) != len(wb):
        raise RuntimeError("word count mismatch in changed_words")
    return [(x, y) for x, y in zip(wa, wb) if x != y]


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    return -sum((c/n)*math.log2(c/n) for c in counts.values() if c)


def pair_same_probability(key_alias_counts: Dict[str, Counter]) -> float:
    num = den = 0
    for c in key_alias_counts.values():
        n = sum(c.values())
        den += n * (n - 1) // 2
        for v in c.values():
            num += v * (v - 1) // 2
    return num / den if den else 0.0


def load_ewok(panel: Path):
    all_tokens = Counter()
    rel_tokens = Counter()
    rel_items = 0
    all_items = 0
    for obj in iter_jsonl(panel):
        all_items += 1
        ts = obj.get("content_tokens_all") or content(" ".join(str(obj.get(k,"")) for k in ["Context1","Context2","Target1","Target2"]))
        all_tokens.update(ts)
        if obj.get("is_step211_relational_domain"):
            rel_items += 1
            rel_tokens.update(ts)
    return {"all_items": all_items, "rel_items": rel_items, "all_tokens": all_tokens, "rel_tokens": rel_tokens}


def sequence_hash_update(h, obj: dict) -> None:
    h.update((str(obj.get("example_id", "")) + "\t" + str(obj["words"]) + "\n").encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig10", default=str(DEFAULT_ORIG10))
    ap.add_argument("--per10", default=str(DEFAULT_PER10))
    ap.add_argument("--stable10", default=str(DEFAULT_STABLE10))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--ewok_panel", default=str(DEFAULT_EWOK_PANEL))
    ap.add_argument("--output_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--max_examples", type=int, default=30)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ewok = load_ewok(Path(args.ewok_panel))
    out_dir = Path(args.output_dir)

    counters = Counter()
    order_hashes = {"orig": hashlib.sha256(), "per": hashlib.sha256(), "stable": hashlib.sha256()}
    stable_key_alias = defaultdict(Counter)
    per_key_alias = defaultdict(Counter)
    alias_key = defaultdict(Counter)
    stable_alias_marginal = Counter()
    per_alias_marginal = Counter()
    changed_support_mismatch = []
    changed_examples = []
    relation_context_rows = Counter()
    transformed_keys = Counter()
    transformed_keys_rellex = Counter()
    transformed_context_rellex = Counter()

    rel_tokens = set(ewok["rel_tokens"])
    all_ewok_tokens = set(ewok["all_tokens"])

    for idx, (o, p, s) in enumerate(zip(iter_jsonl(Path(args.orig10)), iter_jsonl(Path(args.per10)), iter_jsonl(Path(args.stable10))), start=1):
        for name, obj in [("orig", o), ("per", p), ("stable", s)]:
            counters[f"{name}_rows"] += 1
            counters[f"{name}_words"] += int(obj["words"])
            if int(obj["words"]) != len(str(obj["text"]).split()):
                counters[f"{name}_word_mismatch"] += 1
            sequence_hash_update(order_hashes[name], obj)
        if (o.get("example_id"), o["words"]) != (p.get("example_id"), p["words"]) or (o.get("example_id"), o["words"]) != (s.get("example_id"), s["words"]):
            counters["order_mismatch"] += 1
        per_changed = p["text"] != o["text"]
        st_changed = s["text"] != o["text"]
        counters["per_changed_rows"] += int(per_changed)
        counters["stable_changed_rows"] += int(st_changed)
        if per_changed != st_changed:
            counters["changed_row_mismatch"] += 1
            if len(changed_support_mismatch) < args.max_examples:
                changed_support_mismatch.append({"row": idx, "example_id": o.get("example_id"), "per_changed": per_changed, "stable_changed": st_changed})
        if per_changed:
            diffs_st = changed_words(o["text"], s["text"])
            diffs_pr = changed_words(o["text"], p["text"])
            if len(diffs_st) != len(diffs_pr):
                counters["changed_word_count_support_mismatch"] += 1
            if len(changed_examples) < args.max_examples:
                changed_examples.append({"row": idx, "example_id": o.get("example_id"), "stable_diffs": diffs_st[:12], "per_diffs": diffs_pr[:12], "text": o["text"][:350]})
            old_to_st = {}
            old_to_pr = {}
            for old, new in diffs_st:
                key = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", old).lower()
                ali = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", new).lower()
                if key and ali:
                    old_to_st[key] = ali
            for old, new in diffs_pr:
                key = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", old).lower()
                ali = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", new).lower()
                if key and ali:
                    old_to_pr[key] = ali
            if set(old_to_st) != set(old_to_pr):
                counters["changed_key_support_mismatch"] += 1
            ctx_tokens = set(content(o["text"]))
            rel_overlap = len(ctx_tokens & rel_tokens)
            all_overlap = len(ctx_tokens & all_ewok_tokens)
            relation_context_rows["changed_rows"] += 1
            relation_context_rows["rows_with_rel_ewok_token"] += int(rel_overlap > 0)
            relation_context_rows["rows_with_any_ewok_token"] += int(all_overlap > 0)
            relation_context_rows["rel_ewok_token_overlap_sum"] += rel_overlap
            relation_context_rows["all_ewok_token_overlap_sum"] += all_overlap
            for k, a in old_to_st.items():
                stable_key_alias[k][a] += 1
                alias_key[a][k] += 1
                stable_alias_marginal[a] += 1
                transformed_keys[k] += 1
                transformed_keys_rellex[k] += int(k in rel_tokens)
                transformed_context_rellex[k] += rel_overlap
            for k, a in old_to_pr.items():
                per_key_alias[k][a] += 1
                per_alias_marginal[a] += 1

    stable_multi = {k:c for k,c in stable_key_alias.items() if len(c)>1}
    per_multi = {k:c for k,c in per_key_alias.items() if len(c)>1}
    inverse_multi = {a:c for a,c in alias_key.items() if len(c)>1}
    stable_events = sum(sum(c.values()) for c in stable_key_alias.values())
    per_events = sum(sum(c.values()) for c in per_key_alias.values())
    repeat_key_events = sum(sum(c.values()) for k,c in stable_key_alias.items() if sum(c.values()) >= 2)
    singleton_key_events = sum(sum(c.values()) for k,c in stable_key_alias.items() if sum(c.values()) == 1)
    rel_keys = {k for k in stable_key_alias if k in rel_tokens}
    all_ewok_keys = {k for k in stable_key_alias if k in all_ewok_tokens}

    # The per-row output can accidentally reuse the stable alias; count how often.
    same_as_stable = 0
    for k, ca in per_key_alias.items():
        st_alias = next(iter(stable_key_alias[k])) if k in stable_key_alias and len(stable_key_alias[k]) == 1 else None
        if st_alias:
            same_as_stable += ca.get(st_alias, 0)

    summary = {
        "status": "GLOBAL_ORBIT_POSTBUILD_AUDIT",
        "inputs": {"orig10": args.orig10, "per10": args.per10, "stable10": args.stable10, "manifest": args.manifest, "ewok_panel": args.ewok_panel},
        "manifest_status": manifest.get("status"),
        "counts": dict(counters),
        "order_hashes": {k:v.hexdigest() for k,v in order_hashes.items()},
        "manifest_order_contract": manifest.get("order_contract"),
        "stable_alias_entropy": {
            "keys": len(stable_key_alias),
            "keys_with_multiple_aliases": len(stable_multi),
            "assigned_events": stable_events,
            "events_on_multi_alias_keys": sum(sum(c.values()) for c in stable_multi.values()),
            "mean_entropy_bits_per_key": sum(entropy(c) for c in stable_key_alias.values()) / max(1, len(stable_key_alias)),
            "p_same_alias_given_key_pair": pair_same_probability(stable_key_alias),
        },
        "per_row_alias_entropy": {
            "keys": len(per_key_alias),
            "keys_with_multiple_aliases": len(per_multi),
            "assigned_events": per_events,
            "events_on_multi_alias_keys": sum(sum(c.values()) for c in per_multi.values()),
            "mean_entropy_bits_per_key": sum(entropy(c) for c in per_key_alias.values()) / max(1, len(per_key_alias)),
            "p_same_alias_given_key_pair": pair_same_probability(per_key_alias),
            "events_same_as_stable_alias": same_as_stable,
            "fraction_events_same_as_stable_alias": same_as_stable / per_events if per_events else 0.0,
        },
        "inverse_alias_collisions_stable": {
            "aliases": len(alias_key),
            "aliases_with_multiple_source_keys": len(inverse_multi),
            "events_on_multi_key_aliases": sum(sum(c.values()) for c in inverse_multi.values()),
            "mean_entropy_key_given_alias": sum(entropy(c) for c in alias_key.values()) / max(1, len(alias_key)),
            "top_multi_key_aliases": [{"alias": a, "events": sum(c.values()), "n_keys": len(c), "top_keys": c.most_common(8)} for a,c in sorted(inverse_multi.items(), key=lambda kv: (-sum(kv[1].values()), -len(kv[1]), kv[0]))[:40]],
        },
        "marginals": {
            "stable_alias_types": len(stable_alias_marginal),
            "per_alias_types": len(per_alias_marginal),
            "stable_top_aliases": stable_alias_marginal.most_common(30),
            "per_top_aliases": per_alias_marginal.most_common(30),
        },
        "repeated_key_dose": {
            "events_on_repeated_keys": repeat_key_events,
            "events_on_singleton_keys": singleton_key_events,
            "fraction_events_on_repeated_keys": repeat_key_events / stable_events if stable_events else 0.0,
            "keys_repeated": sum(1 for c in stable_key_alias.values() if sum(c.values()) >= 2),
            "keys_singleton": sum(1 for c in stable_key_alias.values() if sum(c.values()) == 1),
        },
        "ewok_lexical_overlap": {
            "ewok_items": {"all": ewok["all_items"], "relational": ewok["rel_items"]},
            "transformed_keys_in_relational_ewok_lexicon": len(rel_keys),
            "transformed_keyevents_in_relational_ewok_lexicon": sum(sum(stable_key_alias[k].values()) for k in rel_keys),
            "transformed_keys_in_any_ewok_lexicon": len(all_ewok_keys),
            "transformed_keyevents_in_any_ewok_lexicon": sum(sum(stable_key_alias[k].values()) for k in all_ewok_keys),
            "changed_row_context_overlap": dict(relation_context_rows),
            "top_transformed_rel_lexicon_keys": [(k, transformed_keys[k]) for k in sorted(rel_keys, key=lambda k: (-transformed_keys[k], k))[:50]],
            "top_transformed_keys_by_context_rel_overlap": transformed_context_rellex.most_common(50),
        },
        "changed_support_mismatch_examples": changed_support_mismatch,
        "changed_examples": changed_examples,
        "interpretation": "This audit is lexical and independent of model training. Zero stable alias entropy and identical changed-row support are required before training; inverse alias collisions, alias marginals, repeated-key dose, and relation/EWoK lexical overlap quantify what a later per-row/stable model contrast could mean.",
    }
    out_json = out_dir / "global_orbit_pair_postbuild_audit.json"
    out_md = out_dir / "global_orbit_pair_postbuild_audit.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = []
    md.append("# research post-build audit of global-compatible orbit pair\n\n")
    md.append(f"Rows/words: orig {counters['orig_rows']}/{counters['orig_words']}, per {counters['per_rows']}/{counters['per_words']}, stable {counters['stable_rows']}/{counters['stable_words']}\n\n")
    md.append(f"Changed rows: per {counters['per_changed_rows']}, stable {counters['stable_changed_rows']}; changed row mismatch {counters['changed_row_mismatch']}; changed key support mismatch {counters['changed_key_support_mismatch']}\n\n")
    md.append("## Alias persistence\n\n")
    md.append(f"Stable: {len(stable_key_alias)} keys, multi-alias keys {len(stable_multi)}, P_same={summary['stable_alias_entropy']['p_same_alias_given_key_pair']:.4f}, mean H(A|K)={summary['stable_alias_entropy']['mean_entropy_bits_per_key']:.4f}\n\n")
    md.append(f"Per-row: {len(per_key_alias)} keys, multi-alias keys {len(per_multi)}, P_same={summary['per_row_alias_entropy']['p_same_alias_given_key_pair']:.4f}, mean H(A|K)={summary['per_row_alias_entropy']['mean_entropy_bits_per_key']:.4f}, same-as-stable events {same_as_stable}/{per_events}={summary['per_row_alias_entropy']['fraction_events_same_as_stable_alias']:.4f}\n\n")
    md.append("## Repeated-key dose\n\n")
    md.append(f"Events on repeated keys: {repeat_key_events}/{stable_events} = {summary['repeated_key_dose']['fraction_events_on_repeated_keys']:.4f}; keys repeated/singleton {summary['repeated_key_dose']['keys_repeated']}/{summary['repeated_key_dose']['keys_singleton']}\n\n")
    md.append("## Inverse alias collisions in stable arm\n\n")
    inv = summary['inverse_alias_collisions_stable']
    md.append(f"Aliases: {inv['aliases']}; aliases with multiple source keys: {inv['aliases_with_multiple_source_keys']}; events on them: {inv['events_on_multi_key_aliases']}; mean H(K|A)={inv['mean_entropy_key_given_alias']:.4f}\n\n")
    md.append("## EWoK lexical overlap\n\n")
    ew = summary['ewok_lexical_overlap']
    md.append(f"Transformed keys in relational EWoK lexicon: {ew['transformed_keys_in_relational_ewok_lexicon']} keys / {ew['transformed_keyevents_in_relational_ewok_lexicon']} key-events\n\n")
    md.append(f"Transformed keys in any EWoK lexicon: {ew['transformed_keys_in_any_ewok_lexicon']} keys / {ew['transformed_keyevents_in_any_ewok_lexicon']} key-events\n\n")
    cro = ew['changed_row_context_overlap']
    md.append(f"Changed rows with >=1 relational EWoK token in context: {cro.get('rows_with_rel_ewok_token',0)}/{cro.get('changed_rows',0)}; with any EWoK token: {cro.get('rows_with_any_ewok_token',0)}/{cro.get('changed_rows',0)}\n\n")
    md.append("Top transformed keys that are themselves in the relational EWoK lexicon:\n\n")
    md.append(str(ew['top_transformed_rel_lexicon_keys'][:30]) + "\n\n")
    md.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(md), encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "stable_multi_alias_keys": len(stable_multi),
        "per_multi_alias_keys": len(per_multi),
        "stable_p_same": summary['stable_alias_entropy']['p_same_alias_given_key_pair'],
        "per_p_same": summary['per_row_alias_entropy']['p_same_alias_given_key_pair'],
        "repeated_event_fraction": summary['repeated_key_dose']['fraction_events_on_repeated_keys'],
        "inverse_multi_aliases": inv['aliases_with_multiple_source_keys'],
        "rel_ewok_keyevents": ew['transformed_keyevents_in_relational_ewok_lexicon'],
        "changed_rows_rel_context": cro.get('rows_with_rel_ewok_token',0),
        "out": str(out_json),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
