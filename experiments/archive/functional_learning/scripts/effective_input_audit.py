#!/usr/bin/env python3
"""research effective-input audit for the PosAlign relational harness.

Scientific purpose
------------------
The research repetition-vs-variation contrast assumed that VARY_CONTEXT forces
abstraction over name tuples.  In the actual PosAlign scorer, after successful
hard matching, name tokens are replaced by learned candidate/other vectors before
the GRU.  Therefore raw name identities are not effective inputs to the
relational learner.  This audit computes canonical effective token patterns for
state/comparison train/eval rows after replacing the two matched names by C/O,
and compares REPEAT/VARY_CONTEXT/VARY_NOISE against eval.

The goal is to decide whether completing long research runs would test a real
invariance mechanism or only repeat an already-supplied invariance.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

A01_SCRIPT_DIR = Path("experiments/archive/representation_and_objectives/training/scripts")
sys.path.insert(0, str(A01_SCRIPT_DIR))

import raw_span_discovery_probe as base  # noqa: E402

# Reuse constants/generators from research.  This import is cheap and does not run training.
FUNCTIONAL_RELATION_STUDIES_SCRIPT_DIR = Path("experiments/archive/functional_learning/scripts")
sys.path.insert(0, str(FUNCTIONAL_RELATION_STUDIES_SCRIPT_DIR))
import repeat_vs_variation as s5  # noqa: E402

OUT = Path("experiments/archive/functional_learning/data/effective_input_audit")
OUT.mkdir(parents=True, exist_ok=True)

NAME_MARKERS = {"cand": "<C>", "other": "<O>"}


def raw_tokens(text: str) -> List[str]:
    toks, _ = base.raw_tokenize(text)
    return toks


def effective_tokens(text: str, names: Tuple[str, str]) -> Tuple[str, ...]:
    """Replace the two surface names by candidate/other markers.

    This mirrors the information available to the GRU under hard matched
    PosAlign when assignment succeeds: the word embedding at a candidate token is
    replaced by cand_param and the word embedding at the other-name token is
    replaced by other_param.  We ignore continuous values and record the symbolic
    pattern.
    """
    n0, n1 = names[0].lower(), names[1].lower()
    out = []
    for tok in raw_tokens(text):
        if tok == n0:
            out.append(NAME_MARKERS["cand"])
        elif tok == n1:
            out.append(NAME_MARKERS["other"])
        else:
            out.append(tok)
    return tuple(out)


def abstract_tokens(eff: Iterable[str], *, object_mode: str = "actual") -> Tuple[str, ...]:
    """Optionally replace the object token by a class marker.

    In this substrate the object appears in the fixed local context
    `during the OBJ episode`.  The first audit used an incomplete object lexicon;
    this context rule correctly abstracts train, generated, and eval objects.
    """
    toks = list(eff)
    out = []
    for i, tok in enumerate(toks):
        if (
            object_mode == "abstract"
            and i >= 2
            and i + 1 < len(toks)
            and toks[i - 2] == "during"
            and toks[i - 1] == "the"
            and toks[i + 1] == "episode"
        ):
            out.append("<OBJ>")
        else:
            out.append(tok)
    return tuple(out)


def voice_of_event(text: str) -> str:
    return "passive" if " was " in text and " by " in text else "active"


def object_of_event(text: str) -> str:
    m = re.search(r"During the ([a-zA-Z]+) episode", text)
    return m.group(1).lower() if m else "?"


def summarize_patterns(rows, event_getter, names_getter, label: str) -> dict:
    eff_counter = Counter()
    abs_counter = Counter()
    object_counter = Counter()
    voice_counter = Counter()
    examples = {}
    for r in rows:
        event = event_getter(r)
        names = names_getter(r)
        eff = effective_tokens(event, names)
        abs_eff = abstract_tokens(eff, object_mode="abstract")
        eff_counter[eff] += 1
        abs_counter[abs_eff] += 1
        object_counter[object_of_event(event)] += 1
        voice_counter[voice_of_event(event)] += 1
        examples.setdefault(" ".join(eff), event)
    return {
        "label": label,
        "n_events": sum(eff_counter.values()),
        "unique_effective_patterns": len(eff_counter),
        "unique_abstract_object_patterns": len(abs_counter),
        "objects": dict(sorted(object_counter.items())),
        "voices": dict(sorted(voice_counter.items())),
        "top_effective_patterns": [(" ".join(k), v, examples[" ".join(k)]) for k, v in eff_counter.most_common(20)],
        "effective_counter": {" ".join(k): v for k, v in eff_counter.items()},
        "abstract_counter": {" ".join(k): v for k, v in abs_counter.items()},
    }


def event_records_from_comps(comps):
    out = []
    for c in comps:
        out.append((c.event1, c.names, c.relation1, c.row_id, "event1"))
        out.append((c.event2, c.names, c.relation2, c.row_id, "event2"))
    return out


def summarize_event_records(records, label: str) -> dict:
    by_rel = defaultdict(list)
    for ev, names, rel, row_id, side in records:
        by_rel[rel].append({"event": ev, "names": names, "row_id": row_id, "side": side})
    overall = summarize_patterns(
        [{"event": ev, "names": names} for ev, names, *_ in records],
        lambda x: x["event"], lambda x: tuple(x["names"]), label,
    )
    per_rel = {}
    for rel, rs in sorted(by_rel.items()):
        per_rel[rel] = summarize_patterns(rs, lambda x: x["event"], lambda x: tuple(x["names"]), f"{label}:{rel}")
    overall["per_relation"] = per_rel
    return overall


def summarize_state_queries(queries, label: str) -> dict:
    """Summarize changed state events overall and per relation."""
    qs = [q for q in queries if q.is_changed]
    overall = summarize_patterns(qs, lambda q: q.event, lambda q: q.names, label)
    by_rel = defaultdict(list)
    for q in qs:
        by_rel[q.relation].append(q)
    per_rel = {}
    for rel, rs in sorted(by_rel.items()):
        per_rel[rel] = summarize_patterns(rs, lambda q: q.event, lambda q: q.names, f"{label}:{rel}")
    overall["per_relation"] = per_rel
    return overall


def set_stats(a_counter: dict, b_counter: dict) -> dict:
    a, b = set(a_counter), set(b_counter)
    return {
        "a_size": len(a),
        "b_size": len(b),
        "intersection": len(a & b),
        "a_minus_b": len(a - b),
        "b_minus_a": len(b - a),
        "jaccard": len(a & b) / len(a | b) if (a | b) else 1.0,
        "examples_a_minus_b": sorted(list(a - b))[:12],
        "examples_b_minus_a": sorted(list(b - a))[:12],
    }


def main():
    ts, tc, es, ec, pe, counts = base.load_dataset(base.DEFAULT_DATA_ROOT, base.DEFAULT_ARM)
    if pe:
        print("parse_errors", pe[:5], file=sys.stderr)

    # Base fixed comparison events.
    fixed_cmp = summarize_event_records(event_records_from_comps(tc), "fixed_train_comparison_events")
    eval_state_changed = summarize_state_queries(es, "eval_changed_state_events")
    train_state_changed = summarize_state_queries(ts, "train_changed_state_events")

    # Generate one or three epochs of research comparison variants without training.
    pool = s5.generate_pool(s5.EXTRA_NAMES, s5.EXTRA_OBJECTS, 48, 30000)
    vary_ctx_e1 = s5.sample_epoch_comps(pool, 48, 1, 30000)
    vary_ctx_e1_e3 = []
    for ep in range(1, 4):
        vary_ctx_e1_e3.extend(s5.sample_epoch_comps(pool, 48, ep, 30000))
    vary_noise_e1 = s5.add_noise_to_comps(tc, 1, 30000)
    vary_noise_e1_e3 = []
    for ep in range(1, 4):
        vary_noise_e1_e3.extend(s5.add_noise_to_comps(tc, ep, 30000))

    summaries = {
        "counts": counts,
        "fixed_train_comparison_events": fixed_cmp,
        "train_changed_state_events": train_state_changed,
        "eval_changed_state_events": eval_state_changed,
        "vary_context_e1_events": summarize_event_records(event_records_from_comps(vary_ctx_e1), "vary_context_e1_events"),
        "vary_context_e1_e3_events": summarize_event_records(event_records_from_comps(vary_ctx_e1_e3), "vary_context_e1_e3_events"),
        "vary_noise_e1_events": summarize_event_records(event_records_from_comps(vary_noise_e1), "vary_noise_e1_events"),
        "vary_noise_e1_e3_events": summarize_event_records(event_records_from_comps(vary_noise_e1_e3), "vary_noise_e1_e3_events"),
    }

    comparisons = {}
    keys = [
        ("fixed_train_comparison_events", "eval_changed_state_events"),
        ("fixed_train_comparison_events", "vary_context_e1_events"),
        ("fixed_train_comparison_events", "vary_noise_e1_events"),
        ("vary_context_e1_events", "eval_changed_state_events"),
        ("vary_noise_e1_events", "eval_changed_state_events"),
    ]
    for a, b in keys:
        comparisons[f"{a}__vs__{b}__effective"] = set_stats(
            summaries[a]["effective_counter"], summaries[b]["effective_counter"]
        )
        comparisons[f"{a}__vs__{b}__abstract_object"] = set_stats(
            summaries[a]["abstract_counter"], summaries[b]["abstract_counter"]
        )

    # Relation-level effective overlap with eval, using abstracted object patterns.
    rel_overlap = {}
    for rel in ["h0_dax", "h1_mep", "h2_norp", "h3_ziv"]:
        rel_overlap[rel] = {}
        eval_rel = summaries["eval_changed_state_events"].get("per_relation", {}).get(rel)
        if not eval_rel:
            continue
        eval_counter = eval_rel["abstract_counter"]
        for src in ["fixed_train_comparison_events", "vary_context_e1_events", "vary_noise_e1_events"]:
            if rel in summaries[src].get("per_relation", {}):
                rel_overlap[rel][src] = set_stats(summaries[src]["per_relation"][rel]["abstract_counter"], eval_counter)

    result = {"summaries": summaries, "comparisons": comparisons, "relation_abstract_overlap_with_eval": rel_overlap}
    (OUT / "effective_input_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# research effective-input audit for research repetition-vs-variation",
        "",
        "PosAlign hard matching replaces each matched surface name by the same candidate/other parameter vectors before the GRU. Raw name identity therefore does not reach the relational learner when matching succeeds. The audit below canonicalizes each event by replacing the two names with `<C>` and `<O>`.",
        "",
        "## Pattern counts after name replacement",
        "",
        "| set | events | unique effective patterns | unique patterns after object abstraction | voices | objects |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for key in ["fixed_train_comparison_events", "vary_context_e1_events", "vary_context_e1_e3_events", "vary_noise_e1_events", "vary_noise_e1_e3_events", "train_changed_state_events", "eval_changed_state_events"]:
        s = summaries[key]
        voices = ", ".join(f"{k}:{v}" for k, v in s["voices"].items())
        lines.append(f"| {key} | {s['n_events']} | {s['unique_effective_patterns']} | {s['unique_abstract_object_patterns']} | {voices} | {len(s['objects'])} |")
    lines.extend(["", "## Set overlap with eval and with fixed train", "", "| comparison | space | A | B | intersection | A minus B | B minus A | Jaccard |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for k, v in comparisons.items():
        space = "abstract_object" if k.endswith("abstract_object") else "effective"
        comp = k.replace("__effective", "").replace("__abstract_object", "")
        lines.append(f"| {comp} | {space} | {v['a_size']} | {v['b_size']} | {v['intersection']} | {v['a_minus_b']} | {v['b_minus_a']} | {v['jaccard']:.3f} |")
    lines.extend(["", "## Top fixed comparison effective patterns", ""])
    for pat, n, ex in fixed_cmp["top_effective_patterns"][:12]:
        lines.append(f"- `{pat}` ×{n}; raw example: {ex}")
    lines.extend(["", "## Top eval changed-state effective patterns", ""])
    for pat, n, ex in eval_state_changed["top_effective_patterns"][:12]:
        lines.append(f"- `{pat}` ×{n}; raw example: {ex}")
    lines.extend(["", "## Consequence for repaired experiment", "", "Name variation in VARY_CONTEXT is not an effective-input manipulation under hard PosAlign. The remaining visible changes are object lexical identity, voice/order, and added filler tokens; the existing base comparison set already contains many objects and both active/passive voices. A stronger contrast must manipulate visible nuisance tokens that can actually compete with the relational solution, or remove the supplied name-invariance and compare learning the matcher with using a frozen matcher.", ""])
    ((OUT.parents[4] / 'research/documents/functional_learning/data/effective_input_audit/effective_input_audit.md')).write_text("\n".join(lines))

    print(json.dumps({
        "status": "EFFECTIVE_INPUT_AUDIT_DONE",
        "out_md": str((OUT.parents[4] / 'research/documents/functional_learning/data/effective_input_audit/effective_input_audit.md')),
        "out_json": str(OUT / "effective_input_audit.json"),
        "fixed_unique_effective": fixed_cmp["unique_effective_patterns"],
        "vary_ctx_e1_unique_effective": summaries["vary_context_e1_events"]["unique_effective_patterns"],
        "vary_noise_e1_unique_effective": summaries["vary_noise_e1_events"]["unique_effective_patterns"],
        "eval_unique_effective": eval_state_changed["unique_effective_patterns"],
        "fixed_vs_eval_abstract_jaccard": comparisons["fixed_train_comparison_events__vs__eval_changed_state_events__abstract_object"]["jaccard"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
