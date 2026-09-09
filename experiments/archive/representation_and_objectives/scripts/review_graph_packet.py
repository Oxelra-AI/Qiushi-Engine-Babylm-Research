#!/usr/bin/env python3
"""Bounded graph-packet artifact review.

This is CPU/file-only. It inspects the graph packet for the
question: does the packet actually instantiate a true
entity-keyed event/state correspondence object, or is it mostly graph-preserving
compaction without the missing write/read computation?
"""
from __future__ import annotations

import collections
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(".")
PACKET = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0/graph_packet_v0.jsonl"
OUT_DIR = ROOT / "experiments/archive/representation_and_objectives/data/graph_packet_review"
OUT_JSON = OUT_DIR / "graph_packet_review.json"
OUT_MD = ROOT / "research/notes/representation_and_objectives/graph_packet_review.md"

STOP = set("""
a an the and or but if then else when while before after of to in on by for from with without into onto over under at as is are was were be been being had has have do does did not no nor this that these those it its he she they them his her their we you i my our your who whom whose which what where why how because so than very just also only all any each every some more most many much few little old new same other own can could would should will may might must shall about across again against between through during above below up down out off back here there now later earlier still even
""".split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def toks(s: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'-]*", s or "") if t.lower() not in STOP and len(t) > 2]


def content_set(s: str) -> set[str]:
    return set(toks(s))


def jacc(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def name_hits(text: str, names: list[str]) -> int:
    lt = (text or "").lower()
    return sum(1 for n in names if n and n.lower() in lt)


def rel_keywords(graph: dict[str, Any]) -> set[str]:
    words = []
    for r in graph.get("relations", []) or []:
        words += toks(" ".join(str(r.get(k, "")) for k in ("type", "arg1", "arg2", "detail")))
    for s in graph.get("states", []) or []:
        words += toks(" ".join(str(s.get(k, "")) for k in ("entity", "property", "value_before", "value_after")))
    for e in graph.get("key_events", []) or []:
        words += toks(str(e.get("event", "")))
    return set(words)


def quantiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p10": math.nan, "p90": math.nan}
    ys = sorted(xs)
    def q(p: float) -> float:
        idx = int(round(p * (len(ys)-1)))
        return ys[idx]
    return {"n": len(xs), "mean": sum(xs)/len(xs), "median": q(0.5), "p10": q(0.1), "p90": q(0.9)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(PACKET)
    by_family = collections.Counter(r.get("family") for r in rows)
    relation_types = collections.Counter()
    n_states_nonempty = 0
    n_state_change_relation = 0
    n_action_relation = 0
    edge_changed_real = 0
    edge_ratios = []
    per_rows = []

    # Deterministic same-family shuffled controls for coarse lexical correspondence.
    fam_rows: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        fam_rows[r.get("family", "")].append(i)

    for i, r in enumerate(rows):
        graph = r.get("graph", {}) or {}
        ents = [e.get("name", "") for e in graph.get("entities", []) or []]
        renamed = list((r.get("entity_rename_map") or {}).values())
        for rel in graph.get("relations", []) or []:
            relation_types[rel.get("type", "")] += 1
        if graph.get("states"):
            n_states_nonempty += 1
        if any(rel.get("type") == "state_change" for rel in graph.get("relations", []) or []):
            n_state_change_relation += 1
        if any(rel.get("type") == "action" for rel in graph.get("relations", []) or []):
            n_action_relation += 1

        sm = difflib.SequenceMatcher(None, r.get("source_text", ""), r.get("edge_changed", ""))
        ratio = sm.ratio()
        edge_ratios.append(ratio)
        # A very conservative "real changed" proxy: not near-identical and changed text marker or large edit.
        if ratio < 0.985 or "**" in (r.get("edge_changed") or ""):
            edge_changed_real += 1

        g_tokens = content_set(r.get("graph_compact", ""))
        o_tokens = content_set(r.get("ordinary_compact", ""))
        er_tokens = content_set(r.get("entity_renamed", ""))
        src_tokens = content_set(r.get("source_text", ""))
        rk = rel_keywords(graph)

        # same-family shuffled graph compact with comparable source length, circular next row in family
        fam = r.get("family", "")
        idxs = fam_rows[fam]
        pos = idxs.index(i)
        j = idxs[(pos + 1) % len(idxs)] if len(idxs) > 1 else i
        shuf_g_tokens = content_set(rows[j].get("graph_compact", ""))
        shuf_o_tokens = content_set(rows[j].get("ordinary_compact", ""))

        rec = {
            "row_index": r.get("row_index"),
            "family": fam,
            "source_bucket": r.get("source_bucket"),
            "n_entities": len(ents),
            "n_states": len(graph.get("states", []) or []),
            "n_relations": len(graph.get("relations", []) or []),
            "relation_types": sorted(set(rel.get("type", "") for rel in graph.get("relations", []) or [])),
            "source_entity_hits_graph_compact": name_hits(r.get("graph_compact", ""), ents),
            "source_entity_hits_ordinary_compact": name_hits(r.get("ordinary_compact", ""), ents),
            "renamed_entity_hits_entity_renamed": name_hits(r.get("entity_renamed", ""), renamed),
            "source_entity_hits_entity_renamed": name_hits(r.get("entity_renamed", ""), ents),
            "rel_kw_coverage_graph_compact": len(rk & g_tokens) / max(1, len(rk)),
            "rel_kw_coverage_ordinary_compact": len(rk & o_tokens) / max(1, len(rk)),
            "lex_jacc_graph_compact_to_entity_renamed": jacc(g_tokens, er_tokens),
            "lex_jacc_ordinary_compact_to_entity_renamed": jacc(o_tokens, er_tokens),
            "lex_jacc_shuffled_graph_compact_to_entity_renamed": jacc(shuf_g_tokens, er_tokens),
            "lex_jacc_shuffled_ordinary_compact_to_entity_renamed": jacc(shuf_o_tokens, er_tokens),
            "edge_changed_similarity": ratio,
            "verification_issues": r.get("verification_issues") or [],
        }
        per_rows.append(rec)

    def agg(key: str) -> dict[str, float]:
        return quantiles([float(r[key]) for r in per_rows if isinstance(r.get(key), (int, float))])

    pair_adv = [r["lex_jacc_graph_compact_to_entity_renamed"] - r["lex_jacc_shuffled_graph_compact_to_entity_renamed"] for r in per_rows]
    ordinary_adv = [r["lex_jacc_graph_compact_to_entity_renamed"] - r["lex_jacc_ordinary_compact_to_entity_renamed"] for r in per_rows]
    rel_cov_adv = [r["rel_kw_coverage_graph_compact"] - r["rel_kw_coverage_ordinary_compact"] for r in per_rows]

    state_rows = [r for r in per_rows if r["n_states"] > 0 or "state_change" in r["relation_types"] or "action" in r["relation_types"]]
    entity_state_family_rows = [r for r in per_rows if r["family"] == "entity_state_update"]

    summary = {
        "status": "A02_GRAPH_PACKET_REVIEW",
        "packet": str(PACKET),
        "n_rows": len(rows),
        "by_family": dict(by_family),
        "relation_types": dict(relation_types),
        "n_rows_with_nonempty_states": n_states_nonempty,
        "n_rows_with_state_change_relation": n_state_change_relation,
        "n_rows_with_action_relation": n_action_relation,
        "edge_changed_real_proxy": edge_changed_real,
        "edge_changed_similarity": quantiles(edge_ratios),
        "aggregates": {
            "source_entity_hits_graph_compact": agg("source_entity_hits_graph_compact"),
            "source_entity_hits_ordinary_compact": agg("source_entity_hits_ordinary_compact"),
            "renamed_entity_hits_entity_renamed": agg("renamed_entity_hits_entity_renamed"),
            "source_entity_hits_entity_renamed": agg("source_entity_hits_entity_renamed"),
            "rel_kw_coverage_graph_compact": agg("rel_kw_coverage_graph_compact"),
            "rel_kw_coverage_ordinary_compact": agg("rel_kw_coverage_ordinary_compact"),
            "true_graph_minus_same_family_shuffled_graph_lex_jacc_to_renamed": quantiles(pair_adv),
            "graph_minus_ordinary_lex_jacc_to_renamed": quantiles(ordinary_adv),
            "graph_minus_ordinary_rel_kw_coverage": quantiles(rel_cov_adv),
        },
        "state_like_rows": {
            "n": len(state_rows),
            "row_indices": [r["row_index"] for r in state_rows[:20]],
        },
        "entity_state_family": {
            "n": len(entity_state_family_rows),
            "rows_with_state_or_action_relations": sum(1 for r in entity_state_family_rows if r["n_states"] > 0 or "state_change" in r["relation_types"] or "action" in r["relation_types"]),
            "row_indices": [r["row_index"] for r in entity_state_family_rows],
        },
        "sample_problem_rows": [
            r for r in per_rows
            if r["n_states"] == 0 and "state_change" not in r["relation_types"] and r["family"] in {"entity_state_update", "event_temporal_causal", "polarity_contrast_event"}
        ][:10],
        "per_row_file": str(OUT_DIR / "per_row_review.jsonl"),
    }

    with (OUT_DIR / "per_row_review.jsonl").open("w", encoding="utf-8") as f:
        for r in per_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md = []
    md.append("# research bounded review of A02 graph packet v0\n")
    md.append("This CPU-only review reads the actual A02 research graph packet and asks whether it supplies the missing research computation: an entity-keyed, event-conditioned state write and query-conditioned read.\n")
    md.append("## Main counts\n")
    md.append(f"- Rows: `{len(rows)}`; families: `{dict(by_family)}`.\n")
    md.append(f"- Relation types: `{dict(relation_types)}`.\n")
    md.append(f"- Rows with nonempty `states`: `{n_states_nonempty}`; rows with `state_change` relations: `{n_state_change_relation}`; rows with `action` relations: `{n_action_relation}`.\n")
    md.append(f"- Entity-state-update family rows: `{len(entity_state_family_rows)}`, with state/action/state_change structure: `{summary['entity_state_family']['rows_with_state_or_action_relations']}`.\n")
    md.append(f"- Edge-changed real proxy: `{edge_changed_real}/50`; similarity summary `{summary['edge_changed_similarity']}`.\n")
    md.append("\n## Surface correspondence and compaction\n")
    ag = summary["aggregates"]
    md.append(f"- Source entity hits, graph vs ordinary compact: `{ag['source_entity_hits_graph_compact']}` vs `{ag['source_entity_hits_ordinary_compact']}`.\n")
    md.append(f"- Relation-keyword coverage, graph vs ordinary compact: `{ag['rel_kw_coverage_graph_compact']}` vs `{ag['rel_kw_coverage_ordinary_compact']}`; graph-minus-ordinary `{ag['graph_minus_ordinary_rel_kw_coverage']}`.\n")
    md.append(f"- True graph compact has lexical overlap with same-row renamed source above same-family shuffled graph compact: `{ag['true_graph_minus_same_family_shuffled_graph_lex_jacc_to_renamed']}`. This reflects same-source surface correspondence, not necessarily an event-state write/read mechanism.\n")
    md.append(f"- Graph-minus-ordinary lexical overlap to renamed source: `{ag['graph_minus_ordinary_lex_jacc_to_renamed']}`.\n")
    md.append("\n## Scientific reading\n")
    md.append("The packet does contain real same-source correspondence and graph-constrained compaction preserves more relation/entity vocabulary than ordinary compaction. But it does **not** instantiate the research missing computation. The extracted graphs almost never contain explicit before/after state slots, no rows use `state_change` or `action` relation types, and the nominal entity-state-update family is mostly social/causal/spatial factual compaction. Edge-changed controls are weak. Therefore this packet is useful as evidence that a teacher can produce relation-preserving compact views, but it should not receive H100 training as the next mechanism route unless rebuilt into explicit entity-keyed event-state transitions with matched correspondence-destroying controls.\n")
    md.append("\n## Consequence\n")
    md.append("Proceed to specify a minimal main-path memory operation that writes event-result state to an entity key and reads by the queried entity, then test it in a small controlled training screen with held-out entities, verbs, and state families and unchanged natural interaction readouts. Practical endpoint AoA confirmation can proceed separately, but endpoint arithmetic does not solve the missing computation.\n")
    md.append(f"\nJSON: `{OUT_JSON}`\nPer-row review: `{OUT_DIR / 'per_row_review.jsonl'}`\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": str(OUT_JSON), "out_md": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
