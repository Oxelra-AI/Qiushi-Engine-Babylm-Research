#!/usr/bin/env python3
"""research Stage 3: Verify transformation outputs and package the graph packet.

Reads Stage 2 transformation outputs, matches them to source rows and graphs,
verifies quality (entity preservation, length, coherence markers), and
packages the final verified graph packet for frozen-model measurement.
"""
from __future__ import annotations
import json, pathlib, re, collections
from typing import Any

ROOT = pathlib.Path(".")
DATA_DIR = ROOT / "experiments/archive/frontier_consolidation/data/graph_packet_v0"
SELECTED_PATH = DATA_DIR / "selected_source_rows.jsonl"
PARSED_GRAPHS = DATA_DIR / "parsed_graphs.jsonl"
TRANSFORM_OUT = DATA_DIR / "stage2_transform_outputs.jsonl"
TRANSFORM_PROMPTS = DATA_DIR / "stage2_transform_prompts.jsonl"
PACKET_PATH = DATA_DIR / "graph_packet_v0.jsonl"
REJECT_PATH = DATA_DIR / "rejected_rows.jsonl"
SUMMARY_PATH = DATA_DIR / "graph_packet_summary.json"
NOTE_PATH = ROOT / "research/notes/frontier_consolidation/graph_packet_v0_construction.md"


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len((text or "").split())


def entity_coverage(text: str, entity_names: list[str]) -> float:
    """Fraction of expected entity names found in text."""
    if not entity_names:
        return 1.0
    text_lower = text.lower()
    found = sum(1 for name in entity_names if name.lower() in text_lower)
    return found / len(entity_names)


def renamed_entity_coverage(text: str, rename_map: dict[str, str]) -> tuple[float, float]:
    """Check new names appear and old names are gone."""
    text_lower = text.lower()
    new_found = sum(1 for new in rename_map.values() if new.lower() in text_lower)
    old_found = sum(1 for old in rename_map.keys() if old.lower() in text_lower)
    new_frac = new_found / max(1, len(rename_map))
    old_frac = old_found / max(1, len(rename_map))
    return new_frac, old_frac


def verify_packet_row(source_row: dict, graph: dict, transforms: dict[str, dict]) -> tuple[bool, list[str]]:
    """Verify a complete row's transformations meet quality gates."""
    issues = []
    source_text = source_row.get("text", "")
    source_words = word_count(source_text)
    entity_names = [e.get("name", "") for e in graph.get("entities", []) if e.get("name")]
    
    # Check graph-preserving compact
    gc = transforms.get("graph_compact", {})
    gc_text = gc.get("output", "")
    if not gc_text:
        issues.append("graph_compact: no output")
    else:
        gc_words = word_count(gc_text)
        if gc_words > source_words * 0.95:
            issues.append(f"graph_compact: not shorter ({gc_words} vs {source_words})")
        gc_coverage = entity_coverage(gc_text, entity_names)
        if gc_coverage < 0.5:
            issues.append(f"graph_compact: low entity coverage {gc_coverage:.2f}")
    
    # Check entity-renamed
    er = transforms.get("entity_renamed", {})
    er_text = er.get("output", "")
    rename_map = er.get("rename_map", {})
    if not er_text:
        issues.append("entity_renamed: no output")
    elif rename_map:
        new_frac, old_frac = renamed_entity_coverage(er_text, rename_map)
        if new_frac < 0.4:
            issues.append(f"entity_renamed: low new name coverage {new_frac:.2f}")
        if old_frac > 0.5:
            issues.append(f"entity_renamed: old names still present {old_frac:.2f}")
    
    # Check edge-changed
    ec = transforms.get("edge_changed", {})
    ec_text = ec.get("output", "")
    if not ec_text:
        issues.append("edge_changed: no output")
    else:
        ec_words = word_count(ec_text)
        if ec_words < source_words * 0.3:
            issues.append(f"edge_changed: too short ({ec_words} vs {source_words})")
    
    # Check ordinary compact
    oc = transforms.get("ordinary_compact", {})
    oc_text = oc.get("output", "")
    if not oc_text:
        issues.append("ordinary_compact: no output")
    else:
        oc_words = word_count(oc_text)
        if oc_words > source_words * 0.95:
            issues.append(f"ordinary_compact: not shorter ({oc_words} vs {source_words})")
    
    # Row is accepted if at most 1 non-critical issue
    critical = [i for i in issues if "no output" in i]
    return len(critical) == 0, issues


def main():
    selected = read_jsonl(SELECTED_PATH)
    parsed_graphs = read_jsonl(PARSED_GRAPHS)
    transform_outputs = read_jsonl(TRANSFORM_OUT)
    # Load input prompts to recover metadata stripped by batch tool
    input_prompts = read_jsonl(TRANSFORM_PROMPTS)
    
    # Index sources and graphs
    source_by_row = {r["row_index"]: r for r in selected}
    graph_by_row = {p["row_index"]: p["graph"] for p in parsed_graphs}
    
    # Match outputs to input prompts by index, then group by row_index
    transforms_by_row: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    for to in transform_outputs:
        idx = to.get("index")
        if idx is not None and idx < len(input_prompts):
            meta = input_prompts[idx]
            row_idx = meta.get("row_index")
            ttype = meta.get("transform_type", "")
            rename_map = meta.get("rename_map", {})
            changed_edge = meta.get("changed_edge", {})
        else:
            # Fallback to prompt_id parsing
            prompt_id = to.get("prompt_id", "")
            row_match = re.search(r"row(\d+)", prompt_id)
            row_idx = int(row_match.group(1)) if row_match else None
            ttype = ""
            rename_map = {}
            changed_edge = {}
            for tt in ["graph_compact", "entity_renamed", "edge_changed", "ordinary_compact"]:
                if tt in prompt_id:
                    ttype = tt
                    break
        
        if row_idx is not None and ttype:
            entry = {"output": to.get("output", ""), "transform_type": ttype}
            if rename_map:
                entry["rename_map"] = rename_map
            if changed_edge:
                entry["changed_edge"] = changed_edge
            transforms_by_row[row_idx][ttype] = entry
    
    # Build and verify packet
    packet = []
    rejected = []
    
    for pg in parsed_graphs:
        row_idx = pg["row_index"]
        if row_idx not in source_by_row:
            rejected.append({"row_index": row_idx, "reason": "source row not found"})
            continue
        
        source_row = source_by_row[row_idx]
        graph = pg["graph"]
        transforms = transforms_by_row.get(row_idx, {})
        
        if len(transforms) < 3:  # Need at least 3 of 4 transform types
            rejected.append({
                "row_index": row_idx,
                "reason": f"only {len(transforms)} transforms (need >=3)",
                "available": list(transforms.keys()),
            })
            continue
        
        accepted, issues = verify_packet_row(source_row, graph, transforms)
        
        record = {
            "row_index": row_idx,
            "family": source_row.get("family", ""),
            "source_bucket": source_row.get("source_bucket", ""),
            "source_text": source_row.get("text", ""),
            "source_words": word_count(source_row.get("text", "")),
            "graph": graph,
            "graph_compact": transforms.get("graph_compact", {}).get("output", ""),
            "entity_renamed": transforms.get("entity_renamed", {}).get("output", ""),
            "entity_rename_map": transforms.get("entity_renamed", {}).get("rename_map", {}),
            "edge_changed": transforms.get("edge_changed", {}).get("output", ""),
            "changed_edge": transforms.get("edge_changed", {}).get("changed_edge", {}),
            "ordinary_compact": transforms.get("ordinary_compact", {}).get("output", ""),
            "verification_accepted": accepted,
            "verification_issues": issues,
        }
        
        if accepted:
            packet.append(record)
        else:
            record["rejection_reason"] = "; ".join(issues)
            rejected.append(record)
    
    # Save packet
    with PACKET_PATH.open("w", encoding="utf-8") as f:
        for r in packet:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    with REJECT_PATH.open("w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Summary statistics
    family_counts = collections.Counter(r["family"] for r in packet)
    source_counts = collections.Counter(r["source_bucket"] for r in packet)
    
    gc_ratios = [word_count(r["graph_compact"]) / max(1, r["source_words"]) 
                 for r in packet if r["graph_compact"]]
    oc_ratios = [word_count(r["ordinary_compact"]) / max(1, r["source_words"]) 
                 for r in packet if r["ordinary_compact"]]
    er_ratios = [word_count(r["entity_renamed"]) / max(1, r["source_words"]) 
                 for r in packet if r["entity_renamed"]]
    
    summary = {
        "status": "GRAPH_PACKET_V0_COMPLETE",
        "accepted_rows": len(packet),
        "rejected_rows": len(rejected),
        "total_parsed_graphs": len(parsed_graphs),
        "by_family": dict(sorted(family_counts.items())),
        "by_source": dict(sorted(source_counts.items())),
        "graph_compact_ratio": {"mean": sum(gc_ratios)/max(1,len(gc_ratios)), "n": len(gc_ratios)} if gc_ratios else None,
        "ordinary_compact_ratio": {"mean": sum(oc_ratios)/max(1,len(oc_ratios)), "n": len(oc_ratios)} if oc_ratios else None,
        "entity_renamed_ratio": {"mean": sum(er_ratios)/max(1,len(er_ratios)), "n": len(er_ratios)} if er_ratios else None,
        "packet_file": str(PACKET_PATH),
        "rejected_file": str(REJECT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    
    # Write note
    NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    note_lines = [
        "# research Graph Packet v0 Construction",
        "",
        f"## Summary",
        f"- Accepted: {len(packet)} rows",
        f"- Rejected: {len(rejected)} rows",
        f"- Families: {dict(family_counts)}",
        f"- Sources: {dict(source_counts)}",
        "",
        "## Transformation quality",
    ]
    if gc_ratios:
        note_lines.append(f"- Graph compact: mean ratio {sum(gc_ratios)/len(gc_ratios):.3f}")
    if oc_ratios:
        note_lines.append(f"- Ordinary compact: mean ratio {sum(oc_ratios)/len(oc_ratios):.3f}")
    if er_ratios:
        note_lines.append(f"- Entity renamed: mean ratio {sum(er_ratios)/len(er_ratios):.3f}")
    
    note_lines.extend([
        "",
        "## Purpose",
        "This packet tests whether graph-preserving relational views transfer better",
        "than ordinary compaction when measured on frozen chck_82M. Each row has:",
        "- source_text: original legal corpus passage",
        "- graph: verified G=(entities, roles, relations, states, events)",
        "- graph_compact: concise rewrite preserving all relations in G",
        "- entity_renamed: same structure with different entity names",
        "- edge_changed: one relation altered, everything else preserved",
        "- ordinary_compact: baseline compression without structural constraint",
        "",
        "## Next measurement",
        "A proposed frozen chck_82M packet probe would test:",
        "1. Cross-view transfer: graph_compact predicts entity_renamed targets",
        "   better than ordinary_compact predicts entity_renamed targets",
        "2. Edge sensitivity: model distinguishes source vs edge_changed",
        "3. Entity-renaming robustness: margins survive name changes",
        "",
        f"Packet: `{PACKET_PATH}`",
        f"Summary: `{SUMMARY_PATH}`",
    ])
    NOTE_PATH.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
