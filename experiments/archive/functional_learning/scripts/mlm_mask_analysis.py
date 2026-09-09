#!/usr/bin/env python3
"""research: MLM mask analysis for contrastive entity-binding packets.

Scientific purpose
------------------
The critical scientific gap is that a beautifully paired packet can still
contribute almost no binding-sensitive training credit if the 15% whole-word masking
rarely masks the answer while leaving the relational evidence visible. This script
quantifies "usable relational supervision" by simulating the actual WWM process on
paired contrastive packets.

For each packet, it measures:
  1. P(answer word-group masked)  
  2. P(critical evidence visible | answer masked)
  3. P(useful relational supervision) = (1) × (2)
  4. P(shortcut available): answer visible elsewhere, source-copy suffices, etc.

It also identifies the actual word-group positions of answer spans, entity names,
update sentences, and source states to support subsequent focused masking interventions.

Usage
-----
  python mlm_mask_analysis.py \\
    --tokenizer-path <path_to_compliant16k> \\
    --packet-jsonl <paired_packets.jsonl> \\
    --out-dir <output_directory> \\
    [--n-simulations 10000]
"""
import argparse, json, pathlib, sys, re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

# ── Constants matching the actual training ──
MASK_PROB = 0.15
SEQ_LENGTH = 256
N_SIM_DEFAULT = 10_000


def is_word_start(token_str: str) -> bool:
    """Match the actual training tokenizer word-start detection."""
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def build_word_groups(token_ids: List[int], tokenizer, special_ids: set) -> List[int]:
    """Reproduce the exact word-group assignment from training."""
    groups = [-1] * len(token_ids)
    gid = -1
    for i, tid in enumerate(token_ids):
        if tid in special_ids:
            continue
        s = tokenizer.convert_ids_to_tokens(int(tid))
        flag = bool(s is not None and is_word_start(str(s)))
        if gid < 0 or flag or i == 0:
            gid += 1
        groups[i] = gid
    return groups


def find_span_word_groups(groups: List[int], token_ids: List[int], 
                          span_token_ids: List[int]) -> List[int]:
    """Find word-group IDs that contain the span tokens (first occurrence)."""
    n = len(span_token_ids)
    if n == 0:
        return []
    for start in range(len(token_ids) - n + 1):
        if token_ids[start:start+n] == span_token_ids:
            gids = set()
            for j in range(start, start+n):
                if groups[j] >= 0:
                    gids.add(groups[j])
            return sorted(gids)
    return []


def find_all_span_word_groups(groups: List[int], token_ids: List[int],
                              span_token_ids: List[int]) -> List[List[int]]:
    """Find ALL occurrences of span tokens and their word-group IDs."""
    n = len(span_token_ids)
    if n == 0:
        return []
    occurrences = []
    for start in range(len(token_ids) - n + 1):
        if token_ids[start:start+n] == span_token_ids:
            gids = set()
            for j in range(start, start+n):
                if groups[j] >= 0:
                    gids.add(groups[j])
            occurrences.append(sorted(gids))
    return occurrences


def simulate_wwm_supervision(
    n_total_groups: int,
    answer_groups: List[int],
    critical_evidence_groups: List[int],
    all_answer_occurrences: List[List[int]],
    n_sims: int = N_SIM_DEFAULT,
    mask_prob: float = MASK_PROB,
) -> Dict[str, float]:
    """Monte Carlo simulation of WWM masking statistics.
    
    Returns:
        answer_masked: P(at least one answer word group is masked)
        evidence_visible: P(all critical evidence groups are unmasked)  
        useful_supervision: P(answer masked AND evidence visible)
        shortcut_available: P(answer also visible elsewhere when answer location is masked)
    """
    import random
    rng = random.Random(42)
    
    answer_set = set(answer_groups)
    evidence_set = set(critical_evidence_groups)
    # Other occurrences of the answer text (beyond the primary in the use sentence)
    other_answer_occ_sets = []
    for occ in all_answer_occurrences:
        occ_set = set(occ)
        if occ_set != answer_set:  # not the primary answer location
            other_answer_occ_sets.append(occ_set)
    
    counts = defaultdict(int)
    for _ in range(n_sims):
        # Each word group independently masked with probability mask_prob
        masked = set()
        for g in range(n_total_groups):
            if rng.random() < mask_prob:
                masked.add(g)
        
        answer_hit = bool(answer_set & masked)
        evidence_all_visible = not bool(evidence_set & masked)
        useful = answer_hit and evidence_all_visible
        
        # Shortcut: answer text visible at another location
        shortcut = False
        if answer_hit:
            for occ_set in other_answer_occ_sets:
                if not (occ_set & masked):  # this other occurrence is fully visible
                    shortcut = True
                    break
        
        counts["answer_masked"] += answer_hit
        counts["evidence_visible"] += evidence_all_visible
        counts["useful_supervision"] += useful
        counts["shortcut_given_useful"] += (useful and shortcut)
        counts["useful_no_shortcut"] += (useful and not shortcut)
    
    n = max(1, n_sims)
    return {
        "answer_masked": counts["answer_masked"] / n,
        "evidence_visible": counts["evidence_visible"] / n,
        "useful_supervision": counts["useful_supervision"] / n,
        "shortcut_given_useful": counts["shortcut_given_useful"] / max(1, counts["useful_supervision"]),
        "useful_no_shortcut": counts["useful_no_shortcut"] / n,
        "n_answer_groups": len(answer_groups),
        "n_evidence_groups": len(critical_evidence_groups),
        "n_total_groups": n_total_groups,
        "n_other_answer_occurrences": len(other_answer_occ_sets),
    }


def analytic_estimates(n_answer_groups: int, n_evidence_groups: int,
                       mask_prob: float = MASK_PROB) -> Dict[str, float]:
    """Closed-form estimates assuming independence between groups."""
    p_ans = 1.0 - (1.0 - mask_prob) ** n_answer_groups
    p_ev = (1.0 - mask_prob) ** n_evidence_groups
    return {
        "analytic_answer_masked": p_ans,
        "analytic_evidence_visible": p_ev,
        "analytic_useful_supervision": p_ans * p_ev,
    }


def analyze_packet(packet: Dict[str, Any], tokenizer, special_ids: set,
                   n_sims: int = N_SIM_DEFAULT) -> Dict[str, Any]:
    """Analyze one contrastive packet for MLM binding supervision.
    
    Expected packet fields:
        full_text: the complete text that would appear in training
        answer_text: the state word(s) that should be predicted
        answer_context: surrounding text to locate the answer position
        entity_name: the queried entity
        update_entity: the entity that receives the update
        update_state_text: the new state text in the update sentence
        source_state_text: the original state text in the source
        packet_type: "UPDATE" or "RETAIN"
    """
    full_text = packet["full_text"]
    
    # Tokenize the full text
    enc = tokenizer(full_text, add_special_tokens=False, return_attention_mask=False)
    token_ids = enc["input_ids"]
    if len(token_ids) > SEQ_LENGTH - 2:  # leave room for special tokens
        token_ids = token_ids[:SEQ_LENGTH - 2]
    
    groups = build_word_groups(token_ids, tokenizer, special_ids)
    n_total_groups = max(groups) + 1 if any(g >= 0 for g in groups) else 0
    
    # Find answer span word groups (in the use sentence)
    answer_token_ids = tokenizer(packet["answer_text"], add_special_tokens=False)["input_ids"]
    answer_groups = find_span_word_groups(groups, token_ids, answer_token_ids)
    all_answer_occ = find_all_span_word_groups(groups, token_ids, answer_token_ids)
    
    # Find critical evidence word groups:
    # 1. Entity name in update sentence
    # 2. Update state in update sentence  
    # 3. Entity name in use sentence (usually part of the answer context)
    critical_evidence_groups = set()
    
    # Update entity name
    ue_tids = tokenizer(packet["update_entity"], add_special_tokens=False)["input_ids"]
    ue_groups = find_span_word_groups(groups, token_ids, ue_tids)
    critical_evidence_groups.update(ue_groups)
    
    # Update state text
    us_tids = tokenizer(packet["update_state_text"], add_special_tokens=False)["input_ids"]
    us_groups = find_span_word_groups(groups, token_ids, us_tids)
    critical_evidence_groups.update(us_groups)
    
    # Queried entity name (find last occurrence, likely in use sentence)
    qe_tids = tokenizer(packet["entity_name"], add_special_tokens=False)["input_ids"]
    qe_all = find_all_span_word_groups(groups, token_ids, qe_tids)
    if qe_all:
        critical_evidence_groups.update(qe_all[-1])  # last occurrence
    
    # Source state text (matters for RETAIN)
    if "source_state_text" in packet and packet["source_state_text"]:
        ss_tids = tokenizer(packet["source_state_text"], add_special_tokens=False)["input_ids"]
        ss_groups = find_span_word_groups(groups, token_ids, ss_tids)
        critical_evidence_groups.update(ss_groups)
    
    # Remove answer groups from critical evidence (they should be masked)
    critical_evidence_groups -= set(answer_groups)
    critical_evidence_list = sorted(critical_evidence_groups)
    
    # Simulate
    sim = simulate_wwm_supervision(
        n_total_groups, answer_groups, critical_evidence_list,
        all_answer_occ, n_sims
    )
    ana = analytic_estimates(len(answer_groups), len(critical_evidence_list))
    
    return {
        "packet_type": packet.get("packet_type", "unknown"),
        "pair_id": packet.get("pair_id", "unknown"),
        "n_tokens": len(token_ids),
        "n_word_groups": n_total_groups,
        "answer_text": packet["answer_text"],
        "answer_groups": answer_groups,
        "n_answer_occurrences": len(all_answer_occ),
        "critical_evidence_groups": critical_evidence_list,
        "simulation": sim,
        "analytic": ana,
    }


def make_hand_crafted_examples() -> List[Dict[str, Any]]:
    """Build hand-crafted paired contrastive examples for initial analysis."""
    examples = []
    
    # Pair 1: Simple state change
    source = "Alice carried a red umbrella and Bob wore a blue hat."
    
    # UPDATE: Alice gets new state, ask about Alice
    examples.append({
        "pair_id": "hand_001",
        "packet_type": "UPDATE",
        "source_sentence": source,
        "update_sentence": "Later that day, Alice started carrying a green umbrella instead.",
        "use_sentence": "When they met for dinner, Alice was holding a green umbrella.",
        "full_text": f"{source} Later that day, Alice started carrying a green umbrella instead. When they met for dinner, Alice was holding a green umbrella.",
        "answer_text": "green",
        "entity_name": "Alice",
        "update_entity": "Alice",
        "update_state_text": "green umbrella",
        "source_state_text": "red umbrella",
    })
    
    # RETAIN: Bob gets new state, ask about Alice (answer unchanged)
    examples.append({
        "pair_id": "hand_001",
        "packet_type": "RETAIN",
        "source_sentence": source,
        "update_sentence": "Later that day, Bob started wearing a yellow hat instead.",
        "use_sentence": "When they met for dinner, Alice was holding a red umbrella.",
        "full_text": f"{source} Later that day, Bob started wearing a yellow hat instead. When they met for dinner, Alice was holding a red umbrella.",
        "answer_text": "red",
        "entity_name": "Alice",
        "update_entity": "Bob",
        "update_state_text": "yellow hat",
        "source_state_text": "red umbrella",
    })
    
    # Pair 2: Longer, more realistic
    source2 = "The old library was managed by Professor Chen, who kept a collection of leather-bound journals, while Dr. Reyes maintained a set of handwritten field notes in her office."
    
    examples.append({
        "pair_id": "hand_002",
        "packet_type": "UPDATE",
        "source_sentence": source2,
        "update_sentence": "After the renovation, Professor Chen replaced the leather-bound journals with digital tablets for cataloging.",
        "use_sentence": "A visiting scholar asked about the materials in Professor Chen's workspace and found digital tablets on the shelves.",
        "full_text": f"{source2} After the renovation, Professor Chen replaced the leather-bound journals with digital tablets for cataloging. A visiting scholar asked about the materials in Professor Chen's workspace and found digital tablets on the shelves.",
        "answer_text": "digital tablets",
        "entity_name": "Professor Chen",
        "update_entity": "Professor Chen",
        "update_state_text": "digital tablets",
        "source_state_text": "leather-bound journals",
    })
    
    examples.append({
        "pair_id": "hand_002",
        "packet_type": "RETAIN",
        "source_sentence": source2,
        "update_sentence": "After the renovation, Dr. Reyes replaced the handwritten field notes with a laptop for recording observations.",
        "use_sentence": "A visiting scholar asked about the materials in Professor Chen's workspace and found leather-bound journals on the shelves.",
        "full_text": f"{source2} After the renovation, Dr. Reyes replaced the handwritten field notes with a laptop for recording observations. A visiting scholar asked about the materials in Professor Chen's workspace and found leather-bound journals on the shelves.",
        "answer_text": "leather-bound journals",
        "entity_name": "Professor Chen",
        "update_entity": "Dr. Reyes",
        "update_state_text": "laptop",
        "source_state_text": "leather-bound journals",
    })
    
    # Pair 3: Short, minimal
    source3 = "Tom had a silver watch and Sarah owned a gold ring."
    
    examples.append({
        "pair_id": "hand_003",
        "packet_type": "UPDATE",
        "source_sentence": source3,
        "update_sentence": "Tom traded his silver watch for a bronze compass.",
        "use_sentence": "When asked about his accessory, Tom showed everyone his bronze compass.",
        "full_text": f"{source3} Tom traded his silver watch for a bronze compass. When asked about his accessory, Tom showed everyone his bronze compass.",
        "answer_text": "bronze compass",
        "entity_name": "Tom",
        "update_entity": "Tom",
        "update_state_text": "bronze compass",
        "source_state_text": "silver watch",
    })
    
    examples.append({
        "pair_id": "hand_003",
        "packet_type": "RETAIN",
        "source_sentence": source3,
        "update_sentence": "Sarah exchanged her gold ring for a jade bracelet.",
        "use_sentence": "When asked about his accessory, Tom showed everyone his silver watch.",
        "full_text": f"{source3} Sarah exchanged her gold ring for a jade bracelet. When asked about his accessory, Tom showed everyone his silver watch.",
        "answer_text": "silver watch",
        "entity_name": "Tom",
        "update_entity": "Sarah",
        "update_state_text": "jade bracelet",
        "source_state_text": "silver watch",
    })
    
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-path", type=str, default=None,
                    help="Path to compliant16k tokenizer")
    ap.add_argument("--packet-jsonl", type=str, default=None,
                    help="JSONL with paired packets (or use --hand-crafted)")
    ap.add_argument("--hand-crafted", action="store_true",
                    help="Use built-in hand-crafted examples")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n-simulations", type=int, default=N_SIM_DEFAULT)
    args = ap.parse_args()
    
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    if args.tokenizer_path:
        tok_path = args.tokenizer_path
    else:
        # Try standard locations (tokenizer is part of model endpoint)
        candidates = [
            "models/frontier",
            "experiments/archive/compact_experience/data/tokenizer/hf_tokenizer_40k",
        ]
        tok_path = None
        for c in candidates:
            if pathlib.Path(c).exists():
                tok_path = c
                break
    
    if tok_path is None:
        print("ERROR: no tokenizer found", file=sys.stderr)
        sys.exit(1)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    special_ids = set(tokenizer.all_special_ids)
    print(f"Tokenizer loaded from {tok_path}, vocab_size={len(tokenizer)}", flush=True)
    
    # Load packets
    if args.hand_crafted:
        packets = make_hand_crafted_examples()
    elif args.packet_jsonl:
        packets = []
        with open(args.packet_jsonl) as f:
            for line in f:
                if line.strip():
                    packets.append(json.loads(line))
    else:
        print("ERROR: specify --packet-jsonl or --hand-crafted", file=sys.stderr)
        sys.exit(1)
    
    print(f"Analyzing {len(packets)} packets with {args.n_simulations} simulations each", flush=True)
    
    # Analyze each packet
    results = []
    for pkt in packets:
        r = analyze_packet(pkt, tokenizer, special_ids, args.n_simulations)
        results.append(r)
        print(f"  {r['pair_id']} {r['packet_type']:8s}  "
              f"ans_groups={r['n_answer_occurrences']}×{len(r['answer_groups'])}  "
              f"ev_groups={len(r['critical_evidence_groups'])}  "
              f"P(useful)={r['simulation']['useful_supervision']:.4f}  "
              f"P(useful_no_shortcut)={r['simulation']['useful_no_shortcut']:.4f}",
              flush=True)
    
    # Aggregate by type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["packet_type"]].append(r)
    
    # Aggregate by pair
    by_pair = defaultdict(list)
    for r in results:
        by_pair[r["pair_id"]].append(r)
    
    summary = {
        "status": "MLM_MASK_ANALYSIS",
        "n_packets": len(results),
        "n_pairs": len(by_pair),
        "by_type": {},
        "by_pair": {},
        "per_packet": results,
    }
    
    for ptype, rlist in by_type.items():
        vals = [r["simulation"] for r in rlist]
        summary["by_type"][ptype] = {
            "n": len(rlist),
            "mean_answer_masked": sum(v["answer_masked"] for v in vals) / len(vals),
            "mean_evidence_visible": sum(v["evidence_visible"] for v in vals) / len(vals),
            "mean_useful_supervision": sum(v["useful_supervision"] for v in vals) / len(vals),
            "mean_useful_no_shortcut": sum(v["useful_no_shortcut"] for v in vals) / len(vals),
            "mean_n_answer_groups": sum(v["n_answer_groups"] for v in vals) / len(vals),
            "mean_n_evidence_groups": sum(v["n_evidence_groups"] for v in vals) / len(vals),
        }
    
    for pair_id, rlist in by_pair.items():
        pair_summary = {}
        for r in rlist:
            pair_summary[r["packet_type"]] = {
                "useful_supervision": r["simulation"]["useful_supervision"],
                "useful_no_shortcut": r["simulation"]["useful_no_shortcut"],
                "n_answer_groups": len(r["answer_groups"]),
                "n_answer_occurrences": r["n_answer_occurrences"],
            }
        # Joint: both UPDATE and RETAIN have useful supervision in the same epoch
        if "UPDATE" in pair_summary and "RETAIN" in pair_summary:
            pu = pair_summary["UPDATE"]["useful_supervision"]
            pr = pair_summary["RETAIN"]["useful_supervision"]
            # P(both get useful supervision in one epoch) ≈ pu × pr
            # P(at least one in one epoch) = 1 - (1-pu)(1-pr)
            # Expected useful-both events per 10 epochs ≈ 10 × pu × pr
            pair_summary["joint_useful_per_epoch"] = pu * pr
            pair_summary["expected_useful_both_per_10epochs"] = 10.0 * pu * pr
        summary["by_pair"][pair_id] = pair_summary
    
    # Write results
    with open(out_dir / "mlm_mask_analysis.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Write markdown report
    lines = ["# research MLM mask analysis for contrastive entity-binding packets\n"]
    lines.append("## Purpose\n")
    lines.append("Quantify how often 15% whole-word masking (WWM) actually presents binding-sensitive\n")
    lines.append("supervision: the answer is masked while the relational evidence (which entity was\n")
    lines.append("updated, what the new state is) remains visible.\n\n")
    
    lines.append("## Per-type aggregates\n\n")
    lines.append("| type | n | P(ans masked) | P(ev visible) | P(useful) | P(useful no shortcut) | mean ans groups | mean ev groups |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for ptype, stats in summary["by_type"].items():
        lines.append(f"| {ptype} | {stats['n']} | {stats['mean_answer_masked']:.4f} "
                      f"| {stats['mean_evidence_visible']:.4f} "
                      f"| {stats['mean_useful_supervision']:.4f} "
                      f"| {stats['mean_useful_no_shortcut']:.4f} "
                      f"| {stats['mean_n_answer_groups']:.1f} "
                      f"| {stats['mean_n_evidence_groups']:.1f} |\n")
    
    lines.append("\n## Per-pair joint supervision\n\n")
    lines.append("| pair_id | UPDATE P(useful) | RETAIN P(useful) | P(both) | Expected both/10ep |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for pair_id, ps in summary["by_pair"].items():
        if "UPDATE" in ps and "RETAIN" in ps:
            lines.append(f"| {pair_id} "
                          f"| {ps['UPDATE']['useful_supervision']:.4f} "
                          f"| {ps['RETAIN']['useful_supervision']:.4f} "
                          f"| {ps.get('joint_useful_per_epoch', 0):.4f} "
                          f"| {ps.get('expected_useful_both_per_10epochs', 0):.2f} |\n")
    
    lines.append("\n## Per-packet detail\n\n")
    for r in results:
        lines.append(f"### {r['pair_id']} — {r['packet_type']}\n")
        lines.append(f"- Answer: `{r['answer_text']}` ({r['n_answer_occurrences']} occurrences, "
                      f"{len(r['answer_groups'])} word groups at primary location)\n")
        lines.append(f"- Critical evidence: {len(r['critical_evidence_groups'])} word groups\n")
        lines.append(f"- Total word groups: {r['n_word_groups']}\n")
        lines.append(f"- **P(answer masked)**: {r['simulation']['answer_masked']:.4f} "
                      f"(analytic: {r['analytic']['analytic_answer_masked']:.4f})\n")
        lines.append(f"- **P(evidence visible)**: {r['simulation']['evidence_visible']:.4f} "
                      f"(analytic: {r['analytic']['analytic_evidence_visible']:.4f})\n")
        lines.append(f"- **P(useful supervision)**: {r['simulation']['useful_supervision']:.4f} "
                      f"(analytic: {r['analytic']['analytic_useful_supervision']:.4f})\n")
        lines.append(f"- **P(shortcut|useful)**: {r['simulation']['shortcut_given_useful']:.4f}\n")
        lines.append(f"- **P(useful, no shortcut)**: {r['simulation']['useful_no_shortcut']:.4f}\n\n")
    
    lines.append("## Interpretation\n\n")
    lines.append("With standard 15% WWM, the probability of useful relational supervision per packet\n")
    lines.append("per epoch is the product P(answer masked) × P(evidence visible). For a typical\n")
    lines.append("1-word-group answer and ~5-10 evidence groups:\n")
    lines.append("- P(answer masked) ≈ 0.15\n")
    lines.append("- P(evidence visible) ≈ 0.85^n_evidence ≈ 0.44-0.72\n")
    lines.append("- P(useful) ≈ 0.07-0.11\n\n")
    lines.append("Over 10 training epochs, each packet provides ~0.7-1.1 useful supervision events.\n")
    lines.append("For paired contrasts, P(both UPDATE and RETAIN useful in same epoch) ≈ 0.005-0.012,\n")
    lines.append("meaning the learner rarely sees both sides of the same pair in the same pass.\n")
    lines.append("This quantifies why volume matters: the model must accumulate binding signal\n")
    lines.append("across different pairs with different entities, not from repeated same-pair exposure.\n\n")
    lines.append("If useful_no_shortcut is much lower than useful_supervision, the packet design has\n")
    lines.append("a leakage problem: the answer text appears at another visible location.\n")
    
    with open(out_dir / "mlm_mask_analysis.md", "w") as f:
        f.writelines(lines)
    
    print(json.dumps({
        "status": "MLM_MASK_ANALYSIS",
        "n_packets": len(results),
        "n_pairs": len(by_pair),
        "out_json": str(out_dir / "mlm_mask_analysis.json"),
        "out_md": str(out_dir / "mlm_mask_analysis.md"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
