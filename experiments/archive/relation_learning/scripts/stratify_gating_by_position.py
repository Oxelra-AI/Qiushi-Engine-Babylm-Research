#!/usr/bin/env python3
"""research: Stratify binding gating by entity position pattern using pair_margins.

For each held pair, classify whether the unchanged entity appears first or the
updated entity appears first in the shared context prefix. Then stratify the
joint correctness by this classification to determine whether position explains
the gating signal.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import pathlib
import sys

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/stratify_gating_by_position.py')
ROOT = _public_path('experiments/archive/relation_learning/data')


def find_first_position(text: str, entity: str) -> int:
    return text.lower().find(entity.lower())


def main():
    balanced_dir = _public_path('experiments/archive/relation_learning/data/binding_factorial_balanced_pairbatch')
    recom_held = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_heldout.jsonl')
    
    # Load heldout rows
    held_by_id = {}
    with open(recom_held) as f:
        for line in f:
            r = json.loads(line.strip())
            held_by_id[r["row_id"]] = r
    
    # Build pairs and classify position
    pairs = {}
    for rid, r in held_by_id.items():
        ph = r.get("pair_half", "")
        pid = r["pair_id"]
        if ph in ("A", "B"):
            pairs.setdefault(pid, {})[ph] = r
    
    pair_position = {}
    for pid, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        row_a = pair["A"]
        row_b = pair["B"]
        
        source = row_a.get("source_sentence", "")
        update = row_a.get("update_sentence", "")
        shared = source + " " + update
        
        unch_entity = row_a["query_entity"]
        upd_entity = row_b["query_entity"]
        
        pos_unch = find_first_position(shared, unch_entity)
        pos_upd = find_first_position(shared, upd_entity)
        
        if pos_unch < 0 or pos_upd < 0:
            pair_position[pid] = "ambiguous"
        elif pos_unch < pos_upd:
            pair_position[pid] = "unchanged_first"
        elif pos_upd < pos_unch:
            pair_position[pid] = "updated_first"
        else:
            pair_position[pid] = "ambiguous"
    
    pos_dist = {}
    for v in pair_position.values():
        pos_dist[v] = pos_dist.get(v, 0) + 1
    print(f"Position distribution: {pos_dist}", flush=True)
    
    # Stratify each arm's pair_margins
    arms = ["answer_clean", "uniform_wwm", "answer_corrupt_update_state"]
    epochs = [0, 5, 10, 15, 20]
    
    results = {}
    for arm in arms:
        arm_dir = balanced_dir / arm
        arm_results = {}
        
        for ep in epochs:
            eval_path = arm_dir / f"eval_epoch_{ep:03d}.json"
            if not eval_path.exists():
                continue
            
            data = json.loads(eval_path.read_text())
            pm = data.get("pair_margins", [])
            
            strata = {}
            for pos_class in ["unchanged_first", "updated_first", "ambiguous"]:
                strata[pos_class] = {"n": 0, "joint": 0, "a_correct": 0, "b_correct": 0, "both_wrong": 0}
            
            for p in pm:
                pid = p["pair_id"]
                pos = pair_position.get(pid, "ambiguous")
                st = strata[pos]
                st["n"] += 1
                a_ok = bool(p["a_correct"])
                b_ok = bool(p["b_correct"])
                if a_ok:
                    st["a_correct"] += 1
                if b_ok:
                    st["b_correct"] += 1
                if a_ok and b_ok:
                    st["joint"] += 1
                if not a_ok and not b_ok:
                    st["both_wrong"] += 1
            
            for pos, st in strata.items():
                if st["n"] > 0:
                    st["gated_frac"] = round(st["joint"] / st["n"], 4)
                    st["a_only"] = st["a_correct"] - st["joint"]
                    st["b_only"] = st["b_correct"] - st["joint"]
            
            arm_results[ep] = strata
        
        results[arm] = arm_results
    
    out_dir = _public_path('experiments/archive/relation_learning/data/position_stratified_gating')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "status": "POSITION_STRATIFIED_GATING",
        "position_distribution": pos_dist,
        "arms": results,
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Markdown: focus on epoch 20 and the critical question
    md_lines = [
        "# research: Position-stratified binding gating",
        "",
        f"Position distribution (n=200 held pairs): {pos_dist}",
        "",
        "If gating is driven by position shortcut, joint successes should concentrate",
        "in unchanged_first pairs (where position is informative) and vanish in",
        "updated_first pairs (where position would give the wrong assignment).",
        "",
        "## Epoch 20 results",
        "",
    ]
    
    for arm in arms:
        arm_data = results.get(arm, {}).get(20, {})
        md_lines.append(f"### {arm}")
        md_lines.append("")
        md_lines.append("| position | n | joint | gated% | a_correct | b_correct | both_wrong |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for pos in ["unchanged_first", "updated_first", "ambiguous"]:
            st = arm_data.get(pos, {})
            if st.get("n", 0) > 0:
                md_lines.append(
                    f"| {pos} | {st['n']} | {st['joint']} | "
                    f"{st.get('gated_frac', 0)*100:.1f}% | "
                    f"{st['a_correct']} | {st['b_correct']} | {st['both_wrong']} |"
                )
        md_lines.append("")
    
    # Trajectory for answer_clean
    md_lines += [
        "## answer_clean trajectory by position",
        "",
        "| epoch | unch_first joint/n | upd_first joint/n | ambiguous joint/n |",
        "|---:|---|---|---|",
    ]
    for ep in epochs:
        arm_data = results.get("answer_clean", {}).get(ep, {})
        uf = arm_data.get("unchanged_first", {})
        up = arm_data.get("updated_first", {})
        amb = arm_data.get("ambiguous", {})
        md_lines.append(
            f"| {ep} | {uf.get('joint',0)}/{uf.get('n',0)} | "
            f"{up.get('joint',0)}/{up.get('n',0)} | "
            f"{amb.get('joint',0)}/{amb.get('n',0)} |"
        )
    
    md_lines += [
        "",
        "## Interpretation",
        "",
        "If joint successes appear in BOTH position classes (unchanged_first AND",
        "updated_first), position alone cannot explain the gating signal.",
        "If joint successes are confined to unchanged_first, position is a viable shortcut.",
    ]
    
    with open(out_dir / "summary.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")
    
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
