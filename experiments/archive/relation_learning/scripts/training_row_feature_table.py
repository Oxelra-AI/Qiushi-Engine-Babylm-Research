#!/usr/bin/env python3
"""research: Tabulate cheap features that predict the answer in binding training rows.

Tabulate over the training rows P(answer = source state | f) for
each cheap feature f: operations present and their count, queried entity first-mentioned,
queried entity named in any operation, last operation targeting the queried entity, answer
containing operation-mentioned items, frame. Identity match must be the only feature that
containing operation-mentioned items, frame. Identity match must be the only feature that
moves that probability off its marginal.
This directly tests WHY the binding objective absorbed cheap credit:
if any feature besides identity predicts the answer, it will capture credit first.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import re
import pathlib
import statistics
from collections import Counter, defaultdict

ROOT = _public_path('.')

def content_words(text: str) -> set[str]:
    """Extract lowercase content words (skip stopwords)."""
    stops = {"the","a","an","is","was","were","are","be","been","being","have","has","had",
             "do","does","did","will","would","shall","should","can","could","may","might",
             "must","in","on","at","to","for","of","with","by","from","up","about","into",
             "over","after","and","but","or","nor","not","so","yet","both","either","neither",
             "that","this","these","those","it","its","he","she","they","them","his","her",
             "their","what","which","who","whom","whose","how","when","where","why"}
    return {w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text) if w.lower() not in stops and len(w) > 1}

def main():
    # Load research frame-varied rows (the actual training data used)
    fv_train = _public_path('experiments/archive/relation_learning/data/frame_varied_recombination_rows/recombination_train_frame_seen.jsonl')
    if not fv_train.exists():
        # Fallback to research original rows
        fv_train = _public_path('experiments/archive/relation_learning/data/recombination_rows/recombination_train.jsonl')
    
    rows = []
    with fv_train.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    
    print(f"Loaded {len(rows)} training rows from {fv_train.name}")
    
    # Classify each row
    features = []
    for row in rows:
        row_id = row.get("row_id", "")
        pair_id = row.get("pair_id", "")
        ptype = row.get("packet_type", "")
        context = row.get("context_text", "")
        answer = row.get("answer_text", "")
        query_entity = row.get("query_entity", "")
        
        # Determine if this is A (unchanged) or B (updated)
        is_unchanged = "unchanged" in row_id.lower() or "UNCHANGED" in ptype
        is_updated = "updated" in row_id.lower() or "UPDATED" in ptype
        
        # answer_is_source_state: True for unchanged (A), False for updated (B)
        answer_is_source = is_unchanged
        
        # Extract entities
        entity_a = row.get("entity_a", "")
        entity_b = row.get("entity_b", "")
        
        # Cheap features
        # 1. Is query entity = entity_a (typically first-mentioned)?
        query_is_entity_a = query_entity.lower().strip() == entity_a.lower().strip() if entity_a else None
        
        # 2. Entity mention order in context
        pos_a = context.lower().find(entity_a.lower()) if entity_a else -1
        pos_b = context.lower().find(entity_b.lower()) if entity_b else -1
        query_first_mentioned = None
        if pos_a >= 0 and pos_b >= 0:
            if query_entity.lower().strip() == entity_a.lower().strip():
                query_first_mentioned = pos_a < pos_b
            else:
                query_first_mentioned = pos_b < pos_a
        
        # 3. Is query entity named in any operation?
        # Operations typically contain "Move/Put/Remove from Box X"
        # Look for the query entity in the middle part of the context
        op_text = ""
        lines = context.split(".")
        for ln in lines:
            ln_lower = ln.lower()
            if any(kw in ln_lower for kw in ["move", "put", "remove", "took", "placed"]):
                op_text += ln + ". "
        query_in_operation = query_entity.lower() in op_text.lower() if op_text else None
        
        # 4. Answer overlaps with operation-mentioned content
        op_words = content_words(op_text)
        answer_words = content_words(answer)
        answer_op_overlap = len(answer_words & op_words) if op_words else 0
        answer_has_op_content = answer_op_overlap > 0
        
        # 5. Frame (if available)
        frame = row.get("frame", "")
        if not frame:
            # Try to infer from row_id
            parts = row_id.split("::")
            frame = parts[-1] if len(parts) > 1 else "original"
        
        # 6. Answer length (words)
        answer_word_count = len(answer.split())
        
        features.append({
            "pair_id": pair_id,
            "row_id": row_id,
            "is_A_unchanged": is_unchanged,
            "answer_is_source": answer_is_source,
            "query_is_entity_a": query_is_entity_a,
            "query_first_mentioned": query_first_mentioned,
            "query_in_operation": query_in_operation,
            "answer_has_op_content": answer_has_op_content,
            "answer_op_overlap": answer_op_overlap,
            "frame": frame,
            "answer_word_count": answer_word_count,
        })
    
    # Compute marginal and conditional P(answer = source state)
    marginal = statistics.mean(f["answer_is_source"] for f in features)
    
    print(f"\nMarginal P(answer = source state) = {marginal:.4f}")
    print(f"  (should be 0.5 for balanced data, or 0.667 for original 2:1 data)")
    print()
    
    # Feature analysis
    feat_names = [
        ("query_is_entity_a", "Query is entity_a (typically first-mentioned)"),
        ("query_first_mentioned", "Query entity mentioned first in context"),
        ("query_in_operation", "Query entity named in operation text"),
        ("answer_has_op_content", "Answer contains operation-mentioned words"),
    ]
    
    print("=" * 80)
    print(f"{'Feature':<50} {'P(src|f=T)':<12} {'P(src|f=F)':<12} {'Δ':<10} {'n_T':<8} {'n_F':<8}")
    print("=" * 80)
    
    for fname, fdesc in feat_names:
        vals_true = [f["answer_is_source"] for f in features if f[fname] is True]
        vals_false = [f["answer_is_source"] for f in features if f[fname] is False]
        if vals_true and vals_false:
            p_true = statistics.mean(vals_true)
            p_false = statistics.mean(vals_false)
            delta = p_true - p_false
            print(f"{fdesc:<50} {p_true:<12.4f} {p_false:<12.4f} {delta:<10.4f} {len(vals_true):<8} {len(vals_false):<8}")
        else:
            print(f"{fdesc:<50} INSUFFICIENT DATA")
    
    # Frame analysis
    print()
    print("Frame analysis:")
    frame_counts = defaultdict(lambda: {"n": 0, "src": 0})
    for f in features:
        fr = f["frame"]
        frame_counts[fr]["n"] += 1
        frame_counts[fr]["src"] += int(f["answer_is_source"])
    
    for fr, cnt in sorted(frame_counts.items()):
        p = cnt["src"] / cnt["n"] if cnt["n"] > 0 else 0
        print(f"  {fr:<40} P(src) = {p:.4f} (n={cnt['n']})")
    
    # Answer word count analysis
    print()
    src_lens = [f["answer_word_count"] for f in features if f["answer_is_source"]]
    new_lens = [f["answer_word_count"] for f in features if not f["answer_is_source"]]
    if src_lens and new_lens:
        print(f"Answer word count: source={statistics.mean(src_lens):.2f}±{statistics.stdev(src_lens):.2f}, "
              f"new={statistics.mean(new_lens):.2f}±{statistics.stdev(new_lens):.2f}")
    
    # Operation-content overlap analysis
    print()
    src_overlaps = [f["answer_op_overlap"] for f in features if f["answer_is_source"]]
    new_overlaps = [f["answer_op_overlap"] for f in features if not f["answer_is_source"]]
    if src_overlaps and new_overlaps:
        print(f"Answer-operation overlap: source={statistics.mean(src_overlaps):.4f}±{statistics.stdev(src_overlaps):.4f}, "
              f"new={statistics.mean(new_overlaps):.4f}±{statistics.stdev(new_overlaps):.4f}")
    
    # Save output
    out_dir = _public_path('experiments/archive/relation_learning/data/training_row_feature_table')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "status": "TRAINING_ROW_FEATURE_TABLE",
        "source": str(fv_train),
        "n_rows": len(features),
        "marginal_p_source": round(marginal, 4),
        "features": {},
    }
    
    for fname, fdesc in feat_names:
        vals_true = [f["answer_is_source"] for f in features if f[fname] is True]
        vals_false = [f["answer_is_source"] for f in features if f[fname] is False]
        if vals_true and vals_false:
            p_true = statistics.mean(vals_true)
            p_false = statistics.mean(vals_false)
            summary["features"][fname] = {
                "description": fdesc,
                "p_source_given_true": round(p_true, 4),
                "p_source_given_false": round(p_false, 4),
                "delta": round(p_true - p_false, 4),
                "n_true": len(vals_true),
                "n_false": len(vals_false),
            }
    
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {out_dir / 'summary.json'}")

if __name__ == "__main__":
    main()
