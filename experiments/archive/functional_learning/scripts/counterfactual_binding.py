#!/usr/bin/env python3
"""research: Counterfactual binding diagnostic.

Scientific purpose
------------------
The baseline probe showed coherent86 scores 7/8 pairs correct. But this could be
driven by surface copying rather than entity-specific binding. This diagnostic
measures the counterfactual: for each pair, when only the update recipient changes
(and EVERYTHING else stays the same), does the model's prediction at the answer
position change accordingly?

Specifically, for each pair we construct four conditions:
  1. UPDATE-as-written: update target → ask target  (answer = new state)
  2. RETAIN-as-written: update distractor → ask target  (answer = source state)
  3. NEUTRAL: no update → ask target  (answer = source state)
  4. SWAPPED: use the UPDATE update but ask about the DISTRACTOR  (answer varies)

True binding means:
  - P(new_state | UPDATE) > P(source_state | UPDATE)  AND
  - P(source_state | RETAIN) > P(new_state | RETAIN)  AND
  - The MARGIN changes between UPDATE and RETAIN conditions

Copy-from-update would give:
  - High P(new_state) in both UPDATE and RETAIN (because the new state is visible)
  
Copy-from-source would give:
  - High P(source_state) in both conditions

Usage
-----
  python counterfactual_binding.py \\
    --model-path <path> \\
    --packet-jsonl <paired_packets.jsonl> \\
    --out-dir <output_directory> \\
    [--gpu 0] [--private-bottleneck 128]
"""
import argparse, json, pathlib, sys, math
from typing import Dict, List, Any
from collections import defaultdict


def score_cloze(model, tokenizer, full_text, answer_text, foil_text, device):
    """Score answer vs foil at the last occurrence of answer_text in full_text."""
    import torch
    
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True,
                    max_length=256, truncation=True)
    input_ids = enc["input_ids"].squeeze(0).tolist()
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    foil_ids = tokenizer(foil_text, add_special_tokens=False)["input_ids"]
    
    # Find LAST occurrence of answer
    answer_start = None
    for start in range(len(input_ids) - len(answer_ids), -1, -1):
        if input_ids[start:start+len(answer_ids)] == answer_ids:
            answer_start = start
            break
    
    if answer_start is None:
        return {"status": "not_found"}
    
    # Mask answer tokens
    mask_id = tokenizer.mask_token_id
    masked = list(input_ids)
    for i in range(answer_start, answer_start + len(answer_ids)):
        masked[i] = mask_id
    
    masked_tensor = torch.tensor([masked], device=device)
    attn_mask = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=masked_tensor, attention_mask=attn_mask).logits
    
    # Score answer and foil
    answer_lp = 0.0
    for i, pos in enumerate(range(answer_start, answer_start + len(answer_ids))):
        lp = torch.log_softmax(logits[0, pos], dim=-1)
        answer_lp += float(lp[answer_ids[i]])
    
    foil_lp = 0.0
    n_compare = min(len(foil_ids), len(answer_ids))
    for i in range(n_compare):
        pos = answer_start + i
        lp = torch.log_softmax(logits[0, pos], dim=-1)
        foil_lp += float(lp[foil_ids[i]])
    if len(foil_ids) != len(answer_ids):
        foil_lp -= abs(len(foil_ids) - len(answer_ids)) * 5.0
    
    # Top prediction
    top_tok = int(logits[0, answer_start].argmax())
    top_str = tokenizer.convert_ids_to_tokens(top_tok)
    
    return {
        "status": "scored",
        "answer_lp": answer_lp,
        "foil_lp": foil_lp,
        "margin": answer_lp - foil_lp,
        "top_prediction": top_str,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--packet-jsonl", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--private-bottleneck", type=int, default=128)
    args = ap.parse_args()
    
    import torch
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    
    # Load model (same logic as binding_probe)
    from transformers import AutoTokenizer, DebertaV2Config
    from safetensors.torch import load_file
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    
    try:
        sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
        from context_credit_trainer import FrozenSlowPrivateDebertaV2ForMaskedLM
        cfg = DebertaV2Config.from_pretrained(str(model_path), local_files_only=True)
        cfg.private_adapter_bottleneck = args.private_bottleneck
        cfg.private_adapter_scale = args.private_scale
        cfg.private_adapter_enabled = True
        model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
        sd = load_file(str(model_path / "model.safetensors"), device="cpu")
        model.load_state_dict(sd, strict=False)
        for m in model.modules():
            if hasattr(m, "private_adapter_enabled"):
                m.private_adapter_enabled = True
        model_type = "private_adapter"
    except:
        from transformers import DebertaV2ForMaskedLM
        model = DebertaV2ForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
        model_type = "standard"
    
    model = model.to(device).eval()
    print(f"Model: {model_type}, device: {device}", flush=True)
    
    # Load paired packets
    packets = []
    with open(args.packet_jsonl) as f:
        for line in f:
            if line.strip():
                packets.append(json.loads(line))
    
    # Group by pair
    by_pair = defaultdict(dict)
    for p in packets:
        by_pair[p["pair_id"]][p["packet_type"]] = p
    
    results = []
    for pair_id, pair in sorted(by_pair.items()):
        if "UPDATE" not in pair or "RETAIN" not in pair:
            continue
        
        u = pair["UPDATE"]
        r = pair["RETAIN"]
        
        # Extract components
        source = u["source_sentence"]
        target_entity = u["entity_name"]
        new_state = u["answer_text"]      # answer for UPDATE
        source_state = r["answer_text"]   # answer for RETAIN (= source state)
        update_sent_target = u["update_sentence"]  # target entity gets updated
        update_sent_distractor = r["update_sentence"]  # distractor gets updated
        use_frame = u["use_sentence_frame"]
        
        # Condition 1: UPDATE-as-written
        update_use = use_frame.replace("{STATE}", new_state)
        update_text = f"{source} {update_sent_target} {update_use}"
        s1 = score_cloze(model, tokenizer, update_text, new_state, source_state, device)
        
        # Condition 2: RETAIN-as-written
        retain_use = use_frame.replace("{STATE}", source_state)
        retain_text = f"{source} {update_sent_distractor} {retain_use}"
        s2 = score_cloze(model, tokenizer, retain_text, source_state, new_state, device)
        
        # Condition 3: NEUTRAL (no update, ask about target → source state)
        neutral_text = f"{source} {retain_use}"
        s3 = score_cloze(model, tokenizer, neutral_text, source_state, new_state, device)
        
        # Condition 4: CROSS-ENTITY (update target, but fill use with source state)
        # This tests: does the model still predict new_state even when the text says source_state?
        # Actually, let's do: update target entity, but the use sentence contains source_state
        # The question: does the model assign higher prob to new_state or source_state?
        cross_text = f"{source} {update_sent_target} {retain_use}"
        s4 = score_cloze(model, tokenizer, cross_text, source_state, new_state, device)
        
        # Condition 5: For deeper test, score new_state in the RETAIN context
        # (source = new, foil = source in RETAIN context)
        s5 = score_cloze(model, tokenizer, retain_text, new_state, source_state, device)
        
        pr = {
            "pair_id": pair_id,
            "target_entity": target_entity,
            "new_state": new_state,
            "source_state": source_state,
            "conditions": {
                "UPDATE": {
                    "description": "target updated → ask target → answer=new_state",
                    "margin_new_vs_source": s1.get("margin", None),
                    "top": s1.get("top_prediction", "?"),
                },
                "RETAIN": {
                    "description": "distractor updated → ask target → answer=source_state",
                    "margin_source_vs_new": s2.get("margin", None),
                    "top": s2.get("top_prediction", "?"),
                },
                "NEUTRAL": {
                    "description": "no update → ask target → answer=source_state",
                    "margin_source_vs_new": s3.get("margin", None),
                    "top": s3.get("top_prediction", "?"),
                },
                "TARGET_UPDATED_ASK_SOURCE": {
                    "description": "target updated → but use sentence has source_state",
                    "margin_source_vs_new": s4.get("margin", None),
                    "top": s4.get("top_prediction", "?"),
                },
                "RETAIN_SCORE_NEW": {
                    "description": "distractor updated → score new_state (should be low)",
                    "margin_new_vs_source": s5.get("margin", None),
                    "top": s5.get("top_prediction", "?"),
                },
            },
        }
        
        # Compute binding measures
        update_margin = s1.get("margin", 0) or 0
        retain_margin = s2.get("margin", 0) or 0
        neutral_margin = s3.get("margin", 0) or 0
        
        # True binding: UPDATE margin > 0 AND RETAIN margin > 0
        pr["update_correct"] = update_margin > 0
        pr["retain_correct"] = retain_margin > 0
        pr["both_correct"] = update_margin > 0 and retain_margin > 0
        
        # Counterfactual test: UPDATE margin and RETAIN margin should have OPPOSITE signs
        # for new_state vs source_state
        # UPDATE: new > source (positive margin)
        # RETAIN_SCORE_NEW: new < source (negative margin for new_vs_source)
        retain_new_margin = s5.get("margin", 0) or 0
        pr["counterfactual_shift"] = update_margin - retain_new_margin
        # If this is large and positive, the model changes its prediction based on who was updated
        
        # Neutral-adjusted: how much does the update add beyond the neutral baseline?
        pr["update_over_neutral"] = update_margin - neutral_margin
        pr["retain_over_neutral"] = retain_margin - neutral_margin
        
        results.append(pr)
        
        print(f"\n{pair_id}: {target_entity}, new={new_state}, source={source_state}", flush=True)
        print(f"  UPDATE  margin(new>source): {update_margin:+.3f}  top={s1.get('top_prediction','?')}", flush=True)
        print(f"  RETAIN  margin(source>new): {retain_margin:+.3f}  top={s2.get('top_prediction','?')}", flush=True)
        print(f"  NEUTRAL margin(source>new): {neutral_margin:+.3f}  top={s3.get('top_prediction','?')}", flush=True)
        print(f"  CROSS   margin(source>new when target updated): {s4.get('margin',0):+.3f}", flush=True)
        print(f"  Counterfactual shift: {pr['counterfactual_shift']:+.3f}", flush=True)
    
    # Summary
    n = len(results)
    n_both = sum(1 for r in results if r["both_correct"])
    mean_shift = sum(r["counterfactual_shift"] for r in results) / max(1, n)
    
    summary = {
        "status": "COUNTERFACTUAL_BINDING",
        "model_path": str(args.model_path),
        "n_pairs": n,
        "n_both_correct": n_both,
        "mean_counterfactual_shift": mean_shift,
        "mean_update_margin": sum(r["conditions"]["UPDATE"]["margin_new_vs_source"] or 0 for r in results) / max(1, n),
        "mean_retain_margin": sum(r["conditions"]["RETAIN"]["margin_source_vs_new"] or 0 for r in results) / max(1, n),
        "mean_neutral_margin": sum(r["conditions"]["NEUTRAL"]["margin_source_vs_new"] or 0 for r in results) / max(1, n),
        "per_pair": results,
    }
    
    with open(out_dir / "counterfactual_binding.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Markdown
    lines = ["# research counterfactual entity-binding diagnostic\n\n"]
    lines.append(f"Model: `{args.model_path}` ({model_type})\n\n")
    lines.append("## Summary\n\n")
    lines.append(f"- Both correct: {n_both}/{n}\n")
    lines.append(f"- Mean counterfactual shift: {mean_shift:+.3f}\n")
    lines.append(f"- Mean UPDATE margin (new > source): {summary['mean_update_margin']:+.3f}\n")
    lines.append(f"- Mean RETAIN margin (source > new): {summary['mean_retain_margin']:+.3f}\n")
    lines.append(f"- Mean NEUTRAL margin (source > new): {summary['mean_neutral_margin']:+.3f}\n\n")
    
    lines.append("## Per-pair conditions\n\n")
    lines.append("| pair | entity | UPDATE | RETAIN | NEUTRAL | CROSS | CF shift | both |\n")
    lines.append("|---|---|---:|---:|---:|---:|---:|:---:|\n")
    for r in results:
        um = r["conditions"]["UPDATE"]["margin_new_vs_source"] or 0
        rm = r["conditions"]["RETAIN"]["margin_source_vs_new"] or 0
        nm = r["conditions"]["NEUTRAL"]["margin_source_vs_new"] or 0
        cm = r["conditions"]["TARGET_UPDATED_ASK_SOURCE"]["margin_source_vs_new"] or 0
        bc = "✓" if r["both_correct"] else "✗"
        lines.append(f"| {r['pair_id']} | {r['target_entity']} | {um:+.1f} | {rm:+.1f} "
                      f"| {nm:+.1f} | {cm:+.1f} | {r['counterfactual_shift']:+.1f} | {bc} |\n")
    
    lines.append("\n## Interpretation\n\n")
    lines.append("**Counterfactual shift** measures how much the model's preference for new_state\n")
    lines.append("changes between UPDATE context (where target entity was updated) and RETAIN context\n")
    lines.append("(where a distractor entity was updated). A large positive shift means the model\n")
    lines.append("tracks which entity was updated and adjusts its prediction accordingly.\n\n")
    lines.append("**NEUTRAL** shows the baseline preference without any update. If RETAIN ≈ NEUTRAL,\n")
    lines.append("the model may be ignoring the distractor update rather than actively binding.\n")
    lines.append("If UPDATE >> NEUTRAL, the model actively incorporates the update information.\n")
    
    with open(out_dir / "counterfactual_binding.md", "w") as f:
        f.writelines(lines)
    
    print(f"\n{json.dumps({'status': summary['status'], 'n_both': n_both, 'mean_shift': mean_shift}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
