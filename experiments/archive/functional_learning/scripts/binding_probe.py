#!/usr/bin/env python3
"""research: Entity-binding cloze probe for contrastive packets.

Scientific purpose
------------------
Score a masked language model on paired contrastive entity-binding packets to measure:
1. Baseline binding competence (before any training intervention)
2. Per-pair UPDATE margin: does the model prefer new state over source state?
3. Per-pair RETAIN margin: does the model prefer source state over new state?
4. Joint binding: do both margins move in the correct direction?

The probe masks only the answer span in the use sentence and computes:
  margin = log P(answer_tokens | context) - log P(foil_tokens | context)

A positive UPDATE margin means the model correctly assigns higher probability to
the new state after the target entity is updated. A positive RETAIN margin means
the model correctly assigns higher probability to the source state when a distractor
entity is updated.

Usage
-----
  python binding_probe.py \\
    --model-path <path_to_model> \\
    --packet-jsonl <paired_packets.jsonl> \\
    --out-dir <output_directory> \\
    [--gpu 0] [--batch-size 8]
"""
import argparse, json, pathlib, sys, math
from typing import Dict, List, Tuple, Any
from collections import defaultdict


def is_word_start(token_str: str) -> bool:
    return token_str.startswith("Ġ") or token_str.startswith("▁")


def build_word_groups(token_ids, tokenizer, special_ids):
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


def find_token_span(token_ids, target_ids, start_from=0):
    """Find first occurrence of target_ids in token_ids starting from start_from."""
    n = len(target_ids)
    for i in range(start_from, len(token_ids) - n + 1):
        if token_ids[i:i+n] == target_ids:
            return i, i+n
    return None, None


def score_packet(model, tokenizer, packet, device, special_ids):
    """Score one packet: mask answer in use sentence, compute answer vs foil logprobs."""
    import torch
    
    full_text = packet["full_text"]
    answer_text = packet["answer_text"]
    foil_text = packet["foil_text"]
    
    # Tokenize
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=True,
                    max_length=256, truncation=True)
    input_ids = enc["input_ids"].squeeze(0).tolist()
    
    # Tokenize answer and foil (without special tokens)
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    foil_ids = tokenizer(foil_text, add_special_tokens=False)["input_ids"]
    
    # Find answer location — we want the LAST occurrence (in the use sentence)
    # Search backwards
    answer_start, answer_end = None, None
    for start in range(len(input_ids) - len(answer_ids), -1, -1):
        if input_ids[start:start+len(answer_ids)] == answer_ids:
            answer_start = start
            answer_end = start + len(answer_ids)
            break
    
    if answer_start is None:
        return {"status": "answer_not_found", "pair_id": packet["pair_id"],
                "packet_type": packet["packet_type"]}
    
    # Create masked input: replace answer tokens with [MASK]
    masked_input = list(input_ids)
    mask_id = tokenizer.mask_token_id
    for i in range(answer_start, answer_end):
        masked_input[i] = mask_id
    
    # Forward pass
    masked_tensor = torch.tensor([masked_input], device=device)
    attn_mask = enc["attention_mask"].to(device)
    
    with torch.no_grad():
        logits = model(input_ids=masked_tensor, attention_mask=attn_mask).logits
    
    # Score answer tokens at masked positions
    answer_logprob = 0.0
    for i, pos in enumerate(range(answer_start, answer_end)):
        token_logits = logits[0, pos]
        log_probs = torch.log_softmax(token_logits, dim=-1)
        answer_logprob += float(log_probs[answer_ids[i]])
    
    # Score foil tokens at masked positions (may have different length)
    foil_logprob = 0.0
    if len(foil_ids) == len(answer_ids):
        for i, pos in enumerate(range(answer_start, answer_end)):
            token_logits = logits[0, pos]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            foil_logprob += float(log_probs[foil_ids[i]])
    else:
        # Different lengths: score each token against position 
        # Use the first min(len(answer), len(foil)) positions
        n_compare = min(len(foil_ids), len(answer_ids))
        foil_logprob = 0.0
        for i in range(n_compare):
            pos = answer_start + i
            token_logits = logits[0, pos]
            log_probs = torch.log_softmax(token_logits, dim=-1)
            foil_logprob += float(log_probs[foil_ids[i]])
        # Penalize length mismatch
        foil_logprob -= abs(len(foil_ids) - len(answer_ids)) * 5.0  # rough penalty
    
    margin = answer_logprob - foil_logprob
    
    # Also measure: what's the model's top prediction at the first masked position?
    first_masked_logits = logits[0, answer_start]
    top_token = int(first_masked_logits.argmax())
    top_token_str = tokenizer.convert_ids_to_tokens(top_token)
    
    return {
        "status": "scored",
        "pair_id": packet["pair_id"],
        "packet_type": packet["packet_type"],
        "answer_text": answer_text,
        "foil_text": foil_text,
        "answer_logprob": answer_logprob,
        "foil_logprob": foil_logprob,
        "margin": margin,
        "answer_len_tokens": len(answer_ids),
        "foil_len_tokens": len(foil_ids),
        "top_prediction_first_pos": top_token_str,
        "correct_direction": margin > 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--packet-jsonl", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=-1, help="-1 for CPU")
    ap.add_argument("--private-scale", type=float, default=0.75)
    ap.add_argument("--private-bottleneck", type=int, default=64)
    args = ap.parse_args()
    
    import torch
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    # Load model
    from transformers import AutoTokenizer, DebertaV2Config
    from safetensors.torch import load_file
    
    model_path = pathlib.Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    
    # Try loading as private-adapter model first, fall back to standard
    try:
        sys.path.insert(0, str(pathlib.Path("experiments/archive/functional_learning/scripts")))
        # Try to import the model class
        from context_credit_trainer import FrozenSlowPrivateDebertaV2ForMaskedLM
        cfg = DebertaV2Config.from_pretrained(str(model_path), local_files_only=True)
        cfg.private_adapter_bottleneck = args.private_bottleneck
        cfg.private_adapter_scale = args.private_scale
        cfg.private_adapter_enabled = True
        model = FrozenSlowPrivateDebertaV2ForMaskedLM(cfg)
        sd = load_file(str(model_path / "model.safetensors"), device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tied_ok = {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}
        real_missing = set(missing) - tied_ok
        if real_missing:
            print(f"WARNING: missing keys: {real_missing}", flush=True)
        # Enable private path
        for m in model.modules():
            if hasattr(m, "private_adapter_enabled"):
                m.private_adapter_enabled = True
        model_type = "private_adapter"
        print(f"Loaded private-adapter model, scale={args.private_scale}", flush=True)
    except Exception as e:
        print(f"Private adapter load failed ({e}), trying standard DeBERTa", flush=True)
        from transformers import DebertaV2ForMaskedLM
        model = DebertaV2ForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
        model_type = "standard"
    
    model = model.to(device)
    model.eval()
    
    # Load packets
    packets = []
    with open(args.packet_jsonl) as f:
        for line in f:
            if line.strip():
                packets.append(json.loads(line))
    
    print(f"Scoring {len(packets)} packets on {model_type} model", flush=True)
    
    # Score each packet
    results = []
    for pkt in packets:
        r = score_packet(model, tokenizer, pkt, device, special_ids)
        results.append(r)
        if r["status"] == "scored":
            direction = "✓" if r["correct_direction"] else "✗"
            print(f"  {r['pair_id']} {r['packet_type']:8s} margin={r['margin']:+.3f} "
                  f"top={r['top_prediction_first_pos']:12s} {direction}", flush=True)
        else:
            print(f"  {r['pair_id']} {r['packet_type']:8s} {r['status']}", flush=True)
    
    # Aggregate by pair
    by_pair = defaultdict(dict)
    for r in results:
        if r["status"] == "scored":
            by_pair[r["pair_id"]][r["packet_type"]] = r
    
    pair_summaries = []
    for pid, pair in sorted(by_pair.items()):
        ps = {"pair_id": pid}
        if "UPDATE" in pair:
            ps["update_margin"] = pair["UPDATE"]["margin"]
            ps["update_correct"] = pair["UPDATE"]["correct_direction"]
        if "RETAIN" in pair:
            ps["retain_margin"] = pair["RETAIN"]["margin"]
            ps["retain_correct"] = pair["RETAIN"]["correct_direction"]
        if "UPDATE" in pair and "RETAIN" in pair:
            ps["both_correct"] = pair["UPDATE"]["correct_direction"] and pair["RETAIN"]["correct_direction"]
            ps["binding_score"] = min(pair["UPDATE"]["margin"], pair["RETAIN"]["margin"])
        pair_summaries.append(ps)
    
    n_both = sum(1 for ps in pair_summaries if ps.get("both_correct", False))
    n_update_correct = sum(1 for ps in pair_summaries if ps.get("update_correct", False))
    n_retain_correct = sum(1 for ps in pair_summaries if ps.get("retain_correct", False))
    n_pairs = len(pair_summaries)
    
    summary = {
        "status": "BINDING_PROBE",
        "model_path": str(args.model_path),
        "model_type": model_type,
        "private_scale": args.private_scale,
        "n_packets": len(results),
        "n_pairs": n_pairs,
        "n_update_correct": n_update_correct,
        "n_retain_correct": n_retain_correct,
        "n_both_correct": n_both,
        "frac_update_correct": n_update_correct / max(1, n_pairs),
        "frac_retain_correct": n_retain_correct / max(1, n_pairs),
        "frac_both_correct": n_both / max(1, n_pairs),
        "mean_update_margin": sum(ps.get("update_margin", 0) for ps in pair_summaries) / max(1, n_pairs),
        "mean_retain_margin": sum(ps.get("retain_margin", 0) for ps in pair_summaries) / max(1, n_pairs),
        "per_pair": pair_summaries,
        "per_packet": results,
    }
    
    with open(out_dir / "binding_probe.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Markdown report
    lines = ["# research entity-binding cloze probe\n\n"]
    lines.append(f"Model: `{args.model_path}` ({model_type}, scale={args.private_scale})\n\n")
    lines.append("## Aggregate\n\n")
    lines.append(f"- Pairs: {n_pairs}\n")
    lines.append(f"- UPDATE correct: {n_update_correct}/{n_pairs} ({100*n_update_correct/max(1,n_pairs):.0f}%)\n")
    lines.append(f"- RETAIN correct: {n_retain_correct}/{n_pairs} ({100*n_retain_correct/max(1,n_pairs):.0f}%)\n")
    lines.append(f"- Both correct: {n_both}/{n_pairs} ({100*n_both/max(1,n_pairs):.0f}%)\n")
    lines.append(f"- Mean UPDATE margin: {summary['mean_update_margin']:+.3f}\n")
    lines.append(f"- Mean RETAIN margin: {summary['mean_retain_margin']:+.3f}\n\n")
    
    lines.append("## Per-pair detail\n\n")
    lines.append("| pair_id | UPDATE margin | RETAIN margin | Both correct |\n")
    lines.append("|---|---:|---:|:---:|\n")
    for ps in pair_summaries:
        um = ps.get("update_margin", float("nan"))
        rm = ps.get("retain_margin", float("nan"))
        bc = "✓" if ps.get("both_correct", False) else "✗"
        lines.append(f"| {ps['pair_id']} | {um:+.3f} | {rm:+.3f} | {bc} |\n")
    
    lines.append("\n## Per-packet detail\n\n")
    lines.append("| pair_id | type | answer | foil | margin | top1 | correct |\n")
    lines.append("|---|---|---|---|---:|---|:---:|\n")
    for r in results:
        if r["status"] == "scored":
            c = "✓" if r["correct_direction"] else "✗"
            lines.append(f"| {r['pair_id']} | {r['packet_type']} | {r['answer_text']} "
                          f"| {r['foil_text']} | {r['margin']:+.3f} "
                          f"| {r['top_prediction_first_pos']} | {c} |\n")
    
    with open(out_dir / "binding_probe.md", "w") as f:
        f.writelines(lines)
    
    print(json.dumps({
        "status": summary["status"],
        "n_pairs": n_pairs,
        "n_both_correct": n_both,
        "mean_update_margin": summary["mean_update_margin"],
        "mean_retain_margin": summary["mean_retain_margin"],
        "out_json": str(out_dir / "binding_probe.json"),
        "out_md": str(out_dir / "binding_probe.md"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
