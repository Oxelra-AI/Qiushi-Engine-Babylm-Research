#!/usr/bin/env python3
"""research: head-free MLM prior readout for research state rows.

This is an inference-only, CPU-compatible probe. It asks whether the underlying
BabyLM DeBERTa masked language model already prefers the true/inverted owner in
state-query prompts before any research classification fine-tuning.

It scores rows in paired candidate groups by replacing the candidate name in
"After the event, <candidate> had the <object>." with one or more mask tokens and
summing the logits assigned to the candidate token IDs. This is not an NLI
classifier; it is a direct cloze-style prior readout from the MLM head.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any, Tuple

# Force CPU unless the caller deliberately overrides before process start.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

STUDY = Path("experiments/archive/representation_and_objectives")
WS = STUDY
SUB = WS / "data" / "equivariant_symmetry_substrate"
CKPT_BASE = WS / "training" / "runs" / "qwen_8x480_16k_wwm_to_token_100M_seed43022" / "hf_model"
OUT = WS / "data" / "mlm_prior_state_readout"

EVAL_SUITES = [
    "paired_state_conservation",
    "cross_template_state_readout",
    "name_permutation_counterfactual",
]

TRAIN_ARMS = [
    "aligned_state_bridge",
    "inverted_state_bridge",
    "neutral_decoupled",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def candidate_ids(tokenizer, name: str) -> List[int]:
    ids = tokenizer.encode(name, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenization for {name!r}")
    return ids


def make_prompt(tokenizer, premise: str, obj: str, n_masks: int) -> str:
    masks = " ".join([tokenizer.mask_token] * n_masks)
    return f"{premise} After the event, {masks} had the {obj}."


def score_candidate(model, tokenizer, device, premise: str, obj: str, name: str) -> float:
    ids = candidate_ids(tokenizer, name)
    prompt = make_prompt(tokenizer, premise, obj, len(ids))
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    mask_positions = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=False).flatten().tolist()
    if len(mask_positions) != len(ids):
        raise RuntimeError(f"expected {len(ids)} masks, found {len(mask_positions)} in {prompt}")
    import torch
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0]
        val = 0.0
        for pos, tid in zip(mask_positions, ids):
            val += float(logits[pos, tid].detach().cpu())
    return val / len(ids)


def object_for_group(rows: List[Dict[str, Any]]) -> str:
    r = rows[0]
    if r.get("query_kind") == "changed":
        return r.get("changed_object") or r.get("object")
    return r.get("static_object") or r.get("object")


def group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("task") != "state_query":
            continue
        groups[(str(r.get("pair_id")), str(r.get("query_kind")))].append(r)
    return groups


def score_groups(model, tokenizer, device, rows: List[Dict[str, Any]], source_name: str, max_groups: int | None = None):
    groups = group_rows(rows)
    items = sorted(groups.items())
    if max_groups is not None:
        items = items[:max_groups]
    scored = []
    tok_counter = Counter()
    for (pid, qk), gr in items:
        # Expect two candidate rows. If duplicated by voice/static orbit, row_id grouping
        # already includes voice/static in pair_id, so this should hold.
        cand_rows = sorted(gr, key=lambda r: r.get("candidate_slot", 99))
        if len(cand_rows) != 2:
            # Some malformed group; record and skip.
            continue
        premise = cand_rows[0]["premise"]
        obj = object_for_group(cand_rows)
        entries = []
        for r in cand_rows:
            name = r["candidate"]
            ids = candidate_ids(tokenizer, name)
            tok_counter[(name, len(ids))] += 1
            s = score_candidate(model, tokenizer, device, premise, obj, name)
            entries.append({"candidate": name, "score": s, "label": bool(r["label"]), "candidate_slot": r.get("candidate_slot"), "correct_slot": r.get("correct_slot")})
        # Positive margin = score(correct candidate) - score(incorrect candidate) under the row labels.
        correct = [e for e in entries if e["label"]]
        incorrect = [e for e in entries if not e["label"]]
        if len(correct) != 1 or len(incorrect) != 1:
            continue
        pred = max(entries, key=lambda e: e["score"])
        margin = correct[0]["score"] - incorrect[0]["score"]
        scored.append({
            "source": source_name,
            "pair_id": pid,
            "query_kind": qk,
            "relation": cand_rows[0].get("relation"),
            "component": cand_rows[0].get("component"),
            "orientation_dependency": cand_rows[0].get("orientation_dependency"),
            "inverted_bridge_label": cand_rows[0].get("inverted_bridge_label"),
            "label_correct_slot": correct[0].get("correct_slot"),
            "pred_candidate": pred["candidate"],
            "pred_slot": pred.get("candidate_slot"),
            "correct_candidate": correct[0]["candidate"],
            "accuracy": 1.0 if pred["label"] else 0.0,
            "margin_correct_minus_incorrect": margin,
            "entries": entries,
        })
    return scored, tok_counter


def summarize(scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    def sub_summary(xs):
        if not xs:
            return {"n": 0}
        accs = [x["accuracy"] for x in xs]
        margins = [x["margin_correct_minus_incorrect"] for x in xs]
        return {
            "n": len(xs),
            "accuracy": mean(accs),
            "accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
            "margin_mean": mean(margins),
            "margin_std": pstdev(margins) if len(margins) > 1 else 0.0,
        }
    out = {"overall": sub_summary(scored)}
    for key in ["source", "query_kind", "relation", "component", "orientation_dependency", "inverted_bridge_label"]:
        by = defaultdict(list)
        for x in scored:
            by[str(x.get(key))].append(x)
        out[f"by_{key}"] = {k: sub_summary(v) for k, v in sorted(by.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="chck_80M")
    ap.add_argument("--max-groups", type=int, default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, DebertaV2ForMaskedLM

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_BASE / args.checkpoint
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = DebertaV2ForMaskedLM.from_pretrained(str(ckpt))
    model.eval().to(device)

    all_scored = []
    tokenization = Counter()

    # Eval state suites: true labels under held nonce relations.
    for suite in EVAL_SUITES:
        rows = load_jsonl(SUB / "eval" / f"{suite}.jsonl")
        scored, tok = score_groups(model, tokenizer, device, rows, f"eval/{suite}", args.max_groups)
        all_scored.extend(scored)
        tokenization.update(tok)

    # Training state rows for bridge arms. These include aligned true labels,
    # inverted labels, and seen-relation neutral labels.
    for arm in TRAIN_ARMS:
        rows = load_jsonl(SUB / "arms" / arm / "train_supervised.jsonl")
        state_rows = [r for r in rows if r.get("task") == "state_query"]
        scored, tok = score_groups(model, tokenizer, device, state_rows, f"train/{arm}", args.max_groups)
        all_scored.extend(scored)
        tokenization.update(tok)

    summary = summarize(all_scored)
    by_source = summary["by_source"]

    result = {
        "checkpoint": str(ckpt),
        "n_scored_groups": len(all_scored),
        "summary": summary,
        "tokenization_name_lengths": {f"{k[0]}|{k[1]}": v for k, v in sorted(tokenization.items())},
        "scored_groups": all_scored,
        "interpretation_note": "Cloze accuracy is a head-free MLM preference for candidate owner names, not the research classifier task. It helps test whether a prior is directly visible before fine-tuning.",
    }
    out_json = OUT / f"mlm_prior_state_readout_{args.checkpoint}.json"
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    lines.append(f"# research MLM prior state readout ({args.checkpoint})")
    lines.append("")
    lines.append("Inference-only cloze probe: premise + `After the event, [MASK] had the object.` Candidate scores are MLM logits for the two names. Positive margin means the MLM prefers the row label's owner.")
    lines.append("")
    lines.append(f"- scored candidate groups: {len(all_scored)}")
    lines.append(f"- checkpoint: `{ckpt}`")
    lines.append("")
    lines.append("## By source")
    lines.append("| source | n | accuracy | margin mean | margin std |")
    lines.append("|---|---:|---:|---:|---:|")
    for src, s in sorted(by_source.items()):
        lines.append(f"| {src} | {s.get('n',0)} | {s.get('accuracy',float('nan')):.3f} | {s.get('margin_mean',float('nan')):.3f} | {s.get('margin_std',float('nan')):.3f} |")
    lines.append("")
    lines.append("## By relation")
    lines.append("| relation | n | accuracy | margin mean |")
    lines.append("|---|---:|---:|---:|")
    for rel, s in sorted(summary["by_relation"].items()):
        lines.append(f"| {rel} | {s.get('n',0)} | {s.get('accuracy',float('nan')):.3f} | {s.get('margin_mean',float('nan')):.3f} |")
    lines.append("")
    lines.append("## By inverted_bridge_label")
    lines.append("| inverted_bridge_label | n | accuracy | margin mean |")
    lines.append("|---|---:|---:|---:|")
    for inv, s in sorted(summary["by_inverted_bridge_label"].items()):
        lines.append(f"| {inv} | {s.get('n',0)} | {s.get('accuracy',float('nan')):.3f} | {s.get('margin_mean',float('nan')):.3f} |")
    lines.append("")
    lines.append("## Reading")
    lines.append("Chance-level held-eval accuracy would mean the research state gain is not directly visible in the MLM before supervised state-format fine-tuning. Above-chance true-direction held-eval accuracy would support a directly accessible pretrained/event-surface prior. Poor accuracy on inverted training labels would be expected if the base model favors the original true direction rather than the intentionally inverted labels.")
    lines.append("")
    lines.append(f"- full JSON: `{out_json}`")
    out_md = OUT / f"mlm_prior_state_readout_{args.checkpoint}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "MLM_PRIOR_STATE_READOUT_COMPLETE",
        "checkpoint": args.checkpoint,
        "summary_md": str(out_md),
        "results_json": str(out_json),
        "n_scored_groups": len(all_scored),
        "overall_acc": summary["overall"].get("accuracy"),
        "no_training_official_eval_upload": True,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
