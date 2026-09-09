#!/usr/bin/env python3
"""research procedural-state learning probe.

This script separates two possibilities after the research result:
1. the procedural state task was not learned even on its own distribution;
2. the procedural state task was learned but did not transfer to official Entity.

For a masked LM, each story is scored by replacing the final state word with a
candidate value and computing a pseudo-log-probability for the candidate word in
context. The coherent context should prefer the recorded target; the corrupted
context should prefer the alternative value implied by the swapped decisive move.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
META = ROOT / "data/state_revision_68/state_materialization_all_seeds.json"
DEFAULT_OUT = ROOT / "data/state_learning_probe.json"
DEFAULT_MODELS = {
    "official_wwm_seed42": ROOT / "training/runs/babylm_masked_wwm_pos512_1M/hf_model/chck_1M",
    "official_wwm_seed43": ROOT / "training/runs/babylm_masked_wwm_pos512_seed43_1M/hf_model/chck_1M",
    "state_coherent_seed42": ROOT / "training/runs/babylm_state_coherent_wwm_seed42_1M/hf_model/chck_1M",
    "state_corrupted_seed42": ROOT / "training/runs/babylm_state_corrupted_wwm_seed42_1M/hf_model/chck_1M",
    "state_coherent_seed43": ROOT / "training/runs/babylm_state_coherent_wwm_seed43_1M/hf_model/chck_1M",
    "state_corrupted_seed43": ROOT / "training/runs/babylm_state_corrupted_wwm_seed43_1M/hf_model/chck_1M",
}


@dataclass
class PairProbe:
    pair_id: str
    coherent_text: str
    corrupted_text: str
    answer: str
    alt: str
    target_word_index: int
    split: str


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def word_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for w in text.split():
        start = text.find(w, pos)
        if start < 0:
            raise RuntimeError(f"could not locate word {w!r} after {pos} in {text!r}")
        end = start + len(w)
        spans.append((start, end))
        pos = end
    return spans


def replace_word(text: str, idx: int, value: str) -> str:
    words = text.split()
    if idx < 0 or idx >= len(words):
        raise IndexError((idx, len(words), text))
    words[idx] = value
    return " ".join(words)


def target_token_indices(tokenizer, text: str, idx: int) -> list[int]:
    spans = word_spans(text)
    start, end = spans[idx]
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    hits = []
    for i, (a, b) in enumerate(offsets):
        if b <= start or a >= end:
            continue
        hits.append(i)
    if not hits:
        raise RuntimeError(f"no token span for word index {idx} {text.split()[idx]!r} in {text!r}; ids={ids[:20]} offsets={offsets[:20]}")
    return hits


@torch.no_grad()
def pseudo_word_score(model, tokenizer, base_text: str, idx: int, candidate: str, device: torch.device) -> dict[str, Any]:
    cand_text = replace_word(base_text, idx, candidate)
    enc = tokenizer(cand_text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    hits = target_token_indices(tokenizer, cand_text, idx)
    token_logps = []
    token_ids = input_ids[0, hits].detach().cpu().tolist()
    for j in hits:
        masked = input_ids.clone()
        masked[0, j] = tokenizer.mask_token_id
        logits = model(input_ids=masked, attention_mask=attention_mask).logits[0, j]
        logp = torch.log_softmax(logits, dim=-1)[int(input_ids[0, j])]
        token_logps.append(float(logp.detach().cpu()))
    return {
        "candidate": candidate,
        "candidate_text": cand_text,
        "target_token_indices": hits,
        "target_token_ids": token_ids,
        "sum_logp": float(sum(token_logps)),
        "mean_logp": float(sum(token_logps) / max(1, len(token_logps))),
        "num_tokens": len(token_logps),
    }


def load_heldout_pairs(meta: dict[str, Any], max_pairs: int, seed: int) -> list[PairProbe]:
    rows = read_jsonl(pathlib.Path(meta["heldout_probe"]["path"]))
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        base = str(r["probe_id"]).rsplit("_", 1)[0]
        grouped.setdefault(base, {})[r["mode"]] = r
    pairs = []
    for base, g in sorted(grouped.items(), key=lambda kv: int(kv[0])):
        if "coherent" not in g or "corrupted" not in g:
            continue
        c, k = g["coherent"], g["corrupted"]
        pairs.append(PairProbe(base, c["text"], k["text"], c["answer"], c["actual_value_corrupted"], int(c["target_word_index"]), "heldout_disjoint"))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def load_train_pairs(meta: dict[str, Any], train_seed: str, max_pairs: int, seed: int) -> list[PairProbe]:
    sm = meta["seeds"][train_seed]
    coh_rows = [r for r in read_jsonl(pathlib.Path(sm["coherent_path"])) if r.get("kind") == "pair_crossview"]
    cor_rows = [r for r in read_jsonl(pathlib.Path(sm["corrupted_path"])) if r.get("kind") == "pair_crossview"]
    cor_by_id = {str(r["state_story_id"]): r for r in cor_rows}
    pairs = []
    for c in coh_rows:
        sid = str(c["state_story_id"])
        k = cor_by_id.get(sid)
        if not k:
            continue
        pairs.append(PairProbe(sid, c["text"], k["text"], c["state_target_value"], c["state_actual_value_corrupted"], int(c["state_target_word_index"]), f"train_seed{train_seed}"))
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs]


def score_pairs(model, tokenizer, pairs: list[PairProbe], device: torch.device, detail_limit: int) -> dict[str, Any]:
    records = []
    for p in pairs:
        coh_ans = pseudo_word_score(model, tokenizer, p.coherent_text, p.target_word_index, p.answer, device)
        coh_alt = pseudo_word_score(model, tokenizer, p.coherent_text, p.target_word_index, p.alt, device)
        cor_ans = pseudo_word_score(model, tokenizer, p.corrupted_text, p.target_word_index, p.answer, device)
        cor_alt = pseudo_word_score(model, tokenizer, p.corrupted_text, p.target_word_index, p.alt, device)
        margin_coh = coh_ans["mean_logp"] - coh_alt["mean_logp"]
        margin_cor = cor_alt["mean_logp"] - cor_ans["mean_logp"]
        records.append({
            "pair_id": p.pair_id,
            "split": p.split,
            "answer": p.answer,
            "alt": p.alt,
            "margin_coherent_answer_minus_alt": margin_coh,
            "margin_corrupted_alt_minus_answer": margin_cor,
            "event_sensitivity": margin_coh + margin_cor,
            "coherent_prefers_supported": margin_coh > 0,
            "corrupted_prefers_supported": margin_cor > 0,
            "answer_token_count": coh_ans["num_tokens"],
            "alt_token_count": coh_alt["num_tokens"],
        })
    n = len(records)
    if n == 0:
        raise RuntimeError("no probe records scored")
    def mean(key: str) -> float:
        return float(sum(float(r[key]) for r in records) / n)
    summary = {
        "num_pairs": n,
        "coherent_accuracy": sum(1 for r in records if r["coherent_prefers_supported"]) / n,
        "corrupted_accuracy": sum(1 for r in records if r["corrupted_prefers_supported"]) / n,
        "both_contexts_accuracy": sum(1 for r in records if r["coherent_prefers_supported"] and r["corrupted_prefers_supported"]) / n,
        "mean_margin_coherent_answer_minus_alt": mean("margin_coherent_answer_minus_alt"),
        "mean_margin_corrupted_alt_minus_answer": mean("margin_corrupted_alt_minus_answer"),
        "mean_event_sensitivity": mean("event_sensitivity"),
        "details_first": records[:detail_limit],
    }
    return summary


def resolve_model_path(path: pathlib.Path) -> pathlib.Path:
    if path.exists():
        return path
    # Some older runs may only have a root hf_model.
    root = path.parent if path.name.startswith("chck_") else path
    if root.exists():
        return root
    raise FileNotFoundError(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max_pairs", type=int, default=120)
    ap.add_argument("--train_seed", default="42")
    ap.add_argument("--sample_seed", type=int, default=7000)
    ap.add_argument("--detail_limit", type=int, default=12)
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    meta = json.loads(META.read_text(encoding="utf-8"))
    heldout_pairs = load_heldout_pairs(meta, args.max_pairs, args.sample_seed)
    train_pairs = load_train_pairs(meta, args.train_seed, args.max_pairs, args.sample_seed)
    model_paths = DEFAULT_MODELS
    if args.models:
        model_paths = {}
        for spec in args.models.split(","):
            name, p = spec.split("=", 1)
            model_paths[name] = pathlib.Path(p)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, Any] = {
        "purpose": "distinguish procedural state task not learned vs learned without official transfer",
        "meta": str(META),
        "device": str(device),
        "max_pairs_per_split": args.max_pairs,
        "train_seed_split": args.train_seed,
        "models": {},
    }
    for name, path in model_paths.items():
        mp = resolve_model_path(path)
        print(f"Loading {name}: {mp}", flush=True)
        tok = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForMaskedLM.from_pretrained(mp).to(device)
        model.eval()
        result["models"][name] = {
            "model_path": str(mp),
            "train_distribution": score_pairs(model, tok, train_pairs, device, args.detail_limit),
            "heldout_disjoint": score_pairs(model, tok, heldout_pairs, device, args.detail_limit),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "STATE_LEARNING_PROBE_DONE", "out": str(out), "models": list(result["models"].keys())}, indent=2))


if __name__ == "__main__":
    main()
