#!/usr/bin/env python3
from __future__ import annotations

import gc
import hashlib
import json
import pathlib

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path("experiments/archive/initial_model_studies")
OUT = ROOT / "data/checkpoint_resolution_probe.json"
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/checkpoint_resolution_probe.md')

CHECKPOINTS = {
    "baseline_1M": ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_1M",
    "baseline_10M": ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_10M",
    "baseline_20M": ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_20M",
    "baseline_100M": ROOT / "training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M",
    "relation_10M": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/hf_model/chck_10M",
    "relation_20M": ROOT / "training/runs/babylm_relation_wwm_deberta_20M/hf_model/chck_20M",
    "shuffled_10M": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/hf_model/chck_10M",
    "shuffled_20M": ROOT / "training/runs/babylm_relation_wwm_shuffled_deberta_20M/hf_model/chck_20M",
}

PROMPTS = [
    ("The cat is on the {mask}.", ["table", "floor", "box", "chair"]),
    ("The boy put the apple in the {mask}.", ["box", "room", "water", "street"]),
    ("Glass is usually {mask}.", ["clear", "heavy", "soft", "alive"]),
    ("Before eating, the child opened the {mask}.", ["box", "door", "book", "apple"]),
]


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_file(path: pathlib.Path) -> pathlib.Path:
    for name in ["model.safetensors", "pytorch_model.bin"]:
        p = path / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no model weights in {path}")


def score_checkpoint(name: str, path: pathlib.Path, device: str) -> dict:
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path).to(device)
    model.eval()
    out = {"path": str(path), "model_file": str(model_file(path)), "model_sha256": sha256_file(model_file(path)), "tokenizer_len": len(tok), "parameter_count": model.num_parameters(), "scores": []}
    mask_id = tok.mask_token_id
    with torch.no_grad():
        for prompt_template, targets in PROMPTS:
            prompt = prompt_template.format(mask=tok.mask_token)
            enc = tok(prompt, return_tensors="pt").to(device)
            positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=False).view(-1)
            if len(positions) != 1:
                raise RuntimeError(f"prompt has {len(positions)} masks: {prompt} (mask_token={tok.mask_token!r}, mask_id={mask_id})")
            logits = model(**enc).logits[0, int(positions[0])]
            log_probs = torch.log_softmax(logits, dim=-1)
            rec = {"prompt_template": prompt_template, "prompt": prompt, "mask_token": tok.mask_token, "target_logprobs": {}, "target_ids": {}}
            for target in targets:
                ids = tok(target, add_special_tokens=False)["input_ids"]
                if len(ids) != 1:
                    rec["target_logprobs"][target] = None
                    rec["target_ids"][target] = ids
                else:
                    rec["target_logprobs"][target] = float(log_probs[int(ids[0])].detach().cpu())
                    rec["target_ids"][target] = ids
            out["scores"].append(rec)
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def flatten_scores(rec: dict) -> dict[str, float]:
    vals = {}
    for pr in rec["scores"]:
        for tgt, val in pr["target_logprobs"].items():
            if val is not None:
                vals[f"{pr['prompt']}::{tgt}"] = val
    return vals


def max_abs_delta(a: dict, b: dict) -> float:
    aa = flatten_scores(a); bb = flatten_scores(b)
    keys = sorted(set(aa) & set(bb))
    return max(abs(aa[k] - bb[k]) for k in keys)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {name: score_checkpoint(name, path, device) for name, path in CHECKPOINTS.items()}
    comparisons = {}
    for a, b in [("baseline_1M", "baseline_10M"), ("baseline_10M", "baseline_20M"), ("baseline_20M", "baseline_100M"), ("relation_10M", "relation_20M"), ("shuffled_10M", "shuffled_20M"), ("relation_20M", "shuffled_20M")]:
        comparisons[f"{a}_vs_{b}"] = {
            "same_hash": results[a]["model_sha256"] == results[b]["model_sha256"],
            "max_abs_logprob_delta_on_probe": max_abs_delta(results[a], results[b]),
        }
    payload = {"status": "CHECKPOINT_RESOLUTION_PROBE_DONE", "device": device, "checkpoints": results, "comparisons": comparisons, "interpretation": "Direct AutoModel loads from checkpoint directories. Non-identical hashes/probe log-probs mean saved checkpoints differ outside the evaluator; identical task scores then point to evaluation sensitivity/cache/revision handling rather than identical weights."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research — checkpoint-resolution probe", "", f"Evidence JSON: `{OUT}`", "", f"Device: {device}", "", "| comparison | same hash | max abs logprob delta |", "|---|---:|---:|"]
    for k, v in comparisons.items():
        lines.append(f"| {k} | {v['same_hash']} | {v['max_abs_logprob_delta_on_probe']:.6f} |")
    NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(OUT), "comparisons": comparisons}, indent=2))

if __name__ == "__main__":
    main()
