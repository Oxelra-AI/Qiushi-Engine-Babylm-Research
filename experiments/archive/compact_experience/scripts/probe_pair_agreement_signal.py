#!/usr/bin/env python3
"""Probe whether clean same-window Qwen training creates an exploitable pair-agreement signal.

This is a no-weight-update, no-official-eval-data probe for an objective/credit-assignment bet.
It uses only research selected training pairs. It compares sentence-level representations for
original->rewrite positives against shuffled rewrites. If aligned positives are not separable
from shuffled negatives, a pair-agreement objective is a weak candidate; if the clean model
shows a stronger margin than official controls, a low-dose objective-level agreement run is
worth a small training test.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import pathlib
import random
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = _public_path('experiments/archive/compact_experience')
PAIR_FILE = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl')
OUT_DIR = _public_path('experiments/archive/compact_experience/data/pair_agreement_probe')
DEFAULT_MODELS = {
    "clean_qwen_seed43022_100M": _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M'),
    "official_lengthmatched_seed43022_100M": _public_path('experiments/archive/compact_experience/training/runs/official_lengthmatched_16k_seed43022/hf_model/chck_100M'),
    "qwen_shuffled_seed43022_100M": _public_path('experiments/archive/compact_experience/training/runs/qwen_shuffled_control_16k_seed43022/hf_model/chck_100M'),
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_pairs(path: pathlib.Path, n: int, seed: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("original") and d.get("rewrite"):
                    rows.append(d)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def deranged_indices(n: int, seed: int) -> List[int]:
    idx = list(range(n))
    rng = random.Random(seed)
    for _ in range(1000):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return idx
    return idx[1:] + idx[:1]


def encode_reps(model, tok, texts: List[str], device: torch.device, max_len: int, batch_size: int) -> torch.Tensor:
    reps = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1]
            mask = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            rep = (h * mask).sum(dim=1) / denom
            rep = F.normalize(rep.float(), p=2, dim=-1)
            reps.append(rep.cpu())
    return torch.cat(reps, dim=0)


def model_probe(name: str, path: pathlib.Path, pairs: List[Dict[str, Any]], args) -> Dict[str, Any]:
    if not path.exists():
        return {"model": name, "path": str(path), "available": False, "error": "missing_model_path"}
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path))
    model.to(device)
    model.eval()
    orig = [p["original"] for p in pairs]
    rew = [p["rewrite"] for p in pairs]
    t0 = time.time()
    ro = encode_reps(model, tok, orig, device, args.max_len, args.batch_size)
    rr = encode_reps(model, tok, rew, device, args.max_len, args.batch_size)
    perm = deranged_indices(len(pairs), args.seed + 991)
    pos = (ro * rr).sum(dim=1)
    neg = (ro * rr[perm]).sum(dim=1)
    logits = ro @ rr.T / args.temperature
    labels = torch.arange(len(pairs))
    top1 = (logits.argmax(dim=1) == labels).float().mean().item()
    loss = F.cross_entropy(logits, labels).item()
    margin = pos - neg
    def stat(x: torch.Tensor) -> Dict[str, float]:
        xs = x.detach().cpu().float().sort().values
        n = xs.numel()
        def q(p: float) -> float:
            if n == 1:
                return float(xs[0])
            loc = p * (n - 1)
            lo = int(math.floor(loc)); hi = min(lo + 1, n - 1); frac = loc - lo
            return float(xs[lo] * (1 - frac) + xs[hi] * frac)
        return {"mean": float(xs.mean()), "std": float(xs.std(unbiased=False)), "min": float(xs[0]), "p10": q(0.10), "median": q(0.50), "p90": q(0.90), "max": float(xs[-1])}
    # Bootstrap a simple margin uncertainty from the fixed sampled training pairs.
    rng = random.Random(args.seed + 12345)
    boot = []
    vals = margin.tolist()
    for _ in range(args.bootstrap):
        sample = [vals[rng.randrange(len(vals))] for _ in vals]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "model": name,
        "path": str(path),
        "available": True,
        "device": str(device),
        "n_pairs": len(pairs),
        "max_len": args.max_len,
        "elapsed_sec": round(time.time() - t0, 3),
        "positive_cosine": stat(pos),
        "shuffled_negative_cosine": stat(neg),
        "positive_minus_negative_margin": stat(margin),
        "margin_bootstrap_ci90": [boot[int(0.05 * (len(boot)-1))], boot[int(0.95 * (len(boot)-1))]] if boot else None,
        "inbatch_retrieval_top1": top1,
        "inbatch_infonce_loss": loss,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=str(PAIR_FILE))
    ap.add_argument("--max_pairs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=41041)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--models", nargs="*", default=[] , help="Optional name=path overrides. Defaults compare clean, official, shuffled if available.")
    args = ap.parse_args()
    pairs = read_pairs(pathlib.Path(args.pairs), args.max_pairs, args.seed)
    model_map = dict(DEFAULT_MODELS)
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"model spec must be name=path: {spec}")
        k, v = spec.split("=", 1)
        model_map[k] = pathlib.Path(v)
    rows = [model_probe(name, path, pairs, args) for name, path in model_map.items()]
    by = {r["model"]: r for r in rows}
    contrasts = []
    def mean_margin(r):
        return None if not r.get("available") else r["positive_minus_negative_margin"]["mean"]
    for a, b in [("clean_qwen_seed43022_100M", "official_lengthmatched_seed43022_100M"), ("clean_qwen_seed43022_100M", "qwen_shuffled_seed43022_100M")]:
        if a in by and b in by and by[a].get("available") and by[b].get("available"):
            contrasts.append({"contrast": f"{a}_minus_{b}", "delta_mean_margin": mean_margin(by[a]) - mean_margin(by[b]), "delta_retrieval_top1": by[a]["inbatch_retrieval_top1"] - by[b]["inbatch_retrieval_top1"]})
    payload = {
        "status": "PAIR_AGREEMENT_SIGNAL_PROBE",
        "started_utc": now(),
        "non_leakage_statement": "Uses only research selected training original-Qwen rewrite pairs and submitted-model representations; no official AoA/CDI items, child curves, AoA scores, SuperGLUE labels, or downstream eval outputs are used.",
        "purpose": "Minimal falsifying probe for a possible pair-agreement objective: require aligned positive representations to separate from shuffled rewrites, and preferably stronger separation in clean-Qwen than in official/shuffled controls, before any weight-changing objective experiment.",
        "pair_file": str(args.pairs),
        "sample_pair_ids": [p.get("pair_id") for p in pairs[:10]],
        "args": vars(args),
        "models": rows,
        "contrasts": contrasts,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _public_path('experiments/archive/compact_experience/data/pair_agreement_probe/pair_agreement_signal.json')
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    note = _public_path('research/documents/compact_experience/data/pair_agreement_probe/pair_agreement_signal.md')
    lines = ["# research pair-agreement representation probe", "", payload["non_leakage_statement"], "", payload["purpose"], "", "## Results"]
    for r in rows:
        if not r.get("available"):
            lines.append(f"- {r['model']}: missing at `{r['path']}`")
        else:
            lines.append(f"- {r['model']}: margin_mean={r['positive_minus_negative_margin']['mean']:.4f}, ci90={r['margin_bootstrap_ci90']}, top1={r['inbatch_retrieval_top1']:.4f}, loss={r['inbatch_infonce_loss']:.4f}")
    lines += ["", "## Contrasts"]
    for c in contrasts:
        lines.append(f"- {c['contrast']}: delta_mean_margin={c['delta_mean_margin']:.4f}, delta_top1={c['delta_retrieval_top1']:.4f}")
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "note": str(note), "contrasts": contrasts}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
