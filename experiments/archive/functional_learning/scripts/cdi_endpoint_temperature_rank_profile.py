#!/usr/bin/env python3
"""research: CDI endpoint surprisal/rank profile under non-benchmark temperature.

This is an explanatory frozen-endpoint analysis, not AoA scoring.  It uses the official
CDI AoA target/context construction but only a deterministic subset, applies the
research temperature fitted on ordinary non-Qwen legal-tail text, and reports target
rank movement.  A global temperature can change NLL but cannot change ranks.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoProcessor

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))

import batched_aoa_extractor as aoa_batch  # noqa: E402
from evaluation_pipeline.AoA_word.eval_util import load_eval  # noqa: E402

DEFAULT_TEMP = _public_path('experiments/archive/functional_learning/data/temperature_source_readout/temperature_source_readout.json')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/cdi_endpoint_temperature_rank_profile')

MODEL_PATHS: Dict[str, pathlib.Path] = {
    "coherent86": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075'),
    "dense_focus_seed62064": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080'),
    "dense_focus_seed62065": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62065_u0080'),
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path, block_size: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def finite(xs: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        if x is None:
            continue
        try:
            xf = float(x)
        except Exception:
            continue
        if math.isfinite(xf):
            out.append(xf)
    return out


def mean(xs: Iterable[Any]) -> Optional[float]:
    vals = finite(xs)
    return sum(vals) / len(vals) if vals else None


def median(xs: Iterable[Any]) -> Optional[float]:
    vals = finite(xs)
    return statistics.median(vals) if vals else None


def sem(xs: Iterable[Any]) -> Optional[float]:
    vals = finite(xs)
    if len(vals) <= 1:
        return None
    return statistics.stdev(vals) / math.sqrt(len(vals))


def rank_from_logits(logits_row: torch.Tensor, target_id: int) -> int:
    return int(torch.sum(logits_row > logits_row[int(target_id)]).item()) + 1


def nll_from_logits(logits_row: torch.Tensor, target_id: int, temperature: float) -> float:
    z = logits_row.float() / float(temperature)
    return float(torch.logsumexp(z, dim=-1).item() - z[int(target_id)].item())


def load_temperatures(path: pathlib.Path) -> Dict[str, float]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fits = obj.get("temperature_fits", {})
    temps: Dict[str, float] = {}
    for name, fit in fits.items():
        if name.startswith("dense_focus") or name == "coherent86":
            temps[name] = float(fit.get("best_temperature", 1.0))
    return temps


def make_subset(max_words: int, seed: int) -> Tuple[List[str], List[List[str]], Dict[str, Any]]:
    words, contexts = load_eval(aoa_batch.CDI_WORDS_PATH, min_context=0, debug=False)
    total_contexts = sum(len(c) for c in contexts)
    indices = list(range(len(words)))
    rng = random.Random(seed)
    if max_words and max_words > 0 and max_words < len(indices):
        indices = rng.sample(indices, max_words)
        indices.sort()
    sub_words = [words[i] for i in indices]
    sub_contexts = [contexts[i] for i in indices]
    return sub_words, sub_contexts, {
        "cdi_words_path": rel(aoa_batch.CDI_WORDS_PATH),
        "cdi_words_sha256": sha256_file(aoa_batch.CDI_WORDS_PATH),
        "total_words_available": len(words),
        "total_contexts_available": total_contexts,
        "sampled_word_indices": indices,
        "sample_seed": int(seed),
        "words_evaluated": len(sub_words),
        "contexts_evaluated": sum(len(c) for c in sub_contexts),
        "target_words_preview": sub_words[:16],
    }


def score_model(model_name: str, model_path: pathlib.Path, temperature: float, target_words: List[str], contexts: List[List[str]],
                device: str, batch_size: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Use research cache repair; it synchronizes Transformers globals too.
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal batch, rows
        if not batch:
            return
        max_len = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
        att = torch.zeros((len(batch), max_len), dtype=torch.long)
        indices = torch.empty((len(batch),), dtype=torch.long)
        targets = torch.empty((len(batch),), dtype=torch.long)
        for i, item in enumerate(batch):
            L = len(item["input_ids"])
            ids[i, :L] = torch.tensor(item["input_ids"], dtype=torch.long)
            att[i, :L] = torch.tensor(item["attention_mask"], dtype=torch.long)
            indices[i] = int(item["index"])
            targets[i] = int(item["target"])
        ids = ids.to(device)
        att = att.to(device)
        indices = indices.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=att)
            logits = out[0] if isinstance(out, tuple) else out["logits"]
            masked = logits[torch.arange(logits.shape[0], device=logits.device), indices].float().detach().cpu()
        for item, z, tid in zip(batch, masked, targets.detach().cpu().tolist(), strict=False):
            rows.append({
                "model": model_name,
                "word_i": int(item["word_i"]),
                "context_i": int(item["context_i"]),
                "target_word": item["target_word"],
                "target_token_id": int(tid),
                "nll_T1": nll_from_logits(z, int(tid), 1.0),
                "nll_Tfit": nll_from_logits(z, int(tid), float(temperature)),
                "target_rank": rank_from_logits(z, int(tid)),
            })
        batch = []

    t0 = time.time()
    variants = 0
    for word_i, (word, ctxs) in enumerate(zip(target_words, contexts, strict=False)):
        for context_i, context in enumerate(ctxs):
            toks, atts, phrase_indices, target_tokens = aoa_batch.prepare_mlm_variants(processor, tokenizer, context, word, False)
            # For multi-subword words, report each target subtoken; this preserves the raw masked-token pressure.
            for ids, att, idx, target in zip(toks, atts, phrase_indices, target_tokens, strict=False):
                batch.append({"word_i": word_i, "context_i": context_i, "target_word": word, "input_ids": ids, "attention_mask": att, "index": idx, "target": target})
                variants += 1
                if len(batch) >= int(batch_size):
                    flush()
    flush()
    info = {
        "model_name": model_name,
        "model_path": rel(model_path),
        "temperature": float(temperature),
        "n_rows": len(rows),
        "variants": variants,
        "elapsed_sec": round(time.time() - t0, 2),
        "checkpoint": aoa_batch.checkpoint_info(model_name, model_path),
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, info


def summarize(rows: List[Dict[str, Any]], parent_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    p = {(int(r["word_i"]), int(r["context_i"]), int(r["target_token_id"])): r for r in parent_rows or []}
    deltas = []
    for r in rows:
        key = (int(r["word_i"]), int(r["context_i"]), int(r["target_token_id"]))
        pr = p.get(key)
        if pr:
            deltas.append({
                "delta_nll_T1": float(r["nll_T1"]) - float(pr["nll_T1"]),
                "delta_nll_Tfit": float(r["nll_Tfit"]) - float(pr["nll_Tfit"]),
                "delta_rank": float(r["target_rank"]) - float(pr["target_rank"]),
                "improved_nll_T1": float(r["nll_T1"]) < float(pr["nll_T1"]),
                "improved_nll_Tfit": float(r["nll_Tfit"]) < float(pr["nll_Tfit"]),
                "improved_rank": float(r["target_rank"]) < float(pr["target_rank"]),
            })
    by_word: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_word[int(r["word_i"])].append(r)
    word_means = []
    p_by_word: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in parent_rows or []:
        p_by_word[int(r["word_i"])].append(r)
    for wi, g in by_word.items():
        obj = {"word_i": wi, "target_word": g[0]["target_word"], "mean_nll_T1": mean(r["nll_T1"] for r in g), "mean_nll_Tfit": mean(r["nll_Tfit"] for r in g), "mean_rank": mean(r["target_rank"] for r in g)}
        pg = p_by_word.get(wi)
        if pg:
            obj.update({
                "delta_word_mean_nll_T1": obj["mean_nll_T1"] - mean(r["nll_T1"] for r in pg),
                "delta_word_mean_nll_Tfit": obj["mean_nll_Tfit"] - mean(r["nll_Tfit"] for r in pg),
                "delta_word_mean_rank": obj["mean_rank"] - mean(r["target_rank"] for r in pg),
            })
        word_means.append(obj)
    return {
        "n_rows": len(rows),
        "mean_nll_T1": mean(r["nll_T1"] for r in rows),
        "mean_nll_Tfit": mean(r["nll_Tfit"] for r in rows),
        "median_nll_T1": median(r["nll_T1"] for r in rows),
        "mean_rank": mean(r["target_rank"] for r in rows),
        "median_rank": median(r["target_rank"] for r in rows),
        "mean_delta_nll_T1_vs_parent": mean(d["delta_nll_T1"] for d in deltas),
        "mean_delta_nll_Tfit_vs_parent": mean(d["delta_nll_Tfit"] for d in deltas),
        "mean_delta_rank_vs_parent": mean(d["delta_rank"] for d in deltas),
        "median_delta_rank_vs_parent": median(d["delta_rank"] for d in deltas),
        "improved_fraction_nll_T1": (sum(1 for d in deltas if d["improved_nll_T1"]) / len(deltas)) if deltas else None,
        "improved_fraction_nll_Tfit": (sum(1 for d in deltas if d["improved_nll_Tfit"]) / len(deltas)) if deltas else None,
        "improved_fraction_rank": (sum(1 for d in deltas if d["improved_rank"]) / len(deltas)) if deltas else None,
        "word_mean_delta_nll_T1": mean(w.get("delta_word_mean_nll_T1") for w in word_means),
        "word_mean_delta_nll_Tfit": mean(w.get("delta_word_mean_nll_Tfit") for w in word_means),
        "word_mean_delta_rank": mean(w.get("delta_word_mean_rank") for w in word_means),
        "word_improved_fraction_rank": (sum(1 for w in word_means if w.get("delta_word_mean_rank") is not None and w["delta_word_mean_rank"] < 0) / len([w for w in word_means if w.get("delta_word_mean_rank") is not None])) if parent_rows else None,
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--temperature-json", type=pathlib.Path, default=DEFAULT_TEMP)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--models", nargs="+", default=["coherent86", "dense_focus_seed62064", "dense_focus_seed62065"], choices=list(MODEL_PATHS.keys()))
    ap.add_argument("--max-words", type=int, default=96)
    ap.add_argument("--sample-seed", type=int, default=86032)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Repair cache env before AutoProcessor/AutoModel dynamic module loading.
    dummy = argparse.Namespace(out_dir=out_dir, device="cpu", mode="shared", target="coherent86", ancestral_limit=1, max_words=1, word_start=0, word_end=0, gpu=0, batch_size=1, use_bos_only=False, progress_every=0)
    aoa_batch.run_extract.__globals__["_ensure_writable_cache_env"]("HF_HOME", out_dir / "hf_cache" / "hf_home")
    aoa_batch.run_extract.__globals__["_ensure_writable_cache_env"]("TRANSFORMERS_CACHE", out_dir / "hf_cache" / "transformers")
    aoa_batch.run_extract.__globals__["_ensure_writable_cache_env"]("HF_MODULES_CACHE", out_dir / "hf_cache" / "modules")
    import transformers.utils.hub as _hf_hub
    import transformers.dynamic_module_utils as _hf_dyn
    _hf_hub.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
    _hf_dyn.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]

    temps = load_temperatures(args.temperature_json)
    words, contexts, subset = make_subset(int(args.max_words), int(args.sample_seed))
    plan = {
        "status": "CDI_ENDPOINT_TEMPERATURE_RANK_PLAN",
        "created_utc": now(),
        "temperature_json": rel(args.temperature_json),
        "temperatures": temps,
        "models": {m: rel(MODEL_PATHS[m]) for m in args.models},
        "subset": subset,
        "boundary": "Explanatory endpoint-only CDI masked-token NLL/rank profile. Temperature was fitted on non-benchmark text; no official AoA or leaderboard score is modified.",
    }
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2), flush=True)
    if args.device == "auto":
        device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        device = f"cuda:{int(args.gpu)}"
    else:
        device = "cpu"
    infos: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {}
    parent_rows: Optional[List[Dict[str, Any]]] = None
    for name in args.models:
        t0 = time.time()
        temp = float(temps.get(name, 1.0))
        print(json.dumps({"event": "load_score_model", "model": name, "temperature": temp, "device": device}), flush=True)
        rows, info = score_model(name, MODEL_PATHS[name], temp, words, contexts, device, int(args.batch_size))
        write_jsonl(out_dir / f"cdi_scores_{name}.jsonl", rows)
        if name == "coherent86":
            parent_rows = rows
        summaries[name] = summarize(rows, parent_rows)
        infos[name] = info
        print(json.dumps({"event": "model_done", "model": name, "elapsed_sec": round(time.time() - t0, 2), **summaries[name]}, ensure_ascii=False), flush=True)
    result = {
        "status": "CDI_ENDPOINT_TEMPERATURE_RANK_DONE",
        "created_utc": now(),
        "plan": rel(out_dir / "plan.json"),
        "model_infos": infos,
        "summaries": summaries,
        "interpretation": "Dense endpoint CDI cost should be read as rank/order degradation only to the extent target ranks worsen; NLL-only changes that shrink under the independent temperature are calibration-sensitive. This remains endpoint-only, not measured AoA.",
    }
    out_json = out_dir / "cdi_endpoint_temperature_rank_profile.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Short markdown.
    lines = ["# research CDI endpoint temperature/rank profile\n\n", result["interpretation"] + "\n\n"]
    for name, s in summaries.items():
        lines.append(f"- `{name}`: mean NLL T1={s.get('mean_nll_T1')}, Tfit={s.get('mean_nll_Tfit')}, mean rank={s.get('mean_rank')}, ΔNLL T1={s.get('mean_delta_nll_T1_vs_parent')}, ΔNLL Tfit={s.get('mean_delta_nll_Tfit_vs_parent')}, Δrank={s.get('mean_delta_rank_vs_parent')}, rank improved fraction={s.get('improved_fraction_rank')}\n")
    (out_dir / "cdi_endpoint_temperature_rank_profile.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_dir / "cdi_endpoint_temperature_rank_profile.md")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
