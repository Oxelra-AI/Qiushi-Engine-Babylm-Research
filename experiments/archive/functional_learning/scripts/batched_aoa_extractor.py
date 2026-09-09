#!/usr/bin/env python3
"""research batched AoA surprisal extraction for early-stop trajectories.

The research endpoint jobs used the public BabyLM AoA extractor literally on CPU and
timed out after only part of one endpoint.  This script preserves the same raw AoA
representation and MLM input semantics, but batches masked-token variants across
contexts and supports word-index shards.  It can be used for candidate endpoints or
for the shared ancestral ladder; merged shards produce the same `surprisal.json`
shape expected by the research assembly/scoring code and research scorer.

The scientific use is evaluation repair: finish truthful AoA measurement without
changing the AoA scorer or endpoint ancestry.  Any measured AoA still requires all
expected word/context/step rows plus platform scoring evidence.
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
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoProcessor, AutoTokenizer

SCRIPT = _public_path('experiments/archive/functional_learning/scripts/batched_aoa_extractor.py')
ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
STRICT = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict')
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))

from evaluation_pipeline.AoA_word.eval_util import JsonProcessor, load_eval  # noqa: E402

CDI_WORDS_PATH = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json')
CDI_HUMAN_PATH = _public_path('experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv')
A02 = _public_path('experiments/archive/frontier_consolidation')
LADDER = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model')
EARLY_STOP_NAMES = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{i*10}M" for i in range(1, 9)]
EARLY_STOP_WORDS = [i * 1_000_000 for i in range(1, 10)] + [10 * i * 1_000_000 for i in range(1, 9)]
ENDPOINTS: dict[str, dict[str, Any]] = {
    "coherent86": {
        "path": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_coherent86_alpha075'),
        "words": 86_005_295,
    },
    "dense_seed62064": {
        "path": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62064_u0080'),
        "words": 89_168_037,
    },
    "dense_seed62065": {
        "path": _public_path('experiments/archive/functional_learning/data/automodel_repair/repaired_dense_seed62065_u0080'),
        "words": 89_168_037,
    },
}
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/batched_aoa')


def rel(path: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_file(path: pathlib.Path, block_size: int = 1 << 20) -> str | None:
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


def checkpoint_info(label: str, path: pathlib.Path) -> dict[str, Any]:
    cfg_path = path / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    weight = None
    for name in ["model.safetensors", "pytorch_model.bin"]:
        p = path / name
        if p.is_file():
            weight = p
            break
    return {
        "label": label,
        "path": rel(path),
        "exists": path.is_dir(),
        "config_sha256": sha256_file(cfg_path),
        "weight_file": weight.name if weight else None,
        "weight_sha256": sha256_file(weight) if weight else None,
        "architectures": cfg.get("architectures"),
        "model_type": cfg.get("model_type"),
        "auto_map": cfg.get("auto_map"),
        "adapter_scale": cfg.get("adapter_scale"),
        "private_adapter_scale": cfg.get("private_adapter_scale"),
    }


def endpoint_step_name(target: str) -> str:
    return f"endpoint_{int(ENDPOINTS[target]['words']) / 1_000_000:.6f}M"


def load_words(word_start: int, word_end: int, max_words: int) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    target_words, contexts = load_eval(CDI_WORDS_PATH, min_context=0, debug=False)
    total_words = len(target_words)
    total_contexts = sum(len(x) for x in contexts)
    start = max(0, int(word_start))
    if max_words and max_words > 0:
        end = start + int(max_words)
    elif word_end and word_end > 0:
        end = int(word_end)
    else:
        end = total_words
    end = min(end, total_words)
    words = target_words[start:end]
    ctxs = contexts[start:end]
    return words, ctxs, {
        "cdi_words_path": rel(CDI_WORDS_PATH),
        "cdi_words_sha256": sha256_file(CDI_WORDS_PATH),
        "cdi_human_path": rel(CDI_HUMAN_PATH),
        "cdi_human_sha256": sha256_file(CDI_HUMAN_PATH),
        "total_words_available": total_words,
        "total_contexts_available": total_contexts,
        "word_start": start,
        "word_end": end,
        "words_evaluated": len(words),
        "contexts_evaluated": sum(len(c) for c in ctxs),
        "target_words_preview": words[:12],
    }


def prepare_mlm_variants(processor, tokenizer, context: str, target_word: str, use_bos_only: bool) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    # Mirrors StepSurprisalExtractor.compute_surprisal/process_mlm_input.
    context = context.strip() + " "
    target_word = target_word.strip()
    if use_bos_only:
        bos_token = tokenizer.bos_token
        input_text = str(bos_token) + target_word
    else:
        input_text = context + target_word
    mask_index = tokenizer.mask_token_id
    tokenizer_output = processor(text=input_text, return_offsets_mapping=True, add_special_tokens=not use_bos_only)
    tokens = list(tokenizer_output["input_ids"])
    attention_mask = list(tokenizer_output["attention_mask"])
    offsets = list(tokenizer_output["offset_mapping"])
    start_char_idx = len(input_text) - len(target_word)
    phrase_indices: list[int] = []
    target_tokens: list[int] = []
    for i, (start, end) in enumerate(offsets):
        if int(end) > start_char_idx:
            phrase_indices.append(int(i))
            target_tokens.append(int(tokens[i]))
    processed_tokens: list[list[int]] = []
    processed_attention_masks: list[list[int]] = []
    for phrase_index in phrase_indices:
        curr_tokens = list(tokens)
        curr_tokens[phrase_index] = int(mask_index)
        processed_tokens.append(curr_tokens)
        processed_attention_masks.append(list(attention_mask))
    return processed_tokens, processed_attention_masks, phrase_indices, target_tokens


def flush_batch(model, batch: list[dict[str, Any]], pad_id: int, device: str, pair_sums: defaultdict[tuple[int, int], float]) -> None:
    if not batch:
        return
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    indices = torch.empty((len(batch),), dtype=torch.long)
    targets = torch.empty((len(batch),), dtype=torch.long)
    keys: list[tuple[int, int]] = []
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        att = item["attention_mask"]
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = torch.tensor(att, dtype=torch.long)
        indices[i] = int(item["index"])
        targets[i] = int(item["target"])
        keys.append(item["key"])
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    indices = indices.to(device)
    targets = targets.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out[0] if isinstance(out, tuple) else out["logits"]
        row_idx = torch.arange(logits.shape[0], device=logits.device)
        masked_logits = logits[row_idx, indices]
        log_probs = F.log_softmax(masked_logits, dim=-1)
        vals = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()
    for key, lp in zip(keys, vals, strict=False):
        pair_sums[key] += -float(lp)


def extract_step(model_path: pathlib.Path, step: str, word_count: int, target_words: list[str], contexts: list[list[str]], device: str, batch_size: int, use_bos_only: bool, progress_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    t0 = time.time()
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, padding_side="right", local_files_only=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pair_sums: defaultdict[tuple[int, int], float] = defaultdict(float)
    pair_has_variant: set[tuple[int, int]] = set()
    batch: list[dict[str, Any]] = []
    tokenized_pairs = 0
    variants_total = 0
    progress = []
    for word_i, (target_word, word_contexts) in enumerate(zip(target_words, contexts, strict=False)):
        for context_i, context in enumerate(word_contexts):
            key = (word_i, context_i)
            try:
                toks, atts, phrase_indices, target_tokens = prepare_mlm_variants(processor, tokenizer, context, target_word, use_bos_only)
            except Exception:
                toks, atts, phrase_indices, target_tokens = [], [], [], []
            if toks:
                pair_has_variant.add(key)
            for ids, att, idx, target in zip(toks, atts, phrase_indices, target_tokens, strict=False):
                batch.append({"input_ids": ids, "attention_mask": att, "index": int(idx), "target": int(target), "key": key})
                variants_total += 1
                if len(batch) >= int(batch_size):
                    flush_batch(model, batch, pad_id, device, pair_sums)
                    batch.clear()
            tokenized_pairs += 1
        if progress_every and ((word_i + 1) % int(progress_every) == 0 or word_i + 1 == len(target_words)):
            rec = {"event": "batched_aoa_progress", "step": step, "words_done": word_i + 1, "elapsed_sec": round(time.time() - t0, 2), "variants": variants_total}
            progress.append(rec)
            print(json.dumps(rec), flush=True)
    flush_batch(model, batch, pad_id, device, pair_sums)
    results: list[dict[str, Any]] = []
    nan_count = 0
    for word_i, (target_word, word_contexts) in enumerate(zip(target_words, contexts, strict=False)):
        for context_i, context in enumerate(word_contexts):
            key = (word_i, context_i)
            if key in pair_has_variant:
                surprisal = float(pair_sums[key])
            else:
                surprisal = float("nan")
                nan_count += 1
            results.append({
                "step": step,
                "word_count": int(word_count),
                "target_word": target_word,
                "context_id": int(context_i),
                "context": "BOS_ONLY" if use_bos_only else context,
                "surprisal": surprisal,
            })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elapsed = time.time() - t0
    return results, {
        "step": step,
        "model_path": rel(model_path),
        "device": device,
        "elapsed_sec": elapsed,
        "n_results": len(results),
        "n_nan_or_missing": nan_count,
        "tokenized_pairs": tokenized_pairs,
        "masked_variants": variants_total,
        "batch_size": int(batch_size),
        "progress": progress,
        "checkpoint_provenance": checkpoint_info(step, model_path),
    }


def summarize_results(results: list[dict[str, Any]], expected_steps: list[str], expected_contexts: int) -> dict[str, Any]:
    by_step = Counter(r.get("step") for r in results)
    finite = 0
    nonfinite = 0
    for r in results:
        try:
            v = float(r.get("surprisal"))
            if math.isfinite(v):
                finite += 1
            else:
                nonfinite += 1
        except Exception:
            nonfinite += 1
    return {
        "n_results": len(results),
        "n_finite": finite,
        "n_nan_or_nonfinite": nonfinite,
        "counts_by_step": dict(by_step),
        "missing_steps": [s for s in expected_steps if by_step.get(s, 0) == 0],
        "steps_with_wrong_counts": {s: by_step.get(s, 0) for s in expected_steps if by_step.get(s, 0) != expected_contexts},
        "first_result": results[0] if results else None,
        "last_result": results[-1] if results else None,
    }


def _cache_path_writable(path_text: str) -> bool:
    try:
        p = pathlib.Path(os.path.expandvars(path_text)).expanduser()
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".qiushi_write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _ensure_writable_cache_env(key: str, default_path: pathlib.Path) -> None:
    cur = os.environ.get(key)
    if cur and _cache_path_writable(cur):
        os.environ[key] = str(pathlib.Path(os.path.expandvars(cur)).expanduser().resolve())
        return
    default_path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(default_path.resolve())


def run_extract(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Dynamic remote-code models need a writable module cache. Some invocations
    # inherit read-only global HF cache paths; override only cache paths that are
    # absent or not writable, so explicit writable caller caches are preserved.
    cache_root = args.out_dir / "hf_cache"
    _ensure_writable_cache_env("HF_HOME", cache_root / "hf_home")
    _ensure_writable_cache_env("TRANSFORMERS_CACHE", cache_root / "transformers")
    _ensure_writable_cache_env("HF_MODULES_CACHE", cache_root / "modules")
    _ensure_writable_cache_env("HF_HUB_CACHE", cache_root / "hf_home" / "hub")
    _ensure_writable_cache_env("HUGGINGFACE_HUB_CACHE", cache_root / "hf_home" / "hub")
    _ensure_writable_cache_env("HF_DATASETS_CACHE", cache_root / "datasets")
    _ensure_writable_cache_env("TMPDIR", cache_root / "tmp")
    # Transformers may have imported cache constants before this function changed the
    # environment. Keep those module globals aligned with the writable paths; otherwise
    # remote-code loading can still try the read-only inherited cache.
    try:
        import transformers.utils.hub as _hf_hub
        import transformers.dynamic_module_utils as _hf_dyn
        _hf_hub.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
        _hf_dyn.HF_MODULES_CACHE = os.environ["HF_MODULES_CACHE"]
        if hasattr(_hf_hub, "TRANSFORMERS_CACHE"):
            _hf_hub.TRANSFORMERS_CACHE = os.environ["TRANSFORMERS_CACHE"]
        if hasattr(_hf_dyn, "TRANSFORMERS_CACHE"):
            _hf_dyn.TRANSFORMERS_CACHE = os.environ["TRANSFORMERS_CACHE"]
    except Exception as e:
        print(json.dumps({"event": "cache_constant_sync_failed", "error": repr(e)}), file=sys.stderr, flush=True)
    if args.device:
        device = args.device
    else:
        device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    target_words, contexts, eval_info = load_words(args.word_start, args.word_end, args.max_words)
    if args.mode == "endpoint":
        step = endpoint_step_name(args.target)
        step_paths = [(step, ENDPOINTS[args.target]["path"], int(ENDPOINTS[args.target]["words"]))]
        model_name = f"batched_endpoint_{args.target}"
    elif args.mode == "shared":
        n = args.ancestral_limit if args.ancestral_limit and int(args.ancestral_limit) > 0 else len(EARLY_STOP_NAMES)
        step_paths = [(s, LADDER / s, int(w)) for s, w in zip(EARLY_STOP_NAMES[:n], EARLY_STOP_WORDS[:n], strict=False)]
        model_name = "batched_shared_ancestry"
    else:
        raise ValueError(args.mode)
    all_results: list[dict[str, Any]] = []
    step_manifests: list[dict[str, Any]] = []
    for step, path, wc in step_paths:
        res, man = extract_step(path, step, wc, target_words, contexts, device, int(args.batch_size), bool(args.use_bos_only), int(args.progress_every))
        all_results.extend(res)
        step_manifests.append(man)
    out = {
        "metadata": {
            "model_name": model_name,
            "use_bos_only": bool(args.use_bos_only),
            "total_steps": len(step_paths),
            "completed_steps": len({r["step"] for r in all_results}),
            "batched_extractor": rel(SCRIPT),
            "schema_goal": "Same raw AoA surprisal representation consumed by BabyLM AoA scoring and research assembly.",
        },
        "results": all_results,
    }
    surp_path = args.out_dir / "surprisal.json"
    JsonProcessor.save_json(out, surp_path)
    expected_contexts = sum(len(c) for c in contexts)
    expected_steps = [s for s, _, _ in step_paths]
    summary = summarize_results(all_results, expected_steps, expected_contexts)
    manifest = {
        "status": "BATCHED_AOA_EXTRACTION_COMPLETE" if not summary["missing_steps"] and not summary["steps_with_wrong_counts"] else "BATCHED_AOA_EXTRACTION_INCOMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": rel(SCRIPT),
        "mode": args.mode,
        "target": args.target if args.mode == "endpoint" else None,
        "out_dir": rel(args.out_dir),
        "surprisal_path": rel(surp_path),
        "device": device,
        "batch_size": int(args.batch_size),
        "word_range": {"start": int(eval_info["word_start"]), "end": int(eval_info["word_end"]), "max_words_arg": int(args.max_words)},
        "eval_info": eval_info,
        "steps": expected_steps,
        "word_counts": [int(w) for _, _, w in step_paths],
        "expected_results": len(expected_steps) * expected_contexts,
        "summary": summary,
        "step_manifests": step_manifests,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    write_md(args.out_dir / "manifest.md", manifest)
    print(json.dumps({"status": manifest["status"], "out_dir": rel(args.out_dir), "surprisal_path": rel(surp_path), "n_results": summary["n_results"], "expected_results": manifest["expected_results"], "device": device}, indent=2), flush=True)
    return manifest


def write_md(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    lines = ["# research batched AoA extraction\n"]
    lines.append(f"Status: `{manifest.get('status')}`\n")
    lines.append(f"Mode: `{manifest.get('mode')}` target `{manifest.get('target')}` device `{manifest.get('device')}` batch `{manifest.get('batch_size')}`.\n")
    info = manifest.get("eval_info", {})
    lines.append(f"Words: `{info.get('word_start')}`..`{info.get('word_end')}` ({info.get('words_evaluated')} words), contexts `{info.get('contexts_evaluated')}`.\n")
    summ = manifest.get("summary", {})
    lines.append(f"Results: `{summ.get('n_results')}` expected `{manifest.get('expected_results')}`, finite `{summ.get('n_finite')}`, nonfinite `{summ.get('n_nan_or_nonfinite')}`.\n")
    lines.append(f"Missing steps: `{summ.get('missing_steps')}`; wrong-count steps: `{summ.get('steps_with_wrong_counts')}`.\n")
    lines.append("\nThis file is raw AoA surprisal evidence only. Platform AoA is produced later by research/research scoring on a complete trajectory.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_results(args: argparse.Namespace) -> dict[str, Any]:
    pred = load_json(args.pred)
    ref = load_json(args.ref)
    pred_rows = pred.get("results", pred if isinstance(pred, list) else [])
    ref_rows = ref.get("results", ref if isinstance(ref, list) else [])
    key = lambda r: (r.get("step"), r.get("target_word"), int(r.get("context_id", -1)), r.get("context"))
    pred_map = {key(r): r for r in pred_rows}
    ref_map = {key(r): r for r in ref_rows}
    common = sorted(set(pred_map) & set(ref_map), key=lambda x: (str(x[0]), str(x[1]), x[2], str(x[3])[:40]))
    diffs = []
    max_abs = 0.0
    n_large = 0
    for k in common:
        pv = float(pred_map[k].get("surprisal"))
        rv = float(ref_map[k].get("surprisal"))
        if math.isfinite(pv) and math.isfinite(rv):
            d = abs(pv - rv)
            max_abs = max(max_abs, d)
            if d > float(args.tolerance):
                n_large += 1
                if len(diffs) < 20:
                    diffs.append({"key": k, "pred": pv, "ref": rv, "abs_diff": d})
        elif not (math.isnan(pv) and math.isnan(rv)):
            n_large += 1
            if len(diffs) < 20:
                diffs.append({"key": k, "pred": pv, "ref": rv, "abs_diff": None})
    out = {
        "status": "BATCHED_AOA_COMPARE_PASS" if n_large == 0 and len(pred_rows) == len(ref_rows) else "BATCHED_AOA_COMPARE_DIFF",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pred": rel(args.pred),
        "ref": rel(args.ref),
        "tolerance": float(args.tolerance),
        "n_pred": len(pred_rows),
        "n_ref": len(ref_rows),
        "n_common": len(common),
        "n_missing_in_pred": len(set(ref_map) - set(pred_map)),
        "n_extra_in_pred": len(set(pred_map) - set(ref_map)),
        "max_abs_diff": max_abs,
        "n_diff_gt_tolerance": n_large,
        "examples": diffs,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "batched_aoa_compare.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": out["status"], "out": rel(out_path), "max_abs_diff": max_abs, "n_diff_gt_tolerance": n_large}, indent=2), flush=True)
    if out["status"].endswith("DIFF"):
        sys.exit(1)
    return out


def discover_shards(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in root.glob("**/surprisal.json") if p.is_file())


def merge_shards(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [pathlib.Path(x) for x in args.shards] if args.shards else discover_shards(args.shard_root)
    all_results: list[dict[str, Any]] = []
    manifests = []
    for p in shard_paths:
        obj = load_json(p)
        rows = obj.get("results", [])
        all_results.extend(rows)
        m = p.parent / "manifest.json"
        if m.is_file():
            manifests.append(load_json(m))
    # Stable official-like order: step order, target word order from CDI, context_id.
    all_words, _, _ = load_words(0, 0, 0)
    word_rank = {w: i for i, w in enumerate(all_words)}
    step_order = {s: i for i, s in enumerate(EARLY_STOP_NAMES + [endpoint_step_name(t) for t in ENDPOINTS])}
    all_results.sort(key=lambda r: (step_order.get(str(r.get("step")), 10_000), word_rank.get(str(r.get("target_word")), 10_000), int(r.get("context_id", -1))))
    seen = set()
    duplicates = 0
    deduped = []
    for r in all_results:
        k = (r.get("step"), r.get("target_word"), int(r.get("context_id", -1)), r.get("context"))
        if k in seen:
            duplicates += 1
            continue
        seen.add(k)
        deduped.append(r)
    out = {"metadata": {"model_name": args.model_name, "use_bos_only": bool(args.use_bos_only), "merged_by": rel(SCRIPT), "n_shards": len(shard_paths)}, "results": deduped}
    surp_path = args.out_dir / "surprisal.json"
    JsonProcessor.save_json(out, surp_path)
    by_step = Counter(r.get("step") for r in deduped)
    manifest = {
        "status": "BATCHED_AOA_SHARDS_MERGED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "out_dir": rel(args.out_dir),
        "surprisal_path": rel(surp_path),
        "n_shards": len(shard_paths),
        "shards": [rel(p) for p in shard_paths],
        "duplicates_removed": duplicates,
        "n_results": len(deduped),
        "counts_by_step": dict(by_step),
        "source_manifests": manifests,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "out_dir": rel(args.out_dir), "n_results": len(deduped), "duplicates_removed": duplicates}, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--mode", choices=["endpoint", "shared"], required=True)
    p.add_argument("--target", choices=list(ENDPOINTS.keys()), default="coherent86")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--word-start", type=int, default=0)
    p.add_argument("--word-end", type=int, default=0)
    p.add_argument("--max-words", type=int, default=0)
    p.add_argument("--ancestral-limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--use-bos-only", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)

    c = sub.add_parser("compare")
    c.add_argument("--pred", type=pathlib.Path, required=True)
    c.add_argument("--ref", type=pathlib.Path, required=True)
    c.add_argument("--out-dir", type=pathlib.Path, required=True)
    c.add_argument("--tolerance", type=float, default=1e-5)

    m = sub.add_parser("merge")
    m.add_argument("--shard-root", type=pathlib.Path, default=DEFAULT_OUT)
    m.add_argument("--shards", nargs="*", default=[])
    m.add_argument("--out-dir", type=pathlib.Path, required=True)
    m.add_argument("--model-name", default="merged_batched_aoa")
    m.add_argument("--use-bos-only", action="store_true")

    args = parser.parse_args()
    if args.cmd == "extract":
        run_extract(args)
    elif args.cmd == "compare":
        compare_results(args)
    elif args.cmd == "merge":
        merge_shards(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
