#!/usr/bin/env python3
"""research: support-floor tokenizer factor audit before any H100 launch.

This script is CPU-only.  It hardens the dormant minfreq50 support-floor route by
quantifying two confounds that matter scientifically before treating the next
screen as a tokenizer/representation experiment:

1. target/visibility accounting: how the support-floor tokenizer changes raw
   tokens, visible tokens, visible WWM groups, expected selected groups, and
   expected target tokens on the exact allowed 10M compact-view reinvest pool;
2. initialization topology: whether changing vocab size while reusing the same
   global random seeds leaves non-vocabulary transformer tensors identical.  If
   not, the standard launcher changes tokenizer AND contextual initialization,
   so a better launcher should copy same-shape non-vocab tensors from a research
   16k reference initialization before training.

No model training, no official evaluation text, and no evaluation predictions are
used here.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import gc
import hashlib
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive/compact_experience/scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

POOL_10M = WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
TOKENS = {
    "legal16k": WORKSPACE / "data/compliant_tokenizer",
    "minfreq50_supportfloor": WORKSPACE / "data/supportfloor_tokenizers/legal_byte_bpe_40k_minfreq50",
    "a01_legal40k": USER_ROOT / "experiments/archive/representation_and_objectives/data/tokenizer_support_spectrum/tokenizers/legal_byte_bpe_40k",
}
EXPECTED_TOKENIZER_SHA = {
    "legal16k": "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9",
    "minfreq50_supportfloor": "9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922",
}
OUT_DIR = WORKSPACE / "data/supportfloor_factor_audit"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_of(obj: dict[str, Any]) -> str:
    s = str(obj.get("source", obj.get("row_source_class", "unknown")))
    # Strip epoch/pass prefixes if present; for the 10M pool they are usually absent.
    if "::" in s:
        s = s.split("::", 1)[1]
    return s or "unknown"


def empty_acc() -> dict[str, float]:
    return {
        "rows": 0,
        "declared_words": 0,
        "raw_tokens": 0,
        "visible_tokens": 0,
        "visible_groups": 0,
        "over256_rows": 0,
        "truncated_tokens": 0,
    }


def add_acc(acc: dict[str, float], *, words: int, raw_tokens: int, visible_tokens: int, visible_groups: int) -> None:
    acc["rows"] += 1
    acc["declared_words"] += words
    acc["raw_tokens"] += raw_tokens
    acc["visible_tokens"] += visible_tokens
    acc["visible_groups"] += visible_groups
    if raw_tokens > 256:
        acc["over256_rows"] += 1
        acc["truncated_tokens"] += raw_tokens - 256


def finalize_acc(acc: dict[str, float], mask_prob: float = 0.15) -> dict[str, Any]:
    words = max(1.0, float(acc["declared_words"]))
    out = {k: int(v) for k, v in acc.items()}
    out.update({
        "raw_tokens_per_word": acc["raw_tokens"] / words,
        "visible_tokens_per_word": acc["visible_tokens"] / words,
        "visible_groups_per_word": acc["visible_groups"] / words,
        "expected_selected_groups": mask_prob * acc["visible_groups"],
        "expected_target_tokens": mask_prob * acc["visible_tokens"],
        "expected_selected_groups_per_word": mask_prob * acc["visible_groups"] / words,
        "expected_target_tokens_per_word": mask_prob * acc["visible_tokens"] / words,
        "target_tokens_per_selected_group_expected": (acc["visible_tokens"] / acc["visible_groups"]) if acc["visible_groups"] else None,
    })
    return out


def token_accounting(max_rows: int = 0) -> dict[str, Any]:
    tokenizers = {}
    tok_info = {}
    for label, path in TOKENS.items():
        if not (path / "tokenizer.json").exists():
            continue
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        tokenizers[label] = tok
        sha = sha256_file(path / "tokenizer.json")
        tok_info[label] = {
            "path": str(path),
            "tokenizer_json_sha256": sha,
            "expected_sha_match": EXPECTED_TOKENIZER_SHA.get(label) in (None, sha),
            "vocab_size_property": tok.vocab_size,
            "len_tokenizer": len(tok),
            "special_token_ids": {
                "unk": tok.unk_token_id,
                "bos": tok.bos_token_id,
                "eos": tok.eos_token_id,
                "pad": tok.pad_token_id,
                "mask": tok.mask_token_id,
            },
        }

    by_tok = {label: empty_acc() for label in tokenizers}
    by_tok_source = {label: defaultdict(empty_acc) for label in tokenizers}
    rows = 0
    words_total = 0
    with POOL_10M.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            words = int(obj.get("words", 0)) or len(text.split())
            if words != len(text.split()):
                raise RuntimeError(f"word mismatch at row {rows+1}: field {words}, actual {len(text.split())}")
            src = source_of(obj)
            rows += 1
            words_total += words
            for label, tok in tokenizers.items():
                enc = tok(text, add_special_tokens=False, truncation=False)
                ids = enc["input_ids"]
                raw_n = len(ids)
                # Match the actual COMPACT_EXPERIENCE trainer's WWM grouping, which does NOT use
                # tokenizer.word_ids().  It starts a new group at token strings with
                # byte-level word-start markers (Ġ/▁) or at position 0, and keeps
                # punctuation/suffix pieces inside the current group.  Using
                # word_ids() over-counts pre-tokenized punctuation and is not the
                # training objective's group geometry.
                visible_n = min(raw_n, 256)
                special_ids = set(tok.all_special_ids)
                gid = -1
                for i, tid in enumerate(ids[:visible_n]):
                    if int(tid) in special_ids:
                        continue
                    tstr = tok.convert_ids_to_tokens(int(tid))
                    if gid < 0 or (tstr is not None and (str(tstr).startswith("Ġ") or str(tstr).startswith("▁"))) or i == 0:
                        gid += 1
                visible_g = gid + 1 if gid >= 0 else 0
                add_acc(by_tok[label], words=words, raw_tokens=raw_n, visible_tokens=visible_n, visible_groups=visible_g)
                add_acc(by_tok_source[label][src], words=words, raw_tokens=raw_n, visible_tokens=visible_n, visible_groups=visible_g)
            if max_rows and rows >= max_rows:
                break
            if rows % 10000 == 0:
                print(json.dumps({"event": "token_accounting_progress", "rows": rows, "time_utc": now_utc()}), flush=True)

    summary = {label: finalize_acc(acc) for label, acc in by_tok.items()}
    source_summary = {label: {src: finalize_acc(acc) for src, acc in sorted(srcs.items())} for label, srcs in by_tok_source.items()}
    comparisons: dict[str, Any] = {}
    base_label = "legal16k"
    for label in summary:
        if label == base_label or base_label not in summary:
            continue
        comp: dict[str, Any] = {}
        for key in ["raw_tokens", "visible_tokens", "visible_groups", "over256_rows", "truncated_tokens", "expected_selected_groups", "expected_target_tokens", "raw_tokens_per_word", "visible_tokens_per_word", "visible_groups_per_word", "expected_selected_groups_per_word", "expected_target_tokens_per_word"]:
            comp[f"delta_{label}_minus_{base_label}_{key}"] = summary[label][key] - summary[base_label][key]
        comp["relative_expected_target_tokens"] = summary[label]["expected_target_tokens"] / summary[base_label]["expected_target_tokens"]
        comp["relative_expected_selected_groups"] = summary[label]["expected_selected_groups"] / summary[base_label]["expected_selected_groups"]
        comparisons[f"{label}_minus_{base_label}"] = comp
    return {
        "pool_path": str(POOL_10M),
        "pool_sha256": sha256_file(POOL_10M),
        "pool_sha_matches_expected": sha256_file(POOL_10M) == EXPECTED_POOL_SHA,
        "max_rows": max_rows,
        "rows_scanned": rows,
        "words_scanned": words_total,
        "tokenizers": tok_info,
        "summary": summary,
        "source_summary": source_summary,
        "comparisons": comparisons,
        "accounting_note": "For the shuffled ten-pass 100M stream, each full 10M pass has the same row multiset; these per-pool token/visibility/expected-mask quantities apply to each full pass and to the 80M prefix as eight passes, up to row-order-independent arithmetic.",
    }


def reset_all_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_position_embeddings=512,
        max_seq_length=256,
        deberta_pos_att_type="p2c,c2p",
        hidden_size=480,
        n_layer=8,
        n_head=8,
        ffn_mult=4,
        max_relative_positions=256,
        position_buckets=256,
    )


def build_model_for(tok_path: pathlib.Path, seed: int = 43, extra_init_seed: int = 43022):
    tok = base.make_portable_tokenizer(str(tok_path))
    args = model_args()
    reset_all_rng(seed)
    if extra_init_seed >= 0:
        reset_all_rng(extra_init_seed)
    return base.build_model(args, tok), tok


def tensor_std(x: torch.Tensor) -> float:
    if x.numel() <= 1 or not torch.is_floating_point(x):
        return 0.0
    return float(x.detach().float().std(unbiased=False).cpu())


def compare_state_dicts(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    same_shape = []
    shape_mismatch = []
    random_like = []
    exact_same_shape = 0
    exact_random_like = 0
    same_numel = 0
    exact_numel = 0
    random_numel = 0
    exact_random_numel = 0
    mismatch_examples = []
    for name in sorted(set(a) & set(b)):
        ta, tb = a[name].detach().cpu(), b[name].detach().cpu()
        if tuple(ta.shape) != tuple(tb.shape):
            shape_mismatch.append({"name": name, "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
            continue
        eq = torch.equal(ta, tb)
        max_abs = 0.0 if eq else float((ta.float() - tb.float()).abs().max().item()) if torch.is_floating_point(ta) else None
        rec = {"name": name, "shape": list(ta.shape), "numel": int(ta.numel()), "exact_equal": bool(eq), "max_abs_diff": max_abs, "std_a": tensor_std(ta), "std_b": tensor_std(tb)}
        same_shape.append(rec)
        same_numel += int(ta.numel())
        if eq:
            exact_same_shape += 1
            exact_numel += int(ta.numel())
        is_random_like = (rec["std_a"] or 0.0) > 0.0 or (rec["std_b"] or 0.0) > 0.0
        if is_random_like:
            random_like.append(rec)
            random_numel += int(ta.numel())
            if eq:
                exact_random_like += 1
                exact_random_numel += int(ta.numel())
        if (not eq) and len(mismatch_examples) < 15:
            mismatch_examples.append(rec)
    return {
        "same_shape_tensor_count": len(same_shape),
        "same_shape_exact_tensor_count": exact_same_shape,
        "same_shape_numel": same_numel,
        "same_shape_exact_numel": exact_numel,
        "same_shape_exact_numel_fraction": exact_numel / same_numel if same_numel else None,
        "random_like_same_shape_tensor_count": len(random_like),
        "random_like_exact_tensor_count": exact_random_like,
        "random_like_numel": random_numel,
        "random_like_exact_numel": exact_random_numel,
        "random_like_exact_numel_fraction": exact_random_numel / random_numel if random_numel else None,
        "shape_mismatch_tensor_count": len(shape_mismatch),
        "shape_mismatches_first20": shape_mismatch[:20],
        "mismatch_examples_first15": mismatch_examples,
    }


def init_audit() -> dict[str, Any]:
    model, tok = build_model_for(TOKENS["legal16k"])
    min_model, min_tok = build_model_for(TOKENS["minfreq50_supportfloor"])
    a = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    b = {k: v.detach().cpu().clone() for k, v in min_model.state_dict().items()}
    standard = compare_state_dicts(a, b)
    param_counts = {
        "legal16k": int(sum(p.numel() for p in model.parameters())),
        "minfreq50_supportfloor": int(sum(p.numel() for p in min_model.parameters())),
        "delta_minfreq50_minus_step35": int(sum(p.numel() for p in min_model.parameters()) - sum(p.numel() for p in model.parameters())),
        "vocab": len(tok),
        "minfreq50_vocab": len(min_tok),
    }

    # Simulate an init-matched target by copying same-shape tensors from a 16k reference.
    matched_sd = {k: v.detach().cpu().clone() for k, v in b.items()}
    copied = []
    skipped = []
    for name, ta in a.items():
        if name in matched_sd and tuple(ta.shape) == tuple(matched_sd[name].shape):
            matched_sd[name] = ta.clone()
            copied.append(name)
        elif name in matched_sd:
            skipped.append({"name": name, "shape_ref": list(ta.shape), "shape_target": list(matched_sd[name].shape)})
    matched = compare_state_dicts(a, matched_sd)
    del model, min_model
    gc.collect()
    return {
        "standard_same_seed_audit": standard,
        "param_counts": param_counts,
        "init_matched_copy_simulation": {
            "copied_same_shape_tensor_count": len(copied),
            "skipped_shape_mismatch_count": len(skipped),
            "skipped_shape_mismatches": skipped,
            "postcopy_compare_to_step35": matched,
        },
        "interpretation": [
            "If random_like_exact_numel_fraction is near zero under standard_same_seed_audit, a plain vocab-size change also changes contextual/body initialization through global RNG consumption.",
            "The init-matched copy simulation shows whether a launcher can isolate the support-floor tokenizer by copying all same-shape tensors from a research 16k random reference and leaving only vocab-shaped tensors independently initialized.",
        ],
    }


def write_md(summary: dict[str, Any], path: pathlib.Path) -> None:
    acc = summary["token_accounting"]
    init = summary["initialization_audit"]
    lines = [
        "# research support-floor factor audit",
        "",
        "CPU-only audit before any support-floor H100 launch. No official evaluation text or model training used.",
        "",
        "## Token/target accounting on the allowed 10M pool",
        "",
        f"Pool SHA matched expected: `{acc['pool_sha_matches_expected']}`. Rows/words scanned: `{acc['rows_scanned']}` / `{acc['words_scanned']}`.",
        "",
        "| tokenizer | vocab | raw tok/word | visible tok/word | visible groups/word | over256 rows | trunc toks | expected selected groups/word | expected target toks/word |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, rec in acc["summary"].items():
        vocab = acc["tokenizers"][label]["len_tokenizer"]
        lines.append(
            f"| {label} | {vocab} | {rec['raw_tokens_per_word']:.6f} | {rec['visible_tokens_per_word']:.6f} | {rec['visible_groups_per_word']:.6f} | {rec['over256_rows']} | {rec['truncated_tokens']} | {rec['expected_selected_groups_per_word']:.6f} | {rec['expected_target_tokens_per_word']:.6f} |"
        )
    lines += ["", "### Deltas versus research legal16k", ""]
    for label, comp in acc["comparisons"].items():
        lines += [
            f"- `{label}`: Δ raw tok/word `{comp.get('delta_' + label.split('_minus_')[0] + '_minus_step35_legal16k_raw_tokens_per_word', float('nan')):.6f}`; "
            f"Δ visible tok/word `{comp.get('delta_' + label.split('_minus_')[0] + '_minus_step35_legal16k_visible_tokens_per_word', float('nan')):.6f}`; "
            f"Δ visible groups/word `{comp.get('delta_' + label.split('_minus_')[0] + '_minus_step35_legal16k_visible_groups_per_word', float('nan')):.6f}`; "
            f"relative expected target tokens `{comp['relative_expected_target_tokens']:.6f}`; relative selected groups `{comp['relative_expected_selected_groups']:.6f}`.",
        ]
    lines += [
        "",
        "## Initialization topology",
        "",
        f"Parameter counts: research `{init['param_counts']['legal16k']}`, minfreq50 `{init['param_counts']['minfreq50_supportfloor']}`, delta `{init['param_counts']['delta_minfreq50_minus_step35']}`.",
        "",
        "### Standard same-seed build (what the existing dormant launcher would do)",
        "",
    ]
    st = init["standard_same_seed_audit"]
    lines += [
        f"- Same-shape tensors exact: `{st['same_shape_exact_tensor_count']}/{st['same_shape_tensor_count']}`; same-shape exact numel fraction `{st['same_shape_exact_numel_fraction']:.6f}`.",
        f"- Random-like same-shape tensors exact: `{st['random_like_exact_tensor_count']}/{st['random_like_same_shape_tensor_count']}`; random-like exact numel fraction `{st['random_like_exact_numel_fraction']:.6f}`.",
        f"- Shape-mismatch tensors (vocab-dependent): `{st['shape_mismatch_tensor_count']}`.",
        "",
        "First standard-build mismatches:",
    ]
    for rec in st["mismatch_examples_first15"][:8]:
        lines.append(f"- `{rec['name']}` shape={rec['shape']} max_abs={rec['max_abs_diff']}")
    mt = init["init_matched_copy_simulation"]["postcopy_compare_to_step35"]
    lines += [
        "",
        "### Init-matched copy simulation",
        "",
        f"- Copied same-shape tensors: `{init['init_matched_copy_simulation']['copied_same_shape_tensor_count']}`; skipped vocab-shaped tensors: `{init['init_matched_copy_simulation']['skipped_shape_mismatch_count']}`.",
        f"- After copying, random-like same-shape exact numel fraction `{mt['random_like_exact_numel_fraction']:.6f}`.",
        "",
        "## Scientific reading",
        "",
    ]
    if st["random_like_exact_numel_fraction"] is not None and st["random_like_exact_numel_fraction"] < 0.99:
        lines.append("- Plain minfreq50 training would not isolate tokenizer/support-floor alone: it also changes the contextual random initialization because the larger embedding/output tensors consume a different random stream.")
        lines.append("- A better fallback launcher should use an init-matched wrapper that copies all same-shape tensors from a research 16k random reference into the minfreq50 model, leaving only vocab-shaped tensors independently initialized.")
    else:
        lines.append("- Standard same-seed initialization appears to preserve non-vocab tensors; the existing launcher is adequate on this confound.")
    lines += [
        "- Minfreq50 should still be read as a representation package: it changes segmentation, expected target-token burden, visible groups, and embedding/output rows; this audit quantifies those changes before any score is interpreted.",
        "",
        f"Full JSON: `{OUT_DIR / 'supportfloor_factor_audit.json'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="0 = full 10M pool")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "SUPPORTFLOOR_FACTOR_AUDIT",
        "created_utc": now_utc(),
        "scope": "CPU-only training-pool accounting and initialization audit for the dormant minfreq50 support-floor route; no training/evaluation text/predictions",
        "token_accounting": token_accounting(max_rows=args.max_rows),
        "initialization_audit": init_audit(),
    }
    out_json = OUT_DIR / "supportfloor_factor_audit.json"
    out_md = OUT_DIR / "supportfloor_factor_audit.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(summary, out_md)
    print(json.dumps({"status": summary["status"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
