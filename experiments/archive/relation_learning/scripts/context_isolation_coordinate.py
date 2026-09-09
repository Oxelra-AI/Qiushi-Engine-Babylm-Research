#!/usr/bin/env python3
"""research: context-vs-isolation MLM coordinate on Strict-complement sentences.

The recent substitution interpretation needs a checkpoint-computable coordinate for
whether a relation intervention shifts prediction toward using adjacent text.  The
research Strict-complement axis was previously scored as whole 160-word rows: masked
tokens were predicted with the rest of the row visible.  Here we split those rows
into sentence-like spans and score the same spans in two modes:

  * row_context: the full 160-word row is visible, but only target-sentence tokens
    are masked/scored.
  * isolation: the sentence span alone is visible and masked/scored.

The primary coordinate is context_gain = isolation_loss - row_context_loss.  Larger
context_gain means the checkpoint benefits more from adjacent row context when
predicting the sentence.  Contrasts of context_gain across matched arms test whether
correct correspondence, wrong correspondence, exact recurrence, splitting, or
marginal restatement dose shifts prediction toward or away from context-supported
computation, rather than only changing unconditional text fit.

This is a research-facing scorer, not an official BabyLM evaluation.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from itertools import combinations
from typing import Any

# Resolve the repository root before importing transformers so that writable dynamic-module
# caches can be installed early enough for trusted adapter checkpoints.
ROOT0 = _public_path('experiments/archive/relation_learning/scripts/context_isolation_coordinate.py')
ROOT = _PUBLIC_ROOT
WS = ROOT / "experiments/archive/relation_learning"

def _preimport_cache() -> None:
    base = pathlib.Path(os.environ.get("CONTEXT_HF_CACHE", str(WS / "data/context_isolation_hf_cache")))
    env_map = {
        "HF_HOME": base / "hf_home",
        "HF_HUB_CACHE": base / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": base / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": base / "transformers",
        "HF_MODULES_CACHE": base / "modules",
        "HF_DATASETS_CACHE": base / "datasets",
        "TMPDIR": base / "tmp",
    }
    for k, v in env_map.items():
        os.environ.setdefault(k, str(v))
    for v in env_map.values():
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)

_preimport_cache()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

COMPACT_EXPERIENCE_RUNS = ROOT / "experiments/archive/compact_experience/training/runs"
REPRESENTATION_FRONTIER_STUDIES_RUNS = ROOT / "experiments/archive/frontier_consolidation/training/runs"
FUNCTIONAL_RELATION_STUDIES_RUNS = ROOT / "experiments/archive/relation_learning/training/runs"
STRICT_AXIS = WS / "data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl"
OUT_DEFAULT = WS / "data/context_isolation_coordinate"
MAX_LEN = 256
MASK_PROB = 0.15

# COMPACT_EXPERIENCE relation text/pairing controls.  These are stock checkpoints.
COMPACT_EXPERIENCE_ARMS: dict[str, dict[str, Any]] = {
    "OFF43022": {"family": "compact_experience", "relation": "OFF", "seed": 43022, "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43022", "trust_remote_code": False, "description": "official lengthmatched, no qwen pair practice"},
    "ALN43022": {"family": "compact_experience", "relation": "ALN", "seed": 43022, "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43022", "trust_remote_code": False, "description": "original plus own rewrite in same window"},
    "SHUF43022": {"family": "compact_experience", "relation": "SHUF", "seed": 43022, "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_shuffled_control_16k_seed43022", "trust_remote_code": False, "description": "original plus wrong rewrite in same window"},
    "SEP43022": {"family": "compact_experience", "relation": "SEP", "seed": 43022, "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_separated_pair_16k_seed43022", "trust_remote_code": False, "description": "same originals/rewrites separated across rows"},
    "DUP43022": {"family": "compact_experience", "relation": "DUP", "seed": 43022, "run_dir": COMPACT_EXPERIENCE_RUNS / "selected_original_dup_all_16k_seed43022", "trust_remote_code": False, "description": "selected original duplicated locally"},
    "OFF43122": {"family": "compact_experience", "relation": "OFF", "seed": 43122, "run_dir": COMPACT_EXPERIENCE_RUNS / "official_lengthmatched_16k_seed43122", "trust_remote_code": False, "description": "official lengthmatched seed43122"},
    "ALN43122": {"family": "compact_experience", "relation": "ALN", "seed": 43122, "run_dir": COMPACT_EXPERIENCE_RUNS / "qwen_clean_aligned_16k_seed43122", "trust_remote_code": False, "description": "aligned seed43122"},
    "SHUF43122": {"family": "compact_experience", "relation": "SHUF", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "qwen_shuffled_control_16k_seed43122", "trust_remote_code": False, "description": "shuffled seed43122 later replication run"},
    "DUP43122": {"family": "compact_experience", "relation": "DUP", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "selected_original_dup_all_16k_seed43122", "trust_remote_code": False, "description": "duplicated-original seed43122 later replication run"},
}

# REPRESENTATION_FRONTIER_STUDIES C/R/V relation-locality family.  These are stock checkpoints.
CRV_ARMS: dict[str, dict[str, Any]] = {
    "C43022": {"family": "crv", "relation": "CLEAN", "seed": 43022, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43022", "trust_remote_code": False, "description": "CLEAN matched rowholdout"},
    "R43022": {"family": "crv", "relation": "REPEAT", "seed": 43022, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43022", "trust_remote_code": False, "description": "exact recurrence local"},
    "V43022": {"family": "crv", "relation": "VIEW", "seed": 43022, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43022", "trust_remote_code": False, "description": "nonidentical restatement local"},
    "RS43022": {"family": "crv", "relation": "REPEAT_SPLIT", "seed": 43022, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43022", "trust_remote_code": False, "description": "exact recurrence split across rows"},
    "VS43022": {"family": "crv", "relation": "VIEW_SPLIT", "seed": 43022, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43022", "trust_remote_code": False, "description": "restatement split across rows"},
    "C43122": {"family": "crv", "relation": "CLEAN", "seed": 43122, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43122", "trust_remote_code": False, "description": "CLEAN seed43122"},
    "R43122": {"family": "crv", "relation": "REPEAT", "seed": 43122, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43122", "trust_remote_code": False, "description": "exact recurrence seed43122"},
    "V43122": {"family": "crv", "relation": "VIEW", "seed": 43122, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43122", "trust_remote_code": False, "description": "restatement seed43122"},
    "RS43122": {"family": "crv", "relation": "REPEAT_SPLIT", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_split_dose2p64x_rowholdout_deberta100M_seed43122", "trust_remote_code": False, "description": "exact recurrence split seed43122"},
    "VS43122": {"family": "crv", "relation": "VIEW_SPLIT", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_split_dose2p64x_rowholdout_deberta100M_seed43122", "trust_remote_code": False, "description": "restatement split seed43122"},
    "C43222": {"family": "crv", "relation": "CLEAN", "seed": 43222, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_clean_dose2p64x_matched_rowholdout_deberta100M_seed43222_parallel", "trust_remote_code": False, "description": "CLEAN third seed"},
    "R43222": {"family": "crv", "relation": "REPEAT", "seed": 43222, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_repeat_dose2p64x_matched_rowholdout_deberta100M_seed43222", "trust_remote_code": False, "description": "exact recurrence third seed"},
    "V43222": {"family": "crv", "relation": "VIEW", "seed": 43222, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "full_p2c_c2p_abs_view_dose2p64x_matched_rowholdout_deberta100M_seed43222", "trust_remote_code": False, "description": "restatement third seed"},
}

# Adapter-scaled base/dose family.  These require trusted loading.
DOSE_ARMS: dict[str, dict[str, Any]] = {
    "base43022": {"family": "dose", "relation": "base0", "seed": 43022, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_h100M100M_seed43022_official_ladder", "trust_remote_code": True, "description": "compact-view-reinvest base seed43022"},
    "dose21_43022": {"family": "dose", "relation": "dose21", "seed": 43022, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose21_seed43022", "trust_remote_code": True, "description": "probe-clean nested restatement dose21 seed43022"},
    "dose25_43022": {"family": "dose", "relation": "dose25", "seed": 43022, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43022", "trust_remote_code": True, "description": "probe-clean nested restatement dose25 seed43022"},
    "base43122": {"family": "dose", "relation": "base0", "seed": 43122, "run_dir": REPRESENTATION_FRONTIER_STUDIES_RUNS / "adapter128_scale1p75_seed43122_dense100M", "trust_remote_code": True, "description": "compact-view-reinvest base seed43122"},
    "dose21_43122": {"family": "dose", "relation": "dose21", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose21_seed43122", "trust_remote_code": True, "description": "probe-clean nested restatement dose21 seed43122"},
    "dose25_43122": {"family": "dose", "relation": "dose25", "seed": 43122, "run_dir": FUNCTIONAL_RELATION_STUDIES_RUNS / "probe_clean_restatement_dose25_seed43122", "trust_remote_code": True, "description": "probe-clean nested restatement dose25 seed43122"},
}

ALL_ARMS: dict[str, dict[str, Any]] = {}
ALL_ARMS.update(COMPACT_EXPERIENCE_ARMS)
ALL_ARMS.update(CRV_ARMS)
ALL_ARMS.update(DOSE_ARMS)
ARM_SETS = {
    "compact_experience": list(COMPACT_EXPERIENCE_ARMS),
    "crv": list(CRV_ARMS),
    "crv2": ["C43022", "R43022", "V43022", "RS43022", "VS43022", "C43122", "R43122", "V43122", "RS43122", "VS43122"],
    "dose": list(DOSE_ARMS),
    "compact_experience_43022": ["OFF43022", "ALN43022", "SHUF43022", "SEP43022", "DUP43022"],
    "compact_experience_43122": ["OFF43122", "ALN43122", "SHUF43122", "DUP43122"],
    "dose43022": ["base43022", "dose21_43022", "dose25_43022"],
    "dose43122": ["base43122", "dose21_43122", "dose25_43122"],
    "all": list(ALL_ARMS),
}
DEFAULT_CKPTS = ["chck_80M", "chck_90M", "chck_100M"]

_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: pathlib.Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def read_jsonl(path: pathlib.Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def stable_u32(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "little") & 0x7fffffff


def model_file_ready(path: pathlib.Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def sentence_like_spans(text: str, min_words: int, max_words: int) -> list[tuple[int, int, str, int]]:
    spans: list[tuple[int, int, str, int]] = []
    start = 0
    parts = []
    for m in _SENT_BOUNDARY.finditer(text):
        end = m.end()
        parts.append((start, end))
        start = end
    if start < len(text):
        parts.append((start, len(text)))
    # If punctuation produced very long spans, split them into word windows while
    # preserving char offsets.  This keeps the isolation condition sentence-like and
    # avoids very long targets where truncation would dominate.
    for st, en in parts:
        raw = text[st:en]
        lead = len(raw) - len(raw.lstrip())
        trail = len(raw.rstrip())
        st2 = st + lead
        en2 = st + trail
        seg = text[st2:en2]
        if not seg:
            continue
        words = list(re.finditer(r"\S+", seg))
        if len(words) < min_words:
            continue
        if len(words) <= max_words:
            spans.append((st2, en2, seg, len(words)))
        else:
            # Split into max_words chunks with a lower bound on the final chunk.
            for i in range(0, len(words), max_words):
                chunk = words[i:i+max_words]
                if len(chunk) < min_words:
                    continue
                cst = st2 + chunk[0].start()
                cen = st2 + chunk[-1].end()
                spans.append((cst, cen, text[cst:cen], len(chunk)))
    return spans


def load_sentence_records(axis_path: pathlib.Path, max_sentences: int, min_words: int, max_words: int, seed: int) -> list[dict[str, Any]]:
    all_recs: list[dict[str, Any]] = []
    for row_i, obj in enumerate(read_jsonl(axis_path)):
        text = str(obj["text"])
        spans = sentence_like_spans(text, min_words=min_words, max_words=max_words)
        for j, (st, en, sent, n_words) in enumerate(spans):
            uid = f"r{int(obj.get('axis_row_index', row_i)):04d}_s{j:02d}"
            all_recs.append({
                "sentence_uid": uid,
                "row_example_id": int(obj.get("example_id", row_i)),
                "axis_row_index": int(obj.get("axis_row_index", row_i)),
                "source": str(obj.get("source", obj.get("origin_file", ""))),
                "row_words": int(obj.get("words", len(text.split()))),
                "sentence_words": int(n_words),
                "char_start": int(st),
                "char_end": int(en),
                "row_text": text,
                "sentence_text": sent,
                "mask_seed": stable_u32(uid),
            })
    if max_sentences and len(all_recs) > max_sentences:
        # Source-balanced deterministic sample.  This prevents a single long-source
        # region from dominating the coordinate and keeps runs cheap enough to repeat.
        rng = random.Random(seed)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in all_recs:
            by_source[r["source"]].append(r)
        sources = sorted(by_source)
        quota = math.ceil(max_sentences / max(1, len(sources)))
        sampled: list[dict[str, Any]] = []
        for src in sources:
            rs = list(by_source[src])
            rng.shuffle(rs)
            sampled.extend(rs[:quota])
        if len(sampled) > max_sentences:
            rng.shuffle(sampled)
            sampled = sampled[:max_sentences]
        all_recs = sorted(sampled, key=lambda r: (r["axis_row_index"], r["char_start"]))
    return all_recs


def special_id_set(tok) -> set[int]:
    out: set[int] = set(int(x) for x in tok.all_special_ids if x is not None)
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"]:
        x = getattr(tok, attr, None)
        if x is not None:
            out.add(int(x))
    return out


def choose_mask_positions(n_target: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    mask = rng.random(n_target) < MASK_PROB
    if n_target > 0 and not bool(mask.any()):
        mask[0] = True
    replace_decisions = rng.random(n_target)
    random_draws = rng.randint(0, 2**31 - 1, size=n_target, dtype=np.int64)
    return mask, replace_decisions, random_draws


def make_masked_batch(tok, recs: list[dict[str, Any]], mode: str, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if mode == "row_context":
        texts = [r["row_text"] for r in recs]
    elif mode == "isolation":
        texts = [r["sentence_text"] for r in recs]
    else:
        raise ValueError(mode)
    enc = tok(texts, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt", return_offsets_mapping=True)
    ids0 = enc["input_ids"]
    att = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    special = special_id_set(tok)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    mask_id = int(tok.mask_token_id)
    mids: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []
    for i, r in enumerate(recs):
        ids = ids0[i].clone()
        labels = ids.clone()
        labels[:] = -100
        if mode == "row_context":
            c0, c1 = int(r["char_start"]), int(r["char_end"])
        else:
            c0, c1 = 0, len(str(r["sentence_text"]))
        target_pos: list[int] = []
        truncated_right = False
        for pos, (a, b) in enumerate(offsets[i].tolist()):
            tid = int(ids[pos])
            if int(att[i, pos]) == 0 or tid == pad_id or tid in special:
                continue
            if b == 0 and a == 0:
                continue
            if mode == "row_context":
                if b > c0 and a < c1:
                    target_pos.append(pos)
                if a < c1 <= b or (a < c1 and pos == MAX_LEN - 1 and int(att[i, pos]) == 1):
                    pass
            else:
                if b > c0 and a < c1:
                    target_pos.append(pos)
        # crude truncation flag: for row context, if the char span reaches beyond the
        # largest non-special token offset seen under MAX_LEN, the target is incomplete.
        max_seen_end = max([int(b) for (a, b) in offsets[i].tolist() if int(b) > 0], default=0)
        if mode == "row_context" and c1 > max_seen_end:
            truncated_right = True
        mask_bool, replace_decisions, random_draws = choose_mask_positions(len(target_pos), int(r["mask_seed"]))
        n_masked = 0
        for j, pos in enumerate(target_pos):
            if not bool(mask_bool[j]):
                continue
            labels[pos] = ids[pos]
            n_masked += 1
            d = float(replace_decisions[j])
            if d < 0.8:
                ids[pos] = mask_id
            elif d < 0.9:
                ids[pos] = int(random_draws[j] % max(1, vocab_size))
            # else keep original token
        mids.append(ids)
        labs.append(labels)
        meta.append({
            "sentence_uid": r["sentence_uid"],
            "source": r["source"],
            "row_example_id": int(r["row_example_id"]),
            "axis_row_index": int(r["axis_row_index"]),
            "sentence_words": int(r["sentence_words"]),
            "row_words": int(r["row_words"]),
            "char_start": int(r["char_start"]),
            "char_end": int(r["char_end"]),
            "mode": mode,
            "n_target_tokens": int(len(target_pos)),
            "n_masked": int(n_masked),
            "target_truncated_right": bool(truncated_right),
        })
    return torch.stack(mids), torch.stack(labs), att, meta


@torch.no_grad()
def score_model(model, tok, sentence_records: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    vocab_size = int(getattr(tok, "vocab_size", len(tok)))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    rows: list[dict[str, Any]] = []
    for mode in ["row_context", "isolation"]:
        for st in range(0, len(sentence_records), batch_size):
            batch = sentence_records[st:st+batch_size]
            mids, labs, att, meta = make_masked_batch(tok, batch, mode, vocab_size)
            mids = mids.to(device)
            labs = labs.to(device)
            att = att.to(device)
            logits = model(input_ids=mids, attention_mask=att).logits.float()
            B, L, V = logits.shape
            losses = loss_fn(logits.view(B * L, V), labs.view(B * L)).view(B, L)
            for i, m in enumerate(meta):
                mpos = labs[i] != -100
                if bool(mpos.any()):
                    loss = float(losses[i][mpos].mean().detach().cpu())
                else:
                    loss = float("nan")
                q = dict(m)
                q["loss"] = loss
                rows.append(q)
    return rows


def model_identity(mp: pathlib.Path, model, trust_remote_code: bool) -> dict[str, Any]:
    cfg_path = mp / "config.json"
    cfg: dict[str, Any] = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    total = sum(int(p.numel()) for p in model.parameters())
    adapter = 0
    for n, p in model.named_parameters():
        if "adapter" in n.lower():
            adapter += int(p.numel())
    return {
        "model_path": rel(mp),
        "loaded_class": type(model).__name__,
        "total_params_loaded": int(total),
        "adapter_params_loaded": int(adapter),
        "model_type": cfg.get("model_type"),
        "auto_map_present": bool(cfg.get("auto_map")),
        "adapter_enabled_config": cfg.get("adapter_enabled"),
        "adapter_scale_config": cfg.get("adapter_scale"),
        "trust_remote_code": bool(trust_remote_code),
        "hf_modules_cache": os.environ.get("HF_MODULES_CACHE"),
    }


def load_tokenizer(run_dir: pathlib.Path):
    for p in [run_dir / "hf_model", run_dir / "hf_model/chck_100M", run_dir / "hf_model/chck_80M"]:
        if p.exists():
            try:
                return AutoTokenizer.from_pretrained(str(p), use_fast=True, local_files_only=True)
            except Exception:
                continue
    return AutoTokenizer.from_pretrained(str(run_dir / "hf_model"), use_fast=True, local_files_only=True)


def load_model(mp: pathlib.Path, trust: bool, device: torch.device):
    kwargs: dict[str, Any] = {"torch_dtype": torch.float32, "local_files_only": True}
    if trust:
        kwargs["trust_remote_code"] = True
    return AutoModelForMaskedLM.from_pretrained(str(mp), **kwargs).eval().to(device)


def summarize_scores(score_rows: list[dict[str, Any]], arms: dict[str, dict[str, Any]]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    df = pd.DataFrame(score_rows)
    if df.empty:
        empty = pd.DataFrame()
        return {"mode_summary": empty, "paired_summary": empty, "selected_contrasts": empty, "source_contrasts": empty}, {}
    mode_summary = df.groupby(["family", "seed", "relation", "arm", "checkpoint", "mode"], dropna=False).agg(
        n=("sentence_uid", "count"),
        mean_loss=("loss", "mean"),
        sd_loss=("loss", "std"),
        mean_target_tokens=("n_target_tokens", "mean"),
        mean_masked=("n_masked", "mean"),
        truncated_frac=("target_truncated_right", "mean"),
    ).reset_index()
    piv = df.pivot_table(index=["family", "seed", "relation", "arm", "checkpoint", "sentence_uid", "source", "sentence_words"], columns="mode", values="loss", aggfunc="mean").reset_index()
    if "row_context" in piv.columns and "isolation" in piv.columns:
        piv["context_gain"] = piv["isolation"] - piv["row_context"]
    else:
        piv["context_gain"] = np.nan
    paired_summary = piv.groupby(["family", "seed", "relation", "arm", "checkpoint"], dropna=False).agg(
        n=("sentence_uid", "count"),
        row_context_loss=("row_context", "mean"),
        isolation_loss=("isolation", "mean"),
        context_gain=("context_gain", "mean"),
        se_context_gain=("context_gain", lambda x: float(np.nanstd(x, ddof=1) / math.sqrt(np.sum(np.isfinite(x)))) if np.sum(np.isfinite(x)) > 1 else float("nan")),
    ).reset_index()

    desired_by_family = {
        "compact_experience": [("ALN", "OFF"), ("SHUF", "OFF"), ("ALN", "SHUF"), ("DUP", "OFF"), ("ALN", "DUP"), ("SEP", "OFF"), ("ALN", "SEP"), ("SHUF", "SEP")],
        "crv": [("VIEW", "CLEAN"), ("REPEAT", "CLEAN"), ("VIEW_SPLIT", "CLEAN"), ("REPEAT_SPLIT", "CLEAN"), ("VIEW", "VIEW_SPLIT"), ("REPEAT", "REPEAT_SPLIT"), ("VIEW", "REPEAT")],
        "dose": [("dose21", "base0"), ("dose25", "base0"), ("dose25", "dose21")],
    }
    contrast_rows: list[dict[str, Any]] = []
    source_contrast_rows: list[dict[str, Any]] = []
    for (family, seed, ck), g in piv.groupby(["family", "seed", "checkpoint"], dropna=False):
        # Relation labels are unique within these family/seed/checkpoint selections except if
        # a caller scores multiple arms of the same relation.  In that case average first.
        rel_df = g.groupby(["relation", "sentence_uid", "source"], dropna=False).agg(
            row_context=("row_context", "mean"),
            isolation=("isolation", "mean"),
            context_gain=("context_gain", "mean"),
        ).reset_index()
        wide = rel_df.pivot(index=["sentence_uid", "source"], columns="relation", values=["row_context", "isolation", "context_gain"])
        desired = desired_by_family.get(str(family), [])
        if not desired:
            rels = sorted(set(rel_df["relation"].astype(str)))
            desired = list(combinations(rels, 2))
        for a, b in desired:
            if ("context_gain", a) not in wide.columns or ("context_gain", b) not in wide.columns:
                continue
            valid = (wide[("context_gain", a)] - wide[("context_gain", b)]).dropna()
            idx = valid.index
            if len(idx) == 0:
                continue
            diffs = {}
            for metric in ["row_context", "isolation", "context_gain"]:
                arr = (wide.loc[idx, (metric, a)] - wide.loc[idx, (metric, b)]).to_numpy(dtype=float)
                diffs[metric] = arr
            row = {
                "family": family, "seed": int(seed), "checkpoint": ck, "contrast": f"{a}-minus-{b}",
                "n_sentences": int(len(idx)),
                "delta_row_context_loss": float(np.nanmean(diffs["row_context"])),
                "delta_isolation_loss": float(np.nanmean(diffs["isolation"])),
                "delta_context_gain": float(np.nanmean(diffs["context_gain"])),
                "se_delta_context_gain": float(np.nanstd(diffs["context_gain"], ddof=1) / math.sqrt(len(idx))) if len(idx) > 1 else float("nan"),
                "fraction_a_higher_context_gain": float(np.nanmean(diffs["context_gain"] > 0)),
            }
            contrast_rows.append(row)
            tmp = pd.DataFrame({"source": [x[1] for x in idx], "d_row": diffs["row_context"], "d_iso": diffs["isolation"], "d_gain": diffs["context_gain"]})
            for src, gg in tmp.groupby("source", dropna=False):
                arr = gg["d_gain"].to_numpy(dtype=float)
                source_contrast_rows.append({
                    "family": family, "seed": int(seed), "checkpoint": ck, "contrast": f"{a}-minus-{b}", "source": src,
                    "n_sentences": int(len(arr)),
                    "delta_row_context_loss": float(gg["d_row"].mean()),
                    "delta_isolation_loss": float(gg["d_iso"].mean()),
                    "delta_context_gain": float(np.nanmean(arr)),
                    "se_delta_context_gain": float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else float("nan"),
                })
    selected = pd.DataFrame(contrast_rows)
    source_contrasts = pd.DataFrame(source_contrast_rows)
    late_rows: list[dict[str, Any]] = []
    if not selected.empty:
        for (family, seed, contrast), g in selected.groupby(["family", "seed", "contrast"], dropna=False):
            late_rows.append({
                "family": family, "seed": int(seed), "checkpoint": "mean_over_scored_checkpoints", "contrast": contrast,
                "n_checkpoint_rows": int(len(g)),
                "mean_delta_row_context_loss": float(g["delta_row_context_loss"].mean()),
                "mean_delta_isolation_loss": float(g["delta_isolation_loss"].mean()),
                "mean_delta_context_gain": float(g["delta_context_gain"].mean()),
            })
    late = pd.DataFrame(late_rows)
    tables = {"mode_summary": mode_summary, "paired_summary": paired_summary, "selected_contrasts": selected, "source_contrasts": source_contrasts, "late_contrasts": late, "paired_rows": piv}
    compact = {
        "selected_contrasts": json.loads(selected.to_json(orient="records")) if not selected.empty else [],
        "late_contrasts": json.loads(late.to_json(orient="records")) if not late.empty else [],
    }
    return tables, compact


def choose_arms(args: argparse.Namespace) -> list[str]:
    arms: list[str] = []
    for aset in args.arm_sets:
        if aset not in ARM_SETS:
            raise SystemExit(f"unknown arm-set {aset}; choices={sorted(ARM_SETS)}")
        arms.extend(ARM_SETS[aset])
    arms.extend(args.arms or [])
    # de-duplicate while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for a in arms:
        if a not in ALL_ARMS:
            raise SystemExit(f"unknown arm {a}; choices include {sorted(ALL_ARMS)[:10]} ...")
        if a not in seen:
            out.append(a); seen.add(a)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--arm-sets", nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--checkpoints", nargs="*", default=["chck_100M"])
    ap.add_argument("--axis", default=str(STRICT_AXIS))
    ap.add_argument("--max-sentences", type=int, default=720)
    ap.add_argument("--min-sentence-words", type=int, default=8)
    ap.add_argument("--max-sentence-words", type=int, default=45)
    ap.add_argument("--sample-seed", type=int, default=7601)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # If the caller did not set caches before launch, keep all cache writes under out_dir
    # from this point onward.  For dynamic modules, shell-level CONTEXT_HF_CACHE
    # is preferred because transformers reads HF_MODULES_CACHE at import time.
    if "CONTEXT_HF_CACHE" not in os.environ:
        cache = out_dir / "hf_cache"
        os.environ["CONTEXT_HF_CACHE"] = str(cache)
        for k, v in {
            "HF_HOME": cache / "hf_home",
            "HF_HUB_CACHE": cache / "hf_home" / "hub",
            "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
            "TRANSFORMERS_CACHE": cache / "transformers",
            "HF_MODULES_CACHE": cache / "modules",
            "HF_DATASETS_CACHE": cache / "datasets",
            "TMPDIR": cache / "tmp",
        }.items():
            os.environ[k] = str(v)
            pathlib.Path(v).mkdir(parents=True, exist_ok=True)

    selected_arms = choose_arms(args)
    if not selected_arms:
        raise SystemExit("no arms selected; pass --arm-sets or --arms")
    axis_path = pathlib.Path(args.axis)
    sentence_records = load_sentence_records(axis_path, max_sentences=args.max_sentences, min_words=args.min_sentence_words, max_words=args.max_sentence_words, seed=args.sample_seed)
    write_jsonl(out_dir / "sentence_records.jsonl", [
        {k: v for k, v in r.items() if k not in {"row_text"}} for r in sentence_records
    ])
    missing: list[str] = []
    arm_plan: dict[str, Any] = {}
    for arm in selected_arms:
        cfg = ALL_ARMS[arm]
        arm_plan[arm] = {k: (rel(v) if isinstance(v, pathlib.Path) else v) for k, v in cfg.items()}
        for ck in args.checkpoints:
            mp = pathlib.Path(cfg["run_dir"]) / "hf_model" / ck
            if not model_file_ready(mp):
                missing.append(rel(mp))
    plan = {
        "status": "CONTEXT_ISOLATION_COORDINATE_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "axis": rel(axis_path),
        "sentence_records": rel(out_dir / "sentence_records.jsonl"),
        "n_sentences": len(sentence_records),
        "source_counts": {str(k): int(v) for k, v in pd.Series([r["source"] for r in sentence_records]).value_counts().sort_index().items()},
        "arms": arm_plan,
        "checkpoints": args.checkpoints,
        "missing": missing,
        "device": args.device,
        "batch_size": args.batch_size,
        "mask_prob": MASK_PROB,
        "hf_modules_cache": os.environ.get("HF_MODULES_CACHE"),
        "scientific_question": "Do relation interventions change context_gain = isolation_loss - row_context_loss on the same Strict-complement sentence spans?",
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(missing[0])

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    all_score_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    ident_rows: list[dict[str, Any]] = []
    for arm in selected_arms:
        cfg = ALL_ARMS[arm]
        run_dir = pathlib.Path(cfg["run_dir"])
        tok = load_tokenizer(run_dir)
        for ck in args.checkpoints:
            mp = run_dir / "hf_model" / ck
            t0 = time.time()
            print(json.dumps({"event": "load", "arm": arm, "checkpoint": ck, "path": rel(mp), "trust_remote_code": bool(cfg.get("trust_remote_code"))}, ensure_ascii=False), flush=True)
            model = load_model(mp, bool(cfg.get("trust_remote_code")), device)
            ident = model_identity(mp, model, bool(cfg.get("trust_remote_code")))
            ident.update({"arm": arm, "family": cfg["family"], "relation": cfg["relation"], "seed": cfg["seed"], "checkpoint": ck})
            ident_rows.append(ident)
            print(json.dumps({"event": "model_identity", **ident}, ensure_ascii=False), flush=True)
            scored = score_model(model, tok, sentence_records, device, args.batch_size)
            for r in scored:
                q = dict(r)
                q.update({
                    "arm": arm,
                    "family": cfg["family"],
                    "relation": cfg["relation"],
                    "seed": cfg["seed"],
                    "checkpoint": ck,
                    "description": cfg.get("description", ""),
                })
                all_score_rows.append(q)
            meta_rows.append({
                "arm": arm, "family": cfg["family"], "relation": cfg["relation"], "seed": cfg["seed"], "checkpoint": ck,
                "n_sentence_records": len(sentence_records),
                "n_scored_mode_rows": len(scored),
                "elapsed_sec": round(time.time() - t0, 2),
                "device": str(device),
                "loaded_class": ident.get("loaded_class"),
                "total_params_loaded": ident.get("total_params_loaded"),
                "adapter_params_loaded": ident.get("adapter_params_loaded"),
            })
            print(json.dumps({"event": "done", **meta_rows[-1]}, ensure_ascii=False), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(out_dir / "context_isolation_scores.csv", all_score_rows)
    write_csv(out_dir / "score_meta.csv", meta_rows)
    with (out_dir / "model_identity_preamble.jsonl").open("w", encoding="utf-8") as f:
        for r in ident_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tables, compact = summarize_scores(all_score_rows, ALL_ARMS)
    for name, df in tables.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / f"{name}.csv", index=False)
    result = {
        "status": "CONTEXT_ISOLATION_COORDINATE_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "axis": rel(axis_path),
        "n_sentences": len(sentence_records),
        "arms": selected_arms,
        "checkpoints": args.checkpoints,
        "rows_scored": len(all_score_rows),
        "outputs": {
            "plan": rel(out_dir / "score_plan.json"),
            "sentence_records": rel(out_dir / "sentence_records.jsonl"),
            "scores": rel(out_dir / "context_isolation_scores.csv"),
            "meta": rel(out_dir / "score_meta.csv"),
            "model_identity_preamble": rel(out_dir / "model_identity_preamble.jsonl"),
            "paired_summary": rel(out_dir / "paired_summary.csv"),
            "selected_contrasts": rel(out_dir / "selected_contrasts.csv"),
            "late_contrasts": rel(out_dir / "late_contrasts.csv"),
            "source_contrasts": rel(out_dir / "source_contrasts.csv"),
            "summary": rel(out_dir / "summary.json"),
            "summary_md": rel(out_dir / "summary.md"),
        },
        "compact_results": compact,
        "interpretation_note": "context_gain = isolation_loss - row_context_loss. A positive A-minus-B delta_context_gain means A benefits more than B from adjacent row context on the same Strict-complement sentence spans; compare row_context and isolation loss deltas to separate in-context fit from context-free fit.",
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# research context-vs-isolation coordinate", "", result["interpretation_note"], "", f"Scored {len(sentence_records)} sentence spans, arms: {', '.join(selected_arms)}, checkpoints: {', '.join(args.checkpoints)}.", ""]
    late = tables.get("late_contrasts")
    if isinstance(late, pd.DataFrame) and not late.empty:
        lines.append("## Mean over scored checkpoints")
        for _, r in late.iterrows():
            lines.append(f"- {r['family']} seed{int(r['seed'])} {r['contrast']}: row_context {float(r['mean_delta_row_context_loss']):+.4f}, isolation {float(r['mean_delta_isolation_loss']):+.4f}, context_gain {float(r['mean_delta_context_gain']):+.4f}")
        lines.append("")
    selected = tables.get("selected_contrasts")
    if isinstance(selected, pd.DataFrame) and not selected.empty:
        lines.append("## Per-checkpoint selected contrasts")
        for _, r in selected.head(80).iterrows():
            lines.append(f"- {r['family']} seed{int(r['seed'])} {r['checkpoint']} {r['contrast']}: d_row {float(r['delta_row_context_loss']):+.4f}, d_iso {float(r['delta_isolation_loss']):+.4f}, d_gain {float(r['delta_context_gain']):+.4f}, n={int(r['n_sentences'])}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "summary": result["outputs"]["summary"], "md": result["outputs"]["summary_md"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
