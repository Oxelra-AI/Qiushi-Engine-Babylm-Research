#!/usr/bin/env python3
"""research: context/isolation coordinate with and without boundary tokens.

research reused the old context/isolation scorer, whose isolation mode included
[CLS]/[SEP]-style boundary tokens.  Coherent-special endpoints were trained on
those tokens, while coherent86 was not; an apparent context-gain movement could
therefore be simple boundary adaptation.  This script rescored the same
Strict-complement sentence coordinate in two input forms:

  * with_special: tokenizer adds special boundary tokens, matching official MLM
    zero-shot input geometry.
  * no_special: tokenizer does not add those tokens; the same row and sentence
    text are scored without boundary-token exposure.

It also records masked-token losses and distances to the nearest text boundary so
endpoint-minus-anchor isolation deltas can be localized to sentence edges versus
interiors.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import importlib.util
import json
import math
import os
import pathlib
import statistics
import time
from typing import Any

import pandas as pd
import torch

ROOT = _public_path('.')
WS = _public_path('experiments/archive/relation_learning')
PATH = _public_path('experiments/archive/relation_learning/scripts/context_isolation_coordinate.py')
AXIS_DEFAULT = _public_path('experiments/archive/relation_learning/data/strict_complement_ngram_axis/strict_complement_ngram_axis_3000_rows.jsonl')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/context_gain_special_token_control')
MAX_LEN = 256
MASK_PROB = 0.15

DEFAULT_ENDPOINTS = {
    "chck82_slow_scale1p75": {
        "path": "experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M",
        "trust_remote_code": True,
        "description": "protected slow-adapter chck82 reference",
    },
    "coherent86_alpha075": {
        "path": "models/frontier",
        "trust_remote_code": True,
        "description": "historical coherent86 alpha0.75 private endpoint",
    },
    "coherent_special_98097_alpha075": {
        "path": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98097/alpha_0.75",
        "trust_remote_code": True,
        "description": "exact coherent suffix trained through research trainer with special tokens, seed98097 alpha0.75",
    },
    "coherent_special_98098_alpha075": {
        "path": "experiments/archive/relation_learning/data/format_replay_corrected/coherent_unsplit_special/seed98098/alpha_0.75",
        "trust_remote_code": True,
        "description": "exact coherent suffix trained through research trainer with special tokens, seed98098 alpha0.75",
    },
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def se(xs: list[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    if len(vals) <= 1:
        return float("nan")
    return statistics.stdev(vals) / math.sqrt(len(vals))


def prepare_cache(out_dir: pathlib.Path) -> None:
    cache = out_dir / "hf_cache"
    env_map = {
        "HF_HOME": cache / "hf_home",
        "HF_HUB_CACHE": cache / "hf_home" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "hf_home" / "hub",
        "TRANSFORMERS_CACHE": cache / "transformers",
        "HF_MODULES_CACHE": cache / "modules",
        "HF_DATASETS_CACHE": cache / "datasets",
        "TMPDIR": cache / "tmp",
    }
    for k, v in env_map.items():
        os.environ[k] = str(v)
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)


def load_step076_module():
    spec = importlib.util.spec_from_file_location("context_isolation_coordinate", PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_endpoint_spec(path: str) -> dict[str, Any]:
    if not path:
        return DEFAULT_ENDPOINTS
    p = pathlib.Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "endpoints" in data:
        data = data["endpoints"]
    out: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[k] = {"path": v, "trust_remote_code": True, "description": ""}
        else:
            out[k] = dict(v)
            out[k].setdefault("trust_remote_code", True)
            out[k].setdefault("description", "")
    return out


def model_file_ready(path: pathlib.Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


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
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def special_id_set(tok) -> set[int]:
    out: set[int] = set(int(x) for x in tok.all_special_ids if x is not None)
    for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "cls_token_id", "sep_token_id", "mask_token_id"]:
        x = getattr(tok, attr, None)
        if x is not None:
            out.add(int(x))
    return out


def choose_mask_positions(n_target: int, seed: int) -> tuple[list[bool], list[float], list[int]]:
    import numpy as np
    rng = np.random.RandomState(seed)
    mask = rng.random(n_target) < MASK_PROB
    if n_target > 0 and not bool(mask.any()):
        mask[0] = True
    replace_decisions = rng.random(n_target).tolist()
    random_draws = rng.randint(0, 2**31 - 1, size=n_target, dtype=np.int64).tolist()
    return [bool(x) for x in mask.tolist()], [float(x) for x in replace_decisions], [int(x) for x in random_draws]


def bin_edge_distance(d: float | int | None) -> str:
    if d is None or not math.isfinite(float(d)):
        return "NA"
    x = int(d)
    if x == 0:
        return "0_edge_token"
    if x == 1:
        return "1"
    if x <= 3:
        return "2-3"
    if x <= 7:
        return "4-7"
    if x <= 15:
        return "8-15"
    return "16+"


def make_masked_batch(tok, recs: list[dict[str, Any]], mode: str, add_special_tokens: bool, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]], list[list[dict[str, Any]]]]:
    if mode == "row_context":
        texts = [r["row_text"] for r in recs]
    elif mode == "isolation":
        texts = [r["sentence_text"] for r in recs]
    else:
        raise ValueError(mode)
    enc = tok(
        texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=bool(add_special_tokens),
    )
    ids0 = enc["input_ids"]
    att = enc["attention_mask"]
    offsets = enc["offset_mapping"]
    special = special_id_set(tok)
    pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    mask_id = int(tok.mask_token_id)
    mids: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    sentence_meta: list[dict[str, Any]] = []
    token_meta_batches: list[list[dict[str, Any]]] = []
    for i, r in enumerate(recs):
        ids = ids0[i].clone()
        labels = ids.clone()
        labels[:] = -100
        if mode == "row_context":
            c0, c1 = int(r["char_start"]), int(r["char_end"])
        else:
            c0, c1 = 0, len(str(r["sentence_text"]))
        # Active content positions in the encoded text, independent of the target span.
        active_content_positions: list[int] = []
        boundary_special_positions: list[int] = []
        for pos, (a, b) in enumerate(offsets[i].tolist()):
            tid = int(ids[pos])
            if int(att[i, pos]) == 0 or tid == pad_id:
                continue
            if tid in special:
                if tid != mask_id:
                    boundary_special_positions.append(pos)
                continue
            if b == 0 and a == 0:
                continue
            active_content_positions.append(pos)
        input_content_index = {pos: j for j, pos in enumerate(active_content_positions)}
        input_content_len = len(active_content_positions)
        target_positions: list[int] = []
        truncated_right = False
        for pos, (a, b) in enumerate(offsets[i].tolist()):
            tid = int(ids[pos])
            if int(att[i, pos]) == 0 or tid == pad_id or tid in special:
                continue
            if b == 0 and a == 0:
                continue
            if b > c0 and a < c1:
                target_positions.append(pos)
        max_seen_end = max([int(b) for (a, b) in offsets[i].tolist() if int(b) > 0], default=0)
        if mode == "row_context" and c1 > max_seen_end:
            truncated_right = True
        mask_bool, replace_decisions, random_draws = choose_mask_positions(len(target_positions), int(r["mask_seed"]))
        masked_token_meta: list[dict[str, Any]] = []
        for j, pos in enumerate(target_positions):
            if not mask_bool[j]:
                continue
            labels[pos] = ids[pos]
            d = float(replace_decisions[j])
            original_tid = int(ids[pos])
            off_a, off_b = offsets[i, pos].tolist()
            token_text = tok.convert_ids_to_tokens([original_tid])[0]
            input_idx = input_content_index.get(pos)
            if input_idx is None or input_content_len <= 0:
                input_edge_distance = None
            else:
                input_edge_distance = int(min(input_idx, input_content_len - 1 - input_idx))
            target_edge_distance = int(min(j, max(0, len(target_positions) - 1 - j))) if target_positions else None
            if boundary_special_positions:
                dist_special = int(min(abs(pos - s) for s in boundary_special_positions))
            else:
                dist_special = None
            masked_token_meta.append({
                "position": int(pos),
                "target_token_ordinal": int(j),
                "token_id": original_tid,
                "token": token_text,
                "char_start_token": int(off_a),
                "char_end_token": int(off_b),
                "input_content_index": None if input_idx is None else int(input_idx),
                "input_content_len": int(input_content_len),
                "input_edge_distance": input_edge_distance,
                "input_edge_bin": bin_edge_distance(input_edge_distance),
                "target_edge_distance": target_edge_distance,
                "target_edge_bin": bin_edge_distance(target_edge_distance),
                "distance_to_boundary_special": dist_special,
                "distance_to_boundary_special_bin": "NA" if dist_special is None else bin_edge_distance(max(0, int(dist_special) - 1)),
            })
            if d < 0.8:
                ids[pos] = mask_id
            elif d < 0.9:
                ids[pos] = int(random_draws[j] % max(1, vocab_size))
        mids.append(ids)
        labs.append(labels)
        sentence_meta.append({
            "sentence_uid": r["sentence_uid"],
            "source": r["source"],
            "row_example_id": int(r["row_example_id"]),
            "axis_row_index": int(r["axis_row_index"]),
            "sentence_words": int(r["sentence_words"]),
            "row_words": int(r["row_words"]),
            "char_start": int(r["char_start"]),
            "char_end": int(r["char_end"]),
            "mode": mode,
            "input_form": "with_special" if add_special_tokens else "no_special",
            "add_special_tokens": bool(add_special_tokens),
            "n_target_tokens": int(len(target_positions)),
            "n_masked": int(len(masked_token_meta)),
            "target_truncated_right": bool(truncated_right),
            "input_content_len": int(input_content_len),
            "min_masked_input_edge_distance": min([m["input_edge_distance"] for m in masked_token_meta if m["input_edge_distance"] is not None], default=None),
            "mean_masked_input_edge_distance": None if not masked_token_meta else float(sum(float(m["input_edge_distance"] or 0) for m in masked_token_meta) / len(masked_token_meta)),
            "min_masked_target_edge_distance": min([m["target_edge_distance"] for m in masked_token_meta if m["target_edge_distance"] is not None], default=None),
            "mean_masked_target_edge_distance": None if not masked_token_meta else float(sum(float(m["target_edge_distance"] or 0) for m in masked_token_meta) / len(masked_token_meta)),
            "masked_target_edge_bins": ";".join(m["target_edge_bin"] for m in masked_token_meta),
        })
        token_meta_batches.append(masked_token_meta)
    return torch.stack(mids), torch.stack(labs), att, sentence_meta, token_meta_batches


@torch.no_grad()
def score_model(model, tok, sentence_records: list[dict[str, Any]], device: torch.device, batch_size: int, add_special_tokens: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vocab_size = int(getattr(tok, "vocab_size", len(tok)))
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    sent_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for mode in ["row_context", "isolation"]:
        for st in range(0, len(sentence_records), batch_size):
            batch = sentence_records[st:st+batch_size]
            mids, labs, att, meta, token_meta_batches = make_masked_batch(tok, batch, mode, add_special_tokens, vocab_size)
            mids = mids.to(device)
            labs = labs.to(device)
            att = att.to(device)
            logits = model(input_ids=mids, attention_mask=att).logits.float()
            bsz, seqlen, vocab = logits.shape
            losses = loss_fn(logits.view(bsz * seqlen, vocab), labs.view(bsz * seqlen)).view(bsz, seqlen)
            for i, m in enumerate(meta):
                mpos = labs[i] != -100
                loss = float(losses[i][mpos].mean().detach().cpu()) if bool(mpos.any()) else float("nan")
                q = dict(m)
                q["loss"] = loss
                sent_rows.append(q)
                for tm in token_meta_batches[i]:
                    pos = int(tm["position"])
                    tr = dict(m)
                    tr.update(tm)
                    tr["token_loss"] = float(losses[i, pos].detach().cpu())
                    token_rows.append(tr)
    return sent_rows, token_rows


def summarize_sentence_rows(score_rows: list[dict[str, Any]], anchors: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.DataFrame(score_rows)
    if df.empty:
        return [], []
    df["target_truncated_right_bool"] = df["target_truncated_right"].astype(str).str.lower().isin(["true", "1"])
    bad = df[(df["mode"] == "row_context") & (df["target_truncated_right_bool"] | (df["n_masked"] >= df["n_target_tokens"]))]
    bad_set = set(zip(bad["endpoint"], bad["input_form"], bad["sentence_uid"]))
    keep = [(e, f, s) not in bad_set for e, f, s in zip(df["endpoint"], df["input_form"], df["sentence_uid"])]
    filt = df.loc[keep].copy()
    piv = filt.pivot_table(index=["endpoint", "input_form", "sentence_uid", "source", "sentence_words"], columns="mode", values="loss", aggfunc="mean").reset_index()
    piv = piv.dropna(subset=["row_context", "isolation"])
    piv["context_gain"] = piv["isolation"] - piv["row_context"]
    summaries: list[dict[str, Any]] = []
    for (endpoint, form), g in piv.groupby(["endpoint", "input_form"], dropna=False):
        summaries.append({
            "endpoint": endpoint,
            "input_form": form,
            "n_sentences_filtered": int(len(g)),
            "row_context_loss": float(g["row_context"].mean()),
            "isolation_loss": float(g["isolation"].mean()),
            "context_gain": float(g["context_gain"].mean()),
            "se_context_gain": se([float(x) for x in g["context_gain"].tolist()]),
            "removed_bad_sentences": int(len(set(df[(df["endpoint"] == endpoint) & (df["input_form"] == form)]["sentence_uid"])) - len(set(g["sentence_uid"]))),
        })
    contrasts: list[dict[str, Any]] = []
    forms = sorted(set(piv["input_form"].astype(str)))
    for form in forms:
        pf = piv[piv["input_form"] == form]
        for anchor in anchors:
            if anchor not in set(pf["endpoint"]):
                continue
            a = pf[pf["endpoint"] == anchor].set_index("sentence_uid")
            for endpoint in sorted(set(pf["endpoint"])):
                if endpoint == anchor:
                    continue
                b = pf[pf["endpoint"] == endpoint].set_index("sentence_uid")
                common = sorted(set(a.index) & set(b.index))
                if not common:
                    continue
                da, db = a.loc[common], b.loc[common]
                d_row = (db["row_context"].to_numpy(dtype=float) - da["row_context"].to_numpy(dtype=float)).tolist()
                d_iso = (db["isolation"].to_numpy(dtype=float) - da["isolation"].to_numpy(dtype=float)).tolist()
                d_gain = (db["context_gain"].to_numpy(dtype=float) - da["context_gain"].to_numpy(dtype=float)).tolist()
                contrasts.append({
                    "endpoint": endpoint,
                    "anchor": anchor,
                    "input_form": form,
                    "n_common_sentences_filtered": int(len(common)),
                    "delta_row_context_loss": float(sum(d_row) / len(d_row)),
                    "delta_isolation_loss": float(sum(d_iso) / len(d_iso)),
                    "delta_context_gain": float(sum(d_gain) / len(d_gain)),
                    "se_delta_context_gain": se(d_gain),
                    "fraction_endpoint_higher_context_gain": float(sum(1 for x in d_gain if x > 0) / len(d_gain)),
                })
    # Within-endpoint special-token increment.
    for endpoint in sorted(set(piv["endpoint"])):
        p = piv[piv["endpoint"] == endpoint]
        if {"with_special", "no_special"}.issubset(set(p["input_form"])):
            a = p[p["input_form"] == "no_special"].set_index("sentence_uid")
            b = p[p["input_form"] == "with_special"].set_index("sentence_uid")
            common = sorted(set(a.index) & set(b.index))
            if common:
                da, db = a.loc[common], b.loc[common]
                d_row = (db["row_context"].to_numpy(dtype=float) - da["row_context"].to_numpy(dtype=float)).tolist()
                d_iso = (db["isolation"].to_numpy(dtype=float) - da["isolation"].to_numpy(dtype=float)).tolist()
                d_gain = (db["context_gain"].to_numpy(dtype=float) - da["context_gain"].to_numpy(dtype=float)).tolist()
                contrasts.append({
                    "endpoint": endpoint,
                    "anchor": endpoint,
                    "input_form": "with_special_minus_no_special",
                    "n_common_sentences_filtered": int(len(common)),
                    "delta_row_context_loss": float(sum(d_row) / len(d_row)),
                    "delta_isolation_loss": float(sum(d_iso) / len(d_iso)),
                    "delta_context_gain": float(sum(d_gain) / len(d_gain)),
                    "se_delta_context_gain": se(d_gain),
                    "fraction_endpoint_higher_context_gain": float(sum(1 for x in d_gain if x > 0) / len(d_gain)),
                })
    return summaries, contrasts


def token_boundary_contrasts(token_rows: list[dict[str, Any]], anchors: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(token_rows)
    if df.empty:
        return []
    df = df[df["mode"] == "isolation"].copy()
    df["pair_key"] = df["sentence_uid"].astype(str) + "::" + df["target_token_ordinal"].astype(str)
    out: list[dict[str, Any]] = []
    for form, pf in df.groupby("input_form", dropna=False):
        for anchor in anchors:
            if anchor not in set(pf["endpoint"]):
                continue
            a = pf[pf["endpoint"] == anchor].set_index("pair_key")
            for endpoint in sorted(set(pf["endpoint"])):
                if endpoint == anchor:
                    continue
                b = pf[pf["endpoint"] == endpoint].set_index("pair_key")
                common = sorted(set(a.index) & set(b.index))
                if not common:
                    continue
                joined = pd.DataFrame({
                    "delta_token_loss": b.loc[common]["token_loss"].to_numpy(dtype=float) - a.loc[common]["token_loss"].to_numpy(dtype=float),
                    "target_edge_bin": b.loc[common]["target_edge_bin"].astype(str).tolist(),
                    "input_edge_bin": b.loc[common]["input_edge_bin"].astype(str).tolist(),
                    "distance_to_boundary_special_bin": b.loc[common]["distance_to_boundary_special_bin"].astype(str).tolist(),
                    "sentence_words": b.loc[common]["sentence_words"].astype(int).tolist(),
                    "source": b.loc[common]["source"].astype(str).tolist(),
                })
                for bin_field in ["target_edge_bin", "input_edge_bin", "distance_to_boundary_special_bin"]:
                    for bin_name, g in joined.groupby(bin_field, dropna=False):
                        vals = [float(x) for x in g["delta_token_loss"].tolist()]
                        out.append({
                            "endpoint": endpoint,
                            "anchor": anchor,
                            "input_form": str(form),
                            "bin_field": bin_field,
                            "bin": str(bin_name),
                            "n_masked_tokens": int(len(vals)),
                            "delta_isolation_token_loss": float(sum(vals) / len(vals)),
                            "se_delta": se(vals),
                            "fraction_improved": float(sum(1 for x in vals if x < 0) / len(vals)),
                        })
    return out


def source_contrasts(score_rows: list[dict[str, Any]], anchors: list[str]) -> list[dict[str, Any]]:
    df = pd.DataFrame(score_rows)
    if df.empty:
        return []
    df["target_truncated_right_bool"] = df["target_truncated_right"].astype(str).str.lower().isin(["true", "1"])
    bad = df[(df["mode"] == "row_context") & (df["target_truncated_right_bool"] | (df["n_masked"] >= df["n_target_tokens"]))]
    bad_set = set(zip(bad["endpoint"], bad["input_form"], bad["sentence_uid"]))
    keep = [(e, f, s) not in bad_set for e, f, s in zip(df["endpoint"], df["input_form"], df["sentence_uid"])]
    filt = df.loc[keep].copy()
    piv = filt.pivot_table(index=["endpoint", "input_form", "sentence_uid", "source", "sentence_words"], columns="mode", values="loss", aggfunc="mean").reset_index()
    piv = piv.dropna(subset=["row_context", "isolation"])
    piv["context_gain"] = piv["isolation"] - piv["row_context"]
    out: list[dict[str, Any]] = []
    for form, pf in piv.groupby("input_form", dropna=False):
        for anchor in anchors:
            if anchor not in set(pf["endpoint"]):
                continue
            a = pf[pf["endpoint"] == anchor].set_index("sentence_uid")
            for endpoint in sorted(set(pf["endpoint"])):
                if endpoint == anchor:
                    continue
                b = pf[pf["endpoint"] == endpoint].set_index("sentence_uid")
                common = sorted(set(a.index) & set(b.index))
                if not common:
                    continue
                tmp = pd.DataFrame({
                    "source": b.loc[common]["source"].tolist(),
                    "d_row": (b.loc[common]["row_context"].to_numpy(dtype=float) - a.loc[common]["row_context"].to_numpy(dtype=float)).tolist(),
                    "d_iso": (b.loc[common]["isolation"].to_numpy(dtype=float) - a.loc[common]["isolation"].to_numpy(dtype=float)).tolist(),
                    "d_gain": (b.loc[common]["context_gain"].to_numpy(dtype=float) - a.loc[common]["context_gain"].to_numpy(dtype=float)).tolist(),
                })
                for src, g in tmp.groupby("source"):
                    out.append({
                        "endpoint": endpoint,
                        "anchor": anchor,
                        "input_form": str(form),
                        "source": src,
                        "n_common_sentences_filtered": int(len(g)),
                        "delta_row_context_loss": float(g["d_row"].mean()),
                        "delta_isolation_loss": float(g["d_iso"].mean()),
                        "delta_context_gain": float(g["d_gain"].mean()),
                        "se_delta_context_gain": se([float(x) for x in g["d_gain"].tolist()]),
                    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--endpoints-json", default="")
    ap.add_argument("--axis", default=str(AXIS_DEFAULT))
    ap.add_argument("--max-sentences", type=int, default=720)
    ap.add_argument("--min-sentence-words", type=int, default=8)
    ap.add_argument("--max-sentence-words", type=int, default=45)
    ap.add_argument("--sample-seed", type=int, default=7601)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--forms", nargs="*", default=["with_special", "no_special"], choices=["with_special", "no_special"])
    ap.add_argument("--anchors", nargs="*", default=["chck82_slow_scale1p75", "coherent86_alpha075"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_cache(out_dir)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    research = load_step076_module()
    endpoints = read_endpoint_spec(args.endpoints_json)
    sentence_records = research.load_sentence_records(pathlib.Path(args.axis), max_sentences=args.max_sentences, min_words=args.min_sentence_words, max_words=args.max_sentence_words, seed=args.sample_seed)
    plan = {
        "status": "CONTEXT_GAIN_SPECIAL_TOKEN_CONTROL_PLAN",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "axis": rel(args.axis),
        "n_sentences": len(sentence_records),
        "input_forms": args.forms,
        "endpoints": {k: {**v, "path": rel(ROOT / v["path"])} for k, v in endpoints.items()},
        "missing": [rel(ROOT / v["path"]) for v in endpoints.values() if not model_file_ready(ROOT / v["path"])],
        "anchors": args.anchors,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    (out_dir / "score_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if plan["missing"]:
        raise FileNotFoundError(plan["missing"][0])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    all_sentence_rows: list[dict[str, Any]] = []
    all_token_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for label, spec in endpoints.items():
        endpoint = ROOT / spec["path"]
        print(json.dumps({"event": "load", "endpoint": label, "path": rel(endpoint)}, ensure_ascii=False), flush=True)
        tok = research.AutoTokenizer.from_pretrained(str(endpoint), use_fast=True, trust_remote_code=bool(spec.get("trust_remote_code", True)), local_files_only=True)
        model = research.load_model(endpoint, bool(spec.get("trust_remote_code", True)), device)
        ident = research.model_identity(endpoint, model, bool(spec.get("trust_remote_code", True)))
        ident.update({"endpoint": label, "description": spec.get("description", "")})
        identities.append(ident)
        for form in args.forms:
            t0 = time.time()
            add_special = form == "with_special"
            sent_rows, tok_rows = score_model(model, tok, sentence_records, device, args.batch_size, add_special)
            for r in sent_rows:
                q = dict(r)
                q.update({"endpoint": label, "endpoint_path": rel(endpoint), "description": spec.get("description", "")})
                all_sentence_rows.append(q)
            for r in tok_rows:
                q = dict(r)
                q.update({"endpoint": label, "endpoint_path": rel(endpoint), "description": spec.get("description", "")})
                all_token_rows.append(q)
            meta_rows.append({
                "endpoint": label,
                "input_form": form,
                "endpoint_path": rel(endpoint),
                "n_sentence_records": len(sentence_records),
                "n_sentence_score_rows": len(sent_rows),
                "n_masked_token_rows": len(tok_rows),
                "elapsed_sec": round(time.time() - t0, 1),
                "loaded_class": ident.get("loaded_class"),
                "total_params_loaded": ident.get("total_params_loaded"),
                "adapter_params_loaded": ident.get("adapter_params_loaded"),
            })
            print(json.dumps({"event": "scored", **meta_rows[-1]}, ensure_ascii=False), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summaries, contrasts = summarize_sentence_rows(all_sentence_rows, args.anchors)
    src_rows = source_contrasts(all_sentence_rows, args.anchors)
    token_bins = token_boundary_contrasts(all_token_rows, args.anchors)
    write_csv(out_dir / "sentence_scores.csv", all_sentence_rows)
    write_csv(out_dir / "masked_token_scores.csv", all_token_rows)
    write_csv(out_dir / "endpoint_summary.csv", summaries)
    write_csv(out_dir / "endpoint_contrasts.csv", contrasts)
    write_csv(out_dir / "source_contrasts.csv", src_rows)
    write_csv(out_dir / "isolation_token_boundary_contrasts.csv", token_bins)
    write_csv(out_dir / "score_meta.csv", meta_rows)
    with (out_dir / "model_identity.jsonl").open("w", encoding="utf-8") as f:
        for r in identities:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # Pull out the contrasts most relevant to the coherent-special interpretation.
    key_rows = [r for r in contrasts if r.get("anchor") == "coherent86_alpha075" or r.get("input_form") == "with_special_minus_no_special"]
    summary = {
        "status": "CONTEXT_GAIN_SPECIAL_TOKEN_CONTROL_DONE",
        "created_utc": now(),
        "out_dir": rel(out_dir),
        "n_sentences": len(sentence_records),
        "endpoints": list(endpoints),
        "input_forms": args.forms,
        "summaries": summaries,
        "contrasts": contrasts,
        "key_contrasts": key_rows,
        "token_boundary_contrasts": token_bins,
        "outputs": {
            "plan": rel(out_dir / "score_plan.json"),
            "sentence_scores": rel(out_dir / "sentence_scores.csv"),
            "masked_token_scores": rel(out_dir / "masked_token_scores.csv"),
            "endpoint_summary": rel(out_dir / "endpoint_summary.csv"),
            "endpoint_contrasts": rel(out_dir / "endpoint_contrasts.csv"),
            "source_contrasts": rel(out_dir / "source_contrasts.csv"),
            "isolation_token_boundary_contrasts": rel(out_dir / "isolation_token_boundary_contrasts.csv"),
            "model_identity": rel(out_dir / "model_identity.jsonl"),
        },
        "reading": "A with-special contrast against a no-special-trained anchor can mix context reliance with learned boundary-token mismatch. A no-special contrast removes that boundary-token channel on the same text coordinate.",
    }
    out_json = out_dir / "context_gain_special_token_control.json"
    out_md = out_dir / "context_gain_special_token_control.md"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research context/isolation coordinate with and without special tokens",
        "",
        summary["reading"],
        "",
        f"Scored {len(sentence_records)} Strict-complement sentence spans.",
        "",
        "## Endpoint summaries",
        "",
        "| endpoint | form | n | row loss | isolated loss | context gain | se | removed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(f"| {r['endpoint']} | {r['input_form']} | {r['n_sentences_filtered']} | {r['row_context_loss']:.4f} | {r['isolation_loss']:.4f} | {r['context_gain']:.4f} | {r['se_context_gain']:.4f} | {r['removed_bad_sentences']} |")
    lines += ["", "## Paired contrasts", "", "| endpoint | anchor | form | n | d_row | d_iso | d_context_gain | se | frac higher |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        lines.append(f"| {r['endpoint']} | {r['anchor']} | {r['input_form']} | {r['n_common_sentences_filtered']} | {r['delta_row_context_loss']:+.4f} | {r['delta_isolation_loss']:+.4f} | {r['delta_context_gain']:+.4f} | {r['se_delta_context_gain']:.4f} | {r['fraction_endpoint_higher_context_gain']:.3f} |")
    lines += ["", "## Isolation token-loss deltas by target edge distance", "", "Shown for endpoint-minus-anchor on masked tokens in isolated sentences.", "", "| endpoint | anchor | form | bin field | bin | n tok | d token loss | se | frac improved |", "|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in token_bins:
        if r["bin_field"] != "target_edge_bin":
            continue
        if r["anchor"] != "coherent86_alpha075":
            continue
        lines.append(f"| {r['endpoint']} | {r['anchor']} | {r['input_form']} | {r['bin_field']} | {r['bin']} | {r['n_masked_tokens']} | {r['delta_isolation_token_loss']:+.4f} | {r['se_delta']:.4f} | {r['fraction_improved']:.3f} |")
    lines += ["", f"JSON: `{rel(out_json)}`"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
