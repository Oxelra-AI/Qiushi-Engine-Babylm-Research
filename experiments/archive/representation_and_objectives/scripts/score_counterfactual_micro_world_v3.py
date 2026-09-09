#!/usr/bin/env python3
"""research: inference-only scorer for the counterfactual micro-world v3.

Scientific purpose
------------------
Score a frozen, model-unseen crossed-sign micro-world that asks whether the model
binds the same alternatives differently under two controlled contexts.  This is a
posthoc measurement on existing checkpoints only.  It does not train, tune a
checkpoint, define a submission metric, or touch the protected chck_82M endpoint.

The predeclared interpretation criterion is: the measure must not be
forced to rank chck_82M highly, because chck_82M is a broad-score peak already
known not to repair matched EWoK/GlobalPIQA binding.  The two-checkpoint pilot
checks mechanical coherence and surface controls only.  Scientific interpretation
requires a broader panel including distinct legal trajectories and the
MLM-only/coupled-aligned/coupled-shuffled turnover controls.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Writable caches must be set before importing transformers.
def find_user_root() -> Path:
    return _PUBLIC_ROOT

USER_ROOT = find_user_root()
os.chdir(USER_ROOT)
A01_WS = USER_ROOT / "experiments/archive" / 'representation_and_objectives'
A02_WS = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
OUT_ROOT = A01_WS / "data" / "counterfactual_micro_world_v3_scores"
CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "default")
CACHE_ROOT = OUT_ROOT / "hf_cache" / CACHE_SUFFIX
for name, path in {
    "HOME": CACHE_ROOT / "home",
    "XDG_CACHE_HOME": CACHE_ROOT / "xdg",
    "HF_HOME": CACHE_ROOT / "hf_home",
    "HF_MODULES_CACHE": CACHE_ROOT / "hf_modules",
    "TRANSFORMERS_CACHE": CACHE_ROOT / "transformers",
    "TORCH_HOME": CACHE_ROOT / "torch",
    "TMPDIR": CACHE_ROOT / "tmp",
}.items():
    os.environ.setdefault(name, str(path))
    Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

FRAME_ROOT = A01_WS / "data" / "counterfactual_micro_world_v3"
FRAME_PATH = FRAME_ROOT / "counterfactual_micro_world_v3_frames.jsonl"
RENAMED_PATH = FRAME_ROOT / "counterfactual_micro_world_v3_renamed_controls.jsonl"
MANIFEST_PATH = FRAME_ROOT / "counterfactual_micro_world_v3_manifest.json"
PANEL_SUMMARY_JSON = OUT_ROOT / "micro_world_v3_panel_summary.json"
PANEL_SUMMARY_MD = A01_WS / "notes" / "micro_world_v3_panel_interpretation.md"

# Existing checkpoints only.  No target here implies training or creates new models.
TARGETS: dict[str, dict[str, Any]] = {
    # Scale1.75 late ladder; 82M is the practical endpoint but not expected to be
    # a binding peak a priori.
    **{
        f"scale1p75_{m}M": {
            "label": f"A02 scale1.75 adapter128 legal16k ladder chck_{m}M",
            "model_path": A02_WS / f"training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_{m}M",
            "family": "scale1p75_late_ladder",
            "exposure_m": m,
        }
        for m in [77, 78, 79, 80, 81, 82, 83, 100]
    },
    # Legal compact-view baselines and alternate seed.
    "legal16k_base_80M_seed43022": {
        "label": "matched legal16k compact-view base seed43022 chck_80M",
        "model_path": A02_WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
        "family": "legal16k_compact_seed43022",
        "exposure_m": 80,
        "ewok_ref_key": "legal16k_80M",
    },
    "legal16k_base_100M_seed43022": {
        "label": "matched legal16k compact-view base seed43022 chck_100M",
        "model_path": A02_WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M",
        "family": "legal16k_compact_seed43022",
        "exposure_m": 100,
        "ewok_ref_key": "legal16k_100M",
    },
    "legal16k_80M_seed43122": {
        "label": "legal16k compact-view alternate seed43122 chck_80M",
        "model_path": A01_WS / "training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_80M",
        "family": "legal16k_compact_seed43122",
        "exposure_m": 80,
    },
    "legal16k_100M_seed43122": {
        "label": "legal16k compact-view alternate seed43122 chck_100M",
        "model_path": A01_WS / "training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_100M",
        "family": "legal16k_compact_seed43122",
        "exposure_m": 100,
    },
    # Legal40k trajectories.
    "legal40k_8x480_100M_seed43022": {
        "label": "legal40k 8x480 compact-view seed43022 chck_100M",
        "model_path": A01_WS / "training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M",
        "family": "legal40k_8x480",
        "exposure_m": 100,
        "ewok_ref_key": "legal40_8x480_43022",
    },
    "legal40k_depth12_100M_seed43022": {
        "label": "legal40k 12x384 depth compact-view seed43022 chck_100M",
        "model_path": A01_WS / "training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M",
        "family": "legal40k_depth12",
        "exposure_m": 100,
        "ewok_ref_key": "legal40_depth_12x384_43022",
    },
    # FineWeb allocation comparators that previously separated broad vs relational tradeoffs.
    "fw_compact_100M_seed43022": {
        "label": "FineWeb compact source-rewrite shared16k chck_100M",
        "model_path": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M",
        "family": "fineweb_allocation",
        "exposure_m": 100,
        "ewok_ref_key": "fw_compact_fullbatch_seed43022",
    },
    "fw_rowblock_100M_seed43022": {
        "label": "FineWeb row-block breadth shared16k chck_100M",
        "model_path": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M",
        "family": "fineweb_allocation",
        "exposure_m": 100,
        "ewok_ref_key": "fw_breadth_rowblock_fullbatch_seed43022",
    },
    # Coupled sparse20 turnover controls.
    "mlm_only_20M": {
        "label": "A02 exact MLM-only 20M baseline for the coupled sparse20 panel",
        "model_path": A02_WS / "training/runs/dualview_mlm_only_20M_seed43022/hf_model/final",
        "family": "coupled_sparse20_turnover_panel",
        "exposure_m": 20,
        "ewok_ref_key": "mlm_only_20M",
    },
    "coupled_aligned_20M": {
        "label": "A02 coupled sparse20 true-correspondence run at 20M",
        "model_path": A02_WS / "training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final",
        "family": "coupled_sparse20_turnover_panel",
        "exposure_m": 20,
        "ewok_ref_key": "coupled_aligned_20M",
    },
    "coupled_shuffled_20M": {
        "label": "A01 matched coupled sparse20 shuffled-correspondence control at 20M",
        "model_path": A01_WS / "training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final",
        "family": "coupled_sparse20_turnover_panel",
        "exposure_m": 20,
        "ewok_ref_key": "coupled_shuffled_20M",
    },
}

PANELS = {
    "pilot": ["scale1p75_82M", "scale1p75_100M"],
    "scale1p75_late": [f"scale1p75_{m}M" for m in [77, 78, 79, 80, 81, 82, 83, 100]],
    "validation": [
        "scale1p75_80M", "scale1p75_82M", "scale1p75_100M",
        "legal16k_base_80M_seed43022", "legal16k_base_100M_seed43022",
        "legal16k_80M_seed43122", "legal16k_100M_seed43122",
        "legal40k_8x480_100M_seed43022", "legal40k_depth12_100M_seed43022",
        "fw_compact_100M_seed43022", "fw_rowblock_100M_seed43022",
        "mlm_only_20M", "coupled_aligned_20M", "coupled_shuffled_20M",
    ],
}
PANELS["all"] = list(dict.fromkeys(PANELS["scale1p75_late"] + PANELS["validation"]))


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    try:
        return str(pp.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(pp)


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def as_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def qstats(vals: Iterable[Any]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {
        "n": len(xs),
        "min": xs[0],
        "p01": q(0.01),
        "p05": q(0.05),
        "p25": q(0.25),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p75": q(0.75),
        "p95": q(0.95),
        "p99": q(0.99),
        "max": xs[-1],
    }


def mean(xs: list[float]) -> float | None:
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(vx * vy)


def ranks(vals: list[float]) -> list[float]:
    order = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    j = 0
    while j < len(order):
        k = j + 1
        while k < len(order) and order[k][0] == order[j][0]:
            k += 1
        r = (j + 1 + k) / 2.0
        for _, i in order[j:k]:
            out[i] = r
        j = k
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    return pearson(rx, ry)


class BatchedMaskScorer:
    def __init__(self, model, tokenizer, device: torch.device, masked_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.masked_batch_size = masked_batch_size
        self.mask_id = tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("Tokenizer has no mask_token_id")

    def score_examples(self, examples: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
        acc: dict[tuple[int, str, str], dict[str, Any]] = {}
        batch_examples: list[dict[str, Any]] = []
        for ex in examples:
            sid = (int(ex["rec_i"]), str(ex["variant"]), str(ex["key"]))
            acc.setdefault(sid, {"sum": 0.0, "n_tokens": 0, "status": "ok"})
            sentence = ex["sentence"]
            span = tuple(ex["span"])
            enc = self.tokenizer(sentence, return_offsets_mapping=True, return_tensors=None)
            token_ids = list(enc["input_ids"])
            attention = list(enc["attention_mask"])
            offsets = list(enc["offset_mapping"])
            selected = []
            s, e = span
            for pos, (a, b) in enumerate(offsets):
                if a == b == 0:
                    continue
                if b > s and a < e:
                    selected.append(pos)
            if not selected:
                acc[sid]["status"] = "no_tokens"
                continue
            for pos in selected:
                cur = list(token_ids)
                cur[pos] = self.mask_id
                batch_examples.append({
                    "sid": sid,
                    "input_ids": cur,
                    "attention_mask": attention,
                    "pos": pos,
                    "target_id": token_ids[pos],
                    "length": len(cur),
                })
        batch_examples.sort(key=lambda x: x["length"])
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        with torch.inference_mode():
            for start in range(0, len(batch_examples), self.masked_batch_size):
                batch = batch_examples[start:start + self.masked_batch_size]
                max_len = max(ex["length"] for ex in batch)
                input_ids = torch.tensor([ex["input_ids"] + [pad_id] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                attn = torch.tensor([ex["attention_mask"] + [0] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                pos = torch.tensor([ex["pos"] for ex in batch], dtype=torch.long, device=self.device)
                target = torch.tensor([ex["target_id"] for ex in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits if hasattr(out, "logits") else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                masked = logits[mb, pos]
                vals = torch.gather(F.log_softmax(masked, dim=-1), -1, target.unsqueeze(-1)).squeeze(-1)
                for sid, val in zip([ex["sid"] for ex in batch], vals.detach().cpu().tolist()):
                    acc[sid]["sum"] += float(val)
                    acc[sid]["n_tokens"] += 1
        for d in acc.values():
            n = d["n_tokens"]
            d["mean"] = d["sum"] / n if n else float("nan")
        return acc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


def build_examples(frames: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    examples = []
    for rec_i, r in enumerate(frames):
        contexts = {
            "c1": r["context1"],
            "c2": r["context2"],
            "none": "",
            "shuf1": r.get("controls", {}).get("shuffled_context1", ""),
            "shuf2": r.get("controls", {}).get("shuffled_context2", ""),
        }
        for ctx_key, ctx in contexts.items():
            for alt_key, alt in [("a", r["alt_a"]), ("b", r["alt_b"])]:
                sent, span = render(ctx, r["query_template"], alt)
                examples.append({
                    "rec_i": rec_i,
                    "frame_id": r["frame_id"],
                    "variant": variant,
                    "key": f"{ctx_key}_{alt_key}",
                    "sentence": sent,
                    "span": span,
                })
    return examples


def attach_records(frames: list[dict[str, Any]], scores: dict[tuple[int, str, str], dict[str, Any]], variant: str, target: str) -> list[dict[str, Any]]:
    out = []
    keys = ["c1_a", "c1_b", "c2_a", "c2_b", "none_a", "none_b", "shuf1_a", "shuf1_b", "shuf2_a", "shuf2_b"]
    for i, r in enumerate(frames):
        row = {
            "target": target,
            "variant": variant,
            "frame_id": r["frame_id"],
            "family": r.get("family"),
            "subtype": r.get("subtype"),
            "depth": r.get("depth"),
            "alt_a": r.get("alt_a"),
            "alt_b": r.get("alt_b"),
            "target_prior_ratio": r.get("target_prior", {}).get("max_min_ratio"),
            "target_prior_balanced_le4": bool(r.get("target_prior", {}).get("balanced_ratio_le_4")),
            "renaming_of": r.get("renaming_of"),
            "shuffled_context_family": r.get("controls", {}).get("shuffled_context_family"),
        }
        for key in keys:
            d = scores.get((i, variant, key), {"sum": float("nan"), "mean": float("nan"), "n_tokens": 0, "status": "missing"})
            row[f"{key}_sum"] = d.get("sum", float("nan"))
            row[f"{key}_mean"] = d.get("mean", float("nan"))
            row[f"{key}_n_tokens"] = d.get("n_tokens", 0)
            row[f"{key}_status"] = d.get("status", "missing")
        c1a, c1b, c2a, c2b = [as_float(row[f"{k}_sum"]) for k in ["c1_a", "c1_b", "c2_a", "c2_b"]]
        sh1a, sh1b, sh2a, sh2b = [as_float(row[f"{k}_sum"]) for k in ["shuf1_a", "shuf1_b", "shuf2_a", "shuf2_b"]]
        na, nb = as_float(row["none_a_sum"]), as_float(row["none_b_sum"])
        row["delta1"] = c1a - c1b
        row["delta2"] = c2a - c2b
        row["correct_margin_c1"] = row["delta1"]
        row["correct_margin_c2"] = -row["delta2"]
        row["min_signed_margin"] = min(row["correct_margin_c1"], row["correct_margin_c2"])
        row["interaction"] = row["delta1"] - row["delta2"]
        row["crossed_success"] = row["delta1"] > 0 and row["delta2"] < 0
        row["c1_success"] = row["delta1"] > 0
        row["c2_success"] = row["delta2"] < 0
        row["same_sign_positive"] = row["delta1"] > 0 and row["delta2"] > 0
        row["same_sign_negative"] = row["delta1"] < 0 and row["delta2"] < 0
        row["none_delta"] = na - nb
        row["delta1_minus_none"] = row["delta1"] - row["none_delta"]
        row["delta2_minus_none"] = row["delta2"] - row["none_delta"]
        row["shuf_delta1"] = sh1a - sh1b
        row["shuf_delta2"] = sh2a - sh2b
        row["shuf_correct_margin_c1"] = row["shuf_delta1"]
        row["shuf_correct_margin_c2"] = -row["shuf_delta2"]
        row["shuf_min_signed_margin"] = min(row["shuf_correct_margin_c1"], row["shuf_correct_margin_c2"])
        row["shuf_interaction"] = row["shuf_delta1"] - row["shuf_delta2"]
        row["shuf_crossed_success"] = row["shuf_delta1"] > 0 and row["shuf_delta2"] < 0
        row["interaction_excess_over_shuffled"] = row["interaction"] - row["shuf_interaction"]
        row["min_margin_excess_over_shuffled"] = row["min_signed_margin"] - row["shuf_min_signed_margin"]
        out.append(row)
    return out


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "crossed_success": sum(1 for r in rows if r["crossed_success"]) / n,
        "shuf_crossed_success": sum(1 for r in rows if r["shuf_crossed_success"]) / n,
        "crossed_excess_over_shuffled": (sum(1 for r in rows if r["crossed_success"]) - sum(1 for r in rows if r["shuf_crossed_success"])) / n,
        "c1_success": sum(1 for r in rows if r["c1_success"]) / n,
        "c2_success": sum(1 for r in rows if r["c2_success"]) / n,
        "same_sign_positive": sum(1 for r in rows if r["same_sign_positive"]) / n,
        "same_sign_negative": sum(1 for r in rows if r["same_sign_negative"]) / n,
        "strong_crossed_min_margin_gt_0p1": sum(1 for r in rows if r["crossed_success"] and as_float(r["min_signed_margin"]) > 0.1) / n,
        "strong_crossed_min_margin_gt_0p5": sum(1 for r in rows if r["crossed_success"] and as_float(r["min_signed_margin"]) > 0.5) / n,
        "interaction": qstats(r["interaction"] for r in rows),
        "shuf_interaction": qstats(r["shuf_interaction"] for r in rows),
        "interaction_excess_over_shuffled": qstats(r["interaction_excess_over_shuffled"] for r in rows),
        "min_signed_margin": qstats(r["min_signed_margin"] for r in rows),
        "shuf_min_signed_margin": qstats(r["shuf_min_signed_margin"] for r in rows),
        "min_margin_excess_over_shuffled": qstats(r["min_margin_excess_over_shuffled"] for r in rows),
        "delta1": qstats(r["delta1"] for r in rows),
        "delta2": qstats(r["delta2"] for r in rows),
        "none_delta": qstats(r["none_delta"] for r in rows),
        "abs_none_delta_mean": mean([abs(as_float(r["none_delta"])) for r in rows]),
        "abs_delta1_minus_none_mean": mean([abs(as_float(r["delta1_minus_none"])) for r in rows]),
        "abs_delta2_minus_none_mean": mean([abs(as_float(r["delta2_minus_none"])) for r in rows]),
    }


def summarize_by(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r.get(field))].append(r)
    out = []
    for k, rs in sorted(groups.items()):
        s = summarize_rows(rs)
        out.append({
            field: k,
            "n": s.get("n"),
            "crossed_success": s.get("crossed_success"),
            "shuf_crossed_success": s.get("shuf_crossed_success"),
            "crossed_excess_over_shuffled": s.get("crossed_excess_over_shuffled"),
            "interaction_median": s.get("interaction", {}).get("median"),
            "min_signed_margin_median": s.get("min_signed_margin", {}).get("median"),
            "strong_crossed_min_margin_gt_0p1": s.get("strong_crossed_min_margin_gt_0p1"),
        })
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def compare_renamed(original_rows: list[dict[str, Any]], renamed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    orig = {r["frame_id"]: r for r in original_rows}
    pairs = []
    for rr in renamed_rows:
        base_id = rr.get("renaming_of")
        if base_id in orig:
            pairs.append((orig[base_id], rr))
    if not pairs:
        return {"n": 0}
    return {
        "n": len(pairs),
        "crossed_success_agreement": sum(1 for a, b in pairs if bool(a["crossed_success"]) == bool(b["crossed_success"])) / len(pairs),
        "original_crossed": sum(1 for a, _ in pairs if a["crossed_success"]) / len(pairs),
        "renamed_crossed": sum(1 for _, b in pairs if b["crossed_success"]) / len(pairs),
        "renamed_minus_original_crossed": (sum(1 for _, b in pairs if b["crossed_success"]) - sum(1 for a, _ in pairs if a["crossed_success"])) / len(pairs),
        "interaction_delta_renamed_minus_original": qstats(as_float(b["interaction"]) - as_float(a["interaction"]) for a, b in pairs),
        "min_margin_delta_renamed_minus_original": qstats(as_float(b["min_signed_margin"]) - as_float(a["min_signed_margin"]) for a, b in pairs),
        "interaction_pearson_original_renamed": pearson([as_float(a["interaction"]) for a, _ in pairs], [as_float(b["interaction"]) for _, b in pairs]),
        "min_margin_pearson_original_renamed": pearson([as_float(a["min_signed_margin"]) for a, _ in pairs], [as_float(b["min_signed_margin"]) for _, b in pairs]),
    }


def load_ewok_references() -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    # research matched scale/base summary.
    p155 = A01_WS / "data/scale1p75_matched_ewok_interaction/scale1p75_matched_ewok_summary.json"
    if p155.exists():
        data = json.loads(p155.read_text(encoding="utf-8"))
        for key, obj in data.get("targets", {}).items():
            s = obj.get("summary", {})
            refs[key] = {
                "source": rel(p155),
                "accuracy": s.get("accuracy"),
                "stable_failure": s.get("stable_failure"),
                "stable_failure_frac_all": s.get("stable_failure_frac_all"),
                "interaction_mean": s.get("interaction_sum_all", {}).get("mean"),
                "interaction_median": s.get("interaction_sum_all", {}).get("median"),
                "wrong_interaction_median": s.get("interaction_sum_wrong", {}).get("median"),
            }
    # research legal40 summary uses slightly different field names.
    p100 = A01_WS / "data/ewok_interaction_synthesis/ewok_interaction_synthesis.json"
    if p100.exists():
        data = json.loads(p100.read_text(encoding="utf-8"))
        for key, s in data.get("per_model", {}).items():
            refs[key] = {
                "source": rel(p100),
                "accuracy": s.get("saved_accuracy"),
                "stable_failure": s.get("stable_failure"),
                "stable_failure_frac_all": s.get("stable_failure_frac_all"),
                "interaction_mean": s.get("interaction_sum", {}).get("mean"),
                "interaction_median": s.get("interaction_sum", {}).get("median"),
                "wrong_interaction_median": s.get("interaction_sum_saved_wrong", {}).get("median"),
            }
    # research FineWeb summary.
    p108 = A01_WS / "data/fw_ewok_interaction_reader/fw_ewok_interaction_reader_summary.json"
    if p108.exists():
        data = json.loads(p108.read_text(encoding="utf-8"))
        for key, obj in data.get("targets", {}).items():
            s = obj.get("summary", {})
            refs[key] = {
                "source": rel(p108),
                "accuracy": s.get("accuracy"),
                "stable_failure": s.get("stable_failure"),
                "stable_failure_frac_all": s.get("stable_failure_frac_all"),
                "interaction_mean": s.get("interaction_sum_all", {}).get("mean"),
                "interaction_median": s.get("interaction_sum_all", {}).get("median"),
                "wrong_interaction_median": s.get("interaction_sum_wrong", {}).get("median"),
            }
    # research coupled turnover summary.
    p181 = A01_WS / "data/full_ewok_coupled_turnover/full_ewok_coupled_turnover_summary.json"
    if p181.exists():
        data = json.loads(p181.read_text(encoding="utf-8"))
        for key, s in data.get("target_summaries", {}).items():
            refs[key] = {
                "source": rel(p181),
                "accuracy": s.get("accuracy"),
                "stable_failure": s.get("stable_failure"),
                "stable_failure_frac_all": s.get("stable_failure_frac_all"),
                "interaction_mean": s.get("interaction_sum", {}).get("mean"),
                "interaction_median": s.get("interaction_sum", {}).get("median"),
                "wrong_interaction_median": None,
            }
    return refs


def target_dir(target: str) -> Path:
    return OUT_ROOT / "targets" / target


def target_summary_path(target: str) -> Path:
    return target_dir(target) / "micro_world_v3_summary.json"


def target_ready(target: str) -> bool:
    sp = target_summary_path(target)
    if not sp.exists():
        return False
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
        return data.get("status") == "SCORED" and data.get("n_original") == 400 and data.get("n_renamed") == 400
    except Exception:
        return False


def score_target(target: str, device_name: str, threads: int, masked_batch_size: int, force: bool) -> dict[str, Any]:
    if target not in TARGETS:
        raise KeyError(target)
    meta = TARGETS[target]
    model_path = Path(meta["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"model_path missing for {target}: {model_path}")
    out_dir = target_dir(target)
    out_dir.mkdir(parents=True, exist_ok=True)
    if target_ready(target) and not force:
        return {"status": "SKIPPED_EXISTING", "target": target, "summary_json": rel(target_summary_path(target))}
    if threads > 0:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "score_target_start", "target": target, "device": str(device), "model_path": rel(model_path), "utc": now_utc()}), flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval().to(device)
    scorer = BatchedMaskScorer(model, tokenizer, device, masked_batch_size)
    frames = load_jsonl(FRAME_PATH)
    renamed = load_jsonl(RENAMED_PATH)
    examples = build_examples(frames, "original") + build_examples(renamed, "renamed")
    scores = scorer.score_examples(examples)
    original_rows = attach_records(frames, scores, "original", target)
    renamed_rows = attach_records(renamed, scores, "renamed", target)
    all_rows = original_rows + renamed_rows
    refs = load_ewok_references()
    ewok_ref = refs.get(meta.get("ewok_ref_key") or target)
    summary = {
        "status": "SCORED",
        "created_utc": now_utc(),
        "target": target,
        "label": meta.get("label"),
        "model_path": rel(model_path),
        "target_family": meta.get("family"),
        "exposure_m": meta.get("exposure_m"),
        "device": str(device),
        "threads": threads,
        "masked_batch_size": masked_batch_size,
        "elapsed_sec": round(time.time() - t0, 2),
        "frame_manifest": rel(MANIFEST_PATH),
        "n_original": len(original_rows),
        "n_renamed": len(renamed_rows),
        "summaries": {
            "original_all": summarize_rows(original_rows),
            "original_prior_balanced_le4": summarize_rows([r for r in original_rows if r["target_prior_balanced_le4"]]),
            "original_prior_imbalanced_gt4": summarize_rows([r for r in original_rows if not r["target_prior_balanced_le4"]]),
            "renamed_all": summarize_rows(renamed_rows),
            "renamed_prior_balanced_le4": summarize_rows([r for r in renamed_rows if r["target_prior_balanced_le4"]]),
            "renamed_prior_imbalanced_gt4": summarize_rows([r for r in renamed_rows if not r["target_prior_balanced_le4"]]),
        },
        "by_family_original": summarize_by(original_rows, "family"),
        "by_depth_original": summarize_by(original_rows, "depth"),
        "by_subtype_original": summarize_by(original_rows, "subtype"),
        "renamed_comparison": compare_renamed(original_rows, renamed_rows),
        "ewok_full_surface_reference": ewok_ref,
        "records_jsonl": rel(out_dir / "micro_world_v3_records.jsonl"),
        "records_csv": rel(out_dir / "micro_world_v3_records.csv"),
    }
    write_jsonl(out_dir / "micro_world_v3_records.jsonl", all_rows)
    write_csv(out_dir / "micro_world_v3_records.csv", all_rows)
    target_summary_path(target).write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "score_target_done",
        "target": target,
        "elapsed_sec": summary["elapsed_sec"],
        "original_crossed": summary["summaries"]["original_all"]["crossed_success"],
        "original_crossed_excess_over_shuffled": summary["summaries"]["original_all"]["crossed_excess_over_shuffled"],
        "renamed_crossed": summary["summaries"]["renamed_all"]["crossed_success"],
        "summary_json": rel(target_summary_path(target)),
    }, ensure_ascii=False), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "SCORED", "target": target, "summary_json": rel(target_summary_path(target))}


def load_target_summary(target: str) -> dict[str, Any] | None:
    p = target_summary_path(target)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def flat_metric(summary: dict[str, Any], key: str) -> float:
    s = summary.get("summaries", {}).get("original_all", {})
    if key == "crossed_success":
        return as_float(s.get("crossed_success"))
    if key == "crossed_excess_over_shuffled":
        return as_float(s.get("crossed_excess_over_shuffled"))
    if key == "interaction_median":
        return as_float(s.get("interaction", {}).get("median"))
    if key == "interaction_mean":
        return as_float(s.get("interaction", {}).get("mean"))
    if key == "min_margin_median":
        return as_float(s.get("min_signed_margin", {}).get("median"))
    if key == "strong_crossed_gt_0p1":
        return as_float(s.get("strong_crossed_min_margin_gt_0p1"))
    if key == "renamed_crossed":
        return as_float(summary.get("summaries", {}).get("renamed_all", {}).get("crossed_success"))
    return float("nan")


def aggregate(scored_targets: list[str] | None = None) -> dict[str, Any]:
    if scored_targets is None:
        scored_targets = [t for t in TARGETS if target_ready(t)]
    rows = []
    for t in scored_targets:
        d = load_target_summary(t)
        if not d:
            continue
        orig = d.get("summaries", {}).get("original_all", {})
        bal = d.get("summaries", {}).get("original_prior_balanced_le4", {})
        ren = d.get("summaries", {}).get("renamed_all", {})
        ew = d.get("ewok_full_surface_reference") or {}
        rows.append({
            "target": t,
            "family": d.get("target_family"),
            "exposure_m": d.get("exposure_m"),
            "crossed": orig.get("crossed_success"),
            "balanced_crossed": bal.get("crossed_success"),
            "shuf_crossed": orig.get("shuf_crossed_success"),
            "crossed_excess": orig.get("crossed_excess_over_shuffled"),
            "strong_crossed_gt_0p1": orig.get("strong_crossed_min_margin_gt_0p1"),
            "interaction_mean": orig.get("interaction", {}).get("mean"),
            "interaction_median": orig.get("interaction", {}).get("median"),
            "min_margin_median": orig.get("min_signed_margin", {}).get("median"),
            "renamed_crossed": ren.get("crossed_success"),
            "renamed_agreement": d.get("renamed_comparison", {}).get("crossed_success_agreement"),
            "ewok_accuracy": ew.get("accuracy"),
            "ewok_stable_failure_frac_all": ew.get("stable_failure_frac_all"),
            "ewok_interaction_median": ew.get("interaction_median"),
            "ewok_interaction_mean": ew.get("interaction_mean"),
            "summary_json": rel(target_summary_path(t)),
        })
    metric_keys = ["crossed", "balanced_crossed", "crossed_excess", "strong_crossed_gt_0p1", "interaction_median", "interaction_mean", "min_margin_median"]
    ewok_keys = ["ewok_accuracy", "ewok_stable_failure_frac_all", "ewok_interaction_median", "ewok_interaction_mean"]
    correlations: dict[str, dict[str, Any]] = {}
    ref_rows = [r for r in rows if finite(r.get("ewok_accuracy"))]
    for mk in metric_keys:
        correlations[mk] = {}
        for ek in ewok_keys:
            pairs = [(as_float(r.get(mk)), as_float(r.get(ek))) for r in ref_rows if finite(r.get(mk)) and finite(r.get(ek))]
            correlations[mk][ek] = {
                "n": len(pairs),
                "pearson": pearson([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None,
                "spearman": spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None,
            }
    coupled = {r["target"]: r for r in rows if r["target"] in {"mlm_only_20M", "coupled_aligned_20M", "coupled_shuffled_20M"}}
    coupled_reading: dict[str, Any] = {"available": sorted(coupled)}
    if set(coupled) == {"mlm_only_20M", "coupled_aligned_20M", "coupled_shuffled_20M"}:
        b = coupled["mlm_only_20M"]
        a = coupled["coupled_aligned_20M"]
        s = coupled["coupled_shuffled_20M"]
        def diff(row, base, field):
            return as_float(row.get(field)) - as_float(base.get(field))
        coupled_reading.update({
            "aligned_minus_mlm_crossed": diff(a, b, "crossed"),
            "shuffled_minus_mlm_crossed": diff(s, b, "crossed"),
            "aligned_minus_shuffled_crossed": diff(a, s, "crossed"),
            "aligned_minus_mlm_crossed_excess": diff(a, b, "crossed_excess"),
            "shuffled_minus_mlm_crossed_excess": diff(s, b, "crossed_excess"),
            "aligned_minus_shuffled_crossed_excess": diff(a, s, "crossed_excess"),
            "full_ewok_reminder": {
                "mlm_accuracy": b.get("ewok_accuracy"),
                "aligned_accuracy": a.get("ewok_accuracy"),
                "shuffled_accuracy": s.get("ewok_accuracy"),
                "mlm_stable_failure_frac": b.get("ewok_stable_failure_frac_all"),
                "aligned_stable_failure_frac": a.get("ewok_stable_failure_frac_all"),
                "shuffled_stable_failure_frac": s.get("ewok_stable_failure_frac_all"),
            },
        })
        # The turnover artifact should not be treated as success.  If both coupled
        # variants improve micro-world crossed substantially while full EWoK accuracy
        # remains near-null/worse, the micro-world would be rewarding the wrong thing.
        coupled_reading["turnover_rejection_reading"] = (
            "rejects_coupled_turnover" if max(abs(diff(a, b, "crossed")), abs(diff(s, b, "crossed"))) < 0.03
            else "sensitive_to_coupled_trajectory_or_turnover; do not use as training selector without deeper analysis"
        )
    panel = {
        "status": "AGGREGATED",
        "created_utc": now_utc(),
        "frame_manifest": rel(MANIFEST_PATH),
        "scored_targets": [r["target"] for r in rows],
        "table": rows,
        "correlations_to_existing_full_ewok_references": correlations,
        "coupled_turnover_reading": coupled_reading,
        "strategist_boundary": "Do not require chck_82M to rank near top; two-checkpoint pilot checks mechanics only; training design requires distinct-trajectory and coupled-turnover validation.",
    }
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PANEL_SUMMARY_JSON.write_text(json.dumps(panel, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_panel_note(panel)
    return panel


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
        if math.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception:
        pass
    return str(x)


def write_panel_note(panel: dict[str, Any]) -> None:
    rows = panel.get("table", [])
    lines = [
        "# research micro-world v3 panel interpretation",
        "",
        f"Status: **{panel.get('status')}**",
        "",
        "This is inference-only scoring of the frozen v3 crossed-sign counterfactual micro-world. It is not a submission metric and was not tuned to make chck_82M rank highly.",
        "",
        "## Scored target table",
        "",
        "| target | family | crossed | balanced crossed | shuffled crossed | crossed excess | interaction median | min-margin median | renamed crossed | EWoK acc | EWoK stable frac |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {target} | {family} | {crossed} | {balanced} | {shuf} | {excess} | {imed} | {mmed} | {ren} | {eacc} | {estab} |".format(
            target=r.get("target"), family=r.get("family"), crossed=fmt(r.get("crossed")), balanced=fmt(r.get("balanced_crossed")),
            shuf=fmt(r.get("shuf_crossed")), excess=fmt(r.get("crossed_excess")), imed=fmt(r.get("interaction_median")),
            mmed=fmt(r.get("min_margin_median")), ren=fmt(r.get("renamed_crossed")), eacc=fmt(r.get("ewok_accuracy")), estab=fmt(r.get("ewok_stable_failure_frac_all")),
        ))
    lines.extend(["", "## Coupled turnover reading", "", "```json", json.dumps(panel.get("coupled_turnover_reading"), indent=2, ensure_ascii=False, sort_keys=True), "```", "", "## Correlations to existing full-EWoK four-cell references", "", "```json", json.dumps(panel.get("correlations_to_existing_full_ewok_references"), indent=2, ensure_ascii=False, sort_keys=True), "```", "", "JSON: `{}`".format(rel(PANEL_SUMMARY_JSON))])
    PANEL_SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    PANEL_SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_targets(args) -> list[str]:
    if args.targets:
        out = []
        for t in args.targets.split(","):
            t = t.strip()
            if t:
                out.append(t)
        return list(dict.fromkeys(out))
    if args.panel:
        if args.panel not in PANELS:
            raise KeyError(f"unknown panel {args.panel}; available {sorted(PANELS)}")
        return PANELS[args.panel]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=sorted(PANELS), default=None)
    ap.add_argument("--targets", default=None, help="comma-separated target keys")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--masked-batch-size", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--list-targets", action="store_true")
    args = ap.parse_args()
    if args.list_targets:
        for k, v in TARGETS.items():
            print(json.dumps({"target": k, "exists": Path(v["model_path"]).exists(), "path": rel(v["model_path"]), "family": v.get("family")}, ensure_ascii=False))
        return
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not FRAME_PATH.exists() or not RENAMED_PATH.exists():
        raise FileNotFoundError("frozen v3 frame files missing; run freeze_counterfactual_micro_world_v3.py first")
    targets = parse_targets(args)
    results = []
    if not args.aggregate_only:
        if not targets:
            raise RuntimeError("no targets selected")
        for t in targets:
            results.append(score_target(t, args.device, args.threads, args.masked_batch_size, args.force))
    panel = aggregate(targets if targets else None)
    print(json.dumps({"status": "DONE", "results": results, "panel_summary_json": rel(PANEL_SUMMARY_JSON), "panel_note": rel(PANEL_SUMMARY_MD), "scored_targets": panel.get("scored_targets")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
