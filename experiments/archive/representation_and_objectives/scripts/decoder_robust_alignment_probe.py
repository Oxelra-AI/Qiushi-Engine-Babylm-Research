#!/usr/bin/env python3
"""research: decoder-robust conditional-ordering readout on existing checkpoints.

This script fits no-official-label layer alignments on legal pretraining-corpus
masked states, uses the research held-out naturalistic bridge only to define an
adjacent-depth readout coordinate later collated outside this script, and saves
EWoK/GlobalPIQA comparative rows for that fixed coordinate. It performs no model
training and does not update checkpoint weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_WS = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT_ROOT = Path(os.environ.get("OUT_ROOT", str(A01_WS / "data/decoder_robust_alignment_probe")))
HF_CACHE = OUT_ROOT / "hf_cache"
NOTE = Path(os.environ.get("NOTE", str(A01_WS / "notes/160_decoder_robust_alignment_probe.md")))
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_MODULES_CACHE"] = str(HF_CACHE / "modules")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

STRICT = USER_ROOT / "experiments/archive/initial_model_studies/repos/babylm-eval/strict"
if str(STRICT) not in sys.path:
    sys.path.insert(0, str(STRICT))
from evaluation_pipeline.sentence_zero_shot.dataset import get_dataloader  # noqa: E402

LEGAL_CORPUS = A02_WS / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
BRIDGE_PAIRS = A01_WS / "data/naturalistic_exchange_bridge/naturalistic_exchange_bridge_pairs.jsonl"
EVAL_EWOK = A01_WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
EWOK_TRANSITIONS = A01_WS / "data/ewok_transition_anatomy/live_minus_matched/all_transition_rows.csv"
GP_DATA_ROOT = A01_WS / "data/globalpiqa_official_lineage/official_dl_scratch/generated_by_current_official_dl/evaluation_data/full_eval"
GP_ANATOMY = A01_WS / "data/globalpiqa_parallel_anatomy/globalpiqa_parallel_anatomy.json"

TARGETS: dict[str, dict[str, Any]] = {
    "matched_base_80M": {
        "model_path": A02_WS / "training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M",
        "label": "A02 matched legal16k compact-view base, 80M",
    },
    "scale1p75_live_80M": {
        "model_path": A02_WS / "training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M",
        "label": "A02 scale1.75 adapter live, 80M",
        "adapter_placement_source": A02_WS / "training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/adapter_scaled_modeling.py",
    },
    "scale1p75_disabled_80M": {
        "model_path": A01_WS / "data/scale1p75_ewok_inference_ablation/proxy_models/scale1p75_disabled",
        "label": "A02 scale1.75 trained trajectory with adapters disabled at inference, 80M",
        "optional": True,
    },
}

WORD_RE = re.compile(r"\S+")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p10": q(0.10), "p90": q(0.90), "min": xs[0], "max": xs[-1]}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


# --------------------------- model / corpus states ---------------------------


def load_model_and_tokenizer(target: str, device: torch.device):
    path = Path(TARGETS[target]["model_path"])
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True)
    model.eval().to(device)
    return tok, model


def read_corpus_texts(sample_texts: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    if not LEGAL_CORPUS.exists():
        raise FileNotFoundError(LEGAL_CORPUS)
    rng = random.Random(seed)
    reservoir: list[tuple[int, str]] = []
    n_seen = 0
    with LEGAL_CORPUS.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            n_seen += 1
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                text = line.strip()
            if not text:
                continue
            item = (line_no, text)
            if len(reservoir) < sample_texts:
                reservoir.append(item)
            else:
                j = rng.randrange(n_seen)
                if j < sample_texts:
                    reservoir[j] = item
    reservoir.sort(key=lambda x: x[0])
    return [t for _, t in reservoir], {"corpus_path": rel(LEGAL_CORPUS), "n_seen_rows": n_seen, "sample_texts": len(reservoir), "sample_line_head": [i for i, _ in reservoir[:10]], "seed": seed}


def build_corpus_mask_cases(tokenizer, texts: list[str], positions_per_text: int, max_len: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed + 9103)
    cases: list[dict[str, Any]] = []
    skipped = Counter()
    special = set(int(x) for x in tokenizer.all_special_ids)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")
    for ti, text in enumerate(texts):
        enc = tokenizer(text, return_attention_mask=True, truncation=True, max_length=max_len, add_special_tokens=True)
        ids = [int(x) for x in enc["input_ids"]]
        attn = [int(x) for x in enc["attention_mask"]]
        cand = [i for i, tid in enumerate(ids) if tid not in special and tid != mask_id and attn[i] == 1]
        if not cand:
            skipped["no_candidate_token"] += 1
            continue
        rng.shuffle(cand)
        for pos in sorted(cand[:positions_per_text]):
            masked = list(ids)
            target_id = masked[pos]
            masked[pos] = int(mask_id)
            cases.append({"text_index": ti, "input_ids": masked, "attention_mask": attn, "pos": pos, "target_id": int(target_id), "seq_len": len(masked)})
    meta = {"n_texts": len(texts), "positions_per_text": positions_per_text, "max_len": max_len, "n_cases": len(cases), "skipped": dict(skipped), "seq_len": qstats(c["seq_len"] for c in cases)}
    return cases, meta


def pad_1pos_batch(batch: list[dict[str, Any]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(c["input_ids"]) for c in batch)
    ids, attn = [], []
    for c in batch:
        dn = max_len - len(c["input_ids"])
        ids.append(c["input_ids"] + [pad_id] * dn)
        attn.append(c["attention_mask"] + [0] * dn)
    pos = torch.tensor([int(c["pos"]) for c in batch], dtype=torch.long, device=device)
    tgt = torch.tensor([int(c["target_id"]) for c in batch], dtype=torch.long, device=device)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(attn, dtype=torch.long, device=device), pos, tgt


def collect_corpus_states(model, tokenizer, device: torch.device, cases: list[dict[str, Any]], batch_size: int, max_alignment_cases: int) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if max_alignment_cases and len(cases) > max_alignment_cases:
        cases = cases[:max_alignment_cases]
    n_layers = int(getattr(model.config, "num_hidden_layers")) + 1
    layer_chunks: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    eq_abs_max = 0.0
    eq_logprob_abs_max = 0.0
    final_lp_vals = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            input_ids, attn, pos, tgt = pad_1pos_batch(batch, int(pad_id), device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            hstates = list(out.hidden_states)
            if len(hstates) != n_layers:
                raise RuntimeError(f"hidden-state count {len(hstates)} != {n_layers}")
            mb = torch.arange(input_ids.shape[0], device=device)
            final_h = hstates[-1][mb, pos, :]
            cls_logits = model.cls(final_h).float()
            direct_logits = out.logits[mb, pos, :].float()
            eq_abs_max = max(eq_abs_max, float((cls_logits - direct_logits).abs().max().item()))
            cls_lp = F.log_softmax(cls_logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            direct_lp = F.log_softmax(direct_logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            eq_logprob_abs_max = max(eq_logprob_abs_max, float((cls_lp - direct_lp).abs().max().item()))
            final_lp_vals.extend(float(x) for x in direct_lp.detach().cpu().tolist())
            for li, hs in enumerate(hstates):
                layer_chunks[li].append(hs[mb, pos, :].detach().float().cpu())
            del out, input_ids, attn, pos, tgt
    states = [torch.cat(chunks, dim=0) for chunks in layer_chunks]
    meta = {
        "n_state_cases": int(states[0].shape[0]) if states else 0,
        "n_layers": n_layers,
        "hidden_size": int(states[0].shape[1]) if states else None,
        "final_head_equivalence_max_abs_logit": eq_abs_max,
        "final_head_equivalence_max_abs_target_logprob": eq_logprob_abs_max,
        "final_target_logprob": qstats(final_lp_vals),
    }
    return states, meta


class AlignmentPack:
    def __init__(self, states: list[torch.Tensor], ridge_lambda: float):
        self.n_layers = len(states)
        self.hidden = int(states[0].shape[1])
        self.mean: list[torch.Tensor] = []
        self.std: list[torch.Tensor] = []
        self.W: list[torch.Tensor | None] = []
        y = states[-1].float()
        y_mean = y.mean(dim=0)
        y_std = y.std(dim=0).clamp_min(1e-5)
        self.final_mean = y_mean
        self.final_std = y_std
        eye = torch.eye(self.hidden, dtype=torch.float32)
        for li, x0 in enumerate(states):
            x = x0.float()
            xm = x.mean(dim=0)
            xs = x.std(dim=0).clamp_min(1e-5)
            self.mean.append(xm)
            self.std.append(xs)
            if li == self.n_layers - 1:
                self.W.append(eye.clone())
                continue
            xc = x - xm
            yc = y - y_mean
            xtx = (xc.T @ xc) / max(1, x.shape[0])
            xty = (xc.T @ yc) / max(1, x.shape[0])
            lam = float(ridge_lambda) * float(torch.diag(xtx).mean().item() + 1e-6)
            try:
                w = torch.linalg.solve(xtx + lam * eye, xty)
            except Exception:
                w = torch.linalg.lstsq(xtx + lam * eye, xty).solution
            self.W.append(w.float().cpu())
        self._device_cache: dict[str, Any] = {}

    def to_device(self, device: torch.device) -> dict[str, Any]:
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = {
                "mean": [x.to(device) for x in self.mean],
                "std": [x.to(device) for x in self.std],
                "final_mean": self.final_mean.to(device),
                "final_std": self.final_std.to(device),
                "W": [w.to(device) if w is not None else None for w in self.W],
            }
        return self._device_cache[key]

    def transform(self, h: torch.Tensor, layer_idx: int, mode: str, device: torch.device) -> torch.Tensor:
        if mode == "raw" or layer_idx == self.n_layers - 1:
            return h
        d = self.to_device(device)
        if mode == "stat":
            return (h - d["mean"][layer_idx]) / d["std"][layer_idx] * d["final_std"] + d["final_mean"]
        if mode == "ridge":
            return (h - d["mean"][layer_idx]).matmul(d["W"][layer_idx]) + d["final_mean"]
        raise ValueError(mode)

    def summary(self) -> dict[str, Any]:
        out = {"n_layers": self.n_layers, "hidden_size": self.hidden, "layers": {}}
        for li in range(self.n_layers):
            w = self.W[li]
            out["layers"][str(li)] = {
                "layer_mean_norm": float(self.mean[li].norm().item()),
                "layer_std_mean": float(self.std[li].mean().item()),
                "final_mean_norm": float(self.final_mean.norm().item()),
                "final_std_mean": float(self.final_std.mean().item()),
                "ridge_W_fro_norm": float(w.norm().item()) if w is not None else None,
            }
        return out


# --------------------------- bridge readout ---------------------------


def read_bridge_pairs(limit: int) -> list[dict[str, Any]]:
    pairs = []
    with BRIDGE_PAIRS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    if limit and len(pairs) > limit:
        pairs = pairs[:limit]
    return pairs


def find_last_subseq(full_ids: list[int], sub_ids: list[int]) -> int | None:
    n = len(sub_ids)
    if n == 0:
        return None
    for i in range(len(full_ids) - n, -1, -1):
        if full_ids[i:i+n] == sub_ids:
            return i
    return None


def bridge_text(pair: dict[str, Any], ctx_key: str, alt_word: str) -> str:
    consequence = pair["consequence_masked"].replace("__TARGET__", alt_word)
    if ctx_key == "erased":
        return consequence
    return pair[f"context_{ctx_key}"] + " " + consequence


def make_bridge_cases(pairs: list[dict[str, Any]], tokenizer, max_len: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = []
    failures = []
    mask_id = int(tokenizer.mask_token_id)
    for local_idx, p in enumerate(pairs):
        for ctx_key in ["AB", "BA", "erased"]:
            for alt_key, alt_word in [("alt0", p["alt_0"]), ("alt1", p["alt_1"] )]:
                text = bridge_text(p, ctx_key, alt_word)
                enc = tokenizer(text, return_attention_mask=True, truncation=True, max_length=max_len, add_special_tokens=True)
                input_ids = [int(x) for x in enc["input_ids"]]
                target_ids = tokenizer.encode(" " + alt_word, add_special_tokens=False)
                start = find_last_subseq(input_ids, [int(x) for x in target_ids])
                token_form = "leading_space"
                if start is None:
                    target_ids = tokenizer.encode(alt_word, add_special_tokens=False)
                    start = find_last_subseq(input_ids, [int(x) for x in target_ids])
                    token_form = "bare"
                target_ids = [int(x) for x in target_ids]
                if start is None or not target_ids:
                    failures.append({"pair_id": p.get("pair_id"), "ctx_key": ctx_key, "alt_key": alt_key, "alt_word": alt_word, "reason": "target_subseq_not_found"})
                    continue
                masked = list(input_ids)
                positions = []
                for j in range(len(target_ids)):
                    positions.append(start + j)
                    masked[start + j] = mask_id
                cases.append({"case_id": len(cases), "local_pair_index": local_idx, "pair_id": p.get("pair_id"), "ctx_key": ctx_key, "alt_key": alt_key, "alt_word": alt_word, "target_ids": target_ids, "target_positions": positions, "input_ids": masked, "attention_mask": [int(x) for x in enc["attention_mask"]], "seq_len": len(masked), "target_token_len": len(target_ids), "token_form": token_form})
    complete = {i for i, n in Counter(c["local_pair_index"] for c in cases).items() if n == 6}
    return cases, {"n_input_pairs": len(pairs), "n_cases": len(cases), "n_complete_pairs": len(complete), "n_failures": len(failures), "failures_head": failures[:20], "target_token_len_hist": dict(Counter(str(c["target_token_len"]) for c in cases))}


def pad_cases_multi(batch: list[dict[str, Any]], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(c["input_ids"]) for c in batch)
    ids, attn = [], []
    for c in batch:
        dn = max_len - len(c["input_ids"])
        ids.append(c["input_ids"] + [pad_id] * dn)
        attn.append(c["attention_mask"] + [0] * dn)
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(attn, dtype=torch.long, device=device)


def score_multi_mask_cases(model, tokenizer, device: torch.device, cases: list[dict[str, Any]], align: AlignmentPack, modes: list[str], batch_size: int) -> dict[str, dict[int, dict[tuple[Any, ...], float]]]:
    n_layers = align.n_layers
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    scores: dict[str, dict[int, dict[tuple[Any, ...], float]]] = {m: {li: {} for li in range(n_layers)} for m in modes}
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start:start + batch_size]
            input_ids, attn = pad_cases_multi(batch, int(pad_id), device)
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            hstates = list(out.hidden_states)
            flat_b, flat_pos, flat_tid, flat_case = [], [], [], []
            for bi, c in enumerate(batch):
                for pos, tid in zip(c["target_positions"], c["target_ids"]):
                    flat_b.append(bi); flat_pos.append(int(pos)); flat_tid.append(int(tid)); flat_case.append(bi)
            fb = torch.tensor(flat_b, dtype=torch.long, device=device)
            fp = torch.tensor(flat_pos, dtype=torch.long, device=device)
            ft = torch.tensor(flat_tid, dtype=torch.long, device=device)
            fc = torch.tensor(flat_case, dtype=torch.long, device=device)
            for li, hs in enumerate(hstates):
                h = hs[fb, fp, :]
                for mode in modes:
                    hh = align.transform(h, li, mode, device)
                    logits = model.cls(hh)
                    vals = F.log_softmax(logits.float(), dim=-1).gather(-1, ft.unsqueeze(-1)).squeeze(-1)
                    sums = torch.zeros(len(batch), dtype=torch.float32, device=device)
                    sums.index_add_(0, fc, vals.float())
                    for bi, c in enumerate(batch):
                        scores[mode][li][(int(c["local_pair_index"]), c["ctx_key"], c["alt_key"])] = float(sums[bi].item())
            del out, input_ids, attn
    return scores


def materialize_bridge_rows(target: str, pairs: list[dict[str, Any]], scores: dict[str, dict[int, dict[tuple[Any, ...], float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, by_layer in scores.items():
        for li, smap in by_layer.items():
            for i, p in enumerate(pairs):
                keys = [(i, ctx, alt) for ctx in ["AB", "BA", "erased"] for alt in ["alt0", "alt1"]]
                if any(k not in smap for k in keys):
                    continue
                s_AB_0 = smap[(i, "AB", "alt0")]; s_AB_1 = smap[(i, "AB", "alt1")]
                s_BA_0 = smap[(i, "BA", "alt0")]; s_BA_1 = smap[(i, "BA", "alt1")]
                s_er_0 = smap[(i, "erased", "alt0")]; s_er_1 = smap[(i, "erased", "alt1")]
                if p["correct_AB"] == p["alt_0"]:
                    margin_AB = s_AB_0 - s_AB_1
                    margin_BA = s_BA_1 - s_BA_0
                    erased_bias = s_er_0 - s_er_1
                else:
                    margin_AB = s_AB_1 - s_AB_0
                    margin_BA = s_BA_0 - s_BA_1
                    erased_bias = s_er_1 - s_er_0
                rows.append({
                    "target": target, "decoder_mode": mode, "layer_index": li, "layer": "emb" if li == 0 else f"L{li}",
                    "pair_id": p.get("pair_id"), "family": p.get("family"), "template_id": p.get("template_id"), "alt_type": p.get("alt_type"),
                    "alt_0": p.get("alt_0"), "alt_1": p.get("alt_1"), "correct_AB": p.get("correct_AB"), "correct_BA": p.get("correct_BA"),
                    "margin_AB": margin_AB, "margin_BA": margin_BA, "four_cell_M": margin_AB + margin_BA,
                    "both_correct": margin_AB > 0 and margin_BA > 0,
                    "swap_both_correct": margin_AB < 0 and margin_BA < 0,
                    "erased_bias": erased_bias, "s_AB_alt0": s_AB_0, "s_AB_alt1": s_AB_1, "s_BA_alt0": s_BA_0, "s_BA_alt1": s_BA_1, "s_erased_alt0": s_er_0, "s_erased_alt1": s_er_1,
                })
    return rows


# --------------------------- EWoK readout ---------------------------


@dataclass
class ScoreTask:
    rec_i: int
    key: str
    sentence: str
    start_boundary: int


def load_ewok_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL_EWOK.glob("*.jsonl")):
        domain = path.stem
        with path.open("r", encoding="utf-8") as f:
            for local_index, line in enumerate(f):
                if line.strip():
                    raw = json.loads(line)
                    raw["_domain"] = domain; raw["_local_index"] = local_index; raw["_global_index"] = len(rows)
                    rows.append(raw)
    return rows


def load_ewok_transition_selection(max_per_bucket: int, seed: int, row_limit: int) -> tuple[dict[int, set[str]], dict[str, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    with EWOK_TRANSITIONS.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            idx = int(r["global_index"])
            a_correct = parse_bool(r.get("a_saved_model_correct_flag")); b_correct = parse_bool(r.get("b_saved_model_correct_flag"))
            a_stable = parse_bool(r.get("a_conditional_reversal_failure_stable")); b_stable = parse_bool(r.get("b_conditional_reversal_failure_stable"))
            if parse_bool(r.get("added_stable_failure")): buckets["added_stable_failure"].append(idx)
            if parse_bool(r.get("removed_stable_failure")): buckets["removed_stable_failure"].append(idx)
            if a_stable and b_stable: buckets["shared_stable_failure"].append(idx)
            if a_correct and b_correct: buckets["shared_correct"].append(idx)
            if a_correct and not b_correct: buckets["base_only_correct"].append(idx)
            if (not a_correct) and b_correct: buckets["scale_only_correct"].append(idx)
    rng = random.Random(seed)
    selected: dict[int, set[str]] = defaultdict(set)
    raw_counts: dict[str, int] = {}
    sampled_counts: dict[str, int] = {}
    for bucket, vals0 in sorted(buckets.items()):
        vals = sorted(set(vals0)); raw_counts[bucket] = len(vals)
        if max_per_bucket and len(vals) > max_per_bucket:
            v2 = list(vals); rng.shuffle(v2); vals = sorted(v2[:max_per_bucket])
        sampled_counts[bucket] = len(vals)
        for idx in vals:
            selected[idx].add(bucket)
    if row_limit and len(selected) > row_limit:
        keys = sorted(selected)
        rng2 = random.Random(seed + 17); rng2.shuffle(keys); keep = set(keys[:row_limit])
        selected = {idx: ss for idx, ss in selected.items() if idx in keep}
    return selected, {"raw": raw_counts, "sampled_pre_union": sampled_counts, "union": len(selected)}


def ewok_completion_task(rec_i: int, key: str, context: str, target: str) -> ScoreTask:
    context = context.rstrip(); target = target.strip()
    sentence = (context + " " + target).strip() if context else target
    completion = " " + target if context else target
    return ScoreTask(rec_i, key, sentence, len(sentence) - len(completion))


def build_ewok_tasks(rows: list[dict[str, Any]], selected: list[int], bucket_map: dict[int, set[str]]) -> tuple[list[dict[str, Any]], list[ScoreTask]]:
    recs = []
    tasks = []
    for rec_i, idx in enumerate(selected):
        row = rows[idx]
        rec = {"global_index": row["_global_index"], "domain": row["_domain"], "local_index": row["_local_index"], "ContextType": row.get("ContextType"), "ContextDiff": row.get("ContextDiff"), "TargetDiff": row.get("TargetDiff"), "ConceptA": row.get("ConceptA"), "ConceptB": row.get("ConceptB"), "buckets": ";".join(sorted(bucket_map.get(idx, set()))), "target1": row["Target1"], "target2": row["Target2"]}
        recs.append(rec)
        c1, c2, t1, t2 = row["Context1"], row["Context2"], row["Target1"], row["Target2"]
        tasks.extend([ewok_completion_task(rec_i, "s11", c1, t1), ewok_completion_task(rec_i, "s21", c2, t1), ewok_completion_task(rec_i, "s12", c1, t2), ewok_completion_task(rec_i, "s22", c2, t2)])
    return recs, tasks


def ewok_examples_from_tasks(tasks: list[ScoreTask], tokenizer) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    acc: dict[tuple[int, str], dict[str, Any]] = {}
    examples = []
    mask_id = int(tokenizer.mask_token_id)
    for t in tasks:
        sid = (t.rec_i, t.key)
        acc.setdefault(sid, {"n_tokens": 0})
        enc = tokenizer(t.sentence, return_offsets_mapping=True, return_tensors=None)
        token_ids = [int(x) for x in enc["input_ids"]]
        attention = [int(x) for x in enc["attention_mask"]]
        offsets = list(enc["offset_mapping"])
        selected = [pos for pos, (a, b) in enumerate(offsets) if b > t.start_boundary]
        for pos in selected:
            cur = list(token_ids); cur[pos] = mask_id
            examples.append({"score_id": sid, "input_ids": cur, "attention_mask": attention, "pos": int(pos), "target_id": int(token_ids[pos]), "length": len(cur)})
    examples.sort(key=lambda x: x["length"])
    return examples, acc


def score_ewok_examples(model, tokenizer, device: torch.device, examples: list[dict[str, Any]], align: AlignmentPack, modes: list[str], batch_size: int) -> dict[str, dict[int, dict[tuple[int, str], dict[str, Any]]]]:
    n_layers = align.n_layers
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out_scores: dict[str, dict[int, dict[tuple[int, str], dict[str, Any]]]] = {m: {li: defaultdict(lambda: {"sum": 0.0, "n_tokens": 0}) for li in range(n_layers)} for m in modes}
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            max_len = max(ex["length"] for ex in batch)
            ids = torch.tensor([ex["input_ids"] + [int(pad_id)] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=device)
            attn = torch.tensor([ex["attention_mask"] + [0] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=device)
            pos = torch.tensor([int(ex["pos"]) for ex in batch], dtype=torch.long, device=device)
            tgt = torch.tensor([int(ex["target_id"]) for ex in batch], dtype=torch.long, device=device)
            mb = torch.arange(ids.shape[0], device=device)
            model_out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
            for li, hs in enumerate(model_out.hidden_states):
                h = hs[mb, pos, :]
                for mode in modes:
                    hh = align.transform(h, li, mode, device)
                    logits = model.cls(hh)
                    vals = F.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).detach().cpu().tolist()
                    for j, ex in enumerate(batch):
                        d = out_scores[mode][li][ex["score_id"]]
                        d["sum"] += float(vals[j]); d["n_tokens"] += 1
            del model_out, ids, attn
    return out_scores


def materialize_ewok_rows(target: str, recs: list[dict[str, Any]], scores: dict[str, dict[int, dict[tuple[int, str], dict[str, Any]]]]) -> list[dict[str, Any]]:
    rows = []
    for mode, by_layer in scores.items():
        for li, smap in by_layer.items():
            for rec_i, rec in enumerate(recs):
                row = dict(rec)
                row.update({"target": target, "decoder_mode": mode, "layer_index": li, "layer": "emb" if li == 0 else f"L{li}"})
                ok = True
                for k in ["s11", "s21", "s12", "s22"]:
                    d = smap.get((rec_i, k))
                    if not d or d.get("n_tokens", 0) == 0:
                        ok = False; row[f"{k}_sum"] = float("nan"); row[f"{k}_n_tokens"] = 0
                    else:
                        row[f"{k}_sum"] = float(d["sum"]); row[f"{k}_n_tokens"] = int(d["n_tokens"])
                s11, s21, s12, s22 = [safe_float(row[f"{k}_sum"]) for k in ["s11", "s21", "s12", "s22"]]
                m1 = s11 - s21; m2 = s22 - s12
                wc1 = s11 - s12; wc2 = s22 - s21
                inter = (s11 + s22) - (s12 + s21)
                row.update({
                    "official_margin_t1_sum": m1, "official_margin_t2_sum": m2, "within_context_margin_c1_sum": wc1, "within_context_margin_c2_sum": wc2, "interaction_sum": inter,
                    "t1_correct": bool(ok and m1 > 0), "t2_correct": bool(ok and m2 > 0), "both_official_positive": bool(ok and m1 > 0 and m2 > 0),
                    "both_swapped_positive": bool(ok and m1 < 0 and m2 < 0), "both_within_context_positive": bool(ok and wc1 > 0 and wc2 > 0), "interaction_positive": bool(ok and inter > 0),
                })
                row["stable_conditional_failure"] = bool((not row["t1_correct"]) and (not row["interaction_positive"]) and (not row["both_within_context_positive"]))
                rows.append(row)
    return rows


# --------------------------- GlobalPIQA readout ---------------------------


def load_gp_always_wrong_ids() -> set[str]:
    if not GP_ANATOMY.exists():
        return set()
    d = read_json(GP_ANATOMY)
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    return {str(r["example_id"]) for r in rows if r.get("n_ok") == 0}


def gp_dataloader_args(model_root: Path, batch_size: int, non_causal_batch_size: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=(GP_DATA_ROOT / "global_piqa_parallel").resolve(), task="global_piqa_parallel", model_path_or_name=str(model_root.resolve()), backend="mlm", output_dir=OUT_ROOT, images_path=None, image_split=None, image_template=None, revision_name=None,
        min_temperature=1.0, max_temperature=None, temperature_interval=0.05, batch_size=batch_size, non_causal_batch_size=non_causal_batch_size, full_sentence_scores=False, save_predictions=False,
    )


def run_gp_rows(model, device: torch.device, target: str, model_path: Path, align: AlignmentPack, modes: list[str], batch_size: int, noncausal_batch_size: int) -> list[dict[str, Any]]:
    dl_args = gp_dataloader_args(model_path, batch_size, noncausal_batch_size)
    dataloader = get_dataloader(dl_args)
    always_wrong = load_gp_always_wrong_ids()
    n_layers = align.n_layers
    rows = []
    processed = 0
    with torch.no_grad():
        for raw_sentences, sentence_dict, labels, metadatas, uids, images in dataloader:
            num_sentences = len([k for k in sentence_dict if k.endswith("attn_mask")])
            prefixes = [f"sentence_{i}" for i in range(num_sentences)]
            cand_scores: dict[str, dict[int, list[list[float]]]] = {m: {li: [[] for _ in prefixes] for li in range(n_layers)} for m in modes}
            cand_lengths: list[list[int]] = []
            for ci, prefix in enumerate(prefixes):
                tokens_all = sentence_dict[f"{prefix}_tokens"]
                attn_all = sentence_dict[f"{prefix}_attn_mask"]
                idx_all = sentence_dict[f"{prefix}_indices"]
                targets_all = sentence_dict[f"{prefix}_targets"]
                indiv: dict[str, dict[int, list[float]]] = {m: {li: [] for li in range(n_layers)} for m in modes}
                for start in range(0, tokens_all.shape[0], noncausal_batch_size):
                    tokens = tokens_all[start:start + noncausal_batch_size].to(device)
                    attn = attn_all[start:start + noncausal_batch_size].to(device)
                    indices = idx_all[start:start + noncausal_batch_size].to(device)
                    targets = targets_all[start:start + noncausal_batch_size].to(device)
                    out = model(input_ids=tokens, attention_mask=attn, output_hidden_states=True, return_dict=True)
                    mb = torch.arange(tokens.shape[0], device=device)
                    for li, hs in enumerate(out.hidden_states):
                        h = hs[mb, indices, :]
                        for mode in modes:
                            hh = align.transform(h, li, mode, device)
                            logits = model.cls(hh)
                            vals = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                            indiv[mode][li].extend(float(x) for x in vals.detach().cpu().tolist())
                    del out, tokens, attn
                lengths = [int(x) for x in sentence_dict[f"{prefix}_examples_per_batch"]]
                cand_lengths.append(lengths)
                for mode in modes:
                    for li in range(n_layers):
                        vals = indiv[mode][li]
                        cur = 0; scores = []
                        for n_ex in lengths:
                            seg = vals[cur:cur+n_ex]; cur += n_ex
                            scores.append(float(sum(seg) / max(n_ex, 1)) if seg else float("nan"))
                        cand_scores[mode][li][ci] = scores
            for bi in range(len(labels)):
                lab = int(labels[bi]); uid = str(uids[bi]); rot = (lab + 1) % num_sentences
                prompt = raw_sentences[bi].get("prefixes", [""])[0]
                completions = raw_sentences[bi].get("completions", [])
                for mode in modes:
                    for li in range(n_layers):
                        scores = [cand_scores[mode][li][ci][bi] for ci in range(num_sentences)]
                        order = sorted(range(num_sentences), key=lambda j: scores[j], reverse=True)
                        choice = order[0]
                        rank = order.index(lab) + 1
                        rot_rank = order.index(rot) + 1
                        rows.append({
                            "target": target, "decoder_mode": mode, "layer_index": li, "layer": "emb" if li == 0 else f"L{li}", "mode": "parallel", "example_id": uid, "is_hard52": uid in always_wrong,
                            "label": lab, "rotated_label": rot, "choice": choice, "true_correct": choice == lab, "rotated_correct": choice == rot,
                            "true_rank": rank, "rotated_rank": rot_rank, "top_minus_true": scores[choice] - scores[lab], "top_minus_rotated": scores[choice] - scores[rot],
                            "true_minus_best_incorrect": scores[lab] - max(scores[j] for j in range(num_sentences) if j != lab),
                            "rotated_minus_best_other": scores[rot] - max(scores[j] for j in range(num_sentences) if j != rot),
                            "scores_json": json.dumps(scores, ensure_ascii=False), "completion_token_lengths_json": json.dumps([cand_lengths[ci][bi] for ci in range(num_sentences)]),
                            "prompt": prompt, "completions_json": json.dumps(completions, ensure_ascii=False),
                        })
                processed += 1
            print(json.dumps({"event": "gp_batch_done", "target": target, "processed": processed, "utc": now()}), flush=True)
    return rows


# --------------------------- summaries ---------------------------


def summarize_bool_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if parse_bool(r.get(key))) / len(rows)


def summarize_by_mode_layer(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["decoder_mode"]), int(r["layer_index"]))].append(r)
    for (mode, li), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        key = f"{mode}:L{li}"
        if kind == "bridge":
            out[key] = {"n": len(rs), "M": qstats(r["four_cell_M"] for r in rs), "both_correct_frac": summarize_bool_metric(rs, "both_correct"), "swap_both_correct_frac": summarize_bool_metric(rs, "swap_both_correct"), "margin_AB": qstats(r["margin_AB"] for r in rs), "margin_BA": qstats(r["margin_BA"] for r in rs)}
        elif kind == "ewok":
            out[key] = {"n": len(rs), "interaction": qstats(r["interaction_sum"] for r in rs), "both_official_frac": summarize_bool_metric(rs, "both_official_positive"), "both_swapped_frac": summarize_bool_metric(rs, "both_swapped_positive"), "t1_accuracy": summarize_bool_metric(rs, "t1_correct"), "stable_failure_frac": summarize_bool_metric(rs, "stable_conditional_failure")}
        elif kind == "gp":
            hard = [r for r in rs if parse_bool(r.get("is_hard52"))]
            def sub(grp: list[dict[str, Any]]) -> dict[str, Any]:
                return {"n": len(grp), "true_accuracy": summarize_bool_metric(grp, "true_correct"), "rotated_accuracy": summarize_bool_metric(grp, "rotated_correct"), "top_minus_true": qstats(r["top_minus_true"] for r in grp), "true_minus_best_incorrect": qstats(r["true_minus_best_incorrect"] for r in grp), "true_rank_counts": dict(Counter(str(r["true_rank"]) for r in grp))}
            out[key] = {"all": sub(rs), "hard52": sub(hard)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=160)
    ap.add_argument("--n_corpus_texts", type=int, default=768)
    ap.add_argument("--positions_per_text", type=int, default=4)
    ap.add_argument("--max_alignment_cases", type=int, default=3072)
    ap.add_argument("--corpus_max_len", type=int, default=192)
    ap.add_argument("--ridge_lambda", type=float, default=1e-2)
    ap.add_argument("--bridge_limit", type=int, default=0)
    ap.add_argument("--bridge_max_len", type=int, default=160)
    ap.add_argument("--ewok_max_per_bucket", type=int, default=160)
    ap.add_argument("--ewok_row_limit", type=int, default=0)
    ap.add_argument("--skip_bridge", action="store_true")
    ap.add_argument("--skip_ewok", action="store_true")
    ap.add_argument("--skip_globalpiqa", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--score_batch_size", type=int, default=96)
    ap.add_argument("--gp_batch_size", type=int, default=8)
    ap.add_argument("--gp_noncausal_batch_size", type=int, default=64)
    ap.add_argument("--decoder_modes", nargs="+", default=["raw", "stat", "ridge"], choices=["raw", "stat", "ridge"])
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.threads))
    OUT_ROOT.mkdir(parents=True, exist_ok=True); HF_CACHE.mkdir(parents=True, exist_ok=True)
    target_dir = OUT_ROOT / args.target
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(TARGETS[args.target]["model_path"])
    readiness = {"model_path": rel(model_path), "exists": model_path.exists(), "config": (model_path / "config.json").exists(), "model": (model_path / "model.safetensors").exists(), "optional": bool(TARGETS[args.target].get("optional")), "legal_corpus": rel(LEGAL_CORPUS), "legal_corpus_exists": LEGAL_CORPUS.exists(), "bridge_pairs": rel(BRIDGE_PAIRS), "bridge_pairs_exists": BRIDGE_PAIRS.exists()}
    pf = {"status": "DECODER_ALIGNMENT_PREFLIGHT", "created_utc": now(), "target": args.target, "readiness": readiness, "args": vars(args)}
    write_json(OUT_ROOT / "preflight.json", pf)
    print(json.dumps(pf, ensure_ascii=False), flush=True)
    if args.preflight:
        return
    if not (readiness["exists"] and readiness["config"] and readiness["model"]):
        raise FileNotFoundError(model_path)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    tok, model = load_model_and_tokenizer(args.target, device)
    n_layers = int(getattr(model.config, "num_hidden_layers")) + 1
    adapter_source = TARGETS[args.target].get("adapter_placement_source")
    adapter_note = None
    if adapter_source and Path(adapter_source).exists():
        text = Path(adapter_source).read_text(encoding="utf-8", errors="ignore")
        adapter_note = {"source": rel(Path(adapter_source)), "post_layer_adapter_addition_line_present": "layer_output = layer_output + self.adapter(layer_output)" in text}

    texts, corpus_sample_meta = read_corpus_texts(args.n_corpus_texts, args.seed)
    corpus_cases, corpus_case_meta = build_corpus_mask_cases(tok, texts, args.positions_per_text, args.corpus_max_len, args.seed)
    states, state_meta = collect_corpus_states(model, tok, device, corpus_cases, args.batch_size, args.max_alignment_cases)
    align = AlignmentPack(states, ridge_lambda=args.ridge_lambda)
    alignment_json = {"target": args.target, "label": TARGETS[args.target]["label"], "created_utc": now(), "no_label_corpus_sample": corpus_sample_meta, "corpus_mask_cases": corpus_case_meta, "state_collection": state_meta, "alignment": align.summary(), "adapter_state_placement": adapter_note}
    write_json(target_dir / "no_label_alignment_summary.json", alignment_json)

    result: dict[str, Any] = {"status": "DECODER_ALIGNMENT_TARGET_DONE", "created_utc": now(), "target": args.target, "label": TARGETS[args.target]["label"], "model_path": rel(model_path), "device": str(device), "n_layers": n_layers, "decoder_modes": args.decoder_modes, "alignment_summary": rel(target_dir / "no_label_alignment_summary.json"), "final_head_equivalence": {"max_abs_logit": state_meta["final_head_equivalence_max_abs_logit"], "max_abs_target_logprob": state_meta["final_head_equivalence_max_abs_target_logprob"]}, "adapter_state_placement": adapter_note}

    if not args.skip_bridge:
        pairs = read_bridge_pairs(args.bridge_limit)
        cases, bridge_case_meta = make_bridge_cases(pairs, tok, args.bridge_max_len)
        b_scores = score_multi_mask_cases(model, tok, device, cases, align, args.decoder_modes, args.score_batch_size)
        b_rows = materialize_bridge_rows(args.target, pairs, b_scores)
        write_csv(target_dir / "bridge_layer_rows.csv", b_rows)
        result["bridge"] = {"case_meta": bridge_case_meta, "n_rows": len(b_rows), "summary_by_mode_layer": summarize_by_mode_layer(b_rows, "bridge"), "rows_csv": rel(target_dir / "bridge_layer_rows.csv")}
        print(json.dumps({"event": "bridge_done", "target": args.target, "rows": len(b_rows), "utc": now()}), flush=True)

    if not args.skip_ewok:
        all_rows = load_ewok_rows()
        bucket_map, selection_meta = load_ewok_transition_selection(args.ewok_max_per_bucket, args.seed, args.ewok_row_limit)
        selected = sorted(bucket_map)
        recs, tasks = build_ewok_tasks(all_rows, selected, bucket_map)
        exs, _ = ewok_examples_from_tasks(tasks, tok)
        e_scores = score_ewok_examples(model, tok, device, exs, align, args.decoder_modes, args.score_batch_size)
        e_rows = materialize_ewok_rows(args.target, recs, e_scores)
        write_csv(target_dir / "ewok_layer_rows.csv", e_rows)
        result["ewok"] = {"selection_meta": selection_meta, "n_selected_rows": len(selected), "n_token_examples": len(exs), "n_rows": len(e_rows), "summary_by_mode_layer": summarize_by_mode_layer(e_rows, "ewok"), "rows_csv": rel(target_dir / "ewok_layer_rows.csv")}
        print(json.dumps({"event": "ewok_done", "target": args.target, "selected": len(selected), "rows": len(e_rows), "utc": now()}), flush=True)

    if not args.skip_globalpiqa:
        gp_rows = run_gp_rows(model, device, args.target, model_path, align, args.decoder_modes, args.gp_batch_size, args.gp_noncausal_batch_size)
        write_csv(target_dir / "globalpiqa_layer_rows.csv", gp_rows)
        result["globalpiqa"] = {"n_rows": len(gp_rows), "summary_by_mode_layer": summarize_by_mode_layer(gp_rows, "gp"), "rows_csv": rel(target_dir / "globalpiqa_layer_rows.csv")}
        print(json.dumps({"event": "globalpiqa_done", "target": args.target, "rows": len(gp_rows), "utc": now()}), flush=True)

    out_path = target_dir / "decoder_alignment_target_summary.json"
    result["summary_path"] = rel(out_path)
    write_json(out_path, result)
    print(json.dumps({"status": result["status"], "target": args.target, "summary": rel(out_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
