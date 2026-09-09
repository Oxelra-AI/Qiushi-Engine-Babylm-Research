#!/usr/bin/env python3
"""research: layerwise residual-state readout for conditional alternative binding.

Scientific question
-------------------
The recent natural fixed-alternative miners failed as training sources.  Before
choosing a new architecture, this script asks where the missing conditional
interaction fails inside existing models: do intermediate residual states contain
correct EWoK four-cell interactions or GlobalPIQA alternative margins that are
lost by the final MLM head/path, or is the signal absent throughout the network?

This is a post-hoc diagnostic on existing checkpoints only.  It performs no
training and must not be used to tune official examples for submission.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import re
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_WS = USER_ROOT / "experiments/archive/frontier_consolidation"
OUT_ROOT = Path(os.environ.get(
    "OUT_ROOT",
    str(A01_WS / "data/midlayer_conditional_interaction_probe"),
))
HF_CACHE = OUT_ROOT / "hf_cache"
NOTE = Path(os.environ.get(
    "NOTE",
    str(A01_WS / "notes/midlayer_conditional_interaction_probe.md"),
))

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
    },
    "scale1p75_disabled_80M": {
        "model_path": A01_WS / "data/scale1p75_ewok_inference_ablation/proxy_models/scale1p75_disabled",
        "label": "A02 scale1.75 trained trajectory with adapters disabled at inference, 80M",
        "optional": True,
    },
}

WORD_RE = re.compile(r"\S+")
PUNCT_STRIP = string.punctuation + "“”‘’«»‹›"


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


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def qstats(vals: Iterable[float]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if is_finite(v))
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
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "p10": q(0.10),
        "p90": q(0.90),
        "min": xs[0],
        "max": xs[-1],
    }


def parse_bool(x: Any) -> bool:
    return str(x).strip().lower() in {"true", "1", "yes"}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


# ------------------------- EWoK four-cell readout -------------------------

def load_ewok_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL_EWOK.glob("*.jsonl")):
        domain = path.stem
        with path.open("r", encoding="utf-8") as f:
            for local_index, line in enumerate(f):
                if not line.strip():
                    continue
                raw = json.loads(line)
                raw["_domain"] = domain
                raw["_local_index"] = local_index
                raw["_global_index"] = len(rows)
                rows.append(raw)
    return rows


def load_ewok_transition_buckets(max_per_bucket: int, seed: int) -> tuple[dict[int, set[str]], dict[str, int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    if not EWOK_TRANSITIONS.exists():
        raise FileNotFoundError(EWOK_TRANSITIONS)
    with EWOK_TRANSITIONS.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            idx = int(r["global_index"])
            a_correct = parse_bool(r.get("a_saved_model_correct_flag"))
            b_correct = parse_bool(r.get("b_saved_model_correct_flag"))
            a_stable = parse_bool(r.get("a_conditional_reversal_failure_stable"))
            b_stable = parse_bool(r.get("b_conditional_reversal_failure_stable"))
            if parse_bool(r.get("added_stable_failure")):
                buckets["added_stable_failure"].append(idx)
            if parse_bool(r.get("removed_stable_failure")):
                buckets["removed_stable_failure"].append(idx)
            if a_stable and b_stable:
                buckets["shared_stable_failure"].append(idx)
            if a_correct and b_correct:
                buckets["shared_correct"].append(idx)
            if a_correct and not b_correct:
                buckets["base_only_correct"].append(idx)
            if (not a_correct) and b_correct:
                buckets["scale_only_correct"].append(idx)
    rng = random.Random(seed)
    selected: dict[int, set[str]] = defaultdict(set)
    raw_counts: dict[str, int] = {}
    for bucket, vals in buckets.items():
        vals = sorted(set(vals))
        raw_counts[bucket] = len(vals)
        if max_per_bucket and len(vals) > max_per_bucket:
            # Deterministic stratified sample after shuffling: avoids first-region bias while
            # preserving reproducibility.
            vals2 = list(vals)
            rng.shuffle(vals2)
            vals = sorted(vals2[:max_per_bucket])
        for idx in vals:
            selected[idx].add(bucket)
    return selected, raw_counts


@dataclass
class ScoreTask:
    rec_i: int
    key: str
    sentence: str
    start_boundary: int


def ewok_completion_task(rec_i: int, key: str, context: str, target: str) -> ScoreTask:
    context = context.rstrip(); target = target.strip()
    sentence = (context + " " + target).strip() if context else target
    completion = " " + target if context else target
    start = len(sentence) - len(completion)
    return ScoreTask(rec_i, key, sentence, start)


def build_ewok_tasks(rows: list[dict[str, Any]], indices: list[int], bucket_map: dict[int, set[str]]) -> tuple[list[dict[str, Any]], list[ScoreTask]]:
    recs: list[dict[str, Any]] = []
    tasks: list[ScoreTask] = []
    for rec_i, idx in enumerate(indices):
        row = rows[idx]
        rec = {
            "global_index": row["_global_index"],
            "domain": row["_domain"],
            "local_index": row["_local_index"],
            "ContextType": row.get("ContextType"),
            "ContextDiff": row.get("ContextDiff"),
            "TargetDiff": row.get("TargetDiff"),
            "ConceptA": row.get("ConceptA"),
            "ConceptB": row.get("ConceptB"),
            "buckets": ";".join(sorted(bucket_map.get(idx, set()))),
            "target1": row["Target1"],
            "target2": row["Target2"],
        }
        recs.append(rec)
        c1, c2, t1, t2 = row["Context1"], row["Context2"], row["Target1"], row["Target2"]
        tasks.extend([
            ewok_completion_task(rec_i, "s11", c1, t1),
            ewok_completion_task(rec_i, "s21", c2, t1),
            ewok_completion_task(rec_i, "s12", c1, t2),
            ewok_completion_task(rec_i, "s22", c2, t2),
        ])
    return recs, tasks


class LayerwiseMaskedScorer:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: torch.device, masked_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.masked_batch_size = masked_batch_size
        self.mask_id = tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("Tokenizer has no mask_token_id")
        self.num_layers = int(getattr(model.config, "num_hidden_layers")) + 1
        self.layer_names = ["emb"] + [f"L{i}" for i in range(1, self.num_layers)]

    def _examples_from_tasks(self, tasks: list[ScoreTask]):
        examples: list[dict[str, Any]] = []
        acc: dict[tuple[int, str], dict[str, Any]] = {}
        for t in tasks:
            sid = (t.rec_i, t.key)
            acc.setdefault(sid, {"sums": [0.0] * self.num_layers, "n_tokens": 0, "status": "ok"})
            enc = self.tokenizer(t.sentence, return_offsets_mapping=True, return_tensors=None)
            token_ids = list(enc["input_ids"])
            attention = list(enc["attention_mask"])
            offsets = list(enc["offset_mapping"])
            selected = [pos for pos, (a, b) in enumerate(offsets) if b > t.start_boundary]
            if not selected:
                acc[sid]["status"] = "no_tokens"
                continue
            for pos in selected:
                cur = list(token_ids)
                cur[pos] = self.mask_id
                examples.append({
                    "score_id": sid,
                    "input_ids": cur,
                    "attention_mask": attention,
                    "pos": pos,
                    "target_id": token_ids[pos],
                    "length": len(cur),
                })
        examples.sort(key=lambda x: x["length"])
        return examples, acc

    def score_tasks(self, tasks: list[ScoreTask]) -> dict[tuple[int, str], dict[str, Any]]:
        examples, acc = self._examples_from_tasks(tasks)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        with torch.no_grad():
            for start in range(0, len(examples), self.masked_batch_size):
                batch = examples[start:start + self.masked_batch_size]
                max_len = max(ex["length"] for ex in batch)
                input_ids = torch.tensor([ex["input_ids"] + [pad_id] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                attn = torch.tensor([ex["attention_mask"] + [0] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                pos = torch.tensor([ex["pos"] for ex in batch], dtype=torch.long, device=self.device)
                target = torch.tensor([ex["target_id"] for ex in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
                hstates = list(out.hidden_states)
                if len(hstates) != self.num_layers:
                    raise RuntimeError(f"hidden state count {len(hstates)} != expected {self.num_layers}")
                mb = torch.arange(input_ids.shape[0], device=self.device)
                vals_by_layer: list[list[float]] = []
                for layer_idx, hs in enumerate(hstates):
                    masked_h = hs[mb, pos, :]
                    logits = self.model.cls(masked_h)
                    vals = torch.gather(F.log_softmax(logits.float(), dim=-1), -1, target.unsqueeze(-1)).squeeze(-1)
                    vals_by_layer.append(vals.detach().cpu().tolist())
                for j, ex in enumerate(batch):
                    d = acc[ex["score_id"]]
                    for li in range(self.num_layers):
                        d["sums"][li] += float(vals_by_layer[li][j])
                    d["n_tokens"] += 1
        return acc


def attach_ewok_layer_scores(recs: list[dict[str, Any]], scores: dict[tuple[int, str], dict[str, Any]], layer_names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = ["s11", "s21", "s12", "s22"]
    for rec_i, rec in enumerate(recs):
        per_key = {k: scores.get((rec_i, k)) for k in keys}
        for li, lname in enumerate(layer_names):
            row = dict(rec)
            row["layer_index"] = li
            row["layer"] = lname
            ok = True
            for k in keys:
                d = per_key[k]
                if not d or d.get("n_tokens", 0) == 0:
                    ok = False
                    row[f"{k}_sum"] = float("nan")
                    row[f"{k}_n_tokens"] = 0
                else:
                    row[f"{k}_sum"] = d["sums"][li]
                    row[f"{k}_n_tokens"] = d["n_tokens"]
            s11, s21, s12, s22 = [safe_float(row[f"{k}_sum"]) for k in keys]
            row["official_margin_t1_sum"] = s11 - s21
            row["official_margin_t2_sum"] = s22 - s12
            row["within_context_margin_c1_sum"] = s11 - s12
            row["within_context_margin_c2_sum"] = s22 - s21
            row["interaction_sum"] = (s11 + s22) - (s12 + s21)
            row["t1_correct"] = bool(ok and row["official_margin_t1_sum"] > 0)
            row["t2_correct"] = bool(ok and row["official_margin_t2_sum"] > 0)
            row["both_official_positive"] = bool(row["t1_correct"] and row["t2_correct"])
            row["both_within_context_positive"] = bool(ok and row["within_context_margin_c1_sum"] > 0 and row["within_context_margin_c2_sum"] > 0)
            row["interaction_positive"] = bool(ok and row["interaction_sum"] > 0)
            row["stable_conditional_failure"] = bool((not row["t1_correct"]) and (not row["interaction_positive"]) and (not row["both_within_context_positive"]))
            out.append(row)
    return out


def summarize_layer_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_layer[int(r["layer_index"])].append(r)
    summary: dict[str, Any] = {}
    for li, rs in sorted(by_layer.items()):
        n = len(rs)
        summary[str(li)] = {
            "layer": rs[0]["layer"],
            "n": n,
            "t1_accuracy": sum(r["t1_correct"] for r in rs) / n if n else None,
            "t2_accuracy": sum(r["t2_correct"] for r in rs) / n if n else None,
            "both_official_frac": sum(r["both_official_positive"] for r in rs) / n if n else None,
            "interaction_positive_frac": sum(r["interaction_positive"] for r in rs) / n if n else None,
            "stable_failure_frac": sum(r["stable_conditional_failure"] for r in rs) / n if n else None,
            "interaction_sum": qstats(r["interaction_sum"] for r in rs),
            "official_margin_t1_sum": qstats(r["official_margin_t1_sum"] for r in rs),
        }
    return summary


def summarize_recoverability(rows: list[dict[str, Any]], final_layer_idx: int) -> dict[str, Any]:
    by_rec: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_rec[int(r["global_index"])].append(r)
    final_wrong = 0
    final_stable = 0
    final_wrong_any_mid_t1 = 0
    final_stable_any_mid_both = 0
    final_stable_any_mid_interaction = 0
    final_stable_any_mid_not_stable = 0
    final_t1_false_but_best_layer = Counter()
    for idx, rs in by_rec.items():
        rs_by_li = {int(r["layer_index"]): r for r in rs}
        fin = rs_by_li[final_layer_idx]
        mids = [r for r in rs if int(r["layer_index"]) < final_layer_idx]
        if not fin["t1_correct"]:
            final_wrong += 1
            if any(r["t1_correct"] for r in mids):
                final_wrong_any_mid_t1 += 1
                best = max(mids, key=lambda r: float(r["official_margin_t1_sum"]))
                final_t1_false_but_best_layer[str(best["layer_index"])] += 1
        if fin["stable_conditional_failure"]:
            final_stable += 1
            if any(r["both_official_positive"] for r in mids):
                final_stable_any_mid_both += 1
            if any(r["interaction_positive"] for r in mids):
                final_stable_any_mid_interaction += 1
            if any(not r["stable_conditional_failure"] for r in mids):
                final_stable_any_mid_not_stable += 1
    return {
        "n_rows": len(by_rec),
        "final_wrong_t1": final_wrong,
        "final_stable_failure": final_stable,
        "final_wrong_any_mid_t1_correct": final_wrong_any_mid_t1,
        "final_wrong_any_mid_t1_correct_frac": final_wrong_any_mid_t1 / final_wrong if final_wrong else None,
        "final_stable_any_mid_both_official": final_stable_any_mid_both,
        "final_stable_any_mid_both_official_frac": final_stable_any_mid_both / final_stable if final_stable else None,
        "final_stable_any_mid_interaction_positive": final_stable_any_mid_interaction,
        "final_stable_any_mid_interaction_positive_frac": final_stable_any_mid_interaction / final_stable if final_stable else None,
        "final_stable_any_mid_not_stable": final_stable_any_mid_not_stable,
        "final_stable_any_mid_not_stable_frac": final_stable_any_mid_not_stable / final_stable if final_stable else None,
        "best_mid_layer_for_final_t1_wrong_counts": dict(final_t1_false_but_best_layer),
    }


def summarize_ewok_by_bucket(rows: list[dict[str, Any]], final_layer_idx: int) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for b in str(r.get("buckets", "")).split(";"):
            if b:
                buckets[b].append(r)
    out: dict[str, Any] = {}
    for b, rs in sorted(buckets.items()):
        out[b] = {
            "layer_summary": summarize_layer_rows(rs),
            "recoverability": summarize_recoverability(rs, final_layer_idx),
        }
    return out


def run_ewok_probe(model, tokenizer, device: torch.device, target: str, args: argparse.Namespace, layer_names: list[str]) -> dict[str, Any]:
    rows = load_ewok_rows()
    bucket_map, raw_counts = load_ewok_transition_buckets(args.ewok_max_per_bucket, args.seed)
    selected = sorted(bucket_map)
    if args.ewok_row_limit and len(selected) > args.ewok_row_limit:
        rng = random.Random(args.seed + 17)
        selected2 = list(selected); rng.shuffle(selected2)
        selected = sorted(selected2[:args.ewok_row_limit])
    recs, tasks = build_ewok_tasks(rows, selected, bucket_map)
    scorer = LayerwiseMaskedScorer(model, tokenizer, device, args.masked_batch_size)
    if scorer.layer_names != layer_names:
        raise RuntimeError("layer name mismatch")
    all_layer_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for b0 in range(0, len(recs), args.ewok_row_batch_size):
        rec_batch = recs[b0:b0 + args.ewok_row_batch_size]
        # task rec_i is local to original recs; rebuild for this slice to avoid sparse ids.
        idx_batch = selected[b0:b0 + args.ewok_row_batch_size]
        rec_batch, task_batch = build_ewok_tasks(rows, idx_batch, bucket_map)
        scores = scorer.score_tasks(task_batch)
        layer_rows = attach_ewok_layer_scores(rec_batch, scores, scorer.layer_names)
        for r in layer_rows:
            r["target"] = target
        all_layer_rows.extend(layer_rows)
        print(json.dumps({"event": "ewok_rows_done", "target": target, "rows_done": min(b0 + args.ewok_row_batch_size, len(recs)), "rows_total": len(recs), "utc": now()}), flush=True)
    final_layer_idx = len(layer_names) - 1
    tdir = OUT_ROOT / target
    write_csv(tdir / "ewok_midlayer_rows.csv", all_layer_rows)
    summary = {
        "target": target,
        "n_selected_rows": len(selected),
        "raw_bucket_counts": raw_counts,
        "sampled_bucket_counts": {b: sum(1 for ss in bucket_map.values() if b in ss and (not args.ewok_row_limit or True)) for b in raw_counts},
        "elapsed_sec": round(time.time() - t0, 2),
        "layer_summary_all_selected": summarize_layer_rows(all_layer_rows),
        "recoverability_all_selected": summarize_recoverability(all_layer_rows, final_layer_idx),
        "by_bucket": summarize_ewok_by_bucket(all_layer_rows, final_layer_idx),
        "rows_csv": rel(tdir / "ewok_midlayer_rows.csv"),
    }
    write_json(tdir / "ewok_midlayer_summary.json", summary)
    return summary


# ------------------------- GlobalPIQA layerwise readout -------------------------

def load_gp_always_wrong_ids() -> set[str]:
    if not GP_ANATOMY.exists():
        return set()
    d = json.loads(GP_ANATOMY.read_text(encoding="utf-8"))
    rows = d.get("agreement", {}).get("parallel", {}).get("rows", [])
    return {r["example_id"] for r in rows if r.get("n_ok") == 0}


def gp_dataloader_args(model_root: Path, batch_size: int, non_causal_batch_size: int) -> argparse.Namespace:
    return argparse.Namespace(
        data_path=(GP_DATA_ROOT / "global_piqa_parallel").resolve(),
        task="global_piqa_parallel",
        model_path_or_name=str(model_root.resolve()),
        backend="mlm",
        output_dir=OUT_ROOT,
        images_path=None,
        image_split=None,
        image_template=None,
        revision_name=None,
        min_temperature=1.0,
        max_temperature=None,
        temperature_interval=0.05,
        batch_size=batch_size,
        non_causal_batch_size=non_causal_batch_size,
        full_sentence_scores=False,
        save_predictions=False,
    )


def run_globalpiqa_probe(model, device: torch.device, target: str, model_path: Path, args: argparse.Namespace, layer_names: list[str]) -> dict[str, Any]:
    always_wrong = load_gp_always_wrong_ids()
    dl_args = gp_dataloader_args(model_path, args.gp_batch_size, args.gp_noncausal_batch_size)
    dataloader = get_dataloader(dl_args)
    num_layers = len(layer_names)
    rows_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for raw_sentences, sentence_dict, labels, metadatas, uids, images in dataloader:
            num_sentences = len([key for key in sentence_dict.keys() if key.endswith("attn_mask")])
            prefixes = [f"sentence_{i}" for i in range(num_sentences)]
            # candidate_scores_by_layer[layer][candidate][batch_item]
            candidate_scores_by_layer: list[list[list[float]]] = [[[] for _ in prefixes] for _ in range(num_layers)]
            candidate_lengths: list[list[int]] = []
            for ci, prefix in enumerate(prefixes):
                tokens_all = sentence_dict[f"{prefix}_tokens"]
                attn_all = sentence_dict[f"{prefix}_attn_mask"]
                idx_all = sentence_dict[f"{prefix}_indices"]
                targets_all = sentence_dict[f"{prefix}_targets"]
                individual_by_layer = [[] for _ in range(num_layers)]
                for start in range(0, tokens_all.shape[0], args.gp_noncausal_batch_size):
                    tokens = tokens_all[start:start + args.gp_noncausal_batch_size].to(device)
                    attn = attn_all[start:start + args.gp_noncausal_batch_size].to(device)
                    indices = idx_all[start:start + args.gp_noncausal_batch_size].to(device)
                    targets = targets_all[start:start + args.gp_noncausal_batch_size].to(device)
                    out = model(input_ids=tokens, attention_mask=attn, output_hidden_states=True, return_dict=True)
                    hstates = list(out.hidden_states)
                    mb = torch.arange(tokens.shape[0], device=device)
                    for li, hs in enumerate(hstates):
                        masked_h = hs[mb, indices, :]
                        logits = model.cls(masked_h)
                        vals = torch.gather(F.log_softmax(logits.float(), dim=-1), -1, targets.unsqueeze(-1)).squeeze(-1)
                        individual_by_layer[li].extend(vals.detach().cpu().tolist())
                lengths = []
                for examples_per_item in sentence_dict[f"{prefix}_examples_per_batch"]:
                    lengths.append(int(examples_per_item))
                candidate_lengths.append(lengths)
                for li in range(num_layers):
                    vals = individual_by_layer[li]
                    curr = 0
                    scores: list[float] = []
                    for n_ex in lengths:
                        seg = vals[curr:curr + n_ex]
                        curr += n_ex
                        scores.append(float(sum(seg) / max(n_ex, 1)) if seg else float("nan"))
                    candidate_scores_by_layer[li][ci] = scores
            batch_n = len(labels)
            for bi in range(batch_n):
                lab = int(labels[bi])
                uid = str(uids[bi])
                prompt = raw_sentences[bi].get("prefixes", [""])[0]
                completions = raw_sentences[bi].get("completions", [])
                for li, lname in enumerate(layer_names):
                    scores = [candidate_scores_by_layer[li][ci][bi] for ci in range(num_sentences)]
                    order = sorted(range(num_sentences), key=lambda j: scores[j], reverse=True)
                    choice = order[0]
                    rank = order.index(lab) + 1
                    correct_score = scores[lab]
                    second = scores[order[1]] if len(order) > 1 else float("nan")
                    row = {
                        "target": target,
                        "mode": "parallel",
                        "example_id": uid,
                        "is_hard52": uid in always_wrong,
                        "label": lab,
                        "layer_index": li,
                        "layer": lname,
                        "choice": choice,
                        "correct": choice == lab,
                        "correct_rank": rank,
                        "top_minus_correct": scores[choice] - correct_score,
                        "correct_minus_second_best": correct_score - (second if choice == lab else scores[choice]),
                        "scores_json": json.dumps(scores, ensure_ascii=False),
                        "completion_token_lengths_json": json.dumps([candidate_lengths[ci][bi] for ci in range(num_sentences)]),
                        "prompt": prompt,
                        "completions_json": json.dumps(completions, ensure_ascii=False),
                    }
                    rows_by_layer[li].append(row)
                processed += 1
            print(json.dumps({"event": "gp_batch_done", "target": target, "processed": processed, "utc": now()}), flush=True)
    all_rows: list[dict[str, Any]] = []
    for li in range(num_layers):
        all_rows.extend(rows_by_layer[li])
    tdir = OUT_ROOT / target
    write_csv(tdir / "globalpiqa_midlayer_rows.csv", all_rows)

    def subsummary(rs: list[dict[str, Any]]) -> dict[str, Any]:
        if not rs:
            return {"n": 0}
        n = len(rs)
        correct = sum(r["correct"] for r in rs)
        margins = [float(r["top_minus_correct"]) for r in rs]
        ranks = Counter(str(r["correct_rank"]) for r in rs)
        return {
            "n": n,
            "accuracy": 100.0 * correct / n,
            "correct_rank_counts": dict(ranks),
            "mean_top_minus_correct": statistics.fmean(margins),
            "median_top_minus_correct": statistics.median(margins),
            "small_wrong_margin_le_0p50_nats": sum((not r["correct"]) and float(r["top_minus_correct"]) <= 0.5 for r in rs),
        }

    layer_summary: dict[str, Any] = {}
    for li, rs in sorted(rows_by_layer.items()):
        hard = [r for r in rs if r["is_hard52"]]
        layer_summary[str(li)] = {
            "layer": layer_names[li],
            "all": subsummary(rs),
            "hard52": subsummary(hard),
        }
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        if r["is_hard52"]:
            by_example[str(r["example_id"])].append(r)
    final_layer = num_layers - 1
    final_wrong = 0
    final_wrong_any_mid_correct = 0
    final_wrong_any_mid_rank1or2 = 0
    final_wrong_best_mid_layer = Counter()
    for uid, rs in by_example.items():
        by_li = {int(r["layer_index"]): r for r in rs}
        fin = by_li[final_layer]
        mids = [r for r in rs if int(r["layer_index"]) < final_layer]
        if not fin["correct"]:
            final_wrong += 1
            if any(r["correct"] for r in mids):
                final_wrong_any_mid_correct += 1
                best = max(mids, key=lambda r: -float(r["top_minus_correct"]))
                final_wrong_best_mid_layer[str(best["layer_index"])] += 1
            if any(int(r["correct_rank"]) <= 2 for r in mids):
                final_wrong_any_mid_rank1or2 += 1
    summary = {
        "target": target,
        "elapsed_sec": round(time.time() - t0, 2),
        "n_layers": num_layers,
        "layer_summary": layer_summary,
        "hard52_recoverability": {
            "hard52_rows": len(by_example),
            "final_wrong": final_wrong,
            "final_wrong_any_mid_correct": final_wrong_any_mid_correct,
            "final_wrong_any_mid_correct_frac": final_wrong_any_mid_correct / final_wrong if final_wrong else None,
            "final_wrong_any_mid_rank_le_2": final_wrong_any_mid_rank1or2,
            "final_wrong_any_mid_rank_le_2_frac": final_wrong_any_mid_rank1or2 / final_wrong if final_wrong else None,
            "best_mid_layer_for_final_wrong_counts": dict(final_wrong_best_mid_layer),
        },
        "rows_csv": rel(tdir / "globalpiqa_midlayer_rows.csv"),
    }
    write_json(tdir / "globalpiqa_midlayer_summary.json", summary)
    return summary


def load_model_and_tokenizer(target: str, device: torch.device):
    meta = TARGETS[target]
    path = Path(meta["model_path"])
    if not path.exists():
        if meta.get("optional"):
            return None, None
        raise FileNotFoundError(path)
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True)
    model.eval().to(device)
    return tok, model


def make_note(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research midlayer conditional-interaction probe\n\n")
    lines.append("Post-hoc layerwise decoding of existing checkpoints only; no training and no official-example tuning. The scientific question is whether correct context-conditioned ordering appears in intermediate residual states and is erased by late computation, or is absent throughout the network.\n\n")
    lines.append(f"Combined summary: `{summary['combined_summary']}`\n\n")
    for target, tres in summary["targets"].items():
        lines.append(f"## `{target}` — {tres['label']}\n\n")
        ew = tres.get("ewok", {})
        if ew:
            final_li = str(tres["n_layers"] - 1)
            final = ew["layer_summary_all_selected"].get(final_li, {})
            best_both_li = max(ew["layer_summary_all_selected"], key=lambda k: ew["layer_summary_all_selected"][k].get("both_official_frac") or -1)
            best_int_li = max(ew["layer_summary_all_selected"], key=lambda k: ew["layer_summary_all_selected"][k].get("interaction_positive_frac") or -1)
            rec = ew["recoverability_all_selected"]
            lines.append(f"- EWoK selected rows: {ew['n_selected_rows']}; final t1_acc={final.get('t1_accuracy')}, final both_official={final.get('both_official_frac')}, final stable_failure={final.get('stable_failure_frac')}.\n")
            lines.append(f"- Best EWoK both_official layer {best_both_li} ({ew['layer_summary_all_selected'][best_both_li].get('both_official_frac')}); best interaction-positive layer {best_int_li} ({ew['layer_summary_all_selected'][best_int_li].get('interaction_positive_frac')}).\n")
            lines.append(f"- EWoK recoverability: final_wrong_any_mid_t1={rec.get('final_wrong_any_mid_t1_correct_frac')}, final_stable_any_mid_both={rec.get('final_stable_any_mid_both_official_frac')}, final_stable_any_mid_interaction={rec.get('final_stable_any_mid_interaction_positive_frac')}.\n")
        gp = tres.get("globalpiqa", {})
        if gp:
            final_li = str(tres["n_layers"] - 1)
            final = gp["layer_summary"].get(final_li, {})
            best_hard_li = max(gp["layer_summary"], key=lambda k: gp["layer_summary"][k]["hard52"].get("accuracy") if gp["layer_summary"][k]["hard52"].get("accuracy") is not None else -1)
            best_all_li = max(gp["layer_summary"], key=lambda k: gp["layer_summary"][k]["all"].get("accuracy") if gp["layer_summary"][k]["all"].get("accuracy") is not None else -1)
            rec = gp["hard52_recoverability"]
            lines.append(f"- GlobalPIQA final all_acc={final.get('all',{}).get('accuracy')}, final hard52_acc={final.get('hard52',{}).get('accuracy')}, final hard52_mean_margin={final.get('hard52',{}).get('mean_top_minus_correct')}.\n")
            lines.append(f"- Best GlobalPIQA all layer {best_all_li} ({gp['layer_summary'][best_all_li]['all'].get('accuracy')}); best hard52 layer {best_hard_li} ({gp['layer_summary'][best_hard_li]['hard52'].get('accuracy')}).\n")
            lines.append(f"- GlobalPIQA hard52 recoverability: final_wrong_any_mid_correct={rec.get('final_wrong_any_mid_correct_frac')}, final_wrong_any_mid_rank_le_2={rec.get('final_wrong_any_mid_rank_le_2_frac')}.\n")
        lines.append("\n")
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    NOTE.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["matched_base_80M", "scale1p75_live_80M"], choices=sorted(TARGETS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=159)
    ap.add_argument("--ewok_max_per_bucket", type=int, default=128, help="0 means all rows in each transition bucket")
    ap.add_argument("--ewok_row_limit", type=int, default=0, help="0 means no cap after bucket union")
    ap.add_argument("--ewok_row_batch_size", type=int, default=48)
    ap.add_argument("--masked_batch_size", type=int, default=192)
    ap.add_argument("--gp_batch_size", type=int, default=8)
    ap.add_argument("--gp_noncausal_batch_size", type=int, default=64)
    ap.add_argument("--skip_ewok", action="store_true")
    ap.add_argument("--skip_globalpiqa", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(args.threads, 1))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

    readiness = {}
    for t in args.targets:
        p = Path(TARGETS[t]["model_path"])
        readiness[t] = {"model_path": rel(p), "exists": p.exists(), "config": (p / "config.json").exists(), "model": (p / "model.safetensors").exists(), "optional": bool(TARGETS[t].get("optional"))}
    pf = {"status": "MIDLAYER_PROBE_PREFLIGHT", "created_utc": now(), "readiness": readiness, "args": vars(args)}
    write_json(OUT_ROOT / "preflight.json", pf)
    print(json.dumps(pf, ensure_ascii=False), flush=True)
    if args.preflight:
        return
    missing = [t for t, r in readiness.items() if not (r["exists"] and r["config"] and r["model"]) and not r["optional"]]
    if missing:
        raise FileNotFoundError(f"Missing required targets: {missing}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    combined: dict[str, Any] = {
        "status": "MIDLAYER_CONDITIONAL_INTERACTION_PROBE_DONE",
        "created_utc": now(),
        "boundary": "post-hoc layerwise residual-state decoding on existing checkpoints; no training or official-example tuning",
        "device": str(device),
        "targets": {},
    }
    for t in args.targets:
        if not readiness[t]["exists"]:
            combined["targets"][t] = {"label": TARGETS[t]["label"], "skipped": "missing optional target"}
            continue
        print(json.dumps({"event": "target_load", "target": t, "model_path": readiness[t]["model_path"], "utc": now()}), flush=True)
        tok, model = load_model_and_tokenizer(t, device)
        assert tok is not None and model is not None
        n_layers = int(getattr(model.config, "num_hidden_layers")) + 1
        layer_names = ["emb"] + [f"L{i}" for i in range(1, n_layers)]
        target_summary: dict[str, Any] = {
            "label": TARGETS[t]["label"],
            "model_path": readiness[t]["model_path"],
            "n_layers": n_layers,
        }
        if not args.skip_ewok:
            target_summary["ewok"] = run_ewok_probe(model, tok, device, t, args, layer_names)
        if not args.skip_globalpiqa:
            target_summary["globalpiqa"] = run_globalpiqa_probe(model, device, t, Path(TARGETS[t]["model_path"]), args, layer_names)
        combined["targets"][t] = target_summary
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined_path = OUT_ROOT / "midlayer_conditional_interaction_summary.json"
    combined["combined_summary"] = rel(combined_path)
    combined["note"] = rel(NOTE)
    write_json(combined_path, combined)
    make_note(combined)
    print(json.dumps({"status": combined["status"], "summary": rel(combined_path), "note": rel(NOTE)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
