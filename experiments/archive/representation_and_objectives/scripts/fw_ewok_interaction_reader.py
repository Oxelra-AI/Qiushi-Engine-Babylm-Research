#!/usr/bin/env python3
"""research: EWoK four-cell interaction reader for FW shared-anchor endpoints.

This script adapts the validated research EWoK interaction measurement to arbitrary
completed FW checkpoints.  Unlike the original research script, it does not rely on
old saved-correctness columns tied to legal40/depth baselines.  For each row it
scores the four context-target combinations:
    s11 = Context1 + Target1, s21 = Context2 + Target1,
    s12 = Context1 + Target2, s22 = Context2 + Target2,
and defines the official row decision by whether s11 > s21, matching the saved
EWoK target-score sign convention validated in research.  It then measures whether
wrong rows are stable conditional-reversal failures: changing context does not
make the two targets prefer their matching contexts after target priors cancel.

Scientific boundary: post-endpoint readout only.  No training, no official-example
shaping, and no custom submission scoring.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import os
import re
import statistics
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

USER_ROOT = Path.cwd()
A01_WS = USER_ROOT / "experiments/archive/representation_and_objectives"
A02_WS = USER_ROOT / "experiments/archive/frontier_consolidation"
EVAL_EWOK = A01_WS / "data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered"
OUT_ROOT = A01_WS / "data/fw_ewok_interaction_reader"
NOTE = A01_WS / "notes/fw_ewok_interaction_reader.md"

DEFAULT_TARGETS: dict[str, dict[str, Any]] = {
    "fw_compact_fullbatch_seed43022": {
        "model_path": A02_WS / "training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M",
        "label": "A02 compact same-proposition recurrence anchor, shared16k full-batch seed43022",
    },
    "fw_breadth_rowblock_fullbatch_seed43022": {
        "model_path": A02_WS / "training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M",
        "label": "A02 row-block whole-sentence source-breadth comparator, shared16k full-batch seed43022",
    },
    "fw_breadth_interleaved_fullbatch_seed43022": {
        "model_path": A01_WS / "training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022/hf_model/chck_100M",
        "label": "A01 interleaved whole-sentence source-breadth comparator, shared16k full-batch seed43022",
    },
}

WORD_RE = re.compile(r"\S+")
PUNCT_STRIP = string.punctuation + "“”‘’«»‹›"


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p95": q(0.95), "max": xs[-1]}


def word_toks(text: str):
    toks = []
    for m in WORD_RE.finditer(text):
        raw = m.group(0)
        norm = raw.lower().strip(PUNCT_STRIP)
        toks.append((raw, norm, m.start(), m.end()))
    return toks


def merged_span(toks, i1: int, i2: int):
    if i1 >= i2:
        return None
    return toks[i1][2], toks[i2 - 1][3]


def diff_spans(c1: str, c2: str) -> dict[str, Any]:
    t1 = word_toks(c1); t2 = word_toks(c2)
    sm = difflib.SequenceMatcher(a=[t[1] for t in t1], b=[t[1] for t in t2], autojunk=False)
    c1_spans = []; c2_spans = []; c1_texts = []; c2_texts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        s1 = merged_span(t1, i1, i2); s2 = merged_span(t2, j1, j2)
        if s1:
            c1_spans.append(s1); c1_texts.append(c1[s1[0]:s1[1]])
        if s2:
            c2_spans.append(s2); c2_texts.append(c2[s2[0]:s2[1]])
    return {"c1_spans": c1_spans, "c2_spans": c2_spans, "c1_texts": c1_texts, "c2_texts": c2_texts, "n_hunks": len(c1_spans) + len(c2_spans)}


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def delete_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    parts = []; last = 0
    for s, e in sorted(spans):
        if s > last:
            parts.append(text[last:s])
        last = max(last, e)
    parts.append(text[last:])
    return normalize_spaces("".join(parts))


def replace_spans(text: str, spans: list[tuple[int, int]], repl_texts: list[str]):
    if not spans or len(spans) != len(repl_texts):
        return text, []
    pairs = sorted(zip(spans, repl_texts), key=lambda x: x[0][0])
    out = []; inserted = []; last = 0; cur = 0
    for (s, e), repl in pairs:
        prefix = text[last:s]
        out.append(prefix); cur += len(prefix)
        ins0 = cur; out.append(repl); cur += len(repl); ins1 = cur
        if repl:
            inserted.append((ins0, ins1))
        last = e
    suffix = text[last:]
    out.append(suffix)
    return "".join(out), inserted


@dataclass
class ScoreTask:
    rec_i: int
    key: str
    sentence: str
    spans: list[tuple[int, int]]
    mode: str = "span"


class BatchedPseudoScorer:
    def __init__(self, model, tokenizer, device: torch.device, masked_batch_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.masked_batch_size = masked_batch_size
        self.mask_id = tokenizer.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("Tokenizer has no mask_token_id")

    def _build_examples(self, tasks: list[ScoreTask]):
        examples = []
        acc: dict[tuple[int, str], dict[str, Any]] = {}
        for task in tasks:
            score_id = (task.rec_i, task.key)
            acc.setdefault(score_id, {"sum": 0.0, "n_tokens": 0, "status": "ok"})
            if not task.sentence or not task.spans:
                acc[score_id]["status"] = "empty"; continue
            enc = self.tokenizer(task.sentence, return_offsets_mapping=True, return_tensors=None)
            token_ids = list(enc["input_ids"]); attention = list(enc["attention_mask"]); offsets = list(enc["offset_mapping"])
            selected = []
            if task.mode == "completion":
                start_boundary = min(s for s, _ in task.spans)
                for pos, (a, b) in enumerate(offsets):
                    if b > start_boundary:
                        selected.append(pos)
            else:
                for pos, (a, b) in enumerate(offsets):
                    if any((b > s and a < e) for s, e in task.spans):
                        selected.append(pos)
            if not selected:
                acc[score_id]["status"] = "no_tokens"; continue
            for pos in selected:
                cur = list(token_ids); cur[pos] = self.mask_id
                examples.append({"score_id": score_id, "input_ids": cur, "attention_mask": attention, "pos": pos, "target_id": token_ids[pos], "length": len(cur)})
        return examples, acc

    def score(self, tasks: list[ScoreTask]):
        examples, acc = self._build_examples(tasks)
        examples.sort(key=lambda x: x["length"])
        with torch.no_grad():
            for start in range(0, len(examples), self.masked_batch_size):
                batch = examples[start:start + self.masked_batch_size]
                if not batch:
                    continue
                max_len = max(ex["length"] for ex in batch)
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.tensor([ex["input_ids"] + [pad_id] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                attn = torch.tensor([ex["attention_mask"] + [0] * (max_len - ex["length"]) for ex in batch], dtype=torch.long, device=self.device)
                pos = torch.tensor([ex["pos"] for ex in batch], dtype=torch.long, device=self.device)
                target = torch.tensor([ex["target_id"] for ex in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits if hasattr(out, "logits") else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                masked = logits[mb, pos]
                vals = torch.gather(F.log_softmax(masked, dim=-1), -1, target.unsqueeze(-1)).squeeze(-1)
                for score_id, val in zip([ex["score_id"] for ex in batch], vals.detach().cpu().tolist()):
                    acc[score_id]["sum"] += float(val)
                    acc[score_id]["n_tokens"] += 1
        for d in acc.values():
            n = d["n_tokens"]
            d["mean"] = d["sum"] / n if n else float("nan")
        return acc


def completion_task(rec_i: int, key: str, context: str, target: str) -> ScoreTask:
    context = context.rstrip(); target = target.strip()
    sentence = (context + " " + target).strip() if context else target
    completion = " " + target
    start_char_idx = len(sentence) - len(completion)
    return ScoreTask(rec_i, key, sentence, [(start_char_idx, len(sentence))], "completion")


def span_task(rec_i: int, key: str, sentence: str, spans: list[tuple[int, int]]) -> ScoreTask:
    return ScoreTask(rec_i, key, sentence, spans, "span")


def load_rows() -> list[dict[str, Any]]:
    rows = []
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


def build_batch_records(rows: list[dict[str, Any]], indices: list[int]):
    recs = []
    tasks = []
    for rec_i, idx in enumerate(indices):
        row = rows[idx]
        c1, c2 = row["Context1"], row["Context2"]
        t1, t2 = row["Target1"], row["Target2"]
        ds = diff_spans(c1, c2)
        c1_del = delete_spans(c1, ds["c1_spans"])
        c2_del = delete_spans(c2, ds["c2_spans"])
        rec = {
            "global_index": row["_global_index"],
            "domain": row["_domain"],
            "local_index": row["_local_index"],
            "ContextType": row.get("ContextType"),
            "ContextDiff": row.get("ContextDiff"),
            "TargetDiff": row.get("TargetDiff"),
            "ConceptA": row.get("ConceptA"),
            "ConceptB": row.get("ConceptB"),
            "context_diff_n_hunks": ds["n_hunks"],
            "context_diff_c1_texts_joined": " ||| ".join(ds["c1_texts"]),
            "context_diff_c2_texts_joined": " ||| ".join(ds["c2_texts"]),
            "target1": t1,
            "target2": t2,
            "context1_deleted_diff": c1_del,
            "context2_deleted_diff": c2_del,
            "deleted_contexts_identical": normalize_spaces(c1_del).lower() == normalize_spaces(c2_del).lower(),
        }
        recs.append(rec)
        tasks.extend([
            completion_task(rec_i, "s11", c1, t1), completion_task(rec_i, "s21", c2, t1),
            completion_task(rec_i, "s12", c1, t2), completion_task(rec_i, "s22", c2, t2),
            completion_task(rec_i, "prior_t1", "", t1), completion_task(rec_i, "prior_t2", "", t2),
            completion_task(rec_i, "del_s11", c1_del, t1), completion_task(rec_i, "del_s21", c2_del, t1),
            completion_task(rec_i, "del_s12", c1_del, t2), completion_task(rec_i, "del_s22", c2_del, t2),
        ])
        if ds["c1_spans"] and ds["c2_spans"] and len(ds["c1_spans"]) == len(ds["c2_texts"]) and len(ds["c2_spans"]) == len(ds["c1_texts"]):
            c1_swapped, c1_inserted = replace_spans(c1, ds["c1_spans"], ds["c2_texts"])
            c2_swapped, c2_inserted = replace_spans(c2, ds["c2_spans"], ds["c1_texts"])
            rec["local_context_status"] = "scored"
            rec["context1_swapped_diff"] = c1_swapped
            rec["context2_swapped_diff"] = c2_swapped
            tasks.extend([
                span_task(rec_i, "local_c1_actual", c1, ds["c1_spans"]),
                span_task(rec_i, "local_c1_swapped", c1_swapped, c1_inserted),
                span_task(rec_i, "local_c2_actual", c2, ds["c2_spans"]),
                span_task(rec_i, "local_c2_swapped", c2_swapped, c2_inserted),
            ])
        else:
            rec["local_context_status"] = "not_scored"
    return recs, tasks


def attach_scores(recs: list[dict[str, Any]], scores: dict[tuple[int, str], dict[str, Any]]) -> None:
    keys = ["s11", "s21", "s12", "s22", "prior_t1", "prior_t2", "del_s11", "del_s21", "del_s12", "del_s22", "local_c1_actual", "local_c1_swapped", "local_c2_actual", "local_c2_swapped"]
    for rec_i, rec in enumerate(recs):
        for key in keys:
            d = scores.get((rec_i, key), {"sum": float("nan"), "mean": float("nan"), "n_tokens": 0, "status": "missing"})
            rec[f"{key}_sum"] = d.get("sum", float("nan"))
            rec[f"{key}_mean"] = d.get("mean", float("nan"))
            rec[f"{key}_n_tokens"] = d.get("n_tokens", 0)
            rec[f"{key}_status"] = d.get("status", "missing")
        for form in ["sum", "mean"]:
            s11 = safe_float(rec[f"s11_{form}"]); s21 = safe_float(rec[f"s21_{form}"]); s12 = safe_float(rec[f"s12_{form}"]); s22 = safe_float(rec[f"s22_{form}"])
            d11 = safe_float(rec[f"del_s11_{form}"]); d21 = safe_float(rec[f"del_s21_{form}"]); d12 = safe_float(rec[f"del_s12_{form}"]); d22 = safe_float(rec[f"del_s22_{form}"])
            p1 = safe_float(rec[f"prior_t1_{form}"]); p2 = safe_float(rec[f"prior_t2_{form}"])
            rec[f"official_margin_t1_{form}"] = s11 - s21
            rec[f"official_margin_t2_{form}"] = s22 - s12
            rec[f"within_context_margin_c1_{form}"] = s11 - s12
            rec[f"within_context_margin_c2_{form}"] = s22 - s21
            rec[f"interaction_{form}"] = (s11 + s22) - (s12 + s21)
            rec[f"deletion_interaction_{form}"] = (d11 + d22) - (d12 + d21)
            rec[f"interaction_minus_deletion_{form}"] = rec[f"interaction_{form}"] - rec[f"deletion_interaction_{form}"]
            rec[f"pmi_within_context_margin_c1_{form}"] = (s11 - p1) - (s12 - p2)
            rec[f"pmi_within_context_margin_c2_{form}"] = (s22 - p2) - (s21 - p1)
        rec["local_c1_actual_over_swapped_sum"] = safe_float(rec.get("local_c1_actual_sum")) - safe_float(rec.get("local_c1_swapped_sum"))
        rec["local_c2_actual_over_swapped_sum"] = safe_float(rec.get("local_c2_actual_sum")) - safe_float(rec.get("local_c2_swapped_sum"))
        rec["saved_model_correct_flag"] = rec["official_margin_t1_sum"] > 0
        rec["saved_model_wrong_flag"] = not rec["saved_model_correct_flag"]
        rec["both_official_sum_positive"] = rec["official_margin_t1_sum"] > 0 and rec["official_margin_t2_sum"] > 0
        rec["both_within_context_sum_positive"] = rec["within_context_margin_c1_sum"] > 0 and rec["within_context_margin_c2_sum"] > 0
        rec["both_within_context_mean_positive"] = rec["within_context_margin_c1_mean"] > 0 and rec["within_context_margin_c2_mean"] > 0
        rec["stable_nonpositive_interaction"] = rec["interaction_sum"] <= 0 and rec["interaction_mean"] <= 0
        rec["conditional_reversal_failure_stable"] = bool(rec["saved_model_wrong_flag"] and rec["stable_nonpositive_interaction"] and not rec["both_within_context_sum_positive"] and not rec["both_within_context_mean_positive"])
        rec["local_both_actual_over_swapped_positive"] = rec["local_c1_actual_over_swapped_sum"] > 0 and rec["local_c2_actual_over_swapped_sum"] > 0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    wrong = [r for r in records if r["saved_model_wrong_flag"]]
    stable = [r for r in records if r["conditional_reversal_failure_stable"]]
    def frac(pred, rows=records):
        return sum(1 for r in rows if pred(r)) / len(rows) if rows else None
    return {
        "n": n,
        "accuracy": frac(lambda r: r["saved_model_correct_flag"]),
        "saved_wrong": len(wrong),
        "stable_failure": len(stable),
        "stable_failure_frac_all": len(stable) / n if n else None,
        "stable_failure_frac_wrong": len(stable) / len(wrong) if wrong else None,
        "interaction_sum_all": qstats(r["interaction_sum"] for r in records),
        "interaction_sum_wrong": qstats(r["interaction_sum"] for r in wrong),
        "interaction_mean_wrong": qstats(r["interaction_mean"] for r in wrong),
        "within_both_positive_wrong_frac": frac(lambda r: r["both_within_context_sum_positive"], wrong),
        "local_both_actual_over_swapped_positive_wrong_frac": frac(lambda r: r["local_both_actual_over_swapped_positive"], wrong),
        "deletion_interaction_sum_wrong": qstats(r["deletion_interaction_sum"] for r in wrong),
        "interaction_minus_deletion_sum_wrong": qstats(r["interaction_minus_deletion_sum"] for r in wrong),
    }


def summarize_by(records: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for r in records:
        groups[str(r.get(field) or "")].append(r)
    rows = []
    for k, rs in groups.items():
        s = summarize(rs)
        rows.append({field: k, "n": len(rs), "accuracy": s["accuracy"], "stable_failure": s["stable_failure"], "stable_failure_frac_all": s["stable_failure_frac_all"], "stable_failure_frac_wrong": s["stable_failure_frac_wrong"], "wrong": s["saved_wrong"], "interaction_sum_median": s["interaction_sum_all"].get("median")})
    return sorted(rows, key=lambda x: (-x["stable_failure"], x[field]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def run_target(target_key: str, model_path: Path, device: torch.device, threads: int, row_limit: int, row_offset: int, row_batch_size: int, masked_batch_size: int):
    if threads > 0:
        torch.set_num_threads(threads)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path missing for {target_key}: {model_path}")
    rows = load_rows()
    selected = list(range(len(rows)))
    if row_offset:
        selected = selected[row_offset:]
    if row_limit:
        selected = selected[:row_limit]
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval().to(device)
    scorer = BatchedPseudoScorer(model, tokenizer, device, masked_batch_size)
    out_records = []
    for b0 in range(0, len(selected), row_batch_size):
        idx_batch = selected[b0:b0 + row_batch_size]
        recs, tasks = build_batch_records(rows, idx_batch)
        scores = scorer.score(tasks)
        attach_scores(recs, scores)
        for rec in recs:
            rec["target"] = target_key
        out_records.extend(recs)
        if (b0 // row_batch_size) % 10 == 0:
            print(json.dumps({"event": "rows_done", "target": target_key, "rows_done": len(out_records), "rows_total": len(selected), "utc": now_utc()}), flush=True)
    return {
        "target": target_key,
        "model_path": str(model_path),
        "device": str(device),
        "threads": threads,
        "row_limit": row_limit,
        "row_offset": row_offset,
        "row_batch_size": row_batch_size,
        "masked_batch_size": masked_batch_size,
        "elapsed_sec": round(time.time() - t0, 2),
        "summary": summarize(out_records),
        "by_domain": summarize_by(out_records, "domain"),
        "by_context_diff": summarize_by(out_records, "ContextDiff"),
        "records": out_records,
    }


def preflight(targets: list[str]) -> dict[str, Any]:
    out = {"status": "READY", "created_utc": now_utc(), "targets": {}}
    for t in targets:
        path = Path(DEFAULT_TARGETS[t]["model_path"])
        rec = {"model_path": str(path), "exists": path.exists(), "errors": []}
        if not path.exists():
            rec["errors"].append("missing chck_100M")
            out["status"] = "NOT_READY"
        out["targets"][t] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), choices=sorted(DEFAULT_TARGETS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--row_limit", type=int, default=0)
    ap.add_argument("--row_offset", type=int, default=0)
    ap.add_argument("--row_batch_size", type=int, default=64)
    ap.add_argument("--masked_batch_size", type=int, default=128)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pf = preflight(args.targets)
    pf_path = OUT_ROOT / "preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "ready": pf["status"], "preflight": str(pf_path)}, indent=2), flush=True)
        return
    if pf["status"] != "READY":
        raise RuntimeError({"preflight": pf, "preflight_path": str(pf_path)})
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    all_summary: dict[str, Any] = {
        "status": "FW_EWOK_INTERACTION_READER_DONE",
        "created_utc": now_utc(),
        "boundary": "post-endpoint EWoK four-cell readout; no training or official-example shaping",
        "targets": {},
    }
    for t in args.targets:
        meta = DEFAULT_TARGETS[t]
        res = run_target(t, Path(meta["model_path"]), device, args.threads, args.row_limit, args.row_offset, args.row_batch_size, args.masked_batch_size)
        tdir = OUT_ROOT / t
        tdir.mkdir(parents=True, exist_ok=True)
        records = res.pop("records")
        write_csv(tdir / "ewok_interaction_records.csv", records)
        write_csv(tdir / "ewok_interaction_by_domain.csv", res["by_domain"])
        write_csv(tdir / "ewok_interaction_by_context_diff.csv", res["by_context_diff"])
        (tdir / "ewok_interaction_summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        all_summary["targets"][t] = {k: v for k, v in res.items() if k not in {"by_domain", "by_context_diff"}}
        print(json.dumps({"event": "target_done", "target": t, "summary": res["summary"], "summary_path": str(tdir / "ewok_interaction_summary.json")}, indent=2), flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = OUT_ROOT / "fw_ewok_interaction_reader_summary.json"
    combined.write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# research — FW EWoK interaction reader\n\n",
        "Post-endpoint four-cell readout for the FW shared-anchor models. It scores EWoK contexts and targets directly and measures stable conditional-reversal failures without using old baseline correctness columns.\n\n",
    ]
    for t, res in all_summary["targets"].items():
        s = res["summary"]
        lines.append(f"- `{t}`: accuracy={s['accuracy']}, wrong={s['saved_wrong']}, stable failures={s['stable_failure']} ({s['stable_failure_frac_wrong']} of wrong), interaction median wrong={s['interaction_sum_wrong'].get('median')}\n")
    lines.append(f"\nCombined summary: `{combined}`\n")
    NOTE.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": all_summary["status"], "combined_summary": str(combined), "note": str(NOTE)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
