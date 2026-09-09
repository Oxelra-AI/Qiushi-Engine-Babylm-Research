#!/usr/bin/env python3
"""research: source-conditioned MLM probe for compact second views.

For each source/view pair, mask matched content-word targets inside the view and
compare NLL(view target) with and without the original source prefix.  The probe is
not a semantic-faithfulness test.  It asks whether a compact candidate still leaves a
source-conditioned reconstruction signal comparable to the inherited rewrite, and how
that signal differs for independently labeled faithful, information-losing, and altered rows.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/functional_learning')
A02 = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/functional_learning/scripts')
A02_SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(A02_SCRIPTS))

import corrected_bridge_trainer as trainer  # noqa: E402

PARENT_PATH = _public_path('models/frontier')
DEFAULT_JOINED = _public_path('experiments/archive/functional_learning/data/compact_pilot/pilot_generation_joined.jsonl')
DEFAULT_LABELS = _public_path('experiments/archive/functional_learning/data/compact_semantic_review/independent_review_semantic_labels_pilot_sample.jsonl')
DEFAULT_OUT = _public_path('experiments/archive/functional_learning/data/compact_source_help_probe')

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "into", "about", "over", "under", "after", "before", "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had", "will", "would", "can", "could", "may", "might", "must", "should", "shall", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "their", "our", "not", "no", "so", "there", "here", "what", "which", "who", "when", "where", "why", "how"
}


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def load_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(label: str) -> str:
    lab = (label or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "unclear_or_information_losing": "supported_summary_with_lost_detail",
        "unclear_or_unsafe_source": "unclear",
    }
    return aliases.get(lab, lab)


def words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def content_word_positions(text: str) -> List[Tuple[int, str, int, int]]:
    out = []
    for wi, m in enumerate(WORD_RE.finditer(text or "")):
        w = m.group(0)
        wl = w.lower().strip("'\"")
        if len(wl) < 4:
            continue
        if wl in STOP:
            continue
        if wl.isdigit():
            continue
        out.append((wi, wl, m.start(), m.end()))
    return out


def choose_targets(source: str, view: str, max_targets: int, rng: random.Random) -> List[Dict[str, Any]]:
    candidates = content_word_positions(view)
    if not candidates:
        return []
    source_vocab = set(words(source))
    copied = [x for x in candidates if x[1] in source_vocab]
    novel = [x for x in candidates if x[1] not in source_vocab]
    chosen: List[Tuple[int, str, int, int]] = []
    # Balance copied and novel targets when both exist; source help on copied words can be lexical copying,
    # while novel words test semantic/contextual prediction.
    if copied and novel and max_targets >= 2:
        n_copy = min(len(copied), max_targets // 2)
        n_novel = min(len(novel), max_targets - n_copy)
        chosen.extend(rng.sample(copied, n_copy))
        chosen.extend(rng.sample(novel, n_novel))
        rem = max_targets - len(chosen)
        if rem > 0:
            rest = [x for x in candidates if x not in chosen]
            if rest:
                chosen.extend(rng.sample(rest, min(rem, len(rest))))
    else:
        chosen = rng.sample(candidates, min(max_targets, len(candidates)))
    chosen.sort(key=lambda x: x[2])
    return [{"word_index": wi, "word": w, "char_start": s, "char_end": e, "copied_from_source": w in source_vocab} for wi, w, s, e in chosen]


def token_span_from_offsets(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> Optional[Tuple[int, int]]:
    idxs = []
    for i, (s, e) in enumerate(offsets):
        s = int(s); e = int(e)
        if e <= s:
            continue
        if s < end and e > start:
            idxs.append(i)
    if not idxs:
        return None
    return min(idxs), max(idxs) + 1


def build_context(tokenizer, source: str, view: str, with_source: bool, max_len: int) -> Tuple[List[int], List[Tuple[int, int]], int]:
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    view_enc = tokenizer(view, add_special_tokens=False, return_offsets_mapping=True)
    view_ids = list(view_enc["input_ids"])
    view_offsets = [(int(a), int(b)) for a, b in view_enc["offset_mapping"]]
    ids: List[int] = []
    if cls_id is not None:
        ids.append(int(cls_id))
    if with_source:
        src_ids = tokenizer.encode(source, add_special_tokens=False)
        ids.extend(src_ids)
        if sep_id is not None:
            ids.append(int(sep_id))
    view_start = len(ids)
    ids.extend(view_ids)
    if sep_id is not None:
        ids.append(int(sep_id))
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids, view_offsets, view_start


def score_batch(model, device, mask_id: int, tasks: List[Dict[str, Any]], max_len: int) -> List[float]:
    if not tasks:
        return []
    B = len(tasks)
    L = min(max(len(t["ids"]) for t in tasks), max_len)
    input_ids = torch.full((B, L), 0, dtype=torch.long, device=device)
    attn = torch.zeros((B, L), dtype=torch.long, device=device)
    orig = torch.full((B, L), -100, dtype=torch.long, device=device)
    for i, t in enumerate(tasks):
        ids = t["ids"][:L]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, :len(ids)] = 1
        orig[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        s, e = t["span"]
        if s < L:
            input_ids[i, s:min(e, L)] = int(mask_id)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    out = []
    for i, t in enumerate(tasks):
        s, e = t["span"]
        e = min(e, L)
        if s >= e or s >= L:
            out.append(float("nan"))
            continue
        lps = torch.log_softmax(logits[i, s:e].float(), dim=-1)
        ids = orig[i, s:e]
        vals = [-float(lps[j, int(tok)].detach().cpu()) for j, tok in enumerate(ids)]
        out.append(sum(vals) / max(1, len(vals)))
    return out


def finite_mean(xs: Iterable[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joined", default=str(DEFAULT_JOINED))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--targets-per-view", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=54055)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(args.seed))

    labels = {r["pair_id"]: normalize_label(r.get("label", "")) for r in load_jsonl(pathlib.Path(args.labels))}
    rows_all = [r for r in load_jsonl(pathlib.Path(args.joined)) if r.get("pair_id") in labels]
    if args.max_records and args.max_records > 0:
        rows_all = rows_all[: int(args.max_records)]

    cache = out_dir / "hf_cache"
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((cache / "transformers").resolve())
    os.environ["HF_MODULES_CACHE"] = str((cache / "modules").resolve())
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    (cache / "modules").mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    model, missing, unexpected = trainer.load_model(device, private_scale=0.75)
    ident = trainer.model_identity(model)
    if ident.get("class") != "FrozenSlowPrivateDebertaV2ForMaskedLM" or int(ident.get("private_adapter_params", 0)) != 995584:
        raise RuntimeError(f"bad model identity: {ident}")
    model.eval()
    tokenizer = trainer.AutoTokenizer.from_pretrained(str(PARENT_PATH), local_files_only=True, use_fast=True)
    if tokenizer.mask_token_id is None:
        raise RuntimeError("tokenizer has no mask token")

    target_records: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for rec in rows_all:
        label = labels[rec["pair_id"]]
        source = rec.get("original", "")
        for view_kind, view in [("current", rec.get("current_rewrite", "")), ("compact", rec.get("compact_rewrite", ""))]:
            targets = choose_targets(source, view, int(args.targets_per_view), rng)
            ids_s, offsets, view_start_s = build_context(tokenizer, source, view, True, int(args.max_length))
            ids_v, offsets_v, view_start_v = build_context(tokenizer, source, view, False, int(args.max_length))
            # offsets and offsets_v are the same because they tokenize the same view text.
            for t in targets:
                span_view = token_span_from_offsets(offsets, int(t["char_start"]), int(t["char_end"]))
                if span_view is None:
                    continue
                span_s = (view_start_s + span_view[0], view_start_s + span_view[1])
                span_v = (view_start_v + span_view[0], view_start_v + span_view[1])
                if span_s[1] > len(ids_s) or span_v[1] > len(ids_v):
                    continue
                rec_id = len(target_records)
                target_records.append({
                    "record_id": rec_id,
                    "pair_id": rec["pair_id"],
                    "semantic_label": label,
                    "source": rec.get("source", ""),
                    "view_kind": view_kind,
                    "target_word": t["word"],
                    "copied_from_source": bool(t["copied_from_source"]),
                    "view_words": len(view.split()),
                    "source_words": len(source.split()),
                })
                tasks.append({"record_id": rec_id, "condition": "with_source", "ids": ids_s, "span": span_s})
                tasks.append({"record_id": rec_id, "condition": "view_only", "ids": ids_v, "span": span_v})

    # Score all tasks in batches.
    scores: Dict[Tuple[int, str], float] = {}
    for start in range(0, len(tasks), int(args.batch_size)):
        mb = tasks[start:start + int(args.batch_size)]
        nlls = score_batch(model, device, int(tokenizer.mask_token_id), mb, int(args.max_length))
        for task, nll in zip(mb, nlls):
            scores[(int(task["record_id"]), task["condition"])] = float(nll)
        if (start // int(args.batch_size)) % 20 == 0:
            print(json.dumps({"event": "score_progress", "scored_tasks": min(start + len(mb), len(tasks)), "total_tasks": len(tasks)}), flush=True)

    rows = []
    for r in target_records:
        ws = scores.get((r["record_id"], "with_source"), float("nan"))
        vo = scores.get((r["record_id"], "view_only"), float("nan"))
        rr = dict(r)
        rr["nll_with_source"] = ws
        rr["nll_view_only"] = vo
        rr["source_help"] = vo - ws if math.isfinite(ws) and math.isfinite(vo) else float("nan")
        rows.append(rr)

    # Summaries by semantic label and view kind.
    def summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_targets": len(group),
            "n_pairs": len(set(r["pair_id"] for r in group)),
            "mean_source_help": finite_mean(r["source_help"] for r in group),
            "mean_nll_with_source": finite_mean(r["nll_with_source"] for r in group),
            "mean_nll_view_only": finite_mean(r["nll_view_only"] for r in group),
            "copied_target_fraction": finite_mean(1.0 if r["copied_from_source"] else 0.0 for r in group),
        }

    by_label_view: Dict[str, Dict[str, Any]] = {}
    for label in sorted(set(r["semantic_label"] for r in rows)):
        for view_kind in ["current", "compact"]:
            g = [r for r in rows if r["semantic_label"] == label and r["view_kind"] == view_kind]
            if g:
                by_label_view[f"{label}/{view_kind}"] = summarize(g)

    # Pair-level compact-current differences on source_help.
    pair_view = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pair_view[(r["pair_id"], r["semantic_label"])][r["view_kind"]].append(r["source_help"])
    diffs = []
    for (pid, label), v in pair_view.items():
        if v.get("current") and v.get("compact"):
            diffs.append({"pair_id": pid, "semantic_label": label, "compact_minus_current_source_help": finite_mean(v["compact"]) - finite_mean(v["current"]), "current_source_help": finite_mean(v["current"]), "compact_source_help": finite_mean(v["compact"])})
    by_label_diff = {}
    for label in sorted(set(d["semantic_label"] for d in diffs)):
        g = [d for d in diffs if d["semantic_label"] == label]
        by_label_diff[label] = {
            "n_pairs": len(g),
            "mean_compact_minus_current_source_help": finite_mean(d["compact_minus_current_source_help"] for d in g),
            "median_compact_minus_current_source_help": statistics.median([d["compact_minus_current_source_help"] for d in g if math.isfinite(d["compact_minus_current_source_help"])]) if g else None,
            "positive_fraction": finite_mean(1.0 if d["compact_minus_current_source_help"] > 0 else 0.0 for d in g),
        }

    # Write outputs.
    rec_path = out_dir / "source_help_target_records.csv"
    with rec_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    diff_path = out_dir / "source_help_pair_diffs.jsonl"
    with diff_path.open("w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    summary = {
        "status": "COMPACT_SOURCE_HELP_PROBE_DONE",
        "scientific_purpose": "Test whether compact candidates preserve source-conditioned MLM reconstruction signal; not a proof of semantic faithfulness or learner improvement.",
        "joined": rel(pathlib.Path(args.joined)),
        "labels": rel(pathlib.Path(args.labels)),
        "n_pairs_scored": len(set(r["pair_id"] for r in rows)),
        "n_target_records": len(rows),
        "device": str(device),
        "model_identity": ident,
        "by_label_view": by_label_view,
        "by_label_pair_diff": by_label_diff,
        "records_csv": rel(rec_path),
        "pair_diffs": rel(diff_path),
    }
    (out_dir / "source_help_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
