#!/usr/bin/env python3
"""research: pair-stratified local response bridge for RoBERTa compact-vs-repeat transfer.

Scientific purpose
------------------
The research RoBERTa experiment gives aggregate official-compatible trajectories for
natural compact second views versus matched first-N repeat views.  The research atlas
shows the data marginal is coupled: tail/source-wide coverage, content density,
source-absent compact content, and legal-tokenizer token load move together.

This script builds a frozen, bounded local-denoising event set stratified by the
already-measured pair attributes.  After the RoBERTa arms finish, the same script can
measure compact-trained versus repeat-trained checkpoint response by stratum.  The
stratification is interpretive evidence only: concordance with downstream stable-family
gains can strengthen a mechanism hypothesis; discordance can choose one clean future
intervention.  It is not a causal result by itself and it does not launch training.

Modes
-----
--build_only: create frozen_events.jsonl + stratum_manifest.json from research pair_atlas.
              This mode loads only the legal tokenizer and local text files.
--evaluate:   load completed compact/repeat checkpoints and evaluate masked target spans
              on the frozen events, reporting local NLL advantages by strata.

No leaderboard submission, no upload, no training.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019-][A-Za-z0-9]+)?")


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
PAIR_FILE = WS / "data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl"
ATLAS_CSV = WS / "data/natural_compact_repeat_mechanism_atlas/pair_atlas.csv"
ATLAS_JSON = WS / "data/natural_compact_repeat_mechanism_atlas/mechanism_atlas.json"
TOKENIZER_DIR = WS / "data/compliant_tokenizer"
DEFAULT_OUT = WS / "data/roberta_pair_stratified_response_probe"
COMPACT_RUN = WS / "training/runs/roberta_compact_reinvest_100M_seed43022"
REPEAT_RUN = WS / "training/runs/roberta_repeat_compact_reinvest_100M_seed43022"
SEQ_LEN = 256
DEFAULT_CHECKPOINTS = ["chck_60M", "chck_70M", "chck_80M", "chck_90M", "chck_100M"]
FEATURES = [
    "compact_tail_content_coverage",
    "compact_content_fraction",
    "compact_source_absent_content_fraction_of_content",
]
VIEW_CATS = [
    ("compact", "source_absent_content"),
    ("compact", "retained_content"),
    ("compact", "function_other"),
    ("repeat", "retained_content"),
    ("repeat", "function_other"),
]
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "so", "because", "as", "than",
    "to", "of", "in", "on", "for", "with", "without", "by", "from", "at", "into", "onto", "over",
    "under", "about", "between", "among", "through", "during", "before", "after", "above", "below",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "shall", "should",
    "will", "would", "this", "that", "these", "those", "there", "here", "it", "its", "they", "them",
    "their", "theirs", "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you",
    "your", "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what", "where",
    "when", "why", "how", "not", "no", "nor", "only", "just", "also", "very", "more", "most", "less",
    "least", "much", "many", "some", "any", "all", "each", "every", "other", "another", "such", "own",
    "same", "too", "again", "still", "already", "yet", "up", "down", "out", "off", "back", "away",
    "into", "within", "across", "per", "via", "using", "used", "use", "uses", "become", "became",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def norm(w: str) -> str:
    parts = WORD_RE.findall(w)
    if not parts:
        return ""
    return "".join(parts).lower()


def is_content(w: str) -> bool:
    n = norm(w)
    if not n or n in STOPWORDS:
        return False
    if n.isdigit():
        return True
    return len(n) >= 4


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fnum(x: Any, default: float = 0.0) -> float:
    return float(x) if finite(x) else default


def quantile(vals: list[float], p: float) -> float:
    vals = sorted(vals)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    k = p * (len(vals) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def read_pairs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with PAIR_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[str(r["pair_id"])] = r
    return out


def read_atlas() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with ATLAS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: dict[str, Any] = dict(r)
            for k, v in list(row.items()):
                if k in {"pair_id", "key"}:
                    continue
                if v in ("True", "False"):
                    row[k] = (v == "True")
                elif v == "":
                    row[k] = None
                else:
                    try:
                        if "." in str(v) or "e" in str(v).lower():
                            row[k] = float(v)
                        else:
                            row[k] = int(v)
                    except Exception:
                        row[k] = v
            rows[str(row["pair_id"])] = row
    return rows


def build_thresholds(atlas_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    th: dict[str, Any] = {}
    for feature in FEATURES:
        vals = [fnum(r.get(feature)) for r in atlas_rows.values() if finite(r.get(feature))]
        q33 = quantile(vals, 1.0 / 3.0)
        q67 = quantile(vals, 2.0 / 3.0)
        rec: dict[str, Any] = {"kind": "tertile", "q33": q33, "q67": q67, "n": len(vals)}
        if feature == "compact_source_absent_content_fraction_of_content":
            pos = [v for v in vals if v > 0]
            rec.update({"kind": "zero_low_high_positive", "positive_median": quantile(pos, 0.5), "positive_n": len(pos), "zero_n": len(vals) - len(pos)})
        th[feature] = rec
    return th


def assign_bins(row: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, str]:
    bins: dict[str, str] = {}
    for feature in FEATURES:
        raw = row.get(feature)
        rec = thresholds[feature]
        if not finite(raw):
            # Atlas value is undefined (e.g. no source-tail content region exists);
            # keep it as an explicit undefined bin instead of silently coercing to 0.0/low.
            bins[feature] = "undefined"
            continue
        x = fnum(raw)
        if rec.get("kind") == "zero_low_high_positive":
            if x <= 0:
                b = "zero"
            elif x <= float(rec["positive_median"]):
                b = "low_positive"
            else:
                b = "high_positive"
        else:
            if x <= float(rec["q33"]):
                b = "low"
            elif x <= float(rec["q67"]):
                b = "mid"
            else:
                b = "high"
        bins[feature] = b
    return bins


def token_count(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def target_span(tok, source_text: str, side_words: list[str], word_i: int) -> tuple[int, int] | None:
    before_side = " ".join(side_words[:word_i])
    through_side = " ".join(side_words[:word_i + 1])
    prefix_before = source_text if not before_side else source_text + " " + before_side
    prefix_after = source_text + " " + through_side
    st = token_count(tok, prefix_before)
    en = token_count(tok, prefix_after)
    if st < en and st < SEQ_LEN:
        return st, min(en, SEQ_LEN)
    return None


def side_text_for(row: dict[str, Any], view_type: str) -> str:
    if view_type == "compact":
        return str(row["view_text"])
    src_words = str(row["source_text"]).split()
    n = int(row.get("view_words", len(str(row["view_text"]).split())))
    return " ".join(src_words[:n])


def category_for_word(w: str, source_norms: set[str], view_type: str) -> str:
    n = norm(w)
    if is_content(w):
        if view_type == "compact" and n not in source_norms:
            return "source_absent_content"
        return "retained_content"
    return "function_other"


def all_candidate_events(tok, pairs: dict[str, dict[str, Any]], atlas: dict[str, dict[str, Any]], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for pair_id, row in pairs.items():
        if pair_id not in atlas:
            continue
        a = atlas[pair_id]
        bins = assign_bins(a, thresholds)
        src = str(row["source_text"])
        source_norms = {norm(w) for w in src.split() if norm(w)}
        for view_type in ["compact", "repeat"]:
            side = side_text_for(row, view_type)
            if not side.strip():
                continue
            side_words = side.split()
            full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
            if len(full_ids) < 4:
                continue
            for wi, w in enumerate(side_words):
                nw = norm(w)
                if not nw:
                    continue
                cat = category_for_word(w, source_norms, view_type)
                if (view_type, cat) not in VIEW_CATS:
                    continue
                span = target_span(tok, src, side_words, wi)
                if span is None or span[0] >= span[1] or span[1] > len(full_ids):
                    continue
                candidates.append({
                    "pair_id": pair_id,
                    "key": str(row.get("key")),
                    "view_type": view_type,
                    "category": cat,
                    "word": w,
                    "norm_word": nw,
                    "word_index": int(wi),
                    "span": [int(span[0]), int(span[1])],
                    "n_pieces": int(span[1] - span[0]),
                    "input_ids": [int(x) for x in full_ids],
                    "features": {feature: (fnum(a.get(feature)) if finite(a.get(feature)) else None) for feature in FEATURES},
                    "feature_bins": bins,
                    "aux_features": {
                        "compact_minus_repeat_content_coverage": fnum(a.get("compact_minus_repeat_content_coverage")),
                        "compact_minus_repeat_active_tokens": fnum(a.get("compact_minus_repeat_active_tokens")),
                        "compact_tail_only_content_fraction_of_content": fnum(a.get("compact_tail_only_content_fraction_of_content")),
                    },
                })
    return candidates


def balanced_select(candidates: list[dict[str, Any]], per_cell: int, seed: int, max_events: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str, str, str], list[int]] = collections.defaultdict(list)
    for idx, e in enumerate(candidates):
        for feature in FEATURES:
            b = e["feature_bins"].get(feature)
            by_cell[(feature, b, e["view_type"], e["category"])].append(idx)
    chosen: set[int] = set()
    cell_counts: dict[str, int] = {}
    for cell, idxs in sorted(by_cell.items()):
        rng.shuffle(idxs)
        per_pair = collections.Counter()
        keep: list[int] = []
        for idx in idxs:
            pid = candidates[idx]["pair_id"]
            # Avoid one pair dominating a feature cell, but allow the same pair in different cells/features.
            if per_pair[pid] >= 1:
                continue
            keep.append(idx)
            per_pair[pid] += 1
            if len(keep) >= per_cell:
                break
        chosen.update(keep)
        cell_counts["|".join(str(x) for x in cell)] = len(keep)
    selected = [candidates[i] for i in sorted(chosen)]
    rng.shuffle(selected)
    if max_events > 0 and len(selected) > max_events:
        selected = selected[:max_events]
    for i, e in enumerate(selected):
        e["event_index"] = i
    meta = {
        "candidate_events": len(candidates),
        "selected_events": len(selected),
        "unique_selected_pairs": len({e["pair_id"] for e in selected}),
        "per_cell_target": per_cell,
        "max_events": max_events,
        "seed": seed,
        "cell_selected_counts": cell_counts,
        "selected_by_view_category": dict(collections.Counter(f"{e['view_type']}|{e['category']}" for e in selected)),
        "selected_by_feature_bin_view_category": feature_bin_counts(selected),
    }
    return selected, meta


def feature_bin_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    ctr: collections.Counter[str] = collections.Counter()
    for e in events:
        for feature in FEATURES:
            ctr[f"{feature}|{e['feature_bins'].get(feature)}|{e['view_type']}|{e['category']}"] += 1
    return dict(ctr)


def write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def short_stats(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p25": quantile(s, 0.25),
        "p75": quantile(s, 0.75),
        "min": s[0],
        "max": s[-1],
    }


def build_only(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
    pairs = read_pairs()
    atlas = read_atlas()
    thresholds = build_thresholds(atlas)
    candidates = all_candidate_events(tok, pairs, atlas, thresholds)
    selected, select_meta = balanced_select(candidates, args.per_cell, args.seed, args.max_events)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "frozen_events.jsonl"
    write_jsonl(events_path, selected)
    # Remove a large field from the small manifest; frozen_events remains the executable artifact.
    per_feature_value_stats = {feature: short_stats([fnum(r.get(feature)) for r in atlas.values()]) for feature in FEATURES}
    source_absent_bin_counter = collections.Counter(
        e["feature_bins"].get("compact_source_absent_content_fraction_of_content")
        for e in selected
        if e["view_type"] == "compact" and e["category"] == "source_absent_content"
    )
    category_consistency = {
        "content_definition": "Matches research atlas STOPWORDS + len>=4/digit content definition; this avoids auxiliary/short-word drift in source-absent strata.",
        "compact_source_absent_events_by_source_absent_pair_bin": dict(source_absent_bin_counter),
        "compact_source_absent_events_in_zero_source_absent_pairs": int(source_absent_bin_counter.get("zero", 0)),
        "ok_no_source_absent_events_in_zero_pairs": int(source_absent_bin_counter.get("zero", 0)) == 0,
    }
    manifest = {
        "status": "ROBERTA_PAIR_STRATIFIED_EVENTS_BUILT",
        "created_utc": now_utc(),
        "meaning": "Frozen local-denoising event set stratified by research pair-level features; to be evaluated only after the paired RoBERTa compact/repeat checkpoints and official-compatible trajectories arrive.",
        "inputs": {
            "pair_file": str(PAIR_FILE),
            "pair_file_sha256": sha256_file(PAIR_FILE),
            "atlas_csv": str(ATLAS_CSV),
            "atlas_csv_sha256": sha256_file(ATLAS_CSV),
            "atlas_json": str(ATLAS_JSON),
            "tokenizer": str(args.tokenizer),
            "tokenizer_json_sha256": sha256_file(args.tokenizer / "tokenizer.json"),
        },
        "features": FEATURES,
        "thresholds": thresholds,
        "feature_value_stats": per_feature_value_stats,
        "selection": select_meta,
        "category_consistency": category_consistency,
        "frozen_events_jsonl": str(events_path),
        "frozen_events_sha256": sha256_file(events_path),
        "interpretation": {
            "response_quantity": "Later evaluation reports compact-trained minus repeat-trained local NLL; negative local loss delta or positive repeat_minus_compact_advantage means the compact-trained model predicts the masked event better.",
            "feature_response_not_causal": "Bins are post-hoc strata over the natural compact-vs-repeat marginal. They can connect aggregate RoBERTa behavior to DeBERTa evidence and choose one future clean intervention, but they cannot by themselves identify a cause.",
            "official_scores_still_primary": "This local probe must be interpreted together with selected official-compatible stable-family deltas from the research trajectory reader; it is not a substitute for BabyLM evaluation.",
        },
    }
    write_json(out_dir / "stratum_manifest.json", manifest)
    md = render_manifest_md(manifest)
    (out_dir / "stratum_manifest.md").write_text(md, encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "out_dir": str(out_dir), "selected_events": len(selected), "unique_pairs": select_meta["unique_selected_pairs"], "events_sha256": manifest["frozen_events_sha256"]}, indent=2), flush=True)


def render_manifest_md(m: dict[str, Any]) -> str:
    lines = [
        "# research RoBERTa pair-stratified response probe event set",
        "",
        f"Created: `{m['created_utc']}`",
        "",
        "## Purpose",
        "",
        "This frozen event set bridges the aggregate RoBERTa compact-vs-repeat trajectory to the measured natural data marginal. It is designed to ask whether local compact-trained response is concentrated in pairs with high tail/source-wide coverage, high compact content density, or high source-absent compact-content fraction.",
        "",
        "The strata are not causal interventions. They only become informative when read together with official-compatible stable-family gains or losses and existing DeBERTa/GPT2/ordered-scrambled evidence.",
        "",
        "## Inputs",
        "",
        f"- Pair file SHA: `{m['inputs']['pair_file_sha256']}`",
        f"- Atlas CSV SHA: `{m['inputs']['atlas_csv_sha256']}`",
        f"- Legal tokenizer SHA: `{m['inputs']['tokenizer_json_sha256']}`",
        "",
        "## Frozen selection",
        "",
        f"- Candidate events: `{m['selection']['candidate_events']}`",
        f"- Selected events: `{m['selection']['selected_events']}`",
        f"- Unique selected pairs: `{m['selection']['unique_selected_pairs']}`",
        f"- Frozen events SHA: `{m['frozen_events_sha256']}`",
        "",
        "Selected events by view/category:",
        "",
    ]
    for k, v in sorted(m["selection"]["selected_by_view_category"].items()):
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## Feature thresholds", ""]
    for feature, rec in m["thresholds"].items():
        lines.append(f"- `{feature}`: {json.dumps(rec, ensure_ascii=False)}`")
    lines += [
        "",
        "## Later evaluation reading",
        "",
        "Run with `--evaluate` only after both RoBERTa training arms and their selected official-compatible trajectory have completed. The local response quantity is compact-trained minus repeat-trained NLL on the same masked event; negative is a compact-trained local advantage. The derived `repeat_minus_compact_advantage` flips the sign so positive means compact-trained better.",
        "",
        "Concordant evidence would be: stable late official-family gains plus local compact advantages concentrated in the same high-tail/high-density/high-source-absent strata that the DeBERTa mechanism predicts. Discordance should choose one clean future intervention, not launch a broad sweep.",
    ]
    return "\n".join(lines) + "\n"


# ---------------- evaluation mode ----------------

def make_batch(events: list[dict[str, Any]], mask_id: int, pad_id: int):
    import torch

    max_len = min(max(len(e["input_ids"]) for e in events), SEQ_LEN)
    x = torch.full((len(events), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(events), max_len), -100, dtype=torch.long)
    for i, e in enumerate(events):
        ids = [int(t) for t in e["input_ids"][:max_len]]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        st, en = [int(z) for z in e["span"]]
        en = min(en, max_len)
        if st < en:
            y[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            x[i, st:en] = mask_id
    return x, y


def eval_arm(model_path: Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> dict[int, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    out: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for i in range(0, len(events), batch_size):
            batch = events[i:i + batch_size]
            x, y = make_batch(batch, tok.mask_token_id, tok.pad_token_id)
            x = x.to(device)
            y = y.to(device)
            attn = (x != tok.pad_token_id).long().to(device)
            logits = model(input_ids=x, attention_mask=attn).logits
            per_tok = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none", ignore_index=-100).reshape(y.shape)
            mask = y != -100
            for b, e in enumerate(batch):
                vals = per_tok[b][mask[b]]
                out[int(e["event_index"])] = {
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "loss_mean": float(vals.mean().detach().cpu()),
                    "pieces": int(vals.numel()),
                }
    del model
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    return out


def weighted_delta(rows: list[dict[str, Any]]) -> float | None:
    den = sum(r["pieces"] for r in rows)
    if den <= 0:
        return None
    return sum((r["compact_loss_sum"] - r["repeat_loss_sum"]) for r in rows) / den


def cluster_bootstrap_advantage(rows: list[dict[str, Any]], reps: int, seed: int) -> dict[str, Any] | None:
    """Pair-cluster bootstrap for repeat_minus_compact_advantage (= -delta)."""
    if reps <= 0 or not rows:
        return None
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_pair[str(r["pair_id"])].append(r)
    keys = list(by_pair)
    if not keys:
        return None
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(reps):
        sample: list[dict[str, Any]] = []
        for _j in range(len(keys)):
            sample.extend(by_pair[rng.choice(keys)])
        d = weighted_delta(sample)
        if d is not None and finite(d):
            vals.append(-float(d))
    if not vals:
        return None
    vals.sort()
    def q(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
        return vals[idx]
    return {
        "reps": len(vals),
        "clusters": len(keys),
        "mean": statistics.mean(vals),
        "p05": q(0.05),
        "p50": q(0.50),
        "p95": q(0.95),
        "p_gt_0": sum(1 for v in vals if v > 0) / len(vals),
    }


def summarize_records(records: list[dict[str, Any]], bootstrap_reps: int = 0, bootstrap_seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {}
    overall_delta = weighted_delta(records)
    out["overall"] = {
        "events": len(records),
        "pieces": sum(r["pieces"] for r in records),
        "compact_minus_repeat_nll": overall_delta,
        "repeat_minus_compact_advantage": (-overall_delta) if overall_delta is not None else None,
        "pair_cluster_bootstrap": cluster_bootstrap_advantage(records, bootstrap_reps, bootstrap_seed),
    }
    # View/category summaries.
    by_vc: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_vc[(r["view_type"], r["category"])].append(r)
    out["by_view_category"] = {}
    for (view, cat), rows in sorted(by_vc.items()):
        d = weighted_delta(rows)
        out["by_view_category"][f"{view}|{cat}"] = {"events": len(rows), "pieces": sum(x["pieces"] for x in rows), "compact_minus_repeat_nll": d, "repeat_minus_compact_advantage": (-d) if d is not None else None, "pair_cluster_bootstrap": cluster_bootstrap_advantage(rows, bootstrap_reps, bootstrap_seed + 17 * (len(out["by_view_category"]) + 1))}

    # Feature/bin summaries and simple high-minus-low or high-positive-minus-zero contrasts.
    out["by_feature_bin"] = {}
    out["feature_contrasts"] = {}
    for feature in FEATURES:
        feature_summary: dict[str, Any] = {}
        for view, cat in VIEW_CATS:
            rows0 = [r for r in records if r["view_type"] == view and r["category"] == cat]
            bins = sorted({str(r["feature_bins"].get(feature)) for r in rows0})
            bybin: dict[str, Any] = {}
            adv_by_bin: dict[str, float] = {}
            for b in bins:
                rowsb = [r for r in rows0 if str(r["feature_bins"].get(feature)) == b]
                d = weighted_delta(rowsb)
                adv = (-d) if d is not None else None
                bybin[b] = {"events": len(rowsb), "pieces": sum(x["pieces"] for x in rowsb), "compact_minus_repeat_nll": d, "repeat_minus_compact_advantage": adv}
                if adv is not None:
                    adv_by_bin[b] = adv
            feature_summary[f"{view}|{cat}"] = bybin
            contrast: float | None = None
            contrast_label: str | None = None
            if "high" in adv_by_bin and "low" in adv_by_bin:
                contrast = adv_by_bin["high"] - adv_by_bin["low"]
                contrast_label = "high_minus_low"
            elif "high_positive" in adv_by_bin and "zero" in adv_by_bin:
                contrast = adv_by_bin["high_positive"] - adv_by_bin["zero"]
                contrast_label = "high_positive_minus_zero"
            elif "high_positive" in adv_by_bin and "low_positive" in adv_by_bin:
                contrast = adv_by_bin["high_positive"] - adv_by_bin["low_positive"]
                contrast_label = "high_positive_minus_low_positive"
            if contrast is not None:
                out["feature_contrasts"][f"{feature}|{view}|{cat}"] = {"contrast": contrast, "contrast_label": contrast_label}
        out["by_feature_bin"][feature] = feature_summary

    # Difference-in-difference: compact-view advantage minus repeat-view advantage within each feature bin.
    out["compact_specific_did_by_feature_bin"] = {}
    for feature in FEATURES:
        bins = sorted({str(r["feature_bins"].get(feature)) for r in records})
        recs = {}
        for b in bins:
            comp_rows = [r for r in records if r["feature_bins"].get(feature) == b and r["view_type"] == "compact"]
            rep_rows = [r for r in records if r["feature_bins"].get(feature) == b and r["view_type"] == "repeat"]
            dc = weighted_delta(comp_rows)
            dr = weighted_delta(rep_rows)
            ac = (-dc) if dc is not None else None
            ar = (-dr) if dr is not None else None
            did = (ac - ar) if ac is not None and ar is not None else None
            recs[b] = {"compact_view_advantage": ac, "repeat_view_advantage": ar, "compact_specific_advantage_did": did, "compact_events": len(comp_rows), "repeat_events": len(rep_rows)}
        out["compact_specific_did_by_feature_bin"][feature] = recs
    return out


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    out_dir = args.out_dir
    events_path = args.events or (out_dir / "frozen_events.jsonl")
    if not events_path.exists():
        raise SystemExit(f"Frozen events not found: {events_path}. Run --build_only first.")
    events = read_jsonl(events_path)
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
    all_ck: dict[str, Any] = {}
    for ck in args.checkpoints:
        cpath = args.compact_run / "hf_model" / ck
        rpath = args.repeat_run / "hf_model" / ck
        if not cpath.exists() or not rpath.exists():
            all_ck[ck] = {"missing": True, "compact_exists": cpath.exists(), "repeat_exists": rpath.exists()}
            continue
        compact = eval_arm(cpath, events, tok, args.device, args.batch_size)
        repeat = eval_arm(rpath, events, tok, args.device, args.batch_size)
        records = []
        for e in events:
            idx = int(e["event_index"])
            if idx not in compact or idx not in repeat:
                continue
            pieces = int(compact[idx]["pieces"])
            records.append({
                "event_index": idx,
                "pair_id": e["pair_id"],
                "view_type": e["view_type"],
                "category": e["category"],
                "features": e["features"],
                "feature_bins": e["feature_bins"],
                "pieces": pieces,
                "compact_loss_sum": float(compact[idx]["loss_sum"]),
                "repeat_loss_sum": float(repeat[idx]["loss_sum"]),
                "compact_loss_mean": float(compact[idx]["loss_mean"]),
                "repeat_loss_mean": float(repeat[idx]["loss_mean"]),
            })
        all_ck[ck] = {"missing": False, "summary": summarize_records(records, bootstrap_reps=args.bootstrap_reps, bootstrap_seed=args.seed + 1009 * (1 + args.checkpoints.index(ck)))}
        if args.write_event_losses:
            loss_path = out_dir / f"event_losses_{ck}.jsonl"
            write_jsonl(loss_path, records)
    payload = {
        "status": "ROBERTA_PAIR_STRATIFIED_RESPONSE_EVALUATED",
        "created_utc": now_utc(),
        "meaning": "Local compact-vs-repeat checkpoint response by frozen research pair strata. Read with official selected trajectories; not causal by itself.",
        "events": str(events_path),
        "events_sha256": sha256_file(events_path),
        "checkpoints": args.checkpoints,
        "compact_run": str(args.compact_run),
        "repeat_run": str(args.repeat_run),
        "device": args.device,
        "per_checkpoint": all_ck,
        "interpretation_warning": "Negative compact_minus_repeat_nll / positive repeat_minus_compact_advantage means compact-trained local advantage. Stratum concentration without stable official-family gains is not a learning-principle result.",
    }
    write_json(out_dir / "stratified_response.json", payload)
    (out_dir / "stratified_response.md").write_text(render_response_md(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_dir / "stratified_response.json"), "checkpoints": args.checkpoints}, indent=2), flush=True)
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()


def render_response_md(p: dict[str, Any]) -> str:
    lines = [
        "# research RoBERTa pair-stratified local response",
        "",
        f"Created: `{p['created_utc']}`",
        "",
        "Positive `repeat_minus_compact_advantage` means the compact-trained model predicts the masked event better than the repeat-trained model. Interpret only with selected official-compatible trajectory results.",
        "",
    ]
    for ck, rec in p["per_checkpoint"].items():
        lines.append(f"## {ck}")
        if rec.get("missing"):
            lines.append(f"Missing compact/repeat checkpoint: {rec}")
            lines.append("")
            continue
        overall = rec["summary"]["overall"]
        boot = overall.get('pair_cluster_bootstrap') or {}
        boot_txt = f", cluster bootstrap p05/p50/p95 `{boot.get('p05')}/{boot.get('p50')}/{boot.get('p95')}`, p>0 `{boot.get('p_gt_0')}`" if boot else ""
        lines.append(f"- Overall local advantage: `{overall.get('repeat_minus_compact_advantage')}` over `{overall.get('events')}` events / `{overall.get('pieces')}` pieces{boot_txt}")
        lines.append("- View/category advantages:")
        for k, v in sorted(rec["summary"].get("by_view_category", {}).items()):
            b = v.get('pair_cluster_bootstrap') or {}
            ci = f", bootstrap p05/p50/p95 `{b.get('p05')}/{b.get('p50')}/{b.get('p95')}`, p>0 `{b.get('p_gt_0')}`" if b else ""
            lines.append(f"  - `{k}`: advantage `{v.get('repeat_minus_compact_advantage')}`, events `{v.get('events')}`{ci}")
        lines.append("- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):")
        for k, v in sorted(rec["summary"].get("feature_contrasts", {}).items()):
            lines.append(f"  - `{k}` ({v.get('contrast_label')}): `{v.get('contrast')}`")
        lines.append("")
    lines.append("This file is a mechanism bridge, not an endpoint score or causal intervention.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build_only", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    ap.add_argument("--per_cell", type=int, default=120)
    ap.add_argument("--max_events", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=21543)
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--compact_run", type=Path, default=COMPACT_RUN)
    ap.add_argument("--repeat_run", type=Path, default=REPEAT_RUN)
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--write_event_losses", action="store_true")
    ap.add_argument("--bootstrap_reps", type=int, default=0)
    args = ap.parse_args()
    if args.build_only:
        build_only(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
