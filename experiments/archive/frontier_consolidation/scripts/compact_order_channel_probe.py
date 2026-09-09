#!/usr/bin/env python3
"""research source-absent compact-channel probe for ordered-vs-scrambled checkpoints.

Evaluates fixed compact-side denoising events (retained content / source-absent content /
function-other) on the *same ordered compact probe text* for both research arms.  This
adds the source-absent-channel readout to the matched ordered-vs-scrambled
real-training experiment, so official family scores can be interpreted together with
whether coherent compact order preserves the rare source-absent target channel.

The probe is CPU/GPU inference only; it does not train or change checkpoints.
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

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts/compact_order_channel_probe.py')
for _ in range(12):
    if (_public_path("experiments")).exists():
        break
    USER_ROOT = _public_path('experiments/archive/frontier_consolidation/scripts')
STUDY = _public_path('experiments/archive/frontier_consolidation')
WS = _public_path('experiments/archive/frontier_consolidation')
PAIR_FILE = _public_path('experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl')
TOKENIZER = _public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer')
DEFAULT_OUT = _public_path('experiments/archive/frontier_consolidation/data/compact_order_channel_probe')
SEQ_LEN = 256
CATEGORIES = ["retained_content", "source_absent_content", "function_other"]
CHECKPOINTS = ["chck_20M", "chck_40M"]
ARMS = {
    "ordered": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_ordered_40M_seed43022/hf_model'),
    "scrambled": _public_path('experiments/archive/frontier_consolidation/training/runs/compact_order_scrambled_40M_seed43022/hf_model'),
}
FUNC_WORDS = frozenset("a an the this that these those my your his her its our their "
    "is am are was were be been being have has had do does did will would shall should "
    "can could may might must need ought to of in on at by for with from into through "
    "during before after above below between under over about against along across "
    "around behind beside beyond down near off since toward upon within without "
    "and but or nor so yet both either neither not no nor if then else than as "
    "which who whom whose what when where how while until because although though "
    "even also just only still already very much more most less least too quite "
    "really rather than such same other another each every all some any few many "
    "no more several enough another each own same different many little much few "
    "i me we us you he him she her it they them myself yourself himself herself "
    "itself ourselves themselves one ones there here up out off away again back "
    "now then so however therefore thus hence moreover furthermore nevertheless "
    "meanwhile otherwise instead indeed certainly perhaps maybe probably "
    "been being getting going doing having making taking".split())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def is_content(w: str) -> bool:
    n = norm(w)
    return n not in FUNC_WORDS and len(n) > 1


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


def read_pairs() -> list[dict[str, Any]]:
    rows = []
    with PAIR_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_events(tok, per_category: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
    pairs = read_pairs()
    counts_before = collections.Counter()
    for r in pairs:
        src = str(r["source_text"])
        side = str(r["view_text"])
        source_norms = {norm(w) for w in src.split() if norm(w)}
        side_words = side.split()
        full_ids = tok(src + " " + side, add_special_tokens=False, truncation=True, max_length=SEQ_LEN)["input_ids"]
        if len(full_ids) < 4:
            continue
        for wi, w in enumerate(side_words):
            nw = norm(w)
            if not nw:
                continue
            span = target_span(tok, src, side_words, wi)
            if span is None or span[0] >= span[1] or span[1] > len(full_ids):
                continue
            if is_content(w) and nw in source_norms:
                cat = "retained_content"
            elif is_content(w) and nw not in source_norms:
                cat = "source_absent_content"
            else:
                cat = "function_other"
            counts_before[cat] += 1
            buckets[cat].append({
                "pair_id": str(r["pair_id"]),
                "source_text": src,
                "side_text": side,
                "word": w,
                "word_index": wi,
                "category": cat,
                "span": [int(span[0]), int(span[1])],
                "n_pieces": int(span[1] - span[0]),
                "input_ids": [int(x) for x in full_ids],
            })
    selected: list[dict[str, Any]] = []
    selected_counts = collections.Counter()
    for cat in CATEGORIES:
        evs = list(buckets[cat])
        rng.shuffle(evs)
        by_pair = collections.Counter()
        keep = []
        for e in evs:
            if by_pair[e["pair_id"]] >= 2:
                continue
            keep.append(e)
            by_pair[e["pair_id"]] += 1
            if len(keep) >= per_category:
                break
        selected_counts[cat] = len(keep)
        selected.extend(keep)
    rng.shuffle(selected)
    meta = {
        "pairs": len(pairs),
        "counts_before_pair_cap": dict(counts_before),
        "selected_counts": dict(selected_counts),
        "per_category_requested": per_category,
        "seed": seed,
        "n_events": len(selected),
    }
    return selected, meta


def make_batch(events: list[dict[str, Any]], mask_id: int, pad_id: int):
    max_len = min(max(len(e["input_ids"]) for e in events), SEQ_LEN)
    x = torch.full((len(events), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(events), max_len), -100, dtype=torch.long)
    for i, e in enumerate(events):
        ids = list(e["input_ids"][:max_len])
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        st, en = e["span"]
        en = min(int(en), max_len)
        st = int(st)
        if st < en:
            y[i, st:en] = torch.tensor(ids[st:en], dtype=torch.long)
            x[i, st:en] = mask_id
    return x, y


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    s = sorted(float(x) for x in xs)
    n = len(s)
    def q(p: float) -> float:
        if n == 1:
            return s[0]
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {
        "n": n,
        "mean": round(statistics.mean(s), 6),
        "median": round(statistics.median(s), 6),
        "p10": round(q(0.10), 6),
        "p25": round(q(0.25), 6),
        "p75": round(q(0.75), 6),
        "p90": round(q(0.90), 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
    }


def eval_model(model_path: Path, events: list[dict[str, Any]], tok, device: str, batch_size: int) -> dict[int, dict[str, Any]]:
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
            per_tok = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="none", ignore_index=-100).view(y.shape)
            mask = y != -100
            for b, e in enumerate(batch):
                vals = per_tok[b][mask[b]]
                out[int(e["event_index"])] = {
                    "loss_mean": float(vals.mean().detach().cpu()),
                    "loss_sum": float(vals.sum().detach().cpu()),
                    "pieces": int(vals.numel()),
                }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def weighted_delta(records: list[dict[str, Any]], lhs: str, rhs: str) -> float:
    num = sum(r[lhs]["loss_sum"] - r[rhs]["loss_sum"] for r in records)
    den = sum(r[lhs]["pieces"] for r in records)
    return num / den if den else float("nan")


def mean_event_delta(records: list[dict[str, Any]], lhs: str, rhs: str) -> float:
    return statistics.mean(r[lhs]["loss_mean"] - r[rhs]["loss_mean"] for r in records) if records else float("nan")


def finite_float(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def r6_or_none(x: Any) -> float | None:
    return round(float(x), 6) if finite_float(x) else None


def category_contrasts_from_deltas(deltas: dict[str, float]) -> dict[str, float]:
    """Contrasts where negative means larger ordered advantage for source-absent content."""
    src = deltas.get("source_absent_content", float("nan"))
    ret = deltas.get("retained_content", float("nan"))
    fun = deltas.get("function_other", float("nan"))
    controls = [x for x in [ret, fun] if finite_float(x)]
    ctrl_mean = statistics.mean(controls) if controls else float("nan")
    return {
        "source_absent_minus_retained_content": src - ret if finite_float(src) and finite_float(ret) else float("nan"),
        "source_absent_minus_function_other": src - fun if finite_float(src) and finite_float(fun) else float("nan"),
        "source_absent_minus_controls_mean": src - ctrl_mean if finite_float(src) and finite_float(ctrl_mean) else float("nan"),
    }


def category_interaction_bootstrap(records_by_cat: dict[str, list[dict[str, Any]]], lhs: str, rhs: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Cluster-bootstrap category interactions for the ordered-vs-scrambled NLL delta.

    The fixed probe uses ordered compact text, so the ordered arm is evaluated on its native
    sequence distribution and the scrambled arm is not. A lower source_absent_content NLL is
    source-absent-selective only if the ordered advantage is materially larger than the retained
    content and function-word advantages measured on the same fixed events.
    """
    by_pair: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(lambda: {cat: [] for cat in CATEGORIES})
    for cat, recs in records_by_cat.items():
        for r in recs:
            by_pair[str(r["pair_id"])][cat].append(r)
    pairs = sorted(by_pair)
    orig_piece = {cat: weighted_delta(records_by_cat.get(cat, []), lhs, rhs) for cat in CATEGORIES}
    orig_event = {cat: mean_event_delta(records_by_cat.get(cat, []), lhs, rhs) for cat in CATEGORIES}
    orig_piece_contrasts = category_contrasts_from_deltas(orig_piece)
    orig_event_contrasts = category_contrasts_from_deltas(orig_event)
    rng = random.Random(seed)
    boot_piece: dict[str, list[float]] = {k: [] for k in orig_piece_contrasts}
    boot_event: dict[str, list[float]] = {k: [] for k in orig_event_contrasts}
    if pairs and n_boot > 0:
        for _ in range(n_boot):
            sample_by_cat: dict[str, list[dict[str, Any]]] = {cat: [] for cat in CATEGORIES}
            for _j in range(len(pairs)):
                p = rng.choice(pairs)
                for cat in CATEGORIES:
                    sample_by_cat[cat].extend(by_pair[p][cat])
            pd = {cat: weighted_delta(sample_by_cat.get(cat, []), lhs, rhs) for cat in CATEGORIES}
            ed = {cat: mean_event_delta(sample_by_cat.get(cat, []), lhs, rhs) for cat in CATEGORIES}
            for k, v in category_contrasts_from_deltas(pd).items():
                if finite_float(v):
                    boot_piece[k].append(float(v))
            for k, v in category_contrasts_from_deltas(ed).items():
                if finite_float(v):
                    boot_event[k].append(float(v))
    def pack(orig: dict[str, float], boots: dict[str, list[float]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in orig.items():
            vals = boots.get(k, [])
            out[k] = {
                "contrast_delta": r6_or_none(v),
                "bootstrap_interval": quantiles(vals) if vals else {},
                "p_contrast_lt0": round(sum(x < 0 for x in vals) / len(vals), 4) if vals else None,
                "n_boot": len(vals),
                "meaning": "negative means source_absent_content has a larger ordered-training NLL advantage than the comparison category",
            }
        return out
    return {
        "category_piece_weighted_deltas": {cat: r6_or_none(v) for cat, v in orig_piece.items()},
        "category_event_mean_deltas": {cat: r6_or_none(v) for cat, v in orig_event.items()},
        "piece_weighted_contrasts": pack(orig_piece_contrasts, boot_piece),
        "event_mean_contrasts": pack(orig_event_contrasts, boot_event),
        "n_pair_clusters_union": len(pairs),
    }

def quantiles(xs: list[float]) -> dict[str, float]:
    s = sorted(float(x) for x in xs)
    if not s:
        return {}
    n = len(s)
    def q(p: float) -> float:
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"p025": round(q(0.025), 6), "p05": round(q(0.05), 6), "median": round(q(0.5), 6), "p95": round(q(0.95), 6), "p975": round(q(0.975), 6)}


def bootstrap(records: list[dict[str, Any]], lhs: str, rhs: str, n_boot: int, seed: int) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)
    pairs = sorted(by_pair)
    rng = random.Random(seed)
    ev_vals = []
    wt_vals = []
    for _ in range(n_boot):
        sample = []
        for _j in range(len(pairs)):
            sample.extend(by_pair[rng.choice(pairs)])
        ev_vals.append(mean_event_delta(sample, lhs, rhs))
        wt_vals.append(weighted_delta(sample, lhs, rhs))
    orig_ev = mean_event_delta(records, lhs, rhs)
    orig_wt = weighted_delta(records, lhs, rhs)
    return {
        "n_events": len(records),
        "n_pair_clusters": len(pairs),
        "event_mean_delta": round(orig_ev, 6),
        "piece_weighted_delta": round(orig_wt, 6),
        "event_cluster_bootstrap": quantiles(ev_vals),
        "piece_cluster_bootstrap": quantiles(wt_vals),
        "p_event_delta_gt0": round(sum(v > 0 for v in ev_vals) / max(1, len(ev_vals)), 4),
        "p_piece_delta_gt0": round(sum(v > 0 for v in wt_vals) / max(1, len(wt_vals)), 4),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize_training(run_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run_root": str(run_root)}
    mpath = run_root / "scientific_metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        out["cum_words_exposed"] = m.get("cum_words_exposed") or m.get("max_word_exposure")
        out["param_count"] = m.get("param_count")
        out["checkpoints"] = m.get("checkpoints")
        out["num_consumed_examples"] = m.get("num_consumed_examples")
        out["data_source_type"] = m.get("data_source_type")
    logp = run_root / "training_log.jsonl"
    if logp.exists():
        rows = [json.loads(l) for l in logp.read_text(encoding="utf-8").splitlines() if l.strip()]
        out["training_log_rows"] = len(rows)
        if rows:
            out["first_log"] = rows[0]
            out["last_log"] = rows[-1]
            tail = rows[-min(50, len(rows)):]
            out["tail50_loss_mean"] = statistics.mean(r["loss"] for r in tail if "loss" in r)
            out["loss_finite"] = all(math.isfinite(float(r.get("loss", float("nan")))) for r in rows)
            out["cum_words_last_log"] = rows[-1].get("cumulative_word_exposure")
    for ck in CHECKPOINTS:
        out[f"{ck}_exists"] = (run_root / "hf_model" / ck / "model.safetensors").exists()
    return out


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# research compact ordered-vs-scrambled source-absent channel probe", ""]
    lines.append(payload["meaning"])
    lines.append("")
    lines.append(f"Events: {payload['inputs']['events']} from {payload['inputs']['pair_file']} with selected counts {payload['inputs']['event_meta']['selected_counts']}.")
    lines.append("")
    if payload.get("loader_confound_summary"):
        c = payload["loader_confound_summary"]
        lines.append("## Realized loader/WWM small differences from research")
        lines.append(f"compact-minus-scrambled changed block: Δ active tokens {c.get('delta_active_tokens_sum')}, Δ candidate groups {c.get('delta_candidate_groups_sum')}, Δ view active tokens {c.get('delta_view_active_tokens_sum')}, Δ CPU-proxy masked tokens {c.get('delta_masked_tokens_cpu_proxy_sum')}, Δ CPU-proxy masked view/source {c.get('delta_masked_view_tokens_cpu_proxy_sum')}/{c.get('delta_masked_source_tokens_cpu_proxy_sum')}.")
        lines.append("")
    for ck in CHECKPOINTS:
        if ck not in payload.get("summary", {}):
            continue
        lines.append(f"## {ck}: ordered minus scrambled fixed-event NLL")
        lines.append("Positive means ordered arm has *higher/worse* loss than scrambled; negative means ordered arm has lower/better fixed-event loss.")
        lines.append("Because this fixed probe uses ordered compact text, the ordered arm is on its native sequence distribution and the scrambled arm is not; source-absent mechanism evidence requires a category interaction, not only a negative source_absent_content row.")
        lines.append("| category | piece Δ | 95% CI | P(Δ<0) | events | pairs |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cat in CATEGORIES:
            rec = payload["summary"][ck][cat]["ordered_minus_scrambled"]
            ci = rec.get("piece_cluster_bootstrap", {})
            p_lt0 = 1.0 - rec.get("p_piece_delta_gt0", 0.0)
            lines.append(f"| {cat} | {rec.get('piece_weighted_delta')} | [{ci.get('p025')}, {ci.get('p975')}] | {p_lt0:.4f} | {rec.get('n_events')} | {rec.get('n_pair_clusters')} |")
        inter = payload.get("category_interactions", {}).get(ck, {}).get("piece_weighted_contrasts", {})
        if inter:
            lines.append("")
            lines.append("Category interaction: negative means the source_absent_content ordered advantage is larger than the comparison category.")
            lines.append("| contrast | Δ difference | 95% interval | P(diff<0) |")
            lines.append("|---|---:|---:|---:|")
            for name, rec in inter.items():
                ci = rec.get("bootstrap_interval", {})
                lines.append(f"| {name} | {rec.get('contrast_delta')} | [{ci.get('p025')}, {ci.get('p975')}] | {rec.get('p_contrast_lt0')} |")
        lines.append("")
    if payload.get("missing"):
        lines.append("## Missing checkpoints")
        for m in payload["missing"]:
            lines.append(f"- {m}")
        lines.append("")
    lines.append(f"JSON: `{payload['json_path']}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--per-category", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=22443023)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "compact_order_channel_probe.json"
    if out_json.exists() and not args.force:
        print(out_json)
        return
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), local_files_only=True, use_fast=True)
    events, event_meta = build_events(tok, args.per_category, args.seed)
    for i, e in enumerate(events):
        e["event_index"] = i
    event_public = [{k: v for k, v in e.items() if k != "input_ids"} for e in events]
    write_jsonl(args.out_dir / "probe_events.jsonl", event_public)
    missing = []
    losses: dict[str, dict[str, dict[int, dict[str, Any]]]] = {arm: {} for arm in ARMS}
    for arm, hf_root in ARMS.items():
        for ck in args.checkpoints:
            mp = hf_root / ck
            if not (mp / "model.safetensors").exists():
                missing.append(str(mp))
                continue
            print(f"eval {arm}/{ck}", flush=True)
            losses[arm][ck] = eval_model(mp, events, tok, args.device, args.batch_size)
    event_loss_rows: list[dict[str, Any]] = []
    records_by_ck_cat: dict[str, dict[str, list[dict[str, Any]]]] = {ck: {cat: [] for cat in CATEGORIES} for ck in args.checkpoints}
    for e in events:
        erow = {"event_index": e["event_index"], "pair_id": e["pair_id"], "category": e["category"], "word": e.get("word"), "n_pieces": e.get("n_pieces")}
        for ck in args.checkpoints:
            if ck in losses["ordered"] and ck in losses["scrambled"]:
                rec = {"pair_id": e["pair_id"], "category": e["category"], "ordered": losses["ordered"][ck][e["event_index"]], "scrambled": losses["scrambled"][ck][e["event_index"]]}
                records_by_ck_cat[ck][e["category"]].append(rec)
                erow[ck] = {"ordered": rec["ordered"], "scrambled": rec["scrambled"], "ordered_minus_scrambled_loss_mean": rec["ordered"]["loss_mean"] - rec["scrambled"]["loss_mean"]}
        event_loss_rows.append(erow)
    write_jsonl(args.out_dir / "event_losses.jsonl", event_loss_rows)
    # Also save a compact CSV view for quick filtering.
    flat_rows = []
    for r in event_loss_rows:
        base = {k: r[k] for k in ["event_index", "pair_id", "category", "word", "n_pieces"]}
        for ck in args.checkpoints:
            if ck in r:
                flat_rows.append({**base, "checkpoint": ck, "ordered_loss_mean": r[ck]["ordered"]["loss_mean"], "scrambled_loss_mean": r[ck]["scrambled"]["loss_mean"], "delta_ordered_minus_scrambled": r[ck]["ordered_minus_scrambled_loss_mean"]})
    write_csv(args.out_dir / "event_loss_delta_rows.csv", flat_rows)
    summary: dict[str, Any] = {}
    category_interactions: dict[str, Any] = {}
    for ck in args.checkpoints:
        if ck not in losses["ordered"] or ck not in losses["scrambled"]:
            continue
        summary[ck] = {}
        for ci, cat in enumerate(CATEGORIES):
            recs = records_by_ck_cat[ck][cat]
            summary[ck][cat] = {
                "ordered_loss_event": stats([r["ordered"]["loss_mean"] for r in recs]),
                "scrambled_loss_event": stats([r["scrambled"]["loss_mean"] for r in recs]),
                "ordered_minus_scrambled": bootstrap(recs, "ordered", "scrambled", args.n_boot, seed=204000 + 1000 * args.checkpoints.index(ck) + 17 * ci),
            }
        category_interactions[ck] = category_interaction_bootstrap(records_by_ck_cat[ck], "ordered", "scrambled", args.n_boot, seed=205000 + 1000 * args.checkpoints.index(ck))
    loader_confound = None
    audit_path = _public_path('experiments/archive/frontier_consolidation/data/deberta_loader_realized_context_audit/deberta_loader_realized_context_audit.json')
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        loader_confound = audit.get("contrasts", {}).get("compact_minus_compact_scrambled")
    training = {arm: summarize_training((ARMS[arm]).parent) for arm in ARMS}
    payload = {
        "status": "COMPACT_ORDER_CHANNEL_PROBE",
        "meaning": "Fixed ordered compact-side denoising readout on the research arms. Negative ordered_minus_scrambled means ordered training gives lower NLL on the same ordered source+compact masked event. Because this probe uses ordered compact text, source_absent_content only identifies the compact source-absent channel when its ordered advantage is materially larger than retained_content and function_other; category_interactions report that comparison.",
        "inputs": {
            "pair_file": str(PAIR_FILE),
            "pair_file_sha256": sha256_file(PAIR_FILE),
            "tokenizer": str(TOKENIZER),
            "tokenizer_sha256": sha256_file(_public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json')),
            "events": len(events),
            "event_meta": event_meta,
            "checkpoints": args.checkpoints,
            "arms": {k: str(v) for k, v in ARMS.items()},
            "device": args.device,
            "batch_size": args.batch_size,
            "n_boot": args.n_boot,
        },
        "missing": missing,
        "training_summaries": training,
        "loader_confound_summary": loader_confound,
        "event_file": str(args.out_dir / "probe_events.jsonl"),
        "event_losses_jsonl": str(args.out_dir / "event_losses.jsonl"),
        "event_loss_delta_csv": str(args.out_dir / "event_loss_delta_rows.csv"),
        "summary": summary,
        "category_interactions": category_interactions,
        "elapsed_sec": round(time.time() - t0, 1),
        "json_path": str(out_json),
    }
    write_json(out_json, payload)
    (args.out_dir / "compact_order_channel_probe.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out_json), "missing": missing, "main": {ck: {cat: summary[ck][cat]["ordered_minus_scrambled"]["piece_weighted_delta"] for cat in CATEGORIES} for ck in summary}, "category_interactions": {ck: category_interactions[ck].get("piece_weighted_contrasts", {}) for ck in category_interactions}, "elapsed_sec": payload["elapsed_sec"]}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
