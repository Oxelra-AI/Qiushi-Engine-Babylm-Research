#!/usr/bin/env python3
"""research: score interference ladder on existing checkpoints.

No-training inference-only scorer.  For each of 800 frames across 10 interference
conditions, computes four PLL values (c1_a, c1_b, c2_a, c2_b) and derives crossed-
sign success, margins, and interaction.  Groups results by condition to measure the
interference gradient across training trajectories.

Reuses the D-state ablation scoring infrastructure from research.
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
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)

A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder_scores')
CACHE_SUFFIX = os.environ.get("CACHE_SUFFIX", "default")
CACHE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder_scores/hf_cache') / CACHE_SUFFIX
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

FRAME_PATH = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_frames.jsonl')
RENAMED_PATH = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder/interference_ladder_renamed_controls.jsonl')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder_scores/interference_ladder_panel_summary.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/interference_ladder_panel.md')

TARGETS = {
    **{f"scale1p75_{m}M": A02_WS / f"training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_{m}M"
       for m in [77, 78, 79, 80, 81, 82, 83, 100]},
    "legal16k_base_80M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'),
    "legal16k_base_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_100M'),
    "legal16k_80M_seed43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_80M'),
    "legal16k_100M_seed43122": _public_path('experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122/hf_model/chck_100M'),
    "legal40k_8x480_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "legal40k_depth12_100M_seed43022": _public_path('experiments/archive/representation_and_objectives/training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M'),
    "fw_compact_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/hf_model/chck_100M'),
    "fw_rowblock_100M_seed43022": _public_path('experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/hf_model/chck_100M'),
    "mlm_only_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/hf_model/final'),
    "coupled_aligned_20M": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final'),
    "coupled_shuffled_20M": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final'),
}

# EWoK references from previous work for correlation
EWOK_REFS = {
    "scale1p75_80M": {"ewok_accuracy": 0.492780, "ewok_stable_failure_frac": 0.3304},
    "scale1p75_100M": {"ewok_accuracy": 0.496456, "ewok_stable_failure_frac": 0.3262},
    "legal16k_base_80M_seed43022": {"ewok_accuracy": 0.509189, "ewok_stable_failure_frac": 0.2836},
    "legal16k_base_100M_seed43022": {"ewok_accuracy": 0.507482, "ewok_stable_failure_frac": 0.2893},
    "legal16k_80M_seed43122": {"ewok_accuracy": 0.498029, "ewok_stable_failure_frac": 0.3131},
    "legal16k_100M_seed43122": {"ewok_accuracy": 0.504988, "ewok_stable_failure_frac": 0.2932},
    "legal40k_8x480_100M_seed43022": {"ewok_accuracy": 0.511288, "ewok_stable_failure_frac": 0.2706},
    "legal40k_depth12_100M_seed43022": {"ewok_accuracy": 0.512916, "ewok_stable_failure_frac": 0.2752},
    "fw_compact_100M_seed43022": {"ewok_accuracy": 0.509324, "ewok_stable_failure_frac": 0.2878},
    "fw_rowblock_100M_seed43022": {"ewok_accuracy": 0.499475, "ewok_stable_failure_frac": 0.3114},
    "mlm_only_20M": {"ewok_accuracy": 0.499212, "ewok_stable_failure_frac": 0.3432},
}

# D-state ablation references for consistency check
DSTATE_REFS = {
    "scale1p75_82M": {"last_event": 0.41, "contradict_bare": 0.0, "explicit_final": 1.0},
    "legal16k_base_100M_seed43022": {"last_event": 0.89, "contradict_bare": 0.0, "explicit_final": 1.0},
    "legal40k_depth12_100M_seed43022": {"last_event": 0.03, "contradict_bare": 0.0, "explicit_final": 1.0},
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p: Path | str) -> str:
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


def qstats(vals: Iterable[Any]) -> dict[str, Any]:
    xs = sorted(float(v) for v in vals if finite(v))
    if not xs:
        return {"n": 0}
    def q(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p * (len(xs) - 1)
        lo, hi = math.floor(idx), math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25),
            "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def load_frames(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


class Scorer:
    def __init__(self, model, tok, device, batch_size: int):
        self.model = model
        self.tok = tok
        self.device = device
        self.batch_size = batch_size
        self.mask_id = tok.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("no mask token")

    def score(self, examples: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
        batches = []
        vals = {}
        for ex in examples:
            enc = self.tok(ex["sentence"], return_offsets_mapping=True, return_tensors=None)
            ids = list(enc["input_ids"])
            att = list(enc["attention_mask"])
            offs = list(enc["offset_mapping"])
            s, e = ex["span"]
            selected = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
            if len(selected) != 1:
                vals[(ex["rec_i"], ex["key"])] = float("nan")
                continue
            pos = selected[0]
            cur = list(ids)
            cur[pos] = self.mask_id
            batches.append({"sid": (ex["rec_i"], ex["key"]), "ids": cur, "att": att,
                           "pos": pos, "tid": ids[pos], "length": len(cur)})
        batches.sort(key=lambda x: x["length"])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.inference_mode():
            for st in range(0, len(batches), self.batch_size):
                batch = batches[st:st + self.batch_size]
                ml = max(x["length"] for x in batch)
                ids_t = torch.tensor([x["ids"] + [pad] * (ml - x["length"]) for x in batch],
                                     dtype=torch.long, device=self.device)
                att_t = torch.tensor([x["att"] + [0] * (ml - x["length"]) for x in batch],
                                     dtype=torch.long, device=self.device)
                pos_t = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=self.device)
                tid_t = torch.tensor([x["tid"] for x in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=ids_t, attention_mask=att_t)
                logits = out.logits if hasattr(out, "logits") else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                lp = torch.gather(F.log_softmax(logits[mb, pos_t], dim=-1), -1,
                                  tid_t.unsqueeze(-1)).squeeze(-1)
                for sid, val in zip([x["sid"] for x in batch], lp.detach().cpu().tolist()):
                    vals[sid] = float(val)
        return vals


def make_examples(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exs = []
    for i, r in enumerate(frames):
        c1, c2 = r["context1"], r["context2"]
        a, b = r["alt_a"], r["alt_b"]
        qt = r["query_template"]
        for ck, ctx in [("c1", c1), ("c2", c2)]:
            for ak, alt in [("a", a), ("b", b)]:
                sent, span = render(ctx, qt, alt)
                exs.append({"rec_i": i, "key": f"{ck}_{ak}", "sentence": sent, "span": span})
    return exs


def attach(frames: list[dict[str, Any]], scores: dict[tuple[int, str], float]) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(frames):
        c1a = scores.get((i, "c1_a"), float("nan"))
        c1b = scores.get((i, "c1_b"), float("nan"))
        c2a = scores.get((i, "c2_a"), float("nan"))
        c2b = scores.get((i, "c2_b"), float("nan"))
        d1 = c1a - c1b
        d2 = c2a - c2b
        rows.append({
            "frame_id": r["frame_id"],
            "base_id": r["base_id"],
            "condition": r["condition"],
            "family": r["family"],
            "object": r.get("object", ""),
            "alt_a": r["alt_a"],
            "alt_b": r["alt_b"],
            "delta1": d1,
            "delta2": d2,
            "interaction": d1 - d2,
            "min_signed_margin": min(d1, -d2),
            "crossed_success": d1 > 0 and d2 < 0,
            "same_a_bias": d1 > 0 and d2 > 0,
            "same_b_bias": d1 < 0 and d2 < 0,
        })
    return rows


def summarize_by_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    out = {}
    for cond, rs in sorted(groups.items()):
        n = len(rs)
        out[cond] = {
            "n": n,
            "crossed_success": sum(1 for r in rs if r["crossed_success"]) / n,
            "same_a_bias": sum(1 for r in rs if r["same_a_bias"]) / n,
            "same_b_bias": sum(1 for r in rs if r["same_b_bias"]) / n,
            "interaction": qstats(r["interaction"] for r in rs),
            "min_signed_margin": qstats(r["min_signed_margin"] for r in rs),
        }
    return out


def summarize_by_condition_family(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["condition"], r["family"])].append(r)
    out = {}
    for (cond, fam), rs in sorted(groups.items()):
        n = len(rs)
        key = f"{cond}/{fam}"
        out[key] = {
            "n": n,
            "crossed_success": sum(1 for r in rs if r["crossed_success"]) / n,
            "same_a_bias": sum(1 for r in rs if r["same_a_bias"]) / n,
            "same_b_bias": sum(1 for r in rs if r["same_b_bias"]) / n,
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sx = math.sqrt(sum((x - mx)**2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my)**2 for y in ys) / (n - 1))
    if sx < 1e-12 or sy < 1e-12:
        return None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((n - 1) * sx * sy)
    return r


def spearman_r(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    def rank(vals):
        s = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[s[j]] == vals[s[i]]:
                j += 1
            avg = (i + j - 1) / 2
            for k in range(i, j):
                r[s[k]] = avg
            i = j
        return r
    return pearson_r(rank(xs), rank(ys))


def score_target(target: str, frames: list[dict[str, Any]], device_name: str,
                 threads: int, batch_size: int) -> dict[str, Any]:
    path = TARGETS[target]
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    if threads > 0:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "target": target, "device": str(device),
                       "model": rel(path)}), flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True).eval().to(device)
    scorer = Scorer(model, tok, device, batch_size)
    examples = make_examples(frames)
    scores = scorer.score(examples)
    rows = attach(frames, scores)
    out_dir = _public_path('experiments/archive/representation_and_objectives/data/interference_ladder_scores/targets') / target
    write_csv(out_dir / "interference_ladder_records.csv", rows)
    by_cond = summarize_by_condition(rows)
    by_cond_fam = summarize_by_condition_family(rows)
    summary = {
        "target": target,
        "model_path": rel(path),
        "elapsed_sec": round(time.time() - t0, 2),
        "n_frames": len(frames),
        "n_examples": len(examples),
        "by_condition": by_cond,
        "by_condition_family": by_cond_fam,
    }
    (out_dir / "interference_ladder_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Short printout
    cond_crossed = {c: v["crossed_success"] for c, v in by_cond.items()}
    print(json.dumps({"event": "done", "target": target,
                       "crossed_by_condition": cond_crossed,
                       "elapsed_sec": summary["elapsed_sec"]}), flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def compute_correlations(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute correlations between condition-specific crossed success and EWoK references."""
    corrs = {}
    conditions = sorted(set(c for s in summaries for c in s["by_condition"]))

    for cond in conditions:
        targets_with_refs = []
        for s in summaries:
            t = s["target"]
            if t in EWOK_REFS and cond in s["by_condition"]:
                targets_with_refs.append((t, s["by_condition"][cond]["crossed_success"]))

        if len(targets_with_refs) < 3:
            continue

        xs_acc = [EWOK_REFS[t]["ewok_accuracy"] for t, _ in targets_with_refs]
        xs_sf = [EWOK_REFS[t]["ewok_stable_failure_frac"] for t, _ in targets_with_refs]
        ys = [cs for _, cs in targets_with_refs]

        corrs[cond] = {
            "n": len(targets_with_refs),
            "ewok_accuracy_pearson": pearson_r(xs_acc, ys),
            "ewok_accuracy_spearman": spearman_r(xs_acc, ys),
            "ewok_stable_failure_pearson": pearson_r(xs_sf, ys),
            "ewok_stable_failure_spearman": spearman_r(xs_sf, ys),
        }
    return corrs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(TARGETS.keys()))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    frames = load_frames(FRAME_PATH)
    print(json.dumps({"event": "loaded_frames", "count": len(frames)}), flush=True)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    summaries = []
    for t in targets:
        summaries.append(score_target(t, frames, args.device, args.threads, args.batch_size))

    # Correlations to EWoK references
    corrs = compute_correlations(summaries)

    # D-state consistency check
    dstate_checks = {}
    for s in summaries:
        t = s["target"]
        if t in DSTATE_REFS:
            ref = DSTATE_REFS[t]
            cond_map = {
                "last_event": "last_event",
                "contradict_bare": "contradict_bare",
                "explicit_final": "explicit_final",
            }
            checks = {}
            for dstate_key, ladder_key in cond_map.items():
                if ladder_key in s["by_condition"]:
                    ladder_val = s["by_condition"][ladder_key]["crossed_success"]
                    ref_val = ref.get(dstate_key)
                    if ref_val is not None:
                        checks[dstate_key] = {
                            "dstate_ref": ref_val,
                            "ladder_val": ladder_val,
                            "delta": ladder_val - ref_val,
                        }
            if checks:
                dstate_checks[t] = checks

    # Build panel summary
    panel = {
        "status": "INTERFERENCE_LADDER_SCORED",
        "created_utc": now(),
        "frame_path": rel(FRAME_PATH),
        "n_targets": len(summaries),
        "n_frames": len(frames),
        "ewok_correlations_by_condition": corrs,
        "dstate_consistency_checks": dstate_checks,
        "targets": [{
            "target": s["target"],
            "elapsed_sec": s["elapsed_sec"],
            "by_condition": {c: {"crossed": v["crossed_success"],
                                  "margin_median": v["min_signed_margin"].get("median")}
                             for c, v in s["by_condition"].items()},
        } for s in summaries],
    }
    OUT_JSON.write_text(json.dumps(panel, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")

    # Markdown table
    conditions_order = [
        "no_context", "explicit_final", "last_event", "consistent_prior",
        "distractor_event", "contradict_bare", "contradict_temporal",
        "contradict_reinforced", "explicit_override", "reported_event",
    ]
    lines = [
        "# research interference ladder panel",
        "",
        f"Status: **SCORED** ({len(summaries)} targets, {len(frames)} frames)",
        "",
        "## Crossed success by condition",
        "",
    ]
    # Header
    hdr = "| target |"
    sep = "|---|"
    for c in conditions_order:
        short = c[:12]
        hdr += f" {short} |"
        sep += "---:|"
    lines.append(hdr)
    lines.append(sep)
    for s in summaries:
        row = f"| {s['target']} |"
        for c in conditions_order:
            v = s["by_condition"].get(c, {}).get("crossed_success")
            row += f" {v:.3f} |" if v is not None else " |"
        lines.append(row)

    # EWoK correlations
    lines.extend(["", "## EWoK correlations by condition", ""])
    lines.append("| condition | n | EWoK acc Pearson | EWoK acc Spearman | EWoK SF Pearson | EWoK SF Spearman |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in conditions_order:
        if c in corrs:
            cr = corrs[c]
            ap = cr.get("ewok_accuracy_pearson")
            asp = cr.get("ewok_accuracy_spearman")
            sfp = cr.get("ewok_stable_failure_pearson")
            sfs = cr.get("ewok_stable_failure_spearman")
            def fv(v): return f"{v:.4f}" if v is not None else "n/a"
            lines.append(f"| {c} | {cr['n']} | {fv(ap)} | {fv(asp)} | {fv(sfp)} | {fv(sfs)} |")

    # D-state checks
    if dstate_checks:
        lines.extend(["", "## D-state consistency", ""])
        for t, checks in dstate_checks.items():
            for k, v in checks.items():
                lines.append(f"- {t}/{k}: D-state ref={v['dstate_ref']:.2f}, ladder={v['ladder_val']:.3f}, delta={v['delta']:.3f}")

    lines.extend(["", f"JSON: `{rel(OUT_JSON)}`"])
    _public_path('research/notes/representation_and_objectives').mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"event": "panel_complete", "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}),
          flush=True)


if __name__ == "__main__":
    main()
