#!/usr/bin/env python3
"""research: no-training D-state surface/polarity ablation.

The v3 micro-world panel found a striking universal failure on D_state_update_order.
Before interpreting it as a noncommutative state-update deficit, test whether the
failure is caused by the redundant context wording (initial state + same-state event
+ final opposite event) rather than inability to use the last update.  This script
scores existing checkpoints only on D rows under three fixed context forms:
  - full_v3: original frozen v3 context.
  - last_event_only: only the final event sentence, e.g. "Kate opened the bag."
  - initial_plus_last: initial-state sentence plus final event.
It also scores opposite polarity alternatives and no-context priors.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import collections
import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

# writable caches before transformers
USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/d_state_surface_ablation')
CACHE_SUFFIX = os.environ.get("DSTATE_CACHE_SUFFIX", "default")
CACHE_ROOT = _public_path('experiments/archive/representation_and_objectives/data/d_state_surface_ablation/hf_cache') / CACHE_SUFFIX
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

FRAME_PATH = _public_path('experiments/archive/representation_and_objectives/data/counterfactual_micro_world_v3/counterfactual_micro_world_v3_frames.jsonl')
OUT_JSON = _public_path('experiments/archive/representation_and_objectives/data/d_state_surface_ablation/d_state_surface_ablation_summary.json')
OUT_MD = _public_path('research/notes/representation_and_objectives/d_state_surface_ablation.md')

TARGETS = {
    **{
        f"scale1p75_{m}M": A02_WS / f"training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_{m}M"
        for m in [77, 78, 79, 80, 81, 82, 83, 100]
    },
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


def f(x: Any) -> float:
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
        lo = math.floor(idx); hi = math.ceil(idx)
        if lo == hi:
            return xs[lo]
        return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)
    return {"n": len(xs), "min": xs[0], "p05": q(0.05), "p25": q(0.25), "mean": statistics.fmean(xs), "median": statistics.median(xs), "p75": q(0.75), "p95": q(0.95), "max": xs[-1]}


def load_d_frames() -> list[dict[str, Any]]:
    rows = []
    with FRAME_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r.get("family") == "D_state_update_order":
                    rows.append(r)
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 D rows, got {len(rows)}")
    return rows


def render(context: str, query_template: str, target: str) -> tuple[str, tuple[int, int]]:
    pre, post = query_template.split("{target}", 1)
    q = pre + target + post
    lead = (context.strip() + " ") if context.strip() else ""
    sent = lead + q
    return sent, (len(lead) + len(pre), len(lead) + len(pre) + len(target))


def context_forms(r: dict[str, Any]) -> dict[str, tuple[str, str]]:
    # Frozen slots make this robust; fall back to sentence splitting if needed.
    s = r.get("slots", {})
    actor, obj = s.get("actor"), s.get("object")
    ev_a, ev_b = s.get("event_a"), s.get("event_b")
    a, b = r.get("alt_a"), r.get("alt_b")
    if actor and obj and ev_a and ev_b and a and b:
        full1, full2 = r["context1"], r["context2"]
        last1 = f"{actor} {ev_a} the {obj}."
        last2 = f"{actor} {ev_b} the {obj}."
        init_last1 = f"The {obj} was {b}. {actor} {ev_a} the {obj}."
        init_last2 = f"The {obj} was {a}. {actor} {ev_b} the {obj}."
        final_only_state1 = f"The {obj} became {a}."
        final_only_state2 = f"The {obj} became {b}."
        return {
            "full_v3": (full1, full2),
            "last_event_only": (last1, last2),
            "initial_plus_last": (init_last1, init_last2),
            "explicit_final_state": (final_only_state1, final_only_state2),
        }
    parts1 = [x.strip() + "." for x in r["context1"].split(".") if x.strip()]
    parts2 = [x.strip() + "." for x in r["context2"].split(".") if x.strip()]
    return {"full_v3": (r["context1"], r["context2"]), "last_event_only": (parts1[-1], parts2[-1]), "initial_plus_last": (" ".join([parts1[0], parts1[-1]]), " ".join([parts2[0], parts2[-1]]))}


class Scorer:
    def __init__(self, model, tok, device, batch_size: int):
        self.model = model
        self.tok = tok
        self.device = device
        self.batch_size = batch_size
        self.mask_id = tok.mask_token_id
        if self.mask_id is None:
            raise RuntimeError("no mask token")

    def score(self, examples: list[dict[str, Any]]) -> dict[tuple[int, str, str], float]:
        batches = []
        vals = {}
        for ex in examples:
            enc = self.tok(ex["sentence"], return_offsets_mapping=True, return_tensors=None)
            ids = list(enc["input_ids"]); att = list(enc["attention_mask"]); offs = list(enc["offset_mapping"])
            s, e = ex["span"]
            selected = [i for i, (a, b) in enumerate(offs) if not (a == b == 0) and b > s and a < e]
            if len(selected) != 1:
                vals[(ex["rec_i"], ex["form"], ex["key"])] = float("nan")
                continue
            pos = selected[0]
            cur = list(ids); cur[pos] = self.mask_id
            batches.append({"sid": (ex["rec_i"], ex["form"], ex["key"]), "ids": cur, "att": att, "pos": pos, "tid": ids[pos], "length": len(cur)})
        batches.sort(key=lambda x: x["length"])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.inference_mode():
            for st in range(0, len(batches), self.batch_size):
                batch = batches[st:st+self.batch_size]
                ml = max(x["length"] for x in batch)
                ids = torch.tensor([x["ids"] + [pad]*(ml-x["length"]) for x in batch], dtype=torch.long, device=self.device)
                att = torch.tensor([x["att"] + [0]*(ml-x["length"]) for x in batch], dtype=torch.long, device=self.device)
                pos = torch.tensor([x["pos"] for x in batch], dtype=torch.long, device=self.device)
                tid = torch.tensor([x["tid"] for x in batch], dtype=torch.long, device=self.device)
                out = self.model(input_ids=ids, attention_mask=att)
                logits = out.logits if hasattr(out, "logits") else out[0]
                mb = torch.arange(logits.shape[0], device=self.device)
                lp = torch.gather(F.log_softmax(logits[mb, pos], dim=-1), -1, tid.unsqueeze(-1)).squeeze(-1)
                for sid, val in zip([x["sid"] for x in batch], lp.detach().cpu().tolist()):
                    vals[sid] = float(val)
        return vals


def make_examples(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exs = []
    for i, r in enumerate(frames):
        forms = context_forms(r)
        forms["no_context"] = ("", "")
        for form, (c1, c2) in forms.items():
            for ck, ctx in [("c1", c1), ("c2", c2)]:
                for ak, alt in [("a", r["alt_a"]), ("b", r["alt_b"] )]:
                    sent, span = render(ctx, r["query_template"], alt)
                    exs.append({"rec_i": i, "form": form, "key": f"{ck}_{ak}", "sentence": sent, "span": span})
    return exs


def attach(frames: list[dict[str, Any]], scores: dict[tuple[int, str, str], float], target: str) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(frames):
        for form in sorted(context_forms(r).keys() | {"no_context"}):
            c1a = scores.get((i, form, "c1_a"), float("nan")); c1b = scores.get((i, form, "c1_b"), float("nan"))
            c2a = scores.get((i, form, "c2_a"), float("nan")); c2b = scores.get((i, form, "c2_b"), float("nan"))
            d1 = c1a - c1b; d2 = c2a - c2b
            rows.append({
                "target": target,
                "frame_id": r["frame_id"],
                "subtype": r["subtype"],
                "alt_a": r["alt_a"], "alt_b": r["alt_b"],
                "form": form,
                "delta1": d1, "delta2": d2,
                "correct_margin_c1": d1, "correct_margin_c2": -d2,
                "interaction": d1 - d2,
                "min_signed_margin": min(d1, -d2),
                "crossed_success": d1 > 0 and d2 < 0,
                "same_a_bias": d1 > 0 and d2 > 0,
                "same_b_bias": d1 < 0 and d2 < 0,
                "prior_ratio": r.get("target_prior", {}).get("max_min_ratio"),
            })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["form"]].append(r)
    for form, rs in sorted(groups.items()):
        n = len(rs)
        out[form] = {
            "n": n,
            "crossed_success": sum(1 for r in rs if r["crossed_success"]) / n,
            "same_a_bias": sum(1 for r in rs if r["same_a_bias"]) / n,
            "same_b_bias": sum(1 for r in rs if r["same_b_bias"]) / n,
            "delta1": qstats(r["delta1"] for r in rs),
            "delta2": qstats(r["delta2"] for r in rs),
            "interaction": qstats(r["interaction"] for r in rs),
            "min_signed_margin": qstats(r["min_signed_margin"] for r in rs),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def fmt(x: Any) -> str:
    try:
        v = float(x)
        return f"{v:.4f}" if math.isfinite(v) else ""
    except Exception:
        return str(x)


def score_target(target: str, device_name: str, threads: int, batch_size: int) -> dict[str, Any]:
    path = TARGETS[target]
    if not path.exists():
        raise FileNotFoundError(path)
    if threads > 0:
        torch.set_num_threads(threads)
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "target": target, "device": str(device), "model": rel(path)}), flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(path), trust_remote_code=True).eval().to(device)
    frames = load_d_frames()
    scorer = Scorer(model, tok, device, batch_size)
    scores = scorer.score(make_examples(frames))
    rows = attach(frames, scores, target)
    out_dir = _public_path('experiments/archive/representation_and_objectives/data/d_state_surface_ablation/targets') / target
    write_csv(out_dir / "d_state_surface_records.csv", rows)
    summary = {"target": target, "model_path": rel(path), "elapsed_sec": round(time.time()-t0, 2), "summary_by_form": summarize(rows), "records_csv": rel(out_dir / "d_state_surface_records.csv")}
    (out_dir / "d_state_surface_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "target": target, "summary": {k: {"crossed": v["crossed_success"], "minmed": v["min_signed_margin"].get("median")} for k, v in summary["summary_by_form"].items()}}), flush=True)
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="scale1p75_82M,legal16k_base_100M_seed43022,legal40k_depth12_100M_seed43022,mlm_only_20M,coupled_shuffled_20M")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    summaries = [score_target(t, args.device, args.threads, args.batch_size) for t in targets]
    panel = {"status": "D_STATE_SURFACE_ABLATION_DONE", "created_utc": now(), "frame_path": rel(FRAME_PATH), "targets": summaries}
    OUT_JSON.write_text(json.dumps(panel, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# research D-state surface ablation", "", "Status: **DONE**", "", "This is no-training scoring of the D_state_update_order rows under different context surfaces.", "", "| target | form | crossed | same A bias | same B bias | Δ1 med | Δ2 med | M med | min-margin med |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        for form, d in s["summary_by_form"].items():
            lines.append(f"| {s['target']} | {form} | {fmt(d['crossed_success'])} | {fmt(d['same_a_bias'])} | {fmt(d['same_b_bias'])} | {fmt(d['delta1'].get('median'))} | {fmt(d['delta2'].get('median'))} | {fmt(d['interaction'].get('median'))} | {fmt(d['min_signed_margin'].get('median'))} |")
    lines.extend(["", "Scientific reading: if last_event_only and explicit_final_state flip to high crossed success while full_v3 fails, v3 D failure is largely a redundancy/order-surface artifact. If all forms fail, it is stronger evidence for a state-update representation deficit.", "", f"JSON: `{OUT_JSON.relative_to(USER_ROOT)}`"])
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": panel["status"], "out_json": rel(OUT_JSON), "out_md": rel(OUT_MD)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
