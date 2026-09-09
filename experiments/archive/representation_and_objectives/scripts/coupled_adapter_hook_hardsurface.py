#!/usr/bin/env python3
"""research: runtime adapter-output hook for coupled sparse20 hard-surface decomposition.

Scientific question
-------------------
The matched shuffled control showed that coupled sparse20 hard-surface repair is
mostly correspondence-free.  Before any new training, separate the live adapter
output from stock/backbone trajectory displacement on the existing coupled aligned
and coupled shuffled checkpoints.  research sparse20 checkpoints ignore
config.adapter_scale, so this script changes adapter output by runtime hooks on
`adapter.forward`, not by editing config or weights.

It evaluates fixed diagnostics only:
  * research GlobalPIQA parallel/nonparallel rank/margin reader;
  * research EWoK stable-reversal subset reader.

No training, no checkpoint mutation, no chck_82M access.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

# CPU-only: this is a posthoc mechanism readout, not a training/eval spend on H100.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

USER_ROOT = _public_path('.')
os.chdir(USER_ROOT)
A01_WS = _public_path('experiments/archive/representation_and_objectives')
A02_WS = _public_path('experiments/archive/frontier_consolidation')
SCRIPT_DIR = _public_path('experiments/archive/representation_and_objectives/scripts')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface')
STABLE_EWOK_CSV = _public_path('experiments/archive/representation_and_objectives/data/ewok_interaction_synthesis/both_models_stable_failure_rows.csv')

# Transformers dynamic-module cache paths are captured at import time by some
# utilities. Set writable local caches before importing the research/108
# readers or any HF code they import.
for _d in [OUT_ROOT, _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/home'), _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/xdg_cache'), _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/hf_cache'), _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/hf_cache/modules'), _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/torch_cache'), _public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/tmp')]:
    _d.mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/home'))
os.environ["XDG_CACHE_HOME"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/xdg_cache'))
os.environ["HF_HOME"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/hf_cache'))
os.environ["TRANSFORMERS_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/hf_cache'))
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/hf_cache/modules'))
os.environ["TORCH_HOME"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/torch_cache'))
os.environ["TMPDIR"] = str(_public_path('experiments/archive/representation_and_objectives/data/coupled_adapter_hook_hardsurface/tmp'))

TARGETS: dict[str, dict[str, Any]] = {
    "mlm_only_20M": {
        "label": "A02 exact MLM-only 20M baseline (adapter initialized/unused)",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/dualview_mlm_only_20M_seed43022/scientific_metrics.json'),
        "family": "mlm_only",
    },
    "coupled_aligned_20M": {
        "label": "A02 coupled sparse20 true-correspondence aligned 20M",
        "model_path": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/frontier_consolidation/training/runs/coupled_sparse20_aligned_20M_seed43022/scientific_metrics.json'),
        "family": "coupled_aligned",
    },
    "coupled_shuffled_20M": {
        "label": "A01 coupled sparse20 shuffled-correspondence matched control 20M",
        "model_path": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/hf_model/final'),
        "metrics_path": _public_path('experiments/archive/representation_and_objectives/training/runs/coupled_sparse20_shuffled_20M_seed43022/scientific_metrics.json'),
        "family": "coupled_shuffled",
    },
}


def import_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


GP = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/globalpiqa_margin_reader.py'), "gp_for_step180_adapter_hook")
EW = import_from(_public_path('experiments/archive/representation_and_objectives/scripts/fw_ewok_interaction_reader.py'), "ewok_for_step180_adapter_hook")


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


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def qstats(vals) -> dict[str, Any]:
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


def load_stable_indices(max_rows: int | None = None) -> tuple[list[int], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    with STABLE_EWOK_CSV.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    indices = [int(r["global_index"]) for r in rows]
    if max_rows is not None and max_rows > 0:
        indices = indices[:max_rows]
    return indices, {
        "source": rel(STABLE_EWOK_CSV),
        "definition": "research rows that were stable conditional-reversal failures for both earlier legal models; fixed before dual-view route.",
        "available_rows": len(rows),
        "used_rows": len(indices),
    }


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            out[k] = json.dumps(v, ensure_ascii=False, sort_keys=True)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_for_csv(r) for r in rows]
    if not flat:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in flat:
        for k in r:
            if k not in seen:
                seen.add(k); keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(flat)


def install_adapter_scale_hooks(model: Any, scale: float) -> dict[str, Any]:
    """Multiply every research adapter output by `scale` at runtime.

    scale=1 preserves the checkpoint. scale=0 disables live adapter output while
    preserving all stock parameters and all non-adapter computation.
    """
    adapters = []
    for li, layer in enumerate(getattr(model.deberta.encoder, "layer", [])):
        adapter = getattr(layer, "adapter", None)
        if adapter is None or not hasattr(adapter, "forward"):
            continue
        orig_forward = adapter.forward
        def make_forward(orig: Callable, a: float):
            def wrapped(hidden_states):
                return orig(hidden_states) * a
            return wrapped
        adapter.forward = make_forward(orig_forward, scale)  # type: ignore[method-assign]
        adapters.append(li)
    return {"scale": scale, "n_adapters_hooked": len(adapters), "adapter_layers": adapters}


def load_hooked(model_path: Path, scale: float, device: Any) -> tuple[Any, Any, dict[str, Any]]:
    tok = EW.AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = EW.AutoModelForMaskedLM.from_pretrained(str(model_path), trust_remote_code=True)
    hook = install_adapter_scale_hooks(model, scale)
    model.eval().to(device)
    return tok, model, hook


def compact_gp(res: dict[str, Any]) -> dict[str, Any]:
    return {
        "target": res.get("target"),
        "label": res.get("label"),
        "model_root": rel(res.get("model_root")),
        "revision": res.get("revision"),
        "device": res.get("device"),
        "model_load_sec": res.get("model_load_sec"),
        "modes": {mode: payload.get("summary", {}) for mode, payload in res.get("modes", {}).items() if isinstance(payload, dict)},
    }


def run_gp_variant(target_name: str, meta: dict[str, Any], scale: float, out_root: Path, gp_max_items: int | None, threads: int) -> tuple[dict[str, Any], dict[str, Any]]:
    # Patch the imported research module's model loader to add hooks after load.
    orig_auto = GP.AutoModelForMaskedLM
    hook_records: list[dict[str, Any]] = []
    class HookedAuto:
        @staticmethod
        def from_pretrained(model_root, *args, **kwargs):
            model = orig_auto.from_pretrained(model_root, *args, **kwargs)
            hook_records.append(install_adapter_scale_hooks(model, scale))
            return model
    GP.AutoModelForMaskedLM = HookedAuto
    try:
        GP.OUT_ROOT = out_root / "raw"
        GP.NOTE = out_root / "globalpiqa_hook_note.md"
        GP.TARGETS = {target_name: {"label": f"{meta['label']} | adapter_output_scale={scale}", "model_root": Path(meta["model_path"]), "revision": None, "family": meta.get("family")}}
        print(json.dumps({"event": "globalpiqa_hook_start", "target": target_name, "scale": scale, "utc": now_utc()}), flush=True)
        res = GP.run_target(target_name, ["parallel", "nonparallel"], batch_size=8, non_causal_batch_size=32, max_items=gp_max_items, threads=threads)
    finally:
        GP.AutoModelForMaskedLM = orig_auto
    compact = compact_gp(res)
    compact["adapter_hook"] = hook_records[-1] if hook_records else {"scale": scale, "n_adapters_hooked": 0}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / f"{target_name}_scale{scale:g}_globalpiqa_margins.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    row_dir = out_root / "rows"; row_dir.mkdir(parents=True, exist_ok=True)
    for mode, payload in res.get("modes", {}).items():
        write_csv(row_dir / f"{target_name}_scale{scale:g}_{mode}_rows.csv", payload.get("rows", []))
    par = compact.get("modes", {}).get("parallel", {})
    hard = par.get("always_wrong_subset", {}) if isinstance(par, dict) else {}
    print(json.dumps({"event": "globalpiqa_hook_done", "target": target_name, "scale": scale, "parallel_accuracy": par.get("accuracy") if isinstance(par, dict) else None, "hard52_accuracy": hard.get("accuracy"), "hard52_mean_top_minus_correct": hard.get("mean_top_minus_correct"), "hook": compact["adapter_hook"]}, ensure_ascii=False), flush=True)
    return compact, compact["adapter_hook"]


def run_ewok_variant(target_name: str, meta: dict[str, Any], scale: float, out_root: Path, max_rows: int | None, threads: int, row_batch_size: int, masked_batch_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if threads > 0:
        import torch
        torch.set_num_threads(threads)
    else:
        import torch
    device = torch.device("cpu")
    all_rows = EW.load_rows()
    indices, subset_meta = load_stable_indices(max_rows)
    model_path = Path(meta["model_path"])
    print(json.dumps({"event": "ewok_hook_start", "target": target_name, "scale": scale, "n_rows": len(indices), "utc": now_utc()}), flush=True)
    t0 = time.time()
    tokenizer, model, hook = load_hooked(model_path, scale, device)
    scorer = EW.BatchedPseudoScorer(model, tokenizer, device, masked_batch_size)
    out_records: list[dict[str, Any]] = []
    for b0 in range(0, len(indices), row_batch_size):
        idx_batch = indices[b0:b0 + row_batch_size]
        recs, tasks = EW.build_batch_records(all_rows, idx_batch)
        scores = scorer.score(tasks)
        EW.attach_scores(recs, scores)
        for rec in recs:
            rec["target"] = target_name
            rec["adapter_output_scale"] = scale
            rec["subset_source"] = "both_models_stable_failure_rows"
        out_records.extend(recs)
        if b0 == 0 or ((b0 // row_batch_size) % 10 == 0):
            print(json.dumps({"event": "ewok_hook_rows_done", "target": target_name, "scale": scale, "rows_done": len(out_records), "rows_total": len(indices), "utc": now_utc()}), flush=True)
    summary = EW.summarize(out_records)
    by_domain = EW.summarize_by(out_records, "domain")
    by_context_diff = EW.summarize_by(out_records, "ContextDiff")
    tdir = out_root / f"{target_name}_scale{scale:g}"
    tdir.mkdir(parents=True, exist_ok=True)
    EW.write_csv(tdir / "ewok_stable_subset_records.csv", out_records)
    EW.write_csv(tdir / "ewok_stable_subset_by_domain.csv", by_domain)
    EW.write_csv(tdir / "ewok_stable_subset_by_context_diff.csv", by_context_diff)
    result = {
        "target": target_name,
        "label": meta["label"],
        "model_path": rel(model_path),
        "device": str(device),
        "adapter_hook": hook,
        "subset": subset_meta,
        "threads": threads,
        "row_batch_size": row_batch_size,
        "masked_batch_size": masked_batch_size,
        "elapsed_sec": round(time.time() - t0, 2),
        "summary": summary,
        "by_domain": by_domain,
        "by_context_diff": by_context_diff,
    }
    (tdir / "ewok_stable_subset_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "ewok_hook_done", "target": target_name, "scale": scale, "accuracy": summary.get("accuracy"), "stable_failure": summary.get("stable_failure"), "stable_failure_frac_all": summary.get("stable_failure_frac_all"), "interaction_mean_wrong": summary.get("interaction_sum_wrong", {}).get("mean") if isinstance(summary.get("interaction_sum_wrong"), dict) else None, "hook": hook}, ensure_ascii=False), flush=True)
    del model, tokenizer, scorer, out_records
    gc.collect()
    return {k: v for k, v in result.items() if k not in {"by_domain", "by_context_diff"}}, hook


def gp_delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode in ["parallel", "nonparallel"]:
        b = base.get("modes", {}).get(mode, {})
        c = cand.get("modes", {}).get(mode, {})
        md: dict[str, Any] = {}
        for key in ["accuracy", "chance_adjusted_accuracy"]:
            if isinstance(b.get(key), (int, float)) and isinstance(c.get(key), (int, float)):
                md[key] = c[key] - b[key]
        for key in ["rank1", "rank2", "rank3", "rank4"]:
            br = b.get("correct_rank_counts", {}).get(key[-1]) if isinstance(b.get("correct_rank_counts"), dict) else None
            cr = c.get("correct_rank_counts", {}).get(key[-1]) if isinstance(c.get("correct_rank_counts"), dict) else None
            if isinstance(br, (int, float)) and isinstance(cr, (int, float)):
                md[f"all_{key}"] = cr - br
        bh = b.get("always_wrong_subset"); ch = c.get("always_wrong_subset")
        if isinstance(bh, dict) and isinstance(ch, dict):
            for key in ["accuracy", "mean_top_minus_correct", "median_top_minus_correct", "small_wrong_margin_le_0p25_nats", "small_wrong_margin_le_0p50_nats"]:
                if isinstance(bh.get(key), (int, float)) and isinstance(ch.get(key), (int, float)):
                    md[f"hard52_{key}"] = ch[key] - bh[key]
            brr = bh.get("correct_rank_counts", {}) if isinstance(bh.get("correct_rank_counts"), dict) else {}
            crr = ch.get("correct_rank_counts", {}) if isinstance(ch.get("correct_rank_counts"), dict) else {}
            for r in ["1", "2", "3", "4"]:
                if isinstance(brr.get(r), (int, float)) or isinstance(crr.get(r), (int, float)):
                    md[f"hard52_rank{r}"] = int(crr.get(r, 0)) - int(brr.get(r, 0))
        bm = b.get("all_rows_margin_summary"); cm = c.get("all_rows_margin_summary")
        if isinstance(bm, dict) and isinstance(cm, dict):
            for key in ["mean_top_minus_correct", "median_top_minus_correct"]:
                if isinstance(bm.get(key), (int, float)) and isinstance(cm.get(key), (int, float)):
                    md[f"all_{key}"] = cm[key] - bm[key]
        out[mode] = md
    return out


def ewok_delta(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, Any]:
    b = base.get("summary", {})
    c = cand.get("summary", {})
    out: dict[str, Any] = {}
    for key in ["accuracy", "saved_wrong", "stable_failure", "stable_failure_frac_all", "stable_failure_frac_wrong", "within_both_positive_wrong_frac", "local_both_actual_over_swapped_positive_wrong_frac"]:
        if isinstance(b.get(key), (int, float)) and isinstance(c.get(key), (int, float)):
            out[key] = c[key] - b[key]
    for block in ["interaction_sum_all", "interaction_sum_wrong", "interaction_mean_wrong", "deletion_interaction_sum_wrong", "interaction_minus_deletion_sum_wrong"]:
        bb = b.get(block); cc = c.get(block)
        if isinstance(bb, dict) and isinstance(cc, dict):
            for stat in ["mean", "median", "p05", "p95"]:
                if isinstance(bb.get(stat), (int, float)) and isinstance(cc.get(stat), (int, float)):
                    out[f"{block}_{stat}"] = cc[stat] - bb[stat]
    return out


def summarize_gp(gp: dict[str, Any]) -> dict[str, Any]:
    par = gp.get("modes", {}).get("parallel", {})
    hard = par.get("always_wrong_subset", {}) if isinstance(par, dict) else {}
    return {
        "parallel_accuracy": par.get("accuracy"),
        "parallel_rank_counts": par.get("correct_rank_counts"),
        "parallel_mean_top_minus_correct": (par.get("all_rows_margin_summary") or {}).get("mean_top_minus_correct") if isinstance(par.get("all_rows_margin_summary"), dict) else None,
        "hard52_accuracy": hard.get("accuracy"),
        "hard52_rank_counts": hard.get("correct_rank_counts"),
        "hard52_mean_top_minus_correct": hard.get("mean_top_minus_correct"),
        "hard52_median_top_minus_correct": hard.get("median_top_minus_correct"),
    }


def summarize_ewok(ew: dict[str, Any]) -> dict[str, Any]:
    s = ew.get("summary", {})
    iw = s.get("interaction_sum_wrong", {}) if isinstance(s.get("interaction_sum_wrong"), dict) else {}
    return {
        "n": s.get("n"),
        "accuracy": s.get("accuracy"),
        "saved_wrong": s.get("saved_wrong"),
        "stable_failure": s.get("stable_failure"),
        "stable_failure_frac_all": s.get("stable_failure_frac_all"),
        "interaction_sum_wrong_mean": iw.get("mean"),
        "interaction_sum_wrong_median": iw.get("median"),
    }


def make_note(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# research coupled adapter-output hook hard-surface decomposition")
    lines.append("")
    lines.append(f"Status: **{summary['status']}**")
    lines.append("")
    lines.append("research sparse20 checkpoints ignore `config.adapter_scale`; this readout therefore multiplies each adapter's actual returned tensor at runtime. Scale 0 disables live adapter output without editing weights or config.")
    lines.append("")
    lines.append("## Variants")
    for k, v in summary["variants"].items():
        lines.append(f"- `{k}`: target={v['target']} scale={v['adapter_output_scale']} hook={v['adapter_hook']} metrics_mode={v['training_metrics'].get('mode') if isinstance(v.get('training_metrics'), dict) else None}")
    lines.append("")
    lines.append("## Surface values")
    for k in summary["variant_order"]:
        lines.append(f"- `{k}` GlobalPIQA: {json.dumps(summarize_gp(summary['globalpiqa'][k]), ensure_ascii=False)}")
        lines.append(f"- `{k}` EWoK: {json.dumps(summarize_ewok(summary['ewok_stable_subset'][k]), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Key deltas")
    for k, v in summary["deltas"].items():
        lines.append(f"- `{k}` GlobalPIQA_parallel: {json.dumps(v.get('globalpiqa', {}).get('parallel', {}), ensure_ascii=False)}")
        lines.append(f"- `{k}` EWoK: {json.dumps(v.get('ewok', {}), ensure_ascii=False)}")
    lines.append("")
    lines.append("## Scientific reading")
    for item in summary.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append(f"JSON: `{rel(summary['summary_json'])}`")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def interpret(summary: dict[str, Any]) -> list[str]:
    items: list[str] = []
    d = summary.get("deltas", {})
    for name in ["aligned_live_minus_disabled", "shuffled_live_minus_disabled"]:
        ew = d.get(name, {}).get("ewok", {})
        gp = d.get(name, {}).get("globalpiqa", {}).get("parallel", {})
        if ew:
            items.append(f"{name}: enabling live adapter changes EWoK stable failures by {ew.get('stable_failure')} and accuracy by {ew.get('accuracy')}; GlobalPIQA hard52 accuracy delta {gp.get('hard52_accuracy')} and hard52 mean-margin delta {gp.get('hard52_mean_top_minus_correct')} nats.")
    for name in ["aligned_disabled_minus_mlm", "shuffled_disabled_minus_mlm", "aligned_live_minus_mlm", "shuffled_live_minus_mlm"]:
        ew = d.get(name, {}).get("ewok", {})
        gp = d.get(name, {}).get("globalpiqa", {}).get("parallel", {})
        if ew:
            items.append(f"{name}: EWoK stable_failure delta {ew.get('stable_failure')}, accuracy delta {ew.get('accuracy')}; GlobalPIQA hard52 acc delta {gp.get('hard52_accuracy')}, hard52 margin delta {gp.get('hard52_mean_top_minus_correct')}.")
    # Mechanism-level automatic reading, deliberately narrow.
    al_off = d.get("aligned_disabled_minus_mlm", {}).get("ewok", {}).get("stable_failure")
    al_live = d.get("aligned_live_minus_mlm", {}).get("ewok", {}).get("stable_failure")
    sh_off = d.get("shuffled_disabled_minus_mlm", {}).get("ewok", {}).get("stable_failure")
    sh_live = d.get("shuffled_live_minus_mlm", {}).get("ewok", {}).get("stable_failure")
    if all(isinstance(x, (int, float)) for x in [al_off, al_live, sh_off, sh_live]):
        items.append("EWoK repair carrier: compare disabled-vs-MLM with live-vs-MLM. If disabled deltas are near zero while live deltas are strongly negative, repair is live-adapter output; if disabled remains strongly negative, repair is embedded in stock trajectory. See exact deltas above rather than treating this as a training route.")
    items.append("This readout does not measure official broad cheap7 under disabled adapters. If hard repair is correspondence-free and lives in the same live adapter branch that also damages broad columns, route should move to mechanism rethink or an isolating construction, not endpoint-scale spend.")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="research coupled adapter output hook hard-surface readout")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--ewok-row-batch-size", type=int, default=64)
    ap.add_argument("--ewok-masked-batch-size", type=int, default=160)
    ap.add_argument("--ewok-max-rows", type=int, default=0)
    ap.add_argument("--gp-max-items", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    for d in [out_root, out_root / "home", out_root / "xdg_cache", out_root / "hf_cache", out_root / "hf_cache/modules", out_root / "torch_cache", out_root / "tmp"]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str((out_root / "home").resolve())
    os.environ["XDG_CACHE_HOME"] = str((out_root / "xdg_cache").resolve())
    os.environ["HF_HOME"] = str((out_root / "hf_cache").resolve())
    os.environ["TRANSFORMERS_CACHE"] = str((out_root / "hf_cache").resolve())
    os.environ["HF_MODULES_CACHE"] = str((out_root / "hf_cache/modules").resolve())
    os.environ["TORCH_HOME"] = str((out_root / "torch_cache").resolve())
    os.environ["TMPDIR"] = str((out_root / "tmp").resolve())

    specs = [
        ("mlm_live", "mlm_only_20M", 1.0),
        ("aligned_live", "coupled_aligned_20M", 1.0),
        ("aligned_disabled", "coupled_aligned_20M", 0.0),
        ("shuffled_live", "coupled_shuffled_20M", 1.0),
        ("shuffled_disabled", "coupled_shuffled_20M", 0.0),
    ]
    gp_max = args.gp_max_items if args.gp_max_items > 0 else None
    ew_max = args.ewok_max_rows if args.ewok_max_rows > 0 else None

    variants: dict[str, Any] = {}
    gp_results: dict[str, Any] = {}
    ewok_results: dict[str, Any] = {}
    for variant_name, target_name, scale in specs:
        meta = TARGETS[target_name]
        metrics = read_json(Path(meta["metrics_path"]))
        variants[variant_name] = {
            "target": target_name,
            "label": meta["label"],
            "family": meta["family"],
            "model_path": rel(meta["model_path"]),
            "adapter_output_scale": scale,
            "training_metrics": metrics,
            "adapter_hook": None,
        }
        gp, gp_hook = run_gp_variant(target_name + "__" + variant_name, meta, scale, out_root / "globalpiqa", gp_max, args.threads)
        gp_results[variant_name] = gp
        ew, ew_hook = run_ewok_variant(target_name + "__" + variant_name, meta, scale, out_root / "ewok_stable_subset", ew_max, args.threads, args.ewok_row_batch_size, args.ewok_masked_batch_size)
        ewok_results[variant_name] = ew
        variants[variant_name]["adapter_hook"] = {"globalpiqa": gp_hook, "ewok": ew_hook}
        gc.collect()

    def pair(label: str, base: str, cand: str) -> tuple[str, dict[str, Any]]:
        return label, {"globalpiqa": gp_delta(gp_results[base], gp_results[cand]), "ewok": ewok_delta(ewok_results[base], ewok_results[cand])}

    deltas = dict([
        pair("aligned_live_minus_disabled", "aligned_disabled", "aligned_live"),
        pair("shuffled_live_minus_disabled", "shuffled_disabled", "shuffled_live"),
        pair("aligned_disabled_minus_mlm", "mlm_live", "aligned_disabled"),
        pair("shuffled_disabled_minus_mlm", "mlm_live", "shuffled_disabled"),
        pair("aligned_live_minus_mlm", "mlm_live", "aligned_live"),
        pair("shuffled_live_minus_mlm", "mlm_live", "shuffled_live"),
        pair("aligned_live_minus_shuffled_live", "shuffled_live", "aligned_live"),
        pair("aligned_disabled_minus_shuffled_disabled", "shuffled_disabled", "aligned_disabled"),
    ])

    _, ew_meta = load_stable_indices(ew_max)
    summary: dict[str, Any] = {
        "status": "COUPLED_ADAPTER_HOOK_HARDSURFACE_DONE",
        "created_utc": now_utc(),
        "boundary": "existing-checkpoint runtime adapter-output hook; no training; research config adapter_scale is inert; chck_82M untouched",
        "variant_order": [x[0] for x in specs],
        "variants": variants,
        "globalpiqa_fixed_hard_set": {"source": rel(_public_path('experiments/archive/representation_and_objectives/data/globalpiqa_margin_synthesis/globalpiqa_margin_synthesis.json')), "definition": "research fixed cross-endpoint GlobalPIQA_parallel hard52 rows"},
        "ewok_stable_subset_definition": ew_meta,
        "globalpiqa": gp_results,
        "ewok_stable_subset": ewok_results,
        "deltas": deltas,
        "interpretation": [],
        "summary_json": str(out_root / "coupled_adapter_hook_hardsurface_summary.json"),
    }
    summary["interpretation"] = interpret(summary)
    out_json = out_root / "coupled_adapter_hook_hardsurface_summary.json"
    summary["summary_json"] = str(out_json)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    make_note(summary, out_root / "coupled_adapter_hook_hardsurface_summary.md")
    print(json.dumps({"status": summary["status"], "summary_json": rel(out_json), "summary_md": rel(out_root / "coupled_adapter_hook_hardsurface_summary.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
