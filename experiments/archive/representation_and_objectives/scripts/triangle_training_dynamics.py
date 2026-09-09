#!/usr/bin/env python3
"""research CPU-only compact-view triangle training-dynamics readout.

Reads the completed/known training logs for the matched DeBERTa-v2 compact-view
mechanism triangle:
  - compact_view_reinvest (historical trusted reference, now allowed by repaired
    GC equivalence through the checked horizon)
  - compact_repeat_reinvest (new GC control)
  - adjbreak_reinvest (new GC control)

This is deliberately NOT an official benchmark readout. It provides mechanistic
pressure from learning curves while the guarded official-compatible no-AoA/posthoc
tasks continue asynchronously. Training loss is objective/data-dependent and cannot
by itself identify a transferable principle; it can reveal schedule/hash/log
mismatches and large local predictability signatures that should shape later
interpretation of endpoint scores.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json
import math
import statistics as stats
from pathlib import Path
from typing import Any

ROOT = _public_path('.')
OUT_ROOT = _public_path('experiments/archive/representation_and_objectives/data/triangle_training_dynamics')

RUNS = {
    "compact_view_reinvest_historical": _public_path('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022'),
    "compact_repeat_reinvest_gc": _public_path('experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2'),
    "adjbreak_reinvest_gc": _public_path('experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2'),
}

REPAIRED_EQUIV = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence_repair/gc_reference_equivalence_repair.json')
DOWNSTREAM_EQUIV = _public_path('experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json')


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # stdout records may include event=..., file logs usually do not.
            step = row.get("step", row.get("update"))
            if step is None:
                continue
            row = dict(row)
            row["step_norm"] = int(step)
            if "loss" not in row and "main_loss" in row:
                row["loss"] = row["main_loss"]
            if "cumulative_word_exposure" not in row and "total_consumed_words" in row:
                row["cumulative_word_exposure"] = row["total_consumed_words"]
            rows.append(row)
    rows.sort(key=lambda r: r["step_norm"])
    return rows


def mean(xs: list[float]) -> float | None:
    return float(sum(xs) / len(xs)) if xs else None


def median(xs: list[float]) -> float | None:
    return float(stats.median(xs)) if xs else None


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs2 = sorted(xs)
    if len(xs2) == 1:
        return float(xs2[0])
    pos = q * (len(xs2) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs2[lo])
    return float(xs2[lo] * (hi - pos) + xs2[hi] * (pos - lo))


def summarize_run(run_root: Path, log: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_path = run_root / "scientific_metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    ckpts = metrics.get("checkpoints", [])
    endpoint = run_root / "hf_model/chck_100M"
    losses = [float(r["loss"]) for r in log if r.get("loss") is not None]
    return {
        "run_root": rel(run_root),
        "training_log": rel(run_root / "training_log.jsonl"),
        "scientific_metrics": rel(metrics_path) if metrics_path.exists() else None,
        "metrics_status": metrics.get("status") or metrics.get("variant"),
        "n_steps": len(log),
        "first": {k: log[0].get(k) for k in ["step_norm", "loss", "lr", "batch_words", "cumulative_word_exposure", "masked_tokens", "effective_mask_rate"]} if log else None,
        "last": {k: log[-1].get(k) for k in ["step_norm", "loss", "lr", "batch_words", "cumulative_word_exposure", "masked_tokens", "effective_mask_rate"]} if log else None,
        "loss_mean_all_steps": mean(losses),
        "loss_mean_last_100_steps": mean(losses[-100:]),
        "loss_mean_first_100_steps": mean(losses[:100]),
        "checkpoint_count_in_metrics": len(ckpts),
        "hf_chck_100M_exists": endpoint.exists(),
        "hf_chck_100M_model_exists": (endpoint / "model.safetensors").exists(),
    }


def compare_pair(name_a: str, log_a: list[dict[str, Any]], name_b: str, log_b: list[dict[str, Any]]) -> dict[str, Any]:
    by_a = {r["step_norm"]: r for r in log_a}
    by_b = {r["step_norm"]: r for r in log_b}
    common = sorted(set(by_a) & set(by_b))
    loss_delta_b_minus_a: list[float] = []
    lr_delta_abs: list[float] = []
    batch_word_mismatches = 0
    cum_word_mismatches = 0
    masked_delta: list[float] = []
    maskrate_delta: list[float] = []
    selected: dict[str, Any] = {}
    for s in common:
        a = by_a[s]
        b = by_b[s]
        if a.get("loss") is not None and b.get("loss") is not None:
            loss_delta_b_minus_a.append(float(b["loss"]) - float(a["loss"]))
        if a.get("lr") is not None and b.get("lr") is not None:
            lr_delta_abs.append(abs(float(b["lr"]) - float(a["lr"])))
        if a.get("batch_words") != b.get("batch_words"):
            batch_word_mismatches += 1
        if a.get("cumulative_word_exposure") != b.get("cumulative_word_exposure"):
            cum_word_mismatches += 1
        if a.get("masked_tokens") is not None and b.get("masked_tokens") is not None:
            masked_delta.append(float(b["masked_tokens"]) - float(a["masked_tokens"]))
        if a.get("effective_mask_rate") is not None and b.get("effective_mask_rate") is not None:
            maskrate_delta.append(float(b["effective_mask_rate"]) - float(a["effective_mask_rate"]))
        if s in {1, 26, 50, 100, 250, 500, 1000, 1500, 2000, 2529}:
            selected[str(s)] = {
                f"{name_a}_loss": a.get("loss"),
                f"{name_b}_loss": b.get("loss"),
                "delta_b_minus_a": (float(b["loss"]) - float(a["loss"])) if a.get("loss") is not None and b.get("loss") is not None else None,
                f"{name_a}_cum_words": a.get("cumulative_word_exposure"),
                f"{name_b}_cum_words": b.get("cumulative_word_exposure"),
            }
    # Decile bins by step index, not exposure, because all three schedules have 2529 updates.
    bins = []
    for start in range(1, max(common) + 1 if common else 1, 253):
        stop = min(start + 252, max(common) if common else 0)
        ds = [float(by_b[s]["loss"]) - float(by_a[s]["loss"]) for s in common if start <= s <= stop and by_a[s].get("loss") is not None and by_b[s].get("loss") is not None]
        if ds:
            bins.append({"step_start": start, "step_stop": stop, "mean_delta_b_minus_a": mean(ds), "median_delta_b_minus_a": median(ds), "n": len(ds)})
    return {
        "a": name_a,
        "b": name_b,
        "n_common_steps": len(common),
        "common_step_minmax": [common[0], common[-1]] if common else None,
        "loss_delta_b_minus_a": {
            "mean_all": mean(loss_delta_b_minus_a),
            "median_all": median(loss_delta_b_minus_a),
            "q10": quantile(loss_delta_b_minus_a, 0.10),
            "q90": quantile(loss_delta_b_minus_a, 0.90),
            "mean_first_100": mean(loss_delta_b_minus_a[:100]),
            "mean_last_100": mean(loss_delta_b_minus_a[-100:]),
            "final_step_delta": loss_delta_b_minus_a[-1] if loss_delta_b_minus_a else None,
            "fraction_b_higher_loss": mean([1.0 if d > 0 else 0.0 for d in loss_delta_b_minus_a]),
        },
        "max_abs_lr_delta": max(lr_delta_abs) if lr_delta_abs else None,
        "batch_word_mismatch_steps": batch_word_mismatches,
        "cumulative_word_mismatch_steps": cum_word_mismatches,
        "mean_masked_tokens_delta_b_minus_a": mean(masked_delta),
        "mean_effective_mask_rate_delta_b_minus_a": mean(maskrate_delta),
        "selected_steps": selected,
        "decile_step_bins": bins,
    }


def compact_interpretation(pairwise: dict[str, Any]) -> list[str]:
    out: list[str] = []
    av = pairwise["compact_view_reinvest_historical__vs__adjbreak_reinvest_gc"]["loss_delta_b_minus_a"]
    rv = pairwise["compact_view_reinvest_historical__vs__compact_repeat_reinvest_gc"]["loss_delta_b_minus_a"]
    ar = pairwise["compact_repeat_reinvest_gc__vs__adjbreak_reinvest_gc"]["loss_delta_b_minus_a"]
    out.append(
        "Training curves are read only as mechanism context: they share the same 2529-step LR schedule and word-exposure grid, but loss values are on different text streams/objective targets and cannot replace the official-compatible no-AoA endpoint readout."
    )
    out.append(
        f"Adjbreak minus compact_view mean loss delta={av['mean_all']:.6f}, final={av['final_step_delta']:.6f}, last100={av['mean_last_100']:.6f}; positive values mean the adjacency-broken corpus is locally harder for the MLM objective."
    )
    out.append(
        f"Repeat minus compact_view mean loss delta={rv['mean_all']:.6f}, final={rv['final_step_delta']:.6f}, last100={rv['mean_last_100']:.6f}."
    )
    out.append(
        f"Adjbreak minus repeat mean loss delta={ar['mean_all']:.6f}, final={ar['final_step_delta']:.6f}, last100={ar['mean_last_100']:.6f}; this isolates source-own adjacency at nearly fixed rewrite marginal more directly than either comparison to view."
    )
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = {name: load_log(root / "training_log.jsonl") for name, root in RUNS.items()}
    runs = {name: summarize_run(root, logs[name]) for name, root in RUNS.items()}
    pairs_to_compare = [
        ("compact_view_reinvest_historical", "compact_repeat_reinvest_gc"),
        ("compact_view_reinvest_historical", "adjbreak_reinvest_gc"),
        ("compact_repeat_reinvest_gc", "adjbreak_reinvest_gc"),
    ]
    pairwise: dict[str, Any] = {}
    for a, b in pairs_to_compare:
        pairwise[f"{a}__vs__{b}"] = compare_pair(a, logs[a], b, logs[b])
    equiv_repair = load_json(REPAIRED_EQUIV) if REPAIRED_EQUIV.exists() else None
    equiv_downstream = load_json(DOWNSTREAM_EQUIV) if DOWNSTREAM_EQUIV.exists() else None
    payload = {
        "status": "TRIANGLE_TRAINING_DYNAMICS_READOUT",
        "purpose": "CPU-only dynamics evidence for compact-view mechanism triangle while official-compatible no-AoA/posthoc tasks continue asynchronously.",
        "implementation_equivalence": {
            "repair_path": rel(REPAIRED_EQUIV),
            "repair_exists": REPAIRED_EQUIV.exists(),
            "repair_interpretation": equiv_repair.get("interpretation") if isinstance(equiv_repair, dict) else None,
            "repair_allows": (equiv_repair.get("allows_historical_reference_for_causal_triangle") if isinstance(equiv_repair, dict) else None),
            "downstream_summary_path": rel(DOWNSTREAM_EQUIV),
            "downstream_summary_exists": DOWNSTREAM_EQUIV.exists(),
            "downstream_allows": ((equiv_downstream.get("allows_historical_reference_for_causal_triangle") or equiv_downstream.get("causal_interpretation_allowed") or equiv_downstream.get("allows")) if isinstance(equiv_downstream, dict) else None),
        },
        "runs": runs,
        "pairwise": pairwise,
        "interpretation_notes": compact_interpretation(pairwise),
        "limits": [
            "No official-compatible benchmark columns are computed here.",
            "A lower training loss may reflect easier local predictability or repetition, not better generalization.",
            "Mechanism conclusion still requires the guarded no-AoA endpoint table, broad-competence profile, and posthoc item turnover under the research seed-spread rules."
        ],
    }
    out_json = _public_path('experiments/archive/representation_and_objectives/data/triangle_training_dynamics/triangle_training_dynamics_mechanism.json')
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = _public_path('research/documents/representation_and_objectives/data/triangle_training_dynamics/triangle_training_dynamics_mechanism.md')
    lines = [
        "# research compact-view triangle training dynamics",
        "",
        "This CPU-only readout compares training logs for the completed matched triangle. It is not a benchmark result.",
        "",
        "## Run endpoints",
    ]
    for name, r in runs.items():
        last = r.get("last") or {}
        lines.append(f"- `{name}`: steps={r['n_steps']}, chck_100M_exists={r['hf_chck_100M_exists']}, final_loss={last.get('loss')}, final_words={last.get('cumulative_word_exposure')}")
    lines.extend(["", "## Pairwise loss deltas (b minus a)"])
    for key, cmp in pairwise.items():
        d = cmp["loss_delta_b_minus_a"]
        lines.append(f"- `{key}`: mean={d['mean_all']:.6f}, median={d['median_all']:.6f}, final={d['final_step_delta']:.6f}, last100={d['mean_last_100']:.6f}, fraction_b_higher={d['fraction_b_higher_loss']:.3f}, LR_max_delta={cmp['max_abs_lr_delta']}")
    lines.extend(["", "## Interpretation notes"])
    for note in payload["interpretation_notes"]:
        lines.append(f"- {note}")
    lines.extend(["", f"JSON: `{rel(out_json)}`"])
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out_json": rel(out_json), "out_md": rel(md), "final_losses": {k: runs[k]["last"]["loss"] for k in runs}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
