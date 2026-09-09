#!/usr/bin/env python3
"""research: parameter-space geometry for late-iterate stabilization candidates.

This CPU/file-only probe measures whether the same-trajectory late checkpoints form a
smooth local path or a rapidly turning path.  It does not evaluate BabyLM tasks, does
not upload models, and does not submit anything.  Its role is to make any later
same-trajectory weight-averaging screen scientifically interpretable after the
pending seed43122 score trajectory arrives.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from safetensors.torch import load_file
import torch


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
STUDY = ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
OUT_DIR = WORKSPACE / "data" / "late_weight_geometry_probe"

TRAJECTORIES = {
    "reference_scale1p75_seed43022": WORKSPACE / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model",
    "scale1p25_seed43022": WORKSPACE / "training/runs/adapter128_scale1p25_seed43022_dense100M/hf_model",
    "scale1p75_seed43122": WORKSPACE / "training/runs/adapter128_scale1p75_seed43122_dense100M/hf_model",
}
ENDPOINTS = [f"chck_{m}M" for m in range(76, 101, 2)]
REFERENCE_SCORE_JSON = WORKSPACE / "data/selected_trajectory_eval_scale1p75_seed43022_reference_common2M/selected_trajectory.json"
AVG_CANDIDATE = WORKSPACE / "data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform/model.safetensors"

ADAPTER_MARKERS = ("adapter.", ".adapter.")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")


def is_adapter_key(k: str) -> bool:
    return any(m in k for m in ADAPTER_MARKERS)


def zero_stats() -> dict[str, float]:
    return {"n_params": 0.0, "sum_sq": 0.0}


def add_tensor_norm(stats: dict[str, float], t: torch.Tensor) -> None:
    x = t.detach().to(dtype=torch.float32)
    stats["n_params"] += float(x.numel())
    stats["sum_sq"] += float(torch.dot(x.flatten(), x.flatten()).item())


def add_diff_stats(stats: dict[str, float], a: torch.Tensor, b: torch.Tensor) -> None:
    d = a.detach().to(dtype=torch.float32) - b.detach().to(dtype=torch.float32)
    stats["n_params"] += float(d.numel())
    stats["sum_sq"] += float(torch.dot(d.flatten(), d.flatten()).item())


def finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    ss = max(float(stats["sum_sq"]), 0.0)
    n = max(float(stats["n_params"]), 1.0)
    return {"n_params": int(stats["n_params"]), "l2": math.sqrt(ss), "rms": math.sqrt(ss / n)}


def partition_norms(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = zero_stats(); adapter = zero_stats(); backbone = zero_stats()
    n_tensors = 0; n_adapter_tensors = 0
    for k, t in state.items():
        n_tensors += 1
        add_tensor_norm(total, t)
        if is_adapter_key(k):
            n_adapter_tensors += 1
            add_tensor_norm(adapter, t)
        else:
            add_tensor_norm(backbone, t)
    return {
        "total": finalize_stats(total),
        "adapter": finalize_stats(adapter),
        "backbone_plus_heads": finalize_stats(backbone),
        "n_tensors": n_tensors,
        "n_adapter_tensors": n_adapter_tensors,
    }


def diff_partition(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, Any]:
    total = zero_stats(); adapter = zero_stats(); backbone = zero_stats()
    for k in sorted(a.keys()):
        if k not in b:
            raise RuntimeError(f"Missing key {k} in second state")
        add_diff_stats(total, a[k], b[k])
        if is_adapter_key(k):
            add_diff_stats(adapter, a[k], b[k])
        else:
            add_diff_stats(backbone, a[k], b[k])
    return {"total": finalize_stats(total), "adapter": finalize_stats(adapter), "backbone_plus_heads": finalize_stats(backbone)}


def diff_state(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: (a[k].detach().to(dtype=torch.float32) - b[k].detach().to(dtype=torch.float32)).contiguous() for k in sorted(a.keys())}


def dot_norm(diff_a: dict[str, torch.Tensor], diff_b: dict[str, torch.Tensor], key_filter: str = "total") -> dict[str, float]:
    dot = 0.0; aa = 0.0; bb = 0.0; n = 0
    for k in sorted(diff_a.keys()):
        use = True
        if key_filter == "adapter":
            use = is_adapter_key(k)
        elif key_filter == "backbone_plus_heads":
            use = not is_adapter_key(k)
        if not use:
            continue
        x = diff_a[k].flatten()
        y = diff_b[k].flatten()
        dot += float(torch.dot(x, y).item())
        aa += float(torch.dot(x, x).item())
        bb += float(torch.dot(y, y).item())
        n += x.numel()
    denom = math.sqrt(max(aa, 0.0) * max(bb, 0.0))
    return {
        "n_params": int(n),
        "dot": dot,
        "norm_a": math.sqrt(max(aa, 0.0)),
        "norm_b": math.sqrt(max(bb, 0.0)),
        "cosine": (dot / denom) if denom > 0 else None,
    }


def load_state(run_dir: Path, endpoint: str) -> dict[str, torch.Tensor]:
    mf = run_dir / endpoint / "model.safetensors"
    if not mf.exists():
        raise FileNotFoundError(mf)
    return load_file(str(mf), device="cpu")


def load_model_file(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(path)
    return load_file(str(path), device="cpu")


def trajectory_geometry(label: str, run_dir: Path, endpoints: list[str]) -> dict[str, Any]:
    endpoint_norms: list[dict[str, Any]] = []
    step_diffs: list[dict[str, Any]] = []
    turn_records: list[dict[str, Any]] = []
    prev_state = None
    prev_ep = None
    prev_diff = None
    for ep in endpoints:
        state = load_state(run_dir, ep)
        norms = partition_norms(state)
        endpoint_norms.append({"endpoint": ep, "model_file": rel(run_dir / ep / "model.safetensors"), **norms})
        if prev_state is not None and prev_ep is not None:
            dpart = diff_partition(state, prev_state)
            cur_diff = diff_state(state, prev_state)
            step_diffs.append({
                "from": prev_ep,
                "to": ep,
                "delta_m_words": int(ep.split("_")[1][:-1]) - int(prev_ep.split("_")[1][:-1]),
                **dpart,
            })
            if prev_diff is not None:
                turn_records.append({
                    "prev_step": f"{step_diffs[-2]['from']}->{step_diffs[-2]['to']}",
                    "step": f"{prev_ep}->{ep}",
                    "total": dot_norm(prev_diff, cur_diff, "total"),
                    "adapter": dot_norm(prev_diff, cur_diff, "adapter"),
                    "backbone_plus_heads": dot_norm(prev_diff, cur_diff, "backbone_plus_heads"),
                })
            prev_diff = cur_diff
        prev_state = state
        prev_ep = ep
    return {
        "label": label,
        "run_dir": rel(run_dir),
        "endpoints": endpoints,
        "endpoint_norms": endpoint_norms,
        "step_diffs": step_diffs,
        "successive_update_cosines": turn_records,
        "summary": summarize_turning(step_diffs, turn_records),
    }


def finite_vals(records: list[dict[str, Any]], key_path: list[str]) -> list[float]:
    vals = []
    for r in records:
        cur: Any = r
        ok = True
        for k in key_path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False; break
            cur = cur[k]
        if ok and cur is not None:
            try:
                vals.append(float(cur))
            except Exception:
                pass
    return vals


def summarize_turning(step_diffs: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in ["total", "adapter", "backbone_plus_heads"]:
        dnorms = finite_vals(step_diffs, [part, "l2"])
        coses = finite_vals(turns, [part, "cosine"])
        out[part] = {
            "mean_step_l2": float(mean(dnorms)) if dnorms else None,
            "std_step_l2": float(pstdev(dnorms)) if len(dnorms) > 1 else (0.0 if dnorms else None),
            "mean_successive_update_cosine": float(mean(coses)) if coses else None,
            "min_successive_update_cosine": float(min(coses)) if coses else None,
            "max_successive_update_cosine": float(max(coses)) if coses else None,
            "n_negative_successive_cosines": sum(1 for c in coses if c < 0.0),
            "n_successive_cosines": len(coses),
        }
    # local turn around the reference 84M peak if present
    for tr in turns:
        if tr.get("prev_step") == "chck_82M->chck_84M" and tr.get("step") == "chck_84M->chck_86M":
            out["turn_around_84M"] = tr
    return out


def reference_scores() -> dict[str, Any]:
    if not REFERENCE_SCORE_JSON.exists():
        return {"status": "missing", "path": rel(REFERENCE_SCORE_JSON)}
    rows = read_json(REFERENCE_SCORE_JSON)
    by_ep = {r.get("endpoint"): r for r in rows if isinstance(r, dict)}
    selected = {ep: by_ep.get(ep) for ep in ["chck_80M", "chck_82M", "chck_84M", "chck_86M", "chck_100M"] if ep in by_ep}
    return {"status": "ok", "path": rel(REFERENCE_SCORE_JSON), "selected_rows": selected}


def average_candidate_geometry() -> dict[str, Any]:
    ref_dir = TRAJECTORIES["reference_scale1p75_seed43022"]
    if not AVG_CANDIDATE.exists():
        return {"status": "missing", "path": rel(AVG_CANDIDATE)}
    avg = load_model_file(AVG_CANDIDATE)
    eps = ["chck_80M", "chck_82M", "chck_84M", "chck_86M", "chck_100M"]
    distances = {}
    states = {}
    for ep in eps:
        states[ep] = load_state(ref_dir, ep)
        distances[ep] = diff_partition(avg, states[ep])
    movement_from_84 = diff_state(avg, states["chck_84M"])
    d80_to_84 = diff_state(states["chck_84M"], states["chck_80M"])
    d82_to_84 = diff_state(states["chck_84M"], states["chck_82M"])
    d84_to_86 = diff_state(states["chck_86M"], states["chck_84M"])
    d84_to_100 = diff_state(states["chck_100M"], states["chck_84M"])
    projections = {
        "avg_minus_84_vs_80_to_84": {part: dot_norm(movement_from_84, d80_to_84, part) for part in ["total", "adapter", "backbone_plus_heads"]},
        "avg_minus_84_vs_82_to_84": {part: dot_norm(movement_from_84, d82_to_84, part) for part in ["total", "adapter", "backbone_plus_heads"]},
        "avg_minus_84_vs_84_to_86": {part: dot_norm(movement_from_84, d84_to_86, part) for part in ["total", "adapter", "backbone_plus_heads"]},
        "avg_minus_84_vs_84_to_100": {part: dot_norm(movement_from_84, d84_to_100, part) for part in ["total", "adapter", "backbone_plus_heads"]},
    }
    return {
        "status": "ok",
        "candidate_model": rel(AVG_CANDIDATE),
        "candidate_reading": "uniform average of reference chck_80M, chck_82M, and chck_84M; built in research and not yet task-evaluated",
        "distances_to_reference_endpoints": distances,
        "projection_cosines": projections,
        "scientific_use": "geometry only; official-compatible selected scoring is still required before any competence statement",
    }


def interpret(result: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    ref = result["trajectories"].get("reference_scale1p75_seed43022", {})
    refsum = ref.get("summary", {})
    turn84 = refsum.get("turn_around_84M", {})
    if turn84:
        notes.append(
            "Reference late path has a directly measured turn from 82->84 to 84->86; "
            f"total cosine={turn84['total'].get('cosine'):.6f}, adapter cosine={turn84['adapter'].get('cosine'):.6f}, "
            f"backbone+heads cosine={turn84['backbone_plus_heads'].get('cosine'):.6f}."
        )
    for label, block in result["trajectories"].items():
        s = block.get("summary", {})
        t = s.get("total", {})
        a = s.get("adapter", {})
        notes.append(
            f"{label}: mean 2M update L2 total={t.get('mean_step_l2'):.6f} with mean successive cosine={t.get('mean_successive_update_cosine'):.6f}; "
            f"adapter mean successive cosine={a.get('mean_successive_update_cosine'):.6f}."
        )
    avg = result.get("average_candidate", {})
    if avg.get("status") == "ok":
        pc = avg["projection_cosines"]["avg_minus_84_vs_84_to_86"]["total"].get("cosine")
        pc100 = avg["projection_cosines"]["avg_minus_84_vs_84_to_100"]["total"].get("cosine")
        notes.append(
            "The built 80/82/84 average moves from 84M in a direction whose cosine with 84->86 is "
            f"{pc:.6f} and with 84->100 is {pc100:.6f}; negative values mean the low-pass point backs away from later-training drift."
        )
    notes.append(
        "These measurements can make a later weight-average screen interpretable, but they do not replace selected BabyLM scoring and should not be used to choose a public endpoint."
    )
    return notes


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# research late-weight geometry probe\n\n")
    lines.append("CPU/file-only parameter geometry for same-trajectory late-iterate stabilization. No BabyLM task evaluation, model upload, or leaderboard submission was performed.\n\n")
    lines.append("## Main readout\n\n")
    for note in result.get("interpretation", []):
        lines.append(f"- {note}\n")
    lines.append("\n## Trajectory summary\n\n")
    lines.append("| trajectory | mean 2M update L2 | mean update cosine | min update cosine | adapter mean cosine | turn 82→84 vs 84→86 total cosine |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for label, block in result["trajectories"].items():
        s = block.get("summary", {})
        total = s.get("total", {})
        adapter = s.get("adapter", {})
        turn = s.get("turn_around_84M", {})
        turn_cos = turn.get("total", {}).get("cosine") if isinstance(turn, dict) else None
        def fmt(x: Any) -> str:
            return "" if x is None else f"{float(x):.6f}"
        lines.append(
            f"| {label} | {fmt(total.get('mean_step_l2'))} | {fmt(total.get('mean_successive_update_cosine'))} | {fmt(total.get('min_successive_update_cosine'))} | {fmt(adapter.get('mean_successive_update_cosine'))} | {fmt(turn_cos)} |\n"
        )
    lines.append("\n## Built average candidate geometry\n\n")
    avg = result.get("average_candidate", {})
    if avg.get("status") == "ok":
        lines.append(f"Candidate: `{avg['candidate_model']}`\n\n")
        lines.append("Distances from the average to selected reference endpoints (total L2):\n\n")
        for ep, d in avg["distances_to_reference_endpoints"].items():
            lines.append(f"- `{ep}`: total L2 `{d['total']['l2']:.6f}`, adapter L2 `{d['adapter']['l2']:.6f}`, backbone+heads L2 `{d['backbone_plus_heads']['l2']:.6f}`\n")
        lines.append("\nProjection cosines for avg - 84M:\n\n")
        for name, parts in avg["projection_cosines"].items():
            lines.append(f"- `{name}`: total `{parts['total']['cosine']:.6f}`, adapter `{parts['adapter']['cosine']:.6f}`, backbone+heads `{parts['backbone_plus_heads']['cosine']:.6f}`\n")
    else:
        lines.append(f"Average candidate unavailable: `{avg}`\n")
    lines.append("\n## Reference score rows used only for orientation\n\n")
    scores = result.get("reference_scores", {})
    if scores.get("status") == "ok":
        for ep, row in scores.get("selected_rows", {}).items():
            if row:
                lines.append(f"- `{ep}` cheap7 `{float(row['cheap7']):.6f}`\n")
    lines.append(f"\nJSON: `{rel(path.with_suffix('.json'))}`\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--trajectories", nargs="+", default=list(TRAJECTORIES.keys()), choices=sorted(TRAJECTORIES.keys()))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = {}
    for label in args.trajectories:
        trajectories[label] = trajectory_geometry(label, TRAJECTORIES[label], ENDPOINTS)
    result = {
        "status": "LATE_WEIGHT_GEOMETRY_PROBE_COMPLETE",
        "created_utc": utc_now(),
        "trajectories": trajectories,
        "average_candidate": average_candidate_geometry(),
        "reference_scores": reference_scores(),
        "official_evaluation_performed": False,
        "leaderboard_submission_performed": False,
        "scientific_reading": "Same-trajectory parameter geometry for interpreting a possible later low-pass stabilization screen; not a task score.",
    }
    result["interpretation"] = interpret(result)
    out_json = args.out_dir / "late_weight_geometry_probe.json"
    out_md = args.out_dir / "late_weight_geometry_probe.md"
    write_json(out_json, result)
    write_markdown(out_md, result)
    print(json.dumps({"status": result["status"], "out_json": rel(out_json), "out_md": rel(out_md)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
