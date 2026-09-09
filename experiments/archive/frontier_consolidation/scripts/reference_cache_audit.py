#!/usr/bin/env python3
"""research: audit existing score payloads for the scale1.75 seed43022 reference grid.

This is CPU/file-only. It does not declare cached files equivalent unless the payload
is complete for cheap7 and the source model path/hash relation is clear. The output
is a launch aid: known reference points can reduce duplicate scoring, but exact
common-grid comparison should keep provenance visible.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from statistics import mean
from typing import Any

CHEAP_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]
ZERO_COLUMNS = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS"]
GP_COLS = ["GlobalPIQA_parallel", "GlobalPIQA_nonparallel"]
ENDPOINTS = [f"chck_{m}M" for m in range(70, 101, 2)]
REFERENCE_RUN = pathlib.Path("experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder")
DATA_ROOT = pathlib.Path("experiments/archive/frontier_consolidation/data")
OUT_DIR = DATA_ROOT / "reference_cache_audit"


def sha256_file(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_scores(payload: dict[str, Any]) -> tuple[dict[str, float | None], float | None]:
    tasks = payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {}
    out: dict[str, float | None] = {}
    for c in ZERO_COLUMNS:
        rec = tasks.get(c, {})
        out[c] = float(rec["score"]) if isinstance(rec, dict) and rec.get("score") is not None else None
    gp_vals: list[float] = []
    for c in GP_COLS:
        rec = tasks.get(c, {})
        if isinstance(rec, dict) and rec.get("score") is not None:
            gp_vals.append(float(rec["score"]))
    out["GlobalPIQA"] = float(mean(gp_vals)) if len(gp_vals) == 2 else None
    rd = tasks.get("Reading", {})
    if isinstance(rd, dict) and isinstance(rd.get("scores"), dict) and rd["scores"].get("Reading") is not None:
        out["Reading"] = float(rd["scores"]["Reading"])
    elif isinstance(rd, dict) and rd.get("score") is not None:
        out["Reading"] = float(rd["score"])
    else:
        out["Reading"] = None
    vals = [out.get(c) for c in CHEAP_COLUMNS]
    cheap7 = float(mean(float(v) for v in vals)) if all(v is not None for v in vals) else None
    return out, cheap7


def endpoint_from_payload(path: pathlib.Path, payload: dict[str, Any]) -> str | None:
    ep = payload.get("endpoint")
    if isinstance(ep, str) and re.fullmatch(r"chck_\d+M", ep):
        return ep
    name = path.name
    m = re.search(r"chck[_-]?(\d+)M", name)
    if m:
        return f"chck_{m.group(1)}M"
    return None


def candidate_payloads() -> list[pathlib.Path]:
    paths = []
    for p in DATA_ROOT.glob("**/per_target/*.json"):
        s = str(p)
        if "scale1p75" in s or "adapter128_scale1p75" in s or "scale1p75" in s:
            paths.append(p)
    return sorted(paths)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_hashes = {ep: sha256_file(REFERENCE_RUN / "hf_model" / ep / "model.safetensors") for ep in ENDPOINTS}
    by_endpoint: dict[str, list[dict[str, Any]]] = {ep: [] for ep in ENDPOINTS}
    for p in candidate_payloads():
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ep = endpoint_from_payload(p, payload)
        if ep not in by_endpoint:
            continue
        scores, cheap7 = extract_scores(payload)
        if cheap7 is None:
            continue
        model_path = pathlib.Path(str(payload.get("model_path", "")))
        model_hash = sha256_file(model_path / "model.safetensors") if model_path else None
        by_endpoint[ep].append({
            "payload": str(p),
            "target": payload.get("target"),
            "run_dir": payload.get("run_dir"),
            "model_path": str(model_path) if model_path else None,
            "payload_endpoint": payload.get("endpoint"),
            "cheap7": cheap7,
            "scores": scores,
            "model_sha256": model_hash,
            "reference_sha256": ref_hashes.get(ep),
            "hash_matches_reference_ladder": (model_hash is not None and ref_hashes.get(ep) is not None and model_hash == ref_hashes.get(ep)),
        })
    chosen: dict[str, Any] = {}
    for ep, recs in by_endpoint.items():
        exact = [r for r in recs if r.get("hash_matches_reference_ladder")]
        if exact:
            chosen[ep] = {"status": "usable_exact_hash", **exact[0]}
        elif recs:
            chosen[ep] = {"status": "prior_payload_hash_unverified_or_mismatch", "candidates": recs}
        else:
            chosen[ep] = {"status": "no_prior_complete_payload"}
    trajectory_seed = []
    for ep, rec in chosen.items():
        if rec.get("status") == "usable_exact_hash":
            words = int(ep[len("chck_"):-1]) * 1_000_000
            trajectory_seed.append({"words": words, "endpoint": ep, **rec["scores"], "cheap7": rec["cheap7"], "source_payload": rec["payload"], "source_model_sha256": rec["model_sha256"]})
    trajectory_seed = sorted(trajectory_seed, key=lambda r: r["words"])
    summary = {
        "status": "REFERENCE_CACHE_AUDIT",
        "reference_run": str(REFERENCE_RUN),
        "endpoints": ENDPOINTS,
        "usable_exact_hash_count": sum(1 for r in chosen.values() if r.get("status") == "usable_exact_hash"),
        "needs_gpu_or_manual_exact_eval": [ep for ep, r in chosen.items() if r.get("status") != "usable_exact_hash"],
        "chosen": chosen,
        "seed_trajectory_json": str(OUT_DIR / "reference_seed_trajectory_from_prior_exact_payloads.json"),
    }
    (OUT_DIR / "reference_cache_audit.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT_DIR / "reference_seed_trajectory_from_prior_exact_payloads.json").write_text(json.dumps(trajectory_seed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# research reference cache audit\n\n"]
    md.append(f"Usable exact-hash prior payloads: {summary['usable_exact_hash_count']} / {len(ENDPOINTS)}.\n\n")
    md.append("| endpoint | status | cheap7 | payload |\n")
    md.append("|---|---|---:|---|\n")
    for ep in ENDPOINTS:
        rec = chosen[ep]
        if rec.get("status") == "usable_exact_hash":
            md.append(f"| {ep} | usable exact hash | {rec['cheap7']:.6f} | `{rec['payload']}` |\n")
        else:
            md.append(f"| {ep} | {rec.get('status')} | — | — |\n")
    md.append("\nUse this only to reduce duplicate reference scoring when provenance is acceptable; keep final trajectory provenance explicit.\n\n")
    md.append(f"JSON: `{OUT_DIR / 'reference_cache_audit.json'}`\n")
    ((OUT_DIR.parents[4] / 'research/documents/frontier_consolidation/data/reference_cache_audit/reference_cache_audit.md')).write_text("".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "usable_exact_hash_count": summary["usable_exact_hash_count"], "needs_gpu_or_manual_exact_eval": summary["needs_gpu_or_manual_exact_eval"], "out_json": str(OUT_DIR / "reference_cache_audit.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
