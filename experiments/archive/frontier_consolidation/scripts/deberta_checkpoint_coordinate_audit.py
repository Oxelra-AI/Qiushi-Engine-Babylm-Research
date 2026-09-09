#!/usr/bin/env python3
"""research: CPU/file-only audit of DeBERTa common-grid checkpoint coordinates.

Checks that the three trajectories to be compared really share the legal tokenizer,
corpus manifest, adapter custom-code path, DeBERTa geometry, and expected common-grid
endpoints. This protects against known checkpoint-comparison failure modes: non-compliant
or mismatched tokenizers, native DeBERTa fallback, missing endpoints, or silently
changed data streams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

EXPECTED_ENDPOINTS = [
    "chck_70M", "chck_72M", "chck_74M", "chck_76M", "chck_78M", "chck_80M",
    "chck_82M", "chck_84M", "chck_86M", "chck_88M", "chck_90M", "chck_92M",
    "chck_94M", "chck_96M", "chck_98M", "chck_100M",
]
HASH_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "adapter_scaled_modeling.py"]
CONFIG_KEYS = [
    "architectures", "auto_map", "model_type", "vocab_size", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "intermediate_size", "max_position_embeddings", "relative_attention",
    "position_buckets", "max_relative_positions", "adapter_bottleneck", "adapter_scale",
    "adapter_enabled", "adapter_activation",
]
METRIC_KEYS = [
    "backend", "model_family", "parameter_count", "vocab_size", "tokenizer_label", "word_exposure",
    "loss_first", "loss_last", "actual_training_steps", "masking_curriculum", "mask_prob_start",
    "mask_prob_end", "seed", "seq_length", "max_seq_length", "hidden_size", "n_layer", "n_head",
]


def sha256(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_labeled_run(spec: str) -> tuple[str, pathlib.Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("Use LABEL=RUN_DIR")
    label, path = spec.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    return label, pathlib.Path(path)


def checkpoint_weight_info(ckpt: pathlib.Path) -> dict[str, Any]:
    st = ckpt / "model.safetensors"
    pt = ckpt / "pytorch_model.bin"
    if st.exists():
        return {"file": "model.safetensors", "size": st.stat().st_size, "sha256": sha256(st)}
    if pt.exists():
        return {"file": "pytorch_model.bin", "size": pt.stat().st_size, "sha256": sha256(pt)}
    return {"file": None, "size": None, "sha256": None}


def audit_one(label: str, run_dir: pathlib.Path, expected_scale: float | None, expected_endpoints: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "run_dir": str(run_dir), "exists": run_dir.exists()}
    issues: list[str] = []
    if not run_dir.exists():
        out["issues"] = ["missing_run_dir"]
        out["ok"] = False
        return out
    hf = run_dir / "hf_model"
    out["hf_model_exists"] = hf.exists()
    if not hf.exists():
        issues.append("missing_hf_model")
        out["issues"] = issues
        out["ok"] = False
        return out
    ckpts = sorted([p.name for p in hf.iterdir() if p.is_dir() and p.name.startswith("chck_")])
    out["n_checkpoint_dirs"] = len(ckpts)
    out["checkpoint_dirs_head"] = ckpts[:5]
    out["checkpoint_dirs_tail"] = ckpts[-5:]
    missing = [e for e in expected_endpoints if e not in ckpts]
    out["missing_common_grid_endpoints"] = missing
    if missing:
        issues.append(f"missing_common_grid:{missing}")

    # Use chck_82M if present because it is the protected reference peak; otherwise first expected endpoint.
    ref_ep = "chck_82M" if (hf / "chck_82M").exists() else (expected_endpoints[0] if expected_endpoints else (ckpts[0] if ckpts else ""))
    ref_dir = hf / ref_ep
    out["reference_endpoint_for_files"] = ref_ep
    if not ref_dir.exists():
        issues.append(f"missing_reference_endpoint_for_files:{ref_ep}")
    else:
        config_path = ref_dir / "config.json"
        config = read_json(config_path) if config_path.exists() else {}
        if not config:
            issues.append(f"missing_or_empty_config:{ref_ep}")
        out["config_subset"] = {k: config.get(k) for k in CONFIG_KEYS}
        arch = config.get("architectures")
        auto = config.get("auto_map")
        if arch != ["AdapterDebertaV2ForMaskedLM"]:
            issues.append(f"unexpected_architectures:{arch}")
        if not isinstance(auto, dict) or "AdapterDebertaV2ForMaskedLM" not in str(auto.get("AutoModelForMaskedLM")):
            issues.append(f"missing_adapter_auto_map:{auto}")
        if expected_scale is not None and config.get("adapter_scale") is not None and abs(float(config.get("adapter_scale")) - expected_scale) > 1e-12:
            issues.append(f"adapter_scale_mismatch:{config.get('adapter_scale')}:{expected_scale}")
        if config.get("vocab_size") != 16384:
            issues.append(f"vocab_size_mismatch:{config.get('vocab_size')}")
        if config.get("hidden_size") != 480 or config.get("num_hidden_layers") != 8 or config.get("num_attention_heads") != 8:
            issues.append("geometry_mismatch")
        out["file_hashes_at_reference_endpoint"] = {fn: sha256(ref_dir / fn) for fn in HASH_FILES + ["config.json"]}
        out["weight_at_reference_endpoint"] = checkpoint_weight_info(ref_dir)

    # Check hashes are stable across the common grid for tokenizer/code/config excluding weights.
    stability: dict[str, Any] = {}
    for fn in HASH_FILES:
        vals = {}
        for ep in expected_endpoints:
            p = hf / ep / fn
            vals[ep] = sha256(p)
        uniq = sorted(set(v for v in vals.values() if v is not None))
        stability[fn] = {"n_present": sum(v is not None for v in vals.values()), "n_unique_sha256": len(uniq), "sha256": uniq, "missing": [ep for ep, v in vals.items() if v is None]}
        if len(uniq) != 1 or stability[fn]["missing"]:
            issues.append(f"unstable_or_missing_{fn}")
    out["common_grid_file_hash_stability"] = stability

    metrics_path = run_dir / "scientific_metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}
    out["scientific_metrics_subset"] = {k: metrics.get(k) for k in METRIC_KEYS} if isinstance(metrics, dict) else {}
    if isinstance(metrics, dict):
        if metrics.get("word_exposure") != 100_000_000:
            issues.append(f"word_exposure_mismatch:{metrics.get('word_exposure')}")
        if metrics.get("parameter_count") != 35463008:
            issues.append(f"parameter_count_mismatch:{metrics.get('parameter_count')}")
        if metrics.get("tokenizer_label") != "compliant16k_reinvest10M":
            issues.append(f"tokenizer_label_mismatch:{metrics.get('tokenizer_label')}")
        saved = metrics.get("saved_checkpoints") if isinstance(metrics.get("saved_checkpoints"), list) else []
        by_name = {r.get("name"): r for r in saved if isinstance(r, dict)}
        out["scientific_metrics_saved_checkpoints"] = {
            "n": len(saved),
            "common_grid_actual_exposure": {ep: by_name.get(ep, {}).get("actual_cumulative_word_exposure") for ep in expected_endpoints},
            "common_grid_target_exposure": {ep: by_name.get(ep, {}).get("target_word_exposure") for ep in expected_endpoints},
        }
        missing_saved = [ep for ep in expected_endpoints if ep not in by_name]
        if missing_saved:
            issues.append(f"missing_saved_checkpoint_records:{missing_saved}")
    else:
        issues.append("missing_scientific_metrics")

    manifest_path = run_dir / "example_order_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    out["example_order_manifest"] = manifest if isinstance(manifest, dict) else {}
    if isinstance(manifest, dict):
        if manifest.get("selected_for_training_words") != 100_000_000:
            issues.append(f"manifest_selected_words_mismatch:{manifest.get('selected_for_training_words')}")
        if manifest.get("example_jsonl") != "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl":
            issues.append(f"manifest_example_jsonl_mismatch:{manifest.get('example_jsonl')}")
        src_words = manifest.get("source_words_consumed") if isinstance(manifest.get("source_words_consumed"), dict) else {}
        out["source_words_total"] = sum(int(v) for v in src_words.values()) if src_words else None
        if out.get("source_words_total") != 100_000_000:
            issues.append(f"source_words_total_mismatch:{out.get('source_words_total')}")
    else:
        issues.append("missing_example_order_manifest")

    out["issues"] = issues
    out["ok"] = not issues
    return out


def compare_shared_hashes(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    comp: dict[str, Any] = {}
    for fn in HASH_FILES:
        vals = {b["label"]: (b.get("file_hashes_at_reference_endpoint", {}) or {}).get(fn) for b in blocks}
        uniq = sorted(set(v for v in vals.values() if v is not None))
        comp[fn] = {"values": vals, "n_unique": len(uniq), "shared": len(uniq) == 1, "sha256": uniq}
    # Config should differ by adapter_scale between 1.25 and 1.75, but other keys should match.
    config_subsets = {b["label"]: b.get("config_subset", {}) for b in blocks}
    key_cmp: dict[str, Any] = {}
    for key in CONFIG_KEYS:
        vals = {label: cfg.get(key) for label, cfg in config_subsets.items()}
        key_cmp[key] = {"values": vals, "n_unique_repr": len(set(json.dumps(v, sort_keys=True) for v in vals.values()))}
    comp["config_key_comparison"] = key_cmp
    return comp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", type=parse_labeled_run, required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--expected-scale", action="append", default=[], help="LABEL=SCALE")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--expected-endpoints", nargs="+", default=EXPECTED_ENDPOINTS)
    args = ap.parse_args()
    scale_map: dict[str, float] = {}
    for spec in args.expected_scale:
        if "=" not in spec:
            raise SystemExit("Use --expected-scale LABEL=SCALE")
        label, val = spec.split("=", 1)
        scale_map[label] = float(val)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blocks = [audit_one(label, path, scale_map.get(label), args.expected_endpoints) for label, path in args.run]
    result = {
        "status": "DEBERTA_CHECKPOINT_COORDINATE_AUDIT",
        "ok": all(b.get("ok") for b in blocks),
        "expected_endpoints": args.expected_endpoints,
        "runs": blocks,
        "shared_coordinate_comparison": compare_shared_hashes(blocks),
        "scientific_use": "Run before interpreting selected scores. It establishes whether the compared trajectories share tokenizer/custom-code/data/geometry and whether the intended adapter-scale contrast is real.",
    }
    out_json = args.out_dir / "deberta_checkpoint_coordinate_audit.json"
    out_md = args.out_dir / "deberta_checkpoint_coordinate_audit.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines: list[str] = ["# research DeBERTa checkpoint coordinate audit\n\n", f"Overall OK: **{result['ok']}**\n\n"]
    lines.append("| label | ok | ckpt dirs | missing common endpoints | adapter scale | params | tokenizer label | 100M exposure | source words total |\n")
    lines.append("|---|---:|---:|---|---:|---:|---|---:|---:|\n")
    for b in blocks:
        cfg = b.get("config_subset", {})
        met = b.get("scientific_metrics_subset", {})
        lines.append(
            f"| {b.get('label')} | {b.get('ok')} | {b.get('n_checkpoint_dirs','')} | {', '.join(b.get('missing_common_grid_endpoints', []))} | "
            f"{cfg.get('adapter_scale','')} | {met.get('parameter_count','')} | {met.get('tokenizer_label','')} | {met.get('word_exposure','')} | {b.get('source_words_total','')} |\n"
        )
    lines.append("\n## Shared file hashes at chck_82M\n\n")
    lines.append("| file | shared | n unique | sha256 |\n|---|---:|---:|---|\n")
    for fn, rec in result["shared_coordinate_comparison"].items():
        if fn == "config_key_comparison":
            continue
        lines.append(f"| {fn} | {rec.get('shared')} | {rec.get('n_unique')} | {', '.join((rec.get('sha256') or []))} |\n")
    lines.append("\n## Issues\n\n")
    for b in blocks:
        lines.append(f"### {b.get('label')}\n")
        if b.get("issues"):
            for issue in b["issues"]:
                lines.append(f"- {issue}\n")
        else:
            lines.append("- none\n")
    lines.append("\nThe config hash may differ across scale settings, but tokenizer and custom adapter modeling code should be shared. Trusted-code loading remains required for these checkpoints.\n\n")
    lines.append(f"JSON: `{out_json}`\n")
    out_md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "ok": result["ok"], "out_json": str(out_json), "out_md": str(out_md)}, indent=2), flush=True)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
