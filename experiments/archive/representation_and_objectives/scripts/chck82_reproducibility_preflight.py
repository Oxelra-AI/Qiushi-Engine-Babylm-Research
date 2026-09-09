#!/usr/bin/env python3
"""research: preflight dossier for the scale1.75 chck_82M from-corpus reproduction.

This is CPU/file work only.  It does not launch training.  It records the exact
legal recipe and fixed selection rule needed if the independent hardened
measurement rerun confirms the research above-frontier score.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import hashlib
import json
import time
from pathlib import Path
from typing import Any


def find_user_root() -> Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/representation_and_objectives"
A02 = USER_ROOT / "experiments/archive/frontier_consolidation"
COMPACT_EXPERIENCE = USER_ROOT / "experiments/archive/compact_experience"

A02_RUN = A02 / "training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder"
ENDPOINT = "chck_82M"
ENDPOINT_DIR = A02_RUN / "hf_model" / ENDPOINT
ENDPOINT_MODEL = ENDPOINT_DIR / "model.safetensors"
ENDPOINT_CONFIG = ENDPOINT_DIR / "config.json"

TRAIN_SCRIPT = A02 / "scripts/adapter_scaled_trainer.py"
MODELING_SCRIPT = A02 / "scripts/adapter_scaled_modeling.py"
BASE_TRAINER = COMPACT_EXPERIENCE / "scripts/masking_curriculum_trainer.py"
DATA_ROOT = A02 / "data/density_cleanqwen_overlay_medium_riskhard"
DATA_10M = DATA_ROOT / "cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
DATA_100M = DATA_ROOT / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
DATA_META = DATA_ROOT / "density_cleanqwen_rowholdout_overlay_metadata.json"
TOKENIZER_DIR = A02 / "data/compliant_tokenizer"
TOKENIZER_META = TOKENIZER_DIR / "tokenizer_metadata.json"

SCORE = STUDY / "data/scale1p75_chck82_full_verification/summary/scale1p75_chck82_full_verification.json"
SWEEP = STUDY / "data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json"
SNAPSHOT = STUDY / "data/chck82_endpoint_snapshot/manifest.json"
REPRO_ROOT = STUDY / "data/scale1p75_chck82_full_eval_reproduction"
REPRO_SUMMARY = REPRO_ROOT / "summary/scale1p75_100M_full_eval_hardened_summary.json"

OUT_ROOT = STUDY / "data/chck82_reproducibility_preflight"
REPRO_RUN_DIR = STUDY / "training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022"
EXPECTED = {
    "pool_10m_sha256": "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23",
    "stream_100m_sha256": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    "chck82_model_sha256": "93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3",
    "chck80_model_sha256": "c37f6665df84109a266428e606e1bf665db70d9a5e7842816b5717d36922edda",
    "chck100_model_sha256": "7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52",
}


def rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_jsonl_words(path: Path) -> dict[str, Any]:
    rows = 0
    words = 0
    first_meta: dict[str, Any] | None = None
    last_meta: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            actual = len(text.split())
            field = int(obj.get("words", actual))
            if field != actual:
                raise RuntimeError({"path": rel(path), "row": rows, "field_words": field, "actual_words": actual})
            words += field
            meta = {k: v for k, v in obj.items() if k != "text"}
            if first_meta is None:
                first_meta = meta
            last_meta = meta
    return {"path": rel(path), "rows": rows, "words": words, "first_meta": first_meta, "last_meta": last_meta}


def tokenizer_vocab(path: Path) -> dict[str, int]:
    data = read_json(path)
    model = data.get("model", {})
    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError(f"No tokenizer model.vocab at {path}")
    return {str(k): int(v) for k, v in vocab.items()}


def find_checkpoint_record(metrics: dict[str, Any], name: str) -> dict[str, Any] | None:
    for rec in metrics.get("saved_checkpoints", []):
        if isinstance(rec, dict) and rec.get("name") == name:
            return rec
    return None


def training_command(gpu: int = 0) -> list[str]:
    return [
        "python", "-B", rel(TRAIN_SCRIPT),
        "--adapter_bottleneck", "128",
        "--adapter_enabled", "1",
        "--adapter_scale", "1.75",
        "--gpu", str(gpu),
        "--example_jsonl", rel(DATA_100M),
        "--example_jsonl_label", "repro_adapter128_scale1p75_matched_100Mhorizon_fixed82M",
        "--example_jsonl_meta", rel(DATA_META),
        "--output_dir", rel(REPRO_RUN_DIR),
        "--tokenizer_path", rel(TOKENIZER_DIR),
        "--tokenizer_label", "compliant16k_reinvest10M",
        "--hidden_size", "480",
        "--n_layer", "8",
        "--n_head", "8",
        "--ffn_mult", "4",
        "--seed", "43",
        "--extra_init_seed", "43022",
        "--train_rng_seed", "43023",
        "--batch_size", "256",
        "--seq_length", "256",
        "--max_seq_length", "256",
        "--learning_rate", "0.001",
        "--warmup_fraction", "0.06",
        "--weight_decay", "0.01",
        "--masking_curriculum", "wwm_fixed",
        "--mask_prob_start", "0.15",
        "--mask_prob_end", "0.15",
        "--checkpoint_words", "1000000",
        "--max_word_exposure", "100000000",
        "--lr_total_steps", "2529",
        "--num_workers", "0",
        "--log_every", "100",
        "--dynamics_trace_every", "500",
    ]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    required_paths = [
        A02_RUN / "scientific_metrics.json", A02_RUN / "example_order_manifest.json",
        ENDPOINT_MODEL, ENDPOINT_CONFIG, TRAIN_SCRIPT, MODELING_SCRIPT, BASE_TRAINER,
        DATA_10M, DATA_100M, DATA_META, TOKENIZER_DIR / "tokenizer.json", TOKENIZER_META,
        SCORE, SWEEP, SNAPSHOT,
    ]
    for p in required_paths:
        if not p.exists():
            errors.append(f"missing required path: {rel(p)}")

    metrics = read_json(A02_RUN / "scientific_metrics.json") if (A02_RUN / "scientific_metrics.json").exists() else {}
    order = read_json(A02_RUN / "example_order_manifest.json") if (A02_RUN / "example_order_manifest.json").exists() else {}
    token_meta = read_json(TOKENIZER_META) if TOKENIZER_META.exists() else {}
    score165 = read_json(SCORE) if SCORE.exists() else {}
    snapshot = read_json(SNAPSHOT) if SNAPSHOT.exists() else {}
    config = read_json(ENDPOINT_CONFIG) if ENDPOINT_CONFIG.exists() else {}

    # Heavy enough to be worth doing once: exact hashes and word counts of the legal pool/stream.
    data_counts = {}
    data_hashes = {}
    for label, path in [("pool_10m", DATA_10M), ("stream_100m", DATA_100M)]:
        if path.exists():
            data_counts[label] = count_jsonl_words(path)
            data_hashes[label] = sha256_file(path)

    endpoint_hashes = {}
    for label, path in [
        ("chck82_model", ENDPOINT_MODEL),
        ("chck80_model", A02_RUN / "hf_model/chck_80M/model.safetensors"),
        ("chck100_model", A02_RUN / "hf_model/chck_100M/model.safetensors"),
        ("chck82_config", ENDPOINT_CONFIG),
        ("training_tokenizer", TOKENIZER_DIR / "tokenizer.json"),
        ("endpoint_tokenizer", ENDPOINT_DIR / "tokenizer.json"),
        ("train_script", TRAIN_SCRIPT),
        ("modeling_script", MODELING_SCRIPT),
        ("base_trainer", BASE_TRAINER),
    ]:
        if path.exists():
            endpoint_hashes[label] = {"path": rel(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

    if data_hashes.get("pool_10m") != EXPECTED["pool_10m_sha256"]:
        errors.append(f"10M pool SHA mismatch: {data_hashes.get('pool_10m')} != {EXPECTED['pool_10m_sha256']}")
    if data_hashes.get("stream_100m") != EXPECTED["stream_100m_sha256"]:
        errors.append(f"100M stream SHA mismatch: {data_hashes.get('stream_100m')} != {EXPECTED['stream_100m_sha256']}")
    if endpoint_hashes.get("chck82_model", {}).get("sha256") != EXPECTED["chck82_model_sha256"]:
        errors.append("chck_82M model SHA mismatch")
    if endpoint_hashes.get("chck80_model", {}).get("sha256") != EXPECTED["chck80_model_sha256"]:
        errors.append("chck_80M model SHA mismatch")
    if endpoint_hashes.get("chck100_model", {}).get("sha256") != EXPECTED["chck100_model_sha256"]:
        errors.append("chck_100M model SHA mismatch")

    if data_counts.get("pool_10m", {}).get("words") != 10_000_000:
        errors.append(f"10M pool word count mismatch: {data_counts.get('pool_10m', {}).get('words')}")
    if data_counts.get("stream_100m", {}).get("words") != 100_000_000:
        errors.append(f"100M stream word count mismatch: {data_counts.get('stream_100m', {}).get('words')}")
    if data_counts.get("pool_10m", {}).get("rows") != 64_740:
        errors.append(f"10M pool row count mismatch: {data_counts.get('pool_10m', {}).get('rows')}")
    if data_counts.get("stream_100m", {}).get("rows") != 647_400:
        errors.append(f"100M stream row count mismatch: {data_counts.get('stream_100m', {}).get('rows')}")

    if token_meta.get("training_data", {}).get("pool_sha256") != data_hashes.get("pool_10m"):
        errors.append("tokenizer metadata pool SHA does not match computed 10M pool SHA")
    if int(token_meta.get("training_data", {}).get("pool_words", -1)) != 10_000_000:
        errors.append("tokenizer metadata pool_words is not 10M")
    if int(token_meta.get("tokenizer", {}).get("vocab_size", -1)) != 16_384:
        errors.append("tokenizer metadata vocab_size is not 16384")

    vocab_compare = {"checked": False}
    try:
        train_vocab = tokenizer_vocab(TOKENIZER_DIR / "tokenizer.json")
        end_vocab = tokenizer_vocab(ENDPOINT_DIR / "tokenizer.json")
        vocab_compare = {
            "checked": True,
            "training_vocab_size": len(train_vocab),
            "endpoint_vocab_size": len(end_vocab),
            "vocab_maps_identical": train_vocab == end_vocab,
        }
        if train_vocab != end_vocab:
            errors.append("endpoint tokenizer vocab map differs from training tokenizer vocab map")
    except Exception as exc:
        errors.append(f"tokenizer vocab compare failed: {exc!r}")

    saved = metrics.get("saved_checkpoints", []) if isinstance(metrics.get("saved_checkpoints"), list) else []
    chck82_record = find_checkpoint_record(metrics, ENDPOINT)
    if metrics.get("word_exposure") != 100_000_000:
        errors.append(f"metrics word_exposure mismatch: {metrics.get('word_exposure')}")
    if metrics.get("actual_training_steps") != 2529:
        errors.append(f"metrics actual_training_steps mismatch: {metrics.get('actual_training_steps')}")
    if len(saved) != 100:
        errors.append(f"saved checkpoint count mismatch: {len(saved)}")
    if not chck82_record:
        errors.append("missing chck_82M checkpoint record in scientific_metrics")
    elif chck82_record.get("target_word_exposure") != 82_000_000 or chck82_record.get("actual_cumulative_word_exposure") != 82_012_495:
        errors.append(f"unexpected chck_82M exposure record: {chck82_record}")

    source_sum = sum(int(v) for v in order.get("source_words_consumed", {}).values()) if isinstance(order.get("source_words_consumed"), dict) else None
    if source_sum != 100_000_000:
        errors.append(f"source_words_consumed sum mismatch: {source_sum}")
    if order.get("example_jsonl") != rel(DATA_100M):
        errors.append(f"example_order_manifest example_jsonl mismatch: {order.get('example_jsonl')}")

    # research's custom summary stores the final nine-column score under
    # `official_overall`, while the hardened research-style summaries store top-level
    # `scores`/`Overall`.  Accept both layouts so this dossier can compare the two.
    official165 = score165.get("official_overall") if isinstance(score165.get("official_overall"), dict) else {}
    score165_scores = score165.get("scores", {}) if isinstance(score165.get("scores"), dict) else {}
    if not score165_scores and isinstance(official165.get("scores"), dict):
        score165_scores = official165["scores"]
    score165_overall = score165.get("Overall", official165.get("Overall"))
    if not isinstance(score165_overall, (int, float)) or float(score165_overall) <= 41.8:
        errors.append(f"research score is not above 41.8: {score165_overall}")
    if abs(float(score165_scores.get("SuperGLUE", -999)) - 69.7602879248243) > 1e-9:
        errors.append("research SuperGLUE differs from recorded expected value")

    if snapshot.get("candidate_model_sha256") != EXPECTED["chck82_model_sha256"]:
        errors.append("research snapshot manifest candidate hash mismatch")

    if REPRO_RUN_DIR.exists():
        warnings.append(f"reproduction run dir already exists; do not overwrite without inspection: {rel(REPRO_RUN_DIR)}")

    repro_summary_status = "pending"
    repro_summary: dict[str, Any] | None = None
    if REPRO_SUMMARY.exists():
        repro_summary_status = "available"
        repro_summary = read_json(REPRO_SUMMARY)

    command_gpu0 = training_command(0)
    command_gpu1 = training_command(1)
    dossier = {
        "status": "PASS" if not errors else "FAIL",
        "created_utc": now(),
        "purpose": "CPU/file-only preflight for launching an independent from-corpus reproduction of the legal scale1.75 trajectory only after the independent hardened chck_82M evaluation rerun agrees with research.",
        "errors": errors,
        "warnings": warnings,
        "fixed_selection_rule": {
            "selected_checkpoint": ENDPOINT,
            "target_word_exposure": 82_000_000,
            "actual_original_cumulative_word_exposure": 82_012_495,
            "no_neighbor_search_in_reproduction": True,
            "post_training_interpretation": "If reproduced chck_82M is bit-identical to the original hash, research/research evaluation evidence transfers. If not bit-identical, run the hardened full official-compatible evaluator on reproduced chck_82M before any score statement.",
        },
        "reproduction_dependency": {
            "must_wait_for": "independent hardened full-evaluation reproduction",
            "reproduction_summary_path": rel(REPRO_SUMMARY),
            "reproduction_summary_status": repro_summary_status,
            "reproduction_overall_if_available": None if repro_summary is None else repro_summary.get("Overall"),
        },
        "original_endpoint": {
            "run_dir": rel(A02_RUN),
            "endpoint_dir": rel(ENDPOINT_DIR),
            "endpoint_model_sha256": endpoint_hashes.get("chck82_model", {}).get("sha256"),
            "checkpoint_record": chck82_record,
            "config_core": {
                "architectures": config.get("architectures"),
                "auto_map": config.get("auto_map"),
                "adapter_enabled": config.get("adapter_enabled"),
                "adapter_scale": config.get("adapter_scale"),
                "adapter_bottleneck": config.get("adapter_bottleneck"),
                "hidden_size": config.get("hidden_size"),
                "num_hidden_layers": config.get("num_hidden_layers"),
                "num_attention_heads": config.get("num_attention_heads"),
                "vocab_size": config.get("vocab_size"),
            },
        },
        "score": {
            "summary_json": rel(SCORE),
            "Overall": score165_overall,
            "scores": score165_scores,
            "cheap7": score165.get("cheap7"),
            "margin_vs_41p8": score165.get("overall_margin_vs_41p8") or score165.get("margin_vs_41p8"),
        },
        "legal_data": {
            "counts": data_counts,
            "hashes": data_hashes,
            "tokenizer_metadata": token_meta,
            "tokenizer_vocab_compare": vocab_compare,
            "overlay_metadata_core": {
                "status": read_json(DATA_META).get("status") if DATA_META.exists() else None,
                "total_words_per_pool": read_json(DATA_META).get("total_words_per_pool") if DATA_META.exists() else None,
                "passes": read_json(DATA_META).get("passes") if DATA_META.exists() else None,
                "compact_reinvest_family": (read_json(DATA_META).get("families", {}) if DATA_META.exists() else {}).get("compact_reinvest"),
            },
        },
        "training_recipe": {
            "train_script": rel(TRAIN_SCRIPT),
            "modeling_script": rel(MODELING_SCRIPT),
            "base_trainer": rel(BASE_TRAINER),
            "reproduction_run_dir": rel(REPRO_RUN_DIR),
            "parameters": {
                "adapter_bottleneck": 128,
                "adapter_enabled": 1,
                "adapter_scale": 1.75,
                "example_jsonl": rel(DATA_100M),
                "example_jsonl_meta": rel(DATA_META),
                "tokenizer_path": rel(TOKENIZER_DIR),
                "hidden_size": 480,
                "n_layer": 8,
                "n_head": 8,
                "ffn_mult": 4,
                "seed": 43,
                "extra_init_seed": 43022,
                "train_rng_seed": 43023,
                "batch_size": 256,
                "seq_length": 256,
                "max_seq_length": 256,
                "learning_rate": 0.001,
                "warmup_fraction": 0.06,
                "weight_decay": 0.01,
                "masking_curriculum": "wwm_fixed",
                "mask_prob_start": 0.15,
                "mask_prob_end": 0.15,
                "checkpoint_words": 1_000_000,
                "max_word_exposure": 100_000_000,
                "lr_total_steps": 2529,
                "num_workers": 0,
                "log_every": 100,
                "dynamics_trace_every": 500,
            },
            "command_gpu0": command_gpu0,
            "command_gpu1": command_gpu1,
        },
        "hashes": endpoint_hashes,
        "scientific_boundary": "This reproduction can establish measurement and training reproducibility of the above-frontier endpoint. It does not by itself establish the requested generalizable data-efficient learning principle; A02's dual-view pathway remains the mechanism route to test.",
    }
    out_json = OUT_ROOT / "chck82_reproducibility_preflight.json"
    out_md = OUT_ROOT / "chck82_reproducibility_preflight.md"
    write_json(out_json, dossier)
    lines = [
        "# research chck_82M reproducibility preflight",
        "",
        f"Status: **{dossier['status']}**",
        f"Original chck_82M model SHA256: `{dossier['original_endpoint']['endpoint_model_sha256']}`",
        f"research Overall: **{score165_overall}**; SuperGLUE: **{score165_scores.get('SuperGLUE')}**; cheap7: **{score165.get('cheap7')}**",
        f"10M pool SHA: `{data_hashes.get('pool_10m')}`; words/rows: {data_counts.get('pool_10m', {}).get('words')}/{data_counts.get('pool_10m', {}).get('rows')}",
        f"100M stream SHA: `{data_hashes.get('stream_100m')}`; words/rows: {data_counts.get('stream_100m', {}).get('words')}/{data_counts.get('stream_100m', {}).get('rows')}",
        f"Tokenizer vocab maps identical: **{vocab_compare.get('vocab_maps_identical')}**",
        f"Fixed selection rule: evaluate `{ENDPOINT}` only; no reproduction checkpoint search.",
        f"Reproduction run dir: `{rel(REPRO_RUN_DIR)}`",
        "",
        "Training command (GPU0):",
        "```bash",
        " \\\n  ".join(command_gpu0),
        "```",
        "",
        f"JSON: `{rel(out_json)}`",
    ]
    if errors:
        lines += ["", "Errors:"] + [f"- {e}" for e in errors]
    if warnings:
        lines += ["", "Warnings:"] + [f"- {w}" for w in warnings]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": dossier["status"], "json": rel(out_json), "md": rel(out_md), "errors": errors, "warnings": warnings, "repro_status": repro_summary_status}, indent=2), flush=True)


if __name__ == "__main__":
    main()
