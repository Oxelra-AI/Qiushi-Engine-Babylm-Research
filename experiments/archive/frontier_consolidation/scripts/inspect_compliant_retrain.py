#!/usr/bin/env python3
"""research/47: inspect compliant-tokenizer retrain artifacts.

Read-only helper for delivered compliant retrains. It verifies whether a run
contains a complete 100M endpoint under the expected legal tokenizer and the
frozen reinvest corpus provenance before any model evaluation is launched.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import pathlib
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive/frontier_consolidation"
WORKSPACE = STUDY
DATA = WORKSPACE / "data"
TOKENIZER = DATA / "compliant_tokenizer"
TOKENIZER_JSON = TOKENIZER / "tokenizer.json"
EXPECTED_TOKENIZER_SHA256 = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_AOA_STEPS = [f"chck_{i}M" for i in range(1, 10)] + [f"chck_{10*i}M" for i in range(1, 11)]
DEFAULT_RUNS = {
    "reinvest": STUDY / "training/runs/complianttok_reinvest_seed43022_r2",
    "clean_qwen": STUDY / "training/runs/complianttok_cleanqwen_seed43022_r2",
}
OUT = DATA / "compliant_retrain_inspection"

FROZEN_REINVEST_POOL_10M = DATA / "density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl"
FROZEN_REINVEST_TRAIN_100M = DATA / "density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
FROZEN_REINVEST_META = DATA / "density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json"
FROZEN_REINVEST_POOL_SHA256 = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
FROZEN_REINVEST_TRAIN_SHA256 = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
FROZEN_REINVEST_META_SHA256 = "92aa4c00b09d201a09da27e7e16677643bc7cda321471e1ec46332600339b3e4"

FROZEN_COMMAND_FLAGS = {
    "--hidden_size": "480",
    "--n_layer": "8",
    "--n_head": "8",
    "--ffn_mult": "4",
    "--seed": "43",
    "--extra_init_seed": "43022",
    "--train_rng_seed": "43023",
    "--batch_size": "256",
    "--seq_length": "256",
    "--max_seq_length": "256",
    "--learning_rate": "0.001",
    "--warmup_fraction": "0.06",
    "--weight_decay": "0.01",
    "--masking_curriculum": "wwm_fixed",
    "--mask_prob_start": "0.15",
    "--mask_prob_end": "0.15",
    "--checkpoint_words": "1000000",
    "--max_word_exposure": "100000000",
    "--num_workers": "0",
}


def user_path(value: str | pathlib.Path | None) -> pathlib.Path | None:
    if value in (None, ""):
        return None
    p = pathlib.Path(value)
    return p if p.is_absolute() else USER_ROOT / p


def rel(path: pathlib.Path | str | None) -> str | None:
    if path in (None, ""):
        return None
    p = pathlib.Path(path)
    try:
        return str(p.resolve().relative_to(USER_ROOT))
    except Exception:
        return str(p)


def same_path(actual: str | pathlib.Path | None, expected: str | pathlib.Path | None) -> bool:
    a = user_path(actual)
    e = user_path(expected)
    if a is None or e is None:
        return False
    try:
        return a.resolve(strict=False) == e.resolve(strict=False)
    except Exception:
        return str(a) == str(e)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tokenizer_canonical_fingerprint(path: pathlib.Path | None) -> str | None:
    """Fingerprint tokenizer semantics while ignoring save_pretrained JSON reserialization drift."""
    if path is None or not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    model = obj.get("model", {}) if isinstance(obj.get("model"), dict) else {}
    keep = {
        "model": {k: model.get(k) for k in sorted(model.keys())},
        "normalizer": obj.get("normalizer"),
        "pre_tokenizer": obj.get("pre_tokenizer"),
        "post_processor": obj.get("post_processor"),
        "decoder": obj.get("decoder"),
        "added_tokens": obj.get("added_tokens"),
    }
    payload = json.dumps(keep, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_jsonl_words(path: pathlib.Path) -> dict[str, Any]:
    rows = 0
    words = 0
    empty_lines = 0
    bad_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                empty_lines += 1
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_rows += 1
                continue
            w = obj.get("words")
            try:
                w_int = int(w)
            except Exception:
                w_int = 0
            if w_int <= 0:
                w_int = len(str(obj.get("text", "")).split())
            words += w_int
    return {"rows": rows, "words": words, "empty_lines": empty_lines, "bad_rows": bad_rows}


def safe_json(path: pathlib.Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc), "path": str(path)}


def value_after(cmd: list[Any], flag: str) -> str | None:
    try:
        i = cmd.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(cmd):
        return None
    return str(cmd[i + 1])


def command_value(command: Any, key: str, flag: str | None = None) -> str | None:
    if isinstance(command, dict):
        val = command.get(key)
        if val not in (None, ""):
            return str(val)
        cmd = command.get("command")
        if flag and isinstance(cmd, list):
            return value_after(cmd, flag)
    return None


def nested_pool_words(tok_meta: Any) -> int | None:
    if not isinstance(tok_meta, dict):
        return None
    td = tok_meta.get("training_data", {})
    for key in ("pool_words", "words"):
        if key in td:
            try:
                return int(td[key])
            except Exception:
                return None
    return None


def nested_pool_rows(tok_meta: Any) -> int | None:
    if not isinstance(tok_meta, dict):
        return None
    td = tok_meta.get("training_data", {})
    for key in ("pool_rows", "rows"):
        if key in td:
            try:
                return int(td[key])
            except Exception:
                return None
    return None


def command_flags_match(command: Any) -> dict[str, bool]:
    cmd = command.get("command") if isinstance(command, dict) else None
    if not isinstance(cmd, list):
        return {flag: False for flag in FROZEN_COMMAND_FLAGS}
    return {flag: value_after(cmd, flag) == expected for flag, expected in FROZEN_COMMAND_FLAGS.items()}


def metadata_provenance(meta: Any, expected_train: pathlib.Path | None, expected_pool: pathlib.Path | None, expected_train_sha: str, expected_pool_sha: str, expected_meta_words: int, expected_train_words: int, expected_pool_rows: int, expected_train_rows: int, expected_passes: int) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    if not isinstance(meta, dict):
        rec["metadata_readable"] = False
        return rec
    rec["metadata_readable"] = True
    rec["metadata_total_words_per_pool"] = meta.get("total_words_per_pool")
    rec["metadata_passes"] = meta.get("passes")
    rec["metadata_audit_all_exact_10M"] = bool(meta.get("audit", {}).get("all_exact_10M"))
    rec["metadata_declares_pool_words_expected"] = (expected_meta_words <= 0) or (meta.get("total_words_per_pool") == expected_meta_words)
    rec["metadata_declares_passes_expected"] = (expected_passes <= 0) or (meta.get("passes") == expected_passes)
    rec["metadata_declares_all_exact_10m"] = bool(meta.get("audit", {}).get("all_exact_10M"))

    sha_map = meta.get("sha256", {}) if isinstance(meta.get("sha256"), dict) else {}
    rec["metadata_pool_sha_matches_expected"] = sha_map.get(expected_pool.name) == expected_pool_sha if expected_pool else False
    rec["metadata_train_sha_matches_expected"] = sha_map.get(expected_train.name) == expected_train_sha if expected_train else False

    files = meta.get("files", {}) if isinstance(meta.get("files"), dict) else {}
    pools = files.get("pools", {}) if isinstance(files.get("pools"), dict) else {}
    training = files.get("training", {}) if isinstance(files.get("training"), dict) else {}
    rec["metadata_pool_path_matches_expected"] = same_path(pools.get("cleanqwen_fineweb_compact_view_reinvest"), expected_pool) if expected_pool else False
    rec["metadata_train_path_matches_expected"] = same_path(training.get("cleanqwen_fineweb_compact_view_reinvest"), expected_train) if expected_train else False

    fam = meta.get("families", {}).get("compact_reinvest", {}) if isinstance(meta.get("families"), dict) else {}
    word_totals = fam.get("word_totals", {}) if isinstance(fam.get("word_totals"), dict) else {}
    row_counts = fam.get("row_counts", {}) if isinstance(fam.get("row_counts"), dict) else {}
    rec["metadata_family_pool_words_exact"] = (expected_meta_words <= 0) or (word_totals.get("cleanqwen_fineweb_compact_view_reinvest") == expected_meta_words)
    rec["metadata_family_pool_rows_expected"] = (expected_pool_rows <= 0) or (row_counts.get("cleanqwen_fineweb_compact_view_reinvest") == expected_pool_rows)
    rec["metadata_expected_train_rows_by_passes"] = (expected_train_rows <= 0 or expected_pool_rows <= 0 or expected_passes <= 0) or (expected_train_rows == expected_pool_rows * expected_passes)
    rec["metadata_expected_train_words_by_passes"] = (expected_train_words <= 0 or expected_meta_words <= 0 or expected_passes <= 0) or (expected_train_words == expected_meta_words * expected_passes)
    return rec


def inspect_run(
    name: str,
    run_dir: pathlib.Path,
    *,
    expected_tokenizer: pathlib.Path | None,
    expected_tokenizer_sha256: str,
    expected_train_file: pathlib.Path | None,
    expected_train_sha256: str,
    expected_pool_10m: pathlib.Path | None,
    expected_pool_sha256: str,
    expected_meta: pathlib.Path | None,
    expected_meta_sha256: str,
    expected_pool_words: int,
    expected_train_words: int,
    expected_pool_rows: int,
    expected_train_rows: int,
    expected_passes: int,
    verify_corpus_content: bool,
) -> dict[str, Any]:
    hf = run_dir / "hf_model"
    metrics_path = run_dir / "scientific_metrics.json"
    launcher_path = run_dir / "launcher_result.json"
    command_path = run_dir / "train_command.json"
    metrics = safe_json(metrics_path)
    launcher = safe_json(launcher_path)
    command = safe_json(command_path)
    available_ckpts = sorted([p.name for p in hf.iterdir() if p.is_dir() and p.name.startswith("chck_")]) if hf.exists() else []
    missing_aoa = [s for s in EXPECTED_AOA_STEPS if s not in set(available_ckpts)]
    chck100 = hf / "chck_100M"
    tok_json = chck100 / "tokenizer.json"
    root_tok_json = hf / "tokenizer.json"
    expected_tok_json = expected_tokenizer / "tokenizer.json" if expected_tokenizer is not None else None
    tok_sha = sha256_file(tok_json) if tok_json.exists() else None
    root_tok_sha = sha256_file(root_tok_json) if root_tok_json.exists() else None
    expected_tok_sha = sha256_file(expected_tok_json) if expected_tok_json is not None and expected_tok_json.exists() else None
    tok_fp = tokenizer_canonical_fingerprint(tok_json)
    root_tok_fp = tokenizer_canonical_fingerprint(root_tok_json)
    expected_tok_fp = tokenizer_canonical_fingerprint(expected_tok_json)

    command_list = command.get("command") if isinstance(command, dict) else None
    if not isinstance(command_list, list):
        command_list = []
    command_tok_meta = command.get("tokenizer_metadata") if isinstance(command, dict) else None
    command_word_info = command.get("word_info") if isinstance(command, dict) else None
    command_flags = command_flags_match(command)

    checks: dict[str, Any] = {
        "run_dir_exists": run_dir.exists(),
        "hf_model_exists": hf.exists(),
        "scientific_metrics_exists": metrics_path.exists(),
        "launcher_result_exists": launcher_path.exists(),
        "train_command_exists": command_path.exists(),
        "checkpoint_100m_exists": chck100.exists(),
        "checkpoint_100m_model_exists": (chck100 / "model.safetensors").exists(),
        "checkpoint_100m_tokenizer_exists": tok_json.exists(),
        "tokenizer_sha_matches_expected": tok_sha == expected_tokenizer_sha256,
        "hf_root_tokenizer_sha_matches_expected": root_tok_sha == expected_tokenizer_sha256,
        "expected_tokenizer_file_exists": expected_tok_json.exists() if expected_tok_json is not None else False,
        "expected_tokenizer_file_sha_matches_expected": expected_tok_sha == expected_tokenizer_sha256 if expected_tok_json is not None else False,
        "checkpoint_tokenizer_canonical_fingerprint_matches_expected": (tok_fp is not None and tok_fp == expected_tok_fp),
        "hf_root_tokenizer_canonical_fingerprint_matches_expected": (root_tok_fp is not None and root_tok_fp == expected_tok_fp),
        "tokenizer_coordinate_matches_expected": (tok_sha == expected_tokenizer_sha256) or (tok_fp is not None and tok_fp == expected_tok_fp),
        "hf_root_tokenizer_coordinate_matches_expected": (root_tok_sha == expected_tokenizer_sha256) or (root_tok_fp is not None and root_tok_fp == expected_tok_fp),
        "aoa_ladder_complete": len(missing_aoa) == 0,
        "word_exposure_100m": isinstance(metrics, dict) and metrics.get("word_exposure") == 100_000_000,
        "vocab_size_16384": isinstance(metrics, dict) and metrics.get("vocab_size") == 16_384,
        "tokenizer_label_compliant": isinstance(metrics, dict) and "compliant" in str(metrics.get("tokenizer_label", "")).lower(),
        "returncode_zero": isinstance(launcher, dict) and launcher.get("returncode") == 0,
        "command_no_pretrained_or_resume_flags": not any(flag in command_list for flag in ["--resume", "--resume_from_checkpoint", "--model_name_or_path", "--from_pretrained"]),
        "command_frozen_recipe_flags_match": all(command_flags.values()),
    }

    provenance: dict[str, Any] = {
        "expected_tokenizer": rel(expected_tokenizer),
        "expected_train_file": rel(expected_train_file),
        "expected_pool_10m": rel(expected_pool_10m),
        "expected_metadata": rel(expected_meta),
        "expected_train_sha256": expected_train_sha256 or None,
        "expected_pool_sha256": expected_pool_sha256 or None,
        "expected_metadata_sha256": expected_meta_sha256 or None,
        "expected_pool_words": expected_pool_words or None,
        "expected_train_words": expected_train_words or None,
        "expected_pool_rows": expected_pool_rows or None,
        "expected_train_rows": expected_train_rows or None,
        "expected_passes": expected_passes or None,
        "verify_corpus_content": verify_corpus_content,
        "command_train_file": command_value(command, "train_file", "--example_jsonl"),
        "command_pool_10m": command_value(command, "pool_10m", None),
        "command_metadata": command_value(command, "metadata", "--example_jsonl_meta"),
        "command_tokenizer_dir": command_value(command, "tokenizer_dir", "--tokenizer_path"),
        "command_max_word_exposure": command_value(command, "max_word_exposure_requested", "--max_word_exposure"),
        "command_word_info": command_word_info,
        "command_frozen_recipe_flags": command_flags,
    }

    if expected_tokenizer is not None:
        checks["command_tokenizer_path_matches_expected"] = same_path(provenance["command_tokenizer_dir"], expected_tokenizer)
        tok_meta_sha = None
        if isinstance(command_tok_meta, dict):
            tok_meta_sha = command_tok_meta.get("tokenizer", {}).get("tokenizer_json_sha256")
        checks["command_tokenizer_metadata_sha_matches_expected"] = tok_meta_sha == expected_tokenizer_sha256 if isinstance(command_tok_meta, dict) else False

    if expected_train_file is not None:
        checks["expected_train_file_exists"] = expected_train_file.is_file()
        checks["expected_pool_10m_exists"] = expected_pool_10m.is_file() if expected_pool_10m else False
        checks["expected_metadata_exists"] = expected_meta.is_file() if expected_meta else False
        checks["command_train_file_matches_expected"] = same_path(provenance["command_train_file"], expected_train_file)
        checks["command_pool_10m_matches_expected"] = same_path(provenance["command_pool_10m"], expected_pool_10m)
        checks["command_metadata_matches_expected"] = same_path(provenance["command_metadata"], expected_meta)
        checks["command_expected_train_sha_matches_expected"] = (not expected_train_sha256) or (isinstance(command, dict) and command.get("expected_train_sha256") == expected_train_sha256)
        checks["command_hash_preflight_passed"] = isinstance(command, dict) and command.get("hash_ok") is True
        try:
            checks["command_max_word_exposure_expected"] = int(str(provenance["command_max_word_exposure"])) == expected_train_words
        except Exception:
            checks["command_max_word_exposure_expected"] = False

        if isinstance(command_tok_meta, dict):
            td = command_tok_meta.get("training_data", {})
            checks["tokenizer_training_pool_sha_matches_expected"] = td.get("pool_sha256") == expected_pool_sha256
            checks["tokenizer_training_words_expected"] = nested_pool_words(command_tok_meta) == expected_pool_words
            checks["tokenizer_training_rows_expected"] = (expected_pool_rows <= 0) or (nested_pool_rows(command_tok_meta) == expected_pool_rows)
            checks["tokenizer_training_pool_path_matches_expected"] = same_path(td.get("pool"), expected_pool_10m)
        else:
            checks["tokenizer_training_pool_sha_matches_expected"] = False
            checks["tokenizer_training_words_expected"] = False
            checks["tokenizer_training_rows_expected"] = False
            checks["tokenizer_training_pool_path_matches_expected"] = False

        corpus_sha: dict[str, str | None] = {"train_100m": None, "pool_10m": None, "metadata": None}
        if expected_train_file.is_file() and expected_train_sha256:
            corpus_sha["train_100m"] = sha256_file(expected_train_file)
            checks["expected_train_sha_matches_expected"] = corpus_sha["train_100m"] == expected_train_sha256
        if expected_pool_10m and expected_pool_10m.is_file() and expected_pool_sha256:
            corpus_sha["pool_10m"] = sha256_file(expected_pool_10m)
            checks["expected_pool_sha_matches_expected"] = corpus_sha["pool_10m"] == expected_pool_sha256
        if expected_meta and expected_meta.is_file() and expected_meta_sha256:
            corpus_sha["metadata"] = sha256_file(expected_meta)
            checks["expected_metadata_sha_matches_expected"] = corpus_sha["metadata"] == expected_meta_sha256
        provenance["computed_sha256"] = corpus_sha

        meta_json = safe_json(expected_meta) if expected_meta else None
        meta_rec = metadata_provenance(meta_json, expected_train_file, expected_pool_10m, expected_train_sha256, expected_pool_sha256, expected_pool_words, expected_train_words, expected_pool_rows, expected_train_rows, expected_passes)
        provenance["metadata_provenance"] = meta_rec
        for key, val in meta_rec.items():
            if key.startswith("metadata_"):
                checks[key] = val

        if verify_corpus_content:
            corpus_counts: dict[str, Any] = {}
            if expected_pool_10m and expected_pool_10m.is_file():
                corpus_counts["pool_10m"] = count_jsonl_words(expected_pool_10m)
                checks["pool_10m_words_expected"] = corpus_counts["pool_10m"].get("words") == expected_pool_words
                checks["pool_10m_rows_expected"] = (expected_pool_rows <= 0) or (corpus_counts["pool_10m"].get("rows") == expected_pool_rows)
                checks["pool_10m_rows_all_parse"] = corpus_counts["pool_10m"].get("bad_rows") == 0
            if expected_train_file.is_file():
                corpus_counts["train_100m"] = count_jsonl_words(expected_train_file)
                checks["train_100m_words_expected"] = corpus_counts["train_100m"].get("words") == expected_train_words
                checks["train_100m_rows_expected"] = (expected_train_rows <= 0) or (corpus_counts["train_100m"].get("rows") == expected_train_rows)
                checks["train_100m_rows_all_parse"] = corpus_counts["train_100m"].get("bad_rows") == 0
            if "pool_10m" in corpus_counts and "train_100m" in corpus_counts and expected_passes > 0:
                checks["train_rows_equal_pool_rows_times_passes"] = corpus_counts["train_100m"].get("rows") == corpus_counts["pool_10m"].get("rows") * expected_passes
                checks["train_words_equal_pool_words_times_passes"] = corpus_counts["train_100m"].get("words") == corpus_counts["pool_10m"].get("words") * expected_passes
            provenance["computed_word_counts"] = corpus_counts
        else:
            if isinstance(command_word_info, dict) and expected_train_words > 0:
                checks["command_word_info_train_words_expected"] = command_word_info.get("words") == expected_train_words
                checks["command_word_info_train_rows_expected"] = (expected_train_rows <= 0) or (command_word_info.get("rows") == expected_train_rows)

    base_required = [
        "hf_model_exists",
        "scientific_metrics_exists",
        "launcher_result_exists",
        "train_command_exists",
        "checkpoint_100m_exists",
        "checkpoint_100m_model_exists",
        "checkpoint_100m_tokenizer_exists",
        "tokenizer_coordinate_matches_expected",
        "aoa_ladder_complete",
        "word_exposure_100m",
        "vocab_size_16384",
        "tokenizer_label_compliant",
        "returncode_zero",
        "command_no_pretrained_or_resume_flags",
        "command_frozen_recipe_flags_match",
    ]
    if expected_tokenizer is not None:
        base_required += ["expected_tokenizer_file_exists", "expected_tokenizer_file_sha_matches_expected", "command_tokenizer_path_matches_expected", "command_tokenizer_metadata_sha_matches_expected"]
    if expected_train_file is not None:
        base_required += [
            "expected_train_file_exists",
            "expected_pool_10m_exists",
            "expected_metadata_exists",
            "expected_train_sha_matches_expected",
            "expected_pool_sha_matches_expected",
            "expected_metadata_sha_matches_expected",
            "command_train_file_matches_expected",
            "command_pool_10m_matches_expected",
            "command_metadata_matches_expected",
            "command_expected_train_sha_matches_expected",
            "command_max_word_exposure_expected",
            "tokenizer_training_pool_sha_matches_expected",
            "tokenizer_training_words_expected",
            "tokenizer_training_rows_expected",
            "tokenizer_training_pool_path_matches_expected",
            "metadata_declares_pool_words_expected",
            "metadata_declares_passes_expected",
            "metadata_declares_all_exact_10m",
            "metadata_pool_sha_matches_expected",
            "metadata_train_sha_matches_expected",
            "metadata_pool_path_matches_expected",
            "metadata_train_path_matches_expected",
            "metadata_family_pool_words_exact",
            "metadata_family_pool_rows_expected",
            "metadata_expected_train_rows_by_passes",
            "metadata_expected_train_words_by_passes",
        ]
        if verify_corpus_content:
            base_required += [
                "pool_10m_words_expected",
                "pool_10m_rows_expected",
                "pool_10m_rows_all_parse",
                "train_100m_words_expected",
                "train_100m_rows_expected",
                "train_100m_rows_all_parse",
                "train_rows_equal_pool_rows_times_passes",
                "train_words_equal_pool_words_times_passes",
            ]
        else:
            base_required += ["command_word_info_train_words_expected", "command_word_info_train_rows_expected"]
    complete = all(bool(checks.get(k)) for k in base_required)

    short_metrics = None
    if isinstance(metrics, dict):
        short_metrics = {k: metrics.get(k) for k in [
            "word_exposure", "actual_training_steps", "loss_first", "loss_last",
            "parameter_count", "vocab_size", "tokenizer_label", "tokenizer_path",
            "seed", "extra_init_seed", "train_rng_seed", "seq_length", "batch_size",
            "learning_rate", "masking_curriculum", "mask_prob_start", "mask_prob_end",
        ] if k in metrics}
        ckpts = metrics.get("saved_checkpoints") or []
        short_metrics["saved_checkpoint_count"] = len(ckpts)
        short_metrics["first_checkpoint"] = ckpts[0].get("name") if ckpts and isinstance(ckpts[0], dict) else None
        short_metrics["last_checkpoint"] = ckpts[-1].get("name") if ckpts and isinstance(ckpts[-1], dict) else None

    failed_required = [k for k in base_required if not checks.get(k)]
    return {
        "arm": name,
        "run_dir": str(run_dir),
        "checks": checks,
        "required_for_complete_100m_endpoint": base_required,
        "failed_required": failed_required,
        "complete_100m_compliant_retrain": complete,
        "available_checkpoint_count": len(available_ckpts),
        "available_checkpoints_tail": available_ckpts[-12:],
        "missing_aoa_steps": missing_aoa,
        "checkpoint_tokenizer_sha256": tok_sha,
        "hf_root_tokenizer_sha256": root_tok_sha,
        "expected_tokenizer_sha256_computed": expected_tok_sha,
        "checkpoint_tokenizer_canonical_fingerprint": tok_fp,
        "hf_root_tokenizer_canonical_fingerprint": root_tok_fp,
        "expected_tokenizer_canonical_fingerprint": expected_tok_fp,
        "metrics_path": str(metrics_path),
        "launcher_result_path": str(launcher_path),
        "train_command_path": str(command_path),
        "metrics_summary": short_metrics,
        "launcher_status": {k: launcher.get(k) for k in ["status", "returncode", "started_utc", "finished_utc", "metrics_exists", "stderr_tail"]} if isinstance(launcher, dict) else launcher,
        "command_tokenizer_dir": provenance.get("command_tokenizer_dir"),
        "command_max_word_exposure": provenance.get("command_max_word_exposure"),
        "provenance": provenance,
        "interpretation": "Only complete=True supports official-compatible evaluation. Any false required field means the run should not be scored as the legal endpoint until the mismatch is understood.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--expected-tokenizer", default=str(TOKENIZER))
    ap.add_argument("--expected-tokenizer-sha256", default=EXPECTED_TOKENIZER_SHA256)
    ap.add_argument("--expected-train-file", default="")
    ap.add_argument("--expected-train-sha256", default="")
    ap.add_argument("--expected-pool-10m", default="")
    ap.add_argument("--expected-pool-sha256", default="")
    ap.add_argument("--expected-meta", default="")
    ap.add_argument("--expected-meta-sha256", default="")
    ap.add_argument("--expected-pool-words", type=int, default=0)
    ap.add_argument("--expected-train-words", type=int, default=0)
    ap.add_argument("--expected-pool-rows", type=int, default=0)
    ap.add_argument("--expected-train-rows", type=int, default=0)
    ap.add_argument("--expected-passes", type=int, default=0)
    ap.add_argument("--verify-corpus-content", action="store_true", help="Count JSONL rows/words for the expected 10M and 100M files before accepting a delivered endpoint.")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_tokenizer = user_path(args.expected_tokenizer)
    expected_train_file = user_path(args.expected_train_file)
    expected_pool_10m = user_path(args.expected_pool_10m)
    expected_meta = user_path(args.expected_meta)

    records = []
    for arm in args.arms:
        if arm in DEFAULT_RUNS:
            run_dir = DEFAULT_RUNS[arm]
            name = arm
        else:
            run_dir = user_path(arm) or pathlib.Path(arm)
            name = run_dir.name
        records.append(inspect_run(
            name,
            run_dir,
            expected_tokenizer=expected_tokenizer,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            expected_train_file=expected_train_file,
            expected_train_sha256=args.expected_train_sha256,
            expected_pool_10m=expected_pool_10m,
            expected_pool_sha256=args.expected_pool_sha256,
            expected_meta=expected_meta,
            expected_meta_sha256=args.expected_meta_sha256,
            expected_pool_words=args.expected_pool_words,
            expected_train_words=args.expected_train_words,
            expected_pool_rows=args.expected_pool_rows,
            expected_train_rows=args.expected_train_rows,
            expected_passes=args.expected_passes,
            verify_corpus_content=args.verify_corpus_content,
        ))
    payload = {
        "status": "COMPLIANT_RETRAIN_PROVENANCE_INSPECTION",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "expected_tokenizer": rel(expected_tokenizer),
        "expected_tokenizer_sha256": args.expected_tokenizer_sha256,
        "expected_corpus": {
            "train_file": rel(expected_train_file),
            "train_sha256": args.expected_train_sha256 or None,
            "pool_10m": rel(expected_pool_10m),
            "pool_sha256": args.expected_pool_sha256 or None,
            "metadata": rel(expected_meta),
            "metadata_sha256": args.expected_meta_sha256 or None,
            "pool_words": args.expected_pool_words or None,
            "train_words": args.expected_train_words or None,
            "pool_rows": args.expected_pool_rows or None,
            "train_rows": args.expected_train_rows or None,
            "passes": args.expected_passes or None,
            "verify_corpus_content": bool(args.verify_corpus_content),
        },
        "expected_aoa_steps": EXPECTED_AOA_STEPS,
        "records": records,
    }
    out_json = out_dir / "compliant_retrain_inspection.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "out_json": str(out_json),
        "records": [
            {
                "arm": r["arm"],
                "complete": r["complete_100m_compliant_retrain"],
                "available_checkpoint_count": r["available_checkpoint_count"],
                "missing_aoa_steps_head": r["missing_aoa_steps"][:5],
                "failed_required_head": r["failed_required"][:12],
                "launcher_status": r["launcher_status"],
            }
            for r in records
        ],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
