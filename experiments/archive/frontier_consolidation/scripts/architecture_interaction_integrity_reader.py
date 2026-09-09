#!/usr/bin/env python3
"""research-hardened integrity reader for the DeBERTa positional-package interaction.

CPU/file-only. It verifies the four cells needed to read the architecture interaction:
  * existing legal full-DeBERTa compact reference from research
  * new legal full-DeBERTa repeat reference from research
  * new legal no-disentangle compact cell from research
  * new legal no-disentangle repeat cell from research

The scientific estimand is the data-treatment interaction:
    (compact - repeat)_full_DeBERTa versus (compact - repeat)_no_disentangle_abs.
Survival of compact-minus-repeat without c2p/p2c score tensors shows those score terms are not
necessary for the compact-view effect in this coordinate. Attenuation implicates the whole removed
score-term package together with parameterization and score-composition changes, not c2p vs p2c alone.

This script does not train, evaluate selected tasks, run SuperGLUE/AoA, upload, or submit.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------------------
# Paths and fixed scientific coordinate
# --------------------------------------------------------------------------------------


def find_user_root() -> Path:
    return _PUBLIC_ROOT


ROOT = find_user_root()
WS = ROOT / "experiments/archive/frontier_consolidation"
DEFAULT_OUT = WS / "data/architecture_interaction_integrity"

EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
TOKENIZER_JSON = WS / "data/compliant_tokenizer/tokenizer.json"

STREAM_SHA = {
    "compact": "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691",
    "repeat": "91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa",
}
STREAM_PATH = {
    "compact": "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl",
    "repeat": "experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl",
}

FULL_PARAM_COUNT = 34_467_424
NODIS_PARAM_COUNT = 30_773_344
REQUIRED_CHECKPOINTS = ["chck_80M", "chck_100M"]

ARMS: dict[str, dict[str, Any]] = {
    "full_compact_existing": {
        "run_dir": WS / "training/runs/complianttok_reinvest_seed43022_r2",
        "variant": "full_p2c_c2p_abs",
        "data_arm": "compact",
        "role": "existing legal full-DeBERTa compact reference; reused only after exact-coordinate checks",
        "expected_param_count": FULL_PARAM_COUNT,
        "expected_pos_att_type": ["p2c", "c2p"],
        "expected_cli_pos_att_type": None,  # older research used trainer default rather than an explicit CLI flag
        "expected_saved_checkpoint_count": 100,
        "expected_checkpoint_words_cli": 1_000_000,
        "existing_reference": True,
    },
    "full_repeat": {
        "run_dir": WS / "training/runs/full_p2c_c2p_abs_repeat_deberta100M_seed43022",
        "variant": "full_p2c_c2p_abs",
        "data_arm": "repeat",
        "role": "new missing legal full-DeBERTa repeat reference",
        "expected_param_count": FULL_PARAM_COUNT,
        "expected_pos_att_type": ["p2c", "c2p"],
        "expected_cli_pos_att_type": "p2c,c2p",
        "expected_saved_checkpoint_count": 10,
        "expected_checkpoint_words_cli": 10_000_000,
        "existing_reference": False,
    },
    "nodis_compact": {
        "run_dir": WS / "training/runs/no_disentangle_abs_compact_deberta100M_seed43022",
        "variant": "no_disentangle_abs",
        "data_arm": "compact",
        "role": "compact arm with c2p/p2c attention-score terms removed and absolute input positions retained",
        "expected_param_count": NODIS_PARAM_COUNT,
        "expected_pos_att_type": [],
        "expected_cli_pos_att_type": "",
        "expected_saved_checkpoint_count": 10,
        "expected_checkpoint_words_cli": 10_000_000,
        "existing_reference": False,
    },
    "nodis_repeat": {
        "run_dir": WS / "training/runs/no_disentangle_abs_repeat_deberta100M_seed43022",
        "variant": "no_disentangle_abs",
        "data_arm": "repeat",
        "role": "repeat arm with c2p/p2c attention-score terms removed and absolute input positions retained",
        "expected_param_count": NODIS_PARAM_COUNT,
        "expected_pos_att_type": [],
        "expected_cli_pos_att_type": "",
        "expected_saved_checkpoint_count": 10,
        "expected_checkpoint_words_cli": 10_000_000,
        "existing_reference": False,
    },
}

METRIC_EXPECTED = {
    "word_exposure": 100_000_000,
    "actual_training_steps": 2529,
    "model_family": "DebertaV2ForMaskedLM",
    "vocab_size": 16_384,
    "tokenizer_label": "compliant16k_reinvest10M",
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "seed": 43,
    "seq_length": 256,
    "max_seq_length": 256,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
}

COMMAND_EXPECTED_COMMON = {
    "--tokenizer_label": "compliant16k_reinvest10M",
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
    "--max_word_exposure": "100000000",
    "--num_workers": "0",
}
COMMAND_EXPECTED_NEW_ONLY = {
    "--lr_total_steps": "2529",
    "--log_every": "50",
    "--dynamics_trace_every": "200",
}

RECIPE_EXPECTED_NEW = {
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
    "weight_decay": 0.01,
    "warmup_fraction": 0.06,
    "lr_total_steps": 2529,
    "masking_curriculum": "wwm_fixed",
    "mask_prob_start": 0.15,
    "mask_prob_end": 0.15,
    "checkpoint_words": 10_000_000,
    "max_word_exposure": 100_000_000,
    "num_workers": 0,
    "log_every": 50,
    "dynamics_trace_every": 200,
}

RECIPE_EXPECTED_EXISTING = {
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
}


# --------------------------------------------------------------------------------------
# Basic IO helpers
# --------------------------------------------------------------------------------------


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_path(path_like: str | Path | None) -> Path | None:
    if path_like is None:
        return None
    p = Path(str(path_like))
    return p if p.is_absolute() else ROOT / p


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def tail(path: Path, n: int = 2400) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-n:]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def values_equal(observed: Any, expected: Any, tol: float = 1e-12) -> bool:
    if isinstance(expected, float):
        try:
            return math.isfinite(float(observed)) and abs(float(observed) - expected) <= tol
        except Exception:
            return False
    return observed == expected


def finite_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Metadata extraction
# --------------------------------------------------------------------------------------


def get_train_command_sha(train_command: dict[str, Any] | None) -> str | None:
    if not train_command:
        return None
    for key in ["train_file_sha256", "actual_train_sha256", "expected_train_sha256"]:
        if train_command.get(key):
            return str(train_command[key])
    return None


def get_tokenizer_sha(train_command: dict[str, Any] | None) -> str | None:
    if not train_command:
        return None
    if train_command.get("tokenizer_json_sha256"):
        return str(train_command["tokenizer_json_sha256"])
    meta = train_command.get("tokenizer_metadata") or {}
    tok = meta.get("tokenizer") or {}
    if tok.get("tokenizer_json_sha256"):
        return str(tok["tokenizer_json_sha256"])
    if meta.get("tokenizer_json_sha256_observed"):
        return str(meta["tokenizer_json_sha256_observed"])
    return None


def get_command_list(train_command: dict[str, Any] | None) -> list[str]:
    if not train_command:
        return []
    cmd = train_command.get("command") or []
    return [str(x) for x in cmd] if isinstance(cmd, list) else []


def command_arg(command: list[str], flag: str) -> str | None:
    try:
        i = command.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(command):
        return None
    return command[i + 1]


def get_recipe(train_command: dict[str, Any] | None) -> dict[str, Any]:
    if not train_command:
        return {}
    recipe = train_command.get("recipe")
    return recipe if isinstance(recipe, dict) else {}


# --------------------------------------------------------------------------------------
# Checkpoint structure
# --------------------------------------------------------------------------------------


def checkpoint_dir(run_dir: Path, ck: str) -> Path:
    return run_dir / "hf_model" / ck


def checkpoint_model_file(run_dir: Path, ck: str) -> Path | None:
    cdir = checkpoint_dir(run_dir, ck)
    for name in ["model.safetensors", "pytorch_model.bin"]:
        path = cdir / name
        if path.exists():
            return path
    return None


def checkpoint_config(run_dir: Path, ck: str) -> dict[str, Any] | None:
    return load_json(checkpoint_dir(run_dir, ck) / "config.json")


def list_state_keys(model_path: Path) -> list[str]:
    if model_path.name == "model.safetensors":
        from safetensors.torch import safe_open

        with safe_open(model_path, framework="pt", device="cpu") as f:
            return list(f.keys())
    import torch

    try:
        obj = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(model_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"unrecognized checkpoint object type: {type(obj)}")
    return [str(k) for k in obj.keys()]


def state_dict_signature(run_dir: Path, ck: str, expected_pos_att_type: list[str]) -> dict[str, Any]:
    path = checkpoint_model_file(run_dir, ck)
    rec: dict[str, Any] = {
        "checkpoint": ck,
        "model_file": str(path) if path else None,
        "model_file_exists": path is not None,
        "ok": False,
        "problems": [],
    }
    if path is None:
        rec["problems"].append(f"{ck} model file missing")
        return rec
    try:
        keys = list_state_keys(path)
    except Exception as exc:
        rec["problems"].append(f"could not list state keys for {ck}: {exc}")
        return rec

    pos_key = [k for k in keys if ".pos_key_proj." in k]
    pos_query = [k for k in keys if ".pos_query_proj." in k]
    has_abs = "deberta.embeddings.position_embeddings.weight" in keys
    has_rel = "deberta.encoder.rel_embeddings.weight" in keys
    expected_full = bool(expected_pos_att_type)
    rec.update(
        {
            "n_total_keys": len(keys),
            "n_pos_key_proj_keys": len(pos_key),
            "n_pos_query_proj_keys": len(pos_query),
            "has_absolute_position_embeddings": has_abs,
            "has_encoder_rel_embeddings": has_rel,
            "sample_pos_key_proj_keys": pos_key[:4],
            "sample_pos_query_proj_keys": pos_query[:4],
        }
    )
    if not has_abs:
        rec["problems"].append("absolute position embedding tensor missing")
    if not has_rel:
        rec["problems"].append("encoder relative embedding tensor missing")
    if expected_full:
        if len(pos_key) != 16:
            rec["problems"].append(f"full variant expected 16 pos_key_proj tensors, found {len(pos_key)}")
        if len(pos_query) != 16:
            rec["problems"].append(f"full variant expected 16 pos_query_proj tensors, found {len(pos_query)}")
    else:
        if len(pos_key) != 0:
            rec["problems"].append(f"no-disentangle variant expected zero pos_key_proj tensors, found {len(pos_key)}")
        if len(pos_query) != 0:
            rec["problems"].append(f"no-disentangle variant expected zero pos_query_proj tensors, found {len(pos_query)}")
    rec["ok"] = len(rec["problems"]) == 0
    return rec


# --------------------------------------------------------------------------------------
# Scientific-coordinate verification
# --------------------------------------------------------------------------------------


def check_recipe(recipe: dict[str, Any], expected: dict[str, Any], *, mandatory: bool) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
    result: dict[str, bool] = {}
    problems: list[str] = []
    brief: dict[str, Any] = {}
    for key, exp in expected.items():
        observed = recipe.get(key)
        brief[key] = observed
        present = key in recipe
        ok = present and values_equal(observed, exp)
        result[f"recipe_{key}_ok"] = ok
        if mandatory and not present:
            problems.append(f"recipe field missing: {key}")
        if present and not ok:
            problems.append(f"recipe {key} mismatch: {observed} != {exp}")
    return result, problems, brief


def check_command_args(command: list[str], spec: dict[str, Any]) -> tuple[dict[str, bool], list[str], dict[str, Any]]:
    result: dict[str, bool] = {}
    problems: list[str] = []
    observed: dict[str, Any] = {}
    is_new = not spec.get("existing_reference")
    expected_flags = dict(COMMAND_EXPECTED_COMMON)
    if is_new:
        expected_flags.update(COMMAND_EXPECTED_NEW_ONLY)
    expected_flags["--checkpoint_words"] = str(spec["expected_checkpoint_words_cli"])
    if spec.get("expected_cli_pos_att_type") is not None:
        expected_flags["--deberta_pos_att_type"] = str(spec["expected_cli_pos_att_type"])

    if not command:
        problems.append("train command list missing")
        return result, problems, observed

    for flag, exp in expected_flags.items():
        val = command_arg(command, flag)
        observed[flag] = val
        ok = val == exp
        result[f"command_{flag.lstrip('-')}_ok"] = ok
        if val is None:
            problems.append(f"command flag missing: {flag}")
        elif not ok:
            problems.append(f"command {flag} mismatch: {val!r} != {exp!r}")

    train_file = command_arg(command, "--example_jsonl")
    tokenizer_path = command_arg(command, "--tokenizer_path")
    observed["--example_jsonl"] = train_file
    observed["--tokenizer_path"] = tokenizer_path
    exp_stream = STREAM_PATH[spec["data_arm"]]
    result["command_example_jsonl_name_ok"] = train_file is not None and Path(train_file).name == Path(exp_stream).name
    result["command_tokenizer_path_ok"] = tokenizer_path is not None and Path(tokenizer_path).name == "compliant_tokenizer"
    if not result["command_example_jsonl_name_ok"]:
        problems.append(f"command --example_jsonl mismatch: {train_file} vs expected {exp_stream}")
    if not result["command_tokenizer_path_ok"]:
        problems.append(f"command --tokenizer_path mismatch: {tokenizer_path}")
    return result, problems, observed


def check_train_command(train_command: dict[str, Any] | None, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    data_arm = spec["data_arm"]
    result: dict[str, Any] = {}
    problems: list[str] = []
    brief: dict[str, Any] = {}
    if train_command is None:
        return {"train_command_exists": False}, ["train_command.json missing"], brief

    result["train_command_exists"] = True
    is_new = not spec.get("existing_reference")
    brief["status"] = train_command.get("status")
    brief["variant"] = train_command.get("variant")
    brief["data_arm"] = train_command.get("data_arm")
    brief["train_file_sha256"] = get_train_command_sha(train_command)
    brief["tokenizer_json_sha256"] = get_tokenizer_sha(train_command)

    if is_new:
        expected_status = "DEBERTA_POSITIONAL_ABLATION_PREFLIGHT_OK"
        for key, exp in [("status", expected_status), ("variant", spec["variant"]), ("data_arm", data_arm)]:
            obs = train_command.get(key)
            ok = obs == exp
            result[f"train_command_{key}_ok"] = ok
            if not ok:
                problems.append(f"train_command {key} mismatch: {obs} != {exp}")

    sha = get_train_command_sha(train_command)
    tok_sha = get_tokenizer_sha(train_command)
    result["train_stream_sha_matches_expected"] = sha == STREAM_SHA[data_arm]
    result["train_tokenizer_sha_matches_legal"] = tok_sha == EXPECTED_TOKENIZER_SHA
    if sha != STREAM_SHA[data_arm]:
        problems.append(f"train stream SHA mismatch: {sha} != {STREAM_SHA[data_arm]}")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        problems.append(f"tokenizer SHA mismatch or absent in train_command: {tok_sha}")

    # New research arms contain exact stream accounting and model-variant signatures in train_command.json.
    word_info = train_command.get("word_info")
    brief["word_info"] = word_info
    if is_new:
        if not isinstance(word_info, dict):
            problems.append("new research train_command word_info missing")
            result["train_command_word_info_ok"] = False
        else:
            expected_word_info = {
                "rows": 647400,
                "words": 100000000,
                "changed_rows": 30050,
                "unique_changed_ids": 3005,
                "exact_100M": True,
            }
            wi_ok = True
            for key, exp in expected_word_info.items():
                ok = word_info.get(key) == exp
                result[f"train_command_word_info_{key}_ok"] = ok
                wi_ok = wi_ok and ok
                if not ok:
                    problems.append(f"word_info {key} mismatch: {word_info.get(key)} != {exp}")
            bad = word_info.get("bad_changed_repeat_counts")
            ok_bad = bad in ({}, None)
            result["train_command_word_info_repeat_counts_ok"] = ok_bad
            wi_ok = wi_ok and ok_bad
            if not ok_bad:
                problems.append(f"bad changed repeat counts: {bad}")
            result["train_command_word_info_ok"] = wi_ok

        mv = train_command.get("model_variant") or {}
        brief["model_variant"] = {
            "parameter_count": mv.get("parameter_count"),
            "vocab_size": mv.get("vocab_size"),
            "has_encoder_rel_embeddings": mv.get("has_encoder_rel_embeddings"),
            "has_pos_key_proj": mv.get("has_pos_key_proj"),
            "has_pos_query_proj": mv.get("has_pos_query_proj"),
        }
        mv_expect = {
            "parameter_count": spec["expected_param_count"],
            "vocab_size": 16_384,
            "has_encoder_rel_embeddings": True,
            "has_pos_key_proj": bool(spec["expected_pos_att_type"]),
            "has_pos_query_proj": bool(spec["expected_pos_att_type"]),
        }
        for key, exp in mv_expect.items():
            ok = mv.get(key) == exp
            result[f"train_command_model_variant_{key}_ok"] = ok
            if not ok:
                problems.append(f"model_variant {key} mismatch: {mv.get(key)} != {exp}")

    recipe = get_recipe(train_command)
    recipe_expected = RECIPE_EXPECTED_NEW if is_new else RECIPE_EXPECTED_EXISTING
    recipe_result, recipe_problems, recipe_brief = check_recipe(recipe, recipe_expected, mandatory=is_new)
    result.update(recipe_result)
    problems.extend(recipe_problems)
    brief["recipe_observed"] = recipe_brief

    cmd_result, cmd_problems, cmd_brief = check_command_args(get_command_list(train_command), spec)
    result.update(cmd_result)
    problems.extend(cmd_problems)
    brief["command_observed"] = cmd_brief
    return result, problems, brief


def check_metrics(metrics: dict[str, Any] | None, spec: dict[str, Any], run_dir: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    result: dict[str, Any] = {"metrics_exists": metrics is not None}
    problems: list[str] = []
    brief: dict[str, Any] = {}
    if metrics is None:
        return result, ["scientific_metrics.json missing"], brief

    for key, exp in METRIC_EXPECTED.items():
        obs = metrics.get(key)
        brief[key] = obs
        ok = values_equal(obs, exp)
        result[f"metrics_{key}_ok"] = ok
        if not ok:
            problems.append(f"metrics {key} mismatch: {obs} != {exp}")

    obs_param = metrics.get("parameter_count")
    brief["parameter_count"] = obs_param
    ok_param = obs_param == spec["expected_param_count"]
    result["metrics_parameter_count_ok"] = ok_param
    if not ok_param:
        problems.append(f"metrics parameter_count mismatch: {obs_param} != {spec['expected_param_count']}")

    for key in ["loss_first", "loss_last"]:
        brief[key] = metrics.get(key)
        ok = finite_number(metrics.get(key))
        result[f"metrics_{key}_finite"] = ok
        if not ok:
            problems.append(f"metrics {key} not finite: {metrics.get(key)}")

    saved = metrics.get("saved_checkpoints", [])
    saved_names = [str(x.get("name")) for x in saved if isinstance(x, dict)] if isinstance(saved, list) else []
    brief["saved_checkpoint_count"] = len(saved_names)
    brief["first_checkpoint"] = saved_names[0] if saved_names else None
    brief["last_checkpoint"] = saved_names[-1] if saved_names else None
    expected_count = int(spec["expected_saved_checkpoint_count"])
    ok_count = len(saved_names) == expected_count
    result["metrics_saved_checkpoint_count_ok"] = ok_count
    if not ok_count:
        problems.append(f"saved checkpoint count {len(saved_names)} != expected {expected_count}")

    present = {}
    in_metrics = {}
    for ck in REQUIRED_CHECKPOINTS:
        present[ck] = checkpoint_model_file(run_dir, ck) is not None
        in_metrics[ck] = ck in saved_names
        if not present[ck]:
            problems.append(f"required checkpoint model missing: {ck}")
        if not in_metrics[ck]:
            problems.append(f"required checkpoint absent from metrics saved list: {ck}")
    result["required_checkpoint_model_files_present"] = present
    result["required_checkpoints_in_metrics"] = in_metrics
    return result, problems, brief


def check_launcher(launcher: dict[str, Any] | None, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    result = {"launcher_result_exists": launcher is not None}
    problems: list[str] = []
    brief: dict[str, Any] = {}
    if spec.get("existing_reference"):
        brief["not_required_for_existing_reference"] = True
        return result, problems, brief
    if launcher is None:
        return result, ["launcher_result.json missing for new research arm"], brief
    brief = {
        "status": launcher.get("status"),
        "variant": launcher.get("variant"),
        "data_arm": launcher.get("data_arm"),
        "gpu": launcher.get("gpu"),
        "returncode": launcher.get("returncode"),
        "metrics_exists": launcher.get("metrics_exists"),
        "started_utc": launcher.get("started_utc"),
        "finished_utc": launcher.get("finished_utc"),
    }
    expected_status = "DEBERTA_POSITIONAL_ABLATION_TRAIN_FINISHED"
    checks = {
        "launcher_status_ok": launcher.get("status") == expected_status,
        "launcher_variant_ok": launcher.get("variant") == spec["variant"],
        "launcher_data_arm_ok": launcher.get("data_arm") == spec["data_arm"],
        "launcher_returncode_zero": launcher.get("returncode") == 0,
        "launcher_metrics_exists_flag": bool(launcher.get("metrics_exists")) is True,
    }
    result.update(checks)
    for key, ok in checks.items():
        if not ok:
            problems.append(f"{key} failed; launcher brief={brief}")
    return result, problems, brief


def check_config_and_state(run_dir: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    result: dict[str, Any] = {}
    problems: list[str] = []
    brief: dict[str, Any] = {"configs": {}, "state_dict_signatures": {}}
    for ck in REQUIRED_CHECKPOINTS:
        cfg = checkpoint_config(run_dir, ck)
        brief["configs"][ck] = None if cfg is None else {
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "hidden_size": cfg.get("hidden_size"),
            "num_hidden_layers": cfg.get("num_hidden_layers"),
            "num_attention_heads": cfg.get("num_attention_heads"),
            "intermediate_size": cfg.get("intermediate_size"),
            "vocab_size": cfg.get("vocab_size"),
            "relative_attention": cfg.get("relative_attention"),
            "position_biased_input": cfg.get("position_biased_input"),
            "pos_att_type": cfg.get("pos_att_type"),
        }
        if cfg is None:
            problems.append(f"{ck} config.json missing")
            result[f"{ck}_config_exists"] = False
        else:
            result[f"{ck}_config_exists"] = True
            expected_config = {
                "hidden_size": 480,
                "num_hidden_layers": 8,
                "num_attention_heads": 8,
                "intermediate_size": 1920,
                "vocab_size": 16_384,
                "relative_attention": True,
                "position_biased_input": True,
            }
            for key, exp in expected_config.items():
                ok = cfg.get(key) == exp
                result[f"{ck}_config_{key}_ok"] = ok
                if not ok:
                    problems.append(f"{ck} config {key} mismatch: {cfg.get(key)} != {exp}")
            pos = cfg.get("pos_att_type") or []
            ok_pos = list(pos) == list(spec["expected_pos_att_type"])
            result[f"{ck}_config_pos_att_type_ok"] = ok_pos
            if not ok_pos:
                problems.append(f"{ck} config pos_att_type mismatch: {pos} != {spec['expected_pos_att_type']}")

        sig = state_dict_signature(run_dir, ck, spec["expected_pos_att_type"])
        brief["state_dict_signatures"][ck] = sig
        result[f"{ck}_state_dict_signature_ok"] = bool(sig.get("ok"))
        problems.extend(str(p) for p in sig.get("problems", []))
    return result, problems, brief


def check_arm(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run_dir"])
    metrics = load_json(run_dir / "scientific_metrics.json")
    launcher = load_json(run_dir / "launcher_result.json")
    train_command = load_json(run_dir / "train_command.json")
    stdout_log = run_dir / "train_stdout.log"
    stderr_log = run_dir / "train_stderr.log"

    problems: list[str] = []
    checks: dict[str, Any] = {"run_dir_exists": run_dir.exists()}
    if not run_dir.exists():
        problems.append("run_dir missing")

    local_tok_sha = sha256_file(TOKENIZER_JSON)
    checks["local_tokenizer_json_sha_ok"] = local_tok_sha == EXPECTED_TOKENIZER_SHA
    if local_tok_sha != EXPECTED_TOKENIZER_SHA:
        problems.append(f"current tokenizer.json SHA mismatch: {local_tok_sha} != {EXPECTED_TOKENIZER_SHA}")

    launcher_checks, launcher_problems, launcher_brief = check_launcher(launcher, spec)
    checks.update(launcher_checks)
    problems.extend(launcher_problems)

    train_checks, train_problems, train_brief = check_train_command(train_command, spec)
    checks.update(train_checks)
    problems.extend(train_problems)

    metric_checks, metric_problems, metrics_brief = check_metrics(metrics, spec, run_dir)
    checks.update(metric_checks)
    problems.extend(metric_problems)

    # Only inspect checkpoint tensors when the required files exist; otherwise a running arm remains clearly not ready.
    if all((checkpoint_model_file(run_dir, ck) is not None and (checkpoint_dir(run_dir, ck) / "config.json").exists()) for ck in REQUIRED_CHECKPOINTS):
        cfg_checks, cfg_problems, cfg_brief = check_config_and_state(run_dir, spec)
        checks.update(cfg_checks)
        problems.extend(cfg_problems)
    else:
        cfg_brief = {"configs": {}, "state_dict_signatures": {}}

    return {
        "arm": name,
        "variant": spec["variant"],
        "data_arm": spec["data_arm"],
        "role": spec["role"],
        "run_dir": str(run_dir),
        "existing_reference": bool(spec.get("existing_reference")),
        "expected": {
            "stream_sha256": STREAM_SHA[spec["data_arm"]],
            "stream_path": STREAM_PATH[spec["data_arm"]],
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
            "parameter_count": spec["expected_param_count"],
            "pos_att_type": spec["expected_pos_att_type"],
            "required_checkpoints": REQUIRED_CHECKPOINTS,
            "saved_checkpoint_count": spec["expected_saved_checkpoint_count"],
            "checkpoint_words_cli": spec["expected_checkpoint_words_cli"],
        },
        "checks": checks,
        "metrics_brief": metrics_brief,
        "train_command_brief": train_brief,
        "launcher_result_brief": launcher_brief,
        "checkpoint_brief": cfg_brief,
        "problems": problems,
        "ok_for_selected_eval": len(problems) == 0,
        "stdout_tail": tail(stdout_log),
        "stderr_tail": tail(stderr_log),
    }


# --------------------------------------------------------------------------------------
# Payloads and command line
# --------------------------------------------------------------------------------------


def selected_arms(names: list[str] | None) -> dict[str, dict[str, Any]]:
    if not names:
        return ARMS
    return {name: ARMS[name] for name in names}


def plan_payload(out_dir: Path, arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ARCHITECTURE_INTERACTION_INTEGRITY_PLAN",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "arms": {name: {k: (str(v) if isinstance(v, Path) else v) for k, v in spec.items()} for name, spec in arms.items()},
        "required_checkpoints": REQUIRED_CHECKPOINTS,
        "scientific_estimand": "Compare compact-minus-repeat under full DeBERTa with compact-minus-repeat under no_disentangle_abs. Survival without c2p/p2c score tensors shows those score terms are not necessary; attenuation implicates the whole removed score-term package plus parameterization and score-composition changes.",
        "file_only_no_training_selected_eval_superglue_aoa_upload_or_leaderboard": True,
    }


def inspect_payload(out_dir: Path, arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    arm_results = {name: check_arm(name, spec) for name, spec in arms.items()}
    all_selected = all(r["ok_for_selected_eval"] for r in arm_results.values()) if arm_results else False
    new_results = [r for r in arm_results.values() if not r["existing_reference"]]
    all_new = bool(new_results) and all(r["ok_for_selected_eval"] for r in new_results)
    all_four_present = set(arm_results) == set(ARMS)
    return {
        "status": "ARCHITECTURE_INTERACTION_INTEGRITY",
        "created_utc": now(),
        "out_dir": str(out_dir),
        "arm_results": arm_results,
        "all_selected_arms_ready_for_selected_eval": all_selected,
        "all_new_step243_arms_terminal_and_valid": all_new,
        "all_four_cells_ready_for_selected_eval": bool(all_four_present and all_selected),
        "readout_condition": "Run the selected panel only after all four cells pass this reader; existing full compact is reused only with matching legal tokenizer, compact stream SHA, recipe, checkpoint config, and direct tensor-key signature.",
        "file_only_boundary": "No selected scoring, SuperGLUE, AoA, upload, leaderboard submission, or training was performed by this reader.",
    }


def write_outputs(payload: dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [f"# {stem}", "", f"Created UTC: `{payload.get('created_utc')}`", ""]
    if "arm_results" in payload:
        lines.append(f"all_selected_arms_ready_for_selected_eval: `{payload.get('all_selected_arms_ready_for_selected_eval')}`")
        lines.append(f"all_four_cells_ready_for_selected_eval: `{payload.get('all_four_cells_ready_for_selected_eval')}`")
        lines.append("")
        for name, rec in payload["arm_results"].items():
            lines.append(f"## {name}")
            lines.append(f"- ok_for_selected_eval: `{rec.get('ok_for_selected_eval')}`")
            lines.append(f"- run_dir: `{rec.get('run_dir')}`")
            lines.append(f"- variant/data: `{rec.get('variant')}` / `{rec.get('data_arm')}`")
            lines.append(f"- metrics: `{json.dumps(rec.get('metrics_brief'), sort_keys=True)}`")
            sigs = rec.get("checkpoint_brief", {}).get("state_dict_signatures", {})
            if sigs:
                lines.append("- tensor signatures:")
                for ck, sig in sigs.items():
                    lines.append(
                        f"  - {ck}: ok={sig.get('ok')}, pos_key={sig.get('n_pos_key_proj_keys')}, pos_query={sig.get('n_pos_query_proj_keys')}, abs={sig.get('has_absolute_position_embeddings')}, rel={sig.get('has_encoder_rel_embeddings')}"
                    )
            if rec.get("problems"):
                lines.append("- problems:")
                for p in rec["problems"]:
                    lines.append(f"  - {p}")
            lines.append("")
    else:
        lines.append(payload.get("scientific_estimand", ""))
        lines.append("")
        for name, spec in payload.get("arms", {}).items():
            lines.append(f"- `{name}`: {spec.get('role')} at `{spec.get('run_dir')}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path), "md": str(md_path)}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=None)
    args = ap.parse_args()
    arms = selected_arms(args.arms)
    if args.plan_only:
        payload = plan_payload(args.out_dir, arms)
        write_outputs(payload, args.out_dir, "architecture_interaction_integrity_plan")
    else:
        payload = inspect_payload(args.out_dir, arms)
        write_outputs(payload, args.out_dir, "architecture_interaction_integrity")


if __name__ == "__main__":
    main()
