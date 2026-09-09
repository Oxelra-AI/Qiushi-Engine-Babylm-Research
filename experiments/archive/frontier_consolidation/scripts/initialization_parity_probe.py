#!/usr/bin/env python3
"""research: CPU-only initialization parity probe for DeBERTa positional ablation.

The active research experiment compares compact-minus-repeat in stock DeBERTa
(full p2c+c2p positional score terms) to compact-minus-repeat in
`no_disentangle_abs` (relative_attention=True, pos_att_type=[]). The training
launcher uses the same numeric seeds, but omitting optional positional projection
modules may shift the PyTorch initialization stream for later shared tensors.
This script empirically measures that at initialization and constructs a
common-copied no-disentangle model to quantify the separate initialization drift.

No training, selected evaluation, upload, AoA, SuperGLUE, or leaderboard action.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
STUDY = USER_ROOT / "experiments/archive" / 'frontier_consolidation'
WORKSPACE = STUDY
TRAINER_PATH = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts" / "masking_curriculum_trainer.py"
TOKENIZER_DIR = WORKSPACE / "data" / "compliant_tokenizer"
STREAM_PATH = WORKSPACE / "data" / "density_cleanqwen_overlay_medium_riskhard" / "cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"

RECIPE = {
    "hidden_size": 480,
    "n_layer": 8,
    "n_head": 8,
    "ffn_mult": 4,
    "max_seq_length": 256,
    "max_position_embeddings": 512,
    "max_relative_positions": 256,
    "position_buckets": 256,
    "seed": 43,
    "extra_init_seed": 43022,
    "train_rng_seed": 43023,
}

VARIANTS = {
    "full_p2c_c2p_abs": "p2c,c2p",
    "c2p_only_abs": "c2p",
    "p2c_only_abs": "p2c",
    "no_disentangle_abs": "",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("compact_experience_masking_curriculum_trainer", str(TRAINER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import trainer from {TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reset_all_rng(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainer_args(pos_att_type: str) -> argparse.Namespace:
    return argparse.Namespace(
        hidden_size=RECIPE["hidden_size"],
        n_layer=RECIPE["n_layer"],
        n_head=RECIPE["n_head"],
        ffn_mult=RECIPE["ffn_mult"],
        max_seq_length=RECIPE["max_seq_length"],
        max_position_embeddings=RECIPE["max_position_embeddings"],
        max_relative_positions=RECIPE["max_relative_positions"],
        position_buckets=RECIPE["position_buckets"],
        deberta_pos_att_type=pos_att_type,
    )


def build_variant(trainer, torch, tokenizer, variant: str):
    reset_all_rng(torch, RECIPE["seed"])
    reset_all_rng(torch, RECIPE["extra_init_seed"])
    model = trainer.build_model(trainer_args(VARIANTS[variant]), tokenizer)
    # Trainer resets train RNG only after model construction. It does not change
    # initial weights, but recording this mirrors the seed path exactly.
    reset_all_rng(torch, RECIPE["train_rng_seed"])
    model.eval()
    return model


def tensor_group(key: str) -> str:
    if key.startswith("deberta.embeddings"):
        return "embeddings"
    if "attention.self.query_proj" in key or "attention.self.key_proj" in key or "attention.self.value_proj" in key:
        return "attention_qkv"
    if "pos_key_proj" in key or "pos_query_proj" in key:
        return "removed_pos_projection"
    if "attention.output" in key:
        return "attention_output_after_optional"
    if ".intermediate." in key:
        return "intermediate_after_optional"
    if ".output." in key and "attention.output" not in key:
        return "layer_output_after_optional"
    if "rel_embeddings" in key:
        return "encoder_rel_embeddings"
    if key.startswith("cls."):
        return "mlm_head_after_optional"
    return "other"


def compare_state_dicts(torch, left, right) -> dict[str, Any]:
    ls = left.state_dict()
    rs = right.state_dict()
    common = sorted(set(ls) & set(rs))
    only_left = sorted(set(ls) - set(rs))
    only_right = sorted(set(rs) - set(ls))
    exact_common = 0
    close_common = 0
    mismatched = []
    groups: dict[str, dict[str, Any]] = {}
    for k in common:
        lt = ls[k]
        rt = rs[k]
        g = tensor_group(k)
        groups.setdefault(g, {"count": 0, "exact": 0, "allclose": 0, "max_abs_values": [], "mean_abs_values": [], "numel": 0})
        rec = groups[g]
        rec["count"] += 1
        rec["numel"] += int(lt.numel())
        exact = bool(torch.equal(lt, rt))
        close = bool(torch.allclose(lt, rt, rtol=0.0, atol=0.0))
        if exact:
            exact_common += 1
            rec["exact"] += 1
        if close:
            close_common += 1
            rec["allclose"] += 1
        if not exact:
            diff = (lt - rt).abs().float()
            max_abs = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
            rec["max_abs_values"].append(max_abs)
            rec["mean_abs_values"].append(mean_abs)
            mismatched.append({
                "key": k,
                "group": g,
                "shape": list(lt.shape),
                "numel": int(lt.numel()),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
            })
    for g, rec in groups.items():
        rec["n_mismatched"] = rec["count"] - rec["exact"]
        rec["max_abs_max"] = max(rec.pop("max_abs_values"), default=0.0)
        means = rec.pop("mean_abs_values")
        rec["mean_abs_mean_over_tensors"] = float(statistics.mean(means)) if means else 0.0
    return {
        "left_n_keys": len(ls),
        "right_n_keys": len(rs),
        "common_key_count": len(common),
        "only_left_count": len(only_left),
        "only_right_count": len(only_right),
        "only_left_sample": only_left[:24],
        "only_right_sample": only_right[:24],
        "exact_common_count": exact_common,
        "nonexact_common_count": len(common) - exact_common,
        "allclose_zero_atol_count": close_common,
        "groups": groups,
        "first_mismatches": mismatched[:80],
        "last_mismatches": mismatched[-20:],
    }


def copy_common_tensors(torch, source, target) -> dict[str, Any]:
    ss = source.state_dict()
    ts = target.state_dict()
    copied = 0
    skipped_shape = []
    new_state = {}
    for k, v in ts.items():
        if k in ss and tuple(ss[k].shape) == tuple(v.shape):
            new_state[k] = ss[k].detach().clone()
            copied += 1
        else:
            new_state[k] = v
            if k in ss:
                skipped_shape.append(k)
    target.load_state_dict(new_state, strict=True)
    return {"copied_common_tensors": copied, "target_total_tensors": len(ts), "skipped_shape_mismatch": skipped_shape}


def read_sample_texts(n: int) -> list[str]:
    out = []
    with STREAM_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj.get("text", ""))
            if text:
                out.append(text)
            if len(out) >= n:
                break
    return out


def logits_compare(torch, tokenizer, models: dict[str, Any], texts: list[str]) -> dict[str, Any]:
    enc = tokenizer(texts, truncation=True, max_length=64, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = {name: model(**enc).logits.float() for name, model in models.items()}
    names = sorted(logits)
    out: dict[str, Any] = {"n_texts": len(texts), "shape": list(next(iter(logits.values())).shape), "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = (logits[a] - logits[b]).abs()
            out["pairs"][f"{a}__vs__{b}"] = {
                "max_abs_logit_diff": float(diff.max().item()),
                "mean_abs_logit_diff": float(diff.mean().item()),
                "rms_logit_diff": float(torch.sqrt((diff ** 2).mean()).item()),
            }
    return out


def write_md(payload: dict[str, Any], path: pathlib.Path) -> None:
    lines = []
    lines.append("# research DeBERTa positional-ablation initialization parity probe")
    lines.append("")
    lines.append(f"Created UTC: {payload['created_utc']}")
    lines.append("")
    lines.append("## Main finding")
    lines.append("")
    lines.append(payload["interpretation"])
    lines.append("")
    lines.append("## Same-seed tensor parity")
    for name, rec in payload["same_seed_comparisons"].items():
        lines.append(f"- `{name}`: common={rec['common_key_count']}, exact={rec['exact_common_count']}, nonexact={rec['nonexact_common_count']}, only_left={rec['only_left_count']}, only_right={rec['only_right_count']}")
        for g, gr in rec["groups"].items():
            lines.append(f"  - {g}: {gr['exact']}/{gr['count']} exact, mismatched={gr['n_mismatched']}, max_abs_max={gr['max_abs_max']:.6g}, mean_abs_mean={gr['mean_abs_mean_over_tensors']:.6g}")
    lines.append("")
    lines.append("## Common-copied repair object")
    lines.append("")
    lines.append(f"- copied common tensors into no-disentangle target: {payload['common_copied_no_disentangle']['copy_record']}")
    cc = payload["common_copied_no_disentangle"]["full_vs_copied_nodis_state"]
    lines.append(f"- after copy, common exact={cc['exact_common_count']}/{cc['common_key_count']}; remaining nonexact should be zero if copy worked.")
    lines.append("")
    lines.append("## Initialization logits on sample compact-stream texts")
    for pair, rec in payload["initial_logits"].get("pairs", {}).items():
        lines.append(f"- {pair}: mean_abs={rec['mean_abs_logit_diff']:.6g}, rms={rec['rms_logit_diff']:.6g}, max_abs={rec['max_abs_logit_diff']:.6g}")
    lines.append("")
    lines.append("## Boundary")
    lines.append("CPU-only fresh initialization probe; no training, selected official-compatible evaluation, SuperGLUE, AoA, upload, or leaderboard submission.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(WORKSPACE / "data" / "initialization_parity_probe"))
    ap.add_argument("--sample-texts", type=int, default=8)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

    trainer = load_trainer_module()
    tok_sha = sha256_file(TOKENIZER_DIR / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer SHA mismatch: {tok_sha}")
    tokenizer = trainer.make_portable_tokenizer(str(TOKENIZER_DIR))

    models = {v: build_variant(trainer, torch, tokenizer, v) for v in VARIANTS}
    same_seed_comparisons = {
        "full_vs_no_disentangle_same_seed": compare_state_dicts(torch, models["full_p2c_c2p_abs"], models["no_disentangle_abs"]),
        "full_vs_c2p_only_same_seed": compare_state_dicts(torch, models["full_p2c_c2p_abs"], models["c2p_only_abs"]),
        "full_vs_p2c_only_same_seed": compare_state_dicts(torch, models["full_p2c_c2p_abs"], models["p2c_only_abs"]),
        "c2p_only_vs_p2c_only_same_seed": compare_state_dicts(torch, models["c2p_only_abs"], models["p2c_only_abs"]),
    }

    copied_nodis = build_variant(trainer, torch, tokenizer, "no_disentangle_abs")
    copy_record = copy_common_tensors(torch, models["full_p2c_c2p_abs"], copied_nodis)
    copied_compare = compare_state_dicts(torch, models["full_p2c_c2p_abs"], copied_nodis)

    texts = read_sample_texts(args.sample_texts)
    initial_logits = logits_compare(torch, tokenizer, {
        "full_same_seed": models["full_p2c_c2p_abs"],
        "nodis_same_seed": models["no_disentangle_abs"],
        "nodis_common_copied_from_full": copied_nodis,
    }, texts)

    nonexact = same_seed_comparisons["full_vs_no_disentangle_same_seed"]["nonexact_common_count"]
    common = same_seed_comparisons["full_vs_no_disentangle_same_seed"]["common_key_count"]
    if nonexact > 0:
        interpretation = (
            "Same numeric seeds do not make the full and no-disentangle variants common-initialized: "
            f"{nonexact}/{common} same-named tensors differ before training. Therefore the running research four-cell result remains a valid architecture-coordinate data-treatment interaction, but attenuation or survival must be interpreted as including architecture-dependent initialization stream changes, not as a perfectly common-weight ablation of only c2p/p2c score terms. A cleaner follow-up, if the research result is scientifically important but ambiguous, is to train no-disentangle compact/repeat models initialized by copying all common tensors from the same full DeBERTa initialization while omitting the positional projection tensors."
        )
    else:
        interpretation = (
            "Same numeric seeds exactly match every common tensor across full and no-disentangle variants at initialization, so the research contrast is clean with respect to common-tensor initialization."
        )

    payload = {
        "status": "INITIALIZATION_PARITY_PROBE",
        "created_utc": now(),
        "trainer_path": str(TRAINER_PATH.relative_to(USER_ROOT)),
        "tokenizer_dir": str(TOKENIZER_DIR.relative_to(USER_ROOT)),
        "tokenizer_json_sha256": tok_sha,
        "recipe": RECIPE,
        "variants": VARIANTS,
        "same_seed_comparisons": same_seed_comparisons,
        "common_copied_no_disentangle": {
            "copy_record": copy_record,
            "full_vs_copied_nodis_state": copied_compare,
        },
        "sample_texts": texts,
        "initial_logits": initial_logits,
        "interpretation": interpretation,
        "boundary": "CPU-only fresh initialization probe; no training, selected official-compatible evaluation, SuperGLUE, AoA, upload, or leaderboard submission.",
    }

    json_path = out_dir / "initialization_parity_probe.json"
    md_path = out_dir / "initialization_parity_probe.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(payload, md_path)

    def display_path(p: pathlib.Path) -> str:
        p_abs = p if p.is_absolute() else USER_ROOT / p
        p_abs = p_abs.resolve()
        try:
            return str(p_abs.relative_to(USER_ROOT))
        except ValueError:
            return str(p_abs)

    print(json.dumps({"status": payload["status"], "json": display_path(json_path), "md": display_path(md_path), "full_vs_nodis_common": common, "full_vs_nodis_nonexact": nonexact}, indent=2))


if __name__ == "__main__":
    main()
