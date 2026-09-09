#!/usr/bin/env python3
"""research: generate a trainer variant that accepts a static token mask prior.

The generated trainer is a conservative source patch of the frozen inherited
masking trainer.  It adds one curriculum mode, `wwm_static_prior`, and one CLI
flag, `--static_prior_json`.  Under that mode, WWM selection probabilities are
formed from token-ID multipliers, averaged within each WWM group, and then
renormalized per example by WWM-group token lengths so the expected number of
masked candidate tokens remains matched to the nominal mask probability.  Data order, corpus, tokenizer,
architecture, optimizer, batch geometry, seeds, and exposure accounting are
unchanged.

This generator only writes and statically validates the source; it does not
train a model.  A future expensive launch must first be justified by the mature
matched legal-tokenizer clean-control comparison.
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
import subprocess
import sys
import time

from transformers import AutoTokenizer


def find_user_root() -> pathlib.Path:
    return _PUBLIC_ROOT


USER_ROOT = find_user_root()
WORKSPACE = USER_ROOT / "experiments/archive/frontier_consolidation"
BASE_TRAINER = USER_ROOT / "experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"
DEFAULT_PRIOR = WORKSPACE / "data/static_token_mask_prior/static_token_mask_prior.json"
DEFAULT_OUT = WORKSPACE / "scripts/masking_curriculum_trainer_static_prior.py"
DEFAULT_VALIDATION = WORKSPACE / "data/static_prior_trainer_validation"
EXPECTED_POOL_SHA = "215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23"
EXPECTED_TOKENIZER_SHA = "91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9"
EXPECTED_100M_SHA = "3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691"
EXPECTED_PRIOR_SCHEMES = {"relation_only_v1", "relation_info_v1"}


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def replace_once(src: str, old: str, new: str) -> str:
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"expected exactly one match, got {n}: {old[:120]!r}")
    return src.replace(old, new)


def build_source(base: pathlib.Path) -> str:
    src = base.read_text(encoding="utf-8")

    src = replace_once(
        src,
        "  amlm_hard       - WWM + per-token difficulty weighting + decay\n  amlm_hard_switch - AMLM-hard with WWM→token switch at switch_frac\n",
        "  amlm_hard       - WWM + per-token difficulty weighting + decay\n  amlm_hard_switch - AMLM-hard with WWM→token switch at switch_frac\n  wwm_static_prior - fixed WWM with corpus-derived static token/group prior\n",
    )
    src = replace_once(
        src,
        "  \"decay_wwm_to_token\", \"amlm_hard\", \"amlm_hard_switch\",\n",
        "  \"decay_wwm_to_token\", \"amlm_hard\", \"amlm_hard_switch\",\n  \"wwm_static_prior\",\n",
    )
    src = replace_once(
        src,
        "    amlm_lambda: float = 0.2  # exponential smoothing for AMLM weights\n",
        "    amlm_lambda: float = 0.2  # exponential smoothing for AMLM weights\n    static_prior_path: str = \"\"\n    static_prior_scheme: str = \"\"\n",
    )
    src = replace_once(
        src,
        "    # Per-token difficulty weights (AMLM-hard)\n    token_weights: torch.Tensor = field(default=None, repr=False)\n",
        "    # Static legal-tokenizer prior weights; path/scheme are configuration\n    # fields above, while this tensor is loaded in initialize().\n    static_prior_weights: torch.Tensor = field(default=None, repr=False)\n\n    # Per-token difficulty weights (AMLM-hard)\n    token_weights: torch.Tensor = field(default=None, repr=False)\n",
    )
    # The inherited file uses a dataclass with a self.uses_amlm guarded block in
    # initialize.  Match that exact indentation; failure here forces a conservative
    # repair rather than silently writing a broken trainer.
    src = replace_once(
        src,
        "        self.vocab_size = vocab_size\n        self.total_steps = total_steps\n        if self.uses_amlm:\n            self.token_weights = torch.ones(vocab_size) * self.mask_prob_start\n            self.correct_counts = torch.zeros(vocab_size, dtype=torch.long)\n            self.total_counts = torch.zeros(vocab_size, dtype=torch.long)\n",
        "        self.vocab_size = vocab_size\n        self.total_steps = total_steps\n        if self.uses_amlm:\n            self.token_weights = torch.ones(vocab_size) * self.mask_prob_start\n            self.correct_counts = torch.zeros(vocab_size, dtype=torch.long)\n            self.total_counts = torch.zeros(vocab_size, dtype=torch.long)\n        if self.curriculum == \"wwm_static_prior\":\n            if not self.static_prior_path:\n                raise RuntimeError(\"wwm_static_prior requires --static_prior_json\")\n            payload = json.loads(Path(self.static_prior_path).read_text(encoding=\"utf-8\"))\n            scheme = self.static_prior_scheme or \"relation_only_v1\"\n            weights = payload.get(\"schemes\", {}).get(scheme, {}).get(\"weights\")\n            if not isinstance(weights, list) or len(weights) != vocab_size:\n                raise RuntimeError(f\"static prior scheme {scheme!r} missing or has wrong vocab size in {self.static_prior_path}\")\n            self.static_prior_weights = torch.tensor(weights, dtype=torch.float32)\n",
    )
    src = replace_once(
        src,
        "    per_token_probs = state.get_per_token_probs(input_ids)  # None if not AMLM\n",
        "    per_token_probs = state.get_per_token_probs(input_ids)  # None if not AMLM\n    static_group_probs = None\n    if state.curriculum == \"wwm_static_prior\":\n        if state.static_prior_weights is None:\n            raise RuntimeError(\"static prior weights were not initialized\")\n        static_group_probs = state.static_prior_weights.to(input_ids.device)\n",
    )
    src = replace_once(
        src,
        "            if per_token_probs is not None:\n                # For AMLM+WWM: use mean token prob within each group\n                group_probs = torch.zeros(valid_groups.max().item() + 1, device=device)\n                for gid in valid_groups:\n                    g_mask = (groups == gid) & candidate[b]\n                    if g_mask.any():\n                        group_probs[gid] = per_token_probs[b][g_mask].mean()\n                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)\n                chosen = valid_groups[gp < group_probs[valid_groups]]\n            else:\n                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)\n                chosen = valid_groups[gp < mask_prob]\n",
        "            if static_group_probs is not None:\n                # Static legal-tokenizer prior: mean token weight within each WWM\n                # group, renormalized per example by group token lengths so the\n                # expected number of masked candidate tokens, not only groups,\n                # remains matched to the nominal mask budget.\n                group_weights = torch.zeros(valid_groups.max().item() + 1, device=device)\n                group_token_counts = torch.zeros(valid_groups.max().item() + 1, device=device)\n                for gid in valid_groups:\n                    g_mask = (groups == gid) & candidate[b]\n                    if g_mask.any():\n                        group_weights[gid] = static_group_probs[input_ids[b][g_mask]].mean()\n                        group_token_counts[gid] = g_mask.sum().float()\n                raw = group_weights[valid_groups]\n                lengths = group_token_counts[valid_groups].clamp_min(1.0)\n                token_weighted_mean = ((raw * lengths).sum() / lengths.sum().clamp_min(1.0)).clamp_min(1e-8)\n                group_probs = (raw / token_weighted_mean * mask_prob).clamp(0.0, 0.95)\n                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)\n                chosen = valid_groups[gp < group_probs]\n            elif per_token_probs is not None:\n                # For AMLM+WWM: use mean token prob within each group\n                group_probs = torch.zeros(valid_groups.max().item() + 1, device=device)\n                for gid in valid_groups:\n                    g_mask = (groups == gid) & candidate[b]\n                    if g_mask.any():\n                        group_probs[gid] = per_token_probs[b][g_mask].mean()\n                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)\n                chosen = valid_groups[gp < group_probs[valid_groups]]\n            else:\n                gp = torch.rand(valid_groups.numel(), generator=gen, device=device)\n                chosen = valid_groups[gp < mask_prob]\n",
    )
    src = replace_once(
        src,
        "    p.add_argument(\"--amlm_lambda\", type=float, default=0.2,\n                   help=\"Exponential smoothing for AMLM weight updates\")\n",
        "    p.add_argument(\"--amlm_lambda\", type=float, default=0.2,\n                   help=\"Exponential smoothing for AMLM weight updates\")\n    p.add_argument(\"--static_prior_json\", default=\"\",\n                   help=\"Static token mask prior JSON for wwm_static_prior\")\n    p.add_argument(\"--static_prior_scheme\", default=\"relation_only_v1\",\n                   help=\"Scheme name inside the static prior JSON\")\n",
    )
    src = replace_once(
        src,
        "        amlm_lambda=args.amlm_lambda,\n    )\n",
        "        amlm_lambda=args.amlm_lambda,\n        static_prior_path=args.static_prior_json,\n        static_prior_scheme=args.static_prior_scheme,\n    )\n",
    )
    src = replace_once(
        src,
        "        \"amlm_window\": args.amlm_window if curriculum_state.uses_amlm else None,\n",
        "        \"amlm_window\": args.amlm_window if curriculum_state.uses_amlm else None,\n        \"static_prior_json\": args.static_prior_json if curriculum_state.curriculum == \"wwm_static_prior\" else None,\n        \"static_prior_scheme\": args.static_prior_scheme if curriculum_state.curriculum == \"wwm_static_prior\" else None,\n",
    )
    src = replace_once(
        src,
        "            n_candidate = int((attention_mask.bool()).sum().item())\n            effective_mask_rate = n_pred / max(1, n_candidate)\n",
        "            candidate_for_rate = attention_mask.bool() & ~torch.isin(input_ids, torch.tensor(sorted(tokenizer.all_special_ids), device=input_ids.device))\n            n_candidate = int(candidate_for_rate.sum().item())\n            effective_mask_rate = n_pred / max(1, n_candidate)\n",
    )
    return src


def validate_prior(prior_path: pathlib.Path, tokenizer_dir: pathlib.Path) -> dict[str, object]:
    tok_sha = sha256_file(tokenizer_dir / "tokenizer.json")
    if tok_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"tokenizer SHA mismatch {tok_sha}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    payload = json.loads(prior_path.read_text(encoding="utf-8"))
    if payload.get("inputs", {}).get("pool_sha256") != EXPECTED_POOL_SHA:
        raise RuntimeError("prior was not built from the frozen legal reinvest 10M pool")
    schemes = payload.get("schemes") or {}
    if set(schemes) != EXPECTED_PRIOR_SCHEMES:
        raise RuntimeError(f"unexpected prior schemes {sorted(schemes)}")
    for name, rec in schemes.items():
        weights = rec.get("weights")
        if not isinstance(weights, list) or len(weights) != len(tokenizer):
            raise RuntimeError(f"scheme {name} has invalid weights")
        if rec.get("occurrence_weighted_mean") is None:
            raise RuntimeError(f"scheme {name} missing occurrence mean")
    return {
        "tokenizer_sha256": tok_sha,
        "vocab_size": len(tokenizer),
        "schemes": sorted(schemes),
        "prior_sha256": sha256_file(prior_path),
    }


def run_unit_validation(trainer_path: pathlib.Path, prior_path: pathlib.Path, out_dir: pathlib.Path) -> dict[str, object]:
    code = r'''
import json, pathlib, sys, torch
from transformers import AutoTokenizer
trainer_path = pathlib.Path(sys.argv[1])
prior_path = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])
import importlib.util
spec = importlib.util.spec_from_file_location("trainer_static_prior", trainer_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
tok = AutoTokenizer.from_pretrained(str(prior_path.parent.parent / "compliant_tokenizer"), use_fast=True)
state = mod.MaskingCurriculumState(
    curriculum="wwm_static_prior", mask_prob_start=0.15, mask_prob_end=0.15,
    static_prior_path=str(prior_path), static_prior_scheme="relation_only_v1",
)
state.initialize(vocab_size=len(tok), total_steps=10)
texts = [
    "The cup is behind the box and moves across the table because the child pushes it.",
    "A simple ordinary sentence with common words stays available for masking.",
]
ds = mod.MaskedChunkDataset([mod.Example(text=t, words=len(t.split()), example_id=i, source="unit") for i, t in enumerate(texts)], tok, 64)
batch = mod.collate([ds[0], ds[1]])
input_ids = batch["input_ids"]
attention = batch["attention_mask"]
word_group = batch["word_group"]
gen = torch.Generator(device="cpu")
gen.manual_seed(43023)
masked, labels = mod.apply_masking_curriculum(input_ids, attention, word_group, tok, state, gen)
selected = (labels != -100)
candidates = attention.bool() & ~torch.isin(input_ids, torch.tensor(sorted(tok.all_special_ids)))
weights = state.static_prior_weights
sel_ids = input_ids[selected]
result = {
    "status": "STATIC_PRIOR_UNIT_OK",
    "selected_tokens": int(selected.sum().item()),
    "candidate_tokens": int(candidates.sum().item()),
    "selected_effective_rate": float(selected.sum().item() / max(1, candidates.sum().item())),
    "selected_token_strings": [tok.convert_ids_to_tokens(int(x)) for x in sel_ids[:40]],
    "selected_mean_prior": float(weights[sel_ids].mean().item()) if sel_ids.numel() else None,
    "candidate_mean_prior": float(weights[input_ids[candidates]].mean().item()),
}
out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
'''
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "static_prior_unit_validation.json"
    proc = subprocess.run(
        [sys.executable, "-c", code, str(trainer_path), str(prior_path), str(out_json)],
        cwd=str(USER_ROOT), text=True, capture_output=True,
    )
    rec = {
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "out_json": str(out_json),
    }
    if proc.returncode != 0:
        raise RuntimeError(f"unit validation failed: {rec}")
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    if payload["selected_tokens"] <= 0:
        raise RuntimeError(f"unit validation selected no tokens: {payload}")
    if not (0.01 <= payload["selected_effective_rate"] <= 0.95):
        raise RuntimeError(f"unit validation selected implausible rate: {payload}")
    rec["payload"] = payload
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-trainer", default=str(BASE_TRAINER))
    ap.add_argument("--prior-json", default=str(DEFAULT_PRIOR))
    ap.add_argument("--out-trainer", default=str(DEFAULT_OUT))
    ap.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION))
    ap.add_argument("--train-100m", default=str(WORKSPACE / "data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl"))
    ap.add_argument("--tokenizer-dir", default=str(WORKSPACE / "data/compliant_tokenizer"))
    args = ap.parse_args()

    base = pathlib.Path(args.base_trainer)
    prior = pathlib.Path(args.prior_json)
    out = pathlib.Path(args.out_trainer)
    val_dir = pathlib.Path(args.validation_dir)
    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    train_100m = pathlib.Path(args.train_100m)

    prior_info = validate_prior(prior, tokenizer_dir)
    train_sha = sha256_file(train_100m)
    if train_sha != EXPECTED_100M_SHA:
        raise RuntimeError(f"100M train SHA mismatch {train_sha}")

    src = build_source(base)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")
    ast = subprocess.run([sys.executable, "-B", "-c", "import ast, pathlib, sys; ast.parse(pathlib.Path(sys.argv[1]).read_text())", str(out)], text=True, capture_output=True)
    if ast.returncode != 0:
        raise RuntimeError(f"generated trainer AST failed: {ast.stderr}")
    unit = run_unit_validation(out, prior, val_dir)
    payload = {
        "status": "STATIC_PRIOR_TRAINER_GENERATED",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "Prepare but do not launch a learning-signal repair that changes only which WWM groups receive prediction pressure under the legal research tokenizer.",
        "base_trainer": str(base),
        "generated_trainer": str(out),
        "generated_trainer_sha256": sha256_file(out),
        "prior_info": prior_info,
        "train_100m_sha256": train_sha,
        "unit_validation": unit,
        "recipe_preserved": {
            "architecture": "DeBERTa-v2 masked LM 8x480 n_head=8 ffn_mult=4",
            "optimizer": "AdamW lr 0.001 weight_decay 0.01 warmup_fraction 0.06",
            "batch_size": 256,
            "seq_length": 256,
            "mask_prob": 0.15,
            "seeds": "seed=43 extra_init_seed=43022 train_rng_seed=43023",
            "exposure": "existing launcher controls max_word_exposure",
        },
        "launch_guard": [
            "Do not launch from this asset until the mature legal-tokenizer clean-control comparison indicates that learning-signal repair is scientifically warranted.",
            "If launched, use the same frozen corpus, tokenizer, seeds, architecture, optimizer, batch geometry, exposure accounting, and official-compatible evaluation path; the only change should be masking_curriculum=wwm_static_prior plus this prior JSON.",
        ],
    }
    val_dir.mkdir(parents=True, exist_ok=True)
    out_json = val_dir / "static_prior_trainer_validation_summary.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md = val_dir / "static_prior_trainer_validation_summary.md"
    out_md.write_text("\n".join([
        "# research static-prior trainer validation",
        "",
        payload["purpose"],
        "",
        f"- generated trainer: `{out}`",
        f"- generated trainer SHA256: `{payload['generated_trainer_sha256']}`",
        f"- prior JSON: `{prior}`",
        f"- prior SHA256: `{prior_info['prior_sha256']}`",
        f"- frozen 100M train SHA256: `{train_sha}`",
        f"- unit selected tokens: {unit['payload']['selected_tokens']}",
        f"- unit effective candidate rate: {unit['payload']['selected_effective_rate']:.4f}",
        f"- unit selected mean prior: {unit['payload']['selected_mean_prior']:.4f}",
        f"- unit candidate mean prior: {unit['payload']['candidate_mean_prior']:.4f}",
        "",
        "Launch remains prohibited until the mature matched legal-tokenizer clean-control comparison selects a learning-signal repair.",
        f"Full JSON: `{out_json}`",
    ]) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "generated_trainer": str(out),
        "summary_json": str(out_json),
        "summary_md": str(out_md),
        "unit_selected_tokens": unit["payload"]["selected_tokens"],
        "unit_effective_rate": unit["payload"]["selected_effective_rate"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
