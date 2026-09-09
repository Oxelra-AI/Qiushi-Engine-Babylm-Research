#!/usr/bin/env python3
"""research: gradient-complementarity probe for MLM-primary same-corruption MNTP.

Scientific purpose
------------------
research showed that replacing whole MLM batches with uncorrupted causal next-token
batches trades away clean-Qwen's bidirectional reconstruction strengths. This probe
tests the sharper hypothesis from research: keep the full MLM stream and ask whether
an auxiliary *same-corruption* masked-next-token prediction (MNTP) loss has stable
positive/near-orthogonal gradient complementarity with the ordinary MLM loss.

The probe reads only legal training data/checkpoints and uses no evaluation item,
AoA/CDI word, leaderboard score, or endpoint-selection signal. It does not train a
model. It computes gradients of two losses on the same masked input:

  MLM:        logits at masked position j predict original token x_j.
  MNTP-token: logits at position j-1 predict original token x_j when j was masked.
  MNTP-word:  same, but only for the first subtoken of each selected word.

The second/third losses reuse exactly the same WWM corruption as the MLM batch.
Thus a possible future trainer can retain every MLM update and add a weighted or
conflict-handled directional component on the same legally counted word exposure.
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
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import DebertaV2Config, DebertaV2ForMaskedLM

import masking_curriculum_trainer as base

ROOT = _public_path('experiments/archive/compact_experience')
DEFAULT_CORPUS = _public_path('experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl')
DEFAULT_TOKENIZER = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M')
DEFAULT_REFERENCE = _public_path('experiments/archive/compact_experience/training/runs/qwen_clean_aligned_16k_seed43022/hf_model/chck_100M')
DEFAULT_OUT = _public_path('experiments/archive/compact_experience/data/gradient_complementarity/mlm_mntp_gradient_probe.json')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_checkpoint_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, value = spec.split("=", 1)
        return label.strip(), value.strip()
    value = spec.strip()
    label = "init" if value == "init" else Path(value).name
    return label, value


def load_sample_examples(path: Path, *, offset_rows: int, num_rows: int, stride: int) -> list[base.Example]:
    examples: list[base.Example] = []
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            if line_no < offset_rows:
                continue
            if ((line_no - offset_rows) % stride) != 0:
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"word-count mismatch at line {line_no + 1}: field={words} actual={actual}")
            ex_id = int(obj.get("example_id", line_no))
            source = str(obj.get("source", "example_jsonl"))
            examples.append(base.Example(text=text, words=words, example_id=ex_id, source=source))
            seen += 1
            if seen >= num_rows:
                break
    if len(examples) < num_rows:
        raise RuntimeError(f"requested {num_rows} rows from {path}, got {len(examples)}")
    return examples


def build_init_model(reference: Path, seed: int, device: torch.device) -> DebertaV2ForMaskedLM:
    cfg = DebertaV2Config.from_pretrained(reference)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DebertaV2ForMaskedLM(cfg)
    model.to(device)
    return model


def load_model(value: str, reference: Path, init_seed: int, device: torch.device) -> tuple[DebertaV2ForMaskedLM, dict]:
    if value == "init":
        model = build_init_model(reference, init_seed, device)
        identity = {"kind": "fresh_init", "reference_config": str(reference), "init_seed": init_seed}
    else:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path if not str(path).startswith("Sessions/") else Path(value)
        model = DebertaV2ForMaskedLM.from_pretrained(path)
        model.to(device)
        identity = {"kind": "checkpoint", "path": str(path)}
        st = path / "model.safetensors"
        if st.exists():
            identity["model_safetensors_sha256"] = sha256_file(st)
    with torch.no_grad():
        identity["word_embedding_sha256"] = hashlib.sha256(
            model.deberta.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()
        ).hexdigest()
        identity["parameter_count"] = sum(p.numel() for p in model.parameters())
    return model, identity


def make_mntp_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    word_group: torch.Tensor,
    mlm_labels: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    """Return labels at predictor position j-1 for targets x_j selected by MLM."""
    labels = torch.full_like(input_ids, -100)
    target_selected = mlm_labels != -100
    valid_pair = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    selected_next = target_selected[:, 1:] & valid_pair
    if variant == "word_start":
        wg = word_group
        current = wg[:, 1:]
        prev = wg[:, :-1]
        # first valid subtoken of a word: group changes between previous token and current token.
        # This also handles words at position 1; position 0 has no predictor and is excluded.
        starts = (current >= 0) & (current != prev)
        selected_next = selected_next & starts
    elif variant != "token_shift":
        raise ValueError(f"unknown MNTP variant: {variant}")
    labels[:, :-1] = torch.where(selected_next, input_ids[:, 1:], labels[:, :-1])
    return labels


def loss_from_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def group_name(param_name: str) -> str:
    if param_name.startswith("deberta.embeddings"):
        return "embeddings"
    if param_name.startswith("cls"):
        return "lm_head"
    marker = "deberta.encoder.layer."
    if param_name.startswith(marker):
        rest = param_name[len(marker):]
        layer = rest.split(".", 1)[0]
        if layer.isdigit():
            return f"layer_{int(layer):02d}"
    return "other"


def grad_pair_stats(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    named_params: list[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph_after: bool = False,
) -> dict:
    params = [p for _, p in named_params if p.requires_grad]
    names = [n for n, p in named_params if p.requires_grad]
    # loss_a and loss_b share the same forward graph. Retain through the first
    # gradient, and optionally through the second when another auxiliary
    # comparison from the same logits will follow.
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=retain_graph_after, allow_unused=True)
    acc: dict[str, dict[str, float]] = defaultdict(lambda: {"dot": 0.0, "norm_a_sq": 0.0, "norm_b_sq": 0.0, "params": 0})
    for name, xa, xb in zip(names, ga, gb):
        if xa is None or xb is None:
            continue
        grp = group_name(name)
        da = xa.detach().float().view(-1)
        db = xb.detach().float().view(-1)
        acc[grp]["dot"] += float(torch.dot(da, db).cpu())
        acc[grp]["norm_a_sq"] += float(torch.dot(da, da).cpu())
        acc[grp]["norm_b_sq"] += float(torch.dot(db, db).cpu())
        acc[grp]["params"] += int(da.numel())
        acc["all"]["dot"] += float(torch.dot(da, db).cpu())
        acc["all"]["norm_a_sq"] += float(torch.dot(da, da).cpu())
        acc["all"]["norm_b_sq"] += float(torch.dot(db, db).cpu())
        acc["all"]["params"] += int(da.numel())
    out = {}
    for grp, v in sorted(acc.items()):
        na = math.sqrt(max(v["norm_a_sq"], 0.0))
        nb = math.sqrt(max(v["norm_b_sq"], 0.0))
        denom = na * nb
        out[grp] = {
            "dot": v["dot"],
            "norm_mlm": na,
            "norm_aux": nb,
            "cosine": (v["dot"] / denom) if denom > 0 else None,
            "aux_over_mlm_norm": (nb / na) if na > 0 else None,
            "negative_dot": bool(v["dot"] < 0),
            "params": v["params"],
        }
    return out


def summarize_batches(rows: list[dict], variant: str) -> dict:
    valid = [r for r in rows if r.get("variant") == variant and r.get("valid")]
    if not valid:
        return {"valid_batches": 0}
    groups = sorted(valid[0]["grad_stats"].keys())
    summary = {"valid_batches": len(valid), "mean_losses": {}, "groups": {}}
    summary["mean_losses"] = {
        "mlm_loss": float(np.mean([r["mlm_loss"] for r in valid])),
        "aux_loss": float(np.mean([r["aux_loss"] for r in valid])),
        "mlm_targets": float(np.mean([r["mlm_targets"] for r in valid])),
        "aux_targets": float(np.mean([r["aux_targets"] for r in valid])),
    }
    for g in groups:
        cos = [r["grad_stats"][g]["cosine"] for r in valid if r["grad_stats"][g]["cosine"] is not None]
        ratio = [r["grad_stats"][g]["aux_over_mlm_norm"] for r in valid if r["grad_stats"][g]["aux_over_mlm_norm"] is not None]
        neg = [r["grad_stats"][g]["negative_dot"] for r in valid]
        summary["groups"][g] = {
            "mean_cosine": float(np.mean(cos)) if cos else None,
            "median_cosine": float(np.median(cos)) if cos else None,
            "min_cosine": float(np.min(cos)) if cos else None,
            "max_cosine": float(np.max(cos)) if cos else None,
            "negative_fraction": float(np.mean(neg)) if neg else None,
            "mean_aux_over_mlm_norm": float(np.mean(ratio)) if ratio else None,
        }
    all_cos = summary["groups"].get("all", {}).get("mean_cosine")
    all_neg = summary["groups"].get("all", {}).get("negative_fraction")
    emb_cos = summary["groups"].get("embeddings", {}).get("mean_cosine")
    summary["interpretive_flags"] = {
        "stable_positive_full_gradient": bool(all_cos is not None and all_cos > 0.05 and (all_neg is not None and all_neg <= 0.25)),
        "near_orthogonal_full_gradient": bool(all_cos is not None and -0.05 <= all_cos <= 0.05),
        "full_gradient_conflict": bool(all_neg is not None and all_neg >= 0.5),
        "embedding_conflict": bool(emb_cos is not None and emb_cos < 0.0),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--reference_checkpoint", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="Checkpoint spec label=path or label=init. Can repeat.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--offset_rows", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seq_length", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--mask_seed", type=int, default=53053)
    ap.add_argument("--init_seed", type=int, default=43022)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train_mode", action="store_true", help="Use train mode/dropout; default eval mode for deterministic geometry.")
    args = ap.parse_args()

    t0 = time.time()
    device = torch.device(args.device)
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    n_rows = args.num_batches * args.batch_size
    examples = load_sample_examples(corpus, offset_rows=args.offset_rows, num_rows=n_rows, stride=args.stride)
    dataset = base.MaskedChunkDataset(examples, tokenizer, args.seq_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=0)

    curriculum = base.MaskingCurriculumState(
        curriculum="wwm_fixed", mask_prob_start=args.mask_prob, mask_prob_end=args.mask_prob
    )
    curriculum.initialize(len(tokenizer), max(1, args.num_batches))

    checkpoint_specs = [parse_checkpoint_spec(x) for x in args.checkpoint]
    manifest = {
        "status": "MLM_MNTP_GRADIENT_PROBE",
        "created_utc_unix": time.time(),
        "scientific_scope": "Gradient relation between ordinary WWM-MLM and same-corruption MNTP auxiliary on legal clean-Qwen training batches; no eval/AoA/CDI/leaderboard signal used.",
        "corpus": str(corpus),
        "corpus_sha256": sha256_file(corpus),
        "tokenizer_path": args.tokenizer_path,
        "reference_checkpoint": args.reference_checkpoint,
        "offset_rows": args.offset_rows,
        "stride": args.stride,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "mask_prob": args.mask_prob,
        "mask_seed": args.mask_seed,
        "init_seed": args.init_seed,
        "device": str(device),
        "model_mode": "train" if args.train_mode else "eval",
        "checkpoint_specs": [{"label": a, "value": b} for a, b in checkpoint_specs],
        "results": [],
    }

    for label, value in checkpoint_specs:
        model, identity = load_model(value, Path(args.reference_checkpoint), args.init_seed, device)
        model.train(mode=bool(args.train_mode))
        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        mask_gen = torch.Generator(device=device)
        mask_gen.manual_seed(args.mask_seed)
        batch_rows: list[dict] = []
        for batch_idx, batch in enumerate(loader):
            # Recompute exact same mask sequence for every checkpoint label.
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            wg = batch["word_group"].to(device)
            curriculum.current_step = batch_idx
            masked, mlm_labels = base.apply_masking_curriculum(ids, am, wg, tokenizer, curriculum, mask_gen)
            with torch.set_grad_enabled(True):
                logits = model(input_ids=masked, attention_mask=am).logits
                mlm_loss = loss_from_labels(logits, mlm_labels)
                variants = ("token_shift", "word_start")
                for variant_i, variant in enumerate(variants):
                    aux_labels = make_mntp_labels(ids, am, wg, mlm_labels, variant)
                    aux_targets = int((aux_labels != -100).sum().detach().cpu())
                    rec = {
                        "batch_index": batch_idx,
                        "variant": variant,
                        "mlm_targets": int((mlm_labels != -100).sum().detach().cpu()),
                        "aux_targets": aux_targets,
                        "batch_words": int(batch["words"].sum().item()),
                    }
                    if aux_targets <= 0:
                        rec["valid"] = False
                        batch_rows.append(rec)
                        continue
                    aux_loss = loss_from_labels(logits, aux_labels)
                    rec.update({
                        "valid": True,
                        "mlm_loss": float(mlm_loss.detach().cpu()),
                        "aux_loss": float(aux_loss.detach().cpu()),
                        "grad_stats": grad_pair_stats(
                            mlm_loss, aux_loss, named_params,
                            retain_graph_after=(variant_i < len(variants) - 1),
                        ),
                    })
                    batch_rows.append(rec)
                    # Free gradients held by autograd.grad outputs; no optimizer grad is used.
                    model.zero_grad(set_to_none=True)
        ckpt_result = {
            "label": label,
            "value": value,
            "identity": identity,
            "batch_rows": batch_rows,
            "summary_by_variant": {
                "token_shift": summarize_batches(batch_rows, "token_shift"),
                "word_start": summarize_batches(batch_rows, "word_start"),
            },
        }
        manifest["results"].append(ckpt_result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest["elapsed_sec"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "out": str(out),
        "elapsed_sec": manifest["elapsed_sec"],
        "summaries": [
            {"label": r["label"], "summary_by_variant": r["summary_by_variant"]}
            for r in manifest["results"]
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
