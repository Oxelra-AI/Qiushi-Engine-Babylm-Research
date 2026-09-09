#!/usr/bin/env python3
"""research: all-parameter mirrored literal-value pilot.

Mechanism-only separation experiment for the research literal-state result.  The
research pilot trained only the fresh private adapters on a frozen chck_82M slow
path.  This script uses the same rows and the same answer-token CE readout, but
lets the entire chck_82M model update.  If unseen-map shared-context pairs or
full quads appear here while the private-only run stayed near zero, the limiting
factor is the frozen-private continuation channel rather than the row family alone.

This is not a BabyLM submission endpoint: it is a controlled mechanism probe on
a narrow augmented family.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import os
import pathlib
import random
import shutil
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = _public_path('.')
STUDY = _public_path('experiments/archive/relation_learning')
SCRIPTS = _public_path('experiments/archive/relation_learning/scripts')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
TRAIN_DEFAULT = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_train_frame_seen.jsonl')
HELD_DEFAULT = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family_position_var/mirrored_heldout_frame_all.jsonl')
OUT_DEFAULT = _public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot')

CACHE = _public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache')
for sub in ["hf_home", "hf_home/hub", "transformers", "modules", "datasets", "tmp"]:
    (CACHE / sub).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/hf_home'))
os.environ["HF_HUB_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/hf_home/hub'))
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/hf_home/hub'))
os.environ["TRANSFORMERS_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/transformers'))
os.environ["HF_MODULES_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/modules'))
os.environ["HF_DATASETS_CACHE"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/datasets'))
os.environ["TMPDIR"] = str(_public_path('experiments/archive/relation_learning/data/allparam_mirrored_pilot/hf_cache/tmp'))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class BindingDataset(Dataset):
    def __init__(self, rows_path: str | pathlib.Path, tokenizer, max_length: int = 512, seed: int = 95095):
        self.rows: list[dict[str, Any]] = []
        with pathlib.Path(rows_path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_id = tokenizer.mask_token_id
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        enc = self.tokenizer(row["row_text"], truncation=True, max_length=self.max_length,
                             return_offsets_mapping=True, padding=False, add_special_tokens=True)
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        labels = [-100] * len(input_ids)
        ans_s = int(row["answer_char_start"])
        ans_e = int(row["answer_char_end"])
        n_labeled = 0
        for i, (s, e) in enumerate(offsets):
            if s is not None and e is not None and s >= ans_s and e <= ans_e and s < e:
                labels[i] = input_ids[i]
                input_ids[i] = self.mask_id
                n_labeled += 1
        return {"input_ids": input_ids, "labels": labels, "n_tokens": len(input_ids),
                "n_labeled": n_labeled, "answer_role": row["answer_role"], "k": row["k"]}


def collate(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    mx = max(b["n_tokens"] for b in batch)
    ids, labs, mask = [], [], []
    for b in batch:
        n = b["n_tokens"]
        ids.append(b["input_ids"] + [pad_id] * (mx - n))
        labs.append(b["labels"] + [-100] * (mx - n))
        mask.append([1] * n + [0] * (mx - n))
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs), "attention_mask": torch.tensor(mask)}


def evaluate_heldout(model, tokenizer, held_path: str | pathlib.Path, device: torch.device,
                     max_rows: int = 600, frame_filter: str = "f00") -> dict[str, Any]:
    rows = []
    with pathlib.Path(held_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if frame_filter and frame_filter not in str(r.get("frame_id", "")):
                continue
            rows.append(r)
    if max_rows > 0:
        rows = rows[:max_rows]

    mask_id = int(tokenizer.mask_token_id)
    cls_id = int(tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0)
    sep_id = int(tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0)
    correct_raw = 0
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    by_k: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0, "upd_n": 0, "upd_c": 0, "ret_n": 0, "ret_c": 0})
    pair_key: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for row in rows:
            prefix = row["context_text"] + " " + row["query_prefix"]
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            best_cand = None
            best_lp = float("-inf")
            for cand_val in row["all_present_values"]:
                cand_ids = tokenizer.encode(str(cand_val), add_special_tokens=False)
                if not cand_ids:
                    continue
                avail = 510 - len(cand_ids)
                trunc = prefix_ids[-avail:] if len(prefix_ids) > avail else prefix_ids
                base = [cls_id] + trunc + cand_ids + [sep_id]
                cs = len(base) - len(cand_ids) - 1
                batch_ids = []
                for j in range(len(cand_ids)):
                    m = list(base)
                    m[cs + j] = mask_id
                    batch_ids.append(m)
                inp = torch.tensor(batch_ids, dtype=torch.long, device=device)
                out = model(input_ids=inp)
                lp = 0.0
                for j, tid in enumerate(cand_ids):
                    lp += float(F.log_softmax(out.logits[j, cs + j], dim=-1)[int(tid)].detach().cpu())
                if lp > best_lp:
                    best_lp = lp
                    best_cand = str(cand_val)
            ok = best_cand == str(row["answer"])
            correct_raw += int(ok)
            role = str(row["answer_role"])
            by_role[role]["n"] += 1
            by_role[role]["correct"] += int(ok)
            k = int(row["k"])
            by_k[k]["n"] += 1
            by_k[k]["correct"] += int(ok)
            if role == "updated":
                by_k[k]["upd_n"] += 1
                by_k[k]["upd_c"] += int(ok)
            else:
                by_k[k]["ret_n"] += 1
                by_k[k]["ret_c"] += int(ok)
            pair_key[(str(row["quad_id"]), str(row["context_variant"]))][role] = ok
    if was_training:
        model.train()
    n = len(rows)
    pairs = [v for v in pair_key.values() if "updated" in v and "retained" in v]
    joint = sum(1 for p in pairs if p["updated"] and p["retained"])
    both_wrong = sum(1 for p in pairs if not p["updated"] and not p["retained"])
    return {
        "n": n, "accuracy": correct_raw / n if n else 0.0,
        "by_role": {r: {"n": s["n"], "acc": s["correct"] / s["n"] if s["n"] else 0.0} for r, s in by_role.items()},
        "by_k": {str(k): {"n": s["n"], "acc": s["correct"] / s["n"] if s["n"] else 0.0,
                           "upd": s["upd_c"] / s["upd_n"] if s["upd_n"] else 0.0,
                           "ret": s["ret_c"] / s["ret_n"] if s["ret_n"] else 0.0}
                 for k, s in sorted(by_k.items())},
        "pairs": len(pairs), "joint": joint, "both_wrong": both_wrong,
        "joint_rate": joint / len(pairs) if pairs else 0.0,
    }


def rel(p: str | pathlib.Path) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def reset(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_allparam_model(model_dir: pathlib.Path, device: torch.device):
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(True)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(json.dumps({
        "event": "model_loaded",
        "model_dir": rel(model_dir),
        "loaded_class": type(model).__module__ + "." + type(model).__qualname__,
        "total_params": total,
        "trainable_params": trainable,
    }), flush=True)
    return model, tokenizer


def save_model(model, tokenizer, dst: pathlib.Path, src_model_dir: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    tokenizer.save_pretrained(str(dst))
    code = src_model_dir / "adapter_scaled_modeling.py"
    if code.exists() and not (dst / code.name).exists():
        shutil.copy2(str(code), str(dst / code.name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(CHCK82))
    ap.add_argument("--train-rows", default=str(TRAIN_DEFAULT))
    ap.add_argument("--held-rows", default=str(HELD_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=95095)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reset(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_allparam_model(pathlib.Path(args.model_dir), device)
    if torch.cuda.is_available():
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

    ds = BindingDataset(args.train_rows, tokenizer, max_length=args.max_length, seed=args.seed)
    if args.smoke:
        ds.rows = ds.rows[:64]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id or 0), drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6)

    cfg = {
        "status": "ALLPARAM_MIRRORED_PILOT_CONFIG",
        "scientific_question": "Can the same literal-value four-row family teach heldout identity-conditioned assignment when the full chck_82M model, not only the private branch, may update?",
        "model_dir": rel(args.model_dir),
        "train_rows": rel(args.train_rows),
        "held_rows": rel(args.held_rows),
        "trainable": "all_parameters",
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "smoke": bool(args.smoke),
    }
    (out / "train_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"event": "config", **cfg}, indent=2), flush=True)

    log_recs: list[dict[str, Any]] = []
    t0 = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_labeled = 0
        batches = 0
        for batch in loader:
            inp = {k: v.to(device) for k, v in batch.items()}
            outp = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], labels=inp["labels"])
            loss = outp.loss
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            labeled = int((inp["labels"] != -100).sum().item())
            total_loss += float(loss.detach().cpu()) * labeled
            total_labeled += labeled
            batches += 1
        avg_loss = total_loss / total_labeled if total_labeled else 0.0
        rec: dict[str, Any] = {"epoch": epoch, "loss": round(avg_loss, 5), "labeled": total_labeled,
                               "batches": batches, "elapsed": round(time.time() - t0, 1)}
        if epoch == 1 or epoch % int(args.eval_every) == 0 or epoch == int(args.epochs):
            rec["eval"] = evaluate_heldout(model, tokenizer, args.held_rows, device, max_rows=600 if not args.smoke else 40)
        log_recs.append(rec)
        print(json.dumps(rec), flush=True)

    with (out / "training_log.jsonl").open("w", encoding="utf-8") as f:
        for r in log_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ckpt = out / "checkpoint"
    save_model(model, tokenizer, ckpt, pathlib.Path(args.model_dir))
    final = log_recs[-1]
    summary = {
        "status": "ALLPARAM_MIRRORED_PILOT_DONE",
        "trainable": "all_parameters",
        "epochs": args.epochs,
        "final_loss": final.get("loss"),
        "final_eval": final.get("eval"),
        "checkpoint": rel(ckpt),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
