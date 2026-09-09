#!/usr/bin/env python3
"""research: Mirrored four-row binding pilot trainer.

Pure binding CE (100% answer masking) on the mirrored family.
Tests whether the private adapter can learn identity-conditioned assignment.
No coherent replay in this pilot — that comes in the composition if this passes.
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import json, os, sys, time, argparse, pathlib, random
from collections import defaultdict

_SCRIPT = _public_path('experiments/archive/relation_learning/scripts/mirrored_pilot_trainer.py')
_ROOT = _public_path('.')

# Set cache BEFORE importing transformers
CACHE = str(_public_path('experiments/archive/relation_learning/data/pilot/hf_cache'))
os.makedirs(CACHE, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = CACHE
os.environ["HF_HOME"] = CACHE

sys.path.insert(0, str(_public_path('experiments/archive/frontier_consolidation/scripts')))
from frozen82_private_modeling import FrozenSlowPrivateDebertaV2ForMaskedLM

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
TRAIN_DEFAULT = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_train_frame_seen.jsonl')
HELD_DEFAULT  = _public_path('experiments/archive/relation_learning/data/mirrored_four_row_family/mirrored_heldout_frame_all.jsonl')
OUT_DEFAULT   = _public_path('experiments/archive/relation_learning/data/pilot')


def rel(p):
    try: return str(pathlib.Path(p).resolve().relative_to(_ROOT))
    except: return str(p)


def load_frozen_model(chck_path, private_scale, device):
    from transformers import DebertaV2Config, AutoTokenizer
    from safetensors.torch import load_file

    tokenizer = AutoTokenizer.from_pretrained(str(chck_path), use_fast=True, local_files_only=True)
    config = DebertaV2Config.from_pretrained(str(chck_path))
    config.private_adapter_bottleneck = getattr(config, "adapter_bottleneck", 128)
    config.private_adapter_scale = private_scale
    config.private_adapter_enabled = True
    config.private_adapter_activation = getattr(config, "adapter_activation", "gelu")

    model = FrozenSlowPrivateDebertaV2ForMaskedLM(config)
    sf = chck_path / "model.safetensors"
    state_dict = load_file(str(sf), device="cpu")
    model.load_state_dict(state_dict, strict=False)

    # Freeze everything, then unfreeze private adapter
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = 0
    for name, p in model.named_parameters():
        if "private_adapter" in name:
            p.requires_grad_(True)
            trainable += p.numel()

    model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {total:,} params, {trainable:,} trainable (private adapter)", flush=True)
    return model, tokenizer


class BindingDataset(Dataset):
    def __init__(self, rows_path, tokenizer, max_length=512, seed=91091):
        self.rows = []
        with open(rows_path) as f:
            for line in f:
                self.rows.append(json.loads(line.strip()))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_id = tokenizer.mask_token_id
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row["row_text"], truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, padding=False, add_special_tokens=True,
        )
        input_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        labels = [-100] * len(input_ids)

        ans_s = row["answer_char_start"]
        ans_e = row["answer_char_end"]
        n_labeled = 0
        for i, (s, e) in enumerate(offsets):
            if s is not None and e is not None and s >= ans_s and e <= ans_e and s < e:
                labels[i] = input_ids[i]
                input_ids[i] = self.mask_id
                n_labeled += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "n_tokens": len(input_ids),
            "n_labeled": n_labeled,
            "answer_role": row["answer_role"],
            "k": row["k"],
        }


def collate(batch, pad_id=0):
    mx = max(b["n_tokens"] for b in batch)
    ids, labs, mask = [], [], []
    for b in batch:
        n = b["n_tokens"]
        ids.append(b["input_ids"] + [pad_id] * (mx - n))
        labs.append(b["labels"] + [-100] * (mx - n))
        mask.append([1] * n + [0] * (mx - n))
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labs),
        "attention_mask": torch.tensor(mask),
    }


def evaluate_heldout(model, tokenizer, held_path, device, max_rows=600, frame_filter="f00"):
    """Quick heldout evaluation using PLL scorer."""
    rows = []
    with open(held_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if frame_filter and frame_filter not in r["frame_id"]:
                continue
            rows.append(r)
    if max_rows > 0:
        rows = rows[:max_rows]

    mask_id = tokenizer.mask_token_id
    cls_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else 0
    sep_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else 0

    correct_raw = 0
    by_role = defaultdict(lambda: {"n": 0, "correct": 0})
    by_k = defaultdict(lambda: {"n": 0, "correct": 0, "upd_n": 0, "upd_c": 0, "ret_n": 0, "ret_c": 0})
    pair_key = defaultdict(dict)

    model.eval()
    with torch.no_grad():
        for row in rows:
            prefix = row["context_text"] + " " + row["query_prefix"]
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

            best_cand = None
            best_lp = float("-inf")
            for cand_val in row["all_present_values"]:
                cand_ids = tokenizer.encode(cand_val, add_special_tokens=False)
                if not cand_ids:
                    continue
                avail = 510 - len(cand_ids)
                trunc = prefix_ids[-avail:] if len(prefix_ids) > avail else prefix_ids
                base = [cls_id] + trunc + cand_ids + [sep_id]
                cs = len(base) - len(cand_ids) - 1

                batch_ids = []
                for i in range(len(cand_ids)):
                    m = list(base)
                    m[cs + i] = mask_id
                    batch_ids.append(m)

                inp = torch.tensor(batch_ids, device=device)
                out = model(input_ids=inp)
                lp = 0.0
                for i in range(len(cand_ids)):
                    logits = out.logits[i, cs + i]
                    lp += F.log_softmax(logits, dim=-1)[cand_ids[i]].item()

                if lp > best_lp:
                    best_lp = lp
                    best_cand = cand_val

            ok = best_cand == row["answer"]
            correct_raw += int(ok)
            role = row["answer_role"]
            by_role[role]["n"] += 1
            by_role[role]["correct"] += int(ok)
            k = row["k"]
            by_k[k]["n"] += 1
            by_k[k]["correct"] += int(ok)
            if role == "updated":
                by_k[k]["upd_n"] += 1; by_k[k]["upd_c"] += int(ok)
            else:
                by_k[k]["ret_n"] += 1; by_k[k]["ret_c"] += int(ok)
            pair_key[(row["quad_id"], row["context_variant"])][role] = ok

    n = len(rows)
    pairs = [v for v in pair_key.values() if "updated" in v and "retained" in v]
    joint = sum(1 for p in pairs if p["updated"] and p["retained"])
    both_wrong = sum(1 for p in pairs if not p["updated"] and not p["retained"])

    return {
        "n": n, "accuracy": correct_raw / n if n else 0,
        "by_role": {r: {"n": s["n"], "acc": s["correct"]/s["n"] if s["n"] else 0}
                    for r, s in by_role.items()},
        "by_k": {str(k): {"n": s["n"], "acc": s["correct"]/s["n"] if s["n"] else 0,
                           "upd": s["upd_c"]/s["upd_n"] if s["upd_n"] else 0,
                           "ret": s["ret_c"]/s["ret_n"] if s["ret_n"] else 0}
                 for k, s in sorted(by_k.items())},
        "pairs": len(pairs), "joint": joint, "both_wrong": both_wrong,
        "joint_rate": joint / len(pairs) if pairs else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chck82", type=pathlib.Path, default=CHCK82)
    ap.add_argument("--train-rows", default=str(TRAIN_DEFAULT))
    ap.add_argument("--held-rows", default=str(HELD_DEFAULT))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--private-scale", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_frozen_model(args.chck82, args.private_scale, device)

    ds = BindingDataset(args.train_rows, tokenizer)
    if args.smoke:
        ds.rows = ds.rows[:64]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(b, tokenizer.pad_token_id or 0),
                        drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    config = {
        "status": "MIRRORED_PILOT",
        "chck82": rel(args.chck82),
        "train_rows": rel(args.train_rows),
        "held_rows": rel(args.held_rows),
        "private_scale": args.private_scale,
        "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_row_count": len(ds),
        "smoke": args.smoke,
    }
    with open(out / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(json.dumps({"event": "config", **config}, indent=2), flush=True)

    trajectory = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_labeled = 0
        n_batches = 0

        for batch in loader:
            inp = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=inp["input_ids"],
                           attention_mask=inp["attention_mask"],
                           labels=inp["labels"])
            loss = outputs.loss
            if loss is None:
                continue

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            labeled = int((inp["labels"] != -100).sum().item())
            epoch_loss += loss.item() * labeled
            epoch_labeled += labeled
            n_batches += 1

        avg_loss = epoch_loss / epoch_labeled if epoch_labeled else 0
        rec = {"epoch": epoch + 1, "loss": round(avg_loss, 4),
               "labeled": epoch_labeled, "batches": n_batches,
               "elapsed": round(time.time() - t0, 1)}

        if (epoch + 1) % args.eval_every == 0 or epoch == 0 or epoch + 1 == args.epochs:
            model.eval()
            ev = evaluate_heldout(model, tokenizer, args.held_rows, device,
                                  max_rows=600 if not args.smoke else 40)
            rec["eval"] = ev
            print(json.dumps(rec), flush=True)
        else:
            print(json.dumps(rec), flush=True)

        trajectory.append(rec)

    # Save final
    with open(out / "training_log.jsonl", "w") as f:
        for rec in trajectory:
            f.write(json.dumps(rec) + "\n")

    # Save checkpoint
    ckpt_dir = out / "checkpoint"
    ckpt_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    final = trajectory[-1]
    print(json.dumps({"status": "PILOT_DONE",
                       "epochs": args.epochs,
                       "final_loss": final["loss"],
                       "final_eval": final.get("eval"),
                       "checkpoint": rel(ckpt_dir),
                       "elapsed": round(time.time() - t0, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
