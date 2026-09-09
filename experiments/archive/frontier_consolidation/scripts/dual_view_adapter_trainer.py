#!/usr/bin/env python3
"""research Dual-View Adapter Trainer.

Implements the validated shared-readout transfer mechanism (research) as a full
pretraining objective within a legal 100M trajectory.

For pair rows (~4.6% of data): two forward passes through the same model:
  1. Source-conditioned: full row (source + rewrite interleaved) with standard WWM
  2. Source-free: rewrite-only text with the SAME rewrite word groups masked
Both predict the same rewrite targets through the shared backbone + adapter.

The auxiliary CE loss on the source-free view forces the adapter to learn
representations that predict rewrite tokens without source context.

Architecture: DeBERTa-v2 8×480 + zero-init bottleneck adapter (from research)
Training: full legal 100M trajectory from random initialization

Modes:
  dual_view_true: correct source-rewrite alignment (treatment)
  mlm_only: no auxiliary view (matched adapter baseline)
"""

import argparse, json, math, os, sys, time, hashlib, shutil
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── Paths ────────────────────────────────────────────────────────────────────
STUDY = Path("experiments/archive/frontier_consolidation")
WORKSPACE = STUDY
SCRIPTS = WORKSPACE / "scripts"

# Add scripts dir for adapter modeling import
sys.path.insert(0, str(SCRIPTS))
from adapter_scaled_modeling import (
    AdapterDebertaV2ForMaskedLM, ZeroOutputBottleneckAdapter
)
from transformers import AutoTokenizer, DebertaV2Config

# Defaults matching the research recipe
DEFAULTS = dict(
    hidden_size=480, n_layer=8, n_head=8, ffn_mult=4,
    seed=43, extra_init_seed=43022, train_rng_seed=43023,
    batch_size=256, seq_length=256,
    learning_rate=0.001, warmup_fraction=0.06, weight_decay=0.01,
    mask_prob=0.15, max_word_exposure=100_000_000,
    checkpoint_words=1_000_000, log_every=50,
    adapter_bottleneck=128, adapter_scale=1.0,
    aux_lambda=1.0,
)


# ─── Data ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainExample:
    text: str
    words: int
    source: str
    example_id: int


def load_examples(path: str, max_words: int) -> list[TrainExample]:
    """Load JSONL examples with example_id tracking."""
    examples = []
    selected = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = str(obj["text"])
            words = int(obj.get("words", len(text.split())))
            actual = len(text.split())
            if words != actual:
                raise RuntimeError(f"Word mismatch: field={words} actual={actual}")
            if selected + words > max_words:
                break
            examples.append(TrainExample(
                text=text, words=words,
                source=str(obj.get("source", "")),
                example_id=int(obj.get("example_id", -1))
            ))
            selected += words
    return examples


def is_word_start(s: str) -> bool:
    return s.startswith("Ġ") or s.startswith("▁")


class DualViewDataset(Dataset):
    def __init__(self, examples: list[TrainExample], tokenizer, seq_length: int,
                 pair_eid_set: set):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self.pair_eid_set = pair_eid_set
        self._ws_cache = {}

    def _word_start(self, tid: int) -> bool:
        v = self._ws_cache.get(tid)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self._ws_cache[tid] = v
        return v

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding="max_length", return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._word_start(tid) or i == 0:
                gid += 1
            group[i] = gid
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_group": group,
            "words": ex.words,
            "example_id": ex.example_id,
            "is_pair": ex.example_id in self.pair_eid_set,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "word_group": torch.stack([x["word_group"] for x in batch]),
        "words": torch.tensor([x["words"] for x in batch], dtype=torch.long),
        "example_id": torch.tensor([x["example_id"] for x in batch], dtype=torch.long),
        "is_pair": torch.tensor([x["is_pair"] for x in batch], dtype=torch.bool),
    }


# ─── Masking ──────────────────────────────────────────────────────────────────
def apply_wwm_masking(input_ids, attention_mask, word_group, tokenizer,
                      mask_prob, gen):
    """Standard whole-word masking with 80/10/10 corruption."""
    device = input_ids.device
    bsz, seq = input_ids.shape
    labels = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = torch.zeros_like(candidate)

    for b in range(bsz):
        groups = word_group[b]
        valid = torch.unique(groups[groups >= 0])
        if valid.numel() == 0:
            continue
        r = torch.rand(valid.numel(), generator=gen, device=device)
        chosen = valid[r < mask_prob]
        if chosen.numel() > 0:
            select[b] = torch.isin(groups, chosen) & candidate[b]

    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[flat[0, 0]] = True

    labels[~select] = -100
    masked_inputs = input_ids.clone()
    r = torch.rand(bsz, seq, generator=gen, device=device)
    masked_inputs[select & (r < 0.8)] = mask_token_id
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked_inputs[rand_tok] = torch.randint(
            0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
    return masked_inputs, labels


# ─── Pair Data ────────────────────────────────────────────────────────────────
@dataclass
class PairRowInfo:
    example_id: int
    rw_input_ids: torch.Tensor      # [seq_len], original rewrite-only tokens
    rw_attention_mask: torch.Tensor  # [seq_len]
    rw_word_group: torch.Tensor     # [seq_len]
    full_rw_wg_set: set             # word groups in full text that are rewrite
    full_wg_to_rw_wg: dict          # full rewrite WG -> rw-only WG


def load_pair_data(pair_data_path: str, seq_length: int, device="cpu") -> dict:
    """Load pre-computed pair data from JSON."""
    with open(pair_data_path) as f:
        raw = json.load(f)

    pair_map = {}
    for eid_str, d in raw["pair_data"].items():
        eid = int(eid_str)
        pair_map[eid] = PairRowInfo(
            example_id=eid,
            rw_input_ids=torch.tensor(d["rewrite_input_ids"][:seq_length], dtype=torch.long),
            rw_attention_mask=torch.tensor(d["rewrite_attention_mask"][:seq_length], dtype=torch.long),
            rw_word_group=torch.tensor(d["rewrite_word_group"][:seq_length], dtype=torch.long),
            full_rw_wg_set=set(d["full_rw_word_groups"]),
            full_wg_to_rw_wg={int(k): v for k, v in d["full_wg_to_rw_wg"].items()},
        )
    return pair_map


def build_auxiliary_view(batch_word_group, batch_labels, pair_indices,
                         pair_data_list, mask_token_id, device):
    """Build rewrite-only masked inputs for pair rows in the batch.

    Returns:
        rw_input_ids: [P, S] masked rewrite-only tokens
        rw_attention_mask: [P, S]
        rw_labels: [P, S] with -100 at non-target positions
        n_aux_targets: total masked rewrite tokens
    """
    P = len(pair_indices)
    S = batch_word_group.shape[1]
    rw_ids_list = []
    rw_mask_list = []
    rw_labels_list = []
    total_targets = 0

    for idx, bi in enumerate(pair_indices):
        pd = pair_data_list[idx]
        wg = batch_word_group[bi]   # [S]
        lab = batch_labels[bi]      # [S]

        # Find masked word groups that are rewrite word groups
        masked_positions = (lab != -100)
        if not masked_positions.any():
            # No masked tokens at all — still add a dummy row
            rw_ids_list.append(pd.rw_input_ids.to(device))
            rw_mask_list.append(pd.rw_attention_mask.to(device))
            rw_labels_list.append(torch.full((S,), -100, dtype=torch.long, device=device))
            continue

        masked_wgs = torch.unique(wg[masked_positions])
        masked_wgs = masked_wgs[masked_wgs >= 0].tolist()

        # Filter to rewrite word groups and map to rw-only WGs
        rw_wgs_to_mask = set()
        for fwg in masked_wgs:
            if fwg in pd.full_rw_wg_set:
                rw_wg = pd.full_wg_to_rw_wg.get(fwg)
                if rw_wg is not None:
                    rw_wgs_to_mask.add(rw_wg)

        # Build rewrite-only masked input
        rw_ids = pd.rw_input_ids.clone().to(device)
        rw_mask = pd.rw_attention_mask.clone().to(device)
        rw_lab = torch.full((S,), -100, dtype=torch.long, device=device)

        if rw_wgs_to_mask:
            rw_wg_tensor = pd.rw_word_group.to(device)
            for rwg in rw_wgs_to_mask:
                positions = (rw_wg_tensor == rwg)
                rw_lab[positions] = rw_ids[positions]  # target = original token
                rw_ids[positions] = mask_token_id       # input = [MASK]
            total_targets += int((rw_lab != -100).sum().item())

        rw_ids_list.append(rw_ids)
        rw_mask_list.append(rw_mask)
        rw_labels_list.append(rw_lab)

    if not rw_ids_list:
        return None, None, None, 0

    return (torch.stack(rw_ids_list),
            torch.stack(rw_mask_list),
            torch.stack(rw_labels_list),
            total_targets)


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model(args, tokenizer):
    """Build DeBERTa-v2 with zero-init adapter."""
    max_pos = max(512, args.seq_length + 8)
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=max_pos,
        max_relative_positions=256,
        position_buckets=256,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_activation="gelu",
        adapter_enabled=True,
        adapter_scale=args.adapter_scale,
    )
    model = AdapterDebertaV2ForMaskedLM(cfg)
    return model


def save_checkpoint(model, tokenizer, dst_path: Path):
    dst_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst_path), safe_serialization=True)
    tokenizer.save_pretrained(str(dst_path))
    # Copy modeling file for trust_remote_code
    src_modeling = SCRIPTS / "adapter_scaled_modeling.py"
    dst_modeling = dst_path / src_modeling.name
    if not dst_modeling.exists():
        shutil.copy2(str(src_modeling), str(dst_modeling))


# ─── LR Schedule ──────────────────────────────────────────────────────────────
def get_lr(step, total_steps, warmup_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Training ─────────────────────────────────────────────────────────────────
def train(args):
    start_time = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Writable HF cache
    hf_cache = str(out / "hf_cache")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache
    os.environ["TRANSFORMERS_CACHE"] = hf_cache
    os.environ["HF_MODULES_CACHE"] = str(Path(hf_cache) / "modules")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "device": str(device),
                       "mode": args.mode, "aux_lambda": args.aux_lambda}), flush=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    mask_token_id = tokenizer.mask_token_id

    # Load pair data
    pair_map = load_pair_data(args.pair_data_path, args.seq_length)
    pair_eid_set = set(pair_map.keys())
    print(json.dumps({"event": "pair_data_loaded", "n_pairs": len(pair_map)}), flush=True)

    # Load training data
    examples = load_examples(args.example_jsonl, args.max_word_exposure)
    actual_words = sum(e.words for e in examples)
    n_pair_examples = sum(1 for e in examples if e.example_id in pair_eid_set)
    print(json.dumps({"event": "data_loaded", "n_examples": len(examples),
                       "words": actual_words, "pair_examples": n_pair_examples}), flush=True)

    # Dataset and loader
    dataset = DualViewDataset(examples, tokenizer, args.seq_length, pair_eid_set)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)

    # Model
    torch.manual_seed(args.seed)
    model = build_model(args, tokenizer)
    # Apply extra init seed
    if args.extra_init_seed > 0:
        g = torch.Generator()
        g.manual_seed(args.extra_init_seed)
        for p in model.parameters():
            if p.requires_grad and p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=0.02, generator=g)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if ".adapter." in n)
    print(json.dumps({"event": "model_built", "total_params": total_params,
                       "adapter_params": adapter_params}), flush=True)

    # Optimizer — exclude zero-init adapter terms from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ".adapter.up." in name or ".adapter.up.bias" in name:
            no_decay_params.append(param)
        elif "bias" in name or "LayerNorm" in name or "layer_norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-6)

    total_steps = len(loader)
    warmup_steps = int(total_steps * args.warmup_fraction)

    # Masking generator
    gen = torch.Generator(device=device)
    gen.manual_seed(args.train_rng_seed)

    # Save training config
    config = {
        "mode": args.mode, "aux_lambda": args.aux_lambda,
        "adapter_scale": args.adapter_scale,
        "adapter_bottleneck": args.adapter_bottleneck,
        "total_steps": total_steps, "warmup_steps": warmup_steps,
        "total_params": total_params, "adapter_params": adapter_params,
        "n_examples": len(examples), "actual_words": actual_words,
        "n_pair_examples": n_pair_examples,
        **{k: getattr(args, k) for k in ["seed", "extra_init_seed", "train_rng_seed",
           "batch_size", "seq_length", "learning_rate", "warmup_fraction",
           "weight_decay", "mask_prob", "max_word_exposure", "checkpoint_words"]},
    }
    (out / "train_config.json").write_text(json.dumps(config, indent=2))

    # Training loop
    model.train()
    cumulative_words = 0
    loss_accum = []
    aux_loss_accum = []
    next_ckpt = args.checkpoint_words if args.checkpoint_words > 0 else None
    log_path = out / "training_log.jsonl"

    with log_path.open("w") as logf:
        for step, batch in enumerate(loader, 1):
            words = int(batch["words"].sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            example_ids = batch["example_id"]
            is_pair = batch["is_pair"]

            # Adjust LR
            lr = get_lr(step - 1, total_steps, warmup_steps, args.learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Standard masking
            masked_inputs, labels = apply_wwm_masking(
                input_ids, attention_mask, word_group, tokenizer,
                args.mask_prob, gen)

            # Main forward pass
            optimizer.zero_grad(set_to_none=True)
            out_model = model(input_ids=masked_inputs, attention_mask=attention_mask,
                              labels=labels)
            main_loss = out_model.loss
            n_main_pred = int((labels != -100).sum().item())

            # Auxiliary forward pass for pair rows
            aux_loss_val = 0.0
            n_aux_targets = 0

            if args.mode != "mlm_only":
                pair_batch_indices = is_pair.nonzero(as_tuple=False).squeeze(-1).tolist()
                if isinstance(pair_batch_indices, int):
                    pair_batch_indices = [pair_batch_indices]

                if pair_batch_indices:
                    pair_data_list = []
                    valid_pair_indices = []
                    for bi in pair_batch_indices:
                        eid = int(example_ids[bi].item())
                        if eid in pair_map:
                            pair_data_list.append(pair_map[eid])
                            valid_pair_indices.append(bi)

                    if valid_pair_indices:
                        rw_ids, rw_mask, rw_lab, n_aux = build_auxiliary_view(
                            word_group, labels, valid_pair_indices,
                            pair_data_list, mask_token_id, device)

                        if rw_ids is not None and n_aux > 0:
                            rw_out = model(input_ids=rw_ids,
                                          attention_mask=rw_mask)
                            V = rw_out.logits.shape[-1]
                            aux_loss = F.cross_entropy(
                                rw_out.logits.view(-1, V),
                                rw_lab.view(-1),
                                ignore_index=-100)
                            aux_loss_val = float(aux_loss.detach().cpu())
                            n_aux_targets = n_aux

                            total_loss = main_loss + args.aux_lambda * aux_loss
                        else:
                            total_loss = main_loss
                    else:
                        total_loss = main_loss
                else:
                    total_loss = main_loss
            else:
                total_loss = main_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            cumulative_words += words
            main_loss_val = float(main_loss.detach().cpu())
            loss_accum.append(main_loss_val)
            if aux_loss_val > 0:
                aux_loss_accum.append(aux_loss_val)

            rec = {
                "step": step, "loss": main_loss_val, "aux_loss": aux_loss_val,
                "lr": lr, "batch_words": words,
                "cumulative_word_exposure": cumulative_words,
                "masked_tokens": n_main_pred, "aux_targets": n_aux_targets,
                "elapsed_sec": round(time.time() - start_time, 1),
            }
            logf.write(json.dumps(rec) + "\n")
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)

            # Checkpointing
            while (next_ckpt is not None and cumulative_words >= next_ckpt
                   and next_ckpt <= args.max_word_exposure):
                if next_ckpt < 1_000_000:
                    name = "chck_1M"
                elif next_ckpt % 1_000_000 == 0:
                    name = f"chck_{next_ckpt // 1_000_000}M"
                else:
                    name = f"chck_{next_ckpt}w"
                cp = out / "hf_model" / name
                save_checkpoint(model, tokenizer, cp)
                print(json.dumps({"event": "checkpoint", "name": name,
                                   "words": cumulative_words, "step": step}), flush=True)
                next_ckpt += args.checkpoint_words

    # Final checkpoint
    save_checkpoint(model, tokenizer, out / "hf_model" / "final")
    print(json.dumps({"event": "checkpoint", "name": "final",
                       "words": cumulative_words, "step": total_steps}), flush=True)

    # Save scientific metrics
    metrics = {
        "status": "DUAL_VIEW_TRAINING",
        "mode": args.mode,
        "aux_lambda": args.aux_lambda,
        "total_steps": total_steps,
        "total_word_exposure": cumulative_words,
        "total_params": total_params,
        "adapter_params": adapter_params,
        "mean_loss": sum(loss_accum) / len(loss_accum) if loss_accum else 0,
        "final_loss": loss_accum[-1] if loss_accum else 0,
        "first_loss": loss_accum[0] if loss_accum else 0,
        "mean_aux_loss": sum(aux_loss_accum) / len(aux_loss_accum) if aux_loss_accum else 0,
        "aux_loss_batches": len(aux_loss_accum),
        "elapsed_sec": round(time.time() - start_time, 2),
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"event": "done", **metrics}), flush=True)


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="research Dual-View Adapter Trainer")
    p.add_argument("--example_jsonl", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--pair_data_path", required=True,
                   help="Pre-computed pair data JSON from build_and_validate_pair_data.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["dual_view_true", "mlm_only"],
                   default="dual_view_true")
    for k, v in DEFAULTS.items():
        flag = f"--{k}"
        if isinstance(v, float):
            p.add_argument(flag, type=float, default=v)
        elif isinstance(v, int):
            p.add_argument(flag, type=int, default=v)
        else:
            p.add_argument(flag, default=v)
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
