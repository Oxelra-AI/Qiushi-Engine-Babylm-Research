from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass, field

from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from .common import get_base_dir, print0, setup_distributed
from .dataloader import batch_iterator, count_dataset_tokens
from .model import RecGPTConfig, RecGPTForCausalLM
from .nl_aux_model import NL_Aux_Model, NL_Aux_Config
from .optimizer import SingleDeviceAuroraWithAuxAdam


@dataclass
class TrainConfig:
    model_config: RecGPTConfig = field(default_factory=RecGPTConfig)
    dataset: str = "bblm10M.parquet"
    tokenizer: str = "bblm10M-bpe"
    run_name: str = "rec-r16-10m"
    seed: int = 0
    data_seed: int = 0

    # Note: Gradient accumulation is considered to be wasteful (https://arxiv.org/pdf/2507.07101)
    # Tune for minimum batch size that is still efficient for your hardware, and adjust other hyperparameters accordingly.
    # Accumulation is still useful for reproducibility.
    microbatch_tok: int = 32768 # Tokens per microbatch (before grad accumulation) per gpu
    total_batch_tok: int = 32768 # Tokens per gradient step. Must be a multiple of microbatch_tok * gpu count.
    sequence_len: int = 512
    epochs: int = 10
    checkpoint_track: Optional[str] = None

    # Token limit (per epoch), -1 means use the entire dataset.
    # Note: Don't use this for multi epoch training on a subset, as it will not reset the dataloader and use the same data.
    max_tokens: int = -1 

    lr_embed: float = 0.005
    lr_block: float = 0.02
    min_lr: float = 0.0 # Minimum for cooldown, ignored during warmup
    wd_adam: float = 0.005
    wd_muon: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.997
    muon_momentum: float = 0.95
    warmup_ratio: float = 0.0
    cooldown_ratio: float = 0.2
    max_grad_norm: float = 2.0 # Not sure this is needed but may help with stability

    nl_mult: float = 2.0 # Multiplier for the NL aux loss
    nl_depth: int = 1
    nl_hidden: int = -1 # Default -1 sets it equal to model hidden
    nl_intermediate: int = 5120
    nl_lr: float = 0.02
    nl_wd: float = 0.1
    nl_momentum: float = 0.95

    torch_compile: bool = True
    use_wandb: bool = False
    wandb_project: str = "bblm26-recgpt"
    log_every: int = 10

def unwrap_model(model: torch.nn.Module) -> RecGPTForCausalLM:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"): # torch.compile
        model = model._orig_mod
    return model

# TODO: Check and maybe simplify this
def build_optimizer(cfg: TrainConfig, model: RecGPTForCausalLM, nl_model: Optional[NL_Aux_Model]):
    adam_params = []
    muon_params = []
    nl_muon_params = []

    param_sets = [(model.named_parameters(), muon_params)]
    if nl_model is not None:
        param_sets.append((nl_model.named_parameters(), nl_muon_params))

    for named_params, matrix_params in param_sets:
        for name, p in named_params:
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "embed_tokens" in name or "lm_head" in name or "norm" in name or "e_to_h" in name or "h_to_e" in name:
                adam_params.append(p)
            else:
                matrix_params.append(p)

    param_groups = [
        {"params": adam_params, "lr": cfg.lr_embed, "use_muon": False, "weight_decay": cfg.wd_adam, "betas": (cfg.adam_beta1, cfg.adam_beta2)},
        {"params": muon_params, "lr": cfg.lr_block, "use_muon": True, "weight_decay": cfg.wd_muon, "momentum": cfg.muon_momentum},
    ]
    if nl_muon_params:
        param_groups.append(
            {"params": nl_muon_params, "lr": cfg.nl_lr, "use_muon": True, "weight_decay": cfg.nl_wd, "momentum": cfg.nl_momentum},
        )

    return SingleDeviceAuroraWithAuxAdam(param_groups)

# Simple function to calculate lr from current tokens seen.
# We use token-based scheduling instead of step-based because of variable padding meaning we don't know exactly how many steps the dataset will be.
# This way we don't waste any tokens while also having an accurate lr schedule.
def token_to_lr(
    tokens_seen: int,
    total_tokens: int,
    warmup_ratio: float,
    cooldown_ratio: float,
    base_lr: float,
    min_lr: float, # Min for cooldown, ignored during warmup
) -> float:
    # Linear warmup, constant phase, then linear cooldown to min_lr.
    assert total_tokens > 0
    warmup_tokens = total_tokens * warmup_ratio
    cooldown_tokens = total_tokens * cooldown_ratio
    cooldown_start = total_tokens - cooldown_tokens

    if warmup_tokens > 0 and tokens_seen < warmup_tokens:
        factor = 1e-8 + (1.0 - 1e-8) * (tokens_seen / warmup_tokens) # Starting lr minimum 1e-8 times base_lr
        return base_lr * factor
    if cooldown_tokens > 0 and tokens_seen >= cooldown_start:
        progress = min(1.0, (tokens_seen - cooldown_start) / cooldown_tokens)
        return base_lr - (progress * (base_lr - min_lr))
    return base_lr

# Bit overkill but supports setting lrs for multiple param groups. 
# We only have 2, but we can have more in the future if we want to for some reason.
def set_lr( 
    optimizer: torch.optim.Optimizer,
    tokens_seen: int,
    total_tokens: int,
    warmup_ratio: float,
    cooldown_ratio: float,
    base_lrs: list[float],
    min_lr: float,
):
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = token_to_lr(
            tokens_seen, total_tokens, warmup_ratio, cooldown_ratio, base_lr, min_lr
        )


def config_to_json(cfg: TrainConfig) -> dict:
    out = {k: v for k, v in cfg.__dict__.items() if k != "model_config"}
    out["model_config"] = cfg.model_config.to_dict()
    return out


def checkpoint_schedule(track: Optional[str], epoch_tokens: int) -> tuple[int, list[tuple[int, int]]]:
    if track is None:
        return 0, []
    if track == "strict-small":
        words_per_epoch = 10 # Million words per epoch
        marks = range(1, 10)
    elif track == "strict":
        words_per_epoch = 100 # Million words per epoch
        marks = (*range(1, 10), *range(10, 100, 10))
    else:
        raise ValueError(f"unknown checkpoint track: {track}")
    return words_per_epoch, [(mark, math.ceil(epoch_tokens * mark / words_per_epoch)) for mark in marks]


def save_checkpoint(model: RecGPTForCausalLM, tokenizer: AutoTokenizer, cfg: TrainConfig, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.config.max_position_embeddings = max(model.config.max_position_embeddings, RecGPTConfig().max_position_embeddings)
    model.config.auto_map = {
        "AutoConfig": "modeling_recgpt.RecGPTConfig",
        "AutoModelForCausalLM": "modeling_recgpt.RecGPTForCausalLM",
    }
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    shutil.copyfile(get_base_dir() / "main" / "model.py", out_dir / "modeling_recgpt.py")
    with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(config_to_json(cfg), f, indent=2)


def train(cfg: TrainConfig):
    device, rank, local_rank, world_size, ddp = setup_distributed()

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    root = get_base_dir()
    parquet_path = root / "data" / "tokenized" / cfg.dataset
    tokenizer_path = root / "tokenizers" / cfg.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    cfg.model_config.vocab_size = len(tokenizer)
    cfg.model_config.pad_token_id = tokenizer.pad_token_id
    cfg.model_config.max_position_embeddings = cfg.sequence_len

    # Model initialization
    raw_model = RecGPTForCausalLM(cfg.model_config).to(device)
    model: torch.nn.Module = torch.compile(raw_model) if cfg.torch_compile else raw_model
    if ddp:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    # NextLat Aux model
    # Only used for auxiliary loss during training, discarded after.
    # Not exposed to saved model checkpoints, as it is not used for inference.
    assert cfg.nl_depth >= 1 # To disable, set nl_mult to 0
    nl_cfg = NL_Aux_Config(
        input_hidden_size=cfg.model_config.hidden_size,
        hidden_size=cfg.nl_hidden,
        intermediate_size=cfg.nl_intermediate,
    )
    nl_raw_model = NL_Aux_Model(nl_cfg).to(device)
    nl_model: torch.nn.Module = torch.compile(nl_raw_model) if cfg.torch_compile else nl_raw_model
    if ddp:
        nl_model = DDP(nl_model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = build_optimizer(cfg, raw_model, nl_raw_model)
    dataset_tokens = count_dataset_tokens(parquet_path)
    epoch_tokens = dataset_tokens if cfg.max_tokens < 0 else min(cfg.max_tokens, dataset_tokens)
    target_tokens = epoch_tokens * cfg.epochs
    checkpoint_words_per_epoch, first_epoch_checkpoints = checkpoint_schedule(cfg.checkpoint_track, epoch_tokens)
    global_microbatch_tok = cfg.microbatch_tok * world_size
    if cfg.total_batch_tok < global_microbatch_tok or cfg.total_batch_tok % global_microbatch_tok != 0:
        raise ValueError("total_batch_tok must be a multiple of microbatch_tok * world_size.")
    grad_acc = cfg.total_batch_tok // global_microbatch_tok
    total_microbatches = max(1, math.ceil(target_tokens / global_microbatch_tok))
    estimated_total_steps = max(1, math.ceil(total_microbatches / grad_acc)) # We can only estimate this due to padding and variable sequence lengths.
    if cfg.warmup_ratio < 0 or cfg.cooldown_ratio < 0 or cfg.warmup_ratio + cfg.cooldown_ratio > 1:
        raise ValueError("warmup_ratio and cooldown_ratio must be nonnegative and sum to at most 1.")
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    set_lr(
        optimizer,
        0,
        target_tokens,
        cfg.warmup_ratio,
        cfg.cooldown_ratio,
        base_lrs,
        cfg.min_lr,
    )

    wandb_run = None
    if cfg.use_wandb and rank == 0:
        import wandb

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=config_to_json(cfg))

    print0(
        f"training {cfg.run_name} | tokens {target_tokens} | estimated steps {estimated_total_steps} | "
        f"microbatch_tok {cfg.microbatch_tok} | total_batch_tok {cfg.total_batch_tok} | "
        f"grad_acc {grad_acc} | world_size {world_size}",
        rank=rank,
    )

    step = 0
    micro_step = 0
    tokens_seen = 0
    step_tokens_accum = 0
    loss_accum = torch.tensor(0.0, device=device) # We have to accumulate this stuff to support grad acc
    nl_loss_accum = torch.tensor(0.0, device=device) # We have to accumulate this stuff to support grad acc
    step_start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    model.train()

    def save_model(out_dir):
        if rank == 0:
            save_checkpoint(unwrap_model(model), tokenizer, cfg, out_dir)
            print(f"saved model to {out_dir}")
        if ddp:
            dist.barrier()

    for epoch in range(cfg.epochs if cfg.max_tokens < 0 else 10**12):
        iterator = batch_iterator(
            parquet_path,
            tokens_per_batch=cfg.microbatch_tok,
            max_sl=cfg.sequence_len,
            pad_token_id=cfg.model_config.pad_token_id,
            device=device,
            seed=cfg.data_seed + epoch,
            rank=rank,
            world_size=world_size,
        )
        for input_ids, labels, segment_ids, token_count in iterator:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, logits, x, e = model(input_ids=input_ids, segment_ids=segment_ids, labels=labels, return_hidden_and_embed=True, return_dict=False)
                loss = loss / grad_acc
                nl_loss = nl_model(hidden=x, embeddings=e, segment_ids=segment_ids, depth=cfg.nl_depth) / grad_acc

            loss_accum += loss.detach()
            nl_loss_accum += nl_loss.detach()
            loss += cfg.nl_mult * nl_loss
            loss.backward()
            step_tokens_accum += token_count
            micro_step += 1

            if micro_step % grad_acc != 0:
                continue

            if ddp:
                # Need to convert token count to tensor to all_reduce.
                # Technically there is no reason why token count sync has to go through the gpu... but here we are. (Only matters for multi-gpu anyways)
                step_tokens_t = torch.tensor(step_tokens_accum, device=device, dtype=torch.long)
                dist.all_reduce(step_tokens_t)
                step_tokens_accum = int(step_tokens_t.item())
            tokens_seen += step_tokens_accum

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for group in optimizer.param_groups for p in group["params"]),
                    cfg.max_grad_norm,
                )
            optimizer.step()
            set_lr(
                optimizer,
                tokens_seen,
                target_tokens,
                cfg.warmup_ratio,
                cfg.cooldown_ratio,
                base_lrs,
                cfg.min_lr,
            )
            optimizer.zero_grad(set_to_none=True)

            now = time.time()
            step_elapsed = max(now - step_start_time, 1e-9)
            step += 1
            if step % cfg.log_every == 0:
                if rank != 0: # To be safe with non-zero ranks
                    loss_accum.zero_()
                    nl_loss_accum.zero_()
                else:
                    loss_accum_float = loss_accum.item() / cfg.log_every
                    nl_loss_accum_float = nl_loss_accum.item() / cfg.log_every
                    loss_accum.zero_()
                    nl_loss_accum.zero_()
                    metrics = {
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": loss_accum_float,
                        "nl_loss": nl_loss_accum_float,
                        "lr_embed": optimizer.param_groups[0]["lr"],
                        "lr_block": optimizer.param_groups[1]["lr"],
                        "lr_nl": optimizer.param_groups[2]["lr"],
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": step_tokens_accum / step_elapsed,
                        "step_time": step_elapsed,
                    }
                    print(
                        f"Epoch {epoch + 1}/{cfg.epochs} "
                        f"Step {step} "
                        f"training loss: {loss_accum_float:.3f} "
                        f"nextlat loss: {nl_loss_accum_float:.3f} "
                        f"lr_embed {optimizer.param_groups[0]['lr']:.5g} "
                        f"lr_block {optimizer.param_groups[1]['lr']:.5g} "
                        f"lr_nl {optimizer.param_groups[2]['lr']:.5g} "
                        f"step_time {step_elapsed:.2f}s "
                        f"tok/s {step_tokens_accum / step_elapsed:.0f} "
                    )
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=tokens_seen)

            step_tokens_accum = 0 # Tok/s and step time reporting not averaged
            while first_epoch_checkpoints and tokens_seen >= first_epoch_checkpoints[0][1]:
                mark, _ = first_epoch_checkpoints.pop(0)
                save_model(root / "models" / f"{cfg.run_name}-checkpoints" / f"chck_{mark}M")
                now = time.time()
            step_start_time = now
            if tokens_seen >= target_tokens:
                break
        if checkpoint_words_per_epoch:
            mark = checkpoint_words_per_epoch * (epoch + 1)
            save_model(root / "models" / f"{cfg.run_name}-checkpoints" / f"chck_{mark}M")
            step_start_time = time.time()
        if tokens_seen >= target_tokens:
            break

    save_model(root / "models" / cfg.run_name)
    if wandb_run is not None:
        wandb_run.finish()
    if ddp:
        torch.distributed.destroy_process_group()
