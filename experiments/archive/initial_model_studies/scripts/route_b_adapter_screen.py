#!/usr/bin/env python3
"""research — Route B: surface/morphology adapter screen on protected DeBERTa 8×480 WWM.

Three arms at 10M exposure:
1. baseline: protected DeBERTa-v2 8×480, baseline16k, WWM, official corpus only.
2. surface: same backbone + char n-gram/prefix/suffix surface adapter gated into embeddings.
3. token_id: parameter-matched token-ID adapter with no surface sharing.

The adapter is applied after the input embedding lookup and before the encoder.
It uses a learned scalar gate initialized to 0 so the model starts as a pure MLM
and can gradually incorporate surface information.

The script reuses the leadershape trainer's data pipeline, word accounting, WWM,
checkpoint conventions, and portable tokenizer save.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, random, sys, time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM, get_cosine_schedule_with_warmup

# Reuse the leadershape trainer's data pipeline and utilities.
ROOT = pathlib.Path("experiments/archive/initial_model_studies")
TRAINER = ROOT / "training/scripts/babylm_masked_train_leadershape.py"
sys.path.insert(0, str(TRAINER.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("babylm_masked_train_leadershape", TRAINER)
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)

# ── Surface feature extraction ──────────────────────────────────────────────

def _normalize_token_str(s: str) -> str:
    """Normalize a tokenizer string for surface feature extraction.
    
    Strips GPT-2 byte-level BPE space prefix 'Ġ' and SentencePiece '▁'.
    Keeps case, punctuation, and digits as-is for char n-gram extraction.
    """
    s = s.strip()
    if s.startswith("Ġ"):
        s = s[1:]
    if s.startswith("▁"):
        s = s[1:]
    return s

def _char_ngram_features(s: str, n_range: tuple[int, int] = (2, 4), max_features: int = 2048) -> torch.Tensor:
    """Extract fixed-dimension char n-gram features via hashing.
    
    Returns a float tensor of shape (max_features,) with count-based features.
    """
    s = _normalize_token_str(s)
    feats = torch.zeros(max_features, dtype=torch.float32)
    if not s:
        return feats
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(s) - n + 1):
            key = s[i:i + n].encode("utf-8")
            h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % max_features
            feats[h] += 1.0
    return feats

def _prefix_suffix_features(s: str, max_len: int = 4, max_features: int = 512) -> torch.Tensor:
    """Extract prefix/suffix features via hashing.
    
    Returns a float tensor of shape (max_features,).
    """
    s = _normalize_token_str(s)
    feats = torch.zeros(max_features, dtype=torch.float32)
    if not s:
        return feats
    for k in range(1, min(max_len, len(s)) + 1):
        pre_key = ("pre:" + s[:k]).encode("utf-8")
        suf_key = ("suf:" + s[-k:]).encode("utf-8")
        h_pre = int.from_bytes(hashlib.blake2b(pre_key, digest_size=8).digest(), "little") % max_features
        h_suf = int.from_bytes(hashlib.blake2b(suf_key, digest_size=8).digest(), "little") % max_features
        feats[h_pre] += 1.0
        feats[h_suf] += 1.0
    return feats

def build_surface_features(tokenizer, max_features: int = 2560) -> torch.Tensor:
    """Build a per-token surface feature matrix.
    
    Returns a tensor of shape (vocab_size, max_features) where each row is the
    concatenated char n-gram and prefix/suffix features for that token.
    """
    vocab_size = len(tokenizer)
    ngram_dim = 2048
    affix_dim = 512
    assert ngram_dim + affix_dim == max_features, f"feature dim mismatch: {ngram_dim}+{affix_dim}!={max_features}"
    features = torch.zeros(vocab_size, max_features, dtype=torch.float32)
    for i in range(vocab_size):
        s = tokenizer.convert_ids_to_tokens(i)
        if s is None:
            continue
        ng = _char_ngram_features(s, n_range=(2, 4), max_features=ngram_dim)
        af = _prefix_suffix_features(s, max_len=4, max_features=affix_dim)
        row = torch.cat([ng, af])
        features[i] = row / row.norm().clamp(min=1e-8)
    return features

def build_token_id_features(vocab_size: int, max_features: int = 2560) -> torch.Tensor:
    """Build a per-token ID-based feature matrix with no surface sharing.
    
    Each token gets a unique random feature vector. This has the same parameter
    count as the surface adapter but no cross-token sharing.
    """
    gen = torch.Generator()
    gen.manual_seed(42)
    features = torch.randn(vocab_size, max_features, generator=gen)
    features = features / features.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return features

# ── Adapter module ───────────────────────────────────────────────────────────

class EmbeddingAdapter(nn.Module):
    """Surface or token-ID adapter that modifies token embeddings before the encoder.
    
    h' = h + gate * proj(feature[token_id])
    
    where gate is a learned scalar initialized to 0, so the model starts as a
    pure MLM and can gradually incorporate adapter information.
    """
    def __init__(self, hidden_size: int, feature_dim: int, feature_matrix: torch.Tensor):
        super().__init__()
        self.hidden_size = hidden_size
        self.feature_dim = feature_dim
        self.register_buffer("features", feature_matrix)
        self.proj = nn.Linear(feature_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """h: (B, L, hidden), input_ids: (B, L)"""
        feat = self.features[input_ids]  # (B, L, feature_dim)
        contribution = self.proj(feat)   # (B, L, hidden)
        return h + self.gate * contribution
    
    def extra_params(self) -> int:
        """Return parameter count of adapter components (proj + gate)."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def vocabulary_delta(self) -> torch.Tensor:
        """Dense vocabulary embedding correction, shape (vocab, hidden)."""
        return self.gate * self.proj(self.features)


def save_fused_hf(model, tokenizer, dst: pathlib.Path) -> None:
    """Export an adapted wrapper as an ordinary HF DeBERTa MLM.

    The training model uses adapter contribution only on the input side. DeBERTa
    normally ties input embeddings and decoder weights, so export must untie them:
    input table receives the adapter delta while decoder retains the original table.
    """
    if model.adapter is None:
        trainer.save_hf_checkpoint(model.base, tokenizer, dst)
        return
    dst.mkdir(parents=True, exist_ok=True)
    emb = model.base.get_input_embeddings()
    decoder = model.base.get_output_embeddings()
    original_input = emb.weight.detach().clone()
    original_output = decoder.weight.detach().clone()
    delta = model.adapter.vocabulary_delta().to(original_input.device, original_input.dtype)
    model.base.config.tie_word_embeddings = False
    emb.weight.data.copy_(original_input + delta)
    decoder.weight = nn.Parameter(original_output)
    model.base.save_pretrained(dst, safe_serialization=True)
    tokenizer.save_pretrained(dst)
    trainer.force_portable_tokenizer_config(dst)
    # Restore the live training wrapper exactly.
    emb.weight.data.copy_(original_input)
    decoder.weight = emb.weight
    model.base.config.tie_word_embeddings = True


class AdaptedDeBERTaForMLM(nn.Module):
    """DeBERTa-v2 MLM with an embedding adapter.
    
    Wraps a standard DebertaV2ForMaskedLM and inserts the adapter between
    the embedding lookup and the encoder. The adapter is disabled when
    adapter_mode is None.
    """
    def __init__(self, base_model: DebertaV2ForMaskedLM, adapter: Optional[EmbeddingAdapter]):
        super().__init__()
        self.base = base_model
        self.adapter = adapter
    
    def forward(self, input_ids, attention_mask, labels=None, adapter_mode: str = "on"):
        # Get raw embeddings
        emb = self.base.deberta.embeddings.word_embeddings(input_ids)
        # Apply adapter if present and enabled
        if self.adapter is not None and adapter_mode == "on":
            emb = self.adapter(emb, input_ids)
        # Run the full DeBERTa embedding pipeline (position embeddings,
        # LayerNorm, dropout and mask handling) from precomputed word embeddings.
        embedding_output = self.base.deberta.embeddings(
            input_ids=None, inputs_embeds=emb, mask=attention_mask,
        )
        encoder_outputs = self.base.deberta.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.base.cls(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type("ModelOutput", (), {"loss": loss, "logits": logits})()


# ── Main ─────────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--adapter_mode", choices=["baseline", "surface", "token_id"], default="baseline")
    p.add_argument("--max_word_exposure", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--tokenizer_path", default="", help="Local tokenizer dir; empty uses baseline repo")
    p.add_argument("--max_steps", type=int, default=0, help="Optional cap for smoke tests; 0 = full")
    p.add_argument("--micro_batch_size", type=int, default=64, help="Activation microbatch; masking stays on the full effective batch")
    return p.parse_args()


def main():
    args = build_args()
    start_time = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = trainer.make_portable_tokenizer(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"
    if tokenizer.mask_token is None:
        tokenizer.mask_token = "<mask>"
    
    # ── Data ──────────────────────────────────────────────────────────────
    raw_dir, manifest_files = trainer.download_dataset(
        argparse.Namespace(
            dataset_id="BabyLM-community/BabyLM-2026-Strict-Small",
            dataset_revision="c92ab16b4f08858304b0815706065b3354d8fc0a",
        ),
        out,
    )
    files = [raw_dir / n for n in trainer.TRAIN_FILES]
    total_words = sum(f["whitespace_words"] for f in manifest_files)
    pool_words = total_words
    selected_words = args.max_word_exposure
    
    pool_examples = list(trainer.iter_examples(files, pool_words, 160))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    pool_actual = sum(ex.words for ex in pool_examples)
    assert pool_actual == pool_words, f"pool word mismatch {pool_actual} vs {pool_words}"
    assert selected_words <= total_words * 10, f"exposure {selected_words} exceeds 10 epochs"
    
    examples: list = []
    actual_words = 0
    epoch = 0
    epoch_metadata = []
    while actual_words < selected_words:
        epoch_examples = list(pool_examples)
        shuffle_seed = args.seed + 1000003 * epoch
        random.Random(shuffle_seed).shuffle(epoch_examples)
        before = actual_words
        epoch_take_examples = 0
        for ex in epoch_examples:
            if actual_words >= selected_words:
                break
            source = f"epoch{epoch + 1}::{ex.source}"
            if actual_words + ex.words <= selected_words:
                examples.append(trainer.Example(ex.text, ex.words, example_id=ex.example_id, source=source))
                actual_words += ex.words
                epoch_take_examples += 1
            else:
                take = selected_words - actual_words
                if take > 0:
                    examples.append(trainer.Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=source))
                    actual_words += take
                    epoch_take_examples += 1
                break
        epoch_metadata.append({"epoch_index": epoch + 1, "shuffle_seed": shuffle_seed,
                               "words_added": actual_words - before, "examples_added": epoch_take_examples})
        epoch += 1
    
    assert actual_words == selected_words, f"word selection mismatch {actual_words} vs {selected_words}"
    
    # ── Model ──────────────────────────────────────────────────────────────
    trainer.reset_all_rng(args.seed)
    max_pos = 512
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=480,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=1920,
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
    )
    base_model = DebertaV2ForMaskedLM(cfg)
    
    adapter = None
    adapter_mode = args.adapter_mode
    if adapter_mode == "surface":
        surf_features = build_surface_features(tokenizer, max_features=2560)
        adapter = EmbeddingAdapter(hidden_size=480, feature_dim=2560, feature_matrix=surf_features)
    elif adapter_mode == "token_id":
        tid_features = build_token_id_features(len(tokenizer), max_features=2560)
        adapter = EmbeddingAdapter(hidden_size=480, feature_dim=2560, feature_matrix=tid_features)
    
    model = AdaptedDeBERTaForMLM(base_model, adapter)
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    adapter_params = adapter.extra_params() if adapter else 0
    base_params = param_count - adapter_params
    
    # ── Dataset and loader ─────────────────────────────────────────────────
    dataset = trainer.MaskedChunkDataset(examples, tokenizer, 256)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        collate_fn=trainer.collate, num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    
    # ── Optimizer ───────────────────────────────────────────────────────────
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-8)
    total_steps = len(loader)
    warmup = max(1, int(total_steps * 0.05))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)
    
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    
    # ── Training loop ───────────────────────────────────────────────────────
    log_path = out / "training_log.jsonl"
    cumulative_words = 0
    loss_values = []
    masked_token_values = []
    next_ckpt = args.checkpoint_words
    saved_checkpoints = []
    model.train()
    
    with log_path.open("w", encoding="utf-8") as logf:
        for step, batch in enumerate(loader, 1):
            if args.max_steps and step > args.max_steps:
                break
            words = int(batch.pop("words").sum().item())
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            word_group = batch["word_group"].to(device, non_blocking=True)
            
            masked_inputs, labels = trainer.apply_masking(
                input_ids, attention_mask, word_group, tokenizer, "wwm", 0.15, gen,
            )
            n_pred = int((labels != -100).sum().item())
            
            optim.zero_grad(set_to_none=True)
            micro = int(args.micro_batch_size or 0)
            if micro and 0 < micro < masked_inputs.shape[0]:
                total_tok = max(1, n_pred)
                loss_sum = 0.0
                for lo in range(0, masked_inputs.shape[0], micro):
                    hi = min(masked_inputs.shape[0], lo + micro)
                    mb_labels = labels[lo:hi]
                    mb_tok = int((mb_labels != -100).sum().item())
                    out_model = model(input_ids=masked_inputs[lo:hi], attention_mask=attention_mask[lo:hi], labels=mb_labels)
                    loss_mb = out_model.loss
                    if loss_mb is None:
                        raise RuntimeError("model returned no loss")
                    (loss_mb * max(1, mb_tok)).backward()
                    loss_sum += float(loss_mb.detach().cpu()) * max(1, mb_tok)
                    del out_model, loss_mb
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(float(total_tok))
                loss_float = loss_sum / float(total_tok)
            else:
                out_model = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
                loss = out_model.loss
                if loss is None:
                    raise RuntimeError("model returned no loss")
                loss.backward()
                loss_float = float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            current_lr = float(sched.get_last_lr()[0])
            
            cumulative_words += words
            loss_values.append(loss_float)
            masked_token_values.append(n_pred)
            
            rec = {
                "step": step, "loss": loss_float, "lr": current_lr,
                "batch_words": words, "cumulative_word_exposure": cumulative_words,
                "masked_tokens": n_pred, "elapsed_sec": time.time() - start_time,
            }
            if adapter is not None:
                rec["adapter_gate"] = float(adapter.gate.detach())
            logf.write(json.dumps(rec) + "\n")
            
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event": "train", **rec}), flush=True)
            
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = f"chck_{next_ckpt // 1_000_000}M"
                cp = out / "hf_model" / name
                save_fused_hf(model, tokenizer, cp)
                saved_checkpoints.append({
                    "name": name, "target_word_exposure": next_ckpt,
                    "actual_cumulative_word_exposure": cumulative_words, "path": str(cp),
                })
                print(json.dumps({"event": "checkpoint_saved", "name": name, "cum_words": cumulative_words}), flush=True)
                next_ckpt += args.checkpoint_words
    
    # Final save
    save_fused_hf(model, tokenizer, out / "hf_model")
    if not saved_checkpoints:
        cp = out / "hf_model" / "chck_1M"
        save_fused_hf(model, tokenizer, cp)
        saved_checkpoints.append({"name": "chck_1M", "target_word_exposure": args.checkpoint_words,
                                  "actual_cumulative_word_exposure": cumulative_words, "path": str(cp)})
    
    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics = {
        "adapter_mode": adapter_mode,
        "parameter_count": param_count,
        "base_parameter_count": base_params,
        "adapter_parameter_count": adapter_params,
        "word_exposure": cumulative_words,
        "loss_first": loss_values[0] if loss_values else None,
        "loss_last": loss_values[-1] if loss_values else None,
        "actual_training_steps": total_steps,
        "masked_tokens_total": sum(masked_token_values),
        "masked_tokens_per_whitespace_word": (sum(masked_token_values) / cumulative_words) if cumulative_words else None,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "batch_size": 256,
        "micro_batch_size": args.micro_batch_size,
        "seed": args.seed,
        "saved_checkpoints": saved_checkpoints,
        "adapter_gate_final": float(adapter.gate.detach()) if adapter else None,
    }
    (out / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event": "done", "adapter_mode": adapter_mode, "loss_first": metrics["loss_first"],
                      "loss_last": metrics["loss_last"], "word_exposure": cumulative_words,
                      "checkpoints": [c["name"] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()