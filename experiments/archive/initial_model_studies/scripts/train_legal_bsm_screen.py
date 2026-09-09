#!/usr/bin/env python3
"""research: Legal BSM trainer — metadata-aware masked LM for binding rows.

Extends the official-compatible DeBERTa-v2 WWM trainer so that:
  - Official rows: standard whole-word masking at 15%
  - Binding rows: mask specifically the answer word identified by metadata
    (mask_char_start, mask_char_end) through the standard MLM head

This produces a legal from-scratch model where binding signal comes from
targeted masking of corpus text, not from added exposure or external models.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, random, sys, time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerFast,
    DebertaV2Config, DebertaV2ForMaskedLM,
    get_cosine_schedule_with_warmup,
)

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
sys.path.insert(0, str((ROOT / 'training/scripts').resolve()))
from babylm_masked_train_fullcycle import (
    make_portable_tokenizer, force_portable_tokenizer_config,
    save_hf_checkpoint, is_word_start, reset_all_rng,
)

CORPUS_DIR = ROOT / 'data/legal_bsm_corpus'
OUT_DIR = ROOT / 'training/runs/legal_bsm_smoke'


def env_setup():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['HF_DATASETS_CACHE'] = str((hf / 'datasets').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


@dataclass
class CorpusRow:
    text: str
    words: int
    kind: str  # "binding" or "official"
    mask_char_start: int
    mask_char_end: int
    correct_tid: int
    distractor_tid: int
    source: str


def load_corpus_jsonl(path: pathlib.Path) -> List[CorpusRow]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(CorpusRow(
                text=obj['text'], words=obj['words'], kind=obj.get('kind', 'official'),
                mask_char_start=obj.get('mask_char_start', -1),
                mask_char_end=obj.get('mask_char_end', -1),
                correct_tid=obj.get('correct_tid', -1),
                distractor_tid=obj.get('distractor_tid', -1),
                source=obj.get('source', 'unknown'),
            ))
    return rows


class BSMDataset(Dataset):
    """Dataset that preserves binding metadata alongside tokenized sequences."""

    def __init__(self, rows: List[CorpusRow], tokenizer, seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.special_ids = set(tokenizer.all_special_ids)
        self._word_start = {}

    def _is_word_start(self, token_id: int) -> bool:
        v = self._word_start.get(token_id)
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(token_id))
            v = bool(s is not None and is_word_start(str(s)))
            self._word_start[token_id] = v
        return v

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(
            row.text, add_special_tokens=False, truncation=True,
            max_length=self.seq_length, padding='max_length', return_tensors='pt',
            return_offsets_mapping=True,
        )
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        offsets = enc['offset_mapping'].squeeze(0)  # (seq_len, 2)

        # Build word groups for WWM
        word_group = torch.full_like(input_ids, -1)
        gid = -1
        for i in range(input_ids.shape[0]):
            if attention_mask[i] == 0:
                continue
            tid = int(input_ids[i])
            if tid in self.special_ids:
                continue
            if gid < 0 or self._is_word_start(tid) or i == 0:
                gid += 1
            word_group[i] = gid

        # Targeted rows (binding OR official_targeted): identify token positions of the answer span.
        # Any row that supplies a valid targeted span is trained with targeted masking, so a
        # matched official reference can reproduce the BSM per-batch target opportunity.
        is_targeted = (row.kind in ('binding', 'official_targeted')) and row.mask_char_start >= 0
        binding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if is_targeted:
            for i in range(input_ids.shape[0]):
                if attention_mask[i] == 0:
                    continue
                start, end = int(offsets[i, 0]), int(offsets[i, 1])
                if end <= 0:
                    continue
                # Token overlaps with the answer character span
                if start < row.mask_char_end and end > row.mask_char_start:
                    binding_mask[i] = True

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'word_group': word_group,
            'binding_mask': binding_mask,
            'is_binding': is_targeted,
            'words': row.words,
        }


def collate_fn(batch):
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'word_group': torch.stack([b['word_group'] for b in batch]),
        'binding_mask': torch.stack([b['binding_mask'] for b in batch]),
        'is_binding': torch.tensor([b['is_binding'] for b in batch], dtype=torch.bool),
        'words': torch.tensor([b['words'] for b in batch]),
    }


def apply_masking(input_ids, attention_mask, word_group, binding_mask, is_binding,
                  tokenizer, mask_prob, gen, device):
    """Apply targeted masking for binding rows and standard WWM for official rows."""
    bsz, seq = input_ids.shape
    mask_token_id = tokenizer.mask_token_id
    vocab_size = len(tokenizer)

    labels = input_ids.clone()
    masked_inputs = input_ids.clone()
    select = torch.zeros(bsz, seq, dtype=torch.bool, device=device)

    for b in range(bsz):
        if is_binding[b]:
            # For binding rows: mask specifically the answer tokens
            select[b] = binding_mask[b]
        else:
            # For official rows: standard WWM at mask_prob
            candidate = (attention_mask[b] == 1) & (word_group[b] >= 0)
            if not candidate.any():
                continue
            # Select word groups to mask
            unique_groups = word_group[b][candidate].unique()
            n_groups_to_mask = max(1, int(round(len(unique_groups) * mask_prob)))
            perm = torch.randperm(len(unique_groups), generator=gen, device=device)
            selected_groups = unique_groups[perm[:n_groups_to_mask]]
            for g in selected_groups:
                select[b] |= (word_group[b] == g)

    # Apply masking: 80% [MASK], 10% random, 10% keep
    r = torch.rand(bsz, seq, generator=gen, device=device)
    mask_tok = select & (r < 0.8)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    masked_inputs[mask_tok] = mask_token_id
    if rand_tok.any():
        rand_ids = torch.randint(0, vocab_size, (int(rand_tok.sum()),), generator=gen, device=device)
        masked_inputs[rand_tok] = rand_ids

    labels[~select] = -100
    return masked_inputs, labels


def build_model(tokenizer, args):
    """Build DeBERTa-v2 from scratch (protected 8×480 config)."""
    pos_att_type = ['p2c', 'c2p']
    cfg = DebertaV2Config(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        intermediate_size=args.hidden_size * args.ffn_mult,
        max_position_embeddings=args.max_position_embeddings,
        position_buckets=args.position_buckets,
        relative_attention=True,
        pos_att_type=pos_att_type,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True, help='Path to JSONL corpus')
    ap.add_argument('--output_dir', default=str(OUT_DIR))
    ap.add_argument('--max_word_exposure', type=int, default=100000)
    ap.add_argument('--seq_length', type=int, default=128)
    ap.add_argument('--max_position_embeddings', type=int, default=512)
    ap.add_argument('--hidden_size', type=int, default=480)
    ap.add_argument('--n_layer', type=int, default=8)
    ap.add_argument('--n_head', type=int, default=8)
    ap.add_argument('--ffn_mult', type=int, default=4)
    ap.add_argument('--position_buckets', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--warmup_fraction', type=float, default=0.05)
    ap.add_argument('--mask_prob', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--checkpoint_words', type=int, default=100000)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--no_shuffle', action='store_true', help='Preserve JSONL file order in DataLoader; required for curriculum/order experiments')
    args = ap.parse_args()

    env_setup()
    t0 = time.time()
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reset_all_rng(args.seed)

    tokenizer = make_portable_tokenizer("")
    corpus_path = pathlib.Path(args.corpus)
    rows = load_corpus_jsonl(corpus_path)

    # Trim to max_word_exposure
    selected = []
    cum_words = 0
    for row in rows:
        if cum_words + row.words > args.max_word_exposure:
            break
        selected.append(row)
        cum_words += row.words

    n_binding = sum(1 for r in selected if r.kind == 'binding')
    n_official = sum(1 for r in selected if r.kind == 'official')
    binding_words = sum(r.words for r in selected if r.kind == 'binding')
    print(json.dumps({
        'event': 'data_loaded', 'corpus': str(corpus_path),
        'total_rows': len(selected), 'binding_rows': n_binding, 'official_rows': n_official,
        'total_words': cum_words, 'binding_words': binding_words,
        'shuffle': (not args.no_shuffle),
    }), flush=True)

    ds = BSMDataset(selected, tokenizer, args.seq_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(not args.no_shuffle), collate_fn=collate_fn,
                        num_workers=2, pin_memory=torch.cuda.is_available())

    model = build_model(tokenizer, args).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(json.dumps({'event': 'model_built', 'params': param_count}), flush=True)

    total_steps = len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    warmup = max(1, int(total_steps * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    model.train()
    logs = []
    cum_words_trained = 0
    saved_checkpoints = []
    next_ckpt = args.checkpoint_words

    for step, batch in enumerate(loader, 1):
        words = int(batch['words'].sum().item())
        cum_words_trained += words

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        word_group = batch['word_group'].to(device)
        binding_mask = batch['binding_mask'].to(device)
        is_binding = batch['is_binding'].to(device)

        masked_inputs, labels = apply_masking(
            input_ids, attention_mask, word_group, binding_mask, is_binding,
            tokenizer, args.mask_prob, gen, device
        )

        outputs = model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Separate binding vs official loss for diagnostics
        with torch.no_grad():
            bind_select = is_binding.unsqueeze(1).expand_as(labels)
            bind_labels = labels.clone()
            bind_labels[~bind_select] = -100
            off_labels = labels.clone()
            off_labels[bind_select] = -100
            bind_targets = (bind_labels != -100).sum().item()
            off_targets = (off_labels != -100).sum().item()

        rec = {
            'step': step, 'cum_words': cum_words_trained,
            'loss': float(loss.detach().cpu()),
            'lr': float(sched.get_last_lr()[0]),
            'bind_targets': bind_targets, 'off_targets': off_targets,
        }
        logs.append(rec)
        if step <= 3 or step % args.log_every == 0 or step == total_steps:
            print(json.dumps({'event': 'train', **rec}), flush=True)

        # Checkpoint
        while next_ckpt is not None and cum_words_trained >= next_ckpt:
            # Use exact k-word checkpoint names to avoid overwriting quarter-M checkpoints.
            name = f"chck_{next_ckpt // 1000}k"
            cp = out / 'hf_model' / name
            save_hf_checkpoint(model, tokenizer, cp)
            saved_checkpoints.append({'name': name, 'words': cum_words_trained})
            print(json.dumps({'event': 'checkpoint', 'name': name, 'words': cum_words_trained}), flush=True)
            next_ckpt += args.checkpoint_words

    # Final save
    save_hf_checkpoint(model, tokenizer, out / 'hf_model')

    metrics = {
        'status': 'LEGAL_BSM_SMOKE_DONE',
        'corpus': str(corpus_path),
        'total_words_trained': cum_words_trained,
        'binding_words': binding_words,
        'binding_fraction': binding_words / max(1, cum_words_trained),
        'total_steps': total_steps,
        'param_count': param_count,
        'loss_first': logs[0]['loss'] if logs else None,
        'loss_last': logs[-1]['loss'] if logs else None,
        'saved_checkpoints': saved_checkpoints,
        'shuffle': (not args.no_shuffle),
        'elapsed_sec': time.time() - t0,
    }
    (out / 'scientific_metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'event': 'done', **{k: v for k, v in metrics.items() if k != 'saved_checkpoints'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
