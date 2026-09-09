#!/usr/bin/env python3
from __future__ import annotations

"""BabyLM masked-LM trainer with primary-objective relational XSpan masks.

This trainer preserves the trusted ordinary WWM stream from babylm_masked_train_fullcycle
and adds a counted XSpan stream where the *standard MLM head* predicts stored s2
semantic target spans.  There is no train-only auxiliary head.  The intended
mechanism is that selected s2 content-span likelihood becomes load-bearing for the
main MLM objective under true-s1 context.
"""

import argparse
import itertools
import json
import pathlib
import random
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import babylm_masked_train_fullcycle as base


@dataclass
class XSpanRow:
    row_id: int
    split: str
    target_type: str
    text: str
    target_span: tuple[int, int]
    words: int
    target_text: str


def load_xspan_rows(path: pathlib.Path, split: str, context: str, max_rows: int = 0) -> tuple[list[XSpanRow], dict]:
    if context not in {"true_s1", "wrong_s1", "no_s1"}:
        raise ValueError(f"unknown context {context}")
    rows: list[XSpanRow] = []
    split_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    counted_words = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            sp = str(obj.get("split", "train"))
            split_counts[sp] = split_counts.get(sp, 0) + 1
            if split != "all" and sp != split:
                continue
            if context == "true_s1":
                text = str(obj["text"])
                span = tuple(int(x) for x in obj["target_span_text"])
                words = int(obj.get("words", len(text.split())))
            elif context == "wrong_s1":
                text = str(obj["text_wrong_s1"])
                span = tuple(int(x) for x in obj["target_span_wrong_s1"])
                words = int(obj.get("words_wrong_s1", len(text.split())))
            else:
                text = str(obj["text_no_s1"])
                span = tuple(int(x) for x in obj["target_span_no_s1"])
                words = int(obj.get("words_no_s1", len(text.split())))
            ttype = str(obj.get("target_type", "unknown"))
            type_counts[ttype] = type_counts.get(ttype, 0) + 1
            rows.append(XSpanRow(
                row_id=int(obj.get("example_id", line_no - 1)),
                split=sp,
                target_type=ttype,
                text=text,
                target_span=(int(span[0]), int(span[1])),
                words=words,
                target_text=str(obj.get("target_text", "")),
            ))
            counted_words += words
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        raise RuntimeError(f"no XSpan rows loaded from {path} split={split} context={context}")
    meta = {
        "path": str(path), "split_requested": split, "context": context, "rows_loaded": len(rows),
        "split_counts_all_seen": split_counts, "type_counts_loaded": type_counts,
        "counted_words_once": counted_words, "max_rows": max_rows,
    }
    return rows, meta


class XSpanDataset(Dataset):
    def __init__(self, rows: list[XSpanRow], tokenizer, seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        r = self.rows[idx]
        enc = self.tokenizer(
            r.text, add_special_tokens=False, truncation=True, max_length=self.seq_length,
            padding="max_length", return_tensors="pt", return_offsets_mapping=True,
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        offsets = enc["offset_mapping"].squeeze(0)
        target_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        a0, b0 = r.target_span
        for i in range(input_ids.shape[0]):
            if int(attention_mask[i].item()) == 0:
                continue
            a = int(offsets[i, 0].item()); b = int(offsets[i, 1].item())
            if b > a0 and a < b0:
                target_mask[i] = True
        return {
            "input_ids": input_ids, "attention_mask": attention_mask, "target_mask": target_mask,
            "words": torch.tensor(r.words, dtype=torch.long), "row_id": torch.tensor(r.row_id, dtype=torch.long),
        }


def collate_xspan(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "target_mask": torch.stack([x["target_mask"] for x in batch]),
        "words": torch.stack([x["words"] for x in batch]),
        "row_id": torch.stack([x["row_id"] for x in batch]),
    }


def apply_xspan_masking(input_ids: torch.Tensor, attention_mask: torch.Tensor, target_mask: torch.Tensor, tokenizer, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = input_ids.device
    labels = input_ids.clone()
    special_ids = torch.tensor(sorted(tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & ~torch.isin(input_ids, special_ids)
    select = target_mask.bool() & candidate
    labels[~select] = -100
    masked_inputs = input_ids.clone()
    if select.sum() > 0:
        r = torch.rand(input_ids.shape, generator=gen, device=device)
        mask_tok = select & (r < 0.8)
        rand_tok = select & (r >= 0.8) & (r < 0.9)
        masked_inputs[mask_tok] = tokenizer.mask_token_id
        if rand_tok.any():
            rand_ids = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
            masked_inputs[rand_tok] = rand_ids
    tele = {
        "xspan_valid_rows": int(select.any(dim=1).sum().item()),
        "xspan_target_tokens": int(select.sum().item()),
        "xspan_batch_rows": int(input_ids.shape[0]),
    }
    return masked_inputs, labels, tele


@torch.no_grad()
def evaluate_xspan_contexts(model, tokenizer, rows_by_context: dict[str, list[XSpanRow]], seq_length: int, batch_size: int, max_batches: int, device) -> dict:
    model.eval()
    out: dict[str, dict] = {}
    for ctx, rows in rows_by_context.items():
        loader = DataLoader(XSpanDataset(rows, tokenizer, seq_length), batch_size=batch_size, shuffle=False, collate_fn=collate_xspan, num_workers=0)
        total_loss = 0.0; total_tok = 0; batches = 0; valid_rows = 0
        for batch in loader:
            if max_batches and batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_mask = batch["target_mask"].to(device)
            labels = input_ids.clone(); labels[~(target_mask.bool() & attention_mask.bool())] = -100
            if int((labels != -100).sum().item()) == 0:
                batches += 1; continue
            res = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ntok = int((labels != -100).sum().item())
            total_loss += float(res.loss.detach().cpu()) * ntok
            total_tok += ntok; valid_rows += int((labels != -100).any(dim=1).sum().item()); batches += 1
        out[ctx] = {"loss_per_token": total_loss / total_tok if total_tok else None, "target_tokens": total_tok, "valid_rows": valid_rows, "batches": batches}
    model.train()
    # Positive delta means true-s1 has lower loss/higher likelihood than wrong/no.
    if out.get("true_s1", {}).get("loss_per_token") is not None:
        t = out["true_s1"]["loss_per_token"]
        for ctx in ["wrong_s1", "no_s1"]:
            if out.get(ctx, {}).get("loss_per_token") is not None:
                out[f"delta_logprob_true_minus_{ctx.replace('_s1','')}"] = out[ctx]["loss_per_token"] - t
    return out


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", default="BabyLM-community/BabyLM-2026-Strict-Small")
    p.add_argument("--dataset_revision", default="c92ab16b4f08858304b0815706065b3354d8fc0a")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_word_exposure", type=int, default=1_000_000, help="ordinary WWM word exposure")
    p.add_argument("--example_pool_words", type=int, default=10_000_000)
    p.add_argument("--checkpoint_words", type=int, default=1_000_000)
    p.add_argument("--words_per_example", type=int, default=160)
    p.add_argument("--tokenizer_path", default="")
    p.add_argument("--tokenizer_label", default="baseline16k")
    p.add_argument("--tokenization_summary_limit", type=int, default=0)
    p.add_argument("--mask_mode", choices=["token", "wwm"], default="wwm")
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--seq_length", type=int, default=256)
    p.add_argument("--max_seq_length", type=int, default=256)
    p.add_argument("--max_position_embeddings", type=int, default=512)
    p.add_argument("--seq_len_schedule", default="")
    p.add_argument("--model_type", choices=["bert", "deberta_v2"], default="deberta_v2")
    p.add_argument("--position_buckets", type=int, default=256)
    p.add_argument("--max_relative_positions", type=int, default=256)
    p.add_argument("--deberta_pos_att_type", default="p2c,c2p")
    p.add_argument("--deberta_relative_attention", choices=["true", "false"], default="true")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_fraction", type=float, default=0.05)
    p.add_argument("--lr_total_steps", type=int, default=0)
    p.add_argument("--hidden_size", type=int, default=480)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra_init_seed", type=int, default=-1)
    p.add_argument("--train_rng_seed", type=int, default=-1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--xspan_jsonl", required=True)
    p.add_argument("--xspan_split", choices=["train", "heldout", "all"], default="train")
    p.add_argument("--xspan_context", choices=["true_s1", "wrong_s1", "no_s1"], default="true_s1")
    p.add_argument("--xspan_max_rows", type=int, default=0)
    p.add_argument("--xspan_batch_size", type=int, default=32)
    p.add_argument("--xspan_rho", type=float, default=0.15)
    p.add_argument("--xspan_every_n_steps", type=int, default=1)
    p.add_argument("--xspan_eval_max_batches", type=int, default=32)
    return p.parse_args()


def prepare_official_examples(args, out: pathlib.Path):
    pool_words = args.example_pool_words
    selected_words = args.max_word_exposure
    if pool_words < selected_words:
        pool_words = selected_words
    raw_dir, manifest_files = base.download_dataset(args, out)
    files = [raw_dir / n for n in base.TRAIN_FILES]
    total_words = sum(f["whitespace_words"] for f in manifest_files)
    if pool_words > total_words:
        pool_words = total_words
    pool_examples = list(base.iter_examples(files, pool_words, args.words_per_example))
    for i, ex in enumerate(pool_examples):
        ex.example_id = i
    pool_actual = sum(ex.words for ex in pool_examples)
    if pool_actual != pool_words:
        raise RuntimeError(f"pool word mismatch {pool_actual} vs {pool_words}")
    if selected_words > total_words * 10:
        raise RuntimeError(f"requested WWM exposure {selected_words} exceeds 10 official epochs ({total_words * 10})")
    examples: list[base.Example] = []
    actual_words = 0; epoch = 0; epoch_metadata = []
    while actual_words < selected_words:
        epoch_examples = list(pool_examples)
        shuffle_seed = args.seed + 1000003 * epoch
        random.Random(shuffle_seed).shuffle(epoch_examples)
        before = actual_words; epoch_take_examples = 0
        for ex in epoch_examples:
            if actual_words >= selected_words:
                break
            source = f"epoch{epoch + 1}::{ex.source}"
            if actual_words + ex.words <= selected_words:
                examples.append(base.Example(ex.text, ex.words, example_id=ex.example_id, source=source)); actual_words += ex.words; epoch_take_examples += 1
            else:
                take = selected_words - actual_words
                if take > 0:
                    examples.append(base.Example(" ".join(ex.text.split()[:take]), take, example_id=ex.example_id, source=source)); actual_words += take; epoch_take_examples += 1
                break
        epoch_metadata.append({"epoch_index": epoch + 1, "shuffle_seed": shuffle_seed, "words_added": actual_words - before, "examples_added": epoch_take_examples})
        epoch += 1
    return examples, actual_words, pool_words, total_words, manifest_files, {"data_source_type":"official_corpus_fullcycle_with_xspan_primary_objective", "selection_epochs": epoch_metadata, "unique_official_pool_words": pool_words, "total_official_corpus_words": total_words}


def main() -> None:
    args = build_args(); start_time = time.time()
    if not (0.0 <= args.xspan_rho <= 1.0):
        raise RuntimeError("--xspan_rho must be in [0,1]")
    out = pathlib.Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = base.make_portable_tokenizer(args.tokenizer_path)
    examples, actual_words, pool_words, total_words, manifest_files, selection_meta = prepare_official_examples(args, out)
    source_words: dict[str, int] = {}
    for ex in examples:
        source_words[ex.source] = source_words.get(ex.source, 0) + ex.words
    (out/"example_order_manifest.json").write_text(json.dumps({"seed":args.seed,"example_pool_words_actual":pool_words,"selected_for_training_words":actual_words,"words_per_example":args.words_per_example,"num_consumed_examples":len(examples),"source_words_consumed":source_words,"consumed_example_ids_in_order":[ex.example_id for ex in examples],"mask_mode":args.mask_mode,"mask_prob":args.mask_prob,"tokenizer_label":args.tokenizer_label,"tokenizer_path":args.tokenizer_path,"tokenizer_vocab_size":len(tokenizer),**selection_meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    tokenization_summary = base.summarize_tokenization_coupling(examples, tokenizer, args.max_seq_length, args.tokenization_summary_limit)
    (out/"tokenization_coupling_summary.json").write_text(json.dumps(tokenization_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    wwm_loader = DataLoader(base.MaskedChunkDataset(examples, tokenizer, args.max_seq_length), batch_size=args.batch_size, shuffle=False, collate_fn=base.collate, num_workers=2, pin_memory=torch.cuda.is_available())
    train_rows, train_meta = load_xspan_rows(pathlib.Path(args.xspan_jsonl), args.xspan_split, args.xspan_context, args.xspan_max_rows)
    xspan_loader = DataLoader(XSpanDataset(train_rows, tokenizer, args.max_seq_length), batch_size=args.xspan_batch_size, shuffle=True, collate_fn=collate_xspan, num_workers=0, pin_memory=torch.cuda.is_available())
    xspan_iter = itertools.cycle(xspan_loader)
    eval_rows_by_context = {ctx: load_xspan_rows(pathlib.Path(args.xspan_jsonl), "heldout", ctx, 0)[0] for ctx in ["true_s1", "wrong_s1", "no_s1"]}
    if args.extra_init_seed >= 0:
        base.reset_all_rng(args.extra_init_seed)
    if args.max_position_embeddings < args.max_seq_length + 8:
        args.max_position_embeddings = args.max_seq_length + 8
    model = base.build_model(args, tokenizer)
    if args.train_rng_seed >= 0:
        base.reset_all_rng(args.train_rng_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    input_embedding_params = model.get_input_embeddings().weight.numel()
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9,0.98))
    total_steps = len(wwm_loader)
    schedule_total = args.lr_total_steps if args.lr_total_steps and args.lr_total_steps > 0 else total_steps
    if schedule_total < total_steps:
        raise RuntimeError(f"lr_total_steps {schedule_total} < actual steps {total_steps}")
    warmup = max(1, int(schedule_total * args.warmup_fraction))
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=schedule_total)
    seq_schedule=[]
    if args.seq_len_schedule:
        for part in args.seq_len_schedule.split(','):
            t,L=part.split(':'); seq_schedule.append((float(t), int(L)))
        seq_schedule.sort()
    gen = torch.Generator(device=device); gen.manual_seed(args.train_rng_seed if args.train_rng_seed >= 0 else args.seed)
    model.train()
    cumulative_words=0; cumulative_xspan_words=0; saved_checkpoints=[]
    loss_values=[]; mlm_loss_values=[]; xspan_loss_values=[]; masked_token_values=[]; xspan_target_tokens=[]; xspan_valid_rows=[]; seq_len_values=[]
    next_ckpt = args.checkpoint_words if args.checkpoint_words and args.checkpoint_words > 0 else None
    log_path=out/"training_log.jsonl"
    with log_path.open('w', encoding='utf-8') as logf:
        for step,batch in enumerate(wwm_loader,1):
            words=int(batch.pop('words').sum().item())
            input_ids=batch['input_ids'].to(device, non_blocking=True); attention_mask=batch['attention_mask'].to(device, non_blocking=True); word_group=batch['word_group'].to(device, non_blocking=True)
            frac=(research)/max(1,schedule_total)
            cur_len=base.seq_length_for_progress(frac, seq_schedule, args.seq_length) if seq_schedule else args.seq_length
            cur_len=min(cur_len, args.max_seq_length)
            input_ids=input_ids[:,:cur_len].contiguous(); attention_mask=attention_mask[:,:cur_len].contiguous(); word_group=word_group[:,:cur_len].contiguous()
            masked_inputs, labels = base.apply_masking(input_ids, attention_mask, word_group, tokenizer, args.mask_mode, args.mask_prob, gen)
            optim.zero_grad(set_to_none=True)
            out_wwm=model(input_ids=masked_inputs, attention_mask=attention_mask, labels=labels)
            mlm_loss=out_wwm.loss
            if mlm_loss is None:
                raise RuntimeError('no WWM MLM loss')
            x_loss=mlm_loss.new_tensor(0.0); xtele={"xspan_valid_rows":0,"xspan_target_tokens":0,"xspan_batch_rows":0,"xspan_words":0}
            if args.xspan_rho > 0 and args.xspan_every_n_steps > 0 and step % args.xspan_every_n_steps == 0:
                xb=next(xspan_iter)
                xid=xb['input_ids'].to(device, non_blocking=True); xatt=xb['attention_mask'].to(device, non_blocking=True); xtm=xb['target_mask'].to(device, non_blocking=True)
                xmasked,xlabels,xtele0=apply_xspan_masking(xid,xatt,xtm,tokenizer,gen)
                if int((xlabels != -100).sum().item()) > 0:
                    xout=model(input_ids=xmasked, attention_mask=xatt, labels=xlabels)
                    x_loss=xout.loss
                xtele.update(xtele0); xtele['xspan_words']=int(xb['words'].sum().item())
                cumulative_xspan_words += xtele['xspan_words']
            loss=(1.0-float(args.xspan_rho))*mlm_loss + float(args.xspan_rho)*x_loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step(); sched.step()
            cumulative_words += words
            loss_values.append(float(loss.detach().cpu())); mlm_loss_values.append(float(mlm_loss.detach().cpu())); xspan_loss_values.append(float(x_loss.detach().cpu()))
            n_pred=int((labels != -100).sum().item()); masked_token_values.append(n_pred); seq_len_values.append(cur_len); xspan_target_tokens.append(xtele['xspan_target_tokens']); xspan_valid_rows.append(xtele['xspan_valid_rows'])
            rec={"step":step,"loss":loss_values[-1],"loss_wwm":mlm_loss_values[-1],"loss_xspan":xspan_loss_values[-1],"lr":float(sched.get_last_lr()[0]),"batch_wwm_words":words,"cumulative_wwm_word_exposure":cumulative_words,"cumulative_xspan_word_exposure":cumulative_xspan_words,"cumulative_total_counted_words":cumulative_words+cumulative_xspan_words,"seq_len":cur_len,"masked_tokens":n_pred,"elapsed_sec":time.time()-start_time,**xtele}
            logf.write(json.dumps(rec)+'\n')
            if step == 1 or step % args.log_every == 0 or step == total_steps:
                print(json.dumps({"event":"train",**rec}), flush=True)
            while next_ckpt is not None and cumulative_words >= next_ckpt and next_ckpt <= args.max_word_exposure:
                name = "chck_1M" if next_ckpt < 1_000_000 else (f"chck_{next_ckpt//1_000_000}M" if next_ckpt % 1_000_000 == 0 else f"chck_{next_ckpt}w")
                cp=out/"hf_model"/name; base.save_hf_checkpoint(model, tokenizer, cp)
                saved_checkpoints.append({"name":name,"target_wwm_word_exposure":next_ckpt,"actual_cumulative_wwm_word_exposure":cumulative_words,"actual_cumulative_total_counted_words":cumulative_words+cumulative_xspan_words,"path":str(cp)})
                print(json.dumps({"event":"checkpoint_saved","name":name,"cum_wwm_words":cumulative_words,"cum_total_words":cumulative_words+cumulative_xspan_words}), flush=True)
                next_ckpt += args.checkpoint_words
    base.save_hf_checkpoint(model, tokenizer, out/"hf_model")
    if not saved_checkpoints:
        cp=out/"hf_model"/"chck_1M"; base.save_hf_checkpoint(model, tokenizer, cp)
        saved_checkpoints.append({"name":"chck_1M","target_wwm_word_exposure":args.checkpoint_words,"actual_cumulative_wwm_word_exposure":cumulative_words,"actual_cumulative_total_counted_words":cumulative_words+cumulative_xspan_words,"path":str(cp)})
    xeval=evaluate_xspan_contexts(model, tokenizer, eval_rows_by_context, args.max_seq_length, args.xspan_batch_size, args.xspan_eval_max_batches, device)
    metrics={
        "variant":f"masked_{args.mask_mode}_primary_xspan", "backend":"mlm", "model_family":model.__class__.__name__, "model_type":args.model_type,
        "parameter_count":param_count, "embedding_parameter_count":input_embedding_params, "non_embedding_parameter_count":param_count-input_embedding_params,
        "vocab_size":len(tokenizer), "tokenizer_label":args.tokenizer_label, "tokenizer_path":args.tokenizer_path,
        "wwm_word_exposure":cumulative_words, "xspan_word_exposure":cumulative_xspan_words, "total_counted_word_exposure":cumulative_words+cumulative_xspan_words,
        "xspan_rho":args.xspan_rho, "xspan_context":args.xspan_context, "xspan_jsonl":args.xspan_jsonl,
        "loss_first":loss_values[0] if loss_values else None, "loss_last":loss_values[-1] if loss_values else None,
        "loss_wwm_first":mlm_loss_values[0] if mlm_loss_values else None, "loss_wwm_last":mlm_loss_values[-1] if mlm_loss_values else None,
        "loss_xspan_first":xspan_loss_values[0] if xspan_loss_values else None, "loss_xspan_last":xspan_loss_values[-1] if xspan_loss_values else None,
        "xspan_eval":xeval, "xspan_target_tokens_total":sum(xspan_target_tokens), "xspan_valid_rows_total":sum(xspan_valid_rows),
        "example_pool_words_actual":pool_words, "selected_for_training_words":actual_words, "lr_schedule_total_steps":schedule_total, "actual_training_steps":total_steps,
        "mask_mode":args.mask_mode, "mask_prob":args.mask_prob, "masked_tokens_total":sum(masked_token_values), "masked_tokens_mean_per_step":sum(masked_token_values)/len(masked_token_values) if masked_token_values else None,
        "masked_tokens_per_wwm_whitespace_word":sum(masked_token_values)/cumulative_words if cumulative_words else None,
        "unique_training_seq_lengths":sorted(set(seq_len_values)), "tokenization_coupling_summary":tokenization_summary,
        "seq_length":args.seq_length, "max_seq_length":args.max_seq_length, "max_position_embeddings":max(args.max_position_embeddings,args.max_seq_length+8),
        "hidden_size":args.hidden_size, "n_layer":args.n_layer, "n_head":args.n_head, "ffn_mult":args.ffn_mult,
        "position_buckets":args.position_buckets, "max_relative_positions":args.max_relative_positions, "deberta_relative_attention":args.deberta_relative_attention, "deberta_pos_att_type":args.deberta_pos_att_type,
        "seed":args.seed, "extra_init_seed":args.extra_init_seed, "train_rng_seed":args.train_rng_seed, "saved_checkpoints":saved_checkpoints,
        "source_words_consumed":source_words, "xspan_train_meta":train_meta, **selection_meta,
    }
    (out/"scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (out/"data_manifest.json").write_text(json.dumps({"dataset_id":args.dataset_id,"dataset_revision":args.dataset_revision,"files":manifest_files,"selected_for_wwm_stream_whitespace_words":actual_words,"xspan_jsonl":args.xspan_jsonl,"xspan_context":args.xspan_context,"xspan_train_meta":train_meta,"total_counted_word_exposure":cumulative_words+cumulative_xspan_words,"model_type":args.model_type,"tokenizer_label":args.tokenizer_label,"tokenizer_path":args.tokenizer_path,"tokenizer_vocab_size":len(tokenizer),"tokenization_coupling_summary_file":str(out/"tokenization_coupling_summary.json"),**selection_meta}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"event":"done","param_count":param_count,"loss_first":metrics['loss_first'],"loss_last":metrics['loss_last'],"wwm_word_exposure":cumulative_words,"xspan_word_exposure":cumulative_xspan_words,"total_counted":cumulative_words+cumulative_xspan_words,"xspan_eval":xeval,"checkpoints":[c['name'] for c in saved_checkpoints]}), flush=True)


if __name__ == "__main__":
    main()
