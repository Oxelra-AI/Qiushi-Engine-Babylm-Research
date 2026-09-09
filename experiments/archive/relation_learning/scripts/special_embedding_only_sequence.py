#!/usr/bin/env python3
"""research: special-token embedding-row-only attribution test.

This tests whether the official special-token mismatch is useful when the instrument is
restricted to the two special input rows rather than a full private branch.  The model
starts from the protected chck_82M slow-adapter checkpoint, trains on the exact
coherent86 suffix rows with add_special_tokens=True, and after every optimizer step
restores all embedding rows except <s> and </s>.  DeBERTa ties input embeddings and
LM decoder weights, so the saved checkpoint also changes the two tied decoder rows;
that caveat is recorded in the endpoint metrics.
"""
from __future__ import annotations

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative


import argparse
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

ROOT = _public_path('.')
CHCK82 = _public_path('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M')
STREAM = _public_path('experiments/archive/relation_learning/data/format_control_streams/coherent_unsplit_replay_3992800w.jsonl')
EVALUATOR = _public_path('experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py')
OUT_ROOT = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control')
WAIT_ON = _public_path('experiments/archive/relation_learning/data/coherent_no_special_control/eval/sequence_summary.json')
WORDS_PER_UPDATE = 3_992_800 // 101
LR_TOTAL_STEPS = 455
WARMUP_STEPS = 27
INITIAL_CONSUMED_WORDS = 82_012_495
CHEAP = ["BLiMP", "Supplement", "EWoK", "Entity", "COMPS", "GlobalPIQA", "Reading"]


def rel(p: pathlib.Path | str) -> str:
    try:
        return str(pathlib.Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_path(path: pathlib.Path, timeout: float, interval: float) -> None:
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"waited {timeout}s for {path}")
        time.sleep(interval)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_word_start(tok: str) -> bool:
    return tok.startswith("Ġ") or tok.startswith("▁")


class WordGrouper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special_ids = set(int(x) for x in tokenizer.all_special_ids)
        self.cache: dict[int, bool] = {}

    def word_start(self, tid: int) -> bool:
        v = self.cache.get(int(tid))
        if v is None:
            s = self.tokenizer.convert_ids_to_tokens(int(tid))
            v = bool(s is not None and is_word_start(str(s)))
            self.cache[int(tid)] = v
        return v

    def tokenize(self, text: str, max_length: int = 256):
        enc = self.tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
        ids = [int(x) for x in enc["input_ids"]]
        groups = [-1] * len(ids)
        gid = -1
        first_non_special = False
        for i, tid in enumerate(ids):
            if tid in self.special_ids:
                continue
            if gid < 0 or self.word_start(tid) or not first_non_special:
                gid += 1
            groups[i] = gid
            first_non_special = True
        return ids, groups


def load_rows(path: pathlib.Path, max_words: int = 0) -> list[dict[str, Any]]:
    rows = []
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            words = int(obj.get("words", len(str(obj.get("text", "")).split())))
            if max_words and rows and total + words > max_words:
                break
            rows.append({"text": str(obj.get("text", "")), "words": words, "source": str(obj.get("source", ""))})
            total += words
            if max_words and total >= max_words:
                break
    return rows


def collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    mx = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), mx), int(pad_id), dtype=torch.long)
    att = torch.zeros((len(batch), mx), dtype=torch.long)
    wg = torch.full((len(batch), mx), -1, dtype=torch.long)
    words = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"], dtype=torch.long)
        att[i, :L] = 1
        wg[i, :L] = torch.tensor(b["wg"], dtype=torch.long)
        words[i] = int(b["words"])
    return {"input_ids": ids, "attention_mask": att, "word_group": wg, "words": words}


def apply_wwm(input_ids, attention_mask, word_group, tokenizer, mask_prob: float, gen: torch.Generator):
    device = input_ids.device
    labels = input_ids.clone()
    special = torch.tensor(sorted(int(x) for x in tokenizer.all_special_ids), device=device)
    candidate = attention_mask.bool() & (word_group >= 0) & ~torch.isin(input_ids, special)
    select = torch.zeros_like(candidate)
    for b in range(input_ids.shape[0]):
        valid = torch.unique(word_group[b][word_group[b] >= 0])
        if valid.numel() == 0:
            continue
        chosen = valid[torch.rand(valid.numel(), generator=gen, device=device) < mask_prob]
        if chosen.numel() > 0:
            select[b] = torch.isin(word_group[b], chosen) & candidate[b]
    if select.sum() == 0:
        flat = candidate.view(-1).nonzero(as_tuple=False)
        if flat.numel() > 0:
            select.view(-1)[int(flat[0, 0])] = True
    labels[~select] = -100
    masked = input_ids.clone()
    r = torch.rand(input_ids.shape, generator=gen, device=device)
    masked[select & (r < 0.8)] = int(tokenizer.mask_token_id)
    rand_tok = select & (r >= 0.8) & (r < 0.9)
    if rand_tok.any():
        masked[rand_tok] = torch.randint(0, len(tokenizer), (int(rand_tok.sum()),), generator=gen, device=device)
    return masked, labels, select, candidate


def build_macro_batches(tokenized: list[dict[str, Any]], words_per_update: int):
    batches = []
    cur = []
    words = 0
    for t in tokenized:
        cur.append(t)
        words += int(t["words"])
        if words >= words_per_update:
            batches.append(cur)
            cur = []
            words = 0
    if cur:
        batches.append(cur)
    return batches


def lr_at(research: int, peak: float) -> float:
    if research < WARMUP_STEPS:
        return peak * research / max(1, WARMUP_STEPS)
    progress = (research - WARMUP_STEPS) / max(1, LR_TOTAL_STEPS - WARMUP_STEPS)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_model_tokenizer(endpoint: pathlib.Path, device: torch.device, cache_root: pathlib.Path):
    for k, p in {"HF_HOME": cache_root / "hf_home", "TRANSFORMERS_CACHE": cache_root / "transformers", "HF_MODULES_CACHE": cache_root / "modules"}.items():
        p.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(p)
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(endpoint), trust_remote_code=True, local_files_only=True)
    model.to(device)
    model.train()
    return model, tok


def train_one(seed: int, gpu: int, force: bool = False) -> pathlib.Path:
    out = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/train') / f"seed{seed}"
    if force and out.exists():
        shutil.rmtree(out)
    if (out / "summary.json").exists() and (out / "checkpoint" / "model.safetensors").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    model, tok = load_model_tokenizer(CHCK82, device, out / "hf_cache")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.deberta.embeddings.word_embeddings.weight
    emb.requires_grad_(True)
    original_emb = emb.detach().clone()
    special_ids = [int(tok.bos_token_id), int(tok.eos_token_id)]
    if any(x is None or x < 0 for x in special_ids):
        raise RuntimeError(f"missing bos/eos ids: {special_ids}")
    grad_mask = torch.zeros_like(emb, device=device)
    for sid in special_ids:
        grad_mask[sid].fill_(1.0)

    rows = load_rows(STREAM, max_words=3_992_800)
    grouper = WordGrouper(tok)
    tokenized = []
    for r in rows:
        ids, wg = grouper.tokenize(r["text"])
        tokenized.append({"ids": ids, "wg": wg, "words": int(r["words"])})
    batches = build_macro_batches(tokenized, WORDS_PER_UPDATE)
    optimizer = torch.optim.AdamW([emb], lr=0.001, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 17)
    pad = int(tok.pad_token_id)
    logs = []
    t0 = time.time()
    cum_words = 0
    with (out / "training_log.jsonl").open("w", encoding="utf-8") as logf:
        for ui, macro in enumerate(batches):
            lr = lr_at(ui, 0.001)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            macro_words = sum(int(t["words"]) for t in macro)
            batch = collate(macro, pad)
            ids = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            wg = batch["word_group"].to(device)
            masked, labels, _select, cand = apply_wwm(ids, att, wg, tok, 0.15, gen)
            out_main = model(input_ids=masked, attention_mask=att)
            vocab = out_main.logits.shape[-1]
            n_targets = int((labels != -100).sum().item())
            loss = F.cross_entropy(out_main.logits.reshape(-1, vocab), labels.reshape(-1), ignore_index=-100, reduction="sum") / max(1, n_targets)
            loss.backward()
            if emb.grad is not None:
                emb.grad.mul_(grad_mask)
            torch.nn.utils.clip_grad_norm_([emb], 1.0)
            optimizer.step()
            with torch.no_grad():
                restore_mask = 1.0 - grad_mask
                emb.mul_(grad_mask).add_(original_emb * restore_mask)
            cum_words += macro_words
            rec = {
                "update": ui + 1,
                "lr": lr,
                "macro_words": macro_words,
                "cum_words": cum_words,
                "macro_rows": len(macro),
                "main_targets": n_targets,
                "non_special_tokens": int(cand.sum().item()),
                "target_ratio": n_targets / max(1, int(cand.sum().item())),
                "main_ce": float(loss.detach().cpu()),
                "special_row_l2_mean": float((emb.detach()[special_ids] - original_emb[special_ids]).float().norm(dim=1).mean().cpu()),
                "nonspecial_abs_max": float((emb.detach() * (1.0 - grad_mask) - original_emb * (1.0 - grad_mask)).abs().max().cpu()),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            logs.append(rec)
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            if ui == 0 or (ui + 1) % 10 == 0:
                print(json.dumps({"event": "train", "seed": seed, **rec}), flush=True)
            del ids, att, wg, masked, labels, out_main, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    ckpt = out / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt), safe_serialization=True)
    tok.save_pretrained(str(ckpt))
    summary = {
        "status": "SPECIAL_EMBEDDING_ONLY_TRAIN_DONE",
        "created_utc": now(),
        "seed": seed,
        "checkpoint": rel(ckpt),
        "updates": len(logs),
        "total_words": cum_words,
        "special_ids": special_ids,
        "special_tokens": [tok.convert_ids_to_tokens(i) for i in special_ids],
        "trainable_object": "deberta.embeddings.word_embeddings.weight rows for <s> and </s>; all other rows restored after every optimizer step",
        "tied_decoder_caveat": "The DeBERTa MLM head ties decoder weights to input embeddings, so the saved checkpoint also changes these two output rows.",
        "first_update": logs[0] if logs else None,
        "last_update": logs[-1] if logs else None,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def read_training_log(path: pathlib.Path) -> list[dict[str, Any]]:
    rows=[]
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip(): rows.append(json.loads(line))
    return rows


def make_runlike(train_dir: pathlib.Path, label: str) -> pathlib.Path:
    summary = read_json(train_dir / "summary.json")
    logs = read_training_log(train_dir / "training_log.jsonl")
    runlike = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/runlike') / label
    hf_final = runlike / "hf_model" / "final"
    if hf_final.exists():
        shutil.rmtree(hf_final)
    hf_final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(train_dir / "checkpoint", hf_final)
    first = logs[0] if logs else summary.get("first_update", {})
    last = logs[-1] if logs else summary.get("last_update", {})
    metrics = {
        "status": "SPECIAL_EMBEDDING_ONLY_ENDPOINT",
        "mode": "special_token_embedding_rows_only",
        "endpoint": rel(train_dir / "checkpoint"),
        "initial_consumed_words": INITIAL_CONSUMED_WORDS,
        "skip_rows": 530944,
        "max_tail_charged_words": int(summary.get("total_words", 3_992_800)),
        "tail_main_word_exposure": int(summary.get("total_words", 3_992_800)),
        "tail_aux_word_exposure": 0,
        "tail_charged_words": int(summary.get("total_words", 3_992_800)),
        "total_consumed_words": INITIAL_CONSUMED_WORDS + int(summary.get("total_words", 3_992_800)),
        "updates": int(summary.get("updates", len(logs))),
        "schedule_total": LR_TOTAL_STEPS,
        "stopped_before_cap": False,
        "trainable": summary.get("trainable_object"),
        "total_params": None,
        "private_params": 0,
        "frozen_slow_params": None,
        "first_main_loss": first.get("main_ce"),
        "final_main_loss": last.get("main_ce"),
        "mean_main_loss": mean([float(x["main_ce"]) for x in logs]) if logs else None,
        "main_loss_batches": len(logs),
        "first_neutral_loss": None,
        "final_neutral_loss": None,
        "mean_neutral_loss": None,
        "deterministic_neutrality_eval_mode": None,
        "first_target_ratio": first.get("target_ratio"),
        "final_target_ratio": last.get("target_ratio"),
        "mean_target_ratio": mean([float(x.get("target_ratio", 0.0)) for x in logs]) if logs else None,
        "special_ids": summary.get("special_ids"),
        "special_tokens": summary.get("special_tokens"),
        "tied_decoder_caveat": summary.get("tied_decoder_caveat"),
        "first_update": first,
        "last_update": last,
        "source_train_dir": rel(train_dir),
        "created_utc": now(),
    }
    (runlike / "scientific_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return runlike


def run_eval(runlike: pathlib.Path, label: str, gpu: int, force: bool, env: dict[str, str]) -> dict[str, Any]:
    out_root = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/cheap7') / label
    collate_root = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/collate') / label
    summary_root = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/summary')
    cmd = [sys.executable, "-B", str(EVALUATOR), "--run-dir", str(runlike), "--target", label, "--endpoint", "final", "--out-root", str(out_root), "--collate-root", str(collate_root), "--summary-root", str(summary_root), "--gpu", str(gpu)]
    if force:
        cmd.append("--force")
    log_dir = _public_path('experiments/archive/relation_learning/data/special_embedding_only_control/eval/launcher_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(json.dumps({"event": "eval_start", "label": label, "cmd": cmd, "utc": now()}), flush=True)
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=9000)
    (log_dir / f"{label}_stdout.log").write_text(p.stdout, encoding="utf-8")
    (log_dir / f"{label}_stderr.log").write_text(p.stderr, encoding="utf-8")
    if p.returncode != 0:
        print(p.stdout[-2000:], flush=True); print(p.stderr[-2000:], flush=True)
        raise RuntimeError(f"eval failed {label} rc={p.returncode}")
    summary_path = summary_root / f"{label}_summary.json"
    rec = {"label": label, "returncode": p.returncode, "elapsed_sec": round(time.time()-t0,1), "summary_path": rel(summary_path), "stdout_log": rel(log_dir / f"{label}_stdout.log"), "stderr_log": rel(log_dir / f"{label}_stderr.log")}
    if summary_path.exists():
        s = read_json(summary_path)
        rec["cheap7"] = s.get("cheap7")
        rec["cheap7_delta_vs_chck82"] = s.get("cheap7_delta_vs_chck82")
        rec["scores"] = s.get("scores")
        rec["deltas_vs_chck82"] = s.get("deltas_vs_chck82")
    print(json.dumps({"event": "eval_done", **rec}, ensure_ascii=False), flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[98197, 98198])
    ap.add_argument("--wait-timeout", type=float, default=28800.0)
    ap.add_argument("--wait-interval", type=float, default=60.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    wait_for_path(WAIT_ON, args.wait_timeout, args.wait_interval)
    w = read_json(WAIT_ON)
    if w.get("status") != "COHERENT_NO_SPECIAL_CONTROL_SEQUENCE_DONE":
        raise RuntimeError(f"upstream no-special control did not finish cleanly: {w.get('status')}")
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu); env["PYTHONUNBUFFERED"] = "1"; env["PYTHONDONTWRITEBYTECODE"] = "1"; env.setdefault("TOKENIZERS_PARALLELISM", "false")
    records=[]
    for seed in args.seeds:
        train_dir = train_one(int(seed), int(args.gpu), force=args.force)
        label = f"special_embedding_only_seed{seed}"
        runlike = make_runlike(train_dir, label)
        ev = run_eval(runlike, label, int(args.gpu), force=args.force, env=env)
        records.append({"seed": int(seed), "train_dir": rel(train_dir), "runlike": rel(runlike), **ev})
    summary = {"status": "SPECIAL_EMBEDDING_ONLY_SEQUENCE_DONE", "created_utc": now(), "waited_on": rel(WAIT_ON), "records": records}
    (_public_path('experiments/archive/relation_learning/data/special_embedding_only_control/sequence_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
