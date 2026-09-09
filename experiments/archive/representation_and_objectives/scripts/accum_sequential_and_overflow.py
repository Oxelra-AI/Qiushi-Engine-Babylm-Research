#!/usr/bin/env python3
"""research stronger CPU checks (no GPU, no active-task polling).

Two upgrades requested by the research verifier review:

(A) Sequential no-reseed masking-stream identity: keep two equally seeded torch.Generators
    alive while iterating a direct 256-row loader and a recombined 4x64 loader over MANY
    consecutive effective batches, including a mid-stream region and the final partial
    effective batch. Compare every masked_inputs, labels, and the final generator state.
    This tests continuous RNG-stream identity, not just single-batch determinism.

(B) Overflow transition matrix between legal16k and legal40k on the exact 10M pool:
    neither / 16k-only / 40k-only / both, so "rescued rows" is not conflated with a net
    count. Also token-per-WWM-group distribution buckets (1,2,3,>=4) for both tokenizers.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

USER_ROOT = pathlib.Path(".").resolve()
COMPACT_EXPERIENCE_SCRIPTS = USER_ROOT / "experiments/archive" / 'compact_experience' / "scripts"
if str(COMPACT_EXPERIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(COMPACT_EXPERIENCE_SCRIPTS))
import masking_curriculum_trainer as base  # noqa: E402

STUDY = pathlib.Path("experiments/archive/representation_and_objectives")
WS = STUDY
OUT_DIR = WS / "data/accum_sequential_and_overflow"
POOL10 = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl")
TOK16 = WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer"
TOK40 = WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k"


def load_examples(n: int) -> list[Any]:
    ex = []
    with POOL10.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            o = json.loads(line)
            ex.append(base.Example(text=o["text"], words=int(o.get("words", len(o["text"].split()))),
                                   example_id=int(o.get("example_id", i)), source=str(o.get("source", "jsonl"))))
    return ex


def combine(mbs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.cat([b[k] for b in mbs], dim=0) for k in ["input_ids", "attention_mask", "word_group", "words"]}


def sequential_identity(tok, n_examples: int, n_effective: int) -> dict[str, Any]:
    examples = load_examples(n_examples)
    dataset = base.MaskedChunkDataset(examples, tok, 256)
    loader256 = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=base.collate, num_workers=0)
    loader64 = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=base.collate, num_workers=0)

    state_a = base.MaskingCurriculumState("wwm_fixed", 0.15, 0.15, 0.7, 10, 0.2)
    state_b = base.MaskingCurriculumState("wwm_fixed", 0.15, 0.15, 0.7, 10, 0.2)
    state_a.initialize(vocab_size=len(tok), total_steps=n_effective)
    state_b.initialize(vocab_size=len(tok), total_steps=n_effective)
    gen_a = torch.Generator(device="cpu"); gen_a.manual_seed(43023)
    gen_b = torch.Generator(device="cpu"); gen_b.manual_seed(43023)

    it256 = iter(loader256)
    micro = list(loader64)
    per_step = []
    all_masked_equal = True
    all_labels_equal = True
    all_tensor_equal = True
    run_count = 0
    last_rows = None
    for step in range(1, n_effective + 1):
        try:
            direct = next(it256)
        except StopIteration:
            break
        run_count += 1
        recomb = combine(micro[(research)*4: step*4])
        te = all(bool(torch.equal(direct[k], recomb[k])) for k in ["input_ids", "attention_mask", "word_group", "words"])
        state_a.current_step = step - 1
        state_b.current_step = step - 1
        ma, la = base.apply_masking_curriculum(direct["input_ids"], direct["attention_mask"], direct["word_group"], tok, state_a, gen_a)
        mb, lb = base.apply_masking_curriculum(recomb["input_ids"], recomb["attention_mask"], recomb["word_group"], tok, state_b, gen_b)
        me = bool(torch.equal(ma, mb)); le = bool(torch.equal(la, lb))
        all_masked_equal &= me; all_labels_equal &= le; all_tensor_equal &= te
        last_rows = int(direct["input_ids"].shape[0])
        if step <= 3 or step % 25 == 0 or step >= n_effective - 1:
            per_step.append({
                "step": step, "rows": last_rows,
                "tensor_equal": te, "masked_equal": me, "labels_equal": le,
                "masked_tokens": int((la != -100).sum().item()),
            })
    gen_state_equal = bool(torch.equal(gen_a.get_state(), gen_b.get_state()))
    return {
        "n_examples": n_examples,
        "n_effective_batches_requested": n_effective,
        "n_effective_batches_run": run_count,
        "all_input_tensors_equal": all_tensor_equal,
        "all_masked_inputs_equal": all_masked_equal,
        "all_labels_equal": all_labels_equal,
        "final_generator_state_equal": gen_state_equal,
        "final_partial_batch_rows": last_rows,
        "sampled_steps": per_step,
    }


def group_token_hist(tok, texts: list[str]) -> dict[str, int]:
    tok.model_max_length = 10**9
    special = set(tok.all_special_ids)
    start_cache: dict[int, bool] = {}

    def word_start(tid: int) -> bool:
        if tid not in start_cache:
            s = tok.convert_ids_to_tokens(int(tid))
            start_cache[tid] = bool(s is not None and (str(s).startswith("Ġ") or str(s).startswith("▁")))
        return start_cache[tid]

    hist = collections.Counter()
    for start in range(0, len(texts), 512):
        encs = tok(texts[start:start+512], add_special_tokens=False, truncation=True, max_length=256, padding=False)
        for ids in encs["input_ids"]:
            gid = -1; count = 0
            for i, tid in enumerate(ids):
                if tid in special:
                    continue
                if gid < 0 or word_start(tid) or i == 0:
                    if gid >= 0:
                        hist[min(count, 4)] += 1
                    gid += 1; count = 1
                else:
                    count += 1
            if gid >= 0:
                hist[min(count, 4)] += 1
    return {str(k): int(v) for k, v in sorted(hist.items())}


def overflow_matrix() -> dict[str, Any]:
    tok16 = AutoTokenizer.from_pretrained(str(TOK16), use_fast=True); tok16.model_max_length = 10**9
    tok40 = AutoTokenizer.from_pretrained(str(TOK40), use_fast=True); tok40.model_max_length = 10**9
    texts = []
    with POOL10.open("r", encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    over16 = [False] * len(texts)
    over40 = [False] * len(texts)
    for start in range(0, len(texts), 512):
        e16 = tok16(texts[start:start+512], add_special_tokens=False, truncation=False, padding=False)
        e40 = tok40(texts[start:start+512], add_special_tokens=False, truncation=False, padding=False)
        for j, (a, b) in enumerate(zip(e16["input_ids"], e40["input_ids"])):
            over16[start+j] = len(a) > 256
            over40[start+j] = len(b) > 256
    mat = {"neither": 0, "only16k": 0, "only40k": 0, "both": 0}
    for a, b in zip(over16, over40):
        if a and b:
            mat["both"] += 1
        elif a and not b:
            mat["only16k"] += 1
        elif b and not a:
            mat["only40k"] += 1
        else:
            mat["neither"] += 1
    return {
        "rows": len(texts),
        "over256_count_16k": sum(over16),
        "over256_count_40k": sum(over40),
        "transition_matrix": mat,
        "net_overflow_reduction_16k_minus_40k": sum(over16) - sum(over40),
        "rows_rescued_16k_to_visible_40k": mat["only16k"],
        "rows_newly_overflowing_only_40k": mat["only40k"],
        "group_token_hist_16k": group_token_hist(tok16, texts),
        "group_token_hist_40k": group_token_hist(tok40, texts),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok16 = AutoTokenizer.from_pretrained(str(TOK16), use_fast=True)
    tok40 = AutoTokenizer.from_pretrained(str(TOK40), use_fast=True)
    seq16 = sequential_identity(tok16, n_examples=1024 + 200, n_effective=4)
    seq40 = sequential_identity(tok40, n_examples=1024 + 200, n_effective=4)
    # Longer sequential run on 16k: 100 full 256-row effective batches plus a 37-row final partial batch.
    seq16_long = sequential_identity(tok16, n_examples=256*100 + 37, n_effective=101)
    overflow = overflow_matrix()
    payload = {
        "status": "ACCUM_SEQUENTIAL_AND_OVERFLOW",
        "purpose": "Stronger continuous-RNG masking identity across many effective batches (incl. final partial) and an exact 16k/40k overflow transition matrix.",
        "sequential_identity": {
            "legal16k_short": seq16,
            "legal40k_short": seq40,
            "legal16k_long_101_effective_incl_partial": seq16_long,
        },
        "overflow": overflow,
        "all_sequential_pass": all(
            r["all_input_tensors_equal"] and r["all_masked_inputs_equal"] and r["all_labels_equal"] and r["final_generator_state_equal"]
            for r in [seq16, seq40, seq16_long]
        ),
        "interpretation": [
            "If all_sequential_pass is true, the 4x64 recombination preserves continuous masking-RNG-stream identity across consecutive effective batches, including the final partial effective batch; the only trainer difference is stochastic forward (dropout/kernel) execution.",
            "The overflow transition matrix replaces the ambiguous net 'rescued rows' with an explicit neither/only16k/only40k/both partition.",
        ],
    }
    out_json = OUT_DIR / "accum_sequential_and_overflow.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "all_sequential_pass": payload["all_sequential_pass"],
        "seq16_long_batches": seq16_long["n_effective_batches_run"],
        "seq16_long_final_partial_rows": seq16_long["final_partial_batch_rows"],
        "overflow_matrix": overflow["transition_matrix"],
        "net_overflow_reduction": overflow["net_overflow_reduction_16k_minus_40k"],
        "out_json": str(out_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
