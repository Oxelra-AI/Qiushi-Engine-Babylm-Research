#!/usr/bin/env python3
"""research CPU check: base batch-256 vs accumulated 4x64 pre-forward identity.

This does not train and does not touch the active GPU runs. It tests the exact part of
research's memory repair that should be identical before model.forward():
  * same JSONL row order
  * same tokenizer/Dataset/collate tensors
  * 4x64 recombined tensors exactly equal a direct 256-row DataLoader batch
  * with the same MaskingCurriculumState and torch.Generator seed, apply_masking_curriculum
    returns identical masked_inputs and labels on CPU

It is a structural identity check, not endpoint evidence and not a substitute for the
post-hoc short accumulated-16k training control.
"""
from __future__ import annotations

import json
import pathlib
import random
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
OUT_DIR = WS / "data/accum_data_mask_identity_check"
TRAIN_100M = pathlib.Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
TOKENIZERS = {
    "legal16k": WS / "data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer",
    "legal40k": WS / "data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k",
}


def load_first_examples(n: int) -> list[Any]:
    examples = []
    with TRAIN_100M.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            obj = json.loads(line)
            text = obj["text"]
            words = int(obj.get("words", len(text.split())))
            ex_id = int(obj.get("example_id", i))
            source = str(obj.get("source", "jsonl"))
            examples.append(base.Example(text=text, words=words, example_id=ex_id, source=source))
    if len(examples) != n:
        raise RuntimeError(f"loaded {len(examples)} examples, expected {n}")
    return examples


def combine(micro_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.cat([b["input_ids"] for b in micro_batches], dim=0),
        "attention_mask": torch.cat([b["attention_mask"] for b in micro_batches], dim=0),
        "word_group": torch.cat([b["word_group"] for b in micro_batches], dim=0),
        "words": torch.cat([b["words"] for b in micro_batches], dim=0),
    }


def make_state(total_steps: int, current_step: int) -> Any:
    st = base.MaskingCurriculumState(
        curriculum="wwm_fixed",
        mask_prob_start=0.15,
        mask_prob_end=0.15,
        switch_frac=0.7,
        amlm_window=10,
        amlm_lambda=0.2,
    )
    # vocab_size is not used for fixed WWM except metadata; set by caller after tokenizer load if desired.
    st.initialize(vocab_size=1, total_steps=total_steps)
    st.current_step = current_step
    return st


def check_tokenizer(label: str, tok_path: pathlib.Path) -> dict[str, Any]:
    tok = base.make_portable_tokenizer(str(tok_path))
    examples = load_first_examples(512)
    dataset = base.MaskedChunkDataset(examples, tok, 256)
    loader256 = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=base.collate, num_workers=0)
    loader64 = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=base.collate, num_workers=0)
    direct_batches = list(loader256)
    micro_batches = list(loader64)
    records = []
    for bi in range(2):
        direct = direct_batches[bi]
        recomb = combine(micro_batches[bi*4:(bi+1)*4])
        tensor_equal = {
            k: bool(torch.equal(direct[k], recomb[k])) for k in ["input_ids", "attention_mask", "word_group", "words"]
        }
        # Test CPU masking identity with two independent but equally seeded CPU generators.
        for step_index in [0, 1, 70]:
            state_a = base.MaskingCurriculumState("wwm_fixed", 0.15, 0.15, 0.7, 10, 0.2)
            state_b = base.MaskingCurriculumState("wwm_fixed", 0.15, 0.15, 0.7, 10, 0.2)
            state_a.initialize(vocab_size=len(tok), total_steps=2529)
            state_b.initialize(vocab_size=len(tok), total_steps=2529)
            state_a.current_step = step_index
            state_b.current_step = step_index
            gen_a = torch.Generator(device="cpu")
            gen_b = torch.Generator(device="cpu")
            gen_a.manual_seed(43023)
            gen_b.manual_seed(43023)
            masked_a, labels_a = base.apply_masking_curriculum(
                direct["input_ids"], direct["attention_mask"], direct["word_group"], tok, state_a, gen_a
            )
            masked_b, labels_b = base.apply_masking_curriculum(
                recomb["input_ids"], recomb["attention_mask"], recomb["word_group"], tok, state_b, gen_b
            )
            records.append({
                "effective_batch_index": bi + 1,
                "current_step": step_index,
                "tensor_equal": tensor_equal,
                "masked_inputs_equal": bool(torch.equal(masked_a, masked_b)),
                "labels_equal": bool(torch.equal(labels_a, labels_b)),
                "masked_tokens": int((labels_a != -100).sum().item()),
                "batch_words": int(direct["words"].sum().item()),
                "candidate_tokens": int(direct["attention_mask"].bool().sum().item()),
            })
    return {
        "tokenizer_label": label,
        "tokenizer_path": str(tok_path),
        "vocab_size": len(tok),
        "records": records,
        "all_input_tensors_equal": all(all(r["tensor_equal"].values()) for r in records),
        "all_masked_inputs_equal": all(r["masked_inputs_equal"] for r in records),
        "all_labels_equal": all(r["labels_equal"] for r in records),
        "first_batch_words": records[0]["batch_words"],
        "first_batch_masked_tokens": records[0]["masked_tokens"],
    }


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [check_tokenizer(label, path) for label, path in TOKENIZERS.items()]
    payload = {
        "status": "ACCUM_DATA_MASK_IDENTITY_CHECK",
        "purpose": "CPU structural test that 4x64 accumulation reconstructs the same direct 256-row tensors and pre-forward masking tensors as the base trainer",
        "train_100m": str(TRAIN_100M),
        "n_examples_tested": 512,
        "effective_batches_tested_per_tokenizer": 2,
        "tokenizers": results,
        "all_pass": all(r["all_input_tensors_equal"] and r["all_masked_inputs_equal"] and r["all_labels_equal"] for r in results),
        "interpretation": "If all_pass is true, the memory repair leaves data order, collation, WWM selection, 80/10/10 replacement, and masking RNG consumption identical before model.forward(); only dropout/forward microbatch stochasticity remains as a trainer-level difference.",
    }
    out_json = OUT_DIR / "accum_data_mask_identity_check.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "all_pass": payload["all_pass"],
        "out_json": str(out_json),
        "summary": [
            {
                "tokenizer": r["tokenizer_label"],
                "vocab_size": r["vocab_size"],
                "first_batch_words": r["first_batch_words"],
                "first_batch_masked_tokens": r["first_batch_masked_tokens"],
                "all_input_tensors_equal": r["all_input_tensors_equal"],
                "all_masked_inputs_equal": r["all_masked_inputs_equal"],
                "all_labels_equal": r["all_labels_equal"],
            } for r in results
        ],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
