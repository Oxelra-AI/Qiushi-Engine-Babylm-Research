#!/usr/bin/env python3
"""research: replay WWM RNG/mask alignment across the first packet block.

The research shared-tokenizer launch proved exact alignment before packets.  This
script asks the next cheapest question raised by independent review: after packet rows with
slightly different tokenization, do the dynamic masker and random-token branch
remain aligned on the immediately following common batch?

It does not train or evaluate models. It imports the same trusted accumulated
trainer and applies its MaskedChunkDataset + apply_masking_curriculum logic on the
prefix of the two research in-place corpora, maintaining separate generator states
as the two actual training arms do.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
from typing import Any

import torch

ROOT = pathlib.Path('.')
TRAINER_PATH = ROOT / 'experiments/archive/representation_and_objectives/scripts/accumulated_masking_curriculum_trainer.py'
TOK_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/shared_intersection_tokenizer/tokenizers/shared_intersection_legal_byte_bpe_40k'
CORPUS_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/inplace_packed_role_switch_replacement_corpus'
OUT_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/mask_replay_packet_block'
BATCH_SIZE = 256
MICRO_BATCH_SIZE = 64
TRAIN_RNG_SEED = 43023
TOTAL_STEPS = 2024


def load_trainer():
    spec = importlib.util.spec_from_file_location('accum_for_step149_replay', TRAINER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def read_jsonl_prefix(path: pathlib.Path, n: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) >= n:
                    break
    if len(rows) < n:
        raise RuntimeError(f'{path} has only {len(rows)} rows, needed {n}')
    return rows


def make_examples(rows: list[dict[str, Any]], trainer) -> list[Any]:
    out = []
    for i, r in enumerate(rows):
        text = str(r['text'])
        words = int(r.get('words', len(text.split())))
        if words != len(text.split()):
            raise RuntimeError(f'row {i} word mismatch')
        out.append(trainer.base.Example(text=text, words=words, example_id=int(r.get('example_id', i)), source=str(r.get('source', ''))))
    return out


def collate_effective(dataset, trainer, start: int, end: int):
    micro_batches = []
    for s in range(start, end, MICRO_BATCH_SIZE):
        items = [dataset[i] for i in range(s, min(end, s + MICRO_BATCH_SIZE))]
        micro_batches.append(trainer.base.collate(items))
    return trainer.combine_microbatches(micro_batches)


def packet_count(rows: list[dict[str, Any]], start: int, end: int, arm: str) -> int:
    prefix = f'step146_{arm}_packet_packed160'
    return sum(1 for r in rows[start:end] if str(r.get('source', '')).startswith(prefix))


def selected_random_count(input_ids: torch.Tensor, masked_inputs: torch.Tensor, labels: torch.Tensor, mask_id: int) -> dict[str, int]:
    selected = labels != -100
    masktok = selected & (masked_inputs == mask_id)
    kept = selected & (masked_inputs == input_ids)
    # If a random id accidentally equals the original id, it is counted as kept;
    # if it equals mask_id, it is counted as masktok. This is negligible and not
    # used as a formal probability estimate.
    random_like = selected & ~(masktok | kept)
    return {
        'selected_tokens': int(selected.sum().item()),
        'mask_token_positions': int(masktok.sum().item()),
        'kept_or_accidental_same_positions': int(kept.sum().item()),
        'random_like_positions': int(random_like.sum().item()),
    }


def tensor_hash(x: torch.Tensor) -> str:
    import hashlib
    arr = x.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=18, help='effective steps from stream start to replay')
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = load_trainer()
    tokenizer = trainer.base.make_portable_tokenizer(str(TOK_DIR))
    n_rows = args.steps * BATCH_SIZE
    paths = {
        'role_switch': CORPUS_DIR / 'role_switch_inplace_packed_replacement_10M.jsonl',
        'role_fixed': CORPUS_DIR / 'role_fixed_inplace_packed_replacement_10M.jsonl',
    }
    rows = {arm: read_jsonl_prefix(path, n_rows) for arm, path in paths.items()}
    datasets = {arm: trainer.base.MaskedChunkDataset(make_examples(rs, trainer), tokenizer, 256) for arm, rs in rows.items()}

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    curriculum = {}
    gens = {}
    for arm in ['role_switch', 'role_fixed']:
        st = trainer.base.MaskingCurriculumState(curriculum='wwm_fixed', mask_prob_start=0.15, mask_prob_end=0.15, switch_frac=0.7, amlm_window=10, amlm_lambda=0.2)
        st.initialize(vocab_size=len(tokenizer), total_steps=TOTAL_STEPS)
        curriculum[arm] = st
        gen = torch.Generator(device=device)
        gen.manual_seed(TRAIN_RNG_SEED)
        gens[arm] = gen

    step_records = []
    first_rng_divergence = None
    first_labels_divergence_on_common_input = None
    first_input_divergence = None
    for step in range(1, args.steps + 1):
        start = (step - 1) * BATCH_SIZE
        end = step * BATCH_SIZE
        batches = {}
        masked = {}
        rec: dict[str, Any] = {'step': step, 'row_start': start, 'row_end_exclusive': end}
        for arm in ['role_switch', 'role_fixed']:
            batch = collate_effective(datasets[arm], trainer, start, end)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            word_group = batch['word_group'].to(device)
            curriculum[arm].current_step = step - 1
            masked_inputs, labels = trainer.base.apply_masking_curriculum(input_ids, attention_mask, word_group, tokenizer, curriculum[arm], gens[arm])
            batches[arm] = {'input_ids': input_ids, 'attention_mask': attention_mask, 'word_group': word_group}
            masked[arm] = {'masked_inputs': masked_inputs, 'labels': labels}
            rec[arm] = {
                'packet_rows': packet_count(rows[arm], start, end, arm),
                'tokens': int(attention_mask.sum().item()),
                'words': sum(int(r['words']) for r in rows[arm][start:end]),
                **selected_random_count(input_ids, masked_inputs, labels, tokenizer.mask_token_id),
                'input_hash': tensor_hash(input_ids),
                'word_group_hash': tensor_hash(word_group),
                'labels_hash': tensor_hash(labels),
                'masked_inputs_hash': tensor_hash(masked_inputs),
                'generator_state_hash': tensor_hash(gens[arm].get_state()),
            }
        input_equal = torch.equal(batches['role_switch']['input_ids'].cpu(), batches['role_fixed']['input_ids'].cpu())
        wg_equal = torch.equal(batches['role_switch']['word_group'].cpu(), batches['role_fixed']['word_group'].cpu())
        labels_equal = torch.equal(masked['role_switch']['labels'].cpu(), masked['role_fixed']['labels'].cpu())
        masked_equal = torch.equal(masked['role_switch']['masked_inputs'].cpu(), masked['role_fixed']['masked_inputs'].cpu())
        gen_equal = torch.equal(gens['role_switch'].get_state().cpu(), gens['role_fixed'].get_state().cpu())
        rec['comparisons'] = {
            'input_ids_equal': bool(input_equal),
            'word_group_equal': bool(wg_equal),
            'labels_equal': bool(labels_equal),
            'masked_inputs_equal': bool(masked_equal),
            'generator_state_equal_after_step': bool(gen_equal),
            'batch_contains_packets': rec['role_switch']['packet_rows'] > 0 or rec['role_fixed']['packet_rows'] > 0,
        }
        if not input_equal and first_input_divergence is None:
            first_input_divergence = step
        if not gen_equal and first_rng_divergence is None:
            first_rng_divergence = step
        if input_equal and (not labels_equal or not masked_equal) and first_labels_divergence_on_common_input is None:
            first_labels_divergence_on_common_input = step
        step_records.append(rec)
        # Keep memory modest.
        del batches, masked

    summary = {
        'status': 'MASK_REPLAY_PACKET_BLOCK',
        'device': str(device),
        'steps_replayed': args.steps,
        'train_rng_seed': TRAIN_RNG_SEED,
        'shared_tokenizer': str(TOK_DIR),
        'first_input_divergence_step': first_input_divergence,
        'first_rng_state_divergence_step': first_rng_divergence,
        'first_mask_or_label_divergence_on_common_input_step': first_labels_divergence_on_common_input,
        'interpretation': {
            'if_common_postpacket_masks_diverge': 'Later common batches are no longer bitwise mask-paired after packet text because tokenizer geometry and random-token draw counts can desynchronize the per-arm training RNG. This is part of the practical recipe effect, but endpoint differences cannot be attributed only to semantic role exchange.',
            'if_common_postpacket_masks_remain_equal': 'The shared tokenizer plus equal word counts preserves mask-pairing at least through this prefix; endpoint contrast is cleaner.'
        },
        'step_records': step_records,
    }
    out_path = OUT_DIR / 'mask_replay_packet_block_summary.json'
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({
        'event': 'MASK_REPLAY_PACKET_BLOCK_DONE',
        'summary': str(out_path),
        'first_input_divergence_step': first_input_divergence,
        'first_rng_state_divergence_step': first_rng_divergence,
        'first_mask_or_label_divergence_on_common_input_step': first_labels_divergence_on_common_input,
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
