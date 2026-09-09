#!/usr/bin/env python3
"""research source-swap antecedent-dependence probe.

Purpose: before spending H100 time on anchored-masking training,
verify that later-mention target tokens actually depend on the CORRECT earlier
antecedent. If the model's fill-in likelihood for a masked later mention does not
depend on which earlier entity is visible, then an anchored objective would learn
target frequency/local syntax, not cross-mention binding.

Design:
  For each high_entity_state window with a repeated capitalized anchor:
    - Find an earlier "source" mention (first occurrence) and a later "target" mention.
    - Mask ONLY the target-mention tokens. The target tokens (labels) are IDENTICAL
      across all conditions.
    - Condition A (correct): earlier source mention left intact/visible.
    - Condition B (swapped): earlier source mention replaced by a DIFFERENT matched
      anchor drawn from another window (same token length when possible).
    - Condition C (masked): earlier source mention replaced by [MASK] tokens.
  Measure mean target-token log-likelihood under each condition with the protected
  100M DeBERTa model. Report A-minus-B and A-minus-C deltas and positive fractions.

A real correct-antecedent advantage (A > B and A > C) justifies building the
target-identical anchored/control training. A null result blocks it.
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import random
import re
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
DATA = ROOT / 'data/structure_density_revision_157/high_entity_state.jsonl'
MODEL_DIR = (ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M').resolve()
OUT = ROOT / 'data/source_swap_antecedent_probe.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/source_swap_antecedent_probe.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/source_swap_probe.log')

CAP_RE = re.compile(r"^[A-Z][a-z]+$")
STOP = {'The','A','An','And','But','Or','If','When','While','He','She','It','They','We','You','I','His','Her','Their','This','That','These','Those','So','As','At','In','On','Of','To','For','With','My','Your'}
MAX_SEQ = 256
N_WINDOWS = 700          # candidate windows to scan
MAX_PROBE_EXAMPLES = 400 # cap probe examples for runtime
SEED = 164


def setup_env():
    hf = ROOT / 'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def find_repeated_cap_words(text: str) -> list[str]:
    toks = text.split()
    counts = collections.Counter()
    for i, t in enumerate(toks):
        clean = re.sub(r"[^A-Za-z]", "", t)
        if i > 0 and CAP_RE.match(clean) and clean not in STOP:
            counts[clean] += 1
    return [w for w, c in counts.items() if c >= 2]


def main():
    setup_env()
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    log_lines = []
    def log(m):
        print(m, flush=True)
        log_lines.append(str(m))

    if not MODEL_DIR.exists():
        raise RuntimeError(f'missing model {MODEL_DIR}')
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL_DIR))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    mask_id = tok.mask_token_id
    log(f'model loaded on {device}; mask_id={mask_id} vocab={len(tok)}')

    rng = random.Random(SEED)
    # Load candidate windows (epoch 0 unique) with a repeated capitalized anchor.
    candidates = []
    with DATA.open('r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('epoch') != 0:
                continue
            text = r['text']
            reps = find_repeated_cap_words(text)
            if reps:
                candidates.append((text, reps))
            if len(candidates) >= N_WINDOWS:
                break
    log(f'candidate windows with repeated cap anchor: {len(candidates)}')

    # Pool of alternate anchors for swap (distinct surface words).
    anchor_pool = sorted({w for _, reps in candidates for w in reps})
    log(f'anchor pool size: {len(anchor_pool)}')

    def token_ids_for_word(word: str, with_space: bool) -> list[int]:
        s = (' ' + word) if with_space else word
        return tok(s, add_special_tokens=False)['input_ids']

    @torch.no_grad()
    def target_ll(ids: list[int], target_positions: list[int], target_ids: list[int]) -> float:
        """Mean log-prob of target_ids at target_positions given a full id list (target masked)."""
        x = list(ids)
        for p in target_positions:
            x[p] = mask_id
        inp = torch.tensor([x[:MAX_SEQ]], device=device)
        att = torch.ones_like(inp)
        logits = model(input_ids=inp, attention_mask=att).logits[0]
        lp = F.log_softmax(logits, dim=-1)
        total = 0.0
        n = 0
        for p, tid in zip(target_positions, target_ids):
            if p >= MAX_SEQ:
                continue
            total += float(lp[p, tid])
            n += 1
        return total / max(1, n)

    examples = []
    a_minus_b = []
    a_minus_c = []
    a_vals = []
    b_vals = []
    c_vals = []

    for text, reps in candidates:
        if len(examples) >= MAX_PROBE_EXAMPLES:
            break
        anchor = reps[0]
        # Build word-level occurrence positions by re-tokenizing with offsets.
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc['input_ids']
        offs = enc['offset_mapping']
        if len(ids) < 8 or len(ids) > MAX_SEQ:
            continue
        # Locate token spans whose surface equals the anchor (allow leading space).
        # Reconstruct token surfaces.
        surfaces = [text[a:b] for (a, b) in offs]
        # Find contiguous spans matching the anchor as a standalone word.
        occ_spans = []  # list of (start_tok, end_tok_exclusive)
        i = 0
        target_word_ids_space = token_ids_for_word(anchor, with_space=True)
        target_word_ids_nospace = token_ids_for_word(anchor, with_space=False)
        L1, L2 = len(target_word_ids_space), len(target_word_ids_nospace)
        while i < len(ids):
            matched = None
            if i + L1 <= len(ids) and ids[i:i+L1] == target_word_ids_space:
                matched = (i, i+L1)
                i += L1
            elif i + L2 <= len(ids) and ids[i:i+L2] == target_word_ids_nospace:
                matched = (i, i+L2)
                i += L2
            else:
                i += 1
            if matched:
                occ_spans.append(matched)
        if len(occ_spans) < 2:
            continue
        source_span = occ_spans[0]
        target_span = occ_spans[-1]
        if target_span[0] <= source_span[1]:
            continue
        target_positions = list(range(target_span[0], target_span[1]))
        target_ids = [ids[p] for p in target_positions]

        # Condition A: correct antecedent visible (original ids).
        llA = target_ll(ids, target_positions, target_ids)

        # Condition B: swap earlier source mention with a different anchor of same token length.
        alt_candidates = [w for w in anchor_pool if w != anchor]
        rng.shuffle(alt_candidates)
        swapped_ids = None
        src_len = source_span[1] - source_span[0]
        for alt in alt_candidates:
            alt_space = token_ids_for_word(alt, with_space=(source_span[0] > 0))
            if len(alt_space) == src_len:
                swapped_ids = list(ids)
                swapped_ids[source_span[0]:source_span[1]] = alt_space
                break
        if swapped_ids is None:
            # fall back: use any alt, pad/truncate to same length
            alt = alt_candidates[0]
            alt_space = token_ids_for_word(alt, with_space=(source_span[0] > 0))
            alt_space = (alt_space + alt_space)[:src_len] if alt_space else [ids[source_span[0]]]
            swapped_ids = list(ids)
            swapped_ids[source_span[0]:source_span[1]] = alt_space
        # target positions unchanged (same length swap)
        llB = target_ll(swapped_ids, target_positions, target_ids)

        # Condition C: mask earlier source mention.
        masked_ids = list(ids)
        for p in range(source_span[0], source_span[1]):
            masked_ids[p] = mask_id
        llC = target_ll(masked_ids, target_positions, target_ids)

        a_vals.append(llA); b_vals.append(llB); c_vals.append(llC)
        a_minus_b.append(llA - llB)
        a_minus_c.append(llA - llC)
        if len(examples) < 15:
            examples.append({
                'anchor': anchor, 'n_occurrences': len(occ_spans),
                'source_span': source_span, 'target_span': target_span,
                'llA_correct': round(llA, 4), 'llB_swapped': round(llB, 4), 'llC_masked': round(llC, 4),
                'A_minus_B': round(llA - llB, 4), 'A_minus_C': round(llA - llC, 4),
                'text_preview': text[:200],
            })

    def stats(xs):
        if not xs:
            return {'n': 0}
        xs_sorted = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        median = xs_sorted[n // 2]
        pos_frac = sum(1 for x in xs if x > 0) / n
        return {'n': n, 'mean': round(mean, 4), 'median': round(median, 4),
                'positive_fraction': round(pos_frac, 4),
                'p10': round(xs_sorted[max(0, n//10)], 4), 'p90': round(xs_sorted[min(n-1, 9*n//10)], 4)}

    payload = {
        'status': 'SOURCE_SWAP_ANTECEDENT_PROBE',
        'model': str(MODEL_DIR),
        'data': str(DATA),
        'n_probe_examples': len(a_minus_b),
        'target_identical_across_conditions': True,
        'condition_meanings': {
            'A_correct': 'earlier correct antecedent visible',
            'B_swapped': 'earlier antecedent replaced by a different matched anchor (same token length)',
            'C_masked': 'earlier antecedent masked',
        },
        'target_ll_stats': {'A_correct': stats(a_vals), 'B_swapped': stats(b_vals), 'C_masked': stats(c_vals)},
        'A_minus_B': stats(a_minus_b),
        'A_minus_C': stats(a_minus_c),
        'examples': examples,
        'decision_rule': 'A>B and A>C with clear positive fraction (>~0.6) and mean delta indicates the later target depends on the correct antecedent, justifying target-identical anchored training. Near-zero deltas block anchored training.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    ab = payload['A_minus_B']; ac = payload['A_minus_C']
    NOTE.write_text('\n'.join([
        '# research — source-swap antecedent-dependence probe',
        '',
        f'Model: protected 100M DeBERTa (`{MODEL_DIR.name}`). Probe examples: {payload["n_probe_examples"]}.',
        '',
        'Target tokens are identical across all three conditions; only the earlier antecedent changes.',
        '',
        '| condition | mean target LL |',
        '|---|---:|',
        f'| A correct antecedent visible | {payload["target_ll_stats"]["A_correct"].get("mean")} |',
        f'| B antecedent swapped | {payload["target_ll_stats"]["B_swapped"].get("mean")} |',
        f'| C antecedent masked | {payload["target_ll_stats"]["C_masked"].get("mean")} |',
        '',
        f'A - B: mean {ab.get("mean")}, median {ab.get("median")}, positive fraction {ab.get("positive_fraction")}',
        f'A - C: mean {ac.get("mean")}, median {ac.get("median")}, positive fraction {ac.get("positive_fraction")}',
        '',
        'If A - B and A - C are near zero, the later target does not depend on the correct antecedent, and anchored-masking training would not localize cross-mention binding. If clearly positive, a target-identical anchored/control training is justified.',
    ]) + '\n', encoding='utf-8')
    LOG.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'n': payload['n_probe_examples'],
                      'A_minus_B': ab, 'A_minus_C': ac, 'out': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
