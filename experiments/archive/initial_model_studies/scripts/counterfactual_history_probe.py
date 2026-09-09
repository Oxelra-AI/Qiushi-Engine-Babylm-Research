#!/usr/bin/env python3
"""research counterfactual history probe.

Scientific purpose:
Confirm whether the research remention-history signal reflects genuine cross-sentence
entity-state tracking vs. mere lexical/topic familiarity. For each remention case,
we create a counterfactual by SWAPPING the earlier mention to a different word (from
the same frequency band) while keeping the local context around the later mention
IDENTICAL. Then measure representation divergence at the later-mention position.

If cosine similarity between original and counterfactual representations is ≈1.0,
the model ignores which entity was in the history → research gain was from topic cues.
If cosine < 0.99 or probe discrimination > 60%, the model genuinely carries
entity-identity information forward.

Cross-seed comparison (wwm42/43, amlm42/43) confirms robustness.
"""
from __future__ import annotations
import argparse, collections, json, os, pathlib, random, re, sys, time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
RAW_DEFAULT = ROOT/'training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/raw_dataset'
OUT_DEFAULT = ROOT/'data/counterfactual_history_probe.json'
NOTE_DEFAULT = (ROOT.parents[2] / 'research/notes/initial_model_studies/counterfactual_history_probe.md')

STOP = set('the a an and or but if then else when while of in on at by for with without from to into onto over under as is are was were be been being do did done have has had i you he she it we they me him her us them my your his its our their this that these those there here not no yes can could would should may might will shall one two three four new old first last good bad little big small great mr mrs miss sir said says say like just very more most some any many much other another'.split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def setup_env():
    hf = ROOT/'training/hf_home'
    os.environ['HF_HOME'] = str(hf.resolve())
    os.environ['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    os.environ['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    os.environ['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def norm_word(w: str) -> str:
    return w.strip("'\".,!?;:()[]{}""''").lower()


def candidate_word(w: str) -> bool:
    n = norm_word(w)
    if len(n) < 4 or n in STOP: return False
    if not re.match(r"^[a-z][a-z'\-]+$", n): return False
    return True


@dataclass
class CounterfactualPair:
    """One pair: original text vs history-swapped text, with positions."""
    target_word: str          # the rementioned word (at position p)
    swap_word: str            # what we put at position q instead
    original_text: str        # full text with real history
    counterfactual_text: str  # full text with swap at position q
    target_char_start: int    # char offset of target in both texts (same local context)
    target_char_end: int
    history_gap_words: int    # word distance between q and p
    source: str


def iter_docs(raw_dir: pathlib.Path, max_docs: int):
    files = ['gutenberg.train.txt', 'simple_wiki.train.txt', 'childes.train.txt',
             'open_subtitles.train.txt', 'bnc_spoken.train.txt', 'switchboard.train.txt']
    n = 0
    for fn in files:
        p = raw_dir / fn
        if not p.exists(): continue
        buf = []
        with p.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                s = line.strip()
                if s:
                    buf.append(s)
                    if sum(len(x.split()) for x in buf) >= 160:
                        yield fn, ' '.join(buf); n += 1; buf = []
                        if n >= max_docs: return
                elif buf:
                    yield fn, ' '.join(buf); n += 1; buf = []
                    if n >= max_docs: return
            if buf:
                yield fn, ' '.join(buf); n += 1
                if n >= max_docs: return


def build_pairs(raw_dir: pathlib.Path, max_docs: int, max_pairs: int, seed: int) -> list[CounterfactualPair]:
    """Extract counterfactual pairs from official text."""
    rng = random.Random(seed)
    # First pass: collect word frequency for swap candidates
    word_freq = collections.Counter()
    all_words_by_freq = {}
    docs_cache = []
    for source, doc in iter_docs(raw_dir, max_docs):
        docs_cache.append((source, doc))
        for m in WORD_RE.finditer(doc):
            n = norm_word(m.group(0))
            if candidate_word(m.group(0)):
                word_freq[n] += 1
    # Build frequency bands for swaps
    freq_bands = collections.defaultdict(list)
    for w, c in word_freq.items():
        band = min(c // 5, 20)  # band by frequency/5, capped at 20
        freq_bands[band].append(w)
    for band in freq_bands:
        rng.shuffle(freq_bands[band])

    def get_swap(word_norm: str) -> Optional[str]:
        c = word_freq[word_norm]
        band = min(c // 5, 20)
        candidates = [w for w in freq_bands[band] if w != word_norm and len(w) >= 3]
        if not candidates:
            candidates = [w for w in freq_bands.get(band + 1, []) + freq_bands.get(max(0, band - 1), []) if w != word_norm]
        if not candidates: return None
        return rng.choice(candidates)

    pairs = []
    for source, doc in docs_cache:
        # Find word occurrences
        matches = [(m.group(0), m.start(), m.end(), norm_word(m.group(0)))
                   for m in WORD_RE.finditer(doc) if candidate_word(m.group(0))]
        if len(matches) < 15: continue
        # Find rementions (word appears at q then at p, gap > 10 word-positions)
        positions_by_word = collections.defaultdict(list)
        for i, (surf, s, e, nw) in enumerate(matches):
            positions_by_word[nw].append(i)
        for nw, idxs in positions_by_word.items():
            if len(idxs) < 2: continue
            # Use first occurrence as q, second as p
            q_idx, p_idx = idxs[0], idxs[1]
            gap = p_idx - q_idx
            if gap < 10: continue  # require substantial gap
            swap = get_swap(nw)
            if swap is None: continue
            # Build texts: keep everything identical except replace word at position q
            q_surf, q_start, q_end, _ = matches[q_idx]
            p_surf, p_start, p_end, _ = matches[p_idx]
            # Swap at q only
            # Capitalize swap to match original casing
            swap_surf = swap.capitalize() if q_surf[0].isupper() else swap
            cf_text = doc[:q_start] + swap_surf + doc[q_end:]
            # Adjust p position if swap changed length
            len_diff = len(swap_surf) - len(q_surf)
            cf_p_start = p_start + len_diff
            cf_p_end = p_end + len_diff
            # Extract windows around p (keeping identical local context)
            # Use a 128-word window centered on p
            window_left = max(0, p_start - 400)
            window_right = min(len(doc), p_end + 200)
            orig_window = doc[window_left:window_right]
            cf_window = cf_text[window_left:window_right]
            target_start = p_start - window_left
            target_end = p_end - window_left
            cf_target_start = cf_p_start - window_left
            cf_target_end = cf_p_end - window_left
            # Verify local context is identical (6 chars each side)
            local_orig = orig_window[max(0, target_start - 30):target_end + 30]
            local_cf = cf_window[max(0, cf_target_start - 30):cf_target_end + 30]
            if local_orig != local_cf:
                continue  # swap changed local context (word appeared nearby) — skip
            pairs.append(CounterfactualPair(
                target_word=nw, swap_word=swap,
                original_text=orig_window, counterfactual_text=cf_window,
                target_char_start=target_start, target_char_end=target_end,
                history_gap_words=gap, source=source
            ))
            if len(pairs) >= max_pairs:
                return pairs
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def token_positions(tokenizer, text: str, start: int, end: int, max_len: int):
    enc = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=max_len, return_offsets_mapping=True, return_tensors='pt')
    offs = enc.pop('offset_mapping')[0].tolist()
    pos = [i for i, (s, e) in enumerate(offs) if e > start and s < end]
    if not pos:
        pos = [min(range(len(offs)), key=lambda i: abs((offs[i][0]+offs[i][1])/2 - (start+end)/2))] if offs else [0]
    return enc, pos


def run_counterfactual_analysis(model_path: pathlib.Path, pairs: list[CounterfactualPair],
                                 layers: list[int], max_len: int, device: str):
    """Compute representation divergence between original and counterfactual at target positions."""
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    model.eval()

    cosines = {l: [] for l in layers}
    l2_dists = {l: [] for l in layers}
    orig_feats = {l: [] for l in layers}
    cf_feats = {l: [] for l in layers}

    with torch.no_grad():
        for pair in pairs:
            # Original
            enc_o, pos_o = token_positions(tok, pair.original_text, pair.target_char_start, pair.target_char_end, max_len)
            enc_o = {k: v.to(device) for k, v in enc_o.items()}
            out_o = model(**enc_o, output_hidden_states=True)
            # Counterfactual
            enc_c, pos_c = token_positions(tok, pair.counterfactual_text, pair.target_char_start, pair.target_char_end, max_len)
            enc_c = {k: v.to(device) for k, v in enc_c.items()}
            out_c = model(**enc_c, output_hidden_states=True)

            for l in layers:
                h_o = out_o.hidden_states[l][0, pos_o, :].mean(dim=0)
                h_c = out_c.hidden_states[l][0, pos_c, :].mean(dim=0)
                cos = torch.nn.functional.cosine_similarity(h_o.unsqueeze(0), h_c.unsqueeze(0)).item()
                l2 = torch.norm(h_o - h_c).item()
                cosines[l].append(cos)
                l2_dists[l].append(l2)
                orig_feats[l].append(h_o.cpu().numpy())
                cf_feats[l].append(h_c.cpu().numpy())

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Discrimination probe: can a linear classifier tell original from counterfactual?
    disc_results = {}
    for l in layers:
        X = np.concatenate([np.stack(orig_feats[l]), np.stack(cf_feats[l])], axis=0).astype('float32')
        y = np.concatenate([np.zeros(len(orig_feats[l])), np.ones(len(cf_feats[l]))]).astype(np.int64)
        # Simple split: first 70% train, last 30% test
        n = len(orig_feats[l])
        n_train = int(0.7 * n)
        idx = list(range(n))
        random.Random(272).shuffle(idx)
        train_idx = idx[:n_train] + [i + n for i in idx[:n_train]]
        test_idx = idx[n_train:] + [i + n for i in idx[n_train:]]
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx], y[test_idx]
        # Standardize and ridge
        mu = Xtr.mean(axis=0, keepdims=True); sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr_n = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1), dtype='float32')], axis=1)
        Xte_n = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1), dtype='float32')], axis=1)
        Y = np.zeros((len(ytr), 2), dtype='float32'); Y[np.arange(len(ytr)), ytr] = 1.0
        A = Xtr_n.T @ Xtr_n + 10.0 * np.eye(Xtr_n.shape[1], dtype='float32')
        W = np.linalg.solve(A, Xtr_n.T @ Y)
        pred = (Xte_n @ W).argmax(axis=1)
        disc_acc = float((pred == yte).mean() * 100.0)
        disc_results[l] = disc_acc

    return {
        'cosine_mean': {l: float(np.mean(cosines[l])) for l in layers},
        'cosine_std': {l: float(np.std(cosines[l])) for l in layers},
        'cosine_min': {l: float(np.min(cosines[l])) for l in layers},
        'l2_mean': {l: float(np.mean(l2_dists[l])) for l in layers},
        'discrimination_accuracy': disc_results,
        'num_pairs': len(pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default=str(RAW_DEFAULT))
    ap.add_argument('--max_docs', type=int, default=3000)
    ap.add_argument('--max_pairs', type=int, default=500)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--layers', nargs='+', type=int, default=[2, 4, 6, 8])
    ap.add_argument('--seed', type=int, default=272)
    ap.add_argument('--out_json', default=str(OUT_DEFAULT))
    ap.add_argument('--out_note', default=str(NOTE_DEFAULT))
    ap.add_argument('--models', nargs='*', default=[
        'wwm42_40M=experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_40M',
        'wwm42_100M=experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M',
        'wwm43_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_40M',
        'wwm43_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model/chck_100M',
        'amlm42_40M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed42_100M_b256/hf_model/chck_40M',
        'amlm42_100M=experiments/archive/initial_model_studies/training/runs/fullcycle_official_amlm_debertav2_8x480_seed42_100M_b256/hf_model/chck_100M',
    ])
    args = ap.parse_args()
    setup_env()
    t0 = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Building counterfactual pairs...", flush=True)
    pairs = build_pairs(pathlib.Path(args.raw_dir), args.max_docs, args.max_pairs, args.seed)
    print(f"  Built {len(pairs)} pairs", flush=True)
    if len(pairs) < 30:
        raise RuntimeError(f"Too few pairs: {len(pairs)}")

    model_specs = []
    for m in args.models:
        name, path = m.split('=', 1)
        model_specs.append((name, pathlib.Path(path)))

    payload = {
        'status': 'COUNTERFACTUAL_HISTORY_PROBE',
        'task': 'history-swap counterfactual: does changing earlier mention alter representation at later mention?',
        'num_pairs': len(pairs),
        'layers': args.layers,
        'max_len': args.max_len,
        'pair_stats': {
            'mean_gap_words': float(np.mean([p.history_gap_words for p in pairs])),
            'median_gap_words': float(np.median([p.history_gap_words for p in pairs])),
            'unique_target_words': len(set(p.target_word for p in pairs)),
            'unique_swap_words': len(set(p.swap_word for p in pairs)),
        },
        'models': {},
        'elapsed_sec': None,
    }

    for name, path in model_specs:
        if not path.exists():
            print(f"  SKIP {name}: {path} not found", flush=True)
            continue
        print(f"  Running {name}...", flush=True)
        result = run_counterfactual_analysis(path, pairs, args.layers, args.max_len, device)
        payload['models'][name] = result
        # Incremental save
        pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out_json).write_text(json.dumps(payload, indent=2) + '\n')

    payload['elapsed_sec'] = time.time() - t0
    pathlib.Path(args.out_json).write_text(json.dumps(payload, indent=2) + '\n')

    # Write note
    lines = [
        '# research counterfactual history probe', '',
        f'Evidence JSON: `{args.out_json}`', '',
        f'Pairs: {len(pairs)}; mean gap {payload["pair_stats"]["mean_gap_words"]:.1f} words; '
        f'{payload["pair_stats"]["unique_target_words"]} unique target words; '
        f'{payload["pair_stats"]["unique_swap_words"]} unique swap words', '',
        '## Results', '',
        '| model | L2 cos | L4 cos | L6 cos | L8 cos | L2 disc | L4 disc | L6 disc | L8 disc |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for name, d in payload['models'].items():
        cos = d['cosine_mean']
        disc = d['discrimination_accuracy']
        lines.append(f"| {name} | {cos.get('2',cos.get(2,0)):.5f} | {cos.get('4',cos.get(4,0)):.5f} | "
                     f"{cos.get('6',cos.get(6,0)):.5f} | {cos.get('8',cos.get(8,0)):.5f} | "
                     f"{disc.get('2',disc.get(2,0)):.1f} | {disc.get('4',disc.get(4,0)):.1f} | "
                     f"{disc.get('6',disc.get(6,0)):.1f} | {disc.get('8',disc.get(8,0)):.1f} |")
    lines += ['', '## Interpretation', '',
              'If mean cosine ≈ 1.0 and discrimination ≈ 50%: model ignores history entity identity → '
              'research gain was from topic/lexical cues, not entity-state tracking.',
              'If cosine < 0.99 and discrimination > 60%: genuine entity-identity signal propagates '
              'across sentences → intermediate-state supervision has a real target.']
    pathlib.Path(args.out_note).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out_note).write_text('\n'.join(lines) + '\n')
    print(json.dumps({
        'out_json': args.out_json, 'out_note': args.out_note,
        'num_pairs': len(pairs), 'elapsed_sec': payload['elapsed_sec']
    }, indent=2))


if __name__ == '__main__':
    main()
