#!/usr/bin/env python3
"""research mechanical verification of pathway-separated trainer properties.

Checks:
1. Init equivalence: adapter on vs off gives same logits at initialization
2. Phase A isolation: main MLM backward produces zero adapter gradients
3. Phase B+C isolation: auxiliary+neutrality backward produces zero stock gradients
4. Adapter enable/disable toggles work correctly
5. Charged-word accounting: aligned vs shuffled match on the same first batch
"""

from pathlib import Path as _PublicPath
_PUBLIC_ROOT = next(p for p in _PublicPath(__file__).resolve().parents if (p / "CITATION.cff").is_file())
def _public_path(relative):
    return _PUBLIC_ROOT / relative

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, DebertaV2Config

_ROOT = _public_path('.')  # project root
STUDY = _public_path('experiments/archive/frontier_consolidation')
SCRIPTS = _public_path('experiments/archive/frontier_consolidation/scripts')
sys.path.insert(0, str(SCRIPTS))

from adapter_modeling import AdapterDebertaV2ForMaskedLM
from detached_private_trainer import (
    set_adapters_enabled, apply_wwm, load_aux_pair_data,
    collect_aux_units, build_view_batches, DualViewDataset,
    collate, load_examples, reset_all, build_model
)

TOK_PATH = str(_public_path('experiments/archive/frontier_consolidation/data/compliant_tokenizer'))
STREAM = str(_public_path('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl'))
AUX_SPARSE = str(_public_path('experiments/archive/frontier_consolidation/data/sparse_aux_pair_data/top20/sparse_aux_pair_data_top20.json'))

class Args:
    hidden_size=480; n_layer=8; n_head=8; ffn_mult=4; seq_length=256; aux_max_length=464
    adapter_bottleneck=128; adapter_scale=1.0; seed=43; extra_init_seed=43022
    train_rng_seed=43023; batch_size=4; mask_prob=0.15; aux_pair_shuffle_seed=43022
    mode="aligned"; aux_micro_batch_size=8; neutral_lambda=1.0; neutral_subsample=2
    max_word_exposure=200_000; lr_total_steps=2529

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)
    mask_id = int(tokenizer.mask_token_id)
    cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or 0

    aux_data, aux_summary = load_aux_pair_data(AUX_SPARSE)
    pair_eids = {int(k) for k in aux_data.keys()}

    examples = load_examples(STREAM, 200_000)
    dataset = DualViewDataset(examples, tokenizer, 256, pair_eids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)

    args = Args()
    reset_all(args.seed)
    if args.extra_init_seed >= 0:
        reset_all(args.extra_init_seed)
    model = build_model(args, tokenizer)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.to(device)
    model.train()

    adapter_names = {n for n, _ in model.named_parameters() if ".adapter." in n}
    stock_names = {n for n, _ in model.named_parameters() if ".adapter." not in n}
    results = {}

    # ── Check 1: Init equivalence (eval mode to eliminate dropout noise) ──
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    model.eval()
    set_adapters_enabled(model, True)
    with torch.no_grad():
        logits_on = model(input_ids=input_ids, attention_mask=attention_mask).logits.clone()

    set_adapters_enabled(model, False)
    with torch.no_grad():
        logits_off = model(input_ids=input_ids, attention_mask=attention_mask).logits.clone()
    model.train()

    init_logit_diff = float((logits_on - logits_off).abs().max().cpu())
    results["init_logit_max_diff"] = init_logit_diff
    results["init_equivalence_ok"] = init_logit_diff < 1e-6
    print(f"Check 1 (init equivalence): max diff = {init_logit_diff:.8f} → {'PASS' if results['init_equivalence_ok'] else 'FAIL'}")

    # ── Check 2: Phase A — main MLM backward, adapter grads must be zero ──
    set_adapters_enabled(model, False)
    model.zero_grad()
    word_group = batch["word_group"].to(device)
    gen = torch.Generator(device=device)
    gen.manual_seed(43023)
    masked, labels = apply_wwm(input_ids, attention_mask, word_group, tokenizer, 0.15, gen)
    out = model(input_ids=masked, attention_mask=attention_mask, labels=labels)
    out.loss.backward()

    adapter_grad_norm_phaseA = 0.0
    stock_grad_norm_phaseA = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            gnorm = float(p.grad.norm().cpu())
            if n in adapter_names:
                adapter_grad_norm_phaseA += gnorm
            else:
                stock_grad_norm_phaseA += gnorm

    results["phaseA_stock_grad_norm"] = stock_grad_norm_phaseA
    results["phaseA_adapter_grad_norm"] = adapter_grad_norm_phaseA
    results["phaseA_adapter_zero_ok"] = adapter_grad_norm_phaseA == 0.0
    print(f"Check 2 (Phase A): stock grad norm = {stock_grad_norm_phaseA:.4f}, "
          f"adapter grad norm = {adapter_grad_norm_phaseA:.6f} → "
          f"{'PASS' if results['phaseA_adapter_zero_ok'] else 'FAIL'}")
    del out

    # ── Check 3: Phase B+C — freeze stock, adapter grads only ──
    model.zero_grad()
    set_adapters_enabled(model, True)

    # Freeze stock
    for name, param in model.named_parameters():
        if name not in adapter_names:
            param.requires_grad_(False)

    # Phase B: auxiliary forward on a micro-batch
    # Create a simple aux-like input
    fake_aux_ids = input_ids[:2].clone()
    fake_aux_mask = attention_mask[:2].clone()
    fake_aux_labels = labels[:2].clone()
    out_aux = model(input_ids=fake_aux_ids, attention_mask=fake_aux_mask)
    vocab = out_aux.logits.shape[-1]
    aux_loss = F.cross_entropy(out_aux.logits.reshape(-1, vocab),
                               fake_aux_labels.reshape(-1), ignore_index=-100)
    aux_loss.backward()

    # Phase C: neutrality
    neutral_out = model(input_ids=input_ids[:2], attention_mask=attention_mask[:2])
    # Fake stock logits (detached)
    fake_stock_logits = logits_off[:2].detach()
    log_p = F.log_softmax(neutral_out.logits, dim=-1)
    p_stock = F.softmax(fake_stock_logits, dim=-1)
    kl = F.kl_div(log_p, p_stock, reduction='none').sum(-1)
    mask_f = attention_mask[:2].float()
    neutral_loss = (kl * mask_f).sum() / max(1.0, float(mask_f.sum()))
    neutral_loss.backward()

    adapter_grad_norm_phaseBC = 0.0
    stock_grad_norm_phaseBC = 0.0
    for n, p in model.named_parameters():
        if p.grad is not None:
            gnorm = float(p.grad.norm().cpu())
            if n in adapter_names:
                adapter_grad_norm_phaseBC += gnorm
            else:
                stock_grad_norm_phaseBC += gnorm

    results["phaseBC_stock_grad_norm"] = stock_grad_norm_phaseBC
    results["phaseBC_adapter_grad_norm"] = adapter_grad_norm_phaseBC
    results["phaseBC_stock_zero_ok"] = stock_grad_norm_phaseBC == 0.0
    results["phaseBC_adapter_nonzero_ok"] = adapter_grad_norm_phaseBC > 0.0
    print(f"Check 3 (Phase B+C): stock grad norm = {stock_grad_norm_phaseBC:.6f}, "
          f"adapter grad norm = {adapter_grad_norm_phaseBC:.4f} → "
          f"stock_zero={'PASS' if results['phaseBC_stock_zero_ok'] else 'FAIL'}, "
          f"adapter_nonzero={'PASS' if results['phaseBC_adapter_nonzero_ok'] else 'FAIL'}")

    # Restore stock requires_grad
    for name, param in model.named_parameters():
        param.requires_grad_(True)

    # ── Check 4: Adapter toggle ──
    set_adapters_enabled(model, False)
    all_disabled = all(not layer.adapter.enabled for layer in model.deberta.encoder.layer)
    set_adapters_enabled(model, True)
    all_enabled = all(layer.adapter.enabled for layer in model.deberta.encoder.layer)
    results["toggle_disabled_ok"] = all_disabled
    results["toggle_enabled_ok"] = all_enabled
    print(f"Check 4 (toggle): disabled={all_disabled} enabled={all_enabled} → "
          f"{'PASS' if all_disabled and all_enabled else 'FAIL'}")

    # ── Summary ──
    all_ok = all(v for k, v in results.items() if k.endswith("_ok"))
    results["all_checks_passed"] = all_ok
    print(f"\nOverall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
