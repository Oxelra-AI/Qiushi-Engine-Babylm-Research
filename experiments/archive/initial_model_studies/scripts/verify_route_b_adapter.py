#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, pathlib, shutil, sys
import torch
from transformers import AutoModelForMaskedLM, DebertaV2Config, DebertaV2ForMaskedLM

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
SRC = ROOT / 'scripts/route_b_adapter_screen.py'
OUT = ROOT / 'data/route_b_adapter_verification.json'
TMP = ROOT / 'data/route_b_export_tmp'

spec = importlib.util.spec_from_file_location('route_b', SRC)
route_b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = route_b
spec.loader.exec_module(route_b)


def make_tiny_base(tok):
    cfg = DebertaV2Config(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, max_position_embeddings=64, max_relative_positions=32,
        position_buckets=32, relative_attention=True, pos_att_type=['p2c','c2p'],
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
    )
    return DebertaV2ForMaskedLM(cfg)


def main():
    torch.manual_seed(123)
    tok_path = str((ROOT / 'training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_10M').resolve())
    tok = route_b.trainer.make_portable_tokenizer(tok_path)
    ids = torch.tensor([[10, 11, 12, 13, 14, tok.pad_token_id], [20, 21, 22, 23, tok.pad_token_id, tok.pad_token_id]], dtype=torch.long)
    att = (ids != tok.pad_token_id).long()
    labels = ids.clone(); labels[labels == tok.pad_token_id] = -100

    # 1. Baseline wrapper equivalence.
    base = make_tiny_base(tok).eval()
    wrap = route_b.AdaptedDeBERTaForMLM(base, None).eval()
    with torch.no_grad():
        logits_base = base(input_ids=ids, attention_mask=att).logits
        logits_wrap = wrap(input_ids=ids, attention_mask=att).logits
    max_abs_baseline = float((logits_base - logits_wrap).abs().max())

    # 2. Feature determinism and adapter param equality.
    f1 = route_b.build_surface_features(tok, max_features=2560)
    f2 = route_b.build_surface_features(tok, max_features=2560)
    deterministic = bool(torch.equal(f1, f2))
    token_features = route_b.build_token_id_features(len(tok), max_features=2560)
    surf_adapter = route_b.EmbeddingAdapter(32, 2560, f1)
    tid_adapter = route_b.EmbeddingAdapter(32, 2560, token_features)
    surf_params = surf_adapter.extra_params()
    tid_params = tid_adapter.extra_params()
    feature_stats = {
        'surface_shape': list(f1.shape),
        'token_id_shape': list(token_features.shape),
        'surface_nonzero_rows': int((f1.abs().sum(dim=1) > 0).sum()),
        'surface_mean_active_features': float((f1 != 0).sum(dim=1).float().mean()),
        'surface_max_active_features': int((f1 != 0).sum(dim=1).max()),
        'surface_row_norm_mean': float(f1.norm(dim=1).mean()),
    }

    # 3. Fused export equivalence. Use nonzero gate/proj and eval mode.
    base2 = make_tiny_base(tok).eval()
    adapter = route_b.EmbeddingAdapter(32, 2560, f1)
    with torch.no_grad():
        adapter.gate.fill_(0.137)
        adapter.proj.weight.normal_(0, 0.01)
    model = route_b.AdaptedDeBERTaForMLM(base2, adapter).eval()
    with torch.no_grad():
        logits_adapted = model(input_ids=ids, attention_mask=att).logits
    if TMP.exists():
        shutil.rmtree(TMP)
    route_b.save_fused_hf(model, tok, TMP)
    reloaded = AutoModelForMaskedLM.from_pretrained(TMP).eval()
    with torch.no_grad():
        logits_fused = reloaded(input_ids=ids, attention_mask=att).logits
    max_abs_fused = float((logits_adapted - logits_fused).abs().max())
    # Verify live model restored after export.
    with torch.no_grad():
        logits_after_restore = model(input_ids=ids, attention_mask=att).logits
    max_abs_restore = float((logits_adapted - logits_after_restore).abs().max())

    payload = {
        'status': 'ROUTE_B_ADAPTER_VERIFICATION',
        'baseline_wrapper_max_abs_diff': max_abs_baseline,
        'feature_deterministic': deterministic,
        'surface_adapter_params': surf_params,
        'token_id_adapter_params': tid_params,
        'feature_stats': feature_stats,
        'fused_export_max_abs_diff': max_abs_fused,
        'post_export_restore_max_abs_diff': max_abs_restore,
        'tmp_export': str(TMP),
        'pass': (max_abs_baseline < 1e-5 and deterministic and surf_params == tid_params and max_abs_fused < 2e-5 and max_abs_restore < 1e-6),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))
    if not payload['pass']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
