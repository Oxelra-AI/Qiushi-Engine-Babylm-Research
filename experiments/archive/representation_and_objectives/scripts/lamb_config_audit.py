#!/usr/bin/env python3
"""research: audit the research LAMB trainer's architecture/config coordinate.

This checks whether the active LAMB endpoints can be interpreted as matched
legal40k architecture comparisons.  The smoke checkpoint already carries the
same create_model defaults used by the full runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path('.')
OUT = ROOT / 'experiments/archive/representation_and_objectives/data/lamb_config_audit'
NOTE = ROOT / 'research/notes/representation_and_objectives/lamb_config_audit.md'
TOKENIZER = ROOT / 'experiments/archive/representation_and_objectives/data/legal_representation_route_map/tokenizers/legal_byte_bpe_40k'
CONFIGS = {
    'legal40k_8x480_adamw_baseline': ROOT / 'experiments/archive/representation_and_objectives/training/runs/legal40k_accum_compact_view_reinvest_seed43022/hf_model/chck_100M/config.json',
    'legal40k_12x384_adamw_depth': ROOT / 'experiments/archive/representation_and_objectives/training/runs/legal40k_12x384_depth_compact_view_reinvest_seed43022/hf_model/chck_100M/config.json',
    'lamb_smoke_12x384': ROOT / 'experiments/archive/representation_and_objectives/data/smoke_test/hf_model/chck_final/config.json',
}
FIELDS = [
    'hidden_size','num_hidden_layers','num_attention_heads','intermediate_size','vocab_size',
    'pad_token_id','bos_token_id','eos_token_id','type_vocab_size',
    'relative_attention','max_relative_positions','position_buckets','pos_att_type','position_biased_input','norm_rel_ebd',
    'max_position_embeddings','layer_norm_eps','hidden_dropout_prob','attention_probs_dropout_prob',
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {'__missing__': str(path)}
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    NOTE.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER), use_fast=True)
    tokenizer_ids = {
        'pad_token': tok.pad_token,
        'pad_token_id': tok.pad_token_id,
        'mask_token': tok.mask_token,
        'mask_token_id': tok.mask_token_id,
        'bos_token': tok.bos_token,
        'bos_token_id': tok.bos_token_id,
        'eos_token': tok.eos_token,
        'eos_token_id': tok.eos_token_id,
        'unk_token': tok.unk_token,
        'unk_token_id': tok.unk_token_id,
        'all_special_ids': list(tok.all_special_ids),
        'vocab_size': tok.vocab_size,
    }
    configs = {k: load_json(v) for k, v in CONFIGS.items()}
    field_table = {}
    for f in FIELDS:
        field_table[f] = {k: cfg.get(f) for k, cfg in configs.items()}
    # Differences of research smoke vs baselines that are not just size.
    mismatch_fields = []
    base12 = configs['legal40k_12x384_adamw_depth']
    smoke = configs['lamb_smoke_12x384']
    for f in FIELDS:
        if f in {'hidden_size','num_hidden_layers','num_attention_heads','intermediate_size'}:
            continue
        if smoke.get(f) != base12.get(f):
            mismatch_fields.append({'field': f, 'smoke': smoke.get(f), 'matched_12x384_baseline': base12.get(f)})
    tokenizer_config_consistency = {
        'pad_token_id_matches_tokenizer': smoke.get('pad_token_id') == tok.pad_token_id,
        'bos_token_id_matches_tokenizer': smoke.get('bos_token_id') == tok.bos_token_id,
        'eos_token_id_matches_tokenizer': smoke.get('eos_token_id') == tok.eos_token_id,
        'baseline_pad_token_id_matches_tokenizer': configs['legal40k_8x480_adamw_baseline'].get('pad_token_id') == tok.pad_token_id,
        'depth_pad_token_id_matches_tokenizer': base12.get('pad_token_id') == tok.pad_token_id,
    }
    summary = {
        'status': 'LAMB_CONFIG_AUDIT',
        'purpose': 'Detect architecture/config mismatches in the research LAMB trainer before interpreting endpoint scores.',
        'tokenizer_ids': tokenizer_ids,
        'config_paths': {k: str(v) for k, v in CONFIGS.items()},
        'field_table': field_table,
        'vs_matched_12x384_non_size_mismatches': mismatch_fields,
        'tokenizer_config_consistency': tokenizer_config_consistency,
        'interpretation': {
            'create_model_does_not_set_pad_bos_eos': True,
            'create_model_does_not_set_pos_att_type_or_max_relative_positions': True,
            'not_a_pure_lamb_curriculum_factor_test': True,
            'if_endpoint_scores_high': 'The run is still a legal trained model candidate, but mechanism attribution requires matched repaired-config controls.',
            'if_endpoint_scores_low': 'Failure could arise from LAMB/update allocation, curriculum geometry, and/or positional/padding config mismatch; do not use it alone to close LAMB or curriculum.',
        },
    }
    sp = OUT / 'lamb_config_audit.json'
    sp.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')

    lines = ['# research — config audit for research LAMB trainer', '']
    lines.append('The research smoke checkpoint was written by the same `create_model` code as the active full runs. It differs from the matched legal40k DeBERTa-v2 coordinate in non-optimizer ways.')
    lines.append('')
    lines.append('## Tokenizer special ids')
    lines.append(f"- tokenizer pad/mask/bos/eos/unk ids: pad={tok.pad_token_id}, mask={tok.mask_token_id}, bos={tok.bos_token_id}, eos={tok.eos_token_id}, unk={tok.unk_token_id}")
    lines.append('')
    lines.append('## Non-size config mismatches: research smoke vs matched 12×384 AdamW depth')
    lines.append('| field | research smoke | matched 12×384 AdamW |')
    lines.append('|---|---:|---:|')
    for m in mismatch_fields:
        lines.append(f"| {m['field']} | `{m['smoke']}` | `{m['matched_12x384_baseline']}` |")
    lines.append('')
    lines.append('Load-bearing mismatches:')
    lines.append('- research config uses `pad_token_id=0`, while the legal40k tokenizer pad id is 3 and matched baselines set pad id 3.')
    lines.append('- research config has `pos_att_type=null` and `max_relative_positions=-1`; matched baselines use `pos_att_type=[p2c,c2p]` and `max_relative_positions=256`.')
    lines.append('- research omits bos/eos ids in the saved config, unlike matched baselines.')
    lines.append('')
    lines.append('Therefore the active endpoints are a joint optimizer × sequence-curriculum × chunking/masking × config-coordinate intervention. They may still be legal models, but their scores cannot separate LAMB/curriculum from these architecture/config differences.')
    lines.append('')
    lines.append(f"Summary JSON: `{sp}`")
    NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': summary['status'], 'summary': str(sp), 'note': str(NOTE), 'mismatch_count': len(mismatch_fields)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
