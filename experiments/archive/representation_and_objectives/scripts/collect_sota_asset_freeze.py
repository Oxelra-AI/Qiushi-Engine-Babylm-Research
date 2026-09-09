#!/usr/bin/env python3
"""Collect immutable references for the research SOTA asset freeze.

This script does not modify protected artifacts. It records existence, sizes, and
selected SHA256 hashes for the current score-bearing BabyLM Strict-Small assets
and the trusted data/tokenizer coordinate used for the compact-view mechanism
study.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path('.')
OUT_DIR = ROOT / 'experiments/archive/representation_and_objectives/data/sota_asset_freeze'
OUT_JSON = OUT_DIR / 'sota_asset_freeze.json'


def sha256_file(path: Path, max_bytes: int | None = None) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    read = 0
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            if max_bytes is not None and read + len(chunk) > max_bytes:
                chunk = chunk[: max_bytes - read]
            h.update(chunk)
            read += len(chunk)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def file_rec(path: str, do_hash: bool = True) -> dict[str, Any]:
    p = ROOT / path
    rec: dict[str, Any] = {'path': path, 'exists': p.exists(), 'type': 'dir' if p.is_dir() else 'file' if p.is_file() else 'missing'}
    if p.exists():
        rec['size'] = p.stat().st_size if p.is_file() else None
    if do_hash and p.is_file():
        rec['sha256'] = sha256_file(p)
    return rec


def read_json(path: str) -> Any:
    p = ROOT / path
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def summarize_hf_model(path: str) -> dict[str, Any]:
    p = ROOT / path
    rec: dict[str, Any] = file_rec(path, do_hash=False)
    if p.is_dir():
        for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json']:
            fp = p / name
            if fp.exists():
                rec[name] = {'size': fp.stat().st_size, 'sha256': sha256_file(fp)}
            else:
                rec[name] = {'missing': True}
        ckpts = sorted([x.name for x in p.iterdir() if x.is_dir() and x.name.startswith('chck_')])
        rec['checkpoint_dir_count'] = len(ckpts)
        rec['first_checkpoints'] = ckpts[:5]
        rec['last_checkpoints'] = ckpts[-5:]
    return rec


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'status': 'SOTA_ASSET_FREEZE',
        'purpose': 'Preserve score-bearing endpoints and exact data/tokenizer/provenance references as a stable base while the session pivots to mechanism-level research.',
        'score_bearing_endpoints': {
            'protected_chck82': {
                'overall': 41.942481167385985,
                'model_sha_recorded': '93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3',
                'carrier_sha_recorded': 'dcad3d8cf285a69d31910a32c62c80689022a3ff459a0956401ab7f4b3542237',
                'source_hf_model': summarize_hf_model('experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M'),
                'from_corpus_repro_hf_model': summarize_hf_model('experiments/archive/representation_and_objectives/training/runs/adapter128_scale1p75_from_corpus_fixed82M_seed43022/hf_model/chck_82M'),
                'hardened_summary': read_json('experiments/archive/representation_and_objectives/data/scale1p75_chck82_full_eval_reproduction/summary/scale1p75_100M_full_eval_hardened_summary.json'),
                'carrier_verifier': read_json('experiments/archive/representation_and_objectives/data/chck82_fast_submission_materialization/fast_submission_verification.json'),
            },
            'coherent86_alpha0p75_projected': {
                'overall_aoa0': 42.1210247099666,
                'cheap7': 44.18142857142857,
                'superglue': 69.81922238969935,
                'model_sha_recorded': 'e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8',
                'carrier_sha_recorded': '40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994',
                'aoa_status': 'unmeasured; current carrier root lacks 19 required checkpoint directories',
                'carrier_manifest': file_rec('research/documents/frontier_consolidation/data/truthful_private_scale_carriers/coherent86_alpha0p75/truthful_coherent86_alpha0p75_carrier_manifest.md', do_hash=True),
                'aoa_preflight': read_json('experiments/archive/representation_and_objectives/data/alpha075_aoa_preflight/official_aoa_min0_coherent86_alpha0p75_dryrun.json'),
            },
            'ordinary84_backup_projected': {
                'overall_aoa0': 42.01583738094448,
                'cheap7': 44.122142857142855,
                'superglue': 69.28753642850036,
                'synthesis': read_json('experiments/archive/representation_and_objectives/data/ordinary84_candidate_synthesis/ordinary84_candidate_synthesis.json'),
            },
        },
        'trusted_compact_view_coordinate': {
            'overlay_metadata': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json', do_hash=True),
            'legal_reinvest_pool_10M': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl', do_hash=True),
            'legal_reinvest_stream_100M': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl', do_hash=True),
            'repeat_reinvest_pool_10M': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_10M.jsonl', do_hash=True),
            'repeat_reinvest_stream_100M': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl', do_hash=True),
            'lengthmatched_reinvest_pool_10M': file_rec('experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_lengthmatched_compact_reinvest_10M.jsonl', do_hash=True),
            'tokenizer_baseline16k': summarize_hf_model('experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model'),
            'shared16k_tokenizer': summarize_hf_model('experiments/archive/representation_and_objectives/data/shared_tokenizer/shared_16k_tokenizer'),
        },
        'existing_mechanism_runs': {
            'compact_view_reinvest_seed43022': summarize_hf_model('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model'),
            'compact_view_reinvest_metrics': read_json('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/scientific_metrics.json'),
            'compact_view_reinvest_full_note': file_rec('research/notes/frontier_consolidation/reinvest_sota_anatomy.md', do_hash=True),
            'compact_view_core_metrics': read_json('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_core_neutral_16k_seed43022/scientific_metrics.json'),
            'compact_repeat_core_metrics': read_json('experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_repeat_compact_core_neutral_16k_seed43022/scientific_metrics.json'),
            'fw_compact_vs_breadth': {
                'compact_metrics': read_json('experiments/archive/frontier_consolidation/training/runs/fw_compact_view_shared16k_seed43022/scientific_metrics.json'),
                'breadth_metrics': read_json('experiments/archive/frontier_consolidation/training/runs/fw_source_breadth_shared16k_seed43022/scientific_metrics.json'),
            },
        },
        'governing_policy': {
            'current_scientific_goal': 'use the SOTA coordinate as a reliable base for generalizable mechanism discovery, not as the endpoint of the research',
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out_json': str(OUT_JSON)}, indent=2))

if __name__ == '__main__':
    main()
