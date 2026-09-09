#!/usr/bin/env python3
"""research final no-truncation/equal-update recheck for paired-restatement route.

Same four-arm experiment as research, but using research token-aware equal-row
JSONLs: exact 1M words, 5988 rows per arm, 47 b128 updates, zero examples over
256 baseline16k tokens. This removes the research truncation/update confounds.
"""
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
TRAIN = ROOT/'training/scripts/babylm_masked_train.py'
DATA = ROOT/'data/same_entity_deranged_1M_tokenaware_equalrows'
META = DATA/'materialization_meta.json'
PYBIN = sys.executable
ARMS = ['true_pair_adjacent', 'hard_negative_same_entity', 'orig_only', 'shuffled_pair_adjacent']
RUN_DIRS = {a: ROOT/f'training/runs/step296_{a}_debertav2_8x480_1M_b128_seed42_tokenaware' for a in ARMS}
OUT_JSON = ROOT/'data/pair_discriminator_tokenaware_1m_profile.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/pair_discriminator_tokenaware_1m_profile.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/pair_discriminator_tokenaware_1m.log')
TASKS = [
    ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
    ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
    ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
    ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
    ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']

def setup_env():
    env = os.environ.copy()
    hf = ROOT/'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf/'hub').resolve())
    env['HF_DATASETS_CACHE'] = str((hf/'datasets').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf/'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT/'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    for k in ['HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, logf, cwd=None):
    line = '$ ' + ' '.join(map(str, cmd))
    print(line, flush=True); logf.write('\n'+line+'\n'); logf.flush()
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-3000:], flush=True); logf.write(p.stdout+f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f'failed {p.returncode}: {line}\n{p.stdout[-8000:]}')

def train_arm(arm, env, logf):
    run_dir = RUN_DIRS[arm]
    jsonl = DATA/f'{arm}_1000000w_tokenaware_equalrows.jsonl'
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)
    cmd = [PYBIN, str(TRAIN),
           '--output_dir', str(run_dir),
           '--example_jsonl', str(jsonl),
           '--example_jsonl_label', f'step296_{arm}_1M_b128_tokenaware_equalrows',
           '--example_jsonl_meta', str(META),
           '--max_word_exposure', '1000000', '--example_pool_words', '1000000', '--checkpoint_words', '1000000',
           '--words_per_example', '160', '--tokenizer_label', 'baseline16k', '--tokenization_summary_limit', '0',
           '--mask_mode', 'wwm', '--mask_prob', '0.15',
           '--seq_length', '256', '--max_seq_length', '256', '--max_position_embeddings', '512',
           '--model_type', 'deberta_v2', '--hidden_size', '480', '--n_layer', '8', '--n_head', '12', '--ffn_mult', '4',
           '--batch_size', '128', '--learning_rate', '0.001', '--weight_decay', '0.01', '--warmup_fraction', '0.05',
           '--seed', '42', '--extra_init_seed', '456', '--train_rng_seed', '789', '--log_every', '10']
    run(cmd, env, logf)

def read_avg(report):
    txt = pathlib.Path(report).read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'cannot parse average from {report}\n{txt[:1000]}')
    return float(m.group(1))

def read_reading(report):
    txt = pathlib.Path(report).read_text(encoding='utf-8', errors='replace'); out = {}
    for label, key in [('EYE TRACKING SCORE','reading_eye_tracking'),('SELF-PACED READING SCORE','reading_self_paced')]:
        m = re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m:
            raise RuntimeError(f'cannot parse {label} from {report}')
        out[key] = float(m.group(1))
    out['Reading_mean'] = (out['reading_eye_tracking']+out['reading_self_paced'])/2.0
    return out

def load_metrics(run_dir):
    p = run_dir/'scientific_metrics.json'
    if not p.exists():
        return {}
    m = json.loads(p.read_text(encoding='utf-8'))
    s = m.get('tokenization_coupling_summary') or {}
    return {'parameter_count': m.get('parameter_count'), 'word_exposure': m.get('word_exposure'),
            'steps': m.get('actual_training_steps'), 'loss_first': m.get('loss_first'), 'loss_last': m.get('loss_last'),
            'truncated_fraction': s.get('truncated_example_fraction'), 'max_untruncated_tokens': s.get('max_untruncated_tokens'),
            'kept_tokens_per_word': s.get('kept_tokens_per_whitespace_word'),
            'masked_tokens_per_word': m.get('masked_tokens_per_whitespace_word')}

def profile_arm(arm, env, logf):
    run_dir = RUN_DIRS[arm]
    ckpt = run_dir/'hf_model'/'chck_1M'
    if not (ckpt/'config.json').exists():
        raise FileNotFoundError(ckpt/'config.json')
    outdir = run_dir/'eval_fast_profile'
    scores = {}; reports = {}
    for task_name, task, data_path, ds_name in TASKS:
        run([PYBIN,'-m','evaluation_pipeline.sentence_zero_shot.run',
             '--model_path_or_name', str((run_dir/'hf_model').resolve()), '--backend','mlm',
             '--task',task,'--data_path',data_path,'--save_predictions','--revision_name','chck_1M',
             '--batch_size','64','--output_dir',str(outdir.resolve())], env, logf, cwd=str(STRICT))
        report = outdir/'hf_model'/'chck_1M'/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name] = read_avg(report); reports[task_name] = str(report)
    run([PYBIN,'-m','evaluation_pipeline.reading.run',
         '--model_path_or_name', str((run_dir/'hf_model').resolve()), '--backend','mlm',
         '--data_path','evaluation_data/fast_eval/reading/reading_data.csv', '--revision_name','chck_1M',
         '--output_dir',str(outdir.resolve())], env, logf, cwd=str(STRICT))
    rr = outdir/'hf_model'/'chck_1M'/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading'] = str(rr)
    return {'run_dir': str(run_dir), 'scores': scores, 'reports': reports, 'metrics_summary': load_metrics(run_dir)}

def main():
    t0 = time.time(); env = setup_env(); LOG.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text())
    for a in ARMS:
        assert meta['arms'][a]['exact_words'] == 1000000
        assert meta['arms'][a]['rows'] == 5988
    rows = {}
    with LOG.open('a', encoding='utf-8') as logf:
        logf.write(f'\n===== research tokenaware discriminator start {time.ctime()} cuda={os.environ.get("CUDA_VISIBLE_DEVICES")} =====\n')
        for a in ARMS:
            logf.write(f'\n--- training {a} ---\n'); logf.flush(); train_arm(a, env, logf)
        for a in ARMS:
            logf.write(f'\n--- profiling {a} ---\n'); logf.flush(); rows[a] = profile_arm(a, env, logf)
    tp = rows['true_pair_adjacent']['scores']; hn = rows['hard_negative_same_entity']['scores']
    oo = rows['orig_only']['scores']; sh = rows['shuffled_pair_adjacent']['scores']
    primary = {k: round(tp[k]-hn[k], 4) for k in COLS}; vs_orig = {k: round(tp[k]-oo[k], 4) for k in COLS}; vs_shuf = {k: round(tp[k]-sh[k], 4) for k in COLS}
    payload = {'status':'PAIR_DISCRIMINATOR_TOKENAWARE_1M_PROFILE', 'arms':rows,
               'true_minus_hardneg':primary, 'true_minus_orig':vs_orig, 'true_minus_shuffled':vs_shuf,
               'elapsed_sec': time.time()-t0,
               'design_note':'Final no-truncation/equal-update recheck: all arms 1M words, 5988 rows, 47 b128 updates, zero examples >256 baseline16k tokens.'}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True); OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = ['# research token-aware/equal-row pair discriminator 1M profile','',f'Evidence JSON: `{OUT_JSON}`','',
             '| column | true_pair | hard_neg | orig | shuffled | true-hardneg | true-orig | true-shuf |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in COLS:
        lines.append(f'| {k} | {tp[k]:.4f} | {hn[k]:.4f} | {oo[k]:.4f} | {sh[k]:.4f} | {primary[k]:+.4f} | {vs_orig[k]:+.4f} | {vs_shuf[k]:+.4f} |')
    lines += ['','## Training/tokenization summary','','| arm | loss first | loss last | steps | trunc frac | max untrunc tokens | kept tokens/word | masked tokens/word |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a,d in rows.items():
        m=d['metrics_summary']; lines.append(f"| {a} | {m.get('loss_first')} | {m.get('loss_last')} | {m.get('steps')} | {m.get('truncated_fraction')} | {m.get('max_untruncated_tokens')} | {m.get('kept_tokens_per_word')} | {m.get('masked_tokens_per_word')} |")
    OUT_NOTE.parent.mkdir(parents=True, exist_ok=True); OUT_NOTE.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(OUT_JSON),'note':str(OUT_NOTE),'true_minus_hardneg':primary,'true_minus_orig':vs_orig,'true_minus_shuffled':vs_shuf,'elapsed_sec':payload['elapsed_sec']}, indent=2))

if __name__ == '__main__':
    main()
