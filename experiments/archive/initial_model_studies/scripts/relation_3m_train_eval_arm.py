#!/usr/bin/env python3
"""research FineWeb relation-vs-random 3M arm trainer/profiler.

Runs one arm (random_quality or relation_explicit) from the research 3M matched
FineWeb materialization. Same training configuration as research except exact 3M
exposure and checkpoints at 1M/2M/3M. Evaluates fast profile at each checkpoint
so relation-explicit EWoK persistence can be read as a trajectory.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT/'repos/babylm-eval/strict'
TRAIN = ROOT/'training/scripts/babylm_masked_train.py'
DATA = ROOT/'data/fineweb_relation_matched_3M'
META = DATA/'fineweb_relation_materialization_meta.json'
PYBIN = sys.executable
TASKS = [
    ('blimp_fast','blimp','evaluation_data/fast_eval/blimp_fast','blimp_fast'),
    ('supplement_fast','blimp','evaluation_data/fast_eval/supplement_fast','supplement_fast'),
    ('ewok_fast','ewok','evaluation_data/fast_eval/evaluation_data/fast_eval/ewok_fast','ewok_fast'),
    ('entity_tracking_fast','entity_tracking','evaluation_data/fast_eval/entity_tracking_fast','entity_tracking_fast'),
    ('comps','comps','evaluation_data/full_eval/comps','comps'),
]
COLS = ['blimp_fast','supplement_fast','ewok_fast','entity_tracking_fast','comps','reading_eye_tracking','reading_self_paced','Reading_mean']
CKPTS = ['chck_1M','chck_2M','chck_3M']
ARM_TO_JSONL = {
    'random_quality': DATA/'fineweb_random_quality_3000000w.jsonl',
    'relation_explicit': DATA/'fineweb_relation_explicit_3000000w.jsonl',
}


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


def count_words_rows(path: pathlib.Path):
    rows = words = 0
    for line in path.open(encoding='utf-8'):
        obj = json.loads(line); rows += 1; words += int(obj.get('words', len(obj['text'].split())))
    return rows, words


def train_arm(arm: str, run_dir: pathlib.Path, env, logf):
    jsonl = ARM_TO_JSONL[arm]
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)
    rows, words = count_words_rows(jsonl)
    if words != 3_000_000:
        raise RuntimeError(f'{arm}: expected 3M words, got {words}')
    logf.write(f'{arm} jsonl rows={rows} words={words}\n')
    cmd = [PYBIN, str(TRAIN),
           '--output_dir', str(run_dir),
           '--example_jsonl', str(jsonl),
           '--example_jsonl_label', f'fineweb_{arm}_3M_b128',
           '--example_jsonl_meta', str(META),
           '--max_word_exposure', '3000000', '--example_pool_words', '3000000', '--checkpoint_words', '1000000',
           '--words_per_example', '160', '--tokenizer_label', 'baseline16k', '--tokenization_summary_limit', '0',
           '--mask_mode', 'wwm', '--mask_prob', '0.15',
           '--seq_length', '256', '--max_seq_length', '256', '--max_position_embeddings', '512',
           '--model_type', 'deberta_v2', '--hidden_size', '480', '--n_layer', '8', '--n_head', '12', '--ffn_mult', '4',
           '--batch_size', '128', '--learning_rate', '0.001', '--weight_decay', '0.01', '--warmup_fraction', '0.05',
           '--seed', '42', '--extra_init_seed', '456', '--train_rng_seed', '789', '--log_every', '20']
    run(cmd, env, logf)


def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'cannot parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))


def read_reading(report: pathlib.Path) -> dict:
    txt = report.read_text(encoding='utf-8', errors='replace'); out = {}
    for label, key in [('EYE TRACKING SCORE','reading_eye_tracking'), ('SELF-PACED READING SCORE','reading_self_paced')]:
        m = re.search(re.escape(label)+r':\s*([0-9.\-]+)', txt)
        if not m:
            raise RuntimeError(f'cannot parse {label} from {report}\n{txt}')
        out[key] = float(m.group(1))
    out['Reading_mean'] = (out['reading_eye_tracking'] + out['reading_self_paced'])/2.0
    return out


def load_metrics(run_dir: pathlib.Path) -> dict:
    p = run_dir/'scientific_metrics.json'
    m = json.loads(p.read_text(encoding='utf-8'))
    s = m.get('tokenization_coupling_summary') or {}
    return {
        'parameter_count': m.get('parameter_count'),
        'word_exposure': m.get('word_exposure'),
        'steps': m.get('actual_training_steps'),
        'loss_first': m.get('loss_first'),
        'loss_last': m.get('loss_last'),
        'truncated_fraction': s.get('truncated_example_fraction'),
        'kept_tokens_per_word': s.get('kept_tokens_per_whitespace_word'),
        'masked_tokens_per_word': m.get('masked_tokens_per_whitespace_word'),
        'saved_checkpoints': m.get('saved_checkpoints'),
    }


def profile_ckpt(arm: str, run_dir: pathlib.Path, ckpt: str, env, logf) -> dict:
    cp = run_dir/'hf_model'/ckpt
    if not (cp/'config.json').exists():
        raise FileNotFoundError(cp/'config.json')
    outdir = run_dir/f'eval_step298_fast_profile_{ckpt}'
    scores = {}; reports = {}
    for task_name, task, data_path, ds_name in TASKS:
        run([PYBIN, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
             '--model_path_or_name', str((run_dir/'hf_model').resolve()), '--backend', 'mlm',
             '--task', task, '--data_path', data_path, '--save_predictions', '--revision_name', ckpt,
             '--batch_size', '64', '--output_dir', str(outdir.resolve())], env, logf, cwd=str(STRICT))
        report = outdir/'hf_model'/ckpt/'zero_shot'/'mlm'/task/ds_name/'best_temperature_report.txt'
        scores[task_name] = read_avg(report); reports[task_name] = str(report)
    run([PYBIN, '-m', 'evaluation_pipeline.reading.run',
         '--model_path_or_name', str((run_dir/'hf_model').resolve()), '--backend', 'mlm',
         '--data_path', 'evaluation_data/fast_eval/reading/reading_data.csv', '--revision_name', ckpt,
         '--output_dir', str(outdir.resolve())], env, logf, cwd=str(STRICT))
    rr = outdir/'hf_model'/ckpt/'zero_shot'/'mlm'/'reading'/'report.txt'
    scores.update(read_reading(rr)); reports['reading'] = str(rr)
    return {'scores': scores, 'reports': reports}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['random_quality','relation_explicit'], required=True)
    args = ap.parse_args()
    arm = args.arm
    run_dir = ROOT/f'training/runs/fineweb_{arm}_debertav2_8x480_3M_b128_seed42'
    out_json = ROOT/f'data/fineweb_{arm}_3m_profile.json'
    out_note = ROOT/f'notes/fineweb_{arm}_3m_profile.md'
    log = ROOT/f'notes/fineweb_{arm}_3m_profile.log'
    t0 = time.time(); env = setup_env(); log.parent.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding='utf-8'))
    assert meta['validation']['random_exact_words'] and meta['validation']['relation_exact_words']
    with log.open('a', encoding='utf-8') as logf:
        logf.write(f'\n===== research {arm} 3M start {time.ctime()} cuda={os.environ.get("CUDA_VISIBLE_DEVICES")} =====\n')
        train_arm(arm, run_dir, env, logf)
        ckpt_rows = {}
        for ckpt in CKPTS:
            logf.write(f'\n--- profiling {arm} {ckpt} ---\n'); logf.flush()
            ckpt_rows[ckpt] = profile_ckpt(arm, run_dir, ckpt, env, logf)
    payload = {'status':'FINEWEB_RELATION_3M_ARM_PROFILE', 'arm':arm, 'run_dir':str(run_dir),
               'checkpoints':ckpt_rows, 'metrics_summary':load_metrics(run_dir), 'elapsed_sec':time.time()-t0,
               'design_note':'Same as research FineWeb relation/random comparison, exact 3M exposure, checkpoints at 1M/2M/3M.'}
    out_json.parent.mkdir(parents=True, exist_ok=True); out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    lines = [f'# research FineWeb {arm} 3M profile', '', f'Evidence JSON: `{out_json}`', '',
             '| checkpoint | BLiMP | Supplement | EWoK | Entity | COMPS | Reading_mean |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for ckpt in CKPTS:
        s = ckpt_rows[ckpt]['scores']
        lines.append(f"| {ckpt} | {s['blimp_fast']:.4f} | {s['supplement_fast']:.4f} | {s['ewok_fast']:.4f} | {s['entity_tracking_fast']:.4f} | {s['comps']:.4f} | {s['Reading_mean']:.4f} |")
    m = payload['metrics_summary']
    lines += ['', '## Training/tokenization summary', '', json.dumps(m, indent=2)]
    out_note.parent.mkdir(parents=True, exist_ok=True); out_note.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'out':str(out_json), 'note':str(out_note), 'arm':arm, 'elapsed_sec':payload['elapsed_sec']}, indent=2))

if __name__ == '__main__':
    main()
