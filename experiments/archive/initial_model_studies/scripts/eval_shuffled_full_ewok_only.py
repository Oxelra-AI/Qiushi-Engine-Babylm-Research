#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
EWOK_OUT = STRICT / 'evaluation_data/full_eval/ewok_filtered_word_tokenize'
SUMMARY = ROOT / 'data/aligned_shuffled_10m_ewok_word_tokenize_filter_summary.json'
ALIGNED_JSON = ROOT / 'data/aligned_10m_full_ewok_word_tokenize_score.json'
SHUFFLED_JSON = ROOT / 'data/shuffled_10m_full_ewok_word_tokenize_score.json'
COMPARISON_JSON = ROOT / 'data/aligned_vs_shuffled_10m_full_ewok_comparison.json'
NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/aligned_vs_shuffled_10m_full_ewok_comparison.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/shuffled_full_ewok_only.log')
RUN = ROOT / 'training/runs/babylm_shuffled_s1_10M'
MODEL = (RUN / 'hf_model/chck_9999996w').resolve()
EVAL_OUT = RUN / 'eval_results_full_ewok_word_tokenize_direct'


def parse_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if not m:
        raise RuntimeError(f'Could not parse average from {report}\n{txt[:1200]}')
    return float(m.group(1))


def setup_env():
    env = os.environ.copy()
    nltk_data = (ROOT / 'data/nltk_data').resolve()
    env['NLTK_DATA'] = str(nltk_data)
    hf = ROOT / 'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def main():
    t0 = time.time()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('w', encoding='utf-8') as logf:
        def log(msg: str):
            print(msg, flush=True)
            logf.write(msg + '\n')
            logf.flush()

        if not MODEL.exists():
            raise RuntimeError(f'Missing shuffled checkpoint {MODEL}')
        if not ALIGNED_JSON.exists():
            raise RuntimeError(f'Missing aligned EWoK JSON {ALIGNED_JSON}')
        if not SUMMARY.exists():
            raise RuntimeError(f'Missing EWoK filter summary {SUMMARY}')
        if not EWOK_OUT.exists() or not any(EWOK_OUT.glob('*.jsonl')):
            raise RuntimeError(f'Missing filtered EWoK data {EWOK_OUT}; rerun full script filter stage')

        # Remove partial output left by the research timeout so the report is unique.
        if EVAL_OUT.exists():
            shutil.rmtree(EVAL_OUT)
        EVAL_OUT.mkdir(parents=True, exist_ok=True)

        env = setup_env()
        cmd = [sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
               '--model_path_or_name', str(MODEL), '--backend', 'mlm', '--task', 'ewok',
               '--data_path', str(EWOK_OUT.resolve()), '--save_predictions', '--batch_size', '64',
               '--output_dir', str(EVAL_OUT.resolve())]
        log('$ ' + ' '.join(cmd))
        p = subprocess.run(cmd, cwd=str(STRICT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log(p.stdout[-10000:])
        if p.returncode != 0:
            raise RuntimeError(f'Shuffled EWoK eval failed {p.returncode}\n{p.stdout[-12000:]}')
        hits = sorted(EVAL_OUT.rglob('best_temperature_report.txt'))
        if len(hits) != 1:
            raise RuntimeError(f'Expected one shuffled EWoK report under {EVAL_OUT}, got {hits[:10]}')
        report = hits[0]
        pred = report.parent / 'predictions.json'
        score = parse_avg(report)
        metrics = json.loads((RUN / 'scientific_metrics.json').read_text())
        shuffled_payload = {
            'status': 'SHUFFLED_S1_10M_FULL_EWOK_WORD_TOKENIZE',
            'arm': 'shuffled',
            'model': 'shuffled S1 12x384 DeBERTa-v2 baseline16k WWM 10M',
            'model_path_direct': str(MODEL),
            'backend': 'mlm',
            'ewok_full_score': score,
            'training_core': {k: metrics.get(k) for k in ['word_exposure','actual_training_steps','loss_first','loss_last','example_jsonl_label','example_jsonl_total_words','example_jsonl_total_rows']},
            'filter_summary': str(SUMMARY),
            'report': str(report),
            'predictions': str(pred),
            'eval_output_dir': str(EVAL_OUT),
            'warning': 'Full local EWoK generated from provided parquet with official nltk.word_tokenize resources and official vocab filter.'
        }
        SHUFFLED_JSON.parent.mkdir(parents=True, exist_ok=True)
        SHUFFLED_JSON.write_text(json.dumps(shuffled_payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

        aligned_payload = json.loads(ALIGNED_JSON.read_text())
        delta = aligned_payload['ewok_full_score'] - shuffled_payload['ewok_full_score']
        comparison = {
            'status': 'ALIGNED_MINUS_SHUFFLED_10M_FULL_EWOK_COMPARISON',
            'mechanism': 'correct semantic correspondence within MLM window vs identical source/target text multisets with wrong correspondence',
            'aligned_json': str(ALIGNED_JSON),
            'shuffled_json': str(SHUFFLED_JSON),
            'scores': {'aligned': aligned_payload['ewok_full_score'], 'shuffled': shuffled_payload['ewok_full_score']},
            'aligned_minus_shuffled_ewok': delta,
            'filter_summary': str(SUMMARY),
            'elapsed_sec_step155': time.time() - t0,
            'interpretation_hint': 'Primary EWoK mechanism signal. Combine with available-coordinate comparison where aligned - shuffled Entity = -0.42.'
        }
        COMPARISON_JSON.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        NOTE.write_text('\n'.join([
            '# research/155 — aligned vs shuffled 10M full EWoK comparison',
            '',
            f'Comparison JSON: `{COMPARISON_JSON}`',
            f'Filter summary: `{SUMMARY}`',
            '',
            '| arm | full EWoK |',
            '|---|---:|',
            f'| aligned | {aligned_payload["ewok_full_score"]:.2f} |',
            f'| shuffled | {shuffled_payload["ewok_full_score"]:.2f} |',
            f'| aligned - shuffled | {delta:+.2f} |',
            '',
            'This completes the primary EWoK half of the aligned-vs-shuffled semantic-redundancy test. The available-coordinate comparison is in `data/aligned_vs_shuffled_10m_available_comparison.json`.'
        ]) + '\n', encoding='utf-8')
        log(json.dumps({'status': comparison['status'], 'shuffled_ewok': score, 'aligned_minus_shuffled_ewok': delta, 'comparison': str(COMPARISON_JSON), 'elapsed_sec': comparison['elapsed_sec_step155']}, indent=2))


if __name__ == '__main__':
    main()
