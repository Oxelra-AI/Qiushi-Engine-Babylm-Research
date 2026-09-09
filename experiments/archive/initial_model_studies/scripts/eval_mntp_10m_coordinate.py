#!/usr/bin/env python3
from __future__ import annotations
import json, os, pathlib, re, subprocess, sys, time

ROOT = pathlib.Path('experiments/archive/initial_model_studies')
STRICT = ROOT / 'repos/babylm-eval/strict'
FULL = STRICT / 'evaluation_data/full_eval'
EWOK_OUT = FULL / 'ewok_filtered_word_tokenize'
EWOK_SUMMARY = ROOT / 'data/aligned_shuffled_10m_ewok_word_tokenize_filter_summary.json'
RUN = ROOT / 'training/runs/babylm_mntp_hybrid_s1_10M'
MODEL = (RUN / 'hf_model/chck_10M').resolve()
OUTDIR = (RUN / 'eval_results_coordinate_direct').resolve()
OUT_JSON = ROOT / 'data/mntp_hybrid_s1_10m_coordinate.json'
OUT_NOTE = (ROOT.parents[2] / 'research/notes/initial_model_studies/mntp_hybrid_s1_10m_coordinate.md')
LOG = (ROOT.parents[2] / 'research/notes/initial_model_studies/mntp_eval.log')
BACKEND = 'mlm'
REV = 'direct_chck_10M'

TASKS = [
    ('blimp', 'blimp', FULL / 'blimp_filtered', 'blimp_filtered'),
    ('supplement', 'blimp', FULL / 'supplement_filtered', 'supplement_filtered'),
    ('entity_tracking', 'entity_tracking', FULL / 'entity_tracking', 'entity_tracking'),
    ('comps', 'comps', FULL / 'comps', 'comps'),
    ('global_piqa_parallel', 'global_piqa_parallel', FULL / 'global_piqa_parallel', 'global_piqa_parallel'),
    ('global_piqa_nonparallel', 'global_piqa_nonparallel', FULL / 'global_piqa_nonparallel', 'global_piqa_nonparallel'),
]

S1_BASELINE = {
    'blimp': 53.37,
    'supplement': 52.34,
    'entity_tracking': 17.73,
    'comps': 50.34,
    'global_piqa_parallel': 17.42,  # derived from S1 mean/nonparallel where available; not used as central alone
    'global_piqa_nonparallel': 53.00,
    'global_piqa_mean': 35.21,
    'reading_mean': 8.35,
    'ewok_full': 50.60,
}

PUBLIC_LEADER = {
    'blimp': 67.20,
    'supplement': 56.04,
    'entity_tracking': 28.45,
    'comps': 53.57,
    'global_piqa_mean': 39.665,
    'reading_mean': 5.425,
    'ewok_full': 56.07,
}

def setup_env():
    env = os.environ.copy()
    env['NLTK_DATA'] = str((ROOT / 'data/nltk_data').resolve())
    hf = ROOT / 'training/hf_home'
    env['HF_HOME'] = str(hf.resolve())
    env['HF_HUB_CACHE'] = str((hf / 'hub').resolve())
    env['TRANSFORMERS_CACHE'] = str((hf / 'transformers').resolve())
    env['HF_MODULES_CACHE'] = str((ROOT / 'training/hf_modules_cache').resolve())
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    for k in ['HF_HOME', 'HF_HUB_CACHE', 'TRANSFORMERS_CACHE', 'HF_MODULES_CACHE']:
        pathlib.Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env

def run(cmd, env, cwd=STRICT, logf=None):
    line = '$ ' + ' '.join(map(str, cmd))
    print(line, flush=True)
    if logf:
        logf.write('\n' + line + '\n'); logf.flush()
    p = subprocess.run([str(x) for x in cmd], cwd=str(cwd), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout[-2500:], flush=True)
    if logf:
        logf.write(p.stdout + f'\n[returncode={p.returncode}]\n'); logf.flush()
    if p.returncode != 0:
        raise RuntimeError(f'command failed {p.returncode}: {line}\n{p.stdout[-10000:]}')

def read_avg(report: pathlib.Path) -> float:
    txt = report.read_text(encoding='utf-8', errors='replace')
    m = re.search(r'AVERAGE ACCURACY\s*\n([0-9.\-]+)', txt)
    if m:
        return float(m.group(1))
    vals = re.findall(r'(?:acc_norm|accuracy|acc)\s*[:=]\s*([0-9.]+)', txt, flags=re.I)
    if vals:
        v = float(vals[-1]); return v * (100 if v <= 1 else 1)
    raise RuntimeError(f'could not parse avg from {report}\n{txt[:1200]}')

def read_reading(report: pathlib.Path) -> dict[str, float]:
    txt = report.read_text(encoding='utf-8', errors='replace')
    out = {}
    for label, key in [('EYE TRACKING SCORE', 'reading_eye_tracking'), ('SELF-PACED READING SCORE', 'reading_self_paced')]:
        m = re.search(re.escape(label) + r':\s*([0-9.\-]+)', txt)
        if not m:
            raise RuntimeError(f'could not parse {label} from {report}\n{txt[:1200]}')
        out[key] = float(m.group(1))
    return out

def report_candidate(outdir: pathlib.Path, task: str, dataset_name: str) -> pathlib.Path:
    report = outdir / 'chck_10M' / REV / 'zero_shot' / BACKEND / task / dataset_name / 'best_temperature_report.txt'
    if report.exists():
        return report
    candidates = list(outdir.rglob(f'zero_shot/{BACKEND}/{task}/{dataset_name}/best_temperature_report.txt'))
    if len(candidates) != 1:
        raise RuntimeError(f'report not found for {task}/{dataset_name}; expected {report}; candidates={candidates[:10]}')
    return candidates[0]

def main():
    t0 = time.time()
    if not MODEL.exists():
        raise RuntimeError(f'Missing direct checkpoint {MODEL}')
    env = setup_env(); OUTDIR.mkdir(parents=True, exist_ok=True); LOG.parent.mkdir(parents=True, exist_ok=True)
    scores = {}; reports = {}; path_status = {}
    with LOG.open('w', encoding='utf-8') as logf:
        for col, task, data_path, dataset_name in TASKS:
            path_status[col] = {'path': str(data_path), 'exists': data_path.exists(), 'num_files': sum(1 for _ in data_path.rglob('*')) if data_path.exists() else 0}
            if not data_path.exists() or not any(data_path.rglob('*')):
                raise RuntimeError(f'missing eval data for {col}: {data_path}')
            run([sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
                 '--model_path_or_name', str(MODEL), '--backend', BACKEND, '--task', task,
                 '--data_path', str(data_path.resolve()), '--save_predictions', '--revision_name', REV,
                 '--batch_size', '64', '--output_dir', str(OUTDIR)], env, logf=logf)
            report = report_candidate(OUTDIR, task, dataset_name)
            reports[col] = str(report); scores[col] = read_avg(report)

        reading_csv = FULL / 'reading/reading_data.csv'
        path_status['reading'] = {'path': str(reading_csv), 'exists': reading_csv.exists(), 'bytes': reading_csv.stat().st_size if reading_csv.exists() else 0}
        run([sys.executable, '-m', 'evaluation_pipeline.reading.run', '--model_path_or_name', str(MODEL), '--backend', BACKEND,
             '--data_path', str(reading_csv.resolve()), '--revision_name', REV, '--output_dir', str(OUTDIR)], env, logf=logf)
        candidates = sorted(OUTDIR.rglob(f'zero_shot/{BACKEND}/reading/report.txt'))
        if not candidates:
            raise RuntimeError('reading report not found')
        reading_report = candidates[-1]
        reports['reading'] = str(reading_report); scores.update(read_reading(reading_report))

        if not EWOK_SUMMARY.exists():
            raise RuntimeError(f'Missing EWoK filter summary {EWOK_SUMMARY}')
        if not EWOK_OUT.exists() or not any(EWOK_OUT.glob('*.jsonl')):
            raise RuntimeError(f'Missing filtered EWoK data {EWOK_OUT}')
        ewok_eval_out = OUTDIR / 'full_ewok_word_tokenize'
        ewok_eval_out.mkdir(parents=True, exist_ok=True)
        run([sys.executable, '-m', 'evaluation_pipeline.sentence_zero_shot.run',
             '--model_path_or_name', str(MODEL), '--backend', BACKEND, '--task', 'ewok',
             '--data_path', str(EWOK_OUT.resolve()), '--save_predictions', '--batch_size', '64',
             '--output_dir', str(ewok_eval_out.resolve())], env, logf=logf)
        hits = sorted(ewok_eval_out.rglob('best_temperature_report.txt'))
        if len(hits) != 1:
            raise RuntimeError(f'Expected one EWoK report under {ewok_eval_out}, got {hits[:10]}')
        reports['ewok_full'] = str(hits[0]); scores['ewok_full'] = read_avg(hits[0])

    scores['global_piqa_mean'] = (scores['global_piqa_parallel'] + scores['global_piqa_nonparallel']) / 2.0
    scores['reading_mean'] = (scores['reading_eye_tracking'] + scores['reading_self_paced']) / 2.0
    metrics = json.loads((RUN / 'scientific_metrics.json').read_text())
    s1_metrics = json.loads((ROOT / 'training/runs/babylm_leadershape_s1_10M_aligned_micro128/scientific_metrics.json').read_text())
    deltas_vs_s1 = {k: (scores[k] - S1_BASELINE[k]) for k in S1_BASELINE if k in scores}
    gaps_vs_leader = {k: (scores[k] - PUBLIC_LEADER[k]) for k in PUBLIC_LEADER if k in scores}
    payload = {
        'status': 'MNTP_HYBRID_S1_10M_COORDINATE',
        'run': str(RUN), 'model_path_direct': str(MODEL), 'backend': BACKEND,
        'scores': scores, 'deltas_vs_s1_10m': deltas_vs_s1, 'deltas_vs_public_leader': gaps_vs_leader,
        'training_core': {k: metrics.get(k) for k in ['variant','word_exposure','actual_training_steps','optimizer_steps','effective_batch_size','micro_batch_size','lr_schedule_total_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word','mntp_every','objective_assignment','objective_step_counts','objective_word_counts','objective_predicted_token_counts','parameter_count','hidden_size','n_layer','n_head','intermediate_size']},
        's1_training_core': {k: s1_metrics.get(k) for k in ['word_exposure','actual_training_steps','optimizer_steps','effective_batch_size','micro_batch_size','lr_schedule_total_steps','loss_first','loss_last','masked_tokens_total','masked_tokens_per_whitespace_word']},
        'reports': reports, 'path_status': path_status, 'elapsed_sec': time.time() - t0,
        'interpretation_hint': '10M matched comparison: same data/order/updates/schedule/init as S1 except deterministic every-16th-batch MNTP objective with causal encoder mask.'
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# research — deterministic MNTP hybrid S1 10M coordinate', '', f'Evidence JSON: `{OUT_JSON}`', f'Run: `{RUN}`', '',
             'Matched against research S1 10M: same data/order/exposure/updates/schedule/init; only every 16th batch uses MNTP.', '',
             '| column/task | MNTP 10M | S1 10M | delta | public leader | MNTP-leader |', '|---|---:|---:|---:|---:|---:|']
    rows = [('blimp','BLiMP'),('supplement','Supplement'),('ewok_full','EWoK full'),('entity_tracking','Entity'),('comps','COMPS'),('global_piqa_mean','GlobalPIQA mean'),('reading_mean','Reading mean')]
    for key, label in rows:
        mn = scores.get(key); s1 = S1_BASELINE.get(key); ld = PUBLIC_LEADER.get(key)
        lines.append(f'| {label} | {mn:.2f} | {s1:.2f} | {mn-s1:+.2f} | {ld:.2f} | {mn-ld:+.2f} |')
    lines += ['', f'Objective counts: `{metrics.get("objective_step_counts")}`, predicted tokens: `{metrics.get("objective_predicted_token_counts")}`.', f'Loss: {metrics.get("loss_first"):.4f} → {metrics.get("loss_last"):.4f} (S1 10M loss_last {s1_metrics.get("loss_last"):.4f}).']
    OUT_NOTE.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'out': str(OUT_JSON), 'note': str(OUT_NOTE), 'scores': scores, 'deltas_vs_s1_10m': deltas_vs_s1}, indent=2))

if __name__ == '__main__':
    main()
