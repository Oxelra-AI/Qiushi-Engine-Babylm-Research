# official eval progress snapshot corrected-tokenizer official-evaluation progress snapshot

This snapshot is a liveness and controller-stage record only. It must not be used as a corrected-tokenizer endpoint score because official EWoK, official min-context-zero AoA, and pristine collation are not complete unless the controller summary and collate roots say so.

Created UTC: `2026-08-30T01:24:52Z`

## Seed 43022 (s51_t41_tool1)
- Stage inference: `full_wrapper_superglue_1_of_7_tasks_recorded`
- Full-wrapper root: exists=True files=32 bytes=294123551
- Official EWoK root: files=0; AoA root: files=0; collate root: files=0
- Wrapper-recorded columns so far: BLiMP: 66.33; Supplement: 59.28; Entity: 26.2; COMPS: 51.81; GlobalPIQA_parallel: 26.21; GlobalPIQA_nonparallel: 46.0; Reading: scores={'Reading_eye': 10.57, 'Reading_self_paced': 4.62, 'Reading': 7.595000000000001}; SuperGLUE: 1/7 tasks
- SuperGLUE `boolq` log: age=1704.811s size=1003152 progress={'percent': 100, 'current': 26, 'total': 26, 'progress_bracket': '00:03<00:00,  6.98it/s'} return_lines=['[returncode=0 elapsed_sec=656.33]']
- SuperGLUE `multirc` log: age=0.123s size=2109897 progress={'percent': 60, 'current': 10189, 'total': 17020, 'progress_bracket': '04:33<21:25,  5.31it/s, accuracy: 0.8750, f1: 0.8571, mcc: 0.7460'} return_lines=[]

## Seed 43122 (s51_t42_tool1)
- Stage inference: `full_wrapper_superglue_1_of_7_tasks_recorded`
- Full-wrapper root: exists=True files=32 bytes=294455897
- Official EWoK root: files=0; AoA root: files=0; collate root: files=0
- Wrapper-recorded columns so far: BLiMP: 66.47; Supplement: 55.77; Entity: 28.59; COMPS: 52.25; GlobalPIQA_parallel: 26.21; GlobalPIQA_nonparallel: 51.0; Reading: scores={'Reading_eye': 11.51, 'Reading_self_paced': 5.47, 'Reading': 8.49}; SuperGLUE: 1/7 tasks
- SuperGLUE `boolq` log: age=2057.579s size=967122 progress={'percent': 100, 'current': 26, 'total': 26, 'progress_bracket': '00:03<00:00,  6.81it/s'} return_lines=['[returncode=0 elapsed_sec=530.29]']
- SuperGLUE `multirc` log: age=0.105s size=2514734 progress={'percent': 70, 'current': 11996, 'total': 17020, 'progress_bracket': '00:13<11:32,  7.26it/s, accuracy: 0.8125, f1: 0.6667, mcc: 0.5449'} return_lines=[]

## Interpretation
Both evaluations had recent SuperGLUE log updates in this snapshot and the downstream official EWoK/AoA/collate roots were still empty, consistent with the controller still inside the full-wrapper SuperGLUE phase. Seven-column wrapper scores are useful only for knowing that upstream columns have run; they are not official corrected endpoints and should not drive route changes before complete evaluation summaries are available.
