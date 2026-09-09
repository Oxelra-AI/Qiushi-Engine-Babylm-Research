# recgpt public local reproduction and reading mismatch — Public RecGPT local reproduction and Reading mismatch

## Local public RecGPT reproduction

The background evaluation from earlier analysis completed successfully using the patched public `Serdar404/RecGPT-10M` copy:

- model: `data/recgpt_local/patched_model`
- score JSON: `data/recgpt_public_causal_scores.json`
- log: `notes/recgpt_public_causal_eval.log`
- reports: `training/runs/recgpt_public_causal_eval/`

Local scores versus public model-card / leaderboard values:

| Column | Local | Public | Delta |
|---|---:|---:|---:|
| BLiMP | 73.20 | 73.11 | +0.09 |
| Supplement | 63.24 | 61.73 | +1.51 |
| EWoK | 52.82 | 52.62 | +0.20 |
| Entity | 16.73 | 16.59 | +0.14 |
| COMPS | 55.47 | 55.43 | +0.04 |
| GlobalPIQA mean | 38.71 | 40.68 | -1.97 |
| Reading mean | 0.32 | 6.92 | -6.60 |

The main zero-shot causal columns reproduce the public RecGPT phenotype very closely: BLiMP, Supplement, EWoK, Entity, and COMPS all match within about 0–1.5 points. Entity 16.73 confirms the known weak-Entity causal profile. GlobalPIQA is lower by about 2 points, driven by local parallel 19.42 and nonparallel 58.00. Reading is the only severe mismatch.

## Reading mismatch evidence

Leaderboard/parsed records list RecGPT Reading 6.92, with eye tracking 9.35 and self-paced 4.49. The local patched run gives eye tracking 0.63 and self-paced 0.01.

The Reading script itself is the same local script that produces normal causal Reading scores for many previous causal checkpoints (for example eye 8–12 and self-paced 2–3 for several earlier causal runs), so this is not a generic failure of the causal Reading path.

A compact no-rerun analysis was saved at:

- `data/recgpt_reading_anomaly_diagnostic.json`

Key findings:

1. RecGPT Reading surprisals are not flat or random:
   - pred mean 15.90, std 4.94, range 1.29–48.07.
   - raw correlations with eye-tracking RT variables are sizable: RTfirstfix 0.533, RTfirstpass 0.557, RTgopast 0.491, RTrightbound 0.564.

2. But RecGPT surprisal is strongly explained by lexical covariates:
   - correlation with word length 0.623;
   - correlation with Subtlex frequency -0.681.
   This leaves little added variance after the Reading regression controls length/frequency/context, producing eye 0.63 and self-paced 0.01.

3. Tokenizer target formatting is a strong suspect:
   - Reading evaluator scores `tokenizer(word, add_special_tokens=False)` after a sentence context.
   - RecGPT's BPE distinguishes no-leading-space and leading-space word tokens for every Reading target inspected.
   - Across 1,726 targets, no-space and leading-space token IDs differ for 100% of words.
   - no-space average token length is 1.377; leading-space average token length is 1.151.
   - no-space single-token fraction is 0.692; leading-space single-token fraction is 0.853.

For example:

- `the`: no-space token decodes `the`, leading-space token decodes ` the`.
- `photograph`: no-space splits into `phot` + `ograph`, leading-space is one token ` photograph`.
- `already`: no-space splits into `al` + `ready`, leading-space is one token ` already`.

Thus local Reading may be scoring many continuation words using token IDs appropriate for beginning-of-string words rather than in-context words. That would inflate length/frequency dependence and explain why the raw correlations are high but controlled incremental Reading score is near zero.

## Scientific interpretation

The public RecGPT causal reference is locally real for the main zero-shot columns. Its official-useful phenotype is now verified except for two unresolved points:

1. GlobalPIQA is lower locally by about 2 points. This may reflect scorer/version details or model-card collation differences; it is not large enough to invalidate the RecGPT causal profile.
2. Reading is not currently reproduced. The mismatch is large and likely comes from Reading target-token formatting for RecGPT's whitespace-sensitive BPE, but this must be checked by rerunning Reading with a patched scorer that uses leading-space target tokens when the sentence context ends before a word boundary.

Until Reading is repaired or explained, do not treat the full RecGPT 41.53 Overall as locally reproduced. But the causal/recursive reference is still strong enough to guide architecture and training work on official data because BLiMP/Supplement/EWoK/Entity/COMPS are already locally aligned, and Entity weakness is confirmed.

## Next concrete work

1. Patch only the Reading scorer for a RecGPT verification run: use leading-space target tokenization for causal continuation words when appropriate, or reproduce the author's evaluator behavior if recoverable from `serdardoesml/babylm-eval`.
2. Rerun Reading only for patched public RecGPT; compare to leaderboard eye 9.35/self-paced 4.49.
3. If Reading recovers, compute the full local RecGPT reference coordinate and proceed to official-corpus same-budget RecGPT training/decomposition.
4. If Reading stays near zero, preserve RecGPT as a verified zero-shot NLP causal reference but separate local Overall estimates from leaderboard Overall until evaluator mismatch is resolved.
