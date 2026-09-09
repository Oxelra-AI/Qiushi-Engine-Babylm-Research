# seqsafe96 interpretive foundation interpretive foundation for the pending seqsafe96 source-breadth result

## Purpose

The independent_review verifier (live fineweb core fact filter) required a tokenizer-level exposure check before interpreting
any word-matched FineWeb arm, because equal whitespace words and row lengths do not
guarantee equal non-padding tokens, masked targets, or truncation under baseline16k. This
note records the audit result and how it constrains the interpretation of the pending
`s14_t33_tool1` FineWeb seqsafe96 source-breadth contrast.

## Audit result (trainer-exact)

Script: `experiments/archive/representation_and_objectives/scripts/tokenizer_exposure_audit_trainer_exact.py`
JSON: `experiments/archive/representation_and_objectives/data/tokenizer_exposure_audit/tokenizer_exposure_audit_trainer_exact.json`
Note: `research/notes/representation_and_objectives/tokenizer_exposure_audit_trainer_exact.md`

Method matches the actual trainer: `add_special_tokens=False`, `max_length=256`,
padding to 256, WWM groups from token word-start markers (`Ġ`/`▁`), fixed WWM p=0.15,
baseline16k tokenizer (`independent orientation cross and context interface` hf_model, vocab 16384).

Per 10M-word epoch, treatment (FineWeb replacement) minus control (official lengthmatched):

- candidate tokens seen: **-0.494% relative** (treatment slightly fewer)
- WWM groups seen: **+0.0018% relative** (essentially identical)
- candidate tok/word: treatment 1.4402 vs control 1.4473
- token truncation fraction delta: ~+0.00005 (negligible)

Source-block detail:
- Treatment FineWeb block (1,753,280 words): candidate tok/word 1.4877 (denser).
- Control official lengthmatched block (1,753,280 words): candidate tok/word 1.5284
  (slightly denser than the FineWeb block).
- Identical tail (6,589,920 words) and qwen_pair block (1,656,800 words) are exactly matched.

The control replacement block is actually slightly MORE token-dense than the FineWeb
block, which is why the treatment arm sees a tiny token deficit overall.

## What this constrains

The two seqsafe96 arms are token-matched and WWM-group-matched. Therefore:

1. A **positive** downstream FineWeb effect from s14_t33_tool1 cannot be attributed to a
   larger treatment prediction-token budget — treatment has a small deficit, so any gain
   is conservative with respect to token exposure and reads as a source-breadth effect.
2. A **flat or negative** effect cannot be blamed on a token-exposure shortfall of a
   material magnitude; the token difference is <0.5%.
3. The WWM prediction-unit budget (groups) is essentially identical, so masked-target
   learning pressure is matched.

This closes the token-confound question the verifier flagged. It does not resolve the
remaining interpretive constraints:
- The FineWeb source here is cached INITIAL_MODEL_STUDIES sample-10BT random-quality single-doc text at
  ~17.5% of the corpus, not the stricter live pool or the leader's full-10M pair corpus.
- Interpret the downstream result by magnitude across EWoK+Entity+COMPS+GlobalPIQA and by
  Supplement/Reading/BLiMP preservation, not by sign alone. A small positive result most
  likely means under-scaled source breadth, per the scale reasoning.

## Interpretation After the Source-Breadth Result

1. Collect the task; read
   `experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json`.
2. Run `python -B experiments/archive/representation_and_objectives/scripts/interpret_fineweb_seqsafe96_noaoa.py`
   and read `research/notes/representation_and_objectives/fineweb_seqsafe96_interpretation.md`.
3. Combine with this token-exposure audit: a positive knowledge-cluster movement at matched
   tokens supports scaling the corrected four-arm family (A_natural, B_breadth, B_repeat,
   C_view) with a cleaner live FineWeb source; a flat/negative result redirects away from
   cached source repetition.
