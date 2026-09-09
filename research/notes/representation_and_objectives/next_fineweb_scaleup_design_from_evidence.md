# FineWeb scale-up design from the source-breadth and view comparisons

## Why this note exists

This historical design note separates source-breadth effects from generated-view
utility. The source-breadth result was pending when the plan was written; the
proposed scale-up experiments below must not be read as completed results.

## Current evidence pieces

1. **SimpleWiki same-source generated views**
   - Full selected endpoint `semantic_view_treatment__chck_80M`: Overall 40.1399.
   - Mechanism real over packet-local repetition (Entity +2.10, EWoK +1.27,
     COMPS +0.60), but endpoint far below inherited clean-Qwen 41.3443 and public
     leader 41.8.
   - Reading: same-source view is not enough to solve SOTA.

2. **FineWeb near-view utility**
   - Changed block only 423,520 words (~4.24% of 10M) on inherited clean-Qwen base.
   - `near_view - near_repeat`: Entity full +3.57, Reading +0.775, Supplement +0.80,
     equal7 fullEnt +0.554; EWoK/COMPS slightly negative, GPIQA flat.
   - Reading: generated FineWeb views carry real entity/state value beyond source
     repetition; this is C−B in the factorization.

3. **Planned seqsafe96 source-breadth test**
   - FineWeb source block 1,753,280 words (~17.53% of 10M) versus matched official
     control, no generated views.
   - It tests B−A: raw source-breadth value at about four times the near-view comparison's changed-block scale.

4. **Live FineWeb-Edu availability**
   - Data source: `HuggingFaceFW/fineweb-edu`.
   - 64-doc pilot: 40,170 doc words -> 21,374 basic factual-sentence candidate
     words; stricter offline filter retained 7,612 strict-anchor words and 3,570
     relation-dense words.
   - Manual sample inspection shows heuristic selection is still noisy; scaled live
     extraction needs stricter boilerplate, event/contact, quote-fragment, pronoun,
     and fake-capital filters before Qwen generation.

## Planned interpretation of the source-breadth result

Read `experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval/fineweb_seqsafe96_delta_summary.json`, then run:

`python -B experiments/archive/representation_and_objectives/scripts/interpret_fineweb_seqsafe96_noaoa.py`

This creates:

`research/notes/representation_and_objectives/fineweb_seqsafe96_interpretation.md`

Use the magnitude/signature:

- **Strong knowledge-cluster gain** (EWoK+Entity+COMPS+GPIQA summed delta ≳ +2 and
  protected BLiMP/Supp/Reading sum not strongly negative): source breadth is alive.
  The next H100 job should not be another 17.5% source-only repetition. Build a
  larger live FineWeb design with at least two arms:
  1. larger source-repeat arm (target 3.0–4.0M FineWeb words, exact matched official
     control);
  2. larger source+near-view arm using relation-dense faithful Qwen rewrites;
     compare to source-repeat using the earlier Entity-positive view result as motivation.
  If no-AoA treatment equal7 approaches/exceeds inherited clean-Qwen and deficits
  are mostly EWoK/Entity/COMPS/GPIQA, select endpoint for full eval.

- **Small positive knowledge signal**: do not stop source breadth; treat 17.5% as
  under-scaled and/or quality-limited. Build a cleaner/larger live FineWeb source
  asset first (CPU/network), run a small Qwen faithfulness slice on relation-dense
  sentences, then train a source+view arm rather than source-only.

- **Flat/negative source-repeat result**: do not spend more H100 time on cached
  source repetition alone. Retain the near-view Entity evidence; consider
  relation-preserving generated views, stronger tokenization/masking interactions,
  or architecture/optimizer mechanisms. If compact/reinvestment results are
  positive, let those determine the next combined arm.

## Minimal-cost scale-up sequence if source breadth remains promising

1. Data extraction: stream FineWeb-Edu into a local data cache;
   collect a relation-dense sentence pool with exact doc ids, source offsets,
   row quality flags, word counts, and dedup hashes. Start with 0.5M accepted
   source words to audit yield before scaling to 3–4M.
2. Faithful rewrite pilot: Qwen-generate 512–1024 relation-preserving near/compact
   views; measure exact-copy, near-copy, entity/number recall, new entity/number
   rate, and manual samples.
3. Materialization dry-run: exact 10M pools preserving clean-Qwen or a deliberate
   replacement of official tail; matched row-length sequence; seq256 visibility
   under baseline16k and available40k; deep string/source reconstruction.
4. Only then launch H100 training. The run must decide between source breadth,
   source+view utility, and protected-column damage; avoid a one-arm blend.
