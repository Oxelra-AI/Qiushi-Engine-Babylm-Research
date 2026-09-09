# compact core joint visibility audit execute state: compact-view density visibility, novelty, and remaining confounds

## Findings and Corrections

While the compact reinvest full eval summary GPU tasks continue asynchronously, converted the compact-view-density route from a fast-score story into a more constrained trainer-interface object. The active route remains `compact_view_reinvest` as the SOTA-facing endpoint and `compact_view_core` as the mechanism anchor, but the mechanism is now stated more precisely:

- compact source-view pairs are short enough to be jointly visible;
- actual row packing preserves almost all source+rewrite units under the baseline16k seq256 trainer interface;
- reinvestment converts the compact-core neutral top-up budget into additional source-view pairs that are mostly not exact or high lexical near-duplicates of the core source keys;
- the current same-seed repeat controls remain scientifically useful but do not isolate consolidation from denoising or lexical diversity because generated compact views tokenize into more subword tokens per matched word row than literal repetition.

No new GPU work was launched and no new BabyLM score was produced.

## Evidence produced

### Reinvest pair and changed-row visibility

Files:

- `experiments/archive/representation_and_objectives/data/compact_reinvest_pair_visibility_audit/compact_reinvest_pair_visibility_audit.json`
- `research/notes/representation_and_objectives/compact_reinvest_pair_visibility_audit.md`
- `experiments/archive/representation_and_objectives/data/compact_reinvest_actual_joint_visibility/compact_reinvest_actual_joint_visibility.json`
- `research/notes/representation_and_objectives/compact_reinvest_actual_joint_visibility.md`

Key values:

- Reinvest selected pairs: 12,155 = 10,094 core + 2,061 added; union check true.
- Source/rewrite/pair words: 261,803 / 161,708 / 423,511; rewrite/source word ratio 0.6177.
- Mean content/entity/number recall: 0.6631 / 0.9960 / 1.0000.
- All 12,155 individual source+rewrite pairs fit within 254 and 256 no-special-token slots; pair token length mean/median/p95/max 50.98 / 47 / 89 / 167.
- Actual packed-row reconstruction from selected pairs: 3,005 of 3,006 changed rows exactly reconstructed; the remaining row is the 9-word top-up row.
- Actual packed seq256 visibility: 12,061/12,155 full source+rewrite visible (0.992267), 12,137/12,155 source visible (0.998519), 91 partial, 3 hidden.
- Loss of full pair visibility is tail-position truncation: non-full-visible positions {3: 46, 4: 29, 2: 17, 1: 1, 5: 1}.

### Core mechanism-anchor visibility

Files:

- `experiments/archive/representation_and_objectives/data/compact_core_joint_visibility_audit/compact_core_joint_visibility_audit.json`
- `research/notes/representation_and_objectives/compact_core_joint_visibility_audit.md`

Key values:

- Compact core changed block: 2,514 pair rows + 435 neutral top-up rows; exact reconstruction of all 2,514 pair rows.
- Actual packed seq256 visibility: 10,012/10,094 full source+rewrite visible (0.991876), 10,078/10,094 source visible (0.998415), 79 partial, 3 hidden.
- This closes the independent_review-identified gap that compact core joint visibility audit had initially audited reinvest but not the core view/repeat mechanism anchor.

### Added-source novelty

Files:

- `experiments/archive/representation_and_objectives/data/compact_reinvest_source_novelty_audit/compact_reinvest_source_novelty_audit.json`
- `research/notes/representation_and_objectives/compact_reinvest_source_novelty_audit.md`

Key values:

- Core/added source-key overlap: 0; exact normalized added-source duplicates in core: 1.
- Added pairs whose document ID also appears in core: 1,779/2,061, so document-level breadth is limited; the added material is mostly new sentence/source keys within already represented documents rather than mostly new documents.
- Best added-to-core lexical similarity is low: trigram Jaccard median/p95/max 0.0204 / 0.0952 / 1.0000; bigram 0.0682 / 0.1379 / 1.0000; content-token 0.0952 / 0.2143 / 1.0000.
- Counts above high near-duplicate thresholds are all 1 (trigram>=0.8, bigram>=0.8, content>=0.8, combined>=0.8), corresponding to the single exact duplicate.

### independent_review verification

File:

- `data/external/independent_review01_verifier1_integration.md`

The verifier agreed that the audits remove word-budget and truncation-artifact explanations for reinvest-vs-core visibility, but emphasized remaining confounds: view rows have more visible subword tokens than repeat rows; content recall around 0.66 does not separate consolidation from denoising; source breadth needs novelty measurement; and the 41.8 projection still depends on unresolved SuperGLUE, AoA, and seed stability. After independent_review, added the core packed-visibility audit and the source-novelty audit.

## Remaining interpretation

The compact-view-density route is stronger after this step because its source+compact-rewrite signal is actually available to the model: it is not hidden by seq256 truncation, and the additional reinvest pairs are mostly not lexical repeats of core source keys. This does not prove downstream competence or the full mechanism.

The important confound is token geometry. Although word totals and row lengths are matched, compact generated views carry more baseline16k subword tokens than literal repeat rows:

- reinvest view-minus-repeat changed-row token delta: mean +8.419, sum +25,308, 2,771/3,006 rows longer;
- core view-minus-repeat changed-row token delta: mean +7.091, sum +20,910, 2,306/2,949 rows longer.

Thus `view > repeat` can reflect anchor-preserving two-view consolidation, cleaner/denser wording, extra lexical-token diversity, or a mixture. The current repeat control remains valuable, but if the route survives full and seed tests, future mechanism work should consider a visible-token-matched repeat or a compact-rewrite-only/source-only control instead of claiming pure consolidation.

## Result-dependent next work

Do not start another expensive run until the active compact reinvest full eval summary tasks report actual results.

1. When `s25_t24_tool1` finishes, read `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/per_target/compact_view_reinvest.json`, `compact_reinvest_full_eval_summary.json`, and the controller log. Interpret the full component vector, especially SuperGLUE and AoA, against 41.8 and against the fast seven-column projection.
2. When `s25_t33_tool1` finishes, read `experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast/compact_view_reinvest_seed43122_fast_summary.json` and compare the fast pattern to seed43022.
3. If reinvest full evaluation crosses or nearly crosses 41.8 and seed43122 keeps the broad pattern, preserve the submit-ready model path and only then choose mechanism controls: matched repeat-reinvest seed43122 first if the main question is seed-paired effect, or visible-token-matched / compact-only controls if the main question is consolidation versus lexical/denoising.
4. If full evaluation fails mainly through SuperGLUE or AoA while hard components survive, repair those components or combine only with validated orthogonal factors; do not discard the density mechanism on a scalar miss.
5. If hard components collapse in full evaluation or seed43122, analyze which family collapsed before extending the same corpus.
