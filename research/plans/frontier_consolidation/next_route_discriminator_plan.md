# route reconstruction — next-route discriminator plan

## Scientific position after route reconstruction

Both post-earlier analysis endpoint routes are complete and below the SOTA target. The current best legal endpoint is scale1.75 adapter128 at Overall **41.5707**, still short of 41.8; U256 at 100M is **41.3292** and U256 at 80M has cheap7 **42.9871**, so U256 does not hide a useful intermediate endpoint.

The earlier analysis private-channel result changes the interpretation of paired views: it proves that a separate pathway can carry information without moving the spatial repair route status language-model function, but it does **not** prove paired-view acquisition. Frozen spatial repair route status already resolves true source/rewrite pairs almost perfectly. The next signal cannot be pair identity or pooled paraphrase alignment.

The next scientific object is therefore:

> Find a legal corpus structure that reduces uncertainty of actual masked tokens beyond what frozen spatial repair route status already represents, then carry that structure through a private or slow pathway that can learn without repeating the mature shared-backbone overwrite seen in scale1.75 and U256.

independent_review route reconstruction and the CPU inventory agree that two real legal candidate structures exist:

1. **Directed edit-state structure** in legal source→compact-rewrite pairs.
2. **Intra-row adjacent-discourse state** in natural legal rows (CHILDES, OpenSubtitles, Gutenberg, SimpleWiki, BNC Spoken, Switchboard), excluding synthetic paired rows.

The inventory file `data/candidate_signal_inventory/candidate_signal_inventory.md` quantifies both:

- 12,155 compact-pair records; 7,027 usable changed-span pairs with enough target change and at least one equal anchor; mean multiset overlap 0.479; mean changed-target fraction 0.265.
- Natural rows excluding synthetic rows: 49,498 rows, 7,919,680 words, 756,689 segments, 707,191 adjacent pairs, 657,864 adjacent triples.

This is enough substrate for both zero-training discriminators. It is not enough to choose either route before measuring token-prediction value.

## Corrections to the initial route reconstruction route note

1. **Do not select discourse-state yet.** It is plausible and abundant, but the incremental token-prediction signal is unmeasured.
2. **Do not select pair alignment.** Pair retrieval is saturated; the viable pair-derived target is directed changed-span token prediction conditioned on aligned source anchors, not source/rewrite matching.
3. **Do not infer damage from raw displacement share.** Embeddings have the largest squared-displacement share partly because they are large. Mid/upper attention shows stronger normalized displacement. The next analysis must use normalized within-group intensity and compare against normal spatial repair route status late movement.
4. **Do not protect noisy score wiggles.** Sentinels should emphasize stable arithmetic drivers: scale1.75 SuperGLUE loss, U256 Reading loss, U256 Supplement subject-aux inversion damage, and robust BLiMP gains/losses; EWoK family movements remain scientifically important but some group-resampling intervals are wide.
5. **Do not build a detached readout as the route.** lead route synthesis after cohmargin closed simple protected late-path/local-logit/frozen-readout variants. Any future private pathway must be connected to token prediction during training while preserving the mature function at initialization.

## Zero-training discriminators to run next

### A. Directed edit-state token value

Legal object: deterministic token-ID edit scripts for source→compact rewrite pairs.

Question: does the true aligned source reduce masked-token NLL on changed rewrite spans beyond overlap/edit/length/source-matched decoy sources?

Minimum test:

- Build monotonic token alignment over the 7,027 usable pairs from the route reconstruction inventory.
- Mask changed target spans in the rewrite while keeping rewrite context visible.
- Compare frozen spatial repair route status pseudo-likelihood under:
  - true aligned source anchors/neighborhood;
  - decoy source matched by source family, overlap, edit distance, and changed-span length;
  - rewrite-only context.
- Stratify by overlap, changed-target fraction, function-word ratio, edit-span length, source/rewrite direction.

Scientific reading:

- If true aligned source gives a substantial changed-token NLL gain over decoys and rewrite-only, directed edit-state is a real incremental signal.
- If decoys match true sources, the apparent structure is generic topical/lexical regularization.
- If gains come only from exact copying, the mechanism is not the missing broad-transfer principle; it may still guide a narrower edit-conditioned pathway but should not get a long run.

### B. Intra-row discourse token value

Legal object: within-row consecutive segments in natural, non-synthetic legal rows. Cross-row order is not available and should not be used.

Question: do true neighboring segments reduce masked-token NLL in the middle segment beyond matched shuffled or reversed neighbors?

Minimum test:

- Exclude `qwen_pair_packed`, FineWeb compact rows, and any generated/rewrite row source from the discourse signal.
- Segment natural rows with source-specific deterministic rules (speaker marks for CHILDES/Switchboard; punctuation/dash split for OpenSubtitles; sentence split for Gutenberg/SimpleWiki/BNC).
- For triples `(u_{t-1}, u_t, u_{t+1})`, mask content/function/pronoun/connective subsets in `u_t` and compare frozen spatial repair route status pseudo-likelihood under:
  - true neighbors;
  - reversed neighbors;
  - same-source shuffled neighbors matched for length and lexical overlap;
  - middle-only context.
- Report by source and by token subset. A useful direction should show residual token value beyond source/register/length/topic matching, not just same-document similarity.

Scientific reading:

- true > reversed > shuffled suggests directional event transitions;
- true ≈ reversed > shuffled suggests topic/entity state;
- true > reversed ≈ shuffled suggests order-specific local coherence;
- true ≈ shuffled means the discourse route does not contain the needed incremental signal.

### C. Importance/displacement relation

Question: were scale1.75 and U256 moving parameters that spatial repair route status corpus MLM treats as important, or were their displacements unrelated to mature-function importance?

Minimum test:

- Estimate spatial repair route status squared-gradient importance from a legal-corpus sample, with several disjoint batches and source stratification.
- Compare endpoint displacement `Δθ²` for scale1.75 and U256 to importance within tensors, by normalized group intensity and by importance deciles.
- Compare against a within-tensor shuffled-importance null and against spatial repair route status's own 80M→100M movement.

Scientific reading:

- If endpoint-specific displacement is enriched in high-importance directions, train-time importance-weighted damping is scientifically motivated.
- If enrichment is similar to spatial repair route status normal late movement or disappears after within-tensor normalization, protection is not a mechanism; it is mostly an anisotropic step-size change.

## Short real-training design if zero-training discriminators are positive

Only after the token-value tests show a real incremental signal:

- Start from spatial repair route status legal checkpoint.
- Preserve ordinary MLM replay as the main objective and exposure ledger.
- Add a zero-output private pathway connected to token prediction: either edit-state cross-attention for changed spans, or a small utterance-state register for discourse segments.
- Run matched 10–20M continuations, not 100M endpoints.
- Required controls:
  - MLM replay;
  - true structure;
  - matched false structure;
  - protection-only;
  - true structure + protection;
  - matched-update-norm control.
- Evaluate cheap7 plus specific sentinels: Supplement subject-aux inversion, Reading, small SuperGLUE panel, EWoK material/spatial/quantitative, and acquisition-specific held-out changed-span or middle-segment token NLL.

A continuation is worth maturation only if the true structure uniquely improves its held-out token target and improves or preserves the sentinel panel relative to controls. A short-run score gain alone is insufficient because U256 and scale1.75 showed early gains that did not survive.

## Relation to the Prior Route Assessment
The preceding route assessment prioritizes context-sensitive representations over further bridge replay, local-logit adjustment or late protection. The proposed directed-edit/discourse token-value discriminators and importance/displacement comparison provide no-training tests of that direction. A distinct context-sensitive construction or independent replication of a discriminator remains a possible follow-up.
