# temporal addressability bridge — Temporal addressability bridge and dissociation

## Scientific Motivation was run

earlier analysis showed that the natural fixed-meaning temporal-change bridge failed: stable ATP ranking facts were learned and secondary stable records were preserved, but changed focal before/after updates did not fit or transfer. representational grounding boundary found that separating the two temporal records or using `Background:`/`Update:` style discourse keys made a changed-only toy version work. The strategist correctly sharpened the interpretation: this should be treated as **addressability**, not as proof that temporal information was absent or that a general third layer had been solved. High cosine does not imply absence, and `Background/Update` might work through exact lexical keys and fixed positions.

This analysis ran the full earlier analysis bridge with section-style addresses as a supplied-address ceiling, then one dissociation of fixed order, exact query-key recurrence, arbitrary tags, and no-tag/swap controls.

## 1. Supplied section-address full bridge

Script: `training/scripts/temporal_section_bridge.py`  
Output: `data/temporal_section_core1/`  
Analysis: `data/temporal_section_core1_analysis/central_analysis.md`

The wrapper imports `temporal_change_bridge_probe.py` and replaces initial/later templates by section addresses:

- context: `Focal background record: Background: ...`; `Focal update record: Update: ...`; plus separate secondary background/update records
- hypothesis: train wording `Based on the background/update, ...`; held wording `According to the background/update record, ...`
- same selected worlds, balanced changed/stable directions, stable base, sparse changed/stable focal updates, held entities, paired AB-vs-BA scoring, one seed, core surfaces only

### Central result

Natural earlier analysis core had `balanced_temporal` sparse changed fit 0.656 and held focal-after 0.200 with secondary-after 1.000. Section-address temporal addressability bridge changed this sharply:

- `balanced_temporal`: train fit 1.000, including `sparse_changed_focal=1.000`
- held changed focal + held stable secondary, train context/train hypothesis: focal_before 1.000, focal_after 1.000, secondary_before 1.000, secondary_after 1.000
- held changed focal + held stable secondary, train context/held hypothesis: focal_before 0.950, focal_after 0.7875, secondary_before 1.000, secondary_after 1.000
- held context remained weaker: train hypothesis focal_before 0.550, focal_after 0.769, secondary_after 0.8875

`oracle_secondary` was also strong on train-context surfaces: held changed/stable train context/held hypothesis gave focal_after 1.000 and secondary_after 1.000.

**Interpretation:** the full mixed stable+changed temporal bridge has a learnable ceiling when the relevant record is made directly addressable. The earlier failure was not a capacity limit for ranking facts or secondary preservation. This is not yet a temporal-meaning principle because exact section keys and familiar query keys remain available.

## 2. Dissociation: fixed position, held query paraphrase, arbitrary tags

Script: `training/scripts/temporal_address_dissociation.py`  
Outputs:

- `data/temporal_address_dissociation_section1/`
- `data/temporal_address_dissociation_tag1/`

Both runs use the non-oracle `balanced_temporal` arm: stable base + sparse changed/stable focal rows, no sparse secondary labels.

### 2a. Section roles with shuffled records and nonidentical held query paraphrases

Mode: `section_rolepara`  
All four focal/secondary background/update records are shuffled deterministically, so fixed order is unavailable. Held query paraphrases use `prior`/`revised` and do **not** repeat the exact words `background`/`update`.

Result:

- train fit 1.000, including changed sparse rows
- train-hypothesis held changed/stable: all four readouts 1.000 despite shuffled record order
- held-hypothesis held changed/stable: focal_before 0.506, focal_after 0.456, secondary_before 1.000, secondary_after 1.000

**Interpretation:** fixed position is not necessary when exact trained section-key query wording is retained; train-hyp queries can retrieve the right record after shuffled order. But `background/update` did not robustly transfer to `prior/revised` on changed focal rows. The secondary stable record remained perfect because it is stable and has strong support; this does not prove focal temporal meaning.

### 2b. Arbitrary per-example tags

Mode: `arbitrary_tag`  
Each context assigns four ordinary-word tags to focal-before, focal-after, secondary-before, and secondary-after records, e.g. `Entry amber: ...`. Tags are sampled per focal/secondary pair and do not have a global before/after meaning. Records are shuffled. The query names a tag.

Result (`data/temporal_address_dissociation_tag1/`):

- train fit 0.9989; `sparse_changed_focal=0.9766`, `sparse_stable_focal=1.000`
- held changed/stable, train query wording: focal_before 0.9875, focal_after 0.99375, secondary_before 1.000, secondary_after 1.000
- held changed/stable, held query wording (`Using the record named TAG...`): focal_before 0.94375, focal_after 0.95625, secondary_before 1.000, secondary_after 1.000
- swapped focal tags in held query while keeping original labels: focal_before 0.05625, focal_after 0.06875, secondary still 1.000

**Interpretation:** arbitrary tags are sufficient for the model to learn a per-context key-value address. Because tags have no global temporal meaning, this is not evidence of temporal semantics. The swapped-tag collapse/inversion shows the model follows the tag address, not merely a participant/order shortcut.

## 3. No-tag ablation

Script: `training/scripts/temporal_tag_notag_ablation.py`  
Canonical fitted rerun: `data/temporal_tag_notag_ablation_seed26700/`

The first seed 26750 did not fit changed sparse rows (`sparse_changed_focal=0.535`) and is not interpretable. The seed-26700 rerun matches the fitted tag model:

- train fit 0.9989; `sparse_changed_focal=0.9766`
- normal held tag query: focal_before 0.94375, focal_after 0.95625, secondary 1.000
- no-tag query (`Using the relevant record...`) with the same tagged context: focal_before 0.450, focal_after 0.550, secondary 1.000
- swapped focal tags: focal_before 0.05625, focal_after 0.06875, secondary 1.000

**Interpretation:** the arbitrary-tag result is genuinely address-dependent. Without an address tag, changed focal before/after is underspecified and collapses; swapping the tag queries the opposite record and inverts the focal answer. Stable secondary remains high because its before and after labels coincide.

## Updated scientific meaning

The strongest result is an **addressability condition** for data-efficient state updating:

> Sparse evidence can install or update a relation state under interference when the relevant record is made addressable through a reusable query-key interface. If the query cannot address the appropriate record, stable priors dominate or focal before/after collapses. The current pretrained bridge can learn key-value record addressing from very little data, including arbitrary per-example tags, but it does not yet show transfer from exact learned addresses to semantically equivalent temporal paraphrases.

This sharpens, but also narrows, the representational grounding boundary interpretation:

- Not established: information absence from high cosine; high cosine was only a clue.
- Not established: `Background/Update` as a general discourse-temporal semantic layer.
- Established at this bridge scale: supplied record addresses rescue full focal updating while preserving stable secondary; record order is not necessary; arbitrary tags work; swapped/no-tag controls show address use.
- Open: temporal meaning and paraphrase robustness. Section `background/update` did not transfer to held `prior/revised` for changed focal rows, even though arbitrary tags transferred from `According to entry TAG` to `Using the record named TAG` because the tag itself was identical.

## Files

- Section full bridge wrapper: `training/scripts/temporal_section_bridge.py`
- Section full bridge outputs: `data/temporal_section_core1/`
- Dissociation script: `training/scripts/temporal_address_dissociation.py`
- Section-role dissociation: `data/temporal_address_dissociation_section1/`
- Arbitrary-tag dissociation: `data/temporal_address_dissociation_tag1/`
- No-tag ablation script: `training/scripts/temporal_tag_notag_ablation.py`
- Fitted no-tag ablation: `data/temporal_tag_notag_ablation_seed26700/`
- Non-fit no-tag run kept only as a failed seed: `data/temporal_tag_notag_ablation1/`

## Next scientific work

The next step should not launch Strict-Small 100M training. The bridge result is still a small supplied-address mechanism. The strongest continuation is to turn the addressability finding into a more general principle by separating **key-value address learning** from **temporal/discourse meaning**. A minimal next experiment should train with multiple semantically equivalent address families (e.g. background/update, prior/revised, original/current, first/latest) and evaluate cross-family transfer under shuffled record order and changed focal/stable secondary worlds. If cross-family transfer remains weak, the principle should be stated as address-interface reuse rather than semantic temporal grounding. If it becomes strong, then the route can move toward a reusable time-indexed state principle under paraphrase-robust address families.

## 4. Two-hop role→tag indirect address probe (attempted, informative failure)

Script: `training/scripts/temporal_indirect_address_probe.py`  
Runs:

- multi-family: `data/temporal_indirect_address1/`
- single-family fit check: `data/temporal_indirect_address_singlefam_fitcheck/`
- component fits: `data/indirect_component_direct1/`, `data/indirect_component_role1/`

### Construction

Each context declares role→tag bindings and stores relations under tags, with all eight sentences shuffled:

```
Focal background snapshot is entry umber. Entry timber: Fiona was ranked below Isaac ...
Separate update snapshot is entry amber. Focal update snapshot is entry elm.
Entry amber: ... Entry elm: ... Entry umber: ... Separate background snapshot is entry timber.
[SEP] According to the background snapshot, Nina was ranked higher than Wendy.
```

Role queries must compose `role word → declared tag → stored record`. Direct-tag queries name the tag. Role-swap evaluation swaps only declarations while keeping intended temporal labels.

### Result

- multi-family (3 train families, 10,752 rows, one seed): train accuracy 0.502 with every train kind at chance; all margins ~1e-5. Not interpretable.
- single family (1,792 rows): train accuracy exactly 0.500, again all kinds at chance.
- length/format audit (`scripts/inspect_indirect_lengths.py`): token lengths 122–151, so no truncation at `max_len=200`; hypothesis and `[SEP]` always present.
- duplicate/conflict audit (`scripts/audit_indirect_probe.py`): 1,792 rows, 1,792 unique texts, zero duplicate texts, zero conflicting labels, balanced labels 896/896.

### Component localization (the informative part)

Training only direct-tag queries in the two-hop context (`data/indirect_component_direct1/`):

- overall train 0.930; `base_direct` 1.000, `sparse_stable_direct` 1.000, but `sparse_changed_direct` **0.508**
- held changed/stable readout: focal_before 1.000, focal_after 0.000 with margin −6.97, secondary_before/after 1.000

Training only role queries (`data/indirect_component_role1/`):

- train 0.500 with all kinds at chance; margins ~1e-4

### What this establishes

The earlier arbitrary-tag success is **not robust to adding role→tag declarations**. In the tagged-entry-only construction, changed focal rows fit at 0.977 and held tag queries reached 0.956. Once each tag also appears in a declaration sentence, the same direct-tag interface still handles stable records perfectly but fails exactly on the contradictory changed focal pair, where focal_after inverts to the initial-state answer with a large negative margin. Role queries do not fit at all in this construction.

So the working mechanism is narrower than "learned key-value addressing":

> The bridge can select between two conflicting records when the query key occurs in exactly one place in the context and identifies that record uniquely. When the same key also appears in another sentence with a different function, the selection collapses on the contradictory pair and the model falls back to the stable/initial relation, while unaffected stable records remain correct.

This is a **key-uniqueness / interference condition** on addressability, not evidence about temporal semantics. It also means the failure of `prior/revised` paraphrase transfer in section mode and the failure of two-hop role queries are consistent with one interference limitation rather than two separate semantic gaps.

### What remains open

- whether role→tag composition can be learned at all with a longer curriculum, more families, or larger scale
- whether the interference is caused by repeated tag tokens specifically, or by any additional sentence that mentions a temporal role word
- whether the stable-record immunity reflects label coincidence only

A useful next construction would keep the two-hop composition but remove tag-token repetition, for example by declaring roles positionally in a separate short preamble with distinct tag surface forms, or by using an explicit two-stage query where the model is first trained to output the tag. Any such run should first be checked for changed-row fit before interpretation.
