# Data Construction and Exact Input Identity

The released recipe uses a 10,000,000-word pool with 64,740 rows. Its full
composition, teacher settings and pool SHA-256 are in the
[model data manifest](../models/frontier/DATA_MANIFEST.json).
Unique pool size and cumulative input presentations are separate quantities.

## Construction Chain

1. [Clean paired-text materialization](../experiments/archive/compact_experience/scripts/clean_materialize_qwen_pairs.py)
   constructs the admitted source/rewrite pairs and packed rows. Keep the
   original selection order and complete-pair rules.
2. [Compact-view overlay](../experiments/archive/frontier_consolidation/scripts/materialize_density_on_cleanqwen_base_rowholdout.py)
   integrates the compact paired block and preserves the specified remaining
   base rows. The associated manifests record selection, displacement, packing
   and pass order; equal word counts alone do not reproduce the stream.
3. [Reference-tail construction](../experiments/archive/functional_learning/scripts/revision_047b_reference_tail_builder.py)
   selects the continuation suffix with its original skip and word-clock policy.
4. [Unchanged-pair annotation](../experiments/archive/functional_learning/scripts/annotate_unchanged_qwen_tail.py)
   adds source/view offsets without changing the reference text.

The final annotation record is
[tail_build_summary.json](../experiments/archive/functional_learning/data/unchanged_pair_segments_tail/tail_build_summary.json).
The complete tail contains 90,609 rows and 13,994,705 words, with SHA-256
`da0225a0a120bb8aa26a0331c56db62f1875133f777452b4b6a09421d776d5ce`.
The actual 80-update prefix contains 20,475 rows and 3,162,742 words, including
3,831 packed paired rows and 11,778 pair segments.

The exact [annotated continuation stream](../experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl)
is retained locally with the original byte-level hash above. This is the realized
input, not a regeneration from the construction recipe.

## Realized Materials

The local archive also contains the selected pairs, packed-row annotations,
repetition and restatement streams, split-window inputs, tokenizer corpora and
item-level measurements. Their original/public hashes are recorded in the
[material manifest](../evidence/materials_manifest.json); the
[coverage map](../evidence/material_coverage.json) connects referenced assets to
their consuming programs.

`distribution: local_only` means that a file is present for local research and
review but is excluded from Git admission and distribution archives while its
source terms and attribution are resolved. It does not mean that the original
was lost or replaced with a placeholder. Metadata identifiers may be relocated;
scientific text, numeric measurements and realized ordering remain intact.

## Selection, Prompts, and Inputs

Original constructors, scientific generation templates, filters, tokenizer
programs, configuration records and construction results are in the
[material manifest](../evidence/materials_manifest.json). Discover them with:

```bash
python3 tools/inspect_materials.py list --kind data_builder
python3 tools/inspect_materials.py list --contains materialization
python3 tools/inspect_materials.py list --contains tokenizer
```

The exact generated output is not determined solely by a model name and
temperature. Preserve admitted pair identities, realized order, output hashes
and tokenizer files; regeneration with a different backend is a new realization.

Mixed natural-text corpora and official evaluation inputs are not covered by the
research code's license. Their source licenses govern redistribution. Local
preservation and public redistribution are distinct: the local archive contains
realized inputs, while source distributions exclude files still marked
`local_only`. The model cards identify upstream sources. New downloads may use
the ignored `data/local/` directory; the archived historical inputs retain the
scientific paths expected by their original programs.

Historical raw examples, overwritten intermediates and withdrawn diagnostic
implementations are not interchangeable with required scientific inputs.
Corrected programs and recorded conclusions are retained;
an absent historical file is not replaced by a newly invented approximation.
