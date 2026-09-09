# Compact-View Density Review

## Review outcome

The compact-view density line should remain the group focus unless the pending full evaluation or the seed43122 endpoint result removes the hard-component movement. The older cached-FineWeb branch may still be useful orthogonal evidence, but it should not reclaim attention merely because it was started earlier.

## Component reading

`compact_view_core` vs same-seed `compact_repeat_core`: seven-column sum +14.890 (+2.127 mean), hard cluster +7.220. Column deltas: BLiMP -0.16, Supplement +7.60, EWoK +2.91, Entity +2.19, COMPS +1.09, GlobalPIQA +1.030, Reading +0.23.

`compact_view_core` vs the inherited clean-Qwen natural coordinate: seven-column sum +5.155 (+0.736 mean), hard cluster +1.815. This matters because the same-source repeat arm is a deliberately harsh comparator; evidence of progress must also survive against the strong natural allocation.

`compact_view_reinvest` is the strongest fast surface now available: seven-column mean 44.2886, with SuperGLUE+AoA needed for 41.8 equal to 66.180. Against clean-Qwen it is +1.176 in seven-column mean and +4.400 in the hard cluster. It is a promising extension, not a substitute for the unresolved core full evaluation and seed evidence.

Fast-seven scenarios: compact_view_core with clean-Qwen SuperGLUE and AoA 0 gives Overall 41.9171; with SuperGLUE 68 and AoA 0 gives 41.6606. Reinvest with SuperGLUE 68 and AoA 0 already gives 42.0022; with SuperGLUE 68 and AoA -2 gives 41.7800. These are arithmetic projections from fast columns, not results.

## Mechanism reading

The best current hypothesis is information-density and consolidation, not generic paraphrase. Compact views keep anchors and numerical facts while deleting redundant wording and some secondary content: core rewrite/source ratio 0.620, content recall 0.662, entity recall 0.996, number recall 1.000. The model receives the original source proposition and a shorter generated view in close proximity, which can stabilize entity/state and predicate abstractions under a fixed word budget. Reinvestment then uses the saved words to add more source-view packets, increasing source count by 1.204x with similar compression and retention.

Trainer-exact exposure does not explain the main contrast: compact_view_core has candidate tokens +0.001416 relative to repeat and WWM groups -0.000027; reinvest vs core has candidate tokens +0.000003 and WWM groups +0.000199. These differences are too small and not aligned enough to replace the data-mechanism explanation, though they should remain in the interpretation.

## Vulnerabilities that matter

The automatic semantic numbers are not proof of perfect meaning preservation. Content recall around 0.66 means many generated views omit propositions, and source samples include extraction artifacts and occasional mixed-topic continuations. The current result may be robust because compression removes noisy surface burden, or it may be partly selected by fast tasks. The next evidence must therefore read the component vector, not just a scalar Overall.

A low full Overall caused mainly by SuperGLUE or AoA would not by itself refute the data mechanism; it would indicate that the data mechanism needs combination or repair. Loss of the EWoK/Entity/COMPS/GlobalPIQA pattern in full evaluation or seed43122 would be much more damaging and should stop unchanged extension.

## Next scientific work

If the full core evaluation preserves the hard-component surface, failure to meet a scalar screening threshold is not scientific evidence against seed43122 replication; the full component vector should determine whether seed43122 view and then matched repeat43122 are warranted. If seed43122 view remains strong, the next expensive same-seed comparison is `compact_repeat_core` seed43122. If core full is healthy, evaluate `compact_view_reinvest` fully because its existing trained endpoint has the stronger fast surface and requires much less SuperGLUE+AoA to cross 41.8.

JSON: `experiments/archive/representation_and_objectives/data/compact_view_density_review/compact_view_density_review.json`
