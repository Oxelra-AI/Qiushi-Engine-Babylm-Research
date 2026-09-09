# accum trainer equivalence and confound corrected WWM target-burden map

This corrected map separates raw row length/overflow from the actual visible seq256 MLM target surface. It supersedes the first accum trainer equivalence and confound target-burden run, which computed overflow after truncation.

## Global 10M pool
- Raw tokens/word: legal16k 1.4669; legal40k 1.3943; raw reduction 4.95%.
- Visible seq256 tokens/word: legal16k 1.4299; legal40k 1.3706; visible target-token reduction 4.14%.
- Raw rows above 256 tokens: legal16k 15143; legal40k 11407; rescued rows 3736.
- Visible WWM groups/word: legal16k 0.9802; legal40k 0.9871.

## Changed compact-view block
- Raw tokens/word: legal16k 1.4362; legal40k 1.2916; raw reduction 10.07%.
- Visible seq256 tokens/word: legal16k 1.4347; legal40k 1.2915; visible target-token reduction 9.98%.
- Raw rows above 256 tokens: legal16k 54; legal40k 2; rescued rows 52.
- Visible WWM groups/word: legal16k 0.9992; legal40k 1.0000.

Detailed files: `wwm_target_burden_map.json`, `source_class_target_burden.csv`, `changed_domain_target_burden_allocated.csv`.
