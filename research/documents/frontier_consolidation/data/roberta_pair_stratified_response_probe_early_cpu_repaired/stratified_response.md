# roberta stratified bridge and early local response RoBERTa pair-stratified local response

Created: `2026-09-02T03:05:38Z`

Positive `repeat_minus_compact_advantage` means the compact-trained model predicts the masked event better than the repeat-trained model. Interpret only with selected official-compatible trajectory results.

## chck_10M
- Overall local advantage: `0.015600721407595868` over `5470` events / `7812` pieces, cluster bootstrap p05/p50/p95 `0.009169194830178116/0.015423315899972697/0.0217129910628264`, p>0 `1.0`
- View/category advantages:
  - `compact|function_other`: advantage `-0.07119420208005882`, events `1129`, bootstrap p05/p50/p95 `-0.08515496000886064/-0.07190766713477129/-0.05708457780576841`, p>0 `0.0`
  - `compact|retained_content`: advantage `0.06504002570643233`, events `1122`, bootstrap p05/p50/p95 `0.05202849653308169/0.06482381984010183/0.07772618961609062`, p>0 `1.0`
  - `compact|source_absent_content`: advantage `0.08037200761933001`, events `965`, bootstrap p05/p50/p95 `0.06402122917974858/0.0803612521132448/0.09523341300990755`, p>0 `1.0`
  - `repeat|function_other`: advantage `-0.1250915149610737`, events `1125`, bootstrap p05/p50/p95 `-0.1402001928176558/-0.12448973934765688/-0.11048286719572323`, p>0 `0.0`
  - `repeat|retained_content`: advantage `0.05300503073049906`, events `1129`, bootstrap p05/p50/p95 `0.04042137609945761/0.05239179590915112/0.06447018455355139`, p>0 `1.0`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `0.02774372865043717`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `0.028098190060530145`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.026045370121724865`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.01634645728354757`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.016234900922757492`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `-0.00467379033245259`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.021368003945739428`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `0.007374237752011667`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.026493838856583185`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `0.02529142488751393`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `-0.002752008201202902`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.02210386190745498`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.011054439024673367`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `-0.0020224472751579625`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `-0.05372222020082413`

## chck_20M
- Overall local advantage: `0.03253553954992182` over `5470` events / `7812` pieces, cluster bootstrap p05/p50/p95 `0.024322607674963137/0.032596785829957314/0.04062124824426587`, p>0 `1.0`
- View/category advantages:
  - `compact|function_other`: advantage `0.012547407023217855`, events `1129`, bootstrap p05/p50/p95 `-0.0018053348586025701/0.012730113195472554/0.02661188324292501`, p>0 `0.9166666666666666`
  - `compact|retained_content`: advantage `0.0448211707402525`, events `1122`, bootstrap p05/p50/p95 `0.02788723891042715/0.04428458349399657/0.059851076499376724`, p>0 `1.0`
  - `compact|source_absent_content`: advantage `0.1080962736715761`, events `965`, bootstrap p05/p50/p95 `0.08898578781679452/0.10785445789109148/0.12767682738223318`, p>0 `1.0`
  - `repeat|function_other`: advantage `-0.014903066714740084`, events `1125`, bootstrap p05/p50/p95 `-0.03142802076535013/-0.01607247676241511/0.002005870399606187`, p>0 `0.07333333333333333`
  - `repeat|retained_content`: advantage `0.0007045918317195756`, events `1129`, bootstrap p05/p50/p95 `-0.02003423011097092/0.0014637085438280318/0.018623936977654836`, p>0 `0.5566666666666666`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `0.02512294497986802`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `0.04173414064267819`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.06974663326565886`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.01356456913990716`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.025630600938856343`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.009480948480941838`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.0156335972283852`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `-0.00012822102351424391`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `0.017861527371527915`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.022912976855847908`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `-0.013320172631666637`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.02756559118836735`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `0.003734188422348364`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `-0.014262618893320315`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `-0.004130603592863955`

This file is a mechanism bridge, not an endpoint score or causal intervention.
