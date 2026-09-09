# roberta stratified bridge and early local response RoBERTa pair-stratified local response

Created: `2026-09-02T02:44:03Z`

Positive `repeat_minus_compact_advantage` means the compact-trained model predicts the masked event better than the repeat-trained model. Interpret only with selected official-compatible trajectory results.

## chck_20M
- Overall local advantage: `0.04398561093088372` over `5224` events / `7499` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.01927199743342629`, events `1076`
  - `compact|retained_content`: advantage `0.049169936150980094`, events `1071`
  - `compact|source_absent_content`: advantage `0.12738112411243002`, events `934`
  - `repeat|function_other`: advantage `0.003069598303900825`, events `1069`
  - `repeat|retained_content`: advantage `0.01012902402045959`, events `1074`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `0.011830105023242714`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `0.02898969820253368`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.07329330186285012`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.016188118238323685`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.04074838580876653`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.02459092196283505`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.0024525550975709673`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `0.006420134662201801`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `0.01442295271345674`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.007826106235571234`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `-0.01978280113479007`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.0244102695578228`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.031708883975432076`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `-0.01951282307253046`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `-0.00042762240066791454`

This file is a mechanism bridge, not an endpoint score or causal intervention.
