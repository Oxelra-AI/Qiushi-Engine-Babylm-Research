# roberta stratified bridge and early local response RoBERTa pair-stratified local response

Created: `2026-09-02T03:23:49Z`

Positive `repeat_minus_compact_advantage` means the compact-trained model predicts the masked event better than the repeat-trained model. Interpret only with selected official-compatible trajectory results.

## chck_60M
- Overall local advantage: `0.2925413478522562` over `5470` events / `7812` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.23567994443899784`, events `1129`
  - `compact|retained_content`: advantage `0.2852064296494752`, events `1122`
  - `compact|source_absent_content`: advantage `0.35412695997981103`, events `965`
  - `repeat|function_other`: advantage `0.29034314417554735`, events `1125`
  - `repeat|retained_content`: advantage `0.2888917949124413`, events `1129`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `-0.012163869181531978`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `0.0930279260018888`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.09633453963657213`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.01742852718084764`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `-0.0037656962279080664`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.02276302801494645`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `0.007456928341888491`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `0.035207184876821285`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.06175077232249265`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `0.04298309464810346`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `-0.01754743915340906`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `-0.007889129652468518`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.00939800972460636`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `0.013472243086365965`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `-0.06576431160750884`

## chck_70M
- Overall local advantage: `0.31265861237935694` over `5470` events / `7812` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.32236948920775216`, events `1129`
  - `compact|retained_content`: advantage `0.2710754158194929`, events `1122`
  - `compact|source_absent_content`: advantage `0.35503785121338055`, events `965`
  - `repeat|function_other`: advantage `0.4114471449178772`, events `1125`
  - `repeat|retained_content`: advantage `0.2560491714568748`, events `1129`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `-0.0687309655645385`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `-0.14132297298207677`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `-0.08099287133701061`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.09371641237969641`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.011050833497652457`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `-0.0011632218240539838`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.027016604538073946`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `0.06981290762281722`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.10732038269590577`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.10950763037232453`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `-0.03225267308341501`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `-0.011373368793018368`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.11965646901926241`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `0.013843606681355203`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `-0.012809925738219458`

## chck_80M
- Overall local advantage: `0.3719394039777544` over `5470` events / `7812` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.26101852673029124`, events `1129`
  - `compact|retained_content`: advantage `0.37929011657219497`, events `1122`
  - `compact|source_absent_content`: advantage `0.5442877004062726`, events `965`
  - `repeat|function_other`: advantage `0.31975419710198294`, events `1125`
  - `repeat|retained_content`: advantage `0.32892319481319576`, events `1129`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `-0.014177939003643836`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `-0.08595313014889211`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `-0.020697881258432305`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.016500601014059313`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.035116794874396506`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.0859938813177277`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.022807138938369886`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `0.031482738311373604`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.06985435507252186`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.02766997270302962`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `0.004327345814090999`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.06137932808409369`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.1170855939054965`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `0.02696693352370849`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `0.014862004588153732`

## chck_90M
- Overall local advantage: `0.36706648190592595` over `5470` events / `7812` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.2778418401906036`, events `1129`
  - `compact|retained_content`: advantage `0.3679016620528056`, events `1122`
  - `compact|source_absent_content`: advantage `0.5002068310442078`, events `965`
  - `repeat|function_other`: advantage `0.30652602909853144`, events `1125`
  - `repeat|retained_content`: advantage `0.3541555529930706`, events `1129`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `-0.004923480288754445`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `-0.060701090877328834`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.03445495756391814`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.012732738441097735`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.037591340859059796`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.10826520726495265`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.00028708326635129566`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `-0.002300060897109213`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.10714763975911762`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.019095983246690984`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `0.01470883385779903`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.06897830434970481`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.08792121433567401`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `0.029601061788746996`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `0.03464990503574267`

## chck_100M
- Overall local advantage: `0.3698290071145694` over `5470` events / `7812` pieces
- View/category advantages:
  - `compact|function_other`: advantage `0.27944349388992856`, events `1129`
  - `compact|retained_content`: advantage `0.3881595748025257`, events `1122`
  - `compact|source_absent_content`: advantage `0.4973058140507263`, events `965`
  - `repeat|function_other`: advantage `0.2883839473267938`, events `1125`
  - `repeat|retained_content`: advantage `0.35606905714986936`, events `1129`
- Feature contrast highlights (positive = stronger compact local advantage in the higher-feature stratum):
  - `compact_content_fraction|compact|function_other` (high_minus_low): `-0.0009239544330624794`
  - `compact_content_fraction|compact|retained_content` (high_minus_low): `-0.06329499595036214`
  - `compact_content_fraction|compact|source_absent_content` (high_minus_low): `0.03552284800432748`
  - `compact_content_fraction|repeat|function_other` (high_minus_low): `-0.02962434389254681`
  - `compact_content_fraction|repeat|retained_content` (high_minus_low): `0.04503293280454895`
  - `compact_source_absent_content_fraction_of_content|compact|function_other` (high_positive_minus_zero): `0.09040806768368731`
  - `compact_source_absent_content_fraction_of_content|compact|retained_content` (high_positive_minus_zero): `-0.00235711150768092`
  - `compact_source_absent_content_fraction_of_content|compact|source_absent_content` (high_positive_minus_low_positive): `-0.002884047671353529`
  - `compact_source_absent_content_fraction_of_content|repeat|function_other` (high_positive_minus_zero): `-0.11929286048177351`
  - `compact_source_absent_content_fraction_of_content|repeat|retained_content` (high_positive_minus_zero): `-0.039555179276468844`
  - `compact_tail_content_coverage|compact|function_other` (high_minus_low): `0.038402063172933376`
  - `compact_tail_content_coverage|compact|retained_content` (high_minus_low): `0.05432893910735331`
  - `compact_tail_content_coverage|compact|source_absent_content` (high_minus_low): `-0.08095283889780419`
  - `compact_tail_content_coverage|repeat|function_other` (high_minus_low): `0.034151454922565494`
  - `compact_tail_content_coverage|repeat|retained_content` (high_minus_low): `0.02110639612948817`

This file is a mechanism bridge, not an endpoint score or causal intervention.
