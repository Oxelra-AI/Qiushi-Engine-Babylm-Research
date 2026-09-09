# relational xspan compact v4 filter — compact v4 XSpan true/wrong/no-s1 likelihood probe

Model: `experiments/archive/initial_model_studies/training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_100M`
XSpan JSONL: `experiments/archive/initial_model_studies/data/xspan_revision_118/relational_xspan_compact_v4_from_v3_seed117_target200000_actual.jsonl`
Evidence JSON: `experiments/archive/initial_model_studies/data/xspan_compact_v4_s1_likelihood_probe.json`

The compact v4 rows were selected by rule-based text filters only; these scores are for mechanism judgment, not training-row selection. Deltas are target-token mean log-probabilities.

| group | n | true-wrong mean | true-wrong median | positive frac | true-no mean | wrong-no mean | mean target tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 212 | +0.7168 | +0.4906 | 0.764 | +0.9441 | +0.2273 | 6.08 |
| definition_property_complement | 56 | +0.5332 | +0.5113 | 0.804 | +0.5531 | +0.0199 | 6.73 |
| semantic_content_continuation | 46 | +0.8435 | +0.5041 | 0.761 | +0.9805 | +0.1370 | 5.50 |
| location_spatial_phrase | 84 | +0.5865 | +0.3099 | 0.714 | +0.9688 | +0.3823 | 6.01 |
| action_object_result_phrase | 26 | +1.3089 | +1.3786 | 0.846 | +1.6417 | +0.3327 | 5.92 |

## Top target prefixes among top-decile true-minus-wrong examples

- `is in`: 7
- `distributed by`: 3
- `black beak`: 1
- `in the`: 1
- `shares in`: 1
- `awarded since`: 1
- `interpretations and`: 1
- `released by`: 1
- `three movements`: 1
- `live single`: 1
- `texas lipan`: 1
- `released on`: 1

## Top positive examples

- Δ=+5.496 action_object_result_phrase `distributed by Columbia Pictures` :: The Mating of Millie is a 1948 American romantic comedy movie directed by Henry  / It was distributed by Columbia Pictures.
- Δ=+5.315 location_spatial_phrase `is in southwestern France` :: Montégut-Arros is a commune in the Gers department. / It is in southwestern France.
- Δ=+4.395 location_spatial_phrase `is in southwestern France` :: Lupiac is a commune in the Gers department. / It is in southwestern France.
- Δ=+4.232 semantic_content_continuation `black beak` :: It has orange-pink legs. / It has a black beak.
- Δ=+4.017 location_spatial_phrase `in the United Kingdom` :: "Keep It Dark" is a 1981 song by British band Genesis and is the third song from / It went to number 33 in the United Kingdom.
- Δ=+3.906 location_spatial_phrase `is in southwestern France` :: Séailles is a commune in the Gers department. / It is in southwestern France.
- Δ=+3.767 semantic_content_continuation `shares in BlackRock` :: BlackRock is a shareholder in many institutional investors. / They own shares in BlackRock.
- Δ=+3.525 location_spatial_phrase `is in southwestern France` :: Pallanne is a commune in the Gers department. / It is in southwestern France.

## Bottom examples

- Δ=-2.515 location_spatial_phrase `is in the San Fernando Valley region` :: The population was 1,725 at the 2020 census. / It is in the San Fernando Valley region.
- Δ=-1.629 location_spatial_phrase `from small ones` :: Denaturing the protein makes it unfold so it is more like a string. / This makes it easier to tell large proteins from small ones.
- Δ=-1.544 location_spatial_phrase `in Great Britain` :: The British Longhair is a breed of cat. / It originated in Great Britain.
- Δ=-1.423 location_spatial_phrase `on the Volga River` :: Cheboksary (; ; , "Šupaškar") is the capital city of Chuvashia, Russia. / It is also a major port on the Volga River.
- Δ=-1.414 location_spatial_phrase `on the Danube river` :: Ruse (also transliterated as Rousse, Russe; ) is a municipality in northern Bulg / It is a port city on the Danube river and is an important cultural and industrial center.
- Δ=-1.407 definition_property_complement `very much well-known` :: "Euthynotus incognitus" is a species of "Euthynotus". / It is very much well-known.
- Δ=-1.196 action_object_result_phrase `provides a high-speed satellite internet service` :: It is headquartered in Germantown, Maryland. / It provides a high-speed satellite internet service.
- Δ=-1.144 location_spatial_phrase `is near Bologna` :: About 32,000 people live there. / It is near Bologna.

## Mechanism interpretation rule

If true-minus-wrong is weak, concentrated in a few templates, or wrong-s1 already captures most true-s1 benefit over no-s1, do not add more boundary rules; change target representation or route.
