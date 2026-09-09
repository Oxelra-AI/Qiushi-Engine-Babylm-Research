# relational xspan materialization — s1-ablation probe pattern analysis (mechanism discovery only)

Input: `experiments/archive/initial_model_studies/data/s1_ablation_likelihood_probe.json`
Evidence JSON: `experiments/archive/initial_model_studies/data/s1_probe_pattern_analysis.json`

**Policy:** use these scores only to discover model-independent relation patterns. Do not select training rows by protected-model delta under Strict-Small.

| target type | n | true-wrong mean | median | positive frac | true-no mean |
|---|---:|---:|---:|---:|---:|
| initial_dependent | 51 | -0.003 | -0.016 | 0.431 | +2.458 |
| first_s2_content | 51 | +0.284 | +0.073 | 0.588 | +0.469 |
| last_s2_content | 51 | +0.314 | +0.130 | 0.588 | +0.244 |
| first_two_s2_content_window | 47 | +0.346 | +0.278 | 0.702 | +0.434 |

## Mechanism observations from high true-vs-wrong content spans

- `India` (last_s2_content, Δ=+3.325, location/spatial complement): The Honnametti bush frog ("Roarchestes honnametti") is a frog. / It lives in India.
- `churches` (last_s2_content, Δ=+3.115, semantic content continuation): The efforts of Western scholars have developed the use of the Latin alphabet to write F... / This set of standards is mainly used in churches.
- `split` (first_s2_content, Δ=+2.934, semantic content continuation): The carbonate deposits were laid down during the Cretaceous Period in what was then the... / It split the continent of North America into two landmasses.
- `teams` (last_s2_content, Δ=+2.491, semantic content continuation): A first-class match is one of three or more days' scheduled duration between two sides ... / It is officially adjudged to be worthy of the status by virtue of the standard of the competing teams.
- `released` (first_s2_content, Δ=+2.340, semantic content continuation): The album's name, "Black Dog Barking", was announced in early February 2013 on their Fa... / It was released on 21 May 2013 via Roadrunner.
- `English` (first_s2_content, Δ=+2.233, location/spatial complement): The open back unrounded vowel is a sound used in some spoken languages. / It is in English.
- `English` (last_s2_content, Δ=+2.233, location/spatial complement): The open back unrounded vowel is a sound used in some spoken languages. / It is in English.
- `lives in India` (first_two_s2_content_window, Δ=+1.982, location/spatial complement): The Honnametti bush frog ("Roarchestes honnametti") is a frog. / It lives in India.
- `young star` (first_two_s2_content_window, Δ=+1.963, definition/property complement): The star is about 505 light years from Earth. / It is a young star, about two million years old.
- `stars Peter` (first_two_s2_content_window, Δ=+1.879, named entity continuation): The Trials of Oscar Wilde is a 1960 British biographical drama movie directed by Ken Hu... / It stars Peter Finch, Yvonne Mitchell, Sonia Dresdel, Emrys Jones, Lionel Jeffries, James Mason, Nigel Patr...
- `lives` (first_s2_content, Δ=+1.685, location/spatial complement): The greater sage-grouse ("Centrocercus urophasianus") is the largest grouse in North Am... / It lives in the western half of the United States and the Alberta and Saskatchewan provinces.
- `books` (last_s2_content, Δ=+1.550, semantic content continuation): The first colleges in the United States were to train clergy members. / These libraries mostly had donated books.

## Rule implication

- Initial pronoun/deictic tokens are not useful XSpan targets: they show true-s1 > no-s1 but essentially no true-s1 > wrong-s1 specificity.
- More promising model-independent targets are semantic content spans after the initial dependent: location/prepositional complements, action-result/object phrases, and definition/property complements following a short definition-style s1.
- The next materializer should use only corpus-intrinsic patterns: same-line adjacent Simple-Wiki definition/description pairs, s2 starts with It/They/This/These/That/Those, target is the first semantic content phrase after the dependent/copula/verb, not selected by model score.
