# earlier analysis row-aware exact bridge pilot

## Result
- selected rows: 55
- selected pairs: 125
- prompts: 500
- pairs with exact-length transformation-like output: 11 (0.088)
- accepted pair word yield: 510/6758 (0.075)
- estimated changed-block word fraction if scaled: 0.075
- whole original rows filled: 0/55 (0.000)

## Decision meaning
This is construction feasibility only. A 100M training contrast remains unjustified unless a full-pool generation can build a large matched subset with fully bridge rows and shared compact/extractive controls; no compact fallback may be used in the measured block.

## Top reasons for rejection
| reason | count |
|---|---:|
| not_exact_word_count | 377 |
| unsupported_content_lemma | 203 |
| not_transformation_like | 136 |
| lost_natural_relation_proxy | 19 |
| missing_natural_number_surface | 17 |
| no_finite_verb | 12 |
| new_entity_like | 9 |
| dropped_natural_polarity_not | 9 |
| dropped_natural_polarity_cannot | 6 |
| extra_polarity_not | 5 |
| grammar_pattern | 5 |
| extra_number_surface | 4 |

## Accepted examples

### 1. compact:frontier_consolidation_fwcompact_medium_025061 / exact_anchor_gapfill / substantive_restructure
Source: When mature, the cucumber fruit is 90% water, and is not particularly high in nutrients, but its flavor and texture have made it popular for use as a fresh addition to salads, as well as pickled and prepared in relishes.
Natural compact: When mature, the cucumber fruit is 90% water, low in nutrients, yet popular in salads, pickles, and relishes due to its flavor and texture.
Bridge: When mature, the cucumber fruit is 90% water, low in nutrients, yet popular in salads, pickles, and relishes due to its flavor and texture.
Metrics: run=0.3333 order=0.8889 gap1=0.8888888888888888 absent=0.08333333333333333

### 2. compact:frontier_consolidation_fwcompact_medium_035533 / exact_anchor_gapfill / substantive_restructure
Source: The rims of the wheels are covered in eyes, watching everything as the wheels turn, something that reminded me of how we interpret the All-Seeing Eye of God in Freemasonry, all-seeing, all-knowing and all-evaluating.
Natural compact: The wheel rims have eyes watching everything as they turn, reminding me of Freemasonry's All-Seeing Eye of God, which is all-seeing, all-knowing, and all-evaluating.
Bridge: The wheel rims have eyes watching everything as they turn, reminding me of Freemasonry's All-Seeing Eye of God, which is all-seeing, all-knowing, and all-evaluating.
Metrics: run=0.1667 order=1.0 gap1=1.0 absent=0.25

### 3. compact:frontier_consolidation_fwcompact_medium_026540 / exact_anchor_gapfill / light_restructure
Source: Instead, the scientific team MacArthur and Bailee of the University of Toronto found that each child is born with a certain amount of enzyme potential which can be either saved or used up according to the wear and tear of living at a faster or slower pace.
Natural compact: MacArthur and Bailee of the University of Toronto found each child is born with enzyme potential saved or used up based on living pace wear and tear.
Bridge: MacArthur and Bailee of the University of Toronto found each child is born with enzyme potential saved or used up based on living pace wear and tear.
Metrics: run=0.3333 order=0.9167 gap1=0.8 absent=0.0

### 4. compact:frontier_consolidation_fwcompact_medium_001244 / exact_anchor_gapfill / light_restructure
Source: Deep learning is a part of a broader family of Machine Learning that is inspired by the functionality of our brain cells called artificial neural network.
Natural compact: Deep learning is machine learning inspired by brain cells, using artificial neural networks.
Bridge: Deep learning is Machine learning inspired by brain cells using artificial neural network.
Metrics: run=0.2308 order=1.0 gap1=0.7 absent=0.0

### 5. compact:frontier_consolidation_fwcompact_medium_006110 / exact_anchor_gapfill / substantive_restructure
Source: Observations of some of the differences in modern species were recorded in a scientific paper published in German in 1906 and another published in Italian in 1947.
Natural compact: A 1906 German paper and a 1947 Italian paper recorded differences in modern species.
Bridge: A 1906 German paper and a 1947 Italian paper recorded differences in modern species.
Metrics: run=0.2143 order=0.25 gap1=0.0 absent=0.1

### 6. compact:frontier_consolidation_fwcompact_medium_046926 / exact_anchor_gapfill / light_restructure
Source: However, she also mentions children, who rarely are included in population-based biobanks, as well as people with early forms of dementia or addiction problems.
Natural compact: She also mentions children, people with early dementia, and those with addiction problems rarely in biobanks.
Bridge: She also mentions children and people with early dementia or addiction problems rarely included in biobanks.
Metrics: run=0.25 order=0.75 gap1=0.8888888888888888 absent=0.0

### 7. compact:frontier_consolidation_fwcompact_medium_012987 / exact_predicate_front / light_restructure
Source: The Attach command is used to bring online an existing database structure.
Natural compact: Attach command brings online existing database structure.
Bridge: Attach command brings online existing database structure.
Metrics: run=0.4286 order=1.0 gap1=0.75 absent=0.14285714285714285

### 8. compact:frontier_consolidation_fwcompact_medium_041299 / exact_anchor_gapfill / light_restructure
Source: Sphincters are little circles of muscle that are tethered in place by tough connective tissue, and control the access between one part of the body or another, as is the case with the anal sphincter.
Natural compact: Anal sphincters are little muscle circles tethered by tough tissue that control access between body parts.
Bridge: Anal sphincters are little muscle circles tethered by tough tissue that control access between body parts.
Metrics: run=0.1875 order=0.8182 gap1=None absent=0.08333333333333333

### 9. compact:frontier_consolidation_fwcompact_medium_009188 / exact_anchor_gapfill / substantive_restructure
Source: In short they are a lot like people, even uncomfortably so - a fact that, ironically, has caused many scientists to ignore them.
Natural compact: Ironically, scientists ignore them because they are uncomfortably like people.
Bridge: Ironically, scientists ignore them because they are uncomfortably like people.
Metrics: run=0.2 order=0.6 gap1=0.3333333333333333 absent=0.0

### 10. compact:frontier_consolidation_fwcompact_medium_004598 / exact_anchor_gapfill / substantive_restructure
Source: The problem with learning is that it requires experience, yet an early encounter with a predator can be deadly, allowing no second chance and no opportunity for learning.
Natural compact: Learning needs experience, but early predator encounters are deadly, allowing no second chance or learning opportunity.
Bridge: Learning requires experience, but early predator encounters are deadly, allowing no second chance or learning opportunity.
Metrics: run=0.3125 order=0.9 gap1=0.7142857142857143 absent=0.08333333333333333

### 11. compact:frontier_consolidation_fwcompact_medium_007814 / exact_anchor_gapfill / light_restructure
Source: This makes them so they are more winter hardy, need less honey and pollen to get through the winter, and it disrupts the varroa mite (bee pest) life cycle.
Natural compact: This makes them more winter hardy, needing less honey and pollen, and disrupting the varroa mite life cycle.
Bridge: This makes them more winter hardy, needing less honey and pollen, and disrupting the varroa mite life cycle.
Metrics: run=0.2222 order=1.0 gap1=0.7692307692307693 absent=0.18181818181818182

