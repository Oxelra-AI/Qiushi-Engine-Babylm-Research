# density cleanqwen overlay medium riskhard medium compact semantic review bundle

Bounded independent semantic assessment of accepted medium compact views before training: relation, negation/modality, roles/coreference, entity and number fidelity.

Rows: `experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_rows.jsonl`
Summary: `experiments/archive/frontier_consolidation/data/medium_compact_analysis/medium_compact_ws_summary.json`
Bundle JSON: `experiments/archive/frontier_consolidation/data/medium_semantic_review/medium_compact_semantic_review_bundle.json`

## Automatic summary

{
  "n": 21465,
  "accepted": 18682,
  "accepted_rate": 0.8703470766363848,
  "source_words_whitespace": 469887,
  "accepted_source_words_whitespace": 400665,
  "accepted_rewrite_words_whitespace": 244568,
  "accepted_pair_words_whitespace": 645233,
  "weighted_rewrite_to_source_ratio_whitespace": 0.610405201352751,
  "length_ratio_stats": {
    "n": 21465,
    "min": 0.18181818181818182,
    "p05": 0.44,
    "mean": 0.6144652095109683,
    "median": 0.6071428571428571,
    "p95": 0.812162162162161,
    "max": 1.2727272727272727,
    "sum": 13189.495722152935
  },
  "content_recall_stats": {
    "n": 21465,
    "min": 0.0,
    "p05": 0.35,
    "mean": 0.6199460442176835,
    "median": 0.625,
    "p95": 0.8888888888888888,
    "max": 1.0,
    "sum": 13307.141839132577
  },
  "entity_recall_stats": {
    "n": 21465,
    "min": 0.0,
    "p05": 0.5,
    "mean": 0.9364196976251483,
    "median": 1.0,
    "p95": 1.0,
    "max": 1.0,
    "sum": 20100.248809523808
  },
  "number_recall_stats": {
    "n": 21465,
    "min": 0.0,
    "p05": 1.0,
    "mean": 0.9805342029660689,
    "median": 1.0,
    "p95": 1.0,
    "max": 1.0,
    "sum": 21047.166666666668
  }
}

## Examples

### 1. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_055740
Metrics: source_words=13 rewrite_words=5 ratio=0.38461538461538464 recall=0.14285714285714285 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Each antler grows from an attachment point on the skull called a pedicle.
REWRITE: Antlers grow from skull pedicles.

### 2. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_002010
Metrics: source_words=12 rewrite_words=5 ratio=0.4166666666666667 recall=0.14285714285714285 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: As the report makes clear, greenbelts can be found around the world.
REWRITE: Reports show greenbelts exist worldwide.

### 3. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_027762
Metrics: source_words=12 rewrite_words=6 ratio=0.5 recall=0.14285714285714285 overlap=0.1 entity=1.0 number=1.0 domains=['causal_relational', 'institutions_society'] risks=[] flags=[] hard=[]
SOURCE: The program allows us a great choice of different features and options.
REWRITE: The program offers many feature choices.

### 4. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_001007
Metrics: source_words=12 rewrite_words=7 ratio=0.5833333333333334 recall=0.14285714285714285 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: The hunting of bats is both for local consumption and commercial purposes.
REWRITE: Bats are hunted for eating and selling.

### 5. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_028476
Metrics: source_words=11 rewrite_words=5 ratio=0.45454545454545453 recall=0.14285714285714285 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
SOURCE: Keeping a baby safe in the crib requires a less-is-more approach.
REWRITE: Safety needs less, not more.

### 6. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_022853
Metrics: source_words=13 rewrite_words=8 ratio=0.6153846153846154 recall=0.14285714285714285 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['POPs'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: In recent decades, vast numbers of POPs have been produced and used worldwide.
REWRITE: Many POPs were made and used globally recently.

### 7. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_040292
Metrics: source_words=11 rewrite_words=7 ratio=0.6363636363636364 recall=0.14285714285714285 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Asbestos products were used extensively in both home and commercial construction.
REWRITE: Asbestos was used in homes and buildings.

### 8. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_035624
Metrics: source_words=11 rewrite_words=5 ratio=0.45454545454545453 recall=0.14285714285714285 overlap=0.125 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Another bad element in them is the excessive amount of salt.
REWRITE: They have too much salt.

### 9. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_059471
Metrics: source_words=10 rewrite_words=5 ratio=0.5 recall=0.16666666666666666 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: The structure of the nerve cells themselves is correspondingly altered.
REWRITE: Nerve cell structures change accordingly.

### 10. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_049172
Metrics: source_words=12 rewrite_words=7 ratio=0.5833333333333334 recall=0.16666666666666666 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=[] missing=[] new=['men'] | numbers source=['0.95'] output=['0.95']
SOURCE: For men, a ratio higher than 0.95 is considered dangerous or unhealthy.
REWRITE: Men with ratios above 0.95 face danger.

### 11. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_035178
Metrics: source_words=13 rewrite_words=8 ratio=0.6153846153846154 recall=0.16666666666666666 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: “As” is an indirect comparison (simile) as opposed to a direct comparison (metaphor).
REWRITE: As" compares indirectly, unlike "metaphor" which compares directly.

### 12. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_008798
Metrics: source_words=13 rewrite_words=8 ratio=0.6153846153846154 recall=0.16666666666666666 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: This difference of opinion is further emphasised in many instances between the generations.
REWRITE: Generational differences are often emphasized in many instances.

### 13. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_021806
Metrics: source_words=10 rewrite_words=7 ratio=0.7 recall=0.16666666666666666 overlap=0.1 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: These categories are woven together in everyday cognition and interaction.
REWRITE: Categories mix in daily thinking and action.

### 14. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_027344
Metrics: source_words=11 rewrite_words=5 ratio=0.45454545454545453 recall=0.16666666666666666 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Otherwise, make sure that your car is running in good condition.
REWRITE: Ensure your car runs well.

### 15. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_044910
Metrics: source_words=12 rewrite_words=6 ratio=0.5 recall=0.16666666666666666 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Did people wear these cones, or were they a purely iconographic device?
REWRITE: Were cones worn or just symbolic?

### 16. compact_low_content_train_rows | frontier_consolidation_fwcompact_medium_037040
Metrics: source_words=13 rewrite_words=7 ratio=0.5384615384615384 recall=0.16666666666666666 overlap=0.1111111111111111 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: The information provided in this article was correct at the time of publishing.
REWRITE: This article's info was correct when published.

### 17. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_042819
Metrics: source_words=47 rewrite_words=34 ratio=0.723404255319149 recall=0.5666666666666667 overlap=0.4594594594594595 entity=0.75 number=1.0 domains=['science_physical'] risks=[] flags=[] hard=[]
Entities: source=['ESA', 'Eneide Mission', 'ISS', 'Italy Greece Portugal', 'Norway', 'Roberto Vittori', 'Russian', 'Sergei Krikalev'] missing=['Italy Greece Portugal', 'Norway'] new=['norwegian'] | numbers source=['2005'] output=['2005']
SOURCE: Filmed during the Eneide Mission in 2005, ESA astronaut Roberto Vittori and Russian cosmonaut Sergei Krikalev are shown investigating the nature of matter from on board the ISS, while students from schools in Italy, Greece, Portugal and Norway present their own inventive ways of exploring the subject.
REWRITE: Filmed in 2005 during the Eneide Mission, ESA astronaut Roberto Vittori and Russian cosmonaut Sergei Krikalev investigated matter from the ISS while Italian, Greek, Portuguese, and Norwegian students presented inventive ways to explore it.

### 18. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_028146
Metrics: source_words=33 rewrite_words=19 ratio=0.5757575757575758 recall=0.5882352941176471 overlap=0.5555555555555556 entity=0.75 number=1.0 domains=['people_history'] risks=[] flags=[] hard=[]
Entities: source=['Artificial Intelligence Laboratory CSAIL', 'CSAIL', 'Computation Group', 'Cryptography', 'Information Security Group', 'MIT', "MIT's Computer Science", 'Theory'] missing=['Artificial Intelligence Laboratory CSAIL', "MIT's Computer Science"] new=[] | numbers source=[] output=[]
SOURCE: Rivest is a member of MIT's Computer Science and Artificial Intelligence Laboratory (CSAIL), a member of the lab's Theory of Computation Group and is a leader of its Cryptography and Information Security Group.
REWRITE: Rivest is a CSAIL member, Theory of Computation Group member, and Cryptography and Information Security Group leader at MIT.

### 19. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_036858
Metrics: source_words=27 rewrite_words=16 ratio=0.5925925925925926 recall=0.7272727272727273 overlap=0.6153846153846154 entity=0.75 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['AN', 'APPLE', 'APPLE SEED', 'OF', 'SEED', 'SHAPE', 'THE', 'THE SHAPE'] missing=['SHAPE', 'THE SHAPE'] new=[] | numbers source=[] output=[]
SOURCE: THE SHAPE OF AN APPLE SEED To the naked eye, the typical shape of a bed bug is similar to an apple seed or dark sesame seed.
REWRITE: A bed bug looks like an apple seed or dark sesame seed to the naked eye.

### 20. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_008787
Metrics: source_words=34 rewrite_words=27 ratio=0.7941176470588235 recall=0.75 overlap=0.7142857142857143 entity=0.75 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Mozambique', 'Near East South Asia', 'Saharan Africa', 'Southeast Asia'] missing=['Saharan Africa'] new=[] | numbers source=['10', '19', '1990', '4'] output=['10', '19', '1990', '4']
SOURCE: Fusarium wilt TR4 was first detected in Southeast Asia in the 1990s and has now been identified at 19 sites in 10 countries, including the Near East, South Asia and Mozambique in sub-Saharan Africa.
REWRITE: Fusarium wilt TR4, first found in Southeast Asia in the 1990s, is now at 19 sites in 10 countries including the Near East, South Asia, and Mozambique.

### 21. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_027979
Metrics: source_words=42 rewrite_words=19 ratio=0.4523809523809524 recall=0.6 overlap=0.5714285714285714 entity=0.75 number=1.0 domains=[] risks=['deictic_time'] flags=['source_risk_deictic_time'] hard=[]
Entities: source=['Illustrated', 'Jurassic', 'MAMMALS', 'MARSUPIAL', 'MARSUPIAL MAMMALS', 'Mammalian Milestones', 'Other', 'Therian Mammals'] missing=['Illustrated', 'Mammalian Milestones'] new=[] | numbers source=[] output=[]
SOURCE: Phylogeny of Therian Mammals originated during the Jurassic survive today Other early mammal groups (now extinct) MARSUPIAL MAMMALS originated during the Jurassic survive today Mammalian Milestones originated during the Jurassic survive today Illustrated by mammalian evolution But two factors are now clear.
REWRITE: Marsupial mammals and therian mammals originated in the Jurassic and survive today, while other early mammal groups are extinct.

### 22. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_030905
Metrics: source_words=47 rewrite_words=24 ratio=0.5106382978723404 recall=0.5238095238095238 overlap=0.4782608695652174 entity=0.75 number=1.0 domains=[] risks=['heading_dash_chain'] flags=['source_risk_heading_dash_chain'] hard=[]
Entities: source=['Biodegradable', 'MSW', 'Recovering', 'Reducing'] missing=['MSW'] new=[] | numbers source=['2010/11', '2015/16', '45%', '67%'] output=['2010/11', '2015/16', '45%', '67%']
SOURCE: Devon's targets include - - Recovering 45% of municipal solid waste (MSW) by 2010/11 - Recovering 67% of municipal solid waste (MSW) by 2015/16 Reducing the amount of biodegradable waste that is buried in landfill sites.'Biodegradable' means anything that rots, for example food scraps and garden waste.
REWRITE: Devon targets recovering 45% of municipal solid waste by 2010/11, 67% by 2015/16, and reducing buried biodegradable waste like food scraps and garden waste.

### 23. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_019155
Metrics: source_words=29 rewrite_words=18 ratio=0.6206896551724138 recall=0.5 overlap=0.4117647058823529 entity=0.75 number=1.0 domains=['people_history', 'quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['East German Falkensee', 'January', 'June', 'Staaken'] missing=['January'] new=['on jan'] | numbers source=['1', '1952', '1961'] output=['1', '1952', '1961']
SOURCE: This ended on January 1, 1961, when western Staaken was incorporated into then East German Falkensee, which had already been under its de facto administration since June 1, 1952.
REWRITE: On Jan 1, 1961, western Staaken joined East German Falkensee, which had administered it since June 1, 1952.

### 24. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_001863
Metrics: source_words=30 rewrite_words=20 ratio=0.6666666666666666 recall=0.6470588235294118 overlap=0.5789473684210527 entity=0.75 number=1.0 domains=['causal_relational', 'geography_places'] risks=[] flags=[] hard=[]
Entities: source=['Canalino Elementary', 'Island Fox Ambassador Schools Canalino Elementary SchoolTeachers Kathleen Tomscha', 'Island Fox Ambassadors', 'Monica Shugart'] missing=['Island Fox Ambassadors'] new=[] | numbers source=['2009', '2014', '4th'] output=['2009', '2014', '4th']
SOURCE: Highlighting two 2014 Island Fox Ambassador Schools: Canalino Elementary SchoolTeachers Kathleen Tomscha and Monica Shugart led the 4th grade teachers at Canalino Elementary to become Island Fox Ambassadors in 2009.
REWRITE: In 2009, Kathleen Tomscha and Monica Shugart led Canalino Elementary's 4th graders to become two 2014 Island Fox Ambassador Schools.

### 25. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_039568
Metrics: source_words=20 rewrite_words=13 ratio=0.65 recall=0.5 overlap=0.45454545454545453 entity=0.75 number=1.0 domains=['quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['August', 'CDC', 'January', 'United States'] missing=['United States'] new=[] | numbers source=['1', '159', '24'] output=['1', '159', '24']
SOURCE: There were 159 cases of measles in the United States from January 1 through August 24, according to the CDC.
REWRITE: The CDC reported 159 US measles cases from January 1 to August 24.

### 26. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_002360
Metrics: source_words=35 rewrite_words=18 ratio=0.5142857142857142 recall=0.5 overlap=0.46153846153846156 entity=0.75 number=1.0 domains=['science_physical'] risks=[] flags=[] hard=[]
Entities: source=['Diabetes', 'Men From', 'Symptoms', 'Type'] missing=['Symptoms'] new=[] | numbers source=['2', '90%', '95%'] output=['2', '90%', '95%']
SOURCE: Signs and Symptoms of Diabetes in Men From different types of diabetes, Type 2 is very common in men and it affects around 90% to 95% of the men who are suffering from the disease.
REWRITE: Type 2 diabetes is very common in men, affecting 90% to 95% of men suffering from the disease.

### 27. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_025583
Metrics: source_words=44 rewrite_words=28 ratio=0.6363636363636364 recall=0.5714285714285714 overlap=0.48 entity=0.75 number=1.0 domains=['institutions_society'] risks=[] flags=[] hard=[]
Entities: source=['DNA', 'Imperial College London', 'Lausanne', 'University'] missing=['DNA'] new=['bmi', 'researchers'] | numbers source=['16', '18.5', '95000'] output=['16', '18.5', '95000']
SOURCE: In a study examining the DNA of over 95,000 people, researchers at Imperial College London and the University of Lausanne have identified that duplication of a part of chromosome 16 is associated with being underweight, defined as a a body mass index below 18.5.
REWRITE: Researchers at Imperial College London and the University of Lausanne found that duplicating part of chromosome 16 links to being underweight (BMI below 18.5) in over 95,000 people.

### 28. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_028201
Metrics: source_words=33 rewrite_words=18 ratio=0.5454545454545454 recall=0.4 overlap=0.3 entity=0.75 number=1.0 domains=['quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['January', "The New Year's Concert", 'Vienna', 'Vienna Philharmonic'] missing=['January'] new=[] | numbers source=['1', '50', '73'] output=['1', '50', '73']
SOURCE: The New Year's Concert of the Vienna Philharmonic takes place each year on the morning of January 1 in Vienna and is broadcast to an estimated audience of 50 million in 73 countries.
REWRITE: Vienna Philharmonic's New Year's Concert airs Jan 1 morning in Vienna to 50 million viewers across 73 countries.

### 29. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_015938
Metrics: source_words=32 rewrite_words=20 ratio=0.625 recall=0.5 overlap=0.5 entity=0.75 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['East Coast', 'Islip New York', 'John', 'Newark New Jersey'] missing=['East Coast'] new=[] | numbers source=['93', '97', '99'] output=['93', '97', '99']
SOURCE: Reuters reported that the East Coast was sweltering too: Newark, New Jersey temperatures hovered at 99 degrees, Islip, New York at 93 degrees, and the mercury hit 97 degrees at John F.
REWRITE: Reuters reported Newark, New Jersey at 99 degrees, Islip, New York at 93 degrees, and John F. at 97 degrees.

### 30. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_018704
Metrics: source_words=19 rewrite_words=15 ratio=0.7894736842105263 recall=0.8888888888888888 overlap=0.8888888888888888 entity=0.75 number=1.0 domains=['geography_places', 'institutions_society', 'people_history'] risks=[] flags=[] hard=[]
Entities: source=['Clerk', 'Common Council', 'New York City', 'Thomas Valentine'] missing=['Thomas Valentine'] new=[] | numbers source=['1801', '1869'] output=['1801', '1869']
SOURCE: David T. (Thomas) Valentine (1801 - 1869) served as the Clerk of the Common Council of New York City.
REWRITE: David T. Valentine served as New York City Common Council Clerk from 1801 to 1869.

### 31. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_044519
Metrics: source_words=21 rewrite_words=12 ratio=0.5714285714285714 recall=0.5 overlap=0.46153846153846156 entity=0.75 number=1.0 domains=['people_history', 'quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['August', 'DeSoto', 'Maddox', 'The Cold War Era Chronology'] missing=['The Cold War Era Chronology'] new=[] | numbers source=['1964', '2'] output=['1964', '2']
SOURCE: "The Cold War Era" Chronology On August 2, 1964, the Maddox was conducting a "DeSoto patrol", referring to an espionage mission.
REWRITE: On August 2, 1964, the Maddox conducted a DeSoto patrol espionage mission.

### 32. compact_entity_or_number_sensitive | frontier_consolidation_fwcompact_medium_057122
Metrics: source_words=22 rewrite_words=15 ratio=0.6818181818181818 recall=0.6363636363636364 overlap=0.5833333333333334 entity=0.75 number=1.0 domains=['geography_places', 'quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['Bartlett', 'February', 'Springfield Illinois', 'The Village'] missing=['The Village'] new=[] | numbers source=['11', '1891'] output=['11', '1891']
SOURCE: The Village of Bartlett got its official start on February 11, 1891, when the petition for incorporation was filed in Springfield, Illinois.
REWRITE: Bartlett started on February 11, 1891, when its incorporation petition was filed in Springfield, Illinois.

### 33. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_056440
Metrics: source_words=12 rewrite_words=8 ratio=0.6666666666666666 recall=0.75 overlap=0.75 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
Entities: source=['Universe'] missing=[] new=['bang'] | numbers source=['13.7'] output=['13.7']
SOURCE: Bang, which formed the Universe 13.7 billion years ago, say its discoverers.
REWRITE: Bang formed the Universe 13.7 billion years ago.

### 34. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_058912
Metrics: source_words=25 rewrite_words=15 ratio=0.6 recall=0.6153846153846154 overlap=0.47058823529411764 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
Entities: source=['Spring'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Honey bee swarms occur mostly in the Spring and are a usually result of a successful colony growing to a point where they can divide.
REWRITE: Honey bee swarms mostly occur in Spring when successful colonies grow large enough to divide.

### 35. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_015503
Metrics: source_words=37 rewrite_words=25 ratio=0.6756756756756757 recall=0.782608695652174 overlap=0.75 entity=1.0 number=1.0 domains=[] risks=['heading_dash_chain'] flags=['source_risk_heading_dash_chain'] hard=[]
Entities: source=['DNA'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: For several years, scientists had used mass spectrometers and high-speed DNA dispensers to find "point mutations" - misspellings in a single letter of genetic code - in people with an inherited risk for various kinds of cancer.
REWRITE: For years, scientists used mass spectrometers and high-speed DNA dispensers to find point mutations, which are single-letter genetic misspellings, in people with inherited cancer risks.

### 36. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_041401
Metrics: source_words=37 rewrite_words=21 ratio=0.5675675675675675 recall=0.4117647058823529 overlap=0.2916666666666667 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
SOURCE: People generally do what they do because of their own karma, We get annoyed because we see our self as the target of rudeness or inconsiderate deeds committed by others, We think ii is happening to us.
REWRITE: People act due to their karma; we feel annoyed because we see ourselves as targets of others' rude or inconsiderate acts.

### 37. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_004889
Metrics: source_words=15 rewrite_words=7 ratio=0.4666666666666667 recall=0.4 overlap=0.18181818181818182 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Is there a risk for those who do not smoke, but breathe in secondhand smoke?
REWRITE: Non-smokers breathing secondhand smoke face health risks.

### 38. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_004523
Metrics: source_words=40 rewrite_words=21 ratio=0.525 recall=0.6 overlap=0.45 entity=1.0 number=1.0 domains=['institutions_society'] risks=['heading_dash_chain'] flags=['source_risk_heading_dash_chain'] hard=[]
Entities: source=['Ruling', 'Yom Tov'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Rashi makes two suggestions: - Ruling that the animal can be eaten is tikkun - it is “fixing” something on Yom Tov - which is forbidden. - Ruling that the animal can be eaten is considered a formal court ruling.
REWRITE: Rashi suggests eating the animal is forbidden on Yom Tov because it fixes something or acts as a formal court ruling.

### 39. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_023205
Metrics: source_words=38 rewrite_words=20 ratio=0.5263157894736842 recall=0.4782608695652174 overlap=0.3793103448275862 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['The California Endowment'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: The California Endowment is committed to helping communities combat the insidious promotion of junk drinks to kids, many of whom reside in low-income neighborhoods with the least access to fresh, healthy foods and affordable alternatives to junk drinks.
REWRITE: The California Endowment helps communities fight junk drink promotion for kids in low-income areas lacking fresh food and affordable alternatives.

### 40. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_003160
Metrics: source_words=22 rewrite_words=17 ratio=0.7727272727272727 recall=0.9285714285714286 overlap=0.9285714285714286 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
SOURCE: Modern lifestyles and upscale urban living have led to a shedding of inhibitions and increased sexual freedom to explore various sexual behaviors.
REWRITE: Modern lifestyles and upscale urban living led to shedding inhibitions and increased sexual freedom to explore behaviors.

### 41. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_017941
Metrics: source_words=26 rewrite_words=16 ratio=0.6153846153846154 recall=0.6153846153846154 overlap=0.5333333333333333 entity=1.0 number=1.0 domains=['people_history'] risks=[] flags=[] hard=[]
SOURCE: However, there are a few differences that will enable even a novice to quickly tell if a crayfish or a lobster is being served to him.
REWRITE: However, a few differences help novices quickly tell if a crayfish or lobster is being served.

### 42. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_023114
Metrics: source_words=30 rewrite_words=16 ratio=0.5333333333333333 recall=0.5294117647058824 overlap=0.45 entity=1.0 number=1.0 domains=['causal_relational'] risks=['deictic_time'] flags=['source_risk_deictic_time'] hard=[]
Entities: source=['Congress', 'FDA'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: "Any such experiment aimed at a pregnancy would need FDA approval, and the agency is currently not allowed to even consider such a request" because of limits set by Congress.
REWRITE: Any pregnancy experiment needs FDA approval, which Congress limits, so the agency cannot consider such requests.

### 43. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_001458
Metrics: source_words=32 rewrite_words=21 ratio=0.65625 recall=0.6111111111111112 overlap=0.5238095238095238 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Okano'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: The injection was given on the ninth day after the injury, considered the most effective timing, and the monkey started to move its limbs again within two to three weeks, Okano said.
REWRITE: Okano said the injection on day nine, the best timing, helped the monkey move limbs again in two to three weeks.

### 44. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_018089
Metrics: source_words=38 rewrite_words=19 ratio=0.5 recall=0.55 overlap=0.4782608695652174 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['MHC', 'MHCs'] missing=[] new=['researchers'] | numbers source=[] output=[]
SOURCE: By matching a group of proteins called major histocompatibility complex proteins (MHCs) of the iPSCs to the MHC of the host, the neuron survival is improved, and the immune response against the neurons is reduced, write the researchers.
REWRITE: Researchers write that matching iPSC MHCs to host MHCs improves neuron survival and reduces immune response against the neurons.

### 45. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_047892
Metrics: source_words=35 rewrite_words=19 ratio=0.5428571428571428 recall=0.5 overlap=0.4166666666666667 entity=1.0 number=1.0 domains=['causal_relational', 'science_physical'] risks=[] flags=[] hard=[]
SOURCE: These reductions are possible because, while delivery vehicles have higher rates of greenhouse gas emissions than private light-duty vehicles, the routing of delivery vehicles to customers is far more efficient than those customers traveling independently.
REWRITE: Though delivery vehicles emit more greenhouse gas than private cars, their efficient routing to customers makes these reductions possible.

### 46. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_019309
Metrics: source_words=41 rewrite_words=28 ratio=0.6829268292682927 recall=0.6923076923076923 overlap=0.6666666666666666 entity=0.8333333333333334 number=1.0 domains=['causal_relational', 'institutions_society'] risks=[] flags=[] hard=[]
Entities: source=['ARV', 'Gowanus Canal', 'New York University', 'Oded Nov', 'Polytechnic Institute', 'The Brooklyn Atlantis'] missing=['ARV'] new=[] | numbers source=[] output=[]
SOURCE: The Brooklyn Atlantis project, spearheaded by Oded Nov, assistant professor at the department of technology management and innovation at the Polytechnic Institute of New York University, has created an aquatic robotic vehicle (ARV) that collects environmental data on the Gowanus Canal.
REWRITE: Oded Nov, a Polytechnic Institute of New York University assistant professor, created an aquatic robotic vehicle for the Brooklyn Atlantis project to collect data on the Gowanus Canal.

### 47. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_014983
Metrics: source_words=28 rewrite_words=14 ratio=0.5 recall=0.5714285714285714 overlap=0.5 entity=1.0 number=1.0 domains=['causal_relational', 'geography_places'] risks=[] flags=[] hard=[]
Entities: source=['America'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: In America, ambitious people were willing to join the "best considers community" because they would join a the business circle of the community which means: credit and partnership.
REWRITE: In America, ambitious people joined the best community's business circle for credit and partnership.

### 48. compact_relation_causal_roles | frontier_consolidation_fwcompact_medium_055365
Metrics: source_words=33 rewrite_words=19 ratio=0.5757575757575758 recall=0.3076923076923077 overlap=0.2222222222222222 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
Entities: source=['Amina'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: As the reality of life with her new husband drew closer, it became clear to Amina that she would not be allowed by her husband to study, work or achieve her own goals.
REWRITE: As Amina's new marriage approached, she realized her husband would not let her study, work, or achieve her goals.

### 49. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_058648
Metrics: source_words=30 rewrite_words=15 ratio=0.5 recall=0.5 overlap=0.4444444444444444 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Despite the vast and continuing challenges facing independent media worldwide, there are signs in many countries that progress is being made, with inclusive media policies being developed or in prospect.
REWRITE: Despite global challenges, many countries show progress with inclusive media policies being developed or planned.

### 50. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_010396
Metrics: source_words=21 rewrite_words=15 ratio=0.7142857142857143 recall=0.5 overlap=0.35714285714285715 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: This often leads to vulnerability and downing leftovers, bread, cookies and more which are not going to do you any good.
REWRITE: This often causes vulnerability and eating leftovers, bread, cookies, and more that won't help you.

### 51. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_053956
Metrics: source_words=22 rewrite_words=12 ratio=0.5454545454545454 recall=0.46153846153846156 overlap=0.42857142857142855 entity=1.0 number=1.0 domains=['quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['Allies', 'Holland', 'The German'] missing=[] new=[] | numbers source=['10th', '1940'] output=['10th', '1940']
SOURCE: The German armoured divisions were finally unleashed on May 10th 1940, making a diversionary attack through Holland to draw the Allies north.
REWRITE: German divisions attacked Holland on May 10th 1940 to draw Allies north.

### 52. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_005339
Metrics: source_words=21 rewrite_words=16 ratio=0.7619047619047619 recall=0.75 overlap=0.6923076923076923 entity=1.0 number=1.0 domains=['media_culture'] risks=[] flags=[] hard=[]
Entities: source=['Vogue'] missing=[] new=[] | numbers source=['1939'] output=['1939']
SOURCE: In 1939 Vogue published their first fashion feature picturing women in trousers The blackout was an unlikely source of fashion inspiration.
REWRITE: In 1939 Vogue published their first fashion feature picturing women in trousers, inspired by the blackout.

### 53. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_048103
Metrics: source_words=23 rewrite_words=15 ratio=0.6521739130434783 recall=0.7142857142857143 overlap=0.625 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Also, with electronic sensors, the direct sun can sometimes cause "blooming" problems where the sensor is overloaded and it won't record properly afterward.
REWRITE: Direct sun can overload electronic sensors, causing blooming problems where they won't record properly afterward.

### 54. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_058101
Metrics: source_words=19 rewrite_words=10 ratio=0.5263157894736842 recall=0.7 overlap=0.5833333333333334 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: On the first day of spring (the spring equinox), they would give each other eggs that were dyed red.
REWRITE: On spring's first day, they gave each other red-dyed eggs.

### 55. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_005720
Metrics: source_words=41 rewrite_words=26 ratio=0.6341463414634146 recall=0.5882352941176471 overlap=0.4 entity=1.0 number=1.0 domains=[] risks=['heading_dash_chain'] flags=['source_risk_heading_dash_chain'] hard=[]
SOURCE: This calculator is not as extensive as the one above, but if your expenses are limited to the basics - rent or mortgage, one or two credit cards, car loan, entertainment, etc - then this calculator is useful for quick results.
REWRITE: This calculator is less extensive than the prior one, yet it yields quick results for basic expenses like rent, mortgages, credit cards, car loans, and entertainment.

### 56. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_009633
Metrics: source_words=15 rewrite_words=7 ratio=0.4666666666666667 recall=0.8333333333333334 overlap=0.7142857142857143 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: If the heart beat is a weak one, it is reflected in a small pulsation.
REWRITE: A weak heart beat causes small pulsation.

### 57. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_050556
Metrics: source_words=12 rewrite_words=9 ratio=0.75 recall=0.8333333333333334 overlap=0.8333333333333334 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: This means that they are not affected by any unprecedented spring frost.
REWRITE: They are not affected by any unprecedented spring frost.

### 58. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_023435
Metrics: source_words=16 rewrite_words=11 ratio=0.6875 recall=0.625 overlap=0.5555555555555556 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: None of these changes occurs in the thalamic electrode at the time that consciousness is lost.
REWRITE: No changes occur in the thalamic electrode when consciousness is lost.

### 59. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_014328
Metrics: source_words=23 rewrite_words=15 ratio=0.6521739130434783 recall=0.7857142857142857 overlap=0.6875 entity=1.0 number=1.0 domains=[] risks=['long_parenthetical'] flags=['source_risk_long_parenthetical'] hard=[]
SOURCE: The blocks were placed in a sand bedding vertically (rather than horizontally as in medieval corduroy roads) creating a kind of wood cobble.
REWRITE: Blocks were placed vertically in sand bedding, creating wood cobble unlike medieval horizontal corduroy roads.

### 60. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_022688
Metrics: source_words=34 rewrite_words=18 ratio=0.5294117647058824 recall=0.5454545454545454 overlap=0.48 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Earth'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: In fact, if further studies indicate that supercritical carbon dioxide can play the role of a viable bio-organic solvent, the implications will stretch far out into anywhere that a super-Earth or gas-giant is found.
REWRITE: If studies show supercritical carbon dioxide works as a bio-organic solvent, implications reach any super-Earth or gas-giant found.

### 61. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_029654
Metrics: source_words=29 rewrite_words=17 ratio=0.5862068965517241 recall=0.5454545454545454 overlap=0.42857142857142855 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
Entities: source=['GUI', 'Ideally'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Haskell not so much. - Ideally it should come with a simple GUI that allows me to set the value of N and D and see the result visually.
REWRITE: Ideally Haskell should include a simple GUI to set N and D values and show results visually.

### 62. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_014876
Metrics: source_words=19 rewrite_words=11 ratio=0.5789473684210527 recall=0.4 overlap=0.3076923076923077 entity=1.0 number=1.0 domains=['geography_places'] risks=['deictic_time'] flags=['source_risk_deictic_time'] hard=[]
SOURCE: Currently, in many technologically progressive societies, the population is not only living longer, but is also offering fewer children.
REWRITE: In many advanced societies, people live longer but have fewer children.

### 63. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_008913
Metrics: source_words=20 rewrite_words=10 ratio=0.5 recall=0.4166666666666667 overlap=0.35714285714285715 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Vollrath, who speculated that spider silk might generate a similar revolutionary shift in the emergent field of biomaterials, was snared.
REWRITE: Vollrath, who speculated spider silk could revolutionize biomaterials, was caught.

### 64. compact_negation_modality_conditionals | frontier_consolidation_fwcompact_medium_059685
Metrics: source_words=25 rewrite_words=14 ratio=0.56 recall=0.5 overlap=0.375 entity=1.0 number=1.0 domains=['quant_numeric'] risks=[] flags=[] hard=[]
SOURCE: Recognizing the methods teens may use in an attempt to lose weight may help the informed parent to determine if his teen is at risk.
REWRITE: Knowing teen weight-loss methods helps informed parents determine if their teen is at risk.

### 65. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_033735
Metrics: source_words=20 rewrite_words=10 ratio=0.5 recall=0.3 overlap=0.23076923076923078 entity=1.0 number=1.0 domains=[] risks=['long_parenthetical'] flags=['source_risk_long_parenthetical'] hard=[]
SOURCE: The default value is zero. - -l, --length length The number of bytes to discard (counting from the starting point).
REWRITE: Default is zero; -l discards specified bytes from the start.

### 66. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_054356
Metrics: source_words=19 rewrite_words=10 ratio=0.5263157894736842 recall=0.5714285714285714 overlap=0.4444444444444444 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=[] missing=[] new=['counting'] | numbers source=['16'] output=['16']
SOURCE: That is how long it would take for it to count to 16 trillion, running at its peak capacity.
REWRITE: Counting to 16 trillion at peak capacity takes that long.

### 67. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_001143
Metrics: source_words=31 rewrite_words=19 ratio=0.6129032258064516 recall=0.6 overlap=0.47368421052631576 entity=1.0 number=1.0 domains=['institutions_society', 'quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=[] missing=[] new=[] | numbers source=['$5000'] output=['$5000']
SOURCE: Only producers who sell less than $5,000 a year in organic foods are exempt from this certification; however, they must follow the same government standards to label their foods as organic.
REWRITE: Only producers selling under $5,000 yearly in organic foods are exempt from certification but must follow government labeling standards.

### 68. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_032517
Metrics: source_words=28 rewrite_words=18 ratio=0.6428571428571429 recall=0.3684210526315789 overlap=0.3181818181818182 entity=1.0 number=1.0 domains=['causal_relational', 'quant_numeric'] risks=[] flags=[] hard=[]
Entities: source=['March'] missing=[] new=[] | numbers source=['2011', '3'] output=['2011', '3']
SOURCE: Thursday, 3 March 2011 A newly identified mechanism that helps explain why some tumours are resistant to specific anti-cancer drugs could help improve patient care and drug design.
REWRITE: On March 3, 2011, a new mechanism explaining tumour drug resistance could improve patient care and drug design.

### 69. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_056802
Metrics: source_words=28 rewrite_words=12 ratio=0.42857142857142855 recall=0.4 overlap=0.35294117647058826 entity=1.0 number=1.0 domains=['institutions_society'] risks=['long_parenthetical'] flags=['source_risk_long_parenthetical'] hard=[]
Entities: source=['Government', 'People'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Authority (eagle's wings) People with authority is shown under the wings of an eagle Government (men sitting down around a table on a stand under the picto authority).
REWRITE: People with authority sit under an eagle's wings in the Government pictogram.

### 70. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_011633
Metrics: source_words=37 rewrite_words=22 ratio=0.5945945945945946 recall=0.47368421052631576 overlap=0.36 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['HF', 'MI', 'VADs'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Ventricular assist devices (VADs) require surgical implantation and are indicated for patients with severe HF after cardiac surgery, in patients who have intractable cardiogenic shock after acute MI, and in patients who deteriorate while awaiting cardiac .
REWRITE: VADs need surgery for severe heart failure after cardiac surgery, intractable shock after acute heart attack, or deterioration while awaiting heart transplant.

### 71. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_060093
Metrics: source_words=29 rewrite_words=17 ratio=0.5862068965517241 recall=0.6 overlap=0.5 entity=1.0 number=1.0 domains=['science_physical'] risks=[] flags=[] hard=[]
Entities: source=['DNA'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Mapping the genome is becoming a standard tool of medical care, with new and cheaper DNA sequencing machines helping doctors unravel inherited diseases that once took years to diagnose.
REWRITE: Cheaper DNA machines now help doctors quickly diagnose inherited diseases, making genome mapping a standard medical tool.

### 72. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_002325
Metrics: source_words=12 rewrite_words=8 ratio=0.6666666666666666 recall=1.0 overlap=0.8333333333333334 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Most of those dust bunnies in your house are from dead skin.
REWRITE: Most house dust bunnies come from dead skin.

### 73. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_054925
Metrics: source_words=18 rewrite_words=11 ratio=0.6111111111111112 recall=0.7777777777777778 overlap=0.7 entity=1.0 number=1.0 domains=[] risks=['probability_hedge'] flags=['source_risk_probability_hedge'] hard=[]
Entities: source=['Dover England'] missing=[] new=['best'] | numbers source=[] output=[]
SOURCE: Perhaps the best known are the flint nodules found in chalk deposits such as those at Dover, England.
REWRITE: Best known are flint nodules in chalk deposits like Dover, England.

### 74. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_033797
Metrics: source_words=15 rewrite_words=9 ratio=0.6 recall=0.3333333333333333 overlap=0.2 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
SOURCE: If they became more successful, they would build up and out and improve their buildings.
REWRITE: Success would cause them to expand and improve buildings.

### 75. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_009002
Metrics: source_words=17 rewrite_words=9 ratio=0.5294117647058824 recall=0.3333333333333333 overlap=0.23076923076923078 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Some individuals might initially notice urine that is darker in color than usual with a pungent odor.
REWRITE: Some people may first notice darker, pungently smelling urine.

### 76. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_025031
Metrics: source_words=18 rewrite_words=10 ratio=0.5555555555555556 recall=0.5 overlap=0.5 entity=1.0 number=1.0 domains=['causal_relational', 'people_history'] risks=[] flags=[] hard=[]
Entities: source=['Gurs'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: How many of them died due to the conditions they were subjected to at Gurs is not known.
REWRITE: How many died at Gurs from conditions is not known.

### 77. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_050622
Metrics: source_words=39 rewrite_words=21 ratio=0.5384615384615384 recall=0.5263157894736842 overlap=0.4166666666666667 entity=1.0 number=1.0 domains=['quant_numeric'] risks=['deictic_time'] flags=['source_risk_deictic_time'] hard=[]
SOURCE: There are many different types of insulin on the market today and, as is the case with oral medications, it may take several different insulin types in order to find the correct balance that will insure good glucose levels.
REWRITE: Many insulin types exist today; like oral meds, several may be needed to find the right balance for good glucose levels.

### 78. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_052032
Metrics: source_words=20 rewrite_words=14 ratio=0.7 recall=0.7272727272727273 overlap=0.6153846153846154 entity=1.0 number=1.0 domains=['geography_places'] risks=[] flags=[] hard=[]
SOURCE: A common man can never understand their language, emotions, mental state and dilemmas in environment in which they are living.
REWRITE: Common people cannot understand their language, emotions, mental state, or dilemmas in their environment.

### 79. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_055211
Metrics: source_words=20 rewrite_words=12 ratio=0.6 recall=0.875 overlap=0.7777777777777778 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: The key to getting the help you need, when you need it, is recognizing the signs of a heart attack.
REWRITE: Recognizing heart attack signs is key to getting needed help when needed.

### 80. compact_coreference_apposition_roles | frontier_consolidation_fwcompact_medium_053244
Metrics: source_words=21 rewrite_words=15 ratio=0.7142857142857143 recall=0.8333333333333334 overlap=0.6666666666666666 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Since BaseObjectData'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Since BaseObjectData has no named attributes, an iterator with m_current_class == BASE_OBJECT_NO is considered to be the end of the map.
REWRITE: Since BaseObjectData lacks named attributes, an iterator with m_current_class equal to BASE_OBJECT_NO ends the map.

### 81. compact_random_train_rows | frontier_consolidation_fwcompact_medium_035369
Metrics: source_words=21 rewrite_words=13 ratio=0.6190476190476191 recall=0.7272727272727273 overlap=0.6153846153846154 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
Entities: source=['FCV'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Some symptoms of FCV are mouth ulcers, sneezing, eye discharge and limping due to the formation of ulcers on the paws.
REWRITE: FCV symptoms include mouth ulcers, sneezing, eye discharge, and limping from paw ulcers.

### 82. compact_random_train_rows | frontier_consolidation_fwcompact_medium_014819
Metrics: source_words=13 rewrite_words=8 ratio=0.6153846153846154 recall=0.8333333333333334 overlap=0.8333333333333334 entity=1.0 number=1.0 domains=[] risks=['deictic_time'] flags=['source_risk_deictic_time'] hard=[]
SOURCE: These ingredients are used in many high quality products on the shelves today.
REWRITE: Many high quality products use these ingredients today.

### 83. compact_random_train_rows | frontier_consolidation_fwcompact_medium_053704
Metrics: source_words=14 rewrite_words=9 ratio=0.6428571428571429 recall=0.75 overlap=0.75 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: Eating correctly and providing your body with right levels of nutrition is also good.
REWRITE: Eating correctly and providing right nutrition is also good.

### 84. compact_random_train_rows | frontier_consolidation_fwcompact_medium_056095
Metrics: source_words=31 rewrite_words=23 ratio=0.7419354838709677 recall=0.8 overlap=0.7058823529411765 entity=1.0 number=1.0 domains=['institutions_society'] risks=[] flags=[] hard=[]
Entities: source=['Windows'] missing=[] new=[] | numbers source=['140'] output=['140']
SOURCE: If you are launching a program that uses the Vcruntime140.dll, Windows will scan your system folders for this dll file as it is necessary for the corresponding application to open successfully.
REWRITE: If launching a program using Vcruntime140.dll, Windows scans system folders for this file because it is necessary for the application to open successfully.

### 85. compact_random_train_rows | frontier_consolidation_fwcompact_medium_034828
Metrics: source_words=13 rewrite_words=9 ratio=0.6923076923076923 recall=0.8571428571428571 overlap=0.8571428571428571 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['SoCs'] missing=[] new=['tiny'] | numbers source=[] output=[]
SOURCE: This is of critical importance for mobile SoCs where power budgets are tiny.
REWRITE: Tiny power budgets make this critical for mobile SoCs.

### 86. compact_random_train_rows | frontier_consolidation_fwcompact_medium_029381
Metrics: source_words=35 rewrite_words=23 ratio=0.6571428571428571 recall=0.5555555555555556 overlap=0.4166666666666667 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Americans', 'CDC', 'Healthy People'] missing=[] new=[] | numbers source=['2020', '40%'] output=['2020', '40%']
SOURCE: On the downside, the new data still falls short of the national and regional initiative for the Healthy People 2020 goal to have only 40% of Americans with uncontrolled blood pressure, according to CDC researchers.
REWRITE: CDC researchers say new data still misses Healthy People 2020 goals to limit uncontrolled blood pressure to 40% of Americans nationally and regionally.

### 87. compact_random_train_rows | frontier_consolidation_fwcompact_medium_000704
Metrics: source_words=22 rewrite_words=13 ratio=0.5909090909090909 recall=0.6428571428571429 overlap=0.5625 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: In fact, so well-developed are its parental instincts that pairs sometimes take-over the broods of unrelated birds with whom they are housed!
REWRITE: Their strong parental instincts cause pairs to sometimes take over unrelated birds' broods.

### 88. compact_random_train_rows | frontier_consolidation_fwcompact_medium_037045
Metrics: source_words=12 rewrite_words=7 ratio=0.5833333333333334 recall=0.8333333333333334 overlap=0.8333333333333334 entity=1.0 number=1.0 domains=['geography_places'] risks=[] flags=[] hard=[]
SOURCE: These types of plants are huge capital investments and are also sophisticated.
REWRITE: These plants are huge, sophisticated capital investments.

### 89. compact_random_train_rows | frontier_consolidation_fwcompact_medium_042714
Metrics: source_words=12 rewrite_words=6 ratio=0.5 recall=0.5555555555555556 overlap=0.5555555555555556 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Avoid Cigarette Smoke Smoking'] missing=[] new=['asthma'] | numbers source=[] output=[]
SOURCE: Avoid Cigarette Smoke Smoking is not healthy for anyone, especially asthma patients.
REWRITE: Asthma patients should avoid cigarette smoke.

### 90. compact_random_train_rows | frontier_consolidation_fwcompact_medium_003069
Metrics: source_words=28 rewrite_words=16 ratio=0.5714285714285714 recall=0.5 overlap=0.42105263157894735 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['Jacksonville', 'Union', 'William Cushing'] missing=[] new=[] | numbers source=['1862'] output=['1862']
SOURCE: In 1862, Jacksonville was raided by Union troops under the command of William Cushing, who succeeded in raising the Union flag over the courthouse and taking several prizes.
REWRITE: In 1862, Union troops led by William Cushing raided Jacksonville, raised the flag, and took prizes.

### 91. compact_random_train_rows | frontier_consolidation_fwcompact_medium_011709
Metrics: source_words=44 rewrite_words=22 ratio=0.5 recall=0.6363636363636364 overlap=0.6086956521739131 entity=1.0 number=1.0 domains=['causal_relational', 'science_physical'] risks=['apostle_or_title_apposition'] flags=['source_risk_apostle_or_title_apposition'] hard=[]
Entities: source=['Ekgmowechashala', 'John Day Formation', 'Josh Samuels'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: New material from the John Day Formation, described by a team led by Josh Samuels, muddies the water even further, suggesting that Ekgmowechashala belonged to a primate group known as adapiforms, not to the omomyids, the group to which it had previously been assigned.
REWRITE: New material from the John Day Formation, described by Josh Samuels' team, suggests Ekgmowechashala belonged to adapiforms, not omomyids, as previously assigned.

### 92. compact_random_train_rows | frontier_consolidation_fwcompact_medium_003584
Metrics: source_words=20 rewrite_words=13 ratio=0.65 recall=0.7272727272727273 overlap=0.6153846153846154 entity=1.0 number=1.0 domains=['institutions_society'] risks=[] flags=[] hard=[]
Entities: source=['The Gombos Company'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: In fact, The Gombos Company purchases alfalfa directly from farmers, and the free market is the basis of alfalfa prices.
REWRITE: The Gombos Company buys alfalfa directly from farmers, with free market setting prices.

### 93. compact_random_train_rows | frontier_consolidation_fwcompact_medium_056197
Metrics: source_words=29 rewrite_words=17 ratio=0.5862068965517241 recall=0.47058823529411764 overlap=0.36363636363636365 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=['East', 'NHS', 'NHS Midlands'] missing=[] new=[] | numbers source=[] output=[]
SOURCE: On the back of the study results, NHS Midlands & East has instigated a regional training programme to standardise service delivery and ensure therapists are competent at phone contacts.
REWRITE: NHS Midlands & East started regional training to standardize services and ensure therapist competence in phone contacts.

### 94. compact_random_train_rows | frontier_consolidation_fwcompact_medium_005206
Metrics: source_words=29 rewrite_words=14 ratio=0.4827586206896552 recall=0.4 overlap=0.2857142857142857 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
Entities: source=["Rocket Lab's"] missing=[] new=[] | numbers source=[] output=[]
SOURCE: Components such as the rocket nozzle and combustion chamber are all manufactured from Rocket Lab's own composite materials which are a fraction of the weight of traditional metal components.
REWRITE: Rocket Lab's lighter composite materials replace heavy metal parts in nozzles and combustion chambers.

### 95. compact_random_train_rows | frontier_consolidation_fwcompact_medium_027396
Metrics: source_words=17 rewrite_words=12 ratio=0.7058823529411765 recall=0.375 overlap=0.2727272727272727 entity=1.0 number=1.0 domains=[] risks=[] flags=[] hard=[]
SOURCE: In other words, your vitamin C requirement is dependent upon how much meat you do not eat.
REWRITE: Your vitamin C need depends on how much meat you don't eat.

### 96. compact_random_train_rows | frontier_consolidation_fwcompact_medium_016000
Metrics: source_words=42 rewrite_words=22 ratio=0.5238095238095238 recall=0.65 overlap=0.5416666666666666 entity=1.0 number=1.0 domains=['causal_relational'] risks=[] flags=[] hard=[]
SOURCE: Portacaval anastomosis, by contrast, is an anastomosis between a vein of the portal circulation and a vein of the systemic circulation, which allows blood to bypass the liver in patients with portal hypertension, often resulting in hemorrhoids, esophageal varices, or caput medusae.
REWRITE: Portacaval anastomosis connects portal and systemic veins, bypassing the liver in portal hypertension patients, often causing hemorrhoids, esophageal varices, or caput medusae.
