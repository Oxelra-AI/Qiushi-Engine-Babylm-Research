# earlier analysis compact-view semantic review packet

Purpose: distinguish faithful shortening from information-losing summary and altered source meaning before any compact text is admitted to learner training.

## Pilot-level summary

- n_records: 512
- current_number_length_pass: 485
- auto_low_risk: 184
- needs_semantic_review: 316
- known_altered: 2
- auto_low_risk_saved_words: 2928
- current_number_length_saved_words: 9723

## Review instructions

For each item, compare the compact rewrite primarily with the original source text, using the current rewrite only as an inherited second view. Label it as one of: faithful_shortening, supported_summary_with_lost_detail, altered_meaning, or unclear. Pay special attention to entity roles, numbers, direction/order, conditional force, modality, negation, temporal relations, and who did what.

## 1. rw2s0_029078 · source=open_subtitles · risk=190 · saved=28

Risk tags: known_altered_claim, navigation_sequence_changed, ordered_navigation, role_reference_weakened

Risk reasons: navigation order reversed near Goddess Mother temple and idol makers | direction sequence original=['straight', 'left', 'up', 'left', 'right', 'straight', 'left', 'up', 'opposite', 'straight', 'up'] compact=['straight', 'left', 'left', 'right', 'left', 'opposite', 'straight'] | role/pronoun preservation 0.00

Original:
Go straight and turn left, further up there's a street, turn left again, there's temple of Goddess Mother, turn right there, further straight you'll see a place where they make idols, turn left there, further up you'll see a barber shop, take the street opposite to it, straight up it's Afzal Gunj!

Current inherited rewrite:
Go straight and turn left until you reach a street, then turn left again to find the temple of Goddess Mother, turn right and proceed straight to where they make idols, turn left and continue until you see a barber shop, take the street opposite it, and go straight up to reach Afzal Gunj!

Compact candidate:
Go straight, turn left, then left again at Goddess Mother temple, right at idol makers, left at barber shop, opposite street leads straight to Afzal Gunj.

## 2. rw_035551 · source=open_subtitles · risk=187 · saved=19

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, known_altered_claim, modal_lost_from_both_references, modality_force_weakened

Risk reasons: conditional self-attribution converted toward a direct assertion | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0

Original:
To bad Amaru I didn't go crazy and nothing is wrong with me If there is something wrong, that would be me in the past What are you doing Amaru?

Current inherited rewrite:
Amaru, I feel terrible that I didn't lose my mind because nothing is currently wrong with me; if there were a problem, it would stem from who I was in the past. What are you doing, Amaru?

Compact candidate:
Amaru, I didn't go crazy; nothing wrong with me, but past me was wrong. What are you doing?

## 3. rw2s1_005144 · source=childes · risk=263 · saved=32

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 13 words, original 49 words | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 3, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
*INV: and so Lori [EX2] and I were trying to get from where we got all these toys at to over here and I I didn't wanna be late because I didn't wanna keep you guys waiting so I was driving probably a little faster then I should have.

Current inherited rewrite:
*INV: and so Lori [EX2] and I were attempting to travel from where we acquired all these toys to over here, and since I didn't want to be late or keep you guys waiting, I was driving probably a little faster then I should have.

Compact candidate:
Lori and I drove fast to avoid being late and keeping you waiting.

## 4. rw2s0_006082 · source=gutenberg · risk=239 · saved=36

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 7 words, original 31 words | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 4, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.10 | compact-current Jaccard 0.10

Original:
"You can't expect a poor music-teacher to break away from her work at this season?" "But I did not know you were a music-teacher." "No, I suppose not," answered Rose, smiling.

Current inherited rewrite:
But I did not know you were a music-teacher," said the speaker, followed by Rose, who smiled and replied, "No, I suppose not," in response to the question, "You can't expect a poor music-teacher to break away from her work at this season?

Compact candidate:
Rose smiled, answering she was a music-teacher.

## 5. rw2s1_006744 · source=gutenberg · risk=237 · saved=44

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 12 words, original 54 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.14 | compact-original Jaccard 0.19

Original:
"Won't take me long when I find the people on shore--and about five minutes will fix that engine when I get back here again." He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return.

Current inherited rewrite:
He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return, having declared that finding the people on shore wouldn't take long and that about five minutes would fix the engine when he got back here again.

Compact candidate:
He rowed to shore, fixed the engine in five minutes, and returned.

## 6. rw2s1_021822 · source=open_subtitles · risk=184 · saved=20

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 3, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 1 | role/pronoun preservation 0.17

Original:
She has lost the capacity to record this memory, store it, because she can't store it, and that's what we call learning and working memory because, if I don't remember what I've been told, I can't do anything.

Current inherited rewrite:
She has lost the capacity to record and store this memory because she cannot store it, which is what we call learning and working memory, because if I don't remember what I've been told, I can't do anything.

Compact candidate:
She lost capacity to record and store memory, which is learning and working memory, so forgetting prevents action.

## 7. rw2s1_011130 · source=simple_wiki · risk=174 · saved=30

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 4, compact count 0 | modality_force marker preservation 0.00; original count 4, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.20

Original:
Semple said: "I thought I might sell one or two, but the website itself would be almost like a piece of performance art, and the pink jar would be like an artwork." This means he did not think many people would care that he was not selling his pink paint to Anish Kapoor.

Current inherited rewrite:
Semple said: "I thought I might sell one or two, but the website itself would be almost like a piece of performance art, and the pink jar would be like an artwork," which indicates he did not believe many people would care that he was not selling his pink paint to Anish Kapoor.

Compact candidate:
Semple said the website was performance art and the pink jar an artwork, implying few cared he wasn't selling paint to Anish Kapoor.

## 8. rw2s1_010040 · source=gutenberg · risk=174 · saved=21

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.29

Original:
He could not feel happy till he did so, and even before he had said anything she knew that the little tug to her sleeve and the whispered "Mother, I want to speak to you," was coming.

Current inherited rewrite:
He could not feel happy until he did so, and even before he had said anything she knew that the little tug to her sleeve and the whispered "Mother, I want to speak to you," was coming.

Compact candidate:
He felt happy only after doing so, and she knew the tug and whisper were coming.

## 9. rw_012099 · source=gutenberg · risk=169 · saved=29

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, negation_force_weakened, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
It is awfully plain--pease pudding and herrings mostly; but I don't mind that if only you'd pay me ten shillings a week and let me come to you every day." "You are the most audacious girl!

Current inherited rewrite:
This fare is dreadfully simple, consisting mainly of pease pudding and herrings, yet I would accept it gladly if you'd hire me for ten shillings a week to work daily at your place," she declared, to which he replied, "You are the most audacious girl!

Compact candidate:
She wants ten shillings weekly to cook pease pudding and herrings daily, calling the girl audacious.

## 10. rw2s1_013165 · source=gutenberg · risk=165 · saved=28

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, role_reference_weakened, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 2 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.29

Original:
She could hear them long after they had vanished from sight, crying out in their fear, plunging among the trees, but gradually the sounds grew fainter, and Bessie, sure that they need fear no more disturbance from Jake Hoover and his brave companions, set out on her return to the camp.

Current inherited rewrite:
Even after they had vanished from sight, she could hear them crying out in their fear as they plunged among the trees, but gradually the sounds grew fainter, and Bessie, sure that they need fear no more disturbance from Jake Hoover and his brave companions, set out on her return to the camp.

Compact candidate:
She heard them crying out in fear among trees until sounds faded, then Bessie returned to camp sure Jake Hoover and companions posed no threat.

## 11. rw2s1_021731 · source=gutenberg · risk=165 · saved=28

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact/original length ratio 0.33 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.11 | compact-current Jaccard 0.11

Original:
Even then, however, it was not good form for a boy to be greatly interested in them; and he had to conceal any little fancy he had about this girl or that unless he wanted to be considered soft by the other fellows.

Current inherited rewrite:
Even then, however, it was not proper for a boy to be greatly interested in them; and he had to conceal any little fancy he had about this girl or that unless he wanted to be considered soft by the other fellows.

Compact candidate:
Even then, boys hiding interest in girls avoided being seen as soft by peers.

## 12. rw2s1_015912 · source=gutenberg · risk=160 · saved=39

Risk tags: conditional_force_partly_changed, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 10 words, original 40 words | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.15 | compact-current Jaccard 0.16

Original:
For a moment the horsemen were too astonished to move; then, recovering from their surprise, they lowered their murderous-looking lances, and would undoubtedly have run all three prisoners through, had not another officer ridden into the circle at that moment.

Current inherited rewrite:
For a brief instant the horsemen were so stunned by the sight that they could not move, but once they recovered from their shock, they lowered their menacing lances and would certainly have pierced all three prisoners had another officer not ridden into the circle at that very moment.

Compact candidate:
Horsemen lowered lances, would kill prisoners, but officer rode in.

## 13. rw2s1_011013 · source=bnc_spoken · risk=160 · saved=24

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, ordered_navigation

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0

Original:
Also more roadworks may well affect your journey to the north of Enstone on the A43, that's because of some resurfacing work, and if you're travelling through on the 417 at East Hendred er near the Hare public house, there's more temporary signals in operation too this evening.

Current inherited rewrite:
Also, your journey to the north of Enstone on the A43 may well be affected by more roadworks due to some resurfacing work, and if you're travelling through on the 417 at East Hendred er near the Hare public house, there's more temporary signals in operation too this evening.

Compact candidate:
Roadworks on the A43 north of Enstone and temporary signals on the 417 at East Hendred near the Hare public house affect journeys this evening.

## 14. rw2s1_009951 · source=open_subtitles · risk=156 · saved=23

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary, temporal_order_partly_changed

Risk reasons: compact/original length ratio 0.39 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
The world's eyes are on the UN summit because the scientific community made it clear that, if a deal on reducing emissions isn't struck this year, then next year may be too late.

Current inherited rewrite:
The scientific community has made it clear that if a deal on reducing emissions isn't struck this year, then next year may be too late, which is why the world's eyes are on the UN summit.

Compact candidate:
UN summit eyes on emissions deal this year or next year too late.

## 15. rw2s1_014406 · source=bnc_spoken · risk=155 · saved=26

Risk tags: modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, ordered_navigation, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.37 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 2, compact count 0

Original:
Again just two lanes open can slow things and a quick glance elsewhere well no real troubles reported just a look at the A One Stanford still the roadworks on the go there both north and south will slow things down.

Current inherited rewrite:
Again, just two lanes open can slow things, and while a quick glance elsewhere reveals no real troubles reported just a look at the A One Stanford, the roadworks on the go there both north and south will slow things down.

Compact candidate:
Two lanes open slow things; A One Stanford roadworks north and south also slow traffic.

## 16. rw2s0_006043 · source=childes · risk=155 · saved=23

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.40 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
*MOT: Cinderella gasped when she saw that she was no longer in rags but in a gorgeous dress of silk and satin embroidered with the sparkling jewels and that on her feet were two dainty glass slippers the prettiest in the world.

Current inherited rewrite:
*MOT: Cinderella gasped upon realizing she was no longer clad in rags but instead wore a magnificent silk and satin dress adorned with sparkling jewels, while on her feet rested two exquisite glass slippers, the most beautiful in the world.

Compact candidate:
Cinderella gasped seeing she wore a gorgeous silk satin dress with sparkling jewels and dainty glass slippers.

## 17. rw2s0_005098 · source=gutenberg · risk=154 · saved=25

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
However, when the trunks stood ready packed, and Edna said good night for the last time before undertaking the journey, she held her mother very tightly around the neck and whispered: "I wish you were going too, Mother." "That can't be, darling," said her mother.

Current inherited rewrite:
However, once the trunks were packed and ready, and Edna said good night for the final time before setting out on the journey, she clung tightly to her mother's neck and whispered: "I wish you were going too, Mother," to which her mother replied, "That can't be, darling."

Compact candidate:
When trunks were packed and Edna said good night, she hugged her mother tightly and whispered, "I wish you were going too, Mother."

## 18. rw2s0_032763 · source=gutenberg · risk=152 · saved=24

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.33; original count 3, compact count 1 | modality_force marker preservation 0.33; original count 3, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 1 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.33

Original:
Chapman, like Marlowe, could write the "mighty line." Jonson had rare lyric power; his verses sing, as witness the wonderful "Do but look on her eyes," which Francis Bacon could no more have written than he could have jumped over the moon.

Current inherited rewrite:
Chapman, similar to Marlowe, was capable of composing the "mighty line," while Jonson possessed rare lyric power, as his verses sing, exemplified by the wonderful "Do but look on her eyes," a piece Francis Bacon could no more have written than he could have jumped over the moon.

Compact candidate:
Chapman, like Marlowe, wrote the "mighty line." Jonson's verses sing, as in "Do but look on her eyes," which Francis Bacon could not write.

## 19. rw2s1_026585 · source=gutenberg · risk=150 · saved=20

Risk tags: causal_or_contrast_force_partly_changed, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modality_force_weakened, negation_force_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.33; original count 3, compact count 1 | negation_force marker preservation 0.33; original count 3, compact count 1 | comparison_relation marker preservation 0.00; original count 2, compact count 1

Original:
In arguing about what Shakespeare "must have" or "could not have" known, we must not forget that at no time or place since history began has human thought fermented more briskly than in London while he was living there.

Current inherited rewrite:
In debating what Shakespeare "must have" or "could not have" known, we must not forget that at no time or place since history began has human thought fermented more briskly than in London while he was living there.

Compact candidate:
Arguing Shakespeare's knowledge, we must not forget human thought fermented most briskly in London while he lived there.

## 20. rw2s0_019852 · source=open_subtitles · risk=147 · saved=28

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, role_reference_weakened, strong_summary, very_low_source_overlap

Risk reasons: compact/original length ratio 0.30 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.17 | compact-original Jaccard 0.15

Original:
I went to Mass that Sunday, and the priest was giving his homily, and he said something about, "Well maybe, you know, "The students have something to say, "and we should start listening to them." Got up and walked out.

Current inherited rewrite:
I attended Mass that Sunday, and while the priest was delivering his homily, he remarked, "Well maybe, you know, 'The students have something to say,' and we should start listening to them," after which I stood up and walked out.

Compact candidate:
I attended Mass Sunday when the priest's homily prompted me to leave.

## 21. rw2s1_016261 · source=open_subtitles · risk=147 · saved=15

Risk tags: conditional_force_partly_changed, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, navigation_sequence_changed, ordered_navigation, role_reference_weakened

Risk reasons: direction sequence original=['left', 'down', 'down'] compact=['down'] | modality_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.20

Original:
Jon called me before he left and was like, "If you come down here, tell me my grandma died." The fake dead grandmother could easily go down as the dirtiest thing ever to be done in this game.

Current inherited rewrite:
Before departing, Jon called me and said, "If you come down here, tell me my grandma died," noting that the fake dead grandmother could easily go down as the dirtiest thing ever to be done in this game.

Compact candidate:
Jon called me before leaving, saying if I came down, tell him his fake dead grandma died, the dirtiest thing in this game.

## 22. rw2s0_002953 · source=gutenberg · risk=145 · saved=33

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.35 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
From that time, the cause received no check; the greatest towns in England began, one after another, to declare for the Prince; and he knew that it was all safe with him when the University of Oxford offered to melt down its plate, if he wanted any money.

Current inherited rewrite:
From that time onward, the cause faced no hindrance as the greatest towns in England began, one after another, to declare for the Prince, and he knew that it was all safe with him when the University of Oxford offered to melt down its plate if he wanted any money.

Compact candidate:
England towns declared for the Prince; Oxford University offered to melt down its plate for his money.

## 23. rw2s1_000855 · source=gutenberg · risk=143 · saved=35

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, role_reference_weakened, strong_summary, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact/original length ratio 0.34 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.18

Original:
My boy's elder brother at once accused him of tracing that bull, which he pretended to have copied; but their father insisted upon taking the child's word for it, though he must have known he was lying; and this gave my boy a far worse conscience than if his father had whipped him.

Current inherited rewrite:
My boy's elder brother immediately accused him of tracing that bull, which he pretended to have copied; yet their father insisted on taking the child's word for it, even though he must have known he was lying, and this gave my boy a far worse conscience than if his father had whipped him.

Compact candidate:
Father trusted child's word over elder brother's accusation about tracing bull, giving boy worse conscience than whipping would.

## 24. rw2s0_028146 · source=gutenberg · risk=142 · saved=16

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
It was blowing hard when the vessels sighted each other, and the captain of the tramp could not read the flags distinctly." "But where was the _Sea Witch_ when sighted, and whither bound?" "Liverpool to New York--a hundred and fifty miles out, twenty-four hours ago.

Current inherited rewrite:
When the vessels sighted each other amidst strong winds, the tramp's captain could not read the flags distinctly; in response to the inquiry about the _Sea Witch_'s location and destination at that moment, the answer was "Liverpool to New York--a hundred and fifty miles out, twenty-four hours ago."

Compact candidate:
It blew hard when vessels sighted each other; the tramp captain couldn't read flags. Where was the Sea Witch when sighted, bound Liverpool to New York, 150 miles out, 24 hours ago?

## 25. rw2s0_018664 · source=bnc_spoken · risk=140 · saved=25

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.39 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.11

Original:
Yeah yeah and I mean you can you you've got to use the negotiating technique of come on fella you know I'm gonna get I'm gonna get chewed over if I don't Oh yeah don't worry I do.

Current inherited rewrite:
Yeah yeah and I mean you have to employ the negotiating technique of saying "come on fella you know I'm gonna get I'm gonna get chewed over if I don't" because, as I said, "Oh yeah don't worry I do."

Compact candidate:
You must use the negotiating technique of come on fella to avoid being chewed over.

## 26. rw_034107 · source=bnc_spoken · risk=140 · saved=19

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
These show some decrease in surface waters but not as severe as the bio- limiting substances calcium, barium carbon would be examples of bio-intermediate substances.

Current inherited rewrite:
While surface waters have shown a decline, it is not as severe as that caused by bio-limiting substances such as calcium and barium, which would instead serve as examples of bio-intermediate substances.

Compact candidate:
These show surface water decrease, less severe than calcium, barium carbon bio-limiting substances.

## 27. rw2s1_030744 · source=simple_wiki · risk=139 · saved=27

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, number_changed_or_removed, ordered_navigation

Risk reasons: one or more source numbers not found in compact text | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
German names like “Adolf” are common in Namibia, in part because Namibia was a German colony called German South West Africa between 1884 and 1915 under the German Empire, and later a mandate of the Union of South Africa.

Current inherited rewrite:
German names such as "Adolf" are prevalent in Namibia, partly due to the territory's history as the German colony German South West Africa from 1884 to 1915 under the German Empire, followed by its status as a mandate of the Union of South Africa.

Compact candidate:
German names like "Adolf" are common in Namibia, a former German colony and later South African mandate.

## 28. rw2s1_000216 · source=bnc_spoken · risk=134 · saved=32

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, negation_force_weakened, negation_lost_from_both_references, ordered_navigation, strong_summary

Risk reasons: compact/original length ratio 0.38 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0

Original:
Gas Prices are going up in March by 7.5%, another Building Society, the Halifax has put up its interest rates, although first time buyers are being spared the increase, and the Chancellor, John Major says the gloomy news isn't over, because inflation isn't coming down as fast as he predicted.

Current inherited rewrite:
In March, gas prices are rising by 7.5%, while another Building Society, the Halifax, has increased its interest rates even though first time buyers are being spared the increase, and the Chancellor, John Major, states that the gloomy news isn't over because inflation isn't coming down as fast as he predicted.

Compact candidate:
Gas prices rise 7.5% in March; Halifax raises rates, sparing first-time buyers; Chancellor John Major warns inflation remains high.

## 29. rw_013824 · source=open_subtitles · risk=133 · saved=27

Risk tags: causal_or_contrast_force_partly_changed, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_current_overlap

Risk reasons: compact/original length ratio 0.39 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-current Jaccard 0.15

Original:
So, we've been talking, and you're a really great cheerleader, and we really want you to be a part of the team, but if you don't want to hang out with us, then maybe you don't have the right kind of school spirit for the job.

Current inherited rewrite:
After our conversation, we recognized your enthusiasm as a valuable asset to the team and would love for you to join; however, if you prefer to keep your distance, it may indicate that your level of school spirit isn't what we require for this role.

Compact candidate:
We want you on the team, but lack of school spirit means you can't hang out with us.

## 30. rw_035658 · source=open_subtitles · risk=133 · saved=17

Risk tags: causal_or_contrast_force_weakened, modal_lost_from_both_references, modality_force_weakened, negation_force_partly_changed, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_current_overlap

Risk reasons: compact length 7 words, original 15 words | modality_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-current Jaccard 0.11

Original:
I can't tell you what's after that, but it was clean, no bodies, no nothing.

Current inherited rewrite:
Although I'm unable to share what came next, I can confirm that the aftermath was spotless, with no bodies or anything else left behind.

Compact candidate:
It was clean, no bodies, no nothing.

## 31. rw_035502 · source=open_subtitles · risk=132 · saved=18

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, role_reference_weakened, severe_summary, very_low_current_overlap

Risk reasons: compact length 5 words, original 12 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33 | compact-current Jaccard 0.12

Original:
Well, your friends are gone and I think they would appreciate it.

Current inherited rewrite:
I believe your departed friends would have valued this, so now that they are no longer here, it is time to let go.

Compact candidate:
Friends gone, they appreciate it.

## 32. rw2s1_023323 · source=gutenberg · risk=132 · saved=14

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened

Risk reasons: modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.33

Original:
"He has just sent off a stud of horses to Dublin for sale, so there isn't one he can offer ye to ride." "I like _you_ very much as a horse, Uncle Fergus," said Margot.

Current inherited rewrite:
"I like _you_ very much as a horse, Uncle Fergus," said Margot, after he had just sent off a stud of horses to Dublin for sale, leaving none he could offer ye to ride.

Compact candidate:
Margot told Uncle Fergus she liked him as a horse, but he sent off his stud to Dublin for sale.

## 33. rw2s0_008006 · source=bnc_spoken · risk=130 · saved=38

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, severe_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 13 words, original 55 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.13 | compact-current Jaccard 0.13

Original:
It does mean that the way in which the case has been handled, the way in which it's been approached, the cooperation between various departments in a particular office, has been done properly, and if the client has the opportunity of winning, he will win and not lose by sloppy work in his solicitor's office.

Current inherited rewrite:
It does mean that the manner in which the case has been handled, the approach taken, the cooperation between various departments in a particular office, has been done properly, and if the client has the opportunity of winning, he will win and not lose by sloppy work in his solicitor's office.

Compact candidate:
Proper department cooperation ensures clients win, not lose due to sloppy solicitor work.

## 34. rw2s1_029884 · source=gutenberg · risk=130 · saved=35

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary

Risk reasons: compact/original length ratio 0.36 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0

Original:
They had all sorts of little secrets that day girls might not share, signs and passwords and mysterious references, which gave them great satisfaction, and were calculated to provoke envy, hatred, malice, and all uncharitableness in the breasts of those who did not understand the allusions, and whom they sternly refused to initiate.

Current inherited rewrite:
On that day, they possessed various small secrets that girls might not disclose, including signs, passwords, and mysterious references, which provided them with great satisfaction and were intended to incite envy, hatred, malice, and all uncharitableness in the hearts of those who failed to grasp the allusions and whom they sternly refused to initiate.

Compact candidate:
They shared little secrets, signs, and passwords that gave satisfaction and provoked envy in those they refused to initiate.

## 35. rw2s0_026429 · source=gutenberg · risk=130 · saved=31

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary

Risk reasons: compact/original length ratio 0.42 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0

Original:
In the New England of those days the general reader had heard a good deal about the Pilgrim Fathers and Salem Witchcraft, and remembered hazily the stories of Hannah Dustin and of Putnam and the wolf, but could not be counted on for much else before the Revolution.

Current inherited rewrite:
Before the Revolution, the general reader in the New England of those days could not be counted on for much else beyond a good deal of hearing about the Pilgrim Fathers and Salem Witchcraft, along with a hazy recollection of the stories of Hannah Dustin and of Putnam and the wolf.

Compact candidate:
New England readers knew Pilgrim Fathers, Salem Witchcraft, Hannah Dustin, Putnam and the wolf, but little else before the Revolution.

## 36. rw2s1_009835 · source=childes · risk=130 · saved=25

Risk tags: causal_or_contrast_force_weakened, navigation_sequence_changed, ordered_navigation, role_reference_weakened, temporal_order_weakened

Risk reasons: direction sequence original=['up', 'up', 'left'] compact=['left'] | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 2, compact count 1 | role/pronoun preservation 0.00

Original:
*UNC: he showed up about nine o'clock on his motorcycle and it was still you know kind o' drizzling out and stuff so he took first truck and went home to pick up Mark he came back he dropped some stuff off at Fred's house came back took his motorcycle and left.

Current inherited rewrite:
*UNC: he arrived around nine o'clock on his motorcycle while it was still kind o' drizzling out and stuff, so he took the first truck to go home and pick up Mark, then returned to drop some stuff off at Fred's house, came back, took his motorcycle, and left.

Compact candidate:
UNC arrived at nine on motorcycle in drizzle, took truck to get Mark, dropped items at Fred's house, then returned on motorcycle and left.

## 37. rw_031652 · source=bnc_spoken · risk=130 · saved=24

Risk tags: conditional_force_weakened, modality_force_weakened, ordered_navigation, role_reference_weakened, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.33; original count 3, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
If you're moving also through Banbury, of course we have our usual restrictions; the high street is affected and also if you're moving through finally on to the A422, I would mention, in Warwickshire, the Stratford to Alcester road, that has some temporary traffic lights at Taylor's wood.

Current inherited rewrite:
If your journey also includes Banbury, please be aware of our standard restrictions on the high street, and if you are proceeding onward to the A422 in Warwickshire via the Stratford to Alcester road, note that temporary traffic lights are in place at Taylor's wood.

Compact candidate:
If moving through Banbury, high street affected; on A422 in Warwickshire, Stratford to Alcester road has temporary lights at Taylor's wood.

## 38. rw2s1_004285 · source=open_subtitles · risk=130 · saved=22

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_partly_changed, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
There's no work here if you haven't got a mule At the bazaars, they load up with stuff for Iraq and bring other stuff back Could you give Ayoub a job, my dear Mr Yassin?

Current inherited rewrite:
If you don't have a mule, there is no work here; at the bazaars they load up with goods for Iraq and bring other items back, so could you give Ayoub a job, my dear Mr Yassin?

Compact candidate:
No work without a mule; bazaars load Iraq goods; ask Mr Yassin for Ayoub's job.

## 39. rw2s1_026614 · source=gutenberg · risk=129 · saved=37

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, negation_force_weakened, negation_lost_from_both_references, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.36 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
You're pluming yourself no end." Opal broke into a fit of delighted giggling, but refused all explanations, and slamming on her hat rushed away home, leaving the juniors still dancing about the cloakroom like pixies and loudly proclaiming: "It's Nicky Nan Night.

Current inherited rewrite:
"It's Nicky Nan Night," the juniors loudly proclaimed as they continued dancing about the cloakroom like pixies, while Opal, who had slammed on her hat and rushed away home after refusing all explanations and breaking into a fit of delighted giggling at the remark, "You're pluming yourself no end," left them behind.

Compact candidate:
Opal giggled, refused explanations, rushed home, leaving juniors dancing like pixies proclaiming Nicky Nan Night.

## 40. rw_032720 · source=open_subtitles · risk=129 · saved=23

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, modality_force_weakened, role_reference_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.39 | modality_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
But you have already begun to trust him You're reading me like a book The merchant ship will leave Dukjin Harbor, and pass through Palgeum and Sadang waters Where will you attack them?

Current inherited rewrite:
You've already started trusting him; you understand me as clearly as one reads an open book. The merchant vessel is departing Dukjin Harbor to navigate through Palgeum and Sadang waters—now, where do you intend to strike?

Compact candidate:
You trust him; merchant ship leaves Dukjin Harbor, passes Palgeum and Sadang waters.

## 41. rw2s1_002985 · source=gutenberg · risk=126 · saved=18

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, temporal_order_partly_changed

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0

Original:
They had seemed to hear the howl of this dreaded phantom more than once during that year, and looked forward to the long hard winter with an anxiety which neither would confess to the other.

Current inherited rewrite:
During that year, they had appeared to hear the howl of this dreaded phantom more than once and looked forward to the long hard winter with an anxiety which neither would confess to the other.

Compact candidate:
They heard the phantom's howl more than once that year, fearing the long winter with unconfessed anxiety.

## 42. rw2s1_002281 · source=bnc_spoken · risk=125 · saved=25

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
Isn't it time that members standing for C E C elections were given the same equality with Regional Secretaries, who as we all know, all year round, in all regions, when a leaflet or a booklet goes out to members, in most of them there is a photograph of the Regional Secretary.

Current inherited rewrite:
Isn't it time that members standing for C E C elections were granted the same equality with Regional Secretaries, who, as we all know, appear in leaflets or booklets sent out to members in all regions throughout the year, with a photograph of the Regional Secretary included in most of them?

Compact candidate:
Members in C E C elections should have same equality as Regional Secretaries whose photos appear in leaflets and booklets sent to members across all regions.

## 43. rw_012254 · source=bnc_spoken · risk=54 · saved=26

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, dialogue_or_question

Risk reasons: comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Any more at one hundred and ten only, one twenty, one thirty one forty, fifty one sixty, seventy one seventy one eighty, the lady's bid sir do you want to come in?

Current inherited rewrite:
The lady is bidding; sir, would you like to enter the auction at one hundred and ten, one hundred twenty, one hundred thirty, one hundred forty, one hundred fifty, one hundred sixty, one hundred seventy, one hundred eighty, or higher?

Compact candidate:
Lady bids one hundred ten to one eighty; do you want to come in?

## 44. rw_020981 · source=childes · risk=54 · saved=23

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.40 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
[very hoarse, distorted, possibly because of recording problem; or, speaker may be BR2] %add: Chi *MOT: did you have fun?

Current inherited rewrite:
The speaker is extremely hoarse and speaking with a distorted quality, likely due to a recording issue or because they are a BR2 participant; Chi *MOT asks, "Did you have fun?"

Compact candidate:
Speaker may be BR2; did you have fun?

## 45. rw_007694 · source=open_subtitles · risk=54 · saved=20

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
When Rachel first told me that she was dating Harvey Dent, I had one thing to say: "The guy that was cut off from those awful campaign commercials, I believe in Harvey Dent?" Yeah nice slogan, Harvey.

Current inherited rewrite:
Upon learning that Rachel was now with Harvey Dent, specifically the man excluded from the negative campaign ads, I couldn't help but retort, "You ask me if I believe in Harvey Dent? That's certainly a catchy slogan, Harvey."

Compact candidate:
Rachel told me she dated Harvey Dent, and I said, "I believe in Harvey Dent?" Nice slogan, Harvey.

## 46. rw2s1_005874 · source=bnc_spoken · risk=54 · saved=19

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
First, Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf; the Iraqi News Agency said the warning had come from the country's Air Force Commander.

Current inherited rewrite:
First, the Iraqi News Agency reported that the country's Air Force Commander issued a warning stating that Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf.

Compact candidate:
Iraq threatened missile attacks on Israel and Saudi Arabia if Gulf war breaks out, per Air Force Commander.

## 47. rw_044687 · source=childes · risk=54 · saved=16

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 2, compact count 0

Original:
[father laughs occasionally during this; Ross and Father make several exasperated and amused comments as Mark repeats] *FAT: okay now what happens?

Current inherited rewrite:
As Mark repeats the line, Ross and his father exchange amused yet exasperated remarks, with the father occasionally chuckling; Father then asks, "Okay now what happens?"

Compact candidate:
Ross and Father comment as Mark repeats while father laughs.

## 48. rw2s1_004011 · source=gutenberg · risk=52 · saved=25

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, modality_force_partly_changed

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
It would need but few awards made on the "eight to seven" principle, as in the Electoral Commission of 1877, to make our arbitrating tribunal the laughing-stock of the world, and to set back for a generation or two the hand upon the timepiece of civilization.

Current inherited rewrite:
It would require only a handful of awards granted under the "eight to seven" principle, similar to those in the Electoral Commission of 1877, to render our arbitrating tribunal the laughing-stock of the world and to set back for a generation or two the hand upon the timepiece of civilization.

Compact candidate:
Few awards on the "eight to seven" principle, like the 1877 Electoral Commission, would make our tribunal a laughing-stock and set back civilization for generations.

## 49. rw_009884 · source=bnc_spoken · risk=52 · saved=24

Risk tags: causal_or_contrast_force_weakened, role_reference_weakened, very_low_current_overlap

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-current Jaccard 0.14

Original:
No Singapore on the way back , didn't get off the plane though, I mean they only re-fuel, your only allowed off for about half an hour.

Current inherited rewrite:
Although the return leg of the journey did not include a stop in Singapore because aircraft are only permitted to disembark passengers for refueling during a brief window of approximately thirty minutes, I did manage to get off the plane.

Compact candidate:
No Singapore on return; didn't disembark as only refueling allowed, exit limited to half an hour.

## 50. rw_036466 · source=open_subtitles · risk=52 · saved=17

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, very_low_current_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | compact-current Jaccard 0.11

Original:
If you hated me half as much as you hate yourself, you'd have used it.

Current inherited rewrite:
If your dislike for me were even a fraction of the hatred you hold for yourself, you would have taken advantage of it long ago.

Compact candidate:
Hate yourself more than me, you'd use it.

## 51. rw_036043 · source=open_subtitles · risk=52 · saved=15

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, temporal_order_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 2, compact count 0

Original:
But then I saw what it was that made him faint in the first place and I almost fainted,too,because it was a mummy,which I told the sheriff, who called the fbi,who called the jeffersonian.

Current inherited rewrite:
However, upon realizing that the object causing his initial fainting spell was a mummy, I nearly lost consciousness myself, so I informed the sheriff, who contacted the FBI, who in turn reached out to the Jeffersonian.

Compact candidate:
I saw what made him faint, almost fainted myself, it was a mummy, told sheriff, who called FBI, who called Jeffersonian.

## 52. rw2s0_001278 · source=gutenberg · risk=51 · saved=22

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, negation_force_partly_changed, strong_summary

Risk reasons: compact/original length ratio 0.38 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Jim, if he could not come back, and she did not expect him even for her sake, to leave such wonderfully grand people as the Holroyds, would at least write a long, explanatory letter to his father.

Current inherited rewrite:
Jim, if he could not return, and she did not expect him even for her sake to leave such wonderfully grand people as the Holroyds, would at least write a long, explanatory letter to his father.

Compact candidate:
Jim would write a long letter to his father if he could not return.

## 53. rw2s1_016216 · source=gutenberg · risk=51 · saved=16

Risk tags: causal_or_contrast_force_partly_changed, negation_force_weakened, negation_lost_from_both_references

Risk reasons: negation_force marker preservation 0.00; original count 2, compact count 0

Original:
Helen further records the trying to get some breakfast in the marketplace and finding nothing but herrings, also the going to mass, and the care she took not to sit upon the holy crown, though she had to sit on its cushion in the sledge.

Current inherited rewrite:
Helen further records the attempt to obtain breakfast in the marketplace and discovering nothing but herrings, as well as the trip to mass and the care she exercised to avoid sitting upon the holy crown, even though she was compelled to sit on its cushion in the sledge.

Compact candidate:
Helen records trying to get breakfast in the marketplace, finding only herrings, going to mass, and carefully avoiding sitting on the holy crown, though she sat on its cushion in the sledge.

## 54. rw2s0_028417 · source=gutenberg · risk=50 · saved=31

Risk tags: causal_or_contrast_force_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 3, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
Passionately he pled with the beautiful creature--spoke of his love--besought her to have pity on him, but although the face in the pool reflected his every look of adoration and of longing, time and again he vainly tried to clasp in his arms what was but the mirrored likeness of himself.

Current inherited rewrite:
Passionately he pleaded with the beautiful creature, speaking of his love and beseeching her to have pity on him, yet although the face in the pool reflected his every look of adoration and longing, time and again he vainly tried to clasp in his arms what was but the mirrored likeness of himself.

Compact candidate:
He passionately pleaded with the beautiful creature, besought pity, yet vainly tried to clasp the mirrored likeness of himself in his arms.

## 55. rw_039796 · source=open_subtitles · risk=50 · saved=28

Risk tags: role_reference_weakened, strong_summary, temporal_order_weakened

Risk reasons: compact/original length ratio 0.38 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
Feb 28 : International Red Army member Fusako Shigenobu departs for Lebanon Late May : RLF's Kosode Base We'll undertake guerrilla warfare here from now on inheriting the spirit of our massacred revolutionary comrade we'll take out all surveillance bases and police stations Steal the cop's guns !

Current inherited rewrite:
On February 28, International Red Army member Fusako Shigenobu left for Lebanon, where the RLF's Kosode Base announced in late May that they would inherit the spirit of their martyred comrades by launching guerrilla warfare to destroy surveillance bases and police stations while seizing officers' weapons.

Compact candidate:
Feb 28: Fusako Shigenobu departs for Lebanon; Late May: RLF's Kosode Base undertakes guerrilla warfare, stealing cop guns.

## 56. rw2s1_005820 · source=gutenberg · risk=50 · saved=25

Risk tags: conditional_force_weakened, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.33

Original:
The party hidden in the Devil's Cleft overheard the peasants in the wood talking of the fall of the giant of Kauffingen, and, becoming alarmed for themselves, they sent to the Governor of the neighboring castle of Hartenstein to offer to restore Prince Ernst, provided they were promised a full pardon.

Current inherited rewrite:
After overhearing peasants in the wood discuss the fall of the giant of Kauffingen, the party concealed in the Devil's Cleft, alarmed for their own safety, sent a message to the Governor of the neighboring castle of Hartenstein offering to restore Prince Ernst on the condition that they be granted a full pardon.

Compact candidate:
The party in Devil's Cleft heard peasants talk of Kauffingen giant's fall, then alarmed, they sent to Hartenstein castle Governor to offer restoring Prince Ernst for full pardon.

## 57. rw2s1_011559 · source=bnc_spoken · risk=50 · saved=25

Risk tags: role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
A neighbour of the victims, ten year old Alexander Ray, has seen the drama unfold: There's this man on this tree, it's about 50 feet tall, the tree, and he's er he's just crouched on there, sitting there now, with a blue rope round his neck, and it's tied on to the tree.

Current inherited rewrite:
Ten-year-old Alexander Ray, a neighbour of the victims, witnessed the unfolding drama: "There's this man on this tree, it's about 50 feet tall, the tree, and he's er he's just crouched on there, sitting there now, with a blue rope round his neck, and it's tied on to the tree."

Compact candidate:
Ten-year-old Alexander Ray, a victim's neighbour, saw a man crouched on a 50-foot tree with a blue rope around his neck tied to the tree.

## 58. rw2s0_010846 · source=bnc_spoken · risk=50 · saved=20

Risk tags: role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.14

Original:
Well they have four cars they have this wee car for, to tow the caravan to the caravan sites on holiday and then he has his new car and the father has a Jaguar and then they've some other wee car just to get them about you know.

Current inherited rewrite:
Well, they have four cars: they have this wee car for towing the caravan to the caravan sites on holiday, then he has his new car, the father has a Jaguar, and then they've some other wee car just to get them about, you know.

Compact candidate:
They have four cars: a small one for towing caravans on holiday, a new car, a father's Jaguar, and another small car for daily use.

## 59. rw2s0_026977 · source=childes · risk=50 · saved=16

Risk tags: role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
[Ross dropped his sand speeder which he was using for light to get down stairs to get a cookie for Mark.] *CHI: oh now what'll I do.

Current inherited rewrite:
[Ross dropped his sand speeder, which he had been using as a light to descend the stairs and retrieve a cookie for Mark.] *CHI: oh now what'll I do.

Compact candidate:
Ross dropped his sand speeder light to get down stairs for Mark's cookie.

## 60. rw2s1_002760 · source=childes · risk=50 · saved=15

Risk tags: role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.33

Original:
*CHI: back to school and then we had quiet time I took my Heman book and slept with it and then I let Brian sleep on my cape Joey slept with his man.

Current inherited rewrite:
*CHI: After returning to school and having quiet time, I took my Heman book and slept with it, then allowed Brian to sleep on my cape while Joey slept with his man.

Compact candidate:
Back to school, quiet time, I slept with Heman book, Brian on cape, Joey with his man.

## 61. rw2s1_025448 · source=switchboard · risk=50 · saved=12

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
B: But I suggest it to anyone who wants to be better off financially to read it because, uh, A: And that includes everyone at T I.

Current inherited rewrite:
A: And that includes everyone at T I, while B suggests it to anyone who wants to be better off financially to read it because, uh.

Compact candidate:
I suggest reading it to anyone wanting financial betterment, including everyone at T I.

## 62. rw_042249 · source=open_subtitles · risk=49 · saved=20

Risk tags: compressed_dialogue, dialogue_or_question, severe_summary

Risk reasons: compact length 7 words, original 16 words

Original:
(barks) - (scoffs) - (barks) (chuckles quietly) (barks softly) You ever hear of the designated survivor?

Current inherited rewrite:
With a series of barks and scoffs, followed by a quiet chuckle and a softer bark, he asks if you have ever heard of the designated survivor.

Compact candidate:
You ever hear of the designated survivor?

## 63. rw2s0_016929 · source=bnc_spoken · risk=0 · saved=22

Risk tags: none

Original:
They're going to be celebrating with a match against the Kew Occasionals fifteen, who include two Irish internationals and six former blues, and afterwards there'll be a firework display, and if they phone me up I'll come along, as long as it's free of charge.

Current inherited rewrite:
They are set to celebrate with a match against the Kew Occasionals fifteen, who include two Irish internationals and six former blues, followed by a firework display, and if they phone me up I'll come along, as long as it's free of charge.

Compact candidate:
They celebrate against Kew Occasionals fifteen, featuring two Irish internationals and six former blues, with fireworks, and I attend if free.

## 64. rw_033941 · source=open_subtitles · risk=12 · saved=10

Risk tags: dialogue_or_question

Original:
Shiomi's Kansai faction called for militancy 956 arrests The Law Against Destructive Activities was used July 2, Saragi's faction began to probe the dissolution of Shiomi's Kansai faction, who had moved to form the RAF (Red Faction Army).

Current inherited rewrite:
On July 2, as Saragi's faction initiated an investigation into Shiomi's Kansai group following 956 arrests prompted by calls for militancy under the Law Against Destructive Activities, Shiomi had already moved to establish the Red Faction Army (RAF).

Compact candidate:
Shiomi's Kansai faction called for militancy; 956 arrests occurred July 2 under The Law Against Destructive Activities as Saragi's faction probed Shiomi's Kansai faction dissolution and RAF formation.

## 65. rw2s0_013580 · source=bnc_spoken · risk=20 · saved=22

Risk tags: causal_or_contrast_force_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
Earnings per share in the period were five p but after taking into account exceptional Carpet Right gain this is reduced to two point eight p and compares to last year's pro forma earnings per share of one point four million which included losses on the closing down of our operations.

Current inherited rewrite:
Earnings per share for the period stood at five p, which falls to two point eight p when the exceptional Carpet Right gain is factored in, compared to last year's pro forma earnings per share of one point four million that incorporated losses from closing down our operations.

Compact candidate:
Earnings per share were five p, reduced to two point eight p after Carpet Right gain, versus last year's one point four million pro forma earnings.

## 66. rw2s1_017873 · source=open_subtitles · risk=0 · saved=17

Risk tags: none

Original:
Christians get a trip to the moon on gossamer wings, Muslims get a little salt and pepper on a fruit salad at the end of Ramadan.

Current inherited rewrite:
At the conclusion of Ramadan, Muslims receive a modest amount of salt and pepper atop a fruit salad, whereas Christians are granted a journey to the moon propelled by gossamer wings.

Compact candidate:
Christians get moon trips on wings; Muslims get salt on fruit salad after Ramadan.

## 67. rw2s0_001657 · source=childes · risk=0 · saved=12

Risk tags: none

Original:
*CHI: Humpty Dumpty sat on a wall Humpty Dumpty had a great fall all the king's horses and all the king's men couldn't put Humpty Dumpty together again.

Current inherited rewrite:
*CHI: Humpty Dumpty had a great fall after sitting on a wall, and despite all the king's horses and all the king's men trying, they couldn't put Humpty Dumpty together again.

Compact candidate:
Humpty Dumpty sat on a wall, fell, and all the king's horses and men couldn't put him together again.

## 68. rw2s0_004620 · source=simple_wiki · risk=20 · saved=9

Risk tags: temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
One year after, on October 20 1827, he and the crew of Azov fought in the Battle of Navarino and with the help of the British and French naval forces, they defeated the Ottoman naval forces.

Current inherited rewrite:
One year later, on October 20 1827, he and the crew of Azov fought in the Battle of Navarino and, with the help of the British and French naval forces, they defeated the Ottoman naval forces.

Compact candidate:
One year later, on October 20 1827, he and the Azov crew fought in the Battle of Navarino, defeating Ottoman naval forces with British and French help.

## 69. rw2s0_009570 · source=open_subtitles · risk=12 · saved=14

Risk tags: dialogue_or_question

Original:
The concert ends, I jump into the car, back on the road, I call her on the way and she tells me: That's lovely, darling, I love you, do you know?

Current inherited rewrite:
After the concert concludes, I jump into the car and get back on the road, calling her along the way, at which point she tells me: That's lovely, darling, I love you, do you know?

Compact candidate:
Concert ends, I jump in car, back on road, call her, she says lovely darling I love you do you know.

## 70. rw_030863 · source=switchboard · risk=0 · saved=7

Risk tags: none

Original:
B: You know, they go B: and they decide if you can move up to the next level.

Current inherited rewrite:
B: Essentially, they review your progress and determine whether you are ready to advance to the next level.

Compact candidate:
They decide if you can move up to the next level.

## 71. rw2s1_030959 · source=childes · risk=0 · saved=11

Risk tags: none

Original:
*CHI: in the frozen cavern in Rebel base the famous star pilot Han Solo watched the blip on the radar screen.

Current inherited rewrite:
*CHI: While the famous star pilot Han Solo observed the blip on the radar screen within the frozen cavern at Rebel base, he was present.

Compact candidate:
Han Solo watched the blip on the radar screen in Rebel base's frozen cavern.

## 72. rw_028231 · source=open_subtitles · risk=12 · saved=17

Risk tags: very_low_current_overlap

Risk reasons: compact-current Jaccard 0.13

Original:
Lucas, you've been a great assistant coach this year, and someday you're gonna take over the team, but for now, you take orders from me.

Current inherited rewrite:
Although I recognize your potential to lead the team in the future and appreciate your outstanding work as assistant coach this year, remember that for now, you must follow my instructions.

Compact candidate:
Lucas, you're a great assistant coach, but for now, you take orders from me.

## 73. rw_017420 · source=simple_wiki · risk=0 · saved=20

Risk tags: none

Original:
After three matches in the Campeonato Paulista Série A2 he went on loan to Ponte Preta but he was only three times on the bench and was sent back to the Red Bulls.

Current inherited rewrite:
Following just three appearances for the Red Bulls in the Campeonato Paulista Série A2, his loan move to Ponte Preta proved unsuccessful after he saw only three minutes of play from the bench, resulting in his return to his original club.

Compact candidate:
After three matches in Campeonato Paulista Série A2, he went on loan to Ponte Preta but returned to the Red Bulls.

## 74. rw2s1_024046 · source=bnc_spoken · risk=20 · saved=11

Risk tags: causal_or_contrast_force_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
Spokesman for Jet John Maples say although there's no immediate threat to jobs he's pleased that Michael Heseltine and John Patten have spoken up in Culham's favour.

Current inherited rewrite:
Spokesman for Jet John Maples says that although there is no immediate threat to jobs, he is pleased that Michael Heseltine and John Patten have spoken up in Culham's favour.

Compact candidate:
Jet John Maples says no immediate job threat but pleased Michael Heseltine and John Patten spoke up for Culham.

## 75. rw2s1_023231 · source=simple_wiki · risk=12 · saved=13

Risk tags: dialogue_or_question

Original:
In the United States, "Like I'm Gonna Lose You" reached number eight on the "Billboard" Hot 100 and was certified 4× Platinum by the Recording Industry Association of America.

Current inherited rewrite:
In the United States, the song "Like I'm Gonna Lose You" achieved a peak position of number eight on the "Billboard" Hot 100 and received a 4× Platinum certification from the Recording Industry Association of America.

Compact candidate:
"Like I'm Gonna Lose You" reached number eight on Billboard Hot 100 and was certified 4× Platinum by RIAA in the United States.

## 76. rw_016699 · source=childes · risk=0 · saved=18

Risk tags: none

Original:
[videotape starts at this point] [SIS returns Very Hungry Caterpillar book to EXP; Chi has the large magnet directly in front of her; she is currently arranging the three colored magnetic balls on the black base magnet] *MOT: here.

Current inherited rewrite:
At this point, the videotape begins: SIS gives the Very Hungry Caterpillar book to EXP while Chi, with a large magnet positioned directly before her, arranges three colored magnetic balls on the black base, saying, "Here."

Compact candidate:
SIS returns Very Hungry Caterpillar book to EXP; Chi arranges three colored magnetic balls on black base magnet.

## 77. rw_024111 · source=switchboard · risk=0 · saved=10

Risk tags: none

Original:
B: And that is it's million of people with anywhere from thousands to tens of thousands I guess of, Of dollars being protected.

Current inherited rewrite:
Speaker B concludes by noting that millions of people have safeguards in place, covering amounts ranging from thousands to tens of thousands of dollars.

Compact candidate:
That is millions of people with thousands to tens of thousands of dollars protected.

## 78. rw2s0_019304 · source=simple_wiki · risk=0 · saved=23

Risk tags: none

Original:
Manning is the son of Ed Manning, who was a longtime NBA and ABA player and professional and college coach.

Current inherited rewrite:
Manning, the son of Ed Manning, was born to a man who served as a professional and college coach and had a long career as a player in both the NBA and ABA.

Compact candidate:
Manning is son of Ed Manning, NBA/ABA player and coach.

## 79. rw_032179 · source=bnc_spoken · risk=0 · saved=21

Risk tags: none

Original:
In the Jordanian capital Amman, a Palestinian armed with a pistol and a knife attacked a group of French tourists, and the Jordanian police shot and killed two youths during a protest by tens of thousands of Palestinians.

Current inherited rewrite:
During a protest involving tens of thousands of Palestinians in Amman, the Jordanian capital, a Palestinian wielding both a knife and a pistol struck a group of French tourists, while Jordanian police subsequently shot and killed two youths.

Compact candidate:
In Amman, a Palestinian attacked French tourists, and Jordanian police killed two youths during a Palestinian protest.

## 80. rw_023748 · source=open_subtitles · risk=0 · saved=19

Risk tags: none

Original:
And the fact that besides the coveted Golden Hatchet, our wonderful arts committee is providing a spectacular grand prize this semester: A college scholarship to the school of your choice!

Current inherited rewrite:
In addition to the highly sought-after Golden Hatchet, our exceptional arts committee is awarding an even more impressive grand prize this semester: a college scholarship that you can use at any institution of your choice!

Compact candidate:
Besides the Golden Hatchet, the arts committee provides a grand prize: a college scholarship this semester.

## 81. rw2s0_002140 · source=simple_wiki · risk=0 · saved=18

Risk tags: none

Original:
He was a practicing clinician at the John Radcliffe Hospital, Oxford and Nuffield Professor of Clinical Medicine and head of the Nuffield Department of Clinical Medicine at the University of Oxford from 2004 to 2016.

Current inherited rewrite:
From 2004 to 2016, he served as a practicing clinician at the John Radcliffe Hospital, Oxford, held the title of Nuffield Professor of Clinical Medicine, and led the Nuffield Department of Clinical Medicine at the University of Oxford.

Compact candidate:
He was a clinician at John Radcliffe Hospital, Oxford and Nuffield Professor at University of Oxford from 2004 to 2016.

## 82. rw_000248 · source=simple_wiki · risk=12 · saved=27

Risk tags: dialogue_or_question

Original:
Renee Schonfeld of Common Sense Media rated it 3 out of 5 stars, saying "By blending solid new musical numbers with the old Oz favorites and including inventive new characters, Spike Brandt, Tony Cerone, and company have made this sequel funny and original." "Tom and Jerry: Cowboy Up!".

Current inherited rewrite:
In her review for *Tom and Jerry: Cowboy Up!*, Renee Schonfeld of Common Sense Media awarded the film 3 out of 5 stars, praising the sequel for its blend of classic Oz favorites with fresh musical numbers and inventive new characters, crediting director Spike Brandt, Tony Cerone, and their team with creating a funny and original result.

Compact candidate:
Renee Schonfeld of Common Sense Media rated Tom and Jerry: Cowboy Up! 3 out of 5 stars, saying Spike Brandt, Tony Cerone, and company made this sequel funny and original.

