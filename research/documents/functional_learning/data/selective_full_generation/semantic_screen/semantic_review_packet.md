# earlier analysis compact-view semantic review packet

Purpose: distinguish faithful shortening from information-losing summary and altered source meaning before any compact text is admitted to learner training.

## Pilot-level summary

- n_records: 26567
- current_number_length_pass: 16628
- auto_low_risk: 13539
- needs_semantic_review: 3262
- known_altered: 2
- auto_low_risk_saved_words: 52319
- current_number_length_saved_words: 69320

## Review instructions

For each item, compare the compact rewrite primarily with the original source text, using the current rewrite only as an inherited second view. Label it as one of: faithful_shortening, supported_summary_with_lost_detail, altered_meaning, or unclear. Pay special attention to entity roles, numbers, direction/order, conditional force, modality, negation, temporal relations, and who did what.

## 1. rw2s0_029078 · source=open_subtitles · risk=190 · saved=11

Risk tags: known_altered_claim, navigation_sequence_changed, ordered_navigation, role_reference_weakened

Risk reasons: navigation order reversed near Goddess Mother temple and idol makers | direction sequence original=['straight', 'left', 'up', 'left', 'right', 'straight', 'left', 'up', 'opposite', 'straight', 'up'] compact=['straight', 'left', 'left', 'right', 'straight', 'left', 'straight', 'opposite', 'straight', 'up'] | role/pronoun preservation 0.33

Original:
Go straight and turn left, further up there's a street, turn left again, there's temple of Goddess Mother, turn right there, further straight you'll see a place where they make idols, turn left there, further up you'll see a barber shop, take the street opposite to it, straight up it's Afzal Gunj!

Current inherited rewrite:
Go straight and turn left until you reach a street, then turn left again to find the temple of Goddess Mother, turn right and proceed straight to where they make idols, turn left and continue until you see a barber shop, take the street opposite it, and go straight up to reach Afzal Gunj!

Compact candidate:
Go straight, turn left to a street, turn left again to the temple of Goddess Mother, turn right, go straight to where they make idols, turn left, go straight to a barber shop, take the opposite street, go straight up to Afzal Gunj!

## 2. rw_035551 · source=open_subtitles · risk=142 · saved=0

Risk tags: dialogue_or_question, known_altered_claim, no_word_saving

Risk reasons: conditional self-attribution converted toward a direct assertion | compact words 37 >= current rewrite words 37

Original:
To bad Amaru I didn't go crazy and nothing is wrong with me If there is something wrong, that would be me in the past What are you doing Amaru?

Current inherited rewrite:
Amaru, I feel terrible that I didn't lose my mind because nothing is currently wrong with me; if there were a problem, it would stem from who I was in the past. What are you doing, Amaru?

Compact candidate:
Amaru, I feel terrible that I didn't lose my mind because nothing is currently wrong with me; if there were a problem, it would stem from who I was in the past. What are you doing, Amaru?

## 3. rw_015509 · source=gutenberg · risk=187 · saved=0

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, navigation_sequence_changed, negation_force_weakened, no_word_saving, ordered_navigation, temporal_order_weakened

Risk reasons: compact words 42 >= current rewrite words 42 | direction sequence original=['straight', 'west'] compact=['west'] | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
Perhaps I throw a stone or so at Mrs Blizzard's tank, Because it's great when I aim straight to hear the stone go "Plank!" Then west I wend from Blizzard's Bend, and not a moment wait, Except, perhaps, at Mr Knapp's, to swing upon his gate.

Current inherited rewrite:
Maybe I toss a stone or two at Mrs. Blizzard's tank, enjoying the satisfying "Plank!" when I hit my mark, before heading west from Blizzard's Bend without delay, save perhaps for a brief stop at Mr. Knapp's to swing on his gate.

Compact candidate:
Maybe I toss a stone or two at Mrs. Blizzard's tank, enjoying the satisfying "Plank!" when I hit my mark, before heading west from Blizzard's Bend without delay, save perhaps for a brief stop at Mr. Knapp's to swing on his gate.

## 4. rw_012713 · source=bnc_spoken · risk=178 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened, very_low_source_overlap

Risk reasons: compact words 24 >= current rewrite words 24 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.00 | compact-original Jaccard 0.19

Original:
Because that's half the battle if you can keep nights, I didn't have a decent night's sleep just for the tickle you know, the .

Current inherited rewrite:
Since securing a full night's rest is half the battle, my lack of decent sleep due to that slight sensation is exactly what happened.

Compact candidate:
Since securing a full night's rest is half the battle, my lack of decent sleep due to that slight sensation is exactly what happened.

## 5. rw_019213 · source=gutenberg · risk=178 · saved=0

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 21 >= current rewrite words 21 | conditional_force marker preservation 0.33; original count 3, compact count 2 | modality_force marker preservation 0.00; original count 2, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
If she would only exert herself she might even get quite well; but Mrs Aldworth had not the least intention of exerting herself.

Current inherited rewrite:
Although Mrs. Aldworth could have recovered perfectly if she had put in some effort, she had absolutely no intention of trying.

Compact candidate:
Although Mrs. Aldworth could have recovered perfectly if she had put in some effort, she had absolutely no intention of trying.

## 6. rw2s0_027591 · source=gutenberg · risk=175 · saved=16

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, ordered_navigation, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.41 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.33

Original:
I got up and dressed, but had to lie down again, and thus I spent the day; and when my sisters came in to see me I would not speak to them.

Current inherited rewrite:
I rose and dressed, yet was compelled to recline once more, thereby passing the day, and upon my sisters' arrival to visit me, I refused to speak with them.

Compact candidate:
I rose, dressed, reclined again, passed the day, and refused my sisters' visit.

## 7. rw2s0_011440 · source=gutenberg · risk=172 · saved=0

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_partly_changed, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 34 >= current rewrite words 34 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.33; original count 3, compact count 2 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0

Original:
"Our horses be not of the best, but neither are they of the worst; and it were an ill beast that could not go thirty-three miles before sunset on the Watling Street." "Ay," agreed Hugo.

Current inherited rewrite:
"Ay," agreed Hugo, in response to the statement that while their horses were neither top-tier nor bottom-tier, it would be a poor beast unable to cover thirty-three miles by sunset on the Watling Street.

Compact candidate:
"Ay," agreed Hugo, in response to the statement that while their horses were neither top-tier nor bottom-tier, it would be a poor beast unable to cover thirty-three miles by sunset on the Watling Street.

## 8. rw_026864 · source=gutenberg · risk=172 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 23 >= current rewrite words 23 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 2 | negation_force marker preservation 0.00; original count 1, compact count 2 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
But after your promise last week--" "I made no promise, and should decline to do so.

Current inherited rewrite:
However, following your assurance from last week, I must clarify that I never made such a promise and will not make one now.

Compact candidate:
However, following your assurance from last week, I must clarify that I never made such a promise and will not make one now.

## 9. rw2s0_010628 · source=gutenberg · risk=163 · saved=14

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.20 | compact-original Jaccard 0.19

Original:
Shun them as you would avoid poison, since once you lose your own self-respect you are sure to fail.

Current inherited rewrite:
Avoid them with the same caution one would use to shun poison, for once you forfeit your own self-respect, failure is inevitable.

Compact candidate:
Shun them like poison; losing self-respect ensures failure.

## 10. rw_018674 · source=gutenberg · risk=163 · saved=0

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, negation_force_weakened, no_word_saving, role_reference_weakened, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact words 20 >= current rewrite words 20 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.25 | compact-original Jaccard 0.09

Original:
We won't get that, and we'll keep falling further and further behind, because they've probably still got more wind than we have.

Current inherited rewrite:
Since they likely possess a greater advantage, failing to secure that will only cause us to fall even further behind.

Compact candidate:
Since they likely possess a greater advantage, failing to secure that will only cause us to fall even further behind.

## 11. rw_007323 · source=bnc_spoken · risk=162 · saved=7

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 1

Original:
Mr Ellis though says it would be more foolish to allow, not to allow reunification.

Current inherited rewrite:
According to Mr. Ellis, it would be far more unwise to prevent reunification than to permit it.

Compact candidate:
Mr Ellis says allowing reunification is wiser than preventing it.

## 12. rw_021821 · source=gutenberg · risk=160 · saved=8

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 3, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.18

Original:
"Mother said Mack could take us for a little ride in the fresh air so we would have a better appetite for dinner." This was quite exciting, for Edna's opportunities for riding in an automobile were not many.

Current inherited rewrite:
Since Edna rarely had a chance to ride in a car, she was thrilled that Mother allowed Mack to take them for a brief drive to freshen the air and whet their appetite for dinner.

Compact candidate:
Since Edna rarely rode in a car, she thrilled that Mother let Mack take them for a brief drive to freshen air and whet appetite for dinner.

## 13. rw_024102 · source=gutenberg · risk=160 · saved=3

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, temporal_order_weakened, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.19

Original:
And then--how it happened I don't think either of the boys could have told--their anger grew from words into deeds.

Current inherited rewrite:
Ultimately, although neither boy could recall exactly how it transpired, their rage escalated from mere words to actions.

Compact candidate:
Ultimately, though neither boy recalled how it happened, their rage escalated from words to actions.

## 14. rw_012178 · source=gutenberg · risk=160 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 14 >= current rewrite words 14 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
done, but still she would not be plain with those about her.

Current inherited rewrite:
Although the task was completed, she continued to remain guarded around those near her.

Compact candidate:
Although the task was completed, she continued to remain guarded around those near her.

## 15. rw_025250 · source=gutenberg · risk=160 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 31 >= current rewrite words 31 | conditional_force marker preservation 0.00; original count 2, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
But when we found him not, we turned about and made towards our own camp, only desiring Van der Kloof, if he should meet with Temple, to bid him follow hard after us.

Current inherited rewrite:
However, upon failing to locate him, we reversed course and headed back to our camp, requesting that Van der Kloof, were he to encounter Temple, urge him to pursue us relentlessly.

Compact candidate:
However, upon failing to locate him, we reversed course and headed back to our camp, requesting that Van der Kloof, were he to encounter Temple, urge him to pursue us relentlessly.

## 16. rw_036106 · source=open_subtitles · risk=159 · saved=8

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, severe_summary

Risk reasons: compact length 7 words, original 13 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1

Original:
Well, don't you think she would have at least told me about it?

Current inherited rewrite:
Wouldn't you agree that she should have informed me about this at the very least?

Compact candidate:
Wouldn't she have told me at least?

## 17. rw2s1_016026 · source=simple_wiki · risk=158 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_partly_changed

Risk reasons: compact words 25 >= current rewrite words 25 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
The first idea for the Ferengi was that they would be the main enemy aliens in "Star Trek: The Next Generation", but they were not scary enough.

Current inherited rewrite:
The initial concept for the Ferengi was to make them the primary alien adversaries in "Star Trek: The Next Generation", yet they lacked sufficient scariness.

Compact candidate:
The initial concept for the Ferengi was to make them the primary alien adversaries in "Star Trek: The Next Generation", yet they lacked sufficient scariness.

## 18. rw_004119 · source=simple_wiki · risk=158 · saved=0

Risk tags: conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact words 26 >= current rewrite words 26 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 2 | temporal_order marker preservation 0.00; original count 1, compact count 1 | compact-original Jaccard 0.18

Original:
Even after the 1940s, when doctors realized that penicillin could cure syphilis, the men were not given this cure, or any other treatment.

Current inherited rewrite:
Despite the fact that physicians discovered penicillin's ability to treat syphilis in the 1940s, the men still received neither this life-saving medicine nor any alternative therapy.

Compact candidate:
Despite the fact that physicians discovered penicillin's ability to treat syphilis in the 1940s, the men still received neither this life-saving medicine nor any alternative therapy.

## 19. rw_005136 · source=gutenberg · risk=158 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 22 >= current rewrite words 22 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | compact-original Jaccard 0.21

Original:
But they could not agree which one was to do it, and in quarreling over the matter, Romulus killed his own twin brother Remus.

Current inherited rewrite:
However, their disagreement over who should perform the task led to a quarrel that ended with Romulus slaying his twin brother, Remus.

Compact candidate:
However, their disagreement over who should perform the task led to a quarrel that ended with Romulus slaying his twin brother, Remus.

## 20. rw_006432 · source=gutenberg · risk=158 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_partly_changed, no_word_saving, role_reference_weakened, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact words 23 >= current rewrite words 23 | conditional_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 3, compact count 0 | role/pronoun preservation 0.25 | compact-original Jaccard 0.19

Original:
"He could but for thee, for we are powerless." "Then again I say, he shall not." "Come nearer still," said Lady De Aldithely.

Current inherited rewrite:
Come closer," Lady De Aldithely commanded, adding that without her, they were impotent, yet in her own voice she insisted, "He shall not.

Compact candidate:
Come closer," Lady De Aldithely commanded, adding that without her, they were impotent, yet in her own voice she insisted, "He shall not.

## 21. rw_024683 · source=bnc_spoken · risk=158 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 22 >= current rewrite words 22 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 2 | compact-original Jaccard 0.14

Original:
I might be going to er Ellesmere Port, there's coffin stones to get up, they've got er hoist thing there so I don't know.

Current inherited rewrite:
I may be heading to Ellesmere Port, but since the building has elevators to address the stairs, I'm unsure if this applies.

Compact candidate:
I may be heading to Ellesmere Port, but since the building has elevators to address the stairs, I'm unsure if this applies.

## 22. rw_028278 · source=gutenberg · risk=158 · saved=0

Risk tags: conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened, very_low_source_overlap

Risk reasons: compact words 29 >= current rewrite words 29 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.00 | compact-original Jaccard 0.16

Original:
garden, and one bore white mulberries and the other black mulberries, and when you had paid your fip to come in, you could eat all the mulberries you wanted, for nothing.

Current inherited rewrite:
Upon paying a small entry fee, guests were welcome to enjoy all the mulberries produced by either the white- or black-fruited tree in the garden at no additional cost.

Compact candidate:
Upon paying a small entry fee, guests were welcome to enjoy all the mulberries produced by either the white- or black-fruited tree in the garden at no additional cost.

## 23. rw_041033 · source=open_subtitles · risk=158 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 20 >= current rewrite words 20 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 1 | compact-original Jaccard 0.18

Original:
But I don't think he's gonna call me back because no one would actually admit that they stole your work.

Current inherited rewrite:
However, I doubt he will return my call, as most people refuse to acknowledge that they have plagiarized your work.

Compact candidate:
However, I doubt he will return my call, as most people refuse to acknowledge that they have plagiarized your work.

## 24. rw_024591 · source=gutenberg · risk=157 · saved=5

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, ordered_navigation

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
But, on the unwarrantable assumption that "boys could not play at dolls," the only part assigned me in the puppet comedy was to take the dolls' dirty clothes to and from an imaginary wash in a miniature wheelbarrow.

Current inherited rewrite:
However, because I was wrongly convinced that "boys cannot play with dolls," my sole role in the puppet show was to transport an imaginary pile of soiled laundry to and from a pretend wash using a tiny wheelbarrow.

Compact candidate:
However, wrongly assuming "boys cannot play with dolls," my sole role in the puppet show was transporting an imaginary pile of soiled laundry to and from a pretend wash using a tiny wheelbarrow.

## 25. rw2s0_013277 · source=gutenberg · risk=155 · saved=0

Risk tags: conditional_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, ordered_navigation

Risk reasons: compact words 18 >= current rewrite words 18 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0

Original:
He would not allow the surgeon to leave the sailors to attend to him till it came to his turn.

Current inherited rewrite:
He refused to let the surgeon abandon the sailors to care for him until it was his turn.

Compact candidate:
He refused to let the surgeon leave the sailors to care for him until it was his turn.

## 26. rw_020285 · source=open_subtitles · risk=152 · saved=3

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
So you're saying that if I don't sign these, I can't come back to work?

Current inherited rewrite:
Are you indicating that refusing to sign these documents will prevent me from returning to work?

Compact candidate:
Are you saying refusing to sign these prevents me from returning to work?

## 27. rw2s1_013769 · source=gutenberg · risk=152 · saved=2

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
They debated the matter for two days; and then, as they would not give him all he asked without promise or inquiry, he dissolved them.

Current inherited rewrite:
After debating the matter for two days, he dissolved them because they would not grant him all he asked without promise or inquiry.

Compact candidate:
After two days of debate, he dissolved them because they refused to grant him all he asked without promise or inquiry.

## 28. rw_003464 · source=gutenberg · risk=152 · saved=0

Risk tags: comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 35 >= current rewrite words 35 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.33; original count 3, compact count 3 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 1

Original:
When I am the Emperor's wife," laughing, "I shall find a way to punish him, for no one can give a man more pain that his wife can, if she desires to do so.

Current inherited rewrite:
As the Emperor's wife," she giggled, "I will surely devise a means to torment him, since a wife can inflict far greater pain on her husband than anyone else, should she choose to do so.

Compact candidate:
As the Emperor's wife," she giggled, "I will surely devise a means to torment him, since a wife can inflict far greater pain on her husband than anyone else, should she choose to do so.

## 29. rw_006819 · source=bnc_spoken · risk=152 · saved=0

Risk tags: comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 27 >= current rewrite words 27 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 2, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
God forbid that you should ever lose parcel and Mrs on Newcastle hasn't got a wig base any more your reputation's going to be in tatters, isn't it?

Current inherited rewrite:
If God spares us from the day you lose the parcel and Mrs. in Newcastle loses her wig base, your reputation will be utterly ruined, won't it?

Compact candidate:
If God spares us from the day you lose the parcel and Mrs. in Newcastle loses her wig base, your reputation will be utterly ruined, won't it?

## 30. rw_015752 · source=gutenberg · risk=152 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 14 >= current rewrite words 14 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
Should we not have captured this very galleon had we come but eleven years ago?

Current inherited rewrite:
Wouldn't we have seized this galleon if we had arrived just eleven years earlier?

Compact candidate:
Wouldn't we have seized this galleon if we had arrived just eleven years earlier?

## 31. rw_018702 · source=bnc_spoken · risk=152 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 22 >= current rewrite words 22 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 2, compact count 0

Original:
rolling over to be tickled but before you could tickle it, it was flipping right over and then jumping up wasn't it?

Current inherited rewrite:
You tried to tickle it while it rolled over, but it quickly flipped back onto its feet and jumped up, didn't it?

Compact candidate:
You tried to tickle it while it rolled over, but it quickly flipped back onto its feet and jumped up, didn't it?

## 32. rw_019977 · source=gutenberg · risk=152 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened

Risk reasons: compact words 33 >= current rewrite words 33 | conditional_force marker preservation 0.00; original count 2, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
When Miss Vincent told us that story yesterday, I couldn't help thinking of Cordelia, and that we might be on the wrong track with _her_, as those horrid girls were with Miss Vincent." "'Those horrid girls'!

Current inherited rewrite:
Upon hearing Miss Vincent recount that tale yesterday, I was reminded of Cordelia and started to doubt whether our approach toward her was misguided, just as the dreadful girls had treated Miss Vincent.

Compact candidate:
Upon hearing Miss Vincent recount that tale yesterday, I was reminded of Cordelia and started to doubt whether our approach toward her was misguided, just as the dreadful girls had treated Miss Vincent.

## 33. rw_024851 · source=gutenberg · risk=152 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 16 >= current rewrite words 16 | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 2, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
It can't be true--but how would he ever think of such a story?

Current inherited rewrite:
Though impossible to believe, one wonders how he could have come up with such a tale.

Compact candidate:
Though impossible to believe, one wonders how he could have come up with such a tale.

## 34. rw_030880 · source=gutenberg · risk=152 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened

Risk reasons: compact words 32 >= current rewrite words 32 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
She was such a pretty little girl, "fair, fair, with" not "golden," I should rather say, "silvern hair," so very pale were the soft silky locks that clustered round her little head.

Current inherited rewrite:
Her soft, silky hair was so pale it seemed silver rather than golden, clustering round her little head with a fair beauty that deserved the description "fair, fair, with" rather than "golden."

Compact candidate:
Her soft, silky hair was so pale it seemed silver rather than golden, clustering round her little head with a fair beauty that deserved the description "fair, fair, with" rather than "golden."

## 35. rw2s1_000371 · source=gutenberg · risk=150 · saved=22

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, ordered_navigation, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
Brennus chose out the hardiest of his mountaineers, and directed them to climb up in the dead of night, one by one, in perfect silence, and thus to surprise the Romans, and complete the slaughter and victory, before the forces assembling at Veii would come to their rescue.

Current inherited rewrite:
Brennus selected the toughest of his mountaineers and ordered them to ascend one by one in absolute silence during the dead of night, aiming to surprise the Romans, finish the slaughter and secure victory before the forces gathering at Veii could arrive to rescue them.

Compact candidate:
Brennus chose the hardiest mountaineers to climb alone in silence at night, surprising Romans, completing slaughter and victory before Veii forces rescued them.

## 36. rw_022526 · source=gutenberg · risk=150 · saved=12

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_partly_changed, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_current_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | compact-current Jaccard 0.15

Original:
All the churches in England were now told by the king what they should do; the pope no longer had anything to say in the matter; the English churches obeyed the king, not the pope.

Current inherited rewrite:
With the king dictating the actions of every church in England and the pope silenced on the matter, ecclesiastical authority had shifted so that English clergy now answered to the monarch rather than the pontiff.

Compact candidate:
The king told all English churches what to do; the pope had nothing to say; English churches obeyed the king, not the pope.

## 37. rw2s1_014511 · source=gutenberg · risk=150 · saved=8

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
I bore a day and a half of his silence and neglect; then I could endure it no longer, and showed him the letter.

Current inherited rewrite:
I endured his silence and neglect for a day and a half, but then I could bear it no longer and showed him the letter.

Compact candidate:
I bore his silence and neglect for a day and a half, then showed him the letter.

## 38. rw_001352 · source=gutenberg · risk=150 · saved=6

Risk tags: comparison_relation_weakened, conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.22

Original:
Most unwillingly was her request granted, for the heart of Eros told him that from their visit no good could come.

Current inherited rewrite:
Though his heart, the vessel of Eros, warned him that their visit would yield nothing but harm, he granted her request with extreme reluctance.

Compact candidate:
Though Eros's heart warned him their visit would yield only harm, he granted her request with extreme reluctance.

## 39. rw_022188 · source=gutenberg · risk=150 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 26 >= current rewrite words 26 | conditional_force marker preservation 0.00; original count 2, compact count 2 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 2, compact count 0 | compact-original Jaccard 0.21

Original:
I wonder if she is really doing it hoping to please Sir John." "Not a bit of it; that would not be Mary's way.

Current inherited rewrite:
I suspected she might be acting solely to curry favor with Sir John, but he assured me that such motives were entirely contrary to Mary's character.

Compact candidate:
I suspected she might be acting solely to curry favor with Sir John, but he assured me that such motives were entirely contrary to Mary's character.

## 40. rw_037283 · source=open_subtitles · risk=150 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 31 >= current rewrite words 31 | conditional_force marker preservation 0.00; original count 1, compact count 2 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 1 | compact-original Jaccard 0.17

Original:
please íf they're not gonna gíve you an award líke "Man of the Year" at least what they could do ís stop havíng you lísted as an ex-convíct whích I thínk ís.

Current inherited rewrite:
If you aren't going to grant you an honor like "Man of the Year," then at the very least, you should remove your status as a convicted felon from their records.

Compact candidate:
If you aren't going to grant you an honor like "Man of the Year," then at the very least, you should remove your status as a convicted felon from their records.

## 41. rw_023230 · source=gutenberg · risk=148 · saved=0

Risk tags: conditional_force_weakened, negation_force_weakened, no_word_saving, ordered_navigation, very_low_source_overlap

Risk reasons: compact words 38 >= current rewrite words 38 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.18

Original:
Wireless telegraphy was too recent an aid to sea-faring to seem real to these simple sailors; this was the first time its workings had touched their lives, and they were not ready to take the burning yacht on faith unseen.

Current inherited rewrite:
For these unsophisticated mariners, wireless telegraphy was such a novel invention that it felt impossible, marking their first encounter with the technology and leaving them unwilling to trust the blazing yacht without seeing its invisible mechanics in action.

Compact candidate:
For these unsophisticated mariners, wireless telegraphy was such a novel invention that it felt impossible, marking their first encounter with the technology and leaving them unwilling to trust the blazing yacht without seeing its invisible mechanics in action.

## 42. rw2s0_027193 · source=gutenberg · risk=145 · saved=11

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
Edna was rather shy of those cousins whom she had not seen for two or three years, and after supper preferred to stay close to her sister Celia and Ben, though her brothers were soon hob-nobbing with Allen and Ted, and were planning expeditions for the morrow.

Current inherited rewrite:
Edna, being somewhat shy of the cousins she hadn't seen in two or three years, chose to remain near her sister Celia and Ben after supper, while her brothers quickly began hob-nobbing with Allen and Ted and were planning expeditions for the morrow.

Compact candidate:
Edna, shy of cousins unseen for two or three years, stayed near sister Celia and Ben after supper, while her brothers quickly hob-nobbed with Allen and Ted, planning expeditions for the morrow.

## 43. rw_017192 · source=gutenberg · risk=54 · saved=12

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
Some of these captives were unwilling to leave the society of the red men; some positively refused to accept the boon of what was called freedom.

Current inherited rewrite:
While a few of the captives did not wish to part with the company of the Red men, others staunchly rejected the so-called gift of liberty.

Compact candidate:
Some captives refused to leave the red men's society; others rejected the so-called freedom.

## 44. rw2s0_031522 · source=gutenberg · risk=54 · saved=9

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
Now _he_ said that there were just as many inhabitants as he had nails in his body; and he was very proud.

Current inherited rewrite:
Now _he_ stated that the number of inhabitants equaled the number of nails in his body, and he was very proud.

Compact candidate:
Now he said inhabitants equaled his body nails; he was very proud.

## 45. rw2s1_026120 · source=gutenberg · risk=54 · saved=5

Risk tags: comparison_relation_partly_changed, conditional_force_weakened, conditional_lost_from_both_references, modality_force_partly_changed

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
I thought you would rather have Edna here with you than elsewhere, and at such a crowded time we have to stow away as we can.

Current inherited rewrite:
I thought you would prefer Edna here with you rather than elsewhere, and given how crowded it is, we must make do with whatever space we can find.

Compact candidate:
I thought you'd prefer Edna here with you than elsewhere, and at such a crowded time we must stow away as we can.

## 46. rw2s0_007282 · source=gutenberg · risk=54 · saved=0

Risk tags: conditional_force_partly_changed, modality_force_partly_changed, no_word_saving, temporal_order_partly_changed

Risk reasons: compact words 32 >= current rewrite words 32

Original:
Still, the way was rugged and circuitous, the Persians would hardly descend before midday, and there was ample time for the Greeks to escape before they could be shut in by the enemy.

Current inherited rewrite:
Still, the route remained rugged and winding, meaning the Persians would not reach the Greeks until after midday, leaving sufficient time for the Greeks to flee before being trapped by the enemy.

Compact candidate:
Still, the route remained rugged and winding, meaning the Persians would not reach the Greeks until after midday, leaving sufficient time for the Greeks to flee before being trapped by the enemy.

## 47. rw2s1_014485 · source=simple_wiki · risk=54 · saved=0

Risk tags: comparison_relation_partly_changed, dialogue_or_question, no_word_saving, temporal_order_partly_changed

Risk reasons: compact words 25 >= current rewrite words 25

Original:
In 1826, she published her first paper, "On the Magnetizing Power of the More Refrangible Solar Rays." She later published two more papers and two books.

Current inherited rewrite:
In 1826, she published her first paper, "On the Magnetizing Power of the More Refrangible Solar Rays," followed by two additional papers and two books.

Compact candidate:
In 1826, she published her first paper, "On the Magnetizing Power of the More Refrangible Solar Rays," followed by two additional papers and two books.

## 48. rw_004327 · source=gutenberg · risk=54 · saved=0

Risk tags: dialogue_or_question, no_word_saving, very_low_current_overlap

Risk reasons: compact words 19 >= current rewrite words 19 | compact-current Jaccard 0.14

Original:
He says if I go I shall not come back; but I do not care, I cannot stay away.

Current inherited rewrite:
Despite his warning that leaving means never returning, I am indifferent to the consequence, for I cannot stay away.

Compact candidate:
He says if I go I shall not come back; but I do not care, I cannot stay away.

## 49. rw_023518 · source=gutenberg · risk=54 · saved=0

Risk tags: dialogue_or_question, no_word_saving, very_low_current_overlap

Risk reasons: compact words 21 >= current rewrite words 20 | compact-current Jaccard 0.11

Original:
But these two cubs o' mine," and he eyed his boys with determination, "has got to give up evil ways right off.

Current inherited rewrite:
And look at my two boys," he declared resolvefully, gazing at them intently, "they must abandon their wicked ways immediately.

Compact candidate:
But these two cubs of mine," he eyed his boys with determination, "has got to give up evil ways right off.

## 50. rw2s0_004196 · source=gutenberg · risk=53 · saved=13

Risk tags: modal_lost_from_both_references, modality_force_weakened, very_low_source_overlap

Risk reasons: modality_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
I'm sure I shall go to sleep in the middle of it, and naturally, too, for even writing to you is enough to bore anybody.

Current inherited rewrite:
I am certain I will fall asleep in the middle of it, and naturally so, for even writing to you is enough to bore anybody.

Compact candidate:
I'm sure I'll fall asleep mid-writing, naturally, for even writing bores anyone.

## 51. rw2s1_005160 · source=childes · risk=53 · saved=10

Risk tags: negation_force_weakened, very_low_source_overlap

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.16

Original:
*MOT: common sense is what makes us safe by telling us what to do in situations that are not covered by rules!

Current inherited rewrite:
*MOT: Common sense ensures our safety by guiding our actions in scenarios that fall outside the scope of established rules!

Compact candidate:
Common sense ensures safety by guiding actions outside established rules!

## 52. rw2s0_028532 · source=simple_wiki · risk=53 · saved=9

Risk tags: causal_or_contrast_force_partly_changed, dialogue_or_question, negation_force_weakened

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
Cavour, not consulted, resigned on 10 July, while King Victor-Emmanuel II gave his agreement "in a personal capacity", thus leaving the door open to any governmental retraction.

Current inherited rewrite:
On 10 July, Cavour resigned without having been consulted, while King Victor-Emmanuel II granted his agreement "in a personal capacity," thereby leaving the door open to any governmental retraction.

Compact candidate:
On 10 July, unconsulted Cavour resigned while King Victor-Emmanuel II agreed "in a personal capacity," leaving open any governmental retraction.

## 53. rw2s0_029645 · source=gutenberg · risk=53 · saved=9

Risk tags: modality_force_partly_changed, negation_force_weakened, negation_lost_from_both_references

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
“You may not be able to secure a stateroom.” “We will try at any rate,” rejoined the tourist.

Current inherited rewrite:
"We will try at any rate," rejoined the tourist, acknowledging that securing a stateroom may not be possible.

Compact candidate:
"We will try at any rate," rejoined the tourist.

## 54. rw_015032 · source=simple_wiki · risk=53 · saved=9

Risk tags: negation_force_weakened, very_low_source_overlap

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 1 | compact-original Jaccard 0.15

Original:
Both mean that a participant in a competition is not required to compete while most of the other participants are.

Current inherited rewrite:
Neither term implies that a competitor is obligated to continue competing while the majority of others remain in the contest.

Compact candidate:
Neither term implies a competitor must compete while most others do.

## 55. rw_024065 · source=gutenberg · risk=53 · saved=8

Risk tags: causal_or_contrast_force_partly_changed, dialogue_or_question, negation_force_weakened

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 1

Original:
to know his "dearest folks" in such a time as this, but he dared stay away no longer from the crowded gangway, so he said good-by to the man whose path had so strangely crossed his own again.

Current inherited rewrite:
Despite knowing his beloved family during such a time, he felt he could not linger any longer on the busy gangway, so he bid farewell to the man whose path had crossed his own once more in such a strange manner.

Compact candidate:
Knowing his dearest folks in such a time, he dared not stay longer on the crowded gangway, so he said good-by to the man whose path had so strangely crossed his own again.

## 56. rw2s1_016573 · source=gutenberg · risk=53 · saved=7

Risk tags: causal_or_contrast_force_partly_changed, dialogue_or_question, negation_force_weakened

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
Upon entering, the cadets do not take seats, but stand behind their chairs, and await the order, "Company A, take seats!" "Company B, take seats!" and so on.

Current inherited rewrite:
Upon entering, the cadets stand behind their chairs rather than taking seats, awaiting the order, "Company A, take seats!" "Company B, take seats!" and so on.

Compact candidate:
Upon entering, cadets stand behind chairs awaiting orders: "Company A, take seats!" "Company B, take seats!" and so on.

## 57. rw_020166 · source=simple_wiki · risk=53 · saved=7

Risk tags: causal_or_contrast_force_weakened, modality_force_partly_changed, severe_summary

Risk reasons: compact length 7 words, original 14 words | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
Tiny houses can stand on a fixed place, but they can also be mobile.

Current inherited rewrite:
While tiny homes may be permanently situated, they are also capable of being moved.

Compact candidate:
Tiny houses can be fixed or mobile.

## 58. rw_024287 · source=gutenberg · risk=53 · saved=6

Risk tags: conditional_force_partly_changed, modality_force_partly_changed, negation_force_weakened

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
He would not consent to any visits, for he feared Mabel's presence would recall Aldred's memory of the fire, and he particularly wished to keep her from all excitement.

Current inherited rewrite:
Because he was determined to spare Mabel from any excitement and feared that her presence might trigger Aldred's recollection of the fire, he refused to allow any visits.

Compact candidate:
He refused visits because he feared Mabel's presence would recall Aldred's memory of the fire and wished to keep her from excitement.

## 59. rw_029811 · source=open_subtitles · risk=53 · saved=6

Risk tags: causal_or_contrast_force_partly_changed, dialogue_or_question, negation_force_weakened

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
Since time is so clearly against us, why don't you show Larita up to the nursery?

Current inherited rewrite:
Given that we are running out of time, could you please take Larita to the nursery?

Compact candidate:
Since time is against us, show Larita to the nursery?

## 60. rw_015625 · source=simple_wiki · risk=53 · saved=5

Risk tags: negation_force_weakened, very_low_source_overlap

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
received criticism for being too exaggerated and not entirely suitable for the target audience of the Pokémon anime, which is children.

Current inherited rewrite:
It faced backlash for its excessive exaggeration and perceived lack of suitability for the Pokémon anime's primary demographic, young children.

Compact candidate:
It faced backlash for excessive exaggeration and unsuitability for the Pokémon anime's young child audience.

## 61. rw_022200 · source=gutenberg · risk=53 · saved=5

Risk tags: negation_force_weakened, very_low_source_overlap

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.16

Original:
France has perhaps no very definite reason for opposition, save the upholding of the rights of the Latin Church.

Current inherited rewrite:
France's opposition lacks a clear rationale other than its commitment to safeguarding the rights of the Latin Church.

Compact candidate:
France's opposition lacks a clear rationale other than safeguarding the Latin Church's rights.

## 62. rw_002570 · source=open_subtitles · risk=53 · saved=3

Risk tags: modality_force_partly_changed, negation_force_weakened, negation_lost_from_both_references

Risk reasons: negation_force marker preservation 0.00; original count 1, compact count 0

Original:
All misunderstandings can be solved and you don't even need a lawyer.

Current inherited rewrite:
No misunderstandings require legal intervention, as every one can be resolved.

Compact candidate:
All misunderstandings can be solved without a lawyer.

## 63. rw_028821 · source=simple_wiki · risk=0 · saved=4

Risk tags: none

Original:
It confronted the North Atlantic Treaty Organisation in Western Europe and helped to keep the Soviet sphere of influence in eastern Europe.

Current inherited rewrite:
By facing the North Atlantic Treaty Organisation in Western Europe, it successfully maintained the Soviet sphere of influence in Eastern Europe.

Compact candidate:
It confronted NATO in Western Europe and helped keep the Soviet sphere of influence in Eastern Europe.

## 64. rw2s0_019197 · source=open_subtitles · risk=0 · saved=3

Risk tags: none

Original:
The little bit of plastic comes out with the first slug and the shells go in the bag.

Current inherited rewrite:
The first slug expels the small amount of plastic while the shells are placed into the bag.

Compact candidate:
The first slug expels the small plastic bit while shells go in the bag.

## 65. rw2s1_016214 · source=simple_wiki · risk=0 · saved=6

Risk tags: none

Original:
Serena received positive feedback for her character portrayal in the anime, which was different from her depiction in the video games.

Current inherited rewrite:
Serena garnered positive feedback for her character portrayal in the anime, a depiction that differed from her representation in the video games.

Compact candidate:
Serena received positive feedback for her anime character portrayal, which differed from her video game depiction.

## 66. rw2s1_003227 · source=gutenberg · risk=0 · saved=4

Risk tags: none

Original:
Loring who had been vainly striving to suppress her emotions, burst into tears, and Guy who was dreadfully shocked and alarmed, cried with her.

Current inherited rewrite:
Loring, who had vainly tried to suppress her emotions, burst into tears, while Guy, who was dreadfully shocked and alarmed, cried with her.

Compact candidate:
Loring, vainly trying to suppress her emotions, burst into tears, while Guy, dreadfully shocked and alarmed, cried with her.

## 67. rw2s1_006636 · source=simple_wiki · risk=0 · saved=3

Risk tags: none

Original:
She wrote multiple works about Muslim feminism, like “Feminine Participation in Islamic Affairs.” Her work reached many people and changed their ideas on Arab-Americans, Muslims, and Islamic feminism.

Current inherited rewrite:
Her work, which included multiple pieces on Muslim feminism such as "Feminine Participation in Islamic Affairs," reached many people and changed their ideas on Arab-Americans, Muslims, and Islamic feminism.

Compact candidate:
She wrote multiple works on Muslim feminism, like "Feminine Participation in Islamic Affairs," reaching many people and changing their ideas on Arab-Americans, Muslims, and Islamic feminism.

## 68. rw_016490 · source=simple_wiki · risk=12 · saved=6

Risk tags: dialogue_or_question

Original:
The largest businesses registered in Mārupe in 2020 were: road transport company "Kreiss", auto parts saler "Inter Cars Latvija", computers saler "Also Latvia", electricals and computers saler "MoonCom".

Current inherited rewrite:
In 2020, the largest registered enterprises in Mārupe were the road transport firm "Kreiss," the auto parts retailer "Inter Cars Latvija," the computer merchant "Also Latvia," and the electronics and computer dealer "MoonCom."

Compact candidate:
In 2020, Mārupe's largest registered enterprises were road transport firm "Kreiss," auto parts retailer "Inter Cars Latvija," computer merchant "Also Latvia," and electronics and computer dealer "MoonCom."

## 69. rw_041841 · source=open_subtitles · risk=0 · saved=8

Risk tags: none

Original:
Another problem is that the people that Ö the criticism isnít just our books, itís so many books.

Current inherited rewrite:
Another issue is that the criticism directed at us extends beyond just our books to encompass a vast number of other titles.

Compact candidate:
Another problem is that the criticism isn't just our books, it's so many books.

## 70. rw_028927 · source=simple_wiki · risk=0 · saved=4

Risk tags: none

Original:
In 2016, WWE created the Cruiserweight Classic, a 32-man tournament for cruiserweights.

Current inherited rewrite:
In 2016, WWE launched the Cruiserweight Classic, an event featuring a 32-person tournament specifically for cruiserweights.

Compact candidate:
In 2016, WWE created the Cruiserweight Classic, a 32-man tournament for cruiserweights.

## 71. rw_037921 · source=open_subtitles · risk=12 · saved=7

Risk tags: dialogue_or_question

Original:
i told them those counterfeit water credits jumped right into my pockets all by themselves.

Current inherited rewrite:
I mentioned that the fake water credits somehow ended up in my pockets on their own.

Compact candidate:
I said counterfeit water credits jumped into my pockets.

## 72. rw2s1_032778 · source=simple_wiki · risk=0 · saved=3

Risk tags: none

Original:
These are not designed to be used on highways and are capable of traveling cross-country over rough land, marsh, swamps, ice or snow.

Current inherited rewrite:
These are not intended for highway use but are capable of traversing cross-country over rough terrain, marsh, swamps, ice, or snow.

Compact candidate:
These are not for highway use but can traverse cross-country over rough terrain, marsh, swamps, ice, or snow.

## 73. rw_016588 · source=gutenberg · risk=0 · saved=1

Risk tags: none

Original:
When a building is being put up nowadays, men use derricks and cranes and engines to haul and raise heavy stones and beams.

Current inherited rewrite:
Modern construction relies on engines, cranes, and derricks to lift and transport massive beams and stones during the erection of buildings.

Compact candidate:
Modern construction uses engines, cranes, and derricks to lift and haul heavy stones and beams when putting up buildings nowadays.

## 74. rw2s1_001173 · source=gutenberg · risk=0 · saved=2

Risk tags: none

Original:
The Swan looked then, as now, a most respectable and sedate bird, with its gold ring round its neck, its spur-boots, and its wings stretched out as if to fly.

Current inherited rewrite:
The Swan appeared then, just as it does now, a highly respectable and sedate bird adorned with a gold ring around its neck, spur-boots, and wings extended as though ready to fly.

Compact candidate:
The Swan looked then, as now, a most respectable and sedate bird, with its gold ring round its neck, its spur-boots, and its wings stretched out as if to fly.

## 75. rw2s0_032248 · source=simple_wiki · risk=12 · saved=12

Risk tags: dialogue_or_question

Original:
According to the Köppen Climate Classification system, Ionia has a humid continental climate, abbreviated "Dfb" on climate maps.

Current inherited rewrite:
Under the Köppen Climate Classification system, Ionia is categorized as having a humid continental climate, which is abbreviated as "Dfb" on climate maps.

Compact candidate:
Under Köppen, Ionia has humid continental climate, abbreviated "Dfb" on maps.

## 76. rw2s0_002664 · source=simple_wiki · risk=12 · saved=1

Risk tags: dialogue_or_question

Original:
In March 2019, Corbyn said that he could vote leave in a second referendum, depending on the Brexit deal on offer.

Current inherited rewrite:
In March 2019, Corbyn stated that he could vote leave in a second referendum, depending on the Brexit deal on offer.

Compact candidate:
In March 2019, Corbyn said he could vote leave in a second referendum, depending on the Brexit deal on offer.

## 77. rw2s1_002054 · source=simple_wiki · risk=0 · saved=5

Risk tags: none

Original:
After the conquest of the Ebro valley, a considerable number of Muslims became subjects of the king of Aragon.

Current inherited rewrite:
Following the conquest of the Ebro valley, a considerable number of Muslims became subjects of the king of Aragon.

Compact candidate:
After conquering the Ebro valley, many Muslims became subjects of the king of Aragon.

## 78. rw_023830 · source=open_subtitles · risk=0 · saved=5

Risk tags: none

Original:
I came to thank the broker woman who wants to unite us.

Current inherited rewrite:
I have come to express my gratitude to the female broker who seeks to bring us together.

Compact candidate:
I came to thank the broker woman who wants to unite us.

## 79. rw2s0_004450 · source=bnc_spoken · risk=0 · saved=8

Risk tags: none

Original:
It was because the people were used to that kind of society with a secret police and er government inspectors and sending people out to Siberia and putting Oh yes.

Current inherited rewrite:
It was because the people were accustomed to that kind of society with a secret police and er government inspectors and sending people out to Siberia and putting Oh yes.

Compact candidate:
It was because people were used to that society with secret police, government inspectors, sending people to Siberia, and putting Oh yes.

## 80. rw2s1_008112 · source=simple_wiki · risk=0 · saved=3

Risk tags: none

Original:
A Group of Nomadic Xoraxane went from Ottoman-Bulgaria to Persia under the reign of Nader Shah and became Shia Muslim.

Current inherited rewrite:
Under the reign of Nader Shah, a Group of Nomadic Xoraxane traveled from Ottoman-Bulgaria to Persia and became Shia Muslim.

Compact candidate:
Under Nader Shah, a Group of Nomadic Xoraxane went from Ottoman-Bulgaria to Persia and became Shia Muslim.

## 81. rw_011151 · source=simple_wiki · risk=0 · saved=5

Risk tags: none

Original:
Following the Labour Party's defeat at the general election on 7 May 2015, Ed Miliband resigned as its party leader.

Current inherited rewrite:
After losing the general election held on May 7, 2015, Ed Miliband stepped down as the leader of the Labour Party.

Compact candidate:
After losing the May 7, 2015 general election, Ed Miliband stepped down as Labour Party leader.

## 82. rw_028960 · source=childes · risk=12 · saved=2

Risk tags: dialogue_or_question

Original:
*CHI: you know what he he does um grind his way to the north pole right?

Current inherited rewrite:
*CHI: I mean, he just grinds his way all the way to the North Pole, right?

Compact candidate:
*CHI: you know what he does, grind his way to the North Pole, right?

