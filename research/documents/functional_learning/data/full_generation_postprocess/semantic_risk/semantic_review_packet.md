# earlier analysis compact-view semantic review packet

Purpose: distinguish faithful shortening from information-losing summary and altered source meaning before any compact text is admitted to learner training.

## Pilot-level summary

- n_records: 26567
- current_number_length_pass: 22566
- auto_low_risk: 13370
- needs_semantic_review: 12836
- known_altered: 2
- auto_low_risk_saved_words: 114462
- current_number_length_saved_words: 229608

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

## 2. rw_035551 · source=open_subtitles · risk=179 · saved=23

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question, known_altered_claim, modality_force_weakened

Risk reasons: conditional self-attribution converted toward a direct assertion | conditional_force marker preservation 0.00; original count 2, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 1

Original:
To bad Amaru I didn't go crazy and nothing is wrong with me If there is something wrong, that would be me in the past What are you doing Amaru?

Current inherited rewrite:
Amaru, I feel terrible that I didn't lose my mind because nothing is currently wrong with me; if there were a problem, it would stem from who I was in the past. What are you doing, Amaru?

Compact candidate:
Amaru, I didn't go crazy; nothing wrong with me, but past me might be.

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

## 5. rw2s0_016420 · source=gutenberg · risk=237 · saved=27

Risk tags: comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 12 words, original 41 words | conditional_force marker preservation 0.00; original count 5, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | compact-original Jaccard 0.18

Original:
[Illustration: "ALL AT ONCE THERE THE INDIANS WERE."] If they could, the boys would rather have been Indians than anything else, but, as there was really no hope of this whatever, they were willing to be settlers, and fight the Indians.

Current inherited rewrite:
[Illustration: "ALL AT ONCE THERE THE INDIANS WERE."] Although the boys would have preferred to be Indians if they could, since there was really no hope of this whatever, they were willing to be settlers and fight the Indians.

Compact candidate:
The boys wanted to be Indians but settled instead to fight them.

## 6. rw2s0_008600 · source=gutenberg · risk=232 · saved=18

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact/original length ratio 0.36 | conditional_force marker preservation 0.00; original count 3, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25 | compact-original Jaccard 0.21

Original:
Laura feared to fall ill if she worked too hard, and then what would become of this pretty young sister who loved her so tenderly and would not be tempted to leave her?

Current inherited rewrite:
Laura feared that working too hard would make her fall ill, leaving her pretty young sister, who loved her so tenderly and would not be tempted to leave, without care.

Compact candidate:
Laura feared illness from hard work, worrying about her pretty young sister.

## 7. rw_027622 · source=gutenberg · risk=229 · saved=27

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened, severe_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 8 words, original 36 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.20 | compact-current Jaccard 0.08

Original:
Many would prefer it to mine, but I don't like to sail under false colors." "It is a whim of mine," said the gentleman, "but I don't think you will be sorry for acceding to it.

Current inherited rewrite:
The gentleman remarked with a touch of whimsy that he did not believe I would regret complying, even though many would likely prefer his proposal over mine, for I refuse to sail under false colors.

Compact candidate:
Gentleman says acceding to whim is not sorry.

## 8. rw2s1_006744 · source=gutenberg · risk=227 · saved=44

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 12 words, original 54 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.14 | compact-original Jaccard 0.17

Original:
"Won't take me long when I find the people on shore--and about five minutes will fix that engine when I get back here again." He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return.

Current inherited rewrite:
He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return, having declared that finding the people on shore wouldn't take long and that about five minutes would fix the engine when he got back here again.

Compact candidate:
He rowed to shore, fixed the engine in five minutes, then returned.

## 9. rw2s1_021812 · source=gutenberg · risk=227 · saved=13

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 7 words, original 18 words | conditional_force marker preservation 0.00; original count 1, compact count 1 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.19

Original:
"I am sure nothing would go well without me." "Do you, then, live in Rome?" asked Ben, curiously.

Current inherited rewrite:
"Do you, then, live in Rome?" asked Ben, curiously, while expressing the certainty that nothing would go well without me.

Compact candidate:
Ben asked if she lived in Rome.

## 10. rw_021990 · source=gutenberg · risk=221 · saved=23

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, severe_summary, very_low_current_overlap

Risk reasons: compact length 6 words, original 24 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | compact-current Jaccard 0.14

Original:
You need not be anxious; Mildred is too healthy to be upset for more than a few hours!" "But I should try the belladonna!

Current inherited rewrite:
You need not worry, for Mildred is too robust to remain distressed for more than a brief period," he replied, only to add, "I must still administer the belladonna!

Compact candidate:
Mildred is healthy; try the belladonna!

## 11. rw2s1_002337 · source=gutenberg · risk=220 · saved=20

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_weakened

Risk reasons: compact/original length ratio 0.41 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.25

Original:
He was so ill, that in four days he could go no more than six miles; still, even at that pace, he went on and resolutely kept his face towards the Border.

Current inherited rewrite:
He was so ill that within four days he could travel no more than six miles; yet, even at that slow pace, he pressed on and resolutely kept his face towards the Border.

Compact candidate:
He was ill, walked six miles in four days, kept face towards Border.

## 12. rw2s0_003393 · source=gutenberg · risk=218 · saved=29

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, very_low_source_overlap

Risk reasons: compact/original length ratio 0.36 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33 | compact-original Jaccard 0.21

Original:
All the boys may have been like my boy in the Boy's Town, in having each an inward being that was not the least like their outward being, but that somehow seemed to be their real self, whether it truly was so or not.

Current inherited rewrite:
All the boys in the Boy's Town may have been like my boy, in that each possessed an inward being that was not the least like their outward being, but that somehow seemed to be their real self, whether it truly was so or not.

Compact candidate:
All boys in Boy's Town had inward beings unlike their outward selves, seeming real though unknown.

## 13. rw2s1_030546 · source=gutenberg · risk=214 · saved=30

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 12 words, original 42 words | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | compact-original Jaccard 0.20 | compact-current Jaccard 0.14

Original:
First, he gathered a basket of chicken feathers, for his father had told him that a few feathers placed at the roots of the young plant would do more to make it strong and healthy than anything else that could be used.

Current inherited rewrite:
First, he collected a basket of chicken feathers because his father had instructed him that a few feathers placed at the roots of the young plant would do more to make it strong and healthy than anything else that could be used.

Compact candidate:
He gathered chicken feathers for his father said they strengthen young plants.

## 14. rw2s1_017656 · source=gutenberg · risk=214 · saved=27

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.33 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.25; original count 4, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.29

Original:
None of them could have told why they used him so ill, for nobody knew; only, the word had gone out that you were not to mind him, but to mock him and fight him; nobody knew where the word first came from.

Current inherited rewrite:
None of them could explain why they treated him so poorly, as nobody knew the reason; only that the word had gone out to mock and fight him rather than mind him, and nobody knew where the word first came from.

Compact candidate:
Nobody knew why they mocked and fought him, though the word had gone out.

## 15. rw_025661 · source=gutenberg · risk=213 · saved=30

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, ordered_navigation, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact/original length ratio 0.33 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
to settle anything with the Pope as to the Spanish marriage; and he now, with a view to the French one, signed a treaty that all Roman Catholics in England should exercise their religion freely, and should never be required to take any oath contrary thereto.

Current inherited rewrite:
With the Spanish marriage having been settled with the Pope, he now signed a treaty aimed at the French alliance, guaranteeing that all Roman Catholics in England could practice their faith freely and would never be compelled to swear any oath contrary to that right.

Compact candidate:
He signed a treaty for French marriage allowing English Catholics free religion without contrary oaths.

## 16. rw2s1_007240 · source=bnc_spoken · risk=213 · saved=27

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, very_low_source_overlap

Risk reasons: compact length 12 words, original 42 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.20 | compact-original Jaccard 0.16

Original:
So I think I can express some of my neighbours concerns, unless, particularly, not particularly worried about those, but there is a general feeling that things are getting worse, and that they're not safe in their own homes, among the more elderly.

Current inherited rewrite:
So I believe I can articulate some of my neighbours' concerns, though I am not particularly worried about those, yet there is a general sentiment that conditions are deteriorating and that the elderly feel unsafe in their own homes.

Compact candidate:
I think neighbours feel unsafe in homes, especially elderly, as things worsen.

## 17. rw2s1_027079 · source=gutenberg · risk=213 · saved=25

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, ordered_navigation, role_reference_weakened, strong_summary, very_low_source_overlap

Risk reasons: compact/original length ratio 0.34 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.33; original count 3, compact count 1 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33 | compact-original Jaccard 0.21

Original:
[Illustration: BETHLEHEM.] Orpah was persuaded to return and settle down among her kindred, and probably did so from a sense of duty; but Ruth would not leave Naomi, although her mother-in-law gave her one more opportunity to go back to Moab.

Current inherited rewrite:
[Illustration: BETHLEHEM.] Although Orpah was persuaded to return and settle among her kindred, likely out of a sense of duty, Ruth refused to leave Naomi even after her mother-in-law offered her one final chance to go back to Moab.

Compact candidate:
Orpah returned to Moab kindred, but Ruth stayed with Naomi despite her mother-in-law's offer.

## 18. rw2s0_032569 · source=gutenberg · risk=206 · saved=22

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_partly_changed, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 8 words, original 33 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 1 | role/pronoun preservation 0.00 | compact-original Jaccard 0.19 | compact-current Jaccard 0.12

Original:
The princess was, however, this much better than before, even in respect of her passions, that they were not quite so bad, and after one was over, she was really ashamed of it.

Current inherited rewrite:
The princess, however, had improved in this regard compared to before, even concerning her passions, as they were not entirely as severe, and following each episode, she genuinely felt ashamed.

Compact candidate:
Princess improved, passions less bad, ashamed after one.

## 19. rw2s1_000930 · source=gutenberg · risk=205 · saved=31

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened, severe_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 14 words, original 48 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.17 | compact-original Jaccard 0.21 | compact-current Jaccard 0.16

Original:
To get a bite was so much clear gain, and when they had wheedled one from the owner of the bread, they took as large a bite as their mouths could stretch to, and they had neither shame nor regret for their behavior, but mocked his just resentment.

Current inherited rewrite:
To secure a bite represented such a clear gain that, after wheedling one from the owner of the bread, they took as large a bite as their mouths could stretch to, feeling neither shame nor regret for their behavior but instead mocking his just resentment.

Compact candidate:
They gained bread, took large bites, felt no shame, and mocked the owner's resentment.

## 20. rw2s1_030763 · source=gutenberg · risk=203 · saved=24

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, very_low_source_overlap

Risk reasons: compact length 8 words, original 33 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25 | compact-original Jaccard 0.20

Original:
Vane was away; no other friend appeared, and no one remembered to invite her, so she bravely hid her girlish longing, and got all the pleasure out of the rehearsals that she could.

Current inherited rewrite:
Since Vane was away, no other friend appeared and no one remembered to invite her, so she bravely hid her girlish longing and extracted all the pleasure she could from the rehearsals.

Compact candidate:
Vane hid her longing and enjoyed rehearsals alone.

## 21. rw2s1_013769 · source=gutenberg · risk=202 · saved=12

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, temporal_order_weakened, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.33 | compact-original Jaccard 0.17

Original:
They debated the matter for two days; and then, as they would not give him all he asked without promise or inquiry, he dissolved them.

Current inherited rewrite:
After debating the matter for two days, he dissolved them because they would not grant him all he asked without promise or inquiry.

Compact candidate:
He dissolved them after two days of debate over his demands.

## 22. rw2s1_015912 · source=gutenberg · risk=200 · saved=37

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_weakened, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact/original length ratio 0.30 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.00 | compact-original Jaccard 0.17 | compact-current Jaccard 0.12

Original:
For a moment the horsemen were too astonished to move; then, recovering from their surprise, they lowered their murderous-looking lances, and would undoubtedly have run all three prisoners through, had not another officer ridden into the circle at that moment.

Current inherited rewrite:
For a brief instant the horsemen were so stunned by the sight that they could not move, but once they recovered from their shock, they lowered their menacing lances and would certainly have pierced all three prisoners had another officer not ridden into the circle at that very moment.

Compact candidate:
Horsemen lowered lances, ready to kill prisoners, until another officer rode in.

## 23. rw2s0_003458 · source=gutenberg · risk=200 · saved=22

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_partly_changed

Risk reasons: compact/original length ratio 0.37 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.33; original count 3, compact count 1 | role/pronoun preservation 0.33

Original:
Mansfield was at first so much startled at seeing her brother that she could find no words to reply, but now they came in what in Ireland might be called not only a flow but a rapid torrent.

Current inherited rewrite:
Mansfield was initially so startled by seeing her brother that she could find no words to reply, but now they came in what in Ireland might be called not only a flow but a rapid torrent.

Compact candidate:
Mansfield was startled seeing her brother, but now words came in a rapid torrent.

## 24. rw_034497 · source=bnc_spoken · risk=199 · saved=11

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary

Risk reasons: compact length 4 words, original 16 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
Wednesday, but if I give it you Tuesday we can be organized for Wednesday can't we?

Current inherited rewrite:
If I provide it to you by Tuesday, we'll be organized for Wednesday, won't we?

Compact candidate:
Tuesday gives Wednesday organization.

## 25. rw_008287 · source=gutenberg · risk=198 · saved=23

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 9 words, original 35 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.21

Original:
faith that gave the earlier Romans such truth and resolution; but latterly they so corrupted it with the Greek myths, that, in after times, they did not even know who the gods of Decius were.

Current inherited rewrite:
The faith that once inspired early Romans with truth and resolve was later corrupted by Greek mythology, a decline so severe that subsequent generations could no longer identify the gods of Decius.

Compact candidate:
Romans corrupted faith with Greek myths, forgetting Decius's gods.

## 26. rw_012099 · source=gutenberg · risk=197 · saved=33

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, negation_force_weakened, role_reference_weakened, strong_summary, very_low_source_overlap

Risk reasons: compact/original length ratio 0.33 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00 | compact-original Jaccard 0.18

Original:
It is awfully plain--pease pudding and herrings mostly; but I don't mind that if only you'd pay me ten shillings a week and let me come to you every day." "You are the most audacious girl!

Current inherited rewrite:
This fare is dreadfully simple, consisting mainly of pease pudding and herrings, yet I would accept it gladly if you'd hire me for ten shillings a week to work daily at your place," she declared, to which he replied, "You are the most audacious girl!

Compact candidate:
She wants ten shillings weekly to cook pease pudding and herrings daily.

## 27. rw_012525 · source=gutenberg · risk=195 · saved=37

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modality_force_weakened, negation_force_weakened, severe_summary, temporal_order_weakened, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 10 words, original 42 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.15 | compact-current Jaccard 0.08

Original:
Being a wise man as well as a good and kindly one, he saw at once that this life would not be safe for the pretty, impulsive, and tenderly reared girl, left so unprotected in a world full of trials and temptations.

Current inherited rewrite:
Recognizing immediately that the world was fraught with trials and temptations, he understood that the young girl, who had been raised tenderly yet possessed an impulsive nature, would be ill-equipped to navigate it safely; thus, his wisdom and kindness compelled him to see her vulnerability without delay.

Compact candidate:
He saw the girl needed protection from a dangerous world.

## 28. rw2s0_026998 · source=gutenberg · risk=195 · saved=35

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary

Risk reasons: compact length 12 words, original 46 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.25

Original:
In all our journey up to that moment we had seen neither man nor any living thing save only the small animals of the woods, and some few wild cattle that smelt us afar off, and vanished from our sight more quickly than eye could follow.

Current inherited rewrite:
Throughout our entire journey up to that moment, we had seen neither man nor any living creature except for the small animals of the woods and a few wild cattle that smelled us from afar and vanished from our sight more quickly than the eye could follow.

Compact candidate:
We saw only small woods animals and wild cattle that vanished quickly.

## 29. rw_011461 · source=gutenberg · risk=195 · saved=30

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, severe_summary, very_low_current_overlap, very_low_source_overlap

Risk reasons: compact length 7 words, original 33 words | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 2, compact count 0 | role/pronoun preservation 0.20 | compact-original Jaccard 0.19 | compact-current Jaccard 0.11

Original:
was, in a certain way, very high among the boys; they would have despised a thief as he deserved, and I cannot remember one of them who might not have been safely trusted.

Current inherited rewrite:
In a sense, their moral standing among the boys was exceptionally high, as they would have scorned a thief for his actions, and I cannot recall a single one who could not have been trusted without question.

Compact candidate:
They despised thieves and trusted boys safely.

## 30. rw_011558 · source=gutenberg · risk=195 · saved=20

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.39 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
And there all the deathless gods were assembled, and Aphrodite no longer looked upon her who had once been her slave with darkened brows, but smiled upon her as the sun smiles upon a new-born flower.

Current inherited rewrite:
Gathered there were the immortal deities, and Aphrodite, no longer frowning with dark brows toward her former servant, now beamed upon her with the radiant warmth of the sun on a newly bloomed flower.

Compact candidate:
Aphrodite smiled upon her former slave as the sun smiles upon a new-born flower.

## 31. rw2s0_027591 · source=gutenberg · risk=195 · saved=13

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, negation_force_weakened, ordered_navigation, role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
I got up and dressed, but had to lie down again, and thus I spent the day; and when my sisters came in to see me I would not speak to them.

Current inherited rewrite:
I rose and dressed, yet was compelled to recline once more, thereby passing the day, and upon my sisters' arrival to visit me, I refused to speak with them.

Compact candidate:
I got up, dressed, lay down, spent the day, and refused to speak to my sisters.

## 32. rw2s1_005158 · source=gutenberg · risk=194 · saved=31

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.38 | conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
Her mother had told her times enough that it was cowardly to blame inanimate objects for things which we were to blame for ourselves, and Aunt Elizabeth went further and said no one but a person without any wits would abuse a senseless thing for what was his own thoughtlessness or carelessness.

Current inherited rewrite:
Her mother had repeatedly warned her that it was cowardly to blame inanimate objects for things which we were to blame for ourselves, and Aunt Elizabeth went further and said no one but a person without any wits would abuse a senseless thing for what was his own thoughtlessness or carelessness.

Compact candidate:
Her mother said blaming objects was cowardly, and Aunt Elizabeth added only fools abuse senseless things for their own carelessness.

## 33. rw_029286 · source=gutenberg · risk=194 · saved=26

Risk tags: comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.32 | conditional_force marker preservation 0.00; original count 3, compact count 0 | modality_force marker preservation 0.00; original count 2, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
If she had been told that she had been there twenty years, she would have believed it—or twenty minutes—it would have been all the same: except for weariness, time was for her no more.

Current inherited rewrite:
Had she been informed that she had resided there for either twenty years or merely twenty minutes, she would have accepted it as fact, for beyond her weariness, the distinction of time held no meaning for her.

Compact candidate:
She believed twenty years or twenty minutes; time meant only weariness.

## 34. rw2s1_016706 · source=gutenberg · risk=194 · saved=17

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.33 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
"I'm sure I don't know, but I think it must be something about Miss Newman." "Let's ask Miss Eloise if she knows," suggested Dorothy.

Current inherited rewrite:
I'm sure I don't know, but I think it must be something about Miss Newman," said Dorothy, suggesting, "Let's ask Miss Eloise if she knows.

Compact candidate:
Dorothy suggested asking Miss Eloise about Miss Newman.

## 35. rw2s0_015050 · source=gutenberg · risk=194 · saved=14

Risk tags: compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_weakened

Risk reasons: compact/original length ratio 0.39 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 1 | role/pronoun preservation 0.20

Original:
He said he would send you a telegram as soon as he had made arrangements, as there was no good troubling you before.

Current inherited rewrite:
He stated that he would dispatch a telegram to you immediately after making arrangements, since there was no point in troubling you beforehand.

Compact candidate:
He promised to send a telegram after making arrangements.

## 36. rw_026608 · source=gutenberg · risk=194 · saved=9

Risk tags: causal_or_contrast_force_weakened, comparison_relation_weakened, compressed_dialogue, conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, modal_lost_from_both_references, modality_force_weakened, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact length 4 words, original 12 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
"Mother!" he exclaimed, but before he could say more she interrupted him.

Current inherited rewrite:
She interrupted him before he could utter another word, having just exclaimed, 'Mother!'

Compact candidate:
Mother! she interrupted him.

## 37. rw2s0_007004 · source=gutenberg · risk=193 · saved=11

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, severe_summary, temporal_order_lost_from_both_references, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact length 7 words, original 19 words | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
Against the power of this new invention the walls of Constantinople could not stand, and finally the city fell.

Current inherited rewrite:
The walls of Constantinople could not withstand the power of this new invention, and finally the city fell.

Compact candidate:
New invention caused Constantinople walls to fall.

## 38. rw2s0_031866 · source=gutenberg · risk=190 · saved=24

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.31 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Then it was made the exact right heat, and Madam and her three daughters--for nothing would keep these old young ladies a minute longer out of the room--superintended the washing and dressing of little Margot.

Current inherited rewrite:
Then the heat was adjusted to the exact right level, and Madam along with her three daughters—who would not stay out of the room for a minute longer—oversaw the washing and dressing of little Margot.

Compact candidate:
Madam and her three daughters superintended washing and dressing little Margot.

## 39. rw2s0_015217 · source=gutenberg · risk=190 · saved=21

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.41 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.20

Original:
For he has humours as well as other people--not only is he fickle in the extreme, but even _black_ sometimes, and he is then, I can assure you, a most disagreeable visitor.

Current inherited rewrite:
For he possesses humours like other people—not only is he fickle to the extreme, but even _black_ at times, and when he is then, I can assure you, he becomes a most disagreeable visitor.

Compact candidate:
He has humours, is fickle and sometimes black, making him a disagreeable visitor.

## 40. rw2s1_002986 · source=gutenberg · risk=190 · saved=18

Risk tags: causal_or_contrast_force_weakened, comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.36 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.33

Original:
He did not like finding other pursuers so near him who might claim part of the reward, at least, when the search was successfully ended.

Current inherited rewrite:
He disliked the prospect of other pursuers being so close by who might claim a share of the reward, at least, once the search was successfully concluded.

Compact candidate:
He disliked near pursuers claiming reward when search ended.

## 41. rw2s0_023457 · source=gutenberg · risk=190 · saved=17

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, role_reference_weakened, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.25

Original:
It looked tired and grey, and very, very old; and the thin hands crossed on her lap, how shrivelled they were!--they trembled all the time as though they could not keep still.

Current inherited rewrite:
It appeared tired and grey, and very, very old; and the thin hands crossed on her lap, how shrivelled they were!--they trembled all the time as though they could not keep still.

Compact candidate:
It looked tired, grey, and very old; thin hands crossed on her lap trembled constantly.

## 42. rw2s1_008191 · source=gutenberg · risk=190 · saved=15

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: compact/original length ratio 0.42 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 2, compact count 0

Original:
She bowed her head again and pressed on through the drifts, feeling her strength would do no more than get her to this refuge.

Current inherited rewrite:
She bowed her head once more and pressed on through the drifts, feeling her strength would do no more than get her to this refuge.

Compact candidate:
She bowed her head, pressed through drifts, strength reached refuge.

## 43. rw2s0_027440 · source=gutenberg · risk=54 · saved=31

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.38 | role/pronoun preservation 0.00

Original:
The people chose red, white, and blue as their colors and the “Marseillaise” as their national song; and everywhere they marched they carried the tricolor, as they called the three-colored flag, and as they marched they sang the “Marseillaise.” Then began what is called the Reign of Terror, and this is a tale of blood.

Current inherited rewrite:
The people selected red, white, and blue as their colors and the "Marseillaise" as their national song; and as they marched everywhere carrying the tricolor, which they called the three-colored flag, they sang the "Marseillaise," after which began what is called the Reign of Terror, and this is a tale of blood.

Compact candidate:
People chose red, white, blue, "Marseillaise," marched with tricolor flag, sang "Marseillaise," then Reign of Terror began, a tale of blood.

## 44. rw_012254 · source=bnc_spoken · risk=54 · saved=26

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, dialogue_or_question

Risk reasons: comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Any more at one hundred and ten only, one twenty, one thirty one forty, fifty one sixty, seventy one seventy one eighty, the lady's bid sir do you want to come in?

Current inherited rewrite:
The lady is bidding; sir, would you like to enter the auction at one hundred and ten, one hundred twenty, one hundred thirty, one hundred forty, one hundred fifty, one hundred sixty, one hundred seventy, one hundred eighty, or higher?

Compact candidate:
Lady bids one hundred ten to one eighty; do you want to come in?

## 45. rw2s0_012733 · source=gutenberg · risk=54 · saved=25

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
As we were examining the shapeless lump of metal, she said, "It's like a little lump of silver that Miss Blomfield has hanging to her watch chain;" which determined me to have a hole made through the remains of my flat iron, and do the same.

Current inherited rewrite:
As we examined the shapeless lump of metal, she remarked, "It's like a little lump of silver that Miss Blomfield has hanging to her watch chain;" a statement that led me to have a hole made through the remains of my flat iron and do the same.

Compact candidate:
She compared the metal lump to Miss Blomfield's silver watch chain pendant, prompting me to punch a hole in my flat iron.

## 46. rw2s0_025816 · source=gutenberg · risk=54 · saved=25

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.35 | role/pronoun preservation 0.33

Original:
The god of the Norsemen was essentially a god of battles, and we are told by great authorities that Baldur was originally a hero who fought on the earth, and who, in time, came to be deified.

Current inherited rewrite:
According to great authorities, Baldur was originally a hero who fought on the earth and, in time, came to be deified, which aligns with the fact that the god of the Norsemen was essentially a god of battles.

Compact candidate:
Norsemen's battle god Baldur was originally an earth hero who later became deified.

## 47. rw_021955 · source=gutenberg · risk=54 · saved=25

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.34 | role/pronoun preservation 0.25

Original:
"Sure enough, the old prophecy is true, and here we have the great man come at last." By the roadside there chanced to be a poor woman and her two children, who, as the carriage passed, held out their hands and asked for help.

Current inherited rewrite:
Indeed, the ancient prophecy has come to pass, for here stands the great man; yet by the roadside, a destitute mother and her two children were seen as the carriage rolled past, reaching out their hands in plea for aid.

Compact candidate:
The old prophecy is true, and here we have the great man come at last.

## 48. rw_012586 · source=gutenberg · risk=54 · saved=23

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.40 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0

Original:
"A hungry one this," he said, "and very bold." So the meal went on, and when Roland had fed his mother with some pieces of the rich food and had seen her gradually revive, yet another thought came to his baby mind.

Current inherited rewrite:
"Such a ravenous and daring one," he remarked, so the feast continued until Roland had nourished his mother with generous portions of the rich fare and watched her slowly recover, only for another idea to surface in his young mind.

Compact candidate:
Roland fed his mother rich food, saw her revive, then another thought came to his baby mind.

## 49. rw_020981 · source=childes · risk=54 · saved=23

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.40 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
[very hoarse, distorted, possibly because of recording problem; or, speaker may be BR2] %add: Chi *MOT: did you have fun?

Current inherited rewrite:
The speaker is extremely hoarse and speaking with a distorted quality, likely due to a recording issue or because they are a BR2 participant; Chi *MOT asks, "Did you have fun?"

Compact candidate:
Speaker may be BR2; did you have fun?

## 50. rw2s0_023060 · source=gutenberg · risk=54 · saved=22

Risk tags: comparison_lost_from_both_references, comparison_relation_weakened, compressed_dialogue, dialogue_or_question

Risk reasons: comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
At last, when one gruff old gentleman had said to Joan, ‘What language do your Voices speak?’ and when Joan had replied to the gruff old gentleman, ‘A pleasanter language than yours,’ they agreed that it was all correct, and that Joan of Arc was inspired from Heaven.

Current inherited rewrite:
At last, after a gruff old gentleman asked Joan, 'What language do your Voices speak?' and Joan responded to the gruff old gentleman, 'A pleasanter language than yours,' they agreed that it was all correct, and that Joan of Arc was inspired from Heaven.

Compact candidate:
Joan told a gruff old gentleman her Voices spoke a pleasanter language, so they agreed Joan of Arc was inspired from Heaven.

## 51. rw2s0_026880 · source=open_subtitles · risk=54 · saved=22

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.35 | role/pronoun preservation 0.33

Original:
Is there any argument for faith, any challenge to your atheism that has given you pause, that has set you back on your heels where you felt you didnít have a ready answer, etc?

Current inherited rewrite:
Is there any argument for faith, any challenge to your atheism that has given you pause, that has set you back on your heels where you felt you didn't have a ready answer, etc?

Compact candidate:
Has any argument for faith challenged your atheism and made you pause?

## 52. rw2s1_001310 · source=gutenberg · risk=54 · saved=22

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.36 | role/pronoun preservation 0.00

Original:
"It's a wonder to me how you ever come home alive when you go out camping by yourselves." "Oh, we manage somehow," boasted Charlie Jamieson.

Current inherited rewrite:
"It's a wonder to me how you ever come home alive when you go out camping by yourselves," remarked the speaker, to which Charlie Jamieson replied, "Oh, we manage somehow," boasting.

Compact candidate:
Charlie Jamieson boasted they manage somehow when camping alone.

## 53. rw2s1_018027 · source=simple_wiki · risk=54 · saved=21

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.39 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
The name "folktronica" seems to have originated in the British press, although (as with the almost non-term post-rock) it has come to encompass performers and bands that include elements of ambient electronica, folk, jazz, classical and even hip-hop.

Current inherited rewrite:
Although "folktronica" appears to have originated in the British press, it has come to encompass performers and bands that include elements of ambient electronica, folk, jazz, classical and even hip-hop, much like the almost non-term post-rock.

Compact candidate:
Folktronica originated in British press, encompassing ambient electronica, folk, jazz, classical, hip-hop bands like post-rock.

## 54. rw_023795 · source=simple_wiki · risk=54 · saved=21

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.33 | role/pronoun preservation 0.00

Original:
Her first role was in 1960 in the television series "Inside Story." Reid is known for her appearances in several British television shows including "Doctor Who", "Casualty" and "Midsomer Murders".

Current inherited rewrite:
Reid first appeared on screen in the 1960 television series "Inside Story" and is well known for her roles in various British shows such as "Doctor Who", "Casualty", and "Midsomer Murders".

Compact candidate:
Reid's first role was 1960 in "Inside Story" TV series.

## 55. rw_007694 · source=open_subtitles · risk=54 · saved=20

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
When Rachel first told me that she was dating Harvey Dent, I had one thing to say: "The guy that was cut off from those awful campaign commercials, I believe in Harvey Dent?" Yeah nice slogan, Harvey.

Current inherited rewrite:
Upon learning that Rachel was now with Harvey Dent, specifically the man excluded from the negative campaign ads, I couldn't help but retort, "You ask me if I believe in Harvey Dent? That's certainly a catchy slogan, Harvey."

Compact candidate:
Rachel told me she dated Harvey Dent, and I said, "I believe in Harvey Dent?" Nice slogan, Harvey.

## 56. rw_012665 · source=simple_wiki · risk=54 · saved=20

Risk tags: compressed_dialogue, conditional_force_weakened, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
Upon his fathers death in 839 he became king of Dalriada styled "Cinaed, son of Alpin" and also king of Scots due to the fact that the Hebrides - Argyll were important parts of the land at the time.

Current inherited rewrite:
Following the death of his father in 839, he ascended to the throne of Dalriada as "Cinaed, son of Alpin" and simultaneously became king of Scots, given that the Hebrides and Argyll formed essential portions of the realm at that era.

Compact candidate:
Upon his father's death in 839 he became king of Dalriada styled "Cinaed, son of Alpin" and also king of Scots.

## 57. rw_013690 · source=simple_wiki · risk=54 · saved=20

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.41 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
Geils Band was influenced by soul music and rhythm and blues, but it moved toward pop and rock by the time the album "Love Stinks" (EMI, 1980) came out.

Current inherited rewrite:
By the release of their 1980 album "Love Stinks" on EMI, The Geils Band had transitioned from their soul and rhythm and blues roots to a sound rooted in pop and rock.

Compact candidate:
Geils Band moved toward pop and rock by 1980's "Love Stinks" album.

## 58. rw2s1_004107 · source=bnc_spoken · risk=54 · saved=19

Risk tags: comparison_relation_weakened, compressed_dialogue, dialogue_or_question, strong_summary

Risk reasons: compact/original length ratio 0.42 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Lot thirty two is the H M V one O one showing now, Lot thirty two and I have fifty offered for these, five, sixty sixty pounds any more at sixty?

Current inherited rewrite:
The H M V one O one currently on display is Lot thirty two, for which I have received fifty offers at sixty sixty pounds, with no further bids at that price.

Compact candidate:
Lot 32 is HMV 101 showing now; I offered fifty, now sixty pounds.

## 59. rw2s1_005874 · source=bnc_spoken · risk=54 · saved=19

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
First, Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf; the Iraqi News Agency said the warning had come from the country's Air Force Commander.

Current inherited rewrite:
First, the Iraqi News Agency reported that the country's Air Force Commander issued a warning stating that Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf.

Compact candidate:
Iraq threatened missile attacks on Israel and Saudi Arabia if Gulf war breaks out, per Air Force Commander.

## 60. rw2s1_030831 · source=gutenberg · risk=54 · saved=19

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.40 | role/pronoun preservation 0.33

Original:
"We have Life and Puck and Judge and--" "I'll take Life and Puck." She accepted the papers handed to her and settled back in the seat she had behind them.

Current inherited rewrite:
"We have Life and Puck and Judge and--" "I'll take Life and Puck," she replied, accepting the papers handed to her and settling back in the seat she had behind them.

Compact candidate:
She accepted Life and Puck papers and settled back in her seat.

## 61. rw_022344 · source=gutenberg · risk=54 · saved=19

Risk tags: compressed_dialogue, dialogue_or_question, temporal_order_lost_from_both_references, temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
The Count was within doors, said the lackey that opened the great gate of the park to us, and we rode up the avenue of chestnuts, just bourgeoning into leaf, and came after some three furlongs to the house.

Current inherited rewrite:
As the lackey opened the park's great gate, he confirmed that the Count was inside, and we then rode the avenue of chestnut trees, which were just beginning to leaf out, until after three furlongs we reached the house.

Compact candidate:
The Count was inside, said the lackey opening the park gate, and we rode the chestnut avenue to the house.

## 62. rw2s0_021939 · source=gutenberg · risk=54 · saved=18

Risk tags: compressed_dialogue, dialogue_or_question, role_reference_weakened, strong_summary

Risk reasons: compact/original length ratio 0.40 | role/pronoun preservation 0.33

Original:
"We mustn't take any chances." They stopped to let the wagon they had heard pass by, but as it came along Bessie cried out suddenly.

Current inherited rewrite:
"They mustn't take any chances," they said as they halted to allow the wagon they had heard to pass, but Bessie cried out suddenly as it came along.

Compact candidate:
They stopped for the wagon, but Bessie cried out suddenly.

## 63. rw2s0_016105 · source=simple_wiki · risk=0 · saved=10

Risk tags: none

Original:
He wrote all the music and lyrics for Bathory, and on some albums he played all the instruments (guitar, bass, drums, etc.).

Current inherited rewrite:
He composed all the music and lyrics for Bathory, and on certain albums he performed all the instruments (guitar, bass, drums, etc.).

Compact candidate:
He wrote Bathory's music, lyrics, and played instruments like guitar, bass, drums.

## 64. rw_005302 · source=simple_wiki · risk=0 · saved=8

Risk tags: none

Original:
Bedford died on 13 January 2016 in Santa Barbara, California from cancer at the age of 80.

Current inherited rewrite:
Santa Barbara, California, was where Bedford, who had turned 80, passed away on January 13, 2016, due to cancer.

Compact candidate:
Bedford died 13 January 2016 Santa Barbara California cancer age 80.

## 65. rw2s0_004791 · source=gutenberg · risk=0 · saved=9

Risk tags: none

Original:
Mr Rimbolt has almost the same craze as Percy for this librarian of his, and I have really no voice in the matter.

Current inherited rewrite:
Mr Rimbolt shares nearly the same obsession as Percy with this librarian of his, and I truly have no say in the matter.

Compact candidate:
Mr Rimbolt shares Percy's craze for his librarian, with no voice in the matter.

## 66. rw_003485 · source=simple_wiki · risk=0 · saved=7

Risk tags: none

Original:
Grand juries trace their roots back to the Assize of Clarendon, an enactment by Henry II of England in 1166.

Current inherited rewrite:
The Assize of Clarendon, a 1166 law enacted by Henry II of England, serves as the origin of grand juries.

Compact candidate:
Grand juries trace roots to Henry II of England's 1166 Assize of Clarendon.

## 67. rw_023406 · source=simple_wiki · risk=12 · saved=3

Risk tags: dialogue_or_question

Original:
In December 2011, Allmyapps said that they had 2.5 million registered users.

Current inherited rewrite:
As of December 2011, Allmyapps reported having 2.5 million registered users.

Compact candidate:
Allmyapps had 2.5 million users in December 2011.

## 68. rw2s1_030489 · source=simple_wiki · risk=0 · saved=9

Risk tags: none

Original:
He goes down and yanks the mic from Paul to tell them to stop banging on the furnace.

Current inherited rewrite:
He descends and pulls the mic from Paul to instruct them to cease banging on the furnace.

Compact candidate:
He yanks Paul's mic to stop furnace banging.

## 69. rw_002184 · source=simple_wiki · risk=20 · saved=13

Risk tags: temporal_order_weakened

Risk reasons: temporal_order marker preservation 0.00; original count 2, compact count 0

Original:
He played for the San Francisco 49ers between 2006 until 2012 and for the New Orleans Saints from 2013 until 2014.

Current inherited rewrite:
From 2006 to 2012, he was a member of the San Francisco 49ers, followed by his tenure with the New Orleans Saints from 2013 through 2014.

Compact candidate:
He played for the San Francisco 49ers 2006-2012 and New Orleans Saints 2013-2014.

## 70. rw2s0_003839 · source=gutenberg · risk=0 · saved=10

Risk tags: none

Original:
The faithful old horse lived in comfort all the rest of his life, for his master, in caring for him, learned to love him again, and treated him as only a faithful friend should be treated.

Current inherited rewrite:
The faithful old horse lived in comfort for the remainder of his life because his master, who learned to love him again through caring for him, treated him as a faithful friend deserves.

Compact candidate:
The faithful old horse lived in comfort, for his master learned to love him again and treated him as a faithful friend should.

## 71. rw2s0_007754 · source=gutenberg · risk=0 · saved=7

Risk tags: none

Original:
I have put an outside price against some, and the others must be mine at any price--you understand.

Current inherited rewrite:
I have set an outside price on some, and the rest must be mine at any cost—you understand.

Compact candidate:
I priced some outside; others must be mine at any price.

## 72. rw2s1_030933 · source=simple_wiki · risk=0 · saved=5

Risk tags: none

Original:
Society has placed limits on them that put them at a disadvantage, like with other people of color.

Current inherited rewrite:
Society has imposed limits on them that place them at a disadvantage, similar to other people of color.

Compact candidate:
Society limits them, putting them at a disadvantage like other people of color.

## 73. rw_041044 · source=open_subtitles · risk=0 · saved=15

Risk tags: none

Original:
Anyway, he mentioned that His parents' 50th anniversary was next month, And I convinced him to let us cater it.

Current inherited rewrite:
In any case, he told me his parents were celebrating their 50th wedding anniversary the following month, and I persuaded him to allow us to handle the catering.

Compact candidate:
He mentioned parents' 50th anniversary next month, I convinced him to cater it.

## 74. rw_014202 · source=open_subtitles · risk=0 · saved=7

Risk tags: none

Original:
The Katakitai tribe will take revenge until the last of us draws breath!

Current inherited rewrite:
We of the Katakitai tribe will exact our vengeance until our final breath is drawn!

Compact candidate:
Katakitai tribe will take revenge until last breath.

## 75. rw2s0_019821 · source=open_subtitles · risk=0 · saved=7

Risk tags: none

Original:
The idea with the climate swoop was to shut down a power station and stop it emitting dangerous chemicals into the atmosphere.

Current inherited rewrite:
The concept behind the climate swoop involved shutting down a power station to prevent it from releasing hazardous chemicals into the atmosphere.

Compact candidate:
The climate swoop aimed to shut down a power station and stop dangerous chemical emissions.

## 76. rw2s1_027731 · source=gutenberg · risk=20 · saved=7

Risk tags: causal_or_contrast_force_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
We therefore determined to quit that room and raise a barricade against its door that opened into the great hall.

Current inherited rewrite:
We therefore resolved to leave that room and construct a barricade against the door which opened into the great hall.

Compact candidate:
We decided to quit the room and raise a barricade against its door.

## 77. rw2s1_022044 · source=simple_wiki · risk=0 · saved=7

Risk tags: none

Original:
Suárez Bértora died on 22 April 2022 in Montevideo, Uruguay from a heart attack at the age of 39.

Current inherited rewrite:
On 22 April 2022, Suárez Bértora passed away at the age of 39 in Montevideo, Uruguay, due to a heart attack.

Compact candidate:
Suárez Bértora died 22 April 2022 in Montevideo, Uruguay from heart attack at 39.

## 78. rw_021416 · source=gutenberg · risk=0 · saved=6

Risk tags: none

Original:
The decks were almost deserted when the skipper of the _Pilgrim_ and Margaret came along very slowly.

Current inherited rewrite:
As Margaret and the skipper of the _Pilgrim_ approached at a leisurely pace, the decks were nearly empty.

Compact candidate:
The decks were deserted when the Pilgrim skipper and Margaret came slowly.

## 79. rw2s0_002236 · source=gutenberg · risk=0 · saved=7

Risk tags: none

Original:
He did not see Effie when she came into the room, but when she bent down and kissed his forehead, he opened his eyes and looked at her.

Current inherited rewrite:
When Effie bent down to kiss his forehead, he opened his eyes and looked at her, having not seen her when she first entered the room.

Compact candidate:
He did not see Effie, but when she kissed his forehead, he opened his eyes and looked at her.

## 80. rw2s1_016302 · source=simple_wiki · risk=0 · saved=5

Risk tags: none

Original:
In 1781, the ship was given to the French, and it was lost off the coast of Madagascar.

Current inherited rewrite:
In 1781, the ship was handed over to the French and subsequently lost off the coast of Madagascar.

Compact candidate:
In 1781, the ship went to the French and was lost off Madagascar.

## 81. rw_005967 · source=gutenberg · risk=0 · saved=14

Risk tags: none

Original:
The rider stares and gasps with astonishment when all he sees is a handful of kilted men standing to arms upon the sloping grass leading to the harbour.

Current inherited rewrite:
Amidst the sloping grass that descends toward the harbour, the rider is left staring and gasping in awe at the sight of a small group of kilted men standing ready for battle.

Compact candidate:
The rider stares and gasps at kilted men standing to arms on sloping grass leading to the harbour.

## 82. rw_019785 · source=open_subtitles · risk=0 · saved=6

Risk tags: none

Original:
You might have made plans for the terror attack together with your brother, J.

Current inherited rewrite:
It is possible that you and your brother, J., coordinated plans for the terror attack.

Compact candidate:
You might have planned terror attack with brother J.

