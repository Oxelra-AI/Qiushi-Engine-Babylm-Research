# earlier analysis compact-view semantic review packet

Purpose: distinguish faithful shortening from information-losing summary and altered source meaning before any compact text is admitted to learner training.

## Pilot-level summary

- n_records: 512
- current_number_length_pass: 296
- auto_low_risk: 219
- needs_semantic_review: 78
- known_altered: 2
- auto_low_risk_saved_words: 1432
- current_number_length_saved_words: 2273

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

## 3. rw_001604 · source=bnc_spoken · risk=130 · saved=0

Risk tags: causal_or_contrast_force_weakened, modality_force_weakened, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 41 >= current rewrite words 41 | modality_force marker preservation 0.00; original count 2, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
The situation in the Gulf may now dwarf the Palestinian issue, but no Middle East crisis can be seen in isolation; a thousand days of protest has assured the plight of Palestinians has rarely been off the world agenda.

Current inherited rewrite:
Although the current crisis in the Gulf might overshadow the Palestinian cause, Middle Eastern conflicts cannot be viewed in isolation, as a year of sustained protests has ensured that the suffering of Palestinians remains a constant priority on the global stage.

Compact candidate:
Although the current crisis in the Gulf might overshadow the Palestinian cause, Middle Eastern conflicts cannot be viewed in isolation, as a year of sustained protests has ensured that the suffering of Palestinians remains a constant priority on the global stage.

## 4. rw_025442 · source=gutenberg · risk=125 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, modality_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 53 >= current rewrite words 53 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
I should have been myself under the circumstances—immediately pulled off his hat with his left hand, and cried, ‘God save the Queen!’ Stubbs was cruelly treated; for the marriage never took place after all, though the Queen pledged herself to the Duke with a ring from her own finger.

Current inherited rewrite:
Given the circumstances, I ought to have acted naturally by instantly removing his hat with my left hand and shouting, "God save the Queen!" Stubbs endured harsh treatment because the wedding ultimately never happened, despite the Queen's promise to the Duke, which she symbolized by giving him a ring from her own finger.

Compact candidate:
Given the circumstances, I ought to have acted naturally by instantly removing his hat with my left hand and shouting, "God save the Queen!" Stubbs endured harsh treatment because the wedding ultimately never happened, despite the Queen's promise to the Duke, which she symbolized by giving him a ring from her own finger.

## 5. rw2s1_000216 · source=bnc_spoken · risk=124 · saved=26

Risk tags: causal_or_contrast_force_weakened, compressed_dialogue, dialogue_or_question, negation_force_weakened, negation_lost_from_both_references, ordered_navigation

Risk reasons: negation_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 0

Original:
Gas Prices are going up in March by 7.5%, another Building Society, the Halifax has put up its interest rates, although first time buyers are being spared the increase, and the Chancellor, John Major says the gloomy news isn't over, because inflation isn't coming down as fast as he predicted.

Current inherited rewrite:
In March, gas prices are rising by 7.5%, while another Building Society, the Halifax, has increased its interest rates even though first time buyers are being spared the increase, and the Chancellor, John Major, states that the gloomy news isn't over because inflation isn't coming down as fast as he predicted.

Compact candidate:
In March, gas prices rise 7.5%; Halifax raises rates, sparing first-time buyers; Chancellor John Major says gloomy news continues as inflation falls slower than predicted.

## 6. rw_009884 · source=bnc_spoken · risk=123 · saved=0

Risk tags: causal_or_contrast_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened, very_low_source_overlap

Risk reasons: compact words 40 >= current rewrite words 40 | negation_force marker preservation 0.00; original count 2, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 2 | role/pronoun preservation 0.33 | compact-original Jaccard 0.16

Original:
No Singapore on the way back , didn't get off the plane though, I mean they only re-fuel, your only allowed off for about half an hour.

Current inherited rewrite:
Although the return leg of the journey did not include a stop in Singapore because aircraft are only permitted to disembark passengers for refueling during a brief window of approximately thirty minutes, I did manage to get off the plane.

Compact candidate:
Although the return leg of the journey did not include a stop in Singapore because aircraft are only permitted to disembark passengers for refueling during a brief window of approximately thirty minutes, I did manage to get off the plane.

## 7. rw2s1_023323 · source=gutenberg · risk=122 · saved=0

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 34 >= current rewrite words 34 | modality_force marker preservation 0.00; original count 1, compact count 1 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
"He has just sent off a stud of horses to Dublin for sale, so there isn't one he can offer ye to ride." "I like _you_ very much as a horse, Uncle Fergus," said Margot.

Current inherited rewrite:
"I like _you_ very much as a horse, Uncle Fergus," said Margot, after he had just sent off a stud of horses to Dublin for sale, leaving none he could offer ye to ride.

Compact candidate:
"I like _you_ very much as a horse, Uncle Fergus," said Margot, after he had just sent off a stud of horses to Dublin for sale, leaving none he could offer ye to ride.

## 8. rw_034351 · source=bnc_spoken · risk=122 · saved=0

Risk tags: comparison_relation_weakened, dialogue_or_question, modality_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 33 >= current rewrite words 33 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Three hundred, three twenty can't see you three fifty, three eighty four hundred and twenty four fifty four fifty down there, any more?

Current inherited rewrite:
Do the three hundred three, two hundred thirty, and three hundred fifty units, along with the three hundred eighty, four hundred twenty-four, and four hundred fifty-four, still have visibility on you down there?

Compact candidate:
Do the three hundred three, two hundred thirty, and three hundred fifty units, along with the three hundred eighty, four hundred twenty-four, and four hundred fifty-four, still have visibility on you down there?

## 9. rw2s0_026429 · source=gutenberg · risk=120 · saved=23

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0

Original:
In the New England of those days the general reader had heard a good deal about the Pilgrim Fathers and Salem Witchcraft, and remembered hazily the stories of Hannah Dustin and of Putnam and the wolf, but could not be counted on for much else before the Revolution.

Current inherited rewrite:
Before the Revolution, the general reader in the New England of those days could not be counted on for much else beyond a good deal of hearing about the Pilgrim Fathers and Salem Witchcraft, along with a hazy recollection of the stories of Hannah Dustin and of Putnam and the wolf.

Compact candidate:
Before the Revolution, New England readers knew well of the Pilgrim Fathers and Salem Witchcraft, vaguely recalled Hannah Dustin and Putnam and the wolf, but knew little else.

## 10. rw_010992 · source=simple_wiki · risk=117 · saved=0

Risk tags: comparison_relation_weakened, dialogue_or_question, no_word_saving, ordered_navigation, temporal_order_weakened

Risk reasons: compact words 40 >= current rewrite words 40 | temporal_order marker preservation 0.00; original count 1, compact count 2 | comparison_relation marker preservation 0.00; original count 1, compact count 2

Original:
He was best known for his role as lawyer Lee Baldwin on the soap opera "General Hospital" playing the role from 1965 to 1976, 1977 to 1986, briefly in 1990, and again from 1992 to 2004.

Current inherited rewrite:
Actor Lee Baldwin, who portrayed a lawyer on the soap opera "General Hospital," was most famous for his performance in that role between 1965 and 1976, from 1977 to 1986, briefly in 1990, and once more from 1992 until 2004.

Compact candidate:
Actor Lee Baldwin, who portrayed a lawyer on the soap opera "General Hospital," was most famous for his performance in that role between 1965 and 1976, from 1977 to 1986, briefly in 1990, and once more from 1992 until 2004.

## 11. rw_007978 · source=open_subtitles · risk=115 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 36 >= current rewrite words 36 | conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
You see, a girl only dresses in a smart hat and coat if she intends to go out of an evening, and yet last night was not her night off.

Current inherited rewrite:
Since a woman only wears her smart hat and coat when planning an evening outing, and last night wasn't her day off, her choice of attire suggests she did intend to go out despite the circumstances.

Compact candidate:
Since a woman only wears her smart hat and coat when planning an evening outing, and last night wasn't her day off, her choice of attire suggests she did intend to go out despite the circumstances.

## 12. rw2s0_020973 · source=simple_wiki · risk=110 · saved=20

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, modal_lost_from_both_references, modality_force_weakened, ordered_navigation

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0

Original:
Composers started their work with a tune (theme) and this tune would be developed in different ways: put in different key, changed from a fast to a slow tune, changed from major to minor or from minor to major.

Current inherited rewrite:
Composers began their work with a tune (theme) that would be developed in various ways, such as being placed in a different key, transformed from a fast to a slow tune, or shifted from major to minor or from minor to major.

Compact candidate:
Composers began with a tune (theme) developed in various ways: different key, fast to slow, major to minor, or minor to major.

## 13. rw_031652 · source=bnc_spoken · risk=110 · saved=13

Risk tags: conditional_force_partly_changed, modality_force_weakened, ordered_navigation, role_reference_weakened, temporal_order_weakened

Risk reasons: modality_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.20

Original:
If you're moving also through Banbury, of course we have our usual restrictions; the high street is affected and also if you're moving through finally on to the A422, I would mention, in Warwickshire, the Stratford to Alcester road, that has some temporary traffic lights at Taylor's wood.

Current inherited rewrite:
If your journey also includes Banbury, please be aware of our standard restrictions on the high street, and if you are proceeding onward to the A422 in Warwickshire via the Stratford to Alcester road, note that temporary traffic lights are in place at Taylor's wood.

Compact candidate:
If your journey includes Banbury, our usual high street restrictions apply; if proceeding onward to the A422 in Warwickshire via the Stratford to Alcester road, note temporary traffic lights at Taylor's wood.

## 14. rw2s0_008006 · source=bnc_spoken · risk=108 · saved=28

Risk tags: modal_lost_from_both_references, modality_force_weakened, negation_force_weakened, negation_lost_from_both_references, strong_summary, very_low_source_overlap

Risk reasons: compact/original length ratio 0.42 | modality_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.17

Original:
It does mean that the way in which the case has been handled, the way in which it's been approached, the cooperation between various departments in a particular office, has been done properly, and if the client has the opportunity of winning, he will win and not lose by sloppy work in his solicitor's office.

Current inherited rewrite:
It does mean that the manner in which the case has been handled, the approach taken, the cooperation between various departments in a particular office, has been done properly, and if the client has the opportunity of winning, he will win and not lose by sloppy work in his solicitor's office.

Compact candidate:
It means proper handling, approach, and departmental cooperation ensure the client wins if they have a chance, avoiding loss from sloppy solicitor work.

## 15. rw2s0_000898 · source=gutenberg · risk=107 · saved=0

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, modality_force_weakened, no_word_saving, role_reference_weakened

Risk reasons: compact words 41 >= current rewrite words 41 | modality_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | role/pronoun preservation 0.00

Original:
"I have to be cautious about what I believe." "But surely you will believe me, knowing that I am perfectly sane?" "That is the question to be determined," said M.

Current inherited rewrite:
"That is the question to be determined," said M., in response to the assertion that one would surely believe the speaker, who claimed to be perfectly sane, given the speaker's own admission of needing to be cautious about what they believe.

Compact candidate:
"That is the question to be determined," said M., in response to the assertion that one would surely believe the speaker, who claimed to be perfectly sane, given the speaker's own admission of needing to be cautious about what they believe.

## 16. rw2s1_000389 · source=simple_wiki · risk=107 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, no_word_saving, ordered_navigation

Risk reasons: compact words 43 >= current rewrite words 43 | conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
It says that if three points, formula_1, formula_2, and formula_3 are on a circle where formula_4 is a diameter (a straight line from one end of the circle to another, through the middle), the angle formula_5 is a right angle of 90 degrees.

Current inherited rewrite:
It states that when three points, formula_1, formula_2, and formula_3 lie on a circle with formula_4 as a diameter (a straight line from one end of the circle to another, through the middle), the angle formula_5 measures a right angle of 90 degrees.

Compact candidate:
It states that when three points, formula_1, formula_2, and formula_3 lie on a circle with formula_4 as a diameter (a straight line from one end of the circle to another, through the middle), the angle formula_5 measures a right angle of 90 degrees.

## 17. rw_001499 · source=switchboard · risk=105 · saved=0

Risk tags: causal_or_contrast_force_weakened, negation_force_weakened, no_word_saving, role_reference_weakened

Risk reasons: compact words 22 >= current rewrite words 22 | negation_force marker preservation 0.00; original count 1, compact count 1 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 2 | role/pronoun preservation 0.33

Original:
B: I mean, it's not a lot because you're just doing a house at a time, B: but you know, every little bit helps.

Current inherited rewrite:
B: Although you're only working on one house at a time, so the total isn't huge, every little contribution makes a difference.

Compact candidate:
B: Although you're only working on one house at a time, so the total isn't huge, every little contribution makes a difference.

## 18. rw_013999 · source=childes · risk=105 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving, ordered_navigation, temporal_order_weakened

Risk reasons: compact words 34 >= current rewrite words 34 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 2, compact count 1

Original:
[BRO is referring to his own magnets] *CHI: [picks up the nickel again for a moment, but then puts it down, picks up a nut, and sticks it to the wand] *INV: [chuckles].

Current inherited rewrite:
*CHI briefly picks up the nickel before setting it aside, opting instead to attach a nut to the wand, prompting *INV to chuckle, while BRO acknowledges that the magnets in question belong to him.

Compact candidate:
*CHI briefly picks up the nickel before setting it aside, opting instead to attach a nut to the wand, prompting *INV to chuckle, while BRO acknowledges that the magnets in question belong to him.

## 19. rw_027638 · source=open_subtitles · risk=103 · saved=0

Risk tags: comparison_relation_weakened, conditional_force_partly_changed, modality_force_partly_changed, negation_force_weakened, no_word_saving

Risk reasons: compact words 38 >= current rewrite words 38 | negation_force marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
And clouds really should be white at least, and oceans should meet us at the beach, and I don't want to overreach when I know what's in store, raw love.

Current inherited rewrite:
Even if I fear what lies ahead, I hold onto the simple truths that clouds ought to be white and oceans should welcome us at the shore, all while longing for that raw love without overstepping my bounds.

Compact candidate:
Even if I fear what lies ahead, I hold onto the simple truths that clouds ought to be white and oceans should welcome us at the shore, all while longing for that raw love without overstepping my bounds.

## 20. rw2s0_013580 · source=bnc_spoken · risk=100 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 48 >= current rewrite words 48 | conditional_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
Earnings per share in the period were five p but after taking into account exceptional Carpet Right gain this is reduced to two point eight p and compares to last year's pro forma earnings per share of one point four million which included losses on the closing down of our operations.

Current inherited rewrite:
Earnings per share for the period stood at five p, which falls to two point eight p when the exceptional Carpet Right gain is factored in, compared to last year's pro forma earnings per share of one point four million that incorporated losses from closing down our operations.

Compact candidate:
Earnings per share for the period stood at five p, which falls to two point eight p when the exceptional Carpet Right gain is factored in, compared to last year's pro forma earnings per share of one point four million that incorporated losses from closing down our operations.

## 21. rw2s1_009835 · source=childes · risk=100 · saved=0

Risk tags: navigation_sequence_changed, no_word_saving, ordered_navigation

Risk reasons: compact words 49 >= current rewrite words 49 | direction sequence original=['up', 'up', 'left'] compact=['up', 'left']

Original:
*UNC: he showed up about nine o'clock on his motorcycle and it was still you know kind o' drizzling out and stuff so he took first truck and went home to pick up Mark he came back he dropped some stuff off at Fred's house came back took his motorcycle and left.

Current inherited rewrite:
*UNC: he arrived around nine o'clock on his motorcycle while it was still kind o' drizzling out and stuff, so he took the first truck to go home and pick up Mark, then returned to drop some stuff off at Fred's house, came back, took his motorcycle, and left.

Compact candidate:
*UNC: he arrived around nine o'clock on his motorcycle while it was still kind o' drizzling out and stuff, so he took the first truck to go home and pick up Mark, then returned to drop some stuff off at Fred's house, came back, took his motorcycle, and left.

## 22. rw2s0_011468 · source=childes · risk=98 · saved=0

Risk tags: causal_or_contrast_force_weakened, conditional_force_weakened, no_word_saving, very_low_source_overlap

Risk reasons: compact words 32 >= current rewrite words 32 | conditional_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.21

Original:
*MOT: Peter gave himself up for lost and shed big tears but his sobs were overheard by some friendly sparrows who flew to him in great excitement and implored him to exert himself.

Current inherited rewrite:
*MOT: Peter, having surrendered to despair and wept profusely, was overheard sobbing by a few amiable sparrows who, bursting with enthusiasm, fluttered to his side and urged him to summon his strength.

Compact candidate:
*MOT: Peter, having surrendered to despair and wept profusely, was overheard sobbing by a few amiable sparrows who, bursting with enthusiasm, fluttered to his side and urged him to summon his strength.

## 23. rw_003348 · source=open_subtitles · risk=98 · saved=0

Risk tags: conditional_force_weakened, no_word_saving, temporal_order_weakened, very_low_source_overlap

Risk reasons: compact words 28 >= current rewrite words 28 | conditional_force marker preservation 0.00; original count 1, compact count 0 | temporal_order marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.11

Original:
If you're using half your concentration to look normal, then you're only half paying attention to whatever else you're doing.

Current inherited rewrite:
When you must split your focus between maintaining a facade and engaging with the present moment, your attention to the task at hand is necessarily diminished by half.

Compact candidate:
When you must split your focus between maintaining a facade and engaging with the present moment, your attention to the task at hand is necessarily diminished by half.

## 24. rw_012099 · source=gutenberg · risk=97 · saved=0

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, negation_force_weakened, no_word_saving

Risk reasons: compact words 45 >= current rewrite words 45 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
It is awfully plain--pease pudding and herrings mostly; but I don't mind that if only you'd pay me ten shillings a week and let me come to you every day." "You are the most audacious girl!

Current inherited rewrite:
This fare is dreadfully simple, consisting mainly of pease pudding and herrings, yet I would accept it gladly if you'd hire me for ten shillings a week to work daily at your place," she declared, to which he replied, "You are the most audacious girl!

Compact candidate:
This fare is dreadfully simple, consisting mainly of pease pudding and herrings, yet I would accept it gladly if you'd hire me for ten shillings a week to work daily at your place," she declared, to which he replied, "You are the most audacious girl!

## 25. rw_012306 · source=simple_wiki · risk=97 · saved=0

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, no_word_saving

Risk reasons: compact words 44 >= current rewrite words 44 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0

Original:
He wrote to his manager, Nicholas Lewis, saying to "dispose of Mary according to her desire, with such of her younger children as she chose." He meant that Bell could buy Mary Hemings and his own children but not Mary Hemings' other children.

Current inherited rewrite:
In a letter to Nicholas Lewis, he instructed that Mary Hemings be freed according to her wishes along with any younger children she selected, clarifying that Bell was authorized to purchase Mary and her own offspring but not the rest of Mary Hemings' children.

Compact candidate:
In a letter to Nicholas Lewis, he instructed that Mary Hemings be freed according to her wishes along with any younger children she selected, clarifying that Bell was authorized to purchase Mary and her own offspring but not the rest of Mary Hemings' children.

## 26. rw_037091 · source=childes · risk=97 · saved=0

Risk tags: dialogue_or_question, negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 21 >= current rewrite words 21 | negation_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
[Mother says Eve still gets "Fraser" and "Cromer" confused] *COL: no in Fraser's briefcase.

Current inherited rewrite:
Despite the note clarifying that Fraser's briefcase is not involved, Mother confirms that Eve continues to mix up "Fraser" and "Cromer."

Compact candidate:
Despite the note clarifying that Fraser's briefcase is not involved, Mother confirms that Eve continues to mix up "Fraser" and "Cromer."

## 27. rw2s0_009801 · source=bnc_spoken · risk=95 · saved=0

Risk tags: conditional_force_weakened, no_word_saving, ordered_navigation

Risk reasons: compact words 45 >= current rewrite words 45 | conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
British Airways were up 1 at 198, B P up 2 at 341, British Gas down 1 at 219, British Steel up 1 at 133, British Telecom up 1 at 296, Rolls Royce up 1 at 169 and T S B up 1 at 136.

Current inherited rewrite:
British Airways rose 1 to 198, B P climbed 2 to 341, British Gas fell 1 to 219, British Steel gained 1 to 133, British Telecom increased 1 to 296, Rolls Royce moved up 1 to 169, and T S B advanced 1 to 136.

Compact candidate:
British Airways rose 1 to 198, B P climbed 2 to 341, British Gas fell 1 to 219, British Steel gained 1 to 133, British Telecom increased 1 to 296, Rolls Royce moved up 1 to 169, and T S B advanced 1 to 136.

## 28. rw2s1_013643 · source=simple_wiki · risk=92 · saved=11

Risk tags: dialogue_or_question, number_changed_or_removed, temporal_order_weakened

Risk reasons: one or more source numbers not found in compact text | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
Court of Appeals found that the seizure of 50 eagle feathers during a 2006 Lipan Apache pow wow violated Robert Soto's rights as a "sincere adherent to an American Indian religion" under the Religious Freedom Restoration Act (RFRA) of 1993.

Current inherited rewrite:
Under the Religious Freedom Restoration Act (RFRA) of 1993, the Court of Appeals determined that the seizure of 50 eagle feathers at a 2006 Lipan Apache pow wow infringed upon Robert Soto's rights as a "sincere adherent to an American Indian religion."

Compact candidate:
The Court of Appeals found that seizing 50 eagle feathers at a 2006 Lipan Apache pow wow violated Robert Soto's RFRA rights as a "sincere adherent to an American Indian religion."

## 29. rw_041960 · source=childes · risk=87 · saved=10

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question, negation_force_weakened

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | negation_force marker preservation 0.00; original count 1, compact count 1

Original:
[all sings] [BRO is singing along with Mot; it is difficult to tell whether Chi is singing also; verifier believes she is not] *MOT: xxx Mikey?

Current inherited rewrite:
In the audio clip, BRO joins Mot's singing, while it remains unclear if Chi is participating, leading the verifier to conclude she is not; Mot then asks, "xxx Mikey?"

Compact candidate:
BRO sings with Mot; Chi's participation is unclear, so the verifier thinks she isn't singing; Mot asks, "xxx Mikey?"

## 30. rw_032720 · source=open_subtitles · risk=87 · saved=0

Risk tags: causal_or_contrast_force_weakened, dialogue_or_question, modality_force_weakened, no_word_saving

Risk reasons: compact words 36 >= current rewrite words 36 | modality_force marker preservation 0.00; original count 2, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
But you have already begun to trust him You're reading me like a book The merchant ship will leave Dukjin Harbor, and pass through Palgeum and Sadang waters Where will you attack them?

Current inherited rewrite:
You've already started trusting him; you understand me as clearly as one reads an open book. The merchant vessel is departing Dukjin Harbor to navigate through Palgeum and Sadang waters—now, where do you intend to strike?

Compact candidate:
You've already started trusting him; you understand me as clearly as one reads an open book. The merchant vessel is departing Dukjin Harbor to navigate through Palgeum and Sadang waters—now, where do you intend to strike?

## 31. rw2s1_002259 · source=childes · risk=85 · saved=10

Risk tags: conditional_force_weakened, dialogue_or_question, modality_force_weakened, very_low_source_overlap

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0 | compact-original Jaccard 0.20

Original:
*MOT: looks like it could go in the Museum of Modern Art in New York to me huh?

Current inherited rewrite:
*MOT: In my opinion, this piece seems worthy of a place in the Museum of Modern Art in New York, don't you think?

Compact candidate:
*MOT: This piece seems worthy of MoMA in New York, don't you think?

## 32. rw2s0_002543 · source=gutenberg · risk=85 · saved=0

Risk tags: comparison_relation_weakened, no_word_saving, ordered_navigation

Risk reasons: compact words 52 >= current rewrite words 52 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Edna picked it up but it had no desire to stay when this, of all hours in the day, was the best to play in, so it scrambled down from her arms and was off like a flash, darting half way up a tree, with ears back and claws outspread.

Current inherited rewrite:
Edna picked it up, but since this was the optimal time to play despite being the middle of the day, the creature had no desire to stay, so it scrambled down from her arms and was off like a flash, darting half way up a tree with ears back and claws outspread.

Compact candidate:
Edna picked it up, but since this was the optimal time to play despite being the middle of the day, the creature had no desire to stay, so it scrambled down from her arms and was off like a flash, darting half way up a tree with ears back and claws outspread.

## 33. rw2s1_002293 · source=simple_wiki · risk=85 · saved=0

Risk tags: causal_or_contrast_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 51 >= current rewrite words 51 | negation_force marker preservation 0.00; original count 1, compact count 0 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
In accordance with the Catholic Church's practice of including the Filioque clause when reciting the Creed in Latin, but not when reciting the Creed in Greek, Popes John Paul II and Benedict XVI have recited the Nicene Creed jointly with Patriarchs Demetrius I and Bartholomew I in Greek without the Filioque clause.

Current inherited rewrite:
In line with the Catholic Church's custom of incorporating the Filioque clause during the Latin recitation of the Creed while omitting it during the Greek recitation, Popes John Paul II and Benedict XVI recited the Nicene Creed together with Patriarchs Demetrius I and Bartholomew I in Greek, excluding the Filioque clause.

Compact candidate:
In line with the Catholic Church's custom of incorporating the Filioque clause during the Latin recitation of the Creed while omitting it during the Greek recitation, Popes John Paul II and Benedict XVI recited the Nicene Creed together with Patriarchs Demetrius I and Bartholomew I in Greek, excluding the Filioque clause.

## 34. rw2s1_005328 · source=open_subtitles · risk=85 · saved=0

Risk tags: negation_force_weakened, no_word_saving, temporal_order_weakened

Risk reasons: compact words 29 >= current rewrite words 29 | negation_force marker preservation 0.00; original count 1, compact count 1 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
It's that the UN summit that's taking place in Copenhagen right now is presenting capitalist solutions to something that shouldn't.

Current inherited rewrite:
It is the fact that the UN summit currently taking place in Copenhagen is offering capitalist solutions to an issue that should not be addressed in such a manner.

Compact candidate:
It is the fact that the UN summit currently taking place in Copenhagen is offering capitalist solutions to an issue that should not be addressed in such a manner.

## 35. rw2s1_006744 · source=gutenberg · risk=85 · saved=0

Risk tags: dialogue_or_question, modality_force_partly_changed, negation_force_weakened, no_word_saving

Risk reasons: compact words 56 >= current rewrite words 56 | negation_force marker preservation 0.00; original count 1, compact count 1

Original:
"Won't take me long when I find the people on shore--and about five minutes will fix that engine when I get back here again." He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return.

Current inherited rewrite:
He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return, having declared that finding the people on shore wouldn't take long and that about five minutes would fix the engine when he got back here again.

Compact candidate:
He rowed off into the darkness, making for a point of light that showed on shore, and they settled back to wait as patiently as they could for his return, having declared that finding the people on shore wouldn't take long and that about five minutes would fix the engine when he got back here again.

## 36. rw2s1_018188 · source=gutenberg · risk=85 · saved=0

Risk tags: dialogue_or_question, modality_force_partly_changed, negation_force_weakened, no_word_saving

Risk reasons: compact words 56 >= current rewrite words 56 | negation_force marker preservation 0.00; original count 1, compact count 2

Original:
"We don't want the others to have a hint of what we mean to do," said Phœbe Stanhope; "they mustn't even know the name of our act." "And we must make all our dresses here too," said Myfanwy, "and any wigs, or moustaches, or anything we need." "Shall we have time?" enquired Aldred.

Current inherited rewrite:
"We must ensure the others know not even the name of our act, for they must not have a hint of what we mean to do," said Phœbe Stanhope; "and we must make all our dresses here too," said Myfanwy, "along with any wigs, moustaches, or other necessities we require." "Shall we have time?" enquired Aldred.

Compact candidate:
"We must ensure the others know not even the name of our act, for they must not have a hint of what we mean to do," said Phœbe Stanhope; "and we must make all our dresses here too," said Myfanwy, "along with any wigs, moustaches, or other necessities we require." "Shall we have time?" enquired Aldred.

## 37. rw2s1_031835 · source=gutenberg · risk=85 · saved=0

Risk tags: causal_or_contrast_force_weakened, negation_force_weakened, no_word_saving

Risk reasons: compact words 50 >= current rewrite words 50 | negation_force marker preservation 0.33; original count 3, compact count 2 | causal_or_contrast_force marker preservation 0.00; original count 2, compact count 3

Original:
They may not want to know us, for we shall be very poor; but I won't be patronised by any one, and I don't want them to call.' Vava looked as if she were going to say something, but thought better of it, and gave the desired promise.

Current inherited rewrite:
They may not wish to know us because we shall be very poor; yet I refuse to be patronised by anyone and do not want them to call, and though Vava looked as if she were going to say something, she thought better of it and gave the desired promise.

Compact candidate:
They may not wish to know us because we shall be very poor; yet I refuse to be patronised by anyone and do not want them to call, and though Vava looked as if she were going to say something, she thought better of it and gave the desired promise.

## 38. rw_024644 · source=childes · risk=85 · saved=0

Risk tags: conditional_force_weakened, modality_force_weakened, no_word_saving

Risk reasons: compact words 28 >= current rewrite words 28 | conditional_force marker preservation 0.00; original count 1, compact count 0 | modality_force marker preservation 0.00; original count 1, compact count 0

Original:
[MOT talks to another sibling who is going somewhere; Chi asks if he can go too; EXP talks about the activity] *CHI: this is for Bingo.

Current inherited rewrite:
In a conversation where MOT discusses an outing with another sibling and EXP elaborates on the plans, CHI inquires about joining and mentions the purpose is for Bingo.

Compact candidate:
In a conversation where MOT discusses an outing with another sibling and EXP elaborates on the plans, CHI inquires about joining and mentions the purpose is for Bingo.

## 39. rw2s1_016261 · source=open_subtitles · risk=82 · saved=6

Risk tags: dialogue_or_question, navigation_sequence_changed, ordered_navigation

Risk reasons: direction sequence original=['left', 'down', 'down'] compact=['down']

Original:
Jon called me before he left and was like, "If you come down here, tell me my grandma died." The fake dead grandmother could easily go down as the dirtiest thing ever to be done in this game.

Current inherited rewrite:
Before departing, Jon called me and said, "If you come down here, tell me my grandma died," noting that the fake dead grandmother could easily go down as the dirtiest thing ever to be done in this game.

Compact candidate:
Before leaving, Jon called me and said, "If you come down here, tell me my grandma died," noting the fake dead grandmother could easily be the dirtiest thing ever in this game.

## 40. rw_004416 · source=bnc_spoken · risk=82 · saved=0

Risk tags: comparison_relation_weakened, dialogue_or_question, no_word_saving, temporal_order_weakened

Risk reasons: compact words 33 >= current rewrite words 33 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
I'll sell at fifty pounds all done, fifty five standing standing at fifty five,sixty sixty there now, sixty five seventy seventy pounds on the right, any more?

Current inherited rewrite:
I will sell the completed item for fifty pounds, the standing version for fifty-five, the sixty-six for sixty, the sixty-five and seventy for seventy pounds on the right, with any additional items available?

Compact candidate:
I will sell the completed item for fifty pounds, the standing version for fifty-five, the sixty-six for sixty, the sixty-five and seventy for seventy pounds on the right, with any additional items available?

## 41. rw_009901 · source=bnc_spoken · risk=82 · saved=0

Risk tags: comparison_relation_weakened, dialogue_or_question, no_word_saving, temporal_order_weakened

Risk reasons: compact words 43 >= current rewrite words 43 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
Standing then at one sixty, are you all done, one seventy, one eighty, one ninety two hundred two twenty two twenty on the left standing, any more at two twenty?

Current inherited rewrite:
With those currently standing at one minute sixty, one minute seventy, one minute eighty, one minute ninety, two minutes, two minutes twenty, and two minutes twenty on the left, are you all finished, or is there anyone else remaining at two minutes twenty?

Compact candidate:
With those currently standing at one minute sixty, one minute seventy, one minute eighty, one minute ninety, two minutes, two minutes twenty, and two minutes twenty on the left, are you all finished, or is there anyone else remaining at two minutes twenty?

## 42. rw_025815 · source=simple_wiki · risk=82 · saved=0

Risk tags: comparison_relation_weakened, dialogue_or_question, no_word_saving, temporal_order_weakened

Risk reasons: compact words 51 >= current rewrite words 51 | temporal_order marker preservation 0.00; original count 1, compact count 0 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
One of her most notable quotes are her final words before her execution in August 1852, "You can kill me as soon as you like, but you cannot stop the emancipation of women." Egyptian jurist Qasim Amin wrote the book "Women's Liberation" ("Tahrir al-Mar'a"), in 1899.

Current inherited rewrite:
In 1899, Egyptian jurist Qasim Amin authored the book "Women's Liberation" ("Tahrir al-Mar'a"), a work that stands in contrast to the powerful declaration by a woman executed in August 1852, whose final words were, "You can kill me as soon as you like, but you cannot stop the emancipation of women."

Compact candidate:
In 1899, Egyptian jurist Qasim Amin authored the book "Women's Liberation" ("Tahrir al-Mar'a"), a work that stands in contrast to the powerful declaration by a woman executed in August 1852, whose final words were, "You can kill me as soon as you like, but you cannot stop the emancipation of women."

## 43. rw2s0_006734 · source=bnc_spoken · risk=53 · saved=0

Risk tags: dialogue_or_question, negation_force_partly_changed, no_word_saving

Risk reasons: compact words 40 >= current rewrite words 40

Original:
She said to her in the week when she rung she won't ring until I thought it was like midnight last night she was getting in and she wasn't gonna ring till today but it'll be tomorrow she'll ring her.

Current inherited rewrite:
She told her during the week when she rang that she wouldn't ring until I thought it was like midnight last night she was getting in and she wasn't gonna ring till today but it'll be tomorrow she'll ring her.

Compact candidate:
She told her during the week when she rang that she wouldn't ring until I thought it was like midnight last night she was getting in and she wasn't gonna ring till today but it'll be tomorrow she'll ring her.

## 44. rw2s1_002540 · source=childes · risk=53 · saved=0

Risk tags: dialogue_or_question, negation_force_partly_changed, no_word_saving

Risk reasons: compact words 35 >= current rewrite words 35

Original:
*FAT: Mark that was terrible you wasted milk don't ever do that again milk is very expensive Mark don't ever do that that's wasting food do you see what Mark did?

Current inherited rewrite:
*FAT: Mark, that was terrible; you wasted milk and must never do that again because milk is very expensive, so don't ever do that as it is wasting food; do you see what Mark did?

Compact candidate:
*FAT: Mark, that was terrible; you wasted milk and must never do that again because milk is very expensive, so don't ever do that as it is wasting food; do you see what Mark did?

## 45. rw2s1_004285 · source=open_subtitles · risk=53 · saved=0

Risk tags: dialogue_or_question, negation_force_partly_changed, no_word_saving

Risk reasons: compact words 37 >= current rewrite words 37

Original:
There's no work here if you haven't got a mule At the bazaars, they load up with stuff for Iraq and bring other stuff back Could you give Ayoub a job, my dear Mr Yassin?

Current inherited rewrite:
If you don't have a mule, there is no work here; at the bazaars they load up with goods for Iraq and bring other items back, so could you give Ayoub a job, my dear Mr Yassin?

Compact candidate:
If you don't have a mule, there is no work here; at the bazaars they load up with goods for Iraq and bring other items back, so could you give Ayoub a job, my dear Mr Yassin?

## 46. rw2s1_032596 · source=gutenberg · risk=52 · saved=17

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 2, compact count 0

Original:
The curtains were drawn, the fire blazed brightly, the lamp on the _console_ at the side of the room threw a soft pleasant glow on the dainty table set out temptingly for "afternoon tea," which, notwithstanding their long residence in France, Auntie and her nieces were very fond of.

Current inherited rewrite:
With the curtains drawn and the fire blazing brightly, the lamp on the _console_ at the side of the room cast a soft pleasant glow on the dainty table set out temptingly for "afternoon tea," a treat that Auntie and her nieces, despite their long residence in France, were very fond of.

Compact candidate:
With curtains drawn and fire blazing, the lamp on the _console_ cast a soft glow on the dainty table set for "afternoon tea," a treat Auntie and her nieces, despite long residence in France, loved.

## 47. rw_004509 · source=simple_wiki · risk=52 · saved=9

Risk tags: conditional_force_weakened, conditional_lost_from_both_references, dialogue_or_question

Risk reasons: conditional_force marker preservation 0.00; original count 1, compact count 0

Original:
In the United States, the Bank Secrecy Act says people must write the government currency transaction reports (CTRs) if they buy, sell, or deposit money worth $10,000 or more.

Current inherited rewrite:
Under the Bank Secrecy Act in the United States, individuals are required to file Currency Transaction Reports (CTRs) with the government whenever they engage in cash transactions involving $10,000 or more, whether through deposits, purchases, or sales.

Compact candidate:
Under the Bank Secrecy Act in the United States, individuals must file Currency Transaction Reports (CTRs) with the government for deposits, purchases, or sales of $10,000 or more.

## 48. rw2s0_004620 · source=simple_wiki · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 36 >= current rewrite words 36 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
One year after, on October 20 1827, he and the crew of Azov fought in the Battle of Navarino and with the help of the British and French naval forces, they defeated the Ottoman naval forces.

Current inherited rewrite:
One year later, on October 20 1827, he and the crew of Azov fought in the Battle of Navarino and, with the help of the British and French naval forces, they defeated the Ottoman naval forces.

Compact candidate:
One year later, on October 20 1827, he and the crew of Azov fought in the Battle of Navarino and, with the help of the British and French naval forces, they defeated the Ottoman naval forces.

## 49. rw2s0_004966 · source=gutenberg · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 54 >= current rewrite words 54 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
I never read such a journal of exertions in my life.' Eight years after, on the taking of Ciudad Rodrigo, in 1812, by the British army under Wellington, Captain William Jones, of the 52nd Regiment, having captured a French officer, employed his prisoner in pointing out quarters for his men.

Current inherited rewrite:
Eight years later, in 1812, when the British army under Wellington took Ciudad Rodrigo, Captain William Jones of the 52nd Regiment, having captured a French officer, employed his prisoner in pointing out quarters for his men, an event that led him to remark, 'I never read such a journal of exertions in my life.'

Compact candidate:
Eight years later, in 1812, when the British army under Wellington took Ciudad Rodrigo, Captain William Jones of the 52nd Regiment, having captured a French officer, employed his prisoner in pointing out quarters for his men, an event that led him to remark, 'I never read such a journal of exertions in my life.'

## 50. rw2s0_013565 · source=switchboard · risk=50 · saved=0

Risk tags: no_word_saving, role_reference_weakened

Risk reasons: compact words 21 >= current rewrite words 21 | role/pronoun preservation 0.33

Original:
A: And I try to cover up when I do the lawn B: Well, I, that, that is really healthier, frankly.

Current inherited rewrite:
B: Well, I, that, that is really healthier, frankly, in contrast to A's attempt to cover up when doing the lawn.

Compact candidate:
B: Well, I, that, that is really healthier, frankly, in contrast to A's attempt to cover up when doing the lawn.

## 51. rw2s0_031925 · source=gutenberg · risk=50 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving

Risk reasons: compact words 52 >= current rewrite words 52 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
At the time of the great Irish rebellion of 1641 the head of the Edgeworth family had left his English wife and her infant son at his castle of Cranallagh in county Longford, thinking them safe there while he joined the royal forces under the Earl of Ormond.

Current inherited rewrite:
When the great Irish rebellion of 1641 broke out, the head of the Edgeworth family departed from his castle of Cranallagh in county Longford, where he had left his English wife and her infant son believing them to be safe, in order to join the royal forces under the Earl of Ormond.

Compact candidate:
When the great Irish rebellion of 1641 broke out, the head of the Edgeworth family departed from his castle of Cranallagh in county Longford, where he had left his English wife and her infant son believing them to be safe, in order to join the royal forces under the Earl of Ormond.

## 52. rw2s1_011668 · source=simple_wiki · risk=50 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving

Risk reasons: compact words 31 >= current rewrite words 31 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
Joséphine survived, but Beauharnais was guillotined, together with his cousin Augustin, on the Place de la Révolution, only a week before the trial and execution of Maximilien Robespierre.

Current inherited rewrite:
On the Place de la Révolution, Beauharnais and his cousin Augustin were guillotined together with Joséphine surviving, an event occurring only a week before the trial and execution of Maximilien Robespierre.

Compact candidate:
On the Place de la Révolution, Beauharnais and his cousin Augustin were guillotined together with Joséphine surviving, an event occurring only a week before the trial and execution of Maximilien Robespierre.

## 53. rw2s1_012381 · source=bnc_spoken · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 39 >= current rewrite words 39 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
On the Buckinghamshire section of the M25 clockwise traffic at junction 16 with the M40 is still very heavy and slow due to the amount of traffic that is on the M25 this evening, most places along 25 are quite heavy.

Current inherited rewrite:
On the Buckinghamshire section of the M25, clockwise traffic at junction 16 with the M40 remains very heavy and slow due to the amount of traffic on the M25 this evening, with most places along 25 being quite heavy.

Compact candidate:
On the Buckinghamshire section of the M25, clockwise traffic at junction 16 with the M40 remains very heavy and slow due to the amount of traffic on the M25 this evening, with most places along 25 being quite heavy.

## 54. rw2s1_017519 · source=simple_wiki · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 42 >= current rewrite words 42 | temporal_order marker preservation 0.00; original count 1, compact count 1

Original:
After working for some time in the Babu Jagjivanram Memorial Hospital in New Delhi, she worked in the North Eastern Indira Gandhi Regional Institute of Health and Medical Sciences (NEIGRIHMS) in Shillong as a junior resident doctor for one year.

Current inherited rewrite:
Following her tenure as a junior resident doctor for one year at the North Eastern Indira Gandhi Regional Institute of Health and Medical Sciences (NEIGRIHMS) in Shillong, she subsequently worked for some time at the Babu Jagjivanram Memorial Hospital in New Delhi.

Compact candidate:
Following her tenure as a junior resident doctor for one year at the North Eastern Indira Gandhi Regional Institute of Health and Medical Sciences (NEIGRIHMS) in Shillong, she subsequently worked for some time at the Babu Jagjivanram Memorial Hospital in New Delhi.

## 55. rw_006812 · source=switchboard · risk=50 · saved=0

Risk tags: no_word_saving, role_reference_weakened

Risk reasons: compact words 15 >= current rewrite words 15 | role/pronoun preservation 0.33

Original:
B: and they put their kids in the U S schools B: and they're appalled.

Current inherited rewrite:
Speaker B expresses shock upon learning that these families enroll their children in U.S. schools.

Compact candidate:
Speaker B expresses shock upon learning that these families enroll their children in U.S. schools.

## 56. rw_014018 · source=simple_wiki · risk=50 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving

Risk reasons: compact words 38 >= current rewrite words 38 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
Two of the oldest libraries we know about are the Library of Alexandria and the Library of Nalanda University, which apparently burned for months because of the sheer number of manuscripts.

Current inherited rewrite:
The Library of Alexandria and the Library of Nalanda University are among the most ancient repositories known to us, yet both reportedly suffered catastrophic fires that raged for months due to the immense volume of manuscripts they housed.

Compact candidate:
The Library of Alexandria and the Library of Nalanda University are among the most ancient repositories known to us, yet both reportedly suffered catastrophic fires that raged for months due to the immense volume of manuscripts they housed.

## 57. rw_014540 · source=open_subtitles · risk=50 · saved=0

Risk tags: comparison_relation_weakened, no_word_saving

Risk reasons: compact words 39 >= current rewrite words 39 | comparison_relation marker preservation 0.00; original count 1, compact count 0

Original:
( applause and cheering ) ( band plays upbeat tune ) ( applause and whistling ) I came to this game with one goal, and that goal was to win a million dollars to better my life and my son's.

Current inherited rewrite:
As the band plays an upbeat tune and the crowd erupts in applause and whistling, I reflect on my singular purpose for attending this game: to win a million dollars to improve the lives of myself and my son.

Compact candidate:
As the band plays an upbeat tune and the crowd erupts in applause and whistling, I reflect on my singular purpose for attending this game: to win a million dollars to improve the lives of myself and my son.

## 58. rw_017420 · source=simple_wiki · risk=50 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving

Risk reasons: compact words 41 >= current rewrite words 41 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 0

Original:
After three matches in the Campeonato Paulista Série A2 he went on loan to Ponte Preta but he was only three times on the bench and was sent back to the Red Bulls.

Current inherited rewrite:
Following just three appearances for the Red Bulls in the Campeonato Paulista Série A2, his loan move to Ponte Preta proved unsuccessful after he saw only three minutes of play from the bench, resulting in his return to his original club.

Compact candidate:
Following just three appearances for the Red Bulls in the Campeonato Paulista Série A2, his loan move to Ponte Preta proved unsuccessful after he saw only three minutes of play from the bench, resulting in his return to his original club.

## 59. rw_020113 · source=simple_wiki · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 43 >= current rewrite words 43 | temporal_order marker preservation 0.00; original count 1, compact count 2

Original:
Gurney is the first of three drivers to have won races in Sports Cars (1958), Formula One (1962), NASCAR (1963), and Indy Car (1967) (the other two being Mario Andretti and Juan Pablo Montoya).

Current inherited rewrite:
Gurney holds the distinction of being the inaugural driver to secure victories across Sports Cars in 1958, Formula One in 1962, NASCAR in 1963, and Indy Car in 1967, a feat previously unmatched until Mario Andretti and Juan Pablo Montoya achieved similar feats.

Compact candidate:
Gurney holds the distinction of being the inaugural driver to secure victories across Sports Cars in 1958, Formula One in 1962, NASCAR in 1963, and Indy Car in 1967, a feat previously unmatched until Mario Andretti and Juan Pablo Montoya achieved similar feats.

## 60. rw_035438 · source=open_subtitles · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 27 >= current rewrite words 27 | temporal_order marker preservation 0.00; original count 1, compact count 0

Original:
Whatever it is, terrorists, a nuclear leak, plants, it's probably safe to get away from people right now.

Current inherited rewrite:
Regardless of the specific threat, whether it involves terrorists, a nuclear leak, or something else, the safest course of action is to distance yourself from others immediately.

Compact candidate:
Regardless of the specific threat, whether it involves terrorists, a nuclear leak, or something else, the safest course of action is to distance yourself from others immediately.

## 61. rw_036730 · source=open_subtitles · risk=50 · saved=0

Risk tags: causal_or_contrast_force_weakened, no_word_saving

Risk reasons: compact words 24 >= current rewrite words 24 | causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
He/she seems strange, but it was for his/her dog, Alfie, that I turned into detective.

Current inherited rewrite:
Although he or she appears peculiar, my transformation into a detective was driven by a desire to investigate what happened to their dog, Alfie.

Compact candidate:
Although he or she appears peculiar, my transformation into a detective was driven by a desire to investigate what happened to their dog, Alfie.

## 62. rw_039875 · source=childes · risk=50 · saved=0

Risk tags: no_word_saving, temporal_order_weakened

Risk reasons: compact words 41 >= current rewrite words 41 | temporal_order marker preservation 0.00; original count 2, compact count 1

Original:
[Mark and Ross stop fighting for a second and then continue] [allen is the bullying older brother of Ross's friend Joey] *CHI: I wouldn't fight with them no way me and Markie like to fight to together.

Current inherited rewrite:
Mark and Ross briefly pause their altercation before resuming, while Allen is the bullying older brother of Ross's friend Joey; in a voice dubbed CHI, someone remarks, "I wouldn't fight with them no way, me and Markie like to fight together."

Compact candidate:
Mark and Ross briefly pause their altercation before resuming, while Allen is the bullying older brother of Ross's friend Joey; in a voice dubbed CHI, someone remarks, "I wouldn't fight with them no way, me and Markie like to fight together."

## 63. rw2s0_009985 · source=childes · risk=0 · saved=16

Risk tags: none

Original:
*CHI: on Saturday he ate through a piece of chocolate cake one icecream cone one pickle one slice of Swish cheese one slice of salami one lollipop one piece of cherry pie one sausage one cupcake one slice of watermelon.

Current inherited rewrite:
*CHI: On Saturday he consumed a piece of chocolate cake, an ice cream cone, a pickle, a slice of Swish cheese, a slice of salami, a lollipop, a piece of cherry pie, a sausage, a cupcake, and a slice of watermelon.

Compact candidate:
*CHI: On Saturday he ate chocolate cake, an ice cream cone, a pickle, Swish cheese, salami, a lollipop, cherry pie, sausage, a cupcake, and watermelon.

## 64. rw2s0_010026 · source=simple_wiki · risk=0 · saved=5

Risk tags: none

Original:
They are seen as part of a wider movement to make Iran a more democratic country since the 1953 Iranian coup d'état, in which the US and the UK overthrow of the democratically elected Prime Minister of Iran, Mohammad Mosaddegh, due to his policy to nationalize Iran's oil industry.

Current inherited rewrite:
They are viewed as part of a broader movement to transform Iran into a more democratic nation since the 1953 Iranian coup d'état, where the US and the UK overthrew the democratically elected Prime Minister of Iran, Mohammad Mosaddegh, because of his policy to nationalize Iran's oil industry.

Compact candidate:
They are seen as part of a wider movement to make Iran more democratic since the 1953 Iranian coup d'état, in which the US and UK overthrew democratically elected Iranian Prime Minister Mohammad Mosaddegh due to his policy to nationalize Iran's oil industry.

## 65. rw_017379 · source=simple_wiki · risk=0 · saved=10

Risk tags: none

Original:
In the early 1970s, Von Brunn briefly worked for Noontide Press, the publishing arm of the Holocaust-denying Institute for Historical Review.

Current inherited rewrite:
During the early 1970s, Von Brunn had a short tenure at Noontide Press, which served as the publishing division of the Institute for Historical Review, an organization that denies the Holocaust.

Compact candidate:
In the early 1970s, Von Brunn briefly worked for Noontide Press, the publishing arm of the Holocaust-denying Institute for Historical Review.

## 66. rw2s0_019852 · source=open_subtitles · risk=12 · saved=2

Risk tags: dialogue_or_question

Original:
I went to Mass that Sunday, and the priest was giving his homily, and he said something about, "Well maybe, you know, "The students have something to say, "and we should start listening to them." Got up and walked out.

Current inherited rewrite:
I attended Mass that Sunday, and while the priest was delivering his homily, he remarked, "Well maybe, you know, 'The students have something to say,' and we should start listening to them," after which I stood up and walked out.

Compact candidate:
I attended Mass that Sunday; while the priest delivered his homily, he said, "Well maybe, you know, 'The students have something to say,' and we should start listening to them," after which I stood up and walked out.

## 67. rw_034543 · source=open_subtitles · risk=0 · saved=7

Risk tags: none

Original:
worried about your artistic vision and being true to yourself, that you'll never be able to do what's best for your company.

Current inherited rewrite:
If you are concerned about maintaining your artistic integrity and staying true to yourself, you may fear that you will never be able to act in your company's best interest.

Compact candidate:
If you worry about your artistic vision and being true to yourself, you may fear you'll never do what's best for your company.

## 68. rw2s0_006924 · source=simple_wiki · risk=0 · saved=6

Risk tags: none

Original:
He was the dean of theology faculty at Baku State University, chairman of a cathedra of Arabic philology and full member of Azerbaijan National Academy of Sciences.

Current inherited rewrite:
He served as the dean of the theology faculty at Baku State University, chaired a cathedra of Arabic philology, and was a full member of the Azerbaijan National Academy of Sciences.

Compact candidate:
He was dean of theology faculty at Baku State University, chaired a cathedra of Arabic philology, and full member of Azerbaijan National Academy of Sciences.

## 69. rw2s0_019304 · source=simple_wiki · risk=0 · saved=16

Risk tags: none

Original:
Manning is the son of Ed Manning, who was a longtime NBA and ABA player and professional and college coach.

Current inherited rewrite:
Manning, the son of Ed Manning, was born to a man who served as a professional and college coach and had a long career as a player in both the NBA and ABA.

Compact candidate:
Manning is the son of Ed Manning, a longtime NBA, ABA player and professional and college coach.

## 70. rw_015472 · source=simple_wiki · risk=0 · saved=10

Risk tags: none

Original:
Madeley's solo projects include the ITV series and his stand-in for The Chris Evans Breakfast Show on BBC Radio 2.

Current inherited rewrite:
Madeley has taken on solo roles such as his appearance in an ITV series and his stint as the stand-in for The Chris Evans Breakfast Show on BBC Radio 2.

Compact candidate:
Madeley's solo projects include an ITV series and his stand-in for The Chris Evans Breakfast Show on BBC Radio 2.

## 71. rw2s0_025051 · source=bnc_spoken · risk=0 · saved=9

Risk tags: none

Original:
The staff from the Abingdon Sorting Office had been due to resume their duties this afternoon at the end of the county-wide dispute, but the workers there refused to go back after part of an agreement with managers was altered.

Current inherited rewrite:
The staff from the Abingdon Sorting Office had been scheduled to resume their duties this afternoon at the end of the county-wide dispute, but the workers there refused to return after part of an agreement with managers was altered.

Compact candidate:
The Abingdon Sorting Office staff were due to resume duties this afternoon after the county-wide dispute, but workers refused to return after part of the agreement with managers was altered.

## 72. rw2s1_005600 · source=gutenberg · risk=0 · saved=8

Risk tags: none

Original:
Clavering was too great a favorite and had too wise and excellent ideas with regard to education ever to be without pupils, and never more, for she believed twenty to be the perfect number to whom she could give every attention and offer every advantage.

Current inherited rewrite:
Clavering, being an exceptionally favored individual endowed with remarkably wise and excellent ideas concerning education, would never lack pupils, and indeed never had more, as she held the conviction that twenty represented the ideal number of students to whom she could devote every attention and provide every advantage.

Compact candidate:
Clavering, too great a favorite with too wise and excellent ideas on education, would never lack pupils, and never more, for she believed twenty to be the perfect number to whom she could give every attention and offer every advantage.

## 73. rw2s0_002308 · source=bnc_spoken · risk=10 · saved=9

Risk tags: conditional_force_partly_changed

Original:
It is submitted that the whole of Skelton Village should be looked at as a unity,that there is extreme environmental value over its north part,and that were the village inset into the greenbelt, the inset boundary should be precisely as shown in the deposit copy of Southern Ryedale's Local Plan.

Current inherited rewrite:
It is submitted that the entirety of Skelton Village should be regarded as a single unit, that its northern section holds extreme environmental value, and that if the village were inset into the greenbelt, the inset boundary should be precisely as shown in the deposit copy of Southern Ryedale's Local Plan.

Compact candidate:
It is submitted that Skelton Village should be regarded as a single unit, its northern section holds extreme environmental value, and if inset into the greenbelt, the boundary should be precisely as shown in the deposit copy of Southern Ryedale's Local Plan.

## 74. rw_000248 · source=simple_wiki · risk=12 · saved=12

Risk tags: dialogue_or_question

Original:
Renee Schonfeld of Common Sense Media rated it 3 out of 5 stars, saying "By blending solid new musical numbers with the old Oz favorites and including inventive new characters, Spike Brandt, Tony Cerone, and company have made this sequel funny and original." "Tom and Jerry: Cowboy Up!".

Current inherited rewrite:
In her review for *Tom and Jerry: Cowboy Up!*, Renee Schonfeld of Common Sense Media awarded the film 3 out of 5 stars, praising the sequel for its blend of classic Oz favorites with fresh musical numbers and inventive new characters, crediting director Spike Brandt, Tony Cerone, and their team with creating a funny and original result.

Compact candidate:
Renee Schonfeld of Common Sense Media rated *Tom and Jerry: Cowboy Up!* 3 out of 5 stars, saying Spike Brandt, Tony Cerone, and company made this sequel funny and original by blending solid new musical numbers with old Oz favorites and including inventive new characters.

## 75. rw_010999 · source=bnc_spoken · risk=0 · saved=5

Risk tags: none

Original:
And Quality Street Egg verdict, the box makes the eggs appear much larger than it actually is cost, a hundred gram, ninety nine .

Current inherited rewrite:
Regarding the Quality Street Egg verdict, the box creates an illusion that makes the eggs seem much larger than they actually are for a cost of ninety-nine cents per hundred grams.

Compact candidate:
Regarding the Quality Street Egg verdict, the box creates an illusion making eggs seem much larger than they actually are for ninety-nine cents per hundred grams.

## 76. rw2s1_024034 · source=bnc_spoken · risk=0 · saved=7

Risk tags: none

Original:
The unset seen scandal of Major's Britain, a hundred and twenty thousand, sixteen to twenty four year olds, not on the register, not in a job, no scheme to join and no benefit available.

Current inherited rewrite:
The unseen scandal of Major's Britain involves one hundred and twenty thousand sixteen to twenty-four year olds who are not on the register, not in a job, have no scheme to join, and have no benefit available.

Compact candidate:
The unseen scandal of Major's Britain involves 120,000 sixteen to twenty-four year olds not on the register, not in a job, with no scheme to join and no benefit available.

## 77. rw_041183 · source=open_subtitles · risk=0 · saved=8

Risk tags: none

Original:
The Bible story of the Hebrew jews begin with Abraham, who was born north of Canaan in the town of Ur, around 2055 BC Abraham married his beautiful sister Sarai.

Current inherited rewrite:
According to the Bible, the narrative of the Hebrew Jews originates with Abraham, a man born circa 2055 BC in the city of Ur, located north of Canaan, who wed his lovely sister Sarai.

Compact candidate:
According to the Bible, the Hebrew Jews' story begins with Abraham, born circa 2055 BC in Ur, north of Canaan, who married his beautiful sister Sarai.

## 78. rw_024111 · source=switchboard · risk=0 · saved=5

Risk tags: none

Original:
B: And that is it's million of people with anywhere from thousands to tens of thousands I guess of, Of dollars being protected.

Current inherited rewrite:
Speaker B concludes by noting that millions of people have safeguards in place, covering amounts ranging from thousands to tens of thousands of dollars.

Compact candidate:
Speaker B concludes that millions of people have safeguards covering amounts from thousands to tens of thousands of dollars.

## 79. rw2s0_007486 · source=bnc_spoken · risk=10 · saved=13

Risk tags: conditional_force_partly_changed

Original:
The amendments to the Bill from the House of Lords were debated in only three hours as the Government guillotined the motion, and the Speaker of the House also ruled that a Lord's amendment saying students should be entitled to Housing Benefit was invalid.

Current inherited rewrite:
The Government guillotined the motion, resulting in the House of Lords debating its amendments to the Bill for only three hours, while the Speaker of the House also ruled that a Lord's amendment stating students should be entitled to Housing Benefit was invalid.

Compact candidate:
The Government guillotined the motion, limiting House of Lords debate on Bill amendments to three hours, while the Speaker ruled a Lord's amendment stating students should receive Housing Benefit invalid.

## 80. rw2s0_014542 · source=gutenberg · risk=20 · saved=7

Risk tags: causal_or_contrast_force_weakened

Risk reasons: causal_or_contrast_force marker preservation 0.00; original count 1, compact count 1

Original:
Whether one hundred thousand or two hundred thousand Protestants were murdered in this outbreak, is uncertain; but, that it was as ruthless and barbarous an outbreak as ever was known among any savage people, is certain.

Current inherited rewrite:
Whether the number of Protestants murdered in this outbreak was one hundred thousand or two hundred thousand remains uncertain, yet it is certain that the outbreak was as ruthless and barbarous as any known among savage people.

Compact candidate:
Whether one hundred thousand or two hundred thousand Protestants were murdered remains uncertain, yet it is certain the outbreak was as ruthless and barbarous as any known among savage people.

## 81. rw2s1_017873 · source=open_subtitles · risk=0 · saved=11

Risk tags: none

Original:
Christians get a trip to the moon on gossamer wings, Muslims get a little salt and pepper on a fruit salad at the end of Ramadan.

Current inherited rewrite:
At the conclusion of Ramadan, Muslims receive a modest amount of salt and pepper atop a fruit salad, whereas Christians are granted a journey to the moon propelled by gossamer wings.

Compact candidate:
Christians get a moon trip on gossamer wings; Muslims get salt and pepper on a fruit salad at Ramadan's end.

## 82. rw2s1_015912 · source=gutenberg · risk=0 · saved=9

Risk tags: none

Original:
For a moment the horsemen were too astonished to move; then, recovering from their surprise, they lowered their murderous-looking lances, and would undoubtedly have run all three prisoners through, had not another officer ridden into the circle at that moment.

Current inherited rewrite:
For a brief instant the horsemen were so stunned by the sight that they could not move, but once they recovered from their shock, they lowered their menacing lances and would certainly have pierced all three prisoners had another officer not ridden into the circle at that very moment.

Compact candidate:
For a moment the horsemen were too astonished to move; then, recovering from their surprise, they lowered their murderous-looking lances, and would undoubtedly have run all three prisoners through, had not another officer ridden into the circle at that moment.

