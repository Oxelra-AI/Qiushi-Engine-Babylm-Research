# Step038b entity-answer grounding suspicious rows

## 1. rw2s0_001965 (simple_wiki; target_earlier)

flags: target_source_answer_closer_to_distractor_than_target, distractor_source_answer_closer_to_target_than_distractor, target_entity_closer_to_competing_source_answer, distractor_entity_closer_to_competing_source_answer, distractor_answer_far_from_distractor_entity

gamma=-7.440767238537471 joint=False update_event=weak_or_unclear

SOURCE: After members of ISIS carried out terrorist attacks in Paris in November 2015, Corbyn said that the only way to deal with the threat by ISIS would be to reach a political settlement and ending the Syrian Civil War.

TARGET=Corbyn ans=political settlement | dist_to_target=76 dist_to_distractor=16 competing_ans_dist_to_target=28

DISTRACTOR=Syrian Civil War ans=terrorist attacks | dist_to_distractor=146 dist_to_target=28 competing_ans_dist_to_distractor=16

## 2. rw_007527 (simple_wiki; target_earlier)

flags: distractor_entity_no_exact_source_span, target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-9.182300429791212 joint=False update_event=weak_or_unclear

SOURCE: Its neighbouring municipalities (clockwise from the north) are: Maroldsweisach, Seßlach, Untermerzbach, Ebern and Burgpreppach.

TARGET=Burgpreppach ans=fifth municipality | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Maroldsweisach Seßlach Untermerzbach Ebern ans=first three municipalities | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 3. rw2s1_010999 (bnc_spoken; target_earlier)

flags: target_source_answer_closer_to_distractor_than_target, target_entity_closer_to_competing_source_answer, target_answer_far_from_target_entity

gamma=-1.057155966758728 joint=False update_event=weak_or_unclear

SOURCE: Thank you It's certainly will be difficult, Chair, but in answer to Mrs Tidley, what's happening to our housing, the answer is we've stopped building it.

TARGET=Chair ans=stopped building it | dist_to_target=84 dist_to_distractor=55 competing_ans_dist_to_target=55

DISTRACTOR=Mrs Tidley ans=housing | dist_to_distractor=26 dist_to_target=55 competing_ans_dist_to_distractor=55

## 4. rw2s0_031693 (simple_wiki; target_later)

flags: target_entity_no_exact_source_span, target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-0.6697009086608894 joint=False update_event=strong_replacement_like

SOURCE: Evans died on 9 August 2022 at his home in London, England from a heart attack at the age of 72.

TARGET=London England ans=location of Evans' death | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=August ans=month of Evans' death | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 5. rw_005618 (gutenberg; target_earlier)

flags: distractor_entity_closer_to_competing_source_answer, distractor_answer_far_from_distractor_entity

gamma=-8.48773156106472 joint=False update_event=weak_or_unclear

SOURCE: It was the little son of the Count von Bardi whom Wilhelm von Mosen brought down by mistake for young Albrecht, and Kunz, while hurrying up to exchange the children, bade the rest of his band hasten on to secure the elder prince without waiting for him.

TARGET=Mosen ans=little son | dist_to_target=41 dist_to_distractor=81 competing_ans_dist_to_target=149

DISTRACTOR=Albrecht ans=elder prince | dist_to_distractor=106 dist_to_target=149 competing_ans_dist_to_distractor=81

## 6. rw_029261 (simple_wiki; target_later)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-7.3566431403160095 joint=False update_event=strong_replacement_like

SOURCE: Flower chose him as the judge for the trial of Bartholomew Shea and John McGough for the murder of Robert Ross.

TARGET=Robert Ross ans=victim | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=John McGough ans=accused | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 7. rw2s1_000670 (gutenberg; target_later)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_closer_to_target_than_distractor

gamma=-5.804875493049622 joint=False update_event=weak_or_unclear

SOURCE: To finish the sad story of Sir John Oldcastle at once, I may mention that he escaped into Wales, and remained there safely, for four years.

TARGET=Wales ans=safely there | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=0

DISTRACTOR=Sir John Oldcastle ans=escaped into Wales | dist_to_distractor=32 dist_to_target=0 competing_ans_dist_to_distractor=inf

## 8. rw2s1_006925 (simple_wiki; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-5.4894990871349965 joint=False update_event=weak_or_unclear

SOURCE: Pope Nicholas I had refused to recognize Patriarch Photios I of Constantinople, who in turn had attacked the pope as a heretic.

TARGET=Nicholas ans=refused recognition | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Patriarch Photios ans=attacked as heretic | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 9. rw_041658 (open_subtitles; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-3.442487021287282 joint=False update_event=additive_or_descriptive

SOURCE: actually of Judaism as a threat in that way at all, until the Zionist movement annexed the messianic, or fused with it, because the messianists didnít used to be Zionists, as you know, so, youíd never know when itís gonna be next.

TARGET=Judaism ans=fused with the messianic movement | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Zionist ans=annexed the messianic movement | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 10. rw2s1_029613 (simple_wiki; target_earlier)

flags: distractor_entity_no_exact_source_span, target_source_answer_no_exact_raw_span

gamma=-3.200116551839388 joint=False update_event=weak_or_unclear

SOURCE: At the age of 15 Villanueva became a member of APRA's "Juventud Aprista Peruana" in opposition to the military dictatorship of Luis Miguel Sánchez Cerro.

TARGET=Villanueva ans=member of APRA's Juventud Aprista Peruana | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=75

DISTRACTOR=APRA's Juventud Aprista Peruana ans=military dictatorship of Luis Miguel Sánchez Cerro | dist_to_distractor=inf dist_to_target=75 competing_ans_dist_to_distractor=inf

## 11. rw2s1_000371 (gutenberg; target_later)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-3.1669741471608477 joint=False update_event=weak_or_unclear

SOURCE: Brennus chose out the hardiest of his mountaineers, and directed them to climb up in the dead of night, one by one, in perfect silence, and thus to surprise the Romans, and complete the slaughter and victory, before the forces assembling at Veii would come to their rescue.

TARGET=Veii ans=assembling forces | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Romans ans=victims of surprise attack | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 12. rw2s0_027269 (simple_wiki; target_later)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-2.8085014820098877 joint=False update_event=weak_or_unclear

SOURCE: He succeeded to the barony in 1998 and lost his seat in the House of Lords in 1999 due to legislative changes.

TARGET=Lords ans=lost seat | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=House ans=succeeded to barony | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 13. rw_007895 (simple_wiki; target_later)

flags: distractor_source_answer_closer_to_target_than_distractor, target_entity_closer_to_competing_source_answer

gamma=-2.7486879229545593 joint=False update_event=additive_or_descriptive

SOURCE: In 1891 he ran for municipal elections is supported by young intellectuals who had recently founded La Nuova Sardegna is the unitary list of Republicans and moderate and, elected with broad suffrage, held for years the city councilor activities.

TARGET=Republicans ans=city councilor activities | dist_to_target=67 dist_to_distractor=102 competing_ans_dist_to_target=4

DISTRACTOR=La Nuova Sardegna ans=unitary list | dist_to_distractor=8 dist_to_target=4 competing_ans_dist_to_distractor=102

## 14. rw2s0_029990 (simple_wiki; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-2.6916272640228263 joint=False update_event=weak_or_unclear

SOURCE: The first choir that recorded it was the "Coro Supramonte" of Orgosolo which, finally, released it in 1974 on an LP for Fonit Cetra.

TARGET=Orgosolo ans=released the recording | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=LP ans=physical format | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 15. rw2s0_027873 (gutenberg; target_earlier)

flags: distractor_source_answer_no_exact_raw_span, target_source_answer_closer_to_distractor_than_target

gamma=-2.6626939455668124 joint=False update_event=weak_or_unclear

SOURCE: He rode to the scaffold in his own carriage, attended by two famous clergymen, Tillotson and Burnet, and sang a psalm to himself very softly, as he went along.

TARGET=Tillotson ans=a psalm | dist_to_target=22 dist_to_distractor=11 competing_ans_dist_to_target=inf

DISTRACTOR=Burnet ans=the procession | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=11

## 16. rw2s1_032450 (simple_wiki; target_earlier)

flags: distractor_source_answer_no_exact_raw_span, target_source_answer_closer_to_distractor_than_target

gamma=-2.5694291353225704 joint=False update_event=additive_or_descriptive

SOURCE: The ruins of Gedi are described and repeatedly mentioned in Andrei Gusev's 2020 novel “Our Wild Sex in Malindi”.

TARGET=Gedi ans=described and repeatedly mentioned | dist_to_target=5 dist_to_distractor=4 competing_ans_dist_to_target=inf

DISTRACTOR=Andrei Gusev's ans=authored a novel | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=4

## 17. rw2s1_027818 (bnc_spoken; target_earlier)

flags: target_source_answer_closer_to_distractor_than_target, distractor_entity_closer_to_competing_source_answer

gamma=-2.215720295906067 joint=False update_event=weak_or_unclear

SOURCE: A delegation of the Islamic world's most senior figures says the ominous call for a Holy War against western forces wouldn't be confined to the Gulf.

TARGET=Islamic ans=ominous call | dist_to_target=38 dist_to_distractor=7 competing_ans_dist_to_target=117

DISTRACTOR=Holy War ans=Gulf | dist_to_distractor=52 dist_to_target=117 competing_ans_dist_to_distractor=7

## 18. rw2s0_007875 (open_subtitles; target_later)

flags: target_source_answer_closer_to_distractor_than_target, distractor_entity_closer_to_competing_source_answer

gamma=-2.1153860648473097 joint=False update_event=additive_or_descriptive

SOURCE: It didn't reappear for more than a thousand years, when knights from the First Crusade discovered secret vaults beneath the Temple of Solomon.

TARGET=Solomon ans=secret vaults | dist_to_target=23 dist_to_distractor=13 competing_ans_dist_to_target=48

DISTRACTOR=Temple ans=knights from the First Crusade | dist_to_distractor=38 dist_to_target=48 competing_ans_dist_to_distractor=13

## 19. rw2s0_001731 (simple_wiki; target_later)

flags: target_source_answer_closer_to_distractor_than_target, distractor_entity_closer_to_competing_source_answer

gamma=-1.4364498355425894 joint=False update_event=weak_or_unclear

SOURCE: It went to number 1 in New Zealand, number 3 in the United States, number 11 in Australia, number 14 in Canada and the Netherlands and number 20 in Belgium.

TARGET=Netherlands ans=number 14 | dist_to_target=19 dist_to_distractor=4 competing_ans_dist_to_target=43

DISTRACTOR=Canada ans=number 11 | dist_to_distractor=28 dist_to_target=43 competing_ans_dist_to_distractor=4

## 20. rw_007461 (gutenberg; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-1.3667196989059445 joint=False update_event=weak_or_unclear

SOURCE: It so happened that no dress could suit Kitty better, and doubtless Sir John had an eye to the appearance of his favorite in such a robe when he ordered it.

TARGET=Kitty ans=perfectly suited robe | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Sir John ans=ordered favorite's robe | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 21. rw2s1_009238 (gutenberg; target_later)

flags: target_entity_closer_to_competing_source_answer, target_answer_far_from_target_entity

gamma=-1.320712248484294 joint=False update_event=strong_replacement_like

SOURCE: The convoy reached Balaclava at dawn, and Phil, with Tony in attendance, and some fifty other wounded men was sent on board a small schooner, which at once weighed anchor, and sailed out of the harbour.

TARGET=Phil ans=schooner | dist_to_target=86 dist_to_distractor=104 competing_ans_dist_to_target=14

DISTRACTOR=Balaclava ans=Balaclava | dist_to_distractor=0 dist_to_target=14 competing_ans_dist_to_distractor=104

## 22. rw_023565 (simple_wiki; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-0.5976500511169434 joint=False update_event=weak_or_unclear

SOURCE: Yirmibeşoğlu died from kidney failure on 2 January 2016 in Istanbul, aged 87.

TARGET=January ans=death date | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Istanbul ans=death location | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 23. rw2s0_008771 (gutenberg; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-0.5354226271311449 joint=False update_event=additive_or_descriptive

SOURCE: Indeed, like many other good things which have loomed up conspicuously in recent times, arbitration can be traced back to the ancient Greeks, for whom it occasionally mitigated the evils attendant upon frequent warfare between their city-states.

TARGET=Greeks ans=mitigated evils | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=city ans=political units | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

## 24. rw2s0_020772 (bnc_spoken; target_later)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_closer_to_target_than_distractor

gamma=-0.22009944915771484 joint=False update_event=additive_or_descriptive

SOURCE: Many children have been injured in increasing numbers over the last year, and it was the idea of one school teacher in Oxford to bring a group of gendarmes over to England to teach children how the French behave on the roads.

TARGET=England ans=French behavior on the roads | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=9

DISTRACTOR=Oxford ans=group of gendarmes | dist_to_distractor=12 dist_to_target=9 competing_ans_dist_to_distractor=inf

## 25. rw2s1_029108 (gutenberg; target_earlier)

flags: target_source_answer_no_exact_raw_span, distractor_source_answer_no_exact_raw_span

gamma=-0.19961094856262207 joint=False update_event=weak_or_unclear

SOURCE: On the day before Scarfe's proposed visit, Walker accosted him as he was going out, with the announcement that my lady would like to speak to him in the morning-room.

TARGET=Scarfe's ans=going out to meet my lady | dist_to_target=inf dist_to_distractor=inf competing_ans_dist_to_target=inf

DISTRACTOR=Walker ans=accosted Scarfe as he was going out | dist_to_distractor=inf dist_to_target=inf competing_ans_dist_to_distractor=inf

