# manual_review_plausible_hard_rows.md

## 1. rw2s1_004551 (gutenberg; target_earlier)

flags: target_answer_not_exact_raw_span, distractor_state_weak_raw_overlap, update_not_explicit_replacement_like

baseline U=11.380 R=-11.751 beta=-0.185 |alpha|=11.565 gamma=-11.751 joint=False

SOURCE: Jonathan Andrewes came to Dacrefield on business connected with his brother's affairs, and he accepted my father's hospitality at the Hall.

TARGET=Dacrefield DISTRACTOR=Hall

TARGET_SOURCE_STATE=served as the location for business affairs | ANSWER=business affairs | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=provided hospitality for the visitor | ANSWER=hospitality | raw_overlap=1.000, exact=True

UPDATE_FRAME: The visitor accepted hospitality at {ENTITY} for the family reunion. [weak_or_unclear]

USE_FRAME: The visitor accepted hospitality at Dacrefield for the {STATE}.

UPDATE_USE: The visitor accepted hospitality at Dacrefield for the family reunion.

RETAIN_USE: The visitor accepted hospitality at Dacrefield for the business affairs.

## 2. rw2s1_025940 (simple_wiki; target_later)

flags: update_not_explicit_replacement_like

baseline U=-11.599 R=11.749 beta=0.075 |alpha|=11.674 gamma=-11.599 joint=False

SOURCE: Fatima Robinson directed the music video for "No", which features Trainor performing choreographed dances in a warehouse and entwining her arms with accompanying female dancers.

TARGET=Trainor DISTRACTOR=No

TARGET_SOURCE_STATE=performing choreographed dances in a warehouse | ANSWER=choreographed dances | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=featured in a music video | ANSWER=music video | raw_overlap=1.000, exact=True

UPDATE_FRAME: The video features {ENTITY} entwining arms with female dancers. [weak_or_unclear]

USE_FRAME: The video shows Trainor performing {STATE} with dancers.

UPDATE_USE: The video shows Trainor performing entwining arms with dancers.

RETAIN_USE: The video shows Trainor performing choreographed dances with dancers.

## 3. rw_019197 (gutenberg; target_earlier)

flags: distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like

baseline U=8.517 R=-9.169 beta=-0.326 |alpha|=8.843 gamma=-9.169 joint=False

SOURCE: I must fetch Merle to look at them at once, and Jessop too.

TARGET=Merle DISTRACTOR=Jessop

TARGET_SOURCE_STATE=fetch Merle to look at them at once | ANSWER=fetch Merle | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=fetch Jessop to look at them at once | ANSWER=fetch Jessop | raw_overlap=1.000, exact=False

UPDATE_FRAME: I must fetch {ENTITY} to examine the documents at once. [weak_or_unclear]

USE_FRAME: I must fetch Merle to examine the {STATE} at once.

UPDATE_USE: I must fetch Merle to examine the examine the documents at once.

RETAIN_USE: I must fetch Merle to examine the fetch Merle at once.

## 4. rw_005618 (gutenberg; target_earlier)

flags: update_not_explicit_replacement_like

baseline U=8.334 R=-8.488 beta=-0.077 |alpha|=8.411 gamma=-8.488 joint=False

SOURCE: It was the little son of the Count von Bardi whom Wilhelm von Mosen brought down by mistake for young Albrecht, and Kunz, while hurrying up to exchange the children, bade the rest of his band hasten on to secure the elder prince without waiting for him.

TARGET=Mosen DISTRACTOR=Albrecht

TARGET_SOURCE_STATE=brought down the little son by mistake | ANSWER=little son | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=was the elder prince to be secured | ANSWER=elder prince | raw_overlap=1.000, exact=True

UPDATE_FRAME: While hurrying up, {ENTITY} exchanged the children with Kunz. [weak_or_unclear]

USE_FRAME: Wilhelm von Mosen brought down the {STATE} by mistake.

UPDATE_USE: Wilhelm von Mosen brought down the children by mistake.

RETAIN_USE: Wilhelm von Mosen brought down the little son by mistake.

## 5. rw2s0_016677 (simple_wiki; target_later)

flags: update_not_explicit_replacement_like

baseline U=-7.693 R=8.237 beta=0.272 |alpha|=7.965 gamma=-7.693 joint=False

SOURCE: He died at the Royal Brompton Hospital in London and was buried in Brookwood Cemetery on 10 June 2011.

TARGET=June DISTRACTOR=hospital

TARGET_SOURCE_STATE=occurred on 10 June 2011 | ANSWER=10 June 2011 | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=located at the Royal Brompton Hospital | ANSWER=Royal Brompton Hospital | raw_overlap=1.000, exact=True

UPDATE_FRAME: The official record states that {ENTITY} was recorded in the official death certificate. [weak_or_unclear]

USE_FRAME: The burial date of June was listed in the {STATE}.

UPDATE_USE: The burial date of June was listed in the official death certificate.

RETAIN_USE: The burial date of June was listed in the 10 June 2011.

## 6. rw2s0_001965 (simple_wiki; target_earlier)

flags: update_not_explicit_replacement_like

baseline U=6.660 R=-7.441 beta=-0.390 |alpha|=7.051 gamma=-7.441 joint=False

SOURCE: After members of ISIS carried out terrorist attacks in Paris in November 2015, Corbyn said that the only way to deal with the threat by ISIS would be to reach a political settlement and ending the Syrian Civil War.

TARGET=Corbyn DISTRACTOR=Syrian Civil War

TARGET_SOURCE_STATE=said the only way to deal with the threat would be to reach a political settlement | ANSWER=political settlement | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=threatened to carry out terrorist attacks in Paris | ANSWER=terrorist attacks | raw_overlap=1.000, exact=True

UPDATE_FRAME: After the Paris attacks, {ENTITY} advocated for ending the conflict through direct negotiations. [weak_or_unclear]

USE_FRAME: Corbyn proposed that the solution required {STATE} to resolve the crisis.

UPDATE_USE: Corbyn proposed that the solution required direct negotiations to resolve the crisis.

RETAIN_USE: Corbyn proposed that the solution required political settlement to resolve the crisis.

## 7. rw2s1_027855 (gutenberg; target_later)

flags: update_not_explicit_replacement_like

baseline U=6.867 R=-6.902 beta=-0.018 |alpha|=6.884 gamma=-6.902 joint=False

SOURCE: His Sowship, having that uncommonly high opinion of himself as a king, had a very low opinion of Parliament as a power that audaciously wanted to control him.

TARGET=Parliament DISTRACTOR=Sowship

TARGET_SOURCE_STATE=had a very low opinion of Parliament as a power | ANSWER=low opinion | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=had that uncommonly high opinion of himself as a king | ANSWER=high opinion | raw_overlap=1.000, exact=True

UPDATE_FRAME: The assembly decided that {ENTITY} developed a mutual respect for shared governance. [weak_or_unclear]

USE_FRAME: Observers noted that Parliament held a distinct {STATE} regarding its authority.

UPDATE_USE: Observers noted that Parliament held a distinct mutual respect regarding its authority.

RETAIN_USE: Observers noted that Parliament held a distinct low opinion regarding its authority.

## 8. rw2s1_000670 (gutenberg; target_later)

flags: target_answer_not_exact_raw_span, update_not_explicit_replacement_like

baseline U=3.611 R=-5.805 beta=-1.097 |alpha|=4.708 gamma=-5.805 joint=False

SOURCE: To finish the sad story of Sir John Oldcastle at once, I may mention that he escaped into Wales, and remained there safely, for four years.

TARGET=Wales DISTRACTOR=Sir John Oldcastle

TARGET_SOURCE_STATE=remained safely there for four years | ANSWER=safely there | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=escaped into Wales at once | ANSWER=escaped into Wales | raw_overlap=1.000, exact=True

UPDATE_FRAME: After the conflict ended, {ENTITY} settled permanently in the northern valley. [weak_or_unclear]

USE_FRAME: Historical records show Wales was the {STATE} for four years.

UPDATE_USE: Historical records show Wales was the northern valley for four years.

RETAIN_USE: Historical records show Wales was the safely there for four years.

## 9. rw2s1_006925 (simple_wiki; target_earlier)

flags: target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like

baseline U=10.066 R=-5.489 beta=2.288 |alpha|=7.778 gamma=-5.489 joint=False

SOURCE: Pope Nicholas I had refused to recognize Patriarch Photios I of Constantinople, who in turn had attacked the pope as a heretic.

TARGET=Nicholas DISTRACTOR=Patriarch Photios

TARGET_SOURCE_STATE=refused to recognize Patriarch Photios I of Constantinople | ANSWER=refused recognition | raw_overlap=0.500, exact=False

DISTRACTOR_SOURCE_STATE=attacked the pope as a heretic | ANSWER=attacked as heretic | raw_overlap=1.000, exact=False

UPDATE_FRAME: After the conflict, {ENTITY} exchanged diplomatic letters with the other leader. [weak_or_unclear]

USE_FRAME: Historical records show Nicholas sent {STATE} to the other leader.

UPDATE_USE: Historical records show Nicholas sent diplomatic letters to the other leader.

RETAIN_USE: Historical records show Nicholas sent refused recognition to the other leader.

## 10. rw2s0_003379 (gutenberg; target_earlier)

flags: none

baseline U=2.910 R=-5.235 beta=-1.163 |alpha|=4.072 gamma=-5.235 joint=False

SOURCE: In general, it means the reddish-brown wood itself; but in jest, it signifies "excessively fine," which arose from an anecdote of Nyboder, in Copenhagen, (the seamen's quarter.) A sailor's wife, who was always proud and fine, in her way, came to her neighbor, and complained that she had got a splinter in her finger.

TARGET=Nyboder DISTRACTOR=Copenhagen

TARGET_SOURCE_STATE=signified excessively fine due to a sailor's wife | ANSWER=excessively fine | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=served as the seamen's quarter in Denmark | ANSWER=seamen's quarter | raw_overlap=1.000, exact=True

UPDATE_FRAME: In the story, {ENTITY} became known for its maritime heritage. [strong_replacement_like]

USE_FRAME: The anecdote revealed that Nyboder was associated with {STATE}.

UPDATE_USE: The anecdote revealed that Nyboder was associated with maritime heritage.

RETAIN_USE: The anecdote revealed that Nyboder was associated with excessively fine.

## 11. rw2s1_006146 (simple_wiki; target_later)

flags: update_not_explicit_replacement_like, use_frame_lexically_favors_new_answer

baseline U=4.726 R=-4.543 beta=0.091 |alpha|=4.635 gamma=-4.543 joint=False

SOURCE: The temple is managed and taken care of by Bhagat family, who also built the recent construction of the temple.

TARGET=Bhagat DISTRACTOR=temple

TARGET_SOURCE_STATE=managed and taken care of by Bhagat family | ANSWER=managed and taken care of | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=built the recent construction of the temple | ANSWER=built the recent construction | raw_overlap=1.000, exact=True

UPDATE_FRAME: The {ENTITY} oversaw the restoration of the ancient structure. [weak_or_unclear]

USE_FRAME: The family praised the Bhagat for the {STATE} of the ancient structure.

UPDATE_USE: The family praised the Bhagat for the restoration of the ancient structure of the ancient structure.

RETAIN_USE: The family praised the Bhagat for the managed and taken care of of the ancient structure.

## 12. rw2s1_025014 (simple_wiki; target_earlier)

flags: update_not_explicit_replacement_like

baseline U=4.233 R=-4.432 beta=-0.099 |alpha|=4.332 gamma=-4.432 joint=False

SOURCE: Freud uses the figure Oedipus, from Greek mythology: Oedipus killed a man in a fight, when he didn't know this man was his father, king Laios of Thebes.

TARGET=Oedipus DISTRACTOR=Greek

TARGET_SOURCE_STATE=killed a man in a fight | ANSWER=killed a man | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=originated from mythology | ANSWER=mythology | raw_overlap=1.000, exact=True

UPDATE_FRAME: In the narrative, {ENTITY} was identified as a tragic hero. [weak_or_unclear]

USE_FRAME: The story reveals that Oedipus was a {STATE}.

UPDATE_USE: The story reveals that Oedipus was a tragic hero.

RETAIN_USE: The story reveals that Oedipus was a killed a man.

## 13. rw_025630 (simple_wiki; target_later)

flags: update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=-3.505 R=4.683 beta=0.589 |alpha|=4.094 gamma=-3.505 joint=False

SOURCE: He directed the art exhibition at the Los Angeles County Fair for many years and brought world-class work to Southern California.

TARGET=Southern California DISTRACTOR=Los Angeles County Fair

TARGET_SOURCE_STATE=brought world-class work to the region | ANSWER=world-class work | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=hosted the art exhibition annually | ANSWER=art exhibition | raw_overlap=1.000, exact=True

UPDATE_FRAME: The cultural program featured {ENTITY} showcasing contemporary installations from global artists. [additive_or_descriptive]

USE_FRAME: Visitors admired the {STATE} displayed at the Southern California venue.

UPDATE_USE: Visitors admired the contemporary installations displayed at the Southern California venue.

RETAIN_USE: Visitors admired the world-class work displayed at the Southern California venue.

## 14. rw_041658 (open_subtitles; target_earlier)

flags: target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=3.376 R=-3.442 beta=-0.033 |alpha|=3.409 gamma=-3.442 joint=False

SOURCE: actually of Judaism as a threat in that way at all, until the Zionist movement annexed the messianic, or fused with it, because the messianists didnít used to be Zionists, as you know, so, youíd never know when itís gonna be next.

TARGET=Judaism DISTRACTOR=Zionist

TARGET_SOURCE_STATE=fused with the messianic movement | ANSWER=fused with the messianic movement | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=annexed the messianic movement | ANSWER=annexed the messianic movement | raw_overlap=1.000, exact=False

UPDATE_FRAME: The movement {ENTITY} adopted a secular political identity to broaden its appeal. [additive_or_descriptive]

USE_FRAME: Observers noted that Judaism shifted toward a {STATE} in the modern era.

UPDATE_USE: Observers noted that Judaism shifted toward a secular political identity in the modern era.

RETAIN_USE: Observers noted that Judaism shifted toward a fused with the messianic movement in the modern era.

## 15. rw2s1_029613 (simple_wiki; target_earlier)

flags: update_not_explicit_replacement_like

baseline U=-0.065 R=-3.200 beta=-1.633 |alpha|=1.567 gamma=-3.200 joint=False

SOURCE: At the age of 15 Villanueva became a member of APRA's "Juventud Aprista Peruana" in opposition to the military dictatorship of Luis Miguel Sánchez Cerro.

TARGET=Villanueva DISTRACTOR=APRA's Juventud Aprista Peruana

TARGET_SOURCE_STATE=became a member of APRA's Juventud Aprista Peruana | ANSWER=member of APRA's Juventud Aprista Peruana | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=represented the military dictatorship of Luis Miguel Sánchez Cerro | ANSWER=military dictatorship of Luis Miguel Sánchez Cerro | raw_overlap=1.000, exact=True

UPDATE_FRAME: At the age of 15, {ENTITY} joined the Workers' Union to oppose the regime. [weak_or_unclear]

USE_FRAME: Historical records show Villanueva was a member of the {STATE}.

UPDATE_USE: Historical records show Villanueva was a member of the Workers' Union.

RETAIN_USE: Historical records show Villanueva was a member of the member of APRA's Juventud Aprista Peruana.

## 16. rw2s0_027269 (simple_wiki; target_later)

flags: target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like

baseline U=-2.809 R=2.531 beta=-0.139 |alpha|=2.670 gamma=-2.809 joint=False

SOURCE: He succeeded to the barony in 1998 and lost his seat in the House of Lords in 1999 due to legislative changes.

TARGET=Lords DISTRACTOR=House

TARGET_SOURCE_STATE=lost seat in the House of Lords in 1999 | ANSWER=lost seat | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=succeeded to the barony in 1998 | ANSWER=succeeded to barony | raw_overlap=1.000, exact=False

UPDATE_FRAME: Due to legislative changes, {ENTITY} gained membership in the upper chamber. [weak_or_unclear]

USE_FRAME: The 1999 legislative changes caused the Lords to lose their {STATE}.

UPDATE_USE: The 1999 legislative changes caused the Lords to lose their gained membership.

RETAIN_USE: The 1999 legislative changes caused the Lords to lose their lost seat.

## 17. rw_007895 (simple_wiki; target_later)

flags: update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=4.632 R=-2.749 beta=0.942 |alpha|=3.691 gamma=-2.749 joint=False

SOURCE: In 1891 he ran for municipal elections is supported by young intellectuals who had recently founded La Nuova Sardegna is the unitary list of Republicans and moderate and, elected with broad suffrage, held for years the city councilor activities.

TARGET=Republicans DISTRACTOR=La Nuova Sardegna

TARGET_SOURCE_STATE=held for years the city councilor activities | ANSWER=city councilor activities | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=founded the unitary list of Republicans and moderate | ANSWER=unitary list | raw_overlap=1.000, exact=True

UPDATE_FRAME: In the legislative session, {ENTITY} served as the primary opposition in the assembly. [additive_or_descriptive]

USE_FRAME: The voters elected Republicans as the {STATE} in the legislative session.

UPDATE_USE: The voters elected Republicans as the primary opposition in the legislative session.

RETAIN_USE: The voters elected Republicans as the city councilor activities in the legislative session.

## 18. rw2s1_032450 (simple_wiki; target_earlier)

flags: distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=2.533 R=-2.569 beta=-0.018 |alpha|=2.551 gamma=-2.569 joint=False

SOURCE: The ruins of Gedi are described and repeatedly mentioned in Andrei Gusev's 2020 novel “Our Wild Sex in Malindi”.

TARGET=Gedi DISTRACTOR=Andrei Gusev's

TARGET_SOURCE_STATE=described and repeatedly mentioned in a novel | ANSWER=described and repeatedly mentioned | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=authored a novel in 2020 | ANSWER=authored a novel | raw_overlap=0.500, exact=False

UPDATE_FRAME: The ruins of {ENTITY} are featured as the central location in a film adaptation. [additive_or_descriptive]

USE_FRAME: Travelers visiting Gedi see the ancient {STATE} of the lost city.

UPDATE_USE: Travelers visiting Gedi see the ancient featured as the central location of the lost city.

RETAIN_USE: Travelers visiting Gedi see the ancient described and repeatedly mentioned of the lost city.

## 19. rw2s0_007875 (open_subtitles; target_later)

flags: update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=-2.115 R=3.260 beta=0.572 |alpha|=2.688 gamma=-2.115 joint=False

SOURCE: It didn't reappear for more than a thousand years, when knights from the First Crusade discovered secret vaults beneath the Temple of Solomon.

TARGET=Solomon DISTRACTOR=Temple

TARGET_SOURCE_STATE=contained secret vaults beneath the structure | ANSWER=secret vaults | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=held knights from the First Crusade above ground | ANSWER=knights from the First Crusade | raw_overlap=1.000, exact=True

UPDATE_FRAME: The excavation team revealed ancient gold artifacts to the public from {ENTITY}. [additive_or_descriptive]

USE_FRAME: Archaeologists discovered the hidden {STATE} within Solomon's domain.

UPDATE_USE: Archaeologists discovered the hidden ancient gold artifacts within Solomon's domain.

RETAIN_USE: Archaeologists discovered the hidden secret vaults within Solomon's domain.

## 20. rw_021557 (simple_wiki; target_earlier)

flags: none

baseline U=1.195 R=-1.541 beta=-0.173 |alpha|=1.368 gamma=-1.541 joint=False

SOURCE: It is the second single from his 16th studio album "Working on a Dream" and went to number 41 in Italy and number 18 on the Billboard Adult Alternative Songs chart.

TARGET=Italy DISTRACTOR=Billboard Adult Alternative Songs

TARGET_SOURCE_STATE=went to number 41 in the country | ANSWER=number 41 | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=went to number 18 on the chart | ANSWER=number 18 | raw_overlap=1.000, exact=True

UPDATE_FRAME: The single went to number 1 in the ranking for {ENTITY}. [strong_replacement_like]

USE_FRAME: The chart position for Italy was {STATE}.

UPDATE_USE: The chart position for Italy was number 1.

RETAIN_USE: The chart position for Italy was number 41.

