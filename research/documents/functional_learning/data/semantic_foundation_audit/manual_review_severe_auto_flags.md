# manual_review_severe_auto_flags.md

## 1. rw2s0_031693 (simple_wiki; target_later)

flags: target_answer_certified_by_generated_state, distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, target_answer_weak_raw_overlap, distractor_answer_weak_raw_overlap, target_state_weak_raw_overlap, distractor_state_weak_raw_overlap, suspect_entity_surface_form, use_frame_lexically_favors_new_answer

baseline U=0.900 R=-0.670 beta=0.115 |alpha|=0.785 gamma=-0.670 joint=False

SOURCE: Evans died on 9 August 2022 at his home in London, England from a heart attack at the age of 72.

TARGET=London England DISTRACTOR=August

TARGET_SOURCE_STATE=was the location of Evans' death | ANSWER=location of Evans' death | raw_overlap=0.333, exact=False

DISTRACTOR_SOURCE_STATE=was the month of Evans' death | ANSWER=month of Evans' death | raw_overlap=0.333, exact=False

UPDATE_FRAME: The committee announced that {ENTITY} became the site of a memorial service. [strong_replacement_like]

USE_FRAME: The memorial service was held at {STATE} in London England.

UPDATE_USE: The memorial service was held at site of a memorial service in London England.

RETAIN_USE: The memorial service was held at location of Evans' death in London England.

## 2. rw_007527 (simple_wiki; target_earlier)

flags: target_answer_certified_by_generated_state, distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, target_answer_weak_raw_overlap, distractor_answer_weak_raw_overlap, target_state_weak_raw_overlap, distractor_state_weak_raw_overlap, update_not_explicit_replacement_like

baseline U=5.917 R=-9.182 beta=-1.633 |alpha|=7.550 gamma=-9.182 joint=False

SOURCE: Its neighbouring municipalities (clockwise from the north) are: Maroldsweisach, Seßlach, Untermerzbach, Ebern and Burgpreppach.

TARGET=Burgpreppach DISTRACTOR=Maroldsweisach Seßlach Untermerzbach Ebern

TARGET_SOURCE_STATE=listed as the fifth municipality in the clockwise sequence | ANSWER=fifth municipality | raw_overlap=0.000, exact=False

DISTRACTOR_SOURCE_STATE=listed as the first three municipalities in the clockwise sequence | ANSWER=first three municipalities | raw_overlap=0.333, exact=False

UPDATE_FRAME: The regional report designated {ENTITY} as the southernmost settlement in the group. [weak_or_unclear]

USE_FRAME: Travelers noted that Burgpreppach is the {STATE} in the clockwise sequence.

UPDATE_USE: Travelers noted that Burgpreppach is the southernmost settlement in the clockwise sequence.

RETAIN_USE: Travelers noted that Burgpreppach is the fifth municipality in the clockwise sequence.

## 3. rw_023565 (simple_wiki; target_earlier)

flags: target_answer_certified_by_generated_state, distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, target_answer_weak_raw_overlap, distractor_answer_weak_raw_overlap, update_not_explicit_replacement_like, suspect_entity_surface_form

baseline U=0.358 R=-0.598 beta=-0.120 |alpha|=0.478 gamma=-0.598 joint=False

SOURCE: Yirmibeşoğlu died from kidney failure on 2 January 2016 in Istanbul, aged 87.

TARGET=January DISTRACTOR=Istanbul

TARGET_SOURCE_STATE=marked the death date of Yirmibeşoğlu | ANSWER=death date | raw_overlap=0.000, exact=False

DISTRACTOR_SOURCE_STATE=hosted the death location of Yirmibeşoğlu | ANSWER=death location | raw_overlap=0.000, exact=False

UPDATE_FRAME: The record for {ENTITY} marked the birth date of Yirmibeşoğlu. [weak_or_unclear]

USE_FRAME: The official record shows {STATE} for January.

UPDATE_USE: The official record shows birth date for January.

RETAIN_USE: The official record shows death date for January.

## 4. rw2s0_029990 (simple_wiki; target_earlier)

flags: distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap, distractor_state_weak_raw_overlap, update_not_explicit_replacement_like, suspect_entity_surface_form

baseline U=1.774 R=-2.692 beta=-0.459 |alpha|=2.233 gamma=-2.692 joint=False

SOURCE: The first choir that recorded it was the "Coro Supramonte" of Orgosolo which, finally, released it in 1974 on an LP for Fonit Cetra.

TARGET=Orgosolo DISTRACTOR=LP

TARGET_SOURCE_STATE=released the recording in 1974 on an LP for Fonit Cetra | ANSWER=released the recording | raw_overlap=0.500, exact=False

DISTRACTOR_SOURCE_STATE=was the physical format for the 1974 release | ANSWER=physical format | raw_overlap=0.000, exact=False

UPDATE_FRAME: The first choir that recorded it was the "Coro Supramonte" of {ENTITY} which issued the recording on a digital download for streaming. [weak_or_unclear]

USE_FRAME: Listeners discovered the {STATE} from Orgosolo in the 1974 release.

UPDATE_USE: Listeners discovered the digital download from Orgosolo in the 1974 release.

RETAIN_USE: Listeners discovered the released the recording from Orgosolo in the 1974 release.

## 5. rw2s0_008771 (gutenberg; target_earlier)

flags: distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap, distractor_state_weak_raw_overlap, update_not_explicit_replacement_like, update_additive_or_coexistence_risk

baseline U=0.674 R=-0.535 beta=0.069 |alpha|=0.605 gamma=-0.535 joint=False

SOURCE: Indeed, like many other good things which have loomed up conspicuously in recent times, arbitration can be traced back to the ancient Greeks, for whom it occasionally mitigated the evils attendant upon frequent warfare between their city-states.

TARGET=Greeks DISTRACTOR=city

TARGET_SOURCE_STATE=occasionally mitigated the evils attendant upon frequent warfare | ANSWER=mitigated evils | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=served as the political units engaging in frequent warfare | ANSWER=political units | raw_overlap=0.000, exact=False

UPDATE_FRAME: Ancient {ENTITY} adopted formal arbitration treaties for conflict resolution. [additive_or_descriptive]

USE_FRAME: Historical records show the Greeks utilized {STATE} to resolve disputes.

UPDATE_USE: Historical records show the Greeks utilized arbitration treaties to resolve disputes.

RETAIN_USE: Historical records show the Greeks utilized mitigated evils to resolve disputes.

## 6. rw_029261 (simple_wiki; target_later)

flags: target_answer_certified_by_generated_state, distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, target_answer_weak_raw_overlap, distractor_answer_weak_raw_overlap

baseline U=7.331 R=-7.357 beta=-0.013 |alpha|=7.344 gamma=-7.357 joint=False

SOURCE: Flower chose him as the judge for the trial of Bartholomew Shea and John McGough for the murder of Robert Ross.

TARGET=Robert Ross DISTRACTOR=John McGough

TARGET_SOURCE_STATE=victim of the murder | ANSWER=victim | raw_overlap=0.000, exact=False

DISTRACTOR_SOURCE_STATE=accused of the murder | ANSWER=accused | raw_overlap=0.000, exact=False

UPDATE_FRAME: The jury declared {ENTITY} guilty of the crime. [strong_replacement_like]

USE_FRAME: The verdict stated Robert Ross was {STATE}.

UPDATE_USE: The verdict stated Robert Ross was guilty.

RETAIN_USE: The verdict stated Robert Ross was victim.

## 7. rw2s1_000371 (gutenberg; target_later)

flags: distractor_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap, distractor_state_weak_raw_overlap, update_not_explicit_replacement_like

baseline U=3.092 R=-3.167 beta=-0.038 |alpha|=3.129 gamma=-3.167 joint=False

SOURCE: Brennus chose out the hardiest of his mountaineers, and directed them to climb up in the dead of night, one by one, in perfect silence, and thus to surprise the Romans, and complete the slaughter and victory, before the forces assembling at Veii would come to their rescue.

TARGET=Veii DISTRACTOR=Romans

TARGET_SOURCE_STATE=assembled forces to come to the rescue | ANSWER=assembling forces | raw_overlap=1.000, exact=False

DISTRACTOR_SOURCE_STATE=were the intended victims of surprise attack | ANSWER=victims of surprise attack | raw_overlap=0.333, exact=False

UPDATE_FRAME: Before the rescue attempt, {ENTITY} had already been defeated in battle. [weak_or_unclear]

USE_FRAME: The soldiers at Veii were surprised to find the city {STATE}.

UPDATE_USE: The soldiers at Veii were surprised to find the city defeated in battle.

RETAIN_USE: The soldiers at Veii were surprised to find the city assembling forces.

## 8. rw_007461 (gutenberg; target_earlier)

flags: target_answer_certified_by_generated_state, target_answer_not_exact_raw_span, distractor_answer_not_exact_raw_span, target_answer_weak_raw_overlap, target_state_weak_raw_overlap, update_not_explicit_replacement_like

baseline U=1.219 R=-1.367 beta=-0.074 |alpha|=1.293 gamma=-1.367 joint=False

SOURCE: It so happened that no dress could suit Kitty better, and doubtless Sir John had an eye to the appearance of his favorite in such a robe when he ordered it.

TARGET=Kitty DISTRACTOR=Sir John

TARGET_SOURCE_STATE=wore a robe that suited her perfectly | ANSWER=perfectly suited robe | raw_overlap=0.333, exact=False

DISTRACTOR_SOURCE_STATE=ordered a robe for his favorite | ANSWER=ordered favorite's robe | raw_overlap=0.667, exact=False

UPDATE_FRAME: The gathering required {ENTITY} to don a gown of silk and lace. [weak_or_unclear]

USE_FRAME: Observers noted how Kitty's {STATE} complemented her features.

UPDATE_USE: Observers noted how Kitty's gown of silk and lace complemented her features.

RETAIN_USE: Observers noted how Kitty's perfectly suited robe complemented her features.

## 9. rw_031559 (childes; target_later)

flags: distractor_answer_certified_by_generated_state, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap, update_not_explicit_replacement_like, suspect_entity_surface_form

baseline U=3.927 R=-4.319 beta=-0.196 |alpha|=4.123 gamma=-4.319 joint=False

SOURCE: *CHI: [smiles] *INV: and I got really nervous because it was really really big dog.

TARGET=*INV DISTRACTOR=*CHI

TARGET_SOURCE_STATE=got really nervous because it was really really big dog | ANSWER=really nervous | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=smiled while the dog was really really big | ANSWER=smiled | raw_overlap=0.000, exact=False

UPDATE_FRAME: After the visit, {ENTITY} felt calm because the dog was friendly. [weak_or_unclear]

USE_FRAME: The observer noted that *INV was {STATE} after the visit.

UPDATE_USE: The observer noted that *INV was calm after the visit.

RETAIN_USE: The observer noted that *INV was really nervous after the visit.

## 10. rw_012466 (childes; target_later)

flags: distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like, suspect_entity_surface_form, use_frame_lexically_favors_new_answer

baseline U=6.645 R=-6.507 beta=0.069 |alpha|=6.576 gamma=-6.507 joint=False

SOURCE: *CHI: [places a silver man in the upside-down position on the tower of silver men] *MOT: uhhuh.

TARGET=*MOT DISTRACTOR=*CHI

TARGET_SOURCE_STATE=acknowledged the silver man placement | ANSWER=uhhuh | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=placed the silver man upside-down on the tower | ANSWER=placed silver man | raw_overlap=0.667, exact=False

UPDATE_FRAME: After the demonstration, {ENTITY} removed the silver man from the tower. [weak_or_unclear]

USE_FRAME: The observer noted that *MOT acknowledged the {STATE} of the silver man.

UPDATE_USE: The observer noted that *MOT acknowledged the removed silver man of the silver man.

RETAIN_USE: The observer noted that *MOT acknowledged the uhhuh of the silver man.

## 11. rw_041658 (open_subtitles; target_earlier)

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

## 12. rw2s0_027873 (gutenberg; target_earlier)

flags: distractor_answer_certified_by_generated_state, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap, update_not_explicit_replacement_like

baseline U=-2.663 R=2.574 beta=-0.045 |alpha|=2.618 gamma=-2.663 joint=False

SOURCE: He rode to the scaffold in his own carriage, attended by two famous clergymen, Tillotson and Burnet, and sang a psalm to himself very softly, as he went along.

TARGET=Tillotson DISTRACTOR=Burnet

TARGET_SOURCE_STATE=sang a psalm to himself very softly | ANSWER=a psalm | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=attended the carriage procession | ANSWER=the procession | raw_overlap=0.000, exact=False

UPDATE_FRAME: The clergyman {ENTITY} recited a hymn aloud during the journey. [weak_or_unclear]

USE_FRAME: Observers noted that Tillotson recited a {STATE} during the journey.

UPDATE_USE: Observers noted that Tillotson recited a a hymn during the journey.

RETAIN_USE: Observers noted that Tillotson recited a a psalm during the journey.

## 13. rw_016168 (simple_wiki; target_later)

flags: distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like, update_additive_or_coexistence_risk, suspect_entity_surface_form

baseline U=1.285 R=-2.487 beta=-0.601 |alpha|=1.886 gamma=-2.487 joint=False

SOURCE: She traveled around the United States and learned more about Native farming and cooking.

TARGET=Native DISTRACTOR=United States

TARGET_SOURCE_STATE=learned more about farming and cooking | ANSWER=farming and cooking | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=traveled around the country | ANSWER=traveled around the country | raw_overlap=0.667, exact=False

UPDATE_FRAME: After the journey, {ENTITY} studied sustainable agriculture methods in depth. [additive_or_descriptive]

USE_FRAME: The guide explained that Native people mastered {STATE} for survival.

UPDATE_USE: The guide explained that Native people mastered sustainable agriculture methods for survival.

RETAIN_USE: The guide explained that Native people mastered farming and cooking for survival.

## 14. rw_022048 (childes; target_later)

flags: distractor_answer_not_exact_raw_span, target_state_weak_raw_overlap, update_not_explicit_replacement_like, suspect_entity_surface_form

baseline U=2.368 R=-2.415 beta=-0.024 |alpha|=2.392 gamma=-2.415 joint=False

SOURCE: [FAT and Ross talk in the background while MOT get's things organized] *CHI: okay.

TARGET=*CHI DISTRACTOR=MOT

TARGET_SOURCE_STATE=said okay to the group | ANSWER=okay | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=got things organized in the room | ANSWER=organized things | raw_overlap=1.000, exact=False

UPDATE_FRAME: During the meeting, {ENTITY} agreed to lead the next task. [weak_or_unclear]

USE_FRAME: The team asked *CHI to confirm the {STATE} for the schedule.

UPDATE_USE: The team asked *CHI to confirm the lead the next task for the schedule.

RETAIN_USE: The team asked *CHI to confirm the okay for the schedule.

## 15. rw2s1_004551 (gutenberg; target_earlier)

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

## 16. rw2s1_006925 (simple_wiki; target_earlier)

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

## 17. rw_012042 (simple_wiki; target_later)

flags: distractor_answer_not_exact_raw_span, update_not_explicit_replacement_like, suspect_entity_surface_form

baseline U=3.767 R=-4.380 beta=-0.307 |alpha|=4.073 gamma=-4.380 joint=False

SOURCE: was the question posed by Time Out Magazine critic Malcolm Hay Personal life.

TARGET=Malcolm Hay Personal DISTRACTOR=Time Out Magazine

TARGET_SOURCE_STATE=posed the question about Personal life | ANSWER=Personal life | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=posed the question about Magazine content | ANSWER=Magazine content | raw_overlap=0.500, exact=False

UPDATE_FRAME: The inquiry was posed by {ENTITY} regarding Career details. [weak_or_unclear]

USE_FRAME: The inquiry posed by Malcolm Hay Personal concerned {STATE}.

UPDATE_USE: The inquiry posed by Malcolm Hay Personal concerned Career details.

RETAIN_USE: The inquiry posed by Malcolm Hay Personal concerned Personal life.

## 18. rw_002445 (simple_wiki; target_later)

flags: distractor_answer_certified_by_generated_state, distractor_answer_not_exact_raw_span, distractor_answer_weak_raw_overlap

baseline U=-3.102 R=3.641 beta=0.270 |alpha|=3.371 gamma=-3.102 joint=False

SOURCE: Overton had Lawrence Reed as the best man at his wedding in 2003 and Overton and his wife died in a plane accident a few months later.

TARGET=Lawrence Reed DISTRACTOR=Overton

TARGET_SOURCE_STATE=served as the best man at the wedding | ANSWER=best man | raw_overlap=1.000, exact=True

DISTRACTOR_SOURCE_STATE=was the groom at the wedding | ANSWER=groom | raw_overlap=0.000, exact=False

UPDATE_FRAME: After the plane crash, {ENTITY} became a widower. [strong_replacement_like]

USE_FRAME: The ceremony honored the {STATE} Lawrence Reed.

UPDATE_USE: The ceremony honored the widower Lawrence Reed.

RETAIN_USE: The ceremony honored the best man Lawrence Reed.

## 19. rw2s0_027269 (simple_wiki; target_later)

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

## 20. rw2s1_032450 (simple_wiki; target_earlier)

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

