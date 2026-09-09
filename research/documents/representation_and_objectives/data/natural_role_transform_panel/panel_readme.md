# earlier analysis natural role-preservation premise panel

Purpose: test whether an approved teacher can provide stable independent-sentence semantic-role labels on source-attested bridge transformations before any new architecture or Strict-Small training.

Cases: 12; contexts per case: 3; facts per case: 4; prompts: 144.

Each prompt contains exactly one context sentence and one hypothesis; no source/bridge pair is shown in the same prompt.

## B006 — program_improves_food_security (A_seed_transform, deletion_reorder)

Source: There is ample evidence to show that HFP program in Bangladesh has improved food security for more than 5 million vulnerable people in diverse agro-ecological zone.

Bridge: Ample evidence shows HFP program in Bangladesh improved food security for more than 5 million vulnerable people in diverse agro-ecological zone.

Natural compact reference: HFP program in Bangladesh improved food security for over 5 million vulnerable people in diverse agro-ecological zones.

Facts:
- ENTAILED / true_actor_effect: The HFP program improved food security.
- ENTAILED / true_scope: More than 5 million vulnerable people benefited from improved food security.
- NOT_ENTAILED / actor_swap_false: Food security improved the HFP program.
- NOT_ENTAILED / conservation_false: More than 5 million vulnerable people improved the HFP program.

## B019 — wwi_effects_bankrupt_europe (A_seed_transform, light_restructure)

Source: In the 1920s, most of Europe was bankrupt due to after effects of WWI.

Bridge: Most of Europe was bankrupt in the 1920s due to WWI effects.

Natural compact reference: WWI's after effects bankrupted most of Europe in the 1920s.

Facts:
- ENTAILED / true_state: Most of Europe was bankrupt in the 1920s.
- ENTAILED / true_cause: WWI effects were a reason most of Europe was bankrupt.
- NOT_ENTAILED / actor_swap_false: Most of Europe caused WWI effects.
- NOT_ENTAILED / conservation_false: WWI effects were bankrupt because of Europe.

## B024 — households_adopt_fuel (A_seed_transform, light_restructure)

Source: Korean households enthusiastically took up the new fuel which _ as they soon discovered _ was so much more convenient than firewood.

Bridge: Korean households enthusiastically took the new fuel, soon discovering it was more convenient than firewood.

Natural compact reference: Korean households enthusiastically adopted the new fuel as it was much more convenient than firewood.

Facts:
- ENTAILED / true_actor_action: Korean households took the new fuel.
- ENTAILED / true_comparison: The new fuel was more convenient than firewood.
- NOT_ENTAILED / actor_swap_false: The new fuel took Korean households.
- NOT_ENTAILED / conservation_false: Firewood was more convenient than the new fuel.

## B039 — frog_allows_photographs (A_seed_transform, light_restructure)

Source: The frog was very cooperative, allowing for close-up photographs with a 50mm lens coupled with a set of macro tubes.

Bridge: The cooperative frog allowed close-up photographs with a 50mm lens coupled with macro tubes.

Natural compact reference: The frog cooperated for close-up photos with a 50mm lens and macro tubes.

Facts:
- ENTAILED / true_actor_action: The frog allowed close-up photographs.
- ENTAILED / true_instrument: The close-up photographs used a 50mm lens.
- NOT_ENTAILED / actor_swap_false: The close-up photographs allowed the frog.
- NOT_ENTAILED / conservation_false: The 50mm lens cooperated for the frog.

## B067 — china_accuses_others (A_seed_transform, deletion_reorder)

Source: China already has accused that countries such as the Philippines and Vietnam have deliberately used US support to escalate tensions in the South China Sea region.

Bridge: China accused the Philippines and Vietnam of deliberately using US support to escalate tensions in the South China Sea region.

Natural compact reference: China accused the Philippines and Vietnam of using US support to escalate South China Sea tensions.

Facts:
- ENTAILED / true_actor_action: China accused the Philippines and Vietnam.
- ENTAILED / true_embedded_role: China accused the Philippines and Vietnam of using US support.
- NOT_ENTAILED / actor_swap_false: The Philippines and Vietnam accused China of using US support.
- NOT_ENTAILED / conservation_false: US support accused China.

## B069 — war_instability_causes_deaths (A_seed_transform, substantive_restructure)

Source: Most of the deaths have been caused by factors provoked by war's instability and destruction.

Bridge: Most deaths were caused by war's instability.

Natural compact reference: War's instability and destruction caused most deaths.

Facts:
- ENTAILED / true_cause: War's instability caused most deaths.
- ENTAILED / true_theme: Most deaths were caused by war's instability.
- NOT_ENTAILED / actor_swap_false: Most deaths caused war's instability.
- NOT_ENTAILED / conservation_false: War's stability caused most deaths.

## B073 — anagram_made_by_rearranging_letters (A_seed_transform, substantive_restructure)

Source: An Anagram is collection of word or phrase made out by rearranging the letters of the word.

Bridge: An anagram is a collection of words made by rearranging letters.

Natural compact reference: An anagram is a word or phrase made by rearranging letters.

Facts:
- ENTAILED / true_definition: An anagram is made by rearranging letters.
- ENTAILED / true_object: Letters are rearranged to make an anagram.
- NOT_ENTAILED / actor_swap_false: Letters are made by rearranging an anagram.
- NOT_ENTAILED / conservation_false: An anagram rearranges the collection.

## B079 — website_claims_palm_valley_remnant (A_seed_transform, substantive_restructure)

Source: “Palm Valley is a remnant of the rainforests that once covered our ancient continent,” claims the text of an NT government website that is yet to be updated.

Bridge: The text of an NT government website claims Palm Valley is a remnant of rainforests that once covered our ancient continent.

Natural compact reference: An NT government website claims Palm Valley is a rainforest remnant from our ancient continent but is not yet updated.

Facts:
- ENTAILED / true_speaker: The NT government website claims Palm Valley is a rainforest remnant.
- ENTAILED / true_theme: Palm Valley is described as a remnant of rainforests.
- NOT_ENTAILED / actor_swap_false: Palm Valley claims the NT government website is a rainforest remnant.
- NOT_ENTAILED / conservation_false: The NT government website is a remnant of Palm Valley.

## B082 — mercury_upsets_microorganism_balance (A_seed_transform, substantive_restructure)

Source: Furthermore, a natural balance of microorganisms in the body is upset by excess mercury in the gastrointestinal tract, and this imbalance may lead to candida.

Bridge: Furthermore, excess mercury upsets the body's natural microorganism balance, leading to candida.

Natural compact reference: Excess mercury in the gut upsets natural microorganism balance, which may lead to candida.

Facts:
- ENTAILED / true_actor_effect: Excess mercury upsets the body's microorganism balance.
- ENTAILED / true_result: The imbalance may lead to candida.
- NOT_ENTAILED / actor_swap_false: The body's microorganism balance upsets excess mercury.
- NOT_ENTAILED / conservation_false: Candida leads to excess mercury.

## B095 — movement_causes_two_entity_specific_outcomes (A_seed_transform, light_restructure)

Source: Chimanbhai Patel was compelled by the movement to resign and it also gave the then Prime Minister Indira Gandhi an excuse to declare emergency on 25 June, 1975.

Bridge: The movement compelled Chimanbhai Patel to resign and gave Prime Minister Indira Gandhi an excuse to declare emergency on 25 June, 1975.

Natural compact reference: The movement forced Chimanbhai Patel to resign and gave Prime Minister Indira Gandhi an excuse to declare emergency on 25 June, 1975.

Facts:
- ENTAILED / true_actor_action: The movement compelled Chimanbhai Patel to resign.
- ENTAILED / true_other_role: The movement gave Indira Gandhi an excuse to declare emergency.
- NOT_ENTAILED / actor_swap_false: Indira Gandhi was compelled by the movement to resign.
- NOT_ENTAILED / conservation_false: Chimanbhai Patel got an excuse to declare emergency.

## B096 — thoughts_control_us_and_feelings (A_seed_transform, light_restructure)

Source: These thoughts are what control us and affect the way we feel.

Bridge: These thoughts control us and affect how we feel.

Natural compact reference: Thoughts control us and affect how we feel.

Facts:
- ENTAILED / true_actor_action: These thoughts control us.
- ENTAILED / true_effect: These thoughts affect how we feel.
- NOT_ENTAILED / actor_swap_false: We control these thoughts.
- NOT_ENTAILED / conservation_false: How we feel controls these thoughts.

## B050 — resurvey_causes_missouri_claim (B_adjudicate_transform, light_restructure)

Source: Honey War line (1836)-- Brown's most controversial survey was the re-survey of the Sullivan Line, which caused Missouri to claim its border extended 13 miles into Iowa.

Bridge: Brown's controversial 1836 re-survey of the Honey War line caused Missouri to claim its border extended 13 miles into Iowa.

Natural compact reference: Brown's 1836 Sullivan Line re-survey caused Missouri to claim its border extended 13 miles into Iowa.

Facts:
- ENTAILED / true_cause: Brown's re-survey caused Missouri to claim its border extended into Iowa.
- ENTAILED / true_theme: Missouri claimed its border extended 13 miles into Iowa.
- NOT_ENTAILED / actor_swap_false: Missouri's claim caused Brown's re-survey.
- NOT_ENTAILED / conservation_false: Iowa claimed Brown's border extended into Missouri.

