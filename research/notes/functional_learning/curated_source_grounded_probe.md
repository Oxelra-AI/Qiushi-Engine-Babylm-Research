# Step038c curated source-grounded probe

This tiny probe materializes six source-grounded recipient contrasts under the curated source grounded probe contract. It is not a training set; it checks that the row schema and scorer can operate on semantically clearer examples while generation and validation remain under revision.

Rows: `experiments/archive/functional_learning/data/curated_source_grounded_probe/curated_source_grounded_training_rows.jsonl`; pairs: `experiments/archive/functional_learning/data/curated_source_grounded_probe/curated_source_grounded_pairs.jsonl`.

## sgp_chart_netherlands_australia

SOURCE: It went to number 1 in New Zealand, number 3 in the United States, number 11 in Australia, number 14 in Canada and the Netherlands and number 20 in Belgium.

TARGET=Netherlands source answer `number 14`; DISTRACTOR=Australia source answer `number 11`; new answer `number 5`.

UPDATE frame: After the reissue, the current chart position for {ENTITY} changed to number 5.

USE frame: The current chart position for Netherlands is {STATE}.

## sgp_chart_italy_billboard

SOURCE: It is the second single from his 16th studio album "Working on a Dream" and went to number 41 in Italy and number 18 on the Billboard Adult Alternative Songs chart.

TARGET=Italy source answer `number 41`; DISTRACTOR=Billboard Adult Alternative Songs source answer `number 18`; new answer `number 1`.

UPDATE frame: After the reissue, the current chart position for {ENTITY} changed to number 1.

USE frame: The current chart position for Italy is {STATE}.

## sgp_pope_action_nicholas_photios

SOURCE: Pope Nicholas I had refused to recognize Patriarch Photios I of Constantinople, who in turn had attacked the pope as a heretic.

TARGET=Pope Nicholas I source answer `refused to recognize`; DISTRACTOR=Patriarch Photios I source answer `attacked the pope as a heretic`; new answer `exchanged diplomatic letters`.

UPDATE frame: After the settlement, the current recorded action for {ENTITY} changed to exchanged diplomatic letters.

USE frame: The current recorded action for Pope Nicholas I is {STATE}.

## sgp_music_video_fatima_trainor

SOURCE: Fatima Robinson directed the music video for "No", which features Trainor performing choreographed dances in a warehouse and entwining her arms with accompanying female dancers.

TARGET=Fatima Robinson source answer `directed`; DISTRACTOR=Trainor source answer `choreographed dances`; new answer `performed live vocals`.

UPDATE frame: The production notes updated the current contribution for {ENTITY} to performed live vocals.

USE frame: The current contribution for Fatima Robinson is {STATE}.

## sgp_military_listing_iraqi_bombers

SOURCE: At the same time, the three thousand five hundred tanks and three hundred and ninety thousand Iraqi troops have dug into defensive positions which make them difficult to knock out, even with the sophistication of British and United States ground attack bombers.

TARGET=Iraqi troops source answer `defensive positions`; DISTRACTOR=British and United States ground attack bombers source answer `ground attack bombers`; new answer `mobile artillery units`.

UPDATE frame: The current military listing for {ENTITY} changed to mobile artillery units.

USE frame: The current military listing for Iraqi troops is {STATE}.

## sgp_brand_label_mcafee_intel

SOURCE: This bore the McAfee brand-name for years, until it was bought by Intel and given the Intel name.

TARGET=McAfee source answer `McAfee brand-name`; DISTRACTOR=Intel source answer `Intel name`; new answer `corporate logo`.

UPDATE frame: After the redesign, the current product label for {ENTITY} changed to corporate logo.

USE frame: The current product label for McAfee is {STATE}.
