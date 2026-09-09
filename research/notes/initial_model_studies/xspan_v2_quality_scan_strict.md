# Relational XSpan target quality scan

JSONL: `experiments/archive/initial_model_studies/data/xspan_revision_116/relational_xspan_v2_seed116_target200000_actual.jsonl`
Evidence JSON: `experiments/archive/initial_model_studies/data/xspan_v2_quality_scan_strict.json`

Rows: **7555**
Flagged rows: **1354** (17.922%)
Flagged by type: {'location_spatial_phrase': 1007, 'action_object_result_phrase': 73, 'semantic_content_continuation': 110, 'definition_property_complement': 164}
Reason counts: {'unbalanced_quote_or_bracket': 386, 'too_few_substantive_tokens': 729, 'too_long': 161, 'mid_clause_trigger': 99, 'bad_end': 10}

## Flagged examples
- location_spatial_phrase `from the previous games in the series, "Splatoon` ['unbalanced_quote_or_bracket'] :: It has many returning elements from the previous games in the series, "Splatoon" and "Splatoon 2".
- location_spatial_phrase `in Grenada` ['too_few_substantive_tokens'] :: It is the capital of the parish of Saint Andrew and the third largest town in Grenada.
- location_spatial_phrase `from their fifth studio album "Torch the Moon` ['unbalanced_quote_or_bracket'] :: It was taken from their fifth studio album "Torch the Moon".
- action_object_result_phrase `released the game for the PlayStation 4, PlayStation 5, Windows, Xbox` ['too_long'] :: They released the game for the PlayStation 4, PlayStation 5, Windows, Xbox One, and Xbox Series X/S.
- action_object_result_phrase `named after British survey-ship "Flying-Fish` ['unbalanced_quote_or_bracket'] :: It was originally named after British survey-ship "Flying-Fish", but many maps call it "The Settlement".
- semantic_content_continuation `seasons` ['too_few_substantive_tokens'] :: It ran for 3 seasons.
- location_spatial_phrase `around` ['too_few_substantive_tokens'] :: It was founded around 1908.
- location_spatial_phrase `in France` ['too_few_substantive_tokens'] :: It is used in France.
- semantic_content_continuation `movements` ['too_few_substantive_tokens'] :: It has ten movements.
- location_spatial_phrase `in October` ['too_few_substantive_tokens'] :: It was one of the first area codes created in October 1947.
- location_spatial_phrase `from his first studio album "Greetings from Asbury Park` ['unbalanced_quote_or_bracket'] :: It is taken from his first studio album "Greetings from Asbury Park, N.J." (1973).
- location_spatial_phrase `on the outside` ['too_few_substantive_tokens'] :: It is crispy on the outside.
- location_spatial_phrase `in a planned series of 12 supersized container ships to be` ['too_long'] :: It is the first in a planned series of 12 supersized container ships to be built for Evergreen Marine.
- location_spatial_phrase `in Rhode Island It was described by "The New` ['unbalanced_quote_or_bracket'] :: It is the only Forbes Five-Star and AAA Five Diamond Hotel in Rhode Island It was described by "The New York Times" as a
- definition_property_complement `not clear` ['too_few_substantive_tokens'] :: It is not clear when the wrecking ball was invented.

## Clean examples
- location_spatial_phrase `in exchange for the payment of special taxes` :: Those Christians who did not convert, called Mozarabs, were tolerated by the Muslim rulers in exchange for the payment o
- definition_property_complement `known for its difficult rhythm patterns` :: It is known for its difficult rhythm patterns, and is inspired by free jazz music.
- semantic_content_continuation `route served vital communites without reliable road access` :: This is because the route served vital communites without reliable road access.
- location_spatial_phrase `lives in cloud forests in Venezuela` :: It lives in cloud forests in Venezuela.
- semantic_content_continuation `borders the municipality of Hengelo` :: It borders the municipality of Hengelo.
- definition_property_complement `band's first album after frontman Dave Mustaine's arm injury` :: It was the band's first album after frontman Dave Mustaine's arm injury.
- semantic_content_continuation `gives the following list for a three-day basic emergency` :: It gives the following list for a three-day basic emergency supply kit.
- action_object_result_phrase `stars Carmel Myers` :: It stars Carmel Myers and Rudolph Valentino before they became famous.
- definition_property_complement `Co-Produced by Nelvana Limited` :: It was Co-Produced by Nelvana Limited and Singaporean animation studio IVL Animation with the participation of Teletoon 
- location_spatial_phrase `on the Stanley Cup had ever been corrected` :: It was the first time that a misspelling on the Stanley Cup had ever been corrected.
- semantic_content_continuation `completely surrounds the Edmonton Capital Region` :: It completely surrounds the Edmonton Capital Region.
- action_object_result_phrase `includes extramarital sex` :: It includes extramarital sex and premarital sex.
