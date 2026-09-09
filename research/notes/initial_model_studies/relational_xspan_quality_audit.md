# relational xspan materialization v2 — relational XSpan quality audit

JSONL: `experiments/archive/initial_model_studies/data/xspan_revision_115/relational_xspan_rule_seed115_target200000_actual.jsonl`
Evidence JSON: `experiments/archive/initial_model_studies/data/relational_xspan_quality_audit.json`

Rows: **7611**
Flagged rows: **1077** (14.151%)
Flagged by type: {'definition_property_complement': 320, 'semantic_content_continuation': 264, 'location_spatial_phrase': 461, 'action_object_result_phrase': 32}
Reason counts: {'bad_end': 1045, 'bad_start': 40, 'too_long': 4}

## Example flagged targets
- definition_property_complement `replaced by "Hey, Slavs" in` ['bad_end'] :: It was replaced by "Hey, Slavs" in 1941.
- semantic_content_continuation `categorises computer network traffic according` ['bad_end'] :: It categorises computer network traffic according to various parameters into a number of "traffic classes".
- semantic_content_continuation `because these people were made` ['bad_start'] :: They were sad because these people were made invisible.
- definition_property_complement `however, exported until` ['bad_start', 'bad_end'] :: They were, however, exported until 1978.
- semantic_content_continuation `received town status in` ['bad_end'] :: It received town status in 1423.
- location_spatial_phrase `is in Normandy in the Manche department in` ['bad_end'] :: It is in Normandy in the Manche department in northwest France.
- location_spatial_phrase `is around 13 kilometres southeast of Sydney CBD in` ['bad_end'] :: It is around 13 kilometres southeast of Sydney CBD in the Local Government Area Randwick City near Malabar .
- definition_property_complement `because all his sons were not worthy` ['bad_start'] :: It was because all his sons were not worthy enough to be the ruler.
- semantic_content_continuation `became official the week of` ['bad_end'] :: It became official the week of October 25, 2003.
- location_spatial_phrase `on The Secret", the predecessor of "The` ['bad_end'] :: It was produced by Drew Heriot, who already worked as a director "on The Secret", the predecessor of "The Mose
- definition_property_complement `sung to the same tune as "The` ['bad_end'] :: It is sung to the same tune as "The Song of Australia" which was made by Carl Linger.
- definition_property_complement `good for automating many small pieces of` ['bad_end'] :: They are good for automating many small pieces of work.
- location_spatial_phrase `outside the village of Cley next the` ['bad_end'] :: It is just outside the village of Cley next the Sea, Norfolk.
- definition_property_complement `official office of the President of the` ['bad_end'] :: It is the official office of the President of the Czech Republic.
- action_object_result_phrase `uses its features to lead consumers directly to` ['bad_end'] :: This is because a well-designed website uses its features to lead consumers directly to what they need without

## Example clean targets
- location_spatial_phrase `is in Brittany in the Côtes-d'Armor` :: It is in Brittany in the Côtes-d'Armor department in northwest France.
- location_spatial_phrase `east of Thurso` :: It is 5 miles east of Thurso.
- action_object_result_phrase `includes two direct sequels, three third-person shooters` :: This includes two direct sequels, three third-person shooters, and two spin-offs.
- location_spatial_phrase `in the United States` :: It appears as a bonus track to her sixth studio album "Cry Pretty" and went to number 47 in the United States.
- definition_property_complement `named clothes irons` :: They are named clothes irons because they used to be made out of the metal iron.
- location_spatial_phrase `west from the Watch Hill business district` :: It extends west from the Watch Hill business district, and Sandy Point.
- definition_property_complement `distributed by Paramount Pictures` :: It was distributed by Paramount Pictures.
- definition_property_complement `later destroyed` :: It was later destroyed.
- location_spatial_phrase `based on 36 critics` :: It was based on 36 critics, which indicates "mixed or average reviews".
- semantic_content_continuation `allows to welcome the President` :: It allows to welcome the President of the French Republic, members of the government, and all foreign guests.
