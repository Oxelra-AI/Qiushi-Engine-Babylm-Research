# validated state stability interpretation validated-state stability smoke

Evidence JSON: `experiments/archive/initial_model_studies/data/validated_state_stability_smoke_pilot.json`

Cases extracted: 3 from 1500 streamed docs / 1064 basic-quality docs

## Per-model effects

| model | n | R_extra mean | R_extra frac>0 | F_repeat mean | relation_over_filler mean | relation_over_filler frac>0 |
|---|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 3 | +0.7617 | 0.667 | -0.0424 | +0.8042 | 0.333 |
| wwm43_80M | 3 | +3.6819 | 0.667 | +0.3601 | +3.3219 | 0.667 |
| wwm43_100M | 3 | +2.3157 | 0.667 | +0.4214 | +1.8943 | 0.667 |
| wwm42_100M | 3 | +5.0697 | 0.667 | +0.1524 | +4.9172 | 0.667 |

## Cross-model stability

Common cases: 3
All-models R_extra>0: 2 (0.667)
All-models relation_over_filler>0: 1 (0.333)

Interpretation: R_extra alone can select exact repetition. relation_over_filler = R_extra - filler-only repeat effect must be positive and stable before materializing validated-state 1M data.

## Sample cases

- 95:possession:christ:right:90:134 family=possession entity=christ cue=left y1=right y2=stay excerpt=e martyrdom. These graphic depictions are intended to help the viewer equate Barbara’s physical torments with those of Christ, whose crucifixion is depicted in the top center composition. The figures at the top left and right, probably the apostles John and Luke, hold sayings taken from the last words of Christ on the cross, which here apply also to Barbara’s death. The bottom center panel shows Barbara enthroned in heaven, wearing a crown and ho
- 366:role_attribute:nasa:northern:24:116 family=role_attribute entity=nasa cue=known y1=northern y2=holidays excerpt=erature Panel: The Secret Behind the Northern Lights In this interactive session, Norwegian solar physicist Pål Brekke utilizes photos and video footage from NASA satellites to discuss the visual phenomenon known as the Northern Lights. - Sat., Mar. 2, 2013, 10:30 AM Please use the event calendar to search for current events. For the duration of Nordic Cool 2013, the Terrace Gallery will become the Cool Club, an extension of the Nordic Design Ill
- 1113:role_attribute:english:letters:4:55 family=role_attribute entity=english cue=called y1=letters y2=video excerpt=The first issue of the English florins, so called because the letters D.G. (“by God's grace”) were omitted for want of room. It happened that Richard Lalor Sheil, the master of the Mint, was a Catholic, and a scandal was raised that the omission was made on religious grounds. The florins were called in and re-cast. (See Mr. Sheil was appointed by the Whig ministry Master of the Mint in 1846; he issued the florin in 1849; was removed in 1850, and 
