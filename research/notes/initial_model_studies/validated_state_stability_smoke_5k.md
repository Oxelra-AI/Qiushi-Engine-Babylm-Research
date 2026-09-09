# validated state stability interpretation validated-state stability smoke

Evidence JSON: `experiments/archive/initial_model_studies/data/validated_state_stability_smoke_5k.json`

Cases extracted: 19 from 5000 streamed docs / 3554 basic-quality docs

## Per-model effects

| model | n | R_extra mean | R_extra frac>0 | F_repeat mean | relation_over_filler mean | relation_over_filler frac>0 |
|---|---:|---:|---:|---:|---:|---:|
| wwm43_40M | 18 | +0.7419 | 0.667 | +0.0409 | +0.7010 | 0.500 |
| wwm43_80M | 18 | +1.8795 | 0.667 | +0.2091 | +1.6703 | 0.556 |
| wwm43_100M | 18 | +1.8310 | 0.611 | +0.2747 | +1.5563 | 0.556 |
| wwm42_100M | 18 | +1.3862 | 0.556 | +0.1885 | +1.1977 | 0.556 |

## Cross-model stability

Common cases: 18
All-models R_extra>0: 10 (0.556)
All-models relation_over_filler>0: 7 (0.389)

Interpretation: R_extra alone can select exact repetition. relation_over_filler = R_extra - filler-only repeat effect must be positive and stable before materializing validated-state 1M data.

## Sample cases

- 95:possession:christ:right:90:134 family=possession entity=christ cue=left y1=right y2=serious excerpt=e martyrdom. These graphic depictions are intended to help the viewer equate Barbara’s physical torments with those of Christ, whose crucifixion is depicted in the top center composition. The figures at the top left and right, probably the apostles John and Luke, hold sayings taken from the last words of Christ on the cross, which here apply also to Barbara’s death. The bottom center panel shows Barbara enthroned in heaven, wearing a crown and ho
- 366:role_attribute:nasa:northern:24:116 family=role_attribute entity=nasa cue=known y1=northern y2=invisible excerpt=erature Panel: The Secret Behind the Northern Lights In this interactive session, Norwegian solar physicist Pål Brekke utilizes photos and video footage from NASA satellites to discuss the visual phenomenon known as the Northern Lights. - Sat., Mar. 2, 2013, 10:30 AM Please use the event calendar to search for current events. For the duration of Nordic Cool 2013, the Terrace Gallery will become the Cool Club, an extension of the Nordic Design Ill
- 1113:role_attribute:english:letters:4:55 family=role_attribute entity=english cue=called y1=letters y2=concept excerpt=The first issue of the English florins, so called because the letters D.G. (“by God's grace”) were omitted for want of room. It happened that Richard Lalor Sheil, the master of the Mint, was a Catholic, and a scandal was raised that the omission was made on religious grounds. The florins were called in and re-cast. (See Mr. Sheil was appointed by the Whig ministry Master of the Mint in 1846; he issued the florin in 1849; was removed in 1850, and 
- 1511:role_attribute:theobald:involved:106:138 family=role_attribute entity=theobald cue=became y1=involved y2=apart excerpt=mply a transition to another realm which could be reached any time. It also appealed to those who were tired of dogmatism and wished to experience God in a personal way. The middle-class Theobald family of London became involved in Spiritualism in the 1860s. Morell Theobald lived with his wife and four children. His spinster sister, Florence, often stayed with them. Florence always stated she’d been born “sensitive” and immediately was drawn to t
- 1536:role_attribute:amphitrite:dancing:22:119 family=role_attribute entity=amphitrite cue=called y1=dancing y2=gentlemen excerpt=f Neptune and Click on image for full size Image courtesy of the Philadelphia Museum of Art: The George W. Elkins Collection. Amphitrite was one of the sea-nymphs called the Nereids. One day the sea god Poseidon saw her dancing and fell desperately in love with her. He promptly asked her to marry him but unfortunately she refused. Not discouraged by Amphitrite's refusal, Poseidon (Neptune) sent one of his servants, a dolphin to convince her. The 
- 1881:action_consequence:parts:causing:80:112 family=action_consequence entity=parts cue=dropped y1=causing y2=towns excerpt=be the first time. Norway has experienced several huge tidal waves along the coastline and in the fjords. The best known disaster is the Tafjord slide that occurred in 1934. Parts of the mountainside simply dropped off, causing a 64-metre high tidal wave to wash about 200 metres inland. It completely wiped out the villages of Tafjord and Fjøra in Møre & Romsdal County. Researchers are painfully aware that disasters like that can happen again. Con
- 2170:location:africa:lake:90:241 family=location entity=africa cue=around y1=lake y2=remaining excerpt=h alluvial deposits. In contrast, they referred to the desert land as Red Land. The river Nile is fed by the White Nile, the Blue Nile and the Atbara rivers of central Africa. Nile enters Egypt near Wadi Halfa in Sudan. Lake Nasser to the south of Egypt is a man-made reservoir resultant from the construction of the Aswan Dam across the Nile. The Aswan Low Dam was constructed at the First Cataract of the Nile in 1902. The High Dam was constructed 
- 2566:location:cockroaches:head:9:144 family=location entity=cockroaches cue=behind y1=head y2=satellite excerpt=American Cockroaches are reddish brown and have a yellowish margin on the body region behind the head. American cockroaches generally live in moist areas, but can survive in dry areas if they have access to water. They prefer warm temperatures around 84 degrees Fahrenheit and do not tolerate cold temperatures. In residential areas, these cockroaches live in basements and sewers, and may move outdoors into yards during warm weather. These cockroac
