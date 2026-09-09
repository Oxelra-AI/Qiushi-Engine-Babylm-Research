# earlier analysis accepted cue-controlled plain-use packet sample

These examples passed step044 copy like diagnosis changed-form state/content checks and the earlier analysis no-use-cue filter.

## UPDATED_USE

### 1. rw_000579

SOURCE: All was indeed well; she was the mother of a genius, a girl who had achieved such high honor that her name in future would always be remembered in the neighborhood of Cherry Court School.

TARGET: Cherry Court School | UPDATED: Cherry Court School

SOURCE_STATE: neighborhood location | NEW_STATE: historic landmark site

UPDATE: The board officially designated Cherry Court School as a historic landmark site for the community.

USE: Tourists visit the historic landmark site known as Cherry Court School.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.375, LCS=3 `cherry court school`; use-update LCS=3 `cherry court school`

STATE_TERMS source=['neighborhood'] new=['historic', 'landmark'] hits_update=['historic', 'landmark'] hits_use_new=['historic', 'landmark'] hits_use_source=[]

### 2. rw2s0_000891

SOURCE: The next excitement was the coming of her father, for whom Dorothy watched and who appeared almost gladder than anyone that his wife and little girl were at home again.

TARGET: Dorothy | UPDATED: Dorothy

SOURCE_STATE: watching her father | NEW_STATE: reading a book

UPDATE: Dorothy stopped watching her father and began reading a book instead.

USE: Dorothy reads a book while sitting in the chair.

USE_CUES: []

SURFACE use-source: J=0.053, overlap_min=0.200, LCS=1 `the`; use-update LCS=3 `read a book`

STATE_TERMS source=['watch', 'father'] new=['read', 'book'] hits_update=['read', 'book'] hits_use_new=['read', 'book'] hits_use_source=[]

### 3. rw2s1_005839

SOURCE: After the play, Abigail says fake Fester must go away again, so the Addamses throw a large party.

TARGET: Abigail | UPDATED: Abigail

SOURCE_STATE: says fake Fester must go away again | NEW_STATE: planners for the Addamses celebration

UPDATE: Following the performance, Abigail joined the Addamses as planners for their large celebration party.

USE: Abigail coordinates decorations for the Addamses celebration party.

USE_CUES: []

SURFACE use-source: J=0.214, overlap_min=0.500, LCS=2 `the addams`; use-update LCS=2 `the addams`

STATE_TERMS source=['says', 'fake', 'fester', 'away', 'again'] new=['planner', 'addams', 'celebration'] hits_update=['planner', 'addams', 'celebration'] hits_use_new=['addams', 'celebration'] hits_use_source=[]

### 4. rw_025416

SOURCE: Still to come, the W I protest about food irradiation, but first, Oxford teacher, Chris Seawright is claiming that the A V A-Level exam in business studies is too easy.

TARGET: Chris Seawright | UPDATED: Chris Seawright

SOURCE_STATE: Oxford teacher | NEW_STATE: leading local food safety initiative

UPDATE: Chris Seawright has begun leading a local food safety initiative after his protest.

USE: The community respects Chris Seawright for his work in the food safety initiative.

USE_CUES: []

SURFACE use-source: J=0.150, overlap_min=0.375, LCS=2 `chri seawright`; use-update LCS=3 `food safety initiative`

STATE_TERMS source=['oxford', 'teacher'] new=['lead', 'food', 'safety', 'initiative'] hits_update=['lead', 'food', 'safety', 'initiative'] hits_use_new=['food', 'safety', 'initiative'] hits_use_source=[]

### 5. rw_017076

SOURCE: Oh, how lovely, Reliance, you have brought up grandma's dear little dishes that were given her when she was a little girl.

TARGET: Reliance | UPDATED: Reliance

SOURCE_STATE: holding grandma's dishes | NEW_STATE: curator of antique ceramics

UPDATE: Reliance now serves as the curator of antique ceramics for the museum.

USE: Reliance organizes the antique ceramics collection with great care.

USE_CUES: []

SURFACE use-source: J=0.062, overlap_min=0.143, LCS=1 `reliance`; use-update LCS=2 `antique ceramic`

STATE_TERMS source=['hold', 'grandma', 'dish'] new=['curator', 'antique', 'ceramic'] hits_update=['curator', 'antique', 'ceramic'] hits_use_new=['antique', 'ceramic'] hits_use_source=[]

### 6. rw_025052

SOURCE: In 135 A.D, the name was once again changed to Palestine after the Romans conquered the land.

TARGET: Palestine | UPDATED: Palestine

SOURCE_STATE: conquered by Romans | NEW_STATE: modern sovereign territory

UPDATE: The ancient land known as Palestine has become a modern sovereign territory.

USE: Palestine operates as a modern sovereign territory.

USE_CUES: []

SURFACE use-source: J=0.083, overlap_min=0.200, LCS=1 `a`; use-update LCS=4 `a modern sovereign territory`

STATE_TERMS source=['conquer', 'roman'] new=['modern', 'sovereign', 'territory'] hits_update=['modern', 'sovereign', 'territory'] hits_use_new=['modern', 'sovereign', 'territory'] hits_use_source=[]

### 7. rw2s0_023427

SOURCE: Molly had lived most of her fourteen years at some Western military post, and she recognized at once this "order" as she rode in.

TARGET: Western | UPDATED: Western

SOURCE_STATE: military post | NEW_STATE: coastal town

UPDATE: The settlers transformed the Western military post into a bustling coastal town.

USE: Fishermen gather at the Western coastal town every morning.

USE_CUES: []

SURFACE use-source: J=0.067, overlap_min=0.167, LCS=1 `at`; use-update LCS=2 `the western`

STATE_TERMS source=['military', 'post'] new=['coastal', 'town'] hits_update=['coastal', 'town'] hits_use_new=['coastal', 'town'] hits_use_source=[]

### 8. rw_026455

SOURCE: The eminent Scottish divine Rutherford gave expression to the prevailing sentiment when he declared, "We regard toleration of all religions as not far removed from blasphemy." Nevertheless, the movement which gave rise to Presbyterianism and to Independency was sure to advance to the announcement of the principle of universal toleration.

TARGET: Rutherford | UPDATED: Rutherford

SOURCE_STATE: Scottish divine | NEW_STATE: historical theologian

UPDATE: The historical theologian Rutherford expressed strong views on religious intolerance previously.

USE: Scholars study the historical theologian Rutherford for his early writings.

USE_CUES: []

SURFACE use-source: J=0.032, overlap_min=0.143, LCS=1 `the`; use-update LCS=4 `the historical theologian rutherford`

STATE_TERMS source=['scottish', 'divine'] new=['historical', 'theologian'] hits_update=['historical', 'theologian'] hits_use_new=['historical', 'theologian'] hits_use_source=[]

### 9. rw2s0_029333

SOURCE: The album is Brown's first release through RCA Records, following the closure of Jive Records in October 2011.

TARGET: Jive Records | UPDATED: Jive Records

SOURCE_STATE: closed in October 2011 | NEW_STATE: defunct music label

UPDATE: Jive Records ceased operations and became a defunct music label after the closure.

USE: The defunct music label Jive Records once signed many pop artists.

USE_CUES: []

SURFACE use-source: J=0.105, overlap_min=0.200, LCS=2 `jive record`; use-update LCS=3 `defunct music label`

STATE_TERMS source=['clos', 'october', '2011'] new=['defunct', 'music', 'label'] hits_update=['defunct', 'music', 'label'] hits_use_new=['defunct', 'music', 'label'] hits_use_source=[]

### 10. rw2s0_029906

SOURCE: Barbara got very friendly with them, and one day Helen and Susan were coming to tea with her, because it was her last day but one.

TARGET: Helen | UPDATED: Helen

SOURCE_STATE: coming to tea | NEW_STATE: leading the bakery team

UPDATE: Helen decided to leave her tea plans to lead the bakery team instead.

USE: Helen directs the morning bake cycle with her bakery team.

USE_CUES: []

SURFACE use-source: J=0.067, overlap_min=0.143, LCS=2 `with her`; use-update LCS=2 `bakery team`

STATE_TERMS source=['coming', 'tea'] new=['lead', 'bakery', 'team'] hits_update=['lead', 'bakery', 'team'] hits_use_new=['bakery', 'team'] hits_use_source=[]

### 11. rw2s0_020563

SOURCE: The weather for the Fox F M area: the evening will start dry and cloudy but outbreaks of rain will reach the area later on.

TARGET: Fox | UPDATED: Fox

SOURCE_STATE: dry and cloudy | NEW_STATE: rain showers

UPDATE: Oscillating weather patterns brought rain showers to the Fox F M area later in the evening.

USE: The Fox F M area experienced rain showers during the late hours.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.286, LCS=5 `the fox f m area`; use-update LCS=5 `the fox f m area`

STATE_TERMS source=['dry', 'cloudy'] new=['rain', 'shower'] hits_update=['rain', 'shower'] hits_use_new=['rain', 'shower'] hits_use_source=[]

### 12. rw_006379

SOURCE: As a member of Bon Jovi, Sambora collected more than seventy gold and platinum CDs.

TARGET: Sambora | UPDATED: Sambora

SOURCE_STATE: member of Bon Jovi | NEW_STATE: electric guitar specialist

UPDATE: After his tenure with Bon Jovi, Sambora established himself as an electric guitar specialist.

USE: Fans admire Sambora for his electric guitar specialist skills.

USE_CUES: []

SURFACE use-source: J=0.062, overlap_min=0.143, LCS=1 `sambora`; use-update LCS=3 `electric guitar specialist`

STATE_TERMS source=['member', 'bon', 'jovi'] new=['electric', 'guitar', 'specialist'] hits_update=['electric', 'guitar', 'specialist'] hits_use_new=['electric', 'guitar', 'specialist'] hits_use_source=[]

### 13. rw2s0_025710

SOURCE: The word "synod" also refers to the standing council of high-ranking bishops governing some of the autocephalous Eastern Orthodox churches.

TARGET: Eastern Orthodox | UPDATED: Eastern Orthodox

SOURCE_STATE: autocephalous churches | NEW_STATE: governed by synod

UPDATE: The standing council of bishops governs some autocephalous Eastern Orthodox churches.

USE: Bishops within the Eastern Orthodox structure meet at the synod.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.571, LCS=2 `eastern orthodox`; use-update LCS=2 `eastern orthodox`

STATE_TERMS source=['autocephalou', 'church'] new=['govern', 'synod'] hits_update=['govern'] hits_use_new=['synod'] hits_use_source=[]

### 14. rw_001190

SOURCE: They laid him in a pretty corner of the little cemetery from which can be seen the old church Ted loved so well, and the beautiful chase, where he so often walked.

TARGET: Ted | UPDATED: Ted

SOURCE_STATE: loved the old church | NEW_STATE: buried in a pretty corner

UPDATE: They laid Ted in a pretty corner of the little cemetery.

USE: Visitors can see Ted resting in that pretty corner.

USE_CUES: []

SURFACE use-source: J=0.176, overlap_min=0.500, LCS=2 `pretty corner`; use-update LCS=2 `pretty corner`

STATE_TERMS source=['loved', 'church'] new=['bury', 'pretty', 'corner'] hits_update=['pretty', 'corner'] hits_use_new=['pretty', 'corner'] hits_use_source=[]

### 15. rw2s0_010446

SOURCE: The museum allows visitors to enter the cab and use various car controls including the motors, the brakes and the pantograph.

TARGET: museum | UPDATED: museum

SOURCE_STATE: allows visitors to enter the cab | NEW_STATE: interactive control center

UPDATE: The museum now operates as an interactive control center for public use.

USE: Guests explore the interactive control center features at this museum.

USE_CUES: []

SURFACE use-source: J=0.111, overlap_min=0.286, LCS=1 `the`; use-update LCS=3 `interactive control center`

STATE_TERMS source=['allow', 'visitor', 'enter', 'cab'] new=['interactive', 'control', 'center'] hits_update=['interactive', 'control', 'center'] hits_use_new=['interactive', 'control', 'center'] hits_use_source=[]

### 16. rw2s1_010519

SOURCE: According to the Köppen Climate Classification system, Bohemia has a humid subtropical climate, abbreviated "Cfa" on climate maps.

TARGET: Bohemia | UPDATED: Bohemia

SOURCE_STATE: humid subtropical climate | NEW_STATE: arid desert environment

UPDATE: Geological shifts transformed Bohemia into an arid desert environment.

USE: The arid desert environment of Bohemia supports sparse vegetation.

USE_CUES: []

SURFACE use-source: J=0.059, overlap_min=0.143, LCS=1 `the`; use-update LCS=3 `arid desert environment`

STATE_TERMS source=['humid', 'subtropical', 'climate'] new=['arid', 'desert', 'environment'] hits_update=['arid', 'desert', 'environment'] hits_use_new=['arid', 'desert', 'environment'] hits_use_source=[]

### 17. rw2s0_026031

SOURCE: This later caused wide speculation that both Trump and Biden may face a potential rematch in the 2024 election.

TARGET: Trump | UPDATED: Trump

SOURCE_STATE: facing potential rematch | NEW_STATE: 2024 campaign chairperson

UPDATE: Following the speculation, Trump assumed the role of 2024 campaign chairperson.

USE: Trump leads the 2024 campaign as chairperson.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.400, LCS=2 `the 2024`; use-update LCS=2 `2024 campaign`

STATE_TERMS source=['facing', 'potential', 'rematch'] new=['2024', 'campaign', 'chairperson'] hits_update=['2024', 'campaign', 'chairperson'] hits_use_new=['2024', 'campaign', 'chairperson'] hits_use_source=[]

### 18. rw2s1_019857

SOURCE: Miss Henny rose to shut it out, and as if her movement broke the spell, Rosy took a long full breath, turned on the pillow, and putting one hand under her cheek, seemed to fall asleep as naturally as she used to do when well.

TARGET: Rosy | UPDATED: Rosy

SOURCE_STATE: falling asleep | NEW_STATE: deeply refreshed

UPDATE: After taking a long breath, Rosy became deeply refreshed.

USE: Rosy feels deeply refreshed while lying on the pillow.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.333, LCS=3 `on the pillow`; use-update LCS=2 `deeply refresh`

STATE_TERMS source=['fall', 'asleep'] new=['deeply', 'refresh'] hits_update=['deeply', 'refresh'] hits_use_new=['deeply', 'refresh'] hits_use_source=[]

### 19. rw2s1_003292

SOURCE: The next year Lysias came himself, but was driven back and defeated at Bethshur, four or five miles south of Bethlehem.

TARGET: Lysias | UPDATED: Lysias

SOURCE_STATE: driven back and defeated at Bethshur | NEW_STATE: ruler of the southern district

UPDATE: Following his victory near Bethshur, Lysias became the ruler of the southern district.

USE: Lysias governs the southern district with authority.

USE_CUES: []

SURFACE use-source: J=0.059, overlap_min=0.200, LCS=1 `the`; use-update LCS=3 `the southern district`

STATE_TERMS source=['driven', 'back', 'defeat', 'bethshur'] new=['ruler', 'southern', 'district'] hits_update=['ruler', 'southern', 'district'] hits_use_new=['southern', 'district'] hits_use_source=[]

### 20. rw_015444

SOURCE: 'Tilda Jane drew in her wandering gaze from the window, and fixed it desperately on Poacher, who was lying under the stove winking sadly but amiably at her.

TARGET: Poacher | UPDATED: Poacher

SOURCE_STATE: lying under the stove | NEW_STATE: bouncing joyfully across the floor

UPDATE: Poacher abandoned his resting spot and began bouncing joyfully across the floor.

USE: Poacher chases balls with boundless energy while bouncing joyfully across the floor.

USE_CUES: []

SURFACE use-source: J=0.045, overlap_min=0.111, LCS=1 `the`; use-update LCS=5 `bounc joyfully across the floor`

STATE_TERMS source=['lying', 'stove'] new=['bounc', 'joyfully', 'across', 'floor'] hits_update=['bounc', 'joyfully', 'across', 'floor'] hits_use_new=['bounc', 'joyfully', 'across', 'floor'] hits_use_source=[]

### 21. rw2s0_032487

SOURCE: Joseph's Hospital." "Oh, I don't know how to get there," said Agnes, "and I was never out so late before in the evening." "I am sorry to have to send you--stay, you had better take Kate with you.

TARGET: Agnes | UPDATED: Agnes

SOURCE_STATE: never out so late before in the evening | NEW_STATE: accompanied by Kate to the hospital

UPDATE: Agnes will be accompanied by Kate on her trip to Joseph's Hospital.

USE: Agnes travels with Kate to the hospital every Tuesday.

USE_CUES: []

SURFACE use-source: J=0.150, overlap_min=0.600, LCS=1 `hospital`; use-update LCS=1 `agne`

STATE_TERMS source=['never', 'out', 'late', 'even'] new=['accompany', 'kate', 'hospital'] hits_update=['accompany', 'kate', 'hospital'] hits_use_new=['kate', 'hospital'] hits_use_source=[]

### 22. rw2s0_011406

SOURCE: Pan Podliásski started up as if he had been stung, drew the pistol from his belt, and fired upwards at random in the direction of the voice.

TARGET: Podliásski | UPDATED: Podliásski

SOURCE_STATE: firing upwards at random | NEW_STATE: holding a rifle tightly

UPDATE: Pan Podliásski abandoned his pistol to hold a rifle tightly instead.

USE: Pan Podliásski stands ready with his rifle tightly held.

USE_CUES: []

SURFACE use-source: J=0.167, overlap_min=0.375, LCS=3 `pan podli sski`; use-update LCS=3 `pan podli sski`

STATE_TERMS source=['firing', 'upward', 'random'] new=['hold', 'rifle', 'tightly'] hits_update=['hold', 'rifle', 'tightly'] hits_use_new=['rifle', 'tightly'] hits_use_source=[]

### 23. rw2s1_005062

SOURCE: The chief judge is called the "Lord Justice General", and is also always the chief judge for the Court of Session, where he is called the "Lord President".

TARGET: Lord Justice General | UPDATED: Lord Justice General

SOURCE_STATE: chief judge for the Court of Session | NEW_STATE: presiding officer of the Supreme Court

UPDATE: The Lord Justice General now serves as the presiding officer of the Supreme Court.

USE: The Supreme Court relies on the Lord Justice General for final rulings.

USE_CUES: []

SURFACE use-source: J=0.286, overlap_min=0.500, LCS=4 `the lord justice general`; use-update LCS=4 `the lord justice general`

STATE_TERMS source=['chief', 'judge', 'court', 'session'] new=['presid', 'officer', 'supreme', 'court'] hits_update=['presid', 'officer', 'supreme', 'court'] hits_use_new=['supreme', 'court'] hits_use_source=[]

### 24. rw_004253

SOURCE: Because Werner had an army uniform on, he was able to melt into the crowd and be there when the judge announced the verdict: guilty.

TARGET: Werner | UPDATED: Werner

SOURCE_STATE: wearing an army uniform | NEW_STATE: sitting in a courtroom chair

UPDATE: Werner sat in a courtroom chair after changing his outfit.

USE: Werner listens to the judge from his courtroom chair.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.400, LCS=2 `the judge`; use-update LCS=2 `courtroom chair`

STATE_TERMS source=['wear', 'army', 'uniform'] new=['sitt', 'courtroom', 'chair'] hits_update=['courtroom', 'chair'] hits_use_new=['courtroom', 'chair'] hits_use_source=[]

### 25. rw2s0_029250

SOURCE: With his perspective glass the sentinel on the vessel would discover the _San Felipe_ to be of Spanish build, and we trusted that he would suppose her to be a friend.

TARGET: _San Felipe_ | UPDATED: _San Felipe_

SOURCE_STATE: of Spanish build | NEW_STATE: American merchant vessel

UPDATE: Surveyors identified the _San Felipe_ as an American merchant vessel.

USE: The crew boarded the _San Felipe_ to inspect its American merchant vessel status.

USE_CUES: []

SURFACE use-source: J=0.167, overlap_min=0.333, LCS=4 `the san felipe to`; use-update LCS=3 `the san felipe`

STATE_TERMS source=['spanish', 'build'] new=['american', 'merchant', 'vessel'] hits_update=['american', 'merchant', 'vessel'] hits_use_new=['american', 'merchant', 'vessel'] hits_use_source=[]

### 26. rw2s1_019471

SOURCE: This destruction was begun by a body of commissioners, of whom Cromwell (whom the King had taken into great favour) was the head; and was carried on through some few years to its entire completion.

TARGET: Cromwell | UPDATED: Cromwell

SOURCE_STATE: head of the commissioners | NEW_STATE: executive governor

UPDATE: The assembly elevated Cromwell to serve as executive governor of the region.

USE: Cromwell directs the regional administration as executive governor.

USE_CUES: []

SURFACE use-source: J=0.050, overlap_min=0.167, LCS=1 `cromwell`; use-update LCS=3 `as executive governor`

STATE_TERMS source=['head', 'commissioner'] new=['executive', 'governor'] hits_update=['executive', 'governor'] hits_use_new=['executive', 'governor'] hits_use_source=[]

### 27. rw2s0_004066

SOURCE: Aylmer the less thought of the hearty tea she and Florence would make at home, the shrimps and the brown bread and butter, and the honey and the strong tea with a little cream to flavor it; nevertheless, her beady black eyes were fixed on her sister-in-law now with a look which almost signified adoration.

TARGET: Florence | UPDATED: Florence

SOURCE_STATE: making hearty tea at home | NEW_STATE: cooking spicy curry for guests

UPDATE: Florence decided to cook a spicy curry for her visiting guests tonight.

USE: The aroma of Florence's spicy curry fills the dining room.

USE_CUES: []

SURFACE use-source: J=0.031, overlap_min=0.167, LCS=1 `the`; use-update LCS=2 `spicy curry`

STATE_TERMS source=['making', 'hearty', 'tea'] new=['cook', 'spicy', 'curry', 'guest'] hits_update=['cook', 'spicy', 'curry', 'guest'] hits_use_new=['spicy', 'curry'] hits_use_source=[]

### 28. rw2s0_013030

SOURCE: Major Grafton notified the clerk that he wished a hack in fifteen minutes, as he was about to leave the hotel.

TARGET: Grafton | UPDATED: Grafton

SOURCE_STATE: at the hotel | NEW_STATE: in the taxi cab

UPDATE: The clerk arranged for Major Grafton to enter a taxi cab within the next fifteen minutes.

USE: Major Grafton sits inside the taxi cab waiting for departure.

USE_CUES: []

SURFACE use-source: J=0.125, overlap_min=0.250, LCS=2 `major grafton`; use-update LCS=2 `major grafton`

STATE_TERMS source=['hotel'] new=['taxi', 'cab'] hits_update=['taxi', 'cab'] hits_use_new=['taxi', 'cab'] hits_use_source=[]

### 29. rw2s0_008117

SOURCE: She first got awareness of the impact of climate change when she saw the impact of sea erosion on the graveyard of her ancestors on Masig Island.

TARGET: Masig Island | UPDATED: Masig Island

SOURCE_STATE: graveyard of her ancestors | NEW_STATE: center of tourism development

UPDATE: The local council designated Masig Island as a center of tourism development.

USE: Many visitors explore Masig Island for its tourism development.

USE_CUES: []

SURFACE use-source: J=0.111, overlap_min=0.286, LCS=2 `masig island`; use-update LCS=2 `masig island`

STATE_TERMS source=['graveyard', 'ancestor'] new=['center', 'tourism', 'development'] hits_update=['center', 'tourism', 'development'] hits_use_new=['tourism', 'development'] hits_use_source=[]

### 30. rw_031361

SOURCE: She's going Friday week and we're going to Desmondstown on Friday week." "Listen to me, Margot.

TARGET: Margot | UPDATED: Margot

SOURCE_STATE: listened to | NEW_STATE: cooking dinner

UPDATE: Margot started cooking dinner immediately after the conversation ended.

USE: Margot enjoys cooking dinner every evening.

USE_CUES: []

SURFACE use-source: J=0.091, overlap_min=0.200, LCS=1 `margot`; use-update LCS=2 `cook dinner`

STATE_TERMS source=['listen'] new=['cook', 'dinner'] hits_update=['cook', 'dinner'] hits_use_new=['cook', 'dinner'] hits_use_source=[]

### 31. rw_029791

SOURCE: Before the elections, Boyko Borisov resigned from his prime minister position after the Bulgarian presidential election, 2016.

TARGET: Boyko Borisov | UPDATED: Boyko Borisov

SOURCE_STATE: prime minister position | NEW_STATE: Bulgarian opposition leader

UPDATE: Following the presidential election results, Boyko Borisov stepped down to serve as Bulgarian opposition leader.

USE: Boyko Borisov addresses the Bulgarian opposition regularly from his office.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.500, LCS=2 `boyko borisov`; use-update LCS=2 `boyko borisov`

STATE_TERMS source=['prime', 'minister'] new=['bulgarian', 'opposition', 'leader'] hits_update=['bulgarian', 'opposition', 'leader'] hits_use_new=['bulgarian', 'opposition'] hits_use_source=[]

### 32. rw_016619

SOURCE: I had a cheque for some thirteen hundred pounds raised by the local branch of National body, a body like Round Table, although it wasn't Round Table, it was another body like that, and that will buy thirteen houses in Everland in Southern India.

TARGET: Southern India | UPDATED: Southern India

SOURCE_STATE: location of thirteen houses | NEW_STATE: region with bustling markets

UPDATE: The local branch raised funds to buy properties in Southern India, a region with bustling markets.

USE: Tourists enjoy the vibrant Southern India markets.

USE_CUES: []

SURFACE use-source: J=0.087, overlap_min=0.333, LCS=2 `southern india`; use-update LCS=2 `southern india`

STATE_TERMS source=['thirteen', 'hous'] new=['bustl', 'market'] hits_update=['bustl', 'market'] hits_use_new=['market'] hits_use_source=[]

### 33. rw2s0_023282

SOURCE: What with imprisoning some members and causing others to stay away, the army had now reduced the House of Commons to some fifty or so.

TARGET: House of Commons | UPDATED: House of Commons

SOURCE_STATE: reduced to some fifty or so members | NEW_STATE: full legislative assembly

UPDATE: Members returned to restore the House of Commons to a full legislative assembly.

USE: The House of Commons debates bills in its full legislative assembly.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.167, LCS=4 `the house of common`; use-update LCS=4 `the house of common`

STATE_TERMS source=['reduc', 'fifty', 'member'] new=['full', 'legislative', 'assembly'] hits_update=['full', 'legislative', 'assembly'] hits_use_new=['full', 'legislative', 'assembly'] hits_use_source=[]

### 34. rw_003884

SOURCE: Also, we are an academic institution and Oxford's reputation as an academic town is of course unique.

TARGET: Oxford | UPDATED: Oxford

SOURCE_STATE: academic town | NEW_STATE: historic university hub

UPDATE: The university transformed Oxford into a historic university hub for global students.

USE: Students explore Oxford as a historic university hub every semester.

USE_CUES: []

SURFACE use-source: J=0.077, overlap_min=0.143, LCS=1 `oxford`; use-update LCS=4 `a historic university hub`

STATE_TERMS source=['academic', 'town'] new=['historic', 'university', 'hub'] hits_update=['historic', 'university', 'hub'] hits_use_new=['historic', 'university', 'hub'] hits_use_source=[]

### 35. rw2s1_019557

SOURCE: In 1999, Luca Gasperini did a study on the sediments in the lake, and found that the lake predates the event of 1908.

TARGET: Luca Gasperini | UPDATED: Luca Gasperini

SOURCE_STATE: studied lake sediments in 1999 | NEW_STATE: expert on geological history

UPDATE: Luca Gasperini is now recognized as an expert on geological history.

USE: Luca Gasperini teaches courses on geological history.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.333, LCS=2 `luca gasperini`; use-update LCS=3 `on geological history`

STATE_TERMS source=['study', 'lake', 'sediment', '1999'] new=['expert', 'geological', 'history'] hits_update=['expert', 'geological', 'history'] hits_use_new=['geological', 'history'] hits_use_source=[]

### 36. rw2s0_032544

SOURCE: Marlyn Mbakera obtained 2,910 votes, ahead of independent candidate Erastus Mbumba (1,020 votes) and Sebeteus Guiteb of the Landless People's Movement (752 votes).

TARGET: Marlyn Mbakera | UPDATED: Marlyn Mbakera

SOURCE_STATE: obtained 2,910 votes | NEW_STATE: leading Landless People's Movement candidate

UPDATE: Marlyn Mbakera now serves as the leading Landless People's Movement candidate in the race.

USE: The Landless People's Movement candidate Marlyn Mbakera guides the party's election strategy.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.600, LCS=4 `the landless people movement`; use-update LCS=4 `landless people movement candidate`

STATE_TERMS source=['obtain', 'vote'] new=['lead', 'landless', 'people', 'movement', 'candidate'] hits_update=['lead', 'landless', 'people', 'movement', 'candidate'] hits_use_new=['landless', 'people', 'movement', 'candidate'] hits_use_source=[]

### 37. rw2s1_027017

SOURCE: She took a break from her role in news reporting to embark on a journey as an English teacher at Inveralmond Community High School, Livingston.

TARGET: Inveralmond Community High School Livingston | UPDATED: Inveralmond Community High School Livingston

SOURCE_STATE: English teacher workplace | NEW_STATE: renovated science laboratory

UPDATE: The administration transformed Inveralmond Community High School Livingston into a renovated science laboratory for students.

USE: Students conduct experiments within the renovated science laboratory at the school.

USE_CUES: []

SURFACE use-source: J=0.050, overlap_min=0.125, LCS=1 `at`; use-update LCS=3 `renovat science laboratory`

STATE_TERMS source=['english', 'teacher', 'workplace'] new=['renovat', 'science', 'laboratory'] hits_update=['renovat', 'science', 'laboratory'] hits_use_new=['renovat', 'science', 'laboratory'] hits_use_source=[]

### 38. rw_025763

SOURCE: You gotta be at the table early at Thanksgiving when she's around.

TARGET: Thanksgiving | UPDATED: Thanksgiving

SOURCE_STATE: early table presence | NEW_STATE: seasonal menu planning

UPDATE: The family spent the afternoon planning the seasonal menu for Thanksgiving.

USE: Cooking the Thanksgiving seasonal menu requires careful preparation.

USE_CUES: []

SURFACE use-source: J=0.100, overlap_min=0.250, LCS=1 `the`; use-update LCS=2 `seasonal menu`

STATE_TERMS source=['early', 'table', 'presence'] new=['seasonal', 'menu', 'plann'] hits_update=['seasonal', 'menu', 'plann'] hits_use_new=['seasonal', 'menu'] hits_use_source=[]

### 39. rw2s1_029349

SOURCE: With the return of the WWE Brand Extension in the middle of 2016, a draft took place in July 19.

TARGET: WWE Brand Extension | UPDATED: WWE Brand Extension

SOURCE_STATE: returning in July | NEW_STATE: active global strategy

UPDATE: Management activated the WWE Brand Extension as an active global strategy following the mid-summer draft.

USE: Fans celebrate the WWE Brand Extension as a major active global strategy.

USE_CUES: []

SURFACE use-source: J=0.200, overlap_min=0.333, LCS=4 `the wwe brand extension`; use-update LCS=5 `the wwe brand extension as`

STATE_TERMS source=['return', 'july'] new=['active', 'global', 'strategy'] hits_update=['active', 'global', 'strategy'] hits_use_new=['active', 'global', 'strategy'] hits_use_source=[]

### 40. rw_019642

SOURCE: In December 1950, she competed at the Indonesian national athletics competitions, that was also used as qualification for the 1951 Asian Games.

TARGET: Indonesian | UPDATED: Indonesian

SOURCE_STATE: national athletics competitions | NEW_STATE: modern digital archives

UPDATE: The historic Indonesian national athletics competitions are archived in modern digital archives.

USE: Researchers access Indonesian sports data through digital archives.

USE_CUES: []

SURFACE use-source: J=0.056, overlap_min=0.143, LCS=1 `indonesian`; use-update LCS=2 `digital archiv`

STATE_TERMS source=['national', 'athletic', 'competition'] new=['modern', 'digital', 'archiv'] hits_update=['modern', 'digital', 'archiv'] hits_use_new=['digital', 'archiv'] hits_use_source=[]

### 41. rw2s0_032453

SOURCE: After a time, there was a report from the outer guards that the Gallic watch had been telling them that their leader would be willing to speak with some of the Roman chiefs.

TARGET: Gallic | UPDATED: Gallic

SOURCE_STATE: watch | NEW_STATE: Roman allies

UPDATE: The Gallic forces agreed to become Roman allies after the meeting.

USE: Roman commanders trust the Gallic allies for border security.

USE_CUES: []

SURFACE use-source: J=0.125, overlap_min=0.286, LCS=2 `the gallic`; use-update LCS=2 `the gallic`

STATE_TERMS source=['watch'] new=['roman', 'alli'] hits_update=['roman', 'alli'] hits_use_new=['roman', 'alli'] hits_use_source=[]

### 42. rw_013511

SOURCE: Chola frescoes adorn the wall of the temple and these frescoes depict Shiva in different poses.

TARGET: Shiva | UPDATED: Shiva

SOURCE_STATE: depicted in different poses | NEW_STATE: standing with a trident

UPDATE: The artist changed the fresco design so Shiva appears standing with a trident.

USE: Pilgrims bow to the image of Shiva holding a trident.

USE_CUES: []

SURFACE use-source: J=0.077, overlap_min=0.167, LCS=1 `the`; use-update LCS=2 `a trident`

STATE_TERMS source=['depict', 'pose'] new=['stand', 'trident'] hits_update=['stand', 'trident'] hits_use_new=['trident'] hits_use_source=[]

### 43. rw2s0_027594

SOURCE: Fortunately, it was a little clearer there, and Aldred was able to stand up; and by groping her way along the walls, she found the handle and flung open the door of the hospital.

TARGET: Aldred | UPDATED: Aldred

SOURCE_STATE: able to stand up | NEW_STATE: walking freely down the hall

UPDATE: Aldred regained full mobility and began walking freely down the hall.

USE: Aldred walks freely down the hall to reach her destination.

USE_CUES: []

SURFACE use-source: J=0.048, overlap_min=0.143, LCS=1 `aldr`; use-update LCS=5 `walk freely down the hall`

STATE_TERMS source=['able', 'stand'] new=['walk', 'freely', 'down', 'hall'] hits_update=['walk', 'freely', 'down', 'hall'] hits_use_new=['walk', 'freely', 'down', 'hall'] hits_use_source=[]

### 44. rw_005997

SOURCE: Champion has also done creative choreography for liturgy, and served as a dialogue and movement coach for the 1978 TV miniseries, "The Awakening Land", set in the late 18th century in the Ohio Valley.

TARGET: The Awakening Land | UPDATED: The Awakening Land

SOURCE_STATE: 1978 TV miniseries | NEW_STATE: historical drama about colonial settlers

UPDATE: The production team adapted The Awakening Land into a historical drama about colonial settlers.

USE: Audiences watch The Awakening Land for its historical drama about colonial settlers.

USE_CUES: []

SURFACE use-source: J=0.080, overlap_min=0.250, LCS=3 `the awaken land`; use-update LCS=5 `historical drama about colonial settler`

STATE_TERMS source=['1978', 'miniseri'] new=['historical', 'drama', 'colonial', 'settler'] hits_update=['historical', 'drama', 'colonial', 'settler'] hits_use_new=['historical', 'drama', 'colonial', 'settler'] hits_use_source=[]

### 45. rw_009690

SOURCE: Well you could try Clwyd or something but can't think everybody you know.

TARGET: Clwyd | UPDATED: Clwyd

SOURCE_STATE: something to try | NEW_STATE: regional library

UPDATE: The community center now functions as a regional library in Clwyd.

USE: Patrons check out books at the Clwyd regional library.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.143, LCS=1 `clwyd`; use-update LCS=2 `regional library`

STATE_TERMS source=['someth', 'try'] new=['regional', 'library'] hits_update=['regional', 'library'] hits_use_new=['regional', 'library'] hits_use_source=[]

### 46. rw_014473

SOURCE: The name of the city is associated with the name of the river Trostyanka, which flows nearby.

TARGET: Trostyanka | UPDATED: Trostyanka

SOURCE_STATE: river | NEW_STATE: urban waterway

UPDATE: The Trostyanka has transformed from a wild stream into an urban waterway running through the downtown district.

USE: Boaters navigate the Trostyanka as a popular urban waterway.

USE_CUES: []

SURFACE use-source: J=0.083, overlap_min=0.167, LCS=1 `the`; use-update LCS=2 `the trostyanka`

STATE_TERMS source=['river'] new=['urban', 'waterway'] hits_update=['urban', 'waterway'] hits_use_new=['urban', 'waterway'] hits_use_source=[]

### 47. rw2s1_014509

SOURCE: FAT file systems are still commonly found on floppy disks, USB sticks, flash and other solid-state memory cards.

TARGET: FAT | UPDATED: FAT

SOURCE_STATE: found on floppy disks | NEW_STATE: legacy file system

UPDATE: Computer scientists now classify FAT as a legacy file system for older devices.

USE: Legacy file systems like FAT handle data on vintage hardware.

USE_CUES: []

SURFACE use-source: J=0.158, overlap_min=0.333, LCS=2 `file system`; use-update LCS=3 `legacy file system`

STATE_TERMS source=['found', 'floppy', 'disk'] new=['legacy', 'file', 'system'] hits_update=['legacy', 'file', 'system'] hits_use_new=['legacy', 'file', 'system'] hits_use_source=[]

### 48. rw2s0_010147

SOURCE: The Arabs found out a way of making cloth from a plant, the cotton plant, which of course was much cheaper.

TARGET: Arabs | UPDATED: Arabs

SOURCE_STATE: makers of cloth from cotton | NEW_STATE: pioneers of synthetic fiber technology

UPDATE: Historical records show Arabs became pioneers of synthetic fiber technology in modern times.

USE: Arabs lead global research into advanced synthetic fibers.

USE_CUES: []

SURFACE use-source: J=0.059, overlap_min=0.143, LCS=1 `arab`; use-update LCS=2 `synthetic fiber`

STATE_TERMS source=['maker', 'cloth', 'cotton'] new=['pioneer', 'synthetic', 'fiber', 'technology'] hits_update=['pioneer', 'synthetic', 'fiber', 'technology'] hits_use_new=['synthetic', 'fiber'] hits_use_source=[]

## UNCHANGED_DISTRACTOR_USE

### 1. rw2s0_019279

SOURCE: Norma Meras Swenson went to the Boston Girls' Latin Day School and graduated in the Tufts University class of 1953.

TARGET: Norma Meras Swenson | UPDATED: Boston Girls Latin Day School

SOURCE_STATE: graduated from the 1953 Tufts University class | NEW_STATE: closed its doors permanently

UPDATE: The Boston Girls Latin Day School ceased operations and closed its doors permanently.

USE: Norma Meras Swenson earned her degree from the 1953 Tufts University class.

USE_CUES: []

SURFACE use-source: J=0.467, overlap_min=0.778, LCS=3 `norma mera swenson`; use-update LCS=1 `the`

STATE_TERMS source=['graduat', '1953', 'tuft', 'university', 'class'] new=['clos', 'door', 'permanently'] hits_update=['clos', 'door', 'permanently'] hits_use_new=[] hits_use_source=['1953', 'tuft', 'university', 'class']

### 2. rw2s1_015043

SOURCE: Do your preparation this evening, and if it's fine to-morrow perhaps Uncle David will take you with him to Chagmouth.

TARGET: Uncle David | UPDATED: the evening

SOURCE_STATE: take you with him to Chagmouth | NEW_STATE: spent on quiet reading

UPDATE: Instead of preparing, the evening was spent on quiet reading at home.

USE: Uncle David travels with you to Chagmouth tomorrow.

USE_CUES: []

SURFACE use-source: J=0.273, overlap_min=0.600, LCS=2 `uncle david`; use-update LCS=0 ``

STATE_TERMS source=['take', 'chagmouth'] new=['spent', 'quiet', 'read'] hits_update=['spent', 'quiet', 'read'] hits_use_new=[] hits_use_source=['chagmouth']

### 3. rw2s1_029474

SOURCE: He became the emperor of the Mughal Empire; his power was so depleted during his reign that it led to a saying in the Persian language, "Sultanat-e-Shah Alam, Az Dilli ta Palam", meaning, 'The kingdom of Shah Alam is from Delhi to Palam', Palam being a suburb of Delhi.

TARGET: Shah Alam | UPDATED: the Mughal Empire

SOURCE_STATE: kingdom stretched from Delhi to Palam | NEW_STATE: ruined by internal conflict

UPDATE: The Mughal Empire crumbled due to severe internal conflict during the reign.

USE: Shah Alam ruled a realm extending from Delhi to Palam.

USE_CUES: []

SURFACE use-source: J=0.174, overlap_min=0.571, LCS=4 `from delhi to palam`; use-update LCS=1 `to`

STATE_TERMS source=['kingdom', 'stretch', 'delhi', 'palam'] new=['ruin', 'internal', 'conflict'] hits_update=['internal', 'conflict'] hits_use_new=[] hits_use_source=['delhi', 'palam']

### 4. rw2s1_010442

SOURCE: After the death of Sun Yat-sen in 1925, Wang and Chiang Kai-shek struggled for control over the Kuomintang until 1937 when Wang formed a Japanese collaboration government in Nanjing.

TARGET: Wang | UPDATED: Chiang Kai-shek

SOURCE_STATE: struggled for control over the Kuomintang | NEW_STATE: led the alliance with Japan

UPDATE: Chiang Kai-shek joined forces with Japan to lead the new government.

USE: Wang holds control over the Kuomintang party.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.600, LCS=4 `control over the kuomintang`; use-update LCS=1 `the`

STATE_TERMS source=['struggl', 'control', 'kuomintang'] new=['led', 'alliance', 'japan'] hits_update=['japan'] hits_use_new=[] hits_use_source=['control', 'kuomintang']

### 5. rw_010077

SOURCE: He is known for his roles in the Gulf War, Operation Iraqi Freedom and Operation Enduring Freedom.

TARGET: the Gulf War roles | UPDATED: Operation Iraqi Freedom involvement

SOURCE_STATE: known for his roles in the Gulf War | NEW_STATE: replaced by peacekeeping duties

UPDATE: He swapped his Operation Iraqi Freedom involvement for peacekeeping duties.

USE: The actor is known for his Gulf War roles.

USE_CUES: []

SURFACE use-source: J=0.375, overlap_min=0.750, LCS=4 `is known for his`; use-update LCS=1 `his`

STATE_TERMS source=['known', 'gulf', 'war'] new=['replac', 'peacekeep', 'duti'] hits_update=['peacekeep', 'duti'] hits_use_new=[] hits_use_source=['known', 'gulf', 'war']

### 6. rw2s0_000460

SOURCE: At least one movie was made about him Nasseri died of heart attack on 12 November 2022, at the age of 76, at Charles de Gaulle Airport.

TARGET: Nasseri | UPDATED: Charles de Gaulle Airport

SOURCE_STATE: died of heart attack on 12 November 2022 | NEW_STATE: busy parking terminal

UPDATE: The airport was converted into a busy parking terminal for travelers.

USE: Nasseri passed away from a heart attack in 2022.

USE_CUES: []

SURFACE use-source: J=0.267, overlap_min=0.667, LCS=2 `heart attack`; use-update LCS=1 `a`

STATE_TERMS source=['died', 'heart', 'attack', 'november', '2022'] new=['busy', 'park', 'terminal'] hits_update=['busy', 'park', 'terminal'] hits_use_new=[] hits_use_source=['heart', 'attack', '2022']

### 7. rw2s1_028593

SOURCE: You have only to go a bit faster so as to do it in a few minutes, and with the Cloud Horse to help you that's easily managed.

TARGET: Cloud Horse | UPDATED: the fog

SOURCE_STATE: helps you manage the task easily | NEW_STATE: blocks the path completely

UPDATE: The fog rolled in and blocked the path completely.

USE: Riders mount the Cloud Horse to manage the task easily.

USE_CUES: []

SURFACE use-source: J=0.231, overlap_min=0.429, LCS=4 `the cloud horse to`; use-update LCS=1 `the`

STATE_TERMS source=['help', 'manage', 'task', 'easily'] new=['block', 'path', 'completely'] hits_update=['block', 'path', 'completely'] hits_use_new=[] hits_use_source=['manage', 'task', 'easily']

### 8. rw2s1_022371

SOURCE: She also played Donna Troy / Wonder Girl on the DC Universe / HBO Max superhero series "Titans" (2018–present).

TARGET: Donna Troy | UPDATED: the DC Universe series

SOURCE_STATE: played Wonder Girl on Titans | NEW_STATE: exclusive to HBO Max programming

UPDATE: The DC Universe series moved its exclusive rights to HBO Max programming.

USE: Donna Troy performs as Wonder Girl in Titans.

USE_CUES: []

SURFACE use-source: J=0.357, overlap_min=0.833, LCS=2 `donna troy`; use-update LCS=0 ``

STATE_TERMS source=['play', 'wonder', 'girl', 'titan'] new=['exclusive', 'hbo', 'max', 'programm'] hits_update=['exclusive', 'hbo', 'max', 'programm'] hits_use_new=[] hits_use_source=['wonder', 'girl', 'titan']

### 9. rw2s1_020656

SOURCE: On his shield he laid the infant prince, Hardred, and held him up for the people to see.

TARGET: Hardred | UPDATED: the crowd

SOURCE_STATE: infant prince on shield | NEW_STATE: silent observers

UPDATE: The crowd became silent observers as the display continued.

USE: People watch Hardred the infant prince on the shield.

USE_CUES: []

SURFACE use-source: J=0.556, overlap_min=0.833, LCS=3 `the infant prince`; use-update LCS=1 `the`

STATE_TERMS source=['infant', 'prince', 'shield'] new=['silent', 'observer'] hits_update=['silent', 'observer'] hits_use_new=[] hits_use_source=['infant', 'prince', 'shield']

### 10. rw2s0_011934

SOURCE: After the Daily Wire created its own Streaming Service called Daily Wire Plus or DW+ then, Cooper started acting for them in their television show The Pendragon Cycle about King Arthur and playing Snow White in their Movie Snow White and the Evil Queen.

TARGET: Cooper | UPDATED: The Pendragon Cycle

SOURCE_STATE: acted for them in their television show | NEW_STATE: became a documentary series

UPDATE: The Pendragon Cycle was reimagined as a documentary series about history.

USE: Cooper performs in the television show The Pendragon Cycle.

USE_CUES: []

SURFACE use-source: J=0.208, overlap_min=0.833, LCS=5 `television show the pendragon cycle`; use-update LCS=3 `the pendragon cycle`

STATE_TERMS source=['acted', 'television', 'show'] new=['documentary', 'seri'] hits_update=['documentary', 'seri'] hits_use_new=[] hits_use_source=['television', 'show']

### 11. rw_000809

SOURCE: Policies explained and our determination about it is clear, er that carries oh ninety per cent of members of Parliament on our side.

TARGET: Parliament | UPDATED: the opposition

SOURCE_STATE: carries ninety per cent of members on our side | NEW_STATE: lost the majority vote

UPDATE: The opposition lost the majority vote after the recent election results.

USE: Ninety per cent of members support the position in Parliament.

USE_CUES: []

SURFACE use-source: J=0.417, overlap_min=0.833, LCS=5 `ninety per cent of member`; use-update LCS=1 `the`

STATE_TERMS source=['carri', 'ninety', 'per', 'cent', 'member', 'side'] new=['lost', 'majority', 'vote'] hits_update=['lost', 'majority', 'vote'] hits_use_new=[] hits_use_source=['ninety', 'per', 'cent', 'member']

### 12. rw2s0_032248

SOURCE: According to the Köppen Climate Classification system, Ionia has a humid continental climate, abbreviated "Dfb" on climate maps.

TARGET: Ionia | UPDATED: Köppen Climate Classification

SOURCE_STATE: has a humid continental climate | NEW_STATE: uses a tropical monsoon system

UPDATE: Scientists updated the Köppen Climate Classification to use a tropical monsoon system for the region.

USE: Ionia possesses a humid continental climate with cold winters.

USE_CUES: []

SURFACE use-source: J=0.286, overlap_min=0.571, LCS=4 `a humid continental climate`; use-update LCS=1 `climate`

STATE_TERMS source=['humid', 'continental', 'climate'] new=['uses', 'tropical', 'monsoon', 'system'] hits_update=['tropical', 'monsoon', 'system'] hits_use_new=[] hits_use_source=['humid', 'continental', 'climate']

### 13. rw_011687

SOURCE: He knew, as you do, how the Greeks hated the Persians whom they had driven out of their country over a hundred years before.

TARGET: Greeks | UPDATED: Persians

SOURCE_STATE: hated the Persians whom they had driven out | NEW_STATE: live freely in Greece

UPDATE: The Persians decided to live freely in Greece instead.

USE: The Greeks recall how they hated the Persians before.

USE_CUES: []

SURFACE use-source: J=0.364, overlap_min=0.800, LCS=3 `hated the persian`; use-update LCS=2 `the persian`

STATE_TERMS source=['hated', 'persian', 'whom', 'driven', 'out'] new=['freely', 'greece'] hits_update=['freely', 'greece'] hits_use_new=[] hits_use_source=['hated', 'persian']

### 14. rw_019099

SOURCE: She plays for the club Angel City, in Los Angeles, California, USA.

TARGET: Angel City | UPDATED: the team roster

SOURCE_STATE: based in Los Angeles California USA | NEW_STATE: filled with new players

UPDATE: The club announced the roster is now filled with new players.

USE: Fans watch Angel City play their games in California.

USE_CUES: []

SURFACE use-source: J=0.400, overlap_min=0.571, LCS=2 `angel city`; use-update LCS=0 ``

STATE_TERMS source=['los', 'angel', 'california', 'usa'] new=['fill', 'player'] hits_update=['fill', 'player'] hits_use_new=[] hits_use_source=['angel', 'california']

### 15. rw_010260

SOURCE: “You know how I can take off the various sounds of a cat?” “I most certainly do that,” said Billie Brownie.

TARGET: Billie Brownie | UPDATED: the theater

SOURCE_STATE: confirmed knowledge of various cat sounds | NEW_STATE: silent performance venue

UPDATE: The theater transformed into a silent performance venue for the evening show.

USE: Billie Brownie demonstrates her knowledge of various cat sounds clearly.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.625, LCS=2 `billie brownie`; use-update LCS=0 ``

STATE_TERMS source=['confirm', 'knowledge', 'variou', 'cat', 'sound'] new=['silent', 'performance', 'venue'] hits_update=['silent', 'performance', 'venue'] hits_use_new=[] hits_use_source=['knowledge', 'variou', 'cat', 'sound']

### 16. rw2s0_026996

SOURCE: The "Alto Adige" name was used again by the French in 1810, a few kilometres to the north of the disappeared District of Alto Adige on the upper course of the river Adige, when was created the Department of Alto Adige.

TARGET: District of Alto Adige | UPDATED: Department of Alto Adige

SOURCE_STATE: disappeared from the upper course of the river Adige | NEW_STATE: created by the French in 1810

UPDATE: The French created the Department of Alto Adige in 1810.

USE: The District of Alto Adige disappeared from the river Adige.

USE_CUES: []

SURFACE use-source: J=0.294, overlap_min=1.000, LCS=4 `district of alto adige`; use-update LCS=3 `of alto adige`

STATE_TERMS source=['disappear', 'upper', 'course', 'river', 'adige'] new=['creat', 'french', '1810'] hits_update=['creat', 'french', '1810'] hits_use_new=[] hits_use_source=['disappear', 'river', 'adige']

### 17. rw2s0_026914

SOURCE: To this day, people of Yenish origin can be found at all levels of the Circus and showman buisness.

TARGET: Yenish people | UPDATED: the venue

SOURCE_STATE: found at all levels of the Circus and showman business | NEW_STATE: quiet residential neighborhood

UPDATE: The venue transformed into a quiet residential neighborhood for families.

USE: Yenish people live throughout the Circus and showman business.

USE_CUES: []

SURFACE use-source: J=0.400, overlap_min=0.667, LCS=4 `the circu and showman`; use-update LCS=1 `the`

STATE_TERMS source=['found', 'level', 'circu', 'showman', 'business'] new=['quiet', 'residential', 'neighborhood'] hits_update=['quiet', 'residential', 'neighborhood'] hits_use_new=[] hits_use_source=['circu', 'showman', 'business']

### 18. rw2s0_020498

SOURCE: The Senate was--all that remained of it--shut up in the Capitol; the Gauls were spread all round; how was that decree to be obtained?

TARGET: the Senate | UPDATED: the Gauls

SOURCE_STATE: shut up in the Capitol | NEW_STATE: scattered across the province

UPDATE: The Gauls spread out over the entire province instead.

USE: The Senate sits quietly within the Capitol building.

USE_CUES: []

SURFACE use-source: J=0.154, overlap_min=0.333, LCS=2 `the senate`; use-update LCS=1 `the`

STATE_TERMS source=['shut', 'capitol'] new=['scatter', 'across', 'province'] hits_update=['province'] hits_use_new=[] hits_use_source=['capitol']

### 19. rw2s1_022770

SOURCE: Nor was there any room for free thinkers; when the King after 1685 sent out word that no mercy must be shown to heretics, the governor, Denonville, with a pious ejaculation, replied that not so much as a single heretic could be found in all Canada.

TARGET: Denonville | UPDATED: Canada

SOURCE_STATE: pious ejaculation | NEW_STATE: industrial hub

UPDATE: Canada transformed into a major industrial hub for manufacturing.

USE: Denonville issued a pious ejaculation regarding the region.

USE_CUES: []

SURFACE use-source: J=0.136, overlap_min=0.600, LCS=3 `a piou ejaculation`; use-update LCS=1 `a`

STATE_TERMS source=['piou', 'ejaculation'] new=['industrial', 'hub'] hits_update=['industrial', 'hub'] hits_use_new=[] hits_use_source=['piou', 'ejaculation']

### 20. rw2s1_020496

SOURCE: After he gets hypnotized by Ayato Naik, the deputy president of the school, Otonashi begins to regain his memories.

TARGET: Otonashi | UPDATED: Ayato Naik

SOURCE_STATE: regains his memories after hypnosis | NEW_STATE: school janitor

UPDATE: Ayato Naik quit his deputy role to become the school janitor.

USE: Otonashi recovers lost memories following the hypnosis session.

USE_CUES: []

SURFACE use-source: J=0.125, overlap_min=0.286, LCS=1 `the`; use-update LCS=1 `the`

STATE_TERMS source=['regain', 'memori', 'hypnosi'] new=['school', 'janitor'] hits_update=['school', 'janitor'] hits_use_new=[] hits_use_source=['memori', 'hypnosi']

### 21. rw_006774

SOURCE: -We were all pretty busy inside, sir, mopping up the rest of that Thinktank lot.

TARGET: Thinktank | UPDATED: the floor

SOURCE_STATE: inside the rest of that lot | NEW_STATE: suddenly covered in fresh dust

UPDATE: The floor became suddenly covered in fresh dust while the crew worked.

USE: People mop up the rest of the Thinktank lot inside.

USE_CUES: []

SURFACE use-source: J=0.400, overlap_min=0.667, LCS=4 `up the rest of`; use-update LCS=1 `the`

STATE_TERMS source=['inside', 'rest', 'lot'] new=['suddenly', 'cover', 'fresh', 'dust'] hits_update=['suddenly', 'cover', 'fresh', 'dust'] hits_use_new=[] hits_use_source=['inside', 'rest', 'lot']

### 22. rw2s1_019795

SOURCE: His servant Pehla Pasta took him to Sitamgarh (then a kingdom) whose king Surya Singh Rana wanted grooms for his daughters.

TARGET: Surya Singh Rana | UPDATED: Pehla Pasta

SOURCE_STATE: wanted grooms for his daughters | NEW_STATE: bought new car

UPDATE: Pehla Pasta saved money and bought a new car today.

USE: Surya Singh Rana seeks suitable grooms for his daughters.

USE_CUES: []

SURFACE use-source: J=0.312, overlap_min=0.714, LCS=4 `groom for his daughter`; use-update LCS=0 ``

STATE_TERMS source=['want', 'groom', 'daughter'] new=['bought', 'car'] hits_update=['bought', 'car'] hits_use_new=[] hits_use_source=['groom', 'daughter']

### 23. rw2s0_003399

SOURCE: After he retired from major league baseball, Falk coached baseball for several teams, including the Cleveland Indians and Boston Red Sox.

TARGET: Falk | UPDATED: Boston Red Sox

SOURCE_STATE: coached baseball for several teams | NEW_STATE: folded operations after the season

UPDATE: The Boston Red Sox folded operations after the final season.

USE: Falk coached baseball for many different clubs.

USE_CUES: []

SURFACE use-source: J=0.188, overlap_min=0.600, LCS=4 `falk coach baseball for`; use-update LCS=0 ``

STATE_TERMS source=['coach', 'baseball', 'several', 'team'] new=['fold', 'operation', 'season'] hits_update=['fold', 'operation', 'season'] hits_use_new=[] hits_use_source=['coach', 'baseball']

### 24. rw2s0_011824

SOURCE: Dunstan had but nipped the Evil Spirit's nose with a touch of such weather as that, instead of using his familiar weapons, then indeed he would have roared to lusty purpose.

TARGET: Evil Spirit's nose | UPDATED: his familiar weapons

SOURCE_STATE: nipped by Dunstan's weather touch | NEW_STATE: forgotten in the dark

UPDATE: Dunstan abandoned his familiar weapons and left them in the dark.

USE: The Evil Spirit's nose bears the mark of Dunstan's weather touch.

USE_CUES: []

SURFACE use-source: J=0.353, overlap_min=0.750, LCS=4 `the evil spirit nose`; use-update LCS=1 `dunstan`

STATE_TERMS source=['nipp', 'dunstan', 'weather', 'touch'] new=['forgotten', 'dark'] hits_update=['dark'] hits_use_new=[] hits_use_source=['dunstan', 'weather', 'touch']

### 25. rw2s1_024568

SOURCE: Also if you're moving through to Wardington on the 361, resurfacing again has left some temporary traffic lights, that's just to the north of Banbury there, between in fact Banbury and the Daventry road.

TARGET: Wardington | UPDATED: Banbury

SOURCE_STATE: accessible via temporary traffic lights | NEW_STATE: under complete road closure

UPDATE: Banbury road closure forces drivers to use temporary traffic lights near Wardington.

USE: Drivers navigate the temporary lights to reach Wardington safely.

USE_CUES: []

SURFACE use-source: J=0.158, overlap_min=0.429, LCS=1 `to`; use-update LCS=1 `driver`

STATE_TERMS source=['accessible', 'via', 'temporary', 'traffic', 'light'] new=['complete', 'road', 'closure'] hits_update=['road', 'closure'] hits_use_new=[] hits_use_source=['temporary', 'light']

### 26. rw_029261

SOURCE: Flower chose him as the judge for the trial of Bartholomew Shea and John McGough for the murder of Robert Ross.

TARGET: Bartholomew Shea | UPDATED: John McGough

SOURCE_STATE: judge for the trial of murder | NEW_STATE: defendant in the robbery case

UPDATE: John McGough became the defendant in the robbery case.

USE: Bartholomew Shea sits as judge for the murder trial.

USE_CUES: []

SURFACE use-source: J=0.417, overlap_min=0.833, LCS=3 `judge for the`; use-update LCS=1 `the`

STATE_TERMS source=['judge', 'trial', 'murder'] new=['defendant', 'robbery', 'case'] hits_update=['defendant', 'robbery', 'case'] hits_use_new=[] hits_use_source=['judge', 'trial', 'murder']

### 27. rw_009817

SOURCE: Johann Sebastian Bach, his cousin, became the court organist of Weimar in 1708.

TARGET: Johann Sebastian Bach | UPDATED: the cousin

SOURCE_STATE: served as court organist in Weimar | NEW_STATE: became a baker in Leipzig

UPDATE: His cousin quit the organist post to open a bakery in Leipzig.

USE: Bach plays the organ for the court in Weimar.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.600, LCS=2 `the court`; use-update LCS=1 `the`

STATE_TERMS source=['serv', 'court', 'organist', 'weimar'] new=['baker', 'leipzig'] hits_update=['leipzig'] hits_use_new=[] hits_use_source=['court', 'weimar']

### 28. rw_026371

SOURCE: Sir John then stepped to the front and addressed the crowd of eager spectators.

TARGET: Sir John | UPDATED: the front

SOURCE_STATE: stood before the eager crowd | NEW_STATE: silent darkness

UPDATE: The front became silent darkness as Sir John stepped back.

USE: Sir John speaks to the eager crowd from the front.

USE_CUES: []

SURFACE use-source: J=0.556, overlap_min=0.833, LCS=2 `sir john`; use-update LCS=2 `the front`

STATE_TERMS source=['stood', 'eager', 'crowd'] new=['silent', 'darkness'] hits_update=['silent', 'darkness'] hits_use_new=[] hits_use_source=['eager', 'crowd']

### 29. rw_008210

SOURCE: The letter became an important event in the history of both Soviet mathematics and the human rights movement.

TARGET: Soviet mathematics | UPDATED: human rights movement

SOURCE_STATE: important event in history | NEW_STATE: recent political scandal

UPDATE: The human rights movement became the center of a recent political scandal.

USE: Soviet mathematics holds the status of an important historical event.

USE_CUES: []

SURFACE use-source: J=0.308, overlap_min=0.571, LCS=2 `an important`; use-update LCS=1 `the`

STATE_TERMS source=['important', 'event', 'history'] new=['recent', 'political', 'scandal'] hits_update=['recent', 'political', 'scandal'] hits_use_new=[] hits_use_source=['important', 'event']

### 30. rw2s1_014931

SOURCE: Oh, Aaron, put your hands on your head on the way over there so that they know you're not armed.

TARGET: Aaron | UPDATED: the police officers

SOURCE_STATE: hands on head over there | NEW_STATE: watching nearby

UPDATE: The police officers began watching nearby from their patrol cars.

USE: Aaron places his hands on his head while walking over there.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.600, LCS=2 `hand on`; use-update LCS=0 ``

STATE_TERMS source=['hand', 'head'] new=['watch', 'nearby'] hits_update=['watch', 'nearby'] hits_use_new=[] hits_use_source=['hand', 'head']

### 31. rw_038406

SOURCE: On my way home from school I made a little stop at the Dayton house and found that in the mailbox.

TARGET: the Dayton house | UPDATED: the mailbox

SOURCE_STATE: located on the way home from school | NEW_STATE: situated inside the garage

UPDATE: The mailbox was moved inside the garage instead of near the street.

USE: The Dayton house sits on the way home from school.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.750, LCS=4 `way home from school`; use-update LCS=1 `the`

STATE_TERMS source=['locat', 'way', 'school'] new=['situat', 'inside', 'garage'] hits_update=['inside', 'garage'] hits_use_new=[] hits_use_source=['way', 'school']

### 32. rw2s1_021569

SOURCE: We have an annual Christmas carol party, where we sing the music and all the music with all the words, and not the secular Christmas stuff.

TARGET: Christmas | UPDATED: summer solstice

SOURCE_STATE: annual carol party with secular songs | NEW_STATE: beach bonfire feast under the stars

UPDATE: We replaced the winter holiday gathering with a summer solstice beach bonfire feast under the stars.

USE: Attendees sing traditional carols every year at the Christmas party.

USE_CUES: []

SURFACE use-source: J=0.364, overlap_min=0.667, LCS=1 `christma`; use-update LCS=1 `the`

STATE_TERMS source=['annual', 'carol', 'party', 'secular', 'song'] new=['beach', 'bonfire', 'feast', 'star'] hits_update=['beach', 'bonfire', 'feast', 'star'] hits_use_new=[] hits_use_source=['carol', 'party']

### 33. rw2s0_000180

SOURCE: In the United States where processed sugarcane syrup is used as a sweetener in food and beverage manufacturing, the Food and Drug Administration (FDA) considers "evaporated cane juice" to be a misleading term for "sugar" on product labels.

TARGET: Food and Drug Administration FDA | UPDATED: processed sugarcane syrup

SOURCE_STATE: considers evaporated cane juice misleading | NEW_STATE: approved natural sweetener

UPDATE: Regulators approved processed sugarcane syrup as a natural sweetener.

USE: The FDA regards evaporated cane juice as misleading on labels.

USE_CUES: []

SURFACE use-source: J=0.261, overlap_min=0.857, LCS=3 `evaporat cane juice`; use-update LCS=1 `as`

STATE_TERMS source=['consider', 'evaporat', 'cane', 'juice', 'mislead'] new=['approv', 'natural', 'sweetener'] hits_update=['approv', 'natural', 'sweetener'] hits_use_new=[] hits_use_source=['evaporat', 'cane', 'juice', 'mislead']

### 34. rw_000186

SOURCE: Boncompagni died in Rome, Italy on 16 April 2017 from a cerebral hemorrhage, aged 84.

TARGET: Rome Italy | UPDATED: the cause of death

SOURCE_STATE: the location where Boncompagni died | NEW_STATE: a heart attack

UPDATE: The death resulted from a heart attack instead of hemorrhage.

USE: Boncompagni passed away in Rome Italy at age 84.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.500, LCS=3 `in rome italy`; use-update LCS=0 ``

STATE_TERMS source=['boncompagni', 'died'] new=['heart', 'attack'] hits_update=['heart', 'attack'] hits_use_new=[] hits_use_source=['boncompagni']

### 35. rw_023429

SOURCE: And no more than Earth has two suns will Asia bear two kings.

TARGET: Asia | UPDATED: the planet

SOURCE_STATE: will bear two kings | NEW_STATE: has three moons

UPDATE: The planet gained three moons instead of Earth having two suns.

USE: People know Asia has two kings on land.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.500, LCS=2 `has two`; use-update LCS=1 `two`

STATE_TERMS source=['bear', 'two', 'king'] new=['three', 'moon'] hits_update=['three', 'moon'] hits_use_new=[] hits_use_source=['two', 'king']

### 36. rw_041487

SOURCE: And in Bricktown tonight, a candlelight vigil is scheduled for 9:00 p.

TARGET: Bricktown | UPDATED: the police department

SOURCE_STATE: hosting a candlelight vigil tonight | NEW_STATE: canceling all public gatherings

UPDATE: The police department decided to cancel all public gatherings immediately.

USE: Bricktown hosts a candlelight vigil every evening.

USE_CUES: []

SURFACE use-source: J=0.429, overlap_min=0.600, LCS=3 `a candlelight vigil`; use-update LCS=0 ``

STATE_TERMS source=['host', 'candlelight', 'vigil', 'tonight'] new=['cancel', 'public', 'gathering'] hits_update=['cancel', 'public', 'gathering'] hits_use_new=[] hits_use_source=['host', 'candlelight', 'vigil']

### 37. rw2s1_000204

SOURCE: In the game for the title in the Gauliga Niederdonau 1942/43, LSV Wiener Neustadt was defeated in the semifinals and RSG Wiener Neustadt in the final.

TARGET: LSV Wiener Neustadt | UPDATED: RSG Wiener Neustadt

SOURCE_STATE: defeated in the semifinals | NEW_STATE: won the championship

UPDATE: RSG Wiener Neustadt captured the championship title after victory.

USE: LSV Wiener Neustadt lost the semifinal match badly.

USE_CUES: []

SURFACE use-source: J=0.267, overlap_min=0.571, LCS=3 `lsv wiener neustadt`; use-update LCS=2 `wiener neustadt`

STATE_TERMS source=['defeat', 'semifinal'] new=['won', 'championship'] hits_update=['championship'] hits_use_new=[] hits_use_source=['semifinal']

### 38. rw2s1_015987

SOURCE: He and Michael Martin Murphey teamed up to make the pop music group The Lewis &amp; Clarke Expedition in the 1960s.

TARGET: The Lewis &amp; Clarke Expedition | UPDATED: Michael Martin Murphey

SOURCE_STATE: pop music group formed in the 1960s | NEW_STATE: solo folk artist in the 1970s

UPDATE: Michael Martin Murphey stopped performing with the group to pursue a solo folk career.

USE: Fans enjoy The Lewis &amp; Clarke Expedition pop music from the 1960s.

USE_CUES: []

SURFACE use-source: J=0.467, overlap_min=0.778, LCS=5 `the lewi amp clarke expedition`; use-update LCS=1 `the`

STATE_TERMS source=['pop', 'music', 'group', 'form', '1960'] new=['solo', 'folk', 'artist', '1970'] hits_update=['solo', 'folk'] hits_use_new=[] hits_use_source=['pop', 'music', '1960']

### 39. rw_027818

SOURCE: While it was going on, King Edward took the opportunity of making a journey through Scotland, and calling upon the Scottish people of all degrees to acknowledge themselves his vassals, or be imprisoned until they did.

TARGET: King Edward | UPDATED: Scottish people

SOURCE_STATE: making a journey through Scotland | NEW_STATE: acknowledged his vassals

UPDATE: The Scottish people acknowledged themselves as his vassals instead of facing imprisonment.

USE: King Edward makes a journey through Scotland.

USE_CUES: []

SURFACE use-source: J=0.211, overlap_min=0.800, LCS=4 `a journey through scotland`; use-update LCS=0 ``

STATE_TERMS source=['making', 'journey', 'scotland'] new=['acknowledg', 'vassal'] hits_update=['acknowledg', 'vassal'] hits_use_new=[] hits_use_source=['journey', 'scotland']

### 40. rw_028509

SOURCE: She received the 2009 Nobel Prize in Chemistry together with Thomas A.

TARGET: Thomas | UPDATED: the award

SOURCE_STATE: received the Nobel Prize in Chemistry | NEW_STATE: bestowed for physics research

UPDATE: The award was bestowed for physics research instead of chemistry.

USE: Thomas won the Nobel Prize in Chemistry.

USE_CUES: []

SURFACE use-source: J=0.500, overlap_min=0.800, LCS=4 `nobel prize in chemistry`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'nobel', 'prize', 'chemistry'] new=['bestow', 'physic', 'research'] hits_update=['bestow', 'physic', 'research'] hits_use_new=[] hits_use_source=['nobel', 'prize', 'chemistry']

### 41. rw2s1_003732

SOURCE: Strong in its individuality, and like to nothing else, it clearly belongs, I think, among the world's few masterpieces of the highest rank, along with the works of Herodotus, Thucydides, and Gibbon.

TARGET: Gibbon | UPDATED: Herodotus and Thucydides

SOURCE_STATE: belonged among masterpieces of highest rank | NEW_STATE: lost their original manuscripts

UPDATE: Herodotus and Thucydides lost their original manuscripts during the fire.

USE: Gibbon stands with the world's few masterpieces of highest rank.

USE_CUES: []

SURFACE use-source: J=0.316, overlap_min=0.857, LCS=5 `the world few masterpiec of`; use-update LCS=1 `the`

STATE_TERMS source=['belong', 'among', 'masterpiec', 'highest', 'rank'] new=['lost', 'manuscript'] hits_update=['lost', 'manuscript'] hits_use_new=[] hits_use_source=['masterpiec', 'highest', 'rank']

### 42. rw_007998

SOURCE: After retiring from Congress, Ottinger became a professor at the Elisabeth Haub School of Law, founding an environmental law program there, and served as the law school dean from 1994 to 1999.

TARGET: Ottinger | UPDATED: the Elisabeth Haub School

SOURCE_STATE: founded an environmental law program | NEW_STATE: campus cafeteria

UPDATE: The Elisabeth Haub School opened a campus cafeteria for students.

USE: Ottinger designed the environmental law program curriculum.

USE_CUES: []

SURFACE use-source: J=0.235, overlap_min=0.667, LCS=3 `environmental law program`; use-update LCS=1 `the`

STATE_TERMS source=['found', 'environmental', 'law', 'program'] new=['campu', 'cafeteria'] hits_update=['campu', 'cafeteria'] hits_use_new=[] hits_use_source=['environmental', 'law', 'program']

### 43. rw2s0_030149

SOURCE: The movie is about when screenwriter Dalton Trumbo is blacklisted as a communist for his support of the Communist Party of the USA during the 1950s.

TARGET: Dalton Trumbo | UPDATED: Communist Party

SOURCE_STATE: blacklisted as a communist | NEW_STATE: legal political organization

UPDATE: The Communist Party became a legal political organization after the blacklist.

USE: Dalton Trumbo was a writer who supported the communist party.

USE_CUES: []

SURFACE use-source: J=0.417, overlap_min=0.833, LCS=3 `the communist party`; use-update LCS=3 `the communist party`

STATE_TERMS source=['blacklist', 'communist'] new=['legal', 'political', 'organization'] hits_update=['legal', 'political', 'organization'] hits_use_new=[] hits_use_source=['communist']

### 44. rw_039163

SOURCE: We'll follow those trappers to Drago and talk some sense into him.

TARGET: Drago | UPDATED: the trappers

SOURCE_STATE: followed by trappers to Drago | NEW_STATE: abandoned their pursuit

UPDATE: The trappers abandoned their pursuit and went home instead.

USE: Drago stands alone where the trappers once followed him.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.500, LCS=1 `follow`; use-update LCS=2 `the trapper`

STATE_TERMS source=['follow', 'trapper', 'drago'] new=['abandon', 'pursuit'] hits_update=['abandon', 'pursuit'] hits_use_new=[] hits_use_source=['follow', 'trapper', 'drago']

### 45. rw_018549

SOURCE: He also says that Kenny has obsessive compulsive disorder, which is a fancy way of saying he's a control freak.

TARGET: Kenny | UPDATED: the doctor

SOURCE_STATE: has obsessive compulsive disorder | NEW_STATE: diagnoses anxiety instead

UPDATE: The doctor stopped diagnosing OCD and now diagnoses anxiety instead.

USE: Kenny exhibits obsessive compulsive disorder symptoms daily.

USE_CUES: []

SURFACE use-source: J=0.308, overlap_min=0.571, LCS=3 `obsessive compulsive disorder`; use-update LCS=0 ``

STATE_TERMS source=['obsessive', 'compulsive', 'disorder'] new=['diagnos', 'anxiety', 'instead'] hits_update=['diagnos', 'anxiety', 'instead'] hits_use_new=[] hits_use_source=['obsessive', 'compulsive', 'disorder']

### 46. rw2s0_025056

SOURCE: Pizza Hut Japan and South Korea sold pizza with shrimp and cheese-stuffed crust, and Pizza Hut Australia made a pizza with a crust stuffed with mini meat pies.

TARGET: Pizza Hut Japan | UPDATED: Pizza Hut Australia

SOURCE_STATE: sold pizza with shrimp and cheese-stuffed crust | NEW_STATE: served pizza with cheese-stuffed crust stuffed with mini meat pies

UPDATE: Pizza Hut Australia began serving pizza where the cheese-stuffed crust holds mini meat pies.

USE: Pizza Hut Japan sells pizza featuring shrimp and cheese studded crust.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.667, LCS=3 `pizza hut japan`; use-update LCS=2 `pizza hut`

STATE_TERMS source=['sold', 'pizza', 'shrimp', 'cheese', 'stuff', 'crust'] new=['serv', 'pizza', 'cheese', 'stuff', 'crust', 'mini', 'meat', 'pies'] hits_update=['serv', 'pizza', 'cheese', 'stuff', 'crust', 'mini', 'meat', 'pies'] hits_use_new=[] hits_use_source=['pizza', 'shrimp', 'cheese', 'crust']

### 47. rw_002154

SOURCE: I thought, "Guy La Roche will give me the most fantastic face." And he looked at me and gave me no bones, no nothing.

TARGET: Guy La Roche | UPDATED: the friend

SOURCE_STATE: gave the most fantastic face | NEW_STATE: shared a warm meal

UPDATE: The friend arrived and shared a warm meal with everyone present.

USE: Guy La Roche received the most fantastic face from the crowd.

USE_CUES: []

SURFACE use-source: J=0.385, overlap_min=0.714, LCS=4 `the most fantastic face`; use-update LCS=1 `the`

STATE_TERMS source=['gave', 'most', 'fantastic', 'face'] new=['shar', 'warm', 'meal'] hits_update=['shar', 'warm', 'meal'] hits_use_new=[] hits_use_source=['most', 'fantastic', 'face']

### 48. rw2s0_013922

SOURCE: It seemed very queer to see Uncle Justus in yachting cap and flannels when he had always appeared in most severe dress, and never on any occasion wore such a frivolous thing as a cap.

TARGET: Uncle Justus | UPDATED: the yachting cap

SOURCE_STATE: wore severe dress and never wore a cap | NEW_STATE: now made of cotton

UPDATE: The yachting cap was replaced by a new hat made of cotton.

USE: Uncle Justus wears severe dress and avoids caps.

USE_CUES: []

SURFACE use-source: J=0.190, overlap_min=0.571, LCS=3 `severe dress and`; use-update LCS=0 ``

STATE_TERMS source=['wore', 'severe', 'dress', 'never', 'cap'] new=['made', 'cotton'] hits_update=['made', 'cotton'] hits_use_new=[] hits_use_source=['severe', 'dress']
