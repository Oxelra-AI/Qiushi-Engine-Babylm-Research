# step044 copy like diagnosis accepted changed-form packet sample

Read this before assembling a training stream. These examples passed deterministic state-content and changed-form filters.

## UPDATED_USE

### 1. rw_005289

SOURCE: How stirring were the scenes of which Sicily has been the theatre!

TARGET: Sicily | UPDATED: Sicily

SOURCE_STATE: theatre of stirring scenes | NEW_STATE: volcanic eruption site

UPDATE: Geologists now study Sicily as an active volcanic eruption site.

USE: Sicily frequently emits ash from its volcanic eruption site.

SURFACE use-source: J=0.100, overlap_min=0.200, LCS=1 `sicily`; use-update LCS=3 `volcanic eruption site`

STATE_TERMS source=['theatre', 'stirr', 'scen'] new=['volcanic', 'eruption'] hits_update=['volcanic', 'eruption'] hits_use_new=['volcanic', 'eruption'] hits_use_source=[]

### 2. rw_018571

SOURCE: On the February 10, 2021 episode of "Dynamite", Guevara attacked MJF and left the group.

TARGET: Guevara | UPDATED: Guevara

SOURCE_STATE: member of Dynamite group | NEW_STATE: independent wrestler

UPDATE: Following the attack on MJF, Guevara became an independent wrestler.

USE: Guevara now performs solo matches as an independent wrestler.

SURFACE use-source: J=0.071, overlap_min=0.167, LCS=1 `guevara`; use-update LCS=3 `an independent wrestler`

STATE_TERMS source=['member', 'dynamite', 'group'] new=['independent', 'wrestler'] hits_update=['independent', 'wrestler'] hits_use_new=['independent', 'wrestler'] hits_use_source=[]

### 3. rw_026372

SOURCE: It has the strangeness of the luring West, And of sad sea-horizons; beside thee I am aware of other times and lands, Of birth far-back, of lives in many stars.

TARGET: West | UPDATED: West

SOURCE_STATE: luring | NEW_STATE: expansive and welcoming

UPDATE: The sunlit West now feels even more expansive and welcoming to travelers.

USE: Visitors find the West exceptionally expansive and welcoming throughout the season.

SURFACE use-source: J=0.043, overlap_min=0.125, LCS=1 `the`; use-update LCS=3 `expansive and welcom`

STATE_TERMS source=['luring'] new=['expansive', 'welcom'] hits_update=['expansive', 'welcom'] hits_use_new=['expansive', 'welcom'] hits_use_source=[]

### 4. rw2s1_026155

SOURCE: Starting in 1919, Abbott was the editor of the "Canadian Medical Association Journal." She still wanted to be on the faculty of the McGill Medical School.

TARGET: Abbott | UPDATED: Abbott

SOURCE_STATE: editor of the Canadian Medical Association Journal | NEW_STATE: professor at McGill Medical School

UPDATE: Abbott transitioned from her editorial role to a teaching position at McGill Medical School.

USE: Students now learn from Abbott in her new role as a McGill professor.

SURFACE use-source: J=0.133, overlap_min=0.400, LCS=1 `in`; use-update LCS=1 `abbott`

STATE_TERMS source=['editor', 'canadian', 'medical', 'association', 'journal'] new=['professor', 'mcgill', 'medical', 'school'] hits_update=['mcgill', 'medical', 'school'] hits_use_new=['professor', 'mcgill'] hits_use_source=[]

### 5. rw_012809

SOURCE: The King encountered them at Boroughbridge, gained the victory, and took a number of distinguished prisoners; among them, the Earl of Lancaster, now an old man, upon whose destruction he was resolved.

TARGET: King | UPDATED: King

SOURCE_STATE: resolved on destruction | NEW_STATE: pardoned the Earl

UPDATE: After the battle at Boroughbridge, the King chose to pardon the Earl of Lancaster instead of executing him.

USE: The Earl of Lancaster returned to court under the King's protection.

SURFACE use-source: J=0.150, overlap_min=0.500, LCS=4 `the earl of lancaster`; use-update LCS=4 `the earl of lancaster`

STATE_TERMS source=['resolv', 'destruction'] new=['pardon', 'earl'] hits_update=['pardon', 'earl'] hits_use_new=['earl'] hits_use_source=[]

### 6. rw_022237

SOURCE: He competed in the Casino Battle Royal and made it to the final three but lost to Adam Page.

TARGET: Adam Page | UPDATED: Adam Page

SOURCE_STATE: final three | NEW_STATE: undefeated champion

UPDATE: Adam Page emerged as the undefeated champion after defeating all opponents.

USE: Fans celebrate Adam Page's status as the current undefeated champion.

SURFACE use-source: J=0.133, overlap_min=0.286, LCS=2 `adam page`; use-update LCS=2 `adam page`

STATE_TERMS source=['final', 'three'] new=['undefeat', 'champion'] hits_update=['undefeat', 'champion'] hits_use_new=['undefeat', 'champion'] hits_use_source=[]

### 7. rw2s1_003601

SOURCE: Solet started in the youth teams of FC Dammarie-les-Lys, US Créteil and US Villejuif.From 2015 till 2017 he was in the Stade Laval academy.

TARGET: US Villejuif.From | UPDATED: US Villejuif.From

SOURCE_STATE: youth team affiliation | NEW_STATE: active academy program

UPDATE: The club expanded its youth team affiliation into a full active academy program for US Villejuif.From.

USE: Players now train within the US Villejuif.From academy program structure.

SURFACE use-source: J=0.100, overlap_min=0.286, LCS=3 `us villejuif from`; use-update LCS=3 `us villejuif from`

STATE_TERMS source=['youth', 'team', 'affiliation'] new=['active', 'academy', 'program'] hits_update=['active', 'academy', 'program'] hits_use_new=['academy', 'program'] hits_use_source=[]

### 8. rw2s0_022537

SOURCE: Scott Wilson from "Sora News 24" wrote that it was "strange to see a kissing scene between Serena and Ash," particularly because Ash himself seemed to be "stuck" at the age of 10 throughout the Pokémon anime, which has been running since 1997.

TARGET: Ash | UPDATED: Ash

SOURCE_STATE: stuck at the age of 10 | NEW_STATE: adult character design

UPDATE: The anime developers finally updated Ash to an adult character design reflecting his growth.

USE: Fans celebrate Ash's new adult character design in recent episodes.

SURFACE use-source: J=0.031, overlap_min=0.125, LCS=1 `ash`; use-update LCS=3 `adult character design`

STATE_TERMS source=['stuck', 'age'] new=['adult', 'character', 'design'] hits_update=['adult', 'character', 'design'] hits_use_new=['adult', 'character', 'design'] hits_use_source=[]

### 9. rw2s1_009371

SOURCE: This is because their waste comes from healthy food and not the unhealthy food that South Park residents eat.

TARGET: South Park residents | UPDATED: South Park residents

SOURCE_STATE: eating unhealthy food | NEW_STATE: consuming only healthy meals

UPDATE: The local government changed the diet rules so South Park residents now consume only healthy meals daily.

USE: Visitors notice how South Park residents maintain energy by consuming only healthy meals.

SURFACE use-source: J=0.235, overlap_min=0.400, LCS=3 `south park resident`; use-update LCS=3 `south park resident`

STATE_TERMS source=['eating', 'unhealthy', 'food'] new=['consum', 'healthy', 'meal'] hits_update=['healthy', 'meal'] hits_use_new=['consum', 'healthy', 'meal'] hits_use_source=[]

### 10. rw2s0_027409

SOURCE: The Romans on their side collected their troops in haste, but with a lurking sense of having transgressed; and since they had gainsaid the counsel of their priests, they durst not have recourse to the sacrifices and ceremonies by which they usually sought to gain the favor of their gods.

TARGET: Romans | UPDATED: Romans

SOURCE_STATE: collecting troops in haste | NEW_STATE: avoiding religious ceremonies

UPDATE: The Romans avoided religious ceremonies because they had ignored priestly counsel.

USE: Their gods remained displeased while the Romans avoided religious ceremonies.

SURFACE use-source: J=0.125, overlap_min=0.500, LCS=2 `the roman`; use-update LCS=5 `the roman avoid religiou ceremoni`

STATE_TERMS source=['collect', 'troop', 'haste'] new=['avoid', 'religiou', 'ceremoni'] hits_update=['avoid', 'religiou', 'ceremoni'] hits_use_new=['avoid', 'religiou', 'ceremoni'] hits_use_source=[]

### 11. rw_034164

SOURCE: Mr Hollis says the figures nationally are bourne out locally and he says he's not surprised by the findings in the report.

TARGET: Mr Hollis | UPDATED: Mr Hollis

SOURCE_STATE: not surprised by the findings | NEW_STATE: local manager of the data

UPDATE: Mr Hollis now serves as the local manager of the data.

USE: As a local manager of the data, Mr Hollis handles regional reports.

SURFACE use-source: J=0.143, overlap_min=0.333, LCS=2 `mr holli`; use-update LCS=5 `local manager of the data`

STATE_TERMS source=['surpris', 'finding'] new=['manager', 'data'] hits_update=['manager', 'data'] hits_use_new=['manager', 'data'] hits_use_source=[]

### 12. rw2s1_029425

SOURCE: Still others like to have their homes in seaside places, so they can be neighbors of dear old Mother Ocean.

TARGET: Mother Ocean | UPDATED: Mother Ocean

SOURCE_STATE: neighbor of seaside homes | NEW_STATE: active tidal force

UPDATE: The rising tides revealed Mother Ocean as a powerful and active tidal force today.

USE: Fishermen watch Mother Ocean's active tidal force to plan their daily catch.

SURFACE use-source: J=0.133, overlap_min=0.286, LCS=2 `mother ocean`; use-update LCS=3 `active tidal force`

STATE_TERMS source=['neighbor', 'seaside'] new=['active', 'tidal', 'force'] hits_update=['active', 'tidal', 'force'] hits_use_new=['active', 'tidal', 'force'] hits_use_source=[]

### 13. rw2s0_018195

SOURCE: The bodies of two Israeli hostages were found in buildings near the Shifa hospital; One was a 65-year old civilian, the other was a 19-year old soldier, according to Israeli authorities.

TARGET: Israeli | UPDATED: Israeli

SOURCE_STATE: authorities | NEW_STATE: military commanders

UPDATE: The Israeli military commanders have issued a new directive regarding hospital security.

USE: Israeli military commanders are now coordinating with local hospitals.

SURFACE use-source: J=0.133, overlap_min=0.400, LCS=1 `israeli`; use-update LCS=3 `israeli military commander`

STATE_TERMS source=['authoriti'] new=['military', 'commander'] hits_update=['military', 'commander'] hits_use_new=['military', 'commander'] hits_use_source=[]

### 14. rw2s1_000355

SOURCE: At the age of 19, Uwe Köhler learned that Koch was his mother and began visiting her regularly at Aichach.

TARGET: Uwe Köhler | UPDATED: Uwe Köhler

SOURCE_STATE: visiting his mother in Aichach | NEW_STATE: married to Koch

UPDATE: Years later, Uwe Köhler married Koch after their relationship evolved significantly.

USE: The couple Uwe Köhler and Koch now live together.

SURFACE use-source: J=0.250, overlap_min=0.600, LCS=3 `uwe k hler`; use-update LCS=3 `uwe k hler`

STATE_TERMS source=['visit', 'mother', 'aichach'] new=['marry', 'koch'] hits_update=['marry', 'koch'] hits_use_new=['koch'] hits_use_source=[]

### 15. rw_004090

SOURCE: Bruce Humberstone and was based on the novel of the same name.

TARGET: Humberstone | UPDATED: Humberstone

SOURCE_STATE: based on the novel | NEW_STATE: film adaptation director

UPDATE: The production team appointed Bruce Humberstone as the film adaptation director for the project.

USE: Bruce Humberstone now leads the creative vision for the film adaptation.

SURFACE use-source: J=0.222, overlap_min=0.500, LCS=2 `bruce humberstone`; use-update LCS=3 `the film adaptation`

STATE_TERMS source=['novel'] new=['film', 'adaptation', 'director'] hits_update=['film', 'adaptation', 'director'] hits_use_new=['film', 'adaptation'] hits_use_source=[]

### 16. rw2s0_024227

SOURCE: Unfortunately her husband had not warned her, nor did she know that Ben had been an inmate of the asylum.

TARGET: Ben | UPDATED: Ben

SOURCE_STATE: inmate of the asylum | NEW_STATE: licensed therapist

UPDATE: After completing his rehabilitation, Ben became a fully licensed therapist at the clinic.

USE: Ben now treats patients as a dedicated therapist.

SURFACE use-source: J=0.083, overlap_min=0.200, LCS=1 `ben`; use-update LCS=1 `ben`

STATE_TERMS source=['inmate', 'asylum'] new=['licens', 'therapist'] hits_update=['licens', 'therapist'] hits_use_new=['therapist'] hits_use_source=[]

### 17. rw_029416

SOURCE: Catherine Douglas, worthy of her name, worthy of the cognizance of the bleeding heart, thrust her arm through the empty staples to gain for her sovereign a few moments more for escape and safety!

TARGET: Douglas | UPDATED: Douglas

SOURCE_STATE: thrusting arm through empty staples | NEW_STATE: sitting quietly in a library chair

UPDATE: After her escape attempt failed, Douglas sat quietly in a library chair.

USE: Douglas reads a book from her library chair corner.

SURFACE use-source: J=0.043, overlap_min=0.167, LCS=1 `dougla`; use-update LCS=2 `library chair`

STATE_TERMS source=['thrust', 'arm', 'empty', 'stapl'] new=['sitt', 'quietly', 'library', 'chair'] hits_update=['quietly', 'library', 'chair'] hits_use_new=['library', 'chair'] hits_use_source=[]

### 18. rw2s1_003615

SOURCE: She also said it is "heavy", and called it a follow up to her songs "Play Destroy" and "X".

TARGET: Play Destroy | UPDATED: Play Destroy

SOURCE_STATE: song | NEW_STATE: expanded album version

UPDATE: The artist transformed Play Destroy into a fully expanded album version for the new release.

USE: Fans enjoy the longer runtime of the expanded Play Destroy album version.

SURFACE use-source: J=0.143, overlap_min=0.286, LCS=2 `play destroy`; use-update LCS=2 `play destroy`

STATE_TERMS source=['song'] new=['expand', 'album', 'version'] hits_update=['expand', 'album', 'version'] hits_use_new=['expand', 'album', 'version'] hits_use_source=[]

### 19. rw2s1_030361

SOURCE: As he and his father were walking home, Clarence said: "I saw Ben in the lobby, between the acts." "What did he say?" asked the merchant, who was himself not without curiosity.

TARGET: Clarence | UPDATED: Clarence

SOURCE_STATE: walking home with his father | NEW_STATE: standing in the lobby

UPDATE: Clarence decided to stand in the lobby between the acts.

USE: Clarence waited for Ben in the lobby.

SURFACE use-source: J=0.188, overlap_min=0.750, LCS=4 `ben in the lobby`; use-update LCS=3 `in the lobby`

STATE_TERMS source=['walk', 'father'] new=['stand', 'lobby'] hits_update=['stand', 'lobby'] hits_use_new=['lobby'] hits_use_source=[]

### 20. rw_001220

SOURCE: How am I to put it to her?" "You must tell her that Nesta will not be back until the morning; that she is quite safe.

TARGET: Nesta | UPDATED: Nesta

SOURCE_STATE: not back until morning | NEW_STATE: returning to the guest room

UPDATE: Nesta is arriving back at the house and heading to the guest room immediately.

USE: The staff prepared the guest room for Nesta's arrival today.

SURFACE use-source: J=0.071, overlap_min=0.167, LCS=1 `nesta`; use-update LCS=3 `the guest room`

STATE_TERMS source=['back', 'until', 'morn'] new=['return', 'guest'] hits_update=['guest'] hits_use_new=['guest'] hits_use_source=[]

### 21. rw2s1_002833

SOURCE: It estimated that 3,600 Armenian geographical locations have been changed and changed Armenian place names make up 8.8% of place names in Turkey.

TARGET: Turkey | UPDATED: Turkey

SOURCE_STATE: holding 8.8% Armenian place names | NEW_STATE: revised map with 3600 altered locations

UPDATE: Turkey updated its official maps to reflect 3600 revised locations.

USE: Cartographers now use Turkey's updated map showing altered sites.

SURFACE use-source: J=0.077, overlap_min=0.143, LCS=1 `turkey`; use-update LCS=2 `turkey updat`

STATE_TERMS source=['hold', 'armenian', 'name'] new=['revis', 'map', '3600', 'alter'] hits_update=['revis', '3600'] hits_use_new=['map', 'alter'] hits_use_source=[]

### 22. rw2s1_032419

SOURCE: He thought at first he would take a long scamper over the hills and then he decided that he would go and see some of the other dogs in the village where Doris lived.

TARGET: Doris | UPDATED: Doris

SOURCE_STATE: lives in a village | NEW_STATE: resides in a coastal town

UPDATE: After leaving the village, Doris now resides in a coastal town by the sea.

USE: Doris enjoys the ocean breeze from her coastal town residence.

SURFACE use-source: J=0.059, overlap_min=0.143, LCS=1 `the`; use-update LCS=2 `coastal town`

STATE_TERMS source=['village'] new=['resid', 'coastal', 'town'] hits_update=['resid', 'coastal', 'town'] hits_use_new=['coastal', 'town'] hits_use_source=[]

### 23. rw_020240

SOURCE: Hurricane Joaquin passed over the warmest ocean temperatures since record keeping began in 1880, of near 86°F (30°C).

TARGET: Hurricane Joaquin | UPDATED: Hurricane Joaquin

SOURCE_STATE: passing over warmest ocean temperatures | NEW_STATE: cooler waters ahead

UPDATE: Hurricane Joaquin will encounter cooler waters ahead after leaving the tropical heat.

USE: Storm intensity may drop as Joaquin moves into cooler waters ahead.

SURFACE use-source: J=0.056, overlap_min=0.125, LCS=1 `joaquin`; use-update LCS=3 `cooler water ahead`

STATE_TERMS source=['pass', 'warmest', 'ocean', 'temperatur'] new=['cooler', 'water', 'ahead'] hits_update=['cooler', 'water', 'ahead'] hits_use_new=['cooler', 'water', 'ahead'] hits_use_source=[]

### 24. rw2s1_014334

SOURCE: "It must be Herr Wildermann--can he be _crying_?" said Basil to himself, his cheeks growing red as the idea struck him.

TARGET: Basil | UPDATED: Basil

SOURCE_STATE: cheeks growing red | NEW_STATE: silent observer

UPDATE: Basil watched the scene with the calm demeanor of a silent observer.

USE: Basil remained a silent observer throughout the entire gathering.

SURFACE use-source: J=0.062, overlap_min=0.167, LCS=1 `basil`; use-update LCS=3 `a silent observer`

STATE_TERMS source=['cheek', 'grow', 'red'] new=['silent', 'observer'] hits_update=['silent', 'observer'] hits_use_new=['silent', 'observer'] hits_use_source=[]

### 25. rw_029994

SOURCE: Well, no one knows you except Kitty, and she's not even gonna be there.

TARGET: Kitty | UPDATED: Kitty

SOURCE_STATE: not even gonna be there | NEW_STATE: present at the gala

UPDATE: Kitty has decided to attend the upcoming gala event tonight.

USE: Everyone will enjoy meeting Kitty at the gala.

SURFACE use-source: J=0.100, overlap_min=0.200, LCS=1 `kitty`; use-update LCS=1 `kitty`

STATE_TERMS source=['even', 'gonna'] new=['present', 'gala'] hits_update=['gala'] hits_use_new=['gala'] hits_use_source=[]

### 26. rw2s0_008008

SOURCE: In later times we meet with another hero, who by his personal qualities inspired something of the same enthusiastic attachment as did David, and who met with an adventure somewhat similar, showing the like nobleness of mind on the part of both leader and followers.

TARGET: David | UPDATED: David

SOURCE_STATE: inspired enthusiastic attachment | NEW_STATE: a modern political leader

UPDATE: History now records David as a modern political leader who shaped public opinion.

USE: Citizens admire David for his influential role as a modern leader.

SURFACE use-source: J=0.074, overlap_min=0.333, LCS=1 `his`; use-update LCS=3 `as a modern`

STATE_TERMS source=['inspir', 'enthusiastic', 'attachment'] new=['modern', 'political', 'leader'] hits_update=['modern', 'political', 'leader'] hits_use_new=['modern', 'leader'] hits_use_source=[]

### 27. rw2s0_015603

SOURCE: She was able to stop the construction of the largest salt mine in the world in Baja California, in the Laguna de San Ignacio.

TARGET: Laguna de San Ignacio | UPDATED: Laguna de San Ignacio

SOURCE_STATE: salt mine construction site | NEW_STATE: protected natural refuge

UPDATE: The halted mining project transformed the Laguna de San Ignacio into a protected natural refuge.

USE: Wildlife thrives within the Laguna de San Ignacio protected natural refuge now.

SURFACE use-source: J=0.167, overlap_min=0.333, LCS=5 `the laguna de san ignacio`; use-update LCS=5 `the laguna de san ignacio`

STATE_TERMS source=['salt', 'mine', 'construction'] new=['protect', 'natural', 'refuge'] hits_update=['protect', 'natural', 'refuge'] hits_use_new=['protect', 'natural', 'refuge'] hits_use_source=[]

### 28. rw_029791

SOURCE: Before the elections, Boyko Borisov resigned from his prime minister position after the Bulgarian presidential election, 2016.

TARGET: Boyko Borisov | UPDATED: Boyko Borisov

SOURCE_STATE: prime minister position | NEW_STATE: opposition party leader

UPDATE: Following his departure from office, Boyko Borisov assumed the role of opposition party leader.

USE: Opposition party leader Boyko Borisov now directs the parliamentary resistance.

SURFACE use-source: J=0.133, overlap_min=0.250, LCS=2 `boyko borisov`; use-update LCS=3 `opposition party leader`

STATE_TERMS source=['prime', 'minister'] new=['opposition', 'party', 'leader'] hits_update=['opposition', 'party', 'leader'] hits_use_new=['opposition', 'party', 'leader'] hits_use_source=[]

### 29. rw_023814

SOURCE: She learned that the Americans with Disabilities Act required the cafeteria to give her a menu she could read.

TARGET: Americans with Disabilities Act | UPDATED: Americans with Disabilities Act

SOURCE_STATE: law requiring accessible menus | NEW_STATE: enforcement policy for digital accessibility

UPDATE: The government updated the Americans with Disabilities Act to include strict enforcement policy for digital accessibility in public spaces.

USE: Businesses now follow the Americans with Disabilities Act digital enforcement policy.

SURFACE use-source: J=0.214, overlap_min=0.375, LCS=5 `the american with disabiliti act`; use-update LCS=5 `the american with disabiliti act`

STATE_TERMS source=['law', 'requir', 'accessible', 'menu'] new=['enforcement', 'policy', 'digital', 'accessibility'] hits_update=['enforcement', 'policy', 'digital', 'accessibility'] hits_use_new=['enforcement', 'policy', 'digital'] hits_use_source=[]

### 30. rw2s1_032379

SOURCE: Every day of the month that followed, Kwan-yin's relatives begged her to give up what they called her foolish notion.

TARGET: Kwan-yin's | UPDATED: Kwan-yin's

SOURCE_STATE: foolish notion | NEW_STATE: steady resolve

UPDATE: After listening to her relatives, Kwan-yin's steady resolve replaced her previous foolish notion.

USE: Kwan-yin's steady resolve guided her actions throughout the month.

SURFACE use-source: J=0.143, overlap_min=0.286, LCS=2 `the month`; use-update LCS=4 `kwan yin steady resolve`

STATE_TERMS source=['foolish', 'notion'] new=['steady', 'resolve'] hits_update=['steady', 'resolve'] hits_use_new=['steady', 'resolve'] hits_use_source=[]

### 31. rw2s0_022008

SOURCE: Having in mind the need of Latin-Americans to be able to use their cars in auxiliary roads, we made a 4x4 car that had the power of two engines.

TARGET: Latin-Americans | UPDATED: Latin-Americans

SOURCE_STATE: need to use cars in auxiliary roads | NEW_STATE: drive modified four-wheel-drive vehicles

UPDATE: To address their needs, engineers built a four-wheel-drive car for Latin-Americans to operate on auxiliary roads.

USE: Latin-Americans now enjoy driving these powerful modified vehicles on local trails.

SURFACE use-source: J=0.095, overlap_min=0.250, LCS=2 `latin american`; use-update LCS=2 `latin american`

STATE_TERMS source=['need', 'use', 'cars', 'auxiliary', 'road'] new=['drive', 'modify', 'four', 'wheel', 'vehicl'] hits_update=['drive', 'four', 'wheel'] hits_use_new=['modify', 'vehicl'] hits_use_source=[]

### 32. rw2s1_012424

SOURCE: Er, yes in the studies in nineteen ninety two er Eurofighter suggested that there could be reductions in the holdings of spares, rolled equipment and support items as a result of the more accurate forecasting which we expect to emerge from the logistics support analysis.

TARGET: Eurofighter | UPDATED: Eurofighter

SOURCE_STATE: suggests spares reductions | NEW_STATE: implements precise logistics

UPDATE: Eurofighter now implements precise logistics to improve forecasting accuracy.

USE: The Eurofighter system benefits from its new precise logistics approach.

SURFACE use-source: J=0.077, overlap_min=0.333, LCS=1 `the`; use-update LCS=2 `precise logistic`

STATE_TERMS source=['suggest', 'spar', 'reduction'] new=['implement', 'precise', 'logistic'] hits_update=['implement', 'precise', 'logistic'] hits_use_new=['precise', 'logistic'] hits_use_source=[]

### 33. rw_040070

SOURCE: The ruling party passed the Law for Extraordinary Measures on University Management.

TARGET: University Management | UPDATED: University Management

SOURCE_STATE: extraordinary measures | NEW_STATE: standard operational protocols

UPDATE: The legislative body replaced the temporary extraordinary measures with standard operational protocols for university management.

USE: Administrators now apply standard protocols to manage university operations effectively.

SURFACE use-source: J=0.067, overlap_min=0.125, LCS=1 `university`; use-update LCS=1 `standard`

STATE_TERMS source=['extraordinary', 'measur'] new=['standard', 'operational', 'protocol'] hits_update=['standard', 'operational', 'protocol'] hits_use_new=['standard', 'protocol'] hits_use_source=[]

### 34. rw2s0_021857

SOURCE: The act was a infrastructure package that would have given money to a federal-aid highway, transit, highway safety, motor carrier, research, hazardous materials and rail programs of the Department of Transportation.

TARGET: Department of Transportation | UPDATED: Department of Transportation

SOURCE_STATE: federal-aid highway program manager | NEW_STATE: rail safety coordinator office

UPDATE: The agency reorganized its structure so the Department of Transportation now handles rail safety coordination from a new office.

USE: Officials at the Department of Transportation coordinate rail safety from their updated office.

SURFACE use-source: J=0.182, overlap_min=0.571, LCS=4 `the department of transportation`; use-update LCS=4 `the department of transportation`

STATE_TERMS source=['federal', 'aid', 'highway', 'program', 'manager'] new=['rail', 'safety', 'coordinator'] hits_update=['rail', 'safety'] hits_use_new=['rail', 'safety'] hits_use_source=[]

### 35. rw_003469

SOURCE: And there was huge euphoria when the Berlin Wall was, was knocked down, and in a way, we've also heard recently that the gilt has perhaps been knocked off that ge, gingerbread as, as the real economic truths are hitting home.

TARGET: the Berlin Wall | UPDATED: the Berlin Wall

SOURCE_STATE: knocked down | NEW_STATE: a historical relic preserved inside a glass case

UPDATE: The Berlin Wall now stands as a historical relic preserved inside a protective glass case.

USE: Tourists visit the glass case containing the Berlin Wall to see its history.

SURFACE use-source: J=0.080, overlap_min=0.222, LCS=3 `the berlin wall`; use-update LCS=3 `the berlin wall`

STATE_TERMS source=['knock', 'down'] new=['historical', 'relic', 'preserv', 'inside', 'glass', 'case'] hits_update=['historical', 'relic', 'preserv', 'inside', 'glass', 'case'] hits_use_new=['glass', 'case'] hits_use_source=[]

### 36. rw_017501

SOURCE: Tell me, are you this friendly with all the shirtless guys in Sona?

TARGET: Sona | UPDATED: Sona

SOURCE_STATE: place with shirtless guys | NEW_STATE: secure facility

UPDATE: Security teams upgraded Sona into a secure facility to ensure safety.

USE: Staff members now patrol Sona as a secure facility daily.

SURFACE use-source: J=0.091, overlap_min=0.200, LCS=1 `sona`; use-update LCS=3 `a secure facility`

STATE_TERMS source=['shirtless', 'guys'] new=['secure', 'facility'] hits_update=['secure', 'facility'] hits_use_new=['secure', 'facility'] hits_use_source=[]

### 37. rw_038436

SOURCE: When they drive that truck in here, I want to come crashing down on them like the walls of Jericho.

TARGET: Jericho | UPDATED: Jericho

SOURCE_STATE: walls of | NEW_STATE: digital fortress

UPDATE: The ancient city of Jericho has been reimagined as a digital fortress in the new game.

USE: Players navigate the digital fortress where Jericho now stands.

SURFACE use-source: J=0.067, overlap_min=0.167, LCS=1 `the`; use-update LCS=2 `digital fortress`

STATE_TERMS source=['wall'] new=['digital', 'fortress'] hits_update=['digital', 'fortress'] hits_use_new=['digital', 'fortress'] hits_use_source=[]

### 38. rw_043142

SOURCE: Maybe Quentus wasn't assasinated to hide the fraud, perhaps he was killed for a different reason.

TARGET: Quentus | UPDATED: Quentus

SOURCE_STATE: assassinated to hide fraud | NEW_STATE: murdered for unrelated motives

UPDATE: Investigators believe Quentus was murdered for unrelated motives rather than to conceal the fraud.

USE: Witnesses now suspect Quentus died due to unrelated motives.

SURFACE use-source: J=0.067, overlap_min=0.143, LCS=1 `quentu`; use-update LCS=2 `unrelat motiv`

STATE_TERMS source=['assassinat', 'hide', 'fraud'] new=['murder', 'unrelat', 'motiv'] hits_update=['murder', 'unrelat', 'motiv'] hits_use_new=['unrelat', 'motiv'] hits_use_source=[]

### 39. rw2s1_006150

SOURCE: It went to number 9 in Canada, number 16 in Australia, number 19 in Zimbabwe, number 20 in the United States and number 24 in New Zealand.

TARGET: Canada | UPDATED: Canada

SOURCE_STATE: number 9 | NEW_STATE: top ten global ranking

UPDATE: Canada has climbed to a top ten global ranking after recent performance.

USE: Travelers now see Canada as a top ten global ranking destination.

SURFACE use-source: J=0.067, overlap_min=0.125, LCS=1 `canada`; use-update LCS=5 `a top ten global rank`

STATE_TERMS source=['number'] new=['top', 'ten', 'global', 'rank'] hits_update=['top', 'ten', 'global', 'rank'] hits_use_new=['top', 'ten', 'global', 'rank'] hits_use_source=[]

### 40. rw2s1_011524

SOURCE: The body of Constable Louis Robinson was removed from the roadside in South Armagh late this afternoon after a day-long security operation.

TARGET: Constable Louis Robinson | UPDATED: Constable Louis Robinson

SOURCE_STATE: body removed from roadside | NEW_STATE: officially deceased

UPDATE: Following the recovery operation, Constable Louis Robinson was confirmed as officially deceased.

USE: His family received news that Constable Louis Robinson is officially deceased.

SURFACE use-source: J=0.167, overlap_min=0.375, LCS=3 `constable loui robinson`; use-update LCS=3 `constable loui robinson`

STATE_TERMS source=['body', 'remov', 'roadside'] new=['officially', 'deceas'] hits_update=['officially', 'deceas'] hits_use_new=['officially', 'deceas'] hits_use_source=[]

### 41. rw2s0_005939

SOURCE: Major Kascambo, with a guard of fifty Cossacks, was going to take the command of the Russian outpost of Lars, one of the forts by which the Russian Czars have slowly been carrying on the aggressive warfare that has nearly absorbed into their vast dominions all the mountains between the Caspian and Black seas.

TARGET: Lars | UPDATED: Lars

SOURCE_STATE: Russian outpost | NEW_STATE: abandoned mining site

UPDATE: After the conflict ended, Lars became a quiet abandoned mining site.

USE: Geologists study the ruins of Lars as an old mining site.

SURFACE use-source: J=0.033, overlap_min=0.200, LCS=2 `of lars`; use-update LCS=2 `mining site`

STATE_TERMS source=['russian', 'outpost'] new=['abandon', 'mining'] hits_update=['abandon', 'mining'] hits_use_new=['mining'] hits_use_source=[]

### 42. rw_007952

SOURCE: A woman drove a van into the garage at Cooksdown in County Tyrone and asked him to fit a new exhaust.

TARGET: Cooksdown | UPDATED: Cooksdown

SOURCE_STATE: garage location in County Tyrone | NEW_STATE: renovated parking facility

UPDATE: The local council transformed the old garage site at Cooksdown into a renovated parking facility.

USE: Drivers now find ample space in the Cooksdown parking facility.

SURFACE use-source: J=0.062, overlap_min=0.143, LCS=1 `the`; use-update LCS=2 `park facility`

STATE_TERMS source=['garage', 'county', 'tyrone'] new=['renovat', 'park', 'facility'] hits_update=['renovat', 'park', 'facility'] hits_use_new=['park', 'facility'] hits_use_source=[]

### 43. rw2s1_015184

SOURCE: Driven along, so as to rest upon the bank, she found them at last, and, by the help of Lucina, she laid them in the grave in the cemetery called Ad Ursum Pileatum.

TARGET: Ad Ursum Pileatum | UPDATED: Ad Ursum Pileatum

SOURCE_STATE: cemetery grave | NEW_STATE: ancient Roman burial site

UPDATE: Historical records identify this location as the ancient Roman burial site known as Ad Ursum Pileatum.

USE: Ad Ursum Pileatum stands as a notable ancient Roman burial site today.

SURFACE use-source: J=0.100, overlap_min=0.250, LCS=3 `ad ursum pileatum`; use-update LCS=4 `ancient roman burial site`

STATE_TERMS source=['cemetery', 'grave'] new=['ancient', 'roman', 'burial'] hits_update=['ancient', 'roman', 'burial'] hits_use_new=['ancient', 'roman', 'burial'] hits_use_source=[]

### 44. rw2s0_007553

SOURCE: Ten European sculptors worked on a series of sculptures that are placed in the public area in Woerden ("Van Kempensingel").

TARGET: Woerden Van Kempensingel | UPDATED: Woerden Van Kempensingel

SOURCE_STATE: public area in Woerden | NEW_STATE: renamed Van Kempensingel Square

UPDATE: The city council officially renamed the public area in Woerden to Van Kempensingel Square.

USE: Visitors now gather at the newly named Van Kempensingel Square for events.

SURFACE use-source: J=0.118, overlap_min=0.250, LCS=2 `van kempensingel`; use-update LCS=3 `van kempensingel square`

STATE_TERMS source=['public', 'woerden'] new=['renam', 'van', 'kempensingel', 'square'] hits_update=['renam', 'van', 'kempensingel', 'square'] hits_use_new=['van', 'kempensingel', 'square'] hits_use_source=[]

### 45. rw2s0_010327

SOURCE: This non-profit local organization provides classes, slide presentations, community activities, a newsletter, its own website and its own museum.

TARGET: museum | UPDATED: museum

SOURCE_STATE: non-profit local organization | NEW_STATE: interactive digital exhibit

UPDATE: The non-profit local organization transformed its museum into an interactive digital exhibit.

USE: Visitors now explore the interactive digital exhibit at the museum.

SURFACE use-source: J=0.056, overlap_min=0.167, LCS=1 `museum`; use-update LCS=3 `interactive digital exhibit`

STATE_TERMS source=['non', 'profit', 'organization'] new=['interactive', 'digital', 'exhibit'] hits_update=['interactive', 'digital', 'exhibit'] hits_use_new=['interactive', 'digital', 'exhibit'] hits_use_source=[]

### 46. rw2s0_027716

SOURCE: No appetite, however, had I; and very cruel I thought Aunt Maria for insisting that I should swallow a certain amount of food, as a condition of being allowed to go at all.

TARGET: Aunt Maria | UPDATED: Aunt Maria

SOURCE_STATE: insisting on food | NEW_STATE: generous hostess

UPDATE: Aunt Maria invited everyone to a generous dinner without any pressure to eat more.

USE: Aunt Maria served delicious meals as the generous hostess of the evening.

SURFACE use-source: J=0.111, overlap_min=0.250, LCS=2 `aunt maria`; use-update LCS=2 `aunt maria`

STATE_TERMS source=['insist', 'food'] new=['generou', 'hostess'] hits_update=['generou'] hits_use_new=['generou', 'hostess'] hits_use_source=[]

### 47. rw2s0_014460

SOURCE: On the morning of August 7th, the North Koreans were able to take over Cloverleaf Hill and Oblong-ni Ridge.

TARGET: North Koreans | UPDATED: North Koreans

SOURCE_STATE: taking over Cloverleaf Hill and Oblong-ni Ridge | NEW_STATE: fortified the southern perimeter

UPDATE: Following the morning assault, the North Koreans fortified the southern perimeter to secure their gains.

USE: The North Koreans now defend the southern perimeter with new trenches.

SURFACE use-source: J=0.133, overlap_min=0.333, LCS=3 `the north korean`; use-update LCS=3 `the north korean`

STATE_TERMS source=['taking', 'cloverleaf', 'hill', 'oblong', 'ridge'] new=['fortify', 'southern', 'perimeter'] hits_update=['fortify', 'southern', 'perimeter'] hits_use_new=['southern', 'perimeter'] hits_use_source=[]

### 48. rw2s1_020531

SOURCE: He was a gay man, who could dance and jest, and sing and drink; and those were the roads to so much, or rather so little, of a heart as King Henry had.

TARGET: King Henry | UPDATED: King Henry

SOURCE_STATE: had so little of a heart | NEW_STATE: possessed a generous spirit

UPDATE: King Henry eventually possessed a generous spirit that changed his reign.

USE: The people admired King Henry for his generous spirit.

SURFACE use-source: J=0.111, overlap_min=0.333, LCS=2 `king henry`; use-update LCS=2 `king henry`

STATE_TERMS source=['little', 'heart'] new=['possess', 'generou', 'spirit'] hits_update=['possess', 'generou', 'spirit'] hits_use_new=['generou', 'spirit'] hits_use_source=[]

### 49. rw2s1_028700

SOURCE: It never occurred to him that it might be painful to Sir Edward to visit his old home under such changed conditions.

TARGET: Sir Edward | UPDATED: Sir Edward

SOURCE_STATE: visiting old home under changed conditions | NEW_STATE: residing in a modern mansion

UPDATE: Sir Edward now resides in a modern mansion after selling his ancestral estate.

USE: Visitors admire Sir Edward's residence in the modern mansion today.

SURFACE use-source: J=0.154, overlap_min=0.286, LCS=2 `sir edward`; use-update LCS=2 `sir edward`

STATE_TERMS source=['visit', 'chang'] new=['resid', 'modern', 'mansion'] hits_update=['resid', 'modern', 'mansion'] hits_use_new=['modern', 'mansion'] hits_use_source=[]

### 50. rw_003670

SOURCE: Well, if Drill's looking for a new friend, - that means he needs access to someone.

TARGET: Drill | UPDATED: Drill

SOURCE_STATE: looking for a new friend | NEW_STATE: granted access to a partner

UPDATE: Drill has been granted access to a partner since he sought a new friend.

USE: Drill now collaborates with his newly accessed partner.

SURFACE use-source: J=0.182, overlap_min=0.400, LCS=1 `drill`; use-update LCS=1 `drill`

STATE_TERMS source=['look', 'friend'] new=['grant', 'access', 'partner'] hits_update=['grant', 'access', 'partner'] hits_use_new=['access', 'partner'] hits_use_source=[]

## UNCHANGED_DISTRACTOR_USE

### 1. rw2s1_023866

SOURCE: To the northeast of the ice cap, it falls to between 350 and 450 mm per year, which is the lowest rainfall in Iceland.

TARGET: Iceland | UPDATED: the ice cap

SOURCE_STATE: receives lowest rainfall in area | NEW_STATE: melts rapidly into water

UPDATE: The ice cap is now melting rapidly into water instead of holding its form.

USE: Iceland continues to receive the lowest rainfall in the entire area.

SURFACE use-source: J=0.250, overlap_min=0.500, LCS=4 `the lowest rainfall in`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'lowest', 'rainfall'] new=['melt', 'rapidly', 'water'] hits_update=['melt', 'rapidly', 'water'] hits_use_new=[] hits_use_source=['lowest', 'rainfall']

### 2. rw2s0_020532

SOURCE: Tsvetaeva left Russia in 1922 and lived with her family in increasing poverty in Paris, Berlin and Prague before returning to Moscow in 1939.

TARGET: Russia | UPDATED: Moscow

SOURCE_STATE: left in 1922 | NEW_STATE: abandoned until 1945

UPDATE: Moscow remained largely abandoned until 1945 after the initial departure.

USE: Travelers still recall Russia being left behind in 1922.

SURFACE use-source: J=0.188, overlap_min=0.500, LCS=2 `in 1922`; use-update LCS=0 ``

STATE_TERMS source=['left', '1922'] new=['abandon', 'until', '1945'] hits_update=['abandon', 'until', '1945'] hits_use_new=[] hits_use_source=['left', '1922']

### 3. rw2s0_015667

SOURCE: The Millard Sheets Art Center first began as the Fine Arts Program of the Los Angeles County Fair in 1922.

TARGET: Millard Sheets Art Center | UPDATED: Los Angeles County Fair

SOURCE_STATE: originated from fine arts program | NEW_STATE: annual agricultural exhibition

UPDATE: The Los Angeles County Fair transformed into an annual agricultural exhibition.

USE: The Millard Sheets Art Center emerged from its fine arts program roots.

SURFACE use-source: J=0.438, overlap_min=0.778, LCS=5 `the millard sheet art center`; use-update LCS=1 `the`

STATE_TERMS source=['originat', 'fine', 'arts', 'program'] new=['annual', 'agricultural', 'exhibition'] hits_update=['annual', 'agricultural', 'exhibition'] hits_use_new=[] hits_use_source=['fine', 'arts', 'program']

### 4. rw_023527

SOURCE: The treaty happened shortly after the last battle in the French and Indian War in North America.

TARGET: North America | UPDATED: the treaty

SOURCE_STATE: the scene of the French and Indian War | NEW_STATE: signed before the last battle

UPDATE: The treaty was signed before the last battle in North America.

USE: The French and Indian War remains the conflict that shaped North America.

SURFACE use-source: J=0.417, overlap_min=0.714, LCS=5 `the french and indian war`; use-update LCS=2 `north america`

STATE_TERMS source=['scene', 'french', 'indian', 'war'] new=['sign', 'last', 'battle'] hits_update=['sign', 'last', 'battle'] hits_use_new=[] hits_use_source=['french', 'indian', 'war']

### 5. rw_033131

SOURCE: This is the gift to the Ambassador Jung Myung So When the sun rises, please wrap it up for me Yes, My Lord Have the doors over here been held in place?

TARGET: Ambassador Jung Myung | UPDATED: the doors

SOURCE_STATE: receives a sun-risen gift | NEW_STATE: left unheld

UPDATE: The doors over here have been left unheld instead of being kept in place.

USE: Ambassador Jung Myung still receives the gift after the sun rises.

SURFACE use-source: J=0.429, overlap_min=0.857, LCS=3 `ambassador jung myung`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'sun', 'risen', 'gift'] new=['left', 'unheld'] hits_update=['left', 'unheld'] hits_use_new=[] hits_use_source=['receiv', 'sun', 'gift']

### 6. rw2s1_021086

SOURCE: After moving to Monticello, Elizabeth Hemings also had at least one child with Joseph Neilson, a white carpenter who worked for Thomas Jefferson.

TARGET: Monticello | UPDATED: Joseph Neilson

SOURCE_STATE: the residence of Thomas Jefferson | NEW_STATE: a blacksmith in Philadelphia

UPDATE: Joseph Neilson left his carpentry work to become a blacksmith in Philadelphia instead.

USE: Monticello remains the historic residence where Thomas Jefferson lived.

SURFACE use-source: J=0.200, overlap_min=0.600, LCS=2 `thoma jefferson`; use-update LCS=0 ``

STATE_TERMS source=['residence', 'thoma', 'jefferson'] new=['blacksmith', 'philadelphia'] hits_update=['blacksmith', 'philadelphia'] hits_use_new=[] hits_use_source=['residence', 'thoma', 'jefferson']

### 7. rw2s0_016363

SOURCE: The word echo come from Greek word ��� ("ēchō) from (ēchos), "sound"." In nature echo happens in empty spaces, woods or mountains.

TARGET: Greek | UPDATED: Latin

SOURCE_STATE: origin of the word echo | NEW_STATE: etymology of the concept silence

UPDATE: Latin scholars now study the etymology of the concept silence.

USE: The original word echo traces back to Greek roots today.

SURFACE use-source: J=0.188, overlap_min=0.429, LCS=2 `word echo`; use-update LCS=1 `the`

STATE_TERMS source=['origin', 'word', 'echo'] new=['etymology', 'concept', 'silence'] hits_update=['etymology', 'concept', 'silence'] hits_use_new=[] hits_use_source=['word', 'echo']

### 8. rw2s1_026265

SOURCE: In fact, so much water is currently stored in Vatnajökull that Ölfusá, Iceland's river with the largest volume of water, would take over 200 years to transport this amount of water into the sea.

TARGET: Vatnajökull | UPDATED: the river

SOURCE_STATE: stores a vast amount of water | NEW_STATE: flows as a dry channel

UPDATE: The river now flows as a dry channel instead of carrying water.

USE: Vatnajökull still holds its vast amount of water while the river dries up.

SURFACE use-source: J=0.278, overlap_min=0.625, LCS=3 `amount of water`; use-update LCS=2 `the river`

STATE_TERMS source=['stor', 'vast', 'amount', 'water'] new=['flow', 'dry', 'channel'] hits_update=['flow', 'dry', 'channel'] hits_use_new=[] hits_use_source=['vast', 'amount', 'water']

### 9. rw2s0_021369

SOURCE: It was covered by Andy Williams as his debut single and went to number 54 in the United States.

TARGET: Andy Williams | UPDATED: United States

SOURCE_STATE: singing his debut single | NEW_STATE: topping the charts in the UK

UPDATE: The song instead topped the charts in the UK rather than staying in the states.

USE: Andy Williams remains famous for singing that debut single.

SURFACE use-source: J=0.364, overlap_min=0.667, LCS=2 `andy william`; use-update LCS=0 ``

STATE_TERMS source=['sing', 'debut', 'single'] new=['topp', 'chart'] hits_update=['topp', 'chart'] hits_use_new=[] hits_use_source=['sing', 'debut', 'single']

### 10. rw_010746

SOURCE: The popular citrus soda Mountain Dew traces its origins to Johnson City.

TARGET: Mountain Dew | UPDATED: Johnson City

SOURCE_STATE: traces its origins to Johnson City | NEW_STATE: a quiet suburb in Ohio

UPDATE: Johnson City was redeveloped into a quiet suburb in Ohio.

USE: Mountain Dew still traces its origins to the original location.

SURFACE use-source: J=0.444, overlap_min=1.000, LCS=4 `trac its origin to`; use-update LCS=0 ``

STATE_TERMS source=['trac', 'origin', 'johnson', 'city'] new=['quiet', 'suburb', 'ohio'] hits_update=['quiet', 'suburb', 'ohio'] hits_use_new=[] hits_use_source=['trac', 'origin']

### 11. rw2s0_006680

SOURCE: A witness said a man who looked like Rodger began the fight by trying to push two women off a ledge.

TARGET: Rodger | UPDATED: the witness

SOURCE_STATE: looked like the man who pushed the women | NEW_STATE: hired by the police department

UPDATE: The witness quit his job to get hired by the police department.

USE: Rodger is still the man who looked like the one who pushed the women.

SURFACE use-source: J=0.429, overlap_min=1.000, LCS=4 `man who look like`; use-update LCS=1 `the`

STATE_TERMS source=['look', 'like', 'man', 'push', 'women'] new=['hired', 'police', 'department'] hits_update=['hired', 'police', 'department'] hits_use_new=[] hits_use_source=['look', 'like', 'man', 'push', 'women']

### 12. rw2s1_022371

SOURCE: She also played Donna Troy / Wonder Girl on the DC Universe / HBO Max superhero series "Titans" (2018–present).

TARGET: Donna Troy | UPDATED: HBO Max

SOURCE_STATE: played in the superhero series Titans | NEW_STATE: streaming platform for animated comedies

UPDATE: HBO Max shifted its lineup to focus on streaming platform for animated comedies.

USE: Audiences continue to watch Donna Troy perform in the superhero series Titans.

SURFACE use-source: J=0.312, overlap_min=0.625, LCS=3 `superhero seri titan`; use-update LCS=1 `to`

STATE_TERMS source=['play', 'superhero', 'seri', 'titan'] new=['stream', 'platform', 'animat', 'comedi'] hits_update=['stream', 'platform', 'animat', 'comedi'] hits_use_new=[] hits_use_source=['superhero', 'seri', 'titan']

### 13. rw2s1_021004

SOURCE: Citta Notabile and Gozo were inland, and their fate would depend upon that of the defenses of the harbor.

TARGET: Citta Notabile | UPDATED: Gozo

SOURCE_STATE: inland and dependent on harbor defenses | NEW_STATE: coastal and independent of harbor defenses

UPDATE: Gozo moved to the coast and now stands independent of harbor defenses.

USE: Citta Notabile remains inland and still depends on harbor defenses.

SURFACE use-source: J=0.667, overlap_min=1.000, LCS=2 `citta notabile`; use-update LCS=2 `harbor defens`

STATE_TERMS source=['inland', 'dependent', 'harbor', 'defens'] new=['coastal', 'independent', 'harbor', 'defens'] hits_update=['independent', 'harbor', 'defens'] hits_use_new=[] hits_use_source=['inland', 'harbor', 'defens']

### 14. rw2s0_000727

SOURCE: Quickly he went to where the mistletoe grew, cut a slender green branch, shaped it into a point, and sought the blind god Hodur.

TARGET: Hodur | UPDATED: the slender green branch

SOURCE_STATE: the blind god sought by him | NEW_STATE: a red twig

UPDATE: He found a red twig instead of the green branch.

USE: Hodur remains a blind god who was searched for.

SURFACE use-source: J=0.200, overlap_min=0.750, LCS=2 `blind god`; use-update LCS=1 `a`

STATE_TERMS source=['blind', 'god', 'sought'] new=['red', 'twig'] hits_update=['red', 'twig'] hits_use_new=[] hits_use_source=['blind', 'god']

### 15. rw2s1_009372

SOURCE: Each night Jenny had new and famous names to add to the list in her journal, and the artless pages were rich in anecdotes, descriptions, and comments on the day's adventures.

TARGET: Jenny | UPDATED: the artless pages

SOURCE_STATE: added famous names to journal | NEW_STATE: filled with travel maps

UPDATE: The artless pages were now filled with detailed travel maps instead.

USE: Jenny continued adding famous names to her journal list as usual.

SURFACE use-source: J=0.294, overlap_min=0.625, LCS=3 `famou name to`; use-update LCS=0 ``

STATE_TERMS source=['added', 'famou', 'name', 'journal'] new=['fill', 'travel', 'maps'] hits_update=['fill', 'travel', 'maps'] hits_use_new=[] hits_use_source=['famou', 'name', 'journal']

### 16. rw_000237

SOURCE: In the 1980s, he created a brad of cigars called "Avo Cigars".

TARGET: Avo Cigars | UPDATED: the cigar brand

SOURCE_STATE: created in the 1980s by him | NEW_STATE: manufactured in the 1990s by her

UPDATE: The cigar brand was manufactured in the 1990s by her instead of the 1980s.

USE: He still remembers the 1980s creation of his cigar brand.

SURFACE use-source: J=0.222, overlap_min=0.400, LCS=2 `the 1980`; use-update LCS=2 `cigar brand`

STATE_TERMS source=['creat', '1980'] new=['manufactur', '1990'] hits_update=['manufactur', '1990'] hits_use_new=[] hits_use_source=['1980']

### 17. rw_022468

SOURCE: Mills plays the character Pam Douglas in "The Bold And The Beautiful".

TARGET: Pam Douglas | UPDATED: the scenic location

SOURCE_STATE: plays character in Bold and Beautiful show | NEW_STATE: dark abandoned warehouse lot

UPDATE: The scenic location transformed into a dark abandoned warehouse lot for filming.

USE: Pam Douglas remains acting out her role in the Bold and Beautiful drama.

SURFACE use-source: J=0.400, overlap_min=0.571, LCS=4 `in the bold and`; use-update LCS=1 `the`

STATE_TERMS source=['play', 'character', 'bold', 'beautiful', 'show'] new=['dark', 'abandon', 'warehouse', 'lot'] hits_update=['dark', 'abandon', 'warehouse', 'lot'] hits_use_new=[] hits_use_source=['bold', 'beautiful']

### 18. rw2s1_007989

SOURCE: On April 15, 2013, The MOGA Pro presented a new design with an added D-pad and four shoulder buttons.

TARGET: The MOGA Pro | UPDATED: four shoulder buttons

SOURCE_STATE: presented new design with added D-pad | NEW_STATE: replaced by four touch triggers

UPDATE: The device now uses four touch triggers instead of shoulder buttons.

USE: The MOGA Pro still includes the new design with added D-pad.

SURFACE use-source: J=0.417, overlap_min=0.833, LCS=3 `the moga pro`; use-update LCS=1 `the`

STATE_TERMS source=['present', 'design', 'added', 'pad'] new=['replac', 'four', 'touch', 'trigger'] hits_update=['four', 'touch', 'trigger'] hits_use_new=[] hits_use_source=['design', 'added', 'pad']

### 19. rw_010477

SOURCE: Dear me, Flo, I have been ill since you saw me last.

TARGET: Flo | UPDATED: the doctor

SOURCE_STATE: has been ill since last seen | NEW_STATE: retired from practice

UPDATE: The doctor retired from medical practice today.

USE: Flo continues feeling ill since she was last seen.

SURFACE use-source: J=0.444, overlap_min=0.667, LCS=2 `ill since`; use-update LCS=0 ``

STATE_TERMS source=['ill', 'since', 'last', 'seen'] new=['retir', 'practice'] hits_update=['retir', 'practice'] hits_use_new=[] hits_use_source=['ill', 'since', 'last', 'seen']

### 20. rw2s0_006067

SOURCE: Van Sambeek started his own design agency with Shigeru Watano and worked on many different assignments, including designing: advertising campaigns, corporate identity, books, magazines and newspapers.

TARGET: Sambeek | UPDATED: Shigeru Watano

SOURCE_STATE: started his own design agency | NEW_STATE: ran a local restaurant

UPDATE: Shigeru Watano left the design field to run a local restaurant.

USE: Sambeek still operates his design agency with various partners.

SURFACE use-source: J=0.136, overlap_min=0.500, LCS=3 `design agency with`; use-update LCS=1 `design`

STATE_TERMS source=['start', 'own', 'design', 'agency'] new=['ran', 'restaurant'] hits_update=['restaurant'] hits_use_new=[] hits_use_source=['design', 'agency']

### 21. rw2s1_006319

SOURCE: "Thou art ready with thy excuses for thy dog." "He is all I have, Humphrey," returned Hugo, quietly.

TARGET: Hugo | UPDATED: Humphrey

SOURCE_STATE: returned Hugo quietly | NEW_STATE: silent observation

UPDATE: Humphrey shifted to silent observation while Hugo spoke.

USE: Hugo remains the one who returned quietly now.

SURFACE use-source: J=0.300, overlap_min=1.000, LCS=1 `return`; use-update LCS=1 `hugo`

STATE_TERMS source=['return', 'hugo', 'quietly'] new=['silent', 'observation'] hits_update=['silent', 'observation'] hits_use_new=[] hits_use_source=['return', 'hugo', 'quietly']

### 22. rw2s1_013474

SOURCE: She was in prison for four years, from 1987 to 1991, because she opposed the Bashar al-Assad government in Syria.

TARGET: Syria | UPDATED: Bashar al-Assad

SOURCE_STATE: opposed by the government of Bashar al-Assad | NEW_STATE: led a democratic reform movement

UPDATE: Bashar al-Assad stepped down to lead a democratic reform movement instead.

USE: Syria remains a nation opposed by the former government of Bashar al-Assad.

SURFACE use-source: J=0.455, overlap_min=0.833, LCS=3 `bashar al assad`; use-update LCS=3 `bashar al assad`

STATE_TERMS source=['oppos', 'government', 'bashar', 'assad'] new=['led', 'democratic', 'reform', 'movement'] hits_update=['democratic', 'reform', 'movement'] hits_use_new=[] hits_use_source=['oppos', 'government', 'bashar', 'assad']

### 23. rw2s0_032354

SOURCE: "Would you call him the best man in Ciscasset?" pursued the farmer, with a wave of his hand toward 'Tilda Jane.

TARGET: Tilda Jane | UPDATED: Ciscasset

SOURCE_STATE: the farmer waved toward her | NEW_STATE: a bustling coastal town

UPDATE: The town of Ciscasset has become a bustling coastal town.

USE: The farmer still waves toward Tilda Jane who remains there.

SURFACE use-source: J=0.455, overlap_min=1.000, LCS=3 `toward tilda jane`; use-update LCS=1 `the`

STATE_TERMS source=['farmer', 'waved', 'toward'] new=['bustl', 'coastal', 'town'] hits_update=['bustl', 'coastal', 'town'] hits_use_new=[] hits_use_source=['farmer', 'toward']

### 24. rw2s0_028853

SOURCE: Afterward one Captain Wiggins visited this country, where he found flax and wheat, highly magnetic iron ore, and rich mines of copper and gold.

TARGET: Captain Wiggins | UPDATED: flax

SOURCE_STATE: visited the country with magnetic ore | NEW_STATE: rare desert cactus

UPDATE: Flax was replaced by rare desert cactus during the visit.

USE: Captain Wiggins still roamed the land with magnetic ore.

SURFACE use-source: J=0.222, overlap_min=0.667, LCS=2 `captain wiggin`; use-update LCS=1 `the`

STATE_TERMS source=['visit', 'country', 'magnetic', 'ore'] new=['rare', 'desert', 'cactu'] hits_update=['rare', 'desert', 'cactu'] hits_use_new=[] hits_use_source=['magnetic', 'ore']

### 25. rw2s0_019909

SOURCE: Assyria reached the height of her power during the reign of Sennacherib and Assur-bani-pal, and everything in Nineveh was so lovely for the Ninevites that the time when Assur-bani-pal reigned was called the Golden Age.

TARGET: Nineveh | UPDATED: Assur-bani-pal

SOURCE_STATE: lovely for the Ninevites | NEW_STATE: ruined kingdom

UPDATE: Assur-bani-pal ended his golden reign and left Nineveh as a ruined kingdom.

USE: Nineveh remains lovely for the people despite the new ruler.

SURFACE use-source: J=0.095, overlap_min=0.400, LCS=3 `lovely for the`; use-update LCS=1 `nineveh`

STATE_TERMS source=['lovely', 'ninevit'] new=['ruin', 'kingdom'] hits_update=['ruin', 'kingdom'] hits_use_new=[] hits_use_source=['lovely']

### 26. rw2s0_012133

SOURCE: Orders here came from Rome that Manlius should return thither, but that Regulus should remain to carry on the war.

TARGET: Regulus | UPDATED: Manlius

SOURCE_STATE: remain to carry on the war | NEW_STATE: farming in the valley

UPDATE: Manlius abandoned the campaign to begin farming in the valley nearby.

USE: Regulus continues his duty to carry on the war against Rome.

SURFACE use-source: J=0.308, overlap_min=0.571, LCS=5 `to carry on the war`; use-update LCS=1 `the`

STATE_TERMS source=['carry', 'war'] new=['farm', 'valley'] hits_update=['farm', 'valley'] hits_use_new=[] hits_use_source=['carry', 'war']

### 27. rw2s1_027590

SOURCE: The match was very well liked by fans and critics, with wrestling journalist Dave Meltzer giving the match a 5 1⁄2 star rating.

TARGET: Dave Meltzer | UPDATED: the fans

SOURCE_STATE: gave the match a 5 1⁄2 star rating | NEW_STATE: loved the chaotic free-for-all style

UPDATE: The fans cheered loudly for the chaotic free-for-all style instead.

USE: Dave Meltzer maintained his high rating for the match.

SURFACE use-source: J=0.286, overlap_min=0.667, LCS=2 `the match`; use-update LCS=2 `for the`

STATE_TERMS source=['gave', 'match', 'star', 'rating'] new=['loved', 'chaotic', 'free', 'style'] hits_update=['chaotic', 'free', 'style'] hits_use_new=[] hits_use_source=['match', 'rating']

### 28. rw2s1_001613

SOURCE: An example of its importance may best be seen in that companies such as Motorola and Durman Esquivel remained located in the district of Calle Blancos along with Coca Cola-FEMSA bottling corporation for decades.

TARGET: Motorola | UPDATED: Coca Cola-FEMSA

SOURCE_STATE: remained located in the district of Calle Blancos | NEW_STATE: moved to the industrial park north of the city

UPDATE: Coca Cola-FEMSA relocated its operations to the industrial park north of the city while other firms stayed.

USE: Motorola continues its long presence within the district of Calle Blancos.

SURFACE use-source: J=0.182, overlap_min=0.500, LCS=5 `the district of calle blanco`; use-update LCS=1 `its`

STATE_TERMS source=['locat', 'district', 'calle', 'blanco'] new=['moved', 'industrial', 'park', 'north', 'city'] hits_update=['industrial', 'park', 'north', 'city'] hits_use_new=[] hits_use_source=['district', 'calle', 'blanco']

### 29. rw_020645

SOURCE: Many a time the old Sword had pierced men's hearts, and then their hot blood flowed along his blade.

TARGET: the old Sword | UPDATED: the silent floor

SOURCE_STATE: pierced men's hearts and flowed with hot blood | NEW_STATE: covered in cold dust

UPDATE: The silent floor now lies covered in cold dust.

USE: The old Sword still holds the memory of pierced hearts and flowing blood.

SURFACE use-source: J=0.417, overlap_min=0.714, LCS=3 `the old sword`; use-update LCS=1 `the`

STATE_TERMS source=['pierc', 'men', 'heart', 'flow', 'hot', 'blood'] new=['cover', 'cold', 'dust'] hits_update=['cover', 'cold', 'dust'] hits_use_new=[] hits_use_source=['pierc', 'heart', 'flow', 'blood']

### 30. rw_012360

SOURCE: Many times the great bell had rung in the little town of Atri, and, as the king had said, the wrongs of which it told, were always righted.

TARGET: Atri | UPDATED: the king

SOURCE_STATE: little town where the great bell rang | NEW_STATE: lost his crown forever

UPDATE: The king lost his crown and lived in exile without power.

USE: People remember Atri as the little town where the great bell once rang.

SURFACE use-source: J=0.278, overlap_min=0.556, LCS=3 `the great bell`; use-update LCS=1 `the`

STATE_TERMS source=['little', 'town', 'great', 'bell', 'rang'] new=['lost', 'crown', 'forever'] hits_update=['lost', 'crown'] hits_use_new=[] hits_use_source=['little', 'town', 'great', 'bell', 'rang']

### 31. rw_028732

SOURCE: The group of fictional characters consists of Twilight Sparkle, Applejack, Rainbow Dash, Pinkie Pie, Fluttershy, and Rarity, all of whom are ponies and best friends from the fictional country of Equestria.

TARGET: Equestria | UPDATED: Twilight Sparkle

SOURCE_STATE: a fictional country for ponies | NEW_STATE: a real human journalist

UPDATE: Twilight Sparkle quit her pony role to work as a human journalist in New York.

USE: The fictional country of Equestria still remains home to the beloved ponies.

SURFACE use-source: J=0.200, overlap_min=0.800, LCS=5 `the fictional country of equestria`; use-update LCS=1 `to`

STATE_TERMS source=['fictional', 'country', 'poni'] new=['real', 'human', 'journalist'] hits_update=['human', 'journalist'] hits_use_new=[] hits_use_source=['fictional', 'country', 'poni']

### 32. rw2s0_008873

SOURCE: Navigating through several streets, Rodger engaged in a harrowing exchange of gunfire with a sheriff's deputy, injuring two pedestrians in the process.

TARGET: Rodger | UPDATED: two pedestrians

SOURCE_STATE: engaged in a harrowing exchange of gunfire | NEW_STATE: safe in a shelter

UPDATE: While Rodger fought, two pedestrians found safety inside a nearby shelter.

USE: Rodger remained involved in the intense gunfire exchange as planned.

SURFACE use-source: J=0.176, overlap_min=0.500, LCS=2 `in the`; use-update LCS=1 `rodger`

STATE_TERMS source=['engag', 'harrow', 'exchange', 'gunfire'] new=['safe', 'shelter'] hits_update=['shelter'] hits_use_new=[] hits_use_source=['exchange', 'gunfire']

### 33. rw2s1_005065

SOURCE: The first rising of the Gauls began by a quarrel at the city we now call Orleans, ending in a massacre of all the Romans there.

TARGET: the Gauls | UPDATED: Orleans

SOURCE_STATE: began by a quarrel at the city | NEW_STATE: modern capital of the Loire valley

UPDATE: Orleans transformed from a quarrelsome border town into the modern capital of the Loire valley.

USE: The Gauls still remember the ancient city where history began.

SURFACE use-source: J=0.214, overlap_min=0.500, LCS=2 `the gaul`; use-update LCS=1 `the`

STATE_TERMS source=['began', 'quarrel', 'city'] new=['modern', 'capital', 'loire', 'valley'] hits_update=['modern', 'capital', 'loire', 'valley'] hits_use_new=[] hits_use_source=['began', 'city']

### 34. rw_007173

SOURCE: "Did I not say it was a day of joy," exclaimed the little Comtesse.

TARGET: the little Comtesse | UPDATED: the grand duke

SOURCE_STATE: exclaimed about a day of joy | NEW_STATE: silent in the cold palace

UPDATE: The grand duke sat silently inside the cold palace while others spoke.

USE: The little Comtesse continues to express joy about the day.

SURFACE use-source: J=0.429, overlap_min=0.600, LCS=3 `the little comtesse`; use-update LCS=1 `the`

STATE_TERMS source=['exclaim', 'joy'] new=['silent', 'cold', 'palace'] hits_update=['cold', 'palace'] hits_use_new=[] hits_use_source=['joy']

### 35. rw2s0_029279

SOURCE: FANFARE My government will continue the work of bringing the public finances under control, and reducing the deficit, so Britain lives within its means.

TARGET: Britain | UPDATED: the public finances

SOURCE_STATE: reducing the deficit to live within means | NEW_STATE: growing rapidly without any constraints

UPDATE: The government shifted focus so the public finances are now growing rapidly.

USE: Britain continues its mission to reduce deficits and live within means.

SURFACE use-source: J=0.267, overlap_min=0.571, LCS=2 `live within`; use-update LCS=0 ``

STATE_TERMS source=['reduc', 'deficit', 'within', 'mean'] new=['grow', 'rapidly', 'without', 'constraint'] hits_update=['grow', 'rapidly'] hits_use_new=[] hits_use_source=['deficit', 'within', 'mean']

### 36. rw2s0_003241

SOURCE: So Ted trotted into the back kitchen, and to prevent cook's thinking there was anything the matter asked her if he might play with the cat.

TARGET: Ted | UPDATED: the cat

SOURCE_STATE: trotted into the back kitchen | NEW_STATE: sleeping in a warm sunbeam

UPDATE: The cat curled up on the rug to sleep in a warm sunbeam.

USE: Ted remains trotting inside the kitchen after his visit.

SURFACE use-source: J=0.214, overlap_min=0.600, LCS=1 `ted`; use-update LCS=1 `the`

STATE_TERMS source=['trott', 'back', 'kitchen'] new=['sleep', 'warm', 'sunbeam'] hits_update=['sleep', 'warm', 'sunbeam'] hits_use_new=[] hits_use_source=['trott', 'kitchen']

### 37. rw2s1_028801

SOURCE: He died the next year without children; his titles went to no one, and his property went to the son of his sister, Charles Sackville, 6th Earl of Dorset.

TARGET: Charles Sackville | UPDATED: his property

SOURCE_STATE: became the sixth Earl of Dorset | NEW_STATE: passed to his nephew's family

UPDATE: His property went to the son of his sister's family instead.

USE: Charles Sackville remained the sixth Earl of Dorset despite the change.

SURFACE use-source: J=0.235, overlap_min=0.571, LCS=3 `earl of dorset`; use-update LCS=1 `the`

STATE_TERMS source=['sixth', 'earl', 'dorset'] new=['pass', 'nephew', 'family'] hits_update=['family'] hits_use_new=[] hits_use_source=['sixth', 'earl', 'dorset']

### 38. rw2s0_008828

SOURCE: Strebe and the scientists said "The project can also be interpreted as a statement against British artist Anish Kapoor's purchase of" Vantablack.

TARGET: Vantablack | UPDATED: Anish Kapoor's

SOURCE_STATE: a statement against British artist Anish Kapoor's purchase of | NEW_STATE: gallery exhibition title

UPDATE: Anish Kapoor's work is now known as the permanent gallery exhibition title.

USE: Vantablack remains a statement against the British purchase of the artwork.

SURFACE use-source: J=0.357, overlap_min=0.833, LCS=3 `a statement against`; use-update LCS=1 `the`

STATE_TERMS source=['statement', 'against', 'british', 'artist', 'anish', 'kapoor', 'purchase'] new=['gallery', 'exhibition', 'title'] hits_update=['gallery', 'exhibition', 'title'] hits_use_new=[] hits_use_source=['statement', 'against', 'british', 'purchase']

### 39. rw_039363

SOURCE: As a matter of fact, we brought a decorator in from Chicago just last year to refurbish this part of the hotel.

TARGET: the hotel | UPDATED: the lobby

SOURCE_STATE: being refurbished by a decorator from Chicago | NEW_STATE: filled with vintage furniture

UPDATE: The lobby now displays vintage furniture instead of renovation tools.

USE: Guests still admire the Chicago decorator's original work at the hotel.

SURFACE use-source: J=0.273, overlap_min=0.500, LCS=2 `the hotel`; use-update LCS=1 `the`

STATE_TERMS source=['refurbish', 'decorator', 'chicago'] new=['fill', 'vintage', 'furniture'] hits_update=['vintage', 'furniture'] hits_use_new=[] hits_use_source=['decorator', 'chicago']

### 40. rw2s1_006353

SOURCE: Upon the Eastern Question he wielded a Varangian axe, and had his advice been heeded, the Commander of the Faithful would ere now have been sent back to Brusa, or beyond.

TARGET: Eastern Question | UPDATED: Brusa

SOURCE_STATE: wielded a Varangian axe | NEW_STATE: modern metropolis

UPDATE: Brusa transformed from a distant province into a modern metropolis today.

USE: The Eastern Question remains a historical topic regarding the wielded axe.

SURFACE use-source: J=0.222, overlap_min=0.571, LCS=3 `the eastern question`; use-update LCS=1 `a`

STATE_TERMS source=['wield', 'varangian', 'axe'] new=['modern', 'metropoli'] hits_update=['modern', 'metropoli'] hits_use_new=[] hits_use_source=['wield', 'axe']

### 41. rw_026330

SOURCE: It is the first album with drummer Dirk Verbeuren, and second album with guitarist Kiko Loureiro.

TARGET: Kiko Loureiro | UPDATED: Dirk Verbeuren

SOURCE_STATE: plays the guitar on the second album | NEW_STATE: switched to the bass guitar role

UPDATE: Dirk Verbeuren replaced his drum kit with a bass guitar for the new track.

USE: Listeners can still hear Kiko Loureiro playing the guitar on that second record.

SURFACE use-source: J=0.214, overlap_min=0.375, LCS=2 `kiko loureiro`; use-update LCS=1 `guitar`

STATE_TERMS source=['play', 'guitar', 'second', 'album'] new=['switch', 'bass', 'guitar'] hits_update=['bass', 'guitar'] hits_use_new=[] hits_use_source=['play', 'guitar', 'second']

### 42. rw2s0_007116

SOURCE: The Guangzhou CTF Finance Centre is situated on a lot along Zhujiang East Road in Zhujiang New Town, Guangzhou's central business district.

TARGET: Guangzhou CTF Finance Centre | UPDATED: Zhujiang New Town

SOURCE_STATE: situated on a lot along Zhujiang East Road | NEW_STATE: modern residential complex

UPDATE: The Zhujiang New Town area was redeveloped into a modern residential complex.

USE: The Guangzhou CTF Finance Centre still sits on its lot along the road.

SURFACE use-source: J=0.462, overlap_min=0.857, LCS=5 `the guangzhou ctf finance centre`; use-update LCS=1 `the`

STATE_TERMS source=['situat', 'lot', 'zhujiang', 'east', 'road'] new=['modern', 'residential', 'complex'] hits_update=['modern', 'residential', 'complex'] hits_use_new=[] hits_use_source=['lot', 'road']

### 43. rw2s0_014408

SOURCE: It will make the most immense difference in your life, my dear Florence, if you gain this Scholarship, and also in the life of your affectionate mother.

TARGET: Florence | UPDATED: your affectionate mother

SOURCE_STATE: will gain the Scholarship | NEW_STATE: receives a promotion

UPDATE: Your affectionate mother received a promotion at work.

USE: Florence keeps studying hard to gain the Scholarship soon.

SURFACE use-source: J=0.200, overlap_min=0.429, LCS=1 `the`; use-update LCS=0 ``

STATE_TERMS source=['gain', 'scholarship'] new=['receiv', 'promotion'] hits_update=['receiv', 'promotion'] hits_use_new=[] hits_use_source=['gain', 'scholarship']

### 44. rw_014663

SOURCE: Peter then takes a glass cross that he had received from the Glass Man.

TARGET: the glass cross | UPDATED: the wooden door

SOURCE_STATE: received from the Glass Man | NEW_STATE: locked tight with iron bolts

UPDATE: The wooden door was locked tight with iron bolts to prevent entry.

USE: Peter still holds the received glass cross he got from the Glass Man.

SURFACE use-source: J=0.625, overlap_min=0.833, LCS=4 `from the glass man`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'glass', 'man'] new=['lock', 'tight', 'iron', 'bolt'] hits_update=['lock', 'tight', 'iron', 'bolt'] hits_use_new=[] hits_use_source=['receiv', 'glass', 'man']

### 45. rw_018973

SOURCE: her big blue eyes filled with tears, and she laid a caressing hand on 'Tilda Jane's shoulder.

TARGET: 'Tilda Jane's shoulder | UPDATED: the big blue eyes

SOURCE_STATE: receives a caressing hand from her | NEW_STATE: filled with laughter instead of tears

UPDATE: The big blue eyes now sparkle with laughter instead of filling with tears.

USE: She keeps her shoulder for a caressing hand from her friend.

SURFACE use-source: J=0.231, overlap_min=0.600, LCS=3 `a caress hand`; use-update LCS=0 ``

STATE_TERMS source=['receiv', 'caress', 'hand'] new=['fill', 'laughter', 'instead', 'tear'] hits_update=['fill', 'laughter', 'instead', 'tear'] hits_use_new=[] hits_use_source=['caress', 'hand']

### 46. rw_013028

SOURCE: companion for you than that coquettish French woman, who will probably leave you in the lurch as soon as you reach Paris.

TARGET: French woman | UPDATED: Paris

SOURCE_STATE: coquettish and will leave you | NEW_STATE: safe haven for all travelers

UPDATE: Paris has become a safe haven for all travelers now.

USE: The coquettish French woman is no longer the only option here.

SURFACE use-source: J=0.231, overlap_min=0.500, LCS=3 `coquettish french woman`; use-update LCS=0 ``

STATE_TERMS source=['coquettish', 'leave'] new=['safe', 'haven', 'traveler'] hits_update=['safe', 'haven', 'traveler'] hits_use_new=[] hits_use_source=['coquettish']

### 47. rw_028655

SOURCE: There is also a large target practice area on the peninsula called the "ANZAC Rifle Range", which is still in use today.

TARGET: ANZAC Rifle Range | UPDATED: the main hall

SOURCE_STATE: large target practice area on the peninsula | NEW_STATE: quiet storage room

UPDATE: The main hall was converted into a quiet storage room for equipment.

USE: Shooters still practice in the large ANZAC Rifle Range on the peninsula.

SURFACE use-source: J=0.545, overlap_min=0.857, LCS=3 `on the peninsula`; use-update LCS=1 `the`

STATE_TERMS source=['large', 'target', 'practice', 'peninsula'] new=['quiet', 'storage'] hits_update=['quiet', 'storage'] hits_use_new=[] hits_use_source=['large', 'practice', 'peninsula']

### 48. rw2s0_019074

SOURCE: In the movie, California Senator Jay Bulworth is losing his bid for re-election to a fiery young populist.

TARGET: California Senator Jay Bulworth | UPDATED: a fiery young populist

SOURCE_STATE: losing his bid for re-election | NEW_STATE: running for mayor

UPDATE: The fiery young populist has shifted focus to run for mayor.

USE: Bulworth remains on the losing side of his re-election bid.

SURFACE use-source: J=0.333, overlap_min=0.800, LCS=2 `re election`; use-update LCS=1 `the`

STATE_TERMS source=['losing', 'bid', 'election'] new=['runn', 'mayor'] hits_update=['mayor'] hits_use_new=[] hits_use_source=['losing', 'bid', 'election']

### 49. rw2s0_031770

SOURCE: Avenue just off Mansfield Road in Nottingham er contact the college for more information that starts at seven tonight.

TARGET: Nottingham | UPDATED: the college

SOURCE_STATE: the city off Mansfield Road | NEW_STATE: closed permanently today

UPDATE: The college ceased operations and is now closed permanently today.

USE: Nottingham remains the city located just off Mansfield Road.

SURFACE use-source: J=0.286, overlap_min=0.667, LCS=4 `just off mansfield road`; use-update LCS=1 `the`

STATE_TERMS source=['city', 'off', 'mansfield', 'road'] new=['clos', 'permanently', 'today'] hits_update=['clos', 'permanently', 'today'] hits_use_new=[] hits_use_source=['city', 'off', 'mansfield', 'road']

### 50. rw_008646

SOURCE: With that she met Fendi, CEO for Brooklyn label Dirty Money Entertainment.

TARGET: Fendi CEO | UPDATED: Dirty Money Entertainment

SOURCE_STATE: runs Brooklyn label Dirty Money Entertainment | NEW_STATE: local indie music collective

UPDATE: Dirty Money Entertainment now operates as a local indie music collective in the city.

USE: Fendi CEO continues to steer Brooklyn label Dirty Money Entertainment forward today.

SURFACE use-source: J=0.583, overlap_min=0.875, LCS=5 `brooklyn label dirty money entertainment`; use-update LCS=3 `dirty money entertainment`

STATE_TERMS source=['runs', 'brooklyn', 'label', 'dirty', 'money', 'entertainment'] new=['indie', 'music', 'collective'] hits_update=['indie', 'music', 'collective'] hits_use_new=[] hits_use_source=['brooklyn', 'label', 'dirty', 'money', 'entertainment']

