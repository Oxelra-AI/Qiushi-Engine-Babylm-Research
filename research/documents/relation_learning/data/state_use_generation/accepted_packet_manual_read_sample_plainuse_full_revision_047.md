# earlier analysis accepted cue-controlled plain-use packet sample

These examples passed step044 copy like diagnosis changed-form state/content checks and the earlier analysis no-use-cue filter.

## UPDATED_USE

### 1. rw_004698

SOURCE: They were defeated by the team of Cora Jade, Io Shirai, Kay Lee Ray, and González in a WarGames match.

TARGET: González | UPDATED: González

SOURCE_STATE: defeated in a WarGames match | NEW_STATE: championship title winner

UPDATE: After the intense WarGames match, González emerged as the championship title winner.

USE: González holds the championship title with pride.

USE_CUES: []

SURFACE use-source: J=0.125, overlap_min=0.333, LCS=2 `gonz lez`; use-update LCS=3 `the championship title`

STATE_TERMS source=['defeat', 'wargam', 'match'] new=['championship', 'title', 'winner'] hits_update=['championship', 'title', 'winner'] hits_use_new=['championship', 'title'] hits_use_source=[]

### 2. rw2s1_029608

SOURCE: They had just mounted to the vessel's deck when Lu-san's father, who had been looking off towards the west, suddenly called the family to his side.

TARGET: Lu-san's father | UPDATED: Lu-san's father

SOURCE_STATE: looking off towards the west | NEW_STATE: standing beside the family

UPDATE: Lu-san's father stood beside the family after calling them over.

USE: Lu-san's father listens to the family while standing nearby.

USE_CUES: []

SURFACE use-source: J=0.188, overlap_min=0.500, LCS=3 `lu san father`; use-update LCS=3 `lu san father`

STATE_TERMS source=['look', 'off', 'toward', 'west'] new=['stand', 'beside', 'family'] hits_update=['beside', 'family'] hits_use_new=['stand', 'family'] hits_use_source=[]

### 3. rw_000762

SOURCE: Right, I'm sure that er, many people will know by now, but I'm gonna say it anyway that at the end of January, we will be on the Breakfast Call, broadcasting to you live from Sydney, to celebrate Australia Day.

TARGET: Sydney | UPDATED: Sydney

SOURCE_STATE: broadcasting location for the Breakfast Call | NEW_STATE: hosting a cultural festival

UPDATE: The city of Sydney will host a major cultural festival next month.

USE: Tourists visit Sydney for its vibrant cultural festival.

USE_CUES: []

SURFACE use-source: J=0.045, overlap_min=0.167, LCS=1 `sydney`; use-update LCS=2 `cultural festival`

STATE_TERMS source=['broadcast', 'breakfast', 'call'] new=['host', 'cultural', 'festival'] hits_update=['host', 'cultural', 'festival'] hits_use_new=['cultural', 'festival'] hits_use_source=[]

### 4. rw2s0_027404

SOURCE: Much to their relief, however, the miser did not go into his bedroom, and they soon heard him hard at work in the garden.

TARGET: garden | UPDATED: garden

SOURCE_STATE: where the miser worked | NEW_STATE: overgrown with weeds

UPDATE: Neglect caused the neglected garden to become overgrown with weeds.

USE: The overgrown with weeds garden attracts no visitors.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.200, LCS=1 `the`; use-update LCS=3 `overgrown with weed`

STATE_TERMS source=['miser', 'work'] new=['overgrown', 'weed'] hits_update=['overgrown', 'weed'] hits_use_new=['overgrown', 'weed'] hits_use_source=[]

### 5. rw_035110

SOURCE: However, above didn't mean Heaven anymore, - but seeing things from a new perspective.

TARGET: Heaven | UPDATED: Heaven

SOURCE_STATE: above | NEW_STATE: a fresh perspective

UPDATE: The concept of Heaven shifted from being above to representing a fresh perspective.

USE: People find comfort in Heaven as a fresh perspective.

USE_CUES: []

SURFACE use-source: J=0.154, overlap_min=0.333, LCS=1 `heaven`; use-update LCS=3 `a fresh perspective`

STATE_TERMS source=['above'] new=['fresh', 'perspective'] hits_update=['fresh', 'perspective'] hits_use_new=['fresh', 'perspective'] hits_use_source=[]

### 6. rw_020883

SOURCE: Haworth emigrated to the Netherlands and worked together with among others Ina Santhagens-Waller, and .

TARGET: Ina Santhagens-Waller | UPDATED: Ina Santhagens-Waller

SOURCE_STATE: worked together with Haworth | NEW_STATE: research lead at Leiden University

UPDATE: After the collaboration ended, Ina Santhagens-Waller became the research lead at Leiden University.

USE: Ina Santhagens-Waller directs the team at Leiden University.

USE_CUES: []

SURFACE use-source: J=0.231, overlap_min=0.429, LCS=3 `ina santhagen waller`; use-update LCS=3 `ina santhagen waller`

STATE_TERMS source=['work', 'together', 'haworth'] new=['research', 'lead', 'leiden', 'university'] hits_update=['research', 'lead', 'leiden', 'university'] hits_use_new=['leiden', 'university'] hits_use_source=[]

### 7. rw_003628

SOURCE: Most Láadan sentences have three short words called particles: In Láadan, the verbs come first.

TARGET: Láadan | UPDATED: Láadan

SOURCE_STATE: uses three short particle words | NEW_STATE: requires long complex clauses

UPDATE: Láadan has evolved to require long complex clauses for expressing ideas.

USE: Speakers of Láadan construct long complex clauses to convey nuance.

USE_CUES: []

SURFACE use-source: J=0.056, overlap_min=0.125, LCS=2 `l adan`; use-update LCS=3 `long complex claus`

STATE_TERMS source=['uses', 'three', 'short', 'particle', 'word'] new=['requir', 'long', 'complex', 'claus'] hits_update=['long', 'complex', 'claus'] hits_use_new=['long', 'complex', 'claus'] hits_use_source=[]

### 8. rw2s0_004583

SOURCE: I am going to stay with you; and, mother, may I introduce you to my friend, Angela St Just?

TARGET: Angela St Just | UPDATED: Angela St Just

SOURCE_STATE: friend | NEW_STATE: professional colleague

UPDATE: I will introduce my friend Angela St Just as a professional colleague for the meeting.

USE: Angela St Just leads the quarterly review as a professional colleague.

USE_CUES: []

SURFACE use-source: J=0.091, overlap_min=0.167, LCS=3 `angela st just`; use-update LCS=4 `as a professional colleague`

STATE_TERMS source=['friend'] new=['professional', 'colleague'] hits_update=['professional', 'colleague'] hits_use_new=['professional', 'colleague'] hits_use_source=[]

### 9. rw2s1_002275

SOURCE: Loring came in then, and sent him away, as Guy had been ill with fever ever since his night's exposure, and could bear but little excitement.

TARGET: Guy | UPDATED: Guy

SOURCE_STATE: ill with fever | NEW_STATE: full recovery

UPDATE: The doctor declared that Guy has reached a full recovery after resting.

USE: Guy returns to work with a full recovery completed.

USE_CUES: []

SURFACE use-source: J=0.053, overlap_min=0.167, LCS=1 `guy`; use-update LCS=3 `a full recovery`

STATE_TERMS source=['ill', 'fever'] new=['full', 'recovery'] hits_update=['full', 'recovery'] hits_use_new=['full', 'recovery'] hits_use_source=[]

### 10. rw2s0_019842

SOURCE: Do you seriously think I led you all the way to the core of Ancor Vantian just to have your Chakras sealed?

TARGET: Ancor Vantian | UPDATED: Ancor Vantian

SOURCE_STATE: core location | NEW_STATE: quantum shielded zone

UPDATE: The scientists activated a device that turned the core of Ancor Vantian into a quantum shielded zone.

USE: Explorers cannot enter the quantum shielded zone of Ancor Vantian.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.250, LCS=3 `of ancor vantian`; use-update LCS=3 `of ancor vantian`

STATE_TERMS source=['core'] new=['quantum', 'shield', 'zone'] hits_update=['quantum', 'shield', 'zone'] hits_use_new=['quantum', 'shield', 'zone'] hits_use_source=[]

### 11. rw2s1_022155

SOURCE: He regretted bitterly now that he had not let Francois off the evening before, as in that case Ben would not have had a chance to escape.

TARGET: Ben | UPDATED: Ben

SOURCE_STATE: chance to escape | NEW_STATE: captured prisoner

UPDATE: Security guards apprehended Ben and secured him as a captured prisoner before dawn.

USE: Ben waits in the holding cell as a captured prisoner.

USE_CUES: []

SURFACE use-source: J=0.067, overlap_min=0.167, LCS=1 `the`; use-update LCS=4 `as a captur prisoner`

STATE_TERMS source=['chance', 'escape'] new=['captur', 'prisoner'] hits_update=['captur', 'prisoner'] hits_use_new=['captur', 'prisoner'] hits_use_source=[]

### 12. rw2s0_024871

SOURCE: Professor Horner in the character of a yachtman was rather a different person from the grave and severe schoolmaster whose school they attended.

TARGET: Horner | UPDATED: Horner

SOURCE_STATE: grave and severe schoolmaster | NEW_STATE: charming yachtman

UPDATE: Professor Horner appeared as a charming yachtman rather than the strict educator.

USE: Horner enjoyed sailing with his friends as a yachtman.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.400, LCS=2 `a yachtman`; use-update LCS=2 `as a`

STATE_TERMS source=['grave', 'severe', 'schoolmaster'] new=['charm', 'yachtman'] hits_update=['charm', 'yachtman'] hits_use_new=['yachtman'] hits_use_source=[]

### 13. rw_010215

SOURCE: like snow," returned Guy, looking intently in the direction she indicated, "but it is now June, and we certainly ought not to encounter such a fall as that appears to be, besides, there is a perfect glare of sunshine there!

TARGET: Guy | UPDATED: Guy

SOURCE_STATE: looking intently | NEW_STATE: wearing a bright yellow coat

UPDATE: Guy put on his bright yellow coat before stepping outside.

USE: Guy walks briskly while wearing his bright yellow coat.

USE_CUES: []

SURFACE use-source: J=0.042, overlap_min=0.143, LCS=1 `guy`; use-update LCS=4 `his bright yellow coat`

STATE_TERMS source=['look', 'intently'] new=['wear', 'bright', 'yellow', 'coat'] hits_update=['bright', 'yellow', 'coat'] hits_use_new=['wear', 'bright', 'yellow', 'coat'] hits_use_source=[]

### 14. rw2s0_003356

SOURCE: Well, I guess I can't tell them what to do on their own free time, as long as they've got their work done and Amy's.

TARGET: Amy's | UPDATED: Amy's

SOURCE_STATE: work done | NEW_STATE: finished cooking dinner

UPDATE: Amy completed the work required for her cooking dinner task.

USE: Amy's kitchen smells like the finished cooking dinner she prepared.

USE_CUES: []

SURFACE use-source: J=0.050, overlap_min=0.125, LCS=1 `amy`; use-update LCS=2 `cook dinner`

STATE_TERMS source=['work', 'done'] new=['finish', 'cook', 'dinner'] hits_update=['cook', 'dinner'] hits_use_new=['finish', 'cook', 'dinner'] hits_use_source=[]

### 15. rw_038500

SOURCE: He will have his own establishment at Durham House, with a chaplain, officer and a retinue befitting his station.

TARGET: Durham House | UPDATED: Durham House

SOURCE_STATE: with a chaplain, officer and a retinue | NEW_STATE: renovated grand library

UPDATE: The duke transformed Durham House into a magnificent renovated grand library.

USE: Scholars gather inside the renovated grand library at Durham House daily.

USE_CUES: []

SURFACE use-source: J=0.067, overlap_min=0.125, LCS=3 `at durham house`; use-update LCS=3 `renovat grand library`

STATE_TERMS source=['chaplain', 'officer', 'retinue'] new=['renovat', 'grand', 'library'] hits_update=['renovat', 'grand', 'library'] hits_use_new=['renovat', 'grand', 'library'] hits_use_source=[]

### 16. rw_004241

SOURCE: He got some money on these conditions, and directly set off with Gaveston to the Border-country, where they spent it in idling away the time, and feasting, while Bruce made ready to drive the English out of Scotland.

TARGET: Gaveston | UPDATED: Gaveston

SOURCE_STATE: in Border-country | NEW_STATE: Scottish nobleman

UPDATE: Gaveston has officially become a Scottish nobleman after their journey.

USE: Gaveston represents his status as a Scottish nobleman during court meetings.

USE_CUES: []

SURFACE use-source: J=0.038, overlap_min=0.143, LCS=1 `gaveston`; use-update LCS=3 `a scottish nobleman`

STATE_TERMS source=['border', 'country'] new=['scottish', 'nobleman'] hits_update=['scottish', 'nobleman'] hits_use_new=['scottish', 'nobleman'] hits_use_source=[]

### 17. rw2s0_014373

SOURCE: The impeller, typically located at the rear of the watercraft, draws water in and forcefully expels it, propelling the PWC forward at high speeds.

TARGET: PWC | UPDATED: PWC

SOURCE_STATE: rear of the watercraft | NEW_STATE: electric motor power

UPDATE: Modern vessels equip the PWC with electric motor power instead of gas engines.

USE: The PWC travels efficiently using electric motor power.

USE_CUES: []

SURFACE use-source: J=0.050, overlap_min=0.143, LCS=2 `the pwc`; use-update LCS=3 `electric motor power`

STATE_TERMS source=['rear', 'watercraft'] new=['electric', 'motor', 'power'] hits_update=['electric', 'motor', 'power'] hits_use_new=['electric', 'motor', 'power'] hits_use_source=[]

### 18. rw_008318

SOURCE: The Onion is similar in that it also has rather hilarious headlines.

TARGET: The Onion | UPDATED: The Onion

SOURCE_STATE: has rather hilarious headlines | NEW_STATE: publishes satirical news

UPDATE: The Onion has shifted from just having funny headlines to publishing satirical news.

USE: Readers enjoy The Onion for its satirical news.

USE_CUES: []

SURFACE use-source: J=0.111, overlap_min=0.200, LCS=2 `the onion`; use-update LCS=2 `the onion`

STATE_TERMS source=['rather', 'hilariou', 'headlin'] new=['publish', 'satirical', 'news'] hits_update=['publish', 'satirical', 'news'] hits_use_new=['satirical', 'news'] hits_use_source=[]

### 19. rw2s0_012723

SOURCE: The women's water polo match was played between a mix of players from West-Germany against the Dutch Hollandsche Dames Zwemclub.

TARGET: West-Germany | UPDATED: West-Germany

SOURCE_STATE: female water polo participant | NEW_STATE: renowned football powerhouse

UPDATE: West-Germany transitioned from water polo fame to becoming a renowned football powerhouse.

USE: Fans celebrate West-Germany as a renowned football powerhouse.

USE_CUES: []

SURFACE use-source: J=0.100, overlap_min=0.286, LCS=2 `west germany`; use-update LCS=4 `a renown football powerhouse`

STATE_TERMS source=['female', 'water', 'polo', 'participant'] new=['renown', 'football', 'powerhouse'] hits_update=['renown', 'football', 'powerhouse'] hits_use_new=['renown', 'football', 'powerhouse'] hits_use_source=[]

### 20. rw2s1_016424

SOURCE: At WrestleDream on October 1, 2023, Swerve faced "Hangman" Adam Page in front of his home state crowd in Washington.

TARGET: Swerve | UPDATED: Swerve

SOURCE_STATE: facing Hangman Adam Page | NEW_STATE: champion of the division

UPDATE: Swerve defeated his opponent to become the official champion of the division.

USE: Fans celebrate Swerve as the reigning champion of the division.

USE_CUES: []

SURFACE use-source: J=0.062, overlap_min=0.167, LCS=1 `swerve`; use-update LCS=4 `champion of the division`

STATE_TERMS source=['facing', 'hangman', 'adam', 'page'] new=['champion', 'division'] hits_update=['champion', 'division'] hits_use_new=['champion', 'division'] hits_use_source=[]

### 21. rw2s0_011122

SOURCE: It was the custom of the school that each girl should keep her desk locked, and Kitty now slipped the key of hers into her pocket.

TARGET: Kitty | UPDATED: Kitty

SOURCE_STATE: keeping desk locked | NEW_STATE: holding school badge

UPDATE: Kitty took her school badge out of her pocket to show the visitor.

USE: Kitty wears her school badge around her neck every day.

USE_CUES: []

SURFACE use-source: J=0.154, overlap_min=0.400, LCS=1 `school`; use-update LCS=3 `her school badge`

STATE_TERMS source=['keep', 'desk', 'lock'] new=['hold', 'school', 'badge'] hits_update=['school', 'badge'] hits_use_new=['school', 'badge'] hits_use_source=[]

### 22. rw_021131

SOURCE: But an hour or two later, just when my mother and aunt were sitting comfortably chatting, and Emilia trying over some old songs on the thin-toned piano, they and the two maid-servants in the kitchen were suddenly startled by piercing screams from my room.

TARGET: Emilia | UPDATED: Emilia

SOURCE_STATE: trying over old songs on the piano | NEW_STATE: cooking dinner in the kitchen

UPDATE: Emilia left the piano to begin preparing a meal in the kitchen.

USE: Emilia serves the family dinner she prepared in the kitchen.

USE_CUES: []

SURFACE use-source: J=0.080, overlap_min=0.333, LCS=3 `in the kitchen`; use-update LCS=3 `in the kitchen`

STATE_TERMS source=['trying', 'song', 'piano'] new=['cook', 'dinner', 'kitchen'] hits_update=['kitchen'] hits_use_new=['dinner', 'kitchen'] hits_use_source=[]

### 23. rw_041058

SOURCE: On my advice, the veep's gonna ask you to be in the Families First talks with Dan Egan.

TARGET: Dan Egan | UPDATED: Dan Egan

SOURCE_STATE: Families First talks participant | NEW_STATE: senior policy advisor

UPDATE: The leadership team appointed Dan Egan to serve as senior policy advisor for the initiative.

USE: Colleagues respect Dan Egan for his work as senior policy advisor.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.250, LCS=2 `dan egan`; use-update LCS=4 `as senior policy advisor`

STATE_TERMS source=['famili', 'first', 'talk', 'participant'] new=['senior', 'policy', 'advisor'] hits_update=['senior', 'policy', 'advisor'] hits_use_new=['senior', 'policy', 'advisor'] hits_use_source=[]

### 24. rw2s1_018601

SOURCE: After leaving botany for a few years, he started collecting plants in 1869, when he made an expedition to collect fossils with Jacob Boll (1828–1880) in West Texas.

TARGET: Jacob Boll | UPDATED: Jacob Boll

SOURCE_STATE: collecting fossils | NEW_STATE: notable paleontologist

UPDATE: Jacob Boll became a notable paleontologist after his work in West Texas.

USE: Jacob Boll is remembered as a notable paleontologist.

USE_CUES: []

SURFACE use-source: J=0.105, overlap_min=0.400, LCS=2 `jacob boll`; use-update LCS=3 `a notable paleontologist`

STATE_TERMS source=['collect', 'fossil'] new=['notable', 'paleontologist'] hits_update=['notable', 'paleontologist'] hits_use_new=['notable', 'paleontologist'] hits_use_source=[]

### 25. rw2s1_032330

SOURCE: He meant no sort of truth in this, and soon afterwards galloped away on horseback to join some tiresome Highland friends, who were always flourishing dirks and broadswords.

TARGET: Highland | UPDATED: Highland

SOURCE_STATE: flourishing dirks and broadswords | NEW_STATE: modern tactical gear

UPDATE: The Highland warriors swapped their traditional weapons for modern tactical gear to fight effectively.

USE: Modern Highland soldiers carry advanced tactical gear in the field.

USE_CUES: []

SURFACE use-source: J=0.043, overlap_min=0.125, LCS=1 `in`; use-update LCS=2 `tactical gear`

STATE_TERMS source=['flourish', 'dirk', 'broadsword'] new=['modern', 'tactical', 'gear'] hits_update=['modern', 'tactical', 'gear'] hits_use_new=['modern', 'tactical', 'gear'] hits_use_source=[]

### 26. rw2s1_004601

SOURCE: At Ted's school, as at all others, there were quarrels and fights sometimes; and many a day he came home with traces of war, in the shapes of bumps and bruises and scratches.

TARGET: Ted's school | UPDATED: Ted's school

SOURCE_STATE: place with quarrels | NEW_STATE: modern learning center

UPDATE: The administration renovated Ted's school into a modern learning center.

USE: Students gather daily in Ted's modern learning center.

USE_CUES: []

SURFACE use-source: J=0.053, overlap_min=0.143, LCS=1 `ted`; use-update LCS=3 `modern learn center`

STATE_TERMS source=['quarrel'] new=['modern', 'learn', 'center'] hits_update=['modern', 'learn', 'center'] hits_use_new=['modern', 'learn', 'center'] hits_use_source=[]

### 27. rw2s1_009740

SOURCE: The four lost no time in questions; they saw two plebes in distress, and they had met Indian on the warpath and learned the cause of the trouble.

TARGET: Indian | UPDATED: Indian

SOURCE_STATE: warpath condition | NEW_STATE: modern peacekeeper

UPDATE: The group learned the cause of trouble from an Indian who serves as a modern peacekeeper.

USE: The Indian peacekeeper helps resolve community conflicts.

USE_CUES: []

SURFACE use-source: J=0.053, overlap_min=0.167, LCS=1 `the`; use-update LCS=1 `the`

STATE_TERMS source=['warpath'] new=['modern', 'peacekeeper'] hits_update=['modern', 'peacekeeper'] hits_use_new=['peacekeeper'] hits_use_source=[]

### 28. rw_018159

SOURCE: And if Jim's got some some new form of escalator that he wants to er sell to the customers, or or or whatever,th then Mm.

TARGET: Jim | UPDATED: Jim

SOURCE_STATE: selling escalators | NEW_STATE: designing roller coasters

UPDATE: Jim stopped selling escalators to start designing thrilling roller coasters for parks.

USE: Jim sketches roller coaster loops during his morning work session.

USE_CUES: []

SURFACE use-source: J=0.062, overlap_min=0.125, LCS=1 `jim`; use-update LCS=2 `roller coaster`

STATE_TERMS source=['sell', 'escalator'] new=['design', 'roller', 'coaster'] hits_update=['design', 'roller', 'coaster'] hits_use_new=['roller', 'coaster'] hits_use_source=[]

### 29. rw_005873

SOURCE: 6,000 technicians and scientists worked on hundreds of projects in the fields of materials, weather, space, medicine and metallurgics guided by the talented physicist, Jorge Sábato.

TARGET: Jorge Sábato | UPDATED: Jorge Sábato

SOURCE_STATE: guided by the talented physicist | NEW_STATE: director of the laboratory

UPDATE: The research center appointed Jorge Sábato as director of the laboratory.

USE: Jorge Sábato leads the laboratory with great authority.

USE_CUES: []

SURFACE use-source: J=0.100, overlap_min=0.333, LCS=3 `jorge s bato`; use-update LCS=3 `jorge s bato`

STATE_TERMS source=['guid', 'talent', 'physicist'] new=['director', 'laboratory'] hits_update=['director', 'laboratory'] hits_use_new=['laboratory'] hits_use_source=[]

### 30. rw2s0_032618

SOURCE: Slowly the fact of Captain John's great good fortune slipped into the background, and bigger and bigger loomed the certainty which he could not bear to face.

TARGET: Captain John | UPDATED: Captain John

SOURCE_STATE: great good fortune | NEW_STATE: overwhelming dread

UPDATE: Captain John abandoned his great good fortune as overwhelming dread seized his mind.

USE: The captain hides from overwhelming dread in his quarters.

USE_CUES: []

SURFACE use-source: J=0.056, overlap_min=0.200, LCS=1 `the`; use-update LCS=2 `overwhelm dread`

STATE_TERMS source=['great', 'good', 'fortune'] new=['overwhelm', 'dread'] hits_update=['overwhelm', 'dread'] hits_use_new=['overwhelm', 'dread'] hits_use_source=[]

### 31. rw2s0_029341

SOURCE: The _Sea Witch_, without a captain, had laid her course for him, and was waiting on the shore.

TARGET: _Sea Witch_ | UPDATED: _Sea Witch_

SOURCE_STATE: waiting on the shore | NEW_STATE: anchored in the bay

UPDATE: The _Sea Witch_ shifted from the shore to an anchored position in the bay.

USE: Fishermen observe the _Sea Witch_ while it is anchored in the bay.

USE_CUES: []

SURFACE use-source: J=0.167, overlap_min=0.333, LCS=3 `the sea witch`; use-update LCS=3 `the sea witch`

STATE_TERMS source=['wait', 'shore'] new=['anchor', 'bay'] hits_update=['anchor', 'bay'] hits_use_new=['anchor', 'bay'] hits_use_source=[]

### 32. rw_013718

SOURCE: "I like the long evenings, when it's dark by tea-time, and we can sit round the fire and talk; it's really far more fun than the summer term." "I love the summer best," said Aldred.

TARGET: Aldred | UPDATED: Aldred

SOURCE_STATE: loves the summer | NEW_STATE: prefers autumn evenings

UPDATE: Aldred prefers autumn evenings when the days grow shorter and the air turns crisp.

USE: Aldred enjoys walking through the colorful leaves during autumn evenings.

USE_CUES: []

SURFACE use-source: J=0.077, overlap_min=0.250, LCS=1 `the`; use-update LCS=2 `autumn evening`

STATE_TERMS source=['love', 'summer'] new=['prefer', 'autumn', 'evening'] hits_update=['prefer', 'autumn', 'evening'] hits_use_new=['autumn', 'evening'] hits_use_source=[]

### 33. rw2s1_031583

SOURCE: It was first proposed by Kristina Lerman, Principal Information Scientist and mathematician at the University of Southern California.

TARGET: Kristina Lerman | UPDATED: Kristina Lerman

SOURCE_STATE: Principal Information Scientist at the University of Southern California | NEW_STATE: Director of Data Science in Austin

UPDATE: Kristina Lerman transitioned her role to become Director of Data Science in Austin.

USE: The Austin team relies on Kristina Lerman for her expertise as Director of Data Science.

USE_CUES: []

SURFACE use-source: J=0.111, overlap_min=0.222, LCS=2 `kristina lerman`; use-update LCS=4 `director of data science`

STATE_TERMS source=['principal', 'information', 'scientist', 'university', 'southern', 'california'] new=['director', 'data', 'science', 'austin'] hits_update=['director', 'data', 'science', 'austin'] hits_use_new=['director', 'data', 'science', 'austin'] hits_use_source=[]

### 34. rw2s1_028652

SOURCE: Successively he was a missionary in Egypt for seven years, whereafter he obtained a masters in Western-Islam relations and the Middle East at the Johns Hopkins University.

TARGET: Johns Hopkins University | UPDATED: Johns Hopkins University

SOURCE_STATE: affiliated with Middle East studies | NEW_STATE: campus library renovation

UPDATE: Construction crews began the campus library renovation at Johns Hopkins University last month.

USE: Students study inside the renovated campus library at Johns Hopkins University.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.333, LCS=3 `john hopkin university`; use-update LCS=4 `at john hopkin university`

STATE_TERMS source=['affiliat', 'middle', 'east', 'studi'] new=['campu', 'library', 'renovation'] hits_update=['campu', 'library', 'renovation'] hits_use_new=['campu', 'library'] hits_use_source=[]

### 35. rw2s0_017287

SOURCE: The Afghans knew that to capture them as they stood, meant the certain annihilation of the British troops as they defiled into the plain.

TARGET: British troops | UPDATED: British troops

SOURCE_STATE: defiling into the plain | NEW_STATE: fortified defensive positions

UPDATE: The exhausted British troops established fortified defensive positions to resist the attack.

USE: Commanders observe the British troops holding their fortified defensive positions.

USE_CUES: []

SURFACE use-source: J=0.125, overlap_min=0.286, LCS=3 `the british troop`; use-update LCS=3 `fortify defensive position`

STATE_TERMS source=['defil', 'plain'] new=['fortify', 'defensive'] hits_update=['fortify', 'defensive'] hits_use_new=['fortify', 'defensive'] hits_use_source=[]

### 36. rw2s0_023442

SOURCE: We can only picture to ourselves the Roman tessellated pavement bestrewn with wine, bones, and fragments of the barbarous revelry.

TARGET: Roman | UPDATED: Roman

SOURCE_STATE: tessellated pavement | NEW_STATE: ancient stone ruins

UPDATE: The Roman site has transformed into a collection of ancient stone ruins.

USE: Tourists explore the Roman ancient stone ruins along the valley.

USE_CUES: []

SURFACE use-source: J=0.059, overlap_min=0.143, LCS=2 `the roman`; use-update LCS=3 `ancient stone ruin`

STATE_TERMS source=['tessellat', 'pavement'] new=['ancient', 'stone', 'ruin'] hits_update=['ancient', 'stone', 'ruin'] hits_use_new=['ancient', 'stone', 'ruin'] hits_use_source=[]

### 37. rw2s0_015171

SOURCE: The weights did not meet the amount of gold ornaments that had been contributed for the purpose, and no doubt the Gauls were resolved to have all that they beheld; for when Sulpitius was about to try to arrange the balance, Brennus insultingly threw his sword into his own scale, exclaiming, Voe victis!

TARGET: Brennus | UPDATED: Brennus

SOURCE_STATE: insultingly threw his sword | NEW_STATE: warrior leader

UPDATE: Brennus stood as the warrior leader who demanded the gold.

USE: Brennus commanded the Gauls with authority as their leader.

USE_CUES: []

SURFACE use-source: J=0.074, overlap_min=0.400, LCS=2 `the gaul`; use-update LCS=1 `brennu`

STATE_TERMS source=['insultingly', 'threw', 'sword'] new=['warrior', 'leader'] hits_update=['warrior', 'leader'] hits_use_new=['leader'] hits_use_source=[]

### 38. rw2s1_013807

SOURCE: According to the Hebrew Bible, God commanded Moses and the Israelites to make councils of judges and to do what they said.

TARGET: God | UPDATED: God

SOURCE_STATE: commanded Moses | NEW_STATE: divine judge of nations

UPDATE: God established a role as a divine judge of nations for all people.

USE: God listens to prayers from the divine judge of nations.

USE_CUES: []

SURFACE use-source: J=0.062, overlap_min=0.167, LCS=1 `to`; use-update LCS=4 `divine judge of nation`

STATE_TERMS source=['command', 'mose'] new=['divine', 'judge', 'nation'] hits_update=['divine', 'judge', 'nation'] hits_use_new=['divine', 'judge', 'nation'] hits_use_source=[]

### 39. rw2s1_024155

SOURCE: Ian was very good as sports and wanted to be an athlete, his grades were also good and so he was able to pass to Rhodes University College in South Africa.

TARGET: Rhodes University College | UPDATED: Rhodes University College

SOURCE_STATE: in South Africa | NEW_STATE: renamed Rhodes University

UPDATE: The institution operating in South Africa changed its official name to Rhodes University.

USE: Students apply to Rhodes University for their undergraduate degrees.

USE_CUES: []

SURFACE use-source: J=0.118, overlap_min=0.333, LCS=3 `to rhod university`; use-update LCS=3 `to rhod university`

STATE_TERMS source=['south', 'africa'] new=['renam', 'rhod', 'university'] hits_update=['rhod', 'university'] hits_use_new=['rhod', 'university'] hits_use_source=[]

### 40. rw2s1_006843

SOURCE: The Pim Fortuyn List was a populist party, established by Pim Fortuyn, and also named after him, in 2002.

TARGET: Pim Fortuyn List | UPDATED: Pim Fortuyn List

SOURCE_STATE: populist party | NEW_STATE: national political movement

UPDATE: The Pim Fortuyn List evolved into a major national political movement.

USE: Supporters view the Pim Fortuyn List as a national political movement.

USE_CUES: []

SURFACE use-source: J=0.231, overlap_min=0.375, LCS=4 `the pim fortuyn list`; use-update LCS=4 `the pim fortuyn list`

STATE_TERMS source=['populist', 'party'] new=['national', 'political', 'movement'] hits_update=['national', 'political', 'movement'] hits_use_new=['national', 'political', 'movement'] hits_use_source=[]

### 41. rw2s1_004216

SOURCE: Aggie whispered to Guy, as she bade him "good-night," that after that prayer she should not be afraid of the Indians, and went very contentedly to her mother's wagon, while Guy followed Gus and George to the one in which they were to sleep.

TARGET: Guy | UPDATED: Guy

SOURCE_STATE: following Gus and George | NEW_STATE: sleeping in the wagon

UPDATE: Guy left the group to sleep in the wagon where they were to rest.

USE: Guy sleeps soundly inside the wagon tonight.

USE_CUES: []

SURFACE use-source: J=0.150, overlap_min=0.500, LCS=1 `guy`; use-update LCS=2 `the wagon`

STATE_TERMS source=['follow', 'gus', 'george'] new=['sleep', 'wagon'] hits_update=['sleep', 'wagon'] hits_use_new=['sleep', 'wagon'] hits_use_source=[]

### 42. rw2s1_013527

SOURCE: “Drifting away, I fear;” and Miss Henny's plump face looked almost beautiful, with the tears on it, as she leaned nearer to listen to the faint breath at the child's lips.

TARGET: Miss Henny's | UPDATED: Miss Henny's

SOURCE_STATE: face | NEW_STATE: cheerful smile

UPDATE: After the sad news, Miss Henny's face broke into a cheerful smile.

USE: Miss Henny's cheerful smile brightened the entire room.

USE_CUES: []

SURFACE use-source: J=0.091, overlap_min=0.333, LCS=2 `miss henny`; use-update LCS=2 `miss henny`

STATE_TERMS source=['face'] new=['cheerful', 'smile'] hits_update=['cheerful', 'smile'] hits_use_new=['cheerful', 'smile'] hits_use_source=[]

### 43. rw2s0_003404

SOURCE: After working as a physician for twenty years in Syria, he retired and devoted himself to literary works.

TARGET: Syria | UPDATED: Syria

SOURCE_STATE: where he worked as a physician | NEW_STATE: a nation with ancient history

UPDATE: Syria is a nation with ancient history that he observed during his medical career.

USE: Travelers often visit Syria to explore its ancient history.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.143, LCS=1 `syria`; use-update LCS=2 `ancient history`

STATE_TERMS source=['work', 'physician'] new=['nation', 'ancient', 'history'] hits_update=['nation', 'ancient', 'history'] hits_use_new=['ancient', 'history'] hits_use_source=[]

### 44. rw2s1_025889

SOURCE: CHAPTER XIV ENGLAND UNDER KING JOHN, CALLED LACKLAND At two-and-thirty years of age, John became King of England.

TARGET: John | UPDATED: John

SOURCE_STATE: King of England | NEW_STATE: crownless monarch

UPDATE: The council stripped John of his title and declared him a crownless monarch.

USE: John ruled the kingdom as a crownless monarch.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.200, LCS=1 `john`; use-update LCS=3 `a crownless monarch`

STATE_TERMS source=['king', 'england'] new=['crownless', 'monarch'] hits_update=['crownless', 'monarch'] hits_use_new=['crownless', 'monarch'] hits_use_source=[]

### 45. rw2s1_007414

SOURCE: In places where the Zika virus lives, people are more likely to have birth defects, neurological problems like Guillain-Barré syndrome, and autoimmune diseases.

TARGET: Zika | UPDATED: Zika

SOURCE_STATE: live in places | NEW_STATE: transmitted by mosquitoes

UPDATE: The Zika virus spreads primarily through the bite of infected mosquitoes.

USE: Mosquitoes carry the Zika virus to nearby communities.

USE_CUES: []

SURFACE use-source: J=0.100, overlap_min=0.333, LCS=3 `the zika viru`; use-update LCS=3 `the zika viru`

STATE_TERMS source=['plac'] new=['transmitt', 'mosquito'] hits_update=['mosquito'] hits_use_new=['mosquito'] hits_use_source=[]

### 46. rw2s0_013030

SOURCE: Major Grafton notified the clerk that he wished a hack in fifteen minutes, as he was about to leave the hotel.

TARGET: Grafton | UPDATED: Grafton

SOURCE_STATE: at the hotel | NEW_STATE: outside the building

UPDATE: Major Grafton stepped out of the hotel and stood outside the building.

USE: Major Grafton waits for the carriage outside the building.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.333, LCS=2 `major grafton`; use-update LCS=3 `outside the build`

STATE_TERMS source=['hotel'] new=['outside', 'build'] hits_update=['outside', 'build'] hits_use_new=['outside', 'build'] hits_use_source=[]

### 47. rw2s1_025733

SOURCE: It may have been seven years late and a billion dollars over budget, but the Hubble space telescope was lifting out of the shuttle cargo bay and on its way to orbit at last, ready to look back across the universe to the very beginnings of time.

TARGET: Hubble | UPDATED: Hubble

SOURCE_STATE: lifting from shuttle cargo bay | NEW_STATE: orbiting in space

UPDATE: The Hubble space telescope successfully reached its orbit in space after the launch.

USE: Astronomers rely on Hubble in orbit to study distant stars.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.286, LCS=1 `hubble`; use-update LCS=1 `hubble`

STATE_TERMS source=['lift', 'shuttle', 'cargo', 'bay'] new=['orbit', 'space'] hits_update=['orbit', 'space'] hits_use_new=['orbit'] hits_use_source=[]

### 48. rw2s0_015944

SOURCE: Ramsey said: "Miss Newman, I want to have a talk with you." So she and Miss Newman took themselves to the further corner of the porch while the little girls gathered around Miss Eloise till the stars came out and it was too cool to sit out longer.

TARGET: Miss Newman | UPDATED: Miss Newman

SOURCE_STATE: on the porch | NEW_STATE: inside the classroom

UPDATE: Miss Newman left the porch and entered the classroom where her students waited.

USE: Miss Newman instructs her students from inside the classroom.

USE_CUES: []

SURFACE use-source: J=0.077, overlap_min=0.333, LCS=2 `miss newman`; use-update LCS=2 `miss newman`

STATE_TERMS source=['porch'] new=['inside', 'classroom'] hits_update=['classroom'] hits_use_new=['inside', 'classroom'] hits_use_source=[]

### 49. rw2s1_024882

SOURCE: It was in the reign of his great-grandson, Hrothgar, that there took place those things that are told in the story of Beowulf.

TARGET: Hrothgar | UPDATED: Hrothgar

SOURCE_STATE: great-grandson of the speaker | NEW_STATE: king of the Geats

UPDATE: The narrative places Hrothgar in the role of king of the Geats during the epic tale.

USE: Hrothgar rules the Geats with strength and courage.

USE_CUES: []

SURFACE use-source: J=0.071, overlap_min=0.200, LCS=1 `the`; use-update LCS=2 `the geat`

STATE_TERMS source=['great', 'grandson', 'speaker'] new=['king', 'geat'] hits_update=['king', 'geat'] hits_use_new=['geat'] hits_use_source=[]

### 50. rw_013743

SOURCE: The Windsor Hotel, on Fifth avenue, is over two miles farther up town than the Metropolitan.

TARGET: Windsor Hotel | UPDATED: Windsor Hotel

SOURCE_STATE: over two miles farther up town than the Metropolitan | NEW_STATE: neighbor to the Grand Plaza

UPDATE: The Windsor Hotel relocated to become a neighbor to the Grand Plaza.

USE: Guests at the Windsor Hotel enjoy views of the Grand Plaza.

USE_CUES: []

SURFACE use-source: J=0.143, overlap_min=0.286, LCS=3 `the windsor hotel`; use-update LCS=3 `the windsor hotel`

STATE_TERMS source=['two', 'mile', 'farther', 'town', 'metropolitan'] new=['neighbor', 'grand', 'plaza'] hits_update=['neighbor', 'grand', 'plaza'] hits_use_new=['grand', 'plaza'] hits_use_source=[]

## UNCHANGED_DISTRACTOR_USE

### 1. rw2s0_025741

SOURCE: This intrusion led to a legal battle between the Ecuadorian government and Sarayaku community before the Inter-American Court of Human Rights, which eventually resulted to a victory for the Sarayaku community.

TARGET: Ecuadorian government | UPDATED: Sarayaku community

SOURCE_STATE: engaged in legal battle against Sarayaku | NEW_STATE: achieved victory in court

UPDATE: The Sarayaku community achieved victory in court.

USE: The Ecuadorian government faced legal action against the Sarayaku.

USE_CUES: []

SURFACE use-source: J=0.200, overlap_min=0.571, LCS=3 `the ecuadorian government`; use-update LCS=2 `the sarayaku`

STATE_TERMS source=['engag', 'legal', 'battle', 'against', 'sarayaku'] new=['achiev', 'victory', 'court'] hits_update=['achiev', 'victory', 'court'] hits_use_new=[] hits_use_source=['legal', 'against', 'sarayaku']

### 2. rw2s1_007708

SOURCE: They had no sooner entered than old Bartlemy said to Hugo, "Thou didst not see the man at the hut?" "Nay," answered Hugo, with a nervous start.

TARGET: Bartlemy | UPDATED: Hugo

SOURCE_STATE: asked Hugo about the man at the hut | NEW_STATE: spoke without starting nervously

UPDATE: Hugo replied with a calm voice and no nervous start.

USE: Bartlemy questions Hugo regarding the man at the hut.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.667, LCS=5 `the man at the hut`; use-update LCS=1 `hugo`

STATE_TERMS source=['asked', 'hugo', 'man', 'hut'] new=['spoke', 'without', 'start', 'nervously'] hits_update=['start'] hits_use_new=[] hits_use_source=['hugo', 'man', 'hut']

### 3. rw2s1_030062

SOURCE: It revolves around the misadventures of best friends Neeladri (who is nicknamed Neel) Walia and Yashpal (who is nicknamed Yash) Mehta, two confident and energetic 13-year-old boys who are skilled dancers and are on the threshold of taking their steps into the world of professional dancing.

TARGET: Neel Walia | UPDATED: Yash Mehta

SOURCE_STATE: skilled dancer on the threshold of professional dancing | NEW_STATE: quiet observer in the corner

UPDATE: Yash Mehta stopped dancing and became a quiet observer in the corner.

USE: Neel Walia performs skilled dances near the professional stage.

USE_CUES: []

SURFACE use-source: J=0.200, overlap_min=0.714, LCS=2 `neel walia`; use-update LCS=1 `danc`

STATE_TERMS source=['skill', 'dancer', 'threshold', 'professional', 'danc'] new=['quiet', 'observer', 'corner'] hits_update=['quiet', 'observer', 'corner'] hits_use_new=[] hits_use_source=['skill', 'professional', 'danc']

### 4. rw_041182

SOURCE: I wouldn't mind the gangrene setting in, then they'll be after whisking me away to a nice, warm hospital bed.

TARGET: hospital | UPDATED: the gangrene

SOURCE_STATE: nice warm bed | NEW_STATE: rapidly spreading infection

UPDATE: The gangrene has now become a rapidly spreading infection in the patient.

USE: Doctors treat the warm hospital bed for comfort and recovery.

USE_CUES: []

SURFACE use-source: J=0.200, overlap_min=0.429, LCS=3 `warm hospital bed`; use-update LCS=1 `the`

STATE_TERMS source=['nice', 'warm', 'bed'] new=['rapidly', 'spread', 'infection'] hits_update=['rapidly', 'spread', 'infection'] hits_use_new=[] hits_use_source=['warm', 'bed']

### 5. rw_019024

SOURCE: In 1966, Tanner was awarded the Lou Marsh Trophy for being Canada's top athlete.

TARGET: Tanner | UPDATED: Canada's representative team

SOURCE_STATE: received the Lou Marsh Trophy for athletic achievement | NEW_STATE: lost the national championship last year

UPDATE: Canada's representative team lost the national championship last year while Tanner focused on other goals.

USE: Tanner earned the Lou Marsh Trophy for his athletic achievement.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.571, LCS=5 `the lou marsh trophy for`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'lou', 'marsh', 'trophy', 'athletic', 'achievement'] new=['lost', 'national', 'championship', 'last'] hits_update=['lost', 'national', 'championship', 'last'] hits_use_new=[] hits_use_source=['lou', 'marsh', 'trophy', 'athletic', 'achievement']

### 6. rw2s1_021025

SOURCE: Things being in this merry state at home, the Merry Monarch undertook a war with the Dutch; principally because they interfered with an African company, established with the two objects of buying gold-dust and slaves, of which the Duke of York was a leading member.

TARGET: the Duke of York | UPDATED: Dutch

SOURCE_STATE: leading member of the African company | NEW_STATE: allied with the African company

UPDATE: The Dutch became allied with the African company instead.

USE: The Duke of York leads the African company efforts.

USE_CUES: []

SURFACE use-source: J=0.217, overlap_min=0.833, LCS=4 `the duke of york`; use-update LCS=3 `the african company`

STATE_TERMS source=['lead', 'member', 'african', 'company'] new=['ally', 'african', 'company'] hits_update=['ally', 'african', 'company'] hits_use_new=[] hits_use_source=['lead', 'african', 'company']

### 7. rw2s0_014476

SOURCE: He did so, but only to learn that the note had been given to Major Grafton, and that both he and Philip had left the hotel.

TARGET: Philip | UPDATED: the note

SOURCE_STATE: left the hotel with Major Grafton | NEW_STATE: delivered to a different recipient

UPDATE: The note was delivered to a different recipient after the note had been given to Major Grafton.

USE: Philip departed the hotel alongside Major Grafton.

USE_CUES: []

SURFACE use-source: J=0.364, overlap_min=0.667, LCS=2 `major grafton`; use-update LCS=2 `major grafton`

STATE_TERMS source=['left', 'hotel', 'major', 'grafton'] new=['deliver', 'recipient'] hits_update=['deliver', 'recipient'] hits_use_new=[] hits_use_source=['hotel', 'major', 'grafton']

### 8. rw_034064

SOURCE: you're lucky I do not file a report that could bury youso deep in Washington you'd never see the sun.

TARGET: Washington | UPDATED: the bureau

SOURCE_STATE: bury you so deep in Washington you'd never see the sun | NEW_STATE: closed down entirely

UPDATE: The bureau closed down entirely after the audit.

USE: Agents search Washington for any buried documents.

USE_CUES: []

SURFACE use-source: J=0.133, overlap_min=0.400, LCS=1 `bury`; use-update LCS=0 ``

STATE_TERMS source=['bury', 'deep', 'washington', "you'd", 'never', 'see', 'sun'] new=['clos', 'down', 'entirely'] hits_update=['clos', 'down', 'entirely'] hits_use_new=[] hits_use_source=['bury', 'washington']

### 9. rw_015206

SOURCE: In reality it was just then that David lifted his head, and feeling himself moving, Ted fancied it was the wings lifting him upward, and gave the pleased smile which David noticed.

TARGET: David | UPDATED: Ted

SOURCE_STATE: lifted his head upward | NEW_STATE: felt the wings pushing him down

UPDATE: Ted felt the wings pushing him downward toward the ground.

USE: David lifted his head to see the sky clearly.

USE_CUES: []

SURFACE use-source: J=0.167, overlap_min=0.500, LCS=4 `david lift his head`; use-update LCS=1 `the`

STATE_TERMS source=['lift', 'head', 'upward'] new=['felt', 'wing', 'push', 'down'] hits_update=['felt', 'wing', 'push'] hits_use_new=[] hits_use_source=['lift', 'head']

### 10. rw2s0_003775

SOURCE: The impression that I got from Miss Hanamura's actual voice was very energetic, but I could tell that she was carefully keeping her distance.

TARGET: Miss Hanamura's actual voice | UPDATED: the audience

SOURCE_STATE: very energetic but carefully keeping distance | NEW_STATE: silent and confused

UPDATE: The audience became silent and confused during the performance.

USE: Miss Hanamura's actual voice sounds very energetic while keeping distance.

USE_CUES: []

SURFACE use-source: J=0.583, overlap_min=0.875, LCS=4 `miss hanamura actual voice`; use-update LCS=0 ``

STATE_TERMS source=['energetic', 'carefully', 'keep', 'distance'] new=['silent', 'confus'] hits_update=['silent', 'confus'] hits_use_new=[] hits_use_source=['energetic', 'keep', 'distance']

### 11. rw2s0_021154

SOURCE: He was always good--and the fire hurt him more than it did anyone else, though it was Maw Hoover and Jake who made all my trouble.

TARGET: Maw Hoover | UPDATED: Jake

SOURCE_STATE: made all my trouble | NEW_STATE: fixed the broken roof

UPDATE: Jake climbed the ladder and fixed the broken roof quickly.

USE: Maw Hoover caused all the trouble for the family.

USE_CUES: []

SURFACE use-source: J=0.200, overlap_min=0.600, LCS=2 `maw hoover`; use-update LCS=1 `the`

STATE_TERMS source=['made', 'trouble'] new=['fixed', 'broken', 'roof'] hits_update=['fixed', 'broken', 'roof'] hits_use_new=[] hits_use_source=['trouble']

### 12. rw_012633

SOURCE: "Publishers Weekly" listed "It" as the best-selling book in the United States in 1986.

TARGET: Publishers Weekly | UPDATED: United States

SOURCE_STATE: listed the book as best-selling | NEW_STATE: ranked in third place

UPDATE: The United States rankings changed so the book ended up in third place.

USE: Publishers Weekly lists the book as a best-seller.

USE_CUES: []

SURFACE use-source: J=0.500, overlap_min=0.833, LCS=3 `publisher weekly list`; use-update LCS=2 `the book`

STATE_TERMS source=['list', 'book', 'best', 'sell'] new=['rank', 'third'] hits_update=['third'] hits_use_new=[] hits_use_source=['list', 'book', 'best']

### 13. rw_007694

SOURCE: When Rachel first told me that she was dating Harvey Dent, I had one thing to say: "The guy that was cut off from those awful campaign commercials, I believe in Harvey Dent?" Yeah nice slogan, Harvey.

TARGET: Harvey Dent | UPDATED: the campaign commercials

SOURCE_STATE: dating Rachel | NEW_STATE: awful and cut off

UPDATE: The awful and cut off campaign commercials were no longer shown.

USE: Harvey Dent believes in the forgotten slogan with Rachel.

USE_CUES: []

SURFACE use-source: J=0.190, overlap_min=0.667, LCS=2 `harvey dent`; use-update LCS=1 `the`

STATE_TERMS source=['dating', 'rachel'] new=['awful', 'cut', 'off'] hits_update=['awful', 'cut', 'off'] hits_use_new=[] hits_use_source=['rachel']

### 14. rw2s0_028151

SOURCE: He had sent a large force in the direction of Zurich with Johann Bonstetten, and advanced himself with 4,000 horse and 1,400 foot upon Sempach.

TARGET: Sempach | UPDATED: Zurich

SOURCE_STATE: received 4,000 horse and 1,400 foot | NEW_STATE: hosted Johann Bonstetten

UPDATE: Johann Bonstetten was hosted in Zurich by the local militia.

USE: Four thousand horse and one thousand four hundred foot marched toward Sempach.

USE_CUES: []

SURFACE use-source: J=0.167, overlap_min=0.375, LCS=2 `horse and`; use-update LCS=0 ``

STATE_TERMS source=['receiv', 'horse', 'foot'] new=['host', 'johann', 'bonstetten'] hits_update=['host', 'johann', 'bonstetten'] hits_use_new=[] hits_use_source=['horse', 'foot']

### 15. rw2s1_023543

SOURCE: In this episode, the South Park boys try to have a fun summer but cannot because there is no snow.

TARGET: the South Park boys | UPDATED: the snow

SOURCE_STATE: try to have a fun summer | NEW_STATE: melting rapidly

UPDATE: The snow melted rapidly while the South Park boys tried to have a fun summer.

USE: The South Park boys attempt to enjoy the fun summer despite no snow.

USE_CUES: []

SURFACE use-source: J=0.462, overlap_min=0.667, LCS=4 `the south park boys`; use-update LCS=4 `the south park boys`

STATE_TERMS source=['try', 'fun', 'summer'] new=['melt', 'rapidly'] hits_update=['melt', 'rapidly'] hits_use_new=[] hits_use_source=['fun', 'summer']

### 16. rw_027543

SOURCE: He later served as a member of the Board of Regents of the University of Maryland, College Park and the University System of Maryland.

TARGET: the University of Maryland | UPDATED: the library

SOURCE_STATE: hosts the Board of Regents | NEW_STATE: digital archive hub

UPDATE: The university library transformed into a digital archive hub.

USE: Students access the University of Maryland for its Board of Regents.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.667, LCS=4 `the university of maryland`; use-update LCS=2 `the university`

STATE_TERMS source=['host', 'board', 'regent'] new=['digital', 'archive', 'hub'] hits_update=['digital', 'archive', 'hub'] hits_use_new=[] hits_use_source=['board', 'regent']

### 17. rw2s0_028371

SOURCE: Clarence regarded Ben in amazement, and turned away his head in a disgust which he did not attempt to conceal.

TARGET: Ben | UPDATED: Clarence

SOURCE_STATE: regarded Clarence in amazement | NEW_STATE: turned away his head in disgust

UPDATE: Clarence turned away his head in disgust instead.

USE: Ben regards Clarence with pure amazement.

USE_CUES: []

SURFACE use-source: J=0.364, overlap_min=0.800, LCS=1 `clarence`; use-update LCS=1 `clarence`

STATE_TERMS source=['regard', 'clarence', 'amazement'] new=['turn', 'away', 'head', 'disgust'] hits_update=['turn', 'away', 'head', 'disgust'] hits_use_new=[] hits_use_source=['regard', 'clarence', 'amazement']

### 18. rw2s1_001821

SOURCE: The Maori people called the North Island takahē "moho" or "mohoau." Scientists think the North Island takahē became extinct in the 1800s.

TARGET: the Maori people | UPDATED: Scientists

SOURCE_STATE: called the North Island takahē moho | NEW_STATE: believe the species survived

UPDATE: Scientists believe the species survived past the 1800s.

USE: The Maori people named the island bird moho.

USE_CUES: []

SURFACE use-source: J=0.286, overlap_min=0.667, LCS=3 `the maori people`; use-update LCS=1 `the`

STATE_TERMS source=['call', 'north', 'island', 'takah', 'moho'] new=['believe', 'speci', 'surviv'] hits_update=['believe', 'speci', 'surviv'] hits_use_new=[] hits_use_source=['island', 'moho']

### 19. rw2s0_016966

SOURCE: "Yes, when I was young," repeated Becky, in exact imitation of the speaker, whose voice was very flat and nasal.

TARGET: Becky | UPDATED: the voice

SOURCE_STATE: repeated the speaker's words in imitation | NEW_STATE: sounded loud and clear

UPDATE: The voice suddenly became loud and clear instead of flat.

USE: Becky repeats the speaker's words with imitation.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.800, LCS=2 `the speaker`; use-update LCS=1 `the`

STATE_TERMS source=['repeat', 'speaker', 'word', 'imitation'] new=['sound', 'loud', 'clear'] hits_update=['loud', 'clear'] hits_use_new=[] hits_use_source=['repeat', 'speaker', 'word', 'imitation']

### 20. rw_020552

SOURCE: Another separate station of the same name can be found on the Circle, Northern as well as the Hammersmith and City Line.

TARGET: the station | UPDATED: Hammersmith

SOURCE_STATE: located on the Circle Northern line | NEW_STATE: serving the Metropolitan Route

UPDATE: Hammersmith now serves the Metropolitan Route instead of its previous connections.

USE: Passengers board the Circle Northern line at the station.

USE_CUES: []

SURFACE use-source: J=0.308, overlap_min=0.667, LCS=3 `the circle northern`; use-update LCS=1 `the`

STATE_TERMS source=['locat', 'circle', 'northern', 'line'] new=['serv', 'metropolitan', 'route'] hits_update=['serv', 'metropolitan', 'route'] hits_use_new=[] hits_use_source=['circle', 'northern', 'line']

### 21. rw_031695

SOURCE: The Grand Monarch intended that in New France it should be absolutely true.

TARGET: Grand Monarch | UPDATED: New France

SOURCE_STATE: intended absolute truth in New France | NEW_STATE: chaotic disorder everywhere

UPDATE: New France descended into chaotic disorder everywhere.

USE: The Grand Monarch intended absolute truth there.

USE_CUES: []

SURFACE use-source: J=0.375, overlap_min=0.600, LCS=4 `the grand monarch intend`; use-update LCS=0 ``

STATE_TERMS source=['intend', 'absolute', 'truth', 'france'] new=['chaotic', 'disorder', 'everywhere'] hits_update=['chaotic', 'disorder', 'everywhere'] hits_use_new=[] hits_use_source=['intend', 'absolute', 'truth']

### 22. rw2s0_031887

SOURCE: 'Alone stood brave Horatius, But constant still in mind, Thrice thirty thousand foes before And the broad flood behind.' A dart had put out one eye, he was wounded in the thigh, and his work was done.

TARGET: Horatius | UPDATED: the broad flood

SOURCE_STATE: stood brave alone with thirty thousand foes | NEW_STATE: a deep river current

UPDATE: The broad flood became a roaring river current that threatened the bridge.

USE: Brave Horatius stands alone with many enemies nearby.

USE_CUES: []

SURFACE use-source: J=0.120, overlap_min=0.429, LCS=2 `brave horatiu`; use-update LCS=0 ``

STATE_TERMS source=['stood', 'brave', 'alone', 'thirty', 'thousand', 'foes'] new=['deep', 'river'] hits_update=['river'] hits_use_new=[] hits_use_source=['brave', 'alone']

### 23. rw2s0_012004

SOURCE: On March 23, 1864, Braman was promoted to Captain of company H in the 93rd New York Regiment.

TARGET: Braman | UPDATED: the 93rd New York Regiment

SOURCE_STATE: promoted to Captain of company H | NEW_STATE: dissolved in 1865

UPDATE: The 93rd New York Regiment was dissolved in 1865 after the war.

USE: Braman serves as a Captain in company H.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.750, LCS=2 `company h`; use-update LCS=1 `in`

STATE_TERMS source=['promot', 'captain', 'company'] new=['dissolv', '1865'] hits_update=['dissolv', '1865'] hits_use_new=[] hits_use_source=['captain', 'company']

### 24. rw_019497

SOURCE: Though it is no longer a church, it is still standing after all these years and is a beautiful sight to see.

TARGET: church | UPDATED: the grounds

SOURCE_STATE: beautiful sight to see | NEW_STATE: parking and picnic area

UPDATE: The grounds have been converted into a parking and picnic area for visitors.

USE: People visit the church to enjoy its beautiful sight.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.500, LCS=2 `beautiful sight`; use-update LCS=1 `the`

STATE_TERMS source=['beautiful', 'sight', 'see'] new=['park', 'picnic'] hits_update=['park', 'picnic'] hits_use_new=[] hits_use_source=['beautiful', 'sight']

### 25. rw2s1_025091

SOURCE: For a minute Jake, who was looking out of one of the windows in front toward the track, did not see her at all.

TARGET: Jake | UPDATED: Marcus

SOURCE_STATE: looking out of one of the windows in front toward the track | NEW_STATE: working on the nearby fence

UPDATE: Marcus began working on the nearby fence while Jake watched the track.

USE: Jake looks out of the front window toward the track.

USE_CUES: []

SURFACE use-source: J=0.778, overlap_min=1.000, LCS=3 `look out of`; use-update LCS=2 `the track`

STATE_TERMS source=['look', 'out', 'window', 'front', 'toward', 'track'] new=['work', 'nearby', 'fence'] hits_update=['work', 'nearby', 'fence'] hits_use_new=[] hits_use_source=['look', 'out', 'window', 'front', 'toward', 'track']

### 26. rw2s1_002252

SOURCE: Nine weeks after its release in France on 2 November 2011, it became the second biggest box office hit in France, just behind the 2008 movie "Welcome to the Sticks".

TARGET: France | UPDATED: Welcome to the Sticks

SOURCE_STATE: released the movie on 2 November 2011 | NEW_STATE: a 2009 comedy film

UPDATE: Welcome to the Sticks changed to a 2009 comedy film instead.

USE: France released a different movie on 2 November 2011.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.800, LCS=4 `on 2 november 2011`; use-update LCS=1 `a`

STATE_TERMS source=['releas', 'movie', 'november', '2011'] new=['2009', 'comedy', 'film'] hits_update=['2009', 'comedy', 'film'] hits_use_new=[] hits_use_source=['releas', 'movie', 'november', '2011']

### 27. rw2s0_023811

SOURCE: She used her platform to better failing systems around her and helped raise awareness of the sexism and homophobia surrounding Latino culture.

TARGET: Latino culture | UPDATED: the local community

SOURCE_STATE: surrounded by sexism and homophobia | NEW_STATE: unified support network

UPDATE: The local community formed a unified support network to help everyone.

USE: Latino culture faces sexism and homophobia from outsiders.

USE_CUES: []

SURFACE use-source: J=0.267, overlap_min=0.667, LCS=3 `sexism and homophobia`; use-update LCS=0 ``

STATE_TERMS source=['surround', 'sexism', 'homophobia'] new=['unify', 'support', 'network'] hits_update=['unify', 'support', 'network'] hits_use_new=[] hits_use_source=['sexism', 'homophobia']

### 28. rw_003826

SOURCE: The Orléans and Bragança family, maternally descended from the Brazilian branch of the House of Bragança and paternally from the French house of Orléans, is considered by monarchists to be the current Brazilian imperial family.

TARGET: the Brazilian imperial family | UPDATED: the House of Bragança

SOURCE_STATE: considered by monarchists as the current ruling lineage | NEW_STATE: active royal government of Portugal

UPDATE: The House of Bragança assumed the active royal government of Portugal instead.

USE: Monarchists recognize the Brazilian imperial family as the ruling lineage.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.571, LCS=3 `brazilian imperial family`; use-update LCS=1 `the`

STATE_TERMS source=['consider', 'monarchist', 'ruling', 'lineage'] new=['active', 'royal', 'government', 'portugal'] hits_update=['active', 'royal', 'government', 'portugal'] hits_use_new=[] hits_use_source=['monarchist', 'ruling', 'lineage']

### 29. rw_031266

SOURCE: Clavering took Kitty's hand, motioned to Florence to follow, and they went into the dining-room.

TARGET: Kitty | UPDATED: Florence

SOURCE_STATE: held Clavering's hand tightly | NEW_STATE: sitting quietly in the living-room

UPDATE: Florence left the dining-room and sat quietly in the living-room.

USE: Kitty grips Clavering's hand tightly as they enter.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.500, LCS=1 `claver`; use-update LCS=0 ``

STATE_TERMS source=['held', 'claver', 'hand', 'tightly'] new=['sitt', 'quietly'] hits_update=['quietly'] hits_use_new=[] hits_use_source=['claver', 'hand', 'tightly']

### 30. rw_025590

SOURCE: Triple H then prsented a new World Heavyweight Championship for the other brand, which was won by Seth "Freakin" Rollins at Night of Champions.

TARGET: Seth Freakin Rollins | UPDATED: the World Heavyweight Championship

SOURCE_STATE: won the World Heavyweight Championship at Night of Champions | NEW_STATE: active with a new design

UPDATE: The organization introduced a new design for the World Heavyweight Championship belt.

USE: Seth Freakin Rollins holds the World Heavyweight Championship title.

USE_CUES: []

SURFACE use-source: J=0.429, overlap_min=0.750, LCS=3 `world heavyweight championship`; use-update LCS=4 `the world heavyweight championship`

STATE_TERMS source=['won', 'world', 'heavyweight', 'championship', 'night', 'champion'] new=['active', 'design'] hits_update=['design'] hits_use_new=[] hits_use_source=['world', 'heavyweight', 'championship']

### 31. rw_009697

SOURCE: The three girls heaved a vast sigh of relief when Miss Drummond, after a tour of inspection into the kitchen and scullery, departed, expressing satisfaction both with the dinner and with the general neatness and order of the establishment.

TARGET: Miss Drummond | UPDATED: the kitchen staff

SOURCE_STATE: departed expressing satisfaction with the dinner and general neatness | NEW_STATE: replaced the dirty dishes with fresh linens

UPDATE: The kitchen staff replaced the dirty dishes with fresh linens while Miss Drummond departed.

USE: Miss Drummond expresses satisfaction with the dinner and general neatness.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=1.000, LCS=4 `with the dinner and`; use-update LCS=2 `miss drummond`

STATE_TERMS source=['depart', 'express', 'satisfaction', 'dinner', 'general', 'neatness'] new=['replac', 'dirty', 'dish', 'fresh', 'linen'] hits_update=['replac', 'dirty', 'dish', 'fresh', 'linen'] hits_use_new=[] hits_use_source=['express', 'satisfaction', 'dinner', 'general', 'neatness']

### 32. rw2s1_017250

SOURCE: "Skin Folk" received the World Fantasy Award and the Sunburst Award for Canadian Literature of the Fantastic in 2003.

TARGET: Skin Folk | UPDATED: the Sunburst Award

SOURCE_STATE: received the World Fantasy Award in 2003 | NEW_STATE: honors works of Canadian fantasy literature

UPDATE: The Sunburst Award shifted its focus to honor works of Canadian fantasy literature.

USE: Skin Folk won the World Fantasy Award back in 2003.

USE_CUES: []

SURFACE use-source: J=0.462, overlap_min=0.750, LCS=4 `the world fantasy award`; use-update LCS=1 `the`

STATE_TERMS source=['receiv', 'world', 'fantasy', 'award', '2003'] new=['honor', 'work', 'canadian', 'fantasy', 'literature'] hits_update=['honor', 'work', 'canadian', 'fantasy', 'literature'] hits_use_new=[] hits_use_source=['world', 'fantasy', 'award', '2003']

### 33. rw2s1_003785

SOURCE: In addition to the State Assembly Chamber, it has the State Senate Chamber and the governor's office (also called the Red Room).

TARGET: State Assembly Chamber | UPDATED: governor's office

SOURCE_STATE: located with the State Senate Chamber | NEW_STATE: renamed the Blue Room

UPDATE: The governor's office was renamed the Blue Room during the renovation.

USE: Legislators meet in the State Assembly Chamber beside the Senate.

USE_CUES: []

SURFACE use-source: J=0.300, overlap_min=0.500, LCS=4 `the state assembly chamber`; use-update LCS=1 `the`

STATE_TERMS source=['locat', 'senate', 'chamber'] new=['renam', 'blue'] hits_update=['renam', 'blue'] hits_use_new=[] hits_use_source=['senate', 'chamber']

### 34. rw2s0_017722

SOURCE: This one has got just about every secondary school in Oxford, county secondary school on it, so it's just where the schools are really.

TARGET: Oxford | UPDATED: the county secondary school

SOURCE_STATE: has county secondary schools on it | NEW_STATE: closed for renovations

UPDATE: The county secondary school was closed for renovations earlier this week.

USE: Oxford has many secondary schools located nearby.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.500, LCS=2 `secondary school`; use-update LCS=2 `secondary school`

STATE_TERMS source=['county', 'secondary', 'school'] new=['clos', 'renovation'] hits_update=['clos', 'renovation'] hits_use_new=[] hits_use_source=['secondary', 'school']

### 35. rw2s0_015673

SOURCE: Even when it was passed and presented to him, the King still thought himself strong enough to discharge Balfour from his command in the Tower, and to put in his place a man of bad character; to whom the Commons instantly objected, and whom he was obliged to abandon.

TARGET: Balfour | UPDATED: the Commons representative

SOURCE_STATE: discharged from the Tower command | NEW_STATE: immediate objection to the decision

UPDATE: The Commons representative lodged an immediate objection to the decision.

USE: The King believes Balfour holds the Tower command firmly.

USE_CUES: []

SURFACE use-source: J=0.160, overlap_min=0.571, LCS=2 `the king`; use-update LCS=1 `the`

STATE_TERMS source=['discharg', 'tower', 'command'] new=['immediate', 'objection', 'decision'] hits_update=['immediate', 'objection', 'decision'] hits_use_new=[] hits_use_source=['tower', 'command']

### 36. rw2s0_027047

SOURCE: I was presented to him on his entrance by Sir Francis Vere, who with a grave countenance related how he had chosen me, as one expert in war and cunning in counsel, to assist the burghers in their extremity.

TARGET: Sir Francis Vere | UPDATED: the burghers

SOURCE_STATE: related how he had chosen him | NEW_STATE: seeking peaceful negotiation

UPDATE: The burghers abandoned their extremity to seek peaceful negotiation.

USE: Sir Francis Vere related how he had chosen an expert.

USE_CUES: []

SURFACE use-source: J=0.412, overlap_min=1.000, LCS=5 `relat how he had chosen`; use-update LCS=0 ``

STATE_TERMS source=['relat', 'how', 'chosen'] new=['seek', 'peaceful', 'negotiation'] hits_update=['seek', 'peaceful', 'negotiation'] hits_use_new=[] hits_use_source=['relat', 'how', 'chosen']

### 37. rw_022107

SOURCE: Central Berlin is being shelled, from the Brandenburg Gate to the Reichstag, up to Friedrichstrasse Station.

TARGET: Central Berlin | UPDATED: Friedrichstrasse Station

SOURCE_STATE: being shelled from the Brandenburg Gate to the Reichstag | NEW_STATE: renovated with a new glass roof

UPDATE: Friedrichstrasse Station was renovated with a new glass roof while construction elsewhere paused.

USE: Tourists walk from the Brandenburg Gate toward the Reichstag in Central Berlin.

USE_CUES: []

SURFACE use-source: J=0.400, overlap_min=0.571, LCS=4 `from the brandenburg gate`; use-update LCS=0 ``

STATE_TERMS source=['shell', 'brandenburg', 'gate', 'reichstag'] new=['renovat', 'glass', 'roof'] hits_update=['renovat', 'glass', 'roof'] hits_use_new=[] hits_use_source=['brandenburg', 'gate', 'reichstag']

### 38. rw_021821

SOURCE: "Mother said Mack could take us for a little ride in the fresh air so we would have a better appetite for dinner." This was quite exciting, for Edna's opportunities for riding in an automobile were not many.

TARGET: Mack | UPDATED: Edna

SOURCE_STATE: takes us for a ride in the fresh air | NEW_STATE: gets a bicycle for herself

UPDATE: Edna found that getting a bicycle for herself was fun.

USE: Mack rides vehicles to take people into fresh air.

USE_CUES: []

SURFACE use-source: J=0.250, overlap_min=0.714, LCS=2 `fresh air`; use-update LCS=0 ``

STATE_TERMS source=['take', 'ride', 'fresh', 'air'] new=['gets', 'bicycle', 'herself'] hits_update=['bicycle', 'herself'] hits_use_new=[] hits_use_source=['take', 'ride', 'fresh', 'air']

### 39. rw2s1_017153

SOURCE: The British women and children seized in Kuwait yesterday while trying to join a convoy to Baghdad, are now being transported in a fleet of buses to the Iraqi capital; authorities say they'll then be free to leave the country.

TARGET: Kuwait | UPDATED: Iraq

SOURCE_STATE: host of seized women and children | NEW_STATE: final destination for the convoy

UPDATE: Iraq received the convoy as the final destination for the group.

USE: Women and children traveled through Kuwait to reach Baghdad.

USE_CUES: []

SURFACE use-source: J=0.174, overlap_min=0.667, LCS=3 `women and children`; use-update LCS=0 ``

STATE_TERMS source=['host', 'seiz', 'women', 'children'] new=['final', 'destination', 'convoy'] hits_update=['final', 'destination', 'convoy'] hits_use_new=[] hits_use_source=['women', 'children']

### 40. rw_029729

SOURCE: She then came to a large lake, where there was neither ship nor boat.

TARGET: boat | UPDATED: ship

SOURCE_STATE: absent from the large lake | NEW_STATE: docked at the shore

UPDATE: A ship docked at the shore near the large lake where no boat exists.

USE: Fishermen paddle across the large lake without seeing any boat nearby.

USE_CUES: []

SURFACE use-source: J=0.231, overlap_min=0.429, LCS=2 `large lake`; use-update LCS=3 `the large lake`

STATE_TERMS source=['absent', 'large', 'lake'] new=['dock', 'shore'] hits_update=['dock', 'shore'] hits_use_new=[] hits_use_source=['large', 'lake']

### 41. rw_025455

SOURCE: We was wondering what kind of fit John Bartlett would have now.

TARGET: John Bartlett | UPDATED: the local tailor

SOURCE_STATE: wondering what fit he would have | NEW_STATE: selling custom suits

UPDATE: The local tailor started selling custom suits to everyone in town.

USE: John Bartlett looks for a better fit at the tailor.

USE_CUES: []

SURFACE use-source: J=0.375, overlap_min=0.600, LCS=2 `john bartlett`; use-update LCS=1 `the`

STATE_TERMS source=['wonder', 'fit'] new=['sell', 'custom', 'suit'] hits_update=['sell', 'custom', 'suit'] hits_use_new=[] hits_use_source=['fit']

### 42. rw_014051

SOURCE: It is the home of the Chautauqua Institution and the birthplace of the Chautauqua Movement.

TARGET: Chautauqua Institution | UPDATED: Chautauqua Movement

SOURCE_STATE: home of the Chautauqua Institution | NEW_STATE: originated in a small town

UPDATE: The Chautauqua Movement originated in a small rural town before expanding.

USE: Visitors travel to the Chautauqua Institution for its historic cultural events.

USE_CUES: []

SURFACE use-source: J=0.222, overlap_min=0.500, LCS=3 `the chautauqua institution`; use-update LCS=2 `the chautauqua`

STATE_TERMS source=['chautauqua', 'institution'] new=['originat', 'small', 'town'] hits_update=['originat', 'small', 'town'] hits_use_new=[] hits_use_source=['chautauqua', 'institution']

### 43. rw_033820

SOURCE: Er she'd held the girl in conversation at Mansfield er for quite a while and she's listened to the tape and she's convinced it's the same girl.

TARGET: Mansfield | UPDATED: a new witness

SOURCE_STATE: held the girl in conversation here | NEW_STATE: saw the tape on screen

UPDATE: A new witness saw the tape on screen instead.

USE: The girl was held in conversation at Mansfield.

USE_CUES: []

SURFACE use-source: J=0.444, overlap_min=1.000, LCS=4 `in conversation at mansfield`; use-update LCS=1 `the`

STATE_TERMS source=['held', 'girl', 'conversation', 'here'] new=['saw', 'tape', 'screen'] hits_update=['saw', 'tape', 'screen'] hits_use_new=[] hits_use_source=['held', 'girl', 'conversation']

### 44. rw2s1_024251

SOURCE: She lives somewhere by the Port, but she just sort of done a detour to come and have lunch with Jan.

TARGET: Jan | UPDATED: the detour

SOURCE_STATE: has lunch with her at the Port | NEW_STATE: a direct route to the cafe

UPDATE: She abandoned the detour and took a direct route to the cafe.

USE: Jan enjoys lunch with her at the port.

USE_CUES: []

SURFACE use-source: J=0.333, overlap_min=0.750, LCS=2 `the port`; use-update LCS=1 `the`

STATE_TERMS source=['lunch', 'port'] new=['direct', 'route', 'cafe'] hits_update=['direct', 'route', 'cafe'] hits_use_new=[] hits_use_source=['lunch', 'port']

### 45. rw2s0_028373

SOURCE: After starting his professional career at Příbram in the Czech First League, he moved on to play for Slavia Prague, before moving to Italy to play in Serie A for Udinese, as well as on loan at other clubs in the same league.

TARGET: Slavia Prague | UPDATED: Udinese

SOURCE_STATE: played in the Czech First League | NEW_STATE: competed in the Italian Serie A

UPDATE: Udinese moved from the Czech league to compete in the Italian Serie A.

USE: Slavia Prague played matches in the Czech First League.

USE_CUES: []

SURFACE use-source: J=0.316, overlap_min=0.857, LCS=5 `in the czech first league`; use-update LCS=2 `the czech`

STATE_TERMS source=['play', 'czech', 'first', 'league'] new=['compet', 'italian', 'serie'] hits_update=['italian', 'serie'] hits_use_new=[] hits_use_source=['play', 'czech', 'first', 'league']

### 46. rw2s1_012551

SOURCE: Bohdan Butenko (8 February 1931 – 14 October 2019) was a Polish cartoonist, illustrator, comic book writer and artist, as well as puppet designer.

TARGET: Bohdan Butenko | UPDATED: puppet designer

SOURCE_STATE: was a Polish cartoonist | NEW_STATE: creates modern digital art

UPDATE: The puppet designer transitioned his career to create modern digital art.

USE: Bohdan Butenko serves as a Polish cartoonist.

USE_CUES: []

SURFACE use-source: J=0.235, overlap_min=0.800, LCS=3 `a polish cartoonist`; use-update LCS=0 ``

STATE_TERMS source=['polish', 'cartoonist'] new=['creat', 'modern', 'digital', 'art'] hits_update=['modern', 'digital', 'art'] hits_use_new=[] hits_use_source=['polish', 'cartoonist']

### 47. rw2s0_011603

SOURCE: The President had no real power and was selected by the government to do ceremonies and act as the head of state.

TARGET: the President | UPDATED: the Prime Minister

SOURCE_STATE: had no real political power | NEW_STATE: holds full executive authority

UPDATE: The Prime Minister now holds full executive authority while the President serves only in ceremonial roles

USE: The President performs ceremonial duties without any real political power.

USE_CUES: []

SURFACE use-source: J=0.231, overlap_min=0.375, LCS=2 `the president`; use-update LCS=2 `the president`

STATE_TERMS source=['real', 'political', 'power'] new=['hold', 'full', 'executive', 'authority'] hits_update=['hold', 'full', 'executive', 'authority'] hits_use_new=[] hits_use_source=['real', 'political', 'power']

### 48. rw2s0_017788

SOURCE: In similar wise, we find the commercial Netherlands in the sixteenth century exhibiting practically such toleration in matters of religion as the British islands attained only much later, and after prolonged and distressing struggle.

TARGET: the Netherlands | UPDATED: the British islands

SOURCE_STATE: exhibited practically such toleration | NEW_STATE: attain this level early

UPDATE: The British islands attained this level of tolerance much earlier than expected.

USE: The Netherlands exhibits practical religious toleration in similar ways.

USE_CUES: []

SURFACE use-source: J=0.174, overlap_min=0.571, LCS=2 `in similar`; use-update LCS=1 `the`

STATE_TERMS source=['exhibit', 'practically', 'toleration'] new=['attain', 'level', 'early'] hits_update=['attain', 'level'] hits_use_new=[] hits_use_source=['exhibit', 'toleration']

### 49. rw_026769

SOURCE: With the support of Nazi Germany and fascist Italy, the Independent State of Croatia (ISC) was created on 10 April 1941.

TARGET: Independent State of Croatia | UPDATED: Nazi Germany

SOURCE_STATE: created with Nazi Germany and fascist Italy support | NEW_STATE: withdrawn all military and financial aid

UPDATE: After the invasion, Nazi Germany withdrew all military and financial aid completely.

USE: The Independent State of Croatia was created with fascist Italy support.

USE_CUES: []

SURFACE use-source: J=0.545, overlap_min=1.000, LCS=5 `the independent state of croatia`; use-update LCS=1 `the`

STATE_TERMS source=['creat', 'nazi', 'germany', 'fascist', 'italy', 'support'] new=['withdrawn', 'military', 'financial', 'aid'] hits_update=['military', 'financial', 'aid'] hits_use_new=[] hits_use_source=['creat', 'fascist', 'italy', 'support']

### 50. rw_038513

SOURCE: Well, as you all know, my youngest son, Derek, couldn't be here because of an important fishing trip.

TARGET: Derek | UPDATED: my mother

SOURCE_STATE: travels on an important fishing trip | NEW_STATE: stays home to cook dinner

UPDATE: My mother decided to stay home instead to prepare a delicious dinner for everyone.

USE: Derek enjoys his time traveling on the fishing trip.

USE_CUES: []

SURFACE use-source: J=0.214, overlap_min=0.500, LCS=2 `fish trip`; use-update LCS=0 ``

STATE_TERMS source=['travel', 'important', 'fish', 'trip'] new=['stay', 'cook', 'dinner'] hits_update=['stay', 'dinner'] hits_use_new=[] hits_use_source=['travel', 'fish', 'trip']

