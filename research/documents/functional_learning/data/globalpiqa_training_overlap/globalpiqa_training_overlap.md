# earlier analysis GlobalPIQA flip overlap with dense training prefix

Scanned first `20475` rows of `experiments/archive/functional_learning/data/unchanged_pair_segments_tail/reference_tail_unchanged_qwen_segments.jsonl`, matching the 80-update dense-focus legal prefix. This is lexical/provenance evidence only; it does not prove semantic learning or its absence.

## Summary
- Shared GlobalPIQA dense gains inspected: `4`.
- Shared GlobalPIQA dense losses inspected: `1`.
- Exact normalized prompt/target/prediction phrase hits in the 80-update prefix: `0` across all inspected flips.
- High lexical-overlap rows are listed below. Treat them as possible thematic exposure, not as confirmed benchmark leakage or causal explanation.

## shared_dense_gains

### `GlobalPIQA_parallel::parallel_ex000000_eng_latn` pattern `011`
Prompt: A plastic bag is filled with air and then sealed. When an object is placed on the bag, what happens?
Target: The amount of air in the bag stays the same
Parent pred: The amount of air in the bag increases
Dense pred: The amount of air in the bag stays the same
Query terms: `amount, filled, increases, object, plastic, sealed, stays`
Exact phrase hits: `0`
Top lexical-overlap rows:
- row `2505` source `qwen_pair_packed` overlap `2` score `0.4788` terms=['filled', 'sealed'] qwen_source_terms=['sealed'] qwen_view_terms=['filled', 'sealed'] snippet="Before mothery got so ill we had our stated times, but now we're never sure when we'll be wanted. Before mothery became so ill, we had our stated times, but now we're never sure when we'll be wanted. The parole board got me into this halfway house called The Brewer, and a job bagging groceries at the Foodway. The parole board secured my placement in the half"
- row `1021` source `qwen_pair_packed` overlap `2` score `0.4222` terms=['amount', 'object'] qwen_source_terms=['object'] qwen_view_terms=['amount'] snippet="You raise this issue though, of whether or not we would wish the churches emptied on Sundays. However, this brings up the question of whether we truly desire churches to be left empty on Sundays. Arriving there, he hunted; he spent quite a while in hunting, for the object of his search was nowhere to be seen. Upon arriving, he spent a considerable amount of "
- row `15751` source `qwen_pair_packed` overlap `1` score `0.3632` terms=['amount'] qwen_source_terms=['amount'] qwen_view_terms=['amount'] snippet="An inset was at one time created for Skelton, and it resulted in a considerable amount of building on the south side. An inset was once created for Skelton, resulting in a considerable amount of building on the south side. She thought that if she could escape the death her parents had talked about, they would be delighted at her leaving them and would not lo"
- row `19972` source `childes` overlap `2` score `0.3592` terms=['object', 'plastic'] snippet="small. She finds a plastic ice-cube container] *CHI: cup. *GLO: cup? *GLO: [asks Mother for something that can fit inside the cup.] *GLO: tell me where the earring is. [inside the cup] *MOT: [gives her an ear+ring] *GLO: okay. [under the cup] *GLO: now where's the earring? [under the cup] *CHI: xxx here. [Gloria tells Sarah to do what she did with the ear+ri"
- row `6722` source `qwen_pair_packed` overlap `1` score `0.3326` terms=['filled'] qwen_source_terms=['filled'] qwen_view_terms=['filled'] snippet="It went to number 4 in the United States, number 7 in the United Kingdom, number 36 in New Zealand, number 27 in Australia and number 3 in Ireland. It reached number 4 in the United States, number 7 in the United Kingdom, number 36 in New Zealand, number 27 in Australia and number 3 in Ireland. In some places Old River was a stagnant pool, covered with thick"

### `GlobalPIQA_parallel::parallel_ex000071_eng_latn` pattern `011`
Prompt: How long is the wick of a candle, compared to the height of the candle wax?
Target: The wick should be longer than the height of the candle wax
Parent pred: The wick should be shorter than the height of the candle wax
Dense pred: The wick should be longer than the height of the candle wax
Query terms: `candle, compared, height, long, longer, shorter, wick`
Exact phrase hits: `0`
Top lexical-overlap rows:
- row `10222` source `qwen_pair_packed` overlap `2` score `0.5092` terms=['long', 'longer'] qwen_source_terms=['long', 'longer'] qwen_view_terms=['long', 'longer'] snippet="He was best known for his novels, including \"A Long Day's Dying\", \"The Book of Bebb\", \"Godric\" and \"Brendan\". His most renowned works were novels, such as \"A Long Day's Dying\", \"The Book of Bebb\", \"Godric\" and \"Brendan\". Her first instinct had been to clear their new desks by tipping the unfortunate museum on to the floor. Her initial impulse was to clear th"
- row `13189` source `qwen_pair_packed` overlap `2` score `0.5036` terms=['long', 'longer'] qwen_source_terms=['long', 'longer'] qwen_view_terms=['long', 'longer'] snippet="I B M employees rarely go to a conference, attend a meeting or take part in a course without being asked whether it meant their expectations, whether it was too long, too short, too dull, or ju or just too much. I B M employees rarely attend a conference, meeting, or course without being asked whether it met their expectations, whether it was too long, too s"
- row `5166` source `qwen_pair_packed` overlap `2` score `0.4881` terms=['long', 'shorter'] qwen_source_terms=['long', 'shorter'] qwen_view_terms=['long', 'shorter'] snippet="He caught up a long chain of bag-puddings, such as had not been seen in Bargen for many a day, and cast it in a merry sport about his neck, as it were insignia of his office. He playfully draped a long chain of bag-puddings, a sight not seen in Bargen for many days, around his neck as if it were a badge of his office. They felt sure there was a shorter way b"
- row `10207` source `qwen_pair_packed` overlap `2` score `0.4623` terms=['long', 'longer'] qwen_source_terms=['longer'] qwen_view_terms=['long'] snippet="At The Grange you'll probably find several other girls who can reel things off a little quicker.\" \"Then I shall go quicker still. At The Grange, you'll likely encounter other girls who can recite things faster; so I shall make them beat me. 'Then,' he answered, 'I have a sort of veil before my eyes, I cannot see distinctly; I always thought my wound was mort"
- row `15333` source `qwen_pair_packed` overlap `2` score `0.4586` terms=['compared', 'long'] qwen_source_terms=['compared', 'long'] qwen_view_terms=['compared', 'long'] snippet="The other girls also obtained prizes, all but Kitty Sharston, who was not long enough in the school to be entitled to one. All the other girls received prizes except Kitty Sharston, who had not been at the school long enough to qualify for one. *MOT: Peter got down very quietly off the wheelbarrow and started running as fast as he could along a straight walk"

### `GlobalPIQA_nonparallel::group0123_ex000032_eng_latn_0_v1` pattern `011`
Prompt: How do you bake small polymer clay sticks to make clay sprinkles?
Target: Place the clay sticks onto a baking sheet and bake at 350°F for one minute.
Parent pred: Place the clay sticks onto a baking sheet and bake at 350°F for twenty minutes.
Dense pred: Place the clay sticks onto a baking sheet and bake at 350°F for one minute.
Query terms: `bake, baking, clay, minute, minutes, polymer, sheet, small, sprinkles, sticks, twenty`
Exact phrase hits: `0`
Top lexical-overlap rows:
- row `10883` source `qwen_pair_packed` overlap `3` score `0.5509` terms=['minute', 'minutes', 'twenty'] qwen_source_terms=['minutes', 'twenty'] qwen_view_terms=['minute', 'twenty'] snippet="Buses are running smoothly this evening, if somewhat slowly, but on the trains from Paddington to Oxford you'll find a delay of about twenty minutes this evening. While bus service is operating smoothly albeit at a sluggish pace, the trains from Paddington to Oxford are experiencing an approximately twenty-minute delay this evening. Therefore, to avoid this "
- row `19579` source `qwen_pair_packed` overlap `2` score `0.5151` terms=['minute', 'twenty'] qwen_source_terms=['minute', 'twenty'] qwen_view_terms=['minute', 'twenty'] snippet="The boys did not think it at all out of the way for him to be in that state; they took it as they took the preparations for the public dinner, and no sense of the shame and sorrow it meant penetrated their tough ignorance of life. The boys did not consider it strange for him to be in that state; they accepted it just as they accepted the preparations for the"
- row `15871` source `qwen_pair_packed` overlap `2` score `0.4703` terms=['small', 'twenty'] qwen_source_terms=['small', 'twenty'] qwen_view_terms=['small', 'twenty'] snippet="Mildred could not make up her mind whether he were old or young, but as he remarked, in the course of conversation, that he had just returned from a fifteen-years sojourn in Ceylon, and that he had left England shortly after his twenty-first birthday, she was able to calculate his age with little difficulty. Mildred was initially unable to determine whether "
- row `9828` source `qwen_pair_packed` overlap `2` score `0.4481` terms=['sheet', 'small'] qwen_source_terms=[] qwen_view_terms=['sheet', 'small'] snippet="The four little girls found themselves in a dingy kitchen whose belongings remained as they had been left years before. Four small girls discovered a neglected kitchen where the items still lay exactly as they had years ago. you will see that we could do it;' and Stella handed the lawyer a second piece of paper, upon which, in a very neat and legible hand, t"
- row `9455` source `qwen_pair_packed` overlap `2` score `0.4357` terms=['minute', 'minutes'] qwen_source_terms=['minutes'] qwen_view_terms=['minute'] snippet="Wouldn't want anyone finding an earring that doesn't belong to the first lady in the Oval Office. It wouldn't be a good idea for anyone to discover an earring in the Oval Office that isn't the First Lady's. I wish you felt that you could talk to me about it. I wish you felt comfortable enough to confide in me. I was in the garage and you said, \"What kind of "

### `GlobalPIQA_nonparallel::group0123_ex000084_eng_latn_0_v1` pattern `011`
Prompt: How do I protect the counter from the heat of the iron ?
Target: Place a dry towel on the counter first.
Parent pred: Place a wet towel on the counter first.
Dense pred: Place a dry towel on the counter first.
Query terms: `counter, heat, iron, protect, towel`
Exact phrase hits: `0`
Top lexical-overlap rows:
- row `8784` source `qwen_pair_packed` overlap `1` score `0.3500` terms=['protect'] qwen_source_terms=['protect'] qwen_view_terms=[] snippet="In World War Two, the US Navy made a deal with the mafia to protect its ships on the waterfront. During World War Two, the US Navy struck a deal with the mafia to safeguard its vessels on the waterfront. British Airways were up 1 at 198, B P up 2 at 341, British Gas down 1 at 219, British Steel up 1 at 133, British Telecom up 1 at 296, Rolls Royce up 1 at 16"
- row `12533` source `cleanqwen_fineweb_compact_view_reinvest` overlap `2` score `0.3482` terms=['heat', 'iron'] snippet="“Thankfully, there are many opportunities for health care providers to prevent headache progression,” he said. He said many opportunities exist to prevent headache progression. Following the principle that metal sticks to metal when it is hot, we heat a five-foot long blowing iron pipe and gather molten glass just at the tip of the blowing iron in a furnace."
- row `8542` source `qwen_pair_packed` overlap `1` score `0.3461` terms=['protect'] qwen_source_terms=['protect'] qwen_view_terms=[] snippet="It went to number 1 in Canada, number 11 in the United States and number 12 in the United Kingdom. It reached number 1 in Canada, number 11 in the United States, and number 12 in the United Kingdom. It is about the struggles of the Meira Paibis to protect the people from crimes in Manipur state. It concerns the efforts of the Meira Paibis to safeguard the pe"
- row `7581` source `qwen_pair_packed` overlap `1` score `0.3425` terms=['heat'] qwen_source_terms=['heat'] qwen_view_terms=['heat'] snippet="It's in these areas that the ocean heat is lost or vented to the atmosphere as this heat vent was carried into the ocean in the lower latitudes. Ocean heat absorbed in lower latitudes is released into the atmosphere through vents in specific regions where it dissipates. I had to sell the salt merchant a girl I had just born. I was forced to sell a newborn da"
- row `8573` source `qwen_pair_packed` overlap `1` score `0.3357` terms=['heat'] qwen_source_terms=[] qwen_view_terms=['heat'] snippet="Later in October, Facebook's Oversight Board said that it will [meet or] speak with Haugen about her knowledge of the business (of Facebook) and its practices. Later in October, Facebook's Oversight Board stated that it will [meet or] speak with Haugen regarding her knowledge of the business (of Facebook) and its practices. Lots of people who had not been ab"
## shared_dense_losses

### `GlobalPIQA_parallel::parallel_ex000094_eng_latn` pattern `100`
Prompt: You put some cookies in the oven before doing some housework. What housework would not be possible to finish before the cookies need to be taken out of the oven?
Target: Drying the laundry
Parent pred: Drying the laundry
Dense pred: Folding the laundry
Query terms: `cookies, doing, drying, finish, folding, housework, laundry, need, oven, possible, taken`
Exact phrase hits: `0`
Top lexical-overlap rows:
- row `7568` source `qwen_pair_packed` overlap `3` score `0.6571` terms=['doing', 'need', 'taken'] qwen_source_terms=['doing', 'need'] qwen_view_terms=['doing', 'need', 'taken'] snippet="The various characters' owners let their characters be used for free because they thought it was a good thing that the special was doing. The owners of the various characters allowed their characters to be used for free because they believed it was beneficial that the special was doing. The fire within had gained a grip of the room, and shone behind her head"
- row `11463` source `childes` overlap `4` score `0.6325` terms=['cookies', 'doing', 'finish', 'need'] snippet="*CHI: xxx falling. *MOT: you gotta be careful. *CHI: let me get another one. *INV2: here it is. *CHI: huh? *CHI: Mommy get the xxx for a minute. *CHI: oh here. *CHI: they were right here. *CHI: then they take one of them? [Chi tries to see how many balls can hang from the bottom.] *CHI: yeah. *CHI: and put two on one. *CHI: like this. *CHI: [laughs]! *CHI: l"
- row `12150` source `childes` overlap `3` score `0.5303` terms=['doing', 'need', 'taken'] snippet="*MOT: you say a busy bulldozer yesterday. *MOT: what was the busy bulldozer doing? *CHI: doing pick dirt up. *MOT: was it pushing the dirt around too? *CHI: oh no xxx moon a pick dirt up. *MOT: \" oh no picking the dirt up.\" *CHI: moon pick dirt up. *MOT: \" moon picks the dirt up?\" *CHI: moon take Adam xxx. *MOT: when do you see the moon? *CHI: moon up up sky"
- row `15160` source `qwen_pair_packed` overlap `2` score `0.4982` terms=['possible', 'taken'] qwen_source_terms=['possible', 'taken'] qwen_view_terms=['possible', 'taken'] snippet="Beholding his plight, one of the burghers in mere kindness, or peradventure out of a licorous appetite, sought to aid him by relieving him of this part of his load; but the Burgomaster clung to it the more closely, protesting vehemently that he would not be robbed, and beseeching us to succour and sustain him. Beholding his plight, one of the burghers, motiv"
- row `3871` source `qwen_pair_packed` overlap `2` score `0.4881` terms=['need', 'taken'] qwen_source_terms=['need', 'taken'] qwen_view_terms=['need', 'taken'] snippet="It was not very often he went into the kitchen, and no one would look for him there. He rarely entered the kitchen, and consequently, no one would search for him there. This time she had no need of the precautions she had taken as she crept in the direction of the disturbing sounds, and she made no effort to conceal herself. As she crept toward the disturbin"

## Interpretation
The GlobalPIQA contribution remains a tiny, same-example official signal. Direct exact phrase matches would strengthen a benchmark-familiarity concern; purely thematic overlap is weaker and should mainly motivate not letting the GlobalPIQA scalar carry the learning-principle argument. The larger scientific evidence for dense focus remains Entity depth movement and controlled source-altered common-target behavior, while BLiMP/Supplement/EWoK costs define the repair target.
