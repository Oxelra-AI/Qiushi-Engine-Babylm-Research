# clean qwen compliance and validity selected-pair stratified spotcheck

Selected-pair file: `experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs.jsonl`
JSON sample: `experiments/archive/compact_experience/data/qwen_clean_aligned/selected_pairs_stratified_spotcheck.json`

Inspect these for: faithful equivalence, minor omission/addition, relation/negation change, speaker/coreference change, incomplete source/rewrite, or formatting corruption.

## 1. rw_021863 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'highov')

words: original=12 rewrite=13 pair=25 len_ratio=1.083 overlap=0.889 entity_recall=1.000

Original: [shows Sarah how to use scissors] *MOT: you hafta hold the paper.

Rewrite: *MOT: you hafta hold the paper* as Sarah demonstrates how to use scissors.

Entities: source=['sarah', '*mot'] rewrite=['*mot', 'sarah']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 2. rw_031252 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'midov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.333 entity_recall=1.000

Original: Oh, dear, I am so thankful to be here instead of out in the cold." "I am thankful, too.

Rewrite: Oh, I'm so grateful to be here rather than shivering in the cold," she replied, to which he agreed, "I feel the same way.

Entities: source=[] rewrite=["i'm"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 3. rw2s0_032782 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'short', 'midov')

words: original=18 rewrite=13 pair=31 len_ratio=0.722 overlap=0.444 entity_recall=1.000

Original: Her sketches and artworks gave a lot of information about what is was like in the internment camps.

Rewrite: Her artworks and sketches provided extensive details regarding life within the internment camps.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 4. rw2s1_029304 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'lowov')

words: original=18 rewrite=16 pair=34 len_ratio=0.889 overlap=0.167 entity_recall=1.000

Original: I'm not really sure, but, uh, she told me to take good care of you, and I will.

Rewrite: I'm not entirely certain, but, uh, she instructed me to look after you, and I will.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 5. rw_015747 — cohort=base source=switchboard bucket=('base', 'switchboard', 'mid', 'midov')

words: original=28 rewrite=23 pair=51 len_ratio=0.821 overlap=0.636 entity_recall=1.000

Original: I, I had a friend from England visit once at Christmas, B: and we could not get her out of the what she called the English back garden.

Rewrite: My friend from England visited for Christmas, and we couldn't get her to leave what she referred to as the English back garden.

Entities: source=['england', 'christmas', 'english'] rewrite=['england', 'christmas', 'english']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 6. rw2s0_004915 — cohort=extra_shard0 source=switchboard bucket=('extra_shard0', 'switchboard', 'mid', 'midov')

words: original=22 rewrite=21 pair=43 len_ratio=0.955 overlap=0.375 entity_recall=1.000

Original: B: But I think this, they way the whole economy is going right now, it's just not good to be in debt.

Rewrite: B: But I think that given the current state of the economy, it is simply not advisable to be in debt.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 7. rw2s0_025542 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'long', 'highov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.800 entity_recall=1.000

Original: I said Paddy what were you doing up there, he was at The Royal, are you bad then Paddy?

Rewrite: I asked Paddy what he was doing up there, noting he was at The Royal, and then inquired if he was bad then, Paddy.

Entities: source=['paddy', 'royal'] rewrite=['paddy', 'royal']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 8. rw2s1_006466 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'highov')

words: original=22 rewrite=15 pair=37 len_ratio=0.682 overlap=0.800 entity_recall=1.000

Original: The narrow road had abruptly expanded into a circular clearing, and in the midst of the clearing stood a small wooden building.

Rewrite: The narrow road suddenly widened into a circular clearing where a small wooden building stood.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 9. rw2s0_010174 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'lowov')

words: original=19 rewrite=23 pair=42 len_ratio=1.211 overlap=0.222 entity_recall=1.000

Original: *FAT: it was like part of a general solution to a problem that everybody was facing at this point.

Rewrite: *FAT: it felt as though it were a component of a comprehensive solution to an issue that everyone was confronting at this moment.

Entities: source=['*fat'] rewrite=['*fat']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 10. rw_042294 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'mid', 'lowov')

words: original=17 rewrite=13 pair=30 len_ratio=0.765 overlap=0.167 entity_recall=1.000

Original: I just realised if she'd had a facelift, any scars would have been removed with the face.

Rewrite: I only now realized that a facelift would have eliminated any facial scarring.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 11. rw_043049 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'highov')

words: original=12 rewrite=12 pair=24 len_ratio=1.000 overlap=0.833 entity_recall=1.000

Original: [sniffs] *MOT: oh xxx I know I know I have another idea.

Rewrite: [sniffs] *MOT: Oh, xxx, I figured it out; I've got another idea.

Entities: source=['*mot'] rewrite=['*mot', 'oh', "i've"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 12. rw_008776 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'mid', 'highov')

words: original=14 rewrite=12 pair=26 len_ratio=0.857 overlap=0.800 entity_recall=1.000

Original: Well We always wo we tend to go back to feet I must say.

Rewrite: I must say that we always tend to return to our feet.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 13. rw_018779 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'long', 'highov')

words: original=13 rewrite=17 pair=30 len_ratio=1.308 overlap=0.875 entity_recall=1.000

Original: Nancy Evans, died on January 26, 2024 from breast cancer at age 90.

Rewrite: Nancy Evans passed away at the age of 90 on January 26, 2024, due to breast cancer.

Entities: source=['evans', 'january'] rewrite=['evans', 'january']

Numbers: source=['26', '2024', '90'] rewrite=['90', '26', '2024']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 14. rw2s1_007615 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'long', 'midov')

words: original=19 rewrite=25 pair=44 len_ratio=1.316 overlap=0.667 entity_recall=1.000

Original: He was removed from office over his handling of the República Cromañón nightclub fire that killed over 190 people.

Rewrite: His removal from office was due to his management of the República Cromañón nightclub fire, which resulted in the deaths of more than 190 people.

Entities: source=['república', 'cromañón'] rewrite=['república', 'cromañón']

Numbers: source=['190'] rewrite=['190']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 15. rw_042889 — cohort=base source=childes bucket=('base', 'childes', 'short', 'midov')

words: original=15 rewrite=11 pair=26 len_ratio=0.733 overlap=0.500 entity_recall=0.750

Original: *MOT: oh Mommy and Daddy watch the television set when Adam went to sleep yes.

Rewrite: Yes, after Adam fell asleep, Mommy and Daddy began watching television.

Entities: source=['*mot', 'mommy', 'daddy', 'adam'] rewrite=['adam', 'mommy', 'daddy']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 16. rw2s1_006293 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'lowov')

words: original=30 rewrite=29 pair=59 len_ratio=0.967 overlap=0.222 entity_recall=1.000

Original: So I thought that if I could just mess up one of your things, just one thing, it would be enough so it wouldn't work out the way you planned.

Rewrite: So I reasoned that if I could only ruin one of your things, just a single item, it would suffice to prevent things from unfolding as you had planned.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 17. rw2s1_017690 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'lowov')

words: original=18 rewrite=17 pair=35 len_ratio=0.944 overlap=0.182 entity_recall=1.000

Original: *CHI: I like the part when he starts eating a little bit each day until he got sick.

Rewrite: *CHI: I enjoy the segment where he begins consuming a small amount daily until he became ill.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 18. rw2s1_028081 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'long', 'midov')

words: original=21 rewrite=27 pair=48 len_ratio=1.286 overlap=0.667 entity_recall=1.000

Original: For him there was but one Jeffreys in the universe, and he jumped at any straw of hope of finding him.

Rewrite: For him, there existed only a single Jeffreys in the entire universe, and he eagerly seized upon any hint of hope that might lead to finding him.

Entities: source=['jeffreys'] rewrite=['jeffreys']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 19. rw2s0_020726 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'highov')

words: original=25 rewrite=23 pair=48 len_ratio=0.920 overlap=0.765 entity_recall=1.000

Original: "Ha'olam Haze", where Stalag ads appeared every week alongside reports of the trial, gave the scandal full coverage and devoted two back covers to it.

Rewrite: "Ha'olam Haze", which featured Stalag ads weekly alongside trial reports, provided full coverage of the scandal and allocated two back covers to it.

Entities: source=['haze', 'stalag'] rewrite=['haze', 'stalag']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 20. rw2s0_000958 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'lowov')

words: original=28 rewrite=17 pair=45 len_ratio=0.607 overlap=0.222 entity_recall=1.000

Original: She was not nearly big enough to be called ‘lady,’ for she was still very young, and she knew quite well that she was not beautiful at all.

Rewrite: She was far too young to be termed 'lady' and fully aware that she lacked beauty entirely.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 21. rw_006553 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'mid', 'midov')

words: original=19 rewrite=18 pair=37 len_ratio=0.947 overlap=0.545 entity_recall=1.000

Original: England was chosen to host the UEFA Women's Euro 2022, so they automatically got a spot in the tournament.

Rewrite: By being selected to host the UEFA Women's Euro 2022, England secured an automatic qualification for the competition.

Entities: source=['uefa', "women's", 'euro'] rewrite=['uefa', "women's", 'euro', 'england']

Numbers: source=['2022'] rewrite=['2022']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 22. rw_041285 — cohort=base source=childes bucket=('base', 'childes', 'short', 'midov')

words: original=32 rewrite=23 pair=55 len_ratio=0.719 overlap=0.462 entity_recall=1.000

Original: *CHI: [places another silver man on top of the tower; it falls; places another silver man on top of the tower again; it falls, as do the two silver sticks] *MOT: [laughs].

Rewrite: *MOT: [laughs] as *CHI repeatedly attempts to stack silver men on the tower, only for the figures and sticks to topple each time.

Entities: source=['*chi', '*mot'] rewrite=['*mot', '*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 23. rw2s0_013505 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'midov')

words: original=35 rewrite=33 pair=68 len_ratio=0.943 overlap=0.533 entity_recall=1.000

Original: She related to him every occurrence of her daily life, all details of his father's conduct except disagreeable ones, and her letters always ended with an urgent request that he would come and visit them.

Rewrite: She recounted to him every event of her daily life, omitted only the disagreeable aspects of his father's conduct, and consistently concluded her letters with an urgent plea for him to visit them.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 24. rw2s0_026029 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'highov')

words: original=25 rewrite=29 pair=54 len_ratio=1.160 overlap=0.933 entity_recall=1.000

Original: I know I'm not the reason my parents split up and my father left, but when you're a kid,that's howt feels,you know,like it's your fault.

Rewrite: I know I'm not the reason my parents split up and my father left, but when you're a kid, that's how it feels, you know, like it's your fault.

Entities: source=["i'm"] rewrite=["i'm"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 25. rw_021351 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'short', 'midov')

words: original=27 rewrite=16 pair=43 len_ratio=0.593 overlap=0.444 entity_recall=1.000

Original: I am feeling as friendly as friendly as can be, and the mothers want their children to come away from me and to go into foolish houses.

Rewrite: Despite my utmost friendliness, the mothers insist that their children leave me and enter foolish houses.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 26. rw_006715 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'long', 'midov')

words: original=13 rewrite=17 pair=30 len_ratio=1.308 overlap=0.571 entity_recall=1.000

Original: Hundreds of thousands of people have been displaced since the beginning of 2017.

Rewrite: Since the start of 2017, hundreds of thousands of individuals have been forced to leave their homes.

Entities: source=[] rewrite=[]

Numbers: source=['2017'] rewrite=['2017']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 27. rw_015021 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'lowov')

words: original=16 rewrite=13 pair=29 len_ratio=0.812 overlap=0.143 entity_recall=1.000

Original: She seems to speak more quickly, with far more running on from one word to another.

Rewrite: It appears she is talking faster, constantly linking one word to the next.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 28. rw_036566 — cohort=base source=childes bucket=('base', 'childes', 'short', 'highov')

words: original=18 rewrite=13 pair=31 len_ratio=0.722 overlap=0.778 entity_recall=0.800

Original: ["Hi Diddle Diddle" is played then everyone joins in on "Old King Cole"] *MOT: where are you going?

Rewrite: Everyone joins in singing "Old King Cole" after "Hi Diddle Diddle" is played.

Entities: source=['diddle', 'old', 'king', 'cole', '*mot'] rewrite=['old', 'king', 'cole', 'hi', 'diddle']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 29. rw_003173 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'short', 'highov')

words: original=14 rewrite=10 pair=24 len_ratio=0.714 overlap=0.833 entity_recall=1.000

Original: The carbonate system is also important in the control of sea water P H.

Rewrite: Sea water pH is also regulated by the carbonate system.

Entities: source=[] rewrite=['ph']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 30. rw2s0_005952 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'mid', 'midov')

words: original=25 rewrite=22 pair=47 len_ratio=0.880 overlap=0.545 entity_recall=1.000

Original: He is mainly known for saying “I tought I taw a puddy tat”, which is a babyish way of saying that he saw a cat.

Rewrite: He is primarily recognized for uttering "I tought I taw a puddy tat," a childish expression indicating that he saw a cat.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 31. rw2s1_021166 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'long', 'midov')

words: original=18 rewrite=27 pair=45 len_ratio=1.500 overlap=0.750 entity_recall=1.000

Original: He is known for portraying the role of Craig Tinker on the ITV longest-running soap opera "Coronation Street".

Rewrite: He is recognized for playing the character Craig Tinker on the ITV soap opera "Coronation Street", which holds the record for the longest-running duration in its genre.

Entities: source=['craig', 'tinker', 'itv', 'coronation', 'street'] rewrite=['craig', 'tinker', 'itv', 'coronation', 'street']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 32. rw_040698 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'highov')

words: original=14 rewrite=15 pair=29 len_ratio=1.071 overlap=0.778 entity_recall=1.000

Original: [tape recorder is shut off while Chi is upstairs in the bath+room] *MOT: okay.

Rewrite: *MOT: okay.* with Chi upstairs in the bathroom, the tape recorder has been turned off.

Entities: source=['chi', '*mot'] rewrite=['*mot', 'chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 33. rw_028652 — cohort=base source=childes bucket=('base', 'childes', 'long', 'highov')

words: original=15 rewrite=19 pair=34 len_ratio=1.267 overlap=0.778 entity_recall=1.000

Original: [indicates the other end of the horseshoe magnet] *MOT: we don't needta talk like that.

Rewrite: The other pole of the horseshoe magnet is marked by the line *MOT: we don't needta talk like that.

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 34. rw_032584 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'short', 'midov')

words: original=30 rewrite=20 pair=50 len_ratio=0.667 overlap=0.333 entity_recall=1.000

Original: Apparently, he was able to hold on to his cell phone until a few moments ago and has given her a great deal of information about what's going on inside.

Rewrite: He managed to keep his cell phone until recently and has shared extensive details with her regarding the situation inside.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 35. rw_027626 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'highov')

words: original=19 rewrite=14 pair=33 len_ratio=0.737 overlap=0.875 entity_recall=1.000

Original: FC KAMAZ Naberezhnye Chelny is an association football club based in Naberezhnye Chelny, playing in the Russian Second Division.

Rewrite: FC KAMAZ Naberezhnye Chelny, a Russian Second Division team, is based in Naberezhnye Chelny.

Entities: source=['fc', 'kamaz', 'naberezhnye', 'chelny', 'russian', 'second', 'division'] rewrite=['fc', 'kamaz', 'naberezhnye', 'chelny', 'russian', 'second', 'division']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 36. rw_003279 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'long', 'highov')

words: original=14 rewrite=18 pair=32 len_ratio=1.286 overlap=0.889 entity_recall=1.000

Original: FERA gave states and cities $3.1 billion (the equivalent of $55.4 billion in 2017).

Rewrite: States and cities received $3.1 billion from the FERA, a sum equivalent to approximately $55.4 billion in 2017.

Entities: source=['fera'] rewrite=['fera']

Numbers: source=['$3.1', '$55.4', '2017'] rewrite=['$3.1', '$55.4', '2017']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 37. rw2s0_020191 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'short', 'highov')

words: original=23 rewrite=17 pair=40 len_ratio=0.739 overlap=0.875 entity_recall=1.000

Original: I hope we will see you back again as soon as possible and that you will continue to avail yourself of our services.

Rewrite: I hope you will continue to avail yourself of our services and return as soon as possible.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 38. rw2s0_012316 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'lowov')

words: original=18 rewrite=16 pair=34 len_ratio=0.889 overlap=0.200 entity_recall=1.000

Original: No, I was just saying if we don't all end up dead maybe we could go for drink.

Rewrite: No, I was merely suggesting that if we all survive, perhaps we could have a drink.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 39. rw2s1_027760 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'short', 'highov')

words: original=30 rewrite=22 pair=52 len_ratio=0.733 overlap=1.000 entity_recall=1.000

Original: It was after Jesus was born so we call that B C, no sorry it wasn't after it was before Jesus was born we call it B C before Christ.

Rewrite: It was before Jesus was born, which is why we call it B C, or before Christ, no sorry it wasn't after.

Entities: source=['jesus', 'christ'] rewrite=['jesus', 'christ']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 40. rw2s0_028723 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'mid', 'lowov')

words: original=19 rewrite=19 pair=38 len_ratio=1.000 overlap=0.222 entity_recall=1.000

Original: Then, people could study all of the body parts, compare them, and learn more about how the body worked.

Rewrite: Then, individuals were able to examine every body part, compare them, and gain further insight into the body's functioning.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 41. rw_033833 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'long', 'midov')

words: original=13 rewrite=17 pair=30 len_ratio=1.308 overlap=0.500 entity_recall=1.000

Original: Or or or f for want of a better name at this stage.

Rewrite: Or, for the time being, let's just call it "F" as there isn't a better name yet.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 42. rw2s0_006818 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'midov')

words: original=28 rewrite=28 pair=56 len_ratio=1.000 overlap=0.500 entity_recall=1.000

Original: [1 laughs] *GMA: it looked like if we stayed on the top where we were that we were gonna end up on the other side of the arboretum.

Rewrite: [1 laughs] *GMA: it seemed as though remaining at the top where we were would have resulted in us ending up on the other side of the arboretum.

Entities: source=['*gma'] rewrite=['*gma']

Numbers: source=['1'] rewrite=['1']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 43. rw2s0_007774 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'long', 'midov')

words: original=26 rewrite=34 pair=60 len_ratio=1.308 overlap=0.643 entity_recall=1.000

Original: It was full seven years before this war ended in a treaty of peace made at Nimeguen, and its details would occupy a very considerable space.

Rewrite: It took a full seven years for this war to conclude with a peace treaty established at Nimeguen, and the specifics of that agreement would require a very considerable amount of space to detail.

Entities: source=['nimeguen'] rewrite=['nimeguen']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 44. rw2s1_015978 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'short', 'highov')

words: original=31 rewrite=23 pair=54 len_ratio=0.742 overlap=0.938 entity_recall=1.000

Original: Gherardini had two wives, Lisa di Giovanni Filippo de' Carducci, whom he married in 1465, and Caterina di Mariotto Rucellai, whom he married in 1473, but, both wives died during childbirth.

Rewrite: Gherardini married Lisa di Giovanni Filippo de' Carducci in 1465 and Caterina di Mariotto Rucellai in 1473, yet both wives died during childbirth.

Entities: source=['lisa', 'giovanni', 'filippo', 'carducci', 'caterina', 'mariotto', 'rucellai'] rewrite=['lisa', 'giovanni', 'filippo', 'carducci', 'caterina', 'mariotto', 'rucellai']

Numbers: source=['1465', '1473'] rewrite=['1465', '1473']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 45. rw2s1_005874 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'mid', 'highov')

words: original=36 rewrite=37 pair=73 len_ratio=1.028 overlap=0.913 entity_recall=1.000

Original: First, Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf; the Iraqi News Agency said the warning had come from the country's Air Force Commander.

Rewrite: First, the Iraqi News Agency reported that the country's Air Force Commander issued a warning stating that Iraq has threatened to attack Israel and Saudia Arabia with missiles and bombs if war breaks out in the Gulf.

Entities: source=['iraq', 'israel', 'saudia', 'arabia', 'gulf', 'iraqi', 'news', 'agency', 'air', 'force', 'commander'] rewrite=['iraqi', 'news', 'agency', 'air', 'force', 'commander', 'iraq', 'israel', 'saudia', 'arabia', 'gulf']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 46. rw2s1_023326 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'short', 'highov')

words: original=19 rewrite=13 pair=32 len_ratio=0.684 overlap=0.857 entity_recall=1.000

Original: *FAT: we walked between Claude's house and the grocery store and then between the grocery store and Claude's house.

Rewrite: *FAT: we walked from Claude's house to the grocery store and back again.

Entities: source=['*fat', "claude's"] rewrite=['*fat', "claude's"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 47. rw2s0_019304 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'long', 'highov')

words: original=20 rewrite=33 pair=53 len_ratio=1.650 overlap=0.889 entity_recall=1.000

Original: Manning is the son of Ed Manning, who was a longtime NBA and ABA player and professional and college coach.

Rewrite: Manning, the son of Ed Manning, was born to a man who served as a professional and college coach and had a long career as a player in both the NBA and ABA.

Entities: source=['ed', 'manning', 'nba', 'aba'] rewrite=['ed', 'manning', 'nba', 'aba']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 48. rw2s1_016362 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'lowov')

words: original=18 rewrite=21 pair=39 len_ratio=1.167 overlap=0.182 entity_recall=1.000

Original: *INV: there's no end goal or you know how many toys you can use or something like that.

Rewrite: *INV: there is no final objective, nor a specified limit on the number of toys you can utilize or anything similar.

Entities: source=['*inv'] rewrite=['*inv']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 49. rw2s1_000176 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'lowov')

words: original=19 rewrite=17 pair=36 len_ratio=0.895 overlap=0.200 entity_recall=1.000

Original: Scientists say this frog is not in danger of dying out because it lives in such a large place.

Rewrite: Scientists state that this frog faces no risk of extinction since it inhabits such an extensive area.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 50. rw2s0_026190 — cohort=extra_shard0 source=switchboard bucket=('extra_shard0', 'switchboard', 'mid', 'midov')

words: original=22 rewrite=19 pair=41 len_ratio=0.864 overlap=0.250 entity_recall=1.000

Original: A: The center that they've got now where you take your stuff in I, I think that should be making some money.

Rewrite: A: I think the center they currently have, where you drop off your items, should be generating some revenue.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 51. rw_010821 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'short', 'midov')

words: original=28 rewrite=19 pair=47 len_ratio=0.679 overlap=0.455 entity_recall=1.000

Original: It's in these areas that the ocean heat is lost or vented to the atmosphere as this heat vent was carried into the ocean in the lower latitudes.

Rewrite: Ocean heat absorbed in lower latitudes is released into the atmosphere through vents in specific regions where it dissipates.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 52. rw2s1_003887 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'highov')

words: original=49 rewrite=34 pair=83 len_ratio=0.694 overlap=0.812 entity_recall=1.000

Original: He wanted a pretext for war, and one which would appeal to his people; and what more powerful one could he have found than a religious one, that is, one in which those of the Greek Church were shown to be the martyrs, for Russia belongs to that persuasion.

Rewrite: Seeking a pretext for war that would resonate with his people, he found nothing more powerful than a religious justification portraying those of the Greek Church as martyrs, since Russia belongs to that persuasion.

Entities: source=['greek', 'church', 'russia'] rewrite=['greek', 'church', 'russia']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 53. rw2s1_016592 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'mid', 'highov')

words: original=38 rewrite=37 pair=75 len_ratio=0.974 overlap=0.857 entity_recall=1.000

Original: There was the haunting, injured look of wounded childhood on her face, and her curled lip showed that she, too, young as she was, had found that all was not good in the world, all was not beautiful.

Rewrite: Her face bore the haunting, injured look of wounded childhood, and her curled lip revealed that even though she was young, she had discovered that all was not good in the world and all was not beautiful.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 54. rw2s1_021228 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'highov')

words: original=30 rewrite=30 pair=60 len_ratio=1.000 overlap=0.938 entity_recall=1.000

Original: The film focuses on the World War II experiences of Desmond Doss, an American pacifist combat medic who did not carry or use a weapon or firearm of any kind.

Rewrite: The film centers on the World War II experiences of Desmond Doss, an American pacifist combat medic who did not carry or use a weapon or firearm of any kind.

Entities: source=['world', 'war', 'ii', 'desmond', 'doss', 'american'] rewrite=['world', 'war', 'ii', 'desmond', 'doss', 'american']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 55. rw2s0_007628 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'short', 'highov')

words: original=20 rewrite=14 pair=34 len_ratio=0.700 overlap=0.800 entity_recall=1.000

Original: They are different from turbojets ("normal" aircraft engines) because they do not have any moving parts which compress the air.

Rewrite: Unlike turbojets ("normal" aircraft engines), they lack any moving parts that compress the air.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 56. rw_026906 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'midov')

words: original=18 rewrite=13 pair=31 len_ratio=0.722 overlap=0.429 entity_recall=1.000

Original: The pupils of this frog's eyes are vertical: they go up and down and open side to side.

Rewrite: This frog's pupils are vertically oriented, moving up and down while opening laterally.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 57. rw_026586 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'lowov')

words: original=21 rewrite=27 pair=48 len_ratio=1.286 overlap=0.222 entity_recall=1.000

Original: I tell thee he would bid the king himself do a task if he chose, and, moreover, the king would obey.

Rewrite: I assure you that if the king were to choose, he would be compelled to undertake such a task, a duty to which he would unquestioningly submit.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 58. rw_038353 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'highov')

words: original=14 rewrite=19 pair=33 len_ratio=1.357 overlap=0.857 entity_recall=1.000

Original: You've been spying on me for weeks, in India, my cab, bugging my phone.

Rewrite: For weeks while I was in India, you've been surveilling me by bugging my phone and tailing my cab.

Entities: source=['india'] rewrite=['india']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 59. rw2s1_027879 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'midov')

words: original=24 rewrite=15 pair=39 len_ratio=0.625 overlap=0.455 entity_recall=1.000

Original: Josceline went first, and was followed by the stranger, who every now and then glanced back to speak a reassuring word to his dog.

Rewrite: Following Josceline, the stranger moved along, occasionally turning to offer his dog a reassuring word.

Entities: source=[] rewrite=['josceline']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 60. rw2s0_031669 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'long', 'midov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.625 entity_recall=1.000

Original: This is why it is safest not to drink any alcohol if a woman thinks she might get pregnant.

Rewrite: This is why the safest course of action is for a woman who suspects she may become pregnant to refrain from consuming any alcohol.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 61. rw2s1_022920 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'short', 'highov')

words: original=28 rewrite=20 pair=48 len_ratio=0.714 overlap=0.857 entity_recall=1.000

Original: *CHI: I had a good day at school and I love my school and I love my friends and l love my teacher and I love my school.

Rewrite: *CHI: I enjoyed a wonderful day at school, and I love my school, my friends, my teacher, and my school.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 62. rw2s1_022327 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'short', 'highov')

words: original=41 rewrite=24 pair=65 len_ratio=0.585 overlap=1.000 entity_recall=1.000

Original: You know, if it wasn't for you guys, You know, if it wasn't for you guys, I would never have known what a good son I would never have known what a good son and what a fine master RJ's become.

Rewrite: You know, if it wasn't for you guys, I would never have known what a good son and what a fine master RJ's become.

Entities: source=["rj's"] rewrite=["rj's"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 63. rw_032415 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'midov')

words: original=14 rewrite=21 pair=35 len_ratio=1.500 overlap=0.714 entity_recall=1.000

Original: I know you stole the research Jor-el surrendered it and I let you live.

Rewrite: Since I am aware that you stole the research Jor-el surrendered and I spared your life, you know what I know.

Entities: source=['jor-el'] rewrite=['jor-el']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 64. rw2s1_030707 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'mid', 'midov')

words: original=22 rewrite=23 pair=45 len_ratio=1.045 overlap=0.667 entity_recall=1.000

Original: The objectives which the objectors are anxious to see achieved are to have these two sites shown as being in the greenbelt.

Rewrite: The objectives that the objectors are eager to see realized are for these two sites to be designated as being within the greenbelt.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 65. rw2s1_009883 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'midov')

words: original=21 rewrite=21 pair=42 len_ratio=1.000 overlap=0.538 entity_recall=1.000

Original: Members of HSReps all over the country do things like join in political events and help out with Republican election campaigns.

Rewrite: Members of HSReps across the nation engage in activities such as participating in political events and assisting with Republican election campaigns.

Entities: source=['hsreps', 'republican'] rewrite=['hsreps', 'republican']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 66. rw_003012 — cohort=base source=childes bucket=('base', 'childes', 'short', 'midov')

words: original=14 rewrite=10 pair=24 len_ratio=0.714 overlap=0.286 entity_recall=1.000

Original: she had a lot of trouble sleeping at night and had many temper tantrums.

Rewrite: She suffered from severe insomnia and frequently threw temper tantrums.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 67. rw_037493 — cohort=base source=childes bucket=('base', 'childes', 'short', 'highov')

words: original=25 rewrite=16 pair=41 len_ratio=0.640 overlap=0.800 entity_recall=1.000

Original: *JAK: he made the hole right here and a hole right there and a hole right there and a hole right there and a hole.

Rewrite: *JAK: He drilled a hole right there, another one right here, and several others scattered throughout.

Entities: source=['*jak'] rewrite=['*jak']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 68. rw_039829 — cohort=base source=childes bucket=('base', 'childes', 'long', 'highov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.818 entity_recall=1.000

Original: [looks at EXP (claims it's an inside joke)] *CHI: um one slice of Swiss cheese one slice of salami.

Rewrite: *CHI: Glancing at EXP, who suggests it's an inside joke, she hesitantly orders, "Um, one slice of Swiss cheese and one slice of salami."

Entities: source=['exp', '*chi', 'swiss'] rewrite=['*chi', 'glancing', 'exp', 'um', 'swiss']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 69. rw_038656 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'midov')

words: original=14 rewrite=18 pair=32 len_ratio=1.286 overlap=0.286 entity_recall=1.000

Original: I get the feeling they weren't very close, and I think I know why.

Rewrite: It seems to me that they weren't particularly close, and I believe I understand the reason behind it.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 70. rw_019858 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'short', 'lowov')

words: original=25 rewrite=18 pair=43 len_ratio=0.720 overlap=0.222 entity_recall=1.000

Original: She didn't know how she would go about that, and she was just, like, it didn't seem fun or interesting in any way at all.

Rewrite: She was uncertain about how to proceed and found the idea utterly unappealing and devoid of any interest.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 71. rw_022137 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'midov')

words: original=21 rewrite=15 pair=36 len_ratio=0.714 overlap=0.444 entity_recall=1.000

Original: An attorney advises his client, the defendant, as to his or her rights and explains all processes of the criminal proceedings.

Rewrite: The attorney guides the defendant through their legal rights and details the entire criminal process.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 72. rw2s1_017251 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'highov')

words: original=39 rewrite=30 pair=69 len_ratio=0.769 overlap=0.947 entity_recall=1.000

Original: THE REACTIVATION These are the ships we have under construction right now: A Bull Carrier, going to Germany, which is a multi-purpose ship, and a tow ship, the grey one down there, for the "Tranzona" company, here in Argentina.

Rewrite: THE REACTIVATION Under construction right now are these ships: a multi-purpose Bull Carrier bound for Germany, and the grey tow ship down there for the "Tranzona" company here in Argentina.

Entities: source=['reactivation', 'bull', 'carrier', 'germany', 'tranzona', 'argentina'] rewrite=['reactivation', 'bull', 'carrier', 'germany', 'tranzona', 'argentina']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 73. rw_020954 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'highov')

words: original=17 rewrite=18 pair=35 len_ratio=1.059 overlap=0.900 entity_recall=1.000

Original: "I never spent a better or more profitable three months, never in my life," said Phil emphatically.

Rewrite: "Never in my life have I spent three months that were more profitable or better," Phil stated emphatically.

Entities: source=['phil'] rewrite=['phil']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 74. rw2s1_006825 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'mid', 'highov')

words: original=23 rewrite=24 pair=47 len_ratio=1.043 overlap=0.909 entity_recall=1.000

Original: His drowning had exactly the value in the child's mind that the jumping up of the little men had, neither more nor less.

Rewrite: In the child's mind, the value of his drowning was precisely equal to that of the little men jumping up, neither more nor less.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 75. rw2s1_021136 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'highov')

words: original=39 rewrite=35 pair=74 len_ratio=0.897 overlap=0.833 entity_recall=1.000

Original: It costs less and is smaller than a normal Nintendo Switch, doesn't use wires or anything and is a standalone device, but can not connect to a television like the normal Nintendo Switch and does not have detachable Joy-Con.

Rewrite: It is a standalone device that costs less and is smaller than a normal Nintendo Switch, operates without wires, cannot connect to a television like the normal Nintendo Switch, and does not have detachable Joy-Con.

Entities: source=['nintendo', 'switch', 'joy-con'] rewrite=['nintendo', 'switch', 'joy-con']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 76. rw2s1_015816 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'long', 'midov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.750 entity_recall=1.000

Original: There was no mark of a princess about her, and never had been since she began to run alone.

Rewrite: She had never shown any sign of being a princess since she started running alone, and there was no such mark about her now.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 77. rw2s1_030880 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'highov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.900 entity_recall=1.000

Original: [Chi continues to succesfully count all the way up to forty nine] *MOT: what comes after forty nine?

Rewrite: [Chi continues to successfully count all the way up to forty nine] *MOT: what comes after forty nine?

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 78. rw_010999 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'long', 'highov')

words: original=24 rewrite=31 pair=55 len_ratio=1.292 overlap=0.875 entity_recall=1.000

Original: And Quality Street Egg verdict, the box makes the eggs appear much larger than it actually is cost, a hundred gram, ninety nine .

Rewrite: Regarding the Quality Street Egg verdict, the box creates an illusion that makes the eggs seem much larger than they actually are for a cost of ninety-nine cents per hundred grams.

Entities: source=['quality', 'street', 'egg'] rewrite=['quality', 'street', 'egg']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 79. rw2s0_029418 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'long', 'highov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.769 entity_recall=1.000

Original: In it, over 3,000 priests were exiled or assassinated, churches desecrated, services mocked, nuns raped and captured priests shot.

Rewrite: In it, more than 3,000 priests faced exile or assassination, churches were desecrated, services were mocked, nuns were raped, and captured priests were shot.

Entities: source=[] rewrite=[]

Numbers: source=['3,000'] rewrite=['3,000']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 80. rw_034292 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'short', 'lowov')

words: original=17 rewrite=11 pair=28 len_ratio=0.647 overlap=0.143 entity_recall=1.000

Original: er we'll have given it him I mean on his on his total cost he'll be down.

Rewrite: Basically, once we've handed it over, his total costs will decrease.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 81. rw_033719 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'short', 'lowov')

words: original=13 rewrite=9 pair=22 len_ratio=0.692 overlap=0.167 entity_recall=1.000

Original: Because I know you, and you wouldn't risk a shot in the dark.

Rewrite: I trust you because you never gamble without certainty.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 82. rw_015280 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'mid', 'highov')

words: original=23 rewrite=19 pair=42 len_ratio=0.826 overlap=0.800 entity_recall=1.000

Original: I think the fact that housing need register, not a waiting list and I therefore ask you to reject er this motion tonight.

Rewrite: Given that the housing need register is not a waiting list, I urge you to reject this motion tonight.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 83. rw2s1_008580 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'short', 'highov')

words: original=21 rewrite=15 pair=36 len_ratio=0.714 overlap=1.000 entity_recall=1.000

Original: I think she's I think she I said Are you gonna vote Conservative like, just cos you've got a canny job.

Rewrite: I said, "I think she's gonna vote Conservative, just cos you've got a canny job."

Entities: source=['conservative'] rewrite=['conservative']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 84. rw_022610 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'long', 'highov')

words: original=12 rewrite=19 pair=31 len_ratio=1.583 overlap=0.833 entity_recall=1.000

Original: This is the Kratolsov collective farm, the biggest in the Yarislavl region.

Rewrite: Located in the Yarislavl region, this is the Kratolsov collective farm, which stands as the largest in the area.

Entities: source=['kratolsov', 'yarislavl'] rewrite=['yarislavl', 'kratolsov']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 85. rw_011556 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'short', 'midov')

words: original=14 rewrite=10 pair=24 len_ratio=0.714 overlap=0.429 entity_recall=1.000

Original: When I get near Tono I get this tight sensation deep in my chest.

Rewrite: Approaching Tono triggers a constricting feeling deep within my chest.

Entities: source=['tono'] rewrite=['tono']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 86. rw2s1_024896 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'highov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.833 entity_recall=1.000

Original: %int: 1 lengthened *MOT: when he's finished giving himself a trim he'll want a broom to sweep up.

Rewrite: %int: 1 lengthened *MOT: after he finishes giving himself a trim, he'll want a broom to sweep up.

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=['1'] rewrite=['1']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 87. rw2s0_009306 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'long', 'highov')

words: original=18 rewrite=23 pair=41 len_ratio=1.278 overlap=0.909 entity_recall=1.000

Original: The Köppen Climate Classification system shows that Coyville has a humid subtropical climate, abbreviated "Cfa" on climate maps.

Rewrite: According to the Köppen Climate Classification system, Coyville is characterized by a humid subtropical climate, which is abbreviated as "Cfa" on climate maps.

Entities: source=['köppen', 'climate', 'classification', 'coyville', 'cfa'] rewrite=['köppen', 'climate', 'classification', 'coyville', 'cfa']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 88. rw_021324 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'midov')

words: original=12 rewrite=19 pair=31 len_ratio=1.583 overlap=0.250 entity_recall=1.000

Original: By sunset the sea grew rough and people began to vanish below.

Rewrite: As the sun dipped below the horizon, the ocean turned turbulent and the people started disappearing beneath the waves.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 89. rw2s1_006318 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'midov')

words: original=55 rewrite=39 pair=94 len_ratio=0.709 overlap=0.640 entity_recall=1.000

Original: The baby tadpole seems much too fat to begin with, and sticks out in front like a little alderman; but soon he gets slimmer again, and you find that he is growing a curly tail (which no alderman ever did), and that there are tiny markings where his eyes and mouth are going to be.

Rewrite: The baby tadpole initially appears excessively fat, protruding forward like a little alderman, but he soon becomes slimmer again, revealing a curly tail (which no alderman ever did) and tiny markings indicating where his eyes and mouth will form.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 90. rw2s0_017124 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'short', 'highov')

words: original=18 rewrite=13 pair=31 len_ratio=0.722 overlap=1.000 entity_recall=1.000

Original: *FAT: what are the bad numbers the guy of the numbers of the guy who loses the game?

Rewrite: *FAT: what are the bad numbers for the guy who loses the game?

Entities: source=['*fat'] rewrite=['*fat']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 91. rw_010386 — cohort=base source=switchboard bucket=('base', 'switchboard', 'short', 'midov')

words: original=24 rewrite=13 pair=37 len_ratio=0.542 overlap=0.556 entity_recall=1.000

Original: B: And a lot of seniors, a lot of elderly people don't even take up golf until they're, you know, in their later years.

Rewrite: Many seniors do not begin playing golf until they reach their later years.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 92. rw_007078 — cohort=base source=childes bucket=('base', 'childes', 'long', 'highov')

words: original=18 rewrite=23 pair=41 len_ratio=1.278 overlap=0.833 entity_recall=1.000

Original: [waves wand over objects again; two silver men and a ball stick to wand] *CHI: one two three!

Rewrite: *CHI: one two three!* as the waves reappear over the objects, with two silver men and a ball now adhering to the wand.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 93. rw_001250 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'short', 'highov')

words: original=13 rewrite=8 pair=21 len_ratio=0.615 overlap=0.800 entity_recall=1.000

Original: It is this prodigious work of construction that we owe to Herbert Spencer.

Rewrite: We owe this monumental construction to Herbert Spencer.

Entities: source=['herbert', 'spencer'] rewrite=['herbert', 'spencer']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 94. rw_030998 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'lowov')

words: original=13 rewrite=22 pair=35 len_ratio=1.692 overlap=0.167 entity_recall=1.000

Original: Ask who wants to be a millionaire, and half the world phones up.

Rewrite: If you inquire about who desires to win a million dollars, you will find that half the globe rushes to call in.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 95. rw2s1_008171 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'long', 'highov')

words: original=21 rewrite=28 pair=49 len_ratio=1.333 overlap=0.917 entity_recall=0.750

Original: "Lez Girls", new fiction, part one of a serial lives novella by author of "Some of Her Parts", Nice one, Jenny.

Rewrite: "Nice one, Jenny," the speaker said, announcing that "Lez Girls," a new fiction novella by the author of "Some of Her Parts," is part one of a serial.

Entities: source=['girls', 'parts', 'nice', 'jenny'] rewrite=['jenny', 'lez', 'girls', 'parts']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 96. rw_002205 — cohort=base source=childes bucket=('base', 'childes', 'short', 'lowov')

words: original=15 rewrite=11 pair=26 len_ratio=0.733 overlap=0.167 entity_recall=1.000

Original: *CHI: maybe he left em up the sky and he didn't come down no more.

Rewrite: *CHI: Perhaps he ascended to the heavens and has never returned.

Entities: source=['*chi'] rewrite=['*chi', 'perhaps']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 97. rw_022469 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'mid', 'highov')

words: original=21 rewrite=19 pair=40 len_ratio=0.905 overlap=0.889 entity_recall=1.000

Original: Ferdinand was born in Graz and grew up in Carinthia, he is the third son of Emperor Ferdinand II of Habsburg.

Rewrite: As the third son of Emperor Ferdinand II of Habsburg, Ferdinand was born in Graz and raised in Carinthia.

Entities: source=['graz', 'carinthia', 'emperor', 'ferdinand', 'ii', 'habsburg'] rewrite=['emperor', 'ferdinand', 'ii', 'habsburg', 'graz', 'carinthia']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 98. rw_000763 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'highov')

words: original=19 rewrite=24 pair=43 len_ratio=1.263 overlap=0.833 entity_recall=1.000

Original: handsome face he shouted to his men to hold out, and fought like a lion beside the foremost gun.

Rewrite: Despite his handsome face, he shouted orders to his men to hold the line and fought with lion-like bravery right beside the front gun.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 99. rw2s1_003413 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'midov')

words: original=36 rewrite=37 pair=73 len_ratio=1.028 overlap=0.278 entity_recall=1.000

Original: *MOT: till Max said be still and tamed them with the magic trick of staring into all their yellow eyes without blinking once and they were frightened and called him the most wild thing of all.

Rewrite: *MOT: until Max commanded them to be still and subdued them by the magical act of gazing unblinking into every yellow eye, causing them to tremble in fear and label him the most wild creature of all.

Entities: source=['*mot', 'max'] rewrite=['*mot', 'max']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 100. rw_004148 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'short', 'highov')

words: original=19 rewrite=11 pair=30 len_ratio=0.579 overlap=1.000 entity_recall=1.000

Original: The button bag was a very large button bag, and the work bag was a very large work bag.

Rewrite: Both the button bag and the work bag were exceptionally large.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 101. rw2s0_031335 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'midov')

words: original=18 rewrite=15 pair=33 len_ratio=0.833 overlap=0.571 entity_recall=1.000

Original: *MOT: if you do not turn any pages we will never get to the end of this book.

Rewrite: *MOT: if no pages are turned, we will never reach the conclusion of this book.

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 102. rw2s0_021083 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'highov')

words: original=22 rewrite=22 pair=44 len_ratio=1.000 overlap=0.857 entity_recall=1.000

Original: *FAT: now Ross if I call this a narf if this is a narf and here's another one then what are these?

Rewrite: *FAT: now Ross, if I label this a narf when this is a narf and here's another one, then what are these?

Entities: source=['*fat', 'ross'] rewrite=['*fat', 'ross']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 103. rw_033154 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'mid', 'midov')

words: original=17 rewrite=19 pair=36 len_ratio=1.118 overlap=0.556 entity_recall=1.000

Original: The German Friends of the Earth's people told me never to do the washing on Monday mornings.

Rewrite: I was advised by members of the German Friends of the Earth to avoid washing clothes on Monday mornings.

Entities: source=['german', 'friends', "earth's", 'monday'] rewrite=['german', 'friends', 'earth', 'monday']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 104. rw2s1_024921 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'short', 'midov')

words: original=24 rewrite=17 pair=41 len_ratio=0.708 overlap=0.500 entity_recall=1.000

Original: The other Wraith didn't want anything to do with him, so he's been kind of holding a little bit of a grudge ever since.

Rewrite: Consequently, the other Wraith has been harboring a grudge ever since he refused to associate with him.

Entities: source=['wraith'] rewrite=['wraith']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 105. rw2s1_031347 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'short', 'highov')

words: original=18 rewrite=13 pair=31 len_ratio=0.722 overlap=1.000 entity_recall=1.000

Original: %add: SIS *UNC: they had a uh they had the live band upstairs and the disc jockey downstairs.

Rewrite: SIS *UNC: they had the disc jockey downstairs and the live band upstairs.

Entities: source=['sis', '*unc'] rewrite=['sis', '*unc']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 106. rw_001919 — cohort=base source=switchboard bucket=('base', 'switchboard', 'mid', 'midov')

words: original=13 rewrite=11 pair=24 len_ratio=0.846 overlap=0.375 entity_recall=1.000

Original: B: You don't, you don't really appreciate it until you're much older anyway.

Rewrite: B: You won't truly grasp its value until you're significantly older.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 107. rw2s0_003260 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'lowov')

words: original=18 rewrite=19 pair=37 len_ratio=1.056 overlap=0.182 entity_recall=1.000

Original: Wholeheartedly believing you'll come closer to discovering the secrets of the universe within the unfathomable abyss of space.

Rewrite: With complete conviction that you will draw nearer to uncovering the universe's secrets inside the incomprehensible void of space.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 108. rw_037386 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'highov')

words: original=15 rewrite=14 pair=29 len_ratio=0.933 overlap=0.857 entity_recall=1.000

Original: [holds a piece of cookie up to her mother's mouth] *CHI: you you eat it.

Rewrite: *CHI: You eat it* while holding a cookie piece up to her mother's mouth.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 109. rw_000041 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'highov')

words: original=14 rewrite=19 pair=33 len_ratio=1.357 overlap=0.778 entity_recall=1.000

Original: Around the Forum were erected temples to the gods, court-houses, and other public buildings.

Rewrite: Temples dedicated to the gods, as well as court-houses and various other public structures, were built around the Forum.

Entities: source=['forum'] rewrite=['forum']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 110. rw2s1_008856 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'highov')

words: original=22 rewrite=18 pair=40 len_ratio=0.818 overlap=1.000 entity_recall=1.000

Original: The school of fish has formed what is known as a baitball in an effort to confuse the predators and protect itself.

Rewrite: To confuse predators and protect itself, the school of fish has formed what is known as a baitball.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 111. rw2s0_023163 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'mid', 'midov')

words: original=22 rewrite=23 pair=45 len_ratio=1.045 overlap=0.750 entity_recall=1.000

Original: H S we are spending more and more on accountants and fi financial consultants and less and less on health care workers.

Rewrite: H S we are allocating an increasing amount to accountants and fi financial consultants while directing a decreasing amount to health care workers.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 112. rw2s0_030784 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'midov')

words: original=54 rewrite=39 pair=93 len_ratio=0.722 overlap=0.714 entity_recall=1.000

Original: So Daffodil as she grew up was only allowed to walk on the beds, and the other children were very jealous of her because they were only allowed to walk on the paths; and they thought what fun it would be if only they were allowed to run about on the beds just once.

Rewrite: So Daffodil, as she grew up, was permitted to walk only on the beds, which made the other children very jealous since they were restricted to the paths and wished they could run about on the beds just once.

Entities: source=['daffodil'] rewrite=['daffodil']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 113. rw2s0_003165 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'long', 'highov')

words: original=21 rewrite=33 pair=54 len_ratio=1.571 overlap=0.900 entity_recall=1.000

Original: "No," replied Nesta, "I haven't got anything with me." "Shall I read aloud to you, Miss?" "No, thank you," replied Nesta.

Rewrite: "No, thank you," replied Nesta, who had previously stated, "No," and added that she hadn't got anything with her, before answering the question "Shall I read aloud to you, Miss?" with a refusal.

Entities: source=['nesta', 'shall', 'miss', 'no'] rewrite=['nesta', 'no', 'shall', 'miss']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 114. rw2s1_018568 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'lowov')

words: original=24 rewrite=22 pair=46 len_ratio=0.917 overlap=0.182 entity_recall=1.000

Original: Here we are, all of us, basically alone, separate creatures, just circling each other, all searching for that slightest hint of a real connection.

Rewrite: Here we all are, essentially solitary and distinct beings, merely orbiting one another while each seeks the faintest sign of genuine connection.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 115. rw_028272 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'midov')

words: original=24 rewrite=34 pair=58 len_ratio=1.417 overlap=0.600 entity_recall=1.000

Original: Effie's little figure, her heavy black dress, her crêpe hat, her white cheeks and dark eyes, all appealed with great pathos to the woman.

Rewrite: The woman felt a profound sense of pathos when she beheld Effie's diminutive form, weighed down by a somber black dress, topped with a crêpe hat, and framed by pale cheeks and dark eyes.

Entities: source=[] rewrite=["effie's"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 116. rw_029782 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'midov')

words: original=14 rewrite=12 pair=26 len_ratio=0.857 overlap=0.500 entity_recall=1.000

Original: *CHI: [tries to open the top of the piano] *CHI: I can't open it.

Rewrite: *CHI: [attempts to pry open the piano lid] *CHI: It won't budge.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 117. rw2s0_005658 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'short', 'highov')

words: original=26 rewrite=14 pair=40 len_ratio=0.538 overlap=1.000 entity_recall=1.000

Original: Was the young fly so wrong in his predictions with the evil, Was the young fly so wrong in his predictions with the evil, ugly chameleon?

Rewrite: Was the young fly so wrong in his predictions with the evil, ugly chameleon?

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 118. rw2s0_029661 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'lowov')

words: original=24 rewrite=20 pair=44 len_ratio=0.833 overlap=0.222 entity_recall=1.000

Original: Her achievements in that line were limited to a donkey at the seaside, but she was not going to confess her lack of experience.

Rewrite: Although her accomplishments in that field were restricted to a donkey at the seaside, she refused to admit her inexperience.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 119. rw2s1_020227 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'mid', 'highov')

words: original=20 rewrite=20 pair=40 len_ratio=1.000 overlap=0.900 entity_recall=1.000

Original: This was the last triumph of this great commander, who had sailed and fought until he was quite worn out.

Rewrite: This was the final triumph of this great commander, who had sailed and fought until he was quite worn out.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 120. rw2s1_021424 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'short', 'midov')

words: original=19 rewrite=12 pair=31 len_ratio=0.632 overlap=0.750 entity_recall=1.000

Original: So are we are we talking er do you see this as a as a as a launch pad?

Rewrite: So, are we talking about this being seen as a launch pad?

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 121. rw2s0_014064 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'midov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.714 entity_recall=1.000

Original: *FAT: if you have a good lovely chocolate heart then you wanna be be good to other people.

Rewrite: *FAT: if you possess a fine, lovely chocolate heart, then you ought to be kind to other people.

Entities: source=['*fat'] rewrite=['*fat']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 122. rw_013598 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'midov')

words: original=27 rewrite=26 pair=53 len_ratio=0.963 overlap=0.538 entity_recall=1.000

Original: The effect on the neighbors is already so surprising that I have literally not been obliged to provide myself with a single meal since the news came.

Rewrite: Since the news arrived, I haven't even had to prepare a single meal for myself, given how astonishing the impact on the neighbors has already been.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 123. rw2s1_028651 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'short', 'highov')

words: original=19 rewrite=13 pair=32 len_ratio=0.684 overlap=0.857 entity_recall=1.000

Original: He arranges to meet with the federal attorney to confess, but before he can do that, guilt destroys him.

Rewrite: Before he can confess to the federal attorney as arranged, guilt destroys him.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 124. rw_017228 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'midov')

words: original=48 rewrite=41 pair=89 len_ratio=0.854 overlap=0.542 entity_recall=0.889

Original: An attempt was made by the Moors four years later to recover the place; but the Infantes Pedro and Henrique hurried from Portugal to succor Menezes, and drove back the besiegers; whereupon the Moors murdered their King, Abu Sayd, on whom they laid the blame of the disaster.

Rewrite: Four years later, the Moors tried to retake the location, but Infantes Pedro and Henrique rushed from Portugal to aid Menezes and repel the attackers; subsequently, the Moors killed their king, Abu Sayd, pinning the blame for the defeat on him.

Entities: source=['moors', 'infantes', 'pedro', 'henrique', 'portugal', 'menezes', 'king', 'abu', 'sayd'] rewrite=['moors', 'infantes', 'pedro', 'henrique', 'portugal', 'menezes', 'abu', 'sayd']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 125. rw_038951 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'lowov')

words: original=12 rewrite=17 pair=29 len_ratio=1.417 overlap=0.200 entity_recall=1.000

Original: I don't want the responsibility, so why don't we all split it?

Rewrite: Since I am unwilling to bear the burden, perhaps we should divide the responsibility among us all.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 126. rw2s1_026250 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'short', 'midov')

words: original=20 rewrite=13 pair=33 len_ratio=0.650 overlap=0.600 entity_recall=1.000

Original: According to WAW, there are many rules and norms which stand in the way of the women getting more freedoms.

Rewrite: According to WAW, numerous rules and norms obstruct women from gaining additional freedoms.

Entities: source=['waw'] rewrite=['waw']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 127. rw2s0_001293 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'highov')

words: original=19 rewrite=22 pair=41 len_ratio=1.158 overlap=0.818 entity_recall=1.000

Original: "He took me into supper, too, poor Steve." Grandma leaned over and laid her hand softly on her sister's.

Rewrite: "He took me into supper, too, poor Steve," Grandma said as she leaned over and gently placed her hand on her sister's.

Entities: source=['steve', 'grandma'] rewrite=['steve', 'grandma']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 128. rw2s1_007221 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'midov')

words: original=18 rewrite=20 pair=38 len_ratio=1.111 overlap=0.455 entity_recall=1.000

Original: Many immigrants had a hard time finding jobs and having money, so many just joined the manufacturing workforce.

Rewrite: Because many immigrants struggled to secure employment and earn money, a large number of them simply entered the manufacturing workforce.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 129. rw2s1_001581 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'short', 'midov')

words: original=37 rewrite=22 pair=59 len_ratio=0.595 overlap=0.700 entity_recall=1.000

Original: The plan doesn't point to Finmere as an area where there's a resumption in favour of mineral working, so what we will have to do with the Finmere proposals, is to look at it on its merits.

Rewrite: Since the plan does not indicate a resumption of mineral working in Finmere, we must evaluate the Finmere proposals on their merits.

Entities: source=['finmere'] rewrite=['finmere']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 130. rw_013512 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'highov')

words: original=14 rewrite=9 pair=23 len_ratio=0.643 overlap=0.800 entity_recall=1.000

Original: In 2019, she launched her own law firm called The Hyman Law Firm, P.A.

Rewrite: She founded The Hyman Law Firm, P.A. in 2019.

Entities: source=['hyman', 'law', 'firm', 'p.a'] rewrite=['hyman', 'law', 'firm', 'p.a']

Numbers: source=['2019'] rewrite=['2019']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 131. rw_021127 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'mid', 'highov')

words: original=18 rewrite=20 pair=38 len_ratio=1.111 overlap=0.800 entity_recall=1.000

Original: Next month he planned to retire from the army, but says he couldn't resist this call to duty.

Rewrite: Although he had planned to retire from the army next month, he admits he couldn't ignore this call to duty.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 132. rw_041143 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'short', 'midov')

words: original=29 rewrite=20 pair=49 len_ratio=0.690 overlap=0.636 entity_recall=1.000

Original: You know, honestly, when Sergeant Gabriel came up with the idea for these transfers, to save your section from budget cuts, by the way, I thought you'd be happy.

Rewrite: Honestly, when Sergeant Gabriel proposed these transfers to shield your section from budget cuts, I assumed you would be pleased.

Entities: source=['sergeant', 'gabriel'] rewrite=['sergeant', 'gabriel']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 133. rw_023108 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'long', 'highov')

words: original=17 rewrite=23 pair=40 len_ratio=1.353 overlap=0.800 entity_recall=1.000

Original: Straight onto a pallet, palletize it, wrap it, forklift it, put it in, stack it three high.

Rewrite: Load the items directly onto a pallet, then palletize, wrap, lift with a forklift, place them inside, and stack them three units high.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 134. rw2s0_021316 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'short', 'midov')

words: original=19 rewrite=13 pair=32 len_ratio=0.684 overlap=0.714 entity_recall=1.000

Original: [Family discussion] *MAR: the things that is hard to do is to get in and out with the cartridges.

Rewrite: [Family discussion] *MAR: what is difficult is getting the cartridges in and out.

Entities: source=['*mar'] rewrite=['*mar']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 135. rw_032696 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'short', 'highov')

words: original=27 rewrite=19 pair=46 len_ratio=0.704 overlap=0.800 entity_recall=1.000

Original: Chair, the point about the schools is that we could put that information in the minutes for the next meeting, schools that are participating in the survey.

Rewrite: Chair, we could include details about the schools participating in the survey in the minutes for the next meeting.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 136. rw2s0_008678 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'short', 'midov')

words: original=49 rewrite=34 pair=83 len_ratio=0.694 overlap=0.579 entity_recall=1.000

Original: Now, is it the intention of the people who asked us to take this action that we should just boycott Nestles products, or we should do the whole range of the conglomerate that owns all these people like Rowntrees and Cross and Blackwell, and many other companies as well.

Rewrite: Now, is the intention of those who requested this action that we merely boycott Nestles products, or that we target the entire conglomerate owning entities such as Rowntrees, Cross, Blackwell, and numerous other companies?

Entities: source=['nestles', 'rowntrees', 'cross', 'blackwell'] rewrite=['nestles', 'rowntrees', 'cross', 'blackwell']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 137. rw_041428 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'lowov')

words: original=20 rewrite=17 pair=37 len_ratio=0.850 overlap=0.143 entity_recall=1.000

Original: *INV: so you know you could tell him what you liked about it or what you didn't like about it.

Rewrite: *INV: Therefore, you are aware that he can be informed regarding your preferences, both positive and negative.

Entities: source=['*inv'] rewrite=['*inv', 'therefore']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 138. rw_001184 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'highov')

words: original=40 rewrite=39 pair=79 len_ratio=0.975 overlap=0.773 entity_recall=0.778

Original: (This file was produced from images generously made available by The Internet Archive.) A CENTURY OF SCIENCE And Other Essays BY JOHN FISKE � _Out of the shadows of night The world rolls into light: It is daybreak everywhere._ LONGFELLOW.

Rewrite: A CENTURY OF SCIENCE And Other Essays BY JOHN FISKE, featuring images generously provided by The Internet Archive, opens with LONGFELLOW's verse: "Out of the shadows of night / The world rolls into light: / It is daybreak everywhere."

Entities: source=['internet', 'archive', 'century', 'science', 'essays', 'john', 'fiske', '_out', 'longfellow'] rewrite=['century', 'science', 'essays', 'john', 'fiske', 'internet', 'archive', "longfellow's", 'out']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 139. rw_030729 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'highov')

words: original=14 rewrite=18 pair=32 len_ratio=1.286 overlap=0.818 entity_recall=1.000

Original: Two bathroom doors, both locked, employees lounge and lockers, warehouse space for additional inventory.

Rewrite: Employees are locked in the lounge and two bathroom doors, while warehouse space remains available for extra inventory.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 140. rw2s0_021729 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'mid', 'lowov')

words: original=18 rewrite=15 pair=33 len_ratio=0.833 overlap=0.182 entity_recall=1.000

Original: She developed many different methods and new types of medicines to help perform more successful surgeries on horses.

Rewrite: She created numerous distinct techniques and novel medicinal formulations to facilitate more successful equine surgeries.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 141. rw2s1_008615 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'midov')

words: original=21 rewrite=20 pair=41 len_ratio=0.952 overlap=0.583 entity_recall=1.000

Original: Mbah Gotho's recollection of his birth date was uncertain, but he vividly remembered the construction of a sugar factory in 1890.

Rewrite: Although Mbah Gotho was unsure about his birth date, he clearly recalled the construction of a sugar factory in 1890.

Entities: source=["gotho's"] rewrite=['mbah', 'gotho']

Numbers: source=['1890'] rewrite=['1890']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 142. rw_036247 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'midov')

words: original=16 rewrite=15 pair=31 len_ratio=0.938 overlap=0.400 entity_recall=0.800

Original: [MOT looks fondly at CHI] *MOT: this is the end of our tour said Nurse Spinner.

Rewrite: As Nurse Spinner spoke of the tour's conclusion, MOT cast a fond glance at CHI.

Entities: source=['mot', 'chi', '*mot', 'nurse', 'spinner'] rewrite=['nurse', 'spinner', 'mot', 'chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 143. rw_024584 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'highov')

words: original=18 rewrite=11 pair=29 len_ratio=0.611 overlap=0.875 entity_recall=0.750

Original: It is the story of a trial lawyer named Ben Holiday who gets a catalog from Rosen's, LTD.

Rewrite: Ben Holiday, a trial lawyer, receives a catalog from Rosen's, LTD.

Entities: source=['ben', 'holiday', "rosen's", 'ltd'] rewrite=['holiday', "rosen's", 'ltd']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 144. rw2s1_029404 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'midov')

words: original=19 rewrite=19 pair=38 len_ratio=1.000 overlap=0.571 entity_recall=1.000

Original: No, and I think Tilly knew exactly who Mr Jackson was, and maybe even what he was up to.

Rewrite: No, and I believe Tilly knew precisely who Mr Jackson was, and perhaps even what he was up to.

Entities: source=['tilly', 'mr', 'jackson'] rewrite=['tilly', 'mr', 'jackson']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 145. rw2s0_028184 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'short', 'highov')

words: original=30 rewrite=18 pair=48 len_ratio=0.600 overlap=1.000 entity_recall=1.000

Original: Also the observations in all parts of the global ocean from ships, from moorings, from drifters on the surface of the ocean and floats in the interior of the ocean.

Rewrite: Also the observations from ships, moorings, surface drifters, and interior floats across all parts of the global ocean.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 146. rw2s0_003305 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'lowov')

words: original=19 rewrite=17 pair=36 len_ratio=0.895 overlap=0.222 entity_recall=1.000

Original: It's one of the first things they do when they think they've got a hostage situation on their hands.

Rewrite: It is among the initial actions they take upon believing they are dealing with a hostage situation.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 147. rw2s1_013587 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'highov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.909 entity_recall=1.000

Original: Her husband, Edwin Legarda, died on December 16 in a hospital in Popayán after receiving three rifle shots.

Rewrite: Her husband, Edwin Legarda, received three rifle shots and died on December 16 in a hospital in Popayán.

Entities: source=['edwin', 'legarda', 'december', 'popayán'] rewrite=['edwin', 'legarda', 'december', 'popayán']

Numbers: source=['16'] rewrite=['16']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 148. rw2s0_007402 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'highov')

words: original=29 rewrite=18 pair=47 len_ratio=0.621 overlap=0.769 entity_recall=1.000

Original: The ornament was in the shape of a pansy; its purple leaves were of amethyst, the yellow of topaz, and in the middle lay a diamond drop of dew.

Rewrite: The pansy-shaped ornament featured amethyst-colored purple leaves, topaz-yellow hues, and a diamond drop of dew at its center.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 149. rw_004683 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'long', 'midov')

words: original=12 rewrite=16 pair=28 len_ratio=1.333 overlap=0.333 entity_recall=1.000

Original: He captained the side in their 5–1 win against Liverpool in 1966.

Rewrite: In 1966, he led his team to a 5–1 victory over Liverpool while serving as captain.

Entities: source=['liverpool'] rewrite=['liverpool']

Numbers: source=['5', '1', '1966'] rewrite=['1966', '5', '1']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 150. rw2s1_020687 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'highov')

words: original=26 rewrite=26 pair=52 len_ratio=1.000 overlap=0.917 entity_recall=1.000

Original: There was a feeling from General Westmoreland all the way on down, like we were doing an awful lot of work and not getting much return.

Rewrite: There was a sentiment from General Westmoreland all the way on down that we were doing an awful lot of work and not getting much return.

Entities: source=['general', 'westmoreland'] rewrite=['general', 'westmoreland']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 151. rw_017271 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'mid', 'lowov')

words: original=14 rewrite=14 pair=28 len_ratio=1.000 overlap=0.167 entity_recall=1.000

Original: She announced in 2013 that she will run for president in the upcoming elections.

Rewrite: In 2013, she declared her intention to seek the presidency in the next election.

Entities: source=[] rewrite=[]

Numbers: source=['2013'] rewrite=['2013']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 152. rw2s1_019558 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'lowov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.167 entity_recall=1.000

Original: *INV: the most important part is that we're actually gonna collect them all and send 'em to him.

Rewrite: *INV: the most crucial aspect is that we will indeed gather them all and dispatch them to him.

Entities: source=['*inv'] rewrite=['*inv']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 153. rw2s0_013315 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'long', 'midov')

words: original=18 rewrite=23 pair=41 len_ratio=1.278 overlap=0.625 entity_recall=1.000

Original: *MOT: think you can put that on top of that and make it stand up think it'll stay?

Rewrite: *MOT: Do you think you can place that on top of that to make it stand upright and believe it will remain stable?

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 154. rw2s1_028618 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'highov')

words: original=24 rewrite=16 pair=40 len_ratio=0.667 overlap=0.800 entity_recall=1.000

Original: Her name was Dolly, and she took my grandparents to church every Sunday for many years, up to a little while before she died.

Rewrite: For many years, until just before her death, Dolly took my grandparents to church every Sunday.

Entities: source=['dolly', 'sunday'] rewrite=['dolly', 'sunday']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 155. rw2s0_015345 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'midov')

words: original=39 rewrite=31 pair=70 len_ratio=0.795 overlap=0.667 entity_recall=1.000

Original: She hated to be on ill terms with anybody, and especially with Alice, of whom she was fond; and as she went forward and swung herself lightly up beside her, she forgot for the moment everything that was unpleasant.

Rewrite: She detested being on ill terms with anyone, particularly with Alice, whom she loved, so as she moved forward and swung herself lightly up beside her, she momentarily forgot everything unpleasant.

Entities: source=['alice'] rewrite=['alice']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 156. rw_006486 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'mid', 'midov')

words: original=18 rewrite=22 pair=40 len_ratio=1.222 overlap=0.455 entity_recall=1.000

Original: Due to the massive migration of Polish people since, the Polish community is deeply anchored in present society.

Rewrite: As a result of the substantial migration of Poles since that time, the Polish community has become firmly established within contemporary society.

Entities: source=['polish'] rewrite=['poles', 'polish']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 157. rw_002039 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'highov')

words: original=12 rewrite=17 pair=29 len_ratio=1.417 overlap=0.833 entity_recall=1.000

Original: A flash of green and crimson light, and something settled under her.

Rewrite: Beneath her, something had taken form as a flash of green and crimson light illuminated the space.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 158. rw2s0_029355 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'short', 'highov')

words: original=44 rewrite=31 pair=75 len_ratio=0.705 overlap=1.000 entity_recall=1.000

Original: *CHI: I want listen to the Christmas tape and I wanna listen to my Christmas tape in your tape recorder and I wanna read my Christmas book and then I wanna read my Charlie Brown Christmas book and listen to my Charlie Brown tape.

Rewrite: *CHI: I want to listen to my Christmas tape in your tape recorder, read my Christmas book and then my Charlie Brown Christmas book, and listen to my Charlie Brown tape.

Entities: source=['*chi', 'christmas', 'charlie', 'brown'] rewrite=['*chi', 'christmas', 'charlie', 'brown']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 159. rw2s1_000511 — cohort=extra_shard1 source=childes bucket=('extra_shard1', 'childes', 'mid', 'highov')

words: original=19 rewrite=20 pair=39 len_ratio=1.053 overlap=0.857 entity_recall=1.000

Original: *MOT: he wanted you to put your arm out so that you could hold him like you're holding Ross.

Rewrite: *MOT: he wanted you to extend your arm so that you could hold him as if you were holding Ross.

Entities: source=['*mot', 'ross'] rewrite=['*mot', 'ross']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 160. rw_024228 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'mid', 'lowov')

words: original=13 rewrite=11 pair=24 len_ratio=0.846 overlap=0.167 entity_recall=1.000

Original: It's just a matter of settling into the habit of writing every day.

Rewrite: The key is simply to cultivate the routine of writing daily.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 161. rw_008574 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'mid', 'highov')

words: original=12 rewrite=12 pair=24 len_ratio=1.000 overlap=0.857 entity_recall=1.000

Original: Before the next number, we'd like to call up a special guest.

Rewrite: We would like to introduce a special guest before the next number.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 162. rw_009913 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'mid', 'highov')

words: original=15 rewrite=12 pair=27 len_ratio=0.800 overlap=0.900 entity_recall=1.000

Original: Font Book allows users to install, manage, preview and organize fonts on their Mac devices.

Rewrite: Font Book enables Mac users to install, manage, preview, and organize fonts.

Entities: source=['book', 'mac'] rewrite=['book', 'mac']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 163. rw2s1_006733 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'mid', 'highov')

words: original=22 rewrite=23 pair=45 len_ratio=1.045 overlap=0.917 entity_recall=1.000

Original: In 2016, the Center for Inquiry merged with the Richard Dawkins Foundation for Reason and Science, keeping one office in Washington DC.

Rewrite: In 2016, the Center for Inquiry merged with the Richard Dawkins Foundation for Reason and Science, retaining a single office in Washington DC.

Entities: source=['center', 'inquiry', 'richard', 'dawkins', 'foundation', 'reason', 'science', 'washington', 'dc'] rewrite=['center', 'inquiry', 'richard', 'dawkins', 'foundation', 'reason', 'science', 'washington', 'dc']

Numbers: source=['2016'] rewrite=['2016']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 164. rw_035941 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'midov')

words: original=19 rewrite=18 pair=37 len_ratio=0.947 overlap=0.667 entity_recall=1.000

Original: [puts another silver man on the silver sticks across from the man he has just placed there] *CHI: probably.

Rewrite: *CHI: Probably.* (He places another silver man on the silver sticks opposite the one he just set there.)

Entities: source=['*chi'] rewrite=['*chi', 'probably.*']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 165. rw2s0_003001 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'long', 'midov')

words: original=20 rewrite=26 pair=46 len_ratio=1.300 overlap=0.636 entity_recall=1.000

Original: On September 25, 2015, John Boehner announced his intention to resign as speaker of the House effective October 30, 2015.

Rewrite: On September 25, 2015, John Boehner declared his plan to step down as speaker of the House, with his resignation taking effect on October 30, 2015.

Entities: source=['september', 'john', 'boehner', 'house', 'october'] rewrite=['september', 'john', 'boehner', 'house', 'october']

Numbers: source=['25', '2015', '30', '2015'] rewrite=['25', '2015', '30', '2015']

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 166. rw_010696 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'lowov')

words: original=32 rewrite=18 pair=50 len_ratio=0.562 overlap=0.200 entity_recall=1.000

Original: In the past, these kinds of calls could only be done with a operator, who would need to ask the person receiving the call if they want to be charged for it.

Rewrite: Previously, placing such calls required an operator to confirm with the recipient whether they wished to be billed.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 167. rw2s1_024583 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'long', 'highov')

words: original=18 rewrite=23 pair=41 len_ratio=1.278 overlap=0.833 entity_recall=1.000

Original: For winning, they got a match at the "Clash of Champions" pay-per-view for the Women's Tag Team Championship.

Rewrite: As a reward for their victory, they were awarded a match at the "Clash of Champions" pay-per-view for the Women's Tag Team Championship.

Entities: source=['clash', 'champions', "women's", 'tag', 'team', 'championship'] rewrite=['clash', 'champions', "women's", 'tag', 'team', 'championship']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 168. rw2s0_014694 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'lowov')

words: original=19 rewrite=18 pair=37 len_ratio=0.947 overlap=0.200 entity_recall=1.000

Original: Do you want to tell me what happened to your dad or would you like me to do it?

Rewrite: Do you wish to share what occurred to your dad, or would you prefer that I recount it?

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 169. rw2s0_007657 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'midov')

words: original=19 rewrite=14 pair=33 len_ratio=0.737 overlap=0.625 entity_recall=1.000

Original: Warburton left them, wondering what the effect of the conversation would be on the minds of her young guests.

Rewrite: Warburton departed, pondering how the conversation would impact the minds of her young guests.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 170. rw2s0_016796 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'mid', 'midov')

words: original=30 rewrite=26 pair=56 len_ratio=0.867 overlap=0.733 entity_recall=1.000

Original: The estate, which has become notorious for high speed races involving stolen cars, and recently attacks on police officers, means that they er will sending increasing numbers of police there.

Rewrite: Due to the estate's notoriety for high-speed races with stolen cars and recent attacks on police officers, they will be sending increasing numbers of police there.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 171. rw2s1_015986 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'short', 'midov')

words: original=26 rewrite=19 pair=45 len_ratio=0.731 overlap=0.357 entity_recall=1.000

Original: A lot of countries that use this system have a coalition government, meaning many parties are in control, and they all work together to make laws.

Rewrite: Many nations employing this system operate under coalition governments, where numerous parties share control and collaborate to enact legislation.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 172. rw_036228 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'mid', 'midov')

words: original=19 rewrite=20 pair=39 len_ratio=1.053 overlap=0.455 entity_recall=1.000

Original: It was an air force base back in the cold war that-- look, I don't know what they're doing.

Rewrite: That facility, once an air force base during the Cold War, remains a mystery as to the current activities there.

Entities: source=[] rewrite=['cold', 'war']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 173. rw2s0_013565 — cohort=extra_shard0 source=switchboard bucket=('extra_shard0', 'switchboard', 'mid', 'highov')

words: original=21 rewrite=21 pair=42 len_ratio=1.000 overlap=0.857 entity_recall=1.000

Original: A: And I try to cover up when I do the lawn B: Well, I, that, that is really healthier, frankly.

Rewrite: B: Well, I, that, that is really healthier, frankly, in contrast to A's attempt to cover up when doing the lawn.

Entities: source=['well'] rewrite=['well', "a's"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 174. rw2s1_029990 — cohort=extra_shard1 source=switchboard bucket=('extra_shard1', 'switchboard', 'mid', 'highov')

words: original=18 rewrite=14 pair=32 len_ratio=0.778 overlap=1.000 entity_recall=1.000

Original: A: So, uh, B: Yes, B: uh, that sounds like a good, that sounds like the right theory.

Rewrite: B: Yes, uh, that sounds like a good, that sounds like the right theory.

Entities: source=['yes'] rewrite=['yes']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 175. rw_003133 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'lowov')

words: original=20 rewrite=26 pair=46 len_ratio=1.300 overlap=0.167 entity_recall=1.000

Original: Tatters upreared and poised himself, stayed poised a moment, then, with a vicious dropping lunge, stabbed with his forefeet downward.

Rewrite: Raising his hindquarters and steadying his stance for a brief instant, the cat suddenly lunged downward with a vicious drop, driving his forepaws into the target.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 176. rw2s0_030741 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'highov')

words: original=18 rewrite=18 pair=36 len_ratio=1.000 overlap=0.778 entity_recall=1.000

Original: Well, the last thing Peter said before he collapsed was that he was the cause of the explosion.

Rewrite: Well, the final thing Peter stated before he collapsed was that he was the cause of the explosion.

Entities: source=['peter'] rewrite=['peter']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 177. rw_008161 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'lowov')

words: original=14 rewrite=19 pair=33 len_ratio=1.357 overlap=0.200 entity_recall=1.000

Original: 'He has it!' and looked up to know whether he should kill or spare.

Rewrite: Realizing he had the upper hand, he glanced upward to decide whether to take a life or show mercy.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 178. rw2s0_010530 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'lowov')

words: original=21 rewrite=18 pair=39 len_ratio=0.857 overlap=0.222 entity_recall=1.000

Original: - If there's a possibility of further murders, I want to know about them, and I want to deal in tangibles.

Rewrite: - Should further murders be possible, I require information on them and wish to engage with concrete evidence.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 179. rw2s1_027024 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'highov')

words: original=26 rewrite=26 pair=52 len_ratio=1.000 overlap=0.909 entity_recall=1.000

Original: It really felt good at a time like this when you got so many other things on your mind, and you do worry about your family.

Rewrite: It truly felt good at a time like this when you had so many other things on your mind and you do worry about your family.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 180. rw_024893 — cohort=base source=switchboard bucket=('base', 'switchboard', 'short', 'midov')

words: original=19 rewrite=13 pair=32 len_ratio=0.684 overlap=0.286 entity_recall=1.000

Original: I'm going to have to golf at a time when the heat of the day does not cook me.

Rewrite: I plan to play golf during cooler hours to avoid the midday heat.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 181. rw2s1_007318 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'long', 'midov')

words: original=21 rewrite=29 pair=50 len_ratio=1.381 overlap=0.556 entity_recall=1.000

Original: A murdersuicide (or murder suicide) is a setting where a person first kills one or more people, and then commits suicide.

Rewrite: A murdersuicide (or murder suicide) is defined as a scenario in which an individual first takes the life of one or more people before subsequently ending their own life.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 182. rw2s0_015486 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'highov')

words: original=25 rewrite=23 pair=48 len_ratio=0.920 overlap=0.786 entity_recall=1.000

Original: Ben stood a poorer chance now than before, for his unexpected defeat, and the raillery of his father, made him angry and reckless of consequences.

Rewrite: Ben's unexpected defeat and his father's raillery made him angry and reckless of consequences, leaving him with a poorer chance now than before.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 183. rw_025279 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'midov')

words: original=12 rewrite=16 pair=28 len_ratio=1.333 overlap=0.600 entity_recall=0.750

Original: [Music playing] [machine spinning] [laughing] [phone ringing] JOSEPH NEWMAN (ON PHONE): Hello.

Rewrite: While music plays, a machine spins, laughter echoes, and a phone rings, Joseph Newman answers, "Hello."

Entities: source=['joseph', 'newman', 'phone', 'hello'] rewrite=['joseph', 'newman', 'hello']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 184. rw2s0_024511 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'midov')

words: original=23 rewrite=16 pair=39 len_ratio=0.696 overlap=0.625 entity_recall=1.000

Original: So he made a machine in which steam lifted a lid called a piston in such a way as to turn a wheel.

Rewrite: So he constructed a machine where steam raised a piston, a lid, to rotate a wheel.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 185. rw_032654 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'long', 'midov')

words: original=15 rewrite=20 pair=35 len_ratio=1.333 overlap=0.600 entity_recall=1.000

Original: So now the Duke of York was down, and the Duke of Somerset was up.

Rewrite: Consequently, the Duke of York fell to a lower standing while the Duke of Somerset rose to a higher one.

Entities: source=['duke', 'york', 'somerset'] rewrite=['duke', 'york', 'somerset']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 186. rw2s1_011846 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'mid', 'midov')

words: original=37 rewrite=31 pair=68 len_ratio=0.838 overlap=0.667 entity_recall=1.000

Original: This is the reason why all that is left of the Tower of Babel and the other buildings that were put up so long ago are now simply hills of clay into which the brick has turned.

Rewrite: This explains why the only remnants of the Tower of Babel and the other structures erected so long ago are now merely hills of clay into which the brick has turned.

Entities: source=['tower', 'babel'] rewrite=['tower', 'babel']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 187. rw_042884 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'lowov')

words: original=14 rewrite=19 pair=33 len_ratio=1.357 overlap=0.143 entity_recall=1.000

Original: The shoal moves and changes shape allmost as if it is a single entity.

Rewrite: The shoal shifts and alters its form with such coordination that it appears to act as one unified being.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 188. rw2s1_014316 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'mid', 'highov')

words: original=24 rewrite=21 pair=45 len_ratio=0.875 overlap=0.917 entity_recall=1.000

Original: The man who was in charge of the storming of the Iranian Embassy ten years ago is to lead Britain's forces in the Gulf.

Rewrite: The man who led Britain's forces in the Gulf is to lead the storming of the Iranian Embassy ten years ago.

Entities: source=['iranian', 'embassy', "britain's", 'gulf'] rewrite=["britain's", 'gulf', 'iranian', 'embassy']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 189. rw2s0_016999 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'midov')

words: original=19 rewrite=20 pair=39 len_ratio=1.053 overlap=0.286 entity_recall=1.000

Original: Hoover will take you there, or nearly there, so that you can easily walk the rest of the way.

Rewrite: Hoover will transport you to a location nearly at your destination, enabling you to complete the remaining distance by walking.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 190. rw_032418 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'long', 'highov')

words: original=19 rewrite=30 pair=49 len_ratio=1.579 overlap=0.786 entity_recall=1.000

Original: [man coughs, gags] [triumphant orchestral finale] [boat horn blows] man speaking native language: You need to have a pee.

Rewrite: As a man coughs and gags, a triumphant orchestral finale swells and a boat horn blasts, prompting him to speak in his native tongue, "You need to have a pee."

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 191. rw2s0_010597 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'long', 'highov')

words: original=23 rewrite=29 pair=52 len_ratio=1.261 overlap=0.778 entity_recall=1.000

Original: For so it was, and all Miss Burton's efforts failed to put her, even for a moment, at the head of his table.

Rewrite: For this was the outcome, and despite all Miss Burton's efforts, she never managed to take her place at the head of his table, not even for a moment.

Entities: source=['miss', "burton's"] rewrite=['miss', "burton's"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 192. rw_000139 — cohort=base source=childes bucket=('base', 'childes', 'short', 'midov')

words: original=23 rewrite=15 pair=38 len_ratio=0.652 overlap=0.455 entity_recall=0.750

Original: *FAT: wait a second and then he he said I wanted what did you say you wanted the whole Empire Strikes Back collection.

Rewrite: FAT paused, then asked what I wanted, specifically mentioning the entire Empire Strikes Back collection.

Entities: source=['*fat', 'empire', 'strikes', 'back'] rewrite=['fat', 'empire', 'strikes', 'back']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 193. rw2s1_027527 — cohort=extra_shard1 source=bnc_spoken bucket=('extra_shard1', 'bnc_spoken', 'mid', 'lowov')

words: original=23 rewrite=24 pair=47 len_ratio=1.043 overlap=0.222 entity_recall=1.000

Original: Organizations like the R S P C A say they're disappointed but promise to continue their battle for a scheme to be introduced.

Rewrite: Organizations such as the R S P C A express disappointment yet vow to persist in their fight for the introduction of the scheme.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 194. rw_029045 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'lowov')

words: original=18 rewrite=15 pair=33 len_ratio=0.833 overlap=0.167 entity_recall=1.000

Original: Then, the people began to be dissatisfied with the Barons, because they did not do enough for them.

Rewrite: Consequently, the populace grew discontented with the Barons for their insufficient efforts on their behalf.

Entities: source=['barons'] rewrite=['barons']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 195. rw_019994 — cohort=base source=gutenberg bucket=('base', 'gutenberg', 'mid', 'lowov')

words: original=19 rewrite=19 pair=38 len_ratio=1.000 overlap=0.182 entity_recall=1.000

Original: But as the Revolution began to make its deadly progress at Paris, a gloom spread over this happy country.

Rewrite: However, as the Revolution commenced its lethal march through Paris, a pervasive melancholy descended upon this once joyful nation.

Entities: source=['revolution', 'paris'] rewrite=['revolution', 'paris']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 196. rw2s0_029687 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'mid', 'highov')

words: original=29 rewrite=24 pair=53 len_ratio=0.828 overlap=0.833 entity_recall=1.000

Original: It will involve more members with decision making taken by those who are responsible for reporting back to the members and for carrying out the effects of such responsibilities.

Rewrite: It will involve more members, with decision-making handled by those responsible for reporting back to the members and executing the effects of such responsibilities.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 197. rw2s0_025030 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'mid', 'lowov')

words: original=19 rewrite=19 pair=38 len_ratio=1.000 overlap=0.200 entity_recall=1.000

Original: Then, if you like, John shall take you home before any one comes to plague you with idle questions.

Rewrite: Then, if you wish, John will escort you home prior to anyone arriving to bother you with pointless inquiries.

Entities: source=['john'] rewrite=['john']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 198. rw_041384 — cohort=base source=open_subtitles bucket=('base', 'open_subtitles', 'mid', 'midov')

words: original=13 rewrite=13 pair=26 len_ratio=1.000 overlap=0.333 entity_recall=1.000

Original: Because, I wanted to be alone, But then your family kept piling in.

Rewrite: Although I intended to spend time in solitude, your family constantly kept arriving.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 199. rw2s0_017162 — cohort=extra_shard0 source=open_subtitles bucket=('extra_shard0', 'open_subtitles', 'mid', 'midov')

words: original=26 rewrite=26 pair=52 len_ratio=1.000 overlap=0.538 entity_recall=1.000

Original: Q wasn't exactly feeling at home in Whitehall, what with the new merger, so he set up shop here, away from prying eyes, as it were.

Rewrite: Q, not feeling entirely at ease in Whitehall due to the new merger, established his operations here to remain out of prying eyes, as it were.

Entities: source=['whitehall'] rewrite=['whitehall']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 200. rw_027558 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'highov')

words: original=14 rewrite=15 pair=29 len_ratio=1.071 overlap=0.857 entity_recall=1.000

Original: lot of time at the ocean and bay beaches and swimming in a lake.

Rewrite: Spending extensive time at ocean and bay beaches as well as swimming in a lake.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 201. rw_004033 — cohort=base source=bnc_spoken bucket=('base', 'bnc_spoken', 'mid', 'lowov')

words: original=14 rewrite=11 pair=25 len_ratio=0.786 overlap=0.167 entity_recall=1.000

Original: It's not enormous but it's perhaps er not the way people er were thinking.

Rewrite: While not enormous, the scale may not align with people's expectations.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 202. rw_043904 — cohort=base source=childes bucket=('base', 'childes', 'short', 'lowov')

words: original=13 rewrite=9 pair=22 len_ratio=0.692 overlap=0.167 entity_recall=1.000

Original: *MOT: didn't they see some things when they were going to the moon?

Rewrite: *MOt: Did they observe anything during their lunar missions?

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 203. rw2s0_028431 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'mid', 'midov')

words: original=28 rewrite=24 pair=52 len_ratio=0.857 overlap=0.643 entity_recall=1.000

Original: Like many other countries, the United Kingdom has tried to ban every new cannabinoid as soon as it is created, but new ones are made all the time.

Rewrite: Like many other countries, the United Kingdom has attempted to ban every new cannabinoid immediately upon its creation, yet new ones are continuously produced.

Entities: source=['united', 'kingdom'] rewrite=['united', 'kingdom']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 204. rw_026931 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'long', 'lowov')

words: original=14 rewrite=18 pair=32 len_ratio=1.286 overlap=0.222 entity_recall=1.000

Original: Subsequently, he faced legal challenges related to these manuscripts but ultimately avoided legal action.

Rewrite: Although he encountered legal hurdles concerning the manuscripts later on, he successfully steered clear of any formal proceedings.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 205. rw2s1_032129 — cohort=extra_shard1 source=open_subtitles bucket=('extra_shard1', 'open_subtitles', 'mid', 'midov')

words: original=19 rewrite=17 pair=36 len_ratio=0.895 overlap=0.571 entity_recall=1.000

Original: How do you come into my house and tell me what to think after what you did to Janey?

Rewrite: How can you enter my house and dictate my thoughts after the harm you inflicted on Janey?

Entities: source=['janey'] rewrite=['janey']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 206. rw2s0_014777 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'short', 'highov')

words: original=21 rewrite=13 pair=34 len_ratio=0.619 overlap=1.000 entity_recall=1.000

Original: *MAR: I saw and I saw a monster open a monster opened the coffin and it had it had five hands.

Rewrite: *MAR: I saw a monster open the coffin, and it had five hands.

Entities: source=['*mar'] rewrite=['*mar']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 207. rw_039738 — cohort=base source=childes bucket=('base', 'childes', 'long', 'highov')

words: original=12 rewrite=17 pair=29 len_ratio=1.417 overlap=0.875 entity_recall=1.000

Original: [hits silver circle with yellow wand; picks up silver circle] *MOT: okay.

Rewrite: Using the yellow wand to strike the silver circle, the character then picks it up. *MOT: okay.

Entities: source=['*mot'] rewrite=['*mot']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 208. rw2s0_024280 — cohort=extra_shard0 source=gutenberg bucket=('extra_shard0', 'gutenberg', 'short', 'highov')

words: original=24 rewrite=17 pair=41 len_ratio=0.708 overlap=0.900 entity_recall=1.000

Original: The dummies consisted of wash basins, buckets, etc., and it was calculated that when these dummies were yanked they would be far from dumb.

Rewrite: The dummies, which included wash basins, buckets, etc., were calculated to be far from dumb when yanked.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 209. rw2s0_011486 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'short', 'highov')

words: original=19 rewrite=14 pair=33 len_ratio=0.737 overlap=0.800 entity_recall=1.000

Original: In three-dimensional space, bisection is usually done by a plane, which is also called the bisector or bisecting plane.

Rewrite: In three-dimensional space, the bisector or bisecting plane is usually used to perform bisection.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 210. rw2s0_029976 — cohort=extra_shard0 source=childes bucket=('extra_shard0', 'childes', 'mid', 'highov')

words: original=48 rewrite=46 pair=94 len_ratio=0.958 overlap=0.842 entity_recall=1.000

Original: [Chi has piled all the washers on the silver circle and has put the silver stick in a v shape below the circle; there is a pair of magnet balls above the silver circle; each has a washer underneath and on top of it] *CHI: I'm all done.

Rewrite: [Chi has stacked all the washers on the silver circle and positioned the silver stick in a v shape beneath the circle; above the silver circle is a pair of magnet balls, each with a washer underneath and on top of it] *CHI: I'm all done.

Entities: source=['*chi', "i'm"] rewrite=['*chi', "i'm"]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 211. rw2s1_004965 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'midov')

words: original=31 rewrite=23 pair=54 len_ratio=0.742 overlap=0.455 entity_recall=1.000

Original: He was so strong a villain that he did not die under the torture, but lived to be afterwards pardoned and rewarded, though not to be ever believed in any more.

Rewrite: He was such a formidable villain that he survived the torture, went on to be pardoned and rewarded, yet never again believed in.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 212. rw2s1_006289 — cohort=extra_shard1 source=gutenberg bucket=('extra_shard1', 'gutenberg', 'short', 'lowov')

words: original=51 rewrite=38 pair=89 len_ratio=0.745 overlap=0.217 entity_recall=1.000

Original: Glyn Williams made the common mistake of thinking that because they rented the Hall, and dispensed large sums in subscriptions, they had the right to order the affairs of their less wealthy neighbours, and to have the first say in everything that was to be done in connection with the place.

Rewrite: Glyn Williams committed the common error of believing that their rental of the Hall and substantial subscription payments entitled them to direct the affairs of their poorer neighbors and hold primary authority over all matters concerning the venue.

Entities: source=['williams', 'hall'] rewrite=['williams', 'hall']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 213. rw_027158 — cohort=base source=childes bucket=('base', 'childes', 'long', 'midov')

words: original=12 rewrite=17 pair=29 len_ratio=1.417 overlap=0.500 entity_recall=1.000

Original: [picks up Millisandy and cuddles her] *MOT: she got nobody to love.

Rewrite: *MOT: She has no one to love* as he lifts Millisandy into his arms for a cuddle.

Entities: source=['millisandy', '*mot'] rewrite=['*mot', 'millisandy']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 214. rw2s0_021777 — cohort=extra_shard0 source=simple_wiki bucket=('extra_shard0', 'simple_wiki', 'short', 'lowov')

words: original=21 rewrite=15 pair=36 len_ratio=0.714 overlap=0.222 entity_recall=1.000

Original: Pseudoscience is when untrue claims are made to appear as though they have a scientific basis, even though they do not.

Rewrite: Pseudoscience occurs when false assertions are presented as having a scientific foundation despite lacking one.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 215. rw2s1_015459 — cohort=extra_shard1 source=simple_wiki bucket=('extra_shard1', 'simple_wiki', 'long', 'midov')

words: original=18 rewrite=24 pair=42 len_ratio=1.333 overlap=0.667 entity_recall=1.000

Original: Diseases spread very quickly because the enslaved people were so crowded together, and because there was no sanitation.

Rewrite: The rapid spread of diseases was caused by the lack of sanitation and the fact that the enslaved people were packed so closely together.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 216. rw_025237 — cohort=base source=childes bucket=('base', 'childes', 'long', 'midov')

words: original=15 rewrite=20 pair=35 len_ratio=1.333 overlap=0.583 entity_recall=1.000

Original: [puts yellow wand close to black base; several objects stick to it] *CHI: xxx stuck.

Rewrite: *CHI: xxx stuck as a yellow wand is positioned near a black base, causing multiple items to adhere to it.

Entities: source=['*chi'] rewrite=['*chi']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 217. rw_022302 — cohort=base source=childes bucket=('base', 'childes', 'mid', 'lowov')

words: original=21 rewrite=21 pair=42 len_ratio=1.000 overlap=0.200 entity_recall=1.000

Original: push them and he got off and he got mad because those kids kind o' took over the swing on him.

Rewrite: He became angry after they pushed him out of the swing, feeling that the children had taken it over from him.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 218. rw_030060 — cohort=base source=switchboard bucket=('base', 'switchboard', 'short', 'midov')

words: original=42 rewrite=29 pair=71 len_ratio=0.690 overlap=0.294 entity_recall=1.000

Original: A: It was really nice that they had, um, that A: they began to cut back because of the oil problems, A: but, um, they would have, um, so many community outdoor theaters and, And like, uh, community country club type things.

Rewrite: Although it was positive that they started reducing expenses due to the oil crisis, they could have established numerous community outdoor theaters and facilities similar to country clubs instead.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 219. rw2s0_018771 — cohort=extra_shard0 source=bnc_spoken bucket=('extra_shard0', 'bnc_spoken', 'mid', 'highov')

words: original=23 rewrite=22 pair=45 len_ratio=0.957 overlap=0.929 entity_recall=1.000

Original: The Oxford University Student's Union has expressed its disappointment at the Government's action on the Student Loans Bill in the Commons last night.

Rewrite: Last night in the Commons, the Oxford University Student's Union voiced its disappointment regarding the Government's action on the Student Loans Bill.

Entities: source=['oxford', 'university', "student's", 'union', "government's", 'student', 'loans', 'bill', 'commons'] rewrite=['commons', 'oxford', 'university', "student's", 'union', "government's", 'student', 'loans', 'bill']

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format

## 220. rw_014209 — cohort=base source=simple_wiki bucket=('base', 'simple_wiki', 'short', 'lowov')

words: original=28 rewrite=17 pair=45 len_ratio=0.607 overlap=0.200 entity_recall=1.000

Original: Scientists say that it is good at living in places that human beings have changed a little, but not in places that human beings have changed a lot.

Rewrite: Scientists indicate that the species thrives in moderately altered environments but struggles where human impact is extensive.

Entities: source=[] rewrite=[]

Numbers: source=[] rewrite=[]

Manual label: [ ] faithful  [ ] minor-change  [ ] meaning-change  [ ] fragment/format
