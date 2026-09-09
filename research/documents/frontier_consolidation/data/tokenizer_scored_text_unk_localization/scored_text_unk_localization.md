# Unknown-token localization in evaluation text

This narrows the broad audit to likely scored text fields in the exact current-coordinate evaluation root, with low-level tokenizer padding/truncation disabled.

## old tokenizer total
strings=441864, tokens=18527632, unk_tokens=584, strings_with_unk=157, unk_rate=3.1520488e-05, string_unk_rate=0.00035531295

### old by family
- BLiMP: strings=119750, tokens=1462751, unk_tokens=0, strings_with_unk=0, unk_rate=0
- EWoK: strings=53326, tokens=449152, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Other: strings=25230, tokens=239909, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Reading_AoA: strings=16934, tokens=219972, unk_tokens=3, strings_with_unk=3, unk_rate=1.3638099e-05
- SuperGLUE: strings=216188, tokens=15962509, unk_tokens=581, strings_with_unk=154, unk_rate=3.6397787e-05
- Supplement: strings=10436, tokens=193339, unk_tokens=0, strings_with_unk=0, unk_rate=0

## new tokenizer total
strings=441864, tokens=18463824, unk_tokens=1513, strings_with_unk=1085, unk_rate=8.1944022e-05, string_unk_rate=0.0024555067

### new by family
- BLiMP: strings=119750, tokens=1467873, unk_tokens=0, strings_with_unk=0, unk_rate=0
- EWoK: strings=53326, tokens=452718, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Other: strings=25230, tokens=240394, unk_tokens=0, strings_with_unk=0, unk_rate=0
- Reading_AoA: strings=16934, tokens=219756, unk_tokens=44, strings_with_unk=9, unk_rate=0.00020022206
- SuperGLUE: strings=216188, tokens=15889475, unk_tokens=523, strings_with_unk=130, unk_rate=3.291487e-05
- Supplement: strings=10436, tokens=193608, unk_tokens=946, strings_with_unk=946, unk_rate=0.0048861617

## New-tokenizer path/field unknown hotspots
- 458 unk / 9427 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/boolq.train.jsonl`, field `passage`
- 244 unk / 280 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/turn_taking.jsonl`, field `sentence_bad`
- 244 unk / 280 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/turn_taking.jsonl`, field `sentence_good`
- 165 unk / 165 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_tricky.jsonl`, field `sentence_bad`
- 165 unk / 165 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_tricky.jsonl`, field `sentence_good`
- 64 unk / 64 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_easy.jsonl`, field `sentence_bad`
- 64 unk / 64 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/supplement_filtered/qa_congruence_easy.jsonl`, field `sentence_good`
- 22 unk / 1635 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl`, field `passage`
- 18 unk / 20215 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/qqp.valid.jsonl`, field `question2`
- 12 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `mad[8].context`
- 12 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `mad[9].context`
- 10 unk / 20215 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/qqp.valid.jsonl`, field `question1`
- 8 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `trash[10].context`
- 7 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `trash[11].context`
- 5 unk / 10000 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/qqp.train.jsonl`, field `question2`
- 4 unk / 10000 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/mnli.train.jsonl`, field `premise`
- 3 unk / 3668 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/mrpc.train.jsonl`, field `sentence1`
- 2 unk / 554 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/wsc.train.jsonl`, field `span1_text`
- 1 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `hand[0].context`
- 1 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `hit[0].context`
- 1 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `pool[3].context`
- 1 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `sticky[17].context`
- 1 unk / 1 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json`, field `table[14].context`
- 1 unk / 10000 strings, family path `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/qqp.train.jsonl`, field `question1`

## New-tokenizer example contexts

The following diagnostic records preserve the original text and code points.

```text
- Reading_AoA field `hand[0].context` n_unk=1 tokens=['Ġat', 'Ġ', '4', '0', 'Ġ', '<unk>', 'Ĭ', 'C', 'Ġor', '</s>'] text=`Care of sheets: – machine at 40˚C or`
- Reading_AoA field `hit[0].context` n_unk=1 tokens=['1', '7', 'Ġ', '~', 'Ġ', '<unk>', 'ł', '´', 'ï¿½', '..', 'ĠN'] text=`Pencarian FILM Untuk "Peace Breaker 2017" yuk mampir ke channel say.. Edges East provides the l.. A corrupt cop makes one w.. Peace Breaker 2017 ~ 破�.. Náo Loạn - Peace Break.. Please subscribe and`
- Reading_AoA field `mad[8].context` n_unk=12 tokens=['Ġsaying', 'Ġwait', 'Ġa', 'Ġsecond', 'Ġ', '<unk>', '§', 'Ù', 'Ī', '<unk>', '³'] text=`But Rumi is saying wait a second اوست دیوانه که دیوانه نشد ‘the one who hasn’t gone`
- Reading_AoA field `mad[9].context` n_unk=12 tokens=['Ġsaying', 'Ġwait', 'Ġa', 'Ġsecond', 'Ġ', '<unk>', '§', 'Ù', 'Ī', '<unk>', '³'] text=`But Rumi is saying wait a second اوست دیوانه که دیوانه نشد ‘the one who hasn’t gone mad is the`
- Reading_AoA field `pool[3].context` n_unk=1 tokens=['ice', 'Ġhas', 'Ġa', 'Ġchildren', 'Ġ', '<unk>', 'ģ', 's', 'Ġplay', 'ground', ','] text=`Lesní Penzion Kobylnice has a children´s playground, an outdoor swimming`
- Reading_AoA field `sticky[17].context` n_unk=1 tokens=['<s>', 'ĠScott', 'Ġ', '<unk>', 'ģ', 's', 'Ġfinally', 'Ġceased', 'Ġbike'] text=`Scott´s finally ceased bike production in the 1950s, when anyone not making dull four stroke singles was on a`
- Reading_AoA field `table[14].context` n_unk=1 tokens=['ice', 'Ġhas', 'Ġa', 'Ġchildren', 'Ġ', '<unk>', 'ģ', 's', 'Ġplay', 'ground', ','] text=`Lesní Penzion Kobylnice has a children´s playground, an outdoor swimming pool for children, volleyball and`
- Reading_AoA field `trash[10].context` n_unk=8 tokens=['<s>', 'ĠSince', 'Ġ', '<unk>', '¾', 'f', 'Ġtheir', 'Ġ', '<unk>'] text=`Since оf their ѕіze plastіс bаgs fіt perfectly іn small`
- Reading_AoA field `trash[11].context` n_unk=7 tokens=['<s>', 'ĠUt', 'il', 'ize', 'Ġth', '<unk>', 'µ', 'm', 'Ġ', '<unk>', 'ĸ'] text=`Utilize thеm іn thе restroom, bed room, workplace, оr anу other place уоu have little`
- SuperGLUE field `passage` n_unk=4 tokens=['ars', 'i', 'Ġ(', 'Ù', 'ģ', '<unk>', '§', '<unk>', '±', '<unk>', '³'] text=`Persian language -- Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily`
- SuperGLUE field `passage` n_unk=1 tokens=['º', '¬', 'å', 'ĸ', '°', '<unk>', '¨', '®', 'Ġ(', 'Ġ', 'ã'] text=`Tokyo Ghoul -- Tokyo Ghoul (Japanese: 東京喰種 ( トーキョーグール ) , Hepburn: Tōkyō Gūru) is a Japanese dark fantasy manga series written and illustrated by Sui Ishida. It was serialized in Shueisha's seinen manga magazine Weekly Young Jump between Se`
- SuperGLUE field `passage` n_unk=1 tokens=['zu', 'Ġ(', 'å', 'Ľ', '½', '<unk>', '¶', '³', ',', 'Ġlit', '.'] text=`China at the FIFA World Cup -- The China national team was founded in 1924 and joined FIFA in 1931--1958, and then from 1979. China first entered World Cup qualification in 1957 in an attempt to qualify for the 1958 FIFA World Cup. China fa`
- SuperGLUE field `passage` n_unk=1 tokens=['zu', 'Ġ(', 'å', 'Ľ', '½', '<unk>', '¶', '³', ',', 'Ġlit', '.'] text=`China at the FIFA World Cup -- The China national team was founded in 1924 and joined FIFA in 1931--1958, and then from 1979. China first entered World Cup qualification in 1957 in an attempt to qualify for the 1958 FIFA World Cup. China fa`
- SuperGLUE field `passage` n_unk=1 tokens=['Ë', 'Ī', 'ma', 'Ê', 'Ĭ', '<unk>', '¯', 'É', 'Ĳ', ')', 'Ġ('] text=`Berlin Wall -- The Berlin Wall (German: Berliner Mauer, pronounced (bɛʁˈliːnɐ ˈmaʊ̯ɐ) ( listen)) was a guarded concrete barrier that physically and ideologically divided Berlin from 1961 to 1989. Constructed by the German Democratic Republi`
- SuperGLUE field `passage` n_unk=2 tokens=['ĠU', 'rd', 'u', ':', 'Ġ', '<unk>', '¨', 'Ù', 'Ĭ', '<unk>', '³'] text=`Gram flour -- Gram flour or chickpea flour or besan (Hindi: बेसन; Burmese: ပဲမှုန့်; Urdu: بيسن‎), is a pulse flour made from a variety of ground chickpea known as Bengal gram. It is a staple ingredient in the cuisine of the Indian subconti`
- SuperGLUE field `passage` n_unk=1 tokens=['Ë', 'Ĳ', 'Ë', 'Ģ', 'n', '<unk>', '©', ')', ';', 'ĠSwedish', ':'] text=`Øresund Bridge -- The Øresund or Öresund Bridge (Danish: Øresundsbroen (ˈøːɐsɔnsˌbʁoːˀn̩); Swedish: Öresundsbron (œːrɛ2sɵnːdsˌbruːn); hybrid name: Øresundsbron) is a combined railway and motorway bridge across the Øresund strait between Swe`
- SuperGLUE field `passage` n_unk=3 tokens=['ann', 'Ġ(', 'Ë', 'Ī', 't', '<unk>', 'ª', 'É', '£', 'u', 'É'] text=`Northern Ireland -- Northern Ireland (Irish: Tuaisceart Éireann (ˈt̪ɣuəʃcəɾɣt̪ɣ ˈeːɾjən̪ɣ) ( listen); Ulster-Scots: Norlin Airlann) is a part of the United Kingdom in the north-east of the island of Ireland, variously described as a country`
- SuperGLUE field `passage` n_unk=7 tokens=['A', 'rab', 'ic', ':', 'Ġ', '<unk>', '¨', '<unk>', '±', '<unk>', '¬'] text=`Burj Al Arab -- The Burj Al Arab (Arabic: برج العرب‎, Tower of the Arabs) is a luxury hotel located in Dubai, United Arab Emirates. It is the third tallest hotel in the world (although 39% of its total height is made up of non-occupiable sp`
- SuperGLUE field `passage` n_unk=2 tokens=[':', 'Ġ', 'å', '¤', '§', '<unk>', 'Ĩ', 'Ĭ', '<unk>', 'Į', '«'] text=`Giant panda -- The giant panda (Ailuropoda melanoleuca, literally ``black and white cat-foot''; Chinese: 大熊猫; pinyin: dà xióng māo, literally ``big bear cat''), also known as panda bear or simply panda, is a bear native to south central Chi`
- SuperGLUE field `passage` n_unk=3 tokens=['ann', 'Ġ(', 'Ë', 'Ī', 't', '<unk>', 'ª', 'É', '£', 'u', 'É'] text=`Northern Ireland -- Northern Ireland (Irish: Tuaisceart Éireann (ˈt̪ɣuəʃcəɾɣt̪ɣ ˈeːɾjən̪ɣ) ( listen); Ulster-Scots: Norlin Airlann) is a part of the United Kingdom in the north-east of the island of Ireland, variously described as a country`

```

## Interpretation
The compliant tokenizer produces `<unk>` in likely scored strings: total 1513, Supplement 946, BLiMP 0, EWoK 0. This is a real tokenizer-construction vulnerability if confirmed by the official scorer's actual serialization.

Full JSON: `experiments/archive/frontier_consolidation/data/tokenizer_scored_text_unk_localization/scored_text_unk_localization.json`
