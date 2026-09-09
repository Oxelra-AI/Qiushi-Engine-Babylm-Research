# Byte-level tokenizer coverage audit
This CPU-only audit checks whether the compliant 10M-trained byte-level BPE tokenizer has a coverage defect from not explicitly passing `ByteLevel.alphabet()` to the trainer. It does not inspect active retrain directories and does not use evaluation text to tune vocabulary.
## Tokenizer identities
- Old tokenizer SHA256: `9cc4f9073675da3f817a6020e7b0833cf3234c0131515d68f4065b00f14933ac`
- Compliant tokenizer SHA256: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- 10M pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
## Byte alphabet coverage
- ByteLevel alphabet size: 256
- Old tokenizer missing byte-level alphabet entries: 66
- Compliant tokenizer missing byte-level alphabet entries: 71
- Missing compliant alphabet preview: `['À', 'Á', 'Ì', 'Í', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', 'Ø', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß', 'ç', 'è', 'ë', 'ì', 'í', 'î', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ', 'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď', 'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ']`
## `<unk>` counts
- Existing eval roots scanned: `['experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval', 'experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data']`; scanned_any_eval_text=True
- old on 10M pool: rows=64740, tokens=16573440, unk_tokens=12, rows_with_unk=4, unk_rate=7.2405005e-07, row_unk_rate=6.1785604e-05
- compliant on 10M pool: rows=64740, tokens=14793999, unk_tokens=0, rows_with_unk=0, unk_rate=0, row_unk_rate=0
- old on eval text: rows=425068, tokens=108817408, unk_tokens=581, rows_with_unk=157, unk_rate=5.3392193e-06, row_unk_rate=0.00036935267
- compliant on eval text: rows=425068, tokens=7527411, unk_tokens=1799, rows_with_unk=1371, unk_rate=0.00023899319, row_unk_rate=0.0032253663
## Eval-family compliant `<unk>` counts
- Reading_AoA: rows=25359, unk_tokens=44, rows_with_unk=9, unk_rate=0.00016456719
- SuperGLUE_or_GLUE: rows=186521, unk_tokens=523, rows_with_unk=130, unk_rate=0.00010990103
- Supplement: rows=10936, unk_tokens=1232, rows_with_unk=1232, unk_rate=0.0060885208
## Example compliant eval `<unk>` contexts

The following diagnostic records preserve the original text and code points.

```text
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=1 chars=['˚', '–'] text=`Care of sheets: – machine at 40˚C or`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=1 chars=['á', 'ạ', '破', '�'] text=`Pencarian FILM Untuk "Peace Breaker 2017" yuk mampir ke channel say.. Edges East provides the l.. A corrupt cop makes one w.. Peace Breaker 2017 ~ 破�.. Náo Loạn - Peace Break.. Please subscribe and`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=12 chars=['ا', 'ت', 'د', 'س', 'ش', 'ن', 'ه', 'و', 'ک', 'ی', '‘', '’'] text=`But Rumi is saying wait a second اوست دیوانه که دیوانه نشد ‘the one who hasn’t gone`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=12 chars=['ا', 'ت', 'د', 'س', 'ش', 'ن', 'ه', 'و', 'ک', 'ی', '‘', '’'] text=`But Rumi is saying wait a second اوست دیوانه که دیوانه نشد ‘the one who hasn’t gone mad is the`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=1 chars=['´', 'í'] text=`Lesní Penzion Kobylnice has a children´s playground, an outdoor swimming`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=1 chars=['´'] text=`Scott´s finally ceased bike production in the 1950s, when anyone not making dull four stroke singles was on a`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=1 chars=['´', 'í'] text=`Lesní Penzion Kobylnice has a children´s playground, an outdoor swimming pool for children, volleyball and`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=8 chars=['а', 'о', 'с', 'ѕ', 'і'] text=`Since оf their ѕіze plastіс bаgs fіt perfectly іn small`
- Reading_AoA `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_childes.json` n_unk=7 chars=['е', 'о', 'у', 'і'] text=`Utilize thеm іn thе restroom, bed room, workplace, оr anу other place уоu have little`
- SuperGLUE_or_GLUE `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/glue_filtered/boolq.train.jsonl` n_unk=4 chars=['ā', 'ɒ', 'ə', 'ɜ', 'ɾ', 'ʃ', 'ʒ', 'ˈ', 'ː', 'ا', 'ر', 'س', 'ف', 'ی'] text=`Persian language -- Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily`
```

## Scientific interpretation
The tokenizer produced no `<unk>` on the 10M training pool but did produce `<unk>` on official evaluation text. This is a coverage vulnerability of the legal tokenizer construction and should be considered if compliant scores fall in families carrying these characters. A repair would be to train a same-pool tokenizer with explicit byte alphabet, not to use evaluation text for vocabulary selection.

Full JSON: `experiments/archive/frontier_consolidation/data/tokenizer_byte_unk_audit/tokenizer_byte_unk_audit.json`
