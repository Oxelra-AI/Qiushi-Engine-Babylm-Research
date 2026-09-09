#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
for arg in sys.argv[1:]:
    p = Path(arg) / 'tokenizer_config.json'
    d = json.loads(p.read_text(encoding='utf-8'))
    before = d.get('tokenizer_class')
    d['tokenizer_class'] = 'PreTrainedTokenizerFast'
    d.setdefault('bos_token', '<s>')
    d.setdefault('eos_token', '</s>')
    d.setdefault('unk_token', '<unk>')
    d.setdefault('pad_token', '<pad>')
    d.setdefault('mask_token', '<mask>')
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'path': str(p), 'before': before, 'after': d.get('tokenizer_class')}))
