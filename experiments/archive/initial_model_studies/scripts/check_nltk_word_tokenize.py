#!/usr/bin/env python3
from __future__ import annotations
import os
import pathlib
import nltk

p = pathlib.Path('experiments/archive/initial_model_studies/data/nltk_data').resolve()
p.mkdir(parents=True, exist_ok=True)
print({'download_dir': str(p)})
for pkg in ['punkt', 'punkt_tab']:
    ok = nltk.download(pkg, download_dir=str(p), quiet=False)
    print({'pkg': pkg, 'ok': ok})
os.environ['NLTK_DATA'] = str(p)
nltk.data.path.insert(0, str(p))
from nltk.tokenize import word_tokenize
print(word_tokenize('This is a test.'))
