#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build tables
bash ../zh/figures/build.sh
python3 -B ../zh/render.py
python3 -B render.py
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.txt
bibtex build/main > build/bibtex.txt
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.txt
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.txt
pdftotext -layout build/main.pdf build/main.txt
python3 -B validate.py
cp build/main.pdf qiushi-engine-babylm-report-en.pdf
pdfinfo build/main.pdf
