#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p build tables figures
bash figures/build.sh
python3 render.py
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/xelatex-pass1.txt
TEXINPUTS=".:../:" BIBINPUTS=".:../:" BSTINPUTS=".:../:" bibtex build/main > build/bibtex.txt
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/xelatex-pass2.txt
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/xelatex-pass3.txt
pdftotext -layout build/main.pdf build/main.txt
python3 validate.py
cp build/main.pdf qiushi-engine-babylm-report-zh.pdf
pdfinfo build/main.pdf
