#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
bash program/build.sh
mkdir -p ../build/figures
for name in input_supervision research_lineage; do
    xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
        -jobname="$name" -output-directory=../build/figures \
        "\\def\\FigureSource{$name.tex}\\input{scientific_figure.tex}" \
        > "../build/figures/$name.txt"
    cp "../build/figures/$name.pdf" "$name.pdf"
done
