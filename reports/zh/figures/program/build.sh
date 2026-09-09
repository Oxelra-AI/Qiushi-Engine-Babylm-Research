#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p ../../build/program
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error \
    -jobname=three_stage_research \
    -output-directory=../../build/program figure_three_stage_research.tex \
    > ../../build/program/three_stage_research.txt
cp ../../build/program/three_stage_research.pdf ../three_stage_research_public.pdf
