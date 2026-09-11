#!/usr/bin/env bash
# Build the paper. fiziko figure needs lualatex + luamplib + fiziko.mp
# (clone https://github.com/jemmybutton/fiziko into paper/fiziko/).
set -e
cd "$(dirname "$0)"

if [ -f fiziko/fiziko.mp ]; then
  TEXINPUTS="fiziko//:" lualatex -interaction=nonstopmode fig_testbed.tex
else
  echo "fiziko not found in paper/fiziko; using TikZ fallback figure"
fi

pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
