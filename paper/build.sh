#!/usr/bin/env bash
# Build the paper with tectonic when available, otherwise with pdflatex.
set -e
cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
  tectonic main.tex
elif [ -x /tmp/tectonic ]; then
  /tmp/tectonic main.tex
else
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
fi
