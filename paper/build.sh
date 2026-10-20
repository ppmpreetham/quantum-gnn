#!/usr/bin/env bash
# Build the paper with tectonic when available, otherwise with pdflatex.
set -e
cd "$(dirname "$0")"

# Use $TECTONIC if set, else tectonic on PATH, else a copy in ~/.local/bin or
# /tmp that this user owns and that no one else can modify.
find_tectonic() {
    if [ -n "${TECTONIC:-}" ]; then echo "$TECTONIC"; return; fi
    if command -v tectonic >/dev/null 2>&1; then command -v tectonic; return; fi
    local f
    for f in "$HOME/.local/bin/tectonic" /tmp/tectonic; do
        if [ -x "$f" ] && [ -O "$f" ] && [ -z "$(find "$f" -perm /022 2>/dev/null)" ]; then
            echo "$f"; return
        fi
    done
    return 1
}

if T=$(find_tectonic); then
  "$T" main.tex
else
  pdflatex -interaction=nonstopmode main.tex
  pdflatex -interaction=nonstopmode main.tex
fi
