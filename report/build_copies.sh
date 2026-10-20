#!/usr/bin/env bash
# Build one personalised copy of the report per team member into copies/.
# Each copy carries only that member's name, register number, programme and
# head of department, and leaves out the "Items to Complete" appendix.
set -euo pipefail
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
TECTONIC=$(find_tectonic) || { echo "tectonic not found" >&2; exit 1; }
mkdir -p copies

build() {  # file-stem, name, register number, programme, head of department
    local job=$1
    cat > "$job.tex" <<TEX
\\def\\finalcopy{}
\\newcommand{\\candidatename}{$2}
\\newcommand{\\candidatereg}{$3}
\\newcommand{\\programme}{$4}
\\newcommand{\\hodname}{$5}
\\input{report.tex}
TEX
    "$TECTONIC" --outdir copies "$job.tex" >/dev/null || { rm -f "$job.tex"; exit 1; }
    rm -f "$job.tex"
    echo "copies/$job.pdf"
}

build 23BRS1346_P_Preetham "P Preetham" 23BRS1346 "Artificial Intelligence and Robotics"         "Dr. Sathian D"
build 23BRS1255_Biresh_Biswas "Biresh Biswas"  23BRS1255 "Artificial Intelligence and Robotics"         "Dr. Sathian D"
build 23BAI1568_Nandana_Satish "Nandana Satish" 23BAI1568 "Artificial Intelligence and Machine Learning" "Dr. Tulasi Prasad Sariki"
