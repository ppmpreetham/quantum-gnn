#!/usr/bin/env bash
# Build one personalised copy of the report per team member into copies/.
# Each copy carries only that member's name, register number, programme and
# head of department, and leaves out the "Items to Complete" appendix.
set -euo pipefail
cd "$(dirname "$0")"
TECTONIC=${TECTONIC:-/tmp/tectonic}
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
