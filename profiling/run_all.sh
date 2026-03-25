#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

NSYS="/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/profilers/Nsight_Systems/bin/nsys"

declare -A CASE_SCRIPTS
CASE_SCRIPTS[1]="profiling/case_1_coil_auglag.py"
CASE_SCRIPTS[2]="profiling/case_2_coil_exact.py"
CASE_SCRIPTS[3]="profiling/case_3_free_boundary.py"
CASE_SCRIPTS[4]="profiling/case_4_single_stage.py"

if [ "$#" -gt 0 ]; then
    CASES=("$@")
else
    CASES=(1 2 3 4)
fi

for c in "${CASES[@]}"; do
    script="${CASE_SCRIPTS[$c]}"
    echo "========================================"
    echo "  Running Case ${c}: ${script}"
    echo "========================================"

    "$NSYS" profile \
        --output="profiling/nsys_case_${c}" \
        --force-overwrite=true \
        --trace=cuda,nvtx \
        python "$script" 2>&1 | tee "profiling/case_${c}_stdout.log"

    "$NSYS" export \
        --type=sqlite \
        --output="profiling/nsys_case_${c}.sqlite" \
        "profiling/nsys_case_${c}.nsys-rep" || true
done

echo "========================================"
echo "  Analyzing Nsight profiles"
echo "========================================"
if ls profiling/nsys_case_*.sqlite 1>/dev/null 2>&1; then
    python profiling/analyze_nsight.py profiling/nsys_case_*.sqlite
else
    echo "No .sqlite files found; skipping analyze_nsight.py"
fi

echo "========================================"
echo "  Generating report"
echo "========================================"
python profiling/report.py
