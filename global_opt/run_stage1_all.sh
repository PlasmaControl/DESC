#!/bin/bash
# Stage-one runs (precise QA example at M = N = 6, boundary modes <= 3), one method
# after another, each pinned to the same cores.
#
# Environment variables:
#   METHODS  methods to run, in order          (default: "baseline multistart basin")
#   BUDGET   wall-clock budget per run [s]     (default: 1800)
#   SEED     random seed                       (default: 0)
#   CORES    taskset core list                 (default: 0-3)
#   VENV     virtualenv to activate, if any    (default: none; use the current python)
#
# Logs go to results/stage1/logs/<method>_seed<SEED>.log; results to results/stage1/.
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
cd "$(dirname "$0")"
mkdir -p results/stage1/logs
BUDGET=${BUDGET:-1800}
SEED=${SEED:-0}
CORES=${CORES:-0-3}
for m in ${METHODS:-baseline multistart basin}; do
  echo "$(date) starting $m seed $SEED"
  PYTHONUNBUFFERED=1 taskset -c $CORES python stage1_precise_QA.py --method $m --budget $BUDGET --seed $SEED \
    > results/stage1/logs/${m}_seed${SEED}.log 2>&1
  echo "$(date) done $m"
done
