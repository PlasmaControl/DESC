#!/bin/bash
# Stage-two runs, one method after another on the same cores, so that every method
# gets the same wall-clock budget on the same hardware.
#
# Environment variables:
#   METHODS  methods to run, in order (stage2_precise_QA.py --method)
#            (default: "baseline converged multistart basin noisy cma")
#   BUDGET   wall-clock budget per run [s]     (default: 1200)
#   SEED     random seed                       (default: 0)
#   CORES    taskset core list                 (default: 0-2)
#   EXTRA    extra driver options, e.g. "--variant matched" or "--free-frame"
#   LOGTAG   suffix for the log file names, e.g. "_matched"
#   VENV     virtualenv to activate, if any    (default: none; use the current python)
#
# Logs go to results/stage2/logs/<method><LOGTAG>_seed<SEED>.log. Results go to
# results/stage2_fixed/ (results/stage2/ with --free-frame).
if [ -n "${VENV:-}" ]; then source "$VENV/bin/activate"; fi
cd "$(dirname "$0")"
mkdir -p results/stage2/logs
BUDGET=${BUDGET:-1200}
SEED=${SEED:-0}
CORES=${CORES:-0-2}
EXTRA=${EXTRA:-}
LOGTAG=${LOGTAG:-}
for m in ${METHODS:-baseline converged multistart basin noisy cma}; do
  echo "$(date) starting $m seed $SEED"
  PYTHONUNBUFFERED=1 taskset -c $CORES python stage2_precise_QA.py --method $m --budget $BUDGET --seed $SEED $EXTRA \
    > results/stage2/logs/${m}${LOGTAG}_seed${SEED}.log 2>&1
  echo "$(date) done $m"
done
