#!/bin/bash
# Run every stage-two method in sequence on the same 3 cores (fair wall-clock budgets).
source /home/user/venv-desc/bin/activate
cd "$(dirname "$0")"
BUDGET=${BUDGET:-1200}
SEED=${SEED:-0}
for m in ${METHODS:-baseline converged multistart basin noisy cma}; do
  echo "$(date) starting $m seed $SEED"
  PYTHONUNBUFFERED=1 taskset -c 0-2 python stage2_precise_QA.py --method $m --budget $BUDGET --seed $SEED \
    > results/stage2/logs/${m}_seed${SEED}.log 2>&1
  echo "$(date) done $m"
done
