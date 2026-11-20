#!/bin/bash
# Stage-one runs (precise QA example at M=N=6, boundary modes <= 3), in sequence.
source /home/user/venv-desc/bin/activate
cd "$(dirname "$0")"
BUDGET=${BUDGET:-1800}
SEED=${SEED:-0}
CORES=${CORES:-0-3}
for m in ${METHODS:-baseline multistart basin}; do
  echo "$(date) starting $m seed $SEED"
  PYTHONUNBUFFERED=1 taskset -c $CORES python stage1_precise_QA.py --method $m --budget $BUDGET --seed $SEED \
    > results/stage1/logs/${m}_seed${SEED}.log 2>&1
  echo "$(date) done $m"
done
