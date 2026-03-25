#!/bin/bash
# Submit a single trial run: k=2, num-simulations=1, 4 cores on the long queue

addqueue -c "trial k=2 nsim=1" -q long -n 1x4 -m 2 \
    -o trial_%j.out \
    -s "/usr/bin/env OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 /usr/bin/python3 -u trial/trial.py --k=2 --num-simulations=1"
