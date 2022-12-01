#!/bin/bash

for i in `seq 1 10`; do
    python pid_optuna.py &
done

wait
