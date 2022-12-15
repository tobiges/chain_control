#!/bin/bash

# This script starts 16 optuna instances in parallel.

for i in {1..16}
do
    python3 nl_optuna_gridsearch.py &
done

wait