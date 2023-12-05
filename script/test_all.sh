#!/bin/bash

ALL_METHODS=(random afl size nmfli greedy simple simple_reverse)
ALL_METHODS=(nmfli)

for policy in ${ALL_METHODS[@]}; do
    echo "=== Run policy ${policy} ==="
    # nohup 
    bash script/train.sh ${policy} show --dataset=mnist --iid=0 --num_users=10 \
        # > src/log/${policy}.txt 2>&1
done