#!/bin/bash

policy=$1
if [[ $2 == "pdb" ]]; then
    # debug
    python3 -m pdb src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0
elif [[ $2 == "show" ]]; then
    python3 src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0
else
    python3 -u src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0 > src/log/${policy}.txt 2>&1 & 
fi

