#!/bin/bash

# Example: bash script/train.sh momentum show --dataset=cifar --iid=1 --num_users=1

policy=$1

if [[ $2 == "pdb" ]]; then
    # debug
    remain_arg=${@:3}
    python3 -m pdb src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0 $remain_arg
elif [[ $2 == "show" ]]; then
    remain_arg=${@:3}
    python3 src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0 $remain_arg
else
    remain_arg=${@:2}
    python3 -u src/federated_main.py \
        --model=cnn  --dataset=mnist --num_users=10 --policy=${policy} --iid=0 $remain_arg > src/log/${policy}.txt 2>&1 & 
fi

