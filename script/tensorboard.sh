#!/bin/bash
# set -x

if [ -z $1 ]; then
    echo "[Error] Please input the log directory"
    exit 0
fi

python3 /home/hphu/.local/lib/python3.8/site-packages/tensorboard/main.py \
    --logdir $1