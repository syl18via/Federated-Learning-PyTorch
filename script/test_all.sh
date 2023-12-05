#!/bin/bash
set -x
export NMFLI_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`

DATASETS=(mnist)
MODELS=(cnn)
TARGET_LABEL_CFGS=(non_overlap overlap identical)
ALL_METHODS=(random afl size nmfli greedy simple)

# TARGET_LABEL_CFGS=(non_overlap)
# ALL_METHODS=(mcafee)

NMFLI_EXP_DIR=save/results/${NMFLI_EXP_DATETIME}
mkdir -p $NMFLI_EXP_DIR
for dataset in ${DATASETS}; do
for target_label in ${TARGET_LABEL_CFGS}; do
for model in ${MODELS}; do
for policy in ${ALL_METHODS[@]}; do
    export NMFLI_EXP_NAME="${NMFLI_EXP_DIR}/${dataset}-${target_label}_label-${model}-${policy}_policy"
    nohup \
    python3 -u src/federated_main.py \
        --model=${model} \
        --dataset=${dataset} \
        --target_label=${target_label} \
        --num_users=10 \
        --policy=${policy} \
        --iid=0 $remain_arg \
        > ${NMFLI_EXP_NAME}.log 2>&1
done
done
done
done