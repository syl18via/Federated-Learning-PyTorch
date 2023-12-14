#!/bin/bash
set -x
export NMFLI_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`

# Hyper-parameters
export UNEVEN_DATASIZE=0
export REQUIRE_CLIENT_NUM="3_2"

# Hyper-parameters that will be tried in one experiment
DATASETS=(cifar)
MODELS=(cnn)
TARGET_LABEL_CFGS=(non_overlap overlap identical)
ALL_METHODS=(random afl size nmfli greedy)

TARGET_LABEL_CFGS=(non_overlap)
# TARGET_LABEL_CFGS=(overlap identical)
# ALL_METHODS=(random)

if [[ ${UNEVEN_DATASIZE} == "1" ]]; then
    NMFLI_EXP_DIR=save/results/${NMFLI_EXP_DATETIME}-req_${REQUIRE_CLIENT_NUM}_client-uneven_datasize
else
    NMFLI_EXP_DIR=save/results/${NMFLI_EXP_DATETIME}-req_${REQUIRE_CLIENT_NUM}_client
fi
mkdir -p $NMFLI_EXP_DIR
for dataset in ${DATASETS[@]}; do
for target_label in ${TARGET_LABEL_CFGS[@]}; do
for model in ${MODELS[@]}; do
for policy in ${ALL_METHODS[@]}; do
    export NMFLI_EXP_NAME="${NMFLI_EXP_DIR}/${dataset}-${target_label}_label-${model}-${policy}_policy"
    if [[ $1 = "debug" ]]; then
        python3 -u src/federated_main.py \
        --gpu 0 \
        --model=${model} \
        --dataset=${dataset} \
        --target_label=${target_label} \
        --num_users=10 \
        --policy=${policy} \
        --iid=0 \
        --verbose=1
    else
        nohup \
        python3 -u src/federated_main.py \
            --gpu 0 \
            --model=${model} \
            --dataset=${dataset} \
            --target_label=${target_label} \
            --num_users=10 \
            --policy=${policy} \
            --iid=0 \
            > ${NMFLI_EXP_NAME}.log 2>&1
    fi
done
done
done
done