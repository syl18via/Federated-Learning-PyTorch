#!/bin/bash
set -x
export NMFLI_EXP_DATETIME=`date '+%Y%m%d-%H%M%S'`

# Hyper-parameters
export UNEVEN_DATASIZE=1
# export REQUIRE_CLIENT_NUM="1_2"
export USE_NOISY_X=0

# Hyper-parameters that will be tried in one experiment
DATASETS=(cifar)
MODELS=(cnn)
TARGET_LABEL_CFGS=(non_overlap overlap identical)
ALL_METHODS=(random afl size nmfli greedy)

TARGET_LABEL_CFGS=(identical)
# TARGET_LABEL_CFGS=(overlap identical)
ALL_METHODS=(nmfli)

# Decide the save data directory
NMFLI_EXP_DIR=save/results/${NMFLI_EXP_DATETIME}
if [[ -n ${REQUIRE_CLIENT_NUM} ]]; then
    NMFLI_EXP_DIR=${NMFLI_EXP_DIR}-req_${REQUIRE_CLIENT_NUM}_client 
fi
if [[ ${UNEVEN_DATASIZE} == "1" ]]; then
    NMFLI_EXP_DIR=${NMFLI_EXP_DIR}-uneven_datasize
else
    NMFLI_EXP_DIR=${NMFLI_EXP_DIR}-even_datasize
fi
if [[ ${USE_NOISY_X} == "1" ]]; then
    NMFLI_EXP_DIR=${NMFLI_EXP_DIR}-noisy_x 
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
        --verbose=1 \
        --noisy=${USE_NOISY_X} \
        --unequal=${UNEVEN_DATASIZE}
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
            --noisy=${USE_NOISY_X} \
            --unequal=${UNEVEN_DATASIZE} \
            > ${NMFLI_EXP_NAME}.log 2>&1
    fi
done
done
done
done