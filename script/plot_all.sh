#!/bin/bash
set -x
# NOTE: must be called from root directory of the project: bash script/plot_all.sh

# Cifar even dataset
## Non-overlap
python3 plot/exp1.py save/results/20231205-170051-even_size_custom-gpu-cifar

## Indentical and overlap 
python3 plot/exp1.py save/results/20231206-144643-even_size_custom-gpu-cifar

## Others
python3 plot/exp1.py save/results/20231215-123505-noisy_x
python3 plot/exp1.py save/results/20231214-040304-req_1_2_client
python3 plot/exp1.py save/results/20231214-040326-req_3_2_client


# Uneven data size

python3 plot/exp1.py save/results/20231216-100943-uneven_datasize


