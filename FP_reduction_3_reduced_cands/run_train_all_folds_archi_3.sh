#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate my_env
for i in {0..9}; do
    python train_FP_reduction_fold_k_archi_3.py $i
done
