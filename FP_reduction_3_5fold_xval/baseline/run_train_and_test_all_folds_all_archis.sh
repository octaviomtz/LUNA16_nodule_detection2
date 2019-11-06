#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate my_env
for i in {0..4}; do
    python train_FP_reduction_fold_k_all_archis.py $i
    python test_FP_reduction_fold_k.py $i
done
python merge_fold_results.py
