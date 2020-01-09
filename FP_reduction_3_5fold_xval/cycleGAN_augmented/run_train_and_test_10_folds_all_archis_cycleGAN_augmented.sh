#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate my_env
for i in {0..9}; do
    python train_FP_reduction_10fold_k_all_archis_cycleGAN_augmented.py $i
    python test_FP_reduction_10fold_k_cycleGAN_augmented.py $i
done
python merge_fold_results_cycleGAN_augmented.py
