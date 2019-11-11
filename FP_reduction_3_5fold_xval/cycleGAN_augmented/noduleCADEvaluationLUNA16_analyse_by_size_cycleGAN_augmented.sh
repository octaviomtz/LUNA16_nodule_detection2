#!/bin/bash

# results_filename should be the full path to the csv file in the final results folder
# out_path should be the current path + FROC_size_analysis_results/
# eval_script_path is the full path to the 'evaluationScript' folder (including 'evaluationScript')

results_filename=/home/se14/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/final_results/multilevel_3D_CNN_FP_reduction.csv
out_path=/home/se14/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/FROC_size_analysis_results/
eval_script_path=/home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/

eval "$(conda shell.bash hook)"
conda activate my_env_p2
echo ${eval_script_path}noduleCADEvaluationLUNA16_analyse_by_size.py ${eval_script_path}annotations/annotations.csv ${eval_script_path}annotations/annotations_excluded.csv ${eval_script_path}annotations/seriesuids.csv $results_filename $out_path