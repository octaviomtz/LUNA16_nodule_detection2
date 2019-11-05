#!/bin/bash

results_filename=/home/om18/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/final_results/multilevel_3D_CNN_FP_reduction.csv
out_path=/home/om18/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/FROC_size_analysis_results/

eval "$(conda shell.bash hook)"
conda activate my_env_p2
python /home/om18/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/noduleCADEvaluationLUNA16_analyse_by_size.py /home/om18/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/annotations.csv /home/om18/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/annotations_excluded.csv /home/om18/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/seriesuids.csv $results_filename $out_path