#!/bin/bash

results_filename=/home/se14/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_1/final_results/3D_CNN_FP_reduction.csv
out_path=/home/se14/Documents/LIDC/LUNA16_nodule_detection/FP_reduction_1/FROC_size_analysis_results/

eval "$(conda shell.bash hook)"
conda activate my_env_p2
python /home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/noduleCADEvaluationLUNA16_analyse_by_size.py /home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/annotations.csv /home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/annotations_excluded.csv /home/se14/Documents/LIDC/LUNA16_nodule_detection/LUNA16_data/evaluationScript/annotations/seriesuids.csv $results_filename $out_path