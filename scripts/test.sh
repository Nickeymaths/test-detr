#!/bin/bash
python test.py --data_path data/detr_data/images/test \
--dataset_file thyroid --resume output/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt/best.pth \
--backbone resnet50 \
--output_dir infer_out/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt/test \
--brighness_levels 3 \
--set_cost_class 2 \
--num_queries 70 \
--thresh 0.8
