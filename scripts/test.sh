#!/bin/bash
python test.py --data_path data/detr_data/images/val \
--dataset_file thyroid --resume output/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt/best.pth \
--backbone resnet50 \
--output_dir infer_out/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt/val \
--brighness_levels 3 \
--set_cost_class 1.5 \
--num_queries 70 \
--thresh 0.8
