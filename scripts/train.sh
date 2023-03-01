#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/detr_data \
--batch_size 8 \
--backbone resnet50 \
--brighness_levels 3 \
--set_cost_class 1.5 \
--num_queries 70 \
--output_dir output/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt \
--resume weights/detr-r50-e632da11.pth