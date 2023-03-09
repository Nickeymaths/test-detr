#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/detr_data \
--iou_thrs_list '0.3,0.5' \
--batch_size 8 \
--backbone resnet50 \
--brighness_levels 4 \
--set_cost_class 2 \
--num_queries 75 \
--output_dir output/resnet50_75nq_coscl2_4bl \
--resume weights/detr-r50-e632da11.pth