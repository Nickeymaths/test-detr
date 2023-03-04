#!/bin/bash
python main.py --batch_size 8 \
--no_aux_loss --dataset_file thyroid \
--iou_thrs_list '0.3,0.5' \
--backbone resnet50 \
--brighness_levels 3 \
--num_queries 70 \
--set_cost_class 1.5 \
--eval --resume output/thyroid_dcm_3_icrsbl_detr_data_r50_cost_class_1.5tt/best.pth \
--data_path data/detr_data
