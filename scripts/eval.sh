#!/bin/bash
python main.py --batch_size 8 \
--no_aux_loss --dataset_file thyroid \
--iou_thrs_list '0.5,0.5' \
--backbone resnet50 \
--brighness_levels 5 \
--num_queries 75 \
--set_cost_class 2 \
--eval --resume output/resnet50_75nq_coscl2_5bl/best.pth \
--data_path data/detr_data
