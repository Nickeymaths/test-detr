#!/bin/bash
python test.py --data_path data/detr_data/images/test \
--dataset_file thyroid --resume output/resnet50_75nq_coscl2_5bl/best.pth \
--backbone resnet50 \
--output_dir infer_out/resnet50_75nq_coscl2_5bl/test \
--brighness_levels 5 \
--set_cost_class 2 \
--num_queries 75 \
--thresh 0.8
