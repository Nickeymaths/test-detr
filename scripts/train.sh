#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/thyroid_493 \
--batch_size 8 \
--brighness_levels 7 \
--set_cost_class 2 \
--num_queries 70 \
--output_dir output/thyroid_dcm_7_icrsbl \
--resume weights/detr-r50-e632da11.pth