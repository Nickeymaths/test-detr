#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/thyroid \
--batch_size 8 \
--dilation \
--brighness_levels 5 \
--set_cost_class 3 \
--output_dir output/thyroid_dcm_5_icrsbl \
--resume weights/detr-r50-e632da11.pth