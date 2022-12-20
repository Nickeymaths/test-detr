#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/thyroid \
--batch_size 8 \
--dilation \
--set_cost_class 3 \
--output_dir output/thyroid_dc5_dcm_icr_f10 \
--resume weights/detr-r50-dc5-panoptic-da08f1b1.pth