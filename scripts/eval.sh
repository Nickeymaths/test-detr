#!/bin/bash
python main.py --batch_size 8 \
--no_aux_loss --dataset_file thyroid \
--brighness_levels 7 \
--num_queries 70 \
--eval --resume output/thyroid_dcm_7_icrsbl/best.pth \
--data_path data/thyroid_493
