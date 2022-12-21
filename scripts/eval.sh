#!/bin/bash
python main.py --batch_size 8 \
--no_aux_loss --dataset_file thyroid \
--eval --resume output/thyroid_dcm_5_icrsbl/checkpoint.pth \
--data_path data/thyroid
