#!/bin/bash
python main.py --batch_size 2 \
--no_aux_loss --dataset_file thyroid \
--eval --resume output/thyroid_dcm_icr_f10/checkpoint.pth \
--data_path data/thyroid
