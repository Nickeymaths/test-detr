#!/bin/bash
python main.py \
--dataset_file thyroid \
--data_path data/thyroid \
--output_dir _output/thyroid \
--resume weights/detr-r50-e632da11.pth
