#!/bin/bash
python test.py --data_path data/thyroid/images/test \
--dataset_file thyroid --resume output/thyroid/checkpoint.pth \
--output_dir infer_out/thyroid