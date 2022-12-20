#!/bin/bash
python test.py --data_path data/thyroid/images/test \
--dataset_file thyroid --resume output/thyroid_dcm_icr_f10/checkpoint.pth \
--output_dir infer_out/thyroid_dcm_icr_f10 \
--thresh 0.5