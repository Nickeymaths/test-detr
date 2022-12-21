#!/bin/bash
python test.py --data_path data/thyroid/images/test \
--dataset_file thyroid --resume output/thyroid_dcm_5_icrsbl/checkpoint.pth \
--output_dir infer_out/thyroid_dcm_5_icrsbl \
--brighness_levels 5 \
--thresh 0.7