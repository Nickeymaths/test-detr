#!/bin/bash
python test.py --data_path data/thyroid_493/images/test \
--dataset_file thyroid --resume output/thyroid_dcm_7_icrsbl/best.pth \
--output_dir infer_out/thyroid_dcm_7_icrsbl_data_493/test \
--brighness_levels 7 \
--num_queries 70 \
--thresh 0.7
