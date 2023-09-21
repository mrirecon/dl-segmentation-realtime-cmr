#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

# create pre-processed nnU-Net raw data
# Task501_rtvolcine_single
# Task502_rtvolcine3d
# Task503_rtvolcine3d_LV
# Task511_rtvolrt_single
# Task516_rtvolrt
# Task512_rtvolrt_Belastung_single
# Task517_rtvolrt_Belastung
# Task513_rtvolrt_Ausbelastung_single
# Task518_rtvolrt_Ausbelastung

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import assess_dl_seg_utils as assess_utils;'
	'assess_utils.prepare_nnunet(img_dir=os.environ["IMG_DIR"],'
	'contour_dir=os.environ["CONTOUR_DIR"],nnunet_dir=os.environ["NNUNET_DIR"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"