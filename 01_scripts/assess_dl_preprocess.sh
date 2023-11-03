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
# Task512_rtvolrt_stress_single
# Task517_rtvolrt_stress
# Task513_rtvolrt_maxstress_single
# Task518_rtvolrt_maxstress

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/01_scripts || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export SCRIPT_DIR=$SCRIPT_REPO/01_scripts
export NNUNET_DIR=$DATA_DIR/nnUNet
export IMG_DIR=$DATA_DIR/scanner_reco/
export CONTOUR_DIR=$DATA_DIR/contour_files/

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg_utils as assess_utils;'
	'assess_utils.prepare_nnunet(img_dir=os.environ["IMG_DIR"],'
	'contour_dir=os.environ["CONTOUR_DIR"],nnunet_dir=os.environ["NNUNET_DIR"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"