#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

SCRIPT_REPO="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/01_scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export SCRIPT_DIR=$SCRIPT_REPO/01_scripts
export FIGURE_OUT=$FIGURE_DIR/Eval_01

# Cardiac function evaluation of comDL and nnU-Net
cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.write_parameter_files_mc_comdl_nnunet(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"

# Cardiac function evaluation of nnU-Net with automatic and semi-automatic phase selection
cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.write_parameter_files_nnunet_auto(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"