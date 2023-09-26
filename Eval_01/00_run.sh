#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" || exit
cd ../01_scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export FIGURE_OUT=$FIGURE_DIR/Eval_01
export NNUNET_OUTPUT=$NNUNET_DIR/output

# Cardiac function evaluation of comDL and nnU-Net
cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.write_parameter_files_mc_comdl_nnunet(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"

# Cardiac function evaluation of nnU-Net with automatic and semi-automatic phase selection
cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.write_parameter_files_nnunet_auto(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"