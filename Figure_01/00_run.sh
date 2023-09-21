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

export FIGURE_OUT=$FIGURE_DIR/Figure_01
export NNUNET_OUTPUT=$RESULTS_FOLDER/output

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.save_fig1(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"