#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"

cd "$SCRIPT_REPO"/01_scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export SCRIPT_DIR=$SCRIPT_REPO/01_scripts
export PARAM_DIR=$SCRIPT_REPO/Eval_01
export FIGURE_OUT=$SCRIPT_REPO/Figure_ba_nnunet_auto

if ! [ -f "$FIGURE_OUT"/DC_vs_bpm_nnunet.png ] ; then
	cmd_array=(	'import sys,os;'
		'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
		'import assess_dl_seg;'
		'assess_dl_seg.save_figba_nnunet_auto(out_dir=os.environ["FIGURE_OUT"],param_dir=os.environ["PARAM_DIR"])')

	cmd="${cmd_array[*]}"
	python3 -c "$cmd"
fi

