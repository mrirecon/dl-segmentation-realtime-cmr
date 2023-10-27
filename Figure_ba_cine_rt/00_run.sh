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
export PARAM_DIR=$SCRIPT_REPO/Eval_01
export FIGURE_OUT=$SCRIPT_REPO/Figure_ba_cine_rt

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.save_figba_cine_rt(out_dir=os.environ["FIGURE_OUT"],param_dir=os.environ["PARAM_DIR"])')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

IN_FIG=("$FIGURE_OUT"/BA_cine_rt_EDV.png "$FIGURE_OUT"/BA_cine_rt_ESV.png "$FIGURE_OUT"/BA_cine_rt_EF.png)
bash "$SCRIPT_DIR"/43_annotate_inkscape.sh "$FIGURE_OUT"/figure_ba_cine_rt_template.svg "$FIGURE_OUT"/figure_b3_cf_cine_rt.png "${IN_FIG[@]}"
