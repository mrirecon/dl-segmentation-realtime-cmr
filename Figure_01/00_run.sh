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
export FIGURE_OUT=$SCRIPT_REPO/Figure_01

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.save_fig1(out_dir=os.environ["FIGURE_OUT"])')

cmd="${cmd_array[*]}"
python3 -c "$cmd"

bash "$SCRIPT_DIR"/43_annotate_inkscape.sh "$FIGURE_OUT"/figure_01_template.svg "$FIGURE_OUT"/figure_01_measurement_ED_ES.png "$FIGURE_OUT"/figure_01_a.png "$FIGURE_OUT"/figure_01_b.png
