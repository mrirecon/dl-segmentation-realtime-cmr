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
export FIGURE_OUT=$SCRIPT_REPO/Figure_ba

if ! [ -f "$FIGURE_OUT"/BA_nnunet_EDV_cine.png ] ; then
	cmd_array=(	'import sys,os;'
		'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
		'import assess_dl_seg;'
		'assess_dl_seg.save_figba(out_dir=os.environ["FIGURE_OUT"])')

	cmd="${cmd_array[*]}"
	python3 -c "$cmd"
fi

IN_FIG=("$FIGURE_OUT"/BA_nnunet_EDV_cine.png "$FIGURE_OUT"/BA_nnunet_EDV_rt.png "$FIGURE_OUT"/BA_nnunet_EDV_rt_stress.png)
IN_FIG+=("$FIGURE_OUT"/BA_nnunet_ESV_cine.png "$FIGURE_OUT"/BA_nnunet_ESV_rt.png "$FIGURE_OUT"/BA_nnunet_ESV_rt_stress.png)
IN_FIG+=("$FIGURE_OUT"/BA_nnunet_EF_cine.png "$FIGURE_OUT"/BA_nnunet_EF_rt.png "$FIGURE_OUT"/BA_nnunet_EF_rt_stress.png)
bash "$SCRIPT_DIR"/43_annotate_inkscape.sh "$FIGURE_OUT"/figure_ba_template.svg "$FIGURE_OUT"/figure_b1_cf_nnunet.png "${IN_FIG[@]}"

IN_FIG=("$FIGURE_OUT"/BA_comDL_EDV_cine.png "$FIGURE_OUT"/BA_comDL_EDV_rt.png "$FIGURE_OUT"/BA_comDL_EDV_rt_stress.png)
IN_FIG+=("$FIGURE_OUT"/BA_comDL_ESV_cine.png "$FIGURE_OUT"/BA_comDL_ESV_rt.png "$FIGURE_OUT"/BA_comDL_ESV_rt_stress.png)
IN_FIG+=("$FIGURE_OUT"/BA_comDL_EF_cine.png "$FIGURE_OUT"/BA_comDL_EF_rt.png "$FIGURE_OUT"/BA_comDL_EF_rt_stress.png)
bash "$SCRIPT_DIR"/43_annotate_inkscape.sh "$FIGURE_OUT"/figure_ba_template.svg "$FIGURE_OUT"/figure_b2_cf_comDL.png "${IN_FIG[@]}"
