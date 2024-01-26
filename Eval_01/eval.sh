#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export SCRIPT_DIR=$SCRIPT_REPO/scripts
export FIGURE_OUT=$SCRIPT_REPO/Eval_01

eval_cardiac_function()
(
	helpstr=$(cat <<- EOF
	Evaluate cardiac function by comparing cardiac function parameters between manually corrected contours (REF) and a reference.
	-h help
	EOF
	)

	usage="Usage: $0 [-h] <reference> <comparison> <output>"

	local OPTIND opt
	while getopts "h" opt; do
		case $opt in
		h)
			echo "$usage"
			echo
			echo "$helpstr"
			exit 0
		;;
		\?)
			echo "$usage" >&2
			exit 1
		;;
		esac
	done

	shift $((OPTIND - 1))

	if [ $# -lt 3 ] ; then

		echo "$usage" >&2
		exit 1
	fi

	local FILE_REF
	FILE_REF=$(readlink -f "$1")
	local FILE_COMP
	FILE_COMP=$(readlink -f "$2")
	local FILE_OUT
	FILE_OUT=$(readlink -f "$3")

	export FILE_REF
	export FILE_COMP

	cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import assess_dl_seg;'
	'assess_dl_seg.print_abs_rel_diff(file_ref=os.environ["FILE_REF"],file_comp=os.environ["FILE_COMP"])')

	cmd="${cmd_array[*]}"
	python3 -c "$cmd" > "$FILE_OUT"

)

REF_CINE=$FIGURE_OUT/cardiac_function_mc_cine.txt
REF_RT=$FIGURE_OUT/cardiac_function_mc_rt.txt
REF_RT_STRESS=$FIGURE_OUT/cardiac_function_mc_rt_stress.txt

eval_cardiac_function "$REF_CINE"  "$FIGURE_OUT"/cardiac_function_nnunet_cine.txt "$FIGURE_OUT"/diff_nnunet_cine.txt
eval_cardiac_function "$REF_RT"  "$FIGURE_OUT"/cardiac_function_nnunet_rt.txt "$FIGURE_OUT"/diff_nnunet_rt.txt
eval_cardiac_function "$REF_RT_STRESS"  "$FIGURE_OUT"/cardiac_function_nnunet_rt_stress.txt "$FIGURE_OUT"/diff_nnunet_rt_stress.txt

eval_cardiac_function "$REF_CINE"  "$FIGURE_OUT"/cardiac_function_comDL_cine.txt "$FIGURE_OUT"/diff_comDL_cine.txt
eval_cardiac_function "$REF_RT"  "$FIGURE_OUT"/cardiac_function_comDL_rt.txt "$FIGURE_OUT"/diff_comDL_rt.txt
eval_cardiac_function "$REF_RT_STRESS"  "$FIGURE_OUT"/cardiac_function_comDL_rt_stress.txt "$FIGURE_OUT"/diff_comDL_rt_stress.txt

eval_cardiac_function "$REF_CINE"  "$FIGURE_OUT"/cardiac_function_mc_cine_intra.txt "$FIGURE_OUT"/diff_mc_cine_intra.txt
eval_cardiac_function "$REF_RT"  "$FIGURE_OUT"/cardiac_function_mc_rt_intra.txt "$FIGURE_OUT"/diff_mc_rt_intra.txt
eval_cardiac_function "$REF_RT_STRESS"  "$FIGURE_OUT"/cardiac_function_mc_rt_stress_intra.txt "$FIGURE_OUT"/diff_mc_rt_stress_intra.txt

eval_cardiac_function "$REF_CINE"  "$FIGURE_OUT"/cardiac_function_mc_cine_inter.txt "$FIGURE_OUT"/diff_mc_cine_inter.txt
eval_cardiac_function "$REF_RT"  "$FIGURE_OUT"/cardiac_function_mc_rt_inter.txt "$FIGURE_OUT"/diff_mc_rt_inter.txt
eval_cardiac_function "$REF_RT_STRESS"  "$FIGURE_OUT"/cardiac_function_mc_rt_stress_inter.txt "$FIGURE_OUT"/diff_mc_rt_stress_inter.txt