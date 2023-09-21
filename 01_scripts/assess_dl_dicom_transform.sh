#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

# To be able to activate conda environments,
# run this script in interactive mode with bash -i assess_dl_nnunet_install.sh

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR" || exit

FILES=()
if [ -d "$DICOM_DIR" ] ; then
	if ! [ "$DICOM_PATTERN" = "" ] ; then
		FILES=("$DICOM_DIR"/*"$DICOM_PATTERN"*)
	else
		FILES=("$DICOM_DIR"/*)
	fi
fi

echo "${FILES[@]}"

for d in "${FILES[@]}"
do
	if [[ "$d" == *".tgz"* ]] ; then
		tar zxf "$d"
		export DICOM_DIR=$PWD
	fi
done

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_CINE"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="cine_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT_STRESS"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_Belastung_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["PYTHON_SCRIPTS"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT_MAX_STRESS"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_Ausbelastung_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"