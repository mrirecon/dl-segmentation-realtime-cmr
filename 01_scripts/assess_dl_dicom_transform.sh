#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

# To be able to activate conda environments,
# run this script in interactive mode with bash -i assess_dl_nnunet_install.sh

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/01_scripts || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR" || exit

# select pattern for DICOM to CFL transform, e.g. vol80
DICOM_PATTERN=""
export DICOM_DIR=$DATA_DIR/dicoms/
export PARAM_FILE_CINE=$DATA_DIR/dicom_transform/rtvol_cine.txt
export PARAM_FILE_RT=$DATA_DIR/dicom_transform/rtvol_rt.txt
export PARAM_FILE_RT_STRESS=$DATA_DIR/dicom_transform/rtvol_rt_stress.txt
export PARAM_FILE_RT_MAX_STRESS=$DATA_DIR/dicom_transform/rtvol_rt_maxstress.txt
export CFL_TARGET_DIR=$DATA_DIR/scanner_reco
export SCRIPT_DIR=$SCRIPT_REPO/01_scripts

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
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_CINE"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="cine_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT_STRESS"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_stress_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import dicom_to_cfl;'
	'dicom_to_cfl.read_dicom_param_file(param_file=os.environ["PARAM_FILE_RT_MAX_STRESS"],'
	'ddir=os.environ["DICOM_DIR"], tdir=os.environ["CFL_TARGET_DIR"], suffix="rt_maxstress_scanner")')
cmd="${cmd_array[*]}"
python3 -c "$cmd"