#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

set -e

SCRIPT_REPO="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

if  [ "$DATA_DIR" = "" ] ; then
	echo "Environment variable DATA_DIR was not set in scripts/assess_dl.env"
	echo "Data will be downloaded into this directory by default"
	export DATA_DIR="$SCRIPT_REPO"
fi

record=10117944
FOLDERS=(contour_files end_expiration_indexes images nnUNet_inference)

cd "${DATA_DIR}"
for name in "${FOLDERS[@]}"
do

	if [[ ! -f ${name}.tgz ]]; then
		echo Downloading "${name}"
		wget -q https://zenodo.org/record/"${record}"/files/"${name}".tgz
	fi

	if [[ ! -d $DATA_DIR/$name  ]] ; then
		tar -xzvf "${name}".tgz
	fi
done