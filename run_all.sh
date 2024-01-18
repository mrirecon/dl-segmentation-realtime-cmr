#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

SCRIPT_REPO="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/scripts/ || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

# Check data repository
FOLDERS=(contour_files end_expiration_indexes images nnUNet_inference)
for f in "${FOLDERS[@]}"
do
	if [[ ! -d $DATA_DIR/$f  ]] ; then
		bash "$SCRIPT_REPO"/load.sh
	fi
done

# Install required conda environment
if ! { conda env list | grep "$CONDA_ENV_EVAL" ; } >/dev/null 2>&1 ; then
	bash "$SCRIPT_REPO"/scripts/assess_dl_conda_env.sh
fi

# Reproduce figures
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_01/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_02/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_03/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_04/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_s1/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_s2/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_s3/run.sh
conda run -n "$CONDA_ENV_EVAL" bash "$SCRIPT_REPO"/Figure_s4/run.sh