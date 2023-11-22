#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

# To be able to activate conda environments,
# run this script in interactive mode with bash -i assess_dl_conda_env.sh

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/01_scripts || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

export NNUNET_DIR=$DATA_DIR/nnUNet

conda create -y -n "$CONDA_ENV_EVAL" python=3.10.9 anaconda
#required for reproducibility of figures
conda run -n "$CONDA_ENV_EVAL" conda install -y -c simpleitk simpleitk
conda run -n "$CONDA_ENV_EVAL" conda install -y -c conda-forge nibabel
#qt plugin installed within conda causes errors
conda run -n "$CONDA_ENV_EVAL" conda remove -y --force pyqt pyqt5-sip pyqtwebengine qt-main qt-webengine qtawesome qtconsole qtpy qtwebkit sphinxcontrib-qthelp