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

CONDA_ENV="mri_recon_nnUNet"
conda create -n $CONDA_ENV python=3.10.9 anaconda

#Install Pytorch with conda https://pytorch.org/get-started/locally/
conda run -n $CONDA_ENV conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

#Install nnUNet
conda run -n $CONDA_ENV pip install nnunet

conda run -n $CONDA_ENV export nnUNet_raw_data_base="$NNUNET_DIR"
conda run -n $CONDA_ENV export nnUNet_preprocessed="$NNUNET_DIR"
conda run -n $CONDA_ENV export RESULTS_FOLDER="$NNUNET_DIR"

#---Check for pretrained weights---
if ! [ -d "$NNUNET_DIR"/nnUNet/2d/Task027_ACDC ] ; then
	echo "Downloading pretrained weights for Task027_ACDC"
	conda run -n $CONDA_ENV nnUNet_download_pretrained_model Task027_ACDC
fi