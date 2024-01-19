#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#
# nnU-Net version 1 has been used for segmentation: https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1
# All images were segmented with pretrained weights, which were downloaded with: nnUNet_download_pretrained_model Task027_ACDC

SCRIPT_REPO="$( cd "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/scripts || exit

read -ra param < <(grep -v '^#' assess_dl.env | xargs)
export "${param[@]}"

NNUNET_DIR=$DATA_DIR/nnUNet

export nnUNet_raw_data_base=$NNUNET_DIR
export nnUNet_preprocessed=$NNUNET_DIR
export RESULTS_FOLDER=$NNUNET_DIR

# speedup options:
# use only single fold:	-f 0
# disable test time data augmentation: --disable_tta

# 2D nnUNet, --save_npz to save softmax probabilities


#---cine MRI---

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task501_cine_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/cine_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --save_npz --overwrite_existing -m 2d

#---nnU-Net ensemble for all slices---

#INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task502_cine3d/imagesTs/
#OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/cine_2d_slice_cv/
#if ! [ -d $OUTPUT_DIRECTORY_2D ] ; then
#	mkdir -p $OUTPUT_DIRECTORY_2D
#fi
#nnUNet_predict -i $INPUT_DIR -o $OUTPUT_DIRECTORY_2D -t 27 --save_npz --overwrite_existing -m 2d
#
#INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task502_cine3d/imagesTs/
#OUTPUT_DIRECTORY_3D=$RESULTS_FOLDER/output/cine_3d_cv/
#if ! [ -d $OUTPUT_DIRECTORY_3D ] ; then
#	mkdir -p $OUTPUT_DIRECTORY_3D
#fi
#nnUNet_predict -i $INPUT_DIR -o $OUTPUT_DIRECTORY_3D -t 27 --save_npz --overwrite_existing -m 3d_fullres
#
#OUTPUT_FOLDER_ENSEMBLE=/scratch/mschi/nnUnet/output/cine_ensemble/
#if ! [ -d $OUTPUT_FOLDER_ENSEMBLE ] ; then
#	mkdir -p $OUTPUT_FOLDER_ENSEMBLE
#fi
#POSTPROCESSING_FILE=$nnUNet_preprocessed/nnUNet/ensembles/Task027_ACDC/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json
##run ensemble
#nnUNet_ensemble -f $OUTPUT_DIRECTORY_3D $OUTPUT_DIRECTORY_2D -o $OUTPUT_FOLDER_ENSEMBLE -pp $POSTPROCESSING_FILE

#---nnU-Net ensemble for slices containing left ventricle---

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task503_cine3d_LV/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/cine_2d_LV_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --save_npz --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task503_cine3d_LV/imagesTs/
OUTPUT_DIRECTORY_3D=$RESULTS_FOLDER/output/cine_3d_LV_cv/
if ! [ -d "$OUTPUT_DIRECTORY_3D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_3D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_3D" -t 27 --save_npz --overwrite_existing -m 3d_fullres

OUTPUT_FOLDER_ENSEMBLE=$RESULTS_FOLDER/output/cine_LV_ensemble/
if ! [ -d "$OUTPUT_FOLDER_ENSEMBLE" ] ; then
	mkdir -p "$OUTPUT_FOLDER_ENSEMBLE"
fi
POSTPROCESSING_FILE=$nnUNet_preprocessed/nnUNet/ensembles/Task027_ACDC/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json
#run ensemble
nnUNet_ensemble -f "$OUTPUT_DIRECTORY_3D" "$OUTPUT_DIRECTORY_2D" -o "$OUTPUT_FOLDER_ENSEMBLE" -pp "$POSTPROCESSING_FILE"


#---real-time MRI---

#---single images---
INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task511_rt_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rt_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task512_rt_stress_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rt_stress_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task513_rt_maxstress_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rt_maxstress_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

#---slices---

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task516_rt/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rt_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task517_rt_stress/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/rt_stress_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task518_rt_maxstress/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rt_maxstress_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d
