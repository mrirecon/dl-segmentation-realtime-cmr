#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

SCRIPT_REPO="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_REPO"/01_scripts || exit

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

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task501_rtvolcine_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_cine_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --save_npz --overwrite_existing -m 2d

#---nnU-Net ensemble for all slices---

#INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task502_rtvolcine3d/imagesTs/
#OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_cine_2d_slice_cv/
#if ! [ -d $OUTPUT_DIRECTORY_2D ] ; then
#	mkdir -p $OUTPUT_DIRECTORY_2D
#fi
#conda run -n "$CONDA_ENV" nnUNet_predict -i $INPUT_DIR -o $OUTPUT_DIRECTORY_2D -t 27 --save_npz --overwrite_existing -m 2d
#
#INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task502_rtvolcine3d/imagesTs/
#OUTPUT_DIRECTORY_3D=$RESULTS_FOLDER/output/rtvol_cine_3d_cv/
#if ! [ -d $OUTPUT_DIRECTORY_3D ] ; then
#	mkdir -p $OUTPUT_DIRECTORY_3D
#fi
#conda run -n "$CONDA_ENV" nnUNet_predict -i $INPUT_DIR -o $OUTPUT_DIRECTORY_3D -t 27 --save_npz --overwrite_existing -m 3d_fullres
#
#OUTPUT_FOLDER_ENSEMBLE=/scratch/mschi/nnUnet/output/rtvol_cine_ensemble/
#if ! [ -d $OUTPUT_FOLDER_ENSEMBLE ] ; then
#	mkdir -p $OUTPUT_FOLDER_ENSEMBLE
#fi
#POSTPROCESSING_FILE=$nnUNet_preprocessed/nnUNet/ensembles/Task027_ACDC/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json
##run ensemble
#conda run -n "$CONDA_ENV" nnUNet_ensemble -f $OUTPUT_DIRECTORY_3D $OUTPUT_DIRECTORY_2D -o $OUTPUT_FOLDER_ENSEMBLE -pp $POSTPROCESSING_FILE

#---nnU-Net ensemble for slices containing left ventricle---

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task503_rtvolcine3d_LV/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_cine_2d_LV_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --save_npz --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task503_rtvolcine3d_LV/imagesTs/
OUTPUT_DIRECTORY_3D=$RESULTS_FOLDER/output/rtvol_cine_3d_LV_cv/
if ! [ -d "$OUTPUT_DIRECTORY_3D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_3D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_3D" -t 27 --save_npz --overwrite_existing -m 3d_fullres

OUTPUT_FOLDER_ENSEMBLE=$RESULTS_FOLDER/output/rtvol_cine_LV_ensemble/
if ! [ -d "$OUTPUT_FOLDER_ENSEMBLE" ] ; then
	mkdir -p "$OUTPUT_FOLDER_ENSEMBLE"
fi
POSTPROCESSING_FILE=$nnUNet_preprocessed/nnUNet/ensembles/Task027_ACDC/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json
#run ensemble
conda run -n "$CONDA_ENV" nnUNet_ensemble -f "$OUTPUT_DIRECTORY_3D" "$OUTPUT_DIRECTORY_2D" -o "$OUTPUT_FOLDER_ENSEMBLE" -pp "$POSTPROCESSING_FILE"

exit 0
#---real-time MRI---

#---single images---
INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task511_rtvolrt_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_rt_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task512_rtvolrt_stress_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_rt_stress_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task513_rtvolrt_maxstress_single/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_rt_maxstress_2d_single_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

#---slices---

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task516_rtvolrt/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_rt_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task517_rtvolrt_stress/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/rtvol_rt_stress_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d

INPUT_DIR=$nnUNet_raw_data_base/nnUNet_raw_data/Task518_rtvolrt_maxstress/imagesTs/
OUTPUT_DIRECTORY_2D=$RESULTS_FOLDER/output/rtvol_rt_maxstress_2d_cv/
if ! [ -d "$OUTPUT_DIRECTORY_2D" ] ; then
	mkdir -p "$OUTPUT_DIRECTORY_2D"
fi
conda run -n "$CONDA_ENV" nnUNet_predict -i "$INPUT_DIR" -o "$OUTPUT_DIRECTORY_2D" -t 27 --overwrite_existing -m 2d
