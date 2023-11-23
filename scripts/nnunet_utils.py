#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Copyright 2023. Uecker Lab, University Medical Center Goettingen.
author: Martin Schilling (martin.schilling@med.uni-goettingen.de)
date: 2023

DESCRIPTION :
These functions help with the conversion to and from the Nifti format demanded by the neural network nnUNet
for 3D biomedical image segmentation.
https://github.com/MIC-DKFZ/nnUNet

Feel free to contact me, if you have problems with this script or have ideas for improvement.
"""

import sys, os
import SimpleITK as sitk
import nibabel as nib
import numpy as np

from inspect import getsourcefile

sys.path.append(os.path.join(os.environ["TOOLBOX_PATH"], "python"))
sys.path.append(os.path.dirname(getsourcefile(lambda:0)))

import cfl

def array2nifti(input:np.array, out_path:os.path, pixel_spacing:tuple=(1.328125,1.328125)):
	"""
	Create a Nifti file from a 2D numpy array in format. [xdim, ydim]
	The pixel spacing will be set according to a given parameter.
	The conversion is done with SimpleITK
	Based on https://simpleitk.readthedocs.io/en/v1.1.0/Examples/DicomSeriesReadModifyWrite/Documentation.html

	:param np.array input: Input 2D array
	:param os.path out_path: Path for output Nifti
	:param tuple pixel_spacing: Pixel spacing [mm] for setting the spacing in the Nifti
	"""
	#Reshape because nnUNet is designed for 3D segmentation
	if 2 == len(input.shape):
		input = np.reshape(input, (1, input.shape[0], input.shape[1]))

	img = sitk.GetImageFromArray(input)
	img.SetSpacing([pixel_spacing[0], pixel_spacing[1],1 ])

	writer = sitk.ImageFileWriter()
	writer.SetFileName(out_path)
	writer.Execute(img)

def z_score(img:np.array):
	"""
	Transform the intensity of 2D image to z-score: Mean of 0 and standard deviation of 1.

	:param np.array img: Input
	:returns: z-scored array
	:rtype:np.array
	"""
	img -= np.mean(img)
	img /= np.std(img)
	return img

def crop_2darray(arr:np.array, cdim:int):
	"""
	Center crop input array of shape [xdim, ydim] to [cdim, cdim]

	:param np.array arr: 2D input array
	:returns: Cropped array
	:rtype: np.array
	"""
	x_center = arr.shape[0] // 2
	y_center = arr.shape[1] // 2
	arr = arr[int(x_center-cdim/2):int(x_center+cdim/2), int(y_center-cdim/2):int(y_center+cdim/2)]
	return arr

def cfl2nifti(input_file:os.path, output_prefix:os.path, selection_array:list=[], selection_output:list=[], pixel_spacing:tuple=(1,1), crop_dim=None):
	"""
	Transform a CFL file into multiple Nifti files featuring single images, which are suitable as input for nnUNet.

	:param os.path input_file: Input CFL file in format [xdim, ydim, 1,1,1,1,1,1,1,1,phase, slice]
	:param os.path output_prefix: Prefix for nnUNet files. Files will have the format <prefi>xxyyy where xx
		represents the slice index, yyy the phase index
	:param list selection_array: Optional input for sole transformation of selected images. Entries have format [slice, phase]
	:param list selection_output: Optional input for naming transformed selected images. Entries have format [slice, phase]
	:param tuple pixel_spacing: Pixel spacing [mm] of input
	:param int crop_dim: Size of square dimensions to crop image to
	"""
	input = cfl.readcfl(input_file)
	if len(input.shape) > 11:
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], input.shape[-1]))
	else:
		# one-slice input
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], 1))

	if 0 == len(selection_output):
		selection_output = selection_array.copy()
	elif len(selection_output) != len(selection_array):
		sys.exit("Array selection and output selection must have the same length!")

	if 0 == len(selection_array):
		for img_slice in range(input.shape[3]):
			for img_phase in range(input.shape[2]):
				output_path = os.path.join(output_prefix + str(img_slice).zfill(2) + str(img_phase).zfill(3) + "_0000.nii.gz")
				input_array = z_score(np.abs(input[:,:,img_phase,img_slice]))
				input_array = crop_2darray(input_array, crop_dim) if None != crop_dim else input_array
				array2nifti(input_array, output_path, pixel_spacing)
	else:
		for (slct_arr, slct_out) in zip(selection_array, selection_output):
			img_slice = slct_arr[0]
			img_phase = slct_arr[1]
			output_path = os.path.join(output_prefix + str(slct_out[0]).zfill(2) + str(slct_out[1]).zfill(3) + "_0000.nii.gz")
			input_array = z_score(np.abs(input[:,:,img_phase,img_slice]))
			input_array = crop_2darray(input_array, crop_dim) if None != crop_dim else input_array
			array2nifti(input_array, output_path, pixel_spacing)

def cfl2nifti_slice(input_file:os.path, output_prefix:os.path, slice_selection:list=[], pixel_spacing:tuple=(1,1)):
	"""
	Transform a CFL file into multiple Nifti files featuring single images, which are suitable as input for nnUNet.

	:param os.path input_file: Input CFL file in format [xdim, ydim, 1,1,1,1,1,1,1,1,phase, slice]
	:param os.path output_prefix: Prefix for nnUNet files. Files will have the format <prefi>xxyyy where xx
		represents the slice index, yyy the phase index
	:param list selection: Optional input for sole transformation selected images. Entries have format [slice, phase]
	:param tuple pixel_spacing: Pixel spacing [mm] of input
	"""
	input = cfl.readcfl(input_file)
	if len(input.shape) > 11:
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], input.shape[-1]))
	else:
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], 1))
	if 0 == len(slice_selection):
		for img_slice in range(input.shape[3]):
			output_path = os.path.join(output_prefix + str(img_slice).zfill(2) + "_0000.nii.gz")
			input_array = z_score(np.abs(input[:,:,:,img_slice]))
			input_array = np.transpose(input_array, [2,0,1])
			array2nifti(input_array, output_path, pixel_spacing)
	else:
		for img_slice in slice_selection:
			output_path = os.path.join(output_prefix + str(img_slice).zfill(2) + "_0000.nii.gz")
			input_array = z_score(np.abs(input[:,:,:,img_slice]))
			input_array = np.transpose(input_array, [2,0,1])
			array2nifti(input_array, output_path, pixel_spacing)


def cfl2nifti_3d(input_file:os.path, output_prefix:os.path, phase_selection:list=[], slice_selection:list=[], pixel_spacing:tuple=(1,1)):
	"""
	Transform a CFL file into multiple Nifti files featuring 3D images, which are suitable as input for nnUNet.

	:param os.path input_file: Input CFL file in format [xdim, ydim, 1,1,1,1,1,1,1,1,phase, slice]
	:param os.path output_prefix: Prefix for nnUNet files. Files will have the format <prefix>xxyyy where xx
		represents the slice index, yyy the phase index
	:param list selection: Optional input for sole transformation selected images. Entries have format [slice, phase]
	:param tuple pixel_spacing: Pixel spacing [mm] of input
	"""
	input = cfl.readcfl(input_file)
	if len(input.shape) > 11:
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], input.shape[-1]))
	else:
		input = np.reshape(input, (input.shape[0], input.shape[1], input.shape[10], 1))
	slice_list = [i for i in range(input.shape[3])] if []==slice_selection else slice_selection
	if 0 == len(phase_selection):
		for img_phase in range(input.shape[2]):
			output_path = os.path.join(output_prefix + str(img_phase).zfill(3) + "_0000.nii.gz")
			input_list = []
			for img_slice in slice_list:
				input_list.append(z_score(np.abs(input[:,:,img_phase,img_slice])))
			input_array = np.stack(input_list, axis=0)
			array2nifti(input_array, output_path, pixel_spacing)
	else:
		for img_phase in phase_selection:
			output_path = os.path.join(output_prefix + str(img_phase).zfill(3) + "_0000.nii.gz")
			input_list = []
			for img_slice in slice_list:
				input_list.append(z_score(np.abs(input[:,:,img_phase,img_slice])))
			input_array = np.stack(input_list, axis=0)
			array2nifti(input_array, output_path, pixel_spacing)

def read_nnunet_segm(file_path:os.path):
	"""
	Convert Nifti output of nnUNet segmentation to a numpy array.

	:param os.path file_path: Input Nifti
	:returns: Image
	:rtype: np.array
	"""
	nib_data = nib.load(file_path)
	img = nib_data.get_fdata()
	img = np.transpose(img, [1,0,2])
	return img