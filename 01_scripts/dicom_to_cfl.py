#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Copyright 2023. Uecker Lab, University Medical Center Goettingen.
author: Martin Schilling (martin.schilling@med.uni-goettingen.de)
date: 2023

DESCRIPTION :
Transform DICOM files into CFL.

Feel free to contact me, if you have problems with this script or have ideas for improvement.
"""

import sys, os
import numpy as np
import pydicom
from inspect import getsourcefile

sys.path.append(os.path.join(os.environ["TOOLBOX_PATH"], "python"))
sys.path.append(os.path.dirname(getsourcefile(lambda:0)))

import cfl

def calc_slice_locations(direc, protocolname=None, reverse_slice=False):
	"""
	Extract slice information from dicom files in specified directory.

	:param str direc: Directory containing dicom images
	:param str protocolname: Name of measurement protocol of dicom
	:param bool reverse_slice: Boolean if measurement was done from top to bottom or vice versa
	:returns: tuple (slice_locations, image_files)
		WHERE
		list slice_locations is list of slice heights
		list image_files is list of file paths to corresponding dicom images
	"""
	slice_locations = []
	image_files = []

	for name in [name for name in os.listdir(direc) if os.path.isfile(os.path.join(direc, name))]:

		dicom =  direc + "/" + name
		ds = pydicom.filereader.dcmread(dicom)

		# check if image belongs to correct measurement
		if None != protocolname and str(ds[0x181030].value) != protocolname:
			continue

		# collect all different slice locations
		if not (ds[0x201041].value in slice_locations):
			slice_locations.append(float(ds[0x201041].value))

		image_files.append(dicom)

	# order slice locations from highest to lowest (slice 1 to last)
	slice_locations.sort(reverse = reverse_slice)
	return slice_locations, image_files

def calc_image_list(slice_locations, image_files):
	"""
	Returns lists of file paths to dicom images corresponding
	to different slices and phases respectively. The dicom images
	are ordered according to their slice height.

	:param list slice_locations: List of slice heights of dicom images
	:param list image_files: List of file paths to dicom images
	:returns: tuple (img_list, img_shape)
		WHERE
		list img_list is list of lists, e.g. img_list[0] contains phases for images of slice 0
		list img_shape is shape of image [x,y]
	"""
	img_per_slice = int(len(image_files) / len(slice_locations))
	phase_list = [""] * img_per_slice
	image_list = [phase_list[:] for i in range(0,len(slice_locations))]
	img_shape = None
	for dicom in image_files:

		ds = pydicom.filereader.dcmread(dicom)

		if not float(ds[0x201041].value) in slice_locations:
			continue

		slice_index = slice_locations.index(float(ds[0x201041].value))
		phase_index = int(ds[0x200013].value % img_per_slice)
		image_list[slice_index][phase_index - 1] = dicom

		if None == img_shape:
			img_shape = ds.pixel_array.astype(np.int).shape

	return image_list, img_shape

def dicom_to_cfl_single(dicom_dir, save_path, reverse=True):
	"""
	Convert DICOM files into single CFL file.
	The input DICOm directory contains single image files in DICOM format for multiple slices.
	Images are assigned to different slices and saved separately for each slice in CFL format.

	:param str dicom_dir: Path to directory containing single images in DICOM format
	:param str output_dir: Path to output directory for CFL files
	:param bool reverse: Flag for reversing the order of slices. Default: True
	"""
	slice_locations, dicom_files = calc_slice_locations(dicom_dir)
	image_list, img_shape = calc_image_list(slice_locations, dicom_files)
	#fix for too many DICOM files in directory. Somehow twice the number of dicom files sometimes appear in a directory.
	img_pslice = []
	for images in image_list:
		img_pslice.append(len([i for i in images if i != ""]))
	images_per_slice = min(img_pslice)

	if reverse:
		image_list.reverse()
	new_array = np.zeros((img_shape[0],img_shape[1],1,1,1,1,1,1,1,1,images_per_slice,1,1,len(image_list)), dtype = complex)
	for sidx,images in enumerate(image_list):
		for idx,dicom in enumerate(images[:images_per_slice]):
			#print(dicom)
			ds = pydicom.filereader.dcmread(dicom)
			img = ds.pixel_array.astype(np.int)
			new_array[:,:,0,0,0,0,0,0,0,0,idx,0,0,sidx] = img
	cfl.writecfl(save_path, new_array)

def dicom_to_cfl_single_from_multi(dicom_dirs, save_path, reverse=True):
	"""
	Convert DICOM files from multiple directories into single CFL file.
	The input DICOm directory contains single image files in DICOM format for multiple slices.
	Images are assigned to different slices and saved separately for each slice in CFL format.

	:param str dicom_dir: Path to directory containing single images in DICOM format
	:param str output_dir: Path to output directory for CFL files
	:param bool reverse: Flag for reversing the order of slices. Default: True
	"""
	image_lists = []
	img_shape = None
	for dicom_dir in dicom_dirs:
		dicom_files = [os.path.join(dicom_dir, name) for name in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, name))]
		images_per_slice = len(dicom_files)
		image_list = ["" for i in range(images_per_slice)]
		for name in dicom_files:
			ds = pydicom.filereader.dcmread(name)

			phase_index = int(ds[0x200013].value % images_per_slice)
			image_list[phase_index - 1] = name

		if None == img_shape:
			img_shape = ds.pixel_array.astype(np.int).shape

		if reverse:
			image_list.reverse()
		image_lists.append(image_list)

	new_array = np.zeros((img_shape[0],img_shape[1],1,1,1,1,1,1,1,1,images_per_slice,1,1,len(image_lists)), dtype = complex)
	for sidx,images in enumerate(image_lists):
		for idx,dicom in enumerate(images[:images_per_slice]):
			ds = pydicom.filereader.dcmread(dicom)
			img = ds.pixel_array.astype(np.int)
			new_array[:,:,0,0,0,0,0,0,0,0,idx,0,0,sidx] = img
	cfl.writecfl(save_path, new_array)

def read_dicom_param_file(param_file:str, ddir:str, tdir:str, suffix:str, meas_id:int=2):
	"""
	Read a parameter file for the transformation of DICOM to CFL.
	DICOMs can either be stored in a single directory, singular measurement ID <measID>, or
	multiple directories, <measID_start measID_end>.

	:param str param_file: File path to parameter file. Format of lines: <identifier> <param2> <param3> <measID>
	:param str ddir: Data directory containing different directories and subdirectories with DICOM files.
	:param str tdir: Target directory for output directories.
	:param str suffix: Output filenames
	:param int meas_id: Position of measurement id when splitting the line content
	"""
	dirlist = [i for i in os.listdir(ddir) if os.path.isdir(os.path.join(ddir,i))]
	dirlist.sort()
	subdir_paths = []
	for num,subdir in enumerate(dirlist):
		subdirs = [os.path.join(ddir,subdir,f) for f in os.listdir(os.path.join(ddir,subdir)) if os.path.isdir(os.path.join(ddir,subdir, f))]
		subdir_paths += subdirs

	count = 0
	with open (param_file, 'rt', encoding="utf8", errors='ignore') as input:
		for line in input:
			count +=1
			acc_path = None
			#skip first line
			if count != 1:
				content = line.split("\t")
				if meas_id < len(content):
					subdir_out = content[0].strip()
					identifier = content[0].strip()
					meas = content[meas_id].strip()
					save_path = os.path.join(tdir, subdir_out, suffix)
					for a in subdir_paths:
						if identifier in a:
							acc_path = a
					if None == acc_path:
						continue

					if not os.path.exists(os.path.join(tdir, subdir_out)):
						os.makedirs(os.path.join(tdir, subdir_out))
					if not os.path.exists(save_path+".hdr"):
						print(os.path.join(acc_path, meas))

						if 1 < len(meas.split()):
							meas_content = meas.split()
							start = int(meas_content[0])
							end = int(meas_content[1])
							dicom_dirs = [os.path.join(acc_path, str(m)) for m in range(start, end+1)]
							dicom_to_cfl_single_from_multi(dicom_dirs, save_path)
						else:
							dicom_to_cfl_single(os.path.join(acc_path, meas), save_path)