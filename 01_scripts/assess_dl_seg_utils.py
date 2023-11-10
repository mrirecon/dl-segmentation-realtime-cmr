#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Copyright 2023. Uecker Lab, University Medical Center Goettingen.
author: Martin Schilling (martin.schilling@med.uni-goettingen.de), 2023

Script with utility functions for the creation of figures and the analysis of data
for the manuscript "Assessment of Deep Learning Segmentation
in Real-Time Free-Breathing Cardiac Magnetic Resonance Imaging".
"""

import numpy as np
import sys, os
import matplotlib #for plotting
import matplotlib.path as mpltPath
import multiprocessing
import matplotlib.pyplot as plt
from inspect import getsourcefile

sys.path.append(os.path.join(os.environ["TOOLBOX_PATH"], "python"))
sys.path.append(os.path.dirname(getsourcefile(lambda:0)))

import nnunet_utils as nnunet
from bart import bart
import cfl

if "DATA_DIR" in os.environ:
	scanner_reco_dir=os.path.join(os.environ["DATA_DIR"], "scanner_reco")
	contour_files_dir=os.path.join(os.environ["DATA_DIR"], "contour_files")
	nnunet_output_dir=os.path.join(os.environ["DATA_DIR"], "nnUNet/output")
	end_exp_dir=os.path.join(os.environ["DATA_DIR"], "end_expiration_indexes")
else:
	scanner_reco_dir=""
	contour_files_dir=""
	nnunet_output_dir=""
	end_exp_dir=""

contour_format = ".txt"
png_dpi=500

#---Analyze Contour file---

def mask_from_polygon_mplt(coordinates:list, img_shape:list):
	"""
	Transfer a polygon into a binary mask by using matplotlib.Path

	:param list coordinates: List of coordinates in format [[x1,y1], [x2,y2], ...]
	:param list img_shape: Shape of the output mask in format [xdim,ydim]
	:returns: Binary 2D mask
	:rtype: np.array
	"""
	path = mpltPath.Path(coordinates)
	points = [[x+0.5,y+0.5] for x in range(img_shape[1]) for y in range(img_shape[0])]
	inside2 = path.contains_points(points)
	new_mask = np.zeros(img_shape)
	count=0
	for y in range(img_shape[1]):
		for x in range(img_shape[0]):
			new_mask[x,y] = int(inside2[count])
			count += 1
	return new_mask

def transform_polygon_mp(args:tuple):
	"""
	Multiprocessing the transformation of a polygon to a binary mask.

	:param tuple args: tuple(coordinates, img_shape)
		WHERE
		list coordinates is list of coordinates in format [[x1,y1], [x2,y2], ...]
		list img_shape is list of dimensions of output mask [xdim, ydim]
	:returns: binary mask
	:rtype:	np.array
	"""
	(coordinates, img_shape) = args
	mask = mask_from_polygon_mplt(coordinates, img_shape)
	return mask

def parameters_from_file(txt_string):
	"""
	Extract parameters from Medis contour file.

	:param str txt_string: File paths to Medis contour file
	:returns: tuple(param_list, coord_list, ccsf)
		list param_list is list of slice and phase information of contours (<slice>, <phase>, <contour_index>)
		list coord_list is list of coordinates for contours
		list max_ccsf is list of contour coordinate scaling factors (0-LV, 1-Myo, 5-RV)
	"""
	line_bool = 0	# managing lines containing information
	line_count = 0	# countdown till end of array
	pattern = {"[XYCONTOUR]"}
	param_list = []
	coord_list = []
	ccsf = []
	add_contour = True

	with open (txt_string, 'rt', encoding="utf8", errors='ignore') as myfile:
		for line in myfile:
			line_bool -= 1

			if all(s in line for s in pattern):	# new array starts
				line_bool = 3
				continue

			if (2 == line_bool):	# line containing parameters of upcoming array

				parameters = [float(i) for i in line.split()] # slice no/  phase no/ contour coordinate scaling factor /  Always 1.0.
				if (int(parameters[0]), int(parameters[1]), int(parameters[2])) not in param_list:
					param_list.append([int(parameters[0]), int(parameters[1]), int(parameters[2])])
					# segm_classes: 0 - lv endocard, 1 - lv epicard, 5 - rv endocard
					segm_class = int(parameters[2])
					add_contour = True
				else:
					add_contour = False

				if int(parameters[2]) not in ccsf:
					ccsf.append(int(parameters[2]))

				continue

			if (1 == line_bool):	# line containing number of coordinate lines

				coordinates = []
				line_count = int(line.rstrip('\n'))  # initialise new line count
				continue

			if (0 != line_count):	# line contains coordinates

				coordinates.append([float(i) for i in line.split()])
				line_count -= 1

				if (0 == line_count) and add_contour:	# last line of coordinate list

					coord_list.append(coordinates)

	ccsf.sort()
	return param_list, coord_list, ccsf

def masks_and_parameters_from_file(txt_string:str, img_shape:list, mp_number=8):
	"""
	Extract images and contour masks from dicom images
	contained in an image list by analyzing contours specified
	in a txt or dicom file.

	:param str txt_string: File paths to txt or dicom file with contours
	:param list img_shape: Shape of image in format [xdim, ydim]
	:param int mp_number: Number of processes for multi-processing.
	:returns: tuple (mask_list, param_list, max_ccsf)
		WHERE
		list mask_list is list of filled contours
		list param_list is list of slice and phase information for filled contours
		list max_ccsf is list of contour coordinate scaling factors (0-LV, 1-Myo, 5-RV)
	"""
	param_list, coord_list, ccsf = parameters_from_file(txt_string)

	#Multiprocessing the transformation of polygons to binary masks
	pool = multiprocessing.Pool(processes=mp_number)
	inputs = [(coord, img_shape) for coord in coord_list]
	mask_list = pool.map(transform_polygon_mp, inputs)
	return mask_list, param_list, ccsf

#---Bland-Altman plot---
def plot_bland_altman_multi(setA:list, setB:list, header:str="", save_paths:list=[], lower_bound:float=None, upper_bound:float=None, point_size:float=15,
	check:bool=False, scale:float=1, precision:int = 2, plot:bool=True, ylim = [],
	labels:list = ["label1", "label2", "label3", "label4", "label5", "label6"],
	colors:list = ["royalblue", "dodgerblue", "palegreen", "yellowgreen", "darkorange", "indianred"],
	xlabel:str = "average of set A and set B", ylabel:str = "set A - set B [-]"):
	"""
	Plot a Bland-Altman plot for two given sets 'A' and 'B'.

	:param list setA: List of first set of values
	:param list setB: List of second set of values
	:param str header: Title addition for the plot
	:param list save_paths: List of file paths (for different file extensions) or single string to save plot
	:param float lower_bound: Lower bound for value of set A
	:param float upper_bound: Upper bound for value of set A
	:param float point_size: Size of points in scatter plot
	:param bool check: Check groups and indexes of sets, if they deviate more than 1.96 standard deviation
	:param float scale: Values are scaled by this value
	:param int precision: Precision for rounding values in plot
	:param bool plot: Flag for showing the plot. This way the plot can be saved without showing the plot.
	:param list ylim: Limits for y-axis
	:param list labels: Labels for groups in sets
	:param list colors: Colors for groups in sets
	:param str xlabel: Label for x-axis
	:param str ylabel: Label for y-axis
	:returns: List index and entry index in format [[list1, entry1], [list2, entry2], ...] for deviating entries
	:rtype: list
	"""
	setA_flat, setB_flat = [], []
	for set in setA:
		for e in set:
			setA_flat.append(e*scale)
	for set in setB:
		for e in set:
			setB_flat.append(e*scale)

	avg, diff = [], []
	for (m,s) in zip(setA_flat, setB_flat):
		if None != lower_bound:
			if m < lower_bound:
				continue
		if None != upper_bound:
			if m > upper_bound:
				continue
		avg.append((m+s) / 2)
		diff.append(m-s)
	stdv = np.std(diff)
	avg_diff=sum(diff) / len(avg)
	upper_limit = sum(diff) / len(diff) + 1.96 * stdv
	lower_limit = sum(diff) / len(diff) - 1.96 * stdv

	avg_list = [[] for i in range(len(setA))]
	diff_list = [[] for i in range(len(setA))]
	dev_list = []
	for num,(al,bl) in enumerate(zip(setA.copy(),setB.copy())):
		for enum,(a,b) in enumerate(zip(al,bl)):
			a = a*scale
			b = b*scale
			avg_list[num].append((a+b) / 2)
			diff_list[num].append(a-b)
			if check:
				if (a - b < lower_limit) or (a - b > upper_limit):
					dev_list.append([num,enum])
	for num,(a,d) in enumerate(zip(avg_list, diff_list)):
		plt.scatter(a,d, s=point_size, label=labels[num], c=colors[num])

	tick_size="large"
	fontsize="medium"
	label_size="x-large"
	legendsize="large"
	framealpha=0.8

	plt.xticks(size=tick_size)
	plt.yticks(size=tick_size)
	plt.xlabel(xlabel, size=label_size)
	plt.ylabel(ylabel, size=label_size)
	plt.title(header, size=label_size)

	vert_offset = 0.2 * (max(avg) - min(avg))
	hori_offset = 1/30 * np.abs(max(diff) - min(diff))
	plt.text(max(avg)-vert_offset, lower_limit+hori_offset, '-1.96 SD = {0:.{1}f}'.format(lower_limit, precision), ha='left', va='center', fontsize=fontsize)
	plt.text(max(avg)-vert_offset, upper_limit+hori_offset, '1.96 SD = {0:.{1}f}'.format(upper_limit, precision), ha='left', va='center', fontsize=fontsize)
	plt.text(max(avg)-vert_offset, avg_diff+hori_offset, 'MEAN = {0:.{1}f}'.format(avg_diff, precision), ha='left', va='center', fontsize=fontsize)
	plt.hlines(y=np.asarray([lower_limit, upper_limit, avg_diff]), xmin=min(avg)-vert_offset, xmax= max(avg)+vert_offset,
		colors=["gray" for i in range(3)], linestyles=["dashed" for i in range(3)])
	plt.xlim((min(avg)-vert_offset/2, max(avg)+vert_offset/2))
	if 0 == len(ylim):
		plt.ylim((min(diff)-hori_offset*3, max(diff)+hori_offset*6))
	else:
		plt.ylim(ylim)
	plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05),
		ncol=3, fancybox=True, shadow=False, framealpha=framealpha, fontsize=legendsize)
	if 0 != len(save_paths):
		if list == type(save_paths):
			for s in save_paths:
				if ".png" in s:
					plt.savefig(s, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
				else:
					plt.savefig(s, bbox_inches='tight', pad_inches=0)
		else:
			if ".png" in save_paths:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
			else:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0)
	if plot:
		plt.show()
	else:
		plt.close()
	return dev_list

def plot_bland_altman_axes(setA:list, setB:list, ax, header:str="", lower_bound:float=None, upper_bound:float=None, point_size:float=15,
	check:bool=False, scale:float=1, precision:int = 2, ylim = [],
	labels:list = ["label1", "label2", "label3", "label4", "label5", "label6"],
	colors:list = ["royalblue", "dodgerblue", "palegreen", "yellowgreen", "darkorange", "indianred"],
	xlabel:str = "average of set A and set B", ylabel:str = "set A - set B [-]"):
	"""
	Plot a Bland-Altman plot for two given sets 'A' and 'B' for an axis of a subplot.

	:param list setA: List of first set of values
	:param list setB: List of second set of values
	:param plt.axes ax: Ax to plot subplot on
	:param str header: Title addition for the plot
	:param float lower_bound: Lower bound for value of set A
	:param float upper_bound: Upper bound for value of set A
	:param float point_size: Size of points in scatter plot
	:param bool check: Check groups and indexes of sets, if they deviate more than 1.96 standard deviation
	:param float scale: Values are scaled by this value
	:param int precision: Precision for rounding values in plot
	:param list ylim: Limits for y-axis
	:param list labels: Labels for groups in sets
	:param list colors: Colors for groups in sets
	:param str xlabel: Label for x-axis
	:param str ylabel: Label for y-axis
	"""
	setA_flat, setB_flat = [], []
	for set in setA:
		for e in set:
			setA_flat.append(e*scale)
	for set in setB:
		for e in set:
			setB_flat.append(e*scale)

	avg, diff = [], []
	for (m,s) in zip(setA_flat, setB_flat):
		if None != lower_bound:
			if m < lower_bound:
				continue
		if None != upper_bound:
			if m > upper_bound:
				continue
		avg.append((m+s) / 2)
		diff.append(m-s)
	stdv = np.std(diff)
	avg_diff=sum(diff) / len(avg)
	upper_limit = sum(diff) / len(diff) + 1.96 * stdv
	lower_limit = sum(diff) / len(diff) - 1.96 * stdv

	avg_list = [[] for i in range(len(setA))]
	diff_list = [[] for i in range(len(setA))]
	dev_list = []
	for num,(al,bl) in enumerate(zip(setA.copy(),setB.copy())):
		for enum,(a,b) in enumerate(zip(al,bl)):
			a = a*scale
			b = b*scale
			avg_list[num].append((a+b) / 2)
			diff_list[num].append(a-b)
			if check:
				if (a - b < lower_limit) or (a - b > upper_limit):
					dev_list.append([num,enum])
	for num,(a,d) in enumerate(zip(avg_list, diff_list)):
		ax.scatter(a,d, s=point_size, label=labels[num], c=colors[num])

	tick_size="x-large"
	fontsize="large"
	label_size="xx-large"
	legendsize="large"
	framealpha=0.8

	ax.tick_params(axis='both', labelsize=tick_size)
	ax.set_xlabel(xlabel, size=label_size)
	ax.set_ylabel(ylabel, size=label_size, labelpad=0)
	ax.set_title(header, size=label_size)

	hori_offset = 0.25 * (max(avg) - min(avg))
	vert_offset = 1/25 * np.abs(max(diff) - min(diff))
	if 0 != len(ylim):
		vert_offset = 1/60 * np.abs(ylim[1] - ylim[0])

	ax.text(max(avg)-hori_offset, lower_limit+vert_offset, '-1.96 SD = {0:.{1}f}'.format(lower_limit, precision), ha='left', va='center', fontsize=fontsize)
	ax.text(max(avg)-hori_offset, upper_limit+vert_offset, '1.96 SD = {0:.{1}f}'.format(upper_limit, precision), ha='left', va='center', fontsize=fontsize)
	ax.text(max(avg)-hori_offset, avg_diff+vert_offset, 'MEAN = {0:.{1}f}'.format(avg_diff, precision), ha='left', va='center', fontsize=fontsize)
	ax.hlines(y=np.asarray([lower_limit, upper_limit, avg_diff]), xmin=min(avg)-hori_offset, xmax= max(avg)+hori_offset,
		colors=["gray" for i in range(3)], linestyles=["dashed" for i in range(3)])
	ax.set_xlim((min(avg)-hori_offset/2, max(avg)+hori_offset/2))
	if 0 == len(ylim):
		ax.set_ylim((min(diff)-hori_offset*3, max(diff)+hori_offset*6))
	else:
		ax.set_ylim(ylim)
	ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05),
		ncol=3, fancybox=True, shadow=False, framealpha=framealpha, fontsize=legendsize)

#---Creating nnU-Net input data---
def extract_img_params(contour_file:os.path):
	"""
	Extract image dimension, field of view, end-systolic and end-diastolic phases
	and the number of slices from a contour file of cardiac cine MRI.

	:param os.path contour_file: File path to file containing contour information
	:returns: tuple (img_shape, fov, ed_phase, es_phase, slices)
		WHERE
		list img_shape is [xdim, ydim]
		list fov is [fov_x, fov_y]
		int slices in number of slices
	"""
	img_res_marker = "Image_resolution"
	fov_marker = "Field_of_view"
	slice_marker = "selection.slice_"

	xdim, ydim = 0, 0
	fov_x, fov_y = 0,0
	ed_phase, es_phase = 0, 0
	number_slices = 0
	with open (contour_file, 'rt', encoding="utf8", errors='ignore') as input:
		for line in input:
			if line.strip():
				content = line.split("\n")
				if img_res_marker in content[0]:
					ydim = int(content[0].split("=")[1].split("x")[0])
					xdim = int(content[0].split("=")[1].split("x")[1])

				if fov_marker in content[0]:
					fov_y = float(content[0].split("=")[1].split()[0].split("x")[0])
					fov_x = float(content[0].split("=")[1].split()[0].split("x")[1])

				if slice_marker in content[0]:
					slice_index = int(content[0].split("_")[1].split("=")[0])
					number_slices = max([number_slices, slice_index])

	input.close()
	img_shape = [xdim, ydim]
	fov = [fov_x, fov_y]
	if 0 == number_slices:
		print("extraction did not work", contour_file)
	return img_shape, fov, number_slices

def single_slice_combine_masks_to_list(mask_list, param_list, medis_slice):
	"""
	Create a single array for a given slice from a list of 2d-arrays and a corresponding parameter list.

	:param list mask_list: List of 2d-array segmentation masks
	:param list param_list: List of parameters of the segmentation masks. Format (<slice>, <phase>, <segmentation class>)
	:param int medis_slice: Slice index for creation of output array
	:returns: List of segmentation masks in format [xdim, ydim]
	:rtype: list
	"""
	max_phase = 0
	slice_phases = []
	for p in param_list:
		if medis_slice == p[0]:
			slice_phases.append(p[1])
	slice_phases = list(set(slice_phases))
	slice_phases.sort()

	img_shape = [mask_list[0].shape[0], mask_list[0].shape[1]]
	slice_masks = [[np.zeros((img_shape[0], img_shape[1]), dtype=float) for i in range(3)] for j in range(len(slice_phases))]
	param_slct = []
	for num, p in enumerate(param_list):
		if medis_slice == p[0]:
			# ignore other classes than LV endo, LV epi, RV endo
			if p[2] not in [0,1,5]:
				continue
			segm_class = 2 if 5 == p[2] else p[2]
			slice_masks[slice_phases.index(p[1])][segm_class] = mask_list[num]
			if [p[0],p[1]] not in param_slct:
				param_slct.append([p[0],p[1]])
	mask_slct = []
	for num,m in enumerate(slice_masks):
		mask = np.zeros((img_shape[0], img_shape[1]), dtype=float)
		#RV endocard
		rv = m[2] - m[1]
		rv[rv < 1] = 0
		mask += rv
		#Myocard
		mask += 2 * (m[1] - m[0])
		#LV endocard
		mask += 3 * m[0]
		mask_slct.append(mask)

	return mask_slct, param_slct

def combine_masks_multi(mask_list, param_list, slices=0, reverse=False):
	"""
	Create a list of arrays for each slice from a list of 2d-arrays and a corresponding parameter list.

	:param list mask_list: List of 2d-array segmentation masks for multiple slices
	:param list param_list: List of parameters of the segmentation masks. Format (<slice>, <phase>, <segmentation class>)
	:param int slices: Number of slices of dataset. Used if the parameter and slice information has to be reversed
	:param bool reverse: Flag to reverse the order of slices. In some cases, slice parameters of the contour files are in reversed order compared to images.
	:param bool transform: Flag for change of orientation of array (rotation and flip)
	:returns: Combined array of segmentation masks in format [1, xdim, ydim, 1, batch_dim]
	:rtype: ndarray
	"""
	mlist = []
	plist = []
	si_list = [] #slice indexes of slices containing masks

	for p in param_list:
		if p[0] not in si_list:
			si_list.append(p[0])

	slices = max(si_list) if 0 == slices else slices

	for s in si_list:
		mask_slct, param_slct = single_slice_combine_masks_to_list(mask_list, param_list, s)
		mlist.append(mask_slct)
		plist.append(param_slct)

	# fix order of parameters and slice information if segmentation of contour file is in reverse order
	if reverse:
		for ps in plist:
			for p in ps:
				p[0] = slices - (p[0] + 1)
		for num in range(len(si_list)):
			si_list[num]  = slices - (si_list[num] + 1)

	return si_list, mlist, plist

def extract_ED_ES_phase_cine(plist:list, mlist:list, segm_class:int=3):
	"""
	Determine the end-diastolic (ED) and end-systolic (ES) phase for cine contour files.
	A parameter list for multiple slices in combination with a list of lists of segmentation masks
	is given as input. The amount of pixels corresponding to a given segmentation class is averaged over all slices
	for each phase. The ED phase is the one with the larger average area, the ES phase the one with the lower average area.

	:param list plist: List of parameters of multiple slices in format (slice, phase)
	:param list mlist: List of segmentation masks of multiple slices in format [xdim,ydim]
	:param int segm_class: Segmentation class to evaluate.
	:returns: tuple(ED_phase, ES_phase)
		WHERE
		int ED_phase is end-diastolic phase
		int ES_Phase is end-systolic phase
	"""
	phases = []
	for pslice in plist:
		for p in pslice:
			phases.append(p[1])
	phases = list(set(phases))
	phase1 = phases[0]
	phase2 = phases[1]
	vol1 = []
	vol2 = []
	for num,ms in enumerate(mlist):
		for enum,m in enumerate(ms):
			lv_count = np.count_nonzero(m[:,:] == segm_class)
			if phase1 == plist[num][enum][1]:
				vol1.append(lv_count)
			else:
				vol2.append(lv_count)
	if sum(vol1) / len(vol1) > sum(vol2) / len(vol2):
		return phase1, phase2
	else:
		return phase2, phase1

def create_nnunet_input_cine(img_file:os.path, output_prefix:os.path, contour_file:os.path="", reverse:bool=False, output_mode:str="2d", pixel_spacing:float=1.328125,
				phase_selection="edes", slice_selection=False):
	"""
	Create input for nnUNet for cine MRI based on contour files.

	:param os.path img_file: Reconstructed MRI images in format [xdim, ydim, 1,1,1,1,1,1,1,1,phase, slice]
	:param os.path output_prefix: Prefix for nnUNet files. Files will have the format <prefi>xxyyy where xx
		represents the slice index, yyy the phase index
	:param os.path contour_file: Text file containing Medis session information and manual contours
	:param bool reverse: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order
	:param str output_mode: Output format of Nifti files for nnU-Net. "2d" for (1,xdim,ydim), "3d" for (batch_dim, xdim, ydim)
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param str phase_selection: Selection of ED and ES phases with 'edes' or all phases
	:param bool slice_selection: Flag for 3D output to only include slices containing the heart (slice_selection=True) or all slices (slice_selection=False)
	"""
	img_data = cfl.readcfl(img_file)
	img_data = np.reshape(img_data, (img_data.shape[0], img_data.shape[1],img_data.shape[10],img_data.shape[13]))
	slice_list = []

	if "" != contour_file:
		img_dims, fov, slices_new = extract_img_params(contour_file)
		mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
		slice_indexes, mlist, plist = combine_masks_multi(mask_list, param_list, slices_new, reverse)

		if img_dims != [img_data.shape[0], img_data.shape[1]]:
				print(contour_file, [img_data.shape[0], img_data.shape[1]], img_dims)

		EDphase, ESphase = extract_ED_ES_phase_cine(plist, mlist, segm_class=3)
		pixel_spacing = round((fov[0] / img_data.shape[0] + fov[1] / img_data.shape[1]) / 2, 7)

		plist = [item for sublist in plist for item in sublist]
		if slice_selection:
			for p in plist:
				if EDphase == p[1] and p[0] not in slice_list:
					slice_list.append(p[0])
			slice_list.sort()

		if "edes" == phase_selection:
			phase_list = [EDphase, ESphase]
		else:
			plist = [[x,y] for x in range(img_data.shape[3]) for y in range(img_data.shape[2])]
			phase_list = [x for x in range(img_data.shape[2])]

	else:
		plist = [[x,y] for x in range(img_data.shape[3]) for y in range(img_data.shape[2])]
		phase_list = [x for x in range(img_data.shape[2])]

	if "2d" == output_mode:
		nnunet.cfl2nifti(img_file, output_prefix, plist, pixel_spacing=(pixel_spacing, pixel_spacing))
	elif "3d" == output_mode:
		nnunet.cfl2nifti_3d(img_file, output_prefix, phase_selection=phase_list, slice_selection=slice_list, pixel_spacing=(pixel_spacing, pixel_spacing))
	else:
		print("Select '2d' or '3d' output mode")

def create_nnunet_input_rt(img_file:os.path, output_prefix:os.path, mode="single", pixel_spacing:float = 1.6,
			   crop_dim=None, slice_id="", contour_file="", reverse_flag=""):
	"""
	Create input for nnUNet for cine MRI based on contour files.

	:param os.path img_file: Reconstructed MRI images in format [xdim, ydim, 1,1,1,1,1,1,1,1,phase, slice]
	:param os.path output_prefix: Prefix for nnUNet files. Files will have the format <prefi>xxyyy where xx
		represents the slice index, yyy the phase index
	:param str mode: Mode for nnU-Net input data: 'single' for single images or 'slice' for all images in a slice
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param int crop_dim: Size of square dimensions to crop image to
	:param str slice_id: String of slice index (input from bash script)
	:param os.path contour_file: Text file containing Medis contours to only create nnU-Net input for indexes with contours
	:param bool reverse_flag: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order.
	"""
	img_data = cfl.readcfl(img_file)
	slices = [0]
	if len(img_data.shape) > 11:
		slices = [i for i in range(img_data.shape[-1])]

	phases = img_data.shape[10]
	offset = 0 if 0 == len(slice_id) else int(slice_id)
	if "single" == mode:
		plist = []
		output_params = []
		if 0 == len(contour_file):
			for s in slices:
				for p in range(phases):
					plist.append([s,p])
					output_params.append([s+offset,p])
		else:
			reverse = int(reverse_flag)
			img_dims, fov, slices = extract_img_params(contour_file)
			param_list, coord_list, ccsf = parameters_from_file(contour_file)
			plist = [[p[0], p[1]] for p in [*set([(p[0],p[1]) for p in param_list])]]
			if reverse:
				for p in plist:
					p[0] = slices - (p[0] + 1)
			if 0 != len(slice_id):
				output_params = [item for item in plist if int(slice_id) == item[0]]
				plist = [[0,item[1]] for item in plist if int(slice_id) == item[0]]

		nnunet.cfl2nifti(img_file, output_prefix, selection_array=plist, selection_output=output_params, pixel_spacing=(pixel_spacing, pixel_spacing), crop_dim=crop_dim)
	elif "slice" == mode:
		nnunet.cfl2nifti_slice(img_file, output_prefix, slices, (pixel_spacing, pixel_spacing))
	else:
		sys.exit("Choose either 'single' or 'slice' as mode.")

#---Calculate DC---

def crop_2darray(arr:np.array, cdim:int):
	"""
	Center crop input array of shape [xdim, ydim] to [cdim, cdim]

	:param np.array arr: 2D input array
	:param int cdim: Dimension of cropped, spherical output. Optional input as list in format [x1,x2,y1,y2] for cropping [x1:x2,y1:y2]
	:returns: Cropped array
	:rtype: np.array
	"""
	if not list == type(cdim):
		x_center = arr.shape[0] // 2
		y_center = arr.shape[1] // 2
		arr = arr[int(x_center-cdim/2):int(x_center+cdim/2), int(y_center-cdim/2):int(y_center+cdim/2)]
	else:
		(x1,x2,y1,y2) = cdim
		arr = arr[x1:x2,y1:y2]
	return arr

def calc_dice_coeff(ref:list, pred:list, segm_classes:int, plist = [], reverse=False)->list:
	"""
	Calculate Dice's coefficient for a list of references and predictions.

	:param list ref: List of references in 2d np.array format
	:param list pred: List of predictions in 2d np.array format
	:param int segm_classes: Number of segmentation classes
	:param list plist: Parameter list in format [[slice1,phase1], [slice2,phase2], ...]
	:param bool reverse: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order
	:returns: tuple (result, dict_list)
		WHERE
		list result is Dice's coefficients for every segmentation classes
		list dict_list is list of dictionaries containing information about Dice coefficients
	"""
	AoO = [0 for i in range(segm_classes)]
	totalNumber = [0 for i in range(segm_classes)]
	single_results = [[] for i in range(segm_classes)]
	slices = []
	if 0 != len(plist):
		for p in plist:
			if p[0] not in slices:
				slices.append(p[0])
		slices.sort()

	dict_list = []
	for num, (r,p) in enumerate(zip(ref, pred)):
		AoO = [0 for i in range(segm_classes)]
		totalNumber = [0 for i in range(segm_classes)]
		ref_flattened = r.flatten()
		pred_flattened = p.flatten()
		for j in range(len(ref_flattened)):
			if (ref_flattened[j] == pred_flattened[j]):
				AoO[int(ref_flattened[j].real)] += 1
			totalNumber[int(ref_flattened[j].real)] += 1
			totalNumber[int(pred_flattened[j].real)] += 1
		if 0 != len(plist):
			d = {'slice' : plist[num][0], 'phase' : plist[num][1] }
		else:
			d = {'slice' : "", 'phase' : ""}
		d['reverse'] = reverse
		d['slices_total'] = slices
		for i in range(segm_classes):
			if 0 == totalNumber[i]:
				d['class'+str(i)] = 1
				single_results[i].append(1)
			else:
				d['class'+str(i)] = 2 * AoO[i] / totalNumber[i]
				single_results[i].append(2 * AoO[i] / totalNumber[i])
		dict_list.append(d)
	result = []
	for i in range(segm_classes):
		result.append(sum(single_results[i]) / len(single_results[i]))
	return result, dict_list

def mask_for_phase_contour_file(mlist:list, plist:list, phase:int):
	"""
	Calculate a list of segmentation masks for a given phase from a list of masks for a given segmentation class.

	:param list mlist: Lists of segmentation masks for multiple slices
	:param list plist: Lists of parameters for multiple slices. Format [slice, phase]
	:param int phase: Phase index among slices
	:returns: Segmentation mask
	:rtype: list
	"""
	mask_list = []
	for num,ms in enumerate(mlist):
		for enum,m in enumerate(ms):
			if phase == plist[num][enum][1]:
				mask_list.append(m)
	return mask_list

def mask_for_param_selection(mlist:list, plist:list, param:list):
	"""
	Extract a list of segmentation masks for a given subset of parameters.

	:param list mlist: Lists of segmentation masks for multiple slices
	:param list plist: Lists of parameters for multiple slices. Format [slice, phase]
	:param list phase: List of parameters in format [[slice1, phase1], [slice2, phase2], ...] which will be extracted
	:returns: Segmentation mask
	:rtype: list
	"""
	mask_list = []
	for ps in param:
		found = False
		for num,pis in enumerate(plist):
			for enum, pi in enumerate(pis):
				if ps == pi:
					mask_list.append(mlist[num][enum])
					found = True
		if not found:
			mask_list.append(np.zeros(mlist[0][0].shape, dtype=float))
	if len(mask_list) != len(param):
		print("Something might have gone wrong")
	return mask_list

def get_phase_mask_from_nnunet(segm_prefix:os.path, plist:list, flag3d=False):
	"""
	Calculate a list of segmentation masks for a segmentation class from a directory containing Nifti files featuring single images.
	Nifti file names have the format <segm_prefix>xxyyy.nii.gz where xx presents the zero-filled slice number and yyy the zero-filled phase.

	:param os.path segm_prefix: Prefix for Nifti file names
	:param list plist: Lists of parameters for multiple slices. Format [slice, phase]
	:param bool flag3d: Flag for 3D segmentation of nnU-Net
	:returns: Segmentation mask
	:rtype: list
	"""
	masks = []

	if flag3d:
		segm_file = segm_prefix + str(plist[0][1]).zfill(3)+".nii.gz"
		img = nnunet.read_nnunet_segm(segm_file)
		for p in plist:
			masks.append(img[:,:,p[0]])
	else:
		for p in plist:
			segm_file = segm_prefix + str(p[0]).zfill(2)+str(p[1]).zfill(3)+".nii.gz"

			if os.path.isfile(segm_file):
				img = nnunet.read_nnunet_segm(segm_file)
			else:
				img = nnunet.read_nnunet_segm(segm_prefix+str(p[0]).zfill(2)+".nii.gz")[:,:,p[1]]

			masks.append(img)
	return masks

def get_ed_es_dice_from_contour_file(contour_file:os.path, reverse:bool, segm_input:str, flag3d=False, phase_select="both", slice_selection=False)->tuple:
	"""
	Determine the end-diastolic (ED) and end-systolic (ES) Dice's coefficient between comDL or nnU-Net segmentation
	and manually corrected contours.

	:param os.path contour_file: Text file containing Medis session information and manual contours.
	:param bool reverse: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order
	:param str segm_input: Single segmentation file (nnet segmentation) or directory/prefix (nnU-Net) containing segmentations
	:param bool flag3d: Flag for 3D segmentation of nnU-Net
	:param str phase_select: Individual output of ED and ES phase with "both" or combined output with "combined"
	:param bool slice_selection:
	:returns: tuple (ed_dc, ed_dict, es_dc, es_dict)
		WHERE
		list ed_dc is Dice's coefficient for end-diastolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ED phases
		list ed_dc is Dice's coefficient for end-systolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ES phases
	"""
	img_dims, fov, slices = extract_img_params(contour_file)
	mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
	slice_indexes, mlist, plist = combine_masks_multi(mask_list, param_list, slices=slices, reverse=reverse)

	slice_indexes.sort()
	EDphase, ESphase = extract_ED_ES_phase_cine(plist, mlist, segm_class=3)

	ed_masks_mc = mask_for_phase_contour_file(mlist, plist, EDphase)
	es_masks_mc = mask_for_phase_contour_file(mlist, plist, ESphase)

	#create one flat parameter list
	ed_plist, es_plist = [], []
	for ps in plist:
		for p in ps:
			if slice_selection:
				p[0] = p[0] - slice_indexes[0]

			if EDphase == p[1]:
				ed_plist.append(p)
			elif ESphase == p[1]:
				es_plist.append(p)

	if os.path.isfile(segm_input):
		img_dims, fov, slices = extract_img_params(segm_input)
		mask_list, param_list, ccsf = masks_and_parameters_from_file(segm_input, img_dims)
		slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list)

		for ps in plist:
			for p in ps:
				p[0] = slices - (p[0] + 1) if reverse else p[0]
		ed_masks_seg = mask_for_param_selection(mlist, plist, ed_plist)
		es_masks_seg = mask_for_param_selection(mlist, plist, es_plist)
	else:
		ed_masks_seg = get_phase_mask_from_nnunet(segm_input, plist=ed_plist, flag3d=flag3d)
		es_masks_seg = get_phase_mask_from_nnunet(segm_input, plist=es_plist, flag3d=flag3d)

	if ed_masks_mc[0].shape[0] != ed_masks_seg[0].shape[0]:
		for num,m in enumerate(ed_masks_mc):
			ed_masks_mc[num] = crop_2darray(m, ed_masks_seg[0].shape[0])

	if es_masks_mc[0].shape[0] != es_masks_seg[0].shape[0]:
		for num,m in enumerate(es_masks_mc):
			es_masks_mc[num] = crop_2darray(es_masks_mc[num], es_masks_seg[0].shape[0])

	segm_classes = 4

	if "both" == phase_select:
		ed_dc, ed_dict = calc_dice_coeff(ed_masks_mc, ed_masks_seg, segm_classes, plist = ed_plist, reverse=reverse)
		es_dc, es_dict = calc_dice_coeff(es_masks_mc, es_masks_seg, segm_classes, plist = es_plist, reverse=reverse)

	elif "combined" == phase_select:
		mask_mc = ed_masks_mc + es_masks_mc
		masks_seg = ed_masks_seg + es_masks_seg
		param_seg = ed_plist + es_plist
		ed_dc, ed_dict = calc_dice_coeff(mask_mc, masks_seg, segm_classes, plist = param_seg, reverse=reverse)
		es_dc = [[] for i in range(segm_classes)]
		es_dict = []
	else:
		sys.exit("Choose either 'both' or 'combined' mode for phase selection.")

	return ed_dc, ed_dict, es_dc, es_dict

def get_ed_es_from_manual_rt(mlist:list, plist:list, segm_class:int=3, pixel_spacing:float=1):
	"""
	Identify the end-systolic and end-diastolic phases from manual real-time contours.
	This is done under the assumption, that only ED and ES phases are segmented in a given slice
	and both phases keep altering between each other so that either all entries with odd and even indexes
	belong to one of the phases.

	:param list mlist: List of lists containing 2D segmentation masks in format [xdim, ydim] for multiple slices
	:param list plist: List of lists containing 2D parameters in format [slice, phase] for multiple slices
	:param int segm_class: Index of segmentation class, which will be analysed (LV endocard)
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:returns: tuple(ed_idx, es_idx, ed_list, es_list)
		WHERE
		list ed_idx is list of indexes belonging to the ED phase for multiple slices
		list es_idx is list of indexes belonging to the ES phase for multiple slices
		list ed_list is list of LV pixels of each ED phase for multiple slices
		list es_list is list of LV pixels of each ES phase for multiple slices
	"""
	ed_list = []
	es_list = []
	ed_idx = []
	es_idx = []
	for enum,ml in enumerate(mlist):
		odd = []
		even = []
		odd_idx = []
		even_idx = []
		for num, m in enumerate(ml):
			lv_count = np.count_nonzero(np.abs(m) == segm_class)
			if num % 2 == 0:
				even.append(lv_count*pixel_spacing*pixel_spacing)
				even_idx.append(plist[enum][num])
			else:
				odd.append(lv_count*pixel_spacing*pixel_spacing)
				odd_idx.append(plist[enum][num])
		if sum(odd) < sum(even):
			ed_list.append(even)
			es_list.append(odd)
			ed_idx.append(even_idx)
			es_idx.append(odd_idx)
		else:
			ed_list.append(odd)
			es_list.append(even)
			ed_idx.append(odd_idx)
			es_idx.append(even_idx)
	return ed_idx, es_idx, ed_list, es_list

def get_ed_es_dice_from_contour_file_rt(contour_file:os.path, reverse:bool, segm_input:str,
			flag3d:bool=False, phase_select="both", slices_string="", mode="output")->tuple:
	"""
	Determine the end-diastolic (ED) and end-systolic (ES) Dice's coefficient (DC) for a given contour file featuring contours in Medis format.
	A dictionary for each image with its DC is given, class labels are in order: "BG", "RV", "Myo", "LV"

	:param os.path contour_file: Text file containing Medis session information and manual contours.
	:param bool reverse: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order
	:param str segm_input: Single segmentation file (nnet segmentation) or directory/prefix (nnU-Net) containing segmentations
	:param bool flag3d: Flag for 3D segmentation of nnU-Net
	:param str phase_select: Selection of phases. 'both' for individual calculation for ED and ES phase. 'combined' for combined calculation.
	:param str slices_string: String of slice indexes for slice selection. Separate slices indexes with spaces "<slice1> <slice2> ..."
	:param str mode: Set mode to "no output" for directly printing Dice's coefficients
	:returns: tuple (ed_dc, ed_dict, es_dc, es_dict)
		WHERE
		list ed_dc is Dice's coefficient for end-diastolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ED phases
		list ed_dc is Dice's coefficient for end-systolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ES phases
	"""
	if not isinstance(reverse, bool):
		reverse = int(reverse)

	img_dims, fov, slices = extract_img_params(contour_file)
	mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
	slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list, slices=slices, reverse=reverse)

	if 0 != len(slices_string):
		slct_slices = slices_string.split()
		slct_slices = [int(x) for x in slct_slices]
		slct_indexes = []
		for num,ps in enumerate(plist):
			if ps[0][0] in slct_slices:
				slct_indexes.append(num)

		mlist = [mlist[i] for i in slct_indexes]
		plist = [plist[i] for i in slct_indexes]

	ed_idx, es_idx, ed_count_list, es_count_list = get_ed_es_from_manual_rt(mlist, plist, segm_class=3)
	ed_idx = [item for sublist in ed_idx for item in sublist]
	es_idx = [item for sublist in es_idx for item in sublist]

	ed_masks_mc = mask_for_param_selection(mlist, plist, ed_idx)
	es_masks_mc = mask_for_param_selection(mlist, plist, es_idx)

	if os.path.isfile(segm_input):
		img_dims, fov, slices = extract_img_params(segm_input)
		mask_list, param_list, ccsf = masks_and_parameters_from_file(segm_input, img_dims)
		slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list)

		for ps in plist:
			for p in ps:
				p[0] = slices - (p[0] + 1) if reverse else p[0]
		ed_masks_seg = mask_for_param_selection(mlist, plist, ed_idx)
		es_masks_seg = mask_for_param_selection(mlist, plist, es_idx)
	else:
		ed_masks_seg = get_phase_mask_from_nnunet(segm_input, plist=ed_idx, flag3d=flag3d)
		es_masks_seg = get_phase_mask_from_nnunet(segm_input, plist=es_idx, flag3d=flag3d)

	if ed_masks_mc[0].shape[0] != ed_masks_seg[0].shape[0]:
		for num,m in enumerate(ed_masks_mc):
			ed_masks_mc[num] = crop_2darray(m, ed_masks_seg[0].shape[0])

	if es_masks_mc[0].shape[0] != es_masks_seg[0].shape[0]:
		for num,m in enumerate(es_masks_mc):
			es_masks_mc[num] = crop_2darray(es_masks_mc[num], es_masks_seg[0].shape[0])

	segm_classes = 4

	if "both" == phase_select:
		ed_dc, ed_dict = calc_dice_coeff(ed_masks_mc, ed_masks_seg, segm_classes, plist = ed_idx, reverse=reverse)
		es_dc, es_dict = calc_dice_coeff(es_masks_mc, es_masks_seg, segm_classes, plist = es_idx, reverse=reverse)

	elif "combined" == phase_select:
		mask_mc = ed_masks_mc + es_masks_mc
		masks_seg = ed_masks_seg + es_masks_seg
		param_seg = ed_idx + es_idx
		ed_dc, ed_dict = calc_dice_coeff(mask_mc, masks_seg, segm_classes, plist = param_seg, reverse=reverse)
		es_dc = [[] for i in range(segm_classes)]
		es_dict = []
	else:
		sys.exit("Choose either 'both' or 'combined' mode for phase selection.")
	if "no output" == mode:
		class_labels=["BG", "RV", "Myo", "LV"]
		for i in range(len(ed_dc)):
			print(class_labels[i] + "\t" + str(round(ed_dc[i],3)))
	else:

		return ed_dc, ed_dict, es_dc, es_dict

def get_ed_es_dice_from_contour_file_multi(contour_dir, data_dict, manual_contour_suffix, comp_contour_suffix="", nnunet_prefix="", phase_select="both",
		title="nnUNet", class_labels=["BG", "RV", "Myo", "LV"], flag3d=True, labels=4, mode="rt", print_output=True, slice_selection=False):
	"""
	Return Dice coefficient for multiple datasets.

	:param str contour_dir: Directory containing contour files in format <contour_dir>/<id><contour_suffix>
	:param list data_dict: List of dictionaries in format [{"id":<id>, "reverse":bool}]
	:param str manual_contour_suffix: Suffix for manual contours, e.g. '_rt_manual.con'
	:param str comp_contour_suffix: Suffix for comparison contour
	:param str nnunet_prefix: Prefix for nnU-Net segmentations
	:param str phase_select: Selection for calculation of ED and ES phase. Either 'both' or 'combined'
	:param str title: Title for output
	:param list class_labels: List of string descriptions for class labels
	:param bool flag3d: Flag for 3D segmentation of nnU-Net
	:param int labels: Number of labels
	:param str mode: Mode of measurement. Either 'rt' or 'cine'
	:param bool print_output:
	:param bool slice_selection:
	:returns: tuple (ed_dc, ed_dict, es_dc, es_dict)
		WHERE
		list ed_dc is Dice's coefficient for end-diastolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ED phases
		list ed_dc is Dice's coefficient for end-systolic phases
		list ed_dict is list of dictionaries containing information about Dice coefficients for ES phases
	"""
	ed_dc_list = [[] for i in range(labels)]
	es_dc_list = [[] for i in range(labels)]
	ed_dict_list = []
	es_dict_list = []
	for data in data_dict:
		vol = data["id"]
		reverse = data["reverse"]
		manual_contour_file = vol+manual_contour_suffix
		contour_file = os.path.join(contour_dir, manual_contour_file)

		# comDL contour
		if "" != comp_contour_suffix:
			comDL_contour_file = vol+comp_contour_suffix
			segm_input = os.path.join(contour_dir, comDL_contour_file)
		# nnUNet segmentation
		elif "" != nnunet_prefix:
			segm_input = nnunet_prefix+vol[3:]

		if "cine" == mode:
			ed_dc, ed_dict, es_dc, es_dict = get_ed_es_dice_from_contour_file(contour_file, reverse, segm_input, flag3d=flag3d, phase_select=phase_select, slice_selection=slice_selection)
		elif "rt" == mode:
			ed_dc, ed_dict, es_dc, es_dict = get_ed_es_dice_from_contour_file_rt(contour_file, reverse, segm_input, flag3d=flag3d, phase_select=phase_select)
		else:
			sys.exit("Choose either mode 'rt' or 'cine'.")

		for e in ed_dict:
			e['id'] = vol
		for e in es_dict:
			e['id'] = vol

		for num,e in enumerate(ed_dc):
			ed_dc_list[num].append(e)
		for num,e in enumerate(es_dc):
			es_dc_list[num].append(e)

		ed_dict_list.extend(ed_dict)
		es_dict_list.extend(es_dict)

	if print_output:
		print(title)
		if "both" == phase_select:
			for i in range(len(ed_dc_list)):
				print("ED: " + class_labels[i], round(sum(ed_dc_list[i]) / len(ed_dc_list[i]),3), round(np.std(ed_dc_list[i]),3))
			for i in range(len(es_dc_list)):
				print("ES: " + class_labels[i], round(sum(es_dc_list[i]) / len(es_dc_list[i]),3), round(np.std(es_dc_list[i]),3))
		else:
			for i in range(len(ed_dc_list)):
				print(class_labels[i], round(sum(ed_dc_list[i]) / len(ed_dc_list[i]),3), round(np.std(ed_dc_list[i]),3))
	return ed_dc_list, es_dc_list, ed_dict_list, es_dict_list

#---Calc bpm---

def calc_bpm(contour_file, reverse, time_res=0.03328, centr_slices = 0):
	"""
	Calculate heartbeats per minute based on a contour file.
	The duration between ED phases is calculated within each slice and averaged over all durations.

	:param str contour_file: File containing contour information in Medis format
	:param bool reverse: Flag for reversing slice order
	:param float time_res: Time resolution in seconds between images
	:param int centr_slices: Number of central slices taken into account for calculation of bpm
	:returns: Beats per minute
	:rtype: float
	"""
	img_dims, fov, slices = extract_img_params(contour_file)
	pixel_spacing = round((fov [0] / img_dims[0] + fov [1] / img_dims[1]) / 2, 3) #1.6 for real-time
	mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
	slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
	ed_idx, es_idx, ed_count_list, es_count_list = get_ed_es_from_manual_rt(mlist, plist, segm_class=3, pixel_spacing=pixel_spacing)
	# lengths between two ED phases
	ed_length = []
	# take slices from mid section
	if centr_slices == 0:
		# take all slices into account
		start = 0
		end = len(ed_idx)
	else:
		start = len(ed_idx) // 2 - centr_slices // 2
		end = start + centr_slices
	for e in ed_idx[start:end]:
		for j in range(len(e)-1):
			ed_length.append(e[j+1][1] - e[j][1])
	beat_length = sum(ed_length) / len(ed_length)
	bps = 1 / (beat_length * time_res)
	bpm = bps * 60

	#standard deviation
	bpm_std = np.std([60/(l*time_res) for l in ed_length])

	##propagation of uncertainty
	#uncertainty = [(60/(l * time_res) - bpm) ** 2 for l in ed_length]
	#bpm_std = math.sqrt(1/(len(ed_length)*(len(ed_length)-1)) * sum(uncertainty))

	return bpm, bpm_std

#---Analyze LV area---

def get_phase_area_from_nnunet(segm_prefix:os.path, plist:list, segm_class:int = 3, pixel_spacing:float = 1, flag3d=False, slice_offset=0):
	"""
	Calculate a list of areas for a segmentation class from a directory containing Nifti files featuring single images.
	Nifti file names have the format <segm_prefix>xxyyy.nii.gz where xx presents the zero-filled slice number and yyy the zero-filled phase.

	:param os.path segm_prefix: Prefix for Nifti file names
	:param list plist: Lists of parameters for multiple slices. Format [slice, phase]
	:param int segm_class: Integer value of pixels of a segmentation class in the 2D segmentation for which the area is calculated. 1 - RV, 2 - Myo, 3 - LV
	:param tuple pixel_spacing: Pixel spacing [mm] for the calculation of the area
	:returns: Area of segmentation class
	:rtype: list
	"""
	area = []
	if flag3d:
		segm_file = segm_prefix + str(plist[0][1]).zfill(3)+".nii.gz"
		img = nnunet.read_nnunet_segm(segm_file)
		for p in plist:
			lv_count = np.count_nonzero(img[:,:,p[0]-slice_offset] == segm_class)
			area.append(lv_count * pixel_spacing * pixel_spacing)
	else:
		for p in plist:
			segm_file = segm_prefix + str(p[0]).zfill(2)+str(p[1]).zfill(3)+".nii.gz"
			if os.path.isfile(segm_file):
				img = nnunet.read_nnunet_segm(segm_file)
			else:
				img = nnunet.read_nnunet_segm(segm_prefix+str(p[0]).zfill(2)+".nii.gz")[:,:,p[1]]

			lv_count = np.count_nonzero(img == segm_class)
			area.append(lv_count * pixel_spacing * pixel_spacing)
	return area

def area_for_phase_contour_file(mlist:list, plist:list, phase:int, segm_class:int = 3, pixel_spacing:float = 1):
	"""
	Calculate a list of areas for a given phase from a list of masks for a given segmentation class.

	:param list mlist: Lists of segmentation masks for multiple slices
	:param list plist: Lists of parameters for multiple slices. Format [slice, phase]
	:param int phase: Phase index among slices
	:param int segm_class: Integer value of pixels of a segmentation class in the 2D segmentation for which the area is calculated
	:param tuple pixel_spacing: Pixel spacing [mm] for the calculation of the area
	:returns: Area of segmentation class
	:rtype: list
	"""
	area = []
	for num,ms in enumerate(mlist):
		for enum,m in enumerate(ms):
			if phase == plist[num][enum][1]:
				lv_count = np.count_nonzero(m == segm_class)
				area.append(lv_count*pixel_spacing*pixel_spacing)
	return area

def get_ed_es_param_from_contour_file(contour_file:os.path, reverse:bool, pixel_spacing:tuple=1, thickness:float=1):
	"""
	Determine the end-diastolic (ED) and end-systolic (ES) phases for a given contour file featuring contours in Medis format.

	:param os.path contour_file: Text file containing Medis session information and manual contours.
	:param bool reverse: Flag for reversing the slice indexing. Some contour files feature the slices in reversed order
	:param tuple pixel_spacing: Pixel spacing [mm] for the calculation of the area
	:param float thickness: Slice thickness [mm] for the calculation of blood volumes. The total thickness hereby refers to the slice thickness + slice gap
	:returns: tuple (ed_mc, es_mc, ed_vol, es_vol, ed_plist, es_plist)
		WHERE
		list ed_mc is areas of ED phase
		list es_mc is areas of ES phase
		float ed_vol is end-diastolic volume
		float es_vol is end-systolic volume
		list ed_plist is ED phase parameters in format [slice, phase]
		list es_plist is ES phase parameters in format [slice, phase]
	"""
	img_dims, fov, slices = extract_img_params(contour_file)
	mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
	slice_indexes, mlist, plist = combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
	EDphase, ESphase = extract_ED_ES_phase_cine(plist, mlist, segm_class=3)
	ed_mc = area_for_phase_contour_file(mlist.copy(), plist, EDphase, pixel_spacing=pixel_spacing)
	es_mc = area_for_phase_contour_file(mlist.copy(), plist, ESphase, pixel_spacing=pixel_spacing)

	ed_vol = sum(ed_mc) * thickness
	if 0 == ed_vol:
		print(contour_file)
	es_vol = sum(es_mc) * thickness

	#create one flat parameter list
	ed_plist, es_plist = [], []
	for ps in plist:
		for p in ps:

			if EDphase == p[1]:
				ed_plist.append(p)
			elif ESphase == p[1]:
				es_plist.append(p)

	return ed_mc, es_mc, ed_vol, es_vol, ed_plist, es_plist

def get_ed_es_from_text(file_path, keyword):
	#Return ED and ES phases from a text file
	readout = False
	param_list = []
	with open (file_path, 'rt', encoding="utf8", errors='ignore') as input:
		for line in input:
			if line.strip():
				content = line.split("\n")
				if readout:
					param_list.append([int(i) for i in content[0].split("\t")])
				else:
					if keyword in content[0]:
						readout = True
			else:
				if readout:
					break
	input.close()
	return param_list

#---Plot figures---

def cmap_map(function, cmap):
	""" Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
	This routine will break any discontinuous points in a colormap.
	"""
	cdict = cmap._segmentdata
	step_dict = {}
	# Firt get the list of points where the segments start or end
	for key in ('red', 'green', 'blue'):
		step_dict[key] = list(map(lambda x: x[0], cdict[key]))
	step_list = sum(step_dict.values(), [])
	step_list = np.array(list(set(step_list)))
	# Then compute the LUT, and apply the function to the LUT
	reduced_cmap = lambda step : np.array(cmap(step)[0:3])
	old_LUT = np.array(list(map(reduced_cmap, step_list)))
	new_LUT = np.array(list(map(function, old_LUT)))
	# Now try to make a minimal segment definition of the new LUT
	cdict = {}
	for i, key in enumerate(['red','green','blue']):
		this_cdict = {}
		for j, step in enumerate(step_list):
			if step in step_dict[key]:
				this_cdict[step] = new_LUT[j, i]
			elif new_LUT[j,i] != old_LUT[j, i]:
				this_cdict[step] = new_LUT[j, i]
		colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
		colorvector.sort()
		cdict[key] = colorvector

	return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)


def flip_rot(arr:np.array, flip:int=1, rot:int=1):
	"""
	Flip and rotate 2D-array.

	:param np.array: Input
	:param int flip: Axis to flip array over. flip < 0 for no flipping
	:param int rot: Number of 90° rotations. rot < 0 for no rotation
	:returns: Transformed array
	:rtype: np.array
	"""
	if -1 < flip:
		arr = np.flip(arr, flip)
	if 0 < rot:
		arr = np.rot90(arr, rot)
	return arr

def plot_measurement_types(vol, reverse, slice_idx, mask_mode=[], phase_mode="es", save_paths=[],
			contour_dir=contour_files_dir,
			img_dir = scanner_reco_dir,
			seg_dir=nnunet_output_dir, crop_dim=160,
			vmax_factor=1, DC=False,
			titles = ["cine", "real-time MRI (rest)", "real-time MRI (stress)", "real-time MRI (max stress)"], plot=True):
	"""
	Visualize different measurement forms (cine, real-time rest, rt stressm rt max stress) in a combined plot with optional segmentation.

	:param str vol: Volunteer string in format vol<id>
	:param bool reverse: Flag for reverse order of indexes of cardiac phase in contour file
	:param str mask_mode: Mode for optional segmentation mask. Either 'none', 'mc', 'comDL' or 'nnunet'
	:param str phase_mode: Mode for cardiac phase. Either 'es' or 'ed'
	:param list save_paths: List of file paths (for different file extensions) or single string to save plot
	:param str contour_dir: Directory containing contour files in Medis format
	:param str img_dir: Directory containing reconstructed images in format <img_dir>/<vol>/cine_scanner.{cfl,hdr}, .../rt_scanner.{cfl,hdr}, etc.
	:param str seg_dir: Directory containing nnU-Net segmentations
	:param int crop_dim: Crop dimension for reconstructed images
	:param list titles: List of titles for individual subplots
	:param bool plot: Flag for showing the plot. This way the plot can be saved without showing the plot.
	"""
	slice_select = [slice_idx for i in range(4)]

	img_files = [os.path.join(img_dir, vol, i) for i in ["cine_scanner", "rt_scanner", "rt_stress_scanner", "rt_maxstress_scanner"]]

	contour_files = [os.path.join(contour_dir, vol+"_" + s+"_manual"+contour_format) for s in ["cine", "rt", "rt_stress", "rt_maxstress"]]

	if 0 != len(mask_mode):
		comDL_contour_files = [os.path.join(contour_dir, vol+"_" + s+"_comDL"+contour_format) for s in ["cine", "rt", "rt_stress", "rt_maxstress"]]
		seg_subdirs = ["rtvol_cine_2d_single_cv", "rtvol_rt_2d_single_cv","rtvol_rt_stress_2d_single_cv", "rtvol_rt_maxstress_2d_single_cv"]
		seg_dirs = [os.path.join(seg_dir, s, "rtvol_"+vol[3:]) for s in seg_subdirs]

	rows = 1 if 0 == len(mask_mode) else len(mask_mode)
	columns = 4
	fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4))
	ax = axes.flatten()
	for i in range(rows*columns):
		ax[i].set_axis_off()

	for num, contour_file in enumerate(contour_files):

		img_dims, fov, slices = extract_img_params(contour_file)
		mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
		slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list, slices=slices, reverse=reverse)

		# find phase from manual contours
		if "cine" in contour_file:
			EDphase, ESphase = extract_ED_ES_phase_cine(plist, mlist, segm_class=3)
			phase = EDphase if "ed" == phase_mode else ESphase

		else:
			ed_idx, es_idx, ed_count_list, es_count_list = get_ed_es_from_manual_rt(mlist, plist, segm_class=3)
			if "ed" == phase_mode:
				for e in ed_idx:
					if slice_select[num] == e[0][0]:
						phase = e[0][1]
						continue
			else:
				for e in es_idx:
					if slice_select[num] == e[0][0]:
						phase = e[0][1]
						continue

		img = cfl.readcfl(img_files[num])
		img = np.abs(img[:,:,0,0,0,0,0,0,0,0,phase,0,0,slice_select[num]])
		img = crop_2darray(img, crop_dim)
		img = flip_rot(img, -1, 1)

		mask_mc = np.zeros((crop_dim, crop_dim))
		for i,ps in enumerate(plist):
			for j,p in enumerate(ps):
				if [slice_select[num], phase] == p:
					mask_mc = mlist[i][j]
					continue

		ax[num].imshow(img, cmap="gray", vmax=np.max(img)*vmax_factor)
		if 0 != len(titles):
			ax[num].set_title(titles[num])

		if 0 != len(mask_mode):

			for enum,m in enumerate(mask_mode):
				# find segmentation for slice and phase combination
				second_row_title = ""
				mask = np.zeros((crop_dim, crop_dim))
				if "mc" == m:
					second_row_title = "Manually corrected contours"
					mask = mask_mc.copy()

				elif "comDL" == m:
					img_dims, fov, slices = extract_img_params(contour_file)
					mask_list, param_list, ccsf = masks_and_parameters_from_file(comDL_contour_files[num], img_dims)
					slice_list_comDL, mlist, plist = combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
					for i,ps in enumerate(plist):
						for j,p in enumerate(ps):
							if [slice_select[num], phase] == p:
								mask = mlist[i][j]
								continue
					if DC:
						dice_coeffs, dict_list = calc_dice_coeff(ref=[mask_mc], pred=[mask], segm_classes=4)
						second_row_title = "DC: LV-"+str(round(dice_coeffs[3],2))+", MYO-"+str(round(dice_coeffs[2],2))+", RV-"+str(round(dice_coeffs[1],2))
					else:
						second_row_title = "comDL"
				elif "nnunet" == m:
					nnunet_file = seg_dirs[num]+str(slice_select[num]).zfill(2)+str(phase).zfill(3)+".nii.gz"
					if os.path.isfile(nnunet_file):
						mask = nnunet.read_nnunet_segm(nnunet_file)
					else:
						mask = nnunet.read_nnunet_segm(seg_dirs[num]+str(slice_select[num]).zfill(2)+".nii.gz")[:,:,phase]

					if DC:
						dice_coeffs, dict_list = calc_dice_coeff(ref=[mask_mc], pred=[mask], segm_classes=4)
						second_row_title = "DC: LV-"+str(round(dice_coeffs[3],2))+", MYO-"+str(round(dice_coeffs[2],2))+", RV-"+str(round(dice_coeffs[1],2))
					else:
						second_row_title = "nnU-Net"

				mask = crop_2darray(mask, crop_dim)
				mask = flip_rot(mask, -1, 1)
				ax[enum*columns+num].imshow(img, cmap="gray", vmax=np.max(img)*vmax_factor)
				masked_plt = np.ma.masked_where(mask == 0, mask)
				ax[enum*columns+num].imshow(masked_plt, cmap=light_jet,interpolation='none', alpha=0.4)
				ax[enum*columns+num].set_title(second_row_title)

	if 0 != len(save_paths):
		if list == type(save_paths):
			for s in save_paths:
				if ".png" in s:
					plt.savefig(s, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
				else:
					plt.savefig(s, bbox_inches='tight', pad_inches=0)
		else:
			if ".png" in save_paths:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
			else:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0)
	if plot:
		plt.show()
	else:
		plt.close()

def plot_measurement_types_axes(vol, axes, reverse, slice_idx, mask_mode=[], phase_mode="es", save_paths=[],
			contour_dir=contour_files_dir,
			img_dir = scanner_reco_dir,
			nnunet_output=nnunet_output_dir, crop_dim=160,
			vmax_factor=1, DC=False,
			titles = ["cine", "real-time MRI (rest)", "real-time MRI (stress)", "real-time MRI (max stress)"]):
	"""
	Visualize different measurement forms (cine, real-time rest, rt stressm rt max stress) in a combined plot with optional segmentation.

	:param str vol: Volunteer string in format vol<id>
	:param bool reverse: Flag for reverse order of indexes of cardiac phase in contour file
	:param str mask_mode: Mode for optional segmentation mask. Either 'none', 'mc', 'comDL' or 'nnunet'
	:param str phase_mode: Mode for cardiac phase. Either 'es' or 'ed'
	:param list save_paths: List of file paths (for different file extensions) or single string to save plot
	:param str contour_dir: Directory containing contour files in Medis format
	:param str img_dir: Directory containing reconstructed images in format <img_dir>/<vol>/cine_scanner.{cfl,hdr}, .../rt_scanner.{cfl,hdr}, etc.
	:param str seg_dir: Directory containing nnU-Net segmentations
	:param int crop_dim: Crop dimension for reconstructed images
	:param list titles: List of titles for individual subplots
	:param bool plot: Flag for showing the plot. This way the plot can be saved without showing the plot.
	"""
	slice_select = [slice_idx for i in range(4)]
	label_size = "xx-large"
	label_size = 18
	img_files = [os.path.join(img_dir, vol, i) for i in ["cine_scanner", "rt_scanner", "rt_stress_scanner", "rt_maxstress_scanner"]]

	contour_files = [os.path.join(contour_dir, vol+"_" + s+"_manual"+contour_format) for s in ["cine", "rt", "rt_stress", "rt_maxstress"]]

	if 0 != len(mask_mode):
		comDL_contour_files = [os.path.join(contour_dir, vol+"_" + s+"_comDL"+contour_format) for s in ["cine", "rt", "rt_stress", "rt_maxstress"]]
		seg_subdirs = ["rtvol_cine_2d_single_cv", "rtvol_rt_2d_single_cv","rtvol_rt_stress_2d_single_cv", "rtvol_rt_maxstress_2d_single_cv"]
		seg_dirs = [os.path.join(nnunet_output, s, "rtvol_"+vol[3:]) for s in seg_subdirs]

	for num, (contour_file, ax) in enumerate(zip(contour_files, axes)):

		img_dims, fov, slices = extract_img_params(contour_file)
		mask_list, param_list, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
		slice_list_mc, mlist, plist = combine_masks_multi(mask_list, param_list, slices=slices, reverse=reverse)

		# find phase from manual contours
		if "cine" in contour_file:
			EDphase, ESphase = extract_ED_ES_phase_cine(plist, mlist, segm_class=3)
			phase = EDphase if "ed" == phase_mode else ESphase

		else:
			ed_idx, es_idx, ed_count_list, es_count_list = get_ed_es_from_manual_rt(mlist, plist, segm_class=3)
			if "ed" == phase_mode:
				for e in ed_idx:
					if slice_select[num] == e[0][0]:
						phase = e[0][1]
						continue
			else:
				for e in es_idx:
					if slice_select[num] == e[0][0]:
						phase = e[0][1]
						continue

		img = cfl.readcfl(img_files[num])
		img = np.abs(img[:,:,0,0,0,0,0,0,0,0,phase,0,0,slice_select[num]])
		img = crop_2darray(img, crop_dim)
		img = flip_rot(img, -1, 1)

		mask_mc = np.zeros((crop_dim, crop_dim))
		for i,ps in enumerate(plist):
			for j,p in enumerate(ps):
				if [slice_select[num], phase] == p:
					mask_mc = mlist[i][j]
					continue

		ax.imshow(img, cmap="gray", vmax=np.max(img)*vmax_factor)
		if 0 != len(titles):
			ax.set_title(titles[num], size=label_size)

		if 0 != len(mask_mode):

			for enum,m in enumerate(mask_mode):
				# find segmentation for slice and phase combination
				second_row_title = ""
				mask = np.zeros((crop_dim, crop_dim))
				if "mc" == m:
					second_row_title = "Manually corrected contours"
					mask = mask_mc.copy()

				elif "comDL" == m:
					img_dims, fov, slices = extract_img_params(contour_file)
					mask_list, param_list, ccsf = masks_and_parameters_from_file(comDL_contour_files[num], img_dims)
					slice_list_comDL, mlist, plist = combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
					for i,ps in enumerate(plist):
						for j,p in enumerate(ps):
							if [slice_select[num], phase] == p:
								mask = mlist[i][j]
								continue
					if DC:
						dice_coeffs, dict_list = calc_dice_coeff(ref=[mask_mc], pred=[mask], segm_classes=4)
						second_row_title = "DC: LV-"+str(round(dice_coeffs[3],2))+", MYO-"+str(round(dice_coeffs[2],2))+", RV-"+str(round(dice_coeffs[1],2))
					else:
						second_row_title = "comDL"
				elif "nnunet" == m:
					nnunet_file = seg_dirs[num]+str(slice_select[num]).zfill(2)+str(phase).zfill(3)+".nii.gz"
					if os.path.isfile(nnunet_file):
						mask = nnunet.read_nnunet_segm(nnunet_file)
					else:
						mask = nnunet.read_nnunet_segm(seg_dirs[num]+str(slice_select[num]).zfill(2)+".nii.gz")[:,:,phase]

					if DC:
						dice_coeffs, dict_list = calc_dice_coeff(ref=[mask_mc], pred=[mask], segm_classes=4)
						second_row_title = "DC: LV-"+str(round(dice_coeffs[3],2))+", MYO-"+str(round(dice_coeffs[2],2))+", RV-"+str(round(dice_coeffs[1],2))
					else:
						second_row_title = "nnU-Net"

				mask = crop_2darray(mask, crop_dim)
				mask = flip_rot(mask, -1, 1)
				ax.imshow(img, cmap="gray", vmax=np.max(img)*vmax_factor)
				masked_plt = np.ma.masked_where(mask == 0, mask)
				ax.imshow(masked_plt, cmap=light_jet,interpolation='none', alpha=0.4)
				ax.set_title(second_row_title, size=label_size)

def plot_mc_nnunet(contour_dir, img_dir, seg_dir, rtvol_dict, param_list, flag3d=True, mode = "nnunet",
			crop_dim=160, contour_suffix = "_cine.txt", save_paths=[], img_suffix="cine_scanner",
			check=False, plot=True):
	"""
	Plot a selection of images and segmentation masks for Medis contours and neural network segmentation.
	Parameters for the image selection are given with idx_list in format (id, slice, phase).

	:param str contour_dir: Directory containing contour files in format <contour_dir>/<id><contour_suffix>
	:param str img_dir: Directory containing images in format <img_dir>/<id>/cine_scanner
	:param str seg_dir: Directory containing BART nnet or nnU-Net segmentations.
	:param list rtvol_dict: List of dictionaries in format [{"id":<id>, "reverse":bool}]
	:param list param_list: List of parameters in format (<id>, slice, phase)
	:param bool flag3d: Flag for 3D segmentation of nnU-Net
	:param str mode: Mode for segmentation. 'nnunet' or 'nnet'
	:param int crop_dim: Crop dimension for center-cropping images and masks
	:param str contour_suffix: Suffix for contour files
	:param list save_paths: List of file paths (for different file extensions) or single string to save plot
	:param str img_suffix: Suffix for image file
	:param bool check: Sets title of subfigures to slice and phase information
	:param bool plot: Flag for showing the plot. This way the plot can be saved without showing the plot.
	"""
	rows = 2
	columns = len(param_list)
	fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4))
	ax = axes.flatten()

	for i in range(rows*columns):
		ax[i].set_axis_off()

	for num,e in enumerate(param_list):
		(vol, slice, phase) = tuple(e)
		reverse = [data["reverse"] for data in rtvol_dict if vol == data["id"]][0]
		if "flip_rot" in rtvol_dict[0]:
			f = [data["flip_rot"] for data in rtvol_dict if vol == data["id"]][0]
		else:
			f = [-1,0]

		#manual contours
		contour_file = os.path.join(contour_dir, vol+contour_suffix)
		img_dims, fov, slices = extract_img_params(contour_file)
		mask_list, plist, ccsf = masks_and_parameters_from_file(contour_file, img_dims)
		slice_indexes, mlist, plist = combine_masks_multi(mask_list, plist, slices, reverse)
		for m,ps in enumerate(plist):
			for n,p in enumerate(ps):
				if [slice,phase] == p:
					mask_mc = mlist[m][n]

		if not list == type(crop_dim):
			mask_mc = crop_2darray(mask_mc, crop_dim)
		else:
			mask_mc = crop_2darray(mask_mc, crop_dim[num])

		mask_dl = np.zeros(mask_mc.shape, dtype=float)

		comp_title = ""
		#BART nnet segmentation, format: <seg_dir>/<id>/cine_scanner
		if "nnet" == mode:
			segm_file = os.path.join(seg_dir, vol, img_suffix)
			segm = cfl.readcfl(segm_file)
			mask_dl = np.abs(segm[0,:,:,0,phase,slice])
			comp_title = "BART nnet Segmentation"
		#nnUNet segmentation, format: <seg_dir>/rtvol_<id[3:]>
		elif "nnunet" == mode:
			nnunet_prefix = os.path.join(seg_dir, "rtvol_" + vol[3:])
			if flag3d:
				mask_nnunet = nnunet.read_nnunet_segm(nnunet_prefix+str(phase).zfill(3)+".nii.gz")
				mask_nnunet = mask_nnunet[:,:,slice]

			else:
				nnunet_file = nnunet_prefix+str(slice).zfill(2)+str(phase).zfill(3)+".nii.gz"
				if os.path.isfile(nnunet_file):
					mask_nnunet = nnunet.read_nnunet_segm(nnunet_file)
				else:
					mask_nnunet = nnunet.read_nnunet_segm(nnunet_prefix+str(slice).zfill(2)+".nii.gz")[:,:,phase]

			if not list == type(crop_dim):
				mask_dl = crop_2darray(mask_nnunet, crop_dim)
			else:
				mask_dl = crop_2darray(mask_nnunet, crop_dim[num])

			comp_title = "nnU-Net segmentation"

		img_file = os.path.join(img_dir, vol, img_suffix)
		images = cfl.readcfl(img_file)
		img = np.abs(images[:,:,0,0,0,0,0,0,0,0,phase,0,0,slice])

		if not list == type(crop_dim):
			img = crop_2darray(img, crop_dim)
		else:
			img = crop_2darray(img, crop_dim[num])

		img = flip_rot(img, f[0], f[1])
		mask_mc = flip_rot(mask_mc, f[0], f[1])
		mask_dl = flip_rot(mask_dl, f[0], f[1])

		ax[num].imshow(img, cmap="gray")
		mask_plt = np.abs(mask_mc)
		masked_plt = np.ma.masked_where(mask_plt == 0, mask_plt)
		ax[num].imshow(masked_plt, cmap=light_jet,interpolation='none', alpha=0.4, vmin=1, vmax=3)
		if check:
			ax[num].set_title(vol + " slc" + str(slice) + " phs" + str(phase))
		else:
			ax[num].set_title("Manually corrected contours")

		ax[columns+num].imshow(np.abs(img), cmap="gray")
		mask_plt = np.abs(mask_dl)
		masked_plt = np.ma.masked_where(mask_plt == 0, mask_plt)
		ax[columns+num].imshow(masked_plt, cmap=light_jet,interpolation='none', alpha=0.4, vmin=1, vmax=3)
		ax[columns+num].set_title(comp_title)

	if 0 != len(save_paths):
		if list == type(save_paths):
			for s in save_paths:
				if ".png" in s:
					plt.savefig(s, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
				else:
					plt.savefig(s, bbox_inches='tight', pad_inches=0)
		else:
			if ".png" in save_paths:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0.1, dpi=png_dpi)
			else:
				plt.savefig(save_paths, bbox_inches='tight', pad_inches=0)
	if plot:
		plt.show()
	else:
		plt.close()

#---Prepare nnU-Net input---

def prepare_nnunet_cine(rtvol, img_dir, contour_dir, nnunet_dir,
			cine_single = "Task501_rtvolcine_single", cine_3d = "Task502_rtvolcine3d", cine_3d_LV = "Task503_rtvolcine3d_LV"):
	""""
	Pre-process reconstructed images of cine MRI for application of nnU-Net.
	Creates directories for single cine 2D images, cine 3D images (along cardiac phase dimension)
	and cine 3D images only containing slices containing the heart.
	"""
	output_dir = os.path.join(nnunet_dir, "nnUNet_raw_data")
	prefix = "rtvol_"
	for data in rtvol:
		vol = data["id"]
		reverse = data["reverse"]
		img_file = os.path.join(img_dir, vol, "cine_scanner")
		contour_file = os.path.join(contour_dir, vol + "_cine_manual"+contour_format)
		if "" != cine_single:
			if not os.path.isdir(os.path.join(output_dir, cine_single, "imagesTs")):
				os.makedirs(os.path.join(output_dir, cine_single, "imagesTs"))
			output_prefix = os.path.join(output_dir, cine_single, "imagesTs", prefix+vol[3:])
			create_nnunet_input_cine(img_file, output_prefix, contour_file=contour_file, reverse=reverse, output_mode="2d", slice_selection=False)
		if "" != cine_3d:
			if not os.path.isdir(os.path.join(output_dir, cine_3d, "imagesTs")):
				os.makedirs(os.path.join(output_dir, cine_3d, "imagesTs"))
			output_prefix = os.path.join(output_dir, cine_3d, "imagesTs", prefix+vol[3:])
			create_nnunet_input_cine(img_file, output_prefix, contour_file=contour_file, reverse=reverse, output_mode="3d", slice_selection=False)
		if "" != cine_3d_LV:
			if not os.path.isdir(os.path.join(output_dir, cine_3d_LV, "imagesTs")):
				os.makedirs(os.path.join(output_dir, cine_3d_LV, "imagesTs"))
			output_prefix = os.path.join(output_dir, cine_3d_LV, "imagesTs", prefix+vol[3:])
			create_nnunet_input_cine(img_file, output_prefix, contour_file=contour_file, reverse=reverse, output_mode="3d", slice_selection=True)

def prepare_nnunet_rt(rtvol, img_dir, nnunet_dir, rt_single = "Task511_rtvolrt_single", rt_slice = "Task516_rtvolrt"):
	""""
	Pre-process reconstructed images of real-time MRI for application of nnU-Net.
	Creates directories for single images and a time series within a slice.
	"""
	output_dir = os.path.join(nnunet_dir, "nnUNet_raw_data")
	prefix = "rtvol_"
	for data in rtvol:
		vol = data["id"]
		img_file = os.path.join(img_dir, vol, "rt_scanner")
		if "" != rt_single:
			if not os.path.isdir(os.path.join(output_dir, rt_single, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_single, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_single, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="single")
		if "" != rt_slice:
			if not os.path.isdir(os.path.join(output_dir, rt_slice, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_slice, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_slice, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="slice")

def prepare_nnunet_rt_stress(rtvol, img_dir, nnunet_dir, rt_single = "Task512_rtvolrt_stress_single", rt_slice = "Task517_rtvolrt_stress"):
	""""
	Pre-process reconstructed images of real-time MRI under stress for application of nnU-Net.
	Creates directories for single images and a time series within a slice.
	"""
	output_dir = os.path.join(nnunet_dir, "nnUNet_raw_data")
	prefix = "rtvol_"
	for data in rtvol:
		vol = data["id"]
		img_file = os.path.join(img_dir, vol, "rt_stress_scanner")
		if "" != rt_single:
			if not os.path.isdir(os.path.join(output_dir, rt_single, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_single, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_single, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="single")
		if "" != rt_slice:
			if not os.path.isdir(os.path.join(output_dir, rt_slice, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_slice, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_slice, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="slice")

def prepare_nnunet_rt_maxstress(rtvol, img_dir, nnunet_dir, rt_single = "Task513_rtvolrt_maxstress_single", rt_slice = "Task518_rtvolrt_maxstress"):
	""""
	Pre-process reconstructed images of real-time MRI under maximal stress for application of nnU-Net.
	Creates directories for single images and a time series within a slice.
	"""
	output_dir = os.path.join(nnunet_dir, "nnUNet_raw_data")
	prefix = "rtvol_"
	for data in rtvol:
		vol = data["id"]
		img_file = os.path.join(img_dir, vol, "rt_maxstress_scanner")
		if "" != rt_single:
			if not os.path.isdir(os.path.join(output_dir, rt_single, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_single, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_single, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="single")
		if "" != rt_slice:
			if not os.path.isdir(os.path.join(output_dir, rt_slice, "imagesTs")):
				os.makedirs(os.path.join(output_dir, rt_slice, "imagesTs"))
			output_prefix = os.path.join(output_dir, rt_slice, "imagesTs", prefix+vol[3:])
			create_nnunet_input_rt(img_file, output_prefix, mode="slice")

def prepare_nnunet(img_dir, contour_dir, nnunet_dir):
	""""
	Pre-process reconstructed images of cine and real-time MRI for application of nnU-Net.
	For cine: Creates directories for single cine 2D images, cine 3D images (along cardiac phase dimension)
	and cine 3D images only containing slices containing the heart.
	For real-time: Creates directories for single images and a time series within a slice.
	"""
	rtvol = [
	{"id":"vol01",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":61},
	{"id":"vol02",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":55},
	{"id":"vol03",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":64},
	{"id":"vol04",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":50},
	{"id":"vol05",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":53},
	{"id":"vol06",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":67},
	{"id":"vol07",	"reverse":True	, "flip_rot":[-1,1], "gender":"f", "age":66},
	{"id":"vol08",	"reverse":True	, "flip_rot":[-1,0], "gender":"m", "age":65},
	{"id":"vol09",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":50},
	{"id":"vol10",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":64},
	{"id":"vol11",	"reverse":True	, "flip_rot":[-1,1], "gender":"m", "age":44},
	{"id":"vol12",	"reverse":True	, "flip_rot":[-1,1], "gender":"f", "age":56},
	{"id":"vol13",	"reverse":True	, "flip_rot":[-1,0], "gender":"m", "age":42},
	{"id":"vol14",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":45},
	{"id":"vol15",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":49}
	]

	prepare_nnunet_cine(rtvol, img_dir, contour_dir, nnunet_dir)
	prepare_nnunet_rt(rtvol, img_dir, nnunet_dir)
	prepare_nnunet_rt_stress(rtvol, img_dir, nnunet_dir)
	prepare_nnunet_rt_maxstress(rtvol, img_dir, nnunet_dir)