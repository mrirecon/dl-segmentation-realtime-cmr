#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Copyright 2023. Uecker Lab, University Medical Center Goettingen.
author: Martin Schilling (martin.schilling@med.uni-goettingen.de), 2023

Script to create figures used in the manuscript "Assessment of Deep Learning Segmentation
in Free-Breathing Real-Time Cardiac Magnetic Resonance Imaging".
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
import math
import argparse

from inspect import getsourcefile

sys.path.append(os.path.join(os.environ["TOOLBOX_PATH"], "python"))
sys.path.append(os.path.dirname(getsourcefile(lambda:0)))

import assess_dl_seg_utils as assess_utils
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

def calc_DC_and_bpm(rtvol_dict, mode=["nnunet"],
		contour_dir = contour_files_dir,
		seg_dir = nnunet_output_dir):
	"""
	Calculate Dice's coefficient (DC) and beats per minute (bpm) for a list of dictionaries
	and add this information to the dictionary entries for Medis DL ACD and/or nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with entries ['id'] and ['reverse']
	:param list mode: List of modes for calculation of Dice's coefficient. Entries can be 'nnunet' and/or 'comDL'
	:param str contour_dir: Path to directory containing contour files in '.con' or '.txt' file format
	:param str seg_dir: Directory containing subdirectories for nnU-Net segmentation outputs
	"""
	segm_classes = 3
	phase_select = "combined"
	seg_subdirs = ["rtvol_rt_2d_single_cv", "rtvol_rt_stress_2d_single_cv", "rtvol_rt_maxstress_2d_single_cv"]
	modes = ["rt", "rt_stress", "rt_maxstress"]
	manual_contour_suffixes = ["_rt_manual"+contour_format, "_rt_stress_manual"+contour_format, "_rt_maxstress_manual"+contour_format]
	comDL_contour_suffixes = ["_rt_comDL"+contour_format, "_rt_stress_comDL"+contour_format, "_rt_maxstress_comDL"+contour_format]
	descr = ["RV", "Myo", "LV"]

	for d in rtvol_dict:
		vol = d["id"]
		reverse = d["reverse"]
		print(vol)
		sessions = [os.path.join(contour_dir, vol+"_" + s+"_manual"+contour_format) for s in modes]
		for i,contour_file in enumerate(sessions):
			if os.path.isfile(contour_file):
				bpm, bpm_std = assess_utils.calc_bpm(contour_file, reverse, time_res=0.03328, centr_slices=3)
				#print(contour_file, "bpm", round(bpm,0), "bpm_std", round(bpm_std,0))
				d["bpm"+modes[i]] = int(bpm)
				d["bpmstd"+modes[i]] = int(bpm_std)

		for i,m in enumerate(modes):
			manual_contour_suffix = manual_contour_suffixes[i]
			contour_file = os.path.join(contour_dir, vol+manual_contour_suffix)
			if not os.path.isfile(contour_file):
				continue

			if "nnunet" in mode:
				class_acc = [[] for _ in range(segm_classes)]
				segm_input = os.path.join(os.path.join(seg_dir, seg_subdirs[i]), "rtvol_"+vol[3:])
				ed_dc, ed_dict, es_dc, es_dict = assess_utils.get_ed_es_dice_from_session_rt(contour_file, reverse, segm_input, phase_select=phase_select)
				for e in ed_dict:
					for j in range(segm_classes):
						class_acc[j].append(e['class'+str(j+1)])
				for j,tag in enumerate(descr):
					d['DC'+"nnunet"+m+tag] = sum(class_acc[j]) / len(class_acc[j])
					d['DCstd'+"nnunet"+m+tag] = np.std(class_acc[j])

			if "comDL" in mode:
				class_acc = [[] for _ in range(segm_classes)]
				segm_input = os.path.join(contour_dir, vol+comDL_contour_suffixes[i])
				ed_dc, ed_dict, es_dc, es_dict = assess_utils.get_ed_es_dice_from_session_rt(contour_file, reverse, segm_input, phase_select=phase_select)
				for e in ed_dict:
					for j in range(segm_classes):
						class_acc[j].append(e['class'+str(j+1)])
				for j,tag in enumerate(descr):
					d['DC'+"comDL"+m+tag] = sum(class_acc[j]) / len(class_acc[j])
					d['DCstd'+"comDL"+m+tag] = np.std(class_acc[j])

def plot_DC_vs_bpm(rtvol_dict, save_paths=[], contour_mode="nnunet", ylim=[], plot=True, mode="noerror", title=""):
	"""
	Plot Dice's coefficient of a list of input dictionaries.

	:param list rtvol_dict: List of dictionaries
	:param str save_path: Path to save plot. Default: Output is not saved
	:param str contour_mode: Contour mode for plotting. Either 'nnunet' or 'comDL'
	"""
	modes = ["rt", "rt_stress", "rt_maxstress"]
	descr = ["LV", "Myo", "RV"]
	markers = ["o", "s", "D"]
	colors = ["crimson", "limegreen", "royalblue"]
	markersize = 15
	dice_scores = [[] for i in descr]
	dice_scores_std = [[] for i in descr]
	heartrates = []
	heartrates_std = []
	for d in rtvol_dict:
		for m in modes:
			heartrate_key = "bpm"+m
			heartrate_std_key = "bpmstd"+m
			if heartrate_key in d:
				heartrates.append(d[heartrate_key])
				heartrates_std.append(d[heartrate_std_key])
				for i, desc in enumerate(descr):
					dice_scores[i].append(d["DC"+contour_mode+m+desc])
					dice_scores_std[i].append(d["DCstd"+contour_mode+m+desc])
	if "error" == mode:
		for i, desc in enumerate(descr):
			plt.errorbar(heartrates, dice_scores[i], xerr=heartrates_std, yerr=dice_scores_std[i], fmt='none',
			label=desc, markersize=markersize, ecolor=colors[i])

	else:
			for i, desc in enumerate(descr):
				plt.scatter(heartrates, dice_scores[i], label=desc, s=markersize, c=colors[i], marker=markers[i])

	tick_size="large"
	label_size="xx-large"
	plt.xlabel("Heart rate [bpm]", size=label_size)
	plt.ylabel("Dice's coefficient", size=label_size)
	if "" != title:
		plt.title(title, size=label_size)

	plt.xticks(ticks=[i*20+60 for i in range(0,6)], size=tick_size)
	plt.yticks(ticks=[i*0.1+round(ylim[0],1) for i in range(0,int((ylim[1]-ylim[0])*10)+1)], size=tick_size)


	if 0 != len(ylim):
		plt.ylim(ylim)

	plt.legend(loc="lower left")
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

def calc_mean_stdv_two_sets(setA, setB, mode="absolute", scale=1, precision=None):
	"""
	Calculate mean and standard deviation of two sets.
	"""
	diff = []
	for (m,s) in zip(setA, setB):
		if "absolute" == mode:
			diff.append(np.abs((m-s) * scale))
		else:
			diff.append(np.abs((m-s)/m) * scale)

	mean = (sum(setA) + sum(setB)) / (len(setA) + len(setB)) * scale
	stdv_mean = np.std(setA+setB) * scale
	mean_diff = sum(diff) / len(diff)
	stdv_diff = np.std(diff)
	if None != precision:
		return round(mean_diff, precision), round(stdv_diff, precision), round(mean,precision), round(stdv_mean, precision)
	return mean_diff, stdv_diff, mean, stdv_mean

def write_output_cardiac_function_parameters(output_file, ed_vol, es_vol, ef, rtvol_dict=rtvol, scale = 0.001, precision=1,
						header="Manually corrected MEDIS DL ACD"):
	"""
	Write output mean and standard deviation parameters.

	:param str output_file: File path to output text file
	:param tuple ed_tuple: Tuple of lists of LV volume for ED phase for manual contours, comDL contours and nnU-Net contours
	:param tuple es_tuple: Tuple of lists of LV volume for ES phase for manual contours, comDL contours and nnU-Net contours
	:param tuple ef_tuple: Tuple of lists of ejection fraction for manual contours, comDL contours and nnU-Net contours
	:param float scale: Scale for parameters
	:param int precision: Precision for output data. Default: 3
	"""
	if "" != output_file:
		with open (output_file, 'w', encoding="utf8", errors='ignore') as output:
			output.write(header+"\n")
			output.write("EDV\n")
			for num, e in enumerate(ed_vol):
				output.write(rtvol_dict[num]["id"]+"\t"+str(round(e*scale,precision))+"\n")
			output.write("ESV\n")
			for num, e in enumerate(es_vol):
				output.write(rtvol_dict[num]["id"]+"\t"+str(round(e*scale,precision))+"\n")
			output.write("EF\n")
			for num, e in enumerate(ef):
				output.write(rtvol_dict[num]["id"]+"\t"+str(round(e,precision))+"\n")
		output.close()

def calc_mean_stdv_parameters_cine(rtvol_dict, seg_dir = os.path.join(nnunet_output_dir, "rtvol_cine_2d_single_cv/"),
				   contour_dir = contour_files_dir, flag3d=False, contour_format=contour_format,
				   pixel_spacing = 1.328125, slice_selection=False):
	"""
	Calculate mean and standard deviation parameters for cine MRI for manual contours, comDL contours and nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with volunteer id and reverse flag
	:param str seg_dir: Directory containing segmentation of nnU-Net for cine measurements
	:param str contour_dir: Directory containing Medis contour files in format <vol_id>_cine_manual.con and <vol_id>_cine_comDL.con
	:param bool flag3d: Flag for marking input data as 2D or 3D data
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param bool slice_selection: Selection of slices for nnU-Net segmentation confined to slices containing the heart
	"""
	thickness = 6.6
	ed_vol_mc, es_vol_mc = [], []
	ed_vol_comDL, es_vol_comDL = [], []
	ed_vol_nnunet, es_vol_nnunet = [], []
	ef_mc, ef_comDL, ef_nnunet = [], [], []

	segm_class = 3

	for data in rtvol_dict:
		vol = data["id"]
		reverse = data["reverse"]
		session = os.path.join(contour_dir, vol+"_cine_manual"+contour_format)
		slice_offset = 0

		ed_mc, es_mc, ed_vol, es_vol, ed_plist, es_plist = assess_utils.get_ed_es_param_from_session(session, reverse, pixel_spacing, thickness)

		if slice_selection:
			slice_list = list(set([p[0] for p in ed_plist]))
			slice_list.sort()
			slice_offset = slice_list[0]

		ed_vol_mc.append(ed_vol)
		es_vol_mc.append(es_vol)
		ef_mc.append( (ed_vol - es_vol) / ed_vol * 100)

		session = os.path.join(contour_dir, vol+"_cine_comDL"+contour_format)
		if os.path.isfile(session):
			img_dims, fov, slices = assess_utils.extract_img_params(session)
			#mask_list, param_list, ccsf = medis.masks_and_parameters_from_file(session, img_dims, no_duplicates=True)
			mask_list, param_list, ccsf = assess_utils.masks_and_parameters_from_file(session, img_dims)
			slice_indexes, mlist, plist = assess_utils.combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
			ed_masks = assess_utils.mask_for_param_selection(mlist, plist, param=ed_plist)
			ed_comDL = []
			for m in ed_masks:
				lv_count = np.count_nonzero(m == segm_class)
				ed_comDL.append(lv_count * pixel_spacing * pixel_spacing)
			es_masks = assess_utils.mask_for_param_selection(mlist, plist, param=es_plist)
			es_comDL = []
			for m in es_masks:
				lv_count = np.count_nonzero(m == segm_class)
				es_comDL.append(lv_count * pixel_spacing * pixel_spacing)
			ed_vol = sum(ed_comDL) * thickness
			es_vol = sum(es_comDL) * thickness
			ed_vol_comDL.append(ed_vol)
			es_vol_comDL.append(es_vol)
			ef_comDL.append( (ed_vol - es_vol) / ed_vol * 100)

		segm_prefix = os.path.join(seg_dir, "rtvol_" + vol[3:])
		if os.path.isdir(seg_dir):
			#for single nnUNet images, ed_plist is list of parameters, for 3D images, it is only the phase value
			ed_nnunet = assess_utils.get_phase_area_from_nnunet(segm_prefix, ed_plist, segm_class = 3, pixel_spacing = pixel_spacing,
							flag3d=flag3d, slice_offset=slice_offset)
			es_nnunet = assess_utils.get_phase_area_from_nnunet(segm_prefix, es_plist, segm_class = 3, pixel_spacing = pixel_spacing,
							flag3d=flag3d, slice_offset=slice_offset)
			ed_vol = sum(ed_nnunet) * thickness
			es_vol = sum(es_nnunet) * thickness
			ed_vol_nnunet.append(ed_vol)
			es_vol_nnunet.append(es_vol)
			ef_nnunet.append( (ed_vol - es_vol) / ed_vol * 100)

	return (ed_vol_mc, ed_vol_comDL, ed_vol_nnunet), (es_vol_mc, es_vol_comDL, es_vol_nnunet), (ef_mc, ef_comDL, ef_nnunet)

def calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_2d_single_cv/"),
				contour_dir=contour_files_dir, flag3d=False,
				pixel_spacing=1.6, session_mc_suffix="_rt_manual"+contour_format, session_comDL_suffix="_rt_comDL"+contour_format,
				exp_dir=end_exp_dir, ed_es_phase_file="_rt.txt"):
	"""
	Calculate mean and standard deviation parameters for rt MRI for manual, comDL and nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with volunteer id and reverse flag
	:param str seg_dir: Directory containing segmentation of nnU-Net for cine measurements
	:param str contour_dir: Directory containing Medis contour files in format <vol_id>_cine_manual.con and <vol_id>_cine_comDL.con
	:param bool flag3d: Flag for marking input data as 2D or 3D data
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param str session_mc_suffix: Suffix for manual contour file
	:param str session_comDL_suffix: Suffix for comDL contour file
	:param str exp_dir: Directory containing text files with indexes for the end-expiration state
	:param str ed_es_phase_file: Suffix for text file in 'exp_dir' containing ED and ES phase information. Format <exp_dir>/<vol_id><ed_es_phase_file>
	"""
	thickness = 6.6
	ed_vol_mc, es_vol_mc = [], []
	ed_vol_comDL, es_vol_comDL = [], []
	ed_vol_nnunet, es_vol_nnunet = [], []
	ef_mc, ef_comDL, ef_nnunet = [], [], []

	segm_class = 3
	for data in rtvol_dict:
		vol = data["id"]
		reverse = data["reverse"]
		file_path = os.path.join(exp_dir, vol+ed_es_phase_file)
		ed_plist = assess_utils.get_ed_es_from_text(file_path, "ED")
		es_plist = assess_utils.get_ed_es_from_text(file_path, "ES")

		session = os.path.join(contour_dir, vol+session_mc_suffix)
		img_dims, fov, slices = assess_utils.extract_img_params(session)
		#mask_list, param_list, ccsf = medis.masks_and_parameters_from_file(session, img_dims, no_duplicates=True)
		mask_list, param_list, ccsf = assess_utils.masks_and_parameters_from_file(session, img_dims)
		slice_indexes, mlist, plist = assess_utils.combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
		ed_masks = assess_utils.mask_for_param_selection(mlist, plist, param=ed_plist)
		ed_mc = []
		for m in ed_masks:
			lv_count = np.count_nonzero(m == segm_class)
			ed_mc.append(lv_count * pixel_spacing * pixel_spacing)
		es_masks = assess_utils.mask_for_param_selection(mlist, plist, param=es_plist)
		es_mc = []
		for m in es_masks:
			lv_count = np.count_nonzero(m == segm_class)
			es_mc.append(lv_count * pixel_spacing * pixel_spacing)
		ed_vol = sum(ed_mc) * thickness
		es_vol = sum(es_mc) * thickness
		ed_vol_mc.append(ed_vol)
		es_vol_mc.append(es_vol)
		ef_mc.append( (ed_vol - es_vol) / ed_vol * 100)

		session = os.path.join(contour_dir, vol+session_comDL_suffix)
		if os.path.isfile(session):
			img_dims, fov, slices = assess_utils.extract_img_params(session)
			#mask_list, param_list, ccsf = medis.masks_and_parameters_from_file(session, img_dims, no_duplicates=True)
			mask_list, param_list, ccsf = assess_utils.masks_and_parameters_from_file(session, img_dims)
			slice_indexes, mlist, plist = assess_utils.combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
			ed_masks = assess_utils.mask_for_param_selection(mlist, plist, param=ed_plist)
			ed_comDL = []
			for m in ed_masks:
				lv_count = np.count_nonzero(m == segm_class)
				ed_comDL.append(lv_count * pixel_spacing * pixel_spacing)
			es_masks = assess_utils.mask_for_param_selection(mlist, plist, param=es_plist)
			es_comDL = []
			for m in es_masks:
				lv_count = np.count_nonzero(m == segm_class)
				es_comDL.append(lv_count * pixel_spacing * pixel_spacing)
			ed_vol = sum(ed_comDL) * thickness
			es_vol = sum(es_comDL) * thickness
			ed_vol_comDL.append(ed_vol)
			es_vol_comDL.append(es_vol)
			ef_comDL.append( (ed_vol - es_vol) / ed_vol * 100)

		segm_prefix = os.path.join(seg_dir, "rtvol_" + vol[3:])
		if os.path.isdir(seg_dir):
			#for single nnUNet images, ed_plist is list of parameters, for 3D images, it is only the phase value
			ed_nnet = assess_utils.get_phase_area_from_nnunet(segm_prefix, ed_plist, segm_class = 3, pixel_spacing = pixel_spacing, flag3d=flag3d)
			es_nnet = assess_utils.get_phase_area_from_nnunet(segm_prefix, es_plist, segm_class = 3, pixel_spacing = pixel_spacing, flag3d=flag3d)
			ed_vol = sum(ed_nnet) * thickness
			es_vol = sum(es_nnet) * thickness
			ed_vol_nnunet.append(ed_vol)
			es_vol_nnunet.append(es_vol)
			ef_nnunet.append( (ed_vol - es_vol) / ed_vol * 100)
	return (ed_vol_mc, ed_vol_comDL, ed_vol_nnunet), (es_vol_mc, es_vol_comDL, es_vol_nnunet), (ef_mc, ef_comDL, ef_nnunet)

def read_cardiac_function_parameters(file_mc, file_comDL, file_nnunet):
	"""
	Read cardiac function parameters
	"""
	edv_mc, esv_mc, ef_mc = read_clinical_measures(file_mc, no_dict=True)
	edv_comDL, esv_comDL, ef_comDL = read_clinical_measures(file_comDL, no_dict=True)
	edv_nnunet, esv_nnunet, ef_nnunet = read_clinical_measures(file_nnunet, no_dict=True)

	return (edv_mc, edv_comDL, edv_nnunet), (esv_mc, esv_comDL, esv_nnunet), (ef_mc, ef_comDL, ef_nnunet)

def plot_ba_ef(save_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_stress, set_colors=["royalblue", "limegreen", "crimson"],
	       plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "comDL"], plot=True, file_extensions=["png"]):
	#Bland-Altman plots for ejection fraction (EF)
	(ef_mc_cine, ef_comDL_cine, ef_nnunet_cine) = ef_tuple_cine
	(ef_mc_rt, ef_comDL_rt, ef_nnunet_rt) = ef_tuple_rt
	(ef_mc_rt_stress, ef_comDL_rt_stress, ef_nnunet_rt_stress) = ef_tuple_rt_stress

	xlabel="EF [%]"
	ylabel="manual contours EF - nnU-Net EF [%]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [ef_mc_cine, ef_mc_rt, ef_mc_rt_stress]
	set_nnunet = [ef_nnunet_cine, ef_nnunet_rt, ef_nnunet_rt_stress]
	set_comDL = [ef_comDL_cine, ef_comDL_rt, ef_comDL_rt_stress]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_paths = [os.path.join(save_dir, "BA_nnunet_EF_"+save_str+"."+f) for f in file_extensions]

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setB, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, plot=plot)

	ylabel="manual contours EF - comDL EF [%]"
	save_paths = [os.path.join(save_dir, "BA_comDL_EF_"+save_str+"."+f) for f in file_extensions]
	if "comDL" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setC, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, plot=plot)

def plot_ba_edv(save_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_stress, set_colors=["royalblue", "limegreen", "crimson"],
		plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "comDL"], plot=True, file_extensions=["png"]):
	#Bland-Altman plots for end-diastolic volume (EDV)
	(ed_vol_mc_cine, ed_vol_comDL_cine, ed_vol_nnunet_cine) = ed_tuple_cine
	(ed_vol_mc_rt, ed_vol_comDL_rt, ed_vol_nnunet_rt) = ed_tuple_rt
	(ed_vol_mc_rt_stress, ed_vol_comDL_rt_stress, ed_vol_nnunet_rt_stress) = ed_tuple_rt_stress

	xlabel="LV end-diastolic volume [ml]"
	ylabel="manual contours EDV - nnU-Net EDV [ml]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [ed_vol_mc_cine, ed_vol_mc_rt, ed_vol_mc_rt_stress]
	set_nnunet = [ed_vol_nnunet_cine, ed_vol_nnunet_rt, ed_vol_nnunet_rt_stress]
	set_comDL = [ed_vol_comDL_cine, ed_vol_comDL_rt, ed_vol_comDL_rt_stress]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_paths = [os.path.join(save_dir, "BA_nnunet_EDV_"+save_str+"."+f) for f in file_extensions]

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setB, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=1, plot=plot)

	ylabel="manual contours EDV - comDL EDV [ml]"
	save_paths = [os.path.join(save_dir, "BA_comDL_EDV_"+save_str+"."+f) for f in file_extensions]
	if "comDL" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setC, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=1, plot=plot)

def plot_ba_esv(save_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_stress, set_colors=["royalblue", "limegreen", "crimson"],
		plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "comDL"], plot=True, file_extensions=["png"]):
	#Bland-Altman plots for end-systolic volume (ESV)
	(es_vol_mc_cine, es_vol_comDL_cine, es_vol_nnunet_cine) = es_tuple_cine
	(es_vol_mc_rt, es_vol_comDL_rt, es_vol_nnunet_rt) = es_tuple_rt
	(es_vol_mc_rt_stress, es_vol_comDL_rt_stress, es_vol_nnunet_rt_stress) = es_tuple_rt_stress
	xlabel="LV end-systolic volume [ml]"
	ylabel="manual contours ESV - nnU-Net ESV [ml]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [es_vol_mc_cine, es_vol_mc_rt, es_vol_mc_rt_stress]
	set_nnunet = [es_vol_nnunet_cine, es_vol_nnunet_rt, es_vol_nnunet_rt_stress]
	set_comDL = [es_vol_comDL_cine, es_vol_comDL_rt, es_vol_comDL_rt_stress]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_paths = [os.path.join(save_dir, "BA_nnunet_ESV_"+save_str+"."+f) for f in file_extensions]

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setB, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=1, plot=plot)

	ylabel="manual contours ESV - comDL ESV [ml]"
	save_paths = [os.path.join(save_dir, "BA_comDL_ESV_"+save_str+"."+f) for f in file_extensions]
	if "comDL" in plot_mode:
		assess_utils.plot_bland_altman_multi(setA, setC, save_paths=save_paths, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=1, plot=plot)

def plot_ba_esv_cine_rt(save_dir, es_tuple_cine, es_tuple_rt, set_colors=["royalblue", "limegreen", "crimson"],
		plot_indexes=[0,1], ylim=[], plot_mode=["nnunet", "auto"]):
	#Bland-Altman plots for end-systolic volume (ESV)
	(es_vol_mc_cine, es_vol_auto_cine, es_vol_nnunet_cine) = es_tuple_cine
	(es_vol_mc_rt, es_vol_auto_rt, es_vol_nnunet_rt) = es_tuple_rt
	xlabel="LV end-systolic volume [ml]"
	ylabel="ESV cine - ESV rt [ml]"
	set_labels = ["mc", "nnunet"]
	set_mc = [es_vol_mc_cine, es_vol_mc_cine]
	set_nnunet = [es_vol_mc_rt, es_vol_nnunet_rt]

	save_str = ""
	if [0,1] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += set_labels[i]
	save_path = os.path.join(save_dir, "BA_cine_rt_ESV_"+save_str+".pdf")

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	#submyo.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
	#			figlayout="tight", ylim=ylim, scale=0.001)
	assess_utils.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=0.001)

def write_parameter_files_nnunet_auto(out_dir, rtvol_dict=rtvol,
				contour_dir=contour_files_dir,
				nnunet_output=nnunet_output_dir,
				exp_dir=end_exp_dir):
	"""
	Bland-Altman plots for comparison of manual contours and nnU-Net segmentations with automatic evaluation.
	"""
	assess_utils.identify_ED_ES_mc_nnunet(rtvol_dict, contour_dir=contour_dir, seg_dir=os.path.join(nnunet_output, "rtvol_rt_2d_single_cv/"), exp_dir=exp_dir,
					restrict_slices=False)
	edv_mc = [data["edv_mc"] for data in rtvol_dict]
	esv_mc = [data["esv_mc"] for data in rtvol_dict]
	ef_mc  = [data["ef_mc"]  for data in rtvol_dict]
	edv_nnunet = [data["edv_nnunet"] for data in rtvol_dict]
	esv_nnunet = [data["esv_nnunet"] for data in rtvol_dict]
	ef_nnunet  = [data["ef_nnunet"]  for data in rtvol_dict]

	output_file = os.path.join(out_dir,"cardiac_function_mc_rt.txt")
	if not os.path.isfile(output_file):
		write_output_cardiac_function_parameters(output_file, edv_mc, esv_mc, ef_mc, rtvol_dict=rtvol_dict, precision=1, header="Manually corrected MEDIS DL ACD")

	output_file = os.path.join(out_dir,"cardiac_function_nnunet_auto_rt.txt")
	write_output_cardiac_function_parameters(output_file, edv_nnunet, esv_nnunet, ef_nnunet, rtvol_dict=rtvol_dict, precision=1, header="Automatic evaluation of nnU-Net")

	assess_utils.identify_ED_ES_mc_nnunet(rtvol_dict, contour_dir=contour_dir, seg_dir=os.path.join(nnunet_output, "rtvol_rt_2d_single_cv/"), exp_dir=exp_dir,
					restrict_slices=True)
	edv_nnunet = [data["edv_nnunet"] for data in rtvol_dict]
	esv_nnunet = [data["esv_nnunet"] for data in rtvol_dict]
	ef_nnunet  = [data["ef_nnunet"]  for data in rtvol_dict]

	output_file = os.path.join(out_dir,"cardiac_function_nnunet_semi-auto_rt.txt")
	write_output_cardiac_function_parameters(output_file, edv_nnunet, esv_nnunet, ef_nnunet, rtvol_dict=rtvol_dict, precision=1, header="Semi-automatic evaluation of nnU-Net with manual slice selection")

def plot_BA_nnunet_auto(rtvol_dict, out_dir_fig, plot=False, file_extensions=["pdf"]):
	"""
	Bland-Altman plots for comparison of manual contours and nnU-Net segmentations with automatic evaluation.
	"""
	assess_utils.identify_ED_ES_mc_nnunet(rtvol_dict, restrict_slices=False)
	edv_mc = [data["edv_mc"] for data in rtvol_dict]
	esv_mc = [data["esv_mc"] for data in rtvol_dict]
	ef_mc  = [data["ef_mc"]  for data in rtvol_dict]
	edv_nnunet = [data["edv_nnunet"] for data in rtvol_dict]
	esv_nnunet = [data["esv_nnunet"] for data in rtvol_dict]
	ef_nnunet  = [data["ef_nnunet"]  for data in rtvol_dict]

	xlabel="LV blood volume [ml]"
	ylabel="manual contours - nnU-Net auto [ml]"
	labels = ["EDV"]
	ylim=[-60,20]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_EDV."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([edv_mc], [edv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_paths=save_paths, plot=plot)
	labels = ["ESV"]
	ylim=[-20,12]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_ESV."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([esv_mc], [esv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_paths=save_paths, plot=plot)
	xlabel="LV ejection fraction [%]"
	ylabel="manual contours - nnU-Net auto [%]"
	labels = ["EF"]
	ylim=[-12,12]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_EF."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([ef_mc], [ef_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim,
					save_paths=save_paths, plot=plot)

	assess_utils.identify_ED_ES_mc_nnunet(rtvol_dict, restrict_slices=True)
	edv_mc = [data["edv_mc"] for data in rtvol_dict]
	esv_mc = [data["esv_mc"] for data in rtvol_dict]
	ef_mc  = [data["ef_mc"]  for data in rtvol_dict]
	edv_nnunet = [data["edv_nnunet"] for data in rtvol_dict]
	esv_nnunet = [data["esv_nnunet"] for data in rtvol_dict]
	ef_nnunet  = [data["ef_nnunet"]  for data in rtvol_dict]

	xlabel="LV blood volume [ml]"
	ylabel="manual contours - nnU-Net auto [ml]"
	labels = ["EDV"]
	ylim=[-60,20]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_EDV."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([edv_mc], [edv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_paths=save_paths, plot=plot)
	labels = ["ESV"]
	ylim=[-20,12]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_ESV."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([esv_mc], [esv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_paths=save_paths, plot=plot)
	xlabel="LV ejection fraction [%]"
	ylabel="manual contours - nnU-Net auto [%]"
	labels = ["EF"]
	ylim=[-12,12]
	save_paths = [os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_EF."+f) for f in file_extensions]
	assess_utils.plot_bland_altman_multi([ef_mc], [ef_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim,
					save_paths=save_paths, plot=plot)

def read_clinical_measures(in_file, no_dict=False):
	EDV_dict = []
	ESV_dict = []
	EF_dict = []
	readout = ""
	with open (in_file, 'rt', encoding="utf8", errors='ignore') as myfile:
		for line in myfile:
			content = line.split()
			content = [c.strip() for c in content]
			if "EDV" == content[0].strip():
				readout = "EDV"
				continue
			if "ESV" == content[0].strip():
				readout = "ESV"
				continue
			if "EF" == content[0].strip():
				readout = "EF"
				continue

			if "EDV" == readout:
				EDV_dict.append({"id":content[0], "value":float(content[1])})
			if "ESV" == readout:
				ESV_dict.append({"id":content[0], "value":float(content[1])})
			if "EF" == readout:
				EF_dict.append({"id":content[0], "value":float(content[1])})
	myfile.close()
	if no_dict:
		return [d["value"] for d in EDV_dict], [d["value"] for d in ESV_dict], [d["value"] for d in EF_dict]
	return EDV_dict, ESV_dict, EF_dict

def print_abs_rel_diff(file_ref, file_comp, precision=1):
	edv_ref, esv_ref, ef_ref = read_clinical_measures(file_ref)
	edv_comp, esv_comp, ef_comp = read_clinical_measures(file_comp)
	ids = []
	for i in edv_comp:
		ids.append(i["id"])
	edv_a = [d["value"] for d in edv_ref if d["id"] in ids]
	esv_a = [d["value"] for d in esv_ref if d["id"] in ids]
	ef_a = [d["value"] for d in ef_ref if d["id"] in ids]
	edv_b = [d["value"] for d in edv_comp]
	esv_b = [d["value"] for d in esv_comp]
	ef_b = [d["value"] for d in ef_comp]
	mdiff_abs_edv, stdv_abs_edv, _, _ = calc_mean_stdv_two_sets(edv_a, edv_b, precision=precision)
	mdiff_abs_esv, stdv_abs_esv, _, _ = calc_mean_stdv_two_sets(esv_a, esv_b, precision=precision)
	mdiff_rel_edv, stdv_rel_edv,_,_ = calc_mean_stdv_two_sets(edv_a, edv_b, mode="relative", precision=precision, scale=100)
	mdiff_rel_esv, stdv_rel_esv,_,_ = calc_mean_stdv_two_sets(esv_a, esv_b, mode="relative", precision=precision, scale=100)
	#ejection fraction
	mdiff_abs_ef, stdv_abs_ef, _, _ = calc_mean_stdv_two_sets(ef_a, ef_b, precision=precision)
	mdiff_rel_ef, stdv_rel_ef, _, _ = calc_mean_stdv_two_sets(ef_a, ef_b, mode="relative", precision=precision, scale=100)
	print("Absolute mean differences")
	print("EDV", mdiff_abs_edv, stdv_abs_edv)
	print("ESV", mdiff_abs_esv, stdv_abs_esv)
	print("EF", mdiff_abs_ef, stdv_abs_ef)
	print("Relative mean differences")
	print("EDV", mdiff_rel_edv, stdv_rel_edv)
	print("ESV", mdiff_rel_esv, stdv_rel_esv)
	print("EF", mdiff_rel_ef, stdv_rel_ef)

def write_cardiac_function_single(out_dir, rtvol_dict, edv_tuple, esv_tuple, ef_tuple, modus="cine"):
	(edv_mc, edv_comDL, edv_nnunet) = edv_tuple
	(esv_mc, esv_comDL, esv_nnunet) = esv_tuple
	(ef_mc, ef_comDL, ef_nnunet) = ef_tuple
	print(modus)
	output_file = os.path.join(out_dir,"cardiac_function_mc_"+modus+".txt")
	write_output_cardiac_function_parameters(output_file, edv_mc, esv_mc, ef_mc, rtvol_dict=rtvol_dict, precision=1, header="Manually corrected MEDIS DL ACD")
	output_file = os.path.join(out_dir,"cardiac_function_comDL_"+modus+".txt")
	write_output_cardiac_function_parameters(output_file, edv_comDL, esv_comDL, ef_comDL, rtvol_dict=rtvol_dict, precision=1, header="MEDIS DL ACD")
	output_file = os.path.join(out_dir,"cardiac_function_nnunet_"+modus+".txt")
	write_output_cardiac_function_parameters(output_file, edv_nnunet, esv_nnunet, ef_nnunet, rtvol_dict=rtvol_dict, precision=1, header="nnU-Net")

def save_fig1(out_dir, plot=False, img_dir=scanner_reco_dir,
			contour_dir=contour_files_dir,
			seg_dir=nnunet_output_dir,
			file_extension="png"):
	"""
	Parameters for plotting the measurement types in the manuscript.
	"""
	file_extensions=file_extension.split(",")
	vol = "vol12"
	reverse = True
	slice_idx=13
	vmax_factor=0.8
	crop_dim=80
	mask_mode = []
	phase_mode = "ed"
	save_paths = [os.path.join(out_dir, "figure_01_a."+f) for f in file_extensions]
	titles = ["cine", "real-time (76 bpm)", "real-time (115 bpm)", "real-time (162 bpm)"]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)

	mask_mode = ["mc"]
	phase_mode = "ed"
	save_paths = [os.path.join(out_dir, "figure_01_b."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)
	mask_mode = []
	phase_mode = "es"
	save_paths = [os.path.join(out_dir, "figure_01_c."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)
	mask_mode = ["mc"]
	phase_mode = "es"
	save_paths = [os.path.join(out_dir, "figure_01_d."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)

def save_fig2(out_dir, rtvol_dict=rtvol, plot=False, contour_dir=contour_files_dir, seg_dir=nnunet_output_dir, file_extension="png"):
	"""
	Create figure for Dice's coefficient depending on heart rate
	"""

	file_extensions=file_extension.split(",")
	ylim = [0.2,1]

	# nnU-Net
	calc_DC_and_bpm(rtvol_dict, mode=["nnunet"], contour_dir = contour_dir, seg_dir = seg_dir)
	contour_mode = "nnunet"
	save_paths = [os.path.join(out_dir, "DC_vs_bpm_"+contour_mode+"."+f) for f in file_extensions]
	title="nnU-Net"
	plot_DC_vs_bpm(rtvol_dict, save_paths=save_paths, contour_mode=contour_mode, ylim=ylim, plot=plot, title=title)

	# comDL
	calc_DC_and_bpm(rtvol_dict, mode=["comDL"], contour_dir = contour_dir, seg_dir = seg_dir)
	#contour_mode = "comDL"
	#save_paths = [os.path.join(out_dir, "DC_vs_bpm_"+contour_mode+"."+f) for f in file_extensions]
	#title="comDL"
	#plot_DC_vs_bpm(rtvol_dict, save_paths=save_paths, contour_mode=contour_mode, ylim=ylim, plot=plot, title=title)

def save_fig3(out_dir, rtvol_dict=rtvol, plot=False, img_dir=scanner_reco_dir,
			contour_dir=contour_files_dir,
			seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_stress_2d_single_cv"),
			file_extension="png"):
	"""
	Create figure for visualizing the limits of neural networks for manuscript.
	"""
	file_extensions=file_extension.split(",")
	crop_dim=[[50,110,30,90],[50,110,30,90],[40,120,40,120], [25,105,15,95]]  #[30,110,50,130]

	save_paths = [os.path.join(out_dir, "figure_03_nn_limits."+f) for f in file_extensions]
	param_list = [['vol10', 2, 25], ['vol10', 3, 123], ['vol11', 15, 69], ['vol15', 11, 126]]
	assess_utils.plot_mc_nnunet(contour_dir, img_dir, seg_dir, rtvol_dict, param_list, flag3d=False, mode = "nnunet",
				crop_dim=crop_dim, contour_suffix = "_rt_stress_manual"+contour_format, img_suffix="rt_stress_scanner", save_paths=save_paths,
				check=False, plot=plot)

def save_figdc(out_dir, plot=False, img_dir=scanner_reco_dir,
			contour_dir=contour_files_dir,
			seg_dir=nnunet_output_dir,
			file_extension="png"):
	"""
	Create a figure depicting an example for Dice's coefficients for comDL and nnU-Net segmentations.
	"""
	file_extensions=file_extension.split(",")
	vol = "vol12"
	reverse = True
	slice_idx=13
	vmax_factor=0.8
	crop_dim=80
	phase_mode = "es"
	mask_mode = []
	titles = ["cine", "real-time (76 bpm)", "real-time (115 bpm)", "real-time (162 bpm)"]
	save_paths = [os.path.join(out_dir, "figure_dc_a."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_stress_2d_single_cv"), crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)
	mask_mode = ["mc"]
	save_paths = [os.path.join(out_dir, "figure_dc_b."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir, DC=False,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)
	mask_mode = ["comDL"]
	save_paths = [os.path.join(out_dir, "figure_dc_c."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir, DC=True,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)
	mask_mode = ["nnunet"]
	save_paths = [os.path.join(out_dir, "figure_dc_d."+f) for f in file_extensions]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_paths=save_paths,
				contour_dir=contour_dir,
				img_dir =img_dir, DC=True,
				seg_dir=seg_dir, crop_dim=crop_dim, vmax_factor=vmax_factor, titles=titles, plot=plot)

def save_figba(out_dir, rtvol_dict=rtvol, param_dir="", nnunet_output=nnunet_output_dir, file_extension="png"):
	"""
	Bland-Altman plots of EDV, ESV and EF for entries of rtvol for cine, real-time and real-time stress.
	"""
	file_extensions=file_extension.split(",")
	if "" == param_dir:
		param_dir = out_dir
		write_cardiac_function_all(out_dir, rtvol_dict=rtvol_dict, nnunet_output=nnunet_output)

	modus = "cine"
	file_mc = os.path.join(param_dir,"cardiac_function_mc_"+modus+".txt")
	file_comDL = os.path.join(param_dir,"cardiac_function_comDL_"+modus+".txt")
	file_nnunet = os.path.join(param_dir,"cardiac_function_nnunet_"+modus+".txt")
	edv_tuple_cine, esv_tuple_cine, ef_tuple_cine = read_cardiac_function_parameters(file_mc, file_comDL, file_nnunet)

	modus = "rt"
	file_mc = os.path.join(param_dir,"cardiac_function_mc_"+modus+".txt")
	file_comDL = os.path.join(param_dir,"cardiac_function_comDL_"+modus+".txt")
	file_nnunet = os.path.join(param_dir,"cardiac_function_nnunet_"+modus+".txt")
	edv_tuple_rt, esv_tuple_rt, ef_tuple_rt = read_cardiac_function_parameters(file_mc, file_comDL, file_nnunet)

	modus = "rt_stress"
	file_mc = os.path.join(param_dir,"cardiac_function_mc_"+modus+".txt")
	file_comDL = os.path.join(param_dir,"cardiac_function_comDL_"+modus+".txt")
	file_nnunet = os.path.join(param_dir,"cardiac_function_nnunet_"+modus+".txt")
	edv_tuple_rt_stress, esv_tuple_rt_stress, ef_tuple_rt_stress = read_cardiac_function_parameters(file_mc, file_comDL, file_nnunet)

	for i in range(3):
		plot_ba_ef(out_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=[-20,20], plot_mode=["nnunet"], file_extensions=file_extensions)
		plot_ba_edv(out_dir, edv_tuple_cine, edv_tuple_rt, edv_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=[-80,80], plot_mode=["nnunet"], file_extensions=file_extensions)
		plot_ba_esv(out_dir, esv_tuple_cine, esv_tuple_rt, esv_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=[-18,18], plot_mode=["nnunet"], file_extensions=file_extensions)

	ef_tuple_rt_stress = [[x for i, x in enumerate(a) if i!= 2] for a in ef_tuple_rt_stress]
	ylims_ef = [[-25,25], [-25,25], [-50,50]]
	ylims_edv= [[-25,25], [-25,25], [-100,100]]
	ylims_esv= [[-25,25], [-25,25], [-25,25]]
	for i in range(3):
		plot_ba_ef(out_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=ylims_ef[i], plot_mode=["comDL"], file_extensions=file_extensions)
		plot_ba_edv(out_dir, edv_tuple_cine, edv_tuple_rt, edv_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=ylims_edv[i], plot_mode=["comDL"], file_extensions=file_extensions)
		plot_ba_esv(out_dir, esv_tuple_cine, esv_tuple_rt, esv_tuple_rt_stress, plot=False,
			plot_indexes=[i], ylim=ylims_esv[i], plot_mode=["comDL"], file_extensions=file_extensions)

def write_cardiac_function_all(out_dir, rtvol_dict=rtvol, contour_dir=contour_files_dir, exp_dir=end_exp_dir,
				nnunet_output=nnunet_output_dir, contour_suffix=contour_format):
	"""
	Write cardiac function parameters EDV, ESV and EF into files for manually corrected, comDL and nnU-Net contours.
	"""
	# cine CMR
	seg_dir = os.path.join(nnunet_output, "rtvol_cine_2d_single_cv/")
	ed_tuple_cine, es_tuple_cine, ef_tuple_cine = calc_mean_stdv_parameters_cine(rtvol_dict, contour_dir=contour_dir, seg_dir=seg_dir,
									flag3d=False, slice_selection=False)
	write_cardiac_function_single(out_dir, rtvol_dict, ed_tuple_cine, es_tuple_cine, ef_tuple_cine, modus="cine")

	# real-time CMR at rest
	seg_dir = os.path.join(nnunet_output, "rtvol_rt_2d_single_cv/")
	ed_tuple_rt, es_tuple_rt, ef_tuple_rt = calc_mean_stdv_parameters_rt(rtvol_dict, contour_dir=contour_dir, seg_dir=seg_dir, exp_dir=exp_dir,
									session_mc_suffix="_rt_manual"+contour_suffix)
	write_cardiac_function_single(out_dir, rtvol_dict, ed_tuple_rt, es_tuple_rt, ef_tuple_rt, modus="rt")

	# real-time CMR at stress
	seg_dir = os.path.join(nnunet_output, "rtvol_rt_stress_2d_single_cv/")
	ed_tuple_rt_stress, es_tuple_rt_stress, ef_tuple_rt_stress = calc_mean_stdv_parameters_rt(rtvol_dict, contour_dir=contour_dir, seg_dir=seg_dir,
							session_mc_suffix="_rt_stress_manual"+contour_suffix, session_comDL_suffix="_rt_stress_comDL"+contour_suffix,
							exp_dir = exp_dir, ed_es_phase_file="_rt_stress.txt")

	write_cardiac_function_single(out_dir, rtvol_dict, ed_tuple_rt_stress, es_tuple_rt_stress, ef_tuple_rt_stress, modus="rt_stress")

def main(out_dir_fig, out_dir_data, plot=False, nnunet_output="/scratch/mschi/nnUnet/"):
	"""
	Create figures used for the abstract and digital poster 'Myocardial T1 Mapping Using Single-Shot Inversion-Recovery
	Radial FLASH: Comparison of Subspace and Nonlinear Model-Based Reconstruction' for ISMRM 2023.

	:param str out_dir: Output directory for figures. Directory is created, if it does not exist.
	:param str data_dir: Directory containing data.
	:param bool plot: Flag for plotting figures or just saving them in figures. Default: False
	:param str suffix: Suffix/File extension for saving figures. Default: '.pdf'
	"""
	# Measurement plot
	save_fig1(out_dir_fig, plot=plot)

	# DC vs bpm
	save_fig2(out_dir_fig, plot=plot)

	# Limits of segmentation network
	save_fig3(out_dir_fig, plot=plot, seg_dir=os.path.join(nnunet_output, "output/rtvol_rt_stress_2d_single_cv"))

	#Bland-Altman Plot
	write_parameter_files_nnunet_auto(out_dir_data, rtvol)
	plot_BA_nnunet_auto(rtvol, out_dir_fig, plot=plot)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Script to create figures used in the abstract and digital poster 'Myocardial T1 Mapping Using Single-Shot Inversion-Recovery Radial FLASH: Comparison of Subspace and Nonlinear Model-Based Reconstruction' for ISMRM 2023.")
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	required.add_argument('figure_dir', type=str, help="Output directory for figures")
	required.add_argument('data_dir', type=str, help="Output directory for data")
	optional.add_argument('-p', '--plot', action='store_true', default=False, help='Flag for showing plots. Default: False')
	parser._action_groups.append(optional)
	args = parser.parse_args()
	main(args.figure_dir, args.data_dir, args.plot)