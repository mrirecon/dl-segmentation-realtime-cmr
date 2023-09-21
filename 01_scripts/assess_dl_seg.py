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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(getsourcefile(lambda:0)))), "python"))

import assess_dl_seg_utils as assess_utils
import cfl

scanner_reco_dir=os.environ["IMG_DIR"] if "IMG_DIR" in os.environ else ""
contour_files_dir=os.environ["CONTOUR_DIR"] if "CONTOUR_DIR" in os.environ else ""
nnunet_output_dir=os.environ["NNUNET_OUTPUT"] if "NNUNET_OUTPUT" in os.environ else ""
end_exp_dir=os.environ["END_EXP"] if "END_EXP" in os.environ else ""

rtvol = [
{"id":"vol60",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":61},
{"id":"vol61",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":55},
{"id":"vol62",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":64},
{"id":"vol64",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":50},
{"id":"vol67",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":53},
{"id":"vol69",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":67},
{"id":"vol70",	"reverse":True	, "flip_rot":[-1,1], "gender":"f", "age":66},
{"id":"vol71",	"reverse":True	, "flip_rot":[-1,0], "gender":"m", "age":65},
{"id":"vol72",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":50},
{"id":"vol78",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":64},
{"id":"vol79",	"reverse":True	, "flip_rot":[-1,1], "gender":"m", "age":44},
{"id":"vol80",	"reverse":True	, "flip_rot":[-1,1], "gender":"f", "age":56},
{"id":"vol82",	"reverse":True	, "flip_rot":[-1,0], "gender":"m", "age":42},
{"id":"vol83",	"reverse":False	, "flip_rot":[-1,0], "gender":"f", "age":45},
{"id":"vol84",	"reverse":False	, "flip_rot":[-1,0], "gender":"m", "age":49}
]

def dice_coeff(rtvol_dict, rtvol_dict_maxstress, mode="single",
		contour_dir=contour_files_dir):
	"""
	Calculate Dice's coefficient for entries in list of dictionaries for
	Medis DL ACD (automatic) and nnU-Net contours for cine and real-time
	measurements at rest, under stress and under max stress.
	As some measurements are not evaluated at max stress, the list of dictionaries
	is different.

	:param list rtvol_dict: List of dictionaries with entries ['id'] and ['reverse']
	:param list rtvol_dict_maxstress: List of dictionaries for maximal stress
	"""

	# cine
	if "single" == mode:
		flag3D = False
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_cine_2d_single_cv")
	else:
		flag3D = True
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_cine_2d_slice_cv/")

	manual_contour_suffix = "_cine_manual.con"
	comp_contour_suffix = "_cine_automatic.con"

	phase_select = "combined"

	print("DC for cine")
	#automatic contours
	ed_dc_list, es_dc_list, ed_dict_list, es_dict_list = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix=comp_contour_suffix, nnunet_prefix="", phase_select=phase_select,
			title="automatic contours", flag3d=True, mode="cine")
	#nnU-Net contours
	nnunet_prefix = os.path.join(seg_dir, "rtvol_")
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix="", nnunet_prefix=nnunet_prefix, phase_select=phase_select,
			title="nnUNet", flag3d=flag3D, mode="cine")

	#rt
	print("DC for real-time rest")
	if "single" == mode:
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_2d_single_cv/")
	else:
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_2d_cv/")

	nnunet_prefix = os.path.join(seg_dir, "rtvol_")
	manual_contour_suffix = "_rt_manual.con"
	comp_contour_suffix = "_rt_automatic.con"
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix=comp_contour_suffix, nnunet_prefix="", phase_select=phase_select,
			title="automatic contours", flag3d=False)
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix="", nnunet_prefix=nnunet_prefix, phase_select=phase_select,
			title="nnUNet", flag3d=False)

	print("DC for real-time stress")
	if "single" == mode:
		flag3D = False
		os.path.join(nnunet_output_dir, "rtvol_rt_Belastung_2d_single_cv/")
	else:
		flag3D = True
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_Belastung_2d_cv/")

	nnunet_prefix = os.path.join(seg_dir, "rtvol_")
	manual_contour_suffix = "_rt_Belastung_manual.con"
	comp_contour_suffix = "_rt_Belastung_automatic.con"
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix=comp_contour_suffix, nnunet_prefix="", phase_select=phase_select,
			title="automatic contours", flag3d=False)
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict, manual_contour_suffix, comp_contour_suffix="", nnunet_prefix=nnunet_prefix, phase_select=phase_select,
			title="nnUNet", flag3d=False)

	print("DC for real-time max stress")

	if "single" == mode:
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_Ausbelastung_2d_single_cv/")
	else:
		seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_Ausbelastung_2d_cv/")

	nnunet_prefix = os.path.join(seg_dir, "rtvol_")
	manual_contour_suffix = "_rt_Ausbelastung_manual.con"
	comp_contour_suffix = "_rt_Ausbelastung_automatic.con"
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict_maxstress, manual_contour_suffix, comp_contour_suffix=comp_contour_suffix, nnunet_prefix="", phase_select=phase_select,
			title="automatic contours", flag3d=False)
	_,_,_,_ = assess_utils.get_ed_es_dice_from_session_multi(contour_dir, rtvol_dict_maxstress, manual_contour_suffix, comp_contour_suffix="", nnunet_prefix=nnunet_prefix, phase_select=phase_select,
			title="nnUNet", flag3d=False)

def calc_DC_and_bpm(rtvol_dict, mode=["nnunet"],
		contour_dir = contour_files_dir,
		seg_dir = nnunet_output_dir):
	"""
	Calculate Dice's coefficient (DC) and beats per minute (bpm) for a list of dictionaries
	and add this information to the dictionary entries for Medis DL ACD and/or nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with entries ['id'] and ['reverse']
	:param list mode: List of modes for calculation of Dice's coefficient. Entries can be 'nnunet' and/or 'auto'
	:param str contour_dir: Path to directory containing contour files in '.con' file format
	:param str seg_dir: Directory containing subdirectories for nnU-Net segmentation outputs
	"""
	segm_classes = 3
	phase_select = "combined"
	seg_subdirs = ["rtvol_rt_2d_single_cv", "rtvol_rt_Belastung_2d_single_cv", "rtvol_rt_Ausbelastung_2d_single_cv"]
	modes = ["rt", "rt_Belastung", "rt_Ausbelastung"]
	manual_contour_suffixes = ["_rt_manual.con", "_rt_Belastung_manual.con", "_rt_Ausbelastung_manual.con"]
	automatic_contour_suffixes = ["_rt_automatic.con", "_rt_Belastung_automatic.con", "_rt_Ausbelastung_automatic.con"]
	descr = ["RV", "Myo", "LV"]

	for d in rtvol_dict:
		vol = d["id"]
		reverse = d["reverse"]
		print(vol)
		sessions = [os.path.join(contour_dir, vol+"_" + s+"_manual.con") for s in modes]
		for i,contour_file in enumerate(sessions):
			if os.path.isfile(contour_file):
				bpm, bpm_std = assess_utils.calc_bpm(contour_file, reverse, time_res=0.03328, centr_slices=5)
				#print(contour_file, round(bpm,0))
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

			if "auto" in mode:
				class_acc = [[] for _ in range(segm_classes)]
				segm_input = os.path.join(contour_dir, vol+automatic_contour_suffixes[i])
				ed_dc, ed_dict, es_dc, es_dict = assess_utils.get_ed_es_dice_from_session_rt(contour_file, reverse, segm_input, phase_select=phase_select)
				for e in ed_dict:
					for j in range(segm_classes):
						class_acc[j].append(e['class'+str(j+1)])
				for j,tag in enumerate(descr):
					d['DC'+"auto"+m+tag] = sum(class_acc[j]) / len(class_acc[j])
					d['DCstd'+"auto"+m+tag] = np.std(class_acc[j])

def plot_DC_vs_bpm(rtvol_dict, save_path="", contour_mode="nnunet", ylim=[], plot=True, mode="noerror", title=""):
	"""
	Plot Dice's coefficient of a list of input dictionaries.

	:param list rtvol_dict: List of dictionaries
	:param str save_path: Path to save plot. Default: Output is not saved
	:param str contour_mode: Contour mode for plotting. Either 'nnunet' or 'auto'
	"""
	modes = ["rt", "rt_Belastung", "rt_Ausbelastung"]
	descr = ["RV", "Myo", "LV"]
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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
			label=desc, markersize=12, ecolor=colors[i])

	else:
			for i, desc in enumerate(descr):
				plt.scatter(heartrates, dice_scores[i], label=desc, s=12)

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
	if "" != save_path:
		if ".png" in save_path:
			plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
		else:
			plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

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

def write_output_stdv_parameters(output_file, ed_tuple, es_tuple, ef_tuple, scale = 0.001, precision=3):
	"""
	Write output mean and standard deviation parameters.

	:param str output_file: File path to output text file
	:param tuple ed_tuple: Tuple of lists of LV volume for ED phase for manual contours, automatic contours and nnU-Net contours
	:param tuple es_tuple: Tuple of lists of LV volume for ES phase for manual contours, automatic contours and nnU-Net contours
	:param tuple ef_tuple: Tuple of lists of ejection fraction for manual contours, automatic contours and nnU-Net contours
	:param float scale: Scale for parameters
	:param int precision: Precision for output data. Default: 3
	"""
	(ed_vol_mc, ed_vol_auto, ed_vol_nnunet) = ed_tuple
	(es_vol_mc, es_vol_auto, es_vol_nnunet) = es_tuple
	(ef_mc, ef_auto, ef_nnunet) = ef_tuple
	mean_diff_auto_ed, stdv_diff_auto_ed, mean_auto_ed, stdv_mean_auto_ed = calc_mean_stdv_two_sets(ed_vol_mc, ed_vol_auto, scale=scale, precision=precision)
	mean_diff_auto_es, stdv_diff_auto_es, mean_auto_es, stdv_mean_auto_es = calc_mean_stdv_two_sets(es_vol_mc, es_vol_auto, scale=scale, precision=precision)
	mean_diff_nnunet_ed, stdv_diff_nnunet_ed, mean_nnunet_ed, stdv_mean_nnunet_ed = calc_mean_stdv_two_sets(ed_vol_mc, ed_vol_nnunet, scale=scale, precision=precision)
	mean_diff_nnunet_es, stdv_diff_nnunet_es, mean_nnunet_es, stdv_mean_nnunet_es = calc_mean_stdv_two_sets(es_vol_mc, es_vol_nnunet, scale=scale, precision=precision)
	mean_rel_auto_ed, stdv_rel_auto_ed,_,_ = calc_mean_stdv_two_sets(ed_vol_mc, ed_vol_auto, mode="relative", precision=precision, scale=100)
	mean_rel_auto_es, stdv_rel_auto_es,_,_ = calc_mean_stdv_two_sets(es_vol_mc, es_vol_auto, mode="relative", precision=precision, scale=100)
	mean_rel_nnunet_ed, stdv_rel_nnunet_ed,_,_ = calc_mean_stdv_two_sets(ed_vol_mc, ed_vol_nnunet, mode="relative", precision=precision, scale=100)
	mean_rel_nnunet_es, stdv_rel_nnunet_es,_,_ = calc_mean_stdv_two_sets(es_vol_mc, es_vol_nnunet, mode="relative", precision=precision, scale=100)
	#ejection fraction
	mean_diff_auto_ef, stdv_diff_auto_ef, mean_auto_ef, stdv_mean_auto_ef = calc_mean_stdv_two_sets(ef_mc, ef_auto, scale=1, precision=precision)
	mean_diff_nnunet_ef, stdv_diff_nnunet_ef, mean_nnunet_ef, stdv_mean_nnunet_ef = calc_mean_stdv_two_sets(ef_mc, ef_nnunet, scale=1, precision=precision)
	mean_rel_auto_ef, stdv_rel_auto_ef, _,_ = calc_mean_stdv_two_sets(ef_mc, ef_auto, mode="relative", precision=precision, scale=100)
	mean_rel_nnunet_ef, stdv_rel_nnunet_ef, _,_ = calc_mean_stdv_two_sets(ef_mc, ef_nnunet, mode="relative", precision=precision, scale=100)
	if "" != output_file:
		with open (output_file, 'w', encoding="utf8", errors='ignore') as output:
			output.write("MEDIS DL ACD\n")
			output.write("mean LVED[ml]:\t"+str(mean_auto_ed)+"\n")
			output.write("stdv LVED[ml]:\t"+str(stdv_mean_auto_ed)+"\n")
			output.write("mean LVES[ml]:\t"+str(mean_auto_es)+"\n")
			output.write("stdv LVES[ml]:\t"+str(stdv_mean_auto_es)+"\n")

			output.write("mean diff LVED[ml]:\t"+str(mean_diff_auto_ed)+"\n")
			output.write("stdv diff LVED[ml]:\t"+str(stdv_diff_auto_ed)+"\n")
			output.write("mean diff LVES[ml]:\t"+str(mean_diff_auto_es)+"\n")
			output.write("stdv diff LVES[ml]:\t"+str(stdv_diff_auto_es)+"\n")

			output.write("mean LVED rel[%]:\t"+str(mean_rel_auto_ed)+"\n")
			output.write("stdv LVED rel[%]:\t"+str(stdv_rel_auto_ed)+"\n")
			output.write("mean LVES rel[%]:\t"+str(mean_rel_auto_es)+"\n")
			output.write("stdv LVES rel[%]:\t"+str(stdv_rel_auto_es)+"\n")

			output.write("mean EF[%]:\t"+str(mean_auto_ef)+"\n")
			output.write("stdv EF[%]:\t"+str(stdv_mean_auto_ef)+"\n")
			output.write("mean diff EF[%]:\t"+str(mean_diff_auto_ef)+"\n")
			output.write("stdv diff EF[%]:\t"+str(stdv_diff_auto_ef)+"\n")
			output.write("mean EF rel[%]:\t"+str(mean_rel_auto_ef)+"\n")
			output.write("stdv EF rel[%]:\t"+str(stdv_rel_auto_ef)+"\n")
			output.write("\n")

			output.write("nnU-Net\n")
			output.write("mean LVED[ml]:\t"+str(mean_nnunet_ed)+"\n")
			output.write("stdv LVED[ml]:\t"+str(stdv_mean_nnunet_ed)+"\n")
			output.write("mean LVES[ml]:\t"+str(mean_nnunet_es)+"\n")
			output.write("stdv LVES[ml]:\t"+str(stdv_mean_nnunet_es)+"\n")

			output.write("mean diff LVED[ml]:\t"+str(mean_diff_nnunet_ed)+"\n")
			output.write("stdv diff LVED[ml]:\t"+str(stdv_diff_nnunet_ed)+"\n")
			output.write("mean diff LVES[ml]:\t"+str(mean_diff_nnunet_es)+"\n")
			output.write("stdv diff LVES[ml]:\t"+str(stdv_diff_nnunet_es)+"\n")

			output.write("mean LVED rel[%]:\t"+str(mean_rel_nnunet_ed)+"\n")
			output.write("stdv LVED rel[%]:\t"+str(stdv_rel_nnunet_ed)+"\n")
			output.write("mean LVES rel[%]:\t"+str(mean_rel_nnunet_es)+"\n")
			output.write("stdv LVES rel[%]:\t"+str(stdv_rel_nnunet_es)+"\n")

			output.write("mean EF[%]:\t"+str(mean_nnunet_ef)+"\n")
			output.write("stdv EF[%]:\t"+str(stdv_mean_nnunet_ef)+"\n")
			output.write("mean diff EF[%]:\t"+str(mean_diff_nnunet_ef)+"\n")
			output.write("stdv diff EF[%]:\t"+str(stdv_diff_nnunet_ef)+"\n")
			output.write("mean EF rel[%]:\t"+str(mean_rel_nnunet_ef)+"\n")
			output.write("stdv EF rel[%]:\t"+str(stdv_rel_nnunet_ef)+"\n")
	output.close()

def calc_mean_stdv_parameters_cine(rtvol_dict, seg_dir = os.path.join(nnunet_output_dir, "rtvol_cine_2d_single_cv/"),
				   contour_dir = contour_files_dir, flag3d=False,
				   pixel_spacing = 1.328125, output_file="", slice_selection=False, precision=1):
	"""
	Calculate mean and standard deviation parameters for cine MRI for manual contours, automatic contours and nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with volunteer id and reverse flag
	:param str seg_dir: Directory containing segmentation of nnU-Net for cine measurements
	:param str contour_dir: Directory containing Medis contour files in format <vol_id>_cine_manual.con and <vol_id>_cine_automatic.con
	:param bool flag3d: Flag for marking input data as 2D or 3D data
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param str output_file: Optional file path to write parameters into output text file
	:param bool slice_selection: Selection of slices for nnU-Net segmentation confined to slices containing the heart
	:param int precision: Precision for output data. Default: 1
	"""
	thickness = 6.6
	ed_vol_mc, es_vol_mc = [], []
	ed_vol_auto, es_vol_auto = [], []
	ed_vol_nnunet, es_vol_nnunet = [], []
	ef_mc, ef_auto, ef_nnunet = [], [], []

	segm_class = 3

	for data in rtvol_dict:
		vol = data["id"]
		reverse = data["reverse"]
		session = os.path.join(contour_dir, vol+"_cine_manual.con")
		slice_offset = 0

		ed_mc, es_mc, ed_vol, es_vol, ed_plist, es_plist = assess_utils.get_ed_es_param_from_session(session, reverse, pixel_spacing, thickness)

		if slice_selection:
			slice_list = list(set([p[0] for p in ed_plist]))
			slice_list.sort()
			slice_offset = slice_list[0]

		ed_vol_mc.append(ed_vol)
		es_vol_mc.append(es_vol)
		ef_mc.append( (ed_vol - es_vol) / ed_vol * 100)

		session = os.path.join(contour_dir, vol+"_cine_automatic.con")
		img_dims, fov, EDphase, ESphase, slices = assess_utils.extract_img_params(session)
		#mask_list, param_list, ccsf = medis.masks_and_parameters_from_file(session, img_dims, no_duplicates=True)
		mask_list, param_list, ccsf = assess_utils.masks_and_parameters_from_file(session, img_dims)
		slice_indexes, mlist, plist = assess_utils.combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
		ed_masks = assess_utils.mask_for_param_selection(mlist, plist, param=ed_plist)
		ed_auto = []
		for m in ed_masks:
			lv_count = np.count_nonzero(m == segm_class)
			ed_auto.append(lv_count * pixel_spacing * pixel_spacing)
		es_masks = assess_utils.mask_for_param_selection(mlist, plist, param=es_plist)
		es_auto = []
		for m in es_masks:
			lv_count = np.count_nonzero(m == segm_class)
			es_auto.append(lv_count * pixel_spacing * pixel_spacing)
		ed_vol = sum(ed_auto) * thickness
		es_vol = sum(es_auto) * thickness
		ed_vol_auto.append(ed_vol)
		es_vol_auto.append(es_vol)
		ef_auto.append( (ed_vol - es_vol) / ed_vol * 100)

		segm_prefix = os.path.join(seg_dir, "rtvol_" + vol[3:])
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

	if "" != output_file:
		write_output_stdv_parameters(output_file, (ed_vol_mc, ed_vol_auto, ed_vol_nnunet), (es_vol_mc, es_vol_auto, es_vol_nnunet),
			       (ef_mc, ef_auto, ef_nnunet), precision=precision)

	return (ed_vol_mc, ed_vol_auto, ed_vol_nnunet), (es_vol_mc, es_vol_auto, es_vol_nnunet), (ef_mc, ef_auto, ef_nnunet)

def calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_2d_single_cv/"),
				contour_dir=contour_files_dir, flag3d=False,
				pixel_spacing=1.6, session_mc_suffix="_rt_manual.con", session_auto_suffix="_rt_automatic.con",
				exp_dir=end_exp_dir, ed_es_phase_file="_rt.txt", output_file="",
				precision=1):
	"""
	Calculate mean and standard deviation parameters for rt MRI for manual, automatic and nnU-Net contours.

	:param list rtvol_dict: List of dictionaries with volunteer id and reverse flag
	:param str seg_dir: Directory containing segmentation of nnU-Net for cine measurements
	:param str contour_dir: Directory containing Medis contour files in format <vol_id>_cine_manual.con and <vol_id>_cine_automatic.con
	:param bool flag3d: Flag for marking input data as 2D or 3D data
	:param float pixel_spacing: Pixel spacing of input in mm, e.g. 1px = 1.6 mm x 1.6 mm --> pixel_spacing = 1.6
	:param str session_mc_suffix: Suffix for manual contour file
	:param str session_auto_suffix: Suffix for automatic contour file
	:param str exp_dir: Directory containing text files with indexes for the end-expiration state
	:param str ed_es_phase_file: Suffix for text file in 'exp_dir' containing ED and ES phase information. Format <exp_dir>/<vol_id><ed_es_phase_file>
	:param str output_file: Optional file path to write parameters into output text file
	:param int precision: Precision for output data. Default: 1
	"""
	thickness = 6.6
	ed_vol_mc, es_vol_mc = [], []
	ed_vol_auto, es_vol_auto = [], []
	ed_vol_nnunet, es_vol_nnunet = [], []
	ef_mc, ef_auto, ef_nnunet = [], [], []

	segm_class = 3
	for data in rtvol_dict:
		vol = data["id"]
		reverse = data["reverse"]
		file_path = os.path.join(exp_dir, vol+ed_es_phase_file)
		ed_plist = assess_utils.get_ed_es_from_text(file_path, "ED")
		es_plist = assess_utils.get_ed_es_from_text(file_path, "ES")

		session = os.path.join(contour_dir, vol+session_mc_suffix)
		img_dims, fov, EDphase, ESphase, slices = assess_utils.extract_img_params(session)
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

		session = os.path.join(contour_dir, vol+session_auto_suffix)
		img_dims, fov, EDphase, ESphase, slices = assess_utils.extract_img_params(session)
		#mask_list, param_list, ccsf = medis.masks_and_parameters_from_file(session, img_dims, no_duplicates=True)
		mask_list, param_list, ccsf = assess_utils.masks_and_parameters_from_file(session, img_dims)
		slice_indexes, mlist, plist = assess_utils.combine_masks_multi(mask_list, param_list, slices, reverse=reverse)
		ed_masks = assess_utils.mask_for_param_selection(mlist, plist, param=ed_plist)
		ed_auto = []
		for m in ed_masks:
			lv_count = np.count_nonzero(m == segm_class)
			ed_auto.append(lv_count * pixel_spacing * pixel_spacing)
		es_masks = assess_utils.mask_for_param_selection(mlist, plist, param=es_plist)
		es_auto = []
		for m in es_masks:
			lv_count = np.count_nonzero(m == segm_class)
			es_auto.append(lv_count * pixel_spacing * pixel_spacing)
		ed_vol = sum(ed_auto) * thickness
		es_vol = sum(es_auto) * thickness
		ed_vol_auto.append(ed_vol)
		es_vol_auto.append(es_vol)
		ef_auto.append( (ed_vol - es_vol) / ed_vol * 100)

		segm_prefix = os.path.join(seg_dir, "rtvol_" + vol[3:])
		#for single nnUNet images, ed_plist is list of parameters, for 3D images, it is only the phase value
		ed_nnet = assess_utils.get_phase_area_from_nnunet(segm_prefix, ed_plist, segm_class = 3, pixel_spacing = pixel_spacing, flag3d=flag3d)
		es_nnet = assess_utils.get_phase_area_from_nnunet(segm_prefix, es_plist, segm_class = 3, pixel_spacing = pixel_spacing, flag3d=flag3d)
		ed_vol = sum(ed_nnet) * thickness
		es_vol = sum(es_nnet) * thickness
		ed_vol_nnunet.append(ed_vol)
		es_vol_nnunet.append(es_vol)
		ef_nnunet.append( (ed_vol - es_vol) / ed_vol * 100)

	if "" != output_file:
		write_output_stdv_parameters(output_file, (ed_vol_mc, ed_vol_auto, ed_vol_nnunet), (es_vol_mc, es_vol_auto, es_vol_nnunet),
				(ef_mc, ef_auto, ef_nnunet), precision=precision)

	return (ed_vol_mc, ed_vol_auto, ed_vol_nnunet), (es_vol_mc, es_vol_auto, es_vol_nnunet), (ef_mc, ef_auto, ef_nnunet)

def plot_ba_ef(save_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_Belastung, set_colors=["royalblue", "palegreen", "indianred"],
	       plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "auto"], plot=True, file_extension=".png"):
	#Bland-Altman plots for ejection fraction (EF)
	(ef_mc_cine, ef_auto_cine, ef_nnunet_cine) = ef_tuple_cine
	(ef_mc_rt, ef_auto_rt, ef_nnunet_rt) = ef_tuple_rt
	(ef_mc_rt_Belastung, ef_auto_rt_Belastung, ef_nnunet_rt_Belastung) = ef_tuple_rt_Belastung

	xlabel="EF [%]"
	ylabel="manual contours EF - nnU-Net EF [%]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [ef_mc_cine, ef_mc_rt, ef_mc_rt_Belastung]
	set_nnunet = [ef_nnunet_cine, ef_nnunet_rt, ef_nnunet_rt_Belastung]
	set_comDL = [ef_auto_cine, ef_auto_rt, ef_auto_rt_Belastung]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_path = os.path.join(save_dir, "BA_nnunet_EF_"+save_str+file_extension)

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim)
		assess_utils.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, plot=plot)

	ylabel="manual contours EF - comDL EF [%]"
	save_path = os.path.join(save_dir, "BA_comDL_EF_"+save_str+file_extension)
	if "auto" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim)
		assess_utils.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, plot=plot)

def plot_ba_edv(save_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_Belastung, set_colors=["royalblue", "palegreen", "indianred"],
		plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "auto"], plot=True, file_extension=".png"):
	#Bland-Altman plots for end-diastolic volume (EDV)
	(ed_vol_mc_cine, ed_vol_auto_cine, ed_vol_nnunet_cine) = ed_tuple_cine
	(ed_vol_mc_rt, ed_vol_auto_rt, ed_vol_nnunet_rt) = ed_tuple_rt
	(ed_vol_mc_rt_Belastung, ed_vol_auto_rt_Belastung, ed_vol_nnunet_rt_Belastung) = ed_tuple_rt_Belastung

	xlabel="LV end-diastolic volume [ml]"
	ylabel="manual contours EDV - nnU-Net EDV [ml]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [ed_vol_mc_cine, ed_vol_mc_rt, ed_vol_mc_rt_Belastung]
	set_nnunet = [ed_vol_nnunet_cine, ed_vol_nnunet_rt, ed_vol_nnunet_rt_Belastung]
	set_comDL = [ed_vol_auto_cine, ed_vol_auto_rt, ed_vol_auto_rt_Belastung]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_path = os.path.join(save_dir, "BA_nnunet_EDV_"+save_str+file_extension)

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim, scale=0.001)
		assess_utils.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=0.001, plot=plot)

	ylabel="manual contours EDV - comDL EDV [ml]"
	save_path = os.path.join(save_dir, "BA_comDL_EDV_"+save_str+file_extension)
	if "auto" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim, scale=0.001)
		assess_utils.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=0.001, plot=plot)

def plot_ba_esv(save_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_Belastung, set_colors=["royalblue", "palegreen", "indianred"],
		plot_indexes=[0,1,2], ylim=[], plot_mode=["nnunet", "auto"], plot=True, file_extension=".png"):
	#Bland-Altman plots for end-systolic volume (ESV)
	(es_vol_mc_cine, es_vol_auto_cine, es_vol_nnunet_cine) = es_tuple_cine
	(es_vol_mc_rt, es_vol_auto_rt, es_vol_nnunet_rt) = es_tuple_rt
	(es_vol_mc_rt_Belastung, es_vol_auto_rt_Belastung, es_vol_nnunet_rt_Belastung) = es_tuple_rt_Belastung
	xlabel="LV end-systolic volume [ml]"
	ylabel="manual contours ESV - nnU-Net ESV [ml]"
	save_labels = ["cine", "rt", "rt_stress"]
	set_labels = ["cine", "rt", "rt stress"]
	set_mc = [es_vol_mc_cine, es_vol_mc_rt, es_vol_mc_rt_Belastung]
	set_nnunet = [es_vol_nnunet_cine, es_vol_nnunet_rt, es_vol_nnunet_rt_Belastung]
	set_comDL = [es_vol_auto_cine, es_vol_auto_rt, es_vol_auto_rt_Belastung]

	save_str = ""
	if [0,1,2] in plot_indexes:
		save_str = "all"
	else:
		for i in plot_indexes:
			save_str += save_labels[i]
	save_path = os.path.join(save_dir, "BA_nnunet_ESV_"+save_str+file_extension)

	setA, setB, setC = [], [], []
	labels, colors = [], []
	for i in plot_indexes:
		setA.append(set_mc[i])
		setB.append(set_nnunet[i])
		setC.append(set_comDL[i])
		labels.append(set_labels[i])
		colors.append(set_colors[i])
	if "nnunet" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim, scale=0.001)
		assess_utils.plot_bland_altman_multi(setA, setB, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=0.001, plot=plot)

	ylabel="manual contours ESV - comDL ESV [ml]"
	save_path = os.path.join(save_dir, "BA_comDL_ESV_"+save_str+file_extension)
	if "auto" in plot_mode:
		#submyo.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
		#		figlayout="tight", ylim=ylim, scale=0.001)
		assess_utils.plot_bland_altman_multi(setA, setC, save_path=save_path, labels=labels, colors=colors, ylabel=ylabel, xlabel=xlabel,
				figlayout="tight", ylim=ylim, scale=0.001, plot=plot)

def plot_ba_esv_cine_rt(save_dir, es_tuple_cine, es_tuple_rt, set_colors=["royalblue", "palegreen", "indianred"],
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

def write_parameter_files_nnunet_auto(rtvol_dict, out_dir_data):
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
	output_file = os.path.join(out_dir_data, "rt_mc_nnunet_auto.txt")
	write_output_stdv_parameters(output_file, (edv_mc, edv_mc, edv_nnunet), (esv_mc, esv_mc, esv_nnunet),
					(ef_mc, ef_mc, ef_nnunet), precision=1)

	assess_utils.identify_ED_ES_mc_nnunet(rtvol_dict, restrict_slices=True)
	edv_mc = [data["edv_mc"] for data in rtvol_dict]
	esv_mc = [data["esv_mc"] for data in rtvol_dict]
	ef_mc  = [data["ef_mc"]  for data in rtvol_dict]
	edv_nnunet = [data["edv_nnunet"] for data in rtvol_dict]
	esv_nnunet = [data["esv_nnunet"] for data in rtvol_dict]
	ef_nnunet  = [data["ef_nnunet"]  for data in rtvol_dict]
	output_file = os.path.join(out_dir_data, "rt_mc_nnunet_auto_restricted_slices.txt")
	write_output_stdv_parameters(output_file, (edv_mc, edv_mc, edv_nnunet), (esv_mc, esv_mc, esv_nnunet),
					(ef_mc, ef_mc, ef_nnunet), precision=1)

def plot_BA_nnunet_auto(rtvol_dict, out_dir_fig, plot=False):
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
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_EDV.pdf")
	assess_utils.plot_bland_altman_multi([edv_mc], [edv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_path=save_path, plot=plot)
	labels = ["ESV"]
	ylim=[-20,12]
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_ESV.pdf")
	assess_utils.plot_bland_altman_multi([esv_mc], [esv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_path=save_path, plot=plot)
	xlabel="LV ejection fraction [%]"
	ylabel="manual contours - nnU-Net auto [%]"
	labels = ["EF"]
	ylim=[-12,12]
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_EF.pdf")
	assess_utils.plot_bland_altman_multi([ef_mc], [ef_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim,
					save_path=save_path, plot=plot)

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
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_EDV.pdf")
	assess_utils.plot_bland_altman_multi([edv_mc], [edv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_path=save_path, plot=plot)
	labels = ["ESV"]
	ylim=[-20,12]
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_ESV.pdf")
	assess_utils.plot_bland_altman_multi([esv_mc], [esv_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim, scale=0.001,
					save_path=save_path, plot=plot)
	xlabel="LV ejection fraction [%]"
	ylabel="manual contours - nnU-Net auto [%]"
	labels = ["EF"]
	ylim=[-12,12]
	save_path = os.path.join(out_dir_fig, "BA_nnunet_auto_restricted_slices_EF.pdf")
	assess_utils.plot_bland_altman_multi([ef_mc], [ef_nnunet], labels=labels, ylabel=ylabel, xlabel=xlabel, ylim=ylim,
					save_path=save_path, plot=plot)

def calc_error_propagation(values, values_std, precision=3):
	sum_error = 0
	for e in values_std:
		sum_error += math.pow(e, 2)
	print("Mean", round(sum(values) / len(values), precision))
	print("Propagated error", round(math.sqrt(sum_error) / len(values), precision))

def plot_BA_cine_rt_stress(rtvol_dict):
	"""
	Bland-Altman plots of EDV, ESV and EF for entries of rtvol for cine, real-time and real-time stress.
	"""
	seg_dir = os.path.join(nnunet_output_dir, "rtvol_cine_2d_single_cv/")
	output_file = ""
	ed_tuple_cine, es_tuple_cine, ef_tuple_cine = calc_mean_stdv_parameters_cine(rtvol_dict, seg_dir=seg_dir, output_file=output_file,
									flag3d=False, slice_selection=False)

	seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_2d_single_cv/")
	ed_tuple_rt, es_tuple_rt, ef_tuple_rt = calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=seg_dir, output_file=output_file)

	seg_dir = os.path.join(nnunet_output_dir, "rtvol_rt_Belastung_2d_single_cv/")
	ed_tuple_rt_Belastung, es_tuple_rt_Belastung, ef_tuple_rt_Belastung = calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=seg_dir,
							session_mc_suffix="_rt_Belastung_manual.con", session_auto_suffix="_rt_Belastung_automatic.con",
				exp_dir = end_exp_dir, ed_es_phase_file="_rt_Belastung.txt", output_file=output_file)

	save_dir = "/home/mschi/Pictures/assess_DL/"
	for i in range(3):
		plot_ba_ef(save_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_Belastung, plot_indexes=[i], ylim=[-20,20], plot_mode=["nnunet"])
		plot_ba_edv(save_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_Belastung, plot_indexes=[i], ylim=[-80,80], plot_mode=["nnunet"])
		plot_ba_esv(save_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_Belastung, plot_indexes=[i], ylim=[-18,18], plot_mode=["nnunet"])
	#for i in range(3):
	#	plot_ba_ef(save_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_Belastung, plot_indexes=[i], ylim=[-25,25], plot_mode=["auto"])
	#	plot_ba_edv(save_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_Belastung, plot_indexes=[i], ylim=[-50,50], plot_mode=["auto"])
	#	plot_ba_esv(save_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_Belastung, plot_indexes=[i], ylim=[-18,18], plot_mode=["auto"])

def save_fig1(out_dir, plot=False, img_dir=scanner_reco_dir,
			contour_dir=contour_files_dir,
			seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_Belastung_2d_single_cv"),
			file_extension=".pdf"):
	"""
	Parameters for plotting the measurement types in the manuscript.
	"""
	vol = "vol80"
	reverse = True
	slice_idx=13
	mask_mode = []
	phase_mode = "es"
	save_path = os.path.join(out_dir, "figure_01_measurement_types"+file_extension)
	titles = ["cine", "real-time (76 bpm)", "real-time (115 bpm)", "real-time (162 bpm)"]
	assess_utils.plot_measurement_types(vol, reverse, slice_idx, mask_mode=mask_mode, phase_mode=phase_mode, save_path=save_path,
				contour_dir=contour_dir,
				img_dir =img_dir,
				seg_dir=seg_dir, crop_dim=160, titles =titles, plot=plot)

def save_fig2(out_dir, rtvol_dict=rtvol, plot=False, contour_dir=contour_files_dir, seg_dir=nnunet_output_dir, file_extension=".png"):
	# Figures for Dice's coefficient depending on heart rate
	ylim = [0.2,1]

	# nnU-Net
	calc_DC_and_bpm(rtvol_dict, mode=["nnunet"], contour_dir = contour_dir, seg_dir = seg_dir)
	contour_mode = "nnunet"
	save_path = os.path.join(out_dir, "DC_vs_bpm_"+contour_mode+file_extension)
	title="nnU-Net"
	plot_DC_vs_bpm(rtvol_dict, save_path=save_path, contour_mode=contour_mode, ylim=ylim, plot=plot, title=title)

	# comDL
	calc_DC_and_bpm(rtvol_dict, mode=["auto"], contour_dir = contour_dir, seg_dir = seg_dir)
	contour_mode = "auto"
	save_path = os.path.join(out_dir, "DC_vs_bpm_"+contour_mode+file_extension)
	title="comDL"
	plot_DC_vs_bpm(rtvol_dict, save_path=save_path, contour_mode=contour_mode, ylim=ylim, plot=plot, title=title)

def save_fig3(out_dir, rtvol_dict=rtvol, plot=False, img_dir=scanner_reco_dir,
			contour_dir=contour_files_dir,
			seg_dir=os.path.join(nnunet_output_dir, "rtvol_rt_Belastung_2d_single_cv"),
			file_extension=".pdf"):
	"""
	Plotting limits of neural networks for manuscript.
	"""
	save_path = os.path.join(out_dir, "figure_03_nn_limits"+file_extension)
	param_list = [['vol78', 2, 25], ['vol78', 3, 123], ['vol79', 15, 69], ['vol84', 11, 126]]
	assess_utils.plot_mc_nnunet(contour_dir, img_dir, seg_dir, rtvol_dict, param_list, flag3d=False, mode = "nnunet",
				crop_dim=160, contour_suffix = "_rt_Belastung_manual.con", img_suffix="rt_Belastung_scanner", save_path=save_path,
				check=False, plot=plot)

def save_fig_ba(out_dir, rtvol_dict=rtvol, nnunet_output=nnunet_output_dir, file_extension=".png"):
	"""
	Bland-Altman plots of EDV, ESV and EF for entries of rtvol for cine, real-time and real-time stress.
	"""
	seg_dir = os.path.join(nnunet_output, "rtvol_cine_2d_single_cv/")
	output_file = ""
	ed_tuple_cine, es_tuple_cine, ef_tuple_cine = calc_mean_stdv_parameters_cine(rtvol_dict, seg_dir=seg_dir, output_file=output_file,
									flag3d=False, slice_selection=False)

	seg_dir = os.path.join(nnunet_output, "rtvol_rt_2d_single_cv/")
	ed_tuple_rt, es_tuple_rt, ef_tuple_rt = calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=seg_dir, output_file=output_file)

	seg_dir = os.path.join(nnunet_output, "rtvol_rt_Belastung_2d_single_cv/")
	ed_tuple_rt_Belastung, es_tuple_rt_Belastung, ef_tuple_rt_Belastung = calc_mean_stdv_parameters_rt(rtvol_dict, seg_dir=seg_dir,
							session_mc_suffix="_rt_Belastung_manual.con", session_auto_suffix="_rt_Belastung_automatic.con",
				exp_dir = end_exp_dir, ed_es_phase_file="_rt_Belastung.txt", output_file=output_file)

	for i in range(3):
		plot_ba_ef(out_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-20,20], plot_mode=["nnunet"], file_extension=file_extension)
		plot_ba_edv(out_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-80,80], plot_mode=["nnunet"], file_extension=file_extension)
		plot_ba_esv(out_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-18,18], plot_mode=["nnunet"], file_extension=file_extension)

	for i in range(3):
		plot_ba_ef(out_dir, ef_tuple_cine, ef_tuple_rt, ef_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-200,200], plot_mode=["auto"], file_extension=file_extension)
		plot_ba_edv(out_dir, ed_tuple_cine, ed_tuple_rt, ed_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-120,120], plot_mode=["auto"], file_extension=file_extension)
		plot_ba_esv(out_dir, es_tuple_cine, es_tuple_rt, es_tuple_rt_Belastung,
			plot_indexes=[i], ylim=[-25,25], plot_mode=["auto"], file_extension=file_extension)

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
	save_fig3(out_dir_fig, plot=plot, seg_dir=os.path.join(nnunet_output, "output/rtvol_rt_Belastung_2d_single_cv"))

	# Evaluation of cardiac function parameters
	seg_dir = os.path.join(nnunet_output, "output/rtvol_cine_2d_single_cv/")
	output_file = os.path.join(out_dir_data, "cine_2d_single.txt")
	_, _, _ = calc_mean_stdv_parameters_cine(rtvol, seg_dir=seg_dir, output_file=output_file,
									flag3d=False, slice_selection=False, precision=1)

	seg_dir = os.path.join(nnunet_output, "output/rtvol_rt_2d_single_cv/")
	output_file = os.path.join(out_dir_data, "rt_2d_single.txt")
	_, _, _ = calc_mean_stdv_parameters_rt(rtvol, seg_dir=seg_dir, output_file=output_file, precision=1)

	seg_dir = os.path.join(nnunet_output, "output/rtvol_rt_Belastung_2d_single_cv/")
	output_file = os.path.join(out_dir_data, "rt_Belastung_2d_single.txt")
	_,_,_ = calc_mean_stdv_parameters_rt(rtvol, seg_dir=seg_dir, output_file=output_file, session_mc_suffix="_rt_Belastung_manual.con",
			session_auto_suffix="_rt_Belastung_automatic.con", ed_es_phase_file="_rt_Belastung.txt", precision=1)

	#Bland-Altman Plot
	write_parameter_files_nnunet_auto(rtvol, out_dir_data)
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