#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

VOL_ARRAY=(vol60 vol61 vol62 vol64 vol67 vol69 vol70 vol71 vol72 vol78 vol79 vol80 vol82 vol83 vol84)

# Copy data
#IDIR=/home/ague/data/mschi/rt_seg/rtvol_data/
#ODIR=/home/ague/archive/projects/2023/mschi/assess_dl_seg/scanner_reco/
#
#for d in "${VOL_ARRAY[@]}"
#do
#	if ! [ -d $ODIR/"$d" ] ; then
#		cp -r $IDIR/"$d" $ODIR/"$d"
#	fi
#done

# Compress DICOM files
IDIR=/media/mschi/WDElements/rt_vol_data/
ODIR=/home/ague/archive/projects/2023/mschi/assess_dl_seg/dicoms/

cd $IDIR || exit

for d in "${VOL_ARRAY[@]}"
do
	echo "$d"
	if ! [ -f $ODIR/"$d".tgz ] ; then
		tar -czf $ODIR/"$d".tgz "$d"/
	fi
done