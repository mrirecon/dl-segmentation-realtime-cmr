#!/bin/bash
#Copyright 2023. TU Graz. Institute of Biomedical Imaging.
#Author: Moritz Blumenthal

set -eu

# Create a new directory, insert reference images as img1.png, img2.png, ...
# Open Inkscape and insert images, with the "Link" option
# Position images as desired
# Resize page to new dimensions. File --> Document Properties --> Page --> Custom size --> Resize page to drawing or selection

usage="Usage: $0 <drawing.svg> <output> <images>"


if [ $# -lt 3 ] ; then

	echo "$usage" >&2
	exit 1
fi

WORKDIR=$(mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir')
trap 'rm -rf "$WORKDIR"' EXIT

cp $1 $WORKDIR/drawing.svg
shift 1

OUT=$(readlink -f "$1")
shift 1


i=1
while (($#)); do

	cp $1 $WORKDIR/img${i}.png
	i=$((i+1))
	shift 1
done

cd $WORKDIR

inkscape --export-filename=$OUT drawing.svg

