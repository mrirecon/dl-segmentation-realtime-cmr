#!/bin/bash
#
# Copyright 2023. Uecker Lab, University Medical Center Goettingen.
#
# Author:
# Martin Schilling, 2023, martin.schilling@med.uni-goettingen.de
#

set -e

usage="Usage: $0 <name> <outdir>"

if [ $# -lt 2 ] ; then

        echo "$usage" >&2
        exit 1
fi

# name in [images, contour_files, end_expiration_indexes, nnUNet_inference]

record=10117944

name=$1
outdir=$(readlink -f "$2")

if [ ! -d "$outdir" ] ; then
        echo "Output directory does not exist." >&2
        echo "$usage" >&2
        exit 1
fi

cd "${outdir}"
if [[ ! -f ${name}.tgz ]]; then
	echo Downloading "${name}"
	wget -q https://zenodo.org/record/"${record}"/files/"${name}".tgz
fi

if [[ ! -d $outdir/$name  ]] ; then
	tar -xzvf "${name}".tgz
fi