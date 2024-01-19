The scripts in this repository reproduce the experiments of:

Martin Schilling*, Christina Unterberg-Buchwald*, Joachim Lotz, Martin Uecker.
**Assessment of Deep Learning Segmentation in Real-Time Free-Breathing Cardiac Magnetic Resonance Imaging**

\* These authors contributed equally to this work.

You can set the environment variable DATA_DIR in scripts/assess_dl.env to a local folder for data storage.
By default, this repository folder will be used for data storage.
All data downloaded from zenodo will be decompressed by default.
Images are stored in CFL format, so the TOOLBOX_PATH environment variable needs to be set and point to a folder containing [BART](https://github.com/mrirecon/bart).

[nnU-Net version 1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1) was used for segmentation, which can be reproduced after a correct installation of nnU-Net (nnunet==1.7.1, commit 579c897) and pre-processing of the images (scripts/assess_dl_preprocess.sh)
with scripts/assess_dl_apply_nnunet.sh.
The application of nnU-Net is optional because the inference can be found on [zenodo](https://doi.org/10.5281/zenodo.10117944).

The visualizations have been tested with Python on version 3.10.9 and require numpy, matplotlib, multiprocessing, inspect, sys, os, math, nibabel, and SimpleITK.
A working example of required packages can be found in requirements.txt.

Prerequisite:
* set TOOLBOX_PATH variable to BART folder
* installed required packages

Run run_all.sh to load data and reproduce figures.

* https://mrirecon.github.io/bart
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10117944.svg)](https://doi.org/10.5281/zenodo.10117944)