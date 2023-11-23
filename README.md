The scripts in this repository reproduce the experiments of:

Martin Schilling*, Christina Unterberg-Buchwald*, Joachim Lotz, Martin Uecker.
**Assessment of Deep Learning Segmentation in Real-Time Free-Breathing Cardiac Magnetic Resonance Imaging**

\* These authors contributed equally to this work.

You can set the environment variable DATA_DIR in scripts/assess_dl.env to a local folder for data storage.
By default this repository folder will be used for data storage.
All data downloaded from zenodo will be decompressed by default.

The visualizations have been tested with Python on version 3.9.10 and require numpy, matplotlib, multiprocessing, inspect, sys, os, math, nibabel, and SimpleITK.

Prerequisite:
* conda installation

Run run_all.sh to load data, create a suitable conda environment and reproduce figures.

* https://mrirecon.github.io/bart
* [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10117944.svg)](https://doi.org/10.5281/zenodo.10117944)