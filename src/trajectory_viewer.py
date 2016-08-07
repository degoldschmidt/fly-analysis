"""

Trajectory Viewer (trajectory_viewer.py)

This script culls all txt files in a given folder to plot their 
trajectories on a subplot grid.

D.Goldschmidt - 07/08/16
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import matplotlib.pyplot as plt  ## package for plotting
from matplotlib.gridspec import GridSpec
from math import *
import os
import shutil

# Put here folder of which data is to be loaded
input_folder = "/Volumes/Elements/raw_data_flies/0508/WTdl/"
__VERBOSE = True

# Func to print out only if VERBOSE
def vprint(*arg):
    if __VERBOSE:
        s= " ".join( map( str, arg ) )
        print(s)
    

"""
File handling (print out how many files are processed)
"""
nfiles=0
files=[]
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        nfiles += 1
vprint(nfiles, "txt-files detected")
for file in os.listdir(input_folder):
    if file.endswith(".txt"):
        files.append(file)
files.sort()

"""
Loading trajectories
"""
alltraj = []
for file in files:
    vprint(file)
    data = np.loadtxt(input_folder+file, usecols=(2,3))
    alltraj.append(data)

"""
Plotting trajectories
"""
# Closing previous windows
plt.close('all')


# Setup grid of subplots   
subp_dim = ceil(sqrt(nfiles))
vprint(subp_dim, "rows/cols needed")
max_grid = 4
nfigs = ceil(nfiles/(max_grid**2))
vprint("Open", nfigs, "figure(s)") 
if subp_dim > max_grid:
    subp_dim = max_grid

pltgrid = GridSpec(subp_dim, subp_dim)
 
for figindex in range(nfigs):
    plt.figure(figindex+1, figsize=(8, 8))


    # Loop through each file
    for index1 in range(subp_dim):
        for index2 in range(subp_dim):
            ax = plt.subplot(pltgrid[index1, index2], aspect=1, adjustable='box-forced')
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(8)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(8)  
            if index2+subp_dim*index1+figindex*max_grid*max_grid < len(files):
                plt.title(files[index2+subp_dim*index1+figindex*max_grid*max_grid], fontsize=10)
                data = alltraj[index2+subp_dim*index1+figindex*max_grid*max_grid]
                if data.size > 0:
                    plt.plot(data[:,0], data[:,1])
    plt.tight_layout()
plt.show()