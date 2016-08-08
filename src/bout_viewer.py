"""

Bout Viewer (bout_viewer.py)

This script takes one specified trajectory file from tmp folder 
to view x, y, forward speed, orientation and turning rate as a 
function of time/frames. Users may set thresholds for forward 
speed and turning rate in order to label bout segments (straights, 
spike turns, etc.).   

D.Goldschmidt - 08/08/16
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
input_folder = "../tmp/"
__VERBOSE = True

# Func to print out only if VERBOSE
def vprint(*arg):
    if __VERBOSE:
        s= " ".join( map( str, arg ) )
        print(s)
    
"""
Argument parser (print out how many files are processed)
"""
parser = argparse.ArgumentParser(description='Get file name.')
parser.add_argument('filename', nargs=1, help='file name for analysis')
args = parser.parse_args()
file = args.filename
if not file[0].endswith(".txt"):
    file[0] = file[0] + ".txt"
data = np.loadtxt(input_folder+file[0], usecols=(1,2,3,4))
spec = {'Frame':0, 
        'X':1, 
        'Y':2, 
        'Phi':3}

"""
Plot data with respect to frames
"""
# Closing previous windows
plt.close('all')

# Subplots
#plt.figure(figsize=(8, 8))
dt = 1./60.
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(dt*data[:, spec['Frame']], data[:, spec['X']], 'r-')
axarr[0].set_ylabel("X [px]")
#axarr[0].set_title('Sharing X axis')
axarr[1].plot(dt*data[:, spec['Frame']], data[:, spec['Y']], 'b-')
axarr[1].set_ylabel("Y [px]")
axarr[2].plot(dt*data[:, spec['Frame']], data[:, spec['Phi']], 'g-')
axarr[2].set_ylabel("Orientation [cont. º]")

axarr[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()