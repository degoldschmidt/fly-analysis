"""

Bout Analyzer (bout_analyzer.py)

This script takes one specified trajectory file from tmp folder 
to view x, y, forward speed, orientation and turning rate as a 
function of time/frames. Users may set thresholds for forward 
speed and turning rate in order to label bout segments (straights, 
spike turns, etc.).   

D.Goldschmidt - 09/08/16
"""

import warnings
from numpy import vstack
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import matplotlib.pyplot as plt  ## package for plotting
from matplotlib.gridspec import GridSpec
from math import *
import os
import shutil
from savitzky_golay import savitzky_golay
from detect_peaks import detect_peaks
import peakutils.peak
import matplotlib.patches as mpatches



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
parser.add_argument('groupname', nargs=1, help='file name for analysis')
parser.add_argument('plotopt', nargs=1, help='plot option')
args = parser.parse_args()
group = args.groupname
group = group[0]
input_folder = "../tmp/" + group + "out/"
outfolder = input_folder + "stats/"
plot_on = int(args.plotopt[0])

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

#alldata = []
#for file in files:
#    alldata.append(np.loadtxt(input_folder+file))

#files = [files[2]]

"""
Group constants
"""
if group == "0714/" or group == "0727/":
    fps = 53.693
    offset = 1
else:
    fps = 60
    offset=0

binss=72*2
allbaseangle = []
allstimangle = []
allbaselens = []
allstimlens = []
for file in files:
    data = np.loadtxt(input_folder+file)
    print(data.shape)
    stimon = 60*60*5
    T       = data[:,0]
    X       = data[:,1]
    Y       = data[:,2]
    Angle   = 180.*data[:,3]/np.pi
    Vf      = data[:,4]
    Vs      = data[:,5]
    Vr      = data[:,6]
    Vt      = data[:,7]
    CState  = data[:,8]
    
    Radius = np.sqrt(np.multiply(X,X) + np.multiply(Y,Y))
    Error = np.sin(Angle - np.pi)
    DState  = np.abs(np.sign(np.diff(CState)))
    DState = np.hstack((np.zeros(1), DState))
    numbouts = int(np.sum(DState) + 2) ## plus 2 because before and after
    ind = np.where(DState > 0)
    print("Detected", numbouts, "bouts")
    
    """
    # TRAJECTORY PLOT
    fig = plt.figure(1, figsize=(8, 8))
    ax = fig.gca()

    colors = ['gray', 'black', 'green', 'blue', 'orange', 'red']
    for index in range(6):
        Xplot = X[CState==index] 
        Yplot = Y[CState==index]
        Rplot = Angle[CState==index]
        quivlen = 1
        plt.quiver(Xplot, Yplot, quivlen*np.cos(Rplot), quivlen*np.sin(Rplot), 
           color=colors[index], scale=2, units="xy", pivot='mid') 
        plt.plot(X[ind[0]],Y[ind[0]],'m*')    

    ax.set_xlim([-45,45])
    ax.set_ylim([-45,45])
    """
    
    bouts = np.zeros((numbouts,3))    ## 0: start, 1: end, 2: ctstate
    bouts[0,2] = CState[0]
    i=0
    for index in ind[0]:
        bouts[i+1,0] = index
        bouts[i, 1] = index-1
        bouts[i+1,2] = CState[index]
        i=i+1
    bouts[-1,1] = T.shape[0]-1  
    
    nullact =  bouts[:,2]==0
    activ =  bouts[:,2]==1
    spike_turn = bouts[:,2]==2
    straights = bouts[:,2] > 2
    shortbouts = bouts[:,2]==3
    mediumbouts = bouts[:,2]==4
    longbouts = bouts[:,2]==5
    #print(bouts[longbouts,:])
    #print(bouts[longbouts,:].shape[0])
    numsbouts = bouts[straights,:].shape[0]
    
    start = bouts[straights,:][0,0]
    end = bouts[straights,:][0,1]
    stimangle = np.zeros((1))
    if start < stimon:
        baseangle = np.remainder(Angle[start:end],360)
    else:
        stimangle = np.remainder(Angle[start:end],360)
    for i in range(bouts[straights,:].shape[0]-1):
        start = bouts[straights,:][i+1,0]
        end = bouts[straights,:][i+1,1]
        if start < stimon:
            baseangle = np.hstack((baseangle, np.remainder(Angle[start:end],360)))
        else:
            if stimangle.shape[0] < 2:
                stimangle = np.remainder(Angle[start:end],360)
            else:
                stimangle = np.hstack((stimangle, np.remainder(Angle[start:end],360)))
    
    
    baselens = bouts[straights,:][:,1] - bouts[straights,:][:,0]
    baselens = baselens[bouts[straights,:][:,0]<stimon]/60.
    stimlens = bouts[straights,:][:,1] - bouts[straights,:][:,0]
    stimlens = stimlens[bouts[straights,:][:,0]>=stimon]/60.
    allbaseangle.append(baseangle)
    allstimangle.append(stimangle)
    allbaselens.append(baselens)
    allstimlens.append(stimlens)  
    

    """
    HISTOGRAMING
    """
    ### ANGLES
    
    plt.figure()
    plt.title("Fly " + file.replace(".txt","") + ", n=" + str(numsbouts))
    plt.xlabel("Fly's heading $\phi$ [$^\circ$]")
    plt.ylabel("$p(\phi)$")
    plt.xlim([0,360])
    plt.hist(baseangle, bins=binss, histtype='stepfilled', normed=True, color='b', label='Baseline')
    plt.hist(stimangle, bins=binss, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Stimulus')
    plt.legend(frameon=False)
    jpgfile = outfolder + "ang_" + file.replace("txt","jpg")
    plt.savefig(jpgfile, dpi=300, format="png")
    #plt.show()
    
    ### LENGTHS
    plt.figure()
    plt.title("Fly " + file.replace(".txt","") + ", n=" + str(numsbouts))
    plt.xlabel("Bout lengths $L_{bout}$ [s]")
    plt.ylabel("$p(L_{bout})$")
    plt.xlim([0.33,5])
    plt.hist(baselens, bins=100, histtype='stepfilled', normed=True, color='b', label='Baseline')
    plt.hist(stimlens, bins=100, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Stimulus')
    plt.legend(frameon=False)
    jpgfile = outfolder + "len_" + file.replace("txt","jpg")
    plt.savefig(jpgfile, dpi=300, format="png")

### ANGLE
allbase = allbaseangle[0]
allstim = allstimangle[0]
for i in range(len(allstimangle)-1):
    print(allstim.shape, allstimangle[i+1].shape)
    allbase = np.hstack((allbase, allbaseangle[i+1]))
    allstim = np.hstack((allstim, allstimangle[i+1]))
    

plt.figure()
plt.title("WT dl (N=16 flies)")
plt.xlabel("Bout lengths $L_{bout}$ [s]")
plt.ylabel("$p(L_{bout})$")
plt.xlim([0,360])
plt.hist(allbase, bins=100, histtype='stepfilled', normed=True, color='b', label='Baseline')
plt.hist(allstim, bins=100, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Stimulus')
plt.legend(frameon=False)
jpgfile = outfolder + "ang_final.jpg"
plt.savefig(jpgfile, dpi=300, format="png")

### LENGTHS
allbasel = allbaselens[0]
allstiml = allstimlens[0]
for i in range(len(allstimlens)-1):
    allbasel = np.hstack((allbasel, allbaselens[i+1]))
    allstiml = np.hstack((allstiml, allstimlens[i+1]))
    
plt.figure()
plt.title("WT dl (N=16 flies)")
plt.xlabel("Bout lengths $L_{bout}$ [s]")
plt.ylabel("$p(L_{bout})$")
plt.xlim([0.33,5])
plt.hist(allbasel, bins=100, histtype='stepfilled', normed=True, color='b', label='Baseline')
plt.hist(allstiml, bins=100, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='Stimulus')
plt.legend(frameon=False)
jpgfile = outfolder + "len_final.jpg"
plt.savefig(jpgfile, dpi=300, format="png")
     
    