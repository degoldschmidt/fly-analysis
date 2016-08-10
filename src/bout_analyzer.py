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
fps = 60.
T = data[:,spec['Frame']]/60. - data[0,spec['Frame']]/60.
X = data[:,spec['X']]/9.67
Y = data[:,spec['Y']]/9.67
#Y = -data[:,spec['Y']]/9.67 + 90.
Rad = np.pi*data[:,spec['Phi']]/180.
Angle = Rad #data[:,spec['Phi']]
#print(T.shape, X.shape, Y.shape, Angle.shape)

"""
Calculate forward, sideward, angular and translational speed 
"""
Vf = np.multiply( (X[1:]-X[:-1]), np.cos(Rad[:-1]) ) + np.multiply( Y[1:]-Y[:-1], np.sin(Rad[:-1])) 
Vf = Vf * fps
Vs = np.multiply( -(X[1:]-X[:-1]), np.sin(Rad[:-1]) ) + np.multiply( Y[1:]-Y[:-1], np.cos(Rad[:-1]))
Vs = Vs * fps
AngleSav = savitzky_golay(Angle, 51., 3.) # window size 51, polynomial order 3
Vr = fps*(np.diff(AngleSav))
Vt = np.sqrt(np.multiply(Vf,Vf) + np.multiply(Vs,Vs));
aVt = np.abs(np.diff(Vt))

"""
Clear jumps
"""
peaks = detect_peaks(aVt, mph=10, mpd=5)
#peaks = peakutils.peak.indexes(aVt, thres=0.3, min_dist=2)

Vf_tmp = Vf
peaks_tmp = np.hstack( (np.zeros(1), peaks, (np.ones(1)*Vf.size)) ) 
numjumps = peaks_tmp.size - 1
for i in range(numjumps):
    aux = Vf_tmp[peaks_tmp[i]:peaks_tmp[i+1]];
    aux[np.isnan(aux)] = [];
    env=10
    #print(i, aux.size, peaks_tmp[i+1])
    if np.mean(aux) < 0 and aux.size > 30:
        Vf[peaks_tmp[i]:peaks_tmp[i+1]] = - Vf[peaks_tmp[i]:peaks_tmp[i+1]]
        Vs[peaks_tmp[i]:peaks_tmp[i+1]] = - Vs[peaks_tmp[i]:peaks_tmp[i+1]]
        #Vr(peaks_tmp(i-1):peaks_tmp(i)) = Vr(peaks_tmp(i-1):peaks_tmp(i))
        #Angle(peaks_tmp(i-1):peaks_tmp(i)) = Angle(peaks_tmp(i-1):peaks_tmp(i)) # -180;             Why not + PI ?
    if peaks_tmp[i+1] > env:
        Vf[peaks_tmp[i+1]-env:peaks_tmp[i+1]+env] = 0 #mean(Vf(peaks_tmp(i)-env:peaks_tmp(i)-3));
        Vs[peaks_tmp[i+1]-env:peaks_tmp[i+1]+env] = 0 #mean(Vs(peaks_tmp(i)-env:peaks_tmp(i)-3));
        Vr[peaks_tmp[i+1]-env:peaks_tmp[i+1]+env] = 0 #mean(Vr(peaks_tmp(i)-env:peaks_tmp(i)-3));
    elif peaks_tmp[i+1] > 2:                                                          ## Why this?
        Vf[peaks_tmp[i+1]-2:peaks_tmp[i+1]+env] = 0 #(Vf(peaks_tmp(i)-2))
        Vs[peaks_tmp[i+1]-2:peaks_tmp[i+1]+env] = 0 #(Vs(peaks_tmp(i)-2))
        Vr[peaks_tmp[i+1]-2:peaks_tmp[i+1]+env] = 0 #(Vr(peaks_tmp(i)-2))
    else:
        Vf[peaks_tmp[i+1]:peaks_tmp[i+1]+env] = 0 #(Vf(peaks_tmp(i)+4))
        Vs[peaks_tmp[i+1]:peaks_tmp[i+1]+env] = 0 #(Vs(peaks_tmp(i)+4))
        Vr[peaks_tmp[i+1]:peaks_tmp[i+1]+env] = 0 #(Vr(peaks_tmp(i)+4))

Vt = np.sqrt(np.multiply(Vf,Vf) + np.multiply(Vs,Vs));
aVt = np.abs(np.diff(Vt))

"""
Postprocessing
"""
#Vf = savitzky_golay(Vf, 51., 3.)
#Vs = savitzky_golay(Vs, 51., 3.)
#Vr = savitzky_golay(Vr, 51., 3.)
#Vt = savitzky_golay(Vt, 51., 3.)
#aVt = savitzky_golay(aVt, 51., 3.)

"""
Activity bouts (0: no activity, 1: activity)
"""
thr_lin = 0.5       ## linear threshold [mm/s]
thr_ang = 20        ## angular threshold [º/s]
fram_len = 20       ## minimum length of bouts
classState = np.zeros(Vr.shape)
ind = np.where(np.logical_or(np.abs(Vr) > thr_ang, np.logical_or(np.abs(Vf) > thr_lin, np.abs(Vs)>thr_lin)))
curr_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
    else:
        if curr_len > fram_len:
            classState[i-curr_len:i-1] = classState[i-curr_len:i-1] + 1
        curr_len = 0
    
"""
Spike turns (2: Spike turn)
"""
thr_spike = np.pi*120.0/180.0      ## angular threshold [º/s]
#ind = np.where(np.logical_and(actState == 1, np.hstack((np.zeros(1), np.abs(fps*np.diff(Vr)))) > thr_spike))
ind = np.where(np.logical_and(classState > 0, np.abs(Vr) > thr_spike))
curr_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
    else:
        if curr_len > fram_len:
            classState[i-curr_len:i-1] = classState[i-curr_len:i-1] + 1
        curr_len = 0

"""
Straight bouts (3: Straights, 4: long straights)
"""
thr_speed = 3
ind = np.where(np.logical_and(classState == 1, np.abs(Vf) > thr_speed))
long_len = 60
curr_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
    else:
        if curr_len > fram_len:
            classState[i-curr_len:i-1] = classState[i-curr_len:i-1] + 2
        if curr_len > long_len:
            classState[i-curr_len:i-1] = classState[i-curr_len:i-1] + 1
        curr_len = 0



"""
Correct array lengths
"""
Vf = np.hstack((np.zeros(1),Vf))
Vs = np.hstack((np.zeros(1),Vs))
Vr = np.hstack((np.zeros(1),Vr))
Vt = np.hstack((np.zeros(1),Vt))
aVt = np.hstack((np.zeros(1),aVt))
#print(Vf.shape, Vs.shape, Vr.shape, Vt.shape)
                              

"""
Plot data with respect to frames
"""
# Closing previous windows
plt.close('all')

"""


# Subplots
#plt.figure(1, figsize=(8, 8))
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(T, X, 'r-')
axarr[0].set_ylabel("X [mm]")
#axarr[0].set_title('Sharing X axis')
axarr[1].plot(T, Y, 'b-')
axarr[1].set_ylabel("Y [mm]")
axarr[2].plot(T, Angle, 'g-')
axarr[2].set_ylabel("Orientation [cont. º]")
axarr[-1].set_xlabel("Time [s]")
"""
f, axarr = plt.subplots(4, sharex=True)
tstart = 60 * 0 
tend = T.size - 1 # 60 * 60 * 19.5     ## 3600 = 1 min
axarr[0].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==1)-20, facecolor='lightsalmon', alpha=0.5, linewidth=0.0)
axarr[0].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==2)-20, facecolor='lightgreen', alpha=0.5, linewidth=0.0)
axarr[0].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==3)-20, facecolor='dodgerblue', alpha=0.5, linewidth=0.0)
axarr[0].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==4)-20, facecolor='red', alpha=0.5, linewidth=0.0)
axarr[0].plot(T[tstart:tend], Vf[tstart:tend], 'r-')
#axarr[0].plot(T[tstart:tend], 10*classState[tstart:tend], 'k--')
axarr[0].set_ylabel("Vf [mm/s]")
axarr[0].set_ylim([-20, 60])

axarr[1].plot(T[tstart:tend], Vs[tstart:tend], 'b-')
axarr[1].set_ylabel("Vs [mm/s]")

axarr[2].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==1)-10, facecolor='lightsalmon', alpha=0.5, linewidth=0.0)
axarr[2].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==2)-10, facecolor='lightgreen', alpha=0.5, linewidth=0.0)
axarr[2].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==3)-10, facecolor='dodgerblue', alpha=0.5, linewidth=0.0)
axarr[2].fill_between(T[tstart:tend], -100, 100*(classState[tstart:tend]==4)-10, facecolor='red', alpha=0.5, linewidth=0.0)
axarr[2].plot(T[tstart:tend], Vr[tstart:tend], 'g-')
axarr[2].set_ylabel("Vr [rad/s]")
axarr[2].set_ylim([-5, 5])
axarr[3].plot(T[tstart:tend], aVt[tstart:tend], 'm-')
axarr[3].plot(T[peaks], Vt[peaks], 'r*')
axarr[3].set_ylabel("Vt [mm/s]")
axarr[-1].set_xlabel("Time [s]")
for ax in axarr:
    ax.set_xlim([T[tstart], T[tend]])


""" TRAJECTORY PLOT
plt.figure(2, figsize=(8, 8))
Xs=X[tstart:tend]
Ys=Y[tstart:tend]
aS=actState[tstart:tend]
spS=spikeState[tstart:tend]
stS=strState[tstart:tend]

Xno = Xs[aS==0]
Xsp = Xs[spS==1]
Xst = Xs[stS==1]
Yno = Ys[aS==0]
Ysp = Ys[spS==1]
Yst = Ys[stS==1]
plt.plot(Xs, Ys,'k.')
plt.plot(Xsp, Ysp,'g.')
plt.plot(Xst, Yst,'b.')
"""

plt.tight_layout()
plt.show()
