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
#fps = 53.693
fps = 60.
end = -1##330 * fps#-1

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
parser.add_argument('filename', nargs=1, help='file name for analysis')
parser.add_argument('plotopt', nargs=1, help='plot option')
parser.add_argument('owrlen', nargs=1, help='overwrite length')
args = parser.parse_args()
group = args.groupname
input_folder = "../tmp/" + group[0]
outfolder = input_folder + "out/"
file = args.filename
plot_on = int(args.plotopt[0])
#print(plot_on)
owrlen = int(args.owrlen[0])
if not file[0].endswith(".txt"):
    file[0] = file[0] + ".txt"
if os.path.isfile(outfolder + file[0]) and not owrlen:
    ll = np.loadtxt(outfolder+file[0], usecols=(1,2))
    end=ll.shape[0]
data = np.loadtxt(input_folder+file[0], usecols=(1,2,3,4))
data = data[0:end,:]
spec = {'Frame':0, 
        'X':1, 
        'Y':2, 
        'Phi':3}
Xo = 505.
Yo = 499.
scale = 45./495.
## for pheromones (600x600 video)
if group[0] == "0727/":
    scale = 45./300.
    Xo = 300.
    Yo = 300.
T = data[:,spec['Frame']]/fps - data[0,spec['Frame']]/fps
X = (data[:,spec['X']]-Xo)*scale
Y = (data[:,spec['Y']]-Yo)*scale
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
print("#Jump events:", numjumps)
for i in range(numjumps):
    aux = Vf_tmp[peaks_tmp[i]:peaks_tmp[i+1]];
    aux[np.isnan(aux)] = [];
    env=10
    #print(i, aux.size, peaks_tmp[i+1])
    if np.mean(aux) < 0 and aux.size > 30:
        Vf[peaks_tmp[i]:peaks_tmp[i+1]] = - Vf[peaks_tmp[i]:peaks_tmp[i+1]]
        Vs[peaks_tmp[i]:peaks_tmp[i+1]] = - Vs[peaks_tmp[i]:peaks_tmp[i+1]]
        Vr[peaks_tmp[i-1]:peaks_tmp[i]] = - Vr[peaks_tmp[i-1]:peaks_tmp[i]]
        Angle[peaks_tmp[i-1]:peaks_tmp[i]] = Angle[peaks_tmp[i-1]:peaks_tmp[i]] -np.pi         
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
Vf = savitzky_golay(Vf, 51., 3.)
Vs = savitzky_golay(Vs, 51., 3.)
Vr = savitzky_golay(Vr, 51., 3.)
Vt = savitzky_golay(Vt, 51., 3.)
aVt = savitzky_golay(aVt, 51., 3.)

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
Activity bouts (0: no activity, 1: activity)
"""
thr_lin = 0.5       ## linear threshold [mm/s]
thr_ang = 20        ## angular threshold [º/s]
fram_len = 20       ## minimum length of bouts
min_len =  0        ## minimum length of stops
classState = np.zeros(Vr.shape)
ind = np.where(np.logical_or(np.abs(Vr) > thr_ang, np.logical_or(np.abs(Vf) > thr_lin, np.abs(Vs)>thr_lin)))
curr_len = 0
stop_len = 0
last_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
        if stop_len > 0 and stop_len < min_len and last_len > 0:
            classState[i-stop_len:i-1] = classState[i-stop_len:i-1] + 1
            curr_len = curr_len + stop_len + last_len
    else:
        if curr_len > fram_len:
            classState[i-curr_len:i-1] = 1
        last_len = curr_len
        curr_len = 0
#print(classState)
if np.any(classState > 1):
    print("Warning: Activity classifier.") 
    
"""
Spike turns (2: Spike turn)
"""
thr_spike = np.pi*140.0/180.0      ## angular threshold [º/s]
#ind = np.where(np.logical_and(actState == 1, np.hstack((np.zeros(1), np.abs(fps*np.diff(Vr)))) > thr_spike))
ind = np.where(np.logical_and(classState == 1, np.abs(Vr) > thr_spike))
curr_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
    else:
        if curr_len > 10:
            classState[i-curr_len:i-1] = 2
        curr_len = 0
        
if np.any(classState > 2):
    print("Warning: Spike turn classifier.") 

"""
Straight bouts (3: Straights, 4: long straights)
"""
thr_speed = 3
str_thr = np.pi*75.0/180.0      ## straightness threshold
ind = np.where(np.logical_and(classState == 1, np.logical_and(np.abs(Vf) > thr_speed, np.abs(Vr) < str_thr)))
long_len = 60 ## 60/90/120
llong_len = 120
curr_len = 0
for i in range(classState.size):
    if np.any(ind[0] == i):
        curr_len = curr_len + 1
    else:
        if curr_len > fram_len:
            classState[i-curr_len:i-1] = 3
        if curr_len > long_len and curr_len <= llong_len:
            classState[i-curr_len:i-1] = 4
        if curr_len > llong_len:
            classState[i-curr_len:i-1] = 5
        curr_len = 0
if np.any(classState > 5):
    print("Warning: Straight classifier.") 
                              

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
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)

"""
Baseline
"""
tstart = 60 * 0
if fps==60.: 
    tend = 60 * 60 * 5 #- data[0,spec['Frame']]# 60 * 60 * 19.5     ## 3600 = 1 min
else:
    tend = 60 * 60 * 5 - data[0,spec['Frame']]# 60 * 60 * 19.5     ## 3600 = 1 min
# TRAJECTORY PLOT
fig = plt.figure(2, figsize=(8, 8))
ax = fig.gca()

circleout = mpatches.Circle((0., 0.), 45.0, color="seashell", alpha=0.25, edgecolor='none') 
circle = mpatches.Circle((0., 0.), 25.0, color="limegreen", alpha=0.25, edgecolor='none') ##28.0
#ax.add_artist(circleout)
#ax.add_artist(circle)

Xs=X[tstart:tend]
Ys=Y[tstart:tend]
cS=classState[tstart:tend]
Rads = Rad[tstart:tend]

colors = ['gray', 'black', 'green', 'blue', 'red', 'red']
for index in range(6):
    Xplot = Xs[cS==index] 
    Yplot = Ys[cS==index]
    Rplot = Rads[cS==index]
    quivlen = 1
    plt.quiver(Xplot, Yplot, quivlen*np.cos(Rplot), quivlen*np.sin(Rplot), 
           color=colors[index], scale=2, units="xy", pivot='mid')     

ax.set_xlim([-45,45])
ax.set_ylim([-45,45])

plt.savefig(outfolder + "Example_base.jpg", dpi=300, format="png")

"""
Stimulus
"""
if fps==60.: 
    tstart = 60 * 60 * 5 + 1 + 300 #- data[0,spec['Frame']]# 60 * 60 * 19.5     ## 3600 = 1 min
else:
    tstart = 60 * 60 * 5 + 1 - data[0,spec['Frame']]# 60 * 60 * 19.5     ## 3600 = 1 min
tend = 60 * 60 * 5 + 60 * 10 + 300 # 60 * 60 * 19.5     ## 3600 = 1 min
# TRAJECTORY PLOT
fig = plt.figure(3, figsize=(8, 8))
ax = fig.gca()

circleout = mpatches.Circle((0., 0.), 45.0, color="seashell", alpha=0.25, edgecolor='none') 
circle = mpatches.Circle((0., 0.), 25.0, color="limegreen", alpha=0.25, edgecolor='none') ##28.0
#ax.add_artist(circleout)
#ax.add_artist(circle)

Xs=X[tstart:tend]
Ys=Y[tstart:tend]
cS=classState[tstart:tend]
Rads = Rad[tstart:tend]

colors = ['gray', 'black', 'green', 'blue', 'red', 'red']
for index in range(6):
    Xplot = Xs[cS==index] 
    Yplot = Ys[cS==index]
    Rplot = Rads[cS==index]
    quivlen = 1
    plt.quiver(Xplot, -Yplot, quivlen*np.cos(Rplot), quivlen*np.sin(Rplot), 
           color=colors[index], scale=2, units="xy", pivot='mid')     

ax.set_xlim([-30+10,30+10])
ax.set_ylim([-30-4,30-4])
ax.set_xlabel("x [mm]")
ax.set_ylabel("x [mm]")

plt.savefig(outfolder + "Example_stim.jpg", dpi=300, format="png")

"""
Xno = Xs[cS==0]
Xsp = Xs[cS==2]
Xst = Xs[cS==3]
Xlst = Xs[cS==4]
Rlst = Rads[cS==4]
Yno = Ys[cS==0]
Ysp = Ys[cS==2]
Yst = Ys[cS==3]
Ylst = Ys[cS==4]
plt.plot(Xs, Ys,'k.')
plt.plot(Xsp, Ysp,'g.')
plt.plot(Xst, Yst,'b.')
plt.plot(Xlst, Ylst,'r.', marker = (3, 0, 45), markersize=20)
"""
outdata = np.zeros((T.size,9))
outdata[:,0] = T
outdata[:,1] = X
outdata[:,2] = Y
outdata[:,3] = Rad
outdata[:,4] = Vf
outdata[:,5] = Vs
outdata[:,6] = Vr
outdata[:,7] = Vt
outdata[:,8] = classState
print(outdata.shape)
np.savetxt(outfolder +file[0], outdata, delimiter="\t")

plt.tight_layout()
if plot_on == 1:
    plt.show()
