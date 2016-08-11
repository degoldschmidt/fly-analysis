"""

Experiment stop (experiment_stop.py)

This script takes a video and calculates the frame number of when 
the experiment was stopped, based on overall pixel changes.   

D.Goldschmidt - 09/08/16
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt  ## package for plotting


__VERBOSE = True
def averag(input):
    sum = 0.*input[0]
    for vals in input:
        sum += vals
    return sum/len(input)

# Func to print out only if VERBOSE
def vprint(*arg):
    if __VERBOSE:
        s= " ".join( map( str, arg ) )
        print(s)

# Local test
#folder = "/Users/degoldschmidt/"
#filename = "output.avi"

folder = "/Volumes/Elements/raw_data_flies/0727/"
filename="VidSave_0726_20-13.avi" 
profolder = "../tmp/vid/"

if not os.path.isfile(profolder + filename):
    os.system("ffmpeg -i " + folder + filename + " -vf fps=fps=4 -f avi -c:v libx264 -s 50x50 " + profolder + filename) ## maybe in fly logger
cap = cv2.VideoCapture(profolder + filename)
if not cap.isOpened():
        print("Error: Could not open")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print("Open video", profolder + filename, "(","#frames:", length, "dims:", (width,height), "fps:", fps,")")

delta = []
i=0
filter = int(500/fps)
motionthr=50
frames = filter*[None]
while(i+1 < length):
    if i%1000==0:
        vprint(i)
        
    # Capture frame-by-frame
    ret, gray = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    # center and radius are the results of HoughCircle
    # mask is a CV_8UC1 image with 0
    mask = np.zeros((gray.shape[0], gray.shape[1]), dtype = "uint8")
    cv2.circle( mask, (int(width/2),int(height/2)), int(width/2-width/20), (255,255,255), -1, 8, 0 )    
    res = np.zeros((gray.shape[0], gray.shape[1]), dtype = "uint8")
    np.bitwise_and(gray, mask, res)  
    
    if i>0:
        frames[(i-1)%filter] = res-oldpx 
        if i > filter-1:
            out = averag(frames)
            if __VERBOSE:
                cv2.imshow('frame', out)
            delta.append(sum(sum(out)))
    oldpx = res
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

ddelta = [j-i for i, j in zip(delta[:-1], delta[1:])]
plt.plot(delta[:],'k--', label="Sum of lowpass-filtered px changes")
plt.plot(ddelta[:],'r-', label= "Temp. difference")
plt.legend()
if __VERBOSE:
    plt.show()

ddelta = np.asarray(ddelta)
stopframes = np.asarray(np.nonzero(ddelta > motionthr))
if stopframes.size > 0:
    print("Experiment stops at frame", stopframes[0,0])
else:
    print("No experiment stop detected")