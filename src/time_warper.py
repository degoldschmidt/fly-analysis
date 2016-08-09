"""

Time Warper (time_warper.py)

This script returns an estimate of the framerate of a given video 
of known duration.  

D.Goldschmidt - 09/08/16
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import os

folder = "/Volumes/Elements/raw_data_flies/vidtest_frameratecalib/"
filename="VidSave_1min.avi" 
profolder = "../tmp/vid/"

if not os.path.isfile(profolder + filename):
    os.system("ffmpeg -i " + folder + filename + " -f avi -c:v libx264 -s 50x50 " + profolder + filename) ## maybe in fly logger
    
cap = cv2.VideoCapture(profolder + filename)
if not cap.isOpened():
        print("Error: Could not open")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
val = input("How many seconds: ")

efps = (1.*length)/float(val)
print("Estimated framerate:", efps)