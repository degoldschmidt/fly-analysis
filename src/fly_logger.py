"""

Fly Logger (fly_logger.py)

This script allows for logging/editing fly data. Log contains 
information about gender, genotype, age, stimulus, temperature, 
misc. comments, as well as datetime and length of the experiment. 
It further calculates number of frames and estimates frame rate.

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
input_folder = "/Volumes/Elements/raw_data_flies/"
parser = argparse.ArgumentParser(description='Get file name.')
parser.add_argument('groupname', nargs=1, help='file name for analysis')
args = parser.parse_args()
group = args.groupname[0]
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
for file in os.listdir(input_folder + group):
    if file.endswith(".txt"):
        nfiles += 1
vprint(nfiles, "txt-files detected")
for file in os.listdir(input_folder + group):
    if file.endswith(".txt"):
        files.append(file)
files.sort()

"""
First menu (File selection)
"""
print("Which files?")
for i, file in enumerate(files):
    vprint("(", i+1, ")", file)

# while loop for 
chosen = []
while True:
    choose = input("Enter a number: ")
    if choose == "":
        break
    choose = int(choose)
    if isinstance(choose,int) and choose-1 < len(files):
        chosen.append(choose)

# These are the chosen files to log
chosen_files = []
for num in chosen:
    chosen_files.append(files[num-1])

"""
Second menu (Input details)
"""
cat = {'File':0,'Gender':1,'Genotype':2,'Age':3, 'Stimulus':4, 'Temperature':5, 'Misc':6, 'Fps':7, 'Len': 8}
gender = ""
geno = ""
age = ""
stim = ""
temp = ""
misc = ""
fpsin = ""
logdata = np.empty((len(chosen_files), 9), dtype='object')
for i, file in enumerate(chosen_files):
    vprint("(", i+1, ")", file)
    logdata[i,cat['File']]        = file
    
    # Gender (possibly batched)
    if "%" not in gender:
        gender = input("Gender: ")        
    logdata[i,cat['Gender']] = gender.replace("%", "")
    
    # Genotype (possibly batched)
    if "%" not in geno:
        geno = input("Genotype: ")        
    logdata[i,cat['Genotype']] = geno.replace("%", "")

    # Age (possibly batched)
    if "%" not in age:
        age = input("Age: ")        
    logdata[i,cat['Age']] = age.replace("%", "")
    
    # Stimulus (possibly batched)
    if "%" not in stim:
        stim = input("Stimulus: ")        
    logdata[i,cat['Stimulus']] = stim.replace("%", "")
    
    # Temperature (possibly batched)
    if "%" not in temp:
        temp = input("Temperature: ")        
    logdata[i,cat['Temperature']] = temp.replace("%", "")
    
    # Misc comments (possibly batched)
    if "%" not in misc:
        misc = input("Misc: ")        
    logdata[i,cat['Misc']] = misc.replace("%", "")
    
    if "%" not in fpsin:
        fpsin = input("Framerate: ")        
    logdata[i,cat['Fps']] = fpsin.replace("%", "")
    
    # FPS 
    #fps = 53.693

# Console output of logged data    
vprint("Output::")
vprint(logdata)

"""
Writing log data into file (append), copying raw data into tmp folder
"""
outdir = '../tmp/' + group
outlog = 'flylog.csv'
for i, file in enumerate(chosen_files):
    shutil.copy(input_folder + group + file, outdir + '{:05d}'.format(i+1) + ".txt")
if not os.path.isfile(outdir + outlog):
    print("log does not exist")
    with open(outdir + outlog,"a+") as f:
         f.write("#File\t\t\t\t#Sex\t#Geno\t#Age\t#Stim\t#Temp\t#Misc\n")
f=open(outdir + outlog,'ab')
np.savetxt(f,logdata,fmt='%s', delimiter='\t')
f.close()