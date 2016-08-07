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
First menu (Choose files)
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
    if isinstance(choose,int) and choose < len(files):
        chosen.append(choose)

# These are the chosen files to log
chosen_files = []
for num in chosen:
    chosen_files.append(files[num-1])

"""
Second menu (Input details)
"""
cat = {'File':0,'Gender':1,'Genotype':2,'Age':3, 'Stimulus':4, 'Temperature':5, 'Misc':6}
gender = ""
geno = ""
age = ""
stim = ""
temp = ""
misc = ""
logdata = np.empty((len(chosen_files), 7), dtype='object')
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
    
vprint("Output::")
vprint(logdata)

outfile = '../tmp/flylog.txt'
outdir = '../tmp/'
for i, file in enumerate(chosen_files):
    shutil.copy(input_folder + file, outdir + file)
if not os.path.isfile(outfile):
    print("log does not exist")
    with open(outfile,"a+") as f:
         f.write("#File\t\t\t\t#Sex\t#Geno\t#Age\t#Stim\t#Temp\t#Misc\n")
f=open(outfile,'ab')
np.savetxt(f,logdata,fmt='%s', delimiter='\t')
f.close()
