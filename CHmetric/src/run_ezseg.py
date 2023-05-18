
# Example routine to run ezseg.py 

import os 
import matplotlib.pyplot as plt
import numpy as npy
import scipy as sp
from scipy.io.idl import readsav
from ezseg import ezseg_algorithm

# read the IDL data
s = readsav('CarringtonMapExample.sav')
nx = s.map.shape[0]
ny = s.map.shape[1]

# setup plot environment
px = 1./plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(ny*px,nx*px))
p = fig.add_subplot()

# add the Carrington image
p.axis('off')
color_map = plt.cm.get_cmap('Blues')
rcmap = color_map.reversed()
p.imshow(s.map, cmap=rcmap,origin='lower')  

# setup and run EZSEG algorithm
SEG = npy.zeros((nx,ny))+1
clevel = 2.
nc = 7 
iters = 100
thresh1 = clevel-0.5
thresh2 = clevel

SEG = ezseg_algorithm(s.map,SEG,nx,ny,thresh1,thresh2,nc,iters)

# make open/closed field map from eszeg output
# CH detected so open
SEG = npy.where(SEG != 0, 1, SEG)
# closed
SEG = npy.where(SEG != 1, 0, SEG)

# overplot the open/closed map
cp = p.contour(SEG,1,colors=['white'])

# save the figure 
plt.savefig('Fig.pdf', bbox_inches='tight')
