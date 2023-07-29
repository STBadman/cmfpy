import coronamagpy.chmetric.chmap.ezsegwrapper as ezsegwrapper
import numpy as npy

# EZSEG
#
#     Python version converted from IDL version converted from FORTRAN 
#       by DH Brooks, Jan 2023
#
#     EZSEG: Routine to segment an image using a two-threshold
#     variable-connectivity region growing method.
#
#     INPUT/OUTPUT:
#        IMG:    Input image.
#        SEG: 
#             ON INPUT:
#                 Matrix of size (nt,np) which contain
#                 1's where there is valid IMG data, and
#                 non-zero values for areas with invalid/no IMG data.
#             ON OUTPUT:
#                 Segmentation map (0:detection ,same as input o.w.).
#        nt,np:   Dimensions of image.
#        thresh1: Seeding threshold value.
#        thresh2: Growing threshold value.
#        nc:      # of consecutive pixels needed for connectivity.
#        iters:
#             ON INPUT: 
#                 maximum limit on number of iterations.
#             ON OUTPUT: 
#                 number of iterations performed.
#
#----------------------------------------------------------------------
#
# Copyright (;) 2015 Predictive Science Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files 
# (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#----------------------------------------------------------------------

# fortran version
def ezseg(*args): ezsegwrapper.ezseg(*args)

# python version
def ezsegpy(IMG,SEG,nt,np,thresh1,thresh2,nc,iters):
      local_vec = npy.arange(15)
      SEG_TMP=SEG
      max_iters=iters
      iters=0
      l1=1
      for k in npy.arange(max_iters):
        if (l1 != 0):
          val_modded=0
          for j in npy.arange(1,np-1): 
            for i in npy.arange(1,nt-1):
              fillit=0
              if (SEG_TMP[i,j] == 1):          
                if (IMG[i,j] <= thresh1):           
                  fillit=1
                else:             
                  if (IMG[i,j] <= thresh2):             
                    local_vec[ 0]=SEG_TMP[i-1,j+1]
                    local_vec[ 1]=SEG_TMP[i  ,j+1]
                    local_vec[ 2]=SEG_TMP[i+1,j+1]
                    local_vec[ 3]=SEG_TMP[i+1,j  ]
                    local_vec[ 4]=SEG_TMP[i+1,j-1]
                    local_vec[ 5]=SEG_TMP[i  ,j-1]
                    local_vec[ 6]=SEG_TMP[i-1,j-1]
                    local_vec[ 7]=SEG_TMP[i-1,j  ]
                    local_vec[ 8]=local_vec[0]
                    local_vec[ 9]=local_vec[1]
                    local_vec[10]=local_vec[2]
                    local_vec[11]=local_vec[3]
                    local_vec[12]=local_vec[4]
                    local_vec[13]=local_vec[5]
                    local_vec[14]=local_vec[6]
                    l2=1
                    for ii in npy.arange(7): 
                      if (l2 != 0):
                        tmp_sum=0
                        for jj in npy.arange(nc):
                          tmp_sum=tmp_sum+local_vec[ii+jj]
                        if (tmp_sum == 0):
                          fillit=1
                          l2=0
                if (fillit == 1):
                  SEG[i,j]=0
                  if (val_modded == 0):
                    val_modded=1
          iters=iters+1
          if (val_modded == 0):
            l1=0          
          else:
            SEG_TMP=SEG

      return SEG