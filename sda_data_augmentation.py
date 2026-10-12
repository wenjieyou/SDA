# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:20:39 2025

    1. Using Pearson Correlation Coefficient (PCC)
       i.e., compute the PCC for the FDA-augmented data (20*20) and sort them
       (using the first data instance (equivalent to covPSSM) as the reference)
    
@author: wenjie
"""

import numpy as np

def Sort_PCC_r0(dat3d):     # PCC    
    tmp2d = dat3d.reshape((-1,20*20))
    r = np.corrcoef(tmp2d)
	# Using gap=0 as the reference, extract the top da instances with the highest 
    # correlation coefficients with it, forming an augmentation factor of da    
    rr0 = r[:,0]               
    idx = np.argsort(rr0)  # Sort rr0 in ascending order, returning the corresponding indices 
    new_dat3d = dat3d[idx[::-1],:,:]    # idx[::-1] reverses the order of idx
    return new_dat3d, idx[::-1]
    
     
da = 150

d = np.load('sda_dat/PDB1075_pssm20_6mer_25gapcbd_DA150_FDA.npz')

dat = d['datX']
ylab = d['ylab']
m,n,k,l = dat.shape
dd = np.zeros((m,n,k,l))

for i in range(m):
    print(i)
    si = dat[i]   
    dd[i] = Sort_PCC_r0(si)[0]        

newdd = dd[:,:da,:,:]   # Extract the top da instances with the highest 
                        # correlation with gap=0 as the augmentation factor
datX = newdd
savfn = 'sda_dat/PDB1075_pssm20_6mer_25gapcbd_DA150_SDA.npz'
np.savez(savfn,datX=datX,ylab=ylab[:,:da])


