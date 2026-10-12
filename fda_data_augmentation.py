# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:33:29 2024
Data augmentation: mmerSM (Definition 2)
                   ngapDM (Definition 3: using L1 distance, i.e., CBD: CityBlockDistance)
                   Combined into: mixcas (our method)
Parameter values: ngap=0..L-1; mmer=1..L
Note: In the feature representation process of data augmentation for deep learning,
      standardization (e.g., sigmoid) is NOT required.
	  
@author: Wenjie You
"""

import scipy.io as sio
import sys
import numpy as np
from scipy.spatial.distance import cdist

import time

start = time.time()

# Load MATLAB data in cell format
data = sio.loadmat('pssm_data/PDB1075_pssmMatrix.mat');
allpssm=data['pssm']
ylab = data['label']
num_seq = allpssm.shape[0]


def gapcbd(pssm, ngap):      # Call scipy package to compute L1 distance
    L, kk = np.shape(pssm)
    # pssm = 1.0/(1+np.exp(-pssm))   # sigmoid transformation：1/(1+e^-x)
    if ngap >= L:
        print("\nERROR!!!, Gap value cannot exceed L-1!\n")
        sys.exit()
    # Front block and Back block of the pssm matrix, with a distance of gap between them	
    col_FrontBlock = pssm[:L-ngap]
    col_BackBlock = pssm[ngap:]
    gap_cbd = np.zeros((kk,kk),dtype=float)   
    gap_cbd = cdist(col_FrontBlock.T,col_BackBlock.T,metric='cityblock')
    # cdist computes the distance between every pair of the two input sets
	# Return a square matrix; Note: distance divided by vector dimension (since dimension L varies)
    return gap_cbd/(L-ngap)


def mersm(pssm, mmer):      # Similar to kmer operation
    L, n = pssm.shape    
    # pssm = 1.0/(1+np.exp(-pssm))   
    if mmer < 1 or mmer > L:
        print("\nERROR!!!, kmer={} > 1 and < L={}!".format(mmer,L))
        return
    if mmer == 1:
        merSM = pssm
    else:
        merSM = np.zeros((L-mmer+1,n),dtype=float)
        for i in range(L-mmer+1):
            merSM[i] = pssm[i:i+mmer-1].sum(axis=0)   # Extract consecutive mmer rows and sum them    
    return merSM


def mixcas(pssm, mmer, ngap): # Generate the kmerPSSM matrix ((L-k+1)*20) from pssm, then compute gapcbd
    L,n = pssm.shape
    if mmer > L or ngap > (L-mmer):
        print("\n ERROR!!!, mmer={} > 1 and < L={}!".format(mmer,L))
        sys.exit()
    mmersm = mersm(pssm, mmer)
    if ngap >= mmersm.shape[0]: 
        print("\n ERROR!!!, ngap={} < {}".format(ngap, mmersm.shape[0]))
        sys.exit()
    mersm_gapcbd = gapcbd(mmersm,ngap)  
    return mersm_gapcbd    

dim = 20    
M = 6   # m_mer value (1..M)
N = 25  # n_gap value (0..N-1); augmentation factor = M * N
mersmgapcbd_4d = np.zeros((num_seq,M*N,dim,dim),dtype=float)
M_N_idx = np.zeros((num_seq,M*N,2),dtype=int)
lenseq = []

for i in range(num_seq):
    print('Sequence No. {}'.format(i))
    pssmmatrix=allpssm[i][0]  
    lenseq.append(pssmmatrix.shape[0])
    
    
    for mmer in range(M):
        # Call mixcas
        for ngap in range(N):
            mersm_gapcbd = mixcas(pssmmatrix, mmer+1, ngap)
            mersmgapcbd_4d[i,mmer*N+ngap,:] = mersm_gapcbd
            M_N_idx[i,mmer*N+ngap,:] = [mmer+1, ngap]    
 
minlen = np.min(lenseq)
print('\n The shortest sequence length among all sequences: %s'%minlen)               
end = time.time()

print('\n CPU running time：%s seconds'%(end-start))

datX = mersmgapcbd_4d
idx = M_N_idx
ylab = np.tile(ylab,(1,M*N))   # Replicate the entire matrix

# Save the numpy.narray 4-dimensional matrix gapdpc_4d as an npy format file

savfn=f'sda_dat/PDB1075_pssm20_{M}mer_{N}gapcbd_DA{M*N}_FDA.npz'
np.savez(savfn,datX=datX,ylab=ylab,idx=idx)   
