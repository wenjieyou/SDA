# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 15:33:29 2024
数据增强： mmerSM (定义2) 
          ngapcbd (定义3：使用 L1 距离, 也即 CBD: CityBlockDistance)
          组合成为：mixcas (our)
参数取值：ngap=0..L-1; mmer=1..L

注意：用于深度学习的数据增强的特征表示过程中，无需进行标准化(如:sigmoid)
@author: Wenjie You
"""

import scipy.io as sio
import sys
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import cdist

import time

start = time.time()

K = 10

# 读入 matlab 数据 cell 格式
data = sio.loadmat(f'../SM_mat/pssmMatrix_matlab/PaCRISPR_uniref90_IND_26_26/PaCRISPR_uniref90_IND_26_26_pssmMatrix_CBalanced_{K}.mat');
allpssm=data['pssm']    # 读取字典对象，字典是另一种可变容器模型
ylab = data['label']
num_seq = allpssm.shape[0]


def gapcbd(pssm, ngap):      # 调用scipy包，计算L1距离(CityBlockDistance)
    L, kk = np.shape(pssm)
    # pssm = 1.0/(1+np.exp(-pssm))   # sigmoid变换：1/(1+e^-x)
    if ngap >= L:
        print("\nERROR!!!, Gap value cannot exceed L-1!\n")
        sys.exit()
    # pssm矩阵的前块和后块，之间距离为gap
    col_FrontBlock = pssm[:L-ngap]
    col_BackBlock = pssm[ngap:]
    gap_cbd = np.zeros((kk,kk),dtype=float)   
    gap_cbd = cdist(col_FrontBlock.T,col_BackBlock.T,metric='cityblock')
    # cdist 计算两组输入的每对之间的距离
    return gap_cbd/(L-ngap)   # 返回方阵, 注：距离除以向量维度(因维度L不同)


def mersm(pssm, mmer):      # 类似kmer操作
    L, n = pssm.shape    
    # pssm = 1.0/(1+np.exp(-pssm))   # sigmoid变换：1/(1+e^-x)
    if mmer < 1 or mmer > L:
        print("\nERROR!!!, kmer={} > 1 and < L={}!".format(mmer,L))
        return
    if mmer == 1:
        merSM = pssm
    else:
        merSM = np.zeros((L-mmer+1,n),dtype=float)
        for i in range(L-mmer+1):
            merSM[i] = pssm[i:i+mmer-1].sum(axis=0)   # 抽取连续mmer行求和    
    return merSM   # 返回第一结果为对称方阵，第二结果为(L-kmer)*20矩阵


def mixcas(pssm, mmer, ngap): #对pssm生成kmerPSSM矩阵((L-k+1)*20),再gapdis
    L,n = pssm.shape
    if mmer > L or ngap > (L-mmer):
        print("\n ERROR!!!, mmer={} > 1 and < L={}!".format(mmer,L))
        sys.exit()
    mmersm = mersm(pssm, mmer)
    if ngap >= mmersm.shape[0]:    # 在之前代码中插入“=”
        print("\n ERROR!!!, ngap={} < {}".format(ngap, mmersm.shape[0]))
        sys.exit()
    mersm_gapcbd = gapcbd(mmersm,ngap)  
    return mersm_gapcbd    

dim = 20    
M = 6   # m_mer值
N = 25  # n_gap值，增强的倍数 = M * N
mersmgapcbd_4d = np.zeros((num_seq,M*N,dim,dim),dtype=float)
M_N_idx = np.zeros((num_seq,M*N,2),dtype=int)
lenseq = []

for i in range(num_seq):
    print('第{}条序列'.format(i))
    pssmmatrix=allpssm[i][0]  
    lenseq.append(pssmmatrix.shape[0])
    
    
    for mmer in range(M):
        # 调用 mixcas
        for ngap in range(N):
            mersm_gapcbd = mixcas(pssmmatrix, mmer+1, ngap)
            mersmgapcbd_4d[i,mmer*N+ngap,:] = mersm_gapcbd
            M_N_idx[i,mmer*N+ngap,:] = [mmer+1, ngap]    
 
minlen = np.min(lenseq)
print('\n 全部序列中，最短序列长度为：%s'%minlen)               
end = time.time()

print('\n CPU运行时间：%s 秒'%(end-start))

datX = mersmgapcbd_4d
idx = M_N_idx
ylab = np.tile(ylab,(1,M*N))   # 实现对矩阵整体复制

# 把 numpy.narray 4维矩阵 gapdpc_4d 保存为 npy 格式文件

savfn=f'PaCRISPR_uniref90_IND_26_26_pssm20_CBalanced{K}_{M}mer_{N}gapcbd_DA{M*N}_DL00.npz'
np.savez(savfn,datX=datX,ylab=ylab,idx=idx)
    
