# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 23:20:39 2025

    1，使用 Pearson Correlation Coefficient (PCC) 
    即：对增强的数据(20*20)计算PCC并排序 
        以第一张数据(等价covPSSM)为基准)
    
@author: wenjie
"""
# from lp_score import *
import numpy as np
from scipy.spatial.distance import cdist

def Sort_PCC_r0(dat3d):     # PCC    
    tmp2d = dat3d.reshape((-1,20*20))
    r = np.corrcoef(tmp2d)
    # 以gap=0为基准，和它相关系数最大的抽出若干da个，组合成增强倍数da
    rr0 = r[:,0]               # 抽取第0列，即gap=0的信息，与所有其它的相关系数
    idx = np.argsort(rr0)      # rr0升序排列，返回对应的索引
    new_dat3d = dat3d[idx[::-1],:,:]    # idx[::-1]对idx倒序操作
    return new_dat3d, idx[::-1]
    
     
cb = 10
da = 500

d = np.load(f'del1/PaCRISPR_uniref100_TRN_98_98_pssm20_CBalanced{cb}_top20mercovgap0_24cityblock_DA500_DL00.npz')

dat = d['datX']
ylab = d['ylab']
m,n,k,l = dat.shape
dd = np.zeros((m,n,k,l))

for i in range(m):
    print(i)
    si = dat[i]
    # dd[i] = Sort_Var(si)[0]
    dd[i] = Sort_PCC_r0(si)[0]
    # dd[i] = Sort_PCC_rall(si)[0]
    # dd[i] = Sort_LpScore(si)[0]    
    

newdd = dd[:,:da,:,:]          # 抽取与gap=0相关性最大的前 da 为增强倍数
datX = newdd
savfn = f'PaCRISPR_uniref100_TRN_98_98_pssm20_CBalanced{cb}_r0Des_top20mercovgap0_24cityblock_DA500_DL00.npz'
np.savez(savfn,datX=datX,ylab=ylab[:,:da])


