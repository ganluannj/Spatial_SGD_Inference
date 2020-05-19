import numpy as np
import random
from numpy import linalg as LA
import pandas as pd
import math
import time
from functions import *
from scipy.sparse import lil_matrix
from scipy.sparse import identity
from numpy.linalg import solve, norm
import scipy.sparse

#%%
N=90000
n=int(N**0.5)
Sumindex=81
W2=spnorm_neigb (n)
#%%
Wdialist=[[]]*N
for j in range(N):
    Wdialist[j]=np.append(Wdialist[j], 1)
temp=W2
for i in range(1, Sumindex):
    for j in range(N):
        Wdialist[j]=np.append(Wdialist[j], temp[j,].toarray()[0,j])
    temp=temp.dot(W2)
    if i==30:
        saveList(Wdialist, 'Wdialist_N_90000_Sumindex_31.npy')
saveList(Wdialist, 'Wdialist_N_90000_Sumindex_81.npy')    
#Wdialist2=loadList('Wdialist_N_90000_Sumindex_31.npy')

#%%
rho=0.3
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_0.3_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

#%%
rho=0.2
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_0.2_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
#%%
rho=0.7
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_0.7_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
#%%
rho=0.8
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_0.8_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
#%%
rho=-0.3
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_-0.3_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')
#%%
rho=-0.7
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_-0.7_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

#%%
rho=-0.2
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_-0.2_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')

#%%
rho=-0.8
rholist=RholistGen(rho, Sumindex)
spInvA=lil_matrix((n**2, n**2))
temp=identity(n*n)
for i in range(Sumindex):
    spInvA=spInvA+temp*rholist[i]
    temp=temp.dot(W2)
scipy.sparse.save_npz('InvA_-0.8_90000.npz', spInvA)
# scipy.sparse.load_npz('/tmp/sparse_matrix.npz')



