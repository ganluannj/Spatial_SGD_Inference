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
# Wdialist=loadList('D:/Googledrive/research/SGD/simulation_result/Kong/sources/Wdialist_N_900_Sumindex_31.npy')
# InvA0=scipy.sparse.load_npz('D:/Googledrive/research/SGD/simulation_result/Kong/sources/InvA_0.3_900.npz')
Wdialist=loadList('/home/g/gl238/sources/Wdialist_N_90000_Sumindex_31.npy')
InvA0=scipy.sparse.load_npz('/home/g/gl238/sgd/sgd2440/InvA_0.3_90000.npz')
# Wdialist=loadList('Wdialist_N_900_Sumindex_31.npy')
# InvA0=scipy.sparse.load_npz('InvA_0.3_900.npz')
#%%
Iter=2
N=90000
n=int(N**0.5)
Seeds=pd.DataFrame({'data':np.zeros(Iter), 'init':np.zeros(Iter)})
for index in range(Iter):
    Seeds.iloc[index,0]=random.randint(1,100000)
    Seeds.iloc[index,1]=random.randint(1,100000)
Seeds=Seeds.astype('int')
Strings='Seeds_'+str(Seeds.iloc[0,0])+'.csv'
Seeds.to_csv(Strings, index=None, header=True)
# Seeds=pd.read_csv(r'Seeds.csv', header=0)
Zeros=np.zeros(Iter)
Inital=pd.DataFrame({'beta0':Zeros, 'beta1':Zeros, 'beta2':Zeros, 'sigsq':Zeros, 
                     'rho':Zeros})
Result=pd.DataFrame({'beta0':Zeros, 'beta1':Zeros, 'beta2':Zeros, 'sigsq':Zeros, 'rho':Zeros})
Result2=pd.DataFrame({'beta0':Zeros, 'beta1':Zeros, 'beta2':Zeros, 'sigsq':Zeros, 'rho':Zeros})
sigmasq=1
beta=np.array([[0.5],
               [0.5],
               [-0.5]]) 
rho=0.3
LRSGD0=0.2
alpha=2/3
Burnin=int(N*0.2)
Sumindex=31
# number of random perturbed sgd estimate
CInumber=200+1
Zeros=np.zeros(Iter*CInumber)
SGDCI=pd.DataFrame({'beta0':Zeros, 'beta1':Zeros, 'beta2':Zeros, 'sigsq':Zeros, 'rho':Zeros})
#%%
# =============================================================================
# SGD Only process
# =============================================================================
# print('first function start!')
for index in range(Iter):    
    # create data
    np.random.seed(Seeds.iloc[index, 0])
    Samples=SpDatagen(N=N, beta=beta,InvA0=InvA0,sigmasq=sigmasq)
    # sgd process
    # initialize rho, theta, beta and sigmasq
    np.random.seed(Seeds.iloc[index, 1])
    # initialized Par
    Par=Init(rho, beta, sigmasq)
    # record initial value
    Inital.loc[index,:]=Par[0:5]    
    Parlist= np.zeros((N+1, 7))
    Parlist[0]=Par    
    LRSGD=LRSGD0*np.arange(1, N+1)**(-alpha)    
    for j in range(1, N+1):
        Parlist[j]=UPdate(Par=Parlist[j-1], Wdialist=Wdialist, Samples=Samples, 
                          Sumindex=Sumindex, liN=n, capN=N, LRSGD=LRSGD, J=j-1)   
    # remove the brunin
    Parlist=Parlist[Burnin+1:, 0:5]
    Result.loc[index,]=np.mean(Parlist, axis=0)
    Strings='sgdestimate_'+str(Seeds.iloc[0,0])+'.csv'
    Result.to_csv(Strings, index=None, header=True)
    # Inital.to_csv(r'initial.csv', index=None, header=True)
  
#%%
# =============================================================================
# SGD CI with different perturbation parameters
# =============================================================================
sd=1.0
Scale=sd*sd
Shape=1/Scale
for index in range(Iter):    
    # create data
    np.random.seed(Seeds.iloc[index, 0])
    Samples=SpDatagen(N=N, beta=beta,InvA0=InvA0,sigmasq=sigmasq)   
    # sgd process
    # initialize rho, theta, beta and sigmasq
    np.random.seed(Seeds.iloc[index, 1])
    Par=Init(rho, beta, sigmasq)
    Inital.loc[index,:]=Par[0:5]    
    # sgdCI process
    ParCIlist=np.zeros(((N+1), 7*CInumber))
    ParCI=np.tile(Par, CInumber)
    ParCIlist[0]=ParCI
    LRSGD=LRSGD0*np.arange(1, N+1)**(-alpha)    
    for j in range(1, N+1):
        ParCIlist[j]=UPdateCI2(ParCI=ParCIlist[j-1], CInumber=CInumber, 
                              Wdialist=Wdialist, Sumindex=Sumindex, Samples=Samples,  
                              Scale=Scale, Shape=Shape, liN=n, capN=N, LRSGD=LRSGD, J=j-1)
    
    # remove the burnin
    ParCIlist=ParCIlist[Burnin+1:,]
    for k in range(CInumber):
        Temp=ParCIlist[:,(k*7):(k*7+5)]
        SGDCI.loc[index*CInumber+k, ]=np.mean(Temp, axis=0)
for k in range(Iter):
    Result2.loc[k,]=SGDCI.loc[k*CInumber,]
Strings='sgdestimateci_'+str(Seeds.iloc[0,0])+'.csv'
Result2.to_csv(Strings, index=None, header=True)    

Droplist=[k*CInumber for k in range(Iter)]
SGDCI.drop(Droplist, 0, inplace=True)
Strings='SGDCI_sd_'+str(sd)+'_'+str(Seeds.iloc[0,0])+'.csv'
SGDCI.to_csv(Strings, index=None, header=True)
Strings='initial_'+str(Seeds.iloc[0,0])+'.csv'
Inital.to_csv(Strings, index=None, header=True)  

      
