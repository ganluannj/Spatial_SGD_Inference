import numpy as np
import random
from numpy import linalg as LA
from numpy.linalg import solve, norm
import pandas as pd
import math
from scipy.sparse import lil_matrix
from scipy.sparse import identity
#%%
# define a function that will return a neighborhood matrix
def norm_neigb (n):
    # this function will create a n^2 by n^2 normalized
    # neighborhood matrix for a n by n grid data, 
    # the neighborhood matrix was normlized
    # by deviding each element over the row sum
    # the grid data looks like:
    # 1        2        3        ...   n
    # n+1      n+2      n+3      ...   n+n
    # .        .        .        ...   . 
    # .        .        .        ...   .
    # .        .        .        ...   .
    # n(n-1)+1 n(n-1)+2 n(n-1)+3 ...  n^2
  
    # create a 0 matrix
    M = np.zeros((n**2, n**2))
    # consider only the left and up neighbor for each element
    # when updating the value in matrix
    
    for i in range(1,n):
        # these are in the first row 
        M[i-1, i]=M[i,i-1]=1
  
    for i in range (n, n**2):
        if (i+1)%n == 1:
            M[i-n,i]=M[i,i-n]=1
        else:
            M[i-n,i]=M[i, i-n]=1
            M[i-1,i]=M[i,i-1]=1
  
    # row sum of M
    Mrowsum=np.sum(M, 1)
  
    # divide M by row sum of M
    M = M/Mrowsum[:,None]

    return(M)

#%%
# define a function to load the list
def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()
# define a function to save list
def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)  

#%%
# define a function for generating samples
def Datagen(N, beta, InvA0, sigmasq):
    # this function generate data for one block
    # this function will generate a arrary of shape (N, 4)
    # the first three column is X, while the last column is y
    # N is the number of data in each block
    # beta is the beta parameter
    # InvA0 is the inverse matrix for spatial dependence
    # sigmasq is the sigma square used for generating data
    x0=np.repeat(1, N)
    x1=np.random.uniform(-1,1, size=N)
    x2=np.random.uniform(-1,1, size=N)
    X=np.concatenate((x0,x1,x2))
    X=np.asmatrix(X)
    X=np.reshape(X,(N,3), order='F')
    MU=X @ beta
    # generate y 
    y_tilde=np.random.normal(loc=0, scale=sigmasq**(1/2), size=N).T
    y_tilde=np.asmatrix(y_tilde)
    y_tilde=np.reshape(y_tilde,(N,1), order='F')
    y_tilde=y_tilde+MU
    y=InvA0@y_tilde
    # convert X to an array of shape (N,3)
    X=np.array(X)
    # convert y to an array of shape (N, 1)
    y=np.array(y)
    # combine X and y to get e new array of shape (N,4)
    # first three columns are X and last column is y
    DATA=np.append(X, y, axis=1)
    return (DATA)

# define a function to generate data for all blocks and combines
# them together
def DatagenAll(Blocks, N, beta, InvA0, sigmasq):
    # this function will generate the data for all the Blocks
    # and combine them together returns a big array of shape (Blocks*N, 4)
    # the first three column is X, while the last column is y
    # Blocks is the total number of Blocks
    # N is the number of data in each block
    # beta is the beta parameter
    # InvA0 is the inverse matrix for spatial dependence
    # sigmasq is the sigma square used for generating data
    Temp=[Datagen(N=N, beta=beta, InvA0=InvA0, sigmasq=sigmasq) 
     for block in range(Blocks)]
    return(np.concatenate(Temp, axis=0))

#%%  
# define a function to generate rholist    
def RholistGen(Rho, N):
    # Rho is the rho value
    # N is the length of the list
    rholist=np.ones(N)
    for k in range(1, N):
        rholist[k]=rholist[k-1]*Rho
    return (rholist)

#%%
# define a function to calculate Neighborhood mean
def NMean (J, capN, ylist):
    # J is the order of the data
    # capN is the total number of data in one block
    # this function returns mean of neighbors
    liN=int(capN**0.5)    
    Mean=np.mean([ylist[k] for k in [J-1, J+1, J-liN, J+liN] 
                  if (k>=0 and k<capN)])
    if(J%liN==0):
        Mean=np.mean([ylist[k] for k in [J+1, J-liN, J+liN] 
                  if (k>=0 and k<capN)])
    if ((J+1)%liN==0):
        Mean=np.mean([ylist[k] for k in [J-1, J-liN, J+liN] 
                  if (k>=0 and k<capN)])
    return(Mean)

#%%
# define a function up date betaest, rhoest, and sigsqest
def UPdate (Par, Wdialist, Samples, Sumindex, capN, LRSGD, J):
    # Par is an numpy array with 
    # Par[0:2] is betaest
    # Par[3] is sigsqest
    # Par[4] is rhoest
    # Par[5] is phiest
    # Par[6] is thetaest
    # Xlist is the list of x-values
    # Wdialist is the list for diagonal element of sum of W^k
    # ylist is the list of y value
    # capN is the total number of data in one block
    # LRSGD is list of learning rate
    # J is the j-th data in the whole datalist
    # fac is the perturbation factors with default value be 1
    
    # first calcuate the 
    betaest=Par[0:3]
    sigsqest=Par[3]
    rhoest=Par[4]
    phiest=Par[5]
    thetaest=Par[6]
    
    ylist=Samples[:,3]
    # generate rholist
    rholist=RholistGen(Rho=rhoest, N=Sumindex)
    # calcuate the jth diagonal element of A inverse
    InvAjj=sum(rholist*Wdialist[J%capN])
    
    Neighmean=NMean(J=J, capN=capN, ylist=ylist)
    
    # littlex=np.array(Xlist[J,:])[0]
    littlex=Samples[J, 0:3]
    sqpart=(ylist[J] - rhoest*Neighmean-sum(littlex*betaest)).item()
            
    derbeta= sqpart*littlex/sigsqest
    betaesttemp=betaest+LRSGD[J]*derbeta
    # update betaest in the end for get the new estimate of rho and sigmasq
        
    # update sigmasq/phi
    dersigsq=-1/(2*sigsqest) + 1/(2*sigsqest**2)*sqpart**2
    derphi=dersigsq*np.exp(phiest)
    phiest=(phiest+LRSGD[J]*derphi).item()
            
    # update rho/theta
    derrho=-(1/rhoest*InvAjj-1/rhoest)+1/sigsqest*sqpart*Neighmean
    dertheta=derrho*math.cos(thetaest)
    thetaest=(thetaest+LRSGD[J]*dertheta).item()
        
    # update rho, beta, and sigsq
    Par2=np.zeros(7)
    Par2[0:3]=betaesttemp
    Par2[3]=np.exp(phiest)
    Par2[4]=math.sin(thetaest)
    Par2[5]=phiest
    Par2[6]=thetaest
    return(Par2)

#%%
# define a update function specificly for CI
def UPdateinner(Par, yJ, xJ, LRJ, Neighmean, Wdia, Sumindex, fac):
    
    betaest=Par[0:3]
    sigsqest=Par[3]
    rhoest=Par[4]
    phiest=Par[5]
    thetaest=Par[6]    
    rholist=RholistGen(Rho=rhoest, N=Sumindex)
    # calcuate the jth diagonal element of A inverse
    InvAjj=sum(rholist*Wdia)    
    sqpart=(yJ - rhoest*Neighmean-sum(xJ*betaest)).item()        
    derbeta= sqpart*xJ/sigsqest
    betaesttemp=betaest+LRJ*derbeta*fac
    # update betaest in the end for get the new estimate of rho and sigmasq
        
    # update sigmasq/phi
    dersigsq=-1/(2*sigsqest) + 1/(2*sigsqest**2)*sqpart**2
    derphi=dersigsq*np.exp(phiest)
    phiest=(phiest+LRJ*derphi*fac).item()
            
    # update rho/theta
    derrho=-(1/rhoest*InvAjj-1/rhoest)+1/sigsqest*sqpart*Neighmean
    dertheta=derrho*math.cos(thetaest)
    thetaest=(thetaest+LRJ*dertheta*fac).item()
        
        
    # update rho, beta, and sigsq
    Par2=np.zeros(7)
    Par2[0:3]=betaesttemp
    Par2[3]=np.exp(phiest)
    Par2[4]=math.sin(thetaest)
    Par2[5]=phiest
    Par2[6]=thetaest
    return(Par2)

def UPdateCI (ParCI, CInumber, Wdialist, Sumindex, Samples, capN, LRSGD, J):
    # Par is an numpy array with a length of CInumber*7
    # inside a repeat of 7
    # Par[0:2] is betaest
    # Par[3] is sigsqest
    # Par[4] is rhoest
    # Par[5] is phiest
    # Par[6] is thetaest
    # Xlist is the list of x-values
    # Wdialist is the list for diagonal element of sum of W^k
    # ylist is the list of y value
    # capN is the total number of data in one block
    # LRSGD is list of learning rate
    # J is the j-th data in the whole datalist
    # fac is the perturbation factors with default value be 1

    ylist=Samples[:,3]
    Neighmean=NMean(J=J, capN=capN, ylist=ylist)
    xJ=Samples[J, 0:3]
    yJ=ylist[J]
    LRJ=LRSGD[J]
    Wdia=Wdialist[J]
    Parsplit=np.split(ParCI, CInumber)
    # faclist=np.ones(CInumber)
    faclist=np.append(1,np.random.gamma(shape=1.0, scale=1.0, size=(CInumber-1)))
    Temp=[UPdateinner(Par=Parsplit[k], yJ=yJ, xJ=xJ, 
                      LRJ=LRJ, Neighmean=Neighmean, Wdia=Wdia, 
                      Sumindex=Sumindex, fac=faclist[k]) 
          for k in range(CInumber)]
    return (np.hstack(Temp))


#%%
# write a function to divide a matrix into a 2 by 2 block matrix
# output its four blocks
def DivM(M):
    # M is the square matrix to be divided
    # evenly split M into 4 parts
    # return each of the 4 blocks
    # in the order of M[1,1], M[1,2], M[2,1], M[2,2]
    LEN=M.shape[0]
    HALF=int(LEN/2)
    return M[0:HALF, 0:HALF], M[0:HALF, HALF:LEN], M[HALF:LEN,0:HALF], M[HALF:LEN, HALF:LEN]

#%%
# write a function to combine four matrices into one matrix
# output one blocks
def ComM(A,B,C,D):
    # A, B, C, D are all square matrixes
    # combine them to a whole matrix M
    # M[1,1]=A, M[1,2]=B, M[2,1]=C, M[2,2]=D
    Temp1 = np.concatenate((A, B), 1)
    Temp2 = np.concatenate((C, D), 1)
    Temp=np.concatenate((Temp1, Temp2), 0)
    return(Temp)

#%%
# write a function to calcuate inverse with block matrix
# links for block matrix inverse
# https://en.wikipedia.org/wiki/Block_matrix
def InvBlock(M):
    # divide matrix into 4 blocks
    A, B, C, D = DivM(M)
    InvD=LA.inv(D)
    BInvD=B@InvD
    ABDC=A-BInvD@C
    InvABDC=LA.inv(ABDC)    
    TempA=InvABDC
    TempB=-InvABDC@BInvD
    TempC=-InvD@C@InvABDC
    TempD=InvD-TempC@BInvD
    Temp=ComM(TempA, TempB, TempC, TempD)
    return(Temp)

#%% write a function to check which subblock does one index belong
def Subblock(x,y,N):
    # x is the index for row
    # y is the index for column
    # N is number of rows of the big matrix
    # return number 1,2,3,4 indicating which subblock does 
    # this index belong
    # 1 means up left
    # 2 means up right
    # 3 means bottom left
    # 4 means bottom right
    halfN=int(N/2)
    if (x<halfN):
        if (y<halfN): return(1)
        else: return(2)
    else:
        if (y<halfN): return(3)
        else: return(4)

#%%
# write a function to assign 1s to each of the four subblocks 
def assignone (A, B, C, D, xassign, yassign, Nassign):
    halfN=int(Nassign/2)
    if Subblock(xassign, yassign, Nassign)==1: 
        A[xassign, yassign]=1
    if Subblock(xassign, yassign, Nassign)==2: 
        B[xassign, yassign-halfN]=1
    if Subblock(xassign, yassign, Nassign)==3: 
        C[xassign-halfN, yassign]=1
    if Subblock(xassign, yassign, Nassign)==4: 
        D[xassign-halfN, yassign-halfN]=1
    return A, B, C, D

#%%
# write a function for generating the subblock of the neighborhood matrix
def norm_neigbblock (n):
    # this function will create a n^2 by n^2 normalized
    # neighborhood matrix for a n by n grid data, 
    # the neighborhood matrix was normlized
    # by deviding each element over the row sum
    # the grid data looks like:
    # 1        2        3        ...   n
    # n+1      n+2      n+3      ...   n+n
    # .        .        .        ...   . 
    # .        .        .        ...   .
    # .        .        .        ...   .
    # n(n-1)+1 n(n-1)+2 n(n-1)+3 ...  n^2
    # then we divided into 4 even submatrix
    # return each of the 4 blocks
    # in the order of M[1,1], M[1,2], M[2,1], M[2,2]
  
    # create a 0 matrix
    Ntotal=n**2
    halfN=int(Ntotal/2)
    A = np.zeros((halfN, halfN))
    B = np.zeros((halfN, halfN))
    C = np.zeros((halfN, halfN))
    D = np.zeros((halfN, halfN))
    # consider only the left and up neighbor for each element
    # when updating the value in matrix
    
    for i in range(1,n):
        # these are in the first row 
        # M[i-1, i]=M[i,i-1]=1
        A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i-1, 
                             yassign=i, Nassign=Ntotal)
        A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i, 
                             yassign=i-1, Nassign=Ntotal)
 
    for i in range (n, n**2):
        if (i+1)%n == 1:
            # M[i-n,i]=M[i,i-n]=1
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i-n, 
                             yassign=i, Nassign=Ntotal)
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i, 
                             yassign=i-n, Nassign=Ntotal)
        else:
            # M[i-n,i]=M[i, i-n]=1
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i-n, 
                             yassign=i, Nassign=Ntotal)
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i, 
                             yassign=i-n, Nassign=Ntotal)
            # M[i-1,i]=M[i,i-1]=1
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i-1, 
                             yassign=i, Nassign=Ntotal)
            A, B, C, D=assignone(A=A, B=B, C=C, D=D, xassign=i, 
                             yassign=i-1, Nassign=Ntotal)
  
    # row sum
    MrowsumAB=np.sum(A, 1)+np.sum(B,1)
    MrowsumCD=np.sum(C, 1)+np.sum(D,1)
  
    # divide M by row sum of M
    # M = M/Mrowsum[:,None]
    A = A/MrowsumAB[:, None]
    B = B/MrowsumAB[:, None]
    C = C/MrowsumCD[:, None]
    D = D/MrowsumCD[:, None]

    return A, B, C, D

#%%
# define a function that will return a neighborhood matrix
# return a sparse matrix
def spnorm_neigb (n):
    # this function will create a n^2 by n^2 normalized
    # neighborhood matrix for a n by n grid data, 
    # the neighborhood matrix was normlized
    # by deviding each element over the row sum
    # the grid data looks like:
    # 1        2        3        ...   n
    # n+1      n+2      n+3      ...   n+n
    # .        .        .        ...   . 
    # .        .        .        ...   .
    # .        .        .        ...   .
    # n(n-1)+1 n(n-1)+2 n(n-1)+3 ...  n^2
  
    # create a 0 matrix
    # M = np.zeros((n**2, n**2))
    M = lil_matrix((n**2, n**2))
    # consider only the left and up neighbor for eacMh element
    # when updating the value in matrix
    
    for i in range(1,n):
        # these are in the first row 
        M[i-1, i]=M[i,i-1]=1
  
    for i in range (n, n**2):
        if (i+1)%n == 1:
            M[i-n,i]=M[i,i-n]=1
        else:
            M[i-n,i]=M[i, i-n]=1
            M[i-1,i]=M[i,i-1]=1
    Mrowsum=np.sum(M, 1)
    M=M/4        
    # row sum of M
    # get the row sum be 3
    index3=np.where(Mrowsum==3)[0]
    M[index3,]=M[index3,]*4/3
    # get the row sum be 2
    index2=np.where(Mrowsum==2)[0]
    M[index2,]=M[index2,]*2
  

    return(M)


#%%
# define a function for data generation with sparse matrix
# define a function for generating samples
def SpDatagen(N, beta, InvA0, sigmasq):
    # this function generate data for one block
    # this function will generate a arrary of shape (N, 4)
    # the first three column is X, while the last column is y
    # N is the number of data in each block
    # beta is the beta parameter
    # InvA0 is the inverse matrix for spatial dependence
    # InvA0 is the matrix in the sparse format
    # sigmasq is the sigma square used for generating data
    x0=np.repeat(1, N)
    x1=np.random.uniform(-1,1, size=N)
    x2=np.random.uniform(-1,1, size=N)
    X=np.concatenate((x0,x1,x2))
    X=np.asmatrix(X)
    X=np.reshape(X,(N,3), order='F')
    MU=X @ beta
    # generate y 
    y_tilde=np.random.normal(loc=0, scale=sigmasq**(1/2), size=N).T
    y_tilde=np.asmatrix(y_tilde)
    y_tilde=np.reshape(y_tilde,(N,1), order='F')
    y_tilde=y_tilde+MU
    y=InvA0.dot(y_tilde)
    # convert X to an array of shape (N,3)
    X=np.array(X)
    # convert y to an array of shape (N, 1)
    y=np.array(y)
    # combine X and y to get e new array of shape (N,4)
    # first three columns are X and last column is y
    DATA=np.append(X, y, axis=1)
    return (DATA)

#%%
# create a function to initialize the parameter
def Init(rho, beta, sigmasq):
    # initialized Par
    Par=np.zeros(7)
    # Par[0:3] is betaest, Par[3] is sigsqest, Par[4] is rhoest
    # Par[5] is phiest, Par[6] is thetaest       
    # initialize rho
    Par[4]=np.random.uniform(0,max(min(1,2*rho),-1),1)[0]
    # initialize theta 
    Par[6]=math.asin(Par[4])
    # initialize beta
    Par[0:3]=np.array([np.random.uniform(0,2*beta[0][0],1)[0],
                   np.random.uniform(0,2*beta[1][0],1)[0],
                   np.random.uniform(0,2*beta[2][0],1)[0]])
    # initialize sigmasq
    Par[3]=np.random.uniform(sigmasq/2,sigmasq*2,1)
    # initialize phi
    Par[5]=math.log(Par[3])
    return (Par)

#%%
# define a update function specificly for CI
# use different perturbation parameter for beta, rho, and sigmasquare
def UPdateinner2(Par, yJ, xJ, LRJ, Neighmean, Wdia, Sumindex, fac1, fac2):
    
    betaest=Par[0:3]
    sigsqest=Par[3]
    rhoest=Par[4]
    phiest=Par[5]
    thetaest=Par[6]    
    rholist=RholistGen(Rho=rhoest, N=Sumindex)
    # calcuate the jth diagonal element of A inverse
    InvAjj=sum(rholist*Wdia)    
    sqpart=(yJ - rhoest*Neighmean-sum(xJ*betaest)).item()        
    derbeta= sqpart*xJ/sigsqest
    betaesttemp=betaest+LRJ*derbeta*fac1
    # update betaest in the end for get the new estimate of rho and sigmasq
        
    # update sigmasq/phi
    dersigsq=-1/(2*sigsqest) + 1/(2*sigsqest**2)*sqpart**2
    derphi=dersigsq*np.exp(phiest)
    phiest=(phiest+LRJ*derphi*fac1).item()
            
    # update rho/theta
    derrho=-(1/rhoest*InvAjj-1/rhoest)+1/sigsqest*sqpart*Neighmean
    dertheta=derrho*math.cos(thetaest)
    thetaest=(thetaest+LRJ*dertheta*fac2).item()
        
        
    # update rho, beta, and sigsq
    Par2=np.zeros(7)
    Par2[0:3]=betaesttemp
    Par2[3]=np.exp(phiest)
    Par2[4]=math.sin(thetaest)
    Par2[5]=phiest
    Par2[6]=thetaest
    return(Par2)
#%%
# define a function to create CI with different perturbation factor
# for rho and for beta, sigmasq
def UPdateCI2 (ParCI, CInumber, Wdialist, Sumindex, Samples,  
               Scale, Shape, capN, LRSGD, J):
    # Par is an numpy array with a length of CInumber*7
    # inside a repeat of 7
    # Par[0:2] is betaest
    # Par[3] is sigsqest
    # Par[4] is rhoest
    # Par[5] is phiest
    # Par[6] is thetaest
    # Xlist is the list of x-values
    # Wdialist is the list for diagonal element of sum of W^k
    # ylist is the list of y value
    # capN is the total number of data in one block
    # LRSGD is list of learning rate
    # J is the j-th data in the whole datalist
    # Scale and Shape are the Scale and Shape parameters for 
    # Gamma distribution for perturbation for rho    

    ylist=Samples[:,3]
    Neighmean=NMean(J=J,capN=capN, ylist=ylist)
    xJ=Samples[J, 0:3]
    yJ=ylist[J]
    LRJ=LRSGD[J]
    Wdia=Wdialist[J%capN]
    Parsplit=np.split(ParCI, CInumber)
    # faclist=np.ones(CInumber)
    faclist1=np.append(1,np.random.gamma(shape=1.0, scale=1.0, size=(CInumber-1)))
    faclist2=np.append(1,np.random.gamma(shape=Shape, scale=Scale, size=(CInumber-1)))
    Temp=[UPdateinner2(Par=Parsplit[k], yJ=yJ, xJ=xJ, 
                      LRJ=LRJ, Neighmean=Neighmean, Wdia=Wdia, 
                      Sumindex=Sumindex, fac1=faclist1[k], fac2=faclist2[k]) 
          for k in range(CInumber)]
    return (np.hstack(Temp))
