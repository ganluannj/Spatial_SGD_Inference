
#' @name SAR
#' @title parameter estimation and confidence interval construction for spatial autoregressive (SAR) model with SGD
#' @description This for applying stochastic gradient descent (SGD) for parameter estimation of
#' SAR model. \code{SGD} apply SGD for parameter estimation and \code{SGDCI}

SARSGD<-function(X, Y, W, lr0, C=2/3, burnin=0.2, K=30){
  # X is the covariates matrix wtih size of N*p
  # Y is the matrix for response variable with size of N*1
  # W is the neighborhood matrix
  # lr0 is the initial learning rate
  # and learning rate for kth data is lr0*C^(-k)
  # burnin is the proportion of interation discard
  # get the sample size
  N = dim(X)[1]
  p = dim(X)[2]
  # get Wdialist
  Wdialist=Powerdiagnoal(W, K)
  # initialize parameters
  set.seed(10)
  beta0=runif(p)
  rho0=runif(1)
  sigsq0 = runif(1,0,1)
  phi0=log(sigsq0)
  theta0=asin(rho0)
  Par0=c(beta0, sigsq0, rho0, phi0, theta0)
  Parlist=matrix(rep(0, (N+1)*(p+4)), nrow=p+4)
  # set the first column of Parlist be Par0
  Parlist[,1]=Par0
  LRSGD=lr0*(1:N)^(-C)
  # SGD process
  for (j in 1:N){
    LR=LRSGD[j]
    Parlist[,j+1]=Update(Par=Parlist[,j], W=W, Wdialist = Wdialist,
                        X=X, Y=Y, LR=LR, j=j, K=K)
  }

  # remove the burnin
  Nburn=as.integer(N*burnin)
  Parlist=Parlist[,(Nburn+2):(N+1)]
  Estimate=rowMeans(Parlist)
  parameters=c()
  for (i in 0:(p-1)){
    parameters=c(parameters, paste0('beta',toString(i)))
  }
  parameters=c(parameters, 'sigmasq', 'rho')
  Result=data.frame(parameters=parameters, estimate=Estimate[1:(p+2)])
  return (Result)
}

SARSGDCI<-function(X, Y, W, lr0, C=2/3, burnin=0.2, K=30, B=200,
                   per_dis='Poi',CI.level=0.95, CI.method='sd'){

  # get the sample size
  N = dim(X)[1]
  p = dim(X)[2]
  # get Wdialist
  Wdialist=Powerdiagnoal(W, K)
  # initialize parameters
  set.seed(10)
  beta0=runif(p)
  rho0=runif(1)
  sigsq0 = runif(1,0,1)
  phi0=log(sigsq0)
  theta0=asin(rho0)
  Par0=c(beta0, sigsq0, rho0, phi0, theta0)
  ParCI0=rep(Par0,B+1)
  ParCIlist=matrix(rep(0, (N+1)*(p+4)*(B+1)), ncol=N+1)
  ParCIlist[,1]=ParCI0
  LRSGD=lr0*(1:N)^(-C)
  # SGDCI process
  for (j in 1:N){
    LR=LRSGD[j]
    if(per_dis=='Poi'){
      faclist=c(1, rpois(B, 1))
    }
    if (per_dis=='exp'){
      faclist=c(1,rexp(B,1))
    }
    ParCIlist[,j+1]=Updateper(ParCI=ParCIlist[,j], W=W, Wdialist = Wdialist,
                         X=X, Y=Y, LR=LR, j=j, K=K, faclist = faclist)
  }
  # remove the burnin
  Nburn=as.integer(N*burnin)
  ParCIlist=ParCIlist[,(Nburn+2):(N+1)]
  # get the estimate
  Estimatelist=ParCIlist[1:(p+4),]
  Estimate=rowMeans(Estimatelist)
  # remove the last two estimates (phi and theta)
  Estimate=Estimate[1:(p+2)]
  # get the ParCIlist for CI construction
  ParCIlist=ParCIlist[(p+5):((p+4)*(B+1)),]
  # first get the mean for each perturbed estiamte
  ParCImean=rowMeans(ParCIlist)
  ParCImean=matrix(ParCImean, nrow=p+4, byrow = FALSE)
  # getrid of the last two rows
  ParCImean=ParCImean[1:(p+2),]
  if (CI.method=='sd'){
    # get the sample standard deviation from perturbed estimates
    SD=apply(X=ParCImean, MARGIN = 1, FUN='sd')
    # get z score
    z=qnorm(1-(1-CI.level)/2)
    # lower bound
    lower=Estimate - z*SD
    # upper bound
    upper=Estimate + z*SD
  }

  if (CI.method=='quantile'){
    lo = (1-CI.level)/2
    up = CI.level+lo
    # lower bound
    lower = 2*Estimate - apply(X=ParCImean,
                              MARGIN = 1, FUN=function(x) Quantile(x,up))
    # upper bound
    upper = 2*Estimate - apply(X=ParCImean,
                              MARGIN = 1, FUN=function(x) Quantile(x,lo))
  }

  # create a vector for parameter names
  parameters=c()
  for (i in 0:(p-1)){
    parameters=c(parameters, paste0('beta',toString(i)))
  }
  parameters=c(parameters, 'sigmasq','rho')
  SGDEstimate=data.frame(parameters=parameters, estimate=Estimate)
  CI=data.frame(parameters=parameters, lowbd=lower, upbd=upper, conf_level = rep(CI.level, p+2))
  return(list(SGDEstimate, CI))
}



###########################################################
##### function to update estimate #########
############################################################
Update<-function (Par, W, Wdialist, X, Y, LR, j, K=K, fac=1){
  # par is a vector with length p+4
  # par[1:p] is betaest
  # par[p+1] is sigsqest
  # par[p+2] is rhoest
  # par[p+3] is phiest
  # par[p+4] is thetaest
  # W neighborhood matrix
  # Wdialist is a list of vectors,
  # with each vector is the ith diagnoal element of W^k
  # X is the dataset of size N*p
  # Y is a response variable of size N*1
  # LR is the learning rate
  # j is the jth data of the whole dataset
  # K is the number of terms used to calcuate dignoal element of A^(-1)
  p=length(Par)-4
  betaest=Par[1:p]
  sigsqest=Par[p+1]
  rhoest=Par[p+2]
  phiest=Par[p+3]
  thetaest=Par[p+4]

  # generate rholist
  rholist=Rholistgen(rhoest, K)
  # calculate the jth diagnoal element of A^(-1)
  # InvAjj=sum(rholist*Wdialist[[j]])
  InvAjj = InvAdig(rholist, Wdialist, j)

  # calculate weightmean of neighbors
  Neighmean = W[j,]%*%Y

  # update beta
  littlex=X[j,]
  sqpart=(Y[j] - rhoest*Neighmean-sum(littlex*betaest))[1,1]
  derbeta= sqpart*littlex/sigsqest
  betaesttemp=betaest+LR*derbeta*fac

  # update sigmasq/phi
  dersigsq=-1/(2*sigsqest) + 1/(2*sigsqest**2)*sqpart**2
  derphi=dersigsq*exp(phiest)
  phiest=phiest+LR*derphi*fac

  # update rho/theta
  derrho=-(1/rhoest*InvAjj-1/rhoest)+1/sigsqest*sqpart*Neighmean
  dertheta=derrho*cos(thetaest)
  thetaest=thetaest+LR*dertheta*fac

  # update rho, beta, and sigsq
  Par2=rep(0,p+4)
  Par2[1:p]=betaesttemp
  Par2[p+1]=exp(phiest)
  Par2[p+2]=sin(thetaest)
  Par2[p+3]=phiest
  Par2[p+4]=thetaest
  return (Par2)

}

###########################################################
##### function to update perturbed estimate #########
############################################################
Updateper<-function(ParCI, W, Wdialist, X, Y, LR, j, K=K, faclist){
  # faclist is the list of perturbation parameters, faclist[1]=1
  # Q=length(faclist)
  # Par is a vector containing all parameters
  # ParCI is a vector containing Q Par
  # with first Par corresponds to regular SGD
  # and the rest correspond to perturbed estimates
  Q=length(faclist)
  Qk=length(ParCI)
  k=as.integer(Qk/Q) # size of the parameter
  ParCI2=rep(0,Qk)
  for (i in 1:Q){
    ParCI2[(k*(i-1)+1):(k*i)]=Update(Par=ParCI[(k*(i-1)+1):(k*i)],W=W, X=X, Y=Y,
                                     Wdialist=Wdialist, LR=LR, j=j, K=K, fac=faclist[i])
  }
  return (ParCI2)
}


###########################################################
##### function to generate digonal element of W^k #########
############################################################
Powerdiagnoal<-function(W, K){
  # W is a neighborhood matrix of size N by N
  # K is an integer
  # let I be the identity matrix of size N by N
  # we have to calculate I  W  W^2  W^3 ... W^(K-1)
  # it returns a list of vectors, the list is of length K
  # the ith element of the diagonal elements of W^(i-1)
  N = length(W[1,])
  L = list()
  L[[1]]=rep(1,N)
  temp = diag(N)
  for (i in 2:K){
    temp = temp%*%W
    L[[i]]=diag(temp)
  }
  return (L)
}

###########################################################
##### function to generate a vector of rho^k #########
############################################################
Rholistgen<-function(Rho, K){
  # this function will generate a vector of length N
  # the ith element of the list is Rho^(i-1)
  V<-rep(1,K)
  for (i in 2:K){
    V[i]=Rho*V[i-1]
  }
  return (V)
}

InvAdig<-function(Rholist, Wdialist,j){
  K=length(Rholist)
  Sum=0
  for (i in 1:K){
    Sum=Sum+Rholist[i]*Wdialist[[i]][j]
  }
  return (Sum)
}


Quantile<-function(x, q){
  # order x by accending order
  # get the upper q quantile of x
  # for example 5% quantile, is the top 5% smallest number in x
  x<-x[order(x)]
  N=as.integer(length(x)*q)
  return(x[N])
}


Neigb<-function(N){
  W=Matrix(nrow=N*N,ncol=N*N,data=0,sparse=TRUE)
  for(i in 2:N){
    W[i-1,i]=W[i,i-1]=1
  }
  for (i in (N+1):(N*N)){
    if (i%%N==1){
      W[i-N,i]=W[i,i-N]=1
    }
    else{
      W[i-N,i]=W[i,i-N]=1
      W[i-1,i]=W[i,i-1]=1
    }
  }
  return (W/rowSums(W))
}

###########################################################
N=100
rho=0.3
beta0=0.5
beta1=0.5
beta2=-0.5
sigsq=1
W=Neigb(N**0.5)

Inverse<-function(W, rho,K=50){
  N=as.integer(length(W)**0.5)
  Rholist<-Rholistgen(rho, K)
  library(Matrix)
  Temp=sparseMatrix(i=1:N, j=1:N, x=rep(1,N))
  Result=Temp
  for (i in 2:K){
    Temp=Temp%*%W
    Result=Result+Temp*Rholist[i]
  }
  return (Result)
}



A=diag(N)-rho*W
InvA=Inverse(W, rho)
x0=rep(1,N)
set.seed(10)
x1=runif(N,-1,1)
x2=runif(N,-1,1)
X=matrix(c(x0,x1,x2),nrow=N, byrow = FALSE)
beta=matrix(c(beta0,beta1,beta2),nrow = 3)
Xbeta=X%*%beta
set.seed(10)
ytilde=rnorm(N, 0, sigsq^(1/2))
ytilde=matrix(ytilde,nrow=N)
ytilde=ytilde+Xbeta
Y=InvA%*%ytilde
#
S1=Sys.time()
SARSGD(X, Y, W, lr0=0.2, C=2/3, burnin=0.2, K=30)
Sys.time()-S1

S2=Sys.time()
SARSGDCI(X, Y, W, lr0=0.2)
Sys.time()-S2
##############################################################
###### Constructing W for all data points are neighbors ######
##############################################################
N=9
ln=matrix(rep(1,N),nrow=1)
W=(t(ln)%*%ln-diag(N))/(N-1)
W_=Neigb(N**0.5)

Statdist<-function(W0){
  # find the stationary distribution for transition matrix W
  N<-length(W0[1,])
  I_W<-diag(N)-W0
  A<-rbind(I_W,rep(1,N))
  b=matrix(c(rep(0,N),1),ncol=1)
  ATA=t(A)%*%A
  InvATA=solve(ATA)
  ATb=t(A)%*%b
  return (InvATA%*%ATb)
}

Limit<-function(W0){
  W5=W0%*%W0%*%W0%*%W0%*%W0
  W10=W5%*%W5
  W20=W10%*%W10
  W40=W20%*%W20
  W80=W40%*%W40
  W160=W80%*%W80
  W200=W160%*%W40
  W400=W200%*%W200
  return (W200)
}

Limitcheck<-function(W0,k=1){
  # W0 is the limit distribution
  # k the step jump k=1 or k=2
  N=length(W0[1,])
  i=1
    while (i+k<=N){
      if (sum(abs(W0[i,]-W0[i+k,])>0.0001)>0){
        return (FALSE)
      }
      i=i+1
    }
  return (TRUE)
}


Statdist(W_)
Limit(W_)
