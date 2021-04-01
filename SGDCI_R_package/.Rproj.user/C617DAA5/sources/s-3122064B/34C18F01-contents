#' @name QuanregSGD
#' @title parameter estimation and confidence interval construction for quantile regression with SGD
#' @description This for applying stochastic gradient descent (SGD) for parameter estimation of
#' linear regression model. \code{QuanregSGD} apply SGD for parameter estimation and \code{QuanregSGDCI}
#' apply SGD for parameter estimation and also construct confidence intervals for estimates.
#' @details Parameter estimation and confidence interval construction are based on Example 4 of Yixin Fang, Jinfeng Xu, and Lei Yang.
#' Online bootstrap confidence intervals for the stochastic gradient descent estimator.
#' The Journal of Machine Learning Research, 19(1):3053-3073, 2018.
#'
#' @param x matrix for independent variable, does not include the constant column
#' @param y response variable
#' @param tau quantile, tau quantile of Y given X is X^T*beta
#' @param lr0 lr0 and C are used to calcluate learning rate, learning rate for k-th data point is
#' calculated as lr_k = lr_0*k **(-C)
#' @param C see description for lr0 and C is between 0.5 and 1
#' @param burnin a number between 0 and 1; this represent the proportion of estimates
#' that are not used for calculating final estimate
#' @param B number of perturbed estimates generated
#' @param Per_dis the distribution to generate perturbation parameter
#' it takes the following values, 'Poi': Poission distribution, Poi(1)
#' 'exp', exponential distribution, exp(1)
#' @param CI.level confidence level for constructing confidence interval
#' @param CI.method the method to construct confidence interval
#' it takes one of the following values 'sd' or 'quantile'
#' 'sd' means using sample standard deviation of perturbed
#' estimates to estimate the standard deviation of SGD estimate
#' 'quantile' means using the upper alpha/2 and lower alpha/2
#' quantile of the perturbed estimates to construct
#' confidence interval
#'
#' @return
#' \strong{\code{QuanregSGD}} \cr
#' This function will return estimated coefficients for quantile regression. \cr
#' \strong{\code{QuanregSGDCI}} \cr
#' This function will return estimated coefficients and confidence intervals for quantile regression. \cr
#'
#'
#' @examples
#' \dontrun{
#' we tried a 90% quantile regression
#' we generate our data by first calculate the expcted 90% quantile as
#' beta0 + beta1*x1 + beta2*x2
#' draw y from a uniform distribution (range 1) with 90% quantile equals to the
#' calculated expected value
#' set.seed(10)
#' N<-50000
#' beta_0<-0.1
#' beta_1<-0.1
#' beta_2<--0.1
#' SD<-1
#' x1<-rnorm(N, 0, 1)
#' x2<-rnorm(N,0,1)
#' x<-matrix(c(x1,x2), nrow=N, byrow = FALSE)
#' Quan.90<-beta_0+beta_1*x1+beta_2*x2
#' Lower<-Quan.90-0.9
#' Upper<-Quan.90+0.1
#' y<-runif(N, Lower, Upper)
#' QuanregSGD(x,y, tau=0.9, lr0=0.1, C=2/3, burnin=0.2)
#' QuanregSGDCI(x,y,tau=0.9,lr0=0.1, C=2/3, burnin=0.2, B=200, CI.method = 'sd', per_dis = 'exp')
#' quantile regression using quantreg package
#' library(quantreg)
#' rq(y ~ x,  tau = 0.9)
#' }
#'
#' @references Yixin Fang, Jinfeng Xu, and Lei Yang.
#' Online bootstrap confidence intervals for the stochastic gradient descent estimator.
#' The Journal of Machine Learning Research, 19(1):3053-3073, 2018.
#' @export

QuanregSGD<-function(X,Y,tau, lr0, C=2/3, burnin=0.2){
  # X is the design matrix, covariates, it does not contain the
  # column of 1s
  # y is the dependent variable, dimension: p*1
  # tau quantile
  # learning rate for the k-th data point is lr0*k^{-C}
  # lr0 and C are for specify learning rate
  # burnin: the proportion of data that will discard

  # get the number of rows of X, ie number of datapoints
  N=nrow(X)
  # add the constant column of 1 to X
  X<-matrix(c(rep(1,N), X), nrow=N, byrow = FALSE)
  # get the dimension of features, including the constant column
  p=ncol(X)
  # get the number of burnin runs
  Nburnin=as.integer(N*burnin)

  # initialize betahat and betasum
  # initialize betahat from uniform distribution (-1,1)
  betahat=runif(p, -1, 1)
  betasum=rep(0,p)

  # generate learning rate
  LR=lr0*seq(1, N)^(-C)

  for (i in 1:N){
    # learning rate for this step
    lr=LR[i]
    # update betahat
    x=X[i,]
    y=Y[i]
    xbeta=(t(x)%*%betahat)[1,1]
    Delta=(tau-(y-xbeta<0))*x
    betahat=betahat+lr*Delta
    # update betasum when i is larger than Nburnin
    if (i > Nburnin){
      betasum = betasum + betahat
    }
  }

  betafinal=betasum/(N-Nburnin)
  # create a vector for parameter names
  parameters=c()
  for (i in 0:(p-1)){
    parameters=c(parameters, paste0('beta',toString(i)))
  }
  Result=data.frame(parameters=parameters, estimate=betafinal)
  return(Result)
}

#' @rdname QuanregSGD
#' @export

QuanregSGDCI<-function(X, Y, tau, lr0, C=2/3, burnin=0.2, B=200, per_dis='Poi', CI.level=0.95, CI.method='sd'){
  # X is the design matrix, covariates, it does not contain the
  # column of 1s
  # y is the dependent variable, dimension: p*1
  # learning rate for the k-th data point is lr0*k^{-alpha}
  # lr0 and alpha are for specify learning rate
  # burnin: the proportion of data that will discard
  # B is the number of perturbed estimates generated
  # Per_dis: the distribution to generate perturbation parameter
  # it takes the following values, 'Poi': Poission distribution, Poi(1)
  # 'exp', exponential distribution, exp(1)
  # CI.level is the confidence level
  # CI.method, the method to construct confidence interval
  # it takes one of the following values 'sd' or 'quantile'
  # 'sd' means using sample standard deviation of perturbed
  # estimates to estimate the standard deviation of SGD estimate
  # 'quantile' means using the upper alpha/2 and lower alpha/2
  # quantile of the perturbed estimates to construct
  # confidence interval

  # get the number of rows of X, ie number of datapoints
  N=nrow(X)
  # add the constant column of 1 to X
  X<-matrix(c(rep(1,N), X), nrow=N, byrow = FALSE)
  # get the dimension of features, including the constant column
  p=ncol(X)
  # get the number of burnin runs
  Nburnin=as.integer(N*burnin)

  # initialize betahatci and betacisum
  # both betahatci and betacisum matrix of size p*(B+1)
  # the first column is for regular SGD and the rest are for perturbed estimates
  # initialize betahat from uniform distribution (-1,1)
  betahat=runif(p, -1, 1)
  betahatci=matrix(rep(betahat, B+1), nrow=p, byrow=FALSE) # p*(B+1)
  betacisum=matrix(rep(0, p*(B+1)), nrow=p, byrow=FALSE) # p*(B+1)
  # generate learning rate
  LR=lr0*seq(1, N)^(-C)

  for (i in 1:N){
    lr=LR[i]
    x=X[i,] # p*1
    y=matrix(rep(Y[i], B+1), nrow=1) # 1*(B+1)
    xbeta=t(x)%*%betahatci # 1*(B+1)
    y_xbeta=y-xbeta # 1*(B+1)
    tau_Ind=tau-(y_xbeta<0)
    Delta= x %*% tau_Ind # p*(B+1)
    if (per_dis=='Poi'){
      lrvec=lr*c(1, rpois(B, 1)) # 1*(B+1)
    }
    if (per_dis=='exp'){
      lrvec=lr*c(1, rexp(B, 1))
    }
    # lrvec=lr*rep(1, (B+1))
    lrDelta=sweep(x=Delta, MARGIN = 2, STATS = lrvec, FUN = '*')
    betahatci = betahatci + lrDelta
    if (i > Nburnin){
      betacisum = betacisum + betahatci
    }
  }
  # average to get the final result
  betacifinal = betacisum/(N-Nburnin) # p*(B+1)
  # get the sgd estimate
  betahat = betacifinal[,1]
  # construct the confidence interval
  # the confidence interval constructed is for each coefficient
  # not familywise confidence interval
  if (CI.method=='sd'){
    # get the sample standard deviation from perturbed estimates
    SD=apply(X=betacifinal[, 2:(B+1)], MARGIN = 1, FUN='sd')
    # get z score
    z=qnorm(1-(1-CI.level)/2)
    # lower bound
    lower=betahat - z*SD
    # upper bound
    upper=betahat + z*SD
  }
  if (CI.method=='quantile'){
    lo = (1-CI.level)/2
    up = CI.level+lo
    # lower bound
    lower = 2*betahat - apply(X=betacifinal[, 2:(B+1)],
                              MARGIN = 1, FUN=function(x) Quantile(x,up))
    # upper bound
    upper = 2*betahat - apply(X=betacifinal[, 2:(B+1)],
                              MARGIN = 1, FUN=function(x) Quantile(x,lo))
  }
  # create a vector for parameter names
  parameters=c()
  for (i in 0:(p-1)){
    parameters=c(parameters, paste0('beta',toString(i)))
  }
  Estimate=data.frame(parameters=parameters, estimate=betahat)
  CI=data.frame(parameters=parameters, lowbd=lower, upbd=upper, conf_level = rep(CI.level, p))
  return(list(Estimate, CI))
}

Quantile<-function(x, q){
  # order x by accending order
  # get the upper q quantile of x
  # for example 5% quantile, is the top 5% smallest number in x
  x<-x[order(x)]
  N=as.integer(length(x)*q)
  return(x[N])
}

