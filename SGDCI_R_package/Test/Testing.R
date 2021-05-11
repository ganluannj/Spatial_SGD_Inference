###########################################################
##  Test for linear regression SGD  #######################
###########################################################
set.seed(100)
# library(SGDCI)
N<-10000
beta_0<-0.2
beta_1<-0.1
beta_2<--0.1
SD<-1
x1<-rnorm(N, 0, 1)
x2<-rnorm(N,0,1)
x<-matrix(c(x1,x2), nrow=N, byrow = FALSE)
y<-beta_0+beta_1*x1+beta_2*x2+rnorm(N, 0, SD)
# direct apply linear regression model
lnmodel<-lm(y~x)
lnmodel

linearSGD(x,y,lr0=0.1, C=2/3, burnin=0.2)
linearSGDCI(x,y,lr0=0.1, C=2/3, burnin=0.2, B=200, CI.method = 'sd', per_dis = 'exp')


###########################################################
##  Test for logistic regression SGD  #######################
###########################################################
set.seed(10)
N<-500000
beta_0<-0.1
beta_1<-0.1
beta_2<--0.1
SD<-1
x1<-rnorm(N, 0, 1)
x2<-rnorm(N,0,1)
x<-matrix(c(x1,x2), nrow=N, byrow = FALSE)
EXP<-exp(beta_0+beta_1*x1+beta_2*x2)
P<- EXP/(1+EXP)
yGLM<-ifelse(runif(N)<P,1,0)
# yGLM<-rbinom
ySGD<-ifelse(yGLM==1, 1, -1)
GLM<-glm(yGLM ~ x1 + x2, family='binomial'(link='logit'))
GLM

LogiSGD(x,ySGD,lr0=0.2, C=2/3, burnin=0.2)
LogiSGDCI(x,ySGD,lr0=0.2, C=2/3, burnin=0.2, B=200, CI.method = 'sd', per_dis = 'exp')


###########################################################
##  Test for quantile regression SGD  #####################
###########################################################
# we tried a 90% quantile regression
# we generate our data by first calculate the expcted 90% quantile as
# beta0 + beta1*x1 + beta2*x2
# draw y from a uniform distribution (range 1) with 90% quantile equals to the
# calculated expected value
set.seed(10)
N<-50000
beta_0<-0.1
beta_1<-0.1
beta_2<--0.1
SD<-1
x1<-rnorm(N, 0, 1)
x2<-rnorm(N,0,1)
x<-matrix(c(x1,x2), nrow=N, byrow = FALSE)
Quan.90<-beta_0+beta_1*x1+beta_2*x2
Lower<-Quan.90-0.9
Upper<-Quan.90+0.1
y<-runif(N, Lower, Upper)
QuanregSGD(x,y, tau=0.9, lr0=0.2, C=2/3, burnin=0.2)
QuanregSGDCI(x,y,tau=0.9,lr0=0.2, C=2/3, burnin=0.2, B=200, CI.method = 'sd', per_dis = 'exp')
# quantile regression using quantreg package
library(quantreg)
rq(y ~ x,  tau = 0.9)
