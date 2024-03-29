% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/LinearRegression.R
\name{linearregSGD}
\alias{linearregSGD}
\alias{linearSGD}
\alias{linearSGDCI}
\title{parameter estimation and confidence interval construction for linear regression with SGD}
\usage{
linearSGD(X, Y, lr0, C = 2/3, burnin = 0.2)

linearSGDCI(
  X,
  Y,
  lr0,
  C = 2/3,
  burnin = 0.2,
  B = 200,
  per_dis = "Poi",
  CI.level = 0.95,
  CI.method = "sd"
)
}
\arguments{
\item{lr0}{lr0 and C are used to calcluate learning rate, learning rate for k-th data point is
calculated as lr_k = lr_0*k **(-C)}

\item{C}{see description for lr0 and C is between 0.5 and 1}

\item{burnin}{a number between 0 and 1; this represent the proportion of estimates
that are not used for calculating final estimate}

\item{B}{number of perturbed estimates generated}

\item{CI.level}{confidence level for constructing confidence interval}

\item{CI.method}{the method to construct confidence interval
it takes one of the following values 'sd' or 'quantile'
'sd' means using sample standard deviation of perturbed
estimates to estimate the standard deviation of SGD estimate
'quantile' means using the upper alpha/2 and lower alpha/2
quantile of the perturbed estimates to construct
confidence interval}

\item{x}{matrix for independent variable, does not include the constant column}

\item{y}{response variable}

\item{Per_dis}{the distribution to generate perturbation parameter
it takes the following values, 'Poi': Poission distribution, Poi(1)
'exp', exponential distribution, exp(1)}
}
\value{
\strong{\code{linearSGD}} \cr
This function will return estimated coefficients for linear regression. \cr
\strong{\code{linearSGDCI}} \cr
This function will return estimated coefficients and confidence intervals for linear regression. \cr
}
\description{
This for applying stochastic gradient descent (SGD) for parameter estimation of
linear regression model. \code{linearSGD} apply SGD for parameter estimation and \code{linearSGDCI}
apply SGD for parameter estimation and also construct confidence intervals for estimates.
}
\details{
Parameter estimation and confidence interval construction are based on Example 1 of Yixin Fang, Jinfeng Xu, and Lei Yang.
Online bootstrap confidence intervals for the stochastic gradient descent estimator.
The Journal of Machine Learning Research, 19(1):3053-3073, 2018.
}
\examples{
\dontrun{
generate x and y
N<-10000
beta_0<-0.2
beta_1<-0.1
beta_2<--0.1
SD<-1
x1<-rnorm(N, 0, 1)
x2<-rnorm(N,0,1)
x<-matrix(c(x1,x2), nrow=N, byrow = FALSE)
y<-beta_0+beta_1*x1+beta_2*x2+rnorm(N, 0, SD)
linearSGD(x,y,lr0=0.1, C=2/3, burnin=0.2)
linearSGDCI(x,y,lr0=0.1, C=2/3, burnin=0.2, B=200, CI.method = 'sd', per_dis = 'exp')
lm function provided by r as a control
lm(y~x)
}

}
\references{
Yixin Fang, Jinfeng Xu, and Lei Yang.
Online bootstrap confidence intervals for the stochastic gradient descent estimator.
The Journal of Machine Learning Research, 19(1):3053-3073, 2018.
}
