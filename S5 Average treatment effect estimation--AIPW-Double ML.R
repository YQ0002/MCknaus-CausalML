# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML5_AIPW.pdf
rm(list = ls())
gc()


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_AIPW_DML.nb.html
# Simulation Notebook: Causal ML: AIPW Double ML (ATE)

# Simulation Notebook: 
library(hdm)
library(grf)
library(causalDML)
library(tidyverse)
library(patchwork)

set.seed(1234)

# Set parameters
n <- 200
p <- 10
theta <- 0.1

# Define and plot functions
x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){pnorm(sin(x))}
m0 <- function(x){sin(x)}
m1 <- function(x){m0(x) + theta}
tau <- function(x){m1(x) - m0(x)}
w <- rbinom(n,1,e(x[,1]))
y <- w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)

g1 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=e,size=1) + ylab("e") + xlab("X1")
g2 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=m1,size=1,aes(colour="Y1")) + 
  stat_function(fun=m0,size=1,aes(colour="Y0")) + ylab("Y") + xlab("X1")
g3 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + xlab("X1")
g1 / g2 / g3


# Hand-coded AIPW without cross-fitting

# No cross-fitting
# Predict propensity score
rf <- regression_forest(x,w,honesty=F)
ehat <- predict(rf,newdata=x)$predictions
# Model control outcome using only control and predict for all
rf0 <- regression_forest(x[w==0,],y[w==0],honesty=F)
m0hat <- predict(rf0,newdata=x)$predictions
# Model control outcome using only control and predict for all
rf1 <- regression_forest(x[w==1,],y[w==1],honesty=F)
m1hat <- predict(rf1,newdata=x)$predictions
# Generate pseudo-outcome
pseudo_y <- m1hat - m0hat + w*(y-m1hat) / ehat - (1-w)*(y-m0hat) / (1-ehat)

mean(pseudo_y)
t.test(pseudo_y)
summary(lm(pseudo_y~1))
# 0.1179403

# Hand-coded AIPW with 2-fold cross-fitting

# 2-fold cross-fitting
m0hat = m1hat = ehat = rep(NA,n)
# Draw random indices for sample 1
set.seed(1234)
index_s1 <- sample(1:n,n/2)
# Create S1
x1 <- x[index_s1,]
w1 <- w[index_s1]
y1 <- y[index_s1]
# Create sample 2 with those not in S1
x2 <- x[-index_s1,]
w2 <- w[-index_s1]
y2 <- y[-index_s1]
# Model in S1, predict in S2
rf <- regression_forest(x1,w1,honesty=F)
ehat[-index_s1] <- predict(rf,newdata=x2)$predictions
rf = regression_forest(x1[w1==0,],y1[w1==0],honesty=F)
m0hat[-index_s1] <- predict(rf,newdata=x2)$predictions
rf = regression_forest(x1[w1==1,],y1[w1==1],honesty=F)
m1hat[-index_s1] <- predict(rf,newdata=x2)$predictions
# Model in S2, predict in S1
rf <- regression_forest(x2,w2,honesty=F)
ehat[index_s1] <- predict(rf,newdata=x1)$predictions
rf = regression_forest(x2[w2==0,],y2[w2==0],honesty=F)
m0hat[index_s1] <- predict(rf,newdata=x1)$predictions
rf = regression_forest(x2[w2==1,],y2[w2==1],honesty=F)
m1hat[index_s1] <- predict(rf,newdata=x1)$predictions
# Generate pseudo-outcome and take and test mean
pseudo_y <- m1hat - m0hat +  w*(y-m1hat) / ehat - (1-w)*(y-m0hat) / (1-ehat)
summary(lm(pseudo_y~1))
# 0.01853    
# -0.06452

# AIPW with 5-fold cross-fitting

# 5-fold cross-fitting with causalDML package
# Create learner
forest <- create_method("forest_grf", args = list(honesty=F))
set.seed(1234)
aipw <- DML_aipw(y,w,x,ml_w=list(forest), ml_y=list(forest), cf=5)
summary(aipw$APO)
help(DML_aipw)

plot(aipw$APO)

summary(aipw$ATE)
# 0.033223 



# Simulation study homogeneous effect setting

# set number of replications
n_rep <- 1000
# initialize storage for results
coverage = results = matrix(NA,n_rep,3)
colnames(coverage) = colnames(results) = c("PL cf5","AIPW no","AIPW cf5")

# start the simulation
for (i in 1:n_rep){
  x=matrix(runif(n*p,-pi,pi),ncol=p)
  w=rbinom(n,1,e(x[,1]))
  y=w*m1(x[,1])+(1-w)*m0(x[,1])+rnorm(n,0,1)
  
  # partially linear model
  pl = DML_partial_linear(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)
  results[i,1] = pl$result[1]
  coverage[i,1] = (pl$result[1] - 1.96*pl$result[2] < theta & pl$result[1] + 1.96*pl$result[2] > theta)
  
  # No cross-fitting
  rf = regression_forest(x,w,honesty=F)
  ehat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==0,],y[w==0],honesty=F)
  m0hat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==1,],y[w==1],honesty=F)
  m1hat = predict(rf,newdata=x)$predictions
  pseudo_y =  m1hat - m0hat +
    w*(y-m1hat) / ehat - (1-w)*(y-m0hat) / (1-ehat)
  results[i,2] = mean(pseudo_y)
  tt = t.test(pseudo_y)
  coverage[i,2] = (tt$conf.int[1]  < theta & tt$conf.int[2] > theta)
  
  # 5-fold cross-fitting with causalDML package reusing the folds and pscores of PL
  aipw = DML_aipw(y,w,x,ml_y=list(forest),cf=5,
                  e_mat = cbind(1-pl$e_hat,pl$e_hat),cf_mat = pl$cf_mat)
  results[i,3] = aipw$ATE$results[1]
  coverage[i,3] = (aipw$ATE$results[1] - 1.96*aipw$ATE$results[2] < theta & aipw$ATE$results[1] + 1.96*aipw$ATE$results[2] > theta)

  print(i)
}

# the estimator distributions
as.data.frame(results) %>% pivot_longer(cols=everything(),names_to = "Estimator",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Estimator)) + geom_density(alpha=0.5) + theme_bw() + geom_vline(xintercept=theta)

data.frame(method = colnames(results),
           bias2 = colMeans(results-theta)^2,
           var = colMeans(sweep(results,2,colMeans(results))^2)) %>% 
  pivot_longer(-method,names_to = "Component",values_to = "MSE") %>%
  ggplot(aes(fill=factor(Component,levels=c("var","bias2")), y=MSE, x=method)) + 
  geom_bar(position="stack", stat="identity") + scale_fill_discrete(name = "Component")

# check the coverage rate
data.frame(method = colnames(results),
           coverage = colMeans(coverage)) %>% 
  ggplot(aes(y=coverage, x=method)) + geom_hline(yintercept=0.95,linetype="dashed") + 
  geom_point(size=5,shape=4) + scale_fill_discrete(name = "Component") + ylim(c(0,1)) +
  geom_hline(yintercept=c(0,1))


# Effect heterogeneity with balanced treatment shares

x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){pnorm(sin(x))}
m1 <- function(x){sin(x)}
m0 <- function(x){cos(x+1/2*pi)}
tau <- function(x){m1(x) - m0(x)}
w <- rbinom(n,1,e(x[,1]))
y <- w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)

g1 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=e,size=1) + ylab("e") + xlab("X1")
g2 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=m1,size=1,aes(colour="Y1")) + 
  stat_function(fun=m0,size=1,aes(colour="Y0")) + ylab("Y") + xlab("X1")
g3 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + xlab("X1")
g1 / g2 / g3


# Simulation study heterogeneous effect setting

# initialize storage for results
coverage_het = results_het = matrix(NA,n_rep,3)
colnames(coverage_het) = colnames(results_het) = c("PL cf5","AIPW no","AIPW cf5")

# start the simulation
for (i in 1:n_rep) {
  x = matrix(runif(n*p,-pi,pi),ncol=p)
  w = rbinom(n,1,e(x[,1]))
  y = w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)
  
  # partially linear model
  pl = DML_partial_linear(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)
  results_het[i,1] = pl$result[1]
  coverage_het[i,1] = (pl$result[1] - 1.96*pl$result[2] < 0 & pl$result[1] + 1.96*pl$result[2] > 0)
  
  # No cross-fitting
  rf = regression_forest(x,w,honesty=F)
  ehat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==0,],y[w==0],honesty=F)
  m0hat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==1,],y[w==1],honesty=F)
  m1hat = predict(rf,newdata=x)$predictions
  pseudo_y =  m1hat - m0hat +
    w*(y-m1hat) / ehat - (1-w)*(y-m0hat) / (1-ehat)
  results_het[i,2] = mean(pseudo_y)
  tt = t.test(pseudo_y)
  coverage_het[i,2] = (tt$conf.int[1]  < 0 & tt$conf.int[2] > 0)
  
  aipw = DML_aipw(y,w,x,ml_y=list(forest),cf=5,
                  e_mat = cbind(1-pl$e_hat,pl$e_hat),cf_mat = pl$cf_mat)
  results_het[i,3] = aipw$ATE$results[1]
  coverage_het[i,3] = (aipw$ATE$results[1] - 1.96*aipw$ATE$results[2] < 0 & aipw$ATE$results[1] + 1.96*aipw$ATE$results[2] > 0)
}

as.data.frame(results_het) %>% pivot_longer(cols=everything(),names_to = "Estimator",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Estimator)) + geom_density(alpha=0.5) + theme_bw() + geom_vline(xintercept=0)


data.frame(method = colnames(results_het),
           bias2 = colMeans(results_het-0)^2,
           var = colMeans(sweep(results_het,2,colMeans(results_het))^2)) %>% 
  pivot_longer(-method,names_to = "Component",values_to = "MSE") %>%
  ggplot(aes(fill=factor(Component,levels=c("var","bias2")), y=MSE, x=method)) + 
  geom_bar(position="stack", stat="identity") + scale_fill_discrete(name = "Component")


data.frame(method = colnames(results_het),
           coverage = colMeans(coverage_het)) %>% 
  ggplot(aes(y=coverage, x=method)) + geom_hline(yintercept=0.95,linetype="dashed") + 
  geom_point(size=5,shape=4) + scale_fill_discrete(name = "Component") + ylim(c(0,1)) +
  geom_hline(yintercept=c(0,1))



# Effect heterogeneity with unbalanced treatment shares

n = 300

x = matrix(runif(n*p,-pi,pi),ncol=p)
e = function(x){pnorm(sin(x)-0.5)}
m1 = function(x){sin(x)}
m0 = function(x){cos(x+1/2*pi)}
tau = function(x){m1(x) - m0(x)}
w = rbinom(n,1,e(x[,1]))
y = w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)

g1 = data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=e,size=1) + ylab("e") + xlab("X1")
g2 = data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=m1,size=1,aes(colour="Y1")) + 
  stat_function(fun=m0,size=1,aes(colour="Y0")) + ylab("Y") + xlab("X1")
g3 = data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + xlab("X1")
g1 / g2 / g3


# initialize storage for results
coverage_unbal = results_unbal = matrix(NA,n_rep,3)
colnames(coverage_unbal) = colnames(results_unbal) = c("PL cf5","AIPW no","AIPW cf5")

# start simulation
for (i in 1:n_rep) {
  x = matrix(runif(n*p,-pi,pi),ncol=p)
  w = rbinom(n,1,e(x[,1]))
  y = w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)
  
  # partially linear model
  pl = DML_partial_linear(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)
  results_unbal[i,1] = pl$result[1]
  coverage_unbal[i,1] = (pl$result[1] - 1.96*pl$result[2] < 0 & pl$result[1] + 1.96*pl$result[2] > 0)
  
  # No cross-fitting
  rf = regression_forest(x,w,honesty=F)
  ehat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==0,],y[w==0],honesty=F)
  m0hat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x[w==1,],y[w==1],honesty=F)
  m1hat = predict(rf,newdata=x)$predictions
  pseudo_y =  m1hat - m0hat +
    w*(y-m1hat) / ehat - (1-w)*(y-m0hat) / (1-ehat)
  results_unbal[i,2] = mean(pseudo_y)
  tt = t.test(pseudo_y)
  coverage_unbal[i,2] = (tt$conf.int[1]  < 0 & tt$conf.int[2] > 0)
  
  # 5-fold cross-fitting with causalDML package reusing the folds and pscores of PL
  aipw = DML_aipw(y,w,x,ml_y=list(forest),cf=5,
                  e_mat = cbind(1-pl$e_hat,pl$e_hat),cf_mat = pl$cf_mat)
  results_unbal[i,3] = aipw$ATE$results[1]
  coverage_unbal[i,3] = (aipw$ATE$results[1] - 1.96*aipw$ATE$results[2] < 0 & aipw$ATE$results[1] + 1.96*aipw$ATE$results[2] > 0)

  print(i)
}

as.data.frame(results_unbal) %>% pivot_longer(cols=everything(),names_to = "Estimator",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Estimator)) + geom_density(alpha=0.5) + theme_bw() + geom_vline(xintercept=0)

data.frame(method = colnames(results_unbal),
           bias2 = colMeans(results_unbal-0,na.rm=T)^2,
           var = colMeans(sweep(results_unbal,2,colMeans(results_unbal,na.rm=T))^2,na.rm=T)) %>% 
  pivot_longer(-method,names_to = "Component",values_to = "MSE") %>%
  ggplot(aes(fill=factor(Component,levels=c("var","bias2")), y=MSE, x=method)) + 
  geom_bar(position="stack", stat="identity") + scale_fill_discrete(name = "Component")

data.frame(method = colnames(results_unbal),
           coverage = colMeans(coverage_unbal,na.rm=T)) %>% 
  ggplot(aes(y=coverage, x=method)) + geom_hline(yintercept=0.95,linetype="dashed") + 
  geom_point(size=5,shape=4) + scale_fill_discrete(name = "Component") + ylim(c(0,1)) +
  geom_hline(yintercept=c(0,1))




# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_AIPW_DML.nb.html
# Application notebook: Causal ML: Double ML for average treatment effects

library(hdm)
library(tidyverse)
library(causalDML)
library(grf)
library(lmtest)
library(sandwich)

set.seed(1234) # for replicability
options(scipen = 10) # Switch off scientific notation

data(pension)
# Outcome
Y = pension$net_tfa
# Treatment
W = pension$p401
# Create main effects matrix
X = model.matrix(~ 0 + age + db + educ + fsize + hown + inc + male + marr + pira + twoearn, data = pension)


# 2-fold cross-fitting
n = length(Y)
m0hat = m1hat = ehat = rep(NA,n)
# Draw random indices for sample 1
index_s1 = sample(1:n,n/2)
# Create S1
x1 = X[index_s1,]
w1 = W[index_s1]
y1 = Y[index_s1]
# Create sample 2 with those not in S1
x2 = X[-index_s1,]
w2 = W[-index_s1]
y2 = Y[-index_s1]
# Model in S1, predict in S2
rf = regression_forest(x1,w1)
ehat[-index_s1] = predict(rf,newdata=x2)$predictions
rf = regression_forest(x1[w1==0,],y1[w1==0])
m0hat[-index_s1] = predict(rf,newdata=x2)$predictions
rf = regression_forest(x1[w1==1,],y1[w1==1])
m1hat[-index_s1] = predict(rf,newdata=x2)$predictions
# Model in S2, predict in S1
rf = regression_forest(x2,w2)
ehat[index_s1] = predict(rf,newdata=x1)$predictions
rf = regression_forest(x2[w2==0,],y2[w2==0])
m0hat[index_s1] = predict(rf,newdata=x1)$predictions
rf = regression_forest(x2[w2==1,],y2[w2==1])
m1hat[index_s1] = predict(rf,newdata=x1)$predictions

Y_t_0 = m0hat + (1-W)*(Y-m0hat)/(1-ehat)
Y_t_1 = m1hat + W*(Y-m1hat)/ehat
summary(lm(Y_t_0 ~ 1))
mean(Y_t_0)
summary(lm(Y_t_1 ~ 1))
mean(Y_t_1)

Y_ate = Y_t_1 - Y_t_0
summary(lm(Y_ate ~ 1))
# 11486


# Double ML for AIPW with causalDML package

# 5-fold cross-fitting with causalDML package
# Create learner
forest = create_method("forest_grf",args=list(tune.parameters = "all"))

aipw = DML_aipw(Y,W,X,ml_w=list(forest),ml_y=list(forest),cf=5)
summary(aipw$APO)
plot(aipw$APO)
summary(aipw$ATE)

APO_att = APO_dml_atet(Y,aipw$APO$m_mat,aipw$APO$w_mat,aipw$APO$e_mat,aipw$APO$cf_mat)
ATT = ATE_dml(APO_att)
summary(ATT)

# Collect the results
Effect = c(aipw$ATE$result[1],ATT$results[1])
se = c(aipw$ATE$result[2],ATT$results[2])
data.frame(Effect,se,
           Target = c("ATE","ATT"),
           cil = Effect - 1.96*se,
           ciu = Effect + 1.96*se)  %>% 
  ggplot(aes(x=Target,y=Effect,ymin=cil,ymax=ciu)) + geom_point(size=2.5) + geom_errorbar(width=0.15)  +
  geom_hline(yintercept=0) + xlab("Target parameter")


