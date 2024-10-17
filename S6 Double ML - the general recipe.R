
# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML6_DML.pdf
# Causal Machine Learning - Double ML - the general recipe


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Influence_Function_OLS.nb.html
# Basics: Influence functions explained using OLS

# OLS influence functions

library(tidyverse)
library(estimatr)
library(MASS)

n = 30
set.seed(1234)

# Draw independent variable
x = rnorm(n)
hist(x)

# Define population parameters
b0 = 1
b1 = 1/2

# Define conditional expectation function
cef = function(x){b0 + b1*x}

# Generate outcome variable
y = cef(x) + runif(n,-1,1)

# Plot sample
df = data.frame(x=x,y=y)
ggplot(df) + stat_function(fun=cef,linewidth=1) + 
  geom_point(aes(x=x,y=y),color="blue",alpha = 0.4)


# Add constant to covariates
X = cbind(rep(1,n),x)
colnames(X) = c("Constant","X")
Q = solve(crossprod(X))
betas = Q %*% t(X) %*% y
betas

# Check that it is identical to lm
summary(lm_robust(y ~ x))


# Recap: General Recipe for Influence Functions

# Calculate IFs for individual 1
solve(crossprod(X) / n) %*% X[1,] %*% (y[1] - X[1,] %*% betas)
IF = (X * as.numeric(y - X %*% betas)) %*% solve(crossprod(X) / n)
dim(IF)

# check whether the means of the influence functions are equal to 0
all.equal(as.numeric(colMeans(IF)),
          rep(0,ncol(X))) 

# Influence Functions for standard errors
sqrt(diag(var(IF)) / n)

lm_robust(y ~ x,se_type = "HC0")$std.error
lm_robust(y ~ x,se_type = "HC1")$std.error
lm_robust(y ~ x,se_type = "HC2")$std.error
lm_robust(y ~ x,se_type = "HC3")$std.error

# Checking coverage rates
rep = 1000 # number of replications for the simulation
se_matrix = matrix(NA, nrow = rep, ncol = 5) # empty matrix to store the standard errors
cover_matrix = matrix(NA, nrow = rep, ncol = 5) # matrix to store whether true value was covered
colnames(se_matrix) = c("HC0", "HC1", "HC2", "HC3", "IF") # give meaningful names to the columns of the matrices
colnames(cover_matrix) = c("HC0", "HC1", "HC2", "HC3", "IF")

n = 100 # number of observations

# True parameter values
b0 = 1
b1 = 1/2

critical_value = qt(0.975, df = n-2)

for (i in 1: rep){
  
  # Draw independent variable
  x = rnorm(n)
  
  # Generate outcome
  y = b0 + b1 * x + runif(n,-1,1)
  
  # use the preimplemented commands, save the standard errors:
  se_matrix[i,1] = lm_robust(y ~ x,se_type = "HC0")$std.error[2]
  se_matrix[i,2] = lm_robust(y ~ x,se_type = "HC1")$std.error[2]
  se_matrix[i,3] = lm_robust(y ~ x,se_type = "HC2")$std.error[2]
  se_matrix[i,4] = lm_robust(y ~ x,se_type = "HC3")$std.error[2]
  
  # influence function, apply the formulas from above:
  X = cbind(rep(1,n),x)
  Q = solve(crossprod(X))
  betas = Q %*% t(X) %*% y
  IF = (X * as.numeric(y - X %*% betas)) %*% solve(crossprod(X) / n)
  se_matrix[i,5] = sqrt(diag(var(IF)) / n)[2]
  
  # check the coverage; i.e. does the true value lie within the bounds of the 95% confidence interval?:
  cover_matrix[i,1] = 1*((betas[2] - critical_value* se_matrix[i,1] < b1) & (betas[2] + critical_value*  se_matrix[i,1] > b1))
  cover_matrix[i,2] = 1*((betas[2] - critical_value* se_matrix[i,2] < b1) & (betas[2] + critical_value*  se_matrix[i,2] > b1))
  cover_matrix[i,3] = 1*((betas[2] - critical_value* se_matrix[i,3] < b1) & (betas[2] + critical_value*  se_matrix[i,3] > b1))
  cover_matrix[i,4] = 1*((betas[2] - critical_value* se_matrix[i,4] < b1) & (betas[2] + critical_value*  se_matrix[i,4] > b1))
  cover_matrix[i,5] = 1*((betas[2] - critical_value* se_matrix[i,5] < b1) & (betas[2] + critical_value*  se_matrix[i,5] > b1))

  print(i)
}

colMeans(cover_matrix)
cor(se_matrix)

IF = (X * as.numeric(y - X %*% betas)) %*% solve(crossprod(X) / n) * sqrt((n-1)/(n-2))
sqrt(diag(var(IF)) / n)

lm_robust(y ~ x,se_type = "HC1")$std.error

# Draw independent variable
x = rnorm(n)

# Generate outcome
y = b0 + b1 * x + runif(n,-1,1)

# influence function, apply the formulas from above:
X = cbind(rep(1,n),x)
Q = solve(crossprod(X))
betas = Q %*% t(X) %*% y
IF = solve(crossprod(X) / n) %*% t(X * as.numeric(y - X %*% betas))

loools = matrix(NA,n,2)

#Calculate coefficients w/o individual i
for (i in 1:n) {
  loools[i,] = solve(crossprod(X[-i,])) %*% t(X[-i,]) %*% y[-i]
}

# plot diff between full sample and leave-on-out against the IF
plot(betas[1] - loools[,1], IF[1,] / n)

plot(betas[2] - loools[,2], IF[2,] / n)


# Chain rule
# Use case 1: Testing for omitted variable bias
# No confounding
n = 1000
mu = c(0,0)
rho = 0
sigma = matrix(c(1,rho,rho,1), nrow = 2)
draw = mvrnorm(n, mu, sigma)
W = draw[,1]
X =  draw[,2]
Y = 0.5 * W + X + rnorm(n)

# run a regression using the pre-implemented command:
summary(lm_robust(Y~ W))

# Calculate the influence functions. In contrast to the code above, here we directly construct the matrix X using model.matrix(lm_robust(Y~ W)) 
# and the OLS coefficients as a vector with matrix(lm_robust(Y~ W)$coefficients) in the same line of code
IF_OVB = (model.matrix(lm_robust(Y~ W))
          * as.numeric(Y - model.matrix(lm_robust(Y~ W)) 
                       %*% matrix(lm_robust(Y~ W)$coefficients))
          ) %*% solve(crossprod(model.matrix(lm_robust(Y~ W))) / n)
dim(IF_OVB)

summary(lm_robust(Y~ W+X))

IF_X = (model.matrix(lm_robust(Y~ W+X)) 
        * as.numeric(Y - model.matrix(lm_robust(Y~ W+X)) 
                     %*% matrix(lm_robust(Y~ W+X)$coefficients))
        ) %*% solve(crossprod(model.matrix(lm_robust(Y~ W+X))) / n)

# display the difference:
Delta = lm_robust(Y ~ W)$coefficients[2] - lm_robust(Y ~ W + X)$coefficients[2]
print(paste0("Delta: ", Delta))

# Influence Function for Delta
IF_Delta = IF_OVB[,2] - IF_X[,2] 

se_Delta = sqrt(var(IF_Delta) / length(IF_Delta))
t_stat = Delta/se_Delta # t-statistic
print(paste0("t-statistic: ", t_stat))

p_val = 2*(1-pnorm(abs(Delta/se_Delta))) # p value
print(paste0("p-value: ", p_val))


# Confounding
rho = 0.7
sigma = matrix(c(1,rho,rho,1), nrow = 2)
draw = mvrnorm(n, mu, sigma)
W = draw[,1]
X =  draw[,2]
Y = 0.5 * W + X + rnorm(n)

summary(lm_robust(Y ~ W))

IF_OVB = (model.matrix(lm_robust(Y~ W)) 
          * as.numeric(Y - model.matrix(lm_robust(Y~ W)) 
                       %*% matrix(lm_robust(Y~ W)$coefficients))
          ) %*% solve(crossprod(model.matrix(lm_robust(Y~ W))) / n)
summary(lm_robust(Y ~ W + X))

IF_X = (model.matrix(lm_robust(Y~ W+X)) 
        * as.numeric(Y - model.matrix(lm_robust(Y~ W+X)) 
                     %*% matrix(lm_robust(Y~ W+X)$coefficients))
        ) %*% solve(crossprod(model.matrix(lm_robust(Y~ W+X))) / n)
# display the difference:
Delta = lm_robust(Y ~ W)$coefficients[2] - lm_robust(Y ~ W + X)$coefficients[2]
print(paste0("Delta: ", Delta))

# Influence Function for Delta
IF_Delta = IF_OVB[,2] - IF_X[,2] 

se_Delta = sqrt(var(IF_Delta) / length(IF_Delta))
t_stat = Delta/se_Delta # t-statistic
print(paste0("t-statistic: ", t_stat))

p_val = 2*(1-pnorm(abs(Delta/se_Delta))) # p value
print(paste0("p-value: ", p_val))


# Use case 2: Standard errors for fitted values using influence functions

x_seq = seq(-3,3,0.01)
n = length(x_seq)

# Draw independent variable
x = rnorm(n)

# True parameter
b0 = -1
b1 = 1/2

# Generate outcome
y = b0 + b1 * x + runif(n,-1,1)

# Add constant
X = cbind(rep(1,n),x)

# Manually obtain the OLS coefficients:
Q = solve(crossprod(X))
betas = Q %*% t(X) %*% y

# And get the influence functions for all individuals
IF = (X * as.numeric(y - X %*% betas)) %*% solve(crossprod(X) / n)

X_seq  = cbind(rep(1, length(x_seq)), x_seq) # construct the matrix, which in the text is called big X tilde

# Get the m x 1 vector of predictions y_hat using the sequence from -3 to 3 as values for x:
y_hat = X_seq %*% betas

# Influence function for the predictions, using the influence functions for the coefficients:
IF_yhat = X_seq %*% t(IF)

# Standard errors:
ses = sqrt(diag(var(t(IF_yhat))) / n) # just like we did it above

# manually compute the bounds of the confidence interval:
CI_lower = y_hat - critical_value*ses
CI_upper = y_hat + critical_value*ses

# plot
tibble(x_seq, y_hat, CI_lower, CI_upper, x , y) %>% ggplot(aes(x = x_seq, y = y_hat))+
  geom_line(color = "red", linetype = "solid")+
  geom_line(aes(x = x_seq, y = CI_lower), color = "red", linetype = "dashed")+
  geom_line(aes(x = x_seq, y = CI_upper), color = "red", linetype = "dashed")+
  geom_point(aes(x = x, y  =y))+
  xlab("x")+
  ylab("")


R = 1000 # number of replications
n = 100 # number of observations

# True parameters (unchanged)
b0 = -1
b1 = 1/2

critical_value = qt(0.975, df = n-2)

# "True" (expected) y:
y_exp = b0 +b1*x_seq # this is the "true" quantity we are estimating

coverage_matrix = matrix(NA, ncol = length(x_seq), nrow =  R)

for (i in 1:R){
  x = rnorm(n)
  
  # Generate outcome
  y = b0 + b1 * x + runif(n,-1,1)
  
  # Add constant
  X = cbind(rep(1,n),x)
  
  # Manually obtain the OLS coefficients:
  Q = solve(crossprod(X))
  betas = Q %*% t(X) %*% y
  
  # And get the influence functions for all individuals
  IF = (X * as.numeric(y - X %*% betas)) %*% solve(crossprod(X) / n)
  
  y_hat = X_seq %*% betas # the predictions
  
  # Influence function for the predictions:
  IF_yhat = X_seq %*% t(IF)
  
  # standard errors:
  ses = sqrt(diag(var(t(IF_yhat))) / n)
  
  coverage_matrix[i,] = 1*(((y_hat - critical_value* ses) <= y_exp) & ((y_hat + critical_value* ses) > y_exp)) # for all observations: check whether the confidence interval
  # includes the true value, save as 0 or 1
  print(i)
}

coverage_rates = colMeans(coverage_matrix) # get the coverage rates at all different values of x

# plot the coverage rates over the grid of x
tibble(x_seq, coverage_rates) %>% ggplot(aes(x = x_seq, y = coverage_rates))+
  geom_hline(yintercept=(0.95), linetype="dashed")+ 
  geom_line(colour = "red", linewidth = 1) +
  ylim(c(0,1)) +
  geom_hline(yintercept=c(0,1)) +
  labs(
    x="x",
    y="Coverage",
    title="Model coverage",
    caption="Based on simulated data"
  ) +
  theme_bw()


# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Generic_DML.nb.html
# Application notebook: Causal ML: Double ML as generic recipe

library(hdm)
library(causalDML)
library(grf)
library(tidyverse)

set.seed(1234)

# Get data
data(pension)
# Outcome
Y = pension$net_tfa
# Treatment
W = pension$p401
# Treatment
Z = pension$e401
# Create main effects matrix
X = model.matrix(~ 0 + age + db + educ + fsize + hown + inc + male + marr + pira + twoearn, data = pension)


# Nuisance parameters
n = length(Y)
nfolds = 5
fold = sample(1:nfolds,n,replace=T)

exhat = mxhat = mwhat0 = mwhat1 = hhat = mzhat0 = mzhat1 = ezhat0 = ezhat1 = rep(NA,n)

for (i in 1:nfolds){
  rfe = regression_forest(X[fold != i,],W[fold != i])
  exhat[fold == i] = predict(rfe, X[fold == i,])$predictions
  
  rfm = regression_forest(X[fold != i,],Y[fold != i])
  mxhat[fold == i] = predict(rfm, X[fold == i,])$predictions
  
  rfm0 = regression_forest(X[fold != i & W==0,],Y[fold != i & W==0])
  mwhat0[fold == i] = predict(rfm0, X[fold == i,])$predictions
  
  rfm1 = regression_forest(X[fold != i & W==1,],Y[fold != i & W==1])
  mwhat1[fold == i] = predict(rfm1, X[fold == i,])$predictions
  
  rfh = regression_forest(X[fold != i,],Z[fold != i])
  hhat[fold == i] = predict(rfh, X[fold == i,])$predictions
  
  rfmz0 = regression_forest(X[fold != i & Z==0,],Y[fold != i & Z==0])
  mzhat0[fold == i] = predict(rfmz0, X[fold == i,])$predictions
  
  rfmz1 = regression_forest(X[fold != i & Z==1,],Y[fold != i & Z==1])
  mzhat1[fold == i] = predict(rfmz1, X[fold == i,])$predictions
  
  rfez0 = regression_forest(X[fold != i & Z==0,],W[fold != i & Z==0])
  ezhat0[fold == i] = predict(rfez0, X[fold == i,])$predictions
  
  rfez1 = regression_forest(X[fold != i & Z==1,],W[fold != i & Z==1])
  ezhat1[fold == i] = predict(rfez1, X[fold == i,])$predictions
}

ehat = mean(W)

# ψa and ψb of each scores
# PL
pa_pl = -(W - exhat)^2
pb_pl = (Y - mxhat) * (W - exhat)
-sum(pb_pl) / sum(pa_pl)
# 13734.91

# ATE
pa_ate = rep(-1,length(Y))
pb_ate = mwhat1 - mwhat0 + W * (Y - mwhat1) / ehat - (1 - W) * (Y - mwhat0) / (1-ehat)
-sum(pb_ate) / sum(pa_ate)
# 12244.7

# ATT
pa_att = -W / ehat
pb_att = W * (Y - mwhat0) / ehat - ( (1 - W) * exhat ) * (Y - mwhat0) / (ehat * (1 - ehat))
-sum(pb_att) / sum(pa_att)
# 14873.26

# PL-IV
pa_iv = -(W - exhat) * (Z - hhat)
pb_iv = (Y - mxhat) * (Z - hhat) 
-sum(pb_iv) / sum(pa_iv)
# 12706.16

# LATE
pa_late = -( ezhat1 - ezhat0 + Z * (W - ezhat1) / hhat - (1 - Z) * (W - ezhat0) / (1-hhat) )
pb_late = mzhat1 - mzhat0 + Z * (Y - mzhat1) / hhat - (1 - Z) * (Y - mzhat0) / (1-hhat)
-sum(pb_late) / sum(pa_late)
# 11281.42

# A generic function for Double ML with linear score
DML_inference = function(psi_a,psi_b) {
  N = length(psi_a)
  theta = -sum(psi_b) / sum(psi_a)
  psi = theta * psi_a + psi_b
  Psi = - psi / mean(psi_a)
  sigma2 = var(Psi)
  # sigma2 = mean(psi^2) / mean(psi_a)^2
  se = sqrt(sigma2 / N)
  t = theta / se
  p = 2 * pt(abs(t),N,lower = FALSE)
  result = c(theta,se,t,p)
  return(result)
}

# Results
results = matrix(NA,5,4)
rownames(results) = c("PL","ATE","ATT","PL-IV","LATE")
colnames(results) = c("Effect","S.E.","t","p")
results[1,] = DML_inference(pa_pl,pb_pl)
results[2,] = DML_inference(pa_ate,pb_ate)
results[3,] = DML_inference(pa_att,pb_att)
results[4,] = DML_inference(pa_iv,pb_iv)
results[5,] = DML_inference(pa_late,pb_late)
printCoefmat(results,has.Pvalue = TRUE)

data.frame(thetas = results[,1],ses = results[,2],
           Estimator = rownames(results),
           cil = results[,1] - 1.96*results[,2],
           ciu = results[,1] + 1.96*results[,2])  %>% 
  ggplot(aes(x=Estimator,y=thetas,ymin=cil,ymax=ciu)) + geom_point(size=2.5) + geom_errorbar(width=0.15)  +
  geom_hline(yintercept=0)

# ATT vs. ATE
Psi_maker = function(psi_a,psi_b) {
  theta = -sum(psi_b) / sum(psi_a)
  psi = theta * psi_a + psi_b
  Psi = - psi / mean(psi_a)
  return(Psi)
}

# Calculate parameters
att = - sum(pb_att) / sum(pa_att)
ate = - sum(pb_ate) / sum(pa_ate)
Delta_att = att - ate
# Create influence function for new parameter
Psi_Delta_att = Psi_maker(pa_att,pb_att) - Psi_maker(pa_ate,pb_ate)
# Calculate standard errors, t and pvalues
se_Delta_att = sqrt(var(Psi_Delta_att)/length(Psi_Delta_att))
t_Delta_att = Delta_att / se_Delta_att
p_Delta_att = 2 * pt(abs(t_Delta_att),length(Psi_Delta_att),lower = FALSE)
# Print results
result = matrix(c(Delta_att,se_Delta_att,t_Delta_att,p_Delta_att),nrow = 1)
rownames(result) = c("ATT-ATE")
colnames(result) = c("Delta","S.E.","t","p")
printCoefmat(result,has.Pvalue = TRUE)

# LATE vs. ATE
# Calculate
late = -sum(pb_late)/sum(pa_late) 
# New target parameter
Delta_late = late - ate 
# Create influence function for new parameter
Psi_Delta_late = Psi_maker(pa_late,pb_late) - Psi_maker(pa_ate,pb_ate)
# Print results
se_Delta_late = sqrt(var(Psi_Delta_late)/length(Psi_Delta_late))
t_Delta_late = Delta_late/se_Delta_late
p_Delta_late = 2 * pt(abs(t_Delta_late),length(Psi_Delta_late),lower = FALSE) # get a p-value (at what level would be not reject?)

results = matrix(c(Delta_late,se_Delta_late,t_Delta_late,p_Delta_late),nrow = 1)
rownames(results) = c("LATE-ATE")
colnames(results) = c("Delta","S.E.","t","p")
printCoefmat(results,has.Pvalue = TRUE)

# How large is the difference between ATT and ATE in %?

# New parameter
Delta_pc = (att-ate)/ate*100 
# New IF
Psi_Delta_pc = 100 / ate * Psi_maker(pa_att, pb_att) - 100 * att / (ate^2) * Psi_maker(pa_ate, pb_ate)
# Results
se_Delta_pc = sqrt(var(Psi_Delta_pc)/length(Psi_Delta_pc))
t_Delta_pc = Delta_pc/se_Delta_pc
p_Delta_pc = 2 * pt(abs(t_Delta_pc),length(Psi_Delta_pc),lower = FALSE)
results = matrix(c(Delta_pc,se_Delta_pc,t_Delta_pc,p_Delta_pc),nrow = 1)
rownames(results) = c("Delta%")
colnames(results) = c("Delta","S.E.","t","p")
printCoefmat(results,has.Pvalue = TRUE)


