# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML7_HTE.pdf
# Predicting effects

# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Causal_tree_forest.nb.html
# Simulation notebook: Causal ML: Causal Tree and Causal Forest

rm(list = ls())

library(grf)
library(tidyverse)
library(rpart)
library(rpart.plot) 
library(partykit)
library(patchwork)
library(causalTree)
library(gganimate)

set.seed(1234)

n <- 1000
p <- 10

x <- matrix(runif(n*p, -pi, pi), ncol = p)
e <- function(x){2/3}
m0 <- function(x){sin(x)}
tau <- function(x){1*(x>-0.5*pi)}
m1 <- function(x){m0(x)+tau(x)}  
w <- rbinom(n,1,e(x))
y <- m0(x[,1])+w*tau(x[,1])+rnorm(n,0,1/2)

g1 <- data.frame(x=c(-pi,pi)) %>% ggplot(aes(x))+
  stat_function(fun=e,size=1)+ylab("e")+xlab("X1")
g2 <- data.frame(x=c(-pi,pi)) %>% ggplot(aes(x))+
  stat_function(fun=m1,size=1,aes(colour="Y1"))+
  stat_function(fun=m0,size=1,aes(colour="Y0"))+ylab("Y")+xlab("X1")
g3 <- data.frame(x=c(-pi,pi)) %>% ggplot(aes(x))+
  stat_function(fun=tau,size=1)+ylab(expression(tau))+xlab("X1")
g1 / g2/ g3


# T-learner with regression trees

# 1. Use regression tree to fit model in control subsample
df <- data.frame(x,y)
tree0 <- rpart(y~x, data=df, subset = (w==0))
rpart.plot(tree0)

# 2. Use regression tree to fit model in treated subsample
tree1 <- rpart(y~x, data=df, subset = (w==1))
rpart.plot(tree1)

# 3. Plot predicted outcomes and CATEs
df$apo_tree0 <- predict(tree0,newdata = data.frame(x))
df$apo_tree1 <- predict(tree1,newdata = data.frame(x))
df$cate_tree <- df$apo_tree1-df$apo_tree0
g1 <- ggplot(df)+stat_function(fun=m1,size=1)+ylab("m1")+
  geom_point(aes(x=x[,1],y=apo_tree1),shape="square",color="blue")
g2 <- ggplot(df)+stat_function(fun=m0,size=1)+ylab("m0")+
  geom_point(aes(x=x[,1],y=apo_tree0),shape="square",color="blue")
g3 <- ggplot(df)+stat_function(fun=tau,size=1)+ylab(expression(tau))+
  geom_point(aes(x=x[,1],y=cate_tree),shape="square",color="blue")
g1 / g2 / g3


# Causal Tree
# Handcoded
grid <- seq(-3,3,0.01)
criterion <- matrix(NA,length(grid),p)
colnames(criterion) = paste0("X",1:p)
for (j in 1:p) {
  for (i in 1:length(grid)) {
    # Indicator for being right of cut-off
    right = (x[,j] > grid[i])
    # Calculate the effect as mean differences in the two leaves
    cate_left = mean(y[w==1 & !right]) - mean(y[w==0 & !right])
    cate_right = mean(y[w==1 & right]) - mean(y[w==0 & right])
    # Calculate and store criterion
    criterion[i,j] = (n-sum(right)) * (cate_left)^2 + sum(right) * (cate_right)^2
  }
}
# Find maximum
index_max <- which(criterion == max(criterion), arr.ind = TRUE)
# Plot criteria
data.frame(x=grid,criterion) %>% 
  pivot_longer(cols=-x,names_to = "Variable",values_to = "Criterion") %>%
  ggplot(aes(x=x ,y=Criterion,colour=Variable)) + geom_line(size=1) + geom_vline(xintercept=-0.5*pi) + 
  geom_vline(xintercept=grid[index_max[1]],linetype = "dashed")


# Package
ctree <- causalTree(y~x, data=df, treatment = w,
                    split.Rule = "CT", cv.option = "CT", split.Honest = T,
                    split.Bucket = F, xval = 5, cp=0, minsize = 20)
rpart.plot(ctree)
opcp <- ctree$cptable[,1][which.min(ctree$cptable[,4])]
opfit <- prune(ctree, opcp)
df$cate_ct <- predict(opfit)
rpart.plot(opfit)

ggplot(df) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + 
  geom_point(aes(x=x[,1],y=cate_ct),shape="square",color="blue") 


# T-learner with random forest
rf0 <- regression_forest(x[w==0,], y[w==0],tune.parameters = "all")
rf1 <- regression_forest(x[w==1,], y[w==1],tune.parameters = "all")

df$apo_rf0 <- predict(rf0,newdata=x)$predictions
df$apo_rf1 <- predict(rf1,newdata=x)$predictions
df$cate_rf <- df$apo_rf1 - df$apo_rf0

g1 <- ggplot(df) + stat_function(fun=m1,size=1) + ylab("m1") + 
  geom_point(aes(x=x[,1],y=apo_rf1),shape="square",color="blue")
g2 <- ggplot(df) + stat_function(fun=m0,size=1) + ylab("m0") + 
  geom_point(aes(x=x[,1],y=apo_rf0),shape="square",color="blue") 
g3 <- ggplot(df) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + 
  geom_point(aes(x=x[,1],y=cate_rf),shape="square",color="blue") 
g1 / g2 / g3


# Causal Forest

cf <- causal_forest(x, y, w,tune.parameters = "all")

df$cate_cf <- predict(cf)$predictions
df$apo_cf0 <- cf$Y.hat - cf$W.hat * df$cate_cf
df$apo_cf1 <- cf$Y.hat + (1-cf$W.hat) * df$cate_cf

g1 <- ggplot(df) + stat_function(fun=m1,size=1) + ylab("m1") + 
  geom_point(aes(x=x[,1],y=apo_cf1),shape="square",color="blue")
g2 <- ggplot(df) + stat_function(fun=m0,size=1) + ylab("m0") + 
  geom_point(aes(x=x[,1],y=apo_cf0),shape="square",color="blue") 
g3 <- ggplot(df) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + 
  geom_point(aes(x=x[,1],y=cate_cf),shape="square",color="blue") 
g1 / g2 / g3


# Comparison
mses <- matrix(NA,4,3)
colnames(mses) <- c("m(0,X)","m(1,X)","tau(X)")

mses[1,1] <- mean( (df$apo_tree0 - m0(x[,1]))^2 )
mses[3,1] <- mean( (df$apo_rf0 - m0(x[,1]))^2 )
mses[4,1] <- mean( (df$apo_cf0 - m0(x[,1]))^2 )

mses[1,2] <- mean( (df$apo_tree1 - m1(x[,1]))^2 )
mses[3,2] <- mean( (df$apo_rf1 - m1(x[,1]))^2 )
mses[4,2] <- mean( (df$apo_cf1 - m1(x[,1]))^2 )

mses[1,3] <- mean( (df$cate_tree - tau(x[,1]))^2 )
mses[2,3] <- mean( (df$cate_ct - tau(x[,1]))^2 )
mses[3,3] <- mean( (df$cate_rf - tau(x[,1]))^2 )
mses[4,3] <- mean( (df$cate_cf - tau(x[,1]))^2 )

data.frame(Method = factor(c("Tree","Causal Tree","Forest","Causal Forest"),
                           levels=c("Tree","Causal Tree","Forest","Causal Forest")),
           mses) %>%
  pivot_longer(cols=-Method,names_to = "Target",values_to = "MSE") %>%
  ggplot(aes(x=Method,y=MSE)) + geom_point() + facet_wrap(~Target) + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))


# Causal Forest behind the scenes

# Set parameters
n <- 200
p <- 2

set.seed(123)
# Draw sample
x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){pnorm(sin(x))}
m0 <- function(x){sin(x)}
tau <- function(x){0 + 1*(x>-0.5*pi)}
m1 <- function(x){m0(x) + tau(x)}
w <- rbinom(n,1,e(x))
y <- m0(x[,1]) + w*tau(x[,1]) + rnorm(n,0,1/10)
g1 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=e,size=1) + ylab("e") + xlab("X1")
g2 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=m1,size=1,aes(colour="Y1")) + 
  stat_function(fun=m0,size=1,aes(colour="Y0")) + ylab("Y") + xlab("X1")
g3 <- data.frame(x = c(-pi, pi)) %>% ggplot(aes(x)) + stat_function(fun=tau,size=1) + ylab(expression(tau)) + xlab("X1")
g1 / g2 / g3


# Weighted residual-on-residual regression at single point

# Run CF
cf <- causal_forest(x, y, w,tune.parameters = "all")
# Define test point 
testx <- matrix(c(-3,rep(0,p-1)),nrow=1)
# Check what package predicts
predict(cf,newdata = testx)$predictions
# 0.1411888


# Get residuals
res_y <- y - cf$Y.hat
res_w <- w - cf$W.hat
# Replicate handcoded
alphax <- get_forest_weights(cf,newdata = testx)[1,]
coef(lm(res_y ~ res_w,weights = alphax))

# Replicate handcoded without constant
coef(lm(res_y ~ 0 + res_w,weights = alphax))


# How the weights move for different predictions

# Run same over grid an see how weights move
grid <- seq(-3,0,0.5)
gridx <- cbind(grid,matrix(0,length(grid),p-1))
grid_hat <- predict(cf,newdata = gridx)$predictions
alpha <- get_forest_weights(cf,newdata = gridx)
for (i in 1:length(grid)) {
  g1 = data.frame(x=grid,tau_hat=grid_hat) %>%
    ggplot(aes(x=x ,y=tau_hat)) + stat_function(fun=tau,size=1) + 
    geom_line(color="blue") + 
    geom_point(aes(x=grid[i],y=grid_hat[i]),size=4,color="blue",shape=4) 
  
  rorr = lm(res_y ~ res_w,weights = alpha[i,])
  
  g2 = data.frame(res_w,res_y,alpha=alpha[i,],x=x) %>%
    ggplot(aes(x=res_w,y=res_y)) + geom_point(aes(size=alpha,color=x[,1]),alpha=0.5) + 
    geom_abline(intercept=rorr$coefficients[1],slope=rorr$coefficients[2]) + 
    annotate("text", x = -0.25, y = 1, label = paste0("tau(",toString(grid[i]),") = slope of line = ",
                                                      toString(round(rorr$coefficients[2],2))))+
    scale_colour_gradient(low = "black", high = "yellow")
  print(g1 / g2)
}


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Meta_learner.nb.html
# Simulation notebook: Causal ML: Meta-learner

# Load packages
library(grf)
library(causalDML)
library(tidyverse)
library(glmnet)
# library(devtools) 
# install_github("xnie/rlearner")
library(rlearner)
library(psych)

set.seed(1234)

# Set parameters
n = 1000
p = 10

# Correct parameters
rho = c(0.3,0.2,0.1,rep(0,p-3))

# Draw sample
x = matrix(runif(n*p,-pi,pi),ncol=p)
e = function(x){pnorm(sin(x))}
m0 = function(x){sin(x)}
tau = x %*% rho
w = rbinom(n,1,e(x[,1]))
y = m0(x[,1]) + w*tau + rnorm(n,0,1)


# Handcoded R-learner with OLS last step

# R-learner with OLS last stage
# 2-fold cross-fitting
mhat = ehat = rep(NA,n)
# Draw random indices for sample 1
index_s1 = sample(1:n,n/2)
# Create S1
x1 = x[index_s1,]
w1 = w[index_s1]
y1 = y[index_s1]
# Create sample 2 with those not in S1
x2 = x[-index_s1,]
w2 = w[-index_s1]
y2 = y[-index_s1]
# Model in S1, predict in S2
rf = regression_forest(x1,w1,tune.parameters = "all")
ehat[-index_s1] = predict(rf,newdata=x2)$predictions
rf = regression_forest(x1,y1,tune.parameters = "all")
mhat[-index_s1] = predict(rf,newdata=x2)$predictions
# Model in S2, predict in S1
rf = regression_forest(x2,w2,tune.parameters = "all")
ehat[index_s1] = predict(rf,newdata=x1)$predictions
rf = regression_forest(x2,y2,tune.parameters = "all")
mhat[index_s1] = predict(rf,newdata=x1)$predictions


# 1. Modify the covariates
# Create residuals
res_y = y-mhat
res_w = w-ehat

# Modify covariates (multiply each column including constant with residual)
x_wc = cbind(rep(1,n),x)
colnames(x_wc) = c("Intercept",paste0("X",1:p))
xstar = x_wc * res_w
# Regress outcome residual on modified covariates
summary(lm(res_y ~ 0 + xstar))


# 2. Pseudo-outcome and weights
# Create pseudo-outcome (outcome res divided by treatment res)
pseudo_rl = res_y / res_w

# Create weights
weights_rl = res_w^2

# Weighted regression of pseudo-outcome on covariates
rols_fit = lm(pseudo_rl ~ x, weights=weights_rl)
summary(rols_fit)
r_ols_est = predict(rols_fit)

# test if all values are equal
all.equal(as.numeric(rols_fit$coefficients), as.numeric(lm(res_y ~ 0 + xstar)$coefficients))

# Estimate CATEs
r_ols_est = predict(rols_fit)


# Handcoded R-learner with Lasso last step
# R-learner with Lasso
rlasso_hand = cv.glmnet(x,pseudo_rl,weights=weights_rl)
plot(rlasso_hand)

rlasso_hand = predict(rlasso_hand,newx = x, s = "lambda.min")


# Using the rlearner package
rlasso_fit = rlasso(x, w, y)
rlasso_est = predict(rlasso_fit, x)


# DR-learner via causalDML package
# DR-learner
dr_est = dr_learner(y,w,x)

# Store and plot predictions
results1k = cbind(tau,r_ols_est,rlasso_hand,rlasso_est,dr_est$cates)
colnames(results1k) = c("True","RL OLS","RL Lasso hand","rlasso","DR RF")
pairs.panels(results1k,method = "pearson")


# Compare MSE
data.frame(MSE = colMeans( (results1k[,-1]-c(tau))^2 ) ,
           Method = factor(colnames(results1k)[-1]) ) %>%
  ggplot(aes(x=Method,y=MSE)) + geom_point(size = 2) + 
  ggtitle(paste(toString(n),"observations")) + geom_hline(yintercept = 0)


# With ensemble/SuperLearner

## Create components of ensemble
# General methods
mean = create_method("mean")
forest =  create_method("forest_grf",args=list(tune.parameters = "all",honesty=F))

# Pscore specific components
ridge_bin = create_method("ridge",args=list(family = "binomial"))
lasso_bin = create_method("lasso",args=list(family = "binomial"))

# Outcome specific components
ols = create_method("ols")
ridge = create_method("ridge")
lasso = create_method("lasso")

# DR-learner with ensemble
dr_ens = dr_learner(y,w,x,ml_w=list(mean,forest,ridge_bin,lasso_bin),
                    ml_y = list(mean,forest,ols,ridge,lasso),
                    ml_tau = list(mean,forest,ols,ridge,lasso),quiet=T)

# Add and plot predictions
label_method = c("RL OLS","RL Lasso hand","rlasso","DR RF","DR Ens")
results1k = cbind(tau,r_ols_est,rlasso_hand,rlasso_est,dr_est$cates,dr_ens$cates)
colnames(results1k) = c("True",label_method)
pairs.panels(results1k,method = "pearson")


# Compare MSE
data.frame(MSE = colMeans( (results1k[,-1]-c(tau))^2 ) ,
           Method = factor(label_method,levels=label_method) ) %>%
  ggplot(aes(x=Method,y=MSE)) + geom_point(size = 2) + 
  ggtitle(paste(toString(n),"observations")) + geom_hline(yintercept = 0)



# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Predicting_effects.nb.html
# Application notebook: Causal ML: Predicting effects

library(hdm)
library(grf)
library(tidyverse)
library(psych)
library(estimatr)
library(causalTree)
library(causalDML)

set.seed(1234) 

data(pension)
Y <- pension$net_tfa
W <- pension$p401
Z <- pension$e401
X <- model.matrix(~ 0 + age + db + educ + fsize + hown + inc + male + marr + pira + twoearn, data = pension)


# S-learner
WX <- cbind(W,X)
rf <- regression_forest(WX,Y)
W0X <- cbind(rep(0,length(Y)),X)
W1X <- cbind(rep(1,length(Y)),X)
cate_sl <- predict(rf,W1X)$predictions - predict(rf,W0X)$predictions
hist(cate_sl)
summary(cate_sl)


# T-learner
# Use ML estimator of your choice to fit model m_hat(1,X) in treated
rfm1 <- regression_forest(X[W==1,],Y[W==1])
# Use ML estimator of your choice to fit model m_hat(0,X) in controls
rfm0 <- regression_forest(X[W==0,],Y[W==0])
# Estimate CATE as τ_hat(X)=m_hat(1,X)−m_hat(0,X)
cate_tl <- predict(rfm1,X)$predictions - predict(rfm0,X)$predictions
hist(cate_tl)
summary(cate_tl)


# Causal Tree

# Prepare data frame
df <- data.frame(X,Y)
# Implemented causalTree adapting specification from R example
ctree <- causalTree(Y~X, data=df, treatment=W,
                    split.Rule = "CT", cv.option = "CT", split.Honest = T,
                    split.Bucket = F, xval=5, cp=0, minsize=20)
opcp <- ctree$cptable[,1][which.min(ctree$cptable[,4])]
opfit <- prune(ctree, opcp)
rpart.plot(opfit)


# Causal Forest
# grf package
cf <- causal_forest(X,Y,W)
cate_cf <- predict(cf)$predictions
hist(cate_cf)


# Replicated using weighted ROR regression
X[1,]

# 1. Extract the nuisance parameters estimates:
mhat <- cf$Y.hat
ehat <- cf$W.hat  
# 2. Extract the weights used for individual one using get_forest_weights:
alphax <- get_forest_weights(cf)[1,]
hist(as.numeric(alphax))
# 3. Create the residuals and run a weighted residual-on-residual regression without constant:
Yres <- Y - mhat
Wres <- W - ehat
manual_cate <- lm(Yres~0+Wres, weights = as.numeric(alphax))$coefficients
manual_cate
# 5306.271 

all.equal(as.numeric(cate_cf[1]),as.numeric(manual_cate))
# "Mean relative difference: 0.002541852"
# Reason: the CF run weighted RORR with constant

# Fix
Yres <- Y - mhat
Wres <- W - ehat
manual_cate_const <- lm(Yres ~ Wres,weights = as.numeric(alphax))
summary(manual_cate_const)

all.equal(as.numeric(cate_cf[1]),as.numeric(manual_cate_const$coefficients[2]))
# TRUE


# R-learner
# Hand-coded using modified covariates (OLS)
# Create residuals
res_y = Y-mhat
res_w = W-ehat

# Modify covariates (multiply each column including constant with residual)
n = length(Y)
X_wc = cbind(rep(1,n),X)
Xstar = X_wc * res_w
# Regress outcome residual on modified covariates
rl_ols = lm(res_y ~ 0 + Xstar)
summary(rl_ols)

cate_rl_ols = X_wc %*% rl_ols$coefficients
hist(cate_rl_ols)


# Hand-coded using pseudo-outcomes and weights (OLS)
# Create pseudo-outcome (outcome res divided by treatment res)
pseudo_rl = res_y / res_w

# Create weights
weights_rl = res_w^2

# Weighted regression of pseudo-outcome on covariates
rols_fit = lm(pseudo_rl ~ X, weights=weights_rl)
summary(rols_fit)

all.equal(as.numeric(rl_ols$coefficients),as.numeric(rols_fit$coefficients))
# TRUE


# Hand-coded using pseudo-outcomes and weights (Random Forest)
# Weighted regression with RF
rrf_fit = regression_forest(X,pseudo_rl, sample.weights = weights_rl)
cate_rl_rf = predict(rrf_fit)$predictions
hist(cate_rl_rf)


# DR-learner
mwhat0 = mwhat1 = rep(NA,length(Y))
rfm0 = regression_forest(X[W==0,],Y[W==0])
mwhat0[W==0] = predict(rfm0)$predictions
mwhat0[W==1] = predict(rfm0,X[W==1,])$predictions

rfm1 = regression_forest(X[W==1,],Y[W==1])
mwhat1 = predict(rfm1)$predictions
mwhat1[W==1] = predict(rfm1)$predictions
mwhat1[W==0] = predict(rfm1,X[W==0,])$predictions

Y_tilde = mwhat1 - mwhat0 + W * (Y - mwhat1) / ehat - (1 - W) * (Y - mwhat0) / (1-ehat)

cate_dr = predict(regression_forest(X,Y_tilde))$predictions
hist(cate_dr)

# Store and plot predictions
results = cbind(cate_sl,cate_tl,cate_cf,cate_rl_rf,cate_dr)
colnames(results) = c("S-learner","T-learner","Causal Forest","R-learner","DR-learner")
pairs.panels(results,method = "pearson")

describe(results)
help(pairs.panels)



# Effect heterogeneity and its validation/inspection

# Determine the number of rows in X
n_rows <- nrow(X)

# Generate a random vector of indices
indices <- sample(1:n_rows, size = 0.5*n_rows)

# Split the data
X_train <- X[indices,]
X_test <- X[-indices,]
W_train <- W[indices]
W_test <- W[-indices]
Y_train <- Y[indices]
Y_test <- Y[-indices]

CF = causal_forest(X_train,Y_train,W_train,tune.parameters = "all")
cates = predict(CF,X_test)$predictions
hist(cates)


# Best linear predictor (BLP)
aipw_test = DML_aipw(Y_test,W_test,X_test)
pseudoY = aipw_test$ATE$delta

demeaned_cates = cates - mean(cates)
lm_blp = lm_robust(pseudoY ~ demeaned_cates)
summary(lm_blp)


# Sorted Group Average Treatment Effects (GATES)
K = 5
slices = factor(as.numeric(cut(cates, breaks=quantile(cates, probs=seq(0,1, length = K+1)),include.lowest=TRUE)))
G_ind = model.matrix(~0+slices)
GATES_woc = lm_robust(pseudoY ~ 0 + G_ind)
# Print results
summary(GATES_woc)

# Plot results
se = GATES_woc$std.error
data.frame(Variable = paste("Group",1:K),
           Coefficient = GATES_woc$coefficients,
           cil = GATES_woc$coefficients - 1.96*se,
           ciu = GATES_woc$coefficients + 1.96*se) %>% 
  ggplot(aes(x=Variable,y=Coefficient,ymin=cil,ymax=ciu)) + geom_point(linewidth=2.5) +
  geom_errorbar(width=0.15) + geom_hline(yintercept=0) + geom_hline(yintercept = lm_blp$coefficients[1], linetype = "dashed")

GATES_wc = lm_robust(pseudoY ~ G_ind[,-1])
summary(GATES_wc)


# Classification analysis (CLAN)

for (i in 1:ncol(X_test)) {
  print(colnames(X_test)[i])
  print(summary(lm_robust(X_test[,i] ~ slices)))
}









