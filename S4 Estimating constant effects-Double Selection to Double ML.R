# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML4_DS_PLR.pdf

rm(list = ls())
gc()

# Estimating constant effects: Double Selection to Double ML


# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_OLS_Frisch_Waugh.nb.html
# Simulation Notebook: Basics: OLS and Frisch-Waugh
# Introducing the data

library(hdm)
library(tidyverse)

set.seed(1234)
options(scipen = 10)

data(pension)
names(pension)


# Handcoding OLS
ols <- lm(net_tfa~age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, 
          data = pension)
summary(ols)

# beta = (X'*X^-1)*X'*Y

X <- model.matrix(~age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, 
                  data = pension)
Y <- pension[, "net_tfa"]

hand_ols <- solve(t(X) %*% X) %*% t(X) %*% Y
hand_ols
coef(ols)
# Check
all.equal(as.numeric(hand_ols), as.numeric(ols$coefficients))


# Frisch-Waugh
# Estimate θ in the LR: Y = Wθ + X'β + U through the following way:

# 1. Run regression Y~X, and extract the estimated residuals
#    Y = α_y + X'π + U_Y∼X

ols_age <- lm(age~db+educ+fsize+hown+inc+male+marr+pira+twoearn, 
              data = pension)
Vhat <- ols_age$residuals

# 2. Run regression W~X, and extract the estimated residuals
#    W = α_w + X0δ + U_W∼X
ols_net_tfa <- lm(net_tfa~db+educ+fsize+hown+inc+male+marr+pira+twoearn, 
              data = pension)
Uhat <- ols_net_tfa$residuals

# 3. Run a residual-on-residual regression, U_Y∼X ~ U_W∼X
# U_hat_Y∼X = θU_hat_W∼X + ε
summary(lm(Uhat ~ 0 + Vhat))
ols$coef[2]

all.equal(as.numeric(coef(lm(Uhat ~ 0 + Vhat))), 
          as.numeric(ols$coef[2]))


# Panel methods

# Fixed-effects regression
# Y_it = W_it*θ + X_it*β + α_i + ε_it
# Y_it - Y_mean = (W_it - W_mean)*θ + (X_it - X_mean)*β + α_i + (ε_it - ε_mean)
## the θ is unaffected


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Double_selection.nb.html
# Simulation notebook: Causal ML: Double Selection
library(tidyverse)
library(hdm)
library(mvtnorm)
library(estimatr)


# Bias and MSE
# Same coefficients for treatment and outcome
set.seed(1234)
n <- 100
p <- 100
n_rep <- 1000

# Define and plot parameters
theta <- 0
delta <- c(seq(1, 0.1, -0.1), rep(0, p-10))
beta <- delta
cov_mat <- toeplitz(0.7^(0:(p - 1)))
plot(delta)
abline(h = 0)

plot(beta)
abline(h = 0)

# Generate one draw
x <- rmvnorm(n = n, mean = rep(0, p), sigma = cov_mat)
w <- x %*% delta + rnorm(n,0,1)
y <- theta*w + x %*% beta + rnorm(n,0,4)

# Select variables in outcome regression
sel_y <- rlasso(x,y)
# Which variables are selected
which(sel_y$beta != 0)

# Run single-selection OLS
x_sel_y <- x[,sel_y$beta != 0]
summary(lm_robust(y ~ w + x_sel_y))
help("lm_robust")

# Select variables in treatment regression
sel_w <- rlasso(x,w)
which(sel_w$beta != 0)

# Double selection
x_sel_union <- x[,sel_y$beta != 0 | sel_w$beta != 0]
dim(x_sel_union)
dim(x_sel_y)
summary(lm_robust(y ~ w + x_sel_union))

ds <- rlassoEffect(x,y,w)
summary(ds)
help(rlassoEffect)


# Repeatedly draw 1000 samples and check 
# the distribution of the resulting coefficient for single and Double Selection:
results <- matrix(NA,n_rep,2)
colnames(results) <- c("Single","Double")

for (i in 1:n_rep) {
  x = rmvnorm(n = n, mean = rep(0, p), sigma = cov_mat)
  w = x %*% delta + rnorm(n,0,1)
  y = theta*w + x %*% beta + rnorm(n,0,4)
  
  sel_y = rlasso(x,y)
  x_sel_y = x[,sel_y$beta != 0]
  results[i,1] = lm(y ~ w + x_sel_y)$coefficients[2]
  results[i,2] = rlassoEffect(x,y,w)$alpha
  
  print(i)
}

as.data.frame(results) %>% 
  pivot_longer(cols=everything(),names_to = "Selection",values_to = "coef") %>%  
  ggplot(aes(x = coef, fill = Selection)) + geom_density(alpha = 0.5) + geom_vline(xintercept = theta)

cat("Bias:\n")
round(colMeans(results)-theta,4)
cat("\nMSE:\n")
round(colMeans((results-theta)^2),4)


# Asymmetric coefficients for treatment and outcome
# Define and plot parameters
delta <- c(seq(0.1,1,0.1),rep(0,p-10))
beta <- c(seq(1,0.1,-0.1),rep(0,p-10))
plot(delta)
abline(h = 0)

plot(beta)
abline(h = 0)

results <- matrix(NA,n_rep,2)
colnames(results) <- c("Single","Double")

# Repeatedly draw 1000 samples and check 
for (i in 1:n_rep) {
  x = rmvnorm(n = n, mean = rep(0, p), sigma = cov_mat)
  w = x %*% delta + rnorm(n,0,1)
  y = theta*w + x %*% beta + rnorm(n,0,4)
  
  sel_y = rlasso(x,y)
  x_sel_y = x[,sel_y$beta != 0]
  results[i,1] = lm(y ~ w + x_sel_y)$coefficients[2]
  results[i,2] = rlassoEffect(x,y,w)$alpha
  
  print(i)
}

as.data.frame(results) %>% 
  pivot_longer(cols=everything(),names_to = "Selection",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Selection)) + geom_density(alpha = 0.5) + geom_vline(xintercept = theta)

cat("Bias:\n")
round(colMeans(results)-theta,4)
cat("\nMSE:\n")
round(colMeans((results-theta)^2),4)


# Dense DGP
# Define and plot parameters
delta <- c(seq(0.1,1,0.1),rep(0.1,p-10))
beta <- c(seq(1,0.1,-0.1),rep(0.1,p-10))
plot(delta, ylim = c(0, 1))
abline(h = 0)

plot(beta, ylim = c(0, 1))
abline(h = 0)

results = matrix(NA,n_rep,2)
colnames(results) = c("Single","Double")

# Repeatedly draw 1000 samples and check 
for (i in 1:n_rep) {
  x = rmvnorm(n = n, mean = rep(0, p), sigma = cov_mat)
  w = x %*% delta + rnorm(n,0,1)
  y = theta*w + x %*% beta + rnorm(n,0,4)
  
  sel_y = rlasso(x,y)
  x_sel_y = x[,sel_y$beta != 0]
  results[i,1] = lm(y ~ w + x_sel_y)$coefficients[2]
  results[i,2] = rlassoEffect(x,y,w)$alpha
  
  print(i)
}

as.data.frame(results) %>% 
  pivot_longer(cols=everything(),names_to = "Selection",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Selection)) + geom_density(alpha = 0.5) + geom_vline(xintercept = theta)


cat("Bias:\n")
round(colMeans(results)-theta,4)
cat("\nMSE:\n")
round(colMeans((results-theta)^2),4)


# Coverage rates
set.seed(1234)

# Simulation settings
n <- 100       # Sample size
p <- 100       # Number of covariates
n_rep <- 1000  # Number of replications
theta <- 0     # True treatment effect

sign_flip <- rep(c(1,-1),p/2)

# DGP specifications
cov_mat <- toeplitz(0.7^(0:(p - 1)))
sparsity_patterns <- list(
  same = rbind(c(seq(1, 0.1, -0.1), rep(0, p-10)),
               c(seq(1, 0.1, -0.1), rep(0, p-10))),
  asymmetric = rbind(c(seq(0.1, 1, 0.1), rep(0, p-10)),
                     c(seq(1, 0.1, -0.1), rep(0, p-10))),
  dense = rbind(c(seq(0.1, 1, 0.1), rep(0.1, p-10)),
                c(seq(1, 0.1, -0.1), rep(0.1, p-10)))
)

# Function to simulate data and return coverage
simulate_coverage <- function(delta, beta) {
  effect = coverage = matrix(NA, n_rep, 2)
  colnames(effect) = colnames(coverage) = c("Single", "Double")
  
  for (i in 1:n_rep) {
    x = rmvnorm(n, mean = rep(0, p), sigma = cov_mat)
    w = x %*% delta + rnorm(n, 0, 1)
    y = theta * w + x %*% beta + rnorm(n, 0, 4)
    
    # Single selection
    sel_y = rlasso(x, y)
    model_single = lm_robust(y ~ w + x[, sel_y$beta != 0])
    effect[i,1] = model_single$coefficients[2]
    ci_single = confint(model_single)["w", ]
    
    # Double selection
    sel_w = rlasso(x, w)
    union_selection = sel_y$beta != 0 | sel_w$beta != 0
    model_double = lm_robust(y ~ w + x[, union_selection])
    effect[i,2] = model_double$coefficients[2]
    ci_double = confint(model_double)["w", ]
    
    # Check coverage
    coverage[i, 1] = theta >= ci_single[1] & theta <= ci_single[2]
    coverage[i, 2] = theta >= ci_double[1] & theta <= ci_double[2]
  }
  
  list(bias = colMeans(effect)-theta, mse = colMeans((effect-theta)^2), coverage = colMeans(coverage))
}

# Run simulations for each DGP
results_bias = results_mse = results_cr = tibble(
  DGP = c("Same", "Asymmetric", "Dense"),
  Single = numeric(length(sparsity_patterns)),
  Double = numeric(length(sparsity_patterns))
)

for (i in seq_along(sparsity_patterns)) {
  delta = sparsity_patterns[[i]][1,]
  beta = sparsity_patterns[[i]][2,]
  run = simulate_coverage(delta, beta)
  results_bias[i, 2:3] = t(run$bias)
  results_mse[i, 2:3] = t(run$mse)
  results_cr[i, 2:3] = t(run$coverage)
}

print(results_cr)


# Ensure the order of DGP is maintained in the plot
results_cr$DGP <- factor(results_cr$DGP, levels = c("Same", "Asymmetric", "Dense"))

# Convert data from wide to long format for ggplot2
results_long <- pivot_longer(results_cr, cols = c(Single, Double), 
                             names_to = "Selection", values_to = "CoverageRate")

# Create the bar plot
ggplot(results_long, aes(x = DGP, y = CoverageRate, fill = Selection)) + 
  geom_bar(stat = "identity", position = position_dodge(width = 0.7)) + 
  scale_y_continuous(labels = scales::percent_format()) + 
  scale_fill_brewer(palette = "Pastel1", direction = -1) + 
  labs(title = "Coverage Rates for Single vs Double Selection", 
       y = "Coverage Rate (%)", 
       x = "DGP Type", 
       fill = "Selection Method") + 
  theme_minimal() + 
  theme(legend.position = "top") + 
  geom_hline(yintercept = c(0,0.95,1), linetype = c("solid","dashed","solid"), 
             color = c("black","red","black"), linewidth = 0.7) +
  geom_text(aes(label=scales::percent(CoverageRate), group=Selection), 
            position=position_dodge(width = 0.7), vjust=-0.25)


# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Double_selection_and_partially_linear_DML.nb.html
# Application notebook: Causal ML: Double Selection and Partially Linear Double ML

# Part 1

# Load the packages required for later
library(hdm)
library(tidyverse)
library(causalDML)
library(grf)
library(estimatr)

set.seed(1234) # for replicability

data(pension)
# Outcome
Y <- pension$net_tfa
# Treatment
W <- pension$p401
# Create main effects matrix
X <- model.matrix(~ 0+age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, data = pension)


# Hand-coded Double Selection

# 1. Select variables in outcome regression
sel_y <- rlasso(X,Y)
# Which variables are selected?
which(sel_y$beta != 0)

# 2. Select variables in treatment regression
sel_w <- rlasso(X,W)
which(sel_w$beta != 0)

# 3. Double selection
X_sel_union <- X[,sel_y$beta != 0 | sel_w$beta != 0]
dim(X_sel_union)
ds_hand <- lm_robust(Y ~ W + X_sel_union)
summary(ds_hand)


# Double Selection with hdm package
ds1 <- rlassoEffect(X,Y,W)
summary(ds1)
all.equal(as.numeric(ds1$alpha),
          as.numeric(ds_hand$coefficients[2]))

# More flexible dictionaries
# X2 with 88 variables
X2 <- model.matrix(~ 0 + (fsize+marr+twoearn+db+pira+hown+male
                          +poly(age,2)+poly(educ,2)+poly(inc,2))^2, data = pension)
# X3 with 567 variables
X3 <- model.matrix(~ 0 + (fsize+marr+twoearn+db+pira+hown+male
                          +poly(age,2)+poly(educ,2)+poly(inc,2))^2, data = pension)

ds2 <- rlassoEffect(X2,Y,W)
summary(ds2)
ds3 <- rlassoEffect(X3,Y,W)
summary(ds3)


# Hand-coded Double ML for partially linear model



# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Partially_linear.nb.html
# Simulation notebook: Causal ML: Partially linear Double ML
# library(devtools)
# install_github(repo="MCKnaus/causalDML")
library(hdm)
library(grf)
library(causalDML)
library(tidyverse)
library(patchwork)

set.seed(1234)

n <- 100
p <- 10
theta <- 0.1

x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){sin(x)}
m <- function(x){theta*e(x) + sin(x)}
w <- e(x[,1]) + rnorm(n,0,1)
y <- theta*w + sin(x[,1]) + rnorm(n,0,1)

df <- data.frame(x=x[,1],w,y)
g1 <- ggplot(df,aes(x=x, y=w)) + stat_function(fun=e,linewidth=1) + ylab("W and e(x)") + geom_point() + xlab("X1")
g2 <- ggplot(df,aes(x=x, y=y)) + stat_function(fun=m,linewidth=1) + ylab("Y and m(x)") + geom_point() + xlab("X1")
g1 | g2


# Hand-coded residual-on-residual regression w/o cross-fitting

# estimate the nuisance parameters e(X)=E[W|X]and m(X)=E[Y|X]
# using random forest without honesty

# No cross-fitting
rf <- regression_forest(x,w,honesty=F)
ehat <- predict(rf,newdata=x)$predictions
# Predict outcome in-sample
rf <- regression_forest(x,y,honesty=F)
mhat <- predict(rf,newdata=x)$predictions
# Get residuals
res_y <- y-mhat
res_w <- w-ehat
# Run residual-on-residual regression (RORR)
lm_robust(res_y ~ 0+res_w)$coefficients

# the τ_hat value
cat("Fully hand-coded:\n",mean(res_y * res_w) / mean(res_w^2))
# 0.1832594


# Hand-coded residual-on-residual regression with 2-fold cross-fitting

# Initialize nuisance vectors
mhat = ehat = rep(NA,n)
# Draw random indices for sample 1
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
rf <- regression_forest(x1,y1,honesty=F)
mhat[-index_s1] <- predict(rf,newdata=x2)$predictions
# Model in S2, predict in S1
rf <- regression_forest(x2,w2,honesty=F)
ehat[index_s1] <- predict(rf,newdata=x1)$predictions
rf <- regression_forest(x2,y2,honesty=F)
mhat[index_s1] <- predict(rf,newdata=x1)$predictions
# RORR
res_y <- y-mhat
res_w <- w-ehat
lm(res_y ~ 0+res_w)$coefficients
# 0.1832594 

all.equal(as.numeric(mean(res_y * res_w) / mean(res_w^2)),
          as.numeric(lm(res_y ~ 0+res_w)$coefficients))


# Residual-on-residual regression with 5-fold cross-fitting

# 5-fold cross-fitting with causalDML package
# Create learner
set.seed(1234)
forest <- create_method("forest_grf",args=list(honesty=F))
# Run partially linear model
pl_cf5 <- DML_partial_linear(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)
summary(pl_cf5)
# Coefficient       SE      t      p
# [1,]    0.119599 0.089044 1.3431 0.1823


# Simulation study

# set number of replications
n_rep <- 100
# Initialize results matrix
results <- matrix(NA,n_rep,8)
colnames(results) <- c("DS1","DS2","DS3","DS4","DS5","PL no cf","PL cf2","PL cf5")
# run the simulation
for (i in 1:n_rep) {
  x = matrix(runif(n*p,-pi,pi),ncol=p)
  w = e(x[,1]) + rnorm(n,0,1)
  y = theta*w + sin(x[,1]) + rnorm(n,0,1)
  
  # double selections
  results[i,1] = rlassoEffect(x,y,w)$alpha
  x2 = cbind(x,x^2)
  results[i,2] = rlassoEffect(x2,y,w)$alpha
  x3 = cbind(x2,x^3)
  results[i,3] = rlassoEffect(x3,y,w)$alpha
  x4 = cbind(x3,x^4)
  results[i,4] = rlassoEffect(x4,y,w)$alpha
  x5 = cbind(x4,x^5)
  results[i,5] = rlassoEffect(x5,y,w)$alpha
  
  
  # No cross-fitting
  rf = regression_forest(x,w,honesty=F)
  ehat = predict(rf,newdata=x)$predictions
  rf = regression_forest(x,y,honesty=F)
  mhat = predict(rf,newdata=x)$predictions
  res_y = y-mhat
  res_w = w-ehat
  results[i,6] = lm(res_y ~ 0+res_w)$coefficients
  
  # 2-fold cross-fitting
  mhat = ehat = rep(NA,n)
  index_s1 = sample(1:n,n/2)
  x1 = x[index_s1,]
  w1 = w[index_s1]
  y1 = y[index_s1]
  x2 = x[-index_s1,]
  w2 = w[-index_s1]
  y2 = y[-index_s1]
  rf = regression_forest(x1,w1,honesty=F)
  ehat[-index_s1] = predict(rf,newdata=x2)$predictions
  rf = regression_forest(x1,y1,honesty=F)
  mhat[-index_s1] = predict(rf,newdata=x2)$predictions
  rf = regression_forest(x2,w2,honesty=F)
  ehat[index_s1] = predict(rf,newdata=x1)$predictions
  rf = regression_forest(x2,y2,honesty=F)
  mhat[index_s1] = predict(rf,newdata=x1)$predictions
  res_y = y-mhat
  res_w = w-ehat
  results[i,7] = lm(res_y ~ 0+res_w)$coefficients
  
  # 5-fold cross-fitting
  results[i,8] = DML_partial_linear(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)$result[1]

  print(i) # time: 5s per i
}

as.data.frame(results[,c(5,6,7,8)]) %>% pivot_longer(cols=everything(),names_to = "Estimator",values_to = "coef") %>%
  ggplot(aes(x = coef, fill = Estimator)) + geom_density(alpha=0.4) + theme_bw() + geom_vline(xintercept=theta)


data.frame(method = colnames(results),
           bias2 = colMeans(results-theta)^2,
           var = colMeans(sweep(results,2,colMeans(results))^2)) %>% 
  pivot_longer(-method,names_to = "Component",values_to = "MSE") %>%
  ggplot(aes(fill=factor(Component,levels=c("var","bias2")), y=MSE, x=method)) + 
  geom_bar(position="stack", stat="identity") + scale_fill_discrete(name = "Component")



# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Double_selection_and_partially_linear_DML.nb.html
# Application notebook: Causal ML: Double Selection and Partially Linear Double ML

# Part 2

# Load the packages required for later
library(hdm)
library(tidyverse)
library(causalDML)
library(grf)
library(estimatr)

set.seed(1234) # for replicability

data(pension)
# Outcome
Y <- pension$net_tfa
# Treatment
W <- pension$p401
# Create main effects matrix
X <- model.matrix(~ 0+age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, data = pension)

# Hand-coded Double ML for partially linear model
# Initialize nuisance vectors
n <- length(Y)
mhat = ehat = rep(NA,n)
# Draw random indices for sample 1
index_s1 <- sample(1:n,n/2)
length(index_s1)
# Create S1
x1 <- X[index_s1,]
w1 <- W[index_s1]
y1 <- Y[index_s1]
# Create sample 2 with those not in S1
x2 <- X[-index_s1,]
w2 <- W[-index_s1]
y2 <- Y[-index_s1]
# Model in S1, predict in S2
rf <- regression_forest(x1,w1)
ehat[-index_s1] <- predict(rf,newdata=x2)$predictions
rf <- regression_forest(x1,y1)
mhat[-index_s1] <- predict(rf,newdata=x2)$predictions
# Model in S2, predict in S1
rf <- regression_forest(x2,w2)
ehat[index_s1] <- predict(rf,newdata=x1)$predictions
rf <- regression_forest(x2,y2)
mhat[index_s1] <- predict(rf,newdata=x1)$predictions
# RORR
res_y <- Y-mhat
res_w <- W-ehat
pl_2f <- lm_robust(res_y ~ 0+res_w)
summary(pl_2f)
#          Estimate Std. Error t value Pr(>|t|) CI Lower CI Upper   DF
# res_w    13944       1517   9.191 4.68e-20    10970    16918 9914

# Double ML for partially linear model with causalDML package

# 5-fold cross-fitting with causalDML package
# Create learner
set.seed(1234)
forest <- create_method("forest_grf",args=list(tune.parameters = "all"))
# Run partially linear model
pl_5f <- DML_partial_linear(Y,W,X,ml_w=list(forest),ml_y=list(forest),cf=5)  
summary(pl_5f)
#          Coefficient      SE      t         p    
# [1,]     13756.4  1514.7 9.0818 < 2.2e-16 ***

help(DML_partial_linear)

# Comparison of results

# Collect the results
Coefficient <- c(ds1$alpha,ds2$alpha,ds3$alpha,pl_2f$coefficients,pl_5f$result[1])
se <- c(ds1$se,ds2$se,ds3$se,pl_2f$std.error,pl_5f$result[2])
data.frame(Coefficient,se,
           Method = c("DS1","DS2","DS3","PL 2-fold","PL 5-fold"),
           cil = Coefficient - 1.96*se,
           ciu = Coefficient + 1.96*se)  %>% 
  ggplot(aes(x=Method,y=Coefficient,ymin=cil,ymax=ciu)) + geom_point(size=2.5) + geom_errorbar(width=0.15)  +
  geom_hline(yintercept=0)



