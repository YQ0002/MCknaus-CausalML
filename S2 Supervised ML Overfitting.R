# https://mcknaus.github.io/assets/notebooks/SNB/SNB_OLS_in_vs_out_of_sample.nb.html
rm(list = ls())


# Supervised ML: Overfitting of OLS and value of training vs. test sample


library(tidyverse)
library(skimr)

set.seed(1234)
n_tr <- 100
n_te <- 10000
p <- 99
beta <- c(0,seq(1,0.1,-0.1),rep(0,p-10))
summary(beta)

# train and test set
x_tr <- cbind(rep(1,n_tr),matrix(rnorm(n_tr*p),ncol=p))
x_te <- cbind(rep(1,n_te),matrix(rnorm(n_te*p),ncol=p))

dim(x_tr)
dim(x_te)
head(x_tr)

# Create the CEF
cfe_tr <- x_tr %*% beta
cfe_te <- x_te %*% beta

y_tr <- cfe_tr + rnorm(n_tr,0,1)
y_te <- cfe_te + rnorm(n_te,0,1)

skim(x_tr)
hist(cfe_te)


# Perfectly specified OLS

perfect_ols <- lm(y_tr~x_tr[,2:11])
summary(perfect_ols)

yhat_pols_tr <- predict(perfect_ols)
plot(yhat_pols_tr,y_tr)

yhat_pols_te <- x_te[,1:11] %*% perfect_ols$coef
plot(yhat_pols_te, y_te)

# MSE
paste("In-sample MSE:", mean((y_tr - yhat_pols_tr)^2))
paste("Out-of-sample MSE:", mean((y_te - yhat_pols_te)^2))
# Note: signal of overfitting


# The failure of in-sample measures


# Container of the results
results_ols <- matrix(NA,p,4)
colnames(results_ols) <- c("Obs MSE train","Obs MSE test","Oracle MSE train","Oracle MSE test")

# Loop that gradually adds variables
for (i in 1:p) {
  temp_ols <- lm(y_tr~x_tr[,2:(i+1)])
  temp_yhat_tr <- predict(temp_ols)
  temp_yhat_te <- x_te[, 1:(i+1)] %*% temp_ols$coef
  
  # Calculate the observable MSEs in training and test sample
  results_ols[i,1] <- mean((y_tr - temp_yhat_tr)^2) # in-sample MSE
  results_ols[i,2] <- mean((y_te - temp_yhat_te)^2) # out-of-sample MSE
  
  # Calculate the oracle MSEs that are only observables b/c we know the true CEF
  results_ols[i,3] <- var(y_tr - cfe_tr) + mean((cfe_tr - temp_yhat_tr)^2)
  results_ols[i,4] <- var(y_te - cfe_te) + mean((cfe_te - temp_yhat_te)^2)
}

df <- data.frame("Number.of.variables"=1:p, results_ols)
ggplot(df, aes(Number.of.variables)) + 
  geom_line(aes(y = Obs.MSE.train, colour = "Obs.MSE.train"),size=1) + 
  ylab("MSE") + geom_hline(yintercept = 0)

paste("In-sample MSE with constant and ",p,"predictors:", round(results_ols[p,1]))


# The value of out-of-sample validation
ggplot(df, aes(Number.of.variables)) + 
  geom_line(aes(y = Obs.MSE.train, colour = "Obs.MSE.train"),size=1) + 
  geom_line(aes(y = Obs.MSE.test, colour = "Obs.MSE.test"),size=1) + ylab("MSE") +
  geom_hline(yintercept = 0)

# cut off the extreme values from the right
ggplot(df[1:80,], aes(Number.of.variables)) + 
  geom_line(aes(y = Obs.MSE.train, colour = "Obs.MSE.train"),size=1) + 
  geom_line(aes(y = Obs.MSE.test, colour = "Obs.MSE.test"),size=1) + ylab("MSE") +
  geom_hline(yintercept = 0)

# the feasible MSE compares to the infeasible oracle MSE
ggplot(df[1:80,], aes(Number.of.variables)) + 
  geom_line(aes(y = Obs.MSE.test, colour = "Obs.MSE.test"),size=1) + 
  geom_line(aes(y = Oracle.MSE.test, colour = "Oracle.MSE.test"),size=1) + ylab("MSE") +
  geom_hline(yintercept = 0)


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Lasso_saves_OLS.nb.html
# Lasso saves the job of OLS


# library(tidyverse)
library(glmnet)
library(hdm)
library(plasso)


# Lasso at work

set.seed(1234)
# glmnet takes inputs in matrix form and not as formulas
lasso <- glmnet(x_tr, y_tr)
plot(lasso, xvar = "lambda")

# Cross-validation
cv_lasso <- cv.glmnet(x_tr, y_tr) # 10-fold
plot(cv_lasso)
bestlam <- cv_lasso$lambda.min
log(bestlam)

# glmnet takes inputs in matrix form and not as formulas
lasso_pred <- predict(lasso, s = bestlam, newx = x_te)
mean((lasso_pred - y_te)^2)


# Post-Lasso at work
# Cross-validated


# Provide the cov matrix w/o the constant x_tr[,-1]
# Increasing lambda.min.ratio ensures that Lasso does not overfit too heavily and reduces running time
post_lasso <- plasso(x_tr[,-1], y_tr, lambda.min.ratio=0.01)
plot(post_lasso, xvar = "lambda")

# Post-Lasso gives the full OLS coefficient as soon as the variable is selected

cv_plasso <- cv.plasso(x_tr, y_tr, lambda.min.ratio=0.01)
plot(cv_plasso, legend_pos = "bottomleft")

names(cv_plasso)
cv_plasso$lambda_min_l


# Fast implementation with hdm

post_hdm <- rlasso(x_tr, y_tr)
summary(post_hdm)


# OLS vs. (Post-)Lasso

# Container of the results
results_ols <- results_lasso <- 
               results_plasso <- 
               results_rlasso <- 
               matrix(NA,p-1,4)
colnames(results_ols) <- colnames(results_lasso) <-
                         colnames(results_plasso) <- 
                         colnames(results_rlasso) <- 
                         c("Obs MSE train","Obs MSE test", 
                           "Oracle MSE train","Oracle MSE test")
p
# Loop that gradually adds variables (start with 2, otherwise glmnet crashes)
for (i in 2:p) {
  # OLS
  temp_ols = lm(y_tr ~ x_tr[,2:(i+1)])
  temp_yhat_tr = predict(temp_ols)
  temp_yhat_te = x_te[,1:(i+1)] %*% temp_ols$coefficients
  # Calculate the observable MSEs in training and test sample
  results_ols[i-1,1] = mean((y_tr - temp_yhat_tr)^2) # in-sample MSE
  results_ols[i-1,2] = mean((y_te - temp_yhat_te)^2) # out-of-sample MSE
  # Calculate the oracle MSEs that are only observables b/c we know the true CEF
  results_ols[i-1,3] = var(y_tr - cfe_tr) + mean((cfe_tr - temp_yhat_tr)^2)
  results_ols[i-1,4] = var(y_te - cfe_te) + mean((cfe_te - temp_yhat_te)^2)
  
  # Lasso
  temp_lasso = cv.glmnet(x_tr[,2:(i+1)],y_tr)
  temp_yhat_tr = predict(temp_lasso, newx = x_tr[,2:(i+1)])
  temp_yhat_te = predict(temp_lasso, newx = x_te[,2:(i+1)])
  # Calculate the observable MSEs in training and test sample
  results_lasso[i-1,1] = mean((y_tr - temp_yhat_tr)^2) # in-sample MSE
  results_lasso[i-1,2] = mean((y_te - temp_yhat_te)^2) # out-of-sample MSE
  # Calculate the oracle MSEs that are only observables b/c we know the true CEF
  results_lasso[i-1,3] = var(y_tr - cfe_tr) + mean((cfe_tr - temp_yhat_tr)^2)
  results_lasso[i-1,4] = var(y_te - cfe_te) + mean((cfe_te - temp_yhat_te)^2)
  
  # plasso
  temp_plasso = cv.plasso(x_tr[,2:(i+1)],y_tr)
  temp_yhat_tr = predict(temp_plasso, newx = x_tr[,2:(i+1)])$plasso
  temp_yhat_te = predict(temp_plasso, newx = x_te[,2:(i+1)])$plasso
  # Calculate the observable MSEs in training and test sample
  results_plasso[i-1,1] = mean((y_tr - temp_yhat_tr)^2) # in-sample MSE
  results_plasso[i-1,2] = mean((y_te - temp_yhat_te)^2) # out-of-sample MSE
  # Calculate the oracle MSEs that are only observables b/c we know the true CEF
  results_plasso[i-1,3] = var(y_tr - cfe_tr) + mean((cfe_tr - temp_yhat_tr)^2)
  results_plasso[i-1,4] = var(y_te - cfe_te) + mean((cfe_te - temp_yhat_te)^2)
  
  # rlasso
  temp_rlasso = rlasso(x_tr[,2:(i+1)],y_tr)
  temp_yhat_tr = predict(temp_rlasso,newdata=x_tr[,2:(i+1)])
  temp_yhat_te = predict(temp_rlasso,newdata=x_te[,2:(i+1)])
  # Calculate the observable MSEs in training and test sample
  results_rlasso[i-1,1] = mean((y_tr - temp_yhat_tr)^2) # in-sample MSE
  results_rlasso[i-1,2] = mean((y_te - temp_yhat_te)^2) # out-of-sample MSE
  # Calculate the oracle MSEs that are only observables b/c we know the true CEF
  results_rlasso[i-1,3] = var(y_tr - cfe_tr) + mean((cfe_tr - temp_yhat_tr)^2)
  results_rlasso[i-1,4] = var(y_te - cfe_te) + mean((cfe_te - temp_yhat_te)^2)
}

df <- data.frame(Estimator = c(rep("OLS",p-1),
                              rep("Lasso",p-1),
                              rep("Post-Lasso CV",p-1),
                              rep("Post-Lasso hdm",p-1)),
                 Number.of.variables = c(2:p,2:p,2:p,2:p),
                 Obs.MSE.test = c(results_ols[,2], 
                                 results_lasso[,2], 
                                 results_plasso[,2], 
                                 results_rlasso[,2]),
                 Oracle.MSE.test = c(results_ols[,4], 
                                    results_lasso[,4], 
                                    results_plasso[,4],
                                    results_plasso[,4]))
dim(df)
ggplot(subset(df), aes(x=Number.of.variables,y=Obs.MSE.test,colour=Estimator)) +
  geom_line(size=1)

ggplot(subset(df,df$Number.of.variables<70), aes(x=Number.of.variables,y=Obs.MSE.test,colour=Estimator))     + geom_line(size=1) + 
  geom_hline(yintercept = 0)


# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Lasso.nb.html
# Application notebook: Lasso

library(glmnet)
library(hdm)
library(tidyverse)
library(skimr)

# Supervised ML: Lasso
set.seed(1234)
options(scipen = 999) # Switch off scientific notation
data(pension)
help(pension)
names(pension)

Y <- pension$net_tfa # net assets
X1<- model.matrix(~0+age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, data=pension)

skim(X1)
hist(Y)
summary(Y)


# First steps with (Post-)Lasso

# Lasso using glmnet package
lasso <- glmnet(X1,Y)
plot(lasso, xvar = "lambda",label=TRUE)

# standardized 
lasso <- glmnet(scale(X1),Y)
plot(lasso, xvar = "lambda",label=TRUE)
cv_lasso <- cv.glmnet(X1,Y)
plot(cv_lasso)

coef(cv_lasso, s = "lambda.min")

# OLS 
summary(lm(Y~X1))

# Post-Lasso using hdm
post_lasso <- rlasso(X1, Y)
summary(post_lasso)

# OLS using selected variables
summary(lm(Y~X1[,post_lasso$coefficients[-1] != 0]))


# Out-of-sample prediction with (Post-)Lasso

# Create training and testing sample
test_fraction <- 1/3
test_size <- floor(test_fraction*length(Y))
# Index for test observations
test_ind <- sample(1:length(Y), size = test_size)
# Create training and test data
X_tr <- X1[-test_ind,]
X_te <- X1[test_ind,]
Y_tr <- Y[-test_ind]
Y_te <- Y[test_ind]

cv_lasso <- cv.glmnet(X_tr, Y_tr)
Y_hat_lasso <- predict(cv_lasso, newx = X_te, s = "lambda.min")

hist(Y_hat_lasso)
plot(Y_hat_lasso, Y_te)

cor(Y_hat_lasso, Y_te)

mse_lasso <- mean((Y_te-Y_hat_lasso)^2)
mse_lasso

# R2
1 - mse_lasso/var(Y_te)

Y_hat_plasso <- predict(post_lasso, newdata = X_te)
mse_plasso <- mean((Y_te - Y_hat_plasso)^2)
1 - mse_plasso/var(Y_te)


# Including interactions and compare performance
# Note: time consuming
X2 <- model.matrix(~0+(fsize+marr+twoearn+db+pira+hown+male+
                         poly(age,2)+poly(educ,2)+poly(inc,2))^2, data=pension)
X3 <- model.matrix(~0+(fsize+marr+twoearn+db+pira+hown+male+
                         poly(age,3)+poly(educ,3)+poly(inc,3))^3, data=pension)
X4 <- model.matrix(~0+(fsize+marr+twoearn+db+pira+hown+male+
                         poly(age,4)+poly(educ,4)+poly(inc,4))^4, data=pension)
dim(X2)
dim(X3)
dim(X4)

# Run the method in the training sample and calculate the test set R2
ols_oos_r2 <- function(x_tr, y_tr, x_te, y_te){
  ols = lm(y_tr~x_tr)
  betas = ols$coef
  betas[is.na(betas)] = 0
  y_hat = cbind(rep(1, nrow(x_te)), x_te) %*% betas
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}
lasso_oos_r2 <- function(x_tr,y_tr,x_te,y_te,min.lambda=1e-04){
  cv_lasso = cv.glmnet(x_tr, y_tr, lambda.min.ratio = min.lambda)
  y_hat <- predict(cv_lasso, newx = x_te, s = "lambda.min")
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}
plasso_oos_r2 <- function(x_tr,y_tr,x_te,y_te,min.lambda=1e-04){
  plasso = rlasso(x_tr, y_tr)
  y_hat = predict(plasso, newdata = x_te)
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}

rep <- 20 # number of replications
# Time Consuming

# Container of the results
results_r2 <- matrix(NA, rep, 12)
colnames(results_r2) <- c("OLS1","OLS2","OLS3","OLS4",
                          "Lasso1","Lasso2","Lasso3","Lasso4",
                          "Post-Lasso1","Post-Lasso2","Post-Lasso3","Post-Lasso4")
# Loop considering different splits
for (i in 1:rep) {
  # Draw index for this round
  temp_ind = sample(1:length(Y), size = test_size)
  
  # Split into training and test samples
  X_tr1 = X1[-temp_ind,]
  X_te1 = X1[temp_ind,]
  X_tr2 = X2[-temp_ind,]
  X_te2 = X2[temp_ind,]
  X_tr3 = X3[-temp_ind,]
  X_te3 = X3[temp_ind,]
  X_tr4 = X4[-temp_ind,]
  X_te4 = X4[temp_ind,]
  Y_tr = Y[-temp_ind]
  Y_te = Y[temp_ind]
  
  # Get test R2 for method-cov matrix combi
  results_r2[i,1] = ols_oos_r2(X_tr1,Y_tr,X_te1,Y_te)
  results_r2[i,2] = ols_oos_r2(X_tr2,Y_tr,X_te2,Y_te)
  results_r2[i,3] = ols_oos_r2(X_tr3,Y_tr,X_te3,Y_te)
  results_r2[i,4] = ols_oos_r2(X_tr4,Y_tr,X_te4,Y_te)
  results_r2[i,5] = lasso_oos_r2(X_tr1,Y_tr,X_te1,Y_te)
  results_r2[i,6] = lasso_oos_r2(X_tr2,Y_tr,X_te2,Y_te)
  
  # Increasing min.lambda to speed up computation
  results_r2[i,7] = lasso_oos_r2(X_tr3,Y_tr,X_te3,Y_te,min.lambda = 0.01)
  results_r2[i,8] = lasso_oos_r2(X_tr4,Y_tr,X_te4,Y_te,min.lambda = 0.05)
  results_r2[i,9] = plasso_oos_r2(X_tr1,Y_tr,X_te1,Y_te)
  results_r2[i,10] = plasso_oos_r2(X_tr2,Y_tr,X_te2,Y_te)
  results_r2[i,11] = plasso_oos_r2(X_tr3,Y_tr,X_te3,Y_te)
  # results_r2[i,12] = plasso_oos_r2(X_tr4,Y_tr,X_te4,Y_te)
}

t(round(colMeans(results_r2),3))

as.data.frame(results_r2[,-c(3:4)]) %>% pivot_longer(cols=everything(),names_to = "Method",values_to = "R2") %>%
  ggplot(aes(x = R2, y = Method)) + geom_boxplot()


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Tree_based.nb.html

# Simulation notebook: Tree-based methods
# Supervised ML: Tree-based methods

# library(tidyverse)
library(rpart)
library(partykit)
library(grf)
library(rpart.plot)

set.seed(1234)
cef <- function(x){-x^3 + 2*x}
n <- 500
x <- runif(n)
y <- cef(x) + rnorm(n,0,1/3)
df <- data.frame(x,y)
ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y),color="black",alpha = 0.4)


# Different ways to fit the data
# OLS
df$y_hat_ols <- predict(lm(y~x))
ggplot(df,aes(x=x,y=y)) + stat_function(fun=cef,size=1) + 
  geom_point(color="black",alpha = 0.4) + 
  geom_point(aes(x=x,y=y_hat_ols),shape="square",color="blue") + 
  geom_smooth(formula="y~x", method='lm') 

# Regression tree
tree <- rpart(y~x, data = df)
rpart.plot(tree)
help(rpart)

df$y_hat_tree <- predict(tree)
ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y),color="black",alpha = 0.4) +
  geom_point(aes(x=x,y=y_hat_tree),shape="square",color="blue") 

# Random Forest
rf <- regression_forest(as.matrix(x), y, tune.parameters = "all", num.trees = 2000)
df$y_hat_rf <- predict(rf)$predictions

ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y),color="black",alpha = 0.4) +
  geom_point(aes(x=x,y=y_hat_rf),shape="square",color="blue")


# Global vs. local predictors
test_point <- 0.2

# OLS
X <- cbind(rep(1,n),x)
predict_ols <- as.numeric(c(1, test_point) %*% solve(t(X) %*% X) %*% t(X) %*% y)
df$w_ols <- t(c(1,test_point) %*% solve(t(X) %*% X) %*% t(X))

w_sign_ols <- rep("negative", n)
w_sign_ols[df$w_ols>0] = 'positive'
w_sign_ols = factor(w_sign_ols, level=c("negative", "positive"))

ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y,size=abs(w_ols),color=w_sign_ols),alpha = 0.2) +
  geom_point(x=test_point,y=predict_ols,shape="cross",size = 3, stroke = 2,color="yellow") + 
  scale_color_manual(values=c("red","blue"))

# Regression trees
predict_tree <- predict(tree, newdata=data.frame(x=test_point))

tree2 <- as.party(tree)
nodes <- predict(tree2, df, type = "node")
node_test <- predict(tree2, newdata=data.frame(x=test_point), type = "node")

df$w_tree <- (nodes==node_test) / sum(nodes==node_test)

w_sign_tree = rep("zero", n)
w_sign_tree[df$w_tree>0] = "positive"
w_sign_tree = factor(w_sign_tree, level=c("zero", "positive"))

ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y,size=w_tree,color=w_sign_tree),alpha = 0.2) +
  geom_point(x=test_point,y=predict_tree,shape="cross",size = 3, stroke = 2,color="yellow") + 
  scale_color_manual(values=c("darkgray","blue"))

# Random Forest
predict_rf <- predict(rf, newdata=as.matrix(test_point))$predictions
df$w_rf <- t(as.matrix( get_forest_weights(rf,newdata=as.matrix(test_point))))

w_sign_rf <- rep("zero",n)
w_sign_rf[df$w_rf>0] <- "positive"
w_sign_rf <- factor(w_sign_rf,level=c("zero","positive"))

ggplot(df) + stat_function(fun=cef,size=1) + 
  geom_point(aes(x=x,y=y,size=w_rf,color=w_sign_rf),alpha = 0.2) +
  geom_point(x=test_point,y=predict_rf,shape="cross",size = 3, stroke = 2,color="yellow") + 
  scale_color_manual(values=c("darkgray","blue"))


# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_Tree_based.nb.html

# Application notebook: Tree-based methods
# Supervised ML: Tree-based methods

library(hdm)
library(tidyverse)
library(rpart)
library(partykit)
library(grf)
library(rpart.plot)

set.seed(1234) # for replicability

data(pension)
Y = pension$net_tfa / 1000
X = model.matrix(~0+age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, data = pension)


# Regression tree
tree <- rpart(Y ~ X)
rpart.plot(tree)

yhat_tree <- predict(tree)
plot(yhat_tree,Y)


# Random Forest regression
rf <- regression_forest(X,Y,tune.parameters = "all")
yhat_rf <- predict(rf)$predictions
plot(yhat_rf,Y)

vi <- variable_importance(rf)
rownames(vi) <- colnames(X)
round(vi,3)


# Compare performance

# Define training and testing sample split
test_fraction <- 1/3
test_size <- floor(test_fraction * length(Y))

# Run the method in the training sample and calculate the test set R2
tree_oos_r2 <- function(x_tr, y_tr, x_te, y_te){
  df_tr = data.frame(x_tr,y_tr)
  df_te = data.frame(x_te,y_te)
  tree = rpart(y_tr ~ ., df_tr)
  y_hat = predict(tree, newdata = df_te)
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}
rf_oos_r2 <- function(x_tr,y_tr,x_te,y_te){
  rf = regression_forest(x_tr,y_tr,tune.parameters = "all", honesty = FALSE)
  y_hat = predict(rf, newdata = x_te)$predictions
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}
rfh_oos_r2 <- function(x_tr,y_tr,x_te,y_te){
  rf = regression_forest(x_tr,y_tr,tune.parameters = "all", honesty = TRUE)
  y_hat = predict(rf, newdata = x_te)$predictions
  mse = mean((y_te - y_hat)^2)
  return(1 - mse/var(y_te))
}

rep <- 20

# Container of the results
results_r2 <- matrix(NA,rep,3)
colnames(results_r2) <- c("Tree","Forest","Honest Forest")

# Loop considering different splits
for (i in 1:rep) {
  # Draw index for this round
  temp_ind = sample(1:length(Y), size = test_size)
  
  # Split into training and test samples
  X_tr = X[-temp_ind,]
  X_te = X[temp_ind,]
  Y_tr = Y[-temp_ind]
  Y_te = Y[temp_ind]
  
  results_r2[i,1] = tree_oos_r2(X_tr,Y_tr,X_te,Y_te)
  results_r2[i,2] = rf_oos_r2(X_tr,Y_tr,X_te,Y_te)
  results_r2[i,3] = rfh_oos_r2(X_tr,Y_tr,X_te,Y_te)
}

round(colMeans(results_r2),3)

as.data.frame(results_r2) %>% pivot_longer(cols=everything(),names_to = "Method",values_to = "R2") %>%
  ggplot(aes(x = R2, y = Method)) + geom_boxplot()


# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Naive_model_selection.nb.html

# Simulation notebook: Why naive model selection fails
# Causal ML: Why naive model selection fails

library(tidyverse)
library(mvtnorm)
library(glmnet)

set.seed(1234) # For replicability

# Define the relevant parameters
n <- 100
var <- 1
cov <- 0.5
p <- 99

beta <- c(0,seq(1,0.1,-0.1),rep(0,p-10))
sig <- matrix(cov,p,p)
diag(sig) <- var

# A simulation study to figure out what works
repl <- 1000

beta1_ols <- beta1_lasso <- beta1_lasso_unpen <- rep(NA,repl)

for (i in 1:repl) {
  x <- cbind(rep(1,n),rmvnorm(n,sigma=sig))
  y <- x %*% beta + rnorm(n,0,2)
  
  # OLS
  beta1_ols[i] <- lm(y~x[,2:11])$coefficients[2]
  
  # Plain Lasso (specifying lambda.min.ratio speeds up)
  cv_lasso <- cv.glmnet(x[,-1],y,lambda.min.ratio=0.001) 
  lasso_temp <- glmnet(x[,-1],y,lambda=cv_lasso$lambda.min)
  beta1_lasso[i] <- lasso_temp$beta[1]
  
  # Lasso with unpenalized X_1
  cv_lasso <- cv.glmnet(x[,-1],y,penalty.factor=c(0,rep(1,p-1)),lambda.min.ratio=0.001)
  lasso_temp <- glmnet(x[,-1],y,lambda=cv_lasso$lambda.min,penalty.factor=c(0,rep(1,p-1)))
  beta1_lasso_unpen[i] <- lasso_temp$beta[1]
}


# OLS
df <- data.frame(x=beta1_ols)
ggplot(df,aes(x=x)) + geom_histogram(bins=30,aes(y =..density..)) + 
  geom_vline(xintercept = c(beta[2],mean(df$x)),linetype=c('solid','dashed'))

# Lasso
df <- data.frame(x=beta1_lasso)
ggplot(df,aes(x=x)) + geom_histogram(bins=30,aes(y =..density..)) + 
  geom_vline(xintercept = c(beta[2],mean(df$x)),linetype=c('solid','dashed'))

# Lasso with unpenalized parameter of interest
df <- data.frame(x=beta1_lasso_unpen)
ggplot(df,aes(x=x)) + geom_histogram(bins=30,aes(y =..density..)) + 
  geom_vline(xintercept = c(beta[2],mean(df$x)),linetype=c('solid','dashed'))

cv_lasso_unpen <- cv.glmnet(x[,-1],y,penalty.factor=c(0,rep(1,p-1)),lambda.min.ratio=0.002)
plot(cv_lasso_unpen)

lasso_unpen <- glmnet(x[,-1],y,penalty.factor=c(0,rep(1,p-1)),lambda.min.ratio=0.002)
plot(lasso_unpen,xvar = "lambda",label=T)





