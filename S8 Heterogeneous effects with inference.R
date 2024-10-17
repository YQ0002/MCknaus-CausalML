# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML8_HTE2.pdf
# Heterogeneous effects with inference

# https://mcknaus.github.io/assets/notebooks/SNB/SNB_GATE.nb.html
# Simulation notebook: Causal ML: Group Average Treatment Effects

rm(list = ls())

# Linear heterogeneity
library(grf)
library(causalDML)
library(tidyverse)
library(patchwork)
library(estimatr)
library(np)
library(crs)

set.seed(1234)

# Set parameters
n <- 1000
p <- 10

rho <- c(0.3,0.2,0.1,rep(0,7))

# Draw sample
x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){pnorm(sin(x))}
m0 <- function(x){sin(x)}
tau <- x %*% rho
w <- rbinom(n,1,e(x[,1]))
y <- m0(x[,1]) + w*tau + rnorm(n,0,1)
hist(tau)

# GATE estimation with OLS
# Hand-coded 2-fold cross-fitting

# 2-fold corss-fitting
m0hat = m1hat = ehat = rep(NA, n)
# Draw random indices for sample 1
index_s1 <- sample(1:n, n/2)
# Create S1
x1 <- x[index_s1,]
w1 <- w[index_s1]
y1 <- y[index_s1]
# Create S2
x2 <- x[-index_s1,]
w2 <- w[-index_s1]
y2 <- y[-index_s1]
# Model in S1, predict in S2
rf <- regression_forest(x1, w1, tune.parameters = "all")
ehat[-index_s1] <- predict(rf, newdata = x2)$predictions
rf <- regression_forest(x1[w1==0,], y1[w1==0], tune.parameters = "all")
m0hat[-index_s1] <- predict(rf, newdata = x2)$predictions
rf <- regression_forest(x1[w1==1,], y1[w1==1], tune.parameters = "all")
m1hat[-index_s1] <- predict(rf, newdata = x2)$predictions
# Model in S2, predict in S1
rf <- regression_forest(x2, w2, tune.parameters = "all")
ehat[index_s1] <- predict(rf, newdata = x1)$predictions
rf <- regression_forest(x2[w2==0,], y2[w2==0], tune.parameters = "all")
m0hat[index_s1] <- predict(rf, newdata = x1)$predictions
rf <- regression_forest(x2[w2==1,], y2[w2==1], tune.parameters = "all")
m1hat[index_s1] <- predict(rf, newdata = x1)$predictions
# Generate pseudo-outcome
pseudo_y <- m1hat + m0hat + w*(y-m1hat)/ehat - (1-w)*(y-m0hat)/(1-ehat)

lm_fit2 <- lm_robust(pseudo_y~x[,1:5])
summary(lm_fit2)


se2 <- lm_fit2$std.error
data.frame(Variable = c("Constant",paste0("X",1:5)),
           Coefficient = lm_fit2$coefficients, 
           cil = lm_fit2$coefficients - 1.96*se2,
           ciu = lm_fit2$coefficients + 1.96*se2,
           truth = c(0,rho[1:5])) %>%
  ggplot(aes(x=Variable, y=Coefficient, ymin=cil, ymax=ciu)) + 
  geom_point(linewidth=2.5, aes(colour="Estimate", shape="Estimate")) + 
  geom_hline(yintercept = 0) + 
  geom_errorbar(width=0.15)  +
  geom_point(aes(x=Variable, y=truth, colour="Truth", shape="Truth"),linewidth=2.5) +
  scale_color_manual(name="Legend", values = c("black", "blue")) + 
  scale_shape_manual(name="Legend", values = c(19,8))

plot(tau, lm_fit2$fitted.values)

# 5-fold cross-fitting with causalDML package
# Create learner
forest <- create_method("forest_grf", args = list(tune.parameters="all"))
# Run
aipw <- DML_aipw(y,w,x,ml_w=list(forest), ml_y=list(forest), cf=5)
summary(aipw)

summary(aipw$APO)
summary(aipw$ATE)

lm_fit5 <- lm_robust(aipw$ATE$delta~x[,1:5])
summary(lm_fit5)

se5 <- lm_fit5$std.error
data.frame(Variable = c("Constant",paste0("X",1:5)),
           Coefficient = lm_fit5$coefficients,
           cil = lm_fit5$coefficients - 1.96*se5,
           ciu = lm_fit5$coefficients + 1.96*se5,
           truth = c(0,rho[1:5])) %>% 
  ggplot(aes(x=Variable,y=Coefficient,ymin=cil,ymax=ciu)) + 
  geom_point(linewidth=2.5,aes(colour="Estimate",shape="Estimate")) + 
  geom_errorbar(width=0.15)  +
  geom_hline(yintercept=0) + 
  geom_point(aes(x=Variable,y=truth,colour="Truth",shape="Truth"),linewidth=2.5) +
  scale_colour_manual(name="Legend", values = c("black","blue")) + 
  scale_shape_manual(name="Legend",values = c(19,8))

plot(tau,lm_fit5$fitted.values)



# Non-parametric heterogeneity

x <- matrix(runif(n*p,-pi,pi),ncol=p)
e <- function(x){pnorm(sin(x))}
m1 <- function(x){sin(x)}
m0 <- function(x){cos(x+1/2*pi)}
tau <- function(x){m1(x) - m0(x)}
w <- rbinom(n,1,e(x[,1]))
y <- w*m1(x[,1]) + (1-w)*m0(x[,1]) + rnorm(n,0,1)

g1 <- data.frame(x=c(-pi,pi)) %>% 
  ggplot(aes(x)) + 
  stat_function(fun=e, linewidth=1,) + 
  ylab("e") + xlab("X1")
g2 <- data.frame(x=c(-pi,pi)) %>% 
  ggplot(aes(x)) + 
  stat_function(fun=m0, linewidth=1, aes(colour="Y1")) + 
  stat_function(fun=m1, linewidth=1, aes(colour="Y0")) + 
  ylab("Y") + xlab("X1")
g3 <- data.frame(x=c(-pi,pi)) %>% 
  ggplot(aes(x)) + 
  stat_function(fun=tau, linewidth=1,) + 
  ylab(expression(tau)) + xlab("X1")
g1 / g2 / g3


# GATE estimation with kernel regression
# Hand-coded 2-fold cross-fitting

m0hat = m1hat = ehat = rep(NA,n)
# Draw random indices for sample 1
set.seed(1234)
index_s <- sample(1:n, n/2)

# Create S1
x1 <- x[index_s1,]
w1 <- w[index_s1]
y1 <- y[index_s1]
# Create S2
x2 <- x[-index_s1,]
w2 <- w[-index_s1]
y2 <- y[-index_s1]

# Model in S1, predict in S2
rf <- regression_forest(x1, w1, tune.parameters = "all")
ehat[-index_s1] <- predict(rf, newdata=x2)$predictions
rf <- regression_forest(x1[w1==0,], y1[w1==0], tune.parameters = "all")
m0hat[-index_s1] <- predict(rf, newdata=x2)$predictions
rf <- regression_forest(x1[w1==1,], y1[w1==1], tune.parameters = "all")
m1hat[-index_s1] <- predict(rf, newdata=x2)$predictions
# Model in S2, predict in S1
rf <- regression_forest(x2, w2, tune.parameters = "all")
ehat[index_s1] <- predict(rf, newdata=x1)$predictions
rf <- regression_forest(x2[w2==0,], y2[w2==0], tune.parameters = "all")
m0hat[index_s1] <- predict(rf, newdata=x1)$predictions
rf <- regression_forest(x2[w2==1,], y2[w2==1], tune.parameters = "all")
m1hat[index_s1] <- predict(rf, newdata=x1)$predictions

# generate pseudo-outcome
pseudo_y <- m1hat - m0hat + w*(y-m1hat)/ehat - (1-w)/(y-m0hat) / (1-ehat)
summary(pseudo_y)

z <- as.data.frame(x[,1])
# Crossvalidate bandwidth
bwobj <- npregbw(ydat=pseudo_y, xdat=z, 
                 ckertype="gaussian", ckerorder=2, regtype="lc", bwmethod="cv.ls")
bws <- bwobj$bw
help("npregbw")
# Undersmoothing, i.e. chose a slightly smaller bandwidth than was cross-validated
bw <- bwobj$bw * 0.9
cate_model <- npreg(tydat=pseudo_y, txdat=z, bws=bw, 
                    ckertype="gaussian", ckerorder=2, regtype="lc")
plot(cate_model)

# spline
library(gam)
data <- as.data.frame(cbind(pseudo_y, x[,1]))
head(data)
spline <- gam(pseudo_y ~ s(V2, 4), data = data)
plot(spline, se = T, col = "blue")


library(splines)
plot(data$V2, pseudo_y, ylim = c(-8, 8), cex = .5, col = "darkgrey")
title("Smoothing Spline")
fit <- smooth.spline(data$V2, pseudo_y, cv = TRUE)
lines(fit, col = "blue", lwd = 2)





# 5-fold cross-fitting with causalDML package
# Create learner
forest <- create_method("forest_grf", args=list(tune.parameters="all"))
# Run
aipw <- DML_aipw(y,w,x,ml_w=list(forest),ml_y=list(forest),cf=5)
summary(aipw$APO)
summary(aipw$ATE)

kernel_reg_x1 <- kr_cate(aipw$ATE$delta,x[, 1])
plot(kernel_reg_x1)

kernel_reg_x2 <- kr_cate(aipw$ATE$delta,x[, 2])
plot(kernel_reg_x2)


# GATE estimation with series regression
# Hand-coded 2-fold cross-fitting

spline_gate <- crs(pseudo_y ~ as.matrix(z))
plot(spline_gate, mean=T)


# 5-fold cross-fitting with causalDML package
spline_reg_x1 <- spline_cate(aipw$ATE$delta,x[, 1])
plot(spline_reg_x1)

spline_reg_x2 <- spline_cate(aipw$ATE$delta,x[, 2])
plot(spline_reg_x2)



# https://mcknaus.github.io/assets/notebooks/appl401k/ANB_401k_GATE.nb.html
# Application Notebook: Causal ML: Double ML for group average treatment effects

# Load the packages required for later
library(hdm)
library(tidyverse)
library(causalDML)
library(grf)
library(estimatr)


set.seed(1234)
options(scipen = 10) # Switch off scientific notation

data(pension)
# Outcome
Y <- pension$net_tfa
# Treatment
W <- pension$p401
# Create main effects matrix
X <- model.matrix(~ 0+age+db+educ+fsize+hown+inc+male+marr+pira+twoearn, data = pension)

# Double ML for AIPW with causalDML package
# 5-fold cross-fitting with causalDML package
aipw <- DML_aipw(Y,W,X)
summary(aipw$ATE)

# # Tune the forest
# forest <- create_method("forest_grf",args=list(tune.parameters = "all"))
# aipw <- DML_aipw(Y,W,X,ml_w=list(forest),ml_y=list(forest),cf=5)


# GATE estimation

# Subgroup effect
male <- X[,7]
blp_male <- lm_robust(aipw$ATE$delta ~ male)
summary(blp_male)

female <- 1-male
blp_male1 <- lm_robust(aipw$ATE$delta ~ 0 + female + male)
summary(blp_male1)


# Best linear prediction
blp <- lm_robust(aipw$ATE$delta ~ X)
summary(blp)


# Non-parametric heterogeneity
# Spline regression

age <- X[,1]
sr_age <- spline_cate(aipw$ATE$delta,age)
plot(sr_age,z_label = "Age")

# Kernel regression

kr_age <- kr_cate(aipw$ATE$delta,age)
plot(kr_age,z_label = "Age")


head(X)
inc <- X[, 6]
sr_inc <- spline_cate(aipw$ATE$delta,inc)
plot(sr_inc,z_label = "Inc")

try <- X[, 2]
sr_try <- spline_cate(aipw$ATE$delta,try)
plot(sr_try, z_label = "Try")

