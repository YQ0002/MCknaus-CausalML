# https://nbviewer.org/github/MCKnaus/causalML-teaching/blob/main/Slides/CML3_CI.pdf
# Causal Machine Learning

# Basics

# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Conditional_Independence.nb.html
# (Conditional) Independence

# Conditional independence implies conditional mean independence

# Formal definition of (conditional) independence
# General Example


library(tidyverse)
library(ggridges)
library(MASS)

set.seed(1234)
n <- 1000000
mu <- c(0,0)
rho <- 0
sigma <- matrix(c(1,rho,rho,1), nrow = 2)
draw <- mvrnorm(n, mu, sigma)
X <- 1*(draw[,1] > 0)
W <- 1*(draw[,2] > 0)
Y <- X + rnorm(n, sd = 0.5)

tibble(Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha = 0.6)+
  xlab("Y") + ylab("Density")

mean(Y)
mean(Y[W==1])
mean(Y[W==0])


# Introduce some correlation between the variables V and  Z
set.seed(1234)
n <- 1000000
mu <- c(0,0)
rho <- 0.6
sigma <- matrix(c(1,rho,rho,1), nrow = 2)
draw <- mvrnorm(n, mu, sigma)
X <- 1*(draw[,1] > 0) 
W <- 1*(draw[,2] > 0)
Y <- X + rnorm(n, sd = 0.5)

tibble(Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha = 0.6)+
  xlab("Y")

tibble(Y,W,X) %>%  mutate(W = as.factor(W), X = as.factor(X), Subgroup = interaction(W,X)) %>%
  ggplot(aes(x = Y, y = fct_rev(Subgroup), fill = Subgroup))+
  geom_density_ridges(alpha = 0.6)+
  scale_fill_discrete(labels = c("W=0, X=0","W=1, X=0","W=0, X=1","W=1, X=1"))+
  xlab("Y") + ylab("Density")

mean(Y)
mean(Y[W==1])
mean(Y[W==0])

tibble(Y,W,X) %>% group_by(X,W) %>% summarise(mean_Y = mean(Y))
tibble(Y,W,X) %>% group_by(X) %>% summarise(mean_Y = mean(Y))


# Example: Potential Outcomes and causal inference
set.seed(1234)
n <- 1000000
mu <- c(0,0)
rho <- 0.6
sigma <- matrix(c(1,rho,rho,1), nrow = 2)
draw <- mvrnorm(n, mu, sigma)
X <- 1*(draw[,1] > 0)
W <- 1*(draw[,2] > 0)
Y0 <- X + rnorm(n, sd = 0.5)
Y1 <- 0.5 + X + rnorm(n, sd = 0.5)
Y <- W*Y1 + (1-W)*Y0

# x = Y1
tibble(Y1,Y0,Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y1, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha = 0.6)+
  xlab("Y(1)")+ ylab("Density")
# x = Y0
tibble(Y1,Y0,Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y0, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha=.6)+
  xlab("Y(0)")+ ylab("Density")

TE_mean_comparison <- mean(Y[W==1]) - mean(Y[W==0])
print(TE_mean_comparison)
# 0.9093149


# Introduce correlation between V and Z
rho <- 0.6
sigma <- matrix(c(1,rho,rho,1), nrow = 2)
draw <- mvrnorm(n, mu, sigma)
X <- 1*(draw[,1] > 0)
W <- 1*(draw[,2] > 0)
Y0 <- X + rnorm(n, sd = 0.5)
Y1 <- 0.5 + X + rnorm(n, sd = 0.5)
Y <- W*Y1 + (1-W)*Y0

# x = Y1
tibble(Y1,Y0,Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y1, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha=.6)+
  xlab("Y(1)")+ylab("Density")
# x = Y0
tibble(Y1,Y0,Y,W,X) %>% mutate(W = as.factor(W)) %>% 
  ggplot(aes(x = Y0, y = fct_rev(W), fill = W))+
  geom_density_ridges(alpha=.6)+
  xlab("Y(0)")+ylab("Density")

TE_mean_comparison <- mean(Y[W==1]) - mean(Y[W==0])
print(TE_mean_comparison)
# 0.9111342


# x = Y1 
tibble(Y,Y1,Y0,W,X) %>%  mutate(W = as.factor(W), X = as.factor(X), Subgroup = interaction(W,X)) %>% 
  ggplot(aes(x = Y1, y = fct_rev(Subgroup), fill = Subgroup))+
  geom_density_ridges(alpha = 0.6)+
  scale_fill_discrete(labels = c("W=0, X=0","W=1, X=0","W=0, X=1","W=1, X=1"))+
  xlab("Y(1)")+ylab("Density")

# x = Y0 
tibble(Y,Y1,Y0,W,X) %>%  mutate(W = as.factor(W), X = as.factor(X), Subgroup = interaction(W,X)) %>% 
  ggplot(aes(x = Y0, y = fct_rev(Subgroup), fill = Subgroup))+
  geom_density_ridges(alpha = 0.7)+
  scale_fill_discrete(labels = c("W=0, X=0","W=1, X=0","W=0, X=1","W=1, X=1"))+
  xlab("Y(0)")+ylab("Density")

# Conditional on X, the potential outcomes are independent of W

TE_cond_mean_comparison_1 <- mean(Y[W==1 & X==1]) - mean(Y[W==0 & X==1])
print(TE_cond_mean_comparison_1)

TE_cond_mean_comparison_0 <- mean(Y[W==1 & X==0]) - mean(Y[W==0 & X==0])
print(TE_cond_mean_comparison_0)

TE_cond_mean_comparison <- mean(X)*TE_cond_mean_comparison_1 + mean(1-X)*TE_cond_mean_comparison_0
print(TE_cond_mean_comparison)


summary(lm(Y~ W + X))



