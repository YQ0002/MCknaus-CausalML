# https://mcknaus.github.io/assets/notebooks/SNB/SNB_Convergence_rates.nb
rm(list = ls())

# Basics: Convergence rates

library(tidyverse)

set.seed(1234)
n <- 500
x <- runif(n)
b0 <- -1
b1 <- 2
cef <- function(x){b0 + b1*x}
y <- cef(x) + runif(n, -1, 1)

df <- data.frame(x, y)
ggplot(df) + stat_function(fun=cef, size=1) + 
  geom_point(aes(x=x, y=y), color="blue", alpha=0.4)
help("ggplot")

summary(lm(y~x))



