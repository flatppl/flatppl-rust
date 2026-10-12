# Independent oracle, run with brms 2.23.2 (c89c4c05ef52fefb4b78d5fb80df0cf6352f48b4).
# R and brms are test dependencies only.
library(brms)
d <- data.frame(y=c(2,4,8,7), x=1:4,
  f=factor(c("b","a","b","a"), levels=c("b","a")))
for (f in list(y~0+Intercept+x, y~0+Intercept+f,
               y~0+I(2*Intercept)+f, y~0+Intercept:x+f,
               y~0+x+Intercept, y~0+Intercept+(0+Intercept+x|f))) {
  print(f)
  print(make_standata(f, data=d))
  print(get_prior(f, data=d))
}
prior <- c(prior(normal(0,2),class="b"),
  prior(normal(1,3),class="b",coef="Intercept"), prior(exponential(1),class="sigma"))
fit <- brm(y~0+Intercept+x, data=d, prior=prior, empty=TRUE)
print(standata(fit, newdata=data.frame(x=c(5,8)), check_response=FALSE)$X)
cat(make_stancode(y~0+Intercept+x, data=d, prior=prior, normalize=TRUE))
