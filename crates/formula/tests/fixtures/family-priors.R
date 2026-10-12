# Independent brms default-prior oracle; run with brms 2.23.2.
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
library(brms)
cases <- list(
  list(lognormal("inverse"), exp(c(-4, -1, 2, 3))),
  list(gaussian("log"), c(-2, 4, 8, 7)),
  list(student("inverse"), c(0, 4, 8, 7)),
  list(poisson("sqrt"), c(0, 4, 8, 7)),
  list(bernoulli("identity"), c(-2, 7, -2, 7)),
  list(bernoulli("identity"), c(FALSE, TRUE, TRUE, FALSE))
)
for (case in cases) {
  data <- data.frame(y = case[[2]], x = 1:4, g = c("A", "B", "A", "B"))
  print(default_prior(y ~ x + (1 | g), data = data, family = case[[1]]))
}
