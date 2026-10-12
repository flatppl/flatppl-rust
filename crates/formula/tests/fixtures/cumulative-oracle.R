# Reproduce the brms source and independent values in cumulative-oracle.json.
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
library(brms)
data <- data.frame(y = ordered(c("medium", "low", "high", "medium"),
                              levels = c("low", "medium", "high")),
                   x = c(1, 2, 4, 5))
family <- cumulative("logit")
priors <- c(prior(normal(0, 2), class = Intercept),
            prior(normal(0, 1.5), class = b))
prepared <- make_standata(y ~ x, data = data, family = family, prior = priors)
stopifnot(identical(as.integer(prepared$Y), c(2L, 1L, 3L, 2L)))
stopifnot(prepared$nthres == 2, mean(prepared$X) == 3)
gaps <- transform(data, y = c(1, 3, 1, 3))
gaps <- make_standata(y ~ x, data = gaps, family = family)
stopifnot(gaps$nthres == 2, identical(as.integer(gaps$Y), c(1L, 3L, 1L, 3L)))
unused <- transform(data, y = ordered(c("high", "low", "high", "low"),
                                     levels = c("low", "medium", "high", "last")))
unused <- make_standata(y ~ x, data = unused, family = family)
stopifnot(unused$nthres == 1, identical(as.integer(unused$Y), c(2L, 1L, 2L, 1L)))
cuts <- c(-0.6, 0.8)
score <- function(b, defaults = FALSE) {
  eta <- (data$x - mean(data$x)) * b
  lo <- cuts[1] - eta
  hi <- cuts[2] - eta
  middle <- plogis(hi, log.p = TRUE) + plogis(-lo, log.p = TRUE) + log(-expm1(lo - hi))
  likelihood <- sum(c(middle[1], plogis(lo[2], log.p = TRUE),
                      plogis(-hi[3], log.p = TRUE), middle[4]))
  prior <- if (defaults) sum(dt(cuts / 2.5, 3, log = TRUE) - log(2.5)) else
    sum(dnorm(cuts, 0, 2, log = TRUE)) + dnorm(b, 0, 1.5, log = TRUE)
  likelihood + prior
}
stopifnot(abs(score(0.4) - (-8.882198854773355)) < 1e-12)
stopifnot(abs(score(10) - (-66.06229096425213)) < 1e-12)
stopifnot(abs(score(0.4, TRUE) - (-8.11259083871886)) < 1e-12)
print(default_prior(y ~ x, data = data, family = family))
print(sessionInfo())
cat(stancode(y ~ x, data = data, family = family, prior = priors, normalize = TRUE))
cat(stancode(y ~ x, data = data, family = family, normalize = TRUE))
