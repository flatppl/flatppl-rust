# Run in the oracle environment, never as a package runtime dependency.
# The Rust tests pin hand-derived design/default values. This script checks
# those values against brms and records versions before capturing Stan code.
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
stopifnot(Sys.setlocale("LC_COLLATE", "C") == "C")
library(brms)
d <- data.frame(y = c(2, 4, 8), x = c(1, 2, 3), g = c(20, 10, 20))
formula <- bf(y ~ x + (1 | g))
standata <- make_standata(formula, data = d, family = gaussian())
stopifnot(identical(unname(standata$X), cbind(rep(1, 3), c(1, 2, 3))))
stopifnot(identical(as.integer(standata$J_1), c(2L, 1L, 2L)))
groups <- transform(d, h = c(2, 2, 1))
reversed <- make_standata(y ~ x + (1 | h) + (1 | g), data = groups,
                         family = gaussian())
stopifnot(identical(as.integer(reversed$J_1), c(2L, 1L, 2L)))
stopifnot(identical(as.integer(reversed$J_2), c(2L, 2L, 1L)))
removed <- make_standata(y ~ x + (1 | g) - (1 || g), data = d,
                        family = gaussian())
stopifnot(identical(as.integer(removed$J_1), c(2L, 1L, 2L)))
duplicate <- try(make_standata(y ~ x + (1 | g) + (1 || g), data = d,
                             family = gaussian()), silent = TRUE)
stopifnot(inherits(duplicate, "try-error"))
defaults <- default_prior(formula, data = d, family = gaussian())
stopifnot(defaults$prior[defaults$class == "Intercept"] == "student_t(3, 4, 3)")
stopifnot(defaults$prior[defaults$class == "sigma"] == "student_t(3, 0, 3)")
priors <- c(prior(normal(0, 2), class = b),
            prior(normal(1, 3), class = Intercept),
            prior(normal(0, 1), class = sd),
            prior(normal(0, 1), class = sigma))
print(sessionInfo())
print(defaults)
cat(stancode(formula, data = d, family = gaussian(), prior = priors, normalize = TRUE))
