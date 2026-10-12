# Run with brms 2.23.2. Evaluate each generated Stan model with
# jacobian=FALSE, propto=FALSE at b=.7, Intercept=1.2 and sigma=.5 when present.
# The JSON fixture retains raw gradients and their constrained-coordinate map.
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
library(brms)
data <- data.frame(y=c(1, 2, 4), x=c(0, 1, 3), s=c(.3, .6, .9))
for (family in c("gaussian", "student")) {
  for (residual in c(FALSE, TRUE)) {
    name <- paste(family, if (residual) "residual" else "default", sep="-")
    formula <- bf(as.formula(paste0("y | se(s, sigma=", residual, ") ~ x")))
    priors <- c(set_prior("normal(0, 2)", class="Intercept"),
                set_prior("normal(0, 1)", class="b"))
    if (residual) priors <- c(priors, set_prior("normal(0, 1)", class="sigma"))
    if (family == "student") priors <- c(priors, set_prior("constant(5)", class="nu"))
    law <- if (family == "student") student() else gaussian()
    writeLines(make_stancode(formula, data=data, family=law, prior=priors,
                            normalize=TRUE), paste0(name, ".stan"))
    dput(unclass(make_standata(formula, data=data, family=law, prior=priors)),
         paste0(name, ".data.R"))
  }
}
