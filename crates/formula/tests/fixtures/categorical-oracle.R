# Reproduce coding, normalized Stan targets and independent softmax derivatives.
# Stan evaluations use propto=FALSE and jacobian=FALSE; all parameters are real.
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
library(brms)
Sys.setlocale("LC_COLLATE", "C")
options(scipen=0)
d <- data.frame(y=factor(c("red","blue","green","red","green","blue"),
                         levels=c("unused","red","blue","green")),
                x=1:6, z=c(0,1,0,2,1,0), w=c(1,.5,2,0,1.25,1))
formulas <- list(default=bf(y~x),
  weighted=bf(y|weights(w)~x,mublue~0+z,mugreen~x+offset(z)))
priors <- list(default=NULL,weighted=c(
  prior(normal(0,1.1),class=b,dpar=mublue),
  prior(normal(0,.7),class=b,dpar=mugreen),
  prior(normal(.2,1.3),class=Intercept,dpar=mugreen)))
points <- list(default=list(c(.4,-.3,-.2,.6),c(-.7,.2,.9,-.4)),
               weighted=list(c(.4,-.2,.6),c(-.7,.9,-.4)))
expected <- list(default=c(-11.410139213373222,-12.486163278702211),
                 weighted=c(-9.550614306617366,-9.602678114990223))
for (name in names(formulas)) {
  prepared <- make_standata(formulas[[name]],d,family=categorical(),prior=priors[[name]])
  stopifnot(identical(as.integer(prepared$Y),c(1L,2L,3L,1L,3L,2L)))
  writeLines(make_stancode(formulas[[name]],d,family=categorical(),
                          prior=priors[[name]],normalize=TRUE),paste0(name,".stan"))
  dput(unclass(prepared),paste0(name,".data.R"))
  for (i in seq_along(points[[name]])) {
    p <- points[[name]][[i]]
    x <- d$x-mean(d$x)
    if (name == "default") {
      eta <- cbind(0,p[1]+p[2]*x,p[3]+p[4]*x)
      lp <- sum(dt(p[c(1,3)]/2.5,3,log=TRUE)-log(2.5))
      grad <- c(-4*p[1]/(18.75+p[1]^2),0,-4*p[3]/(18.75+p[3]^2),0)
      weights <- rep(1,nrow(d))
    } else {
      eta <- cbind(0,p[1]*d$z,p[2]+p[3]*x+d$z)
      lp <- dnorm(p[1],0,1.1,log=TRUE)+dnorm(p[2],.2,1.3,log=TRUE)+dnorm(p[3],0,.7,log=TRUE)
      grad <- c(-p[1]/1.1^2,-(p[2]-.2)/1.3^2,-p[3]/.7^2)
      weights <- d$w
    }
    prob <- exp(eta-apply(eta,1,max))
    prob <- prob/rowSums(prob)
    rows <- log(prob[cbind(seq_len(nrow(d)),prepared$Y)])
    db <- weights*((prepared$Y==2)-prob[,2])
    dg <- weights*((prepared$Y==3)-prob[,3])
    grad <- grad + if (name == "default") c(sum(db),sum(db*x),sum(dg),sum(dg*x)) else
      c(sum(db*d$z),sum(dg),sum(dg*x))
    score <- lp+sum(weights*rows)
    stopifnot(abs(score-expected[[name]][i])<1e-12)
    print(list(model=name,point=p,score=score,gradient=grad,pointwise=rows))
  }
}
fit <- brm(y~x,d,family=categorical(),empty=TRUE)
newdata <- data.frame(y=factor(c("green","blue","red"),
                               levels=c("green","blue","red","unused")),x=c(-1,2,8))
stopifnot(identical(as.integer(standata(fit,newdata=newdata)$Y),c(3L,2L,1L)))
labels <- c(1e5,1.234567890123456,6.5081744082272057e-15,3161646742373.705)
stopifnot(identical(as.character(labels),
  c("1e+05","1.23456789012346","6.50817440822721e-15","3161646742373.7")))
stopifnot(identical(brms:::make_stan_names(c("mua b","mua.b")),c("muab1","muab")))
grouped <- y~x+(1+x|z)
global <- prior(lkj(2),class=cor)
specific <- prior(lkj(4),class=cor,group=z)
code <- make_stancode(grouped,d,family=categorical(),prior=c(global,specific),normalize=TRUE)
stopifnot(grepl("lkj_corr_cholesky_lpdf(L_1 | 4)",code,fixed=TRUE),
          grepl("lkj_corr_cholesky_lpdf(L_2 | 4)",code,fixed=TRUE))
for (p in list(prior(normal(0,2),class=b),prior(normal(0,2),class=sd),
               prior(lkj(3),class=cor,dpar=mublue))) {
  stopifnot(inherits(try(make_stancode(grouped,d,family=categorical(),prior=p),silent=TRUE),"try-error"))
}
# Relabeling keeps the target but deliberately reverses lexical predictor order.
levels(d$y) <- c("unused","red","zeta","alpha")
writeLines(make_stancode(y~x,d,family=categorical(),normalize=TRUE),"reordered.stan")
dput(unclass(make_standata(y~x,d,family=categorical())),"reordered.data.R")
