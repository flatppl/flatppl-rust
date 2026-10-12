# brms 2.23.2 at c89c4c05ef52fefb4b78d5fb80df0cf6352f48b4; R 4.5.3.
# Recreates the design/prior fixtures, normalized Stan sources and independent scores.
suppressPackageStartupMessages(library(brms))
stopifnot(as.character(packageVersion("brms")) == "2.23.2")
Sys.setlocale("LC_COLLATE", "C")
out <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out)) stop("Usage: Rscript distributional-oracle.R OUTPUT_DIR")
dir.create(file.path(out, "models"), recursive = TRUE, showWarnings = FALSE)
encode_json <- function(x) {
  if (is.null(x)) return("null")
  if (is.factor(x)) x <- as.character(x)
  if (is.matrix(x)) x <- lapply(seq_len(nrow(x)), function(i) unname(as.list(x[i, ])))
  if (is.list(x)) {
    if (!length(x)) return(if (is.null(names(x))) "[]" else "{}")
    values <- vapply(x, encode_json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse = ","), "]"))
    keys <- vapply(names(x), encodeString, "", quote = '"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse = ","), "}"))
  }
  if (length(x) != 1L) return(encode_json(unname(as.list(x))))
  if (is.na(x)) return("null")
  if (is.character(x)) return(encodeString(x, quote = '"'))
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(if (is.finite(x)) trimws(formatC(x, digits = 17, format = "g", decimal.mark = ".")) else "null")
  stop("unsupported JSON value")
}
json <- function(x, file) writeLines(encode_json(x), file.path(out, file))
d <- data.frame(y=c(2,4,8,7),x=1:4,g=c(0,1,0,2))
families <- list(gaussian=brmsfamily("gaussian"),student=student(),lognormal=lognormal(),negbinomial=negbinomial(),Gamma=brmsfamily("Gamma"),weibull=weibull(),beta=brmsfamily("beta"),von_mises=von_mises(),zero_inflated_poisson=zero_inflated_poisson())
aux <- list(gaussian="sigma",student=c("sigma","nu"),lognormal="sigma",negbinomial="shape",Gamma="shape",weibull="shape",beta="phi",von_mises="kappa",zero_inflated_poisson="zi")
defaults <- list()
for (family in names(families)) for (dp in aux[[family]]) {
  f <- bf(y~x,as.formula(paste(dp,"~ g")))
  data <- d
  if (family=="beta") data$y <- data$y/10
  if (family=="von_mises") data$y <- c(-1,-.3,.2,1.5)
  if (family %in% c("negbinomial","zero_inflated_poisson")) data$y <- c(0L,2L,4L,0L)
  p <- as.data.frame(get_prior(f,data,family=families[[family]]))
  defaults[[paste(family,dp,sep="_")]] <- list(family=family,dpar=dp,y=as.list(data$y),prior=p$prior[p$class=="Intercept" & p$dpar==dp],link=families[[family]][[paste0("link_",dp)]])
}
json(defaults,"defaults.json")
points <- list(c(.4,.3,.1,-.2),c(1.2,-.4,-.3,.35))
oracle <- list()
for (custom in c(FALSE,TRUE)) {
  name <- if(custom) "gaussian_custom" else "gaussian_default"
  f <- bf(y~x,sigma~g)
  p <- if(custom) c(prior(normal(0,2),class=b),prior(normal(.2,.8),class=Intercept,dpar=sigma),prior(normal(0,.7),class=b,dpar=sigma)) else NULL
  frame <- make_standata(f,d,family=gaussian(),prior=p)
  writeLines(make_stancode(f,d,family=gaussian(),prior=p,normalize=TRUE),file.path(out,"models",paste0(name,".stan")))
  json(frame,paste0("models/",name,".data.json"))
  cases <- lapply(points,function(point) {
    a <- point[1]; b <- point[2]; c <- point[3]; e <- point[4]
    x <- d$x-mean(d$x); g <- d$g-mean(d$g)
    mu <- a+b*x; sigma <- exp(c+e*g); delta <- d$y-mu
    rows <- dnorm(d$y,mu,sigma,log=TRUE)
    dm <- delta/sigma^2; ds <- delta^2/sigma^2-1
    lp <- dt((a-5.5)/3,3,log=TRUE)-log(3)
    grad <- c(sum(dm)-4*(a-5.5)/(27+(a-5.5)^2),sum(dm*x),sum(ds),sum(ds*g))
    if(custom) {
      lp <- lp+dnorm(b,0,2,log=TRUE)+dnorm(c,.2,.8,log=TRUE)+dnorm(e,0,.7,log=TRUE)
      grad <- grad+c(0,-b/4,-(c-.2)/.8^2,-e/.7^2)
    } else {
      lp <- lp+dt(c/2.5,3,log=TRUE)-log(2.5)
      grad[3] <- grad[3]-4*c/(3*2.5^2+c^2)
    }
    list(point=as.list(point),posterior=lp+sum(rows),gradient=as.list(grad),mu=as.list(mu),sigma=as.list(sigma),pointwise=as.list(rows),reported_intercepts=as.list(c(a-mean(d$x)*b,c-mean(d$g)*e)))
  })
  oracle[[name]] <- list(cases=cases,prior=as.data.frame(get_prior(f,d,family=gaussian())),X=frame$X,X_sigma=frame$X_sigma)
}
json(oracle,"gaussian.json")
prediction <- data.frame(x = c(2,5), g = c(-1,3), y = c(1,2))
prediction_cases <- lapply(points, function(point) {
  x <- prediction$x - mean(d$x)
  g <- prediction$g - mean(d$g)
  mu <- point[1] + point[2] * x
  sigma <- exp(point[3] + point[4] * g)
  delta <- prediction$y - mu
  dm <- delta / sigma^2
  ds <- delta^2 / sigma^2 - 1
  list(point = as.list(point), value = sum(dnorm(prediction$y, mu, sigma, log = TRUE)),
       gradient = as.list(c(sum(dm), sum(dm*x), sum(ds), sum(ds*g))),
       mu = as.list(mu), sigma = as.list(sigma))
})
json(list(data = prediction, cases = prediction_cases), "prediction.json")
f <- bf(y~x,sigma~g,nu~x)
frame <- make_standata(f,d,family=student())
writeLines(make_stancode(f,d,family=student(),normalize=TRUE),file.path(out,"models","student.stan"))
json(frame,"models/student.data.json")
cases <- lapply(list(c(.4,.3,.1,-.2,1.5,.25),c(1.2,-.4,-.3,.35,2.3,-.4)),function(point) {
  a<-point[1]; b<-point[2]; c<-point[3]; e<-point[4]; f<-point[5]; h<-point[6]
  x<-d$x-mean(d$x); g<-d$g-mean(d$g)
  mu<-a+b*x; sigma<-exp(c+e*g); nu<-1+exp(f+h*x)
  delta<-d$y-mu; q<-(delta/sigma)^2
  rows<-dt(delta/sigma,nu,log=TRUE)-log(sigma)
  dm<-(nu+1)*delta/(nu*sigma^2+delta^2)
  ds<-nu*(delta^2-sigma^2)/(nu*sigma^2+delta^2)
  dn<-(digamma((nu+1)/2)/2-digamma(nu/2)/2-1/(2*nu)-log1p(q/nu)/2+(nu+1)*q/(2*nu*(nu+q)))*(nu-1)
  lp<-dt((a-5.5)/3,3,log=TRUE)-log(3)+dt(c/2.5,3,log=TRUE)-log(2.5)+dnorm(f,2.7,.8,log=TRUE)
  gradient<-c(sum(dm)-4*(a-5.5)/(27+(a-5.5)^2),sum(dm*x),sum(ds)-4*c/(3*2.5^2+c^2),sum(ds*g),sum(dn)-(f-2.7)/.8^2,sum(dn*x))
  list(point=as.list(point),posterior=lp+sum(rows),gradient=as.list(gradient),mu=as.list(mu),sigma=as.list(sigma),nu=as.list(nu),pointwise=as.list(rows))
})
json(list(cases=cases,X=frame$X,X_sigma=frame$X_sigma,X_nu=frame$X_nu),"student.json")
dg<-d; dg$h<-factor(c("B","A","B","A"),levels=c("B","A"))
fg<-bf(y~x+(1|h),sigma~g+(1+g||h))
pg<-c(prior(normal(0,2),class=sd),prior(normal(0,1),class=sd,group=h,dpar=sigma),prior(normal(0,.5),class=sd,coef=g,group=h,dpar=sigma))
frame<-make_standata(fg,dg,family=gaussian(),prior=pg)
writeLines(make_stancode(fg,dg,family=gaussian(),prior=pg,normalize=TRUE),file.path(out,"models","gaussian_groups.stan"))
json(frame,"models/gaussian_groups.data.json")
point <- c(.4,.3,.1,-.2,.7,.2,-.1,.6,.4,.3,-.2,.1,.25)
a<-point[1]; b<-point[2]; c<-point[3]; e<-point[4]
sd_mu<-point[5]; z_mu<-point[6:7]; sd_i<-point[8]; sd_g<-point[9]
z_i<-point[10:11]; z_g<-point[12:13]; j<-as.integer(dg$h)
x<-d$x-mean(d$x); g<-d$g-mean(d$g)
mu<-a+b*x+sd_mu*z_mu[j]
sigma<-exp(c+e*g+sd_i*z_i[j]+sd_g*z_g[j]*d$g)
delta<-d$y-mu; dm<-delta/sigma^2; ds<-delta^2/sigma^2-1
rows<-dnorm(d$y,mu,sigma,log=TRUE)
lp<-dt((a-5.5)/3,3,log=TRUE)-log(3)+dt(c/2.5,3,log=TRUE)-log(2.5)+
    dnorm(sd_mu,0,2,log=TRUE)+dnorm(sd_i,0,1,log=TRUE)+dnorm(sd_g,0,.5,log=TRUE)+
    3*log(2)+sum(dnorm(c(z_mu,z_i,z_g),log=TRUE))+sum(rows)
sum_group<-function(v) vapply(1:2,function(i) sum(v[j==i]),0.0)
gradient<-c(sum(dm)-4*(a-5.5)/(27+(a-5.5)^2),sum(dm*x),
    sum(ds)-4*c/(3*2.5^2+c^2),sum(ds*g),
    sum(dm*z_mu[j])-sd_mu/4,sd_mu*sum_group(dm)-z_mu,
    sum(ds*z_i[j])-sd_i,sum(ds*z_g[j]*d$g)-sd_g/.5^2,
    sd_i*sum_group(ds)-z_i,sd_g*sum_group(ds*d$g)-z_g)
json(list(cases=list(list(point=as.list(point),posterior=lp,gradient=as.list(gradient),
    mu=as.list(mu),sigma=as.list(sigma),pointwise=as.list(rows))),
    priors=as.data.frame(get_prior(fg,dg,family=gaussian(),prior=pg)),
    X=frame$X,X_sigma=frame$X_sigma,J_mu=as.list(frame$J_1),J_sigma=as.list(frame$J_2),
    Z_mu=as.list(frame$Z_1_1),Z_sigma=list(as.list(frame$Z_2_sigma_1),as.list(frame$Z_2_sigma_2))),
    "gaussian-groups.json")
missing<-data.frame(y=c(2,4,8,NA,9,5),x=c(1,2,3,4,5,NA),g=factor(c("B","A",NA,"A","B","A"),levels=c("B","A","unused")))
frame<-make_standata(bf(y~x,sigma~g),missing,family=gaussian())
json(list(X=frame$X,X_sigma=frame$X_sigma,Y=frame$Y),"missing.json")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
