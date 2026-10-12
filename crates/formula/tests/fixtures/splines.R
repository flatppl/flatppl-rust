# Regenerate splines.json with mgcv 1.9.4 and brms 2.23.2 available.
suppressPackageStartupMessages(library(brms))
stopifnot(as.character(packageVersion("mgcv")) == "1.9.4",
          as.character(packageVersion("brms")) == "2.23.2")
capture <- function(x, rank) {
  term <- if (is.null(rank)) mgcv::s(x) else mgcv::s(x, k = rank)
  data <- data.frame(x)
  smooth <- brms:::smoothCon(term, data, absorb.cons = TRUE,
                            diagonal.penalty = TRUE)[[1]]
  random <- mgcv::smooth2random(smooth, names(data), type = 2)
  prediction <- c(min(x)-2, min(x), mean(x), max(x), max(x)+2)
  predicted <- brms:::s2rPred(smooth, random, data.frame(x = prediction))
  list(x = x, rank = rank, penalty_scale = smooth$S.scale,
       basis = unname(cbind(random$Xf, random$rand[[1]])),
       prediction = prediction,
       predicted = unname(cbind(predicted$Xf, predicted$rand[[1]])))
}
x <- c(-2.7,-2,-1.8,-1.1,-0.6,-0.1,0.2,0.9,1.2,1.8,2.1,2.8,3.4,4)
cases <- list(truncated = capture(x, 6L), default = capture(x, NULL),
              duplicates = capture(c(-2,-2,-1.5,-1,-.3,.1,.1,.8,1.2,1.2,2,3,3), 6L),
              full = capture(c(-2,-1,0,1,2,3), 6L))
json <- function(x) {
  if (is.null(x)) return("null")
  if (is.matrix(x)) return(json(lapply(seq_len(nrow(x)), function(i) unname(as.list(x[i, ])))))
  if (is.list(x)) {
    values <- vapply(x, json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse=","), "]"))
    keys <- vapply(names(x), encodeString, "", quote='"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse=","), "}"))
  }
  if (length(x) != 1L) return(json(unname(as.list(x))))
  trimws(formatC(x, digits=17, format="g", decimal.mark="."))
}
writeLines(json(cases), "splines.json", useBytes=TRUE)
