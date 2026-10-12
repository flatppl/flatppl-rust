# Independent brms oracle; neither R nor brms is a converter dependency.
library(brms)
Sys.setlocale("LC_COLLATE", "C")
options(contrasts=c("contr.treatment", "contr.poly"))
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
  if (is.na(x)) return("null")
  if (is.character(x)) return(encodeString(x, quote='"'))
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(trimws(formatC(x, digits=17, format="g", decimal.mark=".")))
  stop("unsupported JSON value")
}
d <- data.frame(y=c(2,4,8,7), x=c(1,2,3,4),
  g=factor(c("A","B","A","B"), levels=c("B","A")),
  h=c(1,2,2,1), f=factor(c("b","a","b","a"), levels=c("b","a")))
formulas <- c(
  slopes_correlated="y ~ x + (1+x|g)",
  slopes_independent="y ~ x + (1+x||g)",
  slopes_factor="y ~ (0+x+f|g)",
  groups_nested="y ~ (1+x|g/h)",
  groups_nested_with_fixed="y ~ h + (1+x|g/h)",
  slopes_disjoint="y ~ (1|g)+(0+x|g)",
  slopes_duplicate="y ~ (1+x|g)+(1+x|g)")
cases <- lapply(names(formulas), function(name) {
  data <- unclass(make_standata(bf(as.formula(formulas[[name]])), data=d))
  list(name=name, formula=formulas[[name]], standata=data)
})
output <- list(source=paste("brms", packageVersion("brms"), "make_standata, C collation"), cases=cases)
writeLines(json(output), commandArgs(TRUE)[[1]], useBytes=TRUE)
