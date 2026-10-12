# Run: Rscript factor-matrix.R factor-matrix.json
# Base R is an oracle dependency only. brms itself is not executed.
# Its fixed-formula reconstruction and coefficient renaming follow source:
# https://github.com/paul-buerkner/brms/tree/c89c4c05ef52fefb4b78d5fb80df0cf6352f48b4
# R/brmsterms.R terms_fe / validate_terms, R/data-helpers.R validate_data,
# R/data-predictor.R get_model_matrix, R/misc.R rename.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1L, Sys.setlocale("LC_COLLATE", "C") == "C")
options(contrasts = c("contr.treatment", "contr.poly"))
array <- function(x) unname(as.list(x))
json <- function(x) {
  if (is.null(x)) return("null")
  if (is.list(x)) {
    if (!length(x)) return(if (is.null(names(x))) "[]" else "{}")
    values <- vapply(x, json, "")
    if (is.null(names(x))) return(paste0("[", paste(values, collapse = ","), "]"))
    keys <- vapply(names(x), encodeString, "", quote = '"')
    return(paste0("{", paste(paste0(keys, ":", values), collapse = ","), "}"))
  }
  if (length(x) != 1L) return(json(array(x)))
  if (is.na(x)) return("null")
  if (is.character(x)) return(encodeString(x, quote = '"'))
  if (is.logical(x)) return(if (x) "true" else "false")
  if (is.numeric(x)) return(trimws(formatC(x, digits = 17, format = "g", decimal.mark = ".")))
  stop("unsupported JSON value")
}
matrix_record <- function(x) {
  list(row_names = array(rownames(x)), column_names = array(colnames(x)),
       values = lapply(seq_len(nrow(x)), function(i) array(unname(x[i, ]))))
}
table_record <- function(d) {
  columns <- lapply(names(d), function(name) {
    v <- d[[name]]
    valid <- !is.na(v)
    if (is.factor(v)) {
      codes <- as.integer(v) - 1L
      codes[!valid] <- 0L
      values <- list(kind = "factor", values = list(levels = array(levels(v)),
                     codes = array(codes), ordered = is.ordered(v)))
    } else {
      v[!valid] <- if (is.logical(v)) FALSE else if (is.integer(v)) 0L else 0
      values <- list(kind = if (is.logical(v)) "bool" else if (is.integer(v)) "int" else "real64",
                     values = array(v))
    }
    out <- list(name = name, values = values)
    if (!all(valid)) out$valid <- array(valid)
    out
  })
  list(nrows = nrow(d), columns = columns)
}
brms_names <- function(x) {
  from <- c(" ", "(", ")", "[", "]", ",", '"', "'", "?", "+", "-", "*", "/", "^", "=", "$")
  to <- c(rep("", 9), "P", "M", "MU", "D", "E", "EQ", "USD")
  for (i in seq_along(from)) x <- gsub(from[i], to[i], x, fixed = TRUE)
  stopifnot(!anyDuplicated(x))
  array(x)
}

training <- data.frame(
  y = c(2, 4, 8, 3, 6, 7, 5, 9), x = c(-2, 0, 1, 3, 4, 2, -1, 5),
  z = c(3, 1, 2, 4, -1, 0, 5, 2), w = c(1, 2, 1, 3, 2, 4, 1, 2),
  f = factor(c("C", "A", "B", "A", "C", "B", "A", "B"), levels = c("B", "C", "A", "unused")),
  g = factor(c("v", "u", "u", "v", "u", "v", "v", "u"), levels = c("u", "v")),
  o = ordered(c("low", "mid", "high", "top", "mid", "top", "low", "high"),
              levels = c("low", "mid", "high", "top")),
  flag = c(TRUE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE), unused = seq_len(8)
)
missing <- training
missing$z[2] <- NA_real_
missing$f[5] <- NA
missing$unused[4] <- NA_integer_
prediction <- data.frame(
  x = c(8, -2, 5), z = c(-1, 3, 2), w = c(3, 2, 1),
  f = factor(rep("A", 3), levels = c("A", "B", "C", "unused")),
  g = factor(c("v", "u", "v"), levels = c("v", "u")),
  o = ordered(c("top", "mid", "top"), levels = c("top", "high", "mid", "low")),
  flag = c(FALSE, TRUE, FALSE)
)
alltrue <- training
alltrue$flag[] <- TRUE
poly16 <- training[rep(seq_len(nrow(training)), 2L), ]
rownames(poly16) <- NULL
poly_levels <- c("low", "mid", "high", "top", "fifth", "sixth", "seventh", "eighth",
                 "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth")
poly16$o <- ordered(poly_levels, levels = poly_levels)
rare_missing <- missing
rare_missing$f <- factor(c("C", "rare", "B", "A", "C", "B", "A", "B"),
                         levels = c("B", "C", "A", "rare", "unused"))
tables <- list(training = training, missing = missing, prediction = prediction,
               alltrue = alltrue, poly16 = poly16, rare_missing = rare_missing)
specs <- list(
  list("helmert", "y ~ f", "training", c(f = "helmert")),
  list("ordered_poly16", "y ~ o", "poly16"),
  list("first_factor_cells", "y ~ 0 + x + f + g", "training"),
  list("interaction_only", "y ~ f:g", "training"),
  list("partial_hierarchy", "y ~ f + f:g", "training"),
  list("numeric_main", "y ~ x + x:f", "training"),
  list("degree_reconstruction", "y ~ f:g + g + f", "training"),
  list("logical_one_observed", "y ~ flag", "alltrue"),
  list("removed_na", "y ~ x + z - z", "missing"),
  list("level_removed_with_row", "y ~ x + z + f", "rare_missing"),
  list("prediction_interaction", "y ~ x * f + o", "training", c(f = "sum"))
)
cases <- lapply(specs, function(spec) {
  contrasts <- if (length(spec) == 4L) as.list(spec[[4]]) else setNames(list(), character())
  ca <- if (length(contrasts)) lapply(contrasts, function(x) paste0("contr.", x)) else NULL
  terms <- terms(as.formula(spec[[2]]))
  fixed <- as.formula(paste("y ~", paste(c(attr(terms, "intercept"), attr(terms, "term.labels")), collapse = " + ")))
  mf <- model.frame(fixed, tables[[spec[[3]]]], na.action = na.omit, drop.unused.levels = TRUE)
  t <- attr(mf, "terms")
  x <- model.matrix(t, mf, contrasts.arg = ca)
  factors <- mf[vapply(mf, is.factor, TRUE)]
  factor_levels <- lapply(factors, function(v) list(levels = array(levels(v)), ordered = is.ordered(v)))
  means <- as.list(colMeans(x[, colnames(x) != "(Intercept)", drop = FALSE]))
  pt <- delete.response(t)
  pmf <- model.frame(pt, prediction, na.action = na.omit, drop.unused.levels = FALSE,
                     xlev = lapply(factors, levels))
  px <- model.matrix(pt, pmf, contrasts.arg = attr(x, "contrasts"))
  expected <- list(rows = array(as.integer(rownames(mf)) - 1L), factor_levels = factor_levels,
                   x = matrix_record(x), assign = array(attr(x, "assign")), means = means,
                   prediction_x = matrix_record(px), brms_column_names = brms_names(colnames(x)))
  list(name = spec[[1]], formula = spec[[2]], table = spec[[3]], contrasts = contrasts, expected = expected)
})
header <- list(r_version = R.version.string, stats_version = as.character(packageVersion("stats")),
               locale = Sys.getlocale(), brms_executed = FALSE,
               brms_source_sha = "c89c4c05ef52fefb4b78d5fb80df0cf6352f48b4",
               reconstruction = "Base R execution of the fixed formula reconstructed as brms terms_fe: intercept + surviving ordinary term.labels. brms itself is not installed.")
lines <- c("{", vapply(names(header), function(n) paste0("  ", json(n), ": ", json(header[[n]]), ","), ""),
           '  "tables": {',
           paste(vapply(names(tables), function(n) paste0("    ", json(n), ": ", json(table_record(tables[[n]]))), ""), collapse = ",\n"),
           '  },', '  "cases": [', paste(paste0("    ", vapply(cases, json, "")), collapse = ",\n"), '  ]', "}")
writeLines(lines, args[[1]], useBytes = TRUE)
