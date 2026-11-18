# Regenerate factor-smooths.json with mgcv 1.9-4 and jsonlite.
suppressPackageStartupMessages(library(mgcv))
stopifnot(as.character(packageVersion("mgcv")) == "1.9.4")

x <- c(-2.7, -2, -1.8, -1.1, -.6, -.1, .2, .9, 1.2, 1.8, 2.1, 2.8, 3.4, 4)
levels <- c("B", "A")
groups <- c("B", "B", "B", "A", "B", "A", "B", "A", "A", "A", "B", "A", "B", "A")
new_x <- c(-3, .45, 4.5, 2.2)
new_groups <- c("A", "B", "A", "B")

capture <- function(ordered) {
  data <- data.frame(x, f = factor(groups, levels = levels, ordered = ordered))
  new_data <- data.frame(x = new_x, f = factor(new_groups, levels = rev(levels), ordered = ordered))
  smooths <- smoothCon(s(x, k = 4L, by = f), data, absorb.cons = TRUE,
                       diagonal.penalty = TRUE)
  curves <- lapply(smooths, function(smooth) {
    random <- smooth2random(smooth, names(data), type = 2)
    train <- cbind(random$Xf, random$rand[[1]])
    # Transfer the training coefficient transform to mgcv's prediction matrix.
    design <- PredictMat(smooth, new_data) %*% random$trans.U
    design <- sweep(design, 2L, random$trans.D, "*")
    predict <- design[, c(which(random$pen.ind == 0), which(random$pen.ind != 0))]
    rows <- rbind(train, predict)
    # Rust fixes the otherwise arbitrary sign of the affine slope.
    if (sum(train[, 1] * x) < 0) rows[, 1] <- -rows[, 1]
    list(level = smooth$by.level, basis = unname(rows))
  })
  list(ordered = ordered, curves = curves)
}

jsonlite::write_json(list(x = x, levels = levels, groups = groups, new_x = new_x,
                         new_groups = new_groups, cases = lapply(c(FALSE, TRUE), capture)),
                     "factor-smooths.json", auto_unbox = TRUE, digits = NA)
