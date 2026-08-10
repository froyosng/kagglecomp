attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")

xgb_covariates <- c(
  "segmentind","yearind","milesind","milesa","nightind","nighta",
  "pparkind","genderind","ageind","agea","educind",
  "regionind","Urbind","incomeind","incomea"
)

log_loss_matrix <- function(actual, pred, eps = 1e-15) {
  actual <- as.matrix(actual)
  pred <- as.matrix(pred)
  pred <- pred / rowSums(pred)
  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}

canonical_fold_map <- function() {
  saved <- readRDS("data_processed/oof_ensemble_v10.rds")
  saved$fold_of_case
}

sort_long_tasks <- function(df) {
  df[order(df$No, as.integer(df$alt)), , drop = FALSE]
}

long_truth_matrix <- function(df) {
  df <- sort_long_tasks(df)
  stopifnot(nrow(df) %% 4 == 0L)
  out <- matrix(as.integer(df$chosen), ncol = 4, byrow = TRUE)
  stopifnot(all(rowSums(out) == 1L))
  out
}

rank_feature_matrix <- function(df) {
  df <- sort_long_tasks(df)
  df$alt <- as.integer(df$alt)
  df$inside <- as.integer(df$alt != 4L)
  df$d2 <- as.integer(df$alt == 2L)
  df$d3 <- as.integer(df$alt == 3L)
  df$Task_c <- (as.numeric(df$Task) - 10) / 9

  price_mat <- matrix(as.numeric(df$Price), ncol = 4, byrow = TRUE)
  inside_price <- price_mat[, 1:3, drop = FALSE]
  pmin3 <- apply(inside_price, 1, min)
  pmax3 <- apply(inside_price, 1, max)
  df$price_min <- rep(pmin3, each = 4)
  df$price_max <- rep(pmax3, each = 4)
  df$is_cheapest <- as.integer(df$inside == 1L & df$Price == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1L & df$Price == df$price_max)
  df$price_gap_min <- ifelse(df$inside == 1L, df$Price - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1L, df$price_max - df$Price, 0)

  feature_cols <- c(
    attrs, "Price", "alt", "inside", "d2", "d3", "Task_c",
    "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max",
    xgb_covariates
  )
  out <- as.matrix(df[, feature_cols, drop = FALSE])
  storage.mode(out) <- "double"
  out
}

wide_feature_matrix <- function(df) {
  attr_wide <- paste0(rep(attrs, each = 4), rep(1:4, times = length(attrs)))
  price_wide <- paste0("Price", 1:4)
  out <- as.matrix(df[, c(attr_wide, price_wide, xgb_covariates), drop = FALSE])
  storage.mode(out) <- "double"
  out
}

softmax_margins <- function(margins, scale = 1) {
  stopifnot(length(margins) %% 4L == 0L)
  z <- matrix(as.numeric(margins) * scale, ncol = 4, byrow = TRUE)
  z <- z - apply(z, 1, max)
  ez <- exp(z)
  ez / rowSums(ez)
}

best_margin_scale <- function(margins, truth,
                              scales = seq(0.05, 3, by = 0.025)) {
  losses <- vapply(scales, function(s) {
    log_loss_matrix(truth, softmax_margins(margins, s))
  }, numeric(1))
  i <- which.min(losses)
  list(scale = scales[i], logloss = losses[i],
       curve = data.frame(scale = scales, logloss = losses))
}

search_two_way_blend <- function(p1, p2, truth,
                                 weights = seq(0, 1, by = 0.01)) {
  losses <- vapply(weights, function(w) {
    log_loss_matrix(truth, w * p1 + (1 - w) * p2)
  }, numeric(1))
  i <- which.min(losses)
  list(weight1 = weights[i], logloss = losses[i],
       curve = data.frame(weight1 = weights, logloss = losses))
}

search_three_way_blend <- function(p1, p2, p3, truth, step = 0.02) {
  weights <- seq(0, 1, by = step)
  grid <- expand.grid(w1 = weights, w2 = weights)
  grid <- grid[grid$w1 + grid$w2 <= 1 + 1e-12, , drop = FALSE]
  grid$w3 <- 1 - grid$w1 - grid$w2
  grid$logloss <- vapply(seq_len(nrow(grid)), function(i) {
    p <- grid$w1[i] * p1 + grid$w2[i] * p2 + grid$w3[i] * p3
    log_loss_matrix(truth, p)
  }, numeric(1))
  grid[which.min(grid$logloss), , drop = FALSE]
}

write_result_csv <- function(x, path) {
  write.csv(as.data.frame(x), path, row.names = FALSE)
}
