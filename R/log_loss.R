# Multi-class log loss, matching the competition's evaluation metric:
#   LogLoss = -(1/n) * sum_i sum_j y_ij * log(p_ij)
#
# actual: data.frame/matrix with columns Ch1..Ch4 (0/1 indicators), one row per observation
# pred:   data.frame/matrix with columns matching alt order (probabilities), same row order as actual
# eps:    clipping bound to avoid log(0) if a predicted probability is exactly 0
log_loss <- function(actual, pred, eps = 1e-15) {
  actual <- as.matrix(actual)
  pred <- as.matrix(pred)
  stopifnot(all(dim(actual) == dim(pred)))

  # Predictions should sum to 1 per row; competition rescales if not, so we mirror that here
  row_sums <- rowSums(pred)
  pred <- pred / row_sums

  pred <- pmin(pmax(pred, eps), 1 - eps)
  -mean(rowSums(actual * log(pred)))
}
