library(tidyverse)
library(mlogit)
source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
member_vars <- c("segmentind","incomea","agea")

train <- read.csv("csv files/train.csv")
pat <- paste0("^(", paste(c(attrs, "Ch"), collapse = "|"), ")([1-4])$")
train_long <- train %>%
  pivot_longer(cols = matches(pat), names_to = c(".value", "alt"),
               names_pattern = "^(.*)([1-4])$") %>%
  mutate(alt = as.integer(alt), chosen = as.integer(Ch == 1),
         chid = paste(Case, Task, sep = "_"),
         d2 = as.integer(alt == 2), d3 = as.integer(alt == 3))

make_features <- function(df, ctr, scl) {
  df <- as.data.frame(df)
  z <- sweep(sweep(df[scaler_vars], 2, ctr, "-"), 2, scl, "/")
  df$inside <- as.integer(df$alt != 4)
  df$Price_num <- as.numeric(df$Price)
  for (k in 2:12) df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  df$Task_c <- (as.numeric(df$Task) - 10) / 9
  df$P_income <- df$Price_num * z$incomea; df$P_age <- df$Price_num * z$agea
  df$P_miles <- df$Price_num * z$milesa; df$P_night <- df$Price_num * z$nighta
  df$In_income <- df$inside * z$incomea; df$In_age <- df$inside * z$agea
  df$In_miles <- df$inside * z$milesa; df$In_night <- df$inside * z$nighta
  df$In_gender <- df$inside * z$genderind; df$In_urb <- df$inside * z$Urbind; df$In_educ <- df$inside * z$educind
  for (s in 2:6) { df[[paste0("P_seg", s)]] <- df$Price_num * (df$segmentind == s); df[[paste0("In_seg", s)]] <- df$inside * (df$segmentind == s) }
  df$P_task <- df$Price_num * df$Task_c; df$In_task <- df$inside * df$Task_c
  for (s in 2:5) {
    df[[paste0("P_region", s)]] <- df$Price_num * (df$regionind == s); df[[paste0("In_region", s)]] <- df$inside * (df$regionind == s)
    df[[paste0("P_ppark", s)]] <- df$Price_num * (df$pparkind == s); df[[paste0("In_ppark", s)]] <- df$inside * (df$pparkind == s)
  }
  task_stats <- df %>% group_by(chid) %>% summarise(price_min = min(Price_num[inside==1]), price_max = max(Price_num[inside==1]), .groups="drop")
  df <- left_join(df, task_stats, by = "chid")
  df$is_cheapest <- as.integer(df$inside==1 & df$Price_num==df$price_min)
  df$is_dearest <- as.integer(df$inside==1 & df$Price_num==df$price_max)
  df$price_gap_min <- ifelse(df$inside==1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside==1, df$price_max - df$Price_num, 0)
  df
}

attrs_noprice <- attrs[attrs != "Price"]
attr_terms <- paste0("factor(", attrs_noprice, ")")
price_terms <- paste0("Pr_lvl", 2:12)
price_int_terms <- c("P_income","P_age","P_miles","P_night", paste0("P_seg",2:6), "P_task",
                      paste0("P_region",2:5), paste0("P_ppark",2:5),
                      "is_cheapest","is_dearest","price_gap_min","price_gap_max")
nonprice_int_terms <- c("In_income","In_age","In_miles","In_night","In_gender","In_urb","In_educ",
                         paste0("In_seg",2:6), "In_task", paste0("In_region",2:5), paste0("In_ppark",2:5))
fml_full <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2","d3", price_int_terms, nonprice_int_terms), collapse=" + "), "| 0"))

# eta_price computed directly from known feature columns x fitted coefficients
# (avoids the earlier model.matrix-on-new-data column-alignment bug entirely).
# eta_nonprice = log(predicted prob) - eta_price: log(predict()) recovers the
# true linear predictor only up to a per-task additive constant, but that
# constant is identical across all 4 alternatives in a task so it cancels in
# the final softmax regardless of how eta_nonprice/eta_price are recombined
# -- verified exactly (to machine precision) in the single-split screen.
get_eta_split <- function(base_mod, mdat, feat) {
  cf <- coef(base_mod)
  eta_price <- as.numeric(as.matrix(feat[, price_int_terms, drop = FALSE]) %*% cf[price_int_terms] +
                             as.matrix(feat[, price_terms, drop = FALSE]) %*% cf[price_terms])
  pred_shared <- predict(base_mod, newdata = mdat)
  long_pred <- as.data.frame(pred_shared) %>% rownames_to_column("chid") %>%
    pivot_longer(-chid, names_to = "alt_name", values_to = "p") %>%
    mutate(alt = as.integer(gsub("\\D", "", alt_name)))
  feat <- feat %>% select(-any_of("p")) %>% left_join(long_pred %>% select(chid, alt, p), by = c("chid","alt"))
  eta_total <- log(pmax(feat$p, 1e-12))
  list(eta_price = eta_price, eta_nonprice = eta_total - eta_price, feat = feat)
}

fit_em <- function(tr_feat, eta_price, eta_nonprice, n_starts = 4) {
  cases <- unique(tr_feat$Case)
  resp_df <- tr_feat %>% distinct(Case, .keep_all = TRUE) %>% arrange(match(Case, cases))
  member_ctr <- sapply(resp_df[member_vars], mean, na.rm = TRUE)
  member_scl <- sapply(resp_df[member_vars], sd, na.rm = TRUE)
  z_member <- sweep(sweep(resp_df[member_vars], 2, member_ctr, "-"), 2, member_scl, "/")
  Zmat <- as.matrix(cbind(1, z_member))
  task_by_chid <- tr_feat %>% distinct(chid, Case)

  loglik_given_class <- function(lambda) {
    eta <- eta_nonprice + lambda * eta_price
    d <- data.frame(chid = tr_feat$chid, eta = eta, chosen = tr_feat$chosen)
    d <- d %>% group_by(chid) %>% mutate(p = exp(eta - max(eta)) / sum(exp(eta - max(eta))))
    ll <- d %>% filter(chosen == 1) %>% ungroup() %>% select(chid, p) %>% mutate(ll = log(pmax(p, 1e-15)))
    setNames(ll$ll, ll$chid)
  }

  best_ll <- -Inf; best_fit <- NULL
  for (start in 1:n_starts) {
    lambda <- c(1, 1) + rnorm(2, 0, 0.4)
    gamma <- rnorm(ncol(Zmat), 0, 0.2)
    prev_ll <- -Inf; total_ll <- -Inf
    for (iter in 1:30) {
      ll1 <- loglik_given_class(lambda[1]); ll2 <- loglik_given_class(lambda[2])
      resp_ll1 <- tapply(ll1[task_by_chid$chid], task_by_chid$Case, sum)[as.character(cases)]
      resp_ll2 <- tapply(ll2[task_by_chid$chid], task_by_chid$Case, sum)[as.character(cases)]
      prior2 <- as.numeric(1 / (1 + exp(-(Zmat %*% gamma))))
      log_prior1 <- log(pmax(1 - prior2, 1e-12)); log_prior2 <- log(pmax(prior2, 1e-12))
      m <- pmax(resp_ll1 + log_prior1, resp_ll2 + log_prior2)
      total_ll <- sum(m + log(exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m)))
      post2 <- exp(resp_ll2 + log_prior2 - m) / (exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m))
      if (abs(total_ll - prev_ll) < 0.5) break
      prev_ll <- total_ll
      gamma <- coef(glm(post2 ~ Zmat - 1, family = binomial()))
      w1 <- 1 - post2[as.character(tr_feat$Case)]; w2 <- post2[as.character(tr_feat$Case)]
      fit1 <- tryCatch(coef(glm(chosen ~ eta_price - 1, data = tr_feat, weights = w1, offset = eta_nonprice, family = quasibinomial())),
                        error = function(e) c(eta_price = 1))
      fit2 <- tryCatch(coef(glm(chosen ~ eta_price - 1, data = tr_feat, weights = w2, offset = eta_nonprice, family = quasibinomial())),
                        error = function(e) c(eta_price = 1))
      lambda <- c(fit1[["eta_price"]], fit2[["eta_price"]])
    }
    if (total_ll > best_ll) { best_ll <- total_ll; best_fit <- list(lambda = lambda, gamma = gamma, member_ctr = member_ctr, member_scl = member_scl) }
  }
  best_fit
}

predict_mixture <- function(va_feat, eta_price_va, eta_nonprice_va, fit) {
  resp_va <- va_feat %>% distinct(Case, .keep_all = TRUE)
  z_member <- sweep(sweep(resp_va[member_vars], 2, fit$member_ctr, "-"), 2, fit$member_scl, "/")
  Zmat_va <- as.matrix(cbind(1, z_member))
  prior2 <- setNames(1 / (1 + exp(-as.numeric(Zmat_va %*% fit$gamma))), resp_va$Case)

  eta1 <- eta_nonprice_va + fit$lambda[1] * eta_price_va
  eta2 <- eta_nonprice_va + fit$lambda[2] * eta_price_va
  d <- data.frame(chid = va_feat$chid, Case = va_feat$Case, alt = va_feat$alt, eta1 = eta1, eta2 = eta2)
  d <- d %>% group_by(chid) %>% mutate(p1 = exp(eta1-max(eta1))/sum(exp(eta1-max(eta1))),
                                        p2 = exp(eta2-max(eta2))/sum(exp(eta2-max(eta2)))) %>% ungroup()
  w2 <- prior2[as.character(d$Case)]
  d$p_mix <- (1 - w2) * d$p1 + w2 * d$p2
  wide <- d %>% select(chid, alt, p_mix) %>% pivot_wider(names_from = alt, values_from = p_mix, names_prefix = "p")
  list(pred = as.matrix(wide[, c("p1","p2","p3","p4")]), chid_order = wide$chid)
}

set.seed(4821)
cases <- unique(train_long$Case)
folds <- sample(rep(1:5, length.out = length(cases)))
names(folds) <- cases

oof_lc <- matrix(NA_real_, nrow(train), 4)
oof_shared_baseline <- matrix(NA_real_, nrow(train), 4)
oof_truth <- as.matrix(train[, c("Ch1","Ch2","Ch3","Ch4")])

for (k in 1:5) {
  cat("\n========== FOLD", k, "==========\n")
  val_cases_k <- cases[folds == k]
  tr_k <- train_long %>% filter(!(Case %in% val_cases_k))
  va_k <- train_long %>% filter(Case %in% val_cases_k)

  ctr <- sapply(tr_k[scaler_vars], mean, na.rm = TRUE)
  scl <- sapply(tr_k[scaler_vars], sd, na.rm = TRUE); scl[scl == 0] <- 1

  tr_feat <- make_features(tr_k, ctr, scl)
  va_feat <- make_features(va_k, ctr, scl)

  mdat_tr <- dfidx(tr_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")
  mdat_va <- dfidx(va_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")

  base_mod <- mlogit(fml_full, data = mdat_tr)
  cat("Baseline fit. LogLik:", as.numeric(logLik(base_mod)), "\n")

  o_tr <- get_eta_split(base_mod, mdat_tr, tr_feat); tr_feat <- o_tr$feat
  o_va <- get_eta_split(base_mod, mdat_va, va_feat); va_feat <- o_va$feat

  fit <- fit_em(tr_feat, o_tr$eta_price, o_tr$eta_nonprice, n_starts = 4)
  cat("EM done. lambda:", round(fit$lambda, 4), "\n")

  pm <- predict_mixture(va_feat, o_va$eta_price, o_va$eta_nonprice, fit)
  va_map <- va_feat %>% distinct(chid, No)
  row_idx <- match(va_map$No, train$No)
  pred_ordered <- pm$pred[match(va_map$chid, pm$chid_order), ]
  oof_lc[row_idx, ] <- pred_ordered

  pred_shared_va <- predict(base_mod, newdata = mdat_va)
  pred_shared_ordered <- pred_shared_va[match(va_map$chid, rownames(pred_shared_va)), ]
  oof_shared_baseline[row_idx, ] <- pred_shared_ordered

  cat("Fold", k, "log loss -- shared baseline (single population):",
      round(log_loss(oof_truth[row_idx,], pred_shared_ordered), 6),
      " | 2-class price-scale mixture:", round(log_loss(oof_truth[row_idx,], pred_ordered), 6), "\n")
}

cat("\n=== POOLED 5-FOLD CV RESULTS ===\n")
cat("Shared baseline, single population (mlogit_m8trpg):", round(log_loss(oof_truth, oof_shared_baseline), 6), "\n")
cat("2-class price-sensitivity-scale latent mixture:", round(log_loss(oof_truth, oof_lc), 6), "\n")
cat("(reference: mlogit_m8trpg CV = 1.147021)\n")
cat("(reference: full ensemble_v11 CV = 1.145094)\n")
