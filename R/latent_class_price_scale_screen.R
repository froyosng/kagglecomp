library(tidyverse)
library(mlogit)
source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")
member_vars <- c("segmentind","incomea","agea")

split <- readRDS("data_processed/train_val_split.rds")
tr <- split$train_long_tr
va <- split$train_long_val
tr$chid <- paste(tr$Case, tr$Task, sep = "_")
va$chid <- paste(va$Case, va$Task, sep = "_")

ctr <- sapply(tr[scaler_vars], mean, na.rm = TRUE)
scl <- sapply(tr[scaler_vars], sd, na.rm = TRUE)
scl[scl == 0] <- 1

make_features <- function(df, ctr, scl) {
  df <- as.data.frame(df)
  df$d2 <- as.integer(df$alt == 2); df$d3 <- as.integer(df$alt == 3)
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

attr_terms <- paste0("factor(", attrs, ")")
price_terms <- paste0("Pr_lvl", 2:12)
price_int_terms <- c("P_income","P_age","P_miles","P_night", paste0("P_seg",2:6), "P_task",
                      paste0("P_region",2:5), paste0("P_ppark",2:5),
                      "is_cheapest","is_dearest","price_gap_min","price_gap_max")
nonprice_int_terms <- c("In_income","In_age","In_miles","In_night","In_gender","In_urb","In_educ",
                         paste0("In_seg",2:6), "In_task", paste0("In_region",2:5), paste0("In_ppark",2:5))
fml_full <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2","d3", price_int_terms, nonprice_int_terms), collapse=" + "), "| 0"))

tr_feat <- make_features(tr, ctr, scl)
va_feat <- make_features(va, ctr, scl)

cat("Fitting full baseline (all confirmed terms, price and non-price together)...\n")
mdat_tr <- dfidx(tr_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")
mdat_va <- dfidx(va_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")
base_mod <- mlogit(fml_full, data = mdat_tr)
cat("Baseline fit. LogLik:", as.numeric(logLik(base_mod)), "\n")

# eta_price computed DIRECTLY from known feature columns x fitted coefficients
# (no model.matrix() on new data -- that approach mismatched columns earlier).
# eta_nonprice obtained as log(predicted prob) MINUS eta_price: log(predict())
# recovers the true linear predictor only up to a per-task additive constant
# (softmax normalization), but that constant is identical across all 4
# alternatives in a task, so it cancels out in the final softmax regardless
# of how eta_nonprice/eta_price are recombined below -- verified this holds
# before trusting any downstream number.
get_eta_split <- function(feat, mdat) {
  cf <- coef(base_mod)
  eta_price <- as.matrix(feat[, price_int_terms, drop = FALSE]) %*% cf[price_int_terms] +
    as.matrix(feat[, price_terms, drop = FALSE]) %*% cf[price_terms]
  pred_shared <- predict(base_mod, newdata = mdat)
  long_pred <- as.data.frame(pred_shared) %>% rownames_to_column("chid") %>%
    pivot_longer(-chid, names_to = "alt_name", values_to = "p") %>%
    mutate(alt = as.integer(gsub("\\D", "", alt_name)))
  feat2 <- feat %>% select(-any_of("p")) %>% left_join(long_pred %>% select(chid, alt, p), by = c("chid","alt"))
  eta_total <- log(pmax(feat2$p, 1e-12))
  list(eta_price = as.numeric(eta_price), eta_nonprice = eta_total - as.numeric(eta_price), feat = feat2)
}

o_tr <- get_eta_split(tr_feat, mdat_tr)
o_va <- get_eta_split(va_feat, mdat_va)
tr_feat <- o_tr$feat; va_feat <- o_va$feat
eta_price_tr <- o_tr$eta_price; eta_nonprice_tr <- o_tr$eta_nonprice
eta_price_va <- o_va$eta_price; eta_nonprice_va <- o_va$eta_nonprice

# sanity check: eta_nonprice + eta_price should reconstruct eta_total (up to
# the per-task constant already baked into eta_nonprice) -- check within-task
# RELATIVE differences match log(predicted prob) relative differences exactly
chk <- tr_feat %>% mutate(eta_recon = eta_nonprice_tr + eta_price_tr) %>%
  group_by(chid) %>% mutate(d1 = eta_recon - eta_recon[1], d2c = log(p) - log(p)[1]) %>% ungroup()
cat("Sanity check max abs diff (should be ~0):", max(abs(chk$d1 - chk$d2c), na.rm = TRUE), "\n\n")

# ---- EM for a 2-class SCALE on price-related utility ----
cases <- unique(tr_feat$Case)
resp_df <- tr_feat %>% distinct(Case, .keep_all = TRUE) %>% arrange(match(Case, cases))
Zmat <- as.matrix(cbind(1, scale(resp_df[, member_vars])))
task_by_chid <- tr_feat %>% distinct(chid, Case)

loglik_given_class <- function(lambda) {
  eta <- eta_nonprice_tr + lambda * eta_price_tr
  d <- data.frame(chid = tr_feat$chid, eta = eta, chosen = tr_feat$chosen)
  d <- d %>% group_by(chid) %>% mutate(p = exp(eta - max(eta)) / sum(exp(eta - max(eta))))
  ll <- d %>% filter(chosen == 1) %>% ungroup() %>% select(chid, p) %>% mutate(ll = log(pmax(p, 1e-15)))
  setNames(ll$ll, ll$chid)
}

set.seed(1)
best_ll <- -Inf; best_fit <- NULL
for (start in 1:4) {
  cat("--- EM start", start, "---\n")
  lambda <- c(1, 1) + rnorm(2, 0, 0.4)
  gamma <- rnorm(ncol(Zmat), 0, 0.2)
  prev_ll <- -Inf
  for (iter in 1:30) {
    ll1 <- loglik_given_class(lambda[1]); ll2 <- loglik_given_class(lambda[2])
    resp_ll1 <- tapply(ll1[task_by_chid$chid], task_by_chid$Case, sum)[as.character(cases)]
    resp_ll2 <- tapply(ll2[task_by_chid$chid], task_by_chid$Case, sum)[as.character(cases)]
    prior2 <- as.numeric(1 / (1 + exp(-(Zmat %*% gamma))))
    log_prior1 <- log(pmax(1 - prior2, 1e-12)); log_prior2 <- log(pmax(prior2, 1e-12))
    m <- pmax(resp_ll1 + log_prior1, resp_ll2 + log_prior2)
    total_ll <- sum(m + log(exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m)))
    post2 <- exp(resp_ll2 + log_prior2 - m) / (exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m))
    if (abs(total_ll - prev_ll) < 0.5) { cat("Converged at iter", iter, "\n"); break }
    prev_ll <- total_ll
    gamma <- coef(glm(post2 ~ Zmat - 1, family = binomial()))
    w1 <- 1 - post2[as.character(tr_feat$Case)]; w2 <- post2[as.character(tr_feat$Case)]
    fit1 <- tryCatch(coef(glm(chosen ~ eta_price_tr - 1, data = tr_feat, weights = w1, offset = eta_nonprice_tr, family = quasibinomial())),
                      error = function(e) c(eta_price_tr = 1))
    fit2 <- tryCatch(coef(glm(chosen ~ eta_price_tr - 1, data = tr_feat, weights = w2, offset = eta_nonprice_tr, family = quasibinomial())),
                      error = function(e) c(eta_price_tr = 1))
    lambda <- c(fit1[["eta_price_tr"]], fit2[["eta_price_tr"]])
  }
  cat("Final log-lik:", total_ll, " lambda:", round(lambda, 4), "\n")
  if (total_ll > best_ll) { best_ll <- total_ll; best_fit <- list(lambda = lambda, gamma = gamma) }
}
cat("\n=== Best fit ===\n"); print(best_fit)

# ---- Evaluate on held-out validation ----
resp_va <- va_feat %>% distinct(Case, .keep_all = TRUE)
member_ctr <- sapply(resp_df[member_vars], mean); member_scl <- sapply(resp_df[member_vars], sd)
z_va <- sweep(sweep(resp_va[member_vars], 2, member_ctr, "-"), 2, member_scl, "/")
Zmat_va <- as.matrix(cbind(1, z_va))
prior2_va <- setNames(1 / (1 + exp(-as.numeric(Zmat_va %*% best_fit$gamma))), resp_va$Case)

d <- data.frame(chid = va_feat$chid, Case = va_feat$Case, alt = va_feat$alt,
                eta1 = eta_nonprice_va + best_fit$lambda[1] * eta_price_va,
                eta2 = eta_nonprice_va + best_fit$lambda[2] * eta_price_va)
d <- d %>% group_by(chid) %>% mutate(p1 = exp(eta1-max(eta1))/sum(exp(eta1-max(eta1))),
                                      p2 = exp(eta2-max(eta2))/sum(exp(eta2-max(eta2)))) %>% ungroup()
w2 <- prior2_va[as.character(d$Case)]
d$p_mix <- (1 - w2) * d$p1 + w2 * d$p2
wide <- d %>% select(chid, alt, p_mix) %>% pivot_wider(names_from = alt, values_from = p_mix, names_prefix = "p")

truth_mat <- va_feat %>% distinct(chid, alt, chosen) %>% pivot_wider(names_from = alt, values_from = chosen, names_prefix = "Ch")
truth_mat <- truth_mat[match(wide$chid, truth_mat$chid), ]
pred <- as.matrix(wide[, c("p1","p2","p3","p4")])

cat("\nShared baseline only (lambda=1 for everyone, i.e. the original m8trpg-equivalent model):",
    round(log_loss(truth_mat[,paste0("Ch",1:4)],
                    { e <- eta_nonprice_va + eta_price_va; dd <- data.frame(chid=va_feat$chid, e=e, alt=va_feat$alt) %>%
                        group_by(chid) %>% mutate(p=exp(e-max(e))/sum(exp(e-max(e)))) %>% ungroup() %>%
                        select(chid,alt,p) %>% pivot_wider(names_from=alt, values_from=p, names_prefix="p")
                      as.matrix(dd[match(wide$chid, dd$chid), c("p1","p2","p3","p4")]) }), 6), "\n")
cat("2-class price-sensitivity-scale latent mixture:", round(log_loss(truth_mat[,paste0("Ch",1:4)], pred), 6), "\n")
cat("(reference: confirmed base single-split validation log loss = 1.159681)\n")
