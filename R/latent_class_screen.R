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

tr_feat <- make_features(tr, ctr, scl)
va_feat <- make_features(va, ctr, scl)

attr_terms <- paste0("factor(", attrs, ")")
price_terms <- paste0("Pr_lvl", 2:12)
shared_int_terms <- c("P_income","P_age","P_miles","P_night","In_income","In_age","In_miles","In_night",
  "In_gender","In_urb","In_educ", paste0("P_seg",2:6), paste0("In_seg",2:6),
  paste0("P_region",2:5), paste0("In_region",2:5), paste0("P_ppark",2:5), paste0("In_ppark",2:5),
  "is_cheapest","is_dearest","price_gap_min","price_gap_max")
# NOTE: P_task/In_task deliberately excluded from the shared baseline -- they
# become the class-varying dimension below.
fml_baseline <- as.formula(paste("chosen ~", paste(c(attr_terms, price_terms, "d2","d3", shared_int_terms), collapse=" + "), "| 0"))

cat("Fitting shared baseline (all terms except task-fatigue)...\n")
mdat_tr <- dfidx(tr_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")
base_mod <- mlogit(fml_baseline, data = mdat_tr)
cat("Baseline fit done. LogLik:", as.numeric(logLik(base_mod)), "\n")

# Fixed baseline linear predictor (offset) for every row, EXCLUDING P_task/In_task.
# log(predicted prob) is a valid stand-in for the true linear predictor up to
# an irrelevant per-task additive constant (softmax is shift-invariant), and
# sidesteps fragile manual model.matrix column-alignment against new data.
pred_shared_tr <- predict(base_mod, newdata = mdat_tr)
long_pred_tr <- as.data.frame(pred_shared_tr) %>%
  rownames_to_column("chid") %>%
  pivot_longer(-chid, names_to = "alt_name", values_to = "p") %>%
  mutate(alt = as.integer(gsub("\\D", "", alt_name)))
tr_feat <- tr_feat %>% left_join(long_pred_tr %>% select(chid, alt, p), by = c("chid","alt"))
offset_tr <- log(pmax(tr_feat$p, 1e-12))

# ---- EM for a 2-class extension on P_task / In_task ----
cases <- unique(tr_feat$Case)
n_resp <- length(cases)
resp_idx <- match(tr_feat$Case, cases)

# respondent-level membership covariates
resp_df <- tr_feat %>% distinct(Case, .keep_all = TRUE) %>% arrange(match(Case, cases))
Zmat <- as.matrix(cbind(1, scale(resp_df[, member_vars])))

softmax2 <- function(g) 1 / (1 + exp(-g))  # P(class2) from a linear score g

loglik_given_class <- function(beta_task, beta_intask, class_label) {
  eta <- offset_tr + beta_task * tr_feat$P_task + beta_intask * tr_feat$In_task
  eta[is.na(eta)] <- offset_tr[is.na(eta)]
  # softmax within each choice task (chid), then pick the chosen alt's log prob
  df <- data.frame(chid = tr_feat$chid, eta = eta, chosen = tr_feat$chosen)
  df <- df %>% group_by(chid) %>% mutate(p = exp(eta - max(eta)) / sum(exp(eta - max(eta))))
  ll_by_task <- df %>% filter(chosen == 1) %>% pull(p) %>% log()
  chid_order <- df %>% distinct(chid) %>% pull(chid)
  setNames(ll_by_task, chid_order[match(unique(df$chid), chid_order)])
}

set.seed(1)
best_ll <- -Inf
best_fit <- NULL
n_starts <- 3

for (start in 1:n_starts) {
  cat("\n--- EM start", start, "---\n")
  beta_task <- rnorm(2, 0, 0.3); beta_intask <- rnorm(2, 0, 0.3)
  gamma <- rnorm(ncol(Zmat), 0, 0.2)

  prev_ll <- -Inf
  for (iter in 1:25) {
    # E-step: per-respondent log-lik under each class (sum over their tasks)
    task_by_chid <- tr_feat %>% distinct(chid, Case)
    ll1_task <- loglik_given_class(beta_task[1], beta_intask[1], 1)
    ll2_task <- loglik_given_class(beta_task[2], beta_intask[2], 2)
    resp_ll1 <- tapply(ll1_task[task_by_chid$chid], task_by_chid$Case, sum)
    resp_ll2 <- tapply(ll2_task[task_by_chid$chid], task_by_chid$Case, sum)
    resp_ll1 <- resp_ll1[as.character(cases)]; resp_ll2 <- resp_ll2[as.character(cases)]

    prior2 <- as.numeric(softmax2(Zmat %*% gamma))
    log_prior1 <- log(1 - prior2); log_prior2 <- log(prior2)
    m <- pmax(resp_ll1 + log_prior1, resp_ll2 + log_prior2)
    total_ll <- sum(m + log(exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m)))
    post2 <- exp(resp_ll2 + log_prior2 - m) / (exp(resp_ll1 + log_prior1 - m) + exp(resp_ll2 + log_prior2 - m))

    if (abs(total_ll - prev_ll) < 0.5) { cat("Converged at iter", iter, "\n"); break }
    prev_ll <- total_ll

    # M-step (a): membership model via weighted logistic regression (soft labels)
    gamma <- coef(glm(post2 ~ Zmat - 1, family = binomial()))

    # M-step (b): two SEPARATE weighted fits, one per class. (A single fit on
    # duplicated rows with weights w1=1-post2, w2=post2 is WRONG -- w1+w2=1
    # identically, so a combined single-formula fit is mathematically
    # independent of post2 entirely. Confirmed this bug produced identical
    # convergence regardless of random start before this fix.)
    w1 <- 1 - post2[as.character(tr_feat$Case)]
    w2 <- post2[as.character(tr_feat$Case)]
    fit1 <- tryCatch(
      coef(glm(chosen ~ P_task + In_task, data = tr_feat, weights = w1, offset = offset_tr, family = quasibinomial())),
      error = function(e) c(`(Intercept)` = 0, P_task = 0, In_task = 0))
    fit2 <- tryCatch(
      coef(glm(chosen ~ P_task + In_task, data = tr_feat, weights = w2, offset = offset_tr, family = quasibinomial())),
      error = function(e) c(`(Intercept)` = 0, P_task = 0, In_task = 0))
    beta_task <- c(fit1["P_task"], fit2["P_task"])
    beta_intask <- c(fit1["In_task"], fit2["In_task"])
  }
  cat("Final total log-lik:", total_ll, " class2 beta_task:", beta_task[2], " beta_intask:", beta_intask[2], "\n")
  if (total_ll > best_ll) {
    best_ll <- total_ll
    best_fit <- list(beta_task = beta_task, beta_intask = beta_intask, gamma = gamma)
  }
}

cat("\n=== Best EM fit across", n_starts, "starts ===\n")
print(best_fit)
saveRDS(list(base_mod = base_mod, fml_baseline = fml_baseline, best_fit = best_fit,
             ctr = ctr, scl = scl, member_vars = member_vars),
        "data_processed/latent_class_screen_fit.rds")
