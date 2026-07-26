library(tidyverse)
library(mlogit)
source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

split <- readRDS("data_processed/train_val_split.rds")
tr <- split$train_long_tr
va <- split$train_long_val
tr$chid <- paste(tr$Case, tr$Task, sep = "_")
va$chid <- paste(va$Case, va$Task, sep = "_")

fit <- readRDS("data_processed/latent_class_screen_fit.rds")
base_mod <- fit$base_mod
ctr <- fit$ctr; scl <- fit$scl
member_vars <- fit$member_vars
beta_task <- fit$best_fit$beta_task
beta_intask <- fit$best_fit$beta_intask
gamma <- fit$best_fit$gamma

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

tr_feat_for_scale <- make_features(tr, ctr, scl)
resp_tr_for_scale <- tr_feat_for_scale %>% distinct(Case, .keep_all = TRUE)
member_ctr <- sapply(resp_tr_for_scale[member_vars], mean, na.rm = TRUE)
member_scl <- sapply(resp_tr_for_scale[member_vars], sd, na.rm = TRUE)

va_feat <- make_features(va, ctr, scl)
mdat_va <- dfidx(va_feat, idx = list(c("chid","Case"), "alt"), choice = "chosen")
pred_shared_va <- predict(base_mod, newdata = mdat_va)
long_pred_va <- as.data.frame(pred_shared_va) %>%
  rownames_to_column("chid") %>%
  pivot_longer(-chid, names_to = "alt_name", values_to = "p") %>%
  mutate(alt = as.integer(gsub("\\D", "", alt_name)))
va_feat <- va_feat %>% left_join(long_pred_va %>% select(chid, alt, p), by = c("chid","alt"))
offset_va <- log(pmax(va_feat$p, 1e-12))

# class-membership probability for each held-out respondent
resp_va <- va_feat %>% distinct(Case, .keep_all = TRUE)
z_va_member <- sweep(sweep(resp_va[member_vars], 2, member_ctr, "-"), 2, member_scl, "/")
Zmat_va <- as.matrix(cbind(1, z_va_member))
prior2_va <- setNames(1 / (1 + exp(-as.numeric(Zmat_va %*% gamma))), resp_va$Case)

predict_mixture <- function(beta_task, beta_intask, prior2_vec) {
  eta1 <- offset_va + beta_task[1] * va_feat$P_task + beta_intask[1] * va_feat$In_task
  eta2 <- offset_va + beta_task[2] * va_feat$P_task + beta_intask[2] * va_feat$In_task
  d <- data.frame(chid = va_feat$chid, Case = va_feat$Case, alt = va_feat$alt, eta1 = eta1, eta2 = eta2)
  d <- d %>% group_by(chid) %>% mutate(p1 = exp(eta1 - max(eta1)) / sum(exp(eta1 - max(eta1))),
                                        p2 = exp(eta2 - max(eta2)) / sum(exp(eta2 - max(eta2)))) %>% ungroup()
  w2 <- prior2_vec[as.character(d$Case)]
  d$p_mix <- (1 - w2) * d$p1 + w2 * d$p2
  wide <- d %>% select(chid, alt, p_mix) %>% pivot_wider(names_from = alt, values_from = p_mix, names_prefix = "p")
  as.matrix(wide[, c("p1","p2","p3","p4")])
}

pred_lc <- predict_mixture(beta_task, beta_intask, prior2_va)
pred_shared <- predict_mixture(c(0,0), c(0,0), setNames(rep(0, nrow(resp_va)), resp_va$Case))  # shared-only (no class split) for comparison

truth_mat <- va_feat %>% distinct(chid, alt, chosen) %>%
  pivot_wider(names_from = alt, values_from = chosen, names_prefix = "Ch")
wide_order <- va_feat %>% select(chid, alt) %>% distinct(chid) %>% pull(chid)
truth_mat <- truth_mat[match(wide_order, truth_mat$chid), ]

cat("Shared baseline only (no task-fatigue term at all): logloss =",
    round(log_loss(truth_mat[, paste0("Ch",1:4)], pred_shared), 6), "\n")
cat("Full existing m8trpg (with SHARED single P_task/In_task, from earlier session): 1.165734 (reference, single split)\n")
cat("2-class latent mixture on task-fatigue: logloss =",
    round(log_loss(truth_mat[, paste0("Ch",1:4)], pred_lc), 6), "\n")
