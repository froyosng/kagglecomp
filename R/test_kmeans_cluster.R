library(tidyverse)
library(mlogit)

source("R/log_loss.R")

attrs <- c("CC","GN","NS","BU","FA","LD","BZ","FC","FP","RP",
           "PP","KA","SC","TS","NV","MA","LB","AF","HU","Price")
scaler_vars <- c("incomea","agea","milesa","nighta","genderind","Urbind","educind")

split <- readRDS("data_processed/train_val_split.rds")
tr <- split$train_long_tr
va <- split$train_long_val

ctr <- sapply(tr[scaler_vars], mean, na.rm = TRUE)
scl <- sapply(tr[scaler_vars], sd, na.rm = TRUE)
scl[scl == 0] <- 1

# K-means "persona" clustering on respondent-level numeric covariates (income,
# age, miles, night), fit on TRAINING respondents only (one row per Case, not
# per long-format row, so each respondent is weighted once). Zeening's idea
# (branch `zeening`, rf_v2_kmeans), re-tested as a heterogeneity axis for the
# logit rather than a tree-model feature. k=5 to match her choice and mirror
# the existing 5-6 level heterogeneity axes (segment/region/ppark).
cluster_vars <- c("incomea", "agea", "milesa", "nighta")
resp_tr <- tr %>% distinct(Case, .keep_all = TRUE)
resp_z_tr <- sweep(sweep(resp_tr[cluster_vars], 2, ctr[cluster_vars], "-"), 2, scl[cluster_vars], "/")
set.seed(42)
km <- kmeans(resp_z_tr, centers = 5, algorithm = "MacQueen", iter.max = 100)

assign_cluster <- function(df, ctr, scl, centers) {
  resp <- df %>% distinct(Case, .keep_all = TRUE)
  z <- sweep(sweep(resp[cluster_vars], 2, ctr[cluster_vars], "-"), 2, scl[cluster_vars], "/")
  d <- as.matrix(dist(rbind(centers, as.matrix(z))))[-(1:nrow(centers)), 1:nrow(centers)]
  cluster <- apply(d, 1, which.min)
  setNames(cluster, resp$Case)
}
cluster_tr <- assign_cluster(tr, ctr, scl, km$centers)
cluster_va <- assign_cluster(va, ctr, scl, km$centers)  # assigned to TRAINING centroids, not refit
tr$cluster <- cluster_tr[as.character(tr$Case)]
va$cluster <- cluster_va[as.character(va$Case)]
cat("Cluster sizes (train respondents):\n"); print(table(cluster_tr))

make_features <- function(df, ctr, scl) {
  df <- as.data.frame(df)
  df$chid <- paste(df$Case, df$Task, sep = "_")
  df$d2 <- as.integer(df$alt == 2)
  df$d3 <- as.integer(df$alt == 3)
  z <- sweep(sweep(df[scaler_vars], 2, ctr, "-"), 2, scl, "/")

  df$inside <- as.integer(df$alt != 4)
  df$Price_num <- as.numeric(df$Price)
  # Price dummies for levels 2-12 only (reference = level 1, which absorbs
  # opt-out's Price=0 into the same zero-contribution cell). Levels 0 and 1
  # share a cell deliberately: Price=0 occurs only for the opt-out, and the
  # 19-attribute block already encodes "inside vs opt-out" exactly (every
  # inside alt has exactly 9/19 attributes at a non-reference level, a
  # constant of the experimental design), so a full 0:12 price factor
  # duplicates that "inside" indicator via its dummy-sum and produces an
  # exact (rank-1) collinearity with the attribute block. Dropping any one
  # of the 12 price levels' dummies removes the duplication without changing
  # fitted probabilities; levels 2-12 relative to level 1 keeps the whole
  # price curve above the cheapest tier free to estimate.
  for (k in 2:12) {
    df[[paste0("Pr_lvl", k)]] <- as.integer(df$Price_num == k)
  }
  df$Task_c <- (as.numeric(df$Task) - 10) / 9

  df$P_income <- df$Price_num * z$incomea
  df$P_age <- df$Price_num * z$agea
  df$P_miles <- df$Price_num * z$milesa
  df$P_night <- df$Price_num * z$nighta

  df$In_income <- df$inside * z$incomea
  df$In_age <- df$inside * z$agea
  df$In_miles <- df$inside * z$milesa
  df$In_night <- df$inside * z$nighta

  df$In_gender <- df$inside * z$genderind
  df$In_urb <- df$inside * z$Urbind
  df$In_educ <- df$inside * z$educind

  for (s in 2:6) {
    df[[paste0("P_seg", s)]] <- df$Price_num * (df$segmentind == s)
    df[[paste0("In_seg", s)]] <- df$inside * (df$segmentind == s)
  }

  df$P_task <- df$Price_num * df$Task_c
  df$In_task <- df$inside * df$Task_c

  for (s in 2:5) {
    df[[paste0("P_region", s)]] <- df$Price_num * (df$regionind == s)
    df[[paste0("In_region", s)]] <- df$inside * (df$regionind == s)
    df[[paste0("P_ppark", s)]] <- df$Price_num * (df$pparkind == s)
    df[[paste0("In_ppark", s)]] <- df$inside * (df$pparkind == s)
  }

  task_stats <- df %>%
    group_by(chid) %>%
    summarise(
      price_min = min(Price_num[inside == 1], na.rm = TRUE),
      price_max = max(Price_num[inside == 1], na.rm = TRUE),
      .groups = "drop"
    )
  df <- left_join(df, task_stats, by = "chid")
  df$is_cheapest <- as.integer(df$inside == 1 & df$Price_num == df$price_min)
  df$is_dearest <- as.integer(df$inside == 1 & df$Price_num == df$price_max)
  # magnitude versions alongside the rank flags: how far above the cheapest /
  # below the dearest this alternative's price sits (0 for inside alts at the
  # relevant extreme; 0 for the opt-out, which has no price_min/max context)
  df$price_gap_min <- ifelse(df$inside == 1, df$Price_num - df$price_min, 0)
  df$price_gap_max <- ifelse(df$inside == 1, df$price_max - df$Price_num, 0)

  for (c in 2:5) {
    df[[paste0("P_clust", c)]] <- df$Price_num * (df$cluster == c)
    df[[paste0("In_clust", c)]] <- df$inside * (df$cluster == c)
  }

  df
}

tr_feat <- make_features(tr, ctr, scl)
va_feat <- make_features(va, ctr, scl)

attr_terms <- paste0("factor(", attrs[attrs != "Price"], ")")
int_terms <- c(
  "P_income", "P_age", "P_miles", "P_night",
  "In_income", "In_age", "In_miles", "In_night",
  "In_gender", "In_urb", "In_educ",
  paste0("P_seg", 2:6), paste0("In_seg", 2:6),
  "P_task", "In_task",
  paste0("P_region", 2:5), paste0("In_region", 2:5),
  paste0("P_ppark", 2:5), paste0("In_ppark", 2:5),
  "is_cheapest", "is_dearest", "price_gap_min", "price_gap_max",
  paste0("P_clust", 2:5), paste0("In_clust", 2:5)
)

price_terms <- paste0("Pr_lvl", 2:12)

fml <- as.formula(paste(
  "chosen ~",
  paste(c(attr_terms, price_terms, "d2", "d3", int_terms), collapse = " + "),
  "| 0"
))

mdat_tr <- dfidx(tr_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")
mdat_va <- dfidx(va_feat, idx = list(c("chid", "Case"), "alt"), choice = "chosen")

mod <- mlogit(fml, data = mdat_tr)

truth_mat <- va_feat %>%
  select(chid, alt, chosen) %>%
  pivot_wider(names_from = alt, values_from = chosen, names_prefix = "Ch") %>%
  arrange(match(chid, rownames(predict(mod, newdata = mdat_va))))

pred_va <- predict(mod, newdata = mdat_va)
ll <- log_loss(truth_mat[, paste0("Ch", 1:4)], pred_va[truth_mat$chid, ])

cat("Validation log loss:", round(ll, 6), "\n")
print(summary(mod)$CoefTable[grep("P_clust|In_clust", rownames(summary(mod)$CoefTable), value = TRUE), c("Estimate", "Pr(>|z|)")])
