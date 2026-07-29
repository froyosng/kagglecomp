train <- read.csv("csv files/train.csv")
test <- read.csv("csv files/test.csv")
cat("train rows", nrow(train), "unique Case", length(unique(train$Case)), "\n")
cat("test rows", nrow(test), "unique Case", length(unique(test$Case)), "\n")
vars <- c("incomea","agea","milesa","nighta")
for (v in vars) {
  cat(v, "NA train:", sum(is.na(train[[v]])), " NA test:", sum(is.na(test[[v]])), "\n")
}
tr1 <- train[!duplicated(train$Case), vars]
te1 <- test[!duplicated(test$Case), vars]
cat("train respondents:", nrow(tr1), " test respondents:", nrow(te1), "\n")
cat("--- train summary ---\n")
print(summary(tr1))
cat("--- test summary ---\n")
print(summary(te1))
cat("--- train quantiles ---\n")
for (v in vars) print(quantile(tr1[[v]], c(0,.1,.25,.5,.75,.9,1), na.rm=TRUE))
cat("--- test quantiles ---\n")
for (v in vars) print(quantile(te1[[v]], c(0,.1,.25,.5,.75,.9,1), na.rm=TRUE))
