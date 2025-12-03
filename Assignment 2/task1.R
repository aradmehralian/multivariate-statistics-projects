library(MASS)
library(class)
library(nnet)
library(xgboost)
library(HDclassif)
library(tidyverse)


load("Data/task1.Rdata")

# part a

## PCA

# scenario 1
pca_s1 <- prcomp(train.data.s1, center = T,scale. = F)
var_prop_s1 <- cumsum(pca_s1$sdev^2) / sum(pca_s1$sdev^2)
#number of components which explain 90% variance
num_comp_s1 <- which(var_prop_s1 >= 0.9)[1]
cat("Number of PCs for S1:", num_comp_s1, "\n")

# scenario 2
pca_s2 <- prcomp(train.data.s2, center = T, scale. = F)
var_prop_s2 <- cumsum(pca_s2$sdev^2) / sum(pca_s2$sdev^2)
num_comp_s2 <- which(var_prop_s2 >= 0.9)[1]
cat("Number of PCs for S2:", num_comp_s2, "\n")

plot(var_prop_s1, type = "b", pch = 15,
     xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance",
     main = "Cumulative Variance (S1)")
abline(h = 0.9, col = "blue", lty = 2, lwd = 2)  # 90% threshold

plot(var_prop_s2, type = "b", pch = 15,
     xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance",
     main = "Cumulative Variance (S2)")
abline(h = 0.9, col = "blue", lty = 2, lwd = 2)  # 90% threshold

# Project training and test data onto selected PCs
train_pc_s1 <- pca_s1$x[, 1:num_comp_s1]
train_pc_s2 <- pca_s2$x[, 1:num_comp_s2]

test_pc_s1 <- scale(test.data, center = pca_s1$center, scale = F) %*% 
  pca_s1$rotation[, 1:num_comp_s1]

test_pc_s2 <- scale(test.data, center = pca_s2$center, scale = F) %*% 
  pca_s2$rotation[, 1:num_comp_s2]

# we can use Wilks' Lambda to see whether our data is suitable for algorithms
# such as LDA or QDA.

# Wilks' Lambda for S1
wilk_s1 <- manova(as.matrix(train_pc_s1) ~ train.target.s1)
summary(wilk_s1, test = "Wilks")

#Wilks' Lambda for S2 
wilk_s2 <- manova(as.matrix(train_pc_s2) ~ train.target.s2)
summary(wilk_s2, test = "Wilks")

# Wilks’ Lambda equals 0.03 for both scenarios, which is very close to 0,
# indicating strong differences between group centroids as expected.
# p-value small enough to reject the null.
# hence, LDA and QDA are meaningful due to real separation in data.

# part b

## LDA

# scenario 1
lda_s1 <- lda(train_pc_s1, grouping = train.target.s1)
print(lda_s1)
lda_train_pred_s1 <- predict(lda_s1)$class

# Confusion matrix 
table(lda_train_pred_s1, train.target.s1)
# Most errors between visually similar letters (D and O), (G and Q)

# Posterior probabilities
lda_train_post_s1 <- predict(lda_s1)$posterior

# most rows showcase high confidence predictions meaning LDA works decently
# most low confidence predictions between visually similar letters 
# for example row 135 D and Q.

# training error
lda_train_err_s1 <- mean(lda_train_pred_s1 != train.target.s1)

# Predictions
lda_test_pred_s1 <- predict(lda_s1, newdata = test_pc_s1)$class
table(lda_test_pred_s1, test.target)
# predictions error
lda_test_err_s1 <- mean(lda_test_pred_s1 != test.target)

cat("LDA Training Error on s1:", round(lda_train_err_s1, 3), "\n")
cat("LDA Test Error on s1:", round(lda_test_err_s1, 3), "\n")

# scenario 2
lda_s2 <- lda(train_pc_s2, grouping = train.target.s2)
print(lda_s2)

lda_train_pred_s2 <- predict(lda_s2)$class
table(lda_train_pred_s2, train.target.s2)

lda_train_err_s2 <- mean(lda_train_pred_s2 != train.target.s2)

lda_test_pred_s2 <- predict(lda_s2, newdata = test_pc_s2)$class
lda_test_err_s2 <- mean(lda_test_pred_s2 != test.target)

cat("LDA Training Error on s2:", round(lda_train_err_s2, 3), "\n")
cat("LDA Test Error on S2:", round(lda_test_err_s2, 3), "\n")

# mild over fitting, particularly in S2 due to smaller training set
# linear separation works fairly well but cannot capture complex boundaries

## QDA

# scenario 1
qda_s1 <- qda(train_pc_s1, grouping = train.target.s1)
print(qda_s1)
# group means identical to LDA since they work in similar ways,
# but cater for group covariances unlike LDA

qda_train_pred_s1 <- predict(qda_s1)$class
table(qda_train_pred_s1, train.target.s1)
qda_train_err_s1 <- mean(qda_train_pred_s1 != train.target.s1)

# Most errors between D and O and between G and Q,
# but does better than LDA with quadratic decision boundary
qda_test_pred_s1 <- predict(qda_s1, newdata = test_pc_s1)$class
qda_test_err_s1 <- mean(qda_test_pred_s1 != test.target)

cat("QDA Training Error on S1:", round(qda_train_err_s1, 3), "\n")
cat("QDA Test Error on S1:", round(qda_test_err_s1, 3), "\n")

# scenario 2
qda_s2 <- qda(train_pc_s2, grouping = train.target.s2)
print(qda_s2)

qda_train_pred_s2 <- predict(qda_s2)$class
table(qda_train_pred_s2, train.target.s2)
qda_train_err_s2 <- mean(qda_train_pred_s2 != train.target.s2)

qda_test_pred_s2 <- predict(qda_s2, newdata = test_pc_s2)$class
qda_test_err_s2 <- mean(qda_test_pred_s2 != test.target)

cat("QDA Training Error on S2:", round(qda_train_err_s2, 3), "\n")
cat("QDA Test Error on S2:", round(qda_test_err_s2, 3), "\n")

# low training error - much lower than LDA around 4.6% for S1 and 2.6% for S2
# lower test error than LDA as well at 5.8% for S1 and 6.8% for S2
# QDA fits the data with more flexibility thanks to the quadratic decision boundaries, 
# and generalizes better. QDA outperforms LDA 

# KNN

# scenario 1
# Range of k for hyper tuning
k_vals <- seq(1, 50)  
num_k <- length(k_vals)

train_err <- numeric(num_k)
test_err  <- numeric(num_k)

# Loop over k
for (i in seq_along(k_vals)) {
  k <- k_vals[i]
  
  train_pred <- knn(train = train_pc_s1, test = train_pc_s1,
                    cl = train.target.s1, k = k)
  train_err[i] <- mean(train_pred != train.target.s1)
  
  test_pred <- knn(train = train_pc_s1, test = test_pc_s1,
                   cl = train.target.s1, k = k)
  test_err[i] <- mean(test_pred != test.target)
}

# Identify best k based on minimum test error
best_k_val_s1 <- k_vals[which.min(test_err)]
cat("Best k for S1:", best_k_val_s1, "Test Error:", min(test_err), "\n")
cat("Train Error at best k:", round(train_err[which.min(test_err)], 3), "\n")

plot(k_vals, train_err, type="b", col="blue", pch=19, 
     ylim=c(0, max(train_err, test_err)),
     xlab="k value", ylab="Error Rate", main="KNN Errors vs k")
lines(k_vals, test_err, type="b", col="red", pch=19)
legend("topright", legend=c("Train","Test"), col=c("blue","red"), pch=19)

# Train and test predictions using best k

train_pred_knn_s1 <- knn(train = train_pc_s1, test = train_pc_s1,
                         cl = train.target.s1, k = best_k_val_s1)

test_pred_knn_s1 <- knn(train = train_pc_s1, test = test_pc_s1,
                    cl = train.target.s1, k = best_k_val_s1)

knn_train_err_s1 <- mean(train_pred_knn_s1 != train.target.s1)
knn_test_err_s1  <- mean(test_pred_knn_s1 != test.target)

cat("Train Error on S1:", round(knn_train_err_s1, 3), "\n")
cat("Test Error on S1:", round(knn_test_err_s1, 3), "\n")

# scenario 2
k_vals <- seq(1, 50)  
num_k <- length(k_vals)

train_err <- numeric(num_k)
test_err  <- numeric(num_k)

# Loop over k
for (i in seq_along(k_vals)) {
  k <- k_vals[i]
  
  train_pred <- knn(train = train_pc_s2, test = train_pc_s2,
                    cl = train.target.s2, k = k)
  train_err[i] <- mean(train_pred != train.target.s2)
  
  test_pred <- knn(train = train_pc_s2, test = test_pc_s2,
                   cl = train.target.s2, k = k)
  test_err[i] <- mean(test_pred != test.target)
}

# Identify best k based on minimum test error
best_k_val_s2 <- k_vals[which.min(test_err)]
cat("Best k for S2:", best_k_val_s2, "Test Error:", min(test_err), "\n")
cat("Train Error at best k:", round(train_err[which.min(test_err)], 3), "\n")

plot(k_vals, train_err, type="b", col="blue", pch=19,
     ylim=c(0,max(train_err, test_err)),
     xlab="k value", ylab="Error Rate", main="KNN Errors vs k")
lines(k_vals, test_err, type="b", col="red", pch=19)
legend("topright", legend=c("Train","Test"), col=c("blue","red"), pch=19)


train_pred_s2 <- knn(train = train_pc_s2, test = train_pc_s2,
                     cl = train.target.s2, k = best_k_val_s2)
test_pred_s2 <- knn(train = train_pc_s2, test = test_pc_s2,
                    cl = train.target.s2, k = best_k_val_s2)

knn_train_err_s2 <- mean(train_pred_s2 != train.target.s2)
knn_test_err_s2  <- mean(test_pred_s2 != test.target)

cat("Train Error on S2:", round(knn_train_err_s2, 3), "\n")
cat("Test Error on S2:", round(knn_test_err_s2, 3), "\n")


## Multinomial Regression

# scenario 1
mul_s1 <- multinom(
  train.target.s1 ~ .,
  data = as.data.frame(train_pc_s1),
  family = multinomial,
  maxit = 1000,
  hess = TRUE
)

train_pred_s1 <- predict(mul_s1, newdata = as.data.frame(train_pc_s1))
test_pred_s1 <- predict(mul_s1, newdata = as.data.frame(test_pc_s1))

mlr_train_err_s1 <- mean(train_pred_s1 != train.target.s1)
mlr_test_err_s1 <- mean(test_pred_s1 != test.target)

cat("Train Error on S1:", round(mlr_train_err_s1, 3), "\n")
cat("Test Error on S1:", round(mlr_test_err_s1, 3), "\n")

# scenario 2
mul_s2 <- multinom(
  train.target.s2 ~ .,
  data = as.data.frame(train_pc_s2),
  family = multinomial,
  maxit = 1000,
  hess = TRUE
)
train_pred_s2 <- predict(mul_s2, newdata = as.data.frame(train_pc_s2))
test_pred_s2 <- predict(mul_s2, newdata = as.data.frame(test_pc_s2))

mlr_train_err_s2 <- mean(train_pred_s2 != train.target.s2)
mlr_test_err_s2 <- mean(test_pred_s2 != test.target)

cat("Train Error on S2:", round(mlr_train_err_s2, 3), "\n")
cat("Test Error on S2:", round(mlr_test_err_s2, 3), "\n")

# maybe the relationship between principal components and labels is non-linear
# and MLR is linear in the log-odds.

# with a smaller training set, training error decreases to around 5.2%
# but test error increases to 16.7%
# probable under-fitting making the model too simple for the complex decision boundaries


## squaring terms

# scenario 1
train_pc_s1_sq <- as.data.frame(train_pc_s1)
test_pc_s1_sq <- as.data.frame(test_pc_s1)

# squared PCs
sqr_train_s1 <- train_pc_s1_sq^2
colnames(sqr_train_s1) <- paste0(colnames(train_pc_s1_sq), "_sq")

sqr_test_s1 <- test_pc_s1_sq^2
colnames(sqr_test_s1) <- paste0(colnames(test_pc_s1_sq), "_sq")

# Combine original + squared PCs
train_pc_s1_sq <- cbind(train_pc_s1_sq, sqr_train_s1)
test_pc_s1_sq <- cbind(test_pc_s1_sq, sqr_test_s1)

# Scenario 2
train_pc_s2_sq <- as.data.frame(train_pc_s2)
test_pc_s2_sq <- as.data.frame(test_pc_s2)

# Squared PCs
sqr_train_s2 <- train_pc_s2_sq^2
colnames(sqr_train_s2) <- paste0(colnames(train_pc_s2_sq), "_sq")

sqr_test_s2 <- test_pc_s2_sq^2
colnames(sqr_test_s2) <- paste0(colnames(test_pc_s2_sq), "_sq")

# Combine
train_pc_s2_sq <- cbind(train_pc_s2_sq, sqr_train_s2)
test_pc_s2_sq <- cbind(test_pc_s2_sq, sqr_test_s2)


# Scenario 1
mul_sq_s1 <- multinom(
    train.target.s1 ~ .,
    data = train_pc_s1_sq,
    family = multinomial,
    maxit = 1000,
    hess = T
)
train_pred_sq_s1 <- predict(mul_sq_s1, newdata = train_pc_s1_sq)
test_pred_sq_s1 <- predict(mul_sq_s1, newdata = test_pc_s1_sq)

mlr_sq_train_err_s1 <- mean(train_pred_sq_s1 != train.target.s1)
mlr_sq_test_err_s1 <- mean(test_pred_sq_s1 != test.target)

# Scenario 2
mul_sq_s2 <- multinom(
  train.target.s2 ~ .,
  data = train_pc_s2_sq,
  family = multinomial,
  maxit = 1000,
  hess = T
)
train_pred_sq_s2 <- predict(mul_sq_s2, newdata = train_pc_s2_sq)
test_pred_sq_s2 <- predict(mul_sq_s2, newdata = test_pc_s2_sq)

mlr_sq_train_err_s2 <- mean(train_pred_sq_s2 != train.target.s2)
mlr_sq_test_err_s2 <- mean(test_pred_sq_s2 != test.target)

cat("(PC + PC^2) Train Error on S1:", round(mlr_sq_train_err_s1, 3), "\n")
cat("(PC + PC^2) Test Error on S1:", round(mlr_sq_test_err_s1, 3), "\n")
cat("(PC + PC^2) Train Error on S2:", round(mlr_sq_train_err_s2, 3), "\n")
cat("(PC + PC^2) Test Error on S2:", round(mlr_sq_test_err_s2, 3), "\n")


para_s1 <- coef(mul_sq_s1)
para_s2 <- coef(mul_sq_s2)
n_sam_s1 <- length(train.target.s1)
n_sam_s2 <- length(train.target.s2)

cat("S1: K-1 classes =", nrow(para_s1),
    " Vars =", ncol(para_s1),
    " Parameters =", length(para_s1),
    " Training samples =", n_sam_s1, "\n")

cat("S2: K-1 classes =", nrow(para_s2),
    " Vars =", ncol(para_s2),
    " Parameters =", length(para_s2),
    " Training samples =", n_sam_s2, "\n")



# MLR on PCs alone had:
#  S1 → Train 17%, Test 18.8%   vs 23.8% and 24.3% Squared
#  S2 → Train 12.4%, Test 18.4% vs 22.5% and 27.3% Squared

# Adding squared PCs increased both training and test errors significantly maybe because of overparameterization?
# S1-417 variables for only 9600 samples
# S2-393 variables for 1600 samples 
# slight overfitting and poor generalization due to too many correlated predictors/ unstable 
# smaller training set suffers more

## gradient boosting

# gradient boosting requires the labels to be in numeric form and zero-indexed.
# scenario 1
train_lab_num_s1 <- as.numeric(train.target.s1) - 1
test_lab_num  <- as.numeric(test.target) - 1

# optimized data structures for xgboost
dtrain_s1 <- xgb.DMatrix(data = as.matrix(train_pc_s1), label = train_lab_num_s1)
dtest_s1  <- xgb.DMatrix(data = as.matrix(test_pc_s1),  label = test_lab_num)

params_s1 <- list(
  booster          = "gbtree",
  objective        = "multi:softmax",
  num_class        = 4,
  eta              = 0.1,
  max_depth        = 6,
  eval_metric      = "merror"
)

# cross validation to find the best number of rounds to run the algorithm
set.seed(7)
xgb_cv_s1 <- xgb.cv(
  params  = params_s1,
  data    = dtrain_s1,
  nrounds = 100,
  nfold   = 5,
  early_stopping_rounds = 10,
  verbose = 0
)

best_nrounds_s1 <- xgb_cv_s1$best_iteration
cat("Best nrounds for S1:", best_nrounds_s1)

xgb_s1 <- xgb.train(
  params = params_s1,
  data = dtrain_s1,
  nrounds = best_nrounds_s1,
  subsample = 0.8,
  callsample_bytree = 0.9
)

train_pred_xgb_s1 <- predict(xgb_s1, dtrain_s1)
xgb_train_err_s1  <- mean(train_pred_xgb_s1 != train_lab_num_s1)

test_pred_xgb_s1 <- predict(xgb_s1, dtest_s1)
xgb_test_err_s1 <- mean(test_pred_xgb_s1 != test_lab_num)

cat("XGBoost Train Error on S1:", round(xgb_train_err_s1, 3), "\n")
cat("XGBoost Test Error on S1:",  round(xgb_test_err_s1, 3))

# there is a huge difference between train and test errors. possible overfitting
# maybe tweak the parameters to stop overfitting?

# scenario 2
train_lab_num_s2 <- as.numeric(train.target.s2) - 1

dtrain_s2 <- xgb.DMatrix(data = as.matrix(train_pc_s2), label = train_lab_num_s2)
dtest_s2  <- xgb.DMatrix(data = as.matrix(test_pc_s2),  label = test_lab_num)

params_s2 <- list(
  booster          = "gbtree",
  objective        = "multi:softmax",
  num_class        = 4,
  eta              = 0.1,
  max_depth        = 4, # since s2 is a smaller data set
  eval_metric      = "merror"
)

set.seed(10)
xgb_cv_s2 <- xgb.cv(
  params  = params_s2,
  data    = dtrain_s2,
  nrounds = 100,
  nfold   = 5,
  early_stopping_rounds = 10,
  verbose = 0
)

best_nrounds_s2 <- xgb_cv_s2$best_iteration
cat("Best nrounds forS2:", best_nrounds_s2)

xgb_s2 <- xgb.train(
  params  = params_s2,
  data    = dtrain_s2,
  nrounds = best_nrounds_s2,
  subsample = 0.8,
  callsample_bytree = 0.9
)

train_pred_xgb_s2 <- predict(xgb_s2, dtrain_s2)
xgb_train_err_s2  <- mean(train_pred_xgb_s2 != train_lab_num_s2)

test_pred_xgb_s2 <- predict(xgb_s2, dtest_s2)
xgb_test_err_s2 <- mean(test_pred_xgb_s2 != test_lab_num)

cat("XGBoost Train Error on S2:", round(xgb_train_err_s2, 3), "\n")
cat("XGBoost Test Error on S2:",  round(xgb_test_err_s2, 3))

# there is also overfitting in case of s2

## HDDA

# scenario 1
# our train data is centered. so we also have to center our test data.
# no need to use the principal components, since HDDA does its own dimension reduction

s1_train_centered <- scale(train.data.s1, center = T, scale = F)
train_means_s1 <- colMeans(train.data.s1)
test_centered_s1 <- scale(test.data, center = train_means_s1, scale = F)

hdda_s1 <- hdda(
  data      = s1_train_centered,
  cls       = train.target.s1,
  model     = "AKJBKQKD",
  d         = "Cattell",
  threshold = 0.05,
  scaling   = F
)

print(hdda_s1)

train_pred_hdda_s1 <- predict(hdda_s1, s1_train_centered, cls = train.target.s1)
hdda_train_err_s1 <- mean(train_pred_hdda_s1$class != train.target.s1)

test_pred_hdda_s1 <- predict(hdda_s1, test_centered_s1, cls = test.target)
hdda_test_err_s1 <- mean(test_pred_hdda_s1$class != test.target)

cat("HDDA Train Error on S1:", round(hdda_train_err_s1, 3), "\n")
cat("HDDA Test Error on S1:",  round(hdda_test_err_s1, 3))

# no sign of overfitting, train and test error rates are close and below %10

# scenario 2
s2_train_centered <- scale(train.data.s2, center = T, scale = F)
train_means_s2 <- colMeans(train.data.s2)
test_centered_s2 <- scale(test.data, center = train_means_s2, scale = F)

hdda_s2 <- hdda(
  data      = s2_train_centered,
  cls       = train.target.s2,
  model     = "AKJBKQKD",
  d         = "Cattell",
  threshold = 0.05,
  scaling   = F
)

print(hdda_s2)

train_pred_hdda_s2 <- predict(hdda_s2, s2_train_centered, cls = train.target.s2)
hdda_train_err_s2 <- mean(train_pred_hdda_s2$class != train.target.s2)

test_pred_hdda_s2 <- predict(hdda_s2, test_centered_s2, cls = test.target)
hdda_test_err_s2 <- mean(test_pred_hdda_s2$class != test.target)

cat("HDDA Train Error on S2:", round(hdda_train_err_s2, 3), "\n")
cat("HDDA Test Error on S2:",  round(hdda_test_err_s2, 3))

# lower rate of error on train data, but test error rate remains almost the same

# part c
model_names <- rep(
  c(
    "LDA",
    "QDA",
    "KNN",
    "Multinomial LogReg",
    "Multinomial LogReg (Squared)",
    "Gradient Boosting",
    "HDDA"
  ),
  each = 2
)

train_errors <- round(c(lda_train_err_s1, lda_train_err_s2,
                  qda_train_err_s1, qda_train_err_s2,
                  knn_train_err_s1, knn_train_err_s2,
                  mlr_train_err_s1, mlr_train_err_s2,
                  mlr_sq_train_err_s1, mlr_sq_train_err_s2,
                  xgb_train_err_s1, xgb_train_err_s2,
                  hdda_train_err_s1, hdda_train_err_s2), 3)

test_errors <- round(c(lda_test_err_s1, lda_test_err_s2,
                 qda_test_err_s1, qda_test_err_s2,
                 knn_test_err_s1, knn_test_err_s2,
                 mlr_test_err_s1, mlr_test_err_s2,
                 mlr_sq_test_err_s1, mlr_sq_test_err_s2,
                 xgb_test_err_s1, xgb_test_err_s2,
                 hdda_test_err_s1, hdda_test_err_s2), 3)

results <- data.frame(
  Model = model_names,
  Scenario = rep(c("Scenario 1", "Scenario 2"), each = 1),
  Train_Error = train_errors,
  Test_Error = test_errors
)

print(results)

# plotting the error rates

results_long <-results %>%
  pivot_longer(
    cols = c("Train_Error", "Test_Error"),
    names_to = "Error_Type",
    values_to = "Error_Value"
  )

results_long$Error_Type <- factor(results_long$Error_Type, 
                                  levels = c("Train_Error", "Test_Error"))

ggplot(results_long, aes(x = Model, y = Error_Value, fill = Scenario)) +
  geom_bar(stat="identity", position = "dodge") +
  facet_wrap(~Error_Type) +
  labs(
    title = "Overview of model performance",
    y = "Error Rate",
    x = "Method"
  ) +
  theme_bw()+
  theme(axis.text.x = element_text(hjust = 1, angle = 45),
        legend.position = "top") +
  scale_fill_manual(name = "", values = c("#52BDEC", "#00407A"))