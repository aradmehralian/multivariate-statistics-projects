library(MASS)       
library(class)     
library(nnet)       
library(xgboost)    
library(HDclassif)  
library(tidyverse)  

load("task1.Rdata")

#S1 PCA
pca_s1 <- prcomp(train.data.s1, center = TRUE, scale. = FALSE)
var_s1 <- cumsum(pca_s1$sdev^2) / sum(pca_s1$sdev^2)
no_comp_s1 <- which(var_s1 >= 0.9)[1]  #number of components which explain 90% variance
cat("S1: Number of PCs ", no_comp_s1, "\n")

#S2 PCA
pca_s2 <- prcomp(train.data.s2, center = TRUE, scale. = FALSE)
var_s2 <- cumsum(pca_s2$sdev^2) / sum(pca_s2$sdev^2)
no_comp_s2 <- which(var_s2 >= 0.9)[1]
cat("S2: Number of PCs ", no_comp_s2, "\n")

plot(var_s1, type = "b", pch = 15,
     xlab = "No Principal Components",
     ylab = "Cum Prop of Variance",
     main = "Cum Variance - S1")
abline(h = 0.9, col = "blue", lty = 2)  # 90% threshold

plot(var_s2, type = "b", pch = 15,
     xlab = "No Principal Components",
     ylab = "Cum Prop of Variance",
     main = "Cum Variance - S2")
abline(h = 0.9, col = "blue", lty = 2)  # 90% threshold

# Project training and test data onto selected PCs
train.pc.s1 <- pca_s1$x[, 1:no_comp_s1]
train.pc.s2 <- pca_s2$x[, 1:no_comp_s2]

test.pc.s1 <- scale(test.data, center = pca_s1$center, scale = FALSE) %*% pca_s1$rotation[, 1:no_comp_s1]
test.pc.s2 <- scale(test.data, center = pca_s2$center, scale = FALSE) %*% pca_s2$rotation[, 1:no_comp_s2]

#Wilks' Lambda for S1
wilk_s1_pc <- manova(as.matrix(train.pc.s1) ~ train.target.s1)
summary(wilk_s1_pc, test = "Wilks")

#Wilks' Lambda for S2 
wilk_s2_pc <- manova(as.matrix(train.pc.s2) ~ train.target.s2)
summary(wilk_s2_pc, test = "Wilks")

#Wilks’ Lambda (0.03) which is very close to 0 indicating strong differences between group centroids as expected
# p-value small enough to reject the null
# hence, LDA and QDA are meaningful due to real separation in data.

# LDA
# -----------------------
# Scenario 1: LDA
# -----------------------
lda_s1 <- lda(train.pc.s1, grouping = train.target.s1)
print(lda_s1)

#all classes are equally likely -- balanced set
#LD1 contributes most to separation as it helps in explaining most of the trace(between-class variance)

lda_trn_pred_s1 <- predict(lda_s1)$class
#Confusion matrix 
table(lda_trn_pred_s1, train.target.s1)
#Most errors between visually similar letters D to O and G tp Q
# Posterior probabilities
lda_trn_post_s1 <- predict(lda_s1)$posterior

#most rows showcase high confidence predictions meaning LDA works decently
#most low confidence preds between visually similar letters (row 135 D and Q)

# training error
lda_trn_err_s1 <- mean(lda_trn_pred_s1 != train.target.s1)

# Predictons
lda_tst_pred_s1 <- predict(lda_s1, newdata = test.pc.s1)$class
lda_tst_err_s1 <- mean(lda_tst_pred_s1 != test.target)

cat("S1 - LDA Training Error ", lda_trn_err_s1, "\n")
cat("S1 - LDA Test Error ", lda_tst_err_s1, "\n")

# S2: LDA
lda_s2 <- lda(train.pc.s2, grouping = train.target.s2)
print(lda_s2)

lda_trn_pred_s2 <- predict(lda_s2)$class
table(lda_trn_pred_s2, train.target.s2)
lda_trn_err_s2 <- mean(lda_trn_pred_s2 != train.target.s2)

lda_tst_pred_s2 <- predict(lda_s2, newdata = test.pc.s2)$class
lda_tst_err_s2 <- mean(lda_tst_pred_s2 != test.target)

cat("S2 - LDA Training Error:", lda_trn_err_s2, "\n")
cat("S2 - LDA Test Error:", lda_tst_err_s2, "\n")

#training error around 8 to 9% — reasonably good fit
#test error higher at around 11% for S1 and 13% for S2
#mild overfitting, particularly in S2 due to smaller training set
#linear separation works fairly well but cannot capture complex boundaries

# QDA
qda_s1 <- qda(train.pc.s1, grouping = train.target.s1)
print(qda_s1)
#group means identical to LDA since they work in similar ways but cater for group covariances unlike LDA

qda_trn_pred_s1 <- predict(qda_s1)$class
table(qda_trn_pred_s1, train.target.s1)
qda_trn_err_s1 <- mean(qda_trn_pred_s1 != train.target.s1)

#Most errors between D and O and between G and Q, but does better than LDA with quadratic decision boundary
qda_tst_pred_s1 <- predict(qda_s1, newdata = test.pc.s1)$class
qda_tst_err_s1 <- mean(qda_tst_pred_s1 != test.target)

cat("S1 - QDA Training Error", qda_trn_err_s1, "\n")
cat("S1 - QDA Test Error", qda_tst_err_s1, "\n")

# S2: QDA
qda_s2 <- qda(train.pc.s2, grouping = train.target.s2)
print(qda_s2)

qda_trn_pred_s2 <- predict(qda_s2)$class
table(qda_trn_pred_s2, train.target.s2)
qda_trn_err_s2 <- mean(qda_trn_pred_s2 != train.target.s2)

qda_tst_pred_s2 <- predict(qda_s2, newdata = test.pc.s2)$class
qda_tst_err_s2 <- mean(qda_tst_pred_s2 != test.target)

cat("S2 - QDA Training Error ", qda_trn_err_s2, "\n")
cat("S2 - QDA Test Error ", qda_tst_err_s2, "\n")

#low training error - much lower than LDA around 4.6% for S1 and 2.6% for S2
#lower test error than LDA as well at 5.8% for S1 and 6.8% for S2
#QDA fits the data with more flexibility thanks to the quadratic decision boundaries and generalizes better 
#QDA outperforms LDA 

#KNN

# Range of k 
k_vals <- seq(1, 50)  
no_k <- length(k_vals)

train_err <- numeric(no_k)
test_err  <- numeric(no_k)

# Loop over k
for (i in seq_along(k_vals)) {
  k <- k_vals[i]
  
  train_pred <- knn(train = train.pc.s1, test = train.pc.s1,
                    cl = train.target.s1, k = k)
  train_err[i] <- mean(train_pred != train.target.s1)
  
  test_pred <- knn(train = train.pc.s1, test = test.pc.s1,
                   cl = train.target.s1, k = k)
  test_err[i] <- mean(test_pred != test.target)
}

# Identify best k based on minimum test error
best_k_val_s1 <- k_vals[which.min(test_err)]
cat("Best k ", best_k_val_s1, "Test Error:", min(test_err), "\n")
cat("Train Error at best k:", train_err[which.min(test_err)], "\n")

plot(k_vals, train_err, type="b", col="blue", pch=19, ylim=c(0, max(train_err, test_err)),
     xlab="k value", ylab="Error Rate", main="KNN Errors vs k")
lines(k_vals, test_err, type="b", col="red", pch=19)
legend("topright", legend=c("Train","Test"), col=c("blue","red"), pch=10)

# Train and test predictions using best k
train_pred_s1 <- knn(train = train.pc.s1, test = train.pc.s1,
                     cl = train.target.s1, k = 4)
test_pred_s1 <- knn(train = train.pc.s1, test = test.pc.s1,
                    cl = train.target.s1, k = 4)

knn_trn_err_s1 <- mean(train_pred_s1 != train.target.s1)
knn_tst_err_s1  <- mean(test_pred_s1 != test.target)

cat("S1 - Train Error:", knn_trn_err_s1, "\n")
cat("S1 - Test Error:", knn_tst_err_s1, "\n")

# Scenario 2
k_vals <- seq(1, 50)  
no_k <- length(k_vals)

train_err <- numeric(no_k)
test_err  <- numeric(no_k)

# Loop over k
for (i in seq_along(k_vals)) {
  k <- k_vals[i]
  
  train_pred <- knn(train = train.pc.s2, test = train.pc.s2,
                    cl = train.target.s2, k = k)
  train_err[i] <- mean(train_pred != train.target.s2)
  
  test_pred <- knn(train = train.pc.s2, test = test.pc.s2,
                   cl = train.target.s2, k = k)
  test_err[i] <- mean(test_pred != test.target)
}

# Identify best k based on minimum test error
best_k_val_s2 <- k_vals[which.min(test_err)]
cat("Best k ", best_k_val_s2, "Test Error:", min(test_err), "\n")
cat("Train Error at best k:", train_err[which.min(test_err)], "\n")

plot(k_vals, train_err, type="b", col="blue", pch=19, ylim=c(0, max(train_err, test_err)),
     xlab="k value", ylab="Error Rate", main="KNN Errors vs k")
lines(k_vals, test_err, type="b", col="red", pch=19)
legend("topright", legend=c("Train","Test"), col=c("blue","red"), pch=10)


train_pred_s2 <- knn(train = train.pc.s2, test = train.pc.s2,
                     cl = train.target.s2, k = best_k_val_s2)
test_pred_s2 <- knn(train = train.pc.s2, test = test.pc.s2,
                    cl = train.target.s2, k = best_k_val_s2)

knn_trn_err_s2 <- mean(train_pred_s2 != train.target.s2)
knn_tst_err_s2  <- mean(test_pred_s2 != test.target)

cat("S2 - Train Error:", knn_trn_err_s2, "\n")
cat("S2 - Test Error:", knn_tst_err_s2, "\n")

#With larger training set, KNN has 0 training error and test error of 7.6% indicating over-fitting
#With smaller training set, KNN has 6.4% training error and 12.7% test error indicating worse fit
#KNN needs larger datasets 

#Multinomial Regression
mul_s1 <- multinom(train.target.s1 ~ ., data = as.data.frame(train.pc.s1),family=multinomial,maxit=1000,hess=TRUE)
coef_s1 <- coef(mul_s1)

trn_pred_s1 <- predict(mul_s1, newdata = as.data.frame(train.pc.s1))
tst_pred_s1 <- predict(mul_s1, newdata = as.data.frame(test.pc.s1))

mlr_trn_err_s1 <- mean(trn_pred_s1 != train.target.s1)
mlr_tst_err_s1 <- mean(tst_pred_s1 != test.target)

# S2
mul_s2 <- multinom(train.target.s2 ~ ., data = as.data.frame(train.pc.s2),family=multinomial,maxit=1000,hess=TRUE)
trn_pred_s2 <- predict(mul_s2, newdata = as.data.frame(train.pc.s2))
tst_pred_s2 <- predict(mul_s2, newdata = as.data.frame(test.pc.s2))

mlr_trn_err_s2 <- mean(trn_pred_s2 != train.target.s2)
mlr_tst_err_s2 <- mean(tst_pred_s2 != test.target)

#training error around 17% and test error around 18.8% - worse than LDA (11%) and QDA (5.8%).
#maybe the relationship btw principal components and labels is non-linear,and MLR is linear in the log-odds.

#with a smaller training set, training error decreases to around 12% but test error remains at 18.4%
#probable under-fitting making the model too simple for the complex decision boundaries

#similar train and test errors show low over-fitting,but the low scores indicate high bias 

#Squaring terms
# Scenario 1
trn_pc_sq_s1 <- as.data.frame(train.pc.s1)
tst_pc_sq_s1 <- as.data.frame(test.pc.s1)

# Squared PCs
sqr_trn_s1 <- trn_pc_sq_s1^2
colnames(sqr_trn_s1) <- paste0(colnames(trn_pc_sq_s1), "_sq")

sqr_tst_s1 <- tst_pc_sq_s1^2
colnames(sqr_tst_s1) <- paste0(colnames(tst_pc_sq_s1), "_sq")

# Combine original + squared PCs
trn_pc_sq_s1 <- cbind(trn_pc_sq_s1, sqr_trn_s1)
tst_pc_sq_s1 <- cbind(tst_pc_sq_s1, sqr_tst_s1)

# Scenario 2
trn_pc_sq_s2 <- as.data.frame(train.pc.s2)
tst_pc_sq_s2 <- as.data.frame(test.pc.s2)

# Squared PCs
sqr_trn_s2 <- trn_pc_sq_s2^2
colnames(sqr_trn_s2) <- paste0(colnames(trn_pc_sq_s2), "_sq")

sqr_tst_s2 <- tst_pc_sq_s2^2
colnames(sqr_tst_s2) <- paste0(colnames(tst_pc_sq_s2), "_sq")

# Combine
trn_pc_sq_s2 <- cbind(trn_pc_sq_s2, sqr_trn_s2)
tst_pc_sq_s2 <- cbind(tst_pc_sq_s2, sqr_tst_s2)

library(nnet)

# Scenario 1
mul_sq_s1 <- multinom(train.target.s1 ~ ., data = trn_pc_sq_s1,family=multinomial,maxit=1000,hess=TRUE)
trn_prd_sq_s1 <- predict(mul_sq_s1, newdata = trn_pc_sq_s1)
tst_prd_sq_s1 <- predict(mul_sq_s1, newdata = tst_pc_sq_s1)

mlr_sq_trn_err_s1 <- mean(trn_prd_sq_s1 != train.target.s1)
mlr_sq_tst_err_s1 <- mean(tst_prd_sq_s1 != test.target)

# Scenario 2
mul_sq_s2 <- multinom(train.target.s2 ~ ., data = trn_pc_sq_s2,family=multinomial,maxit=1000,hess=TRUE)
trn_prd_sq_s2 <- predict(mul_sq_s2, newdata = trn_pc_sq_s2)
tst_prd_sq_s2 <- predict(mul_sq_s2, newdata = tst_pc_sq_s2)

mlr_sq_trn_err_s2 <- mean(trn_prd_sq_s2 != train.target.s2)
mlr_sq_tst_err_s2 <- mean(tst_prd_sq_s2 != test.target)

cat("S1 (PC + PC^2) Train Error:", mlr_sq_trn_err_s1, "\n")
cat("S1 (PC + PC^2) Test Error:", mlr_sq_tst_err_s1, "\n")
cat("S2 (PC + PC^2) - Train Error:", mlr_sq_trn_err_s2, "\n")
cat("S2 (PC + PC^2) - Test Error:", mlr_sq_tst_err_s2, "\n")


para_s1 <- coef(mul_sq_s1)   #K-1 row, p+1 col
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



#MLR on PCs alone had:
#  S1 → Train 17%, Test 18.8%   vs 23.8% and 24.3% Squared
#  S2 → Train 12.4%, Test 18.4% vs 22.5% and 27.3% Squared

#Adding squared PCs increased both training and test errors significantly maybe because of overparameterization?
#S1-417 variables for only 9600 samples
#S2-393 variables for 1600 samples 
#slight overfitting and poor generalization due to too many correlated predictors/ unstable 
#smaller training set suffers more
