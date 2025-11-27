#Task 2
library(mclust)      
    

load("beer.Rdata")
beer_mat <- as.matrix(beer)   
dim(beer_mat)  
head(beer_mat)

#Hierarchical clustering - Ward's method 

beer_t <- t(beer_mat)  
dist_beer <- dist(beer_t, method = "euclidean",diag = TRUE, upper = TRUE)^2
hc_beer <- hclust(d_beer, method = "ward.D2")

# Dendrogram
plot(hc_beer, main = "Dendrogram Ward's Method")
rect.hclust(hc_beer, k = 3, border = 2:4)  

#Split data
set.seed(1000)
trn_idx <- seq(1, nrow(beer_mat), 2)  
val_idx <- seq(2, nrow(beer_mat), 2) 
trn_mat <- beer_mat[trn_idx, ]
val_mat <- beer_mat[val_idx, ]

dim(trn_mat)  
dim(val_mat)

trn_t <- t(trn_mat)
dist_trn <- dist(trn_t, method = "euclidean")^2
val_t <- t(val_mat)
dist_val <- dist(val_t, method = "euclidean")^2
hc_trn <- hclust(dist_trn, method = "ward.D2")
hc_val <- hclust(dist_val, method = "ward.D2")

trn_cl3 <- cutree(hc_trn, k = 3)
trn_cl4 <- cutree(hc_trn, k = 4)

val_cl3 <- cutree(hc_val, k = 3)
val_cl4 <- cutree(hc_val, k = 4)

# Training set dendrogram
plot(hc_trn, main = "Training Set: HC", xlab = "", sub = "", cex = 0.8, hang = -1)
rect.hclust(hc_trn, k = 3, border = "red")  # 3 clusters
rect.hclust(hc_trn, k = 4, border = "blue") # 4 clusters

# Validation set dendrogram
plot(hc_val, main = "Validation Set: HC", xlab = "", sub = "", cex = 0.8, hang = -1)
rect.hclust(hc_val, k = 3, border = "red")
rect.hclust(hc_val, k = 4, border = "blue")
#lack of stability

#K -Means
set.seed(1000)

km_trn3 <- kmeans(trn_mat, centers = 3, nstart = 100)
km_trn4 <- kmeans(trn_mat, centers = 4, nstart = 100)
km_val3 <- kmeans(val_mat, centers = 3, nstart = 100)
km_val4 <- kmeans(val_mat, centers = 4, nstart = 100)

#Ward + K-means
combined <- function(data, k) 
{
  data <- as.matrix(data)  
  hc <- hclust(dist(data)^2, method = "ward.D2")
  clusters <- cutree(hc, k)
  cent <- rowsum(data, clusters) / as.vector(table(clusters))
  kmeans(data, centers = cent)
}

# Training
comb_trn3 <- combined(trn_mat, 3)
comb_trn4 <- combined(trn_mat, 4)

# Validation
comb_val3 <- combined(val_mat, 3)
comb_val4 <- combined(val_mat, 4)
