library(mclust)
library(smacof)

load("Data/beer.Rdata")
head(beer)
# checking for missing values
sum(is.na(beer))

## part a

# use transpose of the beer data frame to find the distance between beers.
# the dist() function uses euclidean method by default
beer_dist <- dist(t(beer))
hc_beer <- hclust(beer_dist, method = "ward.D2")
plot(
  hc_beer,
  main = "Dendrogram of 12 Beers (Ward's Method)",
  xlab = "Beers",
  sub = "",
  hang = -1
)
rect.hclust(hc_beer, k = 3, border = 2:4)

## part b

# splitting the data to train and valid sets
train_indices <- seq(1, nrow(beer), by = 2)
beer_train <- beer[train_indices, ]
beer_valid <- beer[-train_indices, ]

# we have to compare each observation in the validation set to the centroid
# of each group in the training data. the function below does that. 
# it is in the lecture slides. minimum distance from a centroid implicates that
# the new data should be part of that cluster.

clusters <- function(x, centers) {
  # compute squared euclidean distance from each sample to each cluster center
  tmp <- sapply(seq_len(nrow(x)),
                function(i) apply(centers, 1,
                                  function(v) sum((x[i, ] - v)^2)))
  max.col(-t(tmp)) # find index of min distance
}

# going through the 3 methods with 3, and 4 clusters
num_cluster_1 <- 3
num_cluster_2 <- 4

# method 1: hierarchical clustering
# with 3 clusters
dist_train_1 <- dist(beer_train, method = "euclidean")
hc_train_1 <- hclust(dist_train_1, method = "ward.D2")
train_groups_1 <- cutree(hc_train_1, k = num_cluster_1)

centroids_hc_1 <- as.matrix(aggregate(beer_train, by = list(train_groups_1),
                                      FUN = mean)[, -1])
# predicting the validation data clusters using the train data
valid_pred_hc_1 <- clusters(beer_valid, centroids_hc_1)

dist_valid_1 <- dist(beer_valid, method = "euclidean")
hc_valid_1 <- hclust(dist_valid_1, method = "ward.D2")
valid_independent_hc_1 <- cutree(hc_valid_1, k = num_cluster_1)

ari_hc_1 <- adjustedRandIndex(valid_pred_hc_1, valid_independent_hc_1)
cat("Stability (ARI) - Hierarchical Method with k =",
    num_cluster_1,
    ":",
    round(ari_hc_1, 3))

# with 4 clusters
dist_train_2 <- dist(beer_train, method = "euclidean")
hc_train_2 <- hclust(dist_train_2, method = "ward.D2")
train_groups_2 <- cutree(hc_train_2, k = num_cluster_2)

centroids_hc_2 <- as.matrix(aggregate(beer_train, by = list(train_groups_2),
                                      FUN = mean)[, -1])

# predicting the validation data clusters using the train data
valid_pred_hc_2 <- clusters(beer_valid, centroids_hc_2)

dist_valid_2 <- dist(beer_valid, method = "euclidean")
hc_valid_2 <- hclust(dist_valid_2, method = "ward.D2")
valid_independent_hc_2 <- cutree(hc_valid_2, k = num_cluster_2)

ari_hc_2 <- adjustedRandIndex(valid_pred_hc_2, valid_independent_hc_2)
cat("Stability (ARI) - Hierarchical Method with k =",
    num_cluster_2,
    ":",
    round(ari_hc_2, 3))

# method 2: k-means
# with 3 clusters
set.seed(7)

kmeans_train_1 <- kmeans(beer_train, centers = num_cluster_1, nstart = 100)
centroids_kmeans_1 <- kmeans_train_1$centers

# assign validation points to the nearest training centroid
valid_pred_kmeans_1 <- clusters(beer_valid, centroids_kmeans_1)

# K-means independently on validation data
kmeans_valid_1 <- kmeans(beer_valid, centers = num_cluster_1, nstart = 100)
valid_independent_kmeans_1 <- kmeans_valid_1$cluster

ari_kmeans_1 <- adjustedRandIndex(valid_pred_kmeans_1, valid_independent_kmeans_1)
cat("Stability (ARI) - K-Means with k =",
    num_cluster_1,
    ":",
    round(ari_kmeans_1, 3))

# with 4 clusters
set.seed(10)

kmeans_train_2 <- kmeans(beer_train, centers = num_cluster_2, nstart = 100)
centroids_kmeans_2 <- kmeans_train_2$centers

# assign validation points to the nearest training centroid
valid_pred_kmeans_2 <- clusters(beer_valid, centroids_kmeans_2)

# K-means independently on validation data
kmeans_valid_2 <- kmeans(beer_valid, centers = num_cluster_2, nstart = 100)
valid_independent_kmeans_2 <- kmeans_valid_2$cluster

ari_kmeans_2 <- adjustedRandIndex(valid_pred_kmeans_2, valid_independent_kmeans_2)
cat("Stability (ARI) - K-Means with k =",
    num_cluster_2,
    ":",
    round(ari_kmeans_2, 3))

# method 3: hierarchical followed by k-means
# with 3 clusters
# euclidean distance and clustering groups are already available from method 1
start_centers_train_1 <- as.matrix(aggregate(beer_train, by = list(train_groups_1),
                                           FUN = mean)[, -1])

hc_kmeans_train_1 <- kmeans(beer_train, centers = start_centers_train_1, iter.max = 100)
centroids_train_1 <- hc_kmeans_train_1$centers

# predicting validation data clusters based on training data
valid_pred_1 <- clusters(beer_valid, centroids_train_1)

# independently clustering the validation data
start_centers_valid_1 <- as.matrix(aggregate(beer_valid, by = list(valid_independent_hc_1),
                                             FUN = mean)[, -1])
hc_kmeans_valid_1 <- kmeans(beer_valid, centers = start_centers_valid_1, iter.max = 100)
valid_independent_1 <- hc_kmeans_valid_1$cluster

ari_hc_kmeans_1 <- adjustedRandIndex(valid_pred_1, valid_independent_1)
cat("Stability (ARI) - Hierarchical + K-Means Method with k =",
    num_cluster_1,
    ":",
    round(ari_hc_kmeans_1, 3))

# with 4 clusters
start_centers_train_2 <- as.matrix(aggregate(beer_train, by = list(train_groups_2),
                                             FUN = mean)[, -1])

hc_kmeans_train_2 <- kmeans(beer_train, centers = start_centers_train_2, iter.max = 100)
centroids_train_2 <- hc_kmeans_train_2$centers

# predicting validation data clusters based on training data
valid_pred_2 <- clusters(beer_valid, centroids_train_2)

# independently clustering the validation data
start_centers_valid_2 <- as.matrix(aggregate(beer_valid, by = list(valid_independent_hc_2),
                                             FUN = mean)[, -1])
hc_kmeans_valid_2 <- kmeans(beer_valid, centers = start_centers_valid_2, iter.max = 100)
valid_independent_2 <- hc_kmeans_valid_2$cluster

ari_hc_kmeans_2 <- adjustedRandIndex(valid_pred_2, valid_independent_2)
cat("Stability (ARI) - Hierarchical + K-Means Method with k =",
    num_cluster_2,
    ":",
    round(ari_hc_kmeans_2, 3))

# comparing the results
ari_scores <- c(
  ari_hc_1,        # Hierarchical (k=3)
  ari_hc_2,        # Hierarchical (k=4)
  ari_kmeans_1,    # K-Means (k=3)
  ari_kmeans_2,    # K-Means (k=4)
  ari_hc_kmeans_1, # Hierarchical + K-Means (k=3)
  ari_hc_kmeans_2  # Hierarchical + K-Means (k=4)
)

methods <- rep(c("Hierarchical", "Hierarchical", 
             "K-Means", "K-Means", 
             "Hierarchical + K-Means", "Hierarchical + K-Means"), by = 2)

num_clusters <- rep(c(3, 4), by = 3)

results <- data.frame(
  Method = methods,
  K = num_clusters,
  ARI_Stability = round(ari_scores, 3)
)

results_sorted <- results[order(-results$ARI_Stability), ]
print(results_sorted)

# k-means with 3 clusters has the highest ARI
set.seed(21)
best_k <- 3

final_kmeans <- kmeans(beer, centers = best_k, nstart = 100)
final_clusters <- final_kmeans$cluster
final_centers <- final_kmeans$centers

print(round(final_centers, 2))

## part c

# unfolding_results <- unfolding(beer, type = "ordinal", conditionality = "row")
# the stress value indicates that how well our data fits into 2-d space
# generally it should be below 0.2, but here because the data is about people's 
# preferences, and it's a subjective matter, it is hard to properly map it to 2-d
# space. This is why the stress value is a bit high (0.33)
cat("Stress-1 value:", round(unfolding_results$stress, 4))


# create a plot for the final clustering solution
# Michele and Lucas really like this part
cluster_colors <- c("red", "green", "blue")
person_colors <- cluster_colors[final_clusters]