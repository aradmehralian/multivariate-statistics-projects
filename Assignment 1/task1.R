library(paran) 
library(car) 
library(ggplot2) 

# loading the provided data set
load("Data/life.Rdata")
sum(is.na(life)) # checking for missing values

# exploring the data set
boxplot(life, ylim=c(1,4)) # possible values range from 1-4 	 
summary(life)

## part a

# standardize data 
life_standard <- scale(life, center=TRUE, scale=TRUE)

# conducting PCA
life_pca <-prcomp(life_standard) 
print(life_pca)

# calculating matrix of component scores
standard_scores <- life_standard %*% life_pca$rotation %*% diag(1/life_pca$sdev)  
round(standard_scores, 3) 


# scree plot to determine number of components
screeplot(life_pca, type="lines") 

round(lifePCA$sdev^2, 3) # eigenvalues 
round((lifePCA$sdev^2)/ncol(lifestand), 3) # proportion explained per component 

# Horn's Procedure
# comparing against the mean
set.seed(17) 
paran(life_standard, iterations=5000, graph=TRUE, cfa=FALSE, centile=0) 

# comparing against the 95th percentile
set.seed(21) 
paran(life_standard, iterations=5000, graph=TRUE, cfa=FALSE, centile=95)

# NOTE: both methods suggest using the first two principal components

# computing components loading
loadings <- life_pca$rotation %*% diag(life_pca$sdev)

# rounding the first two components
round(loadings[,1:2], 3)

## part b

par(pty="s", cex=0.65)

biplot(life_pca, pc.biplot=TRUE, xlab="PC1", ylab="PC2",
       xlim=c(-3,3), ylim=c(-3,3)) 

abline(h = 0)
abline(v = 0) 