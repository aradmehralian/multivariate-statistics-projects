library(paran)
library(car)
library(ggplot2)

# loading the provided data set
load("Data/life.Rdata")
sum(is.na(life)) # checking for missing values

# exploring the data set
boxplot(life, ylim = c(1, 4)) # possible values range from 1-4
summary(life)

## part a

# standardize data
life_standard <- scale(life, center = TRUE, scale = TRUE)

# conducting PCA
life_pca <- prcomp(life_standard)
print(life_pca)

# calculating matrix of component scores
standard_scores <- life_standard %*% life_pca$rotation %*% diag(1 / life_pca$sdev)
round(standard_scores, 3)


# scree plot to determine number of components
screeplot(life_pca, type = "lines")

round(lifePCA$sdev^2, 3) # eigenvalues
round((lifePCA$sdev^2) / ncol(lifestand), 3) # proportion explained per component

# Horn's Procedure
# comparing against the mean
set.seed(17)
paran(
  life_standard,
  iterations = 5000,
  graph = TRUE,
  cfa = FALSE,
  centile = 0
)

# comparing against the 95th percentile
set.seed(21)
paran(
  life_standard,
  iterations = 5000,
  graph = TRUE,
  cfa = FALSE,
  centile = 95
)

# NOTE: both methods suggest using the first two principal components

# computing components loading
loadings <- life_pca$rotation %*% diag(life_pca$sdev)

# rounding the first two components
round(loadings[, 1:2], 3)

## part b

par(pty = "s", cex = 0.65)

biplot(
  life_pca,
  pc.biplot = TRUE,
  xlab = "PC1",
  ylab = "PC2",
  xlim = c(-3, 3),
  ylim = c(-3, 3)
)

abline(h = 0)
abline(v = 0)

loadings <- tibble(
  variable = colnames(life),
  load1 = life.PCA$rotation[, 1] * 2,
  load2 = life.PCA$rotation[, 2] * 2
)
scores <- tibble(
  country = rownames(life),
  scores1 = life.PCA$x[, 1] / life.PCA$sdev[1],
  scores2 = life.PCA$x[, 2] / life.PCA$sdev[2]
)

theme_biplot <- theme_classic() +
  theme(
    panel.border = element_rect(
      colour = "black",
      fill = NA,
      linewidth = 0.8
    ),
    text = element_text(family = "serif", size = 14),
    plot.title = element_text(face = "bold", size = 16),
    axis.line = element_line(colour = "black"),
    axis.ticks = element_line(colour = "black"),
    axis.ticks.x.top = element_line(colour = "red"),
    axis.text.x.top = element_text(colour = "red"),
    axis.ticks.y.right = element_line(colour = "red"),
    axis.text.y.right = element_text(colour = "red")
  )

ggplot() +
  scale_x_continuous(
    limits = c(-2.5, 2.5),
    name = "PC1 (39.99%)",
    sec.axis = dup_axis( ~ . / 2, name = NULL)
  ) +
  scale_y_continuous(
    limits = c(-2.5, 2.5),
    name = "PC2 (26.48%)",
    sec.axis = dup_axis( ~ . / 2, name = NULL)
  ) +
  geom_text(
    data = scores,
    aes(x = scores1, y = scores2, label = country),
    color = "black",
    size = 3
  ) +
  geom_segment(
    data = loadings,
    aes(
      x = 0,
      y = 0,
      xend = load1,
      yend = load2
    ),
    arrow = arrow(length = unit(0.25, "cm")),
    color = "red",
    linewidth = 0.8
  ) +
  geom_text(
    data = loadings,
    aes(
      x = load1 * 1.1,
      y = load2 * 1.1,
      label = variable
    ),
    color = "red",
    size = 3.5
  ) +
  geom_hline(
    yintercept = 0,
    color = "black",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.3
  ) +
  geom_vline(
    xintercept = 0,
    color = "black",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.3
  ) +
  ggtitle("Standardized scores on the first two principal components (66.48%)") +
  theme_biplot
