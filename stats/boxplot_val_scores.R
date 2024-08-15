library(ggplot2)
library(dplyr)


# Load data
val_scores = read.csv(file = 'X:/train_output/2_rss2seg_v2_nlabelsfix/fold0/val_preds/val_scores.csv', sep = ';')

# Remove unwanted col
val_scores = val_scores %>% select(-val_idx)


grid(nx=16, ny=16)
boxplot(
  val_scores,
  names=c("mean dice","femcar1","tibcar2","patcar3","fem4","tib5","pat6")
)


mean(val_scores$mean_dices)
