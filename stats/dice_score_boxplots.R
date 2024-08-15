library(ggplot2)
library(dplyr)
library(ggpmisc)

# Load
usampled_fold1 = read.csv(file = 'Y:/train_output/8_r8_rss2seg_weighted/fold1/val_preds/val_scores_.csv', sep = ';') #with label 6 x 1.5 (to dialate a single class)
normal = read.csv(file = 'X:/qvlohuizen/k2s_umcg/train_output/2_rss2seg_v2_nlabelsfix/fold0/val_preds/val_scores.csv', sep = ';')
usampled_0 = read.csv(file = 'X:/sfransen/k2s_umcg/train_output/0_rss_usampled_test/fold0/val_preds/val_scores.csv', sep = ';')
usampled_1 = read.csv(file = 'X:/sfransen/k2s_umcg/train_output/1_rss_usampled_test/fold1/val_preds/val_scores.csv', sep = ';')
weighted_0 = read.csv(file = 'X:/sfransen/k2s_umcg/train_output/0_weighted_dice/fold0/val_preds_original/val_scores.csv', sep = ';')

# Remove unwanted col
normal = normal %>% select(-val_idx)
usampled_0 = usampled_0 %>% select(-val_idx)
usampled_1 = usampled_1 %>% select(-val_idx)
weighted_0 = weighted_0 %>% select(-val_idx)

# add descriptive col (to select the right data lateron)
normal_id <- cbind(ID = 'Normal', normal)
usampled_0_id <- cbind(ID = 'U_fold0', usampled_0)
usampled_1_id <- cbind(ID = 'U_fold1', usampled_1)
weighted_0_id <- cbind(ID = 'U_weighted_dice_fold0', weighted_0)

# add all to one dateframe
df <- rbind(normal_id, usampled_0_id, usampled_1_id, weighted_0_id)

# create boxplot and select proper data by selecting the descriptive column
plt <-
  boxplot(
    df[, -1],
    boxfill = NA,
    border = NA,
    main = "boxplot performances",
    ylab = "dice score"
  ) #invisible boxes - only axes and plot area
plt <-
  boxplot(
    df[df$ID == "Normal",-1],
    xaxt = "n",
    add = TRUE,
    boxfill = "red",
    boxwex = 0.25,
    at = 1:ncol(df[, -1]) - 0.3
  ) #shift these left by -0.3
plt <-
  boxplot(
    df[df$ID == "U_weighted_dice_fold0",-1],
    xaxt = "n",
    add = TRUE,
    boxfill = "green",
    boxwex = 0.25,
    at = 1:ncol(df[, -1])
  ) #shift to the right by 0.0
plt <-
  boxplot(
    df[df$ID == "U_fold0",-1],
    xaxt = "n",
    add = TRUE,
    boxfill = "blue",
    boxwex = 0.25,
    at = 1:ncol(df[, -1]) + 0.3
  ) #shift to the right by +0.3

# create legend information, including calculating the mean
normal_str = paste(c("Normal", round(mean(
  normal_id$mean_dices
), 2)), collapse = " ")
weighted_0_str = paste(c("R8 Weighted 0", round(mean(
  weighted_0_id$mean_dices
), 2)), collapse = " ")
usampled_0_str = paste(c("R8 undersampled 0", round(mean(
  usampled_0_id$mean_dices
), 2)), collapse = " ")


# add legend to plot
plt <-
  legend(
    "bottomright",
    c(normal_str, weighted_0_str, usampled_0_str),
    border = "black",
    fill = c("red", "green", "blue")
  )



normal_id$index <- 1:nrow(normal_id)
usampled_id$index <- 1:nrow(usampled_id)
df <- rbind(normal_id, usampled_id)
test <-
  ggsummarystats(
    df,
    x = "ID",
    y = 'mean_dices',
    ggfunc = ggboxplot,
    add = "jitter"
  )
test <-
  ggsummarystats(df[df$ID == "Bad", -1], x = "mean_dices", y = 'fem_car1', ggfunc = ggboxplot)
test
