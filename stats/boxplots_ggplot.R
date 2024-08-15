library(reshape)
library(ggplot2)
file1 = "X:/sfransen/k2s_umcg/train_output/0_weighted_dice/fold0/val_preds_mirroring/val_scores.csv"
file1 = "Y:/train_output/2_rss2seg_v2_nlabelsfix/fold0/val_preds/conn_mirror/val_scores_conn_mirror.csv"
file2 = "X:/sfransen/k2s_umcg/train_output/0_weighted_dice/fold0/val_preds_mirroring/val_scores_dilation_6.csv"
file3 = "X:/sfransen/k2s_umcg/train_output/0_weighted_dice/fold0/val_preds_mirroring/val_scores_post_processed_perc0.6.csv"

###
# Load first dataframe
df1 = read.csv(file = file1, sep = ';') #with label 6 x 1.5 (to dialate a single class)
df1 = melt(df1, id = c("val_idx"))
colnames(df1) = c("val_idx", "variable", "dice")
df1$process_type = "R1-RSS"

# Second dataloading part
df2 = read.csv(file = file2, sep = ';') #with label 6 x 1.5 (to dialate a single class)
df2 = melt(df2, id = c("val_idx"))
colnames(df2) = c("val_idx", "variable", "dice")
df2$process_type = "dilation_6_weighted"

# Third dataloading part
df3 = read.csv(file = file3, sep = ';') #with label 6 x 1.5 (to dialate a single class)
df3 = melt(df3, id = c("val_idx"))
colnames(df3) = c("val_idx", "variable", "dice")
df3$process_type = "conn_weighted"
####

# Row bind all the dataframes
df <- rbind(df1, df2, df3)

# Make grouped boxplot
bp = ggplot(df, aes(x = variable, y = dice, fill = process_type)) +
  geom_boxplot() +
  facet_wrap( ~ variable, scales = "free")  # try commenting this away

# Visualize
bp

