a <- read.csv("https://raw.githubusercontent.com/neptune-ai/project-tabular-data/main/data/train.csv")

train_idx <- caret::createDataPartition(a$SalePrice, p = .7, 
                    list = FALSE, 
                    times = 1)
train_set <- a[train_idx,]
test_set <- a[-train_idx,]
valid_idx <- caret::createDataPartition(test_set$SalePrice, p = .5, 
                                        list = FALSE, 
                                        times = 1)

valid_set <- test_set[valid_idx, ]
test_set <- test_set[-valid_idx, ]

setwd("~/repos/R-project-tabular-data-version/data")
dir.create('train')
dir.create('valid')
dir.create('test')
write.csv(train_set[,-81],file = 'train/X.csv')
write.csv(valid_set[,-81],file = 'valid/X.csv')
write.csv(test_set[,-81],file = 'test/X.csv')

write.csv(train_set[,81],file = 'train/Y.csv')
write.csv(valid_set[,81],file = 'valid/Y.csv')
write.csv(test_set[,81],file = 'test/Y.csv')
