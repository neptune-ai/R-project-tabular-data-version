library(neptune)

neptune_set_api_token("eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzViMjc1MS1iMWFmLTQyZjUtYjU5Zi04YjVkM2ExZmRkMjIifQ==")
train_path <-  "./data/train"
valid_path <-  "./data/valid"
test_path <-  "./data/test"

setwd("~/repos/R-project-tabular-data-version")
# (neptune) create run
run <-  neptune_init(
  project="common-r/project-tabular-data-version",
  tags=c("training"),
  source_files=c("train.R")
)

##########################
# track data version     #
##########################

# (neptune) track data version
neptune_track_files(run["data/train"], train_path)
neptune_track_files(run["data/valid"], valid_path)
neptune_track_files(run["data/test"], test_path)
neptune_wait(run)

# (neptune) prepare data
neptune_download(run["data/train"], destination = "train")
neptune_download(run["data/valid"], destination = "valid")
neptune_download(run["data/test"], destination = "test")
neptune_wait(run)

setwd("~/repos/R-project-tabular-data-version")
X_train <-  readr::read_csv("./train/X.csv")
y_train <-  readr::read_csv("./train/y.csv")
X_valid <-  readr::read_csv("./valid/X.csv")
y_valid <-  readr::read_csv("./valid/y.csv")
X_test <-  readr::read_csv("./test/X.csv")
y_test <-  readr::read_csv("./test/y.csv")

# make sure test and valid datasets have same factor levels as train
categorical_feature <- c()
for(col in colnames(X_train)){
  if(class(X_train[[col]])=='character' | class(X_train[[col]])=='factor'){
    X_train[[col]] <- as.factor(X_train[[col]])
    if(length(levels(X_train[[col]]))){
      X_valid[[col]] <- NULL
      X_test[[col]] <- NULL
      X_train[[col]] <- NULL
    }else{
      categorical_feature <- c(categorical_feature, col)
      
      X_valid[[col]] <- factor(X_valid[[col]], levels = levels(X_train[[col]]))
      X_test[[col]] <- factor(X_test[[col]], levels = levels(X_train[[col]]))  
    }
  }
}
library(lightgbm)

# (neptune) log train sample
neptune_upload(run["data/train_sample"], neptune_file_as_html(head(X_train, 20)))

###########################
# XGBoost: model training #
###########################
dtrain <- lgb.Dataset(data.matrix(X_train), label = y_train$x, categorical_feature = categorical_feature, colnames = colnames(X_train))
dtest <- lgb.Dataset(data.matrix(X_test), label = y_test$x, reference=dtrain, colnames = colnames(X_train))
dvalid <- lgb.Dataset(data.matrix(X_valid), label = y_valid$x, reference=dtrain, colnames = colnames(X_train))

# define parameters
model_params <- list(
  "eta"= 0.2974,
  "max_depth"= 5,
  "colsample_bytree"= 0.91,
  "subsample"= 0.91,
  "objective"= "regression"
)
valids <-  list(train=dtrain, valid=dvalid)
num_round <-  100
base_namespace <- "model_training"

model <- lgb.train(params = model_params,
          data = dtrain,
          nrounds = num_round,
          valids = valids,
          eval=c('mae','rmse'))
png('importance.png')
lgb.plot.importance(lgb.importance(model))
dev.off()
lgb.save(model, filename = "model.txt")
neptune_upload(run[paste0(base_namespace,'/importance')],value = 'importance.png')
neptune_upload(run[paste0(base_namespace,'/model')], value = 'model.txt')

neptune_sync(run, wait = T)

# (neptune) download model from the run to make predictions on test data
neptune_download(run[paste0(base_namespace, "/model")], "downloaded_model.txt")

model <- lgb.load('downloaded_model.txt')

test_preds <- predict(model, data.matrix(X_test))

# (neptune) log test scores
run[paste0(base_namespace,"/test_score/rmse")] <- caret::RMSE(obs=y_test$x, pred=test_preds)
run[paste0(base_namespace,"/test_score/mae")] <- caret::MAE(obs=y_test$x, pred=test_preds)
neptune_sync(run, wait=T)
neptune_stop(run)
