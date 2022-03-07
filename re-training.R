library(neptune)

neptune_set_api_token("eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMzViMjc1MS1iMWFmLTQyZjUtYjU5Zi04YjVkM2ExZmRkMjIifQ==")
train_path <-  "./data/train"
valid_path <-  "./data/valid"
test_path <-  "./data/test"

setwd("~/repos/R-project-tabular-data-version")
# (neptune) create run that will store re-running metadata
run <-  neptune_init(
  project="common-r/project-tabular-data-version",
  tags=c("training", "from-reference"),
  source_files=c("train.R", "re-training.R")
)

##########################################
# Fetch data info from the reference run #
##########################################

# (neptune) fetch project
reference_run_df <- neptune_fetch_runs_table(project="common-r/project-tabular-data-version", 
                                             tag="reference")


reference_run_id <- tail(reference_run_df$`sys/id`, 1)

# (neptune) resume reference run in the read-only mode
reference_run <- neptune_init(
  project="common-r/project-tabular-data-version",
  run=reference_run_id,
  mode="read-only"
)

# (neptune) download data logged to the reference run
neptune_download(reference_run["data/train"], destination="train")
neptune_download(reference_run["data/valid"], destination="valid")
neptune_download(reference_run["data/test"], destination="test")
neptune_wait(reference_run)

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
neptune_upload(run["data/train_sample"], neptune_file_as_html(head(X_train,20)))

dtrain <- lgb.Dataset(data.matrix(X_train), label = y_train$x, categorical_feature = categorical_feature, colnames = colnames(X_train))
dtest <- lgb.Dataset(data.matrix(X_test), label = y_test$x, reference=dtrain, colnames = colnames(X_train))
dvalid <- lgb.Dataset(data.matrix(X_valid), label = y_valid$x, reference=dtrain, colnames = colnames(X_train))

#######################################
# Assign the same data version to run #
#######################################

neptune_assign(run["data/train"], neptune_fetch(reference_run["data/train"]))
neptune_assign(run["data/valid"], neptune_fetch(reference_run["data/valid"]))
neptune_assign(run["data/test"], neptune_fetch(reference_run["data/test"]))

#######################################
# Fetch params from the reference run #
#######################################

# Fetch the runs parameters
reference_run_params <- neptune_fetch(reference_run["model_params"])
neptune_wait(reference_run)

# (neptune) close reference run
neptune_stop(reference_run)

###########################
# XGBoost: model training #
###########################

evals <- list(train = dtrain, valid = dvalid)
num_round <- 100


# (neptune) pass neptune_callback to the train function and run training
run['model_params'] <- reference_run_params
valids <-  list(train=dtrain, valid=dvalid)
num_round <-  100
base_namespace <- "model_training"

model <- lgb.train(params = reference_run_params,
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

